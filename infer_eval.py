#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import (
    load_examples,
    load_split_assignments,
    select_ids_for_splits,
    stratified_split,
    summarize_split_assignments,
)
from nemotron_baseline.metric import result_to_json, score_prediction, summarize_results
from nemotron_baseline.prompts import build_generation_prompt


def load_config_defaults(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    with Path(config_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit("Run config must contain a JSON object.")
    return payload


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--run-config", default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args()
    defaults = load_config_defaults(bootstrap_args.run_config)

    parser = argparse.ArgumentParser(
        description="Run local held-out evaluation for a Nemotron LoRA adapter."
    )
    parser.add_argument("--run-config", default=bootstrap_args.run_config)
    parser.add_argument("--train-csv", default=defaults.get("train_csv", "data/train.csv"))
    parser.add_argument("--split-csv", default=defaults.get("split_csv"))
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=defaults.get("eval_splits"),
        help="Named splits from --split-csv to evaluate on.",
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-path", default=defaults.get("model_path"))
    parser.add_argument(
        "--kaggle-model-handle",
        default=defaults.get(
            "kaggle_model_handle",
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        ),
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=defaults.get("val_fraction", 0.2),
    )
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--disable-kaggle-triton-fixes", action="store_true")
    parser.add_argument(
        "--predictions-jsonl",
        default=None,
        help="Optional output file for per-example predictions.",
    )
    return parser.parse_args(remaining)


def require_inference_dependencies():
    try:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install transformers, peft, and torch."
        ) from exc

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def apply_kaggle_triton_fixes() -> None:
    try:
        import torch  # type: ignore
        import torch.nn.functional as torch_f  # type: ignore
    except ImportError:
        return

    def _pure_rmsnorm_fn(
        x,
        weight,
        bias=None,
        z=None,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        upcast=True,
    ):
        del group_size, norm_before_gate
        dtype = x.dtype
        if upcast:
            x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        out = x_normed * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None:
            out = out * torch_f.silu(z.float())
        return out.to(dtype)

    for module in list(sys.modules.values()):
        if hasattr(module, "rmsnorm_fn"):
            module.rmsnorm_fn = _pure_rmsnorm_fn


def disable_nemotron_fast_path() -> None:
    for name, module in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(module, "is_fast_path_available"):
            module.is_fast_path_available = False


def resolve_model_path(args: argparse.Namespace, kagglehub_module) -> str:
    if args.model_path:
        return args.model_path
    if kagglehub_module is None:
        raise SystemExit(
            "No --model-path was provided and kagglehub is not installed. "
            "Either install kagglehub or pass a local base-model path."
        )
    return kagglehub_module.model_download(args.kaggle_model_handle)


def batched(sequence, batch_size: int):
    for start in range(0, len(sequence), batch_size):
        yield sequence[start : start + batch_size]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    deps = require_inference_dependencies()
    try:
        import kagglehub  # type: ignore
    except ImportError:
        kagglehub = None

    if not args.disable_kaggle_triton_fixes:
        apply_kaggle_triton_fixes()

    model_path = resolve_model_path(args, kagglehub)
    examples = load_examples(args.train_csv)
    split_assignments = None
    output_split_name = args.split
    if args.split_csv:
        split_assignments = load_split_assignments(args.split_csv)
        eval_split_names = args.eval_splits or ["eval"]
        selected_ids = select_ids_for_splits(split_assignments, eval_split_names)
        eval_examples = [example for example in examples if example.id in selected_ids]
        output_split_name = "_".join(eval_split_names)
        if not eval_examples:
            raise SystemExit(
                f"No evaluation examples matched splits {eval_split_names!r} in {args.split_csv!r}."
            )
    else:
        train_examples, val_examples = stratified_split(
            examples,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        eval_examples = val_examples if args.split == "val" else train_examples
    if args.max_eval_samples is not None:
        eval_examples = eval_examples[: args.max_eval_samples]

    tokenizer = deps["AutoTokenizer"].from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = deps["AutoModelForCausalLM"].from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=deps["torch"].bfloat16,
    )
    disable_nemotron_fast_path()
    model = deps["PeftModel"].from_pretrained(base_model, args.adapter_dir)
    model.eval()

    results = []
    for batch_examples in batched(eval_examples, args.batch_size):
        prompts = [
            build_generation_prompt(tokenizer, example.prompt) for example in batch_examples
        ]
        batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}

        with deps["torch"].inference_mode():
            generated = model.generate(
                **batch,
                do_sample=False,
                top_p=1.0,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = batch["input_ids"].shape[1]
        decoded_predictions = tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
        )

        for example, raw_prediction in zip(batch_examples, decoded_predictions):
            results.append(
                score_prediction(
                    example_id=example.id,
                    category=example.category,
                    raw_prediction=raw_prediction,
                    gold_answer=example.answer or "",
                )
            )

    summary = summarize_results(results)
    summary.update(
        {
            "model_path": model_path,
            "adapter_dir": str(Path(args.adapter_dir).resolve()),
            "split": output_split_name,
            "split_csv": str(Path(args.split_csv).resolve()) if args.split_csv else None,
            "eval_splits": args.eval_splits or (["eval"] if args.split_csv else None),
            "split_counts": summarize_split_assignments(split_assignments) if split_assignments else None,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "evaluated_examples": len(results),
        }
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    adapter_root = Path(args.adapter_dir).resolve().parent
    write_json(adapter_root / f"{output_split_name}_summary.json", summary)

    if args.predictions_jsonl:
        prediction_path = Path(args.predictions_jsonl)
    else:
        prediction_path = adapter_root / f"{output_split_name}_predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        "\n".join(result_to_json(result) for result in results) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
