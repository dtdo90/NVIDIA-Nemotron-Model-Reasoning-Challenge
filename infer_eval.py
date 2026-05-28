#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
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
from nemotron_baseline.prompts import build_generation_prompt, build_user_message
from nemotron_baseline.runtime import (
    check_nemotron_runtime_dependencies,
    disable_transformers_vision_imports,
)

DEFAULT_MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_PATH = os.environ.get("MODEL_PATH") or os.environ.get("BASE_MODEL_PATH") or DEFAULT_MODEL_PATH
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    parser.add_argument(
        "--model-path",
        default=defaults.get("model_path") or MODEL_PATH,
    )
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
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default=defaults.get("backend", "vllm"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=7680,
        help="Match the competition metric max_tokens setting.",
    )
    parser.add_argument("--temperature", type=float, default=defaults.get("temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=defaults.get("top_p", 1.0))
    parser.add_argument("--max-num-seqs", type=int, default=defaults.get("max_num_seqs", 128))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=defaults.get("gpu_memory_utilization", 0.85),
    )
    parser.add_argument("--max-model-len", type=int, default=defaults.get("max_model_len", 8192))
    parser.add_argument("--max-lora-rank", type=int, default=defaults.get("max_lora_rank", 32))
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
        from peft import LoraConfig, get_peft_model  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install transformers, peft, and torch."
        ) from exc

    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def load_adapter_tensors(model, adapter_dir: Path, torch) -> None:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file  # type: ignore

        raw_state = load_file(str(safetensors_path), device="cpu")
        source_path = safetensors_path
    elif bin_path.exists():
        try:
            raw_state = torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            raw_state = torch.load(bin_path, map_location="cpu")
        source_path = bin_path
    else:
        raise SystemExit(f"No adapter weights found in {adapter_dir}")

    model_state = model.state_dict()

    def with_default_adapter(key: str) -> str:
        for marker in (
            ".lora_A.",
            ".lora_B.",
            ".lora_embedding_A.",
            ".lora_embedding_B.",
            ".lora_magnitude_vector.",
        ):
            if marker in key:
                prefix, suffix = key.split(marker, 1)
                if not suffix.startswith("default."):
                    return f"{prefix}{marker}default.{suffix}"
        return key

    remapped = {}
    unmatched = []
    for key, value in raw_state.items():
        candidates = [
            key,
            with_default_adapter(key),
            f"base_model.model.{key}",
            with_default_adapter(f"base_model.model.{key}"),
        ]
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                remapped[candidate] = value
                break
        else:
            unmatched.append(key)

    if not remapped:
        sample_raw = next(iter(raw_state), "<empty>")
        sample_model = next((key for key in model_state if "lora_" in key), "<no lora keys>")
        raise SystemExit(
            "Loaded zero adapter tensors. "
            f"Sample saved key: {sample_raw}. Sample model LoRA key: {sample_model}."
        )
    if unmatched:
        sample_model = next((key for key in model_state if "lora_" in key), "<no lora keys>")
        raise SystemExit(
            f"Only matched {len(remapped)}/{len(raw_state)} adapter tensors from {source_path}. "
            f"Unmatched saved keys: {unmatched[:5]}. Sample model LoRA key: {sample_model}."
        )

    result = model.load_state_dict(remapped, strict=False)
    print(
        f"Loaded {len(remapped)}/{len(raw_state)} adapter tensors from {source_path}. "
        f"Missing keys: {len(result.missing_keys)}, unexpected keys: {len(result.unexpected_keys)}"
    )


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

    for name, module in list(sys.modules.items()):
        if "modeling_nemotron_h" not in name:
            continue
        try:
            rmsnorm_fn = getattr(module, "rmsnorm_fn", None)
        except Exception:
            continue
        if rmsnorm_fn is not None:
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


def load_row_metadata(csv_path: str | Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_id = row.get("id")
            if row_id:
                metadata[row_id] = row
    return metadata


def _stats_payload(total: int, correct: int) -> dict[str, int | float]:
    return {
        "total": total,
        "correct": correct,
        "accuracy": float(correct) / float(total) if total else 0.0,
    }


def classify_transformation_detail(example, row_metadata: dict[str, str]) -> tuple[str, str] | None:
    if example.category != "Transformation Rules":
        return None

    source = row_metadata.get("source", "")
    label = row_metadata.get("category") or row_metadata.get("label") or ""
    source_lower = source.lower()
    label_lower = label.lower()

    if "numeric_equation_transformation_rules" in source_lower or "numeric equation" in label_lower:
        family = "numeric_equation"
        if "direct_template" in source_lower:
            subtype = "direct_template"
        elif "operator_absence" in source_lower:
            subtype = "operator_absence"
        elif "ab_cd" in source_lower:
            subtype = "ab_cd"
        elif "ba_dc" in source_lower:
            subtype = "ba_dc"
        else:
            subtype = "others"
        return family, subtype

    if (
        "symbol_equation_transformation_rules" in source_lower
        or "symbol_transform" in source_lower
        or "symbol transform" in label_lower
    ):
        family = "symbol_equation"
        if "direct_template" in source_lower:
            subtype = "direct_template"
        elif "ba_dc" in source_lower:
            subtype = "ba_dc"
        else:
            subtype = "others"
        return family, subtype

    # Held-out rows without source metadata still count as Transformation Rules,
    # but we do not want to invent a subtype from weak evidence.
    return "unknown", "others"


def summarize_transformation_details(eval_examples, results, metadata_by_id: dict[str, dict[str, str]]) -> dict[str, object]:
    family_order = {
        "numeric_equation": ["ab_cd", "ba_dc", "direct_template", "operator_absence", "others"],
        "symbol_equation": ["ba_dc", "direct_template", "others"],
        "unknown": ["others"],
    }
    counters: dict[str, dict[str, dict[str, int]]] = {}

    for example, result in zip(eval_examples, results):
        detail = classify_transformation_detail(example, metadata_by_id.get(example.id, {}))
        if detail is None:
            continue
        family, subtype = detail
        if family not in counters:
            counters[family] = {}
        if subtype not in counters[family]:
            counters[family][subtype] = {"total": 0, "correct": 0}
        counters[family][subtype]["total"] += 1
        counters[family][subtype]["correct"] += 1 if result.correct else 0

    summary: dict[str, object] = {}
    for family in ("numeric_equation", "symbol_equation", "unknown"):
        subtype_counts = counters.get(family, {})
        if not subtype_counts:
            continue
        total = sum(stats["total"] for stats in subtype_counts.values())
        correct = sum(stats["correct"] for stats in subtype_counts.values())
        ordered_subtypes = family_order.get(family, []) + sorted(
            set(subtype_counts) - set(family_order.get(family, []))
        )
        by_subtype = {
            subtype: _stats_payload(
                subtype_counts.get(subtype, {}).get("total", 0),
                subtype_counts.get(subtype, {}).get("correct", 0),
            )
            for subtype in ordered_subtypes
        }
        summary[family] = {
            **_stats_payload(total, correct),
            "by_subtype": by_subtype,
        }
    return summary


def build_vllm_generation_prompt(tokenizer, prompt: str) -> str:
    user_content = build_user_message(prompt)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return user_content


def generate_with_vllm(args: argparse.Namespace, model_path: str, eval_examples):
    try:
        from vllm import LLM, SamplingParams  # type: ignore
        from vllm.lora.request import LoRARequest  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "vLLM backend requested but vllm could not be imported. "
            "This usually means vllm is missing or one of its runtime dependencies is broken.\n"
            f"Import error: {exc}\n"
            "Install/fix vllm, or rerun with --backend transformers."
        ) from exc

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [build_vllm_generation_prompt(tokenizer, example.prompt) for example in eval_examples]
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, str(Path(args.adapter_dir).resolve())),
    )
    return [output.outputs[0].text for output in outputs]


def generate_with_transformers(args: argparse.Namespace, model_path: str, eval_examples):
    disable_transformers_vision_imports()
    check_nemotron_runtime_dependencies()
    deps = require_inference_dependencies()
    if not args.disable_kaggle_triton_fixes:
        apply_kaggle_triton_fixes()

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
    adapter_dir = Path(args.adapter_dir)
    lora_config = deps["LoraConfig"].from_pretrained(adapter_dir)
    model = deps["get_peft_model"](base_model, lora_config)
    load_adapter_tensors(model, adapter_dir, deps["torch"])
    model.eval()

    decoded_predictions = []
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
            cache_position = deps["torch"].arange(
                batch["input_ids"].shape[1],
                device=batch["input_ids"].device,
            )
            generated = model.generate(
                **batch,
                do_sample=args.temperature > 0,
                temperature=None if args.temperature == 0 else args.temperature,
                top_p=args.top_p,
                cache_position=cache_position,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = batch["input_ids"].shape[1]
        decoded_predictions.extend(
            tokenizer.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
            )
        )
    return decoded_predictions


def main() -> None:
    args = parse_args()
    try:
        import kagglehub  # type: ignore
    except ImportError:
        kagglehub = None

    model_path = resolve_model_path(args, kagglehub)
    examples = load_examples(args.train_csv)
    metadata_by_id = load_row_metadata(args.train_csv)
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

    if args.backend == "vllm":
        raw_predictions = generate_with_vllm(args, model_path, eval_examples)
    else:
        raw_predictions = generate_with_transformers(args, model_path, eval_examples)

    results = [
        score_prediction(
            example_id=example.id,
            category=example.category,
            raw_prediction=raw_prediction,
            gold_answer=example.answer or "",
        )
        for example, raw_prediction in zip(eval_examples, raw_predictions)
    ]

    summary = summarize_results(results)
    summary.update(
        {
            "backend": args.backend,
            "model_path": model_path,
            "adapter_dir": str(Path(args.adapter_dir).resolve()),
            "split": output_split_name,
            "split_csv": str(Path(args.split_csv).resolve()) if args.split_csv else None,
            "eval_splits": args.eval_splits or (["eval"] if args.split_csv else None),
            "split_counts": summarize_split_assignments(split_assignments) if split_assignments else None,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_model_len": args.max_model_len if args.backend == "vllm" else None,
            "transformation_rules_detail": summarize_transformation_details(
                eval_examples,
                results,
                metadata_by_id,
            ),
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
