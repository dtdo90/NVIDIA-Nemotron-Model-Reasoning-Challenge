#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.type_diagnostics.lib.common import (
    CATEGORY_TO_SLUG,
    QUESTION_TYPES,
    classify_subtype,
    is_eval_eligible,
    safe_label,
)

from nemotron_baseline.data import (
    load_examples,
    load_split_assignments,
    select_ids_for_splits,
    stratified_split,
    summarize_split_assignments,
)
from nemotron_baseline.metric import score_prediction, summarize_results
from nemotron_baseline.prompts import build_generation_prompt, build_user_message
from nemotron_baseline.runtime import (
    check_nemotron_runtime_dependencies,
    disable_transformers_vision_imports,
)

DEFAULT_MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
KAGGLE_MODEL_PATH = Path("/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1")


def default_model_path() -> str:
    explicit_model_path = os.environ.get("MODEL_PATH") or os.environ.get("BASE_MODEL_PATH")
    if explicit_model_path:
        return explicit_model_path
    if KAGGLE_MODEL_PATH.exists():
        return str(KAGGLE_MODEL_PATH)
    return DEFAULT_MODEL_PATH


MODEL_PATH = default_model_path()
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
    parser.add_argument("--max-num-seqs", type=int, default=defaults.get("max_num_seqs", 64))
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
    parser.add_argument(
        "--report-dir",
        default=defaults.get("report_dir"),
        help="Optional directory for summary, predictions, and failed trace samples.",
    )
    parser.add_argument(
        "--failed-traces-per-subtype",
        type=int,
        default=defaults.get("failed_traces_per_subtype", 3),
        help="Write and print up to this many failed generations for each non-perfect subtype.",
    )
    parser.add_argument(
        "--no-print-failed-traces",
        action="store_true",
        help="Write failed trace samples to disk but do not print the generations to stdout.",
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


def is_eval_eligible_metadata(row_metadata: dict[str, str]) -> bool:
    return is_eval_eligible(row_metadata)


def _stats_payload(total: int, correct: int) -> dict[str, int | float]:
    return {
        "total": total,
        "correct": correct,
        "accuracy": float(correct) / float(total) if total else 0.0,
    }


def row_category(example, row_metadata: dict[str, str]) -> str:
    return (
        row_metadata.get("category")
        or row_metadata.get("label")
        or example.category
        or "unknown"
    )


def infer_question_type(example, row_metadata: dict[str, str]) -> str:
    diagnostic_type = row_metadata.get("diagnostic_type", "").strip()
    if diagnostic_type in QUESTION_TYPES:
        return diagnostic_type

    category = row_category(example, row_metadata)
    if category in CATEGORY_TO_SLUG:
        return CATEGORY_TO_SLUG[category]

    source_lower = row_metadata.get("source", "").lower()
    label_lower = (row_metadata.get("label") or row_metadata.get("category") or "").lower()
    if "numeric_equation_transformation_rules" in source_lower or "numeric equation" in label_lower:
        return "numeric_equation"
    if (
        "symbol_equation_transformation_rules" in source_lower
        or "symbol_transform" in source_lower
        or "symbol transform" in label_lower
    ):
        return "symbol_transform"

    if example.category == "Transformation Rules":
        return "transformation_rules_unknown"
    return safe_label(category)


def infer_diagnostic_subtype(example, row_metadata: dict[str, str], question_type: str) -> str:
    diagnostic_subtype = row_metadata.get("diagnostic_subtype", "").strip()
    if diagnostic_subtype:
        return diagnostic_subtype

    if question_type in QUESTION_TYPES:
        row_for_classification = dict(row_metadata)
        row_for_classification.setdefault("id", example.id)
        row_for_classification.setdefault("prompt", example.prompt)
        row_for_classification.setdefault("answer", example.answer or "")
        if not row_for_classification.get("category"):
            row_for_classification["category"] = QUESTION_TYPES[question_type]["category"]
        try:
            return classify_subtype(row_for_classification)
        except Exception:
            return "standard"

    return "unknown"


def question_type_display(question_type: str, category: str) -> str:
    if question_type in QUESTION_TYPES:
        return QUESTION_TYPES[question_type]["display"]
    return category or question_type


def build_evaluation_records(eval_examples, results, metadata_by_id: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for example, result in zip(eval_examples, results):
        row_metadata = metadata_by_id.get(example.id, {})
        category = row_category(example, row_metadata)
        question_type = infer_question_type(example, row_metadata)
        diagnostic_subtype = infer_diagnostic_subtype(example, row_metadata, question_type)
        records.append(
            {
                "id": result.example_id,
                "category": category,
                "question_type": question_type,
                "question_type_display": question_type_display(question_type, category),
                "diagnostic_subtype": diagnostic_subtype,
                "source_mode": row_metadata.get("source_mode", "unknown") or "unknown",
                "source": row_metadata.get("source", ""),
                "gold_answer": result.gold_answer,
                "extracted_prediction": result.extracted_prediction,
                "correct": result.correct,
                "prompt": example.prompt,
                "raw_prediction": result.raw_prediction,
            }
        )
    return records


def summarize_by_type(records: list[dict[str, object]]) -> dict[str, object]:
    type_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    subtype_totals: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "correct": 0})
    )
    subtype_sources: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    type_display: dict[str, str] = {}

    for record in records:
        question_type = str(record["question_type"])
        subtype = str(record["diagnostic_subtype"])
        source_mode = str(record.get("source_mode") or "unknown")
        is_correct = bool(record["correct"])
        type_display[question_type] = str(record["question_type_display"])

        type_totals[question_type]["total"] += 1
        type_totals[question_type]["correct"] += 1 if is_correct else 0
        subtype_totals[question_type][subtype]["total"] += 1
        subtype_totals[question_type][subtype]["correct"] += 1 if is_correct else 0
        subtype_sources[question_type][subtype][source_mode] += 1

    summary: dict[str, object] = {}
    for question_type in sorted(type_totals):
        type_stats = type_totals[question_type]
        by_subtype = {
            subtype: {
                **_stats_payload(stats["total"], stats["correct"]),
                "source_modes": dict(sorted(subtype_sources[question_type][subtype].items())),
            }
            for subtype, stats in sorted(subtype_totals[question_type].items())
        }
        summary[question_type] = {
            "display": type_display.get(question_type, question_type),
            **_stats_payload(type_stats["total"], type_stats["correct"]),
            "by_subtype": by_subtype,
        }
    return summary


def prediction_record_to_json(record: dict[str, object]) -> str:
    return json.dumps(
        {
            "id": record["id"],
            "category": record["category"],
            "question_type": record["question_type"],
            "diagnostic_subtype": record["diagnostic_subtype"],
            "source_mode": record["source_mode"],
            "source": record["source"],
            "gold_answer": record["gold_answer"],
            "extracted_prediction": record["extracted_prediction"],
            "correct": record["correct"],
            "raw_prediction": record["raw_prediction"],
        },
        ensure_ascii=False,
    )


def write_failed_trace_samples(
    failed_root: Path,
    records: list[dict[str, object]],
    *,
    per_subtype: int,
) -> tuple[dict[str, dict[str, int]], list[dict[str, object]]]:
    failed_by_subtype: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    totals: Counter = Counter()
    correct: Counter = Counter()
    for record in records:
        key = (str(record["question_type"]), str(record["diagnostic_subtype"]))
        totals[key] += 1
        correct[key] += 1 if bool(record["correct"]) else 0
        if not bool(record["correct"]):
            failed_by_subtype[key].append(record)

    if failed_root.exists():
        shutil.rmtree(failed_root)

    written_counts: dict[str, dict[str, int]] = defaultdict(dict)
    printable_samples: list[dict[str, object]] = []
    for question_type, subtype in sorted(totals):
        if correct[(question_type, subtype)] == totals[(question_type, subtype)]:
            continue
        samples = failed_by_subtype.get((question_type, subtype), [])[:per_subtype]
        written_counts[question_type][subtype] = len(samples)
        subtype_dir = failed_root / safe_label(question_type) / safe_label(subtype)
        subtype_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(samples, start=1):
            payload = {
                "id": record["id"],
                "category": record["category"],
                "question_type": record["question_type"],
                "question_type_display": record["question_type_display"],
                "diagnostic_subtype": record["diagnostic_subtype"],
                "source_mode": record["source_mode"],
                "source": record["source"],
                "gold_answer": record["gold_answer"],
                "extracted_prediction": record["extracted_prediction"],
                "correct": record["correct"],
                "prompt": record["prompt"],
                "raw_prediction": record["raw_prediction"],
            }
            write_json(subtype_dir / f"{index:02d}_{record['id']}.json", payload)
            (subtype_dir / f"{index:02d}_{record['id']}.txt").write_text(
                format_failed_trace_sample(record),
                encoding="utf-8",
            )
            printable_samples.append(record)

    return {key: dict(value) for key, value in sorted(written_counts.items())}, printable_samples


def format_failed_trace_sample(record: dict[str, object]) -> str:
    return "\n".join(
        [
            f"ID: {record['id']}",
            f"Type: {record['question_type']}",
            f"Subtype: {record['diagnostic_subtype']}",
            f"Source mode: {record['source_mode']}",
            f"Gold answer: {record['gold_answer']}",
            f"Extracted prediction: {record['extracted_prediction']}",
            "",
            "Prompt:",
            str(record["prompt"]),
            "",
            "Model generation:",
            str(record["raw_prediction"]),
            "",
        ]
    )


def print_failed_trace_samples(samples: list[dict[str, object]]) -> None:
    if not samples:
        return
    print("\nFAILED TRACE SAMPLES")
    for record in samples:
        print("\n" + "=" * 80)
        print(format_failed_trace_sample(record), end="")


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
    skipped_eval_ineligible = 0
    if args.split_csv:
        split_assignments = load_split_assignments(args.split_csv)
        eval_split_names = args.eval_splits or ["eval"]
        selected_ids = select_ids_for_splits(split_assignments, eval_split_names)
        eval_examples = [example for example in examples if example.id in selected_ids]
        before_filter = len(eval_examples)
        eval_examples = [
            example
            for example in eval_examples
            if is_eval_eligible_metadata(metadata_by_id.get(example.id, {}))
        ]
        skipped_eval_ineligible = before_filter - len(eval_examples)
        output_split_name = "_".join(eval_split_names)
        if not eval_examples:
            raise SystemExit(
                f"No eval-eligible examples matched splits {eval_split_names!r} in {args.split_csv!r}. "
                f"Skipped eval-ineligible rows: {skipped_eval_ineligible}"
            )
    else:
        train_examples, val_examples = stratified_split(
            examples,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        eval_examples = val_examples if args.split == "val" else train_examples
        before_filter = len(eval_examples)
        eval_examples = [
            example
            for example in eval_examples
            if is_eval_eligible_metadata(metadata_by_id.get(example.id, {}))
        ]
        skipped_eval_ineligible = before_filter - len(eval_examples)
    if args.max_eval_samples is not None:
        eval_examples = eval_examples[: args.max_eval_samples]

    if args.backend == "vllm":
        raw_predictions = generate_with_vllm(args, model_path, eval_examples)
    else:
        raw_predictions = generate_with_transformers(args, model_path, eval_examples)

    results = [
        score_prediction(
            example_id=example.id,
            category=row_category(example, metadata_by_id.get(example.id, {})),
            raw_prediction=raw_prediction,
            gold_answer=example.answer or "",
        )
        for example, raw_prediction in zip(eval_examples, raw_predictions)
    ]
    evaluation_records = build_evaluation_records(eval_examples, results, metadata_by_id)
    adapter_root = Path(args.adapter_dir).resolve().parent
    report_dir = Path(args.report_dir).resolve() if args.report_dir else adapter_root
    failed_trace_dir = report_dir / f"{output_split_name}_failed_traces"
    failed_sample_counts, failed_samples = write_failed_trace_samples(
        failed_trace_dir,
        evaluation_records,
        per_subtype=args.failed_traces_per_subtype,
    )

    summary = summarize_results(results)
    summary.update(
        {
            "mode": "full_inference",
            "backend": args.backend,
            "model_path": model_path,
            "adapter_dir": str(Path(args.adapter_dir).resolve()),
            "report_dir": str(report_dir),
            "split": output_split_name,
            "split_csv": str(Path(args.split_csv).resolve()) if args.split_csv else None,
            "eval_splits": args.eval_splits or (["eval"] if args.split_csv else None),
            "split_counts": summarize_split_assignments(split_assignments) if split_assignments else None,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_num_seqs": args.max_num_seqs if args.backend == "vllm" else None,
            "max_model_len": args.max_model_len if args.backend == "vllm" else None,
            "metric": "reference/evaluation.py-compatible",
            "numeric_relative_tolerance": 1e-2,
            "numeric_absolute_tolerance": 1e-5,
            "by_type": summarize_by_type(evaluation_records),
            "failed_trace_dir": str(failed_trace_dir),
            "failed_trace_sample_counts": failed_sample_counts,
            "evaluated_examples": len(results),
            "skipped_eval_ineligible": skipped_eval_ineligible,
        }
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.no_print_failed_traces:
        print_failed_trace_samples(failed_samples)

    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / f"{output_split_name}_summary.json", summary)

    if args.predictions_jsonl:
        prediction_path = Path(args.predictions_jsonl)
    else:
        prediction_path = report_dir / f"{output_split_name}_predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        "\n".join(prediction_record_to_json(record) for record in evaluation_records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
