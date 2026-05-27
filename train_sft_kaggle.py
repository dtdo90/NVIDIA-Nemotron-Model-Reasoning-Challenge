#!/usr/bin/env python3
"""Kaggle-oriented LoRA SFT trainer for the Nemotron competition setup.

The script can use a local --model-path, but its defaults and Triton fixes are
designed for the Kaggle model/runtime environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

COMPETITION_MAX_LORA_RANK = 32
COMPETITION_MAX_MODEL_LEN = 8192
DEFAULT_MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_PATH = os.environ.get("MODEL_PATH") or os.environ.get("BASE_MODEL_PATH") or DEFAULT_MODEL_PATH
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from nemotron_baseline.data import (
    infer_category,
    load_split_assignments,
    select_ids_for_splits,
    stratified_split,
    summarize_categories,
    summarize_split_assignments,
)
from nemotron_baseline.prompts import (
    apply_chat_template,
    build_training_text,
    build_user_message,
    normalize_generated_cot,
)
from nemotron_baseline.runtime import (
    check_nemotron_runtime_dependencies,
    disable_transformers_vision_imports,
)


@dataclass(frozen=True)
class TrainingExample:
    id: str
    prompt: str
    answer: str
    category: str
    generated_cot: str | None = None
    label: str | None = None
    assistant_content: str | None = None


def load_config_defaults(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    with Path(config_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit("Config file must contain a JSON object.")
    return payload


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args()
    defaults = load_config_defaults(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description="Train a Nemotron LoRA baseline with answer-only or CoT supervision."
    )
    parser.add_argument("--config", default=bootstrap_args.config)
    parser.add_argument("--train-csv", default=defaults.get("train_csv", "data/train.csv"))
    parser.add_argument("--split-csv", default=defaults.get("split_csv"))
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=defaults.get("train_splits"),
        help="Named splits from --split-csv to use for SFT training.",
    )
    parser.add_argument(
        "--output-dir",
        default=defaults.get("output_dir", "outputs/baseline_answer_only"),
    )
    parser.add_argument(
        "--init-adapter-dir",
        default=defaults.get("init_adapter_dir"),
        help="Optional LoRA adapter directory to continue training from.",
    )
    parser.add_argument("--model-path", default=defaults.get("model_path") or MODEL_PATH)
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
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=defaults.get("max_train_samples"),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=defaults.get("max_seq_len", 2048),
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=defaults.get("num_epochs", 1.0),
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=defaults.get("per_device_train_batch_size", 1),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.get("gradient_accumulation_steps", 8),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.get("learning_rate", 2e-4),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=defaults.get("lora_rank", 32),
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=defaults.get("lora_alpha", 32),
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=defaults.get("lora_dropout", 0.05),
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=defaults.get("logging_steps", 10),
    )
    parser.add_argument(
        "--save-tokenizer",
        action="store_true",
        default=bool(defaults.get("save_tokenizer", False)),
    )
    parser.add_argument(
        "--supervision-format",
        choices=("auto", "answer_only", "cot"),
        default=defaults.get("supervision_format", "auto"),
        help=(
            "How to build assistant targets. 'auto' uses CoT when the input CSV has "
            "a generated_cot column, otherwise answer-only."
        ),
    )
    parser.add_argument(
        "--cot-column",
        default=defaults.get("cot_column", "generated_cot"),
    )
    parser.add_argument(
        "--append-answer-instruction",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("append_answer_instruction", True)),
        help=(
            "Append the boxed final-answer instruction to user prompts. "
            "Use --no-append-answer-instruction for Phase 1 knowledge injection."
        ),
    )
    parser.add_argument(
        "--answer-style",
        choices=("boxed", "plain"),
        default=defaults.get("answer_style", "boxed"),
        help=(
            "Final target style. 'boxed' emits \\boxed{answer}; 'plain' emits "
            "'Answer: answer' for Phase 1 knowledge injection."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=bool(defaults.get("validate_only", False)),
        help="Validate the configured CSV/split wiring and exit before loading model dependencies.",
    )
    parser.add_argument("--disable-kaggle-triton-fixes", action="store_true")
    return parser.parse_args(remaining)


def require_training_dependencies():
    try:
        import torch  # type: ignore
        import torch.nn.functional as torch_f  # type: ignore
        from datasets import Dataset  # type: ignore
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install transformers, datasets, trl, peft, "
            "and torch in the training environment."
        ) from exc

    return {
        "torch": torch,
        "torch_f": torch_f,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def apply_kaggle_triton_fixes(torch, torch_f) -> None:
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

    src = (
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/"
        "triton/backends/nvidia/bin/ptxas-blackwell"
    )
    dst = "/tmp/ptxas-blackwell"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        import triton.backends.nvidia as nv_backend  # type: ignore

        src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
        dst_bin = "/tmp/triton_nvidia_bin"
        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
        for name in os.listdir(dst_bin):
            file_path = os.path.join(dst_bin, name)
            if os.path.isfile(file_path):
                os.chmod(
                    file_path,
                    os.stat(file_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
                )

        nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
        os.environ["TRITON_PTXAS_PATH"] = dst
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = dst

        import triton.backends.nvidia.compiler as nv_compiler  # type: ignore

        nv_compiler.get_ptxas_version = lambda arch: "12.0"


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


def load_training_examples(csv_path: str, *, cot_column: str) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required_columns = {"id", "prompt", "answer"}
        missing_columns = required_columns - set(fieldnames)
        if missing_columns:
            raise SystemExit(
                f"{csv_path!r} is missing required columns: {sorted(missing_columns)}"
            )
        has_cot_column = cot_column in fieldnames
        has_label_column = "label" in fieldnames
        has_category_column = "category" in fieldnames
        has_assistant_content_column = "assistant_content" in fieldnames
        for row_index, row in enumerate(reader, start=2):
            row_id = (row.get("id") or "").strip()
            prompt = row.get("prompt") or ""
            answer = row.get("answer") or ""
            if not row_id:
                raise SystemExit(f"{csv_path!r} has an empty id at CSV line {row_index}.")
            if not prompt.strip():
                raise SystemExit(f"{csv_path!r} has an empty prompt for id={row_id!r}.")
            if not answer.strip():
                raise SystemExit(f"{csv_path!r} has an empty answer for id={row_id!r}.")
            generated_cot = row.get(cot_column) if has_cot_column else None
            assistant_content = (
                row.get("assistant_content", "").strip()
                if has_assistant_content_column
                else ""
            )
            if has_category_column and row.get("category"):
                category = row["category"]
            elif has_label_column and row.get("label"):
                category = row["label"]
            else:
                category = infer_category(row["prompt"])
            examples.append(
                TrainingExample(
                    id=row_id,
                    prompt=prompt,
                    answer=answer,
                    category=category,
                    generated_cot=normalize_generated_cot(generated_cot),
                    label=row.get("label") if has_label_column else None,
                    assistant_content=assistant_content or None,
                )
            )
    if not examples:
        raise SystemExit(f"{csv_path!r} contains no training rows.")
    return examples


def resolve_supervision_format(
    args: argparse.Namespace,
    examples: list[TrainingExample],
) -> str:
    if args.supervision_format != "auto":
        return args.supervision_format
    if any(example.generated_cot for example in examples):
        return "cot"
    return "answer_only"


def build_dataset(
    dataset_cls,
    tokenizer,
    examples,
    supervision_format: str,
    *,
    append_answer_instruction: bool,
    answer_style: str,
):
    rows = []
    for example in examples:
        if example.assistant_content:
            text = apply_chat_template(
                tokenizer,
                [
                    {
                        "role": "user",
                        "content": build_user_message(
                            example.prompt,
                            append_answer_instruction=append_answer_instruction,
                        ),
                    },
                    {"role": "assistant", "content": example.assistant_content},
                ],
                add_generation_prompt=False,
            )
        else:
            text = build_training_text(
                tokenizer,
                example.prompt,
                example.answer or "",
                generated_cot=example.generated_cot if supervision_format == "cot" else None,
                append_answer_instruction=append_answer_instruction,
                answer_style=answer_style,
            )
        rows.append(
            {
                "id": example.id,
                "category": example.category,
                "label": example.label,
                "text": text,
            }
        )
    return dataset_cls.from_list(rows)


def filter_examples_by_ids(
    examples: list[TrainingExample],
    selected_ids: set[str],
) -> list[TrainingExample]:
    return [example for example in examples if example.id in selected_ids]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_duplicate_ids(examples: list[TrainingExample]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for example in examples:
        if example.id in seen:
            duplicates += 1
        else:
            seen.add(example.id)
    return duplicates


def build_preflight_summary(
    args: argparse.Namespace,
    *,
    examples: list[TrainingExample],
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample],
    supervision_format: str,
    split_assignments: dict[str, str] | None,
    selected_ids: set[str] | None,
) -> dict[str, object]:
    example_ids = {example.id for example in examples}
    selected_missing_ids = sorted((selected_ids or set()) - example_ids)
    generated_cot_count = sum(1 for example in examples if example.generated_cot)
    assistant_content_count = sum(1 for example in examples if example.assistant_content)
    generated_cot_target_count = sum(
        1 for example in examples if example.generated_cot and not example.assistant_content
    )
    answer_only_count = sum(
        1 for example in examples if not example.generated_cot and not example.assistant_content
    )
    warnings: list[str] = []
    duplicate_count = count_duplicate_ids(examples)
    if duplicate_count:
        warnings.append(
            f"Found {duplicate_count} duplicate row ids. This is allowed for weighting, "
            "but it can make split accounting harder to audit."
        )
    if args.supervision_format == "cot" and not (generated_cot_count or assistant_content_count):
        warnings.append(
            "supervision_format is 'cot', but no generated_cot or assistant_content rows were found."
        )
    if selected_missing_ids:
        warnings.append(
            f"{len(selected_missing_ids)} ids selected by the split file are absent from the training CSV."
        )
    return {
        "train_csv": str(Path(args.train_csv).resolve()),
        "split_csv": str(Path(args.split_csv).resolve()) if args.split_csv else None,
        "train_splits": args.train_splits or (["sft_train"] if args.split_csv else None),
        "supervision_format": supervision_format,
        "cot_column": args.cot_column,
        "append_answer_instruction": args.append_answer_instruction,
        "answer_style": args.answer_style,
        "max_seq_len": args.max_seq_len,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "total_rows": len(examples),
        "train_rows": len(train_examples),
        "val_rows": len(val_examples),
        "generated_cot_rows": generated_cot_count,
        "assistant_content_rows": assistant_content_count,
        "generated_cot_target_rows": generated_cot_target_count,
        "answer_only_rows": answer_only_count,
        "duplicate_id_rows": duplicate_count,
        "train_category_counts": summarize_categories(train_examples),
        "val_category_counts": summarize_categories(val_examples),
        "split_counts": summarize_split_assignments(split_assignments) if split_assignments else None,
        "selected_split_ids_missing_from_csv": len(selected_missing_ids),
        "warnings": warnings,
    }


def ensure_trainable_adapter(model) -> None:
    for name, parameter in model.named_parameters():
        if "lora_" in name or "adapter" in name:
            parameter.requires_grad = True
    model.train()


def print_trainable_parameters(model) -> None:
    trainable_params = 0
    total_params = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total_params += count
        if parameter.requires_grad:
            trainable_params += count
    percent = 100 * trainable_params / total_params if total_params else 0
    print(
        "Trainable parameters: "
        f"{trainable_params:,} / {total_params:,} ({percent:.4f}%)"
    )


def main() -> None:
    args = parse_args()
    if args.lora_rank > COMPETITION_MAX_LORA_RANK:
        raise SystemExit(
            f"LoRA rank {args.lora_rank} exceeds competition max_lora_rank "
            f"{COMPETITION_MAX_LORA_RANK}."
        )
    if args.max_seq_len > COMPETITION_MAX_MODEL_LEN:
        raise SystemExit(
            f"max_seq_len {args.max_seq_len} exceeds competition max_model_len "
            f"{COMPETITION_MAX_MODEL_LEN}."
        )
    examples = load_training_examples(args.train_csv, cot_column=args.cot_column)
    supervision_format = resolve_supervision_format(args, examples)
    split_assignments = None
    selected_ids = None
    if args.split_csv:
        split_assignments = load_split_assignments(args.split_csv)
        train_split_names = args.train_splits or ["sft_train"]
        selected_ids = select_ids_for_splits(split_assignments, train_split_names)
        train_examples = filter_examples_by_ids(examples, selected_ids)
        val_examples = [example for example in examples if example.id not in selected_ids]
        if not train_examples:
            raise SystemExit(
                f"No SFT examples matched splits {train_split_names!r} in {args.split_csv!r}."
            )
    else:
        train_examples, val_examples = stratified_split(
            examples,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )

    if args.max_train_samples is not None:
        train_examples = train_examples[: args.max_train_samples]

    if args.validate_only:
        summary = build_preflight_summary(
            args,
            examples=examples,
            train_examples=train_examples,
            val_examples=val_examples,
            supervision_format=supervision_format,
            split_assignments=split_assignments,
            selected_ids=selected_ids,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    disable_transformers_vision_imports()
    check_nemotron_runtime_dependencies()
    deps = require_training_dependencies()
    try:
        import kagglehub  # type: ignore
    except ImportError:
        kagglehub = None

    if not args.disable_kaggle_triton_fixes:
        apply_kaggle_triton_fixes(deps["torch"], deps["torch_f"])

    model_path = resolve_model_path(args, kagglehub)
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = deps["AutoTokenizer"].from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset(
        deps["Dataset"],
        tokenizer,
        train_examples,
        supervision_format,
        append_answer_instruction=args.append_answer_instruction,
        answer_style=args.answer_style,
    )

    model = deps["AutoModelForCausalLM"].from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=deps["torch"].bfloat16,
    )
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    disable_nemotron_fast_path()

    if args.init_adapter_dir:
        try:
            model = deps["PeftModel"].from_pretrained(
                model,
                args.init_adapter_dir,
                is_trainable=True,
            )
        except TypeError:
            model = deps["PeftModel"].from_pretrained(model, args.init_adapter_dir)
            ensure_trainable_adapter(model)
    else:
        lora_config = deps["LoraConfig"](
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=deps["TaskType"].CAUSAL_LM,
        )
        model = deps["get_peft_model"](model, lora_config)
    print_trainable_parameters(model)
    model.train()

    trainer_config = deps["SFTConfig"](
        output_dir=str(output_dir / "trainer_state"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        bf16=True,
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )

    trainer = deps["SFTTrainer"](
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=trainer_config,
    )
    trainer.train()

    trainer.model.save_pretrained(adapter_dir)
    if args.save_tokenizer:
        tokenizer.save_pretrained(adapter_dir)

    submission_path = output_dir / "submission.zip"
    with zipfile.ZipFile(submission_path, "w", zipfile.ZIP_DEFLATED) as zip_handle:
        for file_path in sorted(adapter_dir.iterdir()):
            zip_handle.write(file_path, file_path.name)

    config_payload = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "split_csv": str(Path(args.split_csv).resolve()) if args.split_csv else None,
        "train_splits": args.train_splits or (["sft_train"] if args.split_csv else None),
        "output_dir": str(output_dir.resolve()),
        "adapter_dir": str(adapter_dir.resolve()),
        "submission_zip": str(submission_path.resolve()),
        "init_adapter_dir": str(Path(args.init_adapter_dir).resolve()) if args.init_adapter_dir else None,
        "model_path": model_path,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "max_train_samples": args.max_train_samples,
        "max_seq_len": args.max_seq_len,
        "num_epochs": args.num_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "save_tokenizer": args.save_tokenizer,
        "supervision_format": supervision_format,
        "cot_column": args.cot_column,
        "append_answer_instruction": args.append_answer_instruction,
        "answer_style": args.answer_style,
    }
    dataset_summary = {
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_category_counts": summarize_categories(train_examples),
        "val_category_counts": summarize_categories(val_examples),
        "cot_examples": sum(1 for example in examples if example.generated_cot),
        "split_counts": summarize_split_assignments(split_assignments) if split_assignments else None,
    }

    write_json(output_dir / "run_config.json", config_payload)
    write_json(output_dir / "dataset_summary.json", dataset_summary)

    print(f"Model path: {model_path}")
    print(f"Training rows: {len(train_examples)}")
    print(f"Validation rows held out: {len(val_examples)}")
    print(f"Supervision format: {supervision_format}")
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Submission zip: {submission_path}")


if __name__ == "__main__":
    main()
