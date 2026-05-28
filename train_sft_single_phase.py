#!/usr/bin/env python3
"""Standalone single-phase LoRA SFT trainer on Phase 2 data."""
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PHASE2_CSV = ROOT / "data/training_ready_clean/phase2_sft.csv"
PHASE2_SPLIT_CSV = ROOT / "data/training_ready_clean/phase2_splits_80_10_10.csv"
HF_MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
KAGGLE_MODEL_PATH = Path("/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1")
MAX_LORA_RANK = 32
DEFAULT_MAX_SEQ_LEN = 8192
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from nemotron_baseline.data import (
    infer_category,
    load_split_assignments,
    select_ids_for_splits,
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
class Example:
    id: str
    prompt: str
    answer: str
    category: str
    generated_cot: str = ""
    assistant_content: str = ""


def default_model_path() -> str:
    explicit_model_path = os.environ.get("MODEL_PATH") or os.environ.get("BASE_MODEL_PATH")
    if explicit_model_path:
        return explicit_model_path
    if KAGGLE_MODEL_PATH.exists():
        return str(KAGGLE_MODEL_PATH)
    return HF_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fresh LoRA weights on Phase 2 SFT data only.")
    parser.add_argument(
        "--model-path",
        default=default_model_path(),
        help=(
            "Local model path or HF model id. Defaults to the Kaggle mounted model "
            f"when present, otherwise {HF_MODEL_PATH}."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/sft_single_phase_h200")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-6)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="Trainer optimizer. Use adamw_torch if fused AdamW is unavailable.",
    )
    return parser.parse_args()


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = {"id", "prompt", "answer"} - set(fieldnames)
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")

        has_cot = "generated_cot" in fieldnames
        has_assistant = "assistant_content" in fieldnames
        has_category = "category" in fieldnames
        has_label = "label" in fieldnames

        for line_number, row in enumerate(reader, start=2):
            row_id = (row.get("id") or "").strip()
            prompt = row.get("prompt") or ""
            answer = row.get("answer") or ""
            if not row_id:
                raise SystemExit(f"{path} has an empty id at line {line_number}")
            if not prompt.strip():
                raise SystemExit(f"{path} has an empty prompt for id={row_id}")
            if not answer.strip():
                raise SystemExit(f"{path} has an empty answer for id={row_id}")

            if has_category and row.get("category"):
                category = row["category"]
            elif has_label and row.get("label"):
                category = row["label"]
            else:
                category = infer_category(prompt)

            examples.append(
                Example(
                    id=row_id,
                    prompt=prompt,
                    answer=answer,
                    category=category,
                    generated_cot=normalize_generated_cot(row.get("generated_cot") if has_cot else None),
                    assistant_content=(row.get("assistant_content", "").strip() if has_assistant else ""),
                )
            )
    if not examples:
        raise SystemExit(f"{path} has no rows")
    return examples


def phase2_sft_train_examples() -> tuple[list[Example], list[Example], dict[str, str]]:
    examples = load_examples(PHASE2_CSV)
    split_assignments = load_split_assignments(str(PHASE2_SPLIT_CSV))
    train_ids = select_ids_for_splits(split_assignments, ["sft_train"])
    train = [example for example in examples if example.id in train_ids]
    holdout = [example for example in examples if example.id not in train_ids]
    if not train:
        raise SystemExit("No Phase 2 rows matched split sft_train")
    return train, holdout, split_assignments


def build_dataset(dataset_cls, tokenizer, examples: list[Example]):
    rows = []
    for example in examples:
        if example.assistant_content:
            text = apply_chat_template(
                tokenizer,
                [
                    {"role": "user", "content": build_user_message(example.prompt, append_answer_instruction=True)},
                    {"role": "assistant", "content": example.assistant_content},
                ],
                add_generation_prompt=False,
            )
        else:
            text = build_training_text(
                tokenizer,
                example.prompt,
                example.answer,
                generated_cot=example.generated_cot,
                append_answer_instruction=True,
                answer_style="boxed",
            )
        rows.append({"id": example.id, "text": text})
    return dataset_cls.from_list(rows)


def make_sft_config(sft_config_cls, **kwargs):
    signature = inspect.signature(sft_config_cls.__init__)
    accepts_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_extra_kwargs:
        return sft_config_cls(**kwargs)

    supported = set(signature.parameters) - {"self"}
    skipped = sorted(set(kwargs) - supported)
    if skipped:
        print(f"SFTConfig does not support {skipped}; skipping them")
    return sft_config_cls(**{key: value for key, value in kwargs.items() if key in supported})


def make_min_lr_callback(trainer_callback_cls, min_learning_rate: float):
    class MinLearningRateCallback(trainer_callback_cls):
        def _clamp(self, optimizer) -> None:
            if optimizer is None or min_learning_rate <= 0:
                return
            for group in optimizer.param_groups:
                if group.get("lr", 0.0) < min_learning_rate:
                    group["lr"] = min_learning_rate

        def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
            self._clamp(optimizer)
            return control

        def on_step_end(self, args, state, control, optimizer=None, **kwargs):
            self._clamp(optimizer)
            return control

    return MinLearningRateCallback()


def source_counts(examples: list[Example]) -> dict[str, int]:
    return {"phase2": len(examples)} if examples else {}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def zip_adapter(adapter_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_handle:
        for file_path in sorted(adapter_dir.iterdir()):
            if file_path.is_file():
                zip_handle.write(file_path, file_path.name)


def clear_memory() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"Skipping CUDA cache cleanup: {exc}")


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


def print_summary(args: argparse.Namespace, train_examples: list[Example], holdout_examples: list[Example], split_assignments: dict[str, str]) -> None:
    summary = {
        "mode": "single_phase",
        "output_dir": str(Path(args.output_dir).resolve()),
        "phase2_csv": str(PHASE2_CSV),
        "phase2_split_csv": str(PHASE2_SPLIT_CSV),
        "phase1_train_rows": 0,
        "phase2_train_rows": len(train_examples),
        "total_train_rows": len(train_examples),
        "holdout_rows": len(holdout_examples),
        "train_source_counts": source_counts(train_examples),
        "holdout_source_counts": source_counts(holdout_examples),
        "train_category_counts": summarize_categories(train_examples),
        "holdout_category_counts": summarize_categories(holdout_examples),
        "split_counts": summarize_split_assignments(split_assignments),
        "max_seq_len": args.max_seq_len,
        "lora_rank": MAX_LORA_RANK,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "min_learning_rate": args.min_learning_rate,
        "gradient_checkpointing": args.gradient_checkpointing,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "optim": args.optim,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    train_examples, holdout_examples, split_assignments = phase2_sft_train_examples()
    if args.validate_only:
        print_summary(args, train_examples, holdout_examples, split_assignments)
        return

    disable_transformers_vision_imports()
    check_nemotron_runtime_dependencies()

    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore
    import torch  # type: ignore

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = build_dataset(Dataset, tokenizer, train_examples)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    print(f"Model device map: {getattr(model, 'hf_device_map', None)}")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=MAX_LORA_RANK,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()

    trainer_args = make_sft_config(
        SFTConfig,
        output_dir=str(output_dir / "trainer_state"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1.0,
        learning_rate=args.learning_rate,
        logging_steps=10,
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,
        optim=args.optim,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
        group_by_length=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=trainer_args,
        callbacks=[make_min_lr_callback(TrainerCallback, args.min_learning_rate)],
    )
    print(f"Starting single_phase: rows={len(train_examples)}, learning_rate={args.learning_rate}")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    print(f"Single-phase adapter saved to: {adapter_dir}")
    trainer.model = None
    del trainer
    del dataset
    clear_memory()

    submission_path = output_dir / "submission.zip"
    zip_adapter(adapter_dir, submission_path)
    write_json(
        output_dir / "run_config.json",
        {
            "mode": "single_phase",
            "model_path": args.model_path,
            "adapter_dir": str(adapter_dir.resolve()),
            "submission_zip": str(submission_path.resolve()),
            "phase1_train_rows": 0,
            "phase2_train_rows": len(train_examples),
            "total_train_rows": len(train_examples),
            "holdout_rows": len(holdout_examples),
            "max_seq_len": args.max_seq_len,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "min_learning_rate": args.min_learning_rate,
            "lora_rank": MAX_LORA_RANK,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora_dropout": args.lora_dropout,
            "optim": args.optim,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        },
    )
    write_json(
        output_dir / "dataset_summary.json",
        {
            "train_source_counts": source_counts(train_examples),
            "holdout_source_counts": source_counts(holdout_examples),
            "train_category_counts": summarize_categories(train_examples),
            "holdout_category_counts": summarize_categories(holdout_examples),
            "split_counts": summarize_split_assignments(split_assignments),
        },
    )
    print(f"Phase 2 rows: {len(train_examples)}")
    print(f"Final adapter saved to: {adapter_dir}")
    print(f"Submission zip: {submission_path}")


if __name__ == "__main__":
    main()
