#!/usr/bin/env python3
"""Minimal portable LoRA SFT trainer for Colab, Lightning AI, or local GPUs."""
from __future__ import annotations

import argparse
import csv
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

PHASE1_CSV = ROOT / "data/training_ready_clean/phase1_train.csv"
PHASE2_CSV = ROOT / "data/training_ready_clean/phase2_sft.csv"
PHASE2_SPLIT_CSV = ROOT / "data/training_ready_clean/phase2_splits_80_10_10.csv"
DEFAULT_MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_PATH = os.environ.get("MODEL_PATH") or os.environ.get("BASE_MODEL_PATH") or DEFAULT_MODEL_PATH
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
MAX_LORA_RANK = 32
MAX_SEQ_LEN = 8192
GRADIENT_CHECKPOINTING = False

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
    source: str
    append_answer_instruction: bool
    generated_cot: str = ""
    assistant_content: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean Nemotron LoRA SFT dataset.")
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help=f"Local model path or HF model id. Defaults to {DEFAULT_MODEL_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: outputs/sft_two_stage_h200 or outputs/sft_phase1_h200.",
    )
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    return parser.parse_args()


def load_examples(path: Path, *, source: str, append_answer_instruction: bool) -> list[Example]:
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
                    source=source,
                    append_answer_instruction=append_answer_instruction,
                    generated_cot=normalize_generated_cot(row.get("generated_cot") if has_cot else None),
                    assistant_content=(row.get("assistant_content", "").strip() if has_assistant else ""),
                )
            )
    if not examples:
        raise SystemExit(f"{path} has no rows")
    return examples


def phase2_sft_train_examples() -> tuple[list[Example], list[Example], dict[str, str]]:
    examples = load_examples(PHASE2_CSV, source="phase2", append_answer_instruction=True)
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
                    {
                        "role": "user",
                        "content": build_user_message(
                            example.prompt,
                            append_answer_instruction=example.append_answer_instruction,
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
                example.answer,
                generated_cot=example.generated_cot,
                append_answer_instruction=example.append_answer_instruction,
                answer_style="boxed",
            )
        rows.append({"id": example.id, "text": text})
    return dataset_cls.from_list(rows)


def source_counts(examples: list[Example]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.source] = counts.get(example.source, 0) + 1
    return dict(sorted(counts.items()))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def zip_adapter(adapter_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_handle:
        for file_path in sorted(adapter_dir.iterdir()):
            if file_path.is_file():
                zip_handle.write(file_path, file_path.name)


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


def print_summary(
    *,
    mode: str,
    output_dir: Path,
    phase1_examples: list[Example],
    phase2_train_examples: list[Example],
    holdout_examples: list[Example],
    split_assignments: dict[str, str] | None,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    train_examples = phase1_examples + phase2_train_examples
    summary = {
        "mode": mode,
        "output_dir": str(output_dir.resolve()),
        "phase1_csv": str(PHASE1_CSV),
        "phase2_csv": str(PHASE2_CSV),
        "phase2_split_csv": str(PHASE2_SPLIT_CSV),
        "phase1_train_rows": len(phase1_examples),
        "phase2_train_rows": len(phase2_train_examples),
        "total_train_rows": len(train_examples),
        "holdout_rows": len(holdout_examples),
        "train_source_counts": source_counts(train_examples),
        "holdout_source_counts": source_counts(holdout_examples),
        "train_category_counts": summarize_categories(train_examples),
        "holdout_category_counts": summarize_categories(holdout_examples),
        "split_counts": summarize_split_assignments(split_assignments) if split_assignments else None,
        "max_seq_len": MAX_SEQ_LEN,
        "lora_rank": MAX_LORA_RANK,
        "stage_learning_rates": {
            "phase1": 5e-5,
            "phase2": None if mode == "phase1_only" else 2e-5,
        },
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": per_device_train_batch_size * gradient_accumulation_steps,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def train_stage(
    *,
    stage_name: str,
    model,
    tokenizer,
    examples: list[Example],
    learning_rate: float,
    stage_output_dir: Path,
    dataset_cls,
    sft_config_cls,
    sft_trainer_cls,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
):
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(dataset_cls, tokenizer, examples)
    trainer_args = make_sft_config(
        sft_config_cls,
        output_dir=str(stage_output_dir / "trainer_state"),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1.0,
        learning_rate=learning_rate,
        logging_steps=10,
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=False,
        group_by_length=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
    )
    trainer = sft_trainer_cls(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=trainer_args,
    )
    print(f"Starting {stage_name}: rows={len(examples)}, learning_rate={learning_rate}")
    trainer.train()
    stage_adapter_dir = stage_output_dir / "adapter"
    trainer.model.save_pretrained(stage_adapter_dir)
    print(f"{stage_name} adapter saved to: {stage_adapter_dir}")
    return trainer.model, stage_adapter_dir


def main() -> None:
    args = parse_args()
    mode = "phase1_only" if args.phase1_only else "phase1_then_phase2"
    output_dir = Path(args.output_dir or ("outputs/sft_phase1_h200" if args.phase1_only else "outputs/sft_two_stage_h200"))

    phase2_train_examples: list[Example] = []
    holdout_examples: list[Example] = []
    split_assignments = None

    phase1_examples = load_examples(PHASE1_CSV, source="phase1", append_answer_instruction=False)

    if not args.phase1_only:
        phase2_train, phase2_holdout, split_assignments = phase2_sft_train_examples()
        phase2_train_examples.extend(phase2_train)
        holdout_examples.extend(phase2_holdout)

    if args.validate_only:
        print_summary(
            mode=mode,
            output_dir=output_dir,
            phase1_examples=phase1_examples,
            phase2_train_examples=phase2_train_examples,
            holdout_examples=holdout_examples,
            split_assignments=split_assignments,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        return

    disable_transformers_vision_imports()
    check_nemotron_runtime_dependencies()

    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore
    import torch  # type: ignore

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    if GRADIENT_CHECKPOINTING:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()

    phase1_stage_dir = output_dir if args.phase1_only else output_dir / "phase1"
    model, phase1_adapter_dir = train_stage(
        stage_name="phase1",
        model=model,
        tokenizer=tokenizer,
        examples=phase1_examples,
        learning_rate=5e-5,
        stage_output_dir=phase1_stage_dir,
        dataset_cls=Dataset,
        sft_config_cls=SFTConfig,
        sft_trainer_cls=SFTTrainer,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    final_adapter_dir = phase1_adapter_dir
    if not args.phase1_only:
        model, final_adapter_dir = train_stage(
            stage_name="phase2",
            model=model,
            tokenizer=tokenizer,
            examples=phase2_train_examples,
            learning_rate=2e-5,
            stage_output_dir=output_dir,
            dataset_cls=Dataset,
            sft_config_cls=SFTConfig,
            sft_trainer_cls=SFTTrainer,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

    submission_path = output_dir / "submission.zip"
    zip_adapter(final_adapter_dir, submission_path)

    train_examples = phase1_examples + phase2_train_examples
    write_json(
        output_dir / "run_config.json",
        {
            "mode": mode,
            "model_path": args.model_path,
            "phase1_adapter_dir": str(phase1_adapter_dir.resolve()),
            "adapter_dir": str(final_adapter_dir.resolve()),
            "submission_zip": str(submission_path.resolve()),
            "phase1_train_rows": len(phase1_examples),
            "phase2_train_rows": len(phase2_train_examples),
            "total_train_rows": len(train_examples),
            "holdout_rows": len(holdout_examples),
            "max_seq_len": MAX_SEQ_LEN,
            "stage_learning_rates": {
                "phase1": 5e-5,
                "phase2": None if args.phase1_only else 2e-5,
            },
            "lora_rank": MAX_LORA_RANK,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
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
            "split_counts": summarize_split_assignments(split_assignments) if split_assignments else None,
        },
    )

    print(f"Phase 1 rows: {len(phase1_examples)}")
    print(f"Phase 2 rows: {len(phase2_train_examples)}")
    print(f"Final adapter saved to: {final_adapter_dir}")
    print(f"Submission zip: {submission_path}")


if __name__ == "__main__":
    main()
