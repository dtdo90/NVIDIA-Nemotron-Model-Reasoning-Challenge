#!/usr/bin/env python3
"""Standalone single-phase LoRA SFT trainer on the clean single-phase data mix."""
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

SINGLE_PHASE_CSV = ROOT / "data/single_phase_training_clean/single_phase_sft.csv"
SINGLE_PHASE_SPLIT_CSV = ROOT / "data/single_phase_training_clean/single_phase_splits_80_10_10.csv"
HF_MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
KAGGLE_MODEL_PATH = Path("/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1")
MAX_LORA_RANK = 32
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]
DEFAULT_MAX_SEQ_LEN = 8192
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from nemotron_baseline.data import (
    infer_category,
    summarize_categories,
)
from nemotron_baseline.prompts import (
    build_assistant_trace_content,
    build_competition_prompt,
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
    source_mode: str = "unknown"
    append_answer_instruction: bool = True


def parse_bool(value: str | None, *, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "n", "off"}


def default_model_path() -> str:
    explicit_model_path = os.environ.get("MODEL_PATH") or os.environ.get("BASE_MODEL_PATH")
    if explicit_model_path:
        return explicit_model_path
    if KAGGLE_MODEL_PATH.exists():
        return str(KAGGLE_MODEL_PATH)
    return HF_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fresh LoRA weights on clean single-phase SFT data.")
    parser.add_argument(
        "--model-path",
        default=default_model_path(),
        help=(
            "Local model path or HF model id. Defaults to the Kaggle mounted model "
            f"when present, otherwise {HF_MODEL_PATH}."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/sft_single_phase")
    parser.add_argument("--train-csv", default=str(SINGLE_PHASE_CSV))
    parser.add_argument("--split-csv", default=str(SINGLE_PHASE_SPLIT_CSV))
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["sft_train"],
        help="Split names to use for SFT. Defaults to the 80% sft_train bucket.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Ignore --split-csv and train on every row in --train-csv.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-6)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        help="Trainer optimizer.",
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
                    generated_cot=((row.get("generated_cot") or "").strip() if has_cot else ""),
                    assistant_content=(row.get("assistant_content", "").strip() if has_assistant else ""),
                    source_mode=(row.get("source_mode") or "unknown").strip() or "unknown",
                    append_answer_instruction=parse_bool(
                        row.get("append_answer_instruction"),
                        default=True,
                    ),
                )
            )
    if not examples:
        raise SystemExit(f"{path} has no rows")
    return examples


def load_split_assignments(path: str | Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {"id", "split"} - fieldnames
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            row_id = (row.get("id") or "").strip()
            split = (row.get("split") or "").strip()
            if not row_id or not split:
                raise SystemExit(f"{path} has an invalid row at line {line_number}")
            if row_id in assignments:
                raise SystemExit(f"{path} has duplicate id={row_id}")
            assignments[row_id] = split
    return assignments


def validate_split_assignments(examples: list[Example], assignments: dict[str, str], split_csv: str | Path) -> None:
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for example in examples:
        if example.id in seen:
            duplicate_ids.append(example.id)
        seen.add(example.id)
    if duplicate_ids:
        raise SystemExit(f"{split_csv} cannot be used because train data has duplicate ids: {duplicate_ids[:5]}")
    example_id_set = seen
    assignment_ids = set(assignments)
    missing = sorted(example_id_set - assignment_ids)
    extra = sorted(assignment_ids - example_id_set)
    if missing or extra:
        raise SystemExit(
            f"{split_csv} does not match --train-csv ids. "
            f"Missing={missing[:5]} extra={extra[:5]}"
        )


def select_examples_by_split(
    examples: list[Example],
    assignments: dict[str, str],
    train_splits: list[str],
) -> list[Example]:
    wanted = {split for split in train_splits if split}
    if not wanted:
        raise SystemExit("--train-splits must include at least one split name")
    selected = [example for example in examples if assignments.get(example.id) in wanted]
    if not selected:
        raise SystemExit(f"No examples matched --train-splits {sorted(wanted)}")
    return selected


def split_counts(assignments: dict[str, str] | None) -> dict[str, int] | None:
    if assignments is None:
        return None
    counts: dict[str, int] = {}
    for split in assignments.values():
        counts[split] = counts.get(split, 0) + 1
    return dict(sorted(counts.items()))


def single_phase_train_examples(
    train_csv: str | Path,
    *,
    split_csv: str | Path | None = None,
    train_splits: list[str] | None = None,
    train_all: bool = False,
) -> tuple[list[Example], list[Example], dict[str, str] | None]:
    examples = load_examples(Path(train_csv))
    if not examples:
        raise SystemExit("No single-phase rows were loaded")
    if train_all:
        return examples, examples, None
    if not split_csv:
        raise SystemExit("--split-csv is required unless --train-all is set")
    assignments = load_split_assignments(split_csv)
    validate_split_assignments(examples, assignments, split_csv)
    selected = select_examples_by_split(examples, assignments, train_splits or ["sft_train"])
    return selected, examples, assignments


def assistant_end_token(tokenizer) -> str:
    return tokenizer.eos_token or "<|im_end|>"


def completion_after_generation_prompt(prompt_text: str, assistant_content: str) -> str:
    """Return only the assistant continuation that should be scored.

    Nemotron's chat template opens the generation turn with
    ``<|im_start|>assistant\n<think>\n`` when thinking is enabled. Most of our
    stored CoT traces are self-contained and therefore start with ``<think>``.
    During SFT, we score the continuation after the prompt-opened think tag, so
    the completion must not introduce a second opening tag.
    """

    if prompt_text.rstrip().endswith("<think>") and assistant_content.lstrip().startswith("<think>"):
        content = assistant_content.lstrip()
        content = content[len("<think>") :]
        if content.startswith("\n"):
            content = content[1:]
        return content
    return assistant_content


def tokenize_masked_example(
    tokenizer,
    example: Example,
    *,
    max_seq_len: int,
) -> dict:
    prompt_text = build_competition_prompt(
        tokenizer,
        example.prompt,
        append_answer_instruction=example.append_answer_instruction,
    )
    assistant_content = build_assistant_trace_content(
        example.answer,
        generated_cot=example.generated_cot,
        assistant_content=example.assistant_content,
    )
    end_token = assistant_end_token(tokenizer)
    completion_text = completion_after_generation_prompt(prompt_text, assistant_content)
    if end_token and not completion_text.endswith(end_token):
        completion_text += end_token

    full_text = prompt_text + completion_text
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(full_text, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    if len(input_ids) > max_seq_len:
        raise SystemExit(
            f"id={example.id} has {len(input_ids)} tokens, exceeding max_seq_len={max_seq_len}"
        )
    if len(prompt_ids) >= len(input_ids):
        raise SystemExit(f"id={example.id} has no assistant tokens to score")
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return {
        "id": example.id,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(input_ids) - len(prompt_ids),
        "total_tokens": len(input_ids),
    }


def build_dataset(dataset_cls, tokenizer, examples: list[Example], *, max_seq_len: int):
    rows = []
    for example in examples:
        rows.append(tokenize_masked_example(tokenizer, example, max_seq_len=max_seq_len))
    return dataset_cls.from_list(rows)


class MaskedCausalLMDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]):
        import torch  # type: ignore

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


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
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.source_mode] = counts.get(example.source_mode, 0) + 1
    return dict(sorted(counts.items()))


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


def print_summary(
    args: argparse.Namespace,
    train_examples: list[Example],
    all_examples: list[Example],
    assignments: dict[str, str] | None,
) -> None:
    train_source_counts = source_counts(train_examples)
    summary = {
        "mode": "single_phase",
        "output_dir": str(Path(args.output_dir).resolve()),
        "train_csv": str(Path(args.train_csv).resolve()),
        "split_csv": None if args.train_all else str(Path(args.split_csv).resolve()),
        "train_splits": ["ALL"] if args.train_all else args.train_splits,
        "all_rows": len(all_examples),
        "sft_train_rows": len(train_examples),
        "split_counts": split_counts(assignments),
        "phase1_synthetic_direct_template_rows": train_source_counts.get(
            "phase1_synthetic_direct_template",
            0,
        ),
        "train_source_counts": train_source_counts,
        "train_category_counts": summarize_categories(train_examples),
        "all_category_counts": summarize_categories(all_examples),
        "max_seq_len": args.max_seq_len,
        "lora_rank": MAX_LORA_RANK,
        "lora_target_modules": LORA_TARGET_MODULES,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "min_learning_rate": args.min_learning_rate,
        "gradient_checkpointing": args.gradient_checkpointing,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "optim": args.optim,
        "loss_masking": "assistant_only",
        "prompt_format": "competition_chat_template",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    train_examples, all_examples, assignments = single_phase_train_examples(
        args.train_csv,
        split_csv=args.split_csv,
        train_splits=args.train_splits,
        train_all=args.train_all,
    )
    if args.validate_only:
        print_summary(args, train_examples, all_examples, assignments)
        return

    disable_transformers_vision_imports()
    check_nemotron_runtime_dependencies()

    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments  # type: ignore
    import torch  # type: ignore

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = build_dataset(Dataset, tokenizer, train_examples, max_seq_len=args.max_seq_len)
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
        target_modules=LORA_TARGET_MODULES,
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
        TrainingArguments,
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
        group_by_length=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=MaskedCausalLMDataCollator(tokenizer),
        args=trainer_args,
        callbacks=[make_min_lr_callback(TrainerCallback, args.min_learning_rate)],
    )
    print(
        f"Starting single_phase: rows={len(train_examples)}, "
        f"splits={['ALL'] if args.train_all else args.train_splits}, "
        f"learning_rate={args.learning_rate}"
    )
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
            "train_csv": str(Path(args.train_csv).resolve()),
            "split_csv": None if args.train_all else str(Path(args.split_csv).resolve()),
            "train_splits": ["ALL"] if args.train_all else args.train_splits,
            "sft_train_rows": len(train_examples),
            "all_rows": len(all_examples),
            "split_counts": split_counts(assignments),
            "phase1_synthetic_direct_template_rows": source_counts(train_examples).get(
                "phase1_synthetic_direct_template",
                0,
            ),
            "max_seq_len": args.max_seq_len,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "min_learning_rate": args.min_learning_rate,
            "lora_rank": MAX_LORA_RANK,
            "lora_target_modules": LORA_TARGET_MODULES,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora_dropout": args.lora_dropout,
            "optim": args.optim,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "loss_masking": "assistant_only",
            "prompt_format": "competition_chat_template",
        },
    )
    write_json(
        output_dir / "dataset_summary.json",
        {
            "train_source_counts": source_counts(train_examples),
            "train_category_counts": summarize_categories(train_examples),
            "all_source_counts": source_counts(all_examples),
            "all_category_counts": summarize_categories(all_examples),
            "split_counts": split_counts(assignments),
        },
    )
    print(f"Single-phase rows: {len(train_examples)}")
    print(f"Final adapter saved to: {adapter_dir}")
    print(f"Submission zip: {submission_path}")


if __name__ == "__main__":
    main()
