#!/usr/bin/env python3
"""Minimal portable LoRA SFT trainer for Colab, Lightning AI, or local GPUs."""
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

PHASE1_CSV = ROOT / "data/training_ready_clean/phase1_train.csv"
PHASE2_CSV = ROOT / "data/training_ready_clean/phase2_sft.csv"
PHASE2_SPLIT_CSV = ROOT / "data/training_ready_clean/phase2_splits_80_10_10.csv"
HF_MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
KAGGLE_MODEL_PATH = Path(
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
MAX_LORA_RANK = 32
DEFAULT_MAX_SEQ_LEN = 8192


def default_model_path() -> str:
    explicit_model_path = os.environ.get("MODEL_PATH") or os.environ.get("BASE_MODEL_PATH")
    if explicit_model_path:
        return explicit_model_path
    if KAGGLE_MODEL_PATH.exists():
        return str(KAGGLE_MODEL_PATH)
    return HF_MODEL_PATH


MODEL_PATH = default_model_path()

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
        help=(
            "Local model path or HF model id. Defaults to the Kaggle mounted model "
            f"when present, otherwise {HF_MODEL_PATH}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: outputs/sft_two_stage_h200 or outputs/sft_phase1_h200.",
    )
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--phase2", action="store_true", help="Train only Phase 2 from a saved Phase 1 adapter.")
    parser.add_argument(
        "--phase1-adapter-dir",
        default=None,
        help="Phase 1 adapter to load for --phase2. Default: <output-dir>/phase1/adapter.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--phase1-learning-rate", type=float, default=5e-5)
    parser.add_argument("--phase2-learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="Trainer optimizer. Use adamw_torch if fused AdamW is unavailable.",
    )
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
    max_seq_len: int,
    gradient_checkpointing: bool,
    optim: str,
    phase1_learning_rate: float,
    phase2_learning_rate: float,
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
        "max_seq_len": max_seq_len,
        "lora_rank": MAX_LORA_RANK,
        "stage_learning_rates": {
            "phase1": None if mode == "phase2_only" else phase1_learning_rate,
            "phase2": None if mode == "phase1_only" else phase2_learning_rate,
        },
        "gradient_checkpointing": gradient_checkpointing,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": per_device_train_batch_size * gradient_accumulation_steps,
        "optim": optim,
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
    max_seq_len: int,
    gradient_checkpointing: bool,
    optim: str,
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
        optim=optim,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_len,
        packing=False,
        group_by_length=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=gradient_checkpointing,
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
    trained_model = trainer.model
    trainer.model = None
    del trainer
    del dataset
    clear_memory()
    return trained_model, stage_adapter_dir


def main() -> None:
    args = parse_args()
    if args.phase1_only and args.phase2:
        raise SystemExit("Use either --phase1-only or --phase2, not both")
    mode = "phase2_only" if args.phase2 else ("phase1_only" if args.phase1_only else "phase1_then_phase2")
    output_dir = Path(
        args.output_dir
        or ("outputs/sft_phase1_h200" if args.phase1_only else "outputs/sft_two_stage_h200")
    )

    phase2_train_examples: list[Example] = []
    holdout_examples: list[Example] = []
    split_assignments = None

    phase1_examples = [] if args.phase2 else load_examples(PHASE1_CSV, source="phase1", append_answer_instruction=False)

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
            max_seq_len=args.max_seq_len,
            gradient_checkpointing=args.gradient_checkpointing,
            optim=args.optim,
            phase1_learning_rate=args.phase1_learning_rate,
            phase2_learning_rate=args.phase2_learning_rate,
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
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    phase1_adapter_dir = Path(args.phase1_adapter_dir or (output_dir / "phase1" / "adapter"))
    if args.phase2:
        adapter_config = phase1_adapter_dir / "adapter_config.json"
        if not adapter_config.exists():
            raise SystemExit(
                f"--phase2 requires a saved Phase 1 adapter at {phase1_adapter_dir}. "
                "Pass --phase1-adapter-dir if it is somewhere else."
            )
        print(f"Loading trainable Phase 1 adapter from: {phase1_adapter_dir}")
        model = get_peft_model(model, lora_config)
        load_adapter_tensors(model, phase1_adapter_dir, torch)
    else:
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

    if args.phase2:
        final_adapter_dir = phase1_adapter_dir
    else:
        phase1_stage_dir = output_dir if args.phase1_only else output_dir / "phase1"
        model, phase1_adapter_dir = train_stage(
            stage_name="phase1",
            model=model,
            tokenizer=tokenizer,
            examples=phase1_examples,
            learning_rate=args.phase1_learning_rate,
            stage_output_dir=phase1_stage_dir,
            dataset_cls=Dataset,
            sft_config_cls=SFTConfig,
            sft_trainer_cls=SFTTrainer,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_seq_len=args.max_seq_len,
            gradient_checkpointing=args.gradient_checkpointing,
            optim=args.optim,
        )
        final_adapter_dir = phase1_adapter_dir

    if not args.phase1_only:
        clear_memory()
        model, final_adapter_dir = train_stage(
            stage_name="phase2",
            model=model,
            tokenizer=tokenizer,
            examples=phase2_train_examples,
            learning_rate=args.phase2_learning_rate,
            stage_output_dir=output_dir,
            dataset_cls=Dataset,
            sft_config_cls=SFTConfig,
            sft_trainer_cls=SFTTrainer,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_seq_len=args.max_seq_len,
            gradient_checkpointing=args.gradient_checkpointing,
            optim=args.optim,
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
            "max_seq_len": args.max_seq_len,
            "stage_learning_rates": {
                "phase1": None if args.phase2 else args.phase1_learning_rate,
                "phase2": None if args.phase1_only else args.phase2_learning_rate,
            },
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
            "split_counts": summarize_split_assignments(split_assignments) if split_assignments else None,
        },
    )

    print(f"Phase 1 rows: {len(phase1_examples)}")
    print(f"Phase 2 rows: {len(phase2_train_examples)}")
    print(f"Final adapter saved to: {final_adapter_dir}")
    print(f"Submission zip: {submission_path}")


if __name__ == "__main__":
    main()
