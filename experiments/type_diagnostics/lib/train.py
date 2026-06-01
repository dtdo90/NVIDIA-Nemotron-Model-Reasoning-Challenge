from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .common import (
    DATA_DIR,
    QUESTION_TYPES,
    SPLIT_NAMES,
    assert_type_dataset_fresh,
    load_split_assignments,
    normalize_question_type,
    read_csv_rows,
    select_rows_for_splits,
    summarize_rows,
    type_paths,
    validate_split_assignments,
    write_json,
)

from train_sft_single_phase import (  # type: ignore
    DEFAULT_MAX_SEQ_LEN,
    LORA_TARGET_MODULES,
    MAX_LORA_RANK,
    MaskedCausalLMDataCollator,
    build_dataset,
    clear_memory,
    default_model_path,
    load_examples,
    make_min_lr_callback,
    make_sft_config,
    print_trainable_parameters,
    source_counts,
)
from nemotron_baseline.data import summarize_categories
from nemotron_baseline.runtime import (
    check_nemotron_runtime_dependencies,
    disable_transformers_vision_imports,
)


def parse_args(default_question_type: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one diagnostic LoRA adapter for a single question type."
    )
    if default_question_type is None:
        parser.add_argument("--question-type", required=True)
    else:
        parser.set_defaults(question_type=default_question_type)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-6)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--optim", default="adamw_torch")
    args = parser.parse_args()
    args.question_type = normalize_question_type(args.question_type)
    return args


def load_diagnostic_train_examples(paths) -> tuple[list, list[dict[str, str]], dict[str, str]]:
    if not paths.train_csv.exists() or not paths.split_csv.exists():
        raise SystemExit(
            f"Missing diagnostic data for {paths.slug}. Run:\n"
            f"  python3 experiments/type_diagnostics/prepare_type_datasets.py --question-type {paths.slug}"
        )

    assert_type_dataset_fresh(paths.slug, type_csv=paths.train_csv)
    rows, _ = read_csv_rows(paths.train_csv)
    assignments = load_split_assignments(paths.split_csv)
    validate_split_assignments(rows, assignments, split_csv=paths.split_csv)
    train_ids = {
        row["id"]
        for row in select_rows_for_splits(rows, assignments, ["sft_train"])
    }
    examples = [example for example in load_examples(paths.train_csv) if example.id in train_ids]
    if not examples:
        raise SystemExit(f"No sft_train examples found for {paths.slug}")
    return examples, rows, assignments


def print_summary(args: argparse.Namespace, paths, train_examples, rows, assignments) -> None:
    train_rows = select_rows_for_splits(rows, assignments, ["sft_train"])
    payload = {
        "mode": "type_diagnostic_sft",
        "question_type": args.question_type,
        "category": QUESTION_TYPES[args.question_type]["category"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "train_csv": str(paths.train_csv.resolve()),
        "split_csv": str(paths.split_csv.resolve()),
        "all_rows": len(rows),
        "sft_train_rows": len(train_examples),
        "split_names": list(SPLIT_NAMES),
        "all_data_summary": summarize_rows(rows, assignments),
        "sft_train_summary": summarize_rows(train_rows),
        "train_source_counts": source_counts(train_examples),
        "train_category_counts": summarize_categories(train_examples),
        "max_seq_len": args.max_seq_len,
        "lora_rank": MAX_LORA_RANK,
        "lora_target_modules": LORA_TARGET_MODULES,
        "learning_rate": args.learning_rate,
        "num_train_epochs": 1.0,
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
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(default_question_type: str | None = None) -> None:
    args = parse_args(default_question_type)
    paths = type_paths(args.question_type, data_dir=Path(args.data_dir))
    args.output_dir = args.output_dir or str(paths.output_dir)
    train_examples, rows, assignments = load_diagnostic_train_examples(paths)

    if args.validate_only:
        print_summary(args, paths, train_examples, rows, assignments)
        return

    disable_transformers_vision_imports()
    check_nemotron_runtime_dependencies()

    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments  # type: ignore

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
        f"Starting {args.question_type}: rows={len(train_examples)}, "
        f"learning_rate={args.learning_rate}, num_train_epochs=1.0"
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    print(f"Diagnostic adapter saved to: {adapter_dir}")
    trainer.model = None
    del trainer
    del dataset
    clear_memory()

    write_json(
        output_dir / "run_config.json",
        {
            "mode": "type_diagnostic_sft",
            "question_type": args.question_type,
            "model_path": args.model_path,
            "adapter_dir": str(adapter_dir.resolve()),
            "train_csv": str(paths.train_csv.resolve()),
            "split_csv": str(paths.split_csv.resolve()),
            "sft_train_rows": len(train_examples),
            "all_rows": len(rows),
            "max_seq_len": args.max_seq_len,
            "learning_rate": args.learning_rate,
            "num_train_epochs": 1.0,
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
            "all_data_summary": summarize_rows(rows, assignments),
            "train_source_counts": source_counts(train_examples),
            "train_category_counts": summarize_categories(train_examples),
            "sft_train_subtypes": dict(
                sorted(Counter(row["diagnostic_subtype"] for row in select_rows_for_splits(rows, assignments, ["sft_train"])).items())
            ),
        },
    )
    print(f"{args.question_type} SFT rows: {len(train_examples)}")
    print(f"Final adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
