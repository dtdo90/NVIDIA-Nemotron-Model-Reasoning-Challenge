#!/usr/bin/env python3
"""Continue an existing single-phase LoRA adapter on train-origin v5 rows.

This is intended as a low-LR polish pass after the main mixed-data run:
load the saved adapter, train for exactly one epoch at a flat learning rate,
and keep only rows derived from the original train.csv. By default it uses the
normal sft_train split to avoid validation/eval leakage; pass --train-all-real
intentionally to use every train-origin row in the v5 CSV.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from train_sft_single_phase import (
    DEFAULT_MAX_SEQ_LEN,
    MAX_LORA_RANK,
    LORA_TARGET_MODULES,
    MaskedCausalLMDataCollator,
    build_dataset,
    check_nemotron_runtime_dependencies,
    clear_memory,
    default_model_path,
    disable_transformers_vision_imports,
    filter_trainable_trace_examples,
    load_examples,
    load_split_assignments,
    make_balanced_trainer_cls,
    make_sft_config,
    mirror_saved_outputs,
    paths_equal,
    print_trainable_parameters,
    rescue_adapter_after_train_error,
    resolve_output_and_mirror_dirs,
    resolve_trainer_state_dir,
    save_adapter_with_rescue,
    select_examples_by_split,
    source_counts,
    split_counts,
    summarize_categories,
    validate_split_assignments,
    validate_tokenization_examples,
    write_json,
    zip_adapter,
    SINGLE_PHASE_CSV,
    SINGLE_PHASE_SPLIT_CSV,
)

TRAIN_ORIGIN_SOURCE_MODES = {
    "real",
    "huikang_real_bit",
    "huikang_real_bit_extra_trace",
    "symbol_transform_unreliable_pattern_guess",
    "op_ab_guess_0134_wrong",
    "op_ab_guess_0134_correct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved single-phase LoRA adapter and continue training for "
            "exactly one flat-LR epoch on train-origin v5 rows."
        )
    )
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument(
        "--adapter-dir",
        default="outputs/sft_single_phase/adapter",
        help="Saved LoRA adapter directory to continue from.",
    )
    parser.add_argument("--output-dir", default="outputs/sft_single_phase_real_continue")
    parser.add_argument("--mirror-output-dir", default=None)
    parser.add_argument("--trainer-state-dir", default=None)
    parser.add_argument("--train-csv", default=str(SINGLE_PHASE_CSV))
    parser.add_argument("--split-csv", default=str(SINGLE_PHASE_SPLIT_CSV))
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["sft_train"],
        help="Split names to use before source_mode=real filtering.",
    )
    parser.add_argument(
        "--train-all-real",
        action="store_true",
        help="Ignore --split-csv and train on every train-origin row in --train-csv.",
    )
    parser.add_argument(
        "--decision-weight",
        type=float,
        default=1.0,
        help="Optional decision-token loss weight. Default 1.0 keeps this polish pass unweighted.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-tokenization", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--balanced-accumulation",
        action="store_true",
        help="Balance gradient-accumulation windows by question category.",
    )
    parser.add_argument("--optim", default="adamw_torch")
    return parser.parse_args()


def real_train_examples(
    train_csv: str | Path,
    *,
    split_csv: str | Path | None,
    train_splits: list[str],
    train_all_real: bool,
):
    all_examples = load_examples(Path(train_csv))
    assignments = None
    if train_all_real:
        selected = all_examples
        selection_label = "--train-all-real"
    else:
        if split_csv is None:
            raise SystemExit("--split-csv is required unless --train-all-real is set")
        assignments = load_split_assignments(split_csv)
        validate_split_assignments(all_examples, assignments, split_csv)
        selected = select_examples_by_split(all_examples, assignments, train_splits)
        selection_label = "+".join(train_splits)

    selected = [
        example for example in selected if example.source_mode in TRAIN_ORIGIN_SOURCE_MODES
    ]
    if not selected:
        raise SystemExit(f"No train-origin rows matched {selection_label}")
    selected = filter_trainable_trace_examples(
        selected,
        selection_label=f"{selection_label} train-origin rows",
    )
    return selected, all_examples, assignments


def print_summary(
    args: argparse.Namespace,
    train_examples,
    all_examples,
    assignments,
) -> None:
    output_dir, mirror_output_dir = resolve_output_and_mirror_dirs(
        Path(args.output_dir),
        args.mirror_output_dir,
    )
    summary = {
        "mode": "single_phase_real_continue",
        "continued_from_adapter": str(Path(args.adapter_dir).resolve()),
        "train_origin_source_modes": sorted(TRAIN_ORIGIN_SOURCE_MODES),
        "output_dir": str(output_dir.resolve()),
        "mirror_output_dir": None
        if mirror_output_dir is None
        else str(mirror_output_dir.resolve()),
        "trainer_state_dir": str(
            resolve_trainer_state_dir(output_dir, args.trainer_state_dir).resolve()
        ),
        "train_csv": str(Path(args.train_csv).resolve()),
        "split_csv": None if args.train_all_real else str(Path(args.split_csv).resolve()),
        "train_splits": ["ALL_REAL"] if args.train_all_real else args.train_splits,
        "all_rows": len(all_examples),
        "real_train_rows": len(train_examples),
        "split_counts": split_counts(assignments),
        "train_source_counts": source_counts(train_examples),
        "train_category_counts": summarize_categories(train_examples),
        "all_source_counts": source_counts(all_examples),
        "all_category_counts": summarize_categories(all_examples),
        "max_seq_len": args.max_seq_len,
        "num_train_epochs": 1.0,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "decision_weight": args.decision_weight,
        "gradient_checkpointing": args.gradient_checkpointing,
        "balanced_accumulation": args.balanced_accumulation,
        "balanced_accumulation_group": "category" if args.balanced_accumulation else None,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size
        * args.gradient_accumulation_steps,
        "optim": args.optim,
        "loss_masking": "assistant_only",
        "prompt_format_counts": dict(
            sorted(Counter(example.prompt_format for example in train_examples).items())
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    adapter_dir_arg = Path(args.adapter_dir)
    if not args.validate_only and not args.validate_tokenization and not adapter_dir_arg.exists():
        raise SystemExit(f"--adapter-dir does not exist: {adapter_dir_arg}")

    train_examples, all_examples, assignments = real_train_examples(
        args.train_csv,
        split_csv=args.split_csv,
        train_splits=args.train_splits,
        train_all_real=args.train_all_real,
    )
    if args.validate_only:
        print_summary(args, train_examples, all_examples, assignments)
        return

    if args.validate_tokenization:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        validate_tokenization_examples(
            tokenizer,
            train_examples,
            max_seq_len=args.max_seq_len,
            decision_weight=args.decision_weight,
        )
        return

    disable_transformers_vision_imports()
    check_nemotron_runtime_dependencies()

    from datasets import Dataset  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # type: ignore
    import torch  # type: ignore

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    requested_output_dir = Path(args.output_dir)
    output_dir, mirror_output_dir = resolve_output_and_mirror_dirs(
        requested_output_dir,
        args.mirror_output_dir,
    )
    if not paths_equal(output_dir, requested_output_dir):
        print(
            f"Using local output dir for training artifacts: {output_dir}\n"
            f"Best-effort mirror output dir: {mirror_output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_state_dir = resolve_trainer_state_dir(output_dir, args.trainer_state_dir)
    trainer_state_dir.mkdir(parents=True, exist_ok=True)
    print(f"Trainer state dir: {trainer_state_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = build_dataset(
        Dataset,
        tokenizer,
        train_examples,
        max_seq_len=args.max_seq_len,
        decision_weight=args.decision_weight,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    print(f"Model device map: {getattr(base_model, 'hf_device_map', None)}")
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=True)
    print_trainable_parameters(model)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()

    trainer_args = make_sft_config(
        TrainingArguments,
        output_dir=str(trainer_state_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1.0,
        learning_rate=args.learning_rate,
        logging_steps=10,
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,
        optim=args.optim,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        save_strategy="no",
        report_to="none",
        group_by_length=not args.balanced_accumulation,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
    )
    trainer_cls = make_balanced_trainer_cls(Trainer)
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "data_collator": MaskedCausalLMDataCollator(tokenizer),
        "args": trainer_args,
        # The reused balanced trainer uses this to build a constant scheduler
        # when min_learning_rate == learning_rate.
        "min_learning_rate": args.learning_rate,
    }
    if args.balanced_accumulation:
        trainer_kwargs.update(
            {
                "balanced_accumulation_groups": [example.category for example in train_examples],
                "balanced_accumulation_effective_batch_size": (
                    args.per_device_train_batch_size * args.gradient_accumulation_steps
                ),
                "balanced_accumulation_seed": 42,
            }
        )
    trainer = trainer_cls(**trainer_kwargs)
    print(
        f"Starting real-only continuation: rows={len(train_examples)}, "
        f"splits={['ALL_REAL'] if args.train_all_real else args.train_splits}, "
        f"continued_from={args.adapter_dir}, "
        f"learning_rate={args.learning_rate}, "
        f"epochs=1.0, "
        f"balanced_accumulation={args.balanced_accumulation}"
    )

    expected_adapter_dir = output_dir / "adapter"
    try:
        trainer.train()
    except OSError as exc:
        rescue_adapter_after_train_error(trainer.model, expected_adapter_dir, exc)
        raise

    adapter_dir = save_adapter_with_rescue(trainer.model, expected_adapter_dir)
    print(f"Continued adapter saved to: {adapter_dir}")
    trainer.model = None
    del trainer
    del dataset
    clear_memory()

    metadata_dir = output_dir if adapter_dir == expected_adapter_dir else adapter_dir.parent
    submission_path = metadata_dir / "submission.zip"
    zip_adapter(adapter_dir, submission_path)
    write_json(
        metadata_dir / "run_config.json",
        {
            "mode": "single_phase_real_continue",
            "model_path": args.model_path,
            "continued_from_adapter": str(Path(args.adapter_dir).resolve()),
            "train_origin_source_modes": sorted(TRAIN_ORIGIN_SOURCE_MODES),
            "adapter_dir": str(adapter_dir.resolve()),
            "expected_adapter_dir": str(expected_adapter_dir.resolve()),
            "mirror_output_dir": None
            if mirror_output_dir is None
            else str(mirror_output_dir.resolve()),
            "trainer_state_dir": str(trainer_state_dir.resolve()),
            "submission_zip": str(submission_path.resolve()),
            "train_csv": str(Path(args.train_csv).resolve()),
            "split_csv": None if args.train_all_real else str(Path(args.split_csv).resolve()),
            "train_splits": ["ALL_REAL"] if args.train_all_real else args.train_splits,
            "real_train_rows": len(train_examples),
            "all_rows": len(all_examples),
            "split_counts": split_counts(assignments),
            "train_source_counts": source_counts(train_examples),
            "train_category_counts": summarize_categories(train_examples),
            "max_seq_len": args.max_seq_len,
            "num_train_epochs": 1.0,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.0,
            "decision_weight": args.decision_weight,
            "lora_rank": MAX_LORA_RANK,
            "lora_target_modules": LORA_TARGET_MODULES,
            "gradient_checkpointing": args.gradient_checkpointing,
            "optim": args.optim,
            "balanced_accumulation": args.balanced_accumulation,
            "balanced_accumulation_group": (
                "category" if args.balanced_accumulation else None
            ),
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size
            * args.gradient_accumulation_steps,
            "loss_masking": "assistant_only",
            "prompt_format_counts": dict(
                sorted(Counter(example.prompt_format for example in train_examples).items())
            ),
        },
    )
    write_json(
        metadata_dir / "dataset_summary.json",
        {
            "train_source_counts": source_counts(train_examples),
            "train_category_counts": summarize_categories(train_examples),
            "all_source_counts": source_counts(all_examples),
            "all_category_counts": summarize_categories(all_examples),
            "split_counts": split_counts(assignments),
        },
    )
    mirrored_dir = mirror_saved_outputs(metadata_dir, mirror_output_dir)
    print(f"Real-only continuation rows: {len(train_examples)}")
    print(f"Final adapter saved to: {adapter_dir}")
    if mirrored_dir is not None:
        print(f"Final artifacts mirrored to: {mirrored_dir}")
    print(f"Submission zip: {submission_path}")


if __name__ == "__main__":
    main()
