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
    make_balanced_trainer_cls,
    make_min_lr_callback,
    make_sft_config,
    mirror_saved_outputs,
    paths_equal,
    print_trainable_parameters,
    rescue_adapter_after_train_error,
    resolve_trainer_state_dir,
    resolve_output_and_mirror_dirs,
    save_adapter_with_rescue,
    source_counts,
    validate_tokenization_examples,
)
from nemotron_baseline.data import summarize_categories
from nemotron_baseline.runtime import (
    check_nemotron_runtime_dependencies,
    disable_transformers_vision_imports,
)


def parse_args(
    default_question_type: str | None = None,
    *,
    default_exclude_source_modes: list[str] | None = None,
    default_output_suffix: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one diagnostic LoRA adapter for a single question type."
    )
    if default_question_type is None:
        parser.add_argument("--question-type", required=True)
    else:
        parser.set_defaults(question_type=default_question_type)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--split-csv",
        default=None,
        help="Optional split CSV override. Defaults to the type dataset's splits_80_10_10.csv.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-suffix", default=default_output_suffix)
    parser.add_argument(
        "--mirror-output-dir",
        default=None,
        help=(
            "Optional best-effort mirror directory for final artifacts. On Colab, "
            "default diagnostic output saves locally under /content/outputs first "
            "and mirrors back to the repo output directory when it is on Drive."
        ),
    )
    parser.add_argument(
        "--trainer-state-dir",
        default=None,
        help=(
            "Directory for Trainer scratch state. Defaults to output-dir/trainer_state, "
            "except Colab Google Drive runs use /content/nemotron_trainer_state."
        ),
    )
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--validate-tokenization",
        action="store_true",
        help=(
            "Load only the tokenizer and dry-run exact prompt masking/tokenization "
            "for selected sft_train rows."
        ),
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-6)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--balanced-accumulation",
        action="store_true",
        help=(
            "Order training rows so each gradient-accumulation window is "
            "approximately balanced by diagnostic subtype."
        ),
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument(
        "--exclude-source-modes",
        nargs="*",
        default=list(default_exclude_source_modes or []),
        help="Source modes to exclude from the sft_train examples only.",
    )
    args = parser.parse_args()
    args.question_type = normalize_question_type(args.question_type)
    return args


def load_diagnostic_train_examples(
    paths,
    split_csv: Path,
    *,
    exclude_source_modes: set[str] | None = None,
) -> tuple[list, list[dict[str, str]], dict[str, str]]:
    if not paths.train_csv.exists() or not split_csv.exists():
        raise SystemExit(
            f"Missing diagnostic data for {paths.slug}. Run:\n"
            f"  python3 experiments/type_diagnostics/prepare_type_datasets.py --question-type {paths.slug}"
        )

    assert_type_dataset_fresh(paths.slug, type_csv=paths.train_csv)
    rows, _ = read_csv_rows(paths.train_csv)
    assignments = load_split_assignments(split_csv)
    validate_split_assignments(rows, assignments, split_csv=split_csv)
    train_ids = {
        row["id"]
        for row in select_rows_for_splits(rows, assignments, ["sft_train"])
        if row.get("source_mode", "unknown") not in (exclude_source_modes or set())
    }
    examples = [example for example in load_examples(paths.train_csv) if example.id in train_ids]
    if not examples:
        raise SystemExit(f"No sft_train examples found for {paths.slug}")
    return examples, rows, assignments


def diagnostic_balance_groups(
    rows: list[dict[str, str]],
    train_examples,
) -> list[str]:
    rows_by_id = {row["id"]: row for row in rows}
    groups: list[str] = []
    for example in train_examples:
        row = rows_by_id.get(example.id, {})
        groups.append(
            row.get("diagnostic_subtype")
            or row.get("source_mode")
            or example.source_mode
            or example.category
        )
    return groups


def print_summary(args: argparse.Namespace, paths, train_examples, rows, assignments) -> None:
    excluded = set(args.exclude_source_modes or [])
    train_rows = [
        row
        for row in select_rows_for_splits(rows, assignments, ["sft_train"])
        if row.get("source_mode", "unknown") not in excluded
    ]
    output_dir = Path(args.output_dir)
    mirror_output_dir = Path(args.mirror_output_dir) if args.mirror_output_dir else None
    payload = {
        "mode": "type_diagnostic_sft",
        "question_type": args.question_type,
        "category": QUESTION_TYPES[args.question_type]["category"],
        "output_dir": str(output_dir.resolve()),
        "mirror_output_dir": None
        if mirror_output_dir is None
        else str(mirror_output_dir.resolve()),
        "trainer_state_dir": str(
            resolve_trainer_state_dir(output_dir, args.trainer_state_dir).resolve()
        ),
        "train_csv": str(paths.train_csv.resolve()),
        "split_csv": str(Path(args.split_csv).resolve()),
        "all_rows": len(rows),
        "sft_train_rows": len(train_examples),
        "exclude_source_modes": sorted(excluded),
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
        "lr_scheduler_type": "cosine_with_min_lr",
        "warmup_ratio": 0.05,
        "min_learning_rate": args.min_learning_rate,
        "gradient_checkpointing": args.gradient_checkpointing,
        "balanced_accumulation": args.balanced_accumulation,
        "balanced_accumulation_group": "diagnostic_subtype"
        if args.balanced_accumulation
        else None,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "optim": args.optim,
        "loss_masking": "assistant_only",
        "prompt_format_counts": dict(
            sorted(Counter(example.prompt_format for example in train_examples).items())
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def default_output_dirs_for_runtime(paths) -> tuple[Path, Path | None]:
    output_dir, mirror_output_dir = resolve_output_and_mirror_dirs(paths.output_dir)
    if mirror_output_dir is not None:
        output_dir = Path("/content/outputs/type_diagnostics") / paths.slug
    return output_dir, mirror_output_dir


def apply_output_suffix(
    output_dir: Path,
    mirror_output_dir: Path | None,
    *,
    slug: str,
    suffix: str | None,
) -> tuple[Path, Path | None]:
    if not suffix:
        return output_dir, mirror_output_dir
    name = f"{slug}_{suffix}"
    output_dir = output_dir.parent / name
    if mirror_output_dir is not None:
        mirror_output_dir = mirror_output_dir.parent / name
    return output_dir, mirror_output_dir


def main(
    default_question_type: str | None = None,
    *,
    default_exclude_source_modes: list[str] | None = None,
    default_output_suffix: str | None = None,
) -> None:
    args = parse_args(
        default_question_type,
        default_exclude_source_modes=default_exclude_source_modes,
        default_output_suffix=default_output_suffix,
    )
    paths = type_paths(args.question_type, data_dir=Path(args.data_dir))
    split_csv = Path(args.split_csv) if args.split_csv else paths.split_csv
    args.split_csv = str(split_csv)
    if args.output_dir is None:
        output_dir, mirror_output_dir = default_output_dirs_for_runtime(paths)
        output_dir, mirror_output_dir = apply_output_suffix(
            output_dir,
            mirror_output_dir,
            slug=paths.slug,
            suffix=args.output_suffix,
        )
    else:
        output_dir, mirror_output_dir = resolve_output_and_mirror_dirs(
            Path(args.output_dir),
            args.mirror_output_dir,
        )
    args.output_dir = str(output_dir)
    args.mirror_output_dir = None if mirror_output_dir is None else str(mirror_output_dir)
    excluded_source_modes = set(args.exclude_source_modes or [])
    train_examples, rows, assignments = load_diagnostic_train_examples(
        paths,
        split_csv,
        exclude_source_modes=excluded_source_modes,
    )

    if args.validate_only:
        print_summary(args, paths, train_examples, rows, assignments)
        return

    if args.validate_tokenization:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        validate_tokenization_examples(tokenizer, train_examples, max_seq_len=args.max_seq_len)
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
    mirror_output_dir = Path(args.mirror_output_dir) if args.mirror_output_dir else None
    if mirror_output_dir is not None and not paths_equal(output_dir, mirror_output_dir):
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
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
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
        "callbacks": [make_min_lr_callback(TrainerCallback, args.min_learning_rate)],
        "min_learning_rate": args.min_learning_rate,
    }
    if args.balanced_accumulation:
        trainer_kwargs.update(
            {
                "balanced_accumulation_groups": diagnostic_balance_groups(
                    rows,
                    train_examples,
                ),
                "balanced_accumulation_effective_batch_size": (
                    args.per_device_train_batch_size * args.gradient_accumulation_steps
                ),
                "balanced_accumulation_seed": 42,
            }
        )
    trainer = trainer_cls(**trainer_kwargs)
    print(
        f"Starting {args.question_type}: rows={len(train_examples)}, "
        f"learning_rate={args.learning_rate}, num_train_epochs=1.0, "
        f"balanced_accumulation={args.balanced_accumulation}"
    )
    expected_adapter_dir = output_dir / "adapter"
    try:
        trainer.train()
    except OSError as exc:
        rescue_adapter_after_train_error(trainer.model, expected_adapter_dir, exc)
        raise

    adapter_dir = save_adapter_with_rescue(trainer.model, expected_adapter_dir)
    print(f"Diagnostic adapter saved to: {adapter_dir}")
    trainer.model = None
    del trainer
    del dataset
    clear_memory()

    metadata_dir = output_dir if adapter_dir == expected_adapter_dir else adapter_dir.parent
    write_json(
        metadata_dir / "run_config.json",
        {
            "mode": "type_diagnostic_sft",
            "question_type": args.question_type,
            "model_path": args.model_path,
            "adapter_dir": str(adapter_dir.resolve()),
            "expected_adapter_dir": str(expected_adapter_dir.resolve()),
            "mirror_output_dir": None
            if mirror_output_dir is None
            else str(mirror_output_dir.resolve()),
            "trainer_state_dir": str(trainer_state_dir.resolve()),
            "train_csv": str(paths.train_csv.resolve()),
            "split_csv": str(split_csv.resolve()),
            "sft_train_rows": len(train_examples),
            "all_rows": len(rows),
            "exclude_source_modes": sorted(excluded_source_modes),
            "max_seq_len": args.max_seq_len,
            "learning_rate": args.learning_rate,
            "num_train_epochs": 1.0,
            "lr_scheduler_type": "cosine_with_min_lr",
            "warmup_ratio": 0.05,
            "min_learning_rate": args.min_learning_rate,
            "lora_rank": MAX_LORA_RANK,
            "lora_target_modules": LORA_TARGET_MODULES,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora_dropout": args.lora_dropout,
            "optim": args.optim,
            "balanced_accumulation": args.balanced_accumulation,
            "balanced_accumulation_group": (
                "diagnostic_subtype" if args.balanced_accumulation else None
            ),
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "loss_masking": "assistant_only",
            "prompt_format_counts": dict(
                sorted(Counter(example.prompt_format for example in train_examples).items())
            ),
        },
    )
    write_json(
        metadata_dir / "dataset_summary.json",
        {
            "all_data_summary": summarize_rows(rows, assignments),
            "train_source_counts": source_counts(train_examples),
            "train_category_counts": summarize_categories(train_examples),
            "sft_train_subtypes": dict(
                sorted(
                    Counter(
                        row["diagnostic_subtype"]
                        for row in select_rows_for_splits(rows, assignments, ["sft_train"])
                        if row.get("source_mode", "unknown") not in excluded_source_modes
                    ).items()
                )
            ),
        },
    )
    mirrored_dir = mirror_saved_outputs(metadata_dir, mirror_output_dir)
    print(f"{args.question_type} SFT rows: {len(train_examples)}")
    print(f"Final adapter saved to: {adapter_dir}")
    if mirrored_dir is not None:
        print(f"Final artifacts mirrored to: {mirrored_dir}")


if __name__ == "__main__":
    main()
