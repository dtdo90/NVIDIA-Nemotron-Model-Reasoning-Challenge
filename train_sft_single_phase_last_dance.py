#!/usr/bin/env python3
"""Final single-phase continuation pass for the competition deadline.

This script loads an existing LoRA adapter and trains exactly one more epoch on:
  1. all train-origin rows in the v5 CSV, and
  2. synthetic Numeric Equation Transformation rows.

Unlike the real-only continuation script, this keeps the synthetic Numeric
Equation methodology rows because those traces carry the most explicit
methodology signal. It uses weighted decision tokens for Text Cipher, Symbol
Transform, and Numeric Equation traces, and a cosine schedule from the starting
learning rate down to zero.
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
    SINGLE_PHASE_CSV,
    assistant_end_token,
    build_assistant_trace_content,
    build_competition_prompt,
    completion_after_generation_prompt,
    decision_point_prompt_text,
    check_nemotron_runtime_dependencies,
    clear_memory,
    default_model_path,
    disable_transformers_vision_imports,
    filter_trainable_trace_examples,
    load_examples,
    make_balanced_trainer_cls,
    make_sft_config,
    mirror_saved_outputs,
    paths_equal,
    print_trainable_parameters,
    rescue_adapter_after_train_error,
    resolve_output_and_mirror_dirs,
    resolve_trainer_state_dir,
    save_adapter_with_rescue,
    source_counts,
    summarize_categories,
    write_json,
    zip_adapter,
)

NUMERIC_CATEGORY = "Numeric Equation Transformation Rules"
WEIGHTED_CATEGORIES = {
    "Text Cipher",
    "Symbol Transform",
    "Numeric Equation Transformation Rules",
}
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
            "Load a saved single-phase LoRA adapter and continue for exactly "
            "one epoch on train-origin rows plus synthetic Numeric Equation rows."
        )
    )
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument(
        "--adapter-dir",
        default="outputs/sft_single_phase/adapter",
        help="Saved LoRA adapter directory to continue from.",
    )
    parser.add_argument("--output-dir", default="outputs/sft_single_phase_last_dance")
    parser.add_argument("--mirror-output-dir", default=None)
    parser.add_argument("--trainer-state-dir", default=None)
    parser.add_argument("--train-csv", default=str(SINGLE_PHASE_CSV))
    parser.add_argument(
        "--decision-weight",
        type=float,
        default=2.0,
        help=(
            "High-token weight for Text Cipher, Symbol Transform, and Numeric "
            "Equation decision spans. Default 2.0."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-tokenization", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
        help="Cosine floor. Default 0.0 means no floor.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Default 0.0 so the run starts at --learning-rate.",
    )
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--balanced-accumulation",
        action="store_true",
        help="Balance gradient-accumulation windows by question category.",
    )
    parser.add_argument("--optim", default="adamw_torch")
    return parser.parse_args()


def last_dance_examples(train_csv: str | Path):
    all_examples = load_examples(Path(train_csv))
    selected = [
        example
        for example in all_examples
        if example.source_mode in TRAIN_ORIGIN_SOURCE_MODES
        or (
            example.category == NUMERIC_CATEGORY
            and example.source_mode == "synthetic"
        )
    ]
    if not selected:
        raise SystemExit("No rows matched the last-dance selection.")
    selected = filter_trainable_trace_examples(
        selected,
        selection_label="last-dance train-origin + synthetic Numeric Equation rows",
    )
    return selected, all_examples


def tokenize_last_dance_example(
    tokenizer,
    example,
    *,
    max_seq_len: int,
    decision_weight: float,
) -> dict:
    end_token = assistant_end_token(tokenizer)

    if example.prompt_format == "raw_completion":
        prompt_text = example.prompt.rstrip() + "\n"
        completion_text = (example.assistant_content or example.generated_cot).lstrip()
        if not completion_text:
            raise SystemExit(f"id={example.id} has no raw completion content")
    elif example.prompt_format == "decision_point_chat_template":
        prompt_text, completion_text = decision_point_prompt_text(tokenizer, example)
    elif example.prompt_format == "competition_chat_template":
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
        completion_text = completion_after_generation_prompt(prompt_text, assistant_content)
    else:
        raise SystemExit(
            f"id={example.id} has unsupported prompt_format={example.prompt_format!r}"
        )

    if end_token and not completion_text.endswith(end_token):
        completion_text += end_token

    full_text = prompt_text + completion_text
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(full_text, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise SystemExit(
            f"id={example.id} has tokenizer boundary mismatch between prompt and completion"
        )
    if len(input_ids) > max_seq_len:
        raise SystemExit(
            f"id={example.id} has {len(input_ids)} tokens, exceeding max_seq_len={max_seq_len}"
        )
    if len(prompt_ids) >= len(input_ids):
        raise SystemExit(f"id={example.id} has no assistant tokens to score")

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    record = {
        "id": example.id,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(input_ids) - len(prompt_ids),
        "total_tokens": len(input_ids),
    }

    if decision_weight > 1.0:
        flat = [0.0] * len(prompt_ids) + [1.0] * (len(input_ids) - len(prompt_ids))
        weighter = None
        if example.category == "Text Cipher":
            from nemotron_baseline.text_cipher_loss_weights import (
                completion_label_weights as weighter,
            )
        elif example.category == "Symbol Transform":
            from nemotron_baseline.symbol_transform_loss_weights import (
                completion_label_weights as weighter,
            )
        elif example.category == "Numeric Equation Transformation Rules":
            from nemotron_baseline.numeric_equation_loss_weights import (
                completion_label_weights as weighter,
            )
        if weighter is not None:
            weights = weighter(
                tokenizer,
                prompt_text,
                completion_text,
                high=decision_weight,
                base=1.0,
            )
            record["label_weights"] = weights if len(weights) == len(input_ids) else flat
        else:
            record["label_weights"] = flat

    return record


def build_last_dance_dataset(
    dataset_cls,
    tokenizer,
    examples,
    *,
    max_seq_len: int,
    decision_weight: float,
):
    return dataset_cls.from_list(
        [
            tokenize_last_dance_example(
                tokenizer,
                example,
                max_seq_len=max_seq_len,
                decision_weight=decision_weight,
            )
            for example in examples
        ]
    )


def validate_last_dance_tokenization_examples(
    tokenizer,
    examples,
    *,
    max_seq_len: int,
    decision_weight: float,
) -> None:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    weight_rows: list[dict[str, object]] = []
    for example in examples:
        try:
            tokenized = tokenize_last_dance_example(
                tokenizer,
                example,
                max_seq_len=max_seq_len,
                decision_weight=decision_weight,
            )
        except SystemExit as exc:
            failures.append(
                {
                    "id": example.id,
                    "category": example.category,
                    "source_mode": example.source_mode,
                    "prompt_format": example.prompt_format,
                    "error": str(exc),
                }
            )
            continue
        rows.append(
            {
                "id": example.id,
                "category": example.category,
                "source_mode": example.source_mode,
                "prompt_format": example.prompt_format,
                "prompt_tokens": tokenized["prompt_tokens"],
                "completion_tokens": tokenized["completion_tokens"],
                "total_tokens": tokenized["total_tokens"],
            }
        )
        weights = tokenized.get("label_weights")
        if weights is not None:
            labels = tokenized["labels"]
            label_positions = [
                index for index, label in enumerate(labels) if label != -100
            ]
            label_token_count = len(label_positions)
            weighted_token_count = sum(
                1 for index in label_positions if weights[index] > 1.0
            )
            critical_token_count = sum(
                1 for index in label_positions if weights[index] > decision_weight
            )
            weight_rows.append(
                {
                    "id": example.id,
                    "category": example.category,
                    "source_mode": example.source_mode,
                    "label_tokens": label_token_count,
                    "weighted_tokens": weighted_token_count,
                    "critical_tokens": critical_token_count,
                    "weighted_fraction": (
                        weighted_token_count / label_token_count
                        if label_token_count
                        else 0.0
                    ),
                }
            )

    def summarize_weight_rows(items: list[dict[str, object]]) -> dict[str, object]:
        fractions = sorted(float(item["weighted_fraction"]) for item in items)
        if not fractions:
            return {"rows": 0}
        mid = len(fractions) // 2
        median = (
            fractions[mid]
            if len(fractions) % 2
            else (fractions[mid - 1] + fractions[mid]) / 2
        )
        return {
            "rows": len(items),
            "median_weighted_fraction": median,
            "max_weighted_fraction": max(fractions),
            "total_label_tokens": sum(int(item["label_tokens"]) for item in items),
            "total_weighted_tokens": sum(int(item["weighted_tokens"]) for item in items),
            "total_critical_tokens": sum(int(item["critical_tokens"]) for item in items),
        }

    top_longest = sorted(
        rows,
        key=lambda row: int(row["total_tokens"]),
        reverse=True,
    )[:20]
    by_category: dict[str, list[dict[str, object]]] = {}
    by_source: dict[str, list[dict[str, object]]] = {}
    for item in weight_rows:
        by_category.setdefault(str(item["category"]), []).append(item)
        by_source.setdefault(str(item["source_mode"]), []).append(item)

    summary = {
        "mode": "last_dance_tokenization_validation",
        "rows_checked": len(examples),
        "rows_ok": len(rows),
        "failures": len(failures),
        "max_seq_len": max_seq_len,
        "decision_weight": decision_weight,
        "top_longest": top_longest,
        "failure_samples": failures[:20],
        "weight_summary": {
            "overall": summarize_weight_rows(weight_rows),
            "by_category": {
                category: summarize_weight_rows(items)
                for category, items in sorted(by_category.items())
            },
            "by_source_mode": {
                source_mode: summarize_weight_rows(items)
                for source_mode, items in sorted(by_source.items())
            },
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit("Tokenization validation failed")


def print_summary(args: argparse.Namespace, train_examples, all_examples) -> None:
    output_dir, mirror_output_dir = resolve_output_and_mirror_dirs(
        Path(args.output_dir),
        args.mirror_output_dir,
    )
    summary = {
        "mode": "single_phase_last_dance",
        "continued_from_adapter": str(Path(args.adapter_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "mirror_output_dir": None
        if mirror_output_dir is None
        else str(mirror_output_dir.resolve()),
        "trainer_state_dir": str(
            resolve_trainer_state_dir(output_dir, args.trainer_state_dir).resolve()
        ),
        "train_csv": str(Path(args.train_csv).resolve()),
        "selection": "train-origin rows plus synthetic Numeric Equation rows",
        "train_origin_source_modes": sorted(TRAIN_ORIGIN_SOURCE_MODES),
        "synthetic_numeric_source_mode": "synthetic",
        "all_rows": len(all_examples),
        "train_rows": len(train_examples),
        "train_source_counts": source_counts(train_examples),
        "train_category_counts": summarize_categories(train_examples),
        "all_source_counts": source_counts(all_examples),
        "all_category_counts": summarize_categories(all_examples),
        "weighted_categories": sorted(WEIGHTED_CATEGORIES),
        "max_seq_len": args.max_seq_len,
        "num_train_epochs": 1.0,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine_with_min_lr",
        "warmup_ratio": args.warmup_ratio,
        "min_learning_rate": args.min_learning_rate,
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

    train_examples, all_examples = last_dance_examples(args.train_csv)
    if args.validate_only:
        print_summary(args, train_examples, all_examples)
        return

    if args.validate_tokenization:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        validate_last_dance_tokenization_examples(
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

    print_summary(args, train_examples, all_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = build_last_dance_dataset(
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
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
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
        "min_learning_rate": args.min_learning_rate,
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
        f"Starting last-dance continuation: rows={len(train_examples)}, "
        f"continued_from={args.adapter_dir}, "
        f"learning_rate={args.learning_rate}, "
        f"min_learning_rate={args.min_learning_rate}, "
        f"warmup_ratio={args.warmup_ratio}, "
        f"epochs=1.0, "
        f"decision_weight={args.decision_weight}, "
        f"balanced_accumulation={args.balanced_accumulation}"
    )

    expected_adapter_dir = output_dir / "adapter"
    try:
        trainer.train()
    except OSError as exc:
        rescue_adapter_after_train_error(trainer.model, expected_adapter_dir, exc)
        raise

    adapter_dir = save_adapter_with_rescue(trainer.model, expected_adapter_dir)
    print(f"Last-dance adapter saved to: {adapter_dir}")
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
            "mode": "single_phase_last_dance",
            "model_path": args.model_path,
            "continued_from_adapter": str(Path(args.adapter_dir).resolve()),
            "adapter_dir": str(adapter_dir.resolve()),
            "expected_adapter_dir": str(expected_adapter_dir.resolve()),
            "mirror_output_dir": None
            if mirror_output_dir is None
            else str(mirror_output_dir.resolve()),
            "trainer_state_dir": str(trainer_state_dir.resolve()),
            "submission_zip": str(submission_path.resolve()),
            "train_csv": str(Path(args.train_csv).resolve()),
            "selection": "train-origin rows plus synthetic Numeric Equation rows",
            "train_origin_source_modes": sorted(TRAIN_ORIGIN_SOURCE_MODES),
            "synthetic_numeric_source_mode": "synthetic",
            "train_rows": len(train_examples),
            "all_rows": len(all_examples),
            "train_source_counts": source_counts(train_examples),
            "train_category_counts": summarize_categories(train_examples),
            "all_source_counts": source_counts(all_examples),
            "all_category_counts": summarize_categories(all_examples),
            "weighted_categories": sorted(WEIGHTED_CATEGORIES),
            "max_seq_len": args.max_seq_len,
            "num_train_epochs": 1.0,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": "cosine_with_min_lr",
            "warmup_ratio": args.warmup_ratio,
            "min_learning_rate": args.min_learning_rate,
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
        },
    )
    mirrored_dir = mirror_saved_outputs(metadata_dir, mirror_output_dir)
    print(f"Last-dance continuation rows: {len(train_examples)}")
    print(f"Final adapter saved to: {adapter_dir}")
    if mirrored_dir is not None:
        print(f"Final artifacts mirrored to: {mirrored_dir}")
    print(f"Submission zip: {submission_path}")


if __name__ == "__main__":
    main()
