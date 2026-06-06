#!/usr/bin/env python3
"""Standalone single-phase LoRA SFT trainer on the clean single-phase data mix."""
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import os
import random
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SINGLE_PHASE_CSV = ROOT / "data/single_phase_training_clean/single_phase_sft_v2.csv"
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
    prompt_format: str = "competition_chat_template"


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
    parser.add_argument(
        "--mirror-output-dir",
        default=None,
        help=(
            "Optional best-effort mirror directory for final artifacts. On Colab, "
            "if --output-dir resolves under /content/drive, training saves locally "
            "under /content/outputs first and mirrors back to this output dir by default."
        ),
    )
    parser.add_argument(
        "--trainer-state-dir",
        default=None,
        help=(
            "Directory for Trainer scratch state. Defaults to output-dir/trainer_state, "
            "except Colab Google Drive runs use /content/nemotron_trainer_state to "
            "avoid Drive transport disconnects at the end of training."
        ),
    )
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
    parser.add_argument(
        "--validate-tokenization",
        action="store_true",
        help=(
            "Load only the tokenizer and dry-run exact prompt masking/tokenization "
            "for the selected train rows. Fails before training on boundary or "
            "max_seq_len problems."
        ),
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--balanced-accumulation",
        action="store_true",
        help=(
            "Order training rows so each gradient-accumulation window is "
            "approximately balanced by question category. This preserves one "
            "puzzle per sequence and only changes sampler order."
        ),
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        help="Trainer optimizer.",
    )
    return parser.parse_args()


def path_slug(path: Path) -> str:
    try:
        text = str(path.resolve())
    except OSError:
        text = str(path)
    slug = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return slug[-160:] or "run"


def resolve_trainer_state_dir(output_dir: Path, explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        return Path(explicit_dir)
    try:
        resolved = str(output_dir.resolve())
    except OSError:
        resolved = str(output_dir)
    if resolved.startswith("/content/drive/"):
        return Path("/content/nemotron_trainer_state") / path_slug(output_dir)
    return output_dir / "trainer_state"


def is_colab_drive_path(path: Path) -> bool:
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)
    return resolved.startswith("/content/drive/")


def local_output_dir_for_drive(output_dir: Path) -> Path:
    name = output_dir.name or path_slug(output_dir)
    return Path("/content/outputs") / name


def paths_equal(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def resolve_output_and_mirror_dirs(
    requested_output_dir: Path,
    explicit_mirror_dir: str | None = None,
) -> tuple[Path, Path | None]:
    mirror_dir = Path(explicit_mirror_dir) if explicit_mirror_dir else None
    if is_colab_drive_path(requested_output_dir):
        return local_output_dir_for_drive(requested_output_dir), mirror_dir or requested_output_dir
    return requested_output_dir, mirror_dir


def local_rescue_adapter_dir(adapter_dir: Path) -> Path:
    base = Path("/content/nemotron_rescue") if Path("/content").exists() else Path(tempfile.gettempdir()) / "nemotron_rescue"
    return base / path_slug(adapter_dir.parent) / adapter_dir.name


def mirror_saved_outputs(source_dir: Path, mirror_dir: Path | None) -> Path | None:
    if mirror_dir is None or paths_equal(source_dir, mirror_dir):
        return None

    print(f"Mirroring final artifacts to: {mirror_dir}")
    try:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for name in ("adapter", "submission.zip", "run_config.json", "dataset_summary.json"):
            source = source_dir / name
            if not source.exists():
                continue
            destination = mirror_dir / name
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
    except OSError as exc:
        print(f"WARNING: failed to mirror final artifacts to {mirror_dir}: {exc}")
        return None

    print(f"Final artifacts mirrored to: {mirror_dir}")
    return mirror_dir


def save_adapter_with_rescue(model, adapter_dir: Path) -> Path:
    try:
        model.save_pretrained(adapter_dir)
        return adapter_dir
    except OSError as exc:
        rescue_dir = local_rescue_adapter_dir(adapter_dir)
        print(
            f"WARNING: failed to save adapter to {adapter_dir}: {exc}\n"
            f"Attempting local rescue save to: {rescue_dir}"
        )
        model.save_pretrained(rescue_dir)
        print(f"Adapter rescue save succeeded: {rescue_dir}")
        return rescue_dir


def rescue_adapter_after_train_error(model, adapter_dir: Path, exc: OSError) -> None:
    rescue_dir = local_rescue_adapter_dir(adapter_dir)
    print(
        f"WARNING: trainer.train() raised OSError after/while training: {exc}\n"
        f"Attempting emergency adapter save to: {rescue_dir}"
    )
    try:
        model.save_pretrained(rescue_dir)
    except Exception as save_exc:
        print(f"Emergency adapter save failed: {save_exc}")
        return
    print(f"Emergency adapter save succeeded: {rescue_dir}")


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
                    prompt_format=(
                        row.get("prompt_format") or "competition_chat_template"
                    ).strip()
                    or "competition_chat_template",
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


def balanced_accumulation_order(
    groups: list[str],
    *,
    effective_batch_size: int,
    seed: int,
) -> list[int]:
    """Spread group labels evenly across optimizer-step accumulation windows."""
    if not groups:
        return []
    if effective_batch_size <= 1:
        return list(range(len(groups)))

    rng = random.Random(seed)
    n_windows = math.ceil(len(groups) / effective_batch_size)
    windows: list[list[int]] = [[] for _ in range(n_windows)]
    window_order = list(range(n_windows))
    rng.shuffle(window_order)

    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group or "unknown", []).append(index)

    assigned = 0
    for group in sorted(by_group):
        indices = by_group[group]
        rng.shuffle(indices)
        for index in indices:
            windows[window_order[assigned % n_windows]].append(index)
            assigned += 1

    order: list[int] = []
    for window in windows:
        rng.shuffle(window)
        order.extend(window)
    return order


class BalancedAccumulationSampler:
    """Sampler that preserves examples but balances groups per update window."""

    def __init__(
        self,
        groups: list[str],
        *,
        effective_batch_size: int,
        seed: int,
    ) -> None:
        self.order = balanced_accumulation_order(
            groups,
            effective_batch_size=effective_batch_size,
            seed=seed,
        )

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


def make_balanced_trainer_cls(trainer_cls):
    class BalancedAccumulationTrainer(trainer_cls):
        def __init__(
            self,
            *args,
            balanced_accumulation_groups: list[str] | None = None,
            balanced_accumulation_effective_batch_size: int = 1,
            balanced_accumulation_seed: int = 42,
            min_learning_rate: float = 0.0,
            **kwargs,
        ):
            self._balanced_accumulation_groups = balanced_accumulation_groups
            self._balanced_accumulation_effective_batch_size = (
                balanced_accumulation_effective_batch_size
            )
            self._balanced_accumulation_seed = balanced_accumulation_seed
            self._min_learning_rate = min_learning_rate
            super().__init__(*args, **kwargs)

        def _get_train_sampler(self, *args, **kwargs):
            if self._balanced_accumulation_groups:
                if len(self._balanced_accumulation_groups) != len(self.train_dataset):
                    raise ValueError(
                        "balanced_accumulation_groups length must match train_dataset"
                    )
                return BalancedAccumulationSampler(
                    self._balanced_accumulation_groups,
                    effective_batch_size=(
                        self._balanced_accumulation_effective_batch_size
                    ),
                    seed=self._balanced_accumulation_seed,
                )
            return super()._get_train_sampler(*args, **kwargs)

        def create_scheduler(self, num_training_steps: int, optimizer=None):
            if self.lr_scheduler is None:
                from torch.optim.lr_scheduler import LambdaLR  # type: ignore

                optimizer = self.optimizer if optimizer is None else optimizer
                base_learning_rate = float(getattr(self.args, "learning_rate", 0.0))
                min_lr_ratio = (
                    self._min_learning_rate / base_learning_rate
                    if base_learning_rate > 0
                    else 0.0
                )
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
                self.lr_scheduler = LambdaLR(
                    optimizer,
                    lambda step: cosine_with_min_lr_lambda(
                        step,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_training_steps,
                        min_lr_ratio=min_lr_ratio,
                    ),
                )
            return self.lr_scheduler

    return BalancedAccumulationTrainer


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


def decision_point_prompt_text(tokenizer, example: Example) -> tuple[str, str]:
    """Build HuiKang-style decision-point curriculum masking.

    The chat template opens the assistant turn with ``<think>\n``. We then
    append the already-known partial trace to the prompt side so loss is only
    applied to the remaining continuation.
    """

    partial_trace = example.generated_cot.rstrip()
    completion_text = example.assistant_content.lstrip()
    if not partial_trace:
        raise SystemExit(f"id={example.id} has no decision-point partial trace")
    if not completion_text:
        raise SystemExit(f"id={example.id} has no decision-point completion content")

    prompt_text = build_competition_prompt(
        tokenizer,
        example.prompt,
        append_answer_instruction=example.append_answer_instruction,
    )
    if not prompt_text.rstrip().endswith("<think>"):
        raise SystemExit(
            f"id={example.id} decision-point prompt did not end with the template-opened <think>"
        )
    prompt_text += partial_trace + "\n"
    return prompt_text, completion_text


def tokenize_masked_example(
    tokenizer,
    example: Example,
    *,
    max_seq_len: int,
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
            f"id={example.id} has tokenizer boundary mismatch between prompt and completion; "
            "refusing to build labels because prompt masking would be shifted"
        )
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


def validate_tokenization_examples(tokenizer, examples: list[Example], *, max_seq_len: int) -> None:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for example in examples:
        try:
            tokenized = tokenize_masked_example(tokenizer, example, max_seq_len=max_seq_len)
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

    top_longest = sorted(rows, key=lambda row: int(row["total_tokens"]), reverse=True)[:20]
    summary = {
        "mode": "tokenization_validation",
        "rows_checked": len(examples),
        "rows_ok": len(rows),
        "failures": len(failures),
        "max_seq_len": max_seq_len,
        "prompt_format_counts": dict(
            sorted(Counter(example.prompt_format for example in examples).items())
        ),
        "top_longest": top_longest,
        "failure_samples": failures[:20],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(f"Tokenization validation failed for {len(failures)} rows")


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


def cosine_with_min_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
) -> float:
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    if current_step < num_warmup_steps:
        warmup_ratio = float(current_step) / float(max(1, num_warmup_steps))
        return max(min_lr_ratio, warmup_ratio)

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    progress = max(0.0, min(1.0, progress))
    cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_ratio


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
    output_dir, mirror_output_dir = resolve_output_and_mirror_dirs(
        Path(args.output_dir),
        args.mirror_output_dir,
    )
    summary = {
        "mode": "single_phase",
        "output_dir": str(output_dir.resolve()),
        "mirror_output_dir": None
        if mirror_output_dir is None
        else str(mirror_output_dir.resolve()),
        "trainer_state_dir": str(
            resolve_trainer_state_dir(output_dir, args.trainer_state_dir).resolve()
        ),
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
        "lr_scheduler_type": "cosine_with_min_lr",
        "warmup_ratio": 0.05,
        "min_learning_rate": args.min_learning_rate,
        "gradient_checkpointing": args.gradient_checkpointing,
        "balanced_accumulation": args.balanced_accumulation,
        "balanced_accumulation_group": "category" if args.balanced_accumulation else None,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "optim": args.optim,
        "loss_masking": "assistant_only",
        "prompt_format_counts": dict(
            sorted(Counter(example.prompt_format for example in train_examples).items())
        ),
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

    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments  # type: ignore
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
    balanced_groups = [example.category for example in train_examples] if args.balanced_accumulation else None
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
                "balanced_accumulation_groups": balanced_groups,
                "balanced_accumulation_effective_batch_size": (
                    args.per_device_train_batch_size * args.gradient_accumulation_steps
                ),
                "balanced_accumulation_seed": 42,
            }
        )
    trainer = trainer_cls(**trainer_kwargs)
    print(
        f"Starting single_phase: rows={len(train_examples)}, "
        f"splits={['ALL'] if args.train_all else args.train_splits}, "
        f"learning_rate={args.learning_rate}, "
        f"balanced_accumulation={args.balanced_accumulation}"
    )
    expected_adapter_dir = output_dir / "adapter"
    try:
        trainer.train()
    except OSError as exc:
        rescue_adapter_after_train_error(trainer.model, expected_adapter_dir, exc)
        raise

    adapter_dir = save_adapter_with_rescue(trainer.model, expected_adapter_dir)
    print(f"Single-phase adapter saved to: {adapter_dir}")
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
            "mode": "single_phase",
            "model_path": args.model_path,
            "adapter_dir": str(adapter_dir.resolve()),
            "expected_adapter_dir": str(expected_adapter_dir.resolve()),
            "mirror_output_dir": None
            if mirror_output_dir is None
            else str(mirror_output_dir.resolve()),
            "trainer_state_dir": str(trainer_state_dir.resolve()),
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
                "category" if args.balanced_accumulation else None
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
            "train_source_counts": source_counts(train_examples),
            "train_category_counts": summarize_categories(train_examples),
            "all_source_counts": source_counts(all_examples),
            "all_category_counts": summarize_categories(all_examples),
            "split_counts": split_counts(assignments),
        },
    )
    mirrored_dir = mirror_saved_outputs(metadata_dir, mirror_output_dir)
    print(f"Single-phase rows: {len(train_examples)}")
    print(f"Final adapter saved to: {adapter_dir}")
    if mirrored_dir is not None:
        print(f"Final artifacts mirrored to: {mirrored_dir}")
    print(f"Submission zip: {submission_path}")


if __name__ == "__main__":
    main()
