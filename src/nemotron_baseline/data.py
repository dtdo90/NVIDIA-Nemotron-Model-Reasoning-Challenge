from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CATEGORY_BY_FIRST_SENTENCE = {
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.": "Bit Manipulation",
    "In Alice's Wonderland, the gravitational constant has been secretly changed.": "Gravity",
    "In Alice's Wonderland, a secret unit conversion is applied to measurements.": "Unit Conversion",
    "In Alice's Wonderland, secret encryption rules are used on text.": "Text Cipher",
    "In Alice's Wonderland, numbers are secretly converted into a different numeral system.": "Numeral System",
    "In Alice's Wonderland, a secret set of transformation rules is applied to equations.": "Transformation Rules",
}


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    answer: str | None
    category: str


def normalize_first_sentence(prompt: str) -> str:
    text = re.sub(r"\s+", " ", prompt.replace("\n", " ")).strip()
    match = re.match(r"(.+?[.!?])(?:\s|$)", text)
    return match.group(1) if match else text


def infer_category(prompt: str) -> str:
    first_sentence = normalize_first_sentence(prompt)
    try:
        return CATEGORY_BY_FIRST_SENTENCE[first_sentence]
    except KeyError as exc:
        raise ValueError(
            f"Unrecognized prompt template: {first_sentence!r}"
        ) from exc


def load_examples(csv_path: str | Path) -> list[Example]:
    path = Path(csv_path)
    examples: list[Example] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            answer = row.get("answer")
            examples.append(
                Example(
                    id=row["id"],
                    prompt=row["prompt"],
                    answer=answer if answer not in (None, "") else None,
                    category=infer_category(row["prompt"]),
                )
            )
    return examples


def summarize_categories(examples: Iterable[Example]) -> dict[str, int]:
    counts = {category: 0 for category in CATEGORY_BY_FIRST_SENTENCE.values()}
    for example in examples:
        counts[example.category] = counts.get(example.category, 0) + 1
    return counts


def stratified_partition(
    examples: list[Example],
    split_fractions: dict[str, float],
    *,
    seed: int = 42,
) -> dict[str, list[Example]]:
    if not split_fractions:
        raise ValueError("split_fractions must not be empty.")

    total_fraction = sum(split_fractions.values())
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError("split_fractions must sum to 1.0.")

    rng = random.Random(seed)
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        grouped[example.category].append(example)

    split_names = list(split_fractions)
    partitions = {name: [] for name in split_names}

    for category_examples in grouped.values():
        shuffled = list(category_examples)
        rng.shuffle(shuffled)

        raw_counts = {
            name: len(shuffled) * split_fractions[name]
            for name in split_names
        }
        counts = {name: int(raw_counts[name]) for name in split_names}
        remainder = len(shuffled) - sum(counts.values())
        fractional_order = sorted(
            split_names,
            key=lambda name: (raw_counts[name] - counts[name], split_names.index(name)),
            reverse=True,
        )
        for name in fractional_order[:remainder]:
            counts[name] += 1

        start = 0
        for name in split_names:
            end = start + counts[name]
            partitions[name].extend(shuffled[start:end])
            start = end

    for split_examples in partitions.values():
        rng.shuffle(split_examples)
    return partitions


def stratified_split(
    examples: list[Example],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0).")

    if val_fraction == 0.0:
        return list(examples), []

    partitions = stratified_partition(
        examples,
        {"val": val_fraction, "train": 1.0 - val_fraction},
        seed=seed,
    )
    train_examples = partitions["train"]
    val_examples = partitions["val"]
    if not train_examples or not val_examples:
        raise ValueError("Validation split produced an empty train or validation set.")
    return train_examples, val_examples


def load_split_assignments(csv_path: str | Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    path = Path(csv_path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        split_ids = None
        if isinstance(payload, dict):
            if isinstance(payload.get("split_ids"), dict):
                split_ids = payload["split_ids"]
            elif isinstance(payload.get("splits"), dict):
                split_ids = payload["splits"]
        if split_ids is None:
            raise ValueError(
                "Split JSON must contain a 'split_ids' or 'splits' object mapping split names to ids."
            )
        for split_name, row_ids in split_ids.items():
            if not isinstance(split_name, str) or not split_name.strip():
                raise ValueError(f"Split JSON has an invalid split name: {split_name!r}")
            if not isinstance(row_ids, list):
                raise ValueError(f"Split JSON expected a list of ids for split {split_name!r}")
            for row_id in row_ids:
                if not isinstance(row_id, str) or not row_id.strip():
                    raise ValueError(
                        f"Split JSON has an invalid row id {row_id!r} in split {split_name!r}"
                    )
                if row_id in assignments:
                    raise ValueError(f"Split JSON has duplicate row id: {row_id!r}")
                assignments[row_id] = split_name.strip()
        return assignments

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Split CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            row_id = row["id"]
            split_name = row["split"].strip()
            if not split_name:
                raise ValueError(f"Split CSV has an empty split name for id={row_id!r}")
            if row_id in assignments:
                raise ValueError(f"Split CSV has duplicate row id: {row_id!r}")
            assignments[row_id] = split_name
    return assignments


def select_ids_for_splits(
    assignments: dict[str, str],
    split_names: Iterable[str],
) -> set[str]:
    wanted = {name for name in split_names if name}
    return {
        example_id
        for example_id, split_name in assignments.items()
        if split_name in wanted
    }


def summarize_split_assignments(assignments: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name in assignments.values():
        counts[split_name] = counts.get(split_name, 0) + 1
    return dict(sorted(counts.items()))
