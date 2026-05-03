#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import infer_category


QUERY_RE = re.compile(
    r"write the number\s+([0-9]+)\s+in the Wonderland numeral system",
    re.I,
)

ROMAN_VALUES: list[tuple[int, str]] = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


@dataclass(frozen=True)
class NumeralRow:
    id: str
    prompt: str
    answer: str
    generated_cot: str
    label: str
    category: str
    source: str
    source_mode: str
    target_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Numeral System Phase 2 CoT rows from train.csv. "
            "Rows adapt Huikang's greedy Arabic-to-Roman conversion trace while using "
            "our normalized boxed-answer wording."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Competition train.csv containing id, prompt, and answer.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/numeral_cot.csv",
        help="Destination CSV for Numeral System CoT rows.",
    )
    parser.add_argument(
        "--method-resolved-output-csv",
        default="data/trainable/numeral_cot_method_resolved.csv",
        help="Destination CSV for rows solved by greedy Roman conversion.",
    )
    return parser.parse_args()


def parse_numeral_prompt(prompt: str) -> int:
    match = QUERY_RE.search(prompt)
    if match is None:
        raise ValueError("Failed to parse numeral-system target number.")
    return int(match.group(1))


def to_roman_parts(number: int) -> list[tuple[int, int, str, int]]:
    if number <= 0:
        raise ValueError(f"Roman conversion expects a positive integer, got {number}.")

    remaining = number
    steps: list[tuple[int, int, str, int]] = []
    for value, symbol in ROMAN_VALUES:
        while remaining >= value:
            next_remaining = remaining - value
            steps.append((remaining, value, symbol, next_remaining))
            remaining = next_remaining
    return steps


def to_roman(number: int) -> str:
    return "".join(symbol for _before, _value, symbol, _after in to_roman_parts(number))


def render_trace(target_number: int) -> str:
    steps = to_roman_parts(target_number)
    computed = "".join(symbol for _before, _value, symbol, _after in steps)
    spaced = " ".join(symbol for _before, _value, symbol, _after in steps)

    lines = [
        (
            "This is an Arabic to Roman numeral conversion. I will use the "
            "standard greedy Roman numeral table."
        ),
        "I will put my final answer inside \\boxed{}.",
        "",
        f"Converting {target_number}:",
    ]
    for before, value, symbol, after in steps:
        lines.append(f"  {before} >= {value} -> {symbol}, remainder {after}")
    lines.extend(
        [
            "",
            f"Result: {spaced} -> {computed}",
            "",
            "I will now return the answer in \\boxed{}",
            f"The final answer is \\boxed{{{computed}}}",
        ]
    )
    return "\n".join(lines)


def build_rows(train_csv: Path) -> tuple[list[NumeralRow], Counter[str]]:
    rows: list[NumeralRow] = []
    mode_counts: Counter[str] = Counter()
    with train_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            if infer_category(prompt) != "Numeral System":
                continue

            target_number = parse_numeral_prompt(prompt)
            answer = to_roman(target_number)
            gold_answer = row["answer"].strip()
            if answer != gold_answer:
                raise SystemExit(
                    f"Numeral answer mismatch for {row['id']}: "
                    f"computed {answer!r}, gold {gold_answer!r}"
                )

            mode = "greedy_roman_conversion"
            mode_counts[mode] += 1
            rows.append(
                NumeralRow(
                    id=row["id"],
                    prompt=prompt,
                    answer=answer,
                    generated_cot=render_trace(target_number),
                    label="Numeral System",
                    category="Numeral System",
                    source="deterministic_numeral_cot",
                    source_mode=mode,
                    target_number=target_number,
                )
            )
    return rows, mode_counts


def main() -> None:
    args = parse_args()
    train_path = ROOT / args.train_csv
    output_path = ROOT / args.output_csv
    resolved_path = ROOT / args.method_resolved_output_csv
    rows, mode_counts = build_rows(train_path)

    fieldnames = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "source_mode",
        "target_number",
    ]
    for path, selected_rows in ((output_path, rows), (resolved_path, rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row.__dict__ for row in selected_rows)

    print(f"Wrote {len(rows)} Numeral System CoT rows to {output_path}")
    print(f"  method_resolved: {len(rows)} -> {resolved_path}")
    for key in sorted(mode_counts):
        print(f"  {key}: {mode_counts[key]}")


if __name__ == "__main__":
    main()
