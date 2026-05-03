#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.numeric_equation import (
    classify_equation_vs_symbol,
    solve_numeric_equation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze numeric_equation coverage for the deterministic solver."
    )
    parser.add_argument(
        "--input-csv",
        default="data/train.csv",
        help="Path to the competition training CSV.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=8,
        help="How many solved and unsolved examples to print.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_csv)

    total = 0
    solved = 0
    confidence_counts: Counter[str] = Counter()
    query_example_counts: Counter[int] = Counter()
    solved_examples: list[tuple[str, str, str, str]] = []
    unsolved_examples: list[tuple[str, str, str | None, tuple[str, ...]]] = []

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            if "transformation rules" not in prompt.lower():
                continue
            if classify_equation_vs_symbol(prompt) != "numeric_equation":
                continue

            total += 1
            result = solve_numeric_equation(prompt)
            query_example_counts[result.query_example_count] += 1
            confidence_counts[result.confidence] += 1

            if result.prediction == row["answer"]:
                solved += 1
                if len(solved_examples) < args.show_examples:
                    solved_examples.append(
                        (
                            row["id"],
                            row["answer"],
                            result.confidence,
                            result.chosen_description or "",
                        )
                    )
            elif len(unsolved_examples) < args.show_examples:
                unsolved_examples.append(
                    (
                        row["id"],
                        row["answer"],
                        result.prediction,
                        result.notes,
                    )
                )

    print(f"Total numeric_equation rows: {total}")
    print(f"Solved exactly: {solved}")
    print(f"Coverage: {solved / max(1, total):.2%}")

    print("\nConfidence counts:")
    for confidence, count in sorted(confidence_counts.items()):
        print(f"  {confidence}: {count}")

    print("\nQuery operator example counts:")
    for seen_count, count in sorted(query_example_counts.items()):
        print(f"  seen {seen_count} time(s): {count}")

    print("\nSolved examples:")
    for row_id, answer, confidence, description in solved_examples:
        print(f"  {row_id}: answer={answer!r}, confidence={confidence}, rule={description}")

    print("\nUnsolved examples:")
    for row_id, answer, prediction, notes in unsolved_examples:
        print(
            f"  {row_id}: gold={answer!r}, prediction={prediction!r}, notes={'; '.join(notes) if notes else '-'}"
        )


if __name__ == "__main__":
    main()
