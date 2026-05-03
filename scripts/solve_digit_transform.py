#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nemotron_baseline.data import infer_category
from src.nemotron_baseline.digit_transform import solve_digit_transform
from src.nemotron_baseline.numeric_equation import classify_equation_vs_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve numeric-equation / digit-transform rows with a frequency-ordered "
            "scan over (pairing | operation | output format)."
        )
    )
    parser.add_argument("--input-csv", default="data/train.csv")
    parser.add_argument("--row-id", default=None, help="Solve one row by id.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap in batch mode.")
    parser.add_argument("--output-csv", default=None, help="Optional destination CSV for batch results.")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable exhaustive fallback and report scan-order results only.",
    )
    return parser.parse_args()


def load_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{input_csv} is missing a CSV header.")
        return list(reader)


def is_digit_transform_row(row: dict[str, str]) -> bool:
    prompt = row.get("prompt", "")
    if not prompt:
        return False
    try:
        category = infer_category(prompt)
    except ValueError:
        return False
    return category == "Transformation Rules" and classify_equation_vs_symbol(prompt) == "numeric_equation"


def iter_digit_transform_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if is_digit_transform_row(row)
    ]


def solve_one_row(row: dict[str, str], fallback_to_exhaustive: bool) -> dict[str, object]:
    result = solve_digit_transform(row["prompt"], fallback_to_exhaustive=fallback_to_exhaustive)
    return {
        "id": row.get("id", ""),
        "answer": row.get("answer", ""),
        "prediction": result.prediction or "",
        "correct": result.prediction == row.get("answer", ""),
        "confidence": result.confidence,
        "query_operator": result.query_operator,
        "query_example_count": result.query_example_count,
        "scan_match_rank": result.scan_match_rank or "",
        "chosen_combo": result.chosen_combo.label if result.chosen_combo is not None else "",
        "chosen_width": "" if result.chosen_width is None else result.chosen_width,
        "used_fallback": result.used_fallback,
        "chosen_description": result.chosen_description or "",
        "notes": " | ".join(result.notes),
    }


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    rows = load_rows(input_csv)
    digit_rows = iter_digit_transform_rows(rows)

    if args.row_id:
        for row in digit_rows:
            if row.get("id") == args.row_id:
                print(json.dumps(solve_one_row(row, fallback_to_exhaustive=not args.no_fallback), indent=2))
                return
        raise SystemExit(f"Could not find numeric-equation row id {args.row_id!r} in {input_csv}.")

    if args.max_rows is not None:
        digit_rows = digit_rows[: args.max_rows]

    solved_rows = [solve_one_row(row, fallback_to_exhaustive=not args.no_fallback) for row in digit_rows]
    correct_count = sum(1 for row in solved_rows if row["correct"])
    scan_only_count = sum(1 for row in solved_rows if not row["used_fallback"] and row["prediction"])
    scan_only_correct = sum(1 for row in solved_rows if not row["used_fallback"] and row["correct"])
    fallback_count = sum(1 for row in solved_rows if row["used_fallback"] and row["prediction"])
    fallback_correct = sum(1 for row in solved_rows if row["used_fallback"] and row["correct"])

    summary = {
        "input_csv": str(input_csv.resolve()),
        "examined_rows": len(solved_rows),
        "correct_rows": correct_count,
        "accuracy": (correct_count / len(solved_rows)) if solved_rows else 0.0,
        "scan_only_predictions": scan_only_count,
        "scan_only_correct_rows": scan_only_correct,
        "fallback_predictions": fallback_count,
        "fallback_correct_rows": fallback_correct,
        "fallback_enabled": not args.no_fallback,
    }
    print(json.dumps(summary, indent=2))

    if args.output_csv is None:
        return

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "answer",
        "prediction",
        "correct",
        "confidence",
        "query_operator",
        "query_example_count",
        "scan_match_rank",
        "chosen_combo",
        "chosen_width",
        "used_fallback",
        "chosen_description",
        "notes",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in solved_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
