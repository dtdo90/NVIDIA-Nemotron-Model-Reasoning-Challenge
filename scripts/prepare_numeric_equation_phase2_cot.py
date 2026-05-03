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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from nemotron_baseline.numeric_equation import parse_numeric_equation_puzzle  # noqa: E402
from numeric_equation_detailed_cot import (  # noqa: E402
    parse_rule_label,
    render_detailed_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the extended numeric_equation CoT export from the cursor sandbox "
            "into data/trainable, preserving deterministic/oracle labels."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Competition train.csv used to validate prompt/answer alignment.",
    )
    parser.add_argument(
        "--source-csv",
        default="reference/cursor/transformation_rules/numeric_equation/results/labelled_cot.csv",
        help="Source labelled CoT CSV from the numeric_equation sandbox.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/numeric_equation_labelled_cot.csv",
        help="Destination CSV containing deterministic and oracle rows.",
    )
    parser.add_argument(
        "--deterministic-output-csv",
        default="data/trainable/numeric_equation_deterministic_cot.csv",
        help="Destination CSV containing deterministic rows only.",
    )
    return parser.parse_args()


def load_train_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "prompt", "answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        return {row["id"]: dict(row) for row in reader}


def main() -> None:
    args = parse_args()
    train_path = ROOT / args.train_csv
    source_path = ROOT / args.source_csv
    output_path = ROOT / args.output_csv
    deterministic_output_path = ROOT / args.deterministic_output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deterministic_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows = load_train_rows(train_path)
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {
            "id",
            "prompt",
            "answer",
            "generated_cot",
            "label",
            "category",
            "source",
            "rule_label",
            "source_tier",
            "confidence",
            "is_deterministic",
        }
        missing = required - set(fieldnames)
        if missing:
            raise SystemExit(f"{source_path} is missing required columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]

    counts: Counter[str] = Counter()
    deterministic_rows: list[dict[str, str]] = []
    for row in rows:
        row_id = row["id"].strip()
        if row_id not in train_rows:
            raise SystemExit(f"Numeric CoT row {row_id!r} is not present in {train_path}")
        train_row = train_rows[row_id]
        if row["prompt"] != train_row["prompt"]:
            raise SystemExit(f"Prompt mismatch for numeric CoT row {row_id}")
        if row["answer"].strip() != train_row["answer"].strip():
            raise SystemExit(
                f"Answer mismatch for numeric CoT row {row_id}: "
                f"{row['answer']!r} != {train_row['answer']!r}"
            )
        is_deterministic = row.get("is_deterministic", "").strip()
        if is_deterministic not in {"0", "1"}:
            raise SystemExit(f"Invalid is_deterministic={is_deterministic!r} for row {row_id}")
        puzzle = parse_numeric_equation_puzzle(row["prompt"])
        if puzzle is None:
            raise SystemExit(f"Could not parse numeric CoT row {row_id}")
        candidate = parse_rule_label(row["rule_label"])
        prediction = candidate.predict(
            puzzle.query.left_operand_text,
            puzzle.query.right_operand_text,
            puzzle.query.operator,
        )
        if prediction != row["answer"].strip():
            raise SystemExit(
                f"Rule {row['rule_label']!r} predicts {prediction!r} for row {row_id}, "
                f"expected {row['answer']!r}"
            )
        row["generated_cot"] = render_detailed_trace(
            puzzle=puzzle,
            candidate=candidate,
            is_deterministic=is_deterministic == "1",
            source_tier=row.get("source_tier", ""),
        )
        counts[f"is_deterministic={is_deterministic}"] += 1
        counts[row.get("source_tier", "").strip()] += 1
        if is_deterministic == "1":
            deterministic_rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with deterministic_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deterministic_rows)

    print(f"Wrote {len(rows)} numeric_equation labelled rows -> {output_path}")
    print(
        f"Wrote {len(deterministic_rows)} deterministic numeric_equation rows "
        f"-> {deterministic_output_path}"
    )
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
