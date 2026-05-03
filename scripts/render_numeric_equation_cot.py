#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import infer_category
from nemotron_baseline.numeric_equation import (
    classify_equation_vs_symbol,
    render_numeric_equation_trace,
    solve_numeric_equation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render deterministic, high-confidence reasoning traces for Transformation Rules "
            "rows in the numeric_equation subtype."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="data/train.csv",
        help="Source CSV containing competition rows.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/numeric_equation_solver_high_conf.csv",
        help="Destination CSV for accepted high-confidence rows.",
    )
    parser.add_argument(
        "--skipped-csv",
        default="data/trainable/numeric_equation_solver_high_conf_skipped.csv",
        help="Destination CSV for skipped rows and reasons.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of numeric_equation rows processed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = ROOT / args.input_csv
    output_path = ROOT / args.output_csv
    skipped_path = ROOT / args.skipped_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)

    accepted_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []

    processed = 0
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            if "transformation rules" not in prompt.lower():
                continue
            if classify_equation_vs_symbol(prompt) != "numeric_equation":
                continue
            processed += 1
            if args.max_rows is not None and processed > args.max_rows:
                break

            result = solve_numeric_equation(prompt)
            trace = render_numeric_equation_trace(prompt, result)
            category = infer_category(prompt)

            if result.prediction == row["answer"] and result.confidence == "high" and trace:
                accepted_rows.append(
                    {
                        "id": row["id"],
                        "prompt": prompt,
                        "answer": row["answer"],
                        "generated_cot": trace,
                        "label": category,
                        "category": category,
                        "source": "numeric_equation_solver",
                        "solver_confidence": result.confidence,
                        "solver_prediction": result.prediction,
                        "query_operator": result.query_operator,
                        "query_example_count": str(result.query_example_count),
                        "query_candidate_count": str(result.query_candidate_count),
                    }
                )
            else:
                skipped_rows.append(
                    {
                        "id": row["id"],
                        "prompt": prompt,
                        "answer": row["answer"],
                        "solver_prediction": "" if result.prediction is None else result.prediction,
                        "solver_confidence": result.confidence,
                        "query_operator": result.query_operator,
                        "query_example_count": str(result.query_example_count),
                        "query_candidate_count": str(result.query_candidate_count),
                        "skip_reason": "; ".join(result.notes) if result.notes else "Not exact high-confidence numeric_equation row.",
                    }
                )

    output_fields = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "solver_confidence",
        "solver_prediction",
        "query_operator",
        "query_example_count",
        "query_candidate_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(accepted_rows)

    skipped_fields = [
        "id",
        "prompt",
        "answer",
        "solver_prediction",
        "solver_confidence",
        "query_operator",
        "query_example_count",
        "query_candidate_count",
        "skip_reason",
    ]
    with skipped_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=skipped_fields)
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"Processed {processed} numeric_equation rows from {input_path}")
    print(f"Accepted {len(accepted_rows)} high-confidence exact rows -> {output_path}")
    print(f"Skipped {len(skipped_rows)} rows -> {skipped_path}")


if __name__ == "__main__":
    main()
