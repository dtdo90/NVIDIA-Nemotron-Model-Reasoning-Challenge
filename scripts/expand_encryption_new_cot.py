#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.text_cipher import normalize_text_encryption_cot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add or refresh a generated_cot_explanded column in the Text Encryption CoT CSV "
            "using deterministic full mapping expansion."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="data/trainable/encryption_new_cot.csv",
        help="Source Text Encryption CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV. Defaults to updating the input file in place.",
    )
    parser.add_argument(
        "--output-column",
        default="generated_cot_explanded",
        help="Name of the expanded CoT column to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = ROOT / args.input_csv
    output_path = ROOT / args.output_csv if args.output_csv else input_path

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required_columns = {"id", "prompt", "answer", "generated_cot"}
        missing = required_columns - set(fieldnames)
        if missing:
            raise SystemExit(
                f"Text Encryption CSV is missing required columns: {sorted(missing)}"
            )

        if args.output_column not in fieldnames:
            fieldnames.append(args.output_column)

        rows: list[dict[str, str]] = []
        for row in reader:
            updated = dict(row)
            updated[args.output_column] = normalize_text_encryption_cot(
                row["prompt"],
                row.get("generated_cot"),
            )
            rows.append(updated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path == input_path:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            delete=False,
            dir=str(output_path.parent),
            suffix=".tmp",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            temp_path = Path(handle.name)
        temp_path.replace(output_path)
    else:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(
        f"Wrote {len(rows)} rows with {args.output_column!r} to {output_path}"
    )


if __name__ == "__main__":
    main()
