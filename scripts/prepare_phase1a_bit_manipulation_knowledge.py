#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "id",
    "prompt",
    "answer",
    "generated_cot",
    "label",
    "category",
    "source",
    "source_category",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt the cursor bit-manipulation Phase 1A knowledge cards into "
            "data/trainable/ using the same schema as the other Phase 1A files."
        )
    )
    parser.add_argument(
        "--source-csv",
        default=(
            "reference/cursor/bit_manipulation/results/"
            "phase1a_bit_manipulation_knowledge.csv"
        ),
        help="Cursor sandbox Phase 1A bit knowledge CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1a_bit_manipulation_knowledge.csv",
        help="Destination CSV.",
    )
    return parser.parse_args()


def load_and_validate(source_path: Path) -> list[dict[str, str]]:
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise SystemExit(f"{source_path} is missing required columns: {sorted(missing)}")

        rows: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            row_id = row["id"].strip()
            if not row_id:
                raise SystemExit(f"{source_path}:{row_number} has an empty id.")
            if row_id in seen_ids:
                raise SystemExit(f"{source_path}:{row_number} duplicates id {row_id!r}.")
            seen_ids.add(row_id)

            for column in REQUIRED_COLUMNS:
                if not row.get(column, "").strip():
                    raise SystemExit(
                        f"{source_path}:{row_number} has empty required column {column!r}."
                    )
            rows.append({column: row[column] for column in reader.fieldnames or []})
    return rows


def main() -> None:
    args = parse_args()
    source_path = ROOT / args.source_csv
    output_path = ROOT / args.output_csv
    if not source_path.exists():
        raise SystemExit(f"Source CSV not found: {source_path}")

    rows = load_and_validate(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "source_category",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    print(f"Wrote {len(rows)} Phase 1A bit-manipulation rows to {output_path}")


if __name__ == "__main__":
    main()
