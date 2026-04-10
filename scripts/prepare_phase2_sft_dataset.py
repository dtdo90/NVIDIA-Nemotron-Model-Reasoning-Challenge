#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import infer_category, load_split_assignments, select_ids_for_splits
from nemotron_baseline.prompts import normalize_generated_cot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the phase-2 SFT dataset by anchoring on train.csv, overlaying the main "
            "competition CoT dataset, and replacing Text Cipher rows with doc2lora "
            "encryption CoT for the selected split ids."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Full competition train.csv used as the source of truth for prompts and answers.",
    )
    parser.add_argument(
        "--base-cot-csv",
        default=None,
        help=(
            "Optional cleaned competition CoT CSV. If omitted, the script will auto-detect "
            "a likely cleaned file if present."
        ),
    )
    parser.add_argument(
        "--text-cot-csv",
        default="data/encryption_new_cot.csv",
        help="Text Encryption CoT dataset from doc2lora.",
    )
    parser.add_argument(
        "--split-csv",
        default="data/splits_70_15_15.config.json",
        help="Split assignment file (CSV or JSON split config).",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["sft_train"],
        help="Split names to include in the phase-2 SFT file.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/train_sft_phase2_70_15_15.csv",
        help="Destination CSV in SFT schema.",
    )
    parser.add_argument(
        "--cot-column",
        default="generated_cot",
        help="CoT column to read from the base CoT CSV.",
    )
    return parser.parse_args()


def resolve_base_cot_path(explicit_path: str | None) -> Path | None:
    if explicit_path:
        path = ROOT / explicit_path
        if not path.exists():
            raise SystemExit(f"Base CoT CSV not found: {path}")
        return path

    candidates = [
        ROOT / "data/train_cot_gpt_oss_clean.csv",
        ROOT / "data/train_cot_clean.csv",
        ROOT / "data/train_cot.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def wrap_reasoning_as_think_block(reasoning: str | None) -> str:
    normalized = normalize_generated_cot(reasoning)
    if not normalized:
        return ""
    if normalized.startswith("<think>") and "</think>" in normalized:
        return normalized
    return f"<think>\n{normalized}\n</think>"


def load_rows_by_id(csv_path: Path, *, cot_column: str) -> dict[str, dict[str, str]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "id" not in fieldnames:
            raise SystemExit(f"CSV is missing required column 'id': {csv_path}")
        for row in reader:
            row_id = row["id"].strip()
            if not row_id:
                continue
            copied = dict(row)
            if cot_column in fieldnames and cot_column != "generated_cot":
                copied["generated_cot"] = row.get(cot_column, "")
            rows_by_id[row_id] = copied
    return rows_by_id


def load_train_rows(train_csv: Path) -> dict[str, dict[str, str]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    with train_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"id", "prompt", "answer"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Train CSV is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            rows_by_id[row["id"]] = {
                "id": row["id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
            }
    return rows_by_id


def choose_label(base_row: dict[str, str] | None, category: str, source: str) -> str:
    if source == "doc2lora_text_cot":
        return "Text Cipher"
    if base_row and base_row.get("label"):
        return base_row["label"]
    return category


def main() -> None:
    args = parse_args()
    train_path = ROOT / args.train_csv
    text_cot_path = ROOT / args.text_cot_csv
    split_path = ROOT / args.split_csv
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_cot_path = resolve_base_cot_path(args.base_cot_csv)
    if base_cot_path is None:
        print("No cleaned base CoT CSV found; using answer-only fallback except for doc2lora text rows.")

    train_rows = load_train_rows(train_path)
    split_assignments = load_split_assignments(split_path)
    selected_ids = select_ids_for_splits(split_assignments, args.train_splits)
    text_rows = load_rows_by_id(text_cot_path, cot_column="generated_cot")
    base_rows = (
        load_rows_by_id(base_cot_path, cot_column=args.cot_column)
        if base_cot_path is not None
        else {}
    )

    output_rows: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()

    for row_id in sorted(selected_ids):
        if row_id not in train_rows:
            raise SystemExit(f"Split id {row_id!r} does not exist in {train_path}")

        train_row = train_rows[row_id]
        category = infer_category(train_row["prompt"])
        base_row = base_rows.get(row_id)
        text_row = text_rows.get(row_id)

        chosen_cot = ""
        source = "answer_only"
        if base_row and base_row.get("generated_cot", "").strip():
            chosen_cot = base_row["generated_cot"]
            source = "base_cot"
        if category == "Text Cipher" and text_row and text_row.get("generated_cot", "").strip():
            if text_row.get("answer", "").strip() != train_row["answer"].strip():
                raise SystemExit(
                    f"Answer mismatch for text row {row_id}: "
                    f"{text_row.get('answer')!r} != {train_row['answer']!r}"
                )
            chosen_cot = text_row["generated_cot"]
            source = "doc2lora_text_cot"

        output_rows.append(
            {
                "id": row_id,
                "prompt": train_row["prompt"],
                "answer": train_row["answer"],
                "generated_cot": wrap_reasoning_as_think_block(chosen_cot),
                "label": choose_label(base_row, category, source),
                "category": category,
                "source": source,
            }
        )
        source_counts[source] += 1

    fieldnames = ["id", "prompt", "answer", "generated_cot", "label", "category", "source"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} phase-2 SFT rows to {output_path}")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
