#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert doc2lora knowledge QA rows into the repo's SFT CSV schema."
    )
    parser.add_argument(
        "--input-csv",
        default="data/knowledge_qa.csv",
        help="Source knowledge QA CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/text_knowledge_phase1.csv",
        help="Destination CSV in SFT schema.",
    )
    parser.add_argument(
        "--label",
        default="Text Cipher Knowledge",
        help="Label/category name to attach to generated rows.",
    )
    return parser.parse_args()


def wrap_reasoning_as_think_block(reasoning: str) -> str:
    cleaned = (reasoning or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("<think>") and "</think>" in cleaned:
        return cleaned
    return f"<think>\n{cleaned}\n</think>"


def main() -> None:
    args = parse_args()
    input_path = ROOT / args.input_csv
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"question", "answer", "cot", "category"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Knowledge CSV is missing required columns: {sorted(missing)}"
            )

        for index, row in enumerate(reader, start=1):
            rows.append(
                {
                    "id": f"text_knowledge_{index:05d}",
                    "prompt": row["question"].strip(),
                    "answer": row["answer"].strip(),
                    "generated_cot": wrap_reasoning_as_think_block(row["cot"]),
                    "label": args.label,
                    "category": args.label,
                    "source": "doc2lora_knowledge_qa",
                    "source_category": row["category"].strip(),
                }
            )

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
        writer.writerows(rows)

    print(f"Wrote {len(rows)} phase-1 knowledge rows to {output_path}")


if __name__ == "__main__":
    main()
