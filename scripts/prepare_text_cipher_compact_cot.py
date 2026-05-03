#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import infer_category
from nemotron_baseline.text_cipher import (
    DEFAULT_WONDERLAND_VOCABULARY,
    render_text_cipher_compact_cot,
    solve_text_cipher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact deterministic CoT traces for all Text Cipher rows."
    )
    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Competition train.csv containing id, prompt, and answer columns.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/text_cipher_compact_cot.csv",
        help="Destination CSV with compact deterministic Text Cipher traces.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = ROOT / args.train_csv
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    mismatches: list[tuple[str, str, str]] = []

    with train_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "prompt", "answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{train_path} is missing required columns: {sorted(missing)}")

        for row in reader:
            if infer_category(row["prompt"]) != "Text Cipher":
                continue

            solution = solve_text_cipher(
                row["prompt"],
                vocabulary=DEFAULT_WONDERLAND_VOCABULARY,
            )
            generated_cot = render_text_cipher_compact_cot(
                row["prompt"],
                vocabulary=DEFAULT_WONDERLAND_VOCABULARY,
            )

            expected = row["answer"].strip()
            predicted = solution.decoded_phrase.strip()
            if predicted != expected or not solution.unique:
                mismatches.append((row["id"], expected, predicted))

            rows.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "generated_cot": generated_cot,
                    "type": "Text Encryption",
                    "source": "deterministic_text_cipher_compact",
                }
            )

    if mismatches:
        print(f"Found {len(mismatches)} Text Cipher rows that did not solve cleanly:")
        for row_id, expected, predicted in mismatches[:20]:
            print(f"  {row_id}: expected={expected!r} predicted={predicted!r}")
        raise SystemExit(1)

    fieldnames = ["id", "prompt", "answer", "generated_cot", "type", "source"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lengths = [len(row["generated_cot"]) for row in rows]
    print(f"Wrote {len(rows)} Text Cipher CoT rows to {output_path}")
    print(f"Wonderland vocabulary size: {len(DEFAULT_WONDERLAND_VOCABULARY)}")
    print(
        "Generated CoT chars: "
        f"min={min(lengths)} median={int(statistics.median(lengths))} "
        f"mean={int(statistics.mean(lengths))} max={max(lengths)}"
    )


if __name__ == "__main__":
    main()
