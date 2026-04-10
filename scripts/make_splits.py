#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import (
    load_examples,
    stratified_partition,
    summarize_categories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a persistent stratified train split file."
    )
    parser.add_argument("--input-csv", default="data/train.csv")
    parser.add_argument("--output-csv", default="data/splits_70_15_15.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-fraction", type=float, default=0.70)
    parser.add_argument("--grpo-fraction", type=float, default=0.15)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    split_fractions = {
        "sft_train": args.sft_fraction,
        "grpo_train": args.grpo_fraction,
        "eval": args.eval_fraction,
    }

    examples = load_examples(args.input_csv)
    partitions = stratified_partition(examples, split_fractions, seed=args.seed)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "category", "split"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for split_name, split_examples in partitions.items():
            for example in sorted(split_examples, key=lambda item: item.id):
                writer.writerow(
                    {
                        "id": example.id,
                        "category": example.category,
                        "split": split_name,
                    }
                )

    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "output_csv": str(output_path.resolve()),
        "split_config_json": str(output_path.with_suffix(".config.json").resolve()),
        "seed": args.seed,
        "fractions": split_fractions,
        "total_examples": len(examples),
        "split_sizes": {
            split_name: len(split_examples)
            for split_name, split_examples in partitions.items()
        },
        "split_category_counts": {
            split_name: summarize_categories(split_examples)
            for split_name, split_examples in partitions.items()
        },
    }
    write_json(output_path.with_suffix(".summary.json"), summary)

    split_config = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "output_csv": str(output_path.resolve()),
        "seed": args.seed,
        "fractions": split_fractions,
        "split_ids": {
            split_name: [example.id for example in sorted(split_examples, key=lambda item: item.id)]
            for split_name, split_examples in partitions.items()
        },
    }
    write_json(output_path.with_suffix(".config.json"), split_config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
