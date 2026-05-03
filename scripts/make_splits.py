#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import infer_category
from nemotron_baseline.numeric_equation import classify_equation_vs_symbol


@dataclass(frozen=True)
class SplitRow:
    id: str
    category: str
    subcategory: str
    solve_source: str
    stratum: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a persistent train split stratified by category, useful "
            "subcategory, and current deterministic solve coverage."
        )
    )
    parser.add_argument("--input-csv", default="data/train.csv")
    parser.add_argument("--output-csv", default="data/splits_75_10_15.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-fraction", type=float, default=0.75)
    parser.add_argument("--grpo-fraction", type=float, default=0.10)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument(
        "--text-cot-csv",
        default="data/trainable/text_cipher_compact_cot.csv",
        help="Text Cipher deterministic CoT rows used to mark solved text rows.",
    )
    parser.add_argument(
        "--bit-cot-csv",
        default="data/trainable/bit_manipulation_hybrid_cot.csv",
        help="Bit Manipulation deterministic CoT rows used to mark solved bit rows.",
    )
    parser.add_argument(
        "--unit-cot-csv",
        default="data/trainable/unit_conversion_cot_method_resolved.csv",
        help="Unit Conversion deterministic CoT rows used to mark solved unit rows.",
    )
    parser.add_argument(
        "--gravity-cot-csv",
        default="data/trainable/gravity_cot_method_resolved.csv",
        help="Gravity deterministic CoT rows used to mark solved gravity rows.",
    )
    parser.add_argument(
        "--numeral-cot-csv",
        default="data/trainable/numeral_cot_method_resolved.csv",
        help="Numeral deterministic CoT rows used to mark solved numeral rows.",
    )
    parser.add_argument(
        "--numeric-equation-cot-csv",
        default="data/trainable/numeric_equation_labelled_cot.csv",
        help="numeric_equation deterministic CoT rows used to mark solved rows.",
    )
    parser.add_argument(
        "--symbol-transform-cot-csv",
        default="data/trainable/symbol_transform_phase2_combined.csv",
        help=(
            "Symbol Transform CoT rows used only for split stratification. "
            "Only rows with data_type=real are counted as solved."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_id_set(path_text: str, *, require_real: bool = False) -> set[str]:
    path = ROOT / path_text
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} is missing required column 'id'.")
        for row in reader:
            if require_real and row.get("data_type", "").strip() != "real":
                continue
            row_id = row.get("id", "").strip()
            if row_id:
                ids.add(row_id)
    return ids


def infer_subcategory(prompt: str, category: str) -> str:
    if category == "Transformation Rules":
        return classify_equation_vs_symbol(prompt)
    return "all"


def solve_source_for_row(
    row_id: str,
    *,
    category: str,
    subcategory: str,
    solved_ids: dict[str, set[str]],
) -> str:
    if category == "Text Cipher" and row_id in solved_ids["text"]:
        return "text_cipher_compact_cot"
    if category == "Bit Manipulation" and row_id in solved_ids["bit"]:
        return "bit_manipulation_hybrid_cot"
    if category == "Unit Conversion" and row_id in solved_ids["unit"]:
        return "unit_conversion_method_resolved"
    if category == "Gravity" and row_id in solved_ids["gravity"]:
        return "gravity_weighted_cot"
    if category == "Numeral System" and row_id in solved_ids["numeral"]:
        return "numeral_greedy_roman_cot"
    if (
        category == "Transformation Rules"
        and subcategory == "numeric_equation"
        and row_id in solved_ids["numeric_equation"]
    ):
        return "numeric_equation_solver"
    if (
        category == "Transformation Rules"
        and subcategory == "symbol_transform"
        and row_id in solved_ids["symbol_transform"]
    ):
        return "symbol_transform_solver"
    return "answer_only"


def load_split_rows(args: argparse.Namespace) -> list[SplitRow]:
    solved_ids = {
        "text": read_id_set(args.text_cot_csv),
        "bit": read_id_set(args.bit_cot_csv),
        "unit": read_id_set(args.unit_cot_csv),
        "gravity": read_id_set(args.gravity_cot_csv),
        "numeral": read_id_set(args.numeral_cot_csv),
        "numeric_equation": read_id_set(args.numeric_equation_cot_csv),
        "symbol_transform": read_id_set(args.symbol_transform_cot_csv, require_real=True),
    }

    input_path = ROOT / args.input_csv
    rows: list[SplitRow] = []
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"id", "prompt"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{input_path} is missing required columns: {sorted(missing)}")
        for row in reader:
            row_id = row["id"].strip()
            category = infer_category(row["prompt"])
            subcategory = infer_subcategory(row["prompt"], category)
            solve_source = solve_source_for_row(
                row_id,
                category=category,
                subcategory=subcategory,
                solved_ids=solved_ids,
            )
            stratum = f"{category}::{subcategory}::{solve_source}"
            rows.append(
                SplitRow(
                    id=row_id,
                    category=category,
                    subcategory=subcategory,
                    solve_source=solve_source,
                    stratum=stratum,
                )
            )
    return rows


def allocate_counts(total: int, split_fractions: dict[str, float]) -> dict[str, int]:
    raw_counts = {
        split_name: total * fraction for split_name, fraction in split_fractions.items()
    }
    counts = {split_name: int(raw_counts[split_name]) for split_name in split_fractions}
    remainder = total - sum(counts.values())
    order = sorted(
        split_fractions,
        key=lambda split_name: (
            raw_counts[split_name] - counts[split_name],
            -list(split_fractions).index(split_name),
        ),
        reverse=True,
    )
    for split_name in order[:remainder]:
        counts[split_name] += 1
    return counts


def stratified_partition_rows(
    rows: list[SplitRow],
    split_fractions: dict[str, float],
    *,
    seed: int,
) -> dict[str, list[SplitRow]]:
    rng = random.Random(seed)
    grouped: dict[str, list[SplitRow]] = defaultdict(list)
    for row in rows:
        grouped[row.stratum].append(row)

    split_names = list(split_fractions)
    partitions: dict[str, list[SplitRow]] = {split_name: [] for split_name in split_names}
    for stratum_rows in grouped.values():
        shuffled = list(stratum_rows)
        rng.shuffle(shuffled)
        counts = allocate_counts(len(shuffled), split_fractions)
        start = 0
        for split_name in split_names:
            end = start + counts[split_name]
            partitions[split_name].extend(shuffled[start:end])
            start = end

    for split_rows in partitions.values():
        rng.shuffle(split_rows)
    rebalance_global_counts(partitions, split_fractions)
    return partitions


def rebalance_global_counts(
    partitions: dict[str, list[SplitRow]],
    split_fractions: dict[str, float],
) -> None:
    """Keep exact global split sizes after independent per-stratum rounding."""
    total = sum(len(rows) for rows in partitions.values())
    targets = allocate_counts(total, split_fractions)
    stratum_totals = Counter(
        row.stratum for split_rows in partitions.values() for row in split_rows
    )

    def choose_move(over_split: str, under_split: str) -> SplitRow:
        over_counts = Counter(row.stratum for row in partitions[over_split])
        under_counts = Counter(row.stratum for row in partitions[under_split])
        scored: list[tuple[float, str, SplitRow]] = []
        for row in partitions[over_split]:
            stratum_total = stratum_totals[row.stratum]
            over_ideal = stratum_total * split_fractions[over_split]
            under_ideal = stratum_total * split_fractions[under_split]
            over_surplus = over_counts[row.stratum] - over_ideal
            under_deficit = under_ideal - under_counts[row.stratum]
            scored.append((over_surplus + under_deficit, row.id, row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]

    while True:
        over = [
            split_name
            for split_name, split_rows in partitions.items()
            if len(split_rows) > targets[split_name]
        ]
        under = [
            split_name
            for split_name, split_rows in partitions.items()
            if len(split_rows) < targets[split_name]
        ]
        if not over and not under:
            return
        if not over or not under:
            raise RuntimeError("Split rebalance reached an impossible state.")
        over_split = max(over, key=lambda split_name: len(partitions[split_name]) - targets[split_name])
        under_split = max(under, key=lambda split_name: targets[split_name] - len(partitions[split_name]))
        row = choose_move(over_split, under_split)
        partitions[over_split].remove(row)
        partitions[under_split].append(row)


def summarize_by(rows: list[SplitRow], attr_name: str) -> dict[str, int]:
    return dict(sorted(Counter(getattr(row, attr_name) for row in rows).items()))


def main() -> int:
    args = parse_args()
    split_fractions = {
        "sft_train": args.sft_fraction,
        "grpo_train": args.grpo_fraction,
        "eval": args.eval_fraction,
    }

    total_fraction = sum(split_fractions.values())
    if abs(total_fraction - 1.0) > 1e-9:
        raise SystemExit(f"Split fractions must sum to 1.0, got {total_fraction}.")

    examples = load_split_rows(args)
    partitions = stratified_partition_rows(examples, split_fractions, seed=args.seed)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "category",
                "subcategory",
                "solve_source",
                "stratum",
                "split",
            ],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for split_name, split_examples in partitions.items():
            for example in sorted(split_examples, key=lambda item: item.id):
                writer.writerow(
                    {
                        "id": example.id,
                        "category": example.category,
                        "subcategory": example.subcategory,
                        "solve_source": example.solve_source,
                        "stratum": example.stratum,
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
            split_name: summarize_by(split_examples, "category")
            for split_name, split_examples in partitions.items()
        },
        "split_subcategory_counts": {
            split_name: summarize_by(split_examples, "subcategory")
            for split_name, split_examples in partitions.items()
        },
        "split_solve_source_counts": {
            split_name: summarize_by(split_examples, "solve_source")
            for split_name, split_examples in partitions.items()
        },
        "split_stratum_counts": {
            split_name: summarize_by(split_examples, "stratum")
            for split_name, split_examples in partitions.items()
        },
        "total_stratum_counts": summarize_by(examples, "stratum"),
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
        "stratification": {
            "fields": ["category", "subcategory", "solve_source"],
            "stratum_counts": summarize_by(examples, "stratum"),
        },
    }
    write_json(output_path.with_suffix(".config.json"), split_config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
