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

from nemotron_baseline.numeric_equation import classify_equation_vs_symbol  # noqa: E402
from nemotron_baseline.symbol_transform import (  # noqa: E402
    parse_symbol_transform_puzzle,
    same_operator_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a focused diagnostic table for symbol_transform misses that are "
            "oracle-explainable inside the current core encrypted-digit rule space."
        )
    )
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument(
        "--analysis-csv",
        default="data/symbol_transform_solver_analysis_core_adaptive_safe.csv",
        help="Current deterministic solver analysis CSV.",
    )
    parser.add_argument(
        "--oracle-csv",
        default="data/symbol_transform_oracle_missed_full.csv",
        help="Oracle miss-diagnosis CSV generated from the current analysis.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/symbol_transform_core_gap_diagnostics.csv",
    )
    return parser.parse_args()


def load_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: dict(row) for row in csv.DictReader(handle)}


def split_variants_lossy(text: str) -> list[str]:
    # The analysis CSV joins variants with '|', which is lossy when a symbol
    # answer itself contains '|'. Keep this only as a rough triage signal.
    if not text:
        return []
    return [part for part in text.split("|") if part]


def action_bucket(
    *,
    current_method: str,
    current_confidence: str,
    answer_unseen_symbols: str,
    oracle_candidate_count: int,
) -> str:
    if answer_unseen_symbols:
        return "map_completion_unseen_answer_symbol"
    if current_confidence == "ambiguous":
        if oracle_candidate_count <= 10:
            return "ranking_small_ambiguous_core"
        return "ranking_large_ambiguous_core"
    if current_method == "no_rule":
        if oracle_candidate_count <= 10:
            return "beam_or_map_completion_small_core"
        return "beam_or_map_completion_large_core"
    if current_method.endswith("_unique"):
        return "wrong_unique_candidate_core"
    return "other_core_gap"


def main() -> None:
    args = parse_args()
    train_by_id = load_by_id(ROOT / args.train_csv)
    analysis_by_id = load_by_id(ROOT / args.analysis_csv)
    oracle_by_id = load_by_id(ROOT / args.oracle_csv)

    rows: list[dict[str, str]] = []
    bucket_counts: Counter[str] = Counter()
    motif_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()
    same_count_counts: Counter[str] = Counter()

    for row_id, oracle_row in sorted(oracle_by_id.items()):
        if oracle_row.get("oracle_status") != "current_core_space":
            continue
        current = analysis_by_id.get(row_id)
        train_row = train_by_id.get(row_id)
        if current is None or train_row is None:
            continue
        if current.get("is_exact") == "1":
            continue

        prompt = train_row["prompt"]
        if classify_equation_vs_symbol(prompt) != "symbol_transform":
            continue
        puzzle = parse_symbol_transform_puzzle(prompt)
        if puzzle is None:
            continue
        same = same_operator_examples(puzzle)
        answer = train_row["answer"].strip()
        answer_unseen_symbols = oracle_row.get("answer_unseen_symbols", "").strip()
        oracle_candidate_count = int(oracle_row.get("candidate_count") or 0)
        variants = split_variants_lossy(current.get("prediction_variants", ""))
        bucket = action_bucket(
            current_method=current.get("method", ""),
            current_confidence=current.get("confidence", ""),
            answer_unseen_symbols=answer_unseen_symbols,
            oracle_candidate_count=oracle_candidate_count,
        )

        output_row = {
            "id": row_id,
            "answer": answer,
            "query": puzzle.query,
            "query_operator": puzzle.query_operator,
            "same_operator_count": str(len(same)),
            "example_count": str(len(puzzle.examples)),
            "answer_len": str(len(answer)),
            "answer_unseen_symbols": answer_unseen_symbols,
            "has_unseen_answer_symbol": "1" if answer_unseen_symbols else "0",
            "current_method": current.get("method", ""),
            "current_prediction": current.get("prediction", ""),
            "current_confidence": current.get("confidence", ""),
            "current_candidate_count": current.get("candidate_count", ""),
            "current_variant_count_lossy": str(len(set(variants))),
            "gold_in_current_variants_lossy": "1" if answer in set(variants) else "0",
            "oracle_candidate_count": str(oracle_candidate_count),
            "best_stage": oracle_row.get("best_stage", ""),
            "best_motif": oracle_row.get("best_motif", ""),
            "best_rule": oracle_row.get("best_rule", ""),
            "best_rule_family": oracle_row.get("best_rule_family", ""),
            "best_output_mode": oracle_row.get("best_output_mode", ""),
            "best_label": oracle_row.get("best_label", ""),
            "action_bucket": bucket,
            "prompt": prompt,
        }
        rows.append(output_row)
        bucket_counts[bucket] += 1
        motif_counts[output_row["best_motif"]] += 1
        rule_counts[output_row["best_rule"]] += 1
        output_counts[output_row["best_output_mode"]] += 1
        same_count_counts[output_row["same_operator_count"]] += 1

    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "answer",
        "query",
        "query_operator",
        "same_operator_count",
        "example_count",
        "answer_len",
        "answer_unseen_symbols",
        "has_unseen_answer_symbol",
        "current_method",
        "current_prediction",
        "current_confidence",
        "current_candidate_count",
        "current_variant_count_lossy",
        "gold_in_current_variants_lossy",
        "oracle_candidate_count",
        "best_stage",
        "best_motif",
        "best_rule",
        "best_rule_family",
        "best_output_mode",
        "best_label",
        "action_bucket",
        "prompt",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} current-core symbol_transform gap rows -> {output_path}")
    print("\nAction buckets:")
    for key, value in bucket_counts.most_common():
        print(f"  {key}: {value}")
    print("\nTop motifs:")
    for key, value in motif_counts.most_common(8):
        print(f"  {key}: {value}")
    print("\nTop rules:")
    for key, value in rule_counts.most_common(10):
        print(f"  {key}: {value}")
    print("\nOutput modes:")
    for key, value in output_counts.most_common():
        print(f"  {key}: {value}")
    print("\nSame-operator counts:")
    for key, value in sorted(same_count_counts.items(), key=lambda item: int(item[0])):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
