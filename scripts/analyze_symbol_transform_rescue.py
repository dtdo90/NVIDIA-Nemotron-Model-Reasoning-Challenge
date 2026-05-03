#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and adaptive-retry symbol_transform analyses."
    )
    parser.add_argument(
        "--baseline-csv",
        default="data/symbol_transform_solver_analysis_core.csv",
        help="Baseline solver analysis CSV.",
    )
    parser.add_argument(
        "--retry-csv",
        default="data/symbol_transform_solver_analysis_core_adaptive.csv",
        help="Adaptive-retry solver analysis CSV.",
    )
    parser.add_argument(
        "--oracle-csv",
        default="",
        help="Optional oracle diagnostic CSV to annotate rescued rows.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/symbol_transform_rescue_analysis.csv",
        help="Where to write per-row rescue deltas.",
    )
    return parser


def _load_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def _is_predicted(row: dict[str, str]) -> bool:
    return bool(row.get("prediction", ""))


def _is_exact(row: dict[str, str]) -> bool:
    return row.get("is_exact") == "1"


def main() -> None:
    args = build_parser().parse_args()
    baseline = _load_csv(Path(args.baseline_csv))
    retry = _load_csv(Path(args.retry_csv))
    oracle = _load_csv(Path(args.oracle_csv)) if args.oracle_csv else {}

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    summary: Counter[str] = Counter()
    by_method: Counter[str] = Counter()
    by_oracle: Counter[str] = Counter()

    for row_id, retry_row in retry.items():
        base_row = baseline.get(row_id)
        if base_row is None:
            continue

        base_pred = base_row.get("prediction", "")
        retry_pred = retry_row.get("prediction", "")
        if base_pred == retry_pred:
            continue

        base_exact = _is_exact(base_row)
        retry_exact = _is_exact(retry_row)
        if not _is_predicted(base_row) and _is_predicted(retry_row):
            delta_type = "new_prediction"
        elif _is_predicted(base_row) and not _is_predicted(retry_row):
            delta_type = "lost_prediction"
        else:
            delta_type = "changed_prediction"

        if retry_exact and not base_exact:
            outcome = "rescued_exact"
        elif base_exact and not retry_exact:
            outcome = "lost_exact"
        elif _is_predicted(retry_row) and not retry_exact:
            outcome = "new_or_changed_wrong"
        else:
            outcome = "neutral"

        oracle_row = oracle.get(row_id, {})
        oracle_status = oracle_row.get("oracle_status", "")

        summary[outcome] += 1
        by_method[retry_row.get("method", "")] += 1
        if oracle_status:
            by_oracle[oracle_status] += 1

        records.append(
            {
                "id": row_id,
                "answer": retry_row.get("answer", ""),
                "delta_type": delta_type,
                "outcome": outcome,
                "baseline_prediction": base_pred,
                "retry_prediction": retry_pred,
                "baseline_method": base_row.get("method", ""),
                "retry_method": retry_row.get("method", ""),
                "retry_motif": retry_row.get("motif", ""),
                "retry_rule": retry_row.get("query_rule", ""),
                "retry_candidate_count": retry_row.get("candidate_count", ""),
                "oracle_status": oracle_status,
                "oracle_best_label": oracle_row.get("best_label", ""),
                "oracle_answer_unseen_symbols": oracle_row.get("answer_unseen_symbols", ""),
            }
        )

    fieldnames = [
        "id",
        "answer",
        "delta_type",
        "outcome",
        "baseline_prediction",
        "retry_prediction",
        "baseline_method",
        "retry_method",
        "retry_motif",
        "retry_rule",
        "retry_candidate_count",
        "oracle_status",
        "oracle_best_label",
        "oracle_answer_unseen_symbols",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Changed rows: {len(records)}")
    print(f"Wrote: {output_path}")

    print("\nOutcomes:")
    for key, count in summary.most_common():
        print(f"  {key}: {count}")

    print("\nRetry methods:")
    for key, count in by_method.most_common():
        print(f"  {key}: {count}")

    if oracle:
        print("\nOracle status for changed rows:")
        for key, count in by_oracle.most_common():
            print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
