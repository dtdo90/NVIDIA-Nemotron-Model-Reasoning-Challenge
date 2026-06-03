#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import (  # noqa: E402
    QUESTION_TYPES,
    is_train_only,
    read_csv_rows,
    summarize_rows,
    type_paths,
    validate_split_assignments,
    write_csv_rows,
    write_json,
)


SEED = 20260603
QUESTION_TYPE = "text_cipher"
FORCE_EVAL_IDS = {
    "3094603c",
    "63500e84",
    "6fde02ef",
}


def main() -> None:
    paths = type_paths(QUESTION_TYPE)
    rows, _ = read_csv_rows(paths.train_csv)
    if not rows:
        raise SystemExit(f"No rows found in {paths.train_csv}")

    category = QUESTION_TYPES[QUESTION_TYPE]["category"]
    wrong_category = sorted({row.get("category", "") for row in rows if row.get("category") != category})
    if wrong_category:
        raise SystemExit(f"{paths.train_csv} contains non-text-cipher categories: {wrong_category}")

    rows_by_id = {row["id"]: row for row in rows}
    missing_force_ids = sorted(FORCE_EVAL_IDS - set(rows_by_id))
    if missing_force_ids:
        raise SystemExit(f"Forced eval ids are missing from {paths.train_csv}: {missing_force_ids}")

    for row_id in sorted(FORCE_EVAL_IDS):
        if is_train_only(rows_by_id[row_id]):
            raise SystemExit(f"Forced eval id {row_id} is train-only and cannot be evaluated")

    assignments: dict[str, str] = {}
    train_only_rows = []
    eval_candidate_rows = []
    for row in rows:
        row_id = row["id"]
        if is_train_only(row):
            assignments[row_id] = "sft_train"
            train_only_rows.append(row)
        elif row_id in FORCE_EVAL_IDS:
            assignments[row_id] = "eval_holdout"
        else:
            eval_candidate_rows.append(row)

    total_eval_eligible = len(eval_candidate_rows) + len(FORCE_EVAL_IDS)
    target_eval_count = max(1, round(total_eval_eligible * 0.20))
    remaining_eval_count = max(0, target_eval_count - len(FORCE_EVAL_IDS))

    rng = random.Random(SEED)
    shuffled = list(eval_candidate_rows)
    rng.shuffle(shuffled)
    eval_ids = {row["id"] for row in shuffled[:remaining_eval_count]}
    for row in eval_candidate_rows:
        assignments[row["id"]] = "eval_holdout" if row["id"] in eval_ids else "sft_train"

    validate_split_assignments(rows, assignments, split_csv="text_cipher_retest")

    split_rows = [
        {
            "id": row["id"],
            "split": assignments[row["id"]],
            "diagnostic_type": row.get("diagnostic_type", QUESTION_TYPE),
            "diagnostic_subtype": row.get("diagnostic_subtype", "standard"),
            "source_mode": row.get("source_mode", "unknown"),
            "eval_eligible": row.get("eval_eligible", "true"),
            "split_policy": row.get("split_policy", "auto") or "auto",
        }
        for row in rows
    ]
    fieldnames = [
        "id",
        "split",
        "diagnostic_type",
        "diagnostic_subtype",
        "source_mode",
        "eval_eligible",
        "split_policy",
    ]

    output_csv = paths.data_dir / "splits_retest_80_20.csv"
    summary_json = paths.data_dir / "splits_retest_80_20_summary.json"
    write_csv_rows(output_csv, split_rows, fieldnames)

    split_counts = Counter(assignments.values())
    forced_eval_splits = {row_id: assignments[row_id] for row_id in sorted(FORCE_EVAL_IDS)}
    write_json(
        summary_json,
        {
            "mode": "text_cipher_retest_split",
            "seed": SEED,
            "train_csv": str(paths.train_csv.resolve()),
            "split_csv": str(output_csv.resolve()),
            "total_rows": len(rows),
            "train_only_rows": len(train_only_rows),
            "eval_eligible_rows": total_eval_eligible,
            "target_eval_fraction": 0.20,
            "forced_eval_ids": forced_eval_splits,
            "split_counts": dict(sorted(split_counts.items())),
            "summary": summarize_rows(rows, assignments),
        },
    )
    print(f"Wrote {output_csv}")
    print(f"Wrote {summary_json}")
    print(f"Split counts: {dict(sorted(split_counts.items()))}")
    print(f"Forced eval ids: {forced_eval_splits}")


if __name__ == "__main__":
    main()
