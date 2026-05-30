from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    CATEGORY_TO_SLUG,
    DATA_DIR,
    QUESTION_TYPES,
    ROOT,
    SOURCE_CSV,
    build_stratified_splits,
    classify_subtype,
    load_type_rows,
    normalize_question_type,
    read_csv_rows,
    summarize_rows,
    type_paths,
    write_csv_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-question-type diagnostic datasets and 80/10/10 splits."
    )
    parser.add_argument("--source-csv", default=str(SOURCE_CSV))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--question-type",
        default=None,
        help="Optional single type. Defaults to all seven diagnostic types.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-sync-root-split",
        action="store_true",
        help=(
            "Do not copy the combined all-type split back to "
            "data/single_phase_training_clean/single_phase_splits_80_10_10.csv."
        ),
    )
    return parser.parse_args()


def prepare_one(question_type: str, *, source_csv: Path, data_dir: Path, seed: int) -> dict[str, object]:
    slug = normalize_question_type(question_type)
    rows, fieldnames = load_type_rows(slug, source_csv)
    for row in rows:
        row["diagnostic_type"] = slug
        row["diagnostic_subtype"] = classify_subtype(row)

    output_fields = list(fieldnames)
    for extra in ("diagnostic_type", "diagnostic_subtype"):
        if extra not in output_fields:
            output_fields.append(extra)

    assignments = build_stratified_splits(rows, seed=seed)
    paths = type_paths(slug, data_dir=data_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(paths.train_csv, rows, output_fields)

    split_rows = [
        {
            "id": row["id"],
            "split": assignments[row["id"]],
            "diagnostic_type": slug,
            "diagnostic_subtype": row["diagnostic_subtype"],
            "source_mode": row.get("source_mode", "unknown"),
        }
        for row in rows
    ]
    write_csv_rows(
        paths.split_csv,
        split_rows,
        ["id", "split", "diagnostic_type", "diagnostic_subtype", "source_mode"],
    )

    summary = {
        "question_type": slug,
        "category": QUESTION_TYPES[slug]["category"],
        "source_csv": str(source_csv),
        "train_csv": str(paths.train_csv),
        "split_csv": str(paths.split_csv),
        "seed": seed,
        **summarize_rows(rows, assignments),
    }
    write_json(paths.summary_json, summary)
    return summary


def write_global_split_from_type_splits(
    question_types: list[str],
    *,
    data_dir: Path,
    output_csv: Path,
) -> dict[str, int]:
    split_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for question_type in question_types:
        paths = type_paths(question_type, data_dir=data_dir)
        rows, _ = read_csv_rows(paths.split_csv)
        for row in rows:
            row_id = row["id"]
            if row_id in seen_ids:
                raise SystemExit(f"Duplicate id while building global split: {row_id}")
            seen_ids.add(row_id)
            split_rows.append({"id": row_id, "split": row["split"]})

    split_rows.sort(key=lambda row: row["id"])
    write_csv_rows(output_csv, split_rows, ["id", "split"])

    counts: dict[str, int] = {}
    for row in split_rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return dict(sorted(counts.items()))


def main(default_question_type: str | None = None) -> None:
    args = parse_args()
    source_csv = Path(args.source_csv)
    data_dir = Path(args.data_dir)
    if default_question_type:
        question_types = [default_question_type]
    elif args.question_type:
        question_types = [normalize_question_type(args.question_type)]
    else:
        question_types = list(QUESTION_TYPES)

    summaries = [
        prepare_one(question_type, source_csv=source_csv, data_dir=data_dir, seed=args.seed)
        for question_type in question_types
    ]
    global_split_path = data_dir / "global_splits_80_10_10.csv"
    global_split_counts = write_global_split_from_type_splits(
        [str(summary["question_type"]) for summary in summaries],
        data_dir=data_dir,
        output_csv=global_split_path,
    )
    root_split_path = ROOT / "data/single_phase_training_clean/single_phase_splits_80_10_10.csv"
    root_split_counts = None
    should_sync_root_split = (
        not args.no_sync_root_split
        and set(question_types) == set(QUESTION_TYPES)
        and source_csv.resolve() == SOURCE_CSV.resolve()
    )
    if should_sync_root_split:
        root_split_counts = write_global_split_from_type_splits(
            [str(summary["question_type"]) for summary in summaries],
            data_dir=data_dir,
            output_csv=root_split_path,
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        data_dir / "all_types_summary.json",
        {
            "source_csv": str(source_csv),
            "seed": args.seed,
            "prepared_types": [summary["question_type"] for summary in summaries],
            "global_split_csv": str(global_split_path),
            "global_split_counts": global_split_counts,
            "synced_root_split_csv": str(root_split_path) if should_sync_root_split else None,
            "synced_root_split_counts": root_split_counts,
            "totals": {
                summary["question_type"]: {
                    "total": summary["total"],
                    "split_counts": summary["split_counts"],
                    "subtype_count": len(summary["by_subtype"]),
                }
                for summary in summaries
            },
            "categories": dict(sorted(CATEGORY_TO_SLUG.items())),
        },
    )

    for summary in summaries:
        print(
            f"{summary['question_type']}: total={summary['total']} "
            f"splits={summary['split_counts']} subtypes={len(summary['by_subtype'])}"
        )


if __name__ == "__main__":
    main()
