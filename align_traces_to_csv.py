#!/usr/bin/env python3
"""Export or verify trace text files against the active SFT CSV.

The CSV is the source of truth. For each row, we export the exact training
target stored in `generated_cot`, or `assistant_content` when the row uses a
preformatted assistant completion such as the HuiKang bit traces.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CSV = Path("data/single_phase_training_clean/single_phase_sft.csv")
DEFAULT_OUT_DIR = Path("data/single_phase_training_clean/exported_traces")
DEFAULT_MANIFEST = Path("data/single_phase_training_clean/trace_manifest_from_csv.csv")


@dataclass(frozen=True)
class TraceRecord:
    problem_id: str
    category: str
    source_mode: str
    answer: str
    source_column: str
    text: str
    path: Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def normalize_trace(text: str) -> str:
    return text.strip() + "\n"


def trace_text_from_row(row: dict[str, str]) -> tuple[str, str]:
    generated_cot = (row.get("generated_cot") or "").strip()
    if generated_cot:
        return "generated_cot", normalize_trace(generated_cot)

    assistant_content = (row.get("assistant_content") or "").strip()
    if assistant_content:
        return "assistant_content", normalize_trace(assistant_content)

    raise ValueError(f"row {row.get('id', '<missing id>')} has no trace content")


def load_records(csv_path: Path, out_dir: Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    seen_ids: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            problem_id = (row.get("id") or "").strip()
            if not problem_id:
                raise ValueError(f"missing id at CSV row {row_number}")
            if problem_id in seen_ids:
                raise ValueError(f"duplicate id {problem_id} at CSV row {row_number}")
            seen_ids.add(problem_id)

            category = (row.get("category") or "unknown").strip() or "unknown"
            source_mode = (row.get("source_mode") or "").strip()
            answer = (row.get("answer") or "").strip()
            source_column, text = trace_text_from_row(row)
            path = out_dir / slugify(category) / f"{problem_id}.txt"
            records.append(
                TraceRecord(
                    problem_id=problem_id,
                    category=category,
                    source_mode=source_mode,
                    answer=answer,
                    source_column=source_column,
                    text=text,
                    path=path,
                )
            )

    return records


def write_manifest(records: list[TraceRecord], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "category",
                "source_mode",
                "answer",
                "source_column",
                "trace_path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.problem_id,
                    "category": record.category,
                    "source_mode": record.source_mode,
                    "answer": record.answer,
                    "source_column": record.source_column,
                    "trace_path": record.path.as_posix(),
                }
            )


def export_records(records: list[TraceRecord], overwrite: bool) -> tuple[int, list[str]]:
    written = 0
    problems: list[str] = []

    for record in records:
        if record.path.exists():
            current = record.path.read_text(encoding="utf-8")
            if current == record.text:
                continue
            if not overwrite:
                problems.append(f"different existing trace: {record.path}")
                continue

        record.path.parent.mkdir(parents=True, exist_ok=True)
        record.path.write_text(record.text, encoding="utf-8")
        written += 1

    return written, problems


def check_records(records: list[TraceRecord], out_dir: Path) -> tuple[list[str], list[str]]:
    expected_paths = {record.path for record in records}
    problems: list[str] = []

    for record in records:
        if not record.path.exists():
            problems.append(f"missing trace: {record.path}")
            continue
        current = record.path.read_text(encoding="utf-8")
        if current != record.text:
            problems.append(f"trace differs from CSV: {record.path}")

    extras = []
    if out_dir.exists():
        extras = sorted(
            str(path)
            for path in out_dir.rglob("*.txt")
            if path not in expected_paths
        )

    return problems, extras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--export", action="store_true", help="write trace files from CSV")
    parser.add_argument("--check", action="store_true", help="verify trace files match CSV")
    parser.add_argument("--overwrite", action="store_true", help="overwrite changed trace files")
    parser.add_argument("--no-manifest", action="store_true", help="skip writing manifest on export")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.export and not args.check:
        args.check = True

    records = load_records(args.csv, args.output_dir)
    print(f"Loaded {len(records)} trace records from {args.csv}")

    if args.export:
        written, export_problems = export_records(records, overwrite=args.overwrite)
        print(f"Exported {written} trace files to {args.output_dir}")
        if not args.no_manifest:
            write_manifest(records, args.manifest)
            print(f"Wrote manifest to {args.manifest}")
        if export_problems:
            print("Export found problems:", file=sys.stderr)
            for problem in export_problems[:20]:
                print(f"  {problem}", file=sys.stderr)
            if len(export_problems) > 20:
                print(f"  ... {len(export_problems) - 20} more", file=sys.stderr)
            return 2

    if args.check:
        check_problems, extras = check_records(records, args.output_dir)
        if check_problems:
            print("Trace alignment check failed:", file=sys.stderr)
            for problem in check_problems[:20]:
                print(f"  {problem}", file=sys.stderr)
            if len(check_problems) > 20:
                print(f"  ... {len(check_problems) - 20} more", file=sys.stderr)
            return 3
        print("Trace alignment check passed")
        if extras:
            print(f"Found {len(extras)} extra exported trace files not present in CSV")
            for path in extras[:20]:
                print(f"  {path}")
            if len(extras) > 20:
                print(f"  ... {len(extras) - 20} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
