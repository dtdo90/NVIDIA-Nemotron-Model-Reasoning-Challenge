#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


FINAL_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:Final Answer|Answer)\s*:.*$|^\s*The final answer is\s*.*$|^\s*\\boxed\{.*\}\s*$"
)
BOXED_PATTERN = re.compile(r"\\boxed\{")
ANALYSIS_PREFIX_RE = re.compile(r"^\s*analysis(?=\s|:|-|[A-Z])[:\s-]*", re.IGNORECASE)
ASSISTANTFINAL_PREFIX_RE = re.compile(
    r"^\s*assistantfinal(?=\s|:|-|\*|[A-Z])[:\s-]*",
    re.IGNORECASE,
)
METADATA_LEAK_RE = re.compile(
    r"verified (?:correct )?answer|provided answer|gold answer|ground truth|given answer",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Salvage raw generated CoT traces with deterministic cleanup before "
            "running the supervisor and approver pipeline."
        )
    )
    parser.add_argument("--input-csv", default="data/trainable/train_cot_gpt_oss.csv")
    parser.add_argument("--output-csv", default="data/trainable/train_cot_gpt_oss_salvaged.csv")
    parser.add_argument(
        "--keep-original-column",
        action="store_true",
        default=True,
        help="Keep the original trace in an extra generated_cot_original column.",
    )
    parser.add_argument(
        "--no-keep-original-column",
        dest="keep_original_column",
        action="store_false",
        help="Do not keep the original trace column in the salvaged CSV.",
    )
    return parser.parse_args()


def collapse_blank_lines(lines: list[str]) -> list[str]:
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        if not line.strip():
            if previous_blank:
                continue
            collapsed.append("")
            previous_blank = True
            continue
        collapsed.append(line.rstrip())
        previous_blank = False
    while collapsed and not collapsed[-1].strip():
        collapsed.pop()
    return collapsed


def remove_metadata_leak_lines(raw_text: str) -> tuple[str, bool]:
    kept_lines: list[str] = []
    removed_any = False
    for line in raw_text.splitlines():
        if METADATA_LEAK_RE.search(line):
            removed_any = True
            continue
        kept_lines.append(line.rstrip())
    kept_lines = collapse_blank_lines(kept_lines)
    return "\n".join(kept_lines).strip(), removed_any


def strip_channel_markers(text: str) -> tuple[str, bool]:
    updated_lines: list[str] = []
    changed = False
    for line in text.splitlines():
        cleaned = ANALYSIS_PREFIX_RE.sub("", line, count=1)
        cleaned = ASSISTANTFINAL_PREFIX_RE.sub("", cleaned, count=1)
        if cleaned != line:
            changed = True
        updated_lines.append(cleaned.rstrip())
    updated_lines = collapse_blank_lines(updated_lines)
    return "\n".join(updated_lines).strip(), changed


def count_boxes(text: str) -> int:
    return len(BOXED_PATTERN.findall(text))


def truncate_at_first_box(text: str, gold_answer: str) -> str:
    first_box_index = text.find(r"\boxed{")
    if first_box_index < 0:
        return text.strip()
    prefix = text[:first_box_index].rstrip()
    final_line = f"\\boxed{{{gold_answer}}}"
    if prefix:
        return f"{prefix}\n\n{final_line}"
    return final_line


def salvage_trace(raw_text: str, gold_answer: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    salvaged, removed_metadata_lines = remove_metadata_leak_lines(raw_text)
    if removed_metadata_lines:
        reasons.append("removed_metadata_leak_line")

    salvaged, stripped_channel_markers = strip_channel_markers(salvaged)
    if stripped_channel_markers:
        reasons.append("stripped_channel_marker")

    box_count = count_boxes(salvaged)
    if box_count > 1:
        salvaged = truncate_at_first_box(salvaged, gold_answer)
        reasons.append("truncated_from_first_box")
    elif box_count == 0:
        final_line = f"\\boxed{{{gold_answer}}}"
        salvaged = f"{salvaged}\n\n{final_line}".strip() if salvaged else final_line
        reasons.append("appended_missing_box")

    return salvaged, sorted(set(reasons))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{input_path} is missing a CSV header.")

        fieldnames = list(reader.fieldnames)
        if args.keep_original_column and "generated_cot_original" not in fieldnames:
            fieldnames.append("generated_cot_original")
        if "salvage_notes" not in fieldnames:
            fieldnames.append("salvage_notes")

        salvage_reason_counts: Counter[str] = Counter()
        nonempty_reasoning_count = 0
        rows_written = 0

        with output_path.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                raw_text = row.get("generated_cot", "") or ""
                salvaged_text, reasons = salvage_trace(raw_text, row.get("answer", "") or "")
                if args.keep_original_column:
                    row["generated_cot_original"] = raw_text
                row["generated_cot"] = salvaged_text
                row["salvage_notes"] = ";".join(reasons)
                writer.writerow(row)

                rows_written += 1
                salvage_reason_counts.update(reasons)
                body = FINAL_ANSWER_LINE_RE.sub("", salvaged_text).strip()
                if body:
                    nonempty_reasoning_count += 1

    summary = {
        "input_csv": str(input_path.resolve()),
        "output_csv": str(output_path.resolve()),
        "rows_written": rows_written,
        "rows_with_nonempty_reasoning": nonempty_reasoning_count,
        "salvage_reason_counts": dict(salvage_reason_counts.most_common()),
        "kept_original_column": args.keep_original_column,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
