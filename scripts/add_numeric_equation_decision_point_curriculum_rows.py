#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs/numeric_equation_decision_point_curriculum"
MANIFEST_CSV = SOURCE_DIR / "manifest.csv"
SINGLE_PHASE_CSV = ROOT / "data/single_phase_training_clean/single_phase_sft.csv"
SPLIT_CSV = ROOT / "data/single_phase_training_clean/single_phase_splits_80_10_10.csv"

SOURCE_MODE = "numeric_equation_decision_point_curriculum"
PROMPT_FORMAT = "raw_completion"
ID_PREFIX = "syn_ne_dp_"

EXTRA_COLUMNS = [
    "eval_eligible",
    "split_policy",
    "append_answer_instruction",
    "official_answer",
    "huikang_status",
    "prompt_format",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def with_defaults(row: dict[str, str]) -> dict[str, str]:
    row = dict(row)
    row.setdefault("eval_eligible", "true")
    row.setdefault("split_policy", "auto")
    row.setdefault("append_answer_instruction", "true")
    row.setdefault("official_answer", row.get("answer", ""))
    row.setdefault("huikang_status", "")
    row.setdefault("prompt_format", "competition_chat_template")
    return row


def split_sections(text: str, path: Path) -> tuple[str, str]:
    prompt_marker = "Training prompt:\n"
    target_marker = "\n\nTraining target:\n"
    if prompt_marker not in text or target_marker not in text:
        raise RuntimeError(f"Missing Training prompt/target sections: {path}")
    prompt = text.split(prompt_marker, 1)[1].split(target_marker, 1)[0].strip()
    target = text.split(target_marker, 1)[1].strip()
    if not prompt:
        raise RuntimeError(f"Empty Training prompt: {path}")
    if not target:
        raise RuntimeError(f"Empty Training target: {path}")
    return prompt, target


def load_curriculum_rows(existing_ids: set[str]) -> list[dict[str, str]]:
    manifest_rows, _fieldnames = read_csv(MANIFEST_CSV)
    rows: list[dict[str, str]] = []
    for manifest in manifest_rows:
        row_id = manifest["id"]
        if row_id in existing_ids:
            continue
        path = ROOT / manifest["path"]
        prompt, target = split_sections(path.read_text(encoding="utf-8"), path)
        rows.append(
            {
                "id": row_id,
                "prompt": prompt,
                "answer": manifest["answer"],
                "generated_cot": "",
                "assistant_content": target,
                "label": "Numeric Equation Transformation Rules",
                "category": "Numeric Equation Transformation Rules",
                "source": manifest["path"],
                "source_mode": SOURCE_MODE,
                "eval_eligible": "false",
                "split_policy": "train_only",
                "append_answer_instruction": "false",
                "official_answer": "",
                "huikang_status": "decision_point_curriculum",
                "prompt_format": PROMPT_FORMAT,
            }
        )
    return rows


def update_split_rows(existing_ids_to_keep: set[str], added_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    split_rows, _fieldnames = read_csv(SPLIT_CSV)
    kept = [
        row
        for row in split_rows
        if row.get("id") in existing_ids_to_keep
        and not row.get("id", "").startswith(ID_PREFIX)
    ]
    kept.extend({"id": row["id"], "split": "sft_train"} for row in added_rows)
    return kept


def validate(rows: list[dict[str, str]], split_rows: list[dict[str, str]]) -> None:
    row_ids = [row["id"] for row in rows]
    split_ids = [row["id"] for row in split_rows]
    duplicate_rows = [row_id for row_id, count in Counter(row_ids).items() if count > 1]
    duplicate_splits = [row_id for row_id, count in Counter(split_ids).items() if count > 1]
    if duplicate_rows:
        raise RuntimeError(f"Duplicate train row ids: {duplicate_rows[:10]}")
    if duplicate_splits:
        raise RuntimeError(f"Duplicate split row ids: {duplicate_splits[:10]}")
    if set(row_ids) != set(split_ids):
        missing = sorted(set(row_ids) - set(split_ids))[:10]
        extra = sorted(set(split_ids) - set(row_ids))[:10]
        raise RuntimeError(f"Train/split id mismatch: missing={missing} extra={extra}")

    bad_curriculum = [
        row["id"]
        for row in rows
        if row.get("source_mode") == SOURCE_MODE
        and (
            row.get("prompt_format") != PROMPT_FORMAT
            or row.get("append_answer_instruction") != "false"
            or row.get("split_policy") != "train_only"
            or row.get("eval_eligible") != "false"
            or not row.get("assistant_content", "").strip()
            or "Please put your final answer inside" in row.get("prompt", "")
        )
    ]
    if bad_curriculum:
        raise RuntimeError(f"Bad decision-point curriculum rows: {bad_curriculum[:10]}")


def main() -> None:
    rows, fieldnames = read_csv(SINGLE_PHASE_CSV)
    fieldnames = list(dict.fromkeys([*fieldnames, *EXTRA_COLUMNS]))
    base_rows = [
        with_defaults(row)
        for row in rows
        if row.get("source_mode") != SOURCE_MODE
        and not row.get("id", "").startswith(ID_PREFIX)
    ]
    existing_ids = {row["id"] for row in base_rows}
    added_rows = load_curriculum_rows(existing_ids)
    output_rows = base_rows + added_rows
    split_rows = update_split_rows(existing_ids, added_rows)

    validate(output_rows, split_rows)
    write_csv(SINGLE_PHASE_CSV, output_rows, fieldnames)
    write_csv(SPLIT_CSV, split_rows, ["id", "split"])

    print(
        {
            "total_rows": len(output_rows),
            "added_decision_point_rows": len(added_rows),
            "split_counts": dict(sorted(Counter(row["split"] for row in split_rows).items())),
            "prompt_format_counts": dict(
                sorted(Counter(row.get("prompt_format", "") for row in output_rows).items())
            ),
        }
    )


if __name__ == "__main__":
    main()
