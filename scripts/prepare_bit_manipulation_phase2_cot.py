#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
FINAL_OUTRO = "I will now return the answer in \\boxed{}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt cursor's oracle-filtered bit-manipulation CoT traces into "
            "the main Phase 2 SFT schema by joining them back to data/train.csv."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Competition train CSV with prompt and answer columns.",
    )
    parser.add_argument(
        "--cursor-cot-csv",
        default="reference/cursor/bit_manipulation/results/labelled_cot.csv",
        help="Cursor oracle-filtered labelled bit CoT CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/bit_manipulation_hybrid_cot.csv",
        help="Destination CSV in our Phase 2 schema.",
    )
    return parser.parse_args()


def load_train_rows(train_path: Path) -> dict[str, dict[str, str]]:
    with train_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"id", "prompt", "answer"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{train_path} is missing required columns: {sorted(missing)}")
        return {row["id"]: row for row in reader}


def final_boxed_answer(trace: str) -> str:
    matches = BOXED_RE.findall(trace)
    for match in reversed(matches):
        cleaned = match.strip()
        if cleaned:
            return cleaned
    return ""


def separate_final_answer_paragraph(trace: str) -> str:
    """Ensure normalization strips only the final answer, not query application.

    Some Tier 2 cursor traces append "I will now return..." immediately after
    the query application block with only one newline. Our CoT normalizer
    removes the final paragraph containing answer text, so we insert a blank
    line before the outro to keep the query application paragraph intact.
    """
    marker = "\n" + FINAL_OUTRO
    replacement = "\n\n" + FINAL_OUTRO
    return trace.replace(marker, replacement, 1)


def normalize_placeholder_box(trace: str) -> str:
    return trace.replace("\\boxed{-}", "\\boxed{–}")


def load_cursor_rows(cursor_path: Path, train_rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    with cursor_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id",
            "source_tier",
            "rule",
            "trace",
            "prediction",
            "answer",
            "n_lines",
            "n_chars",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{cursor_path} is missing required columns: {sorted(missing)}")

        rows: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            row_id = row["id"].strip()
            if not row_id:
                raise SystemExit(f"{cursor_path}:{row_number} has an empty id.")
            if row_id in seen_ids:
                raise SystemExit(f"{cursor_path}:{row_number} duplicates id {row_id!r}.")
            if row_id not in train_rows:
                raise SystemExit(f"{cursor_path}:{row_number} id {row_id!r} is absent from train CSV.")
            seen_ids.add(row_id)

            train_answer = train_rows[row_id]["answer"].strip()
            cursor_answer = row["answer"].strip()
            prediction = row["prediction"].strip()
            row["trace"] = normalize_placeholder_box(separate_final_answer_paragraph(row["trace"]))
            boxed = final_boxed_answer(row["trace"])
            if cursor_answer != train_answer:
                raise SystemExit(
                    f"Answer mismatch for {row_id}: cursor {cursor_answer!r} != train {train_answer!r}"
                )
            if prediction != train_answer:
                raise SystemExit(
                    f"Prediction mismatch for {row_id}: prediction {prediction!r} != answer {train_answer!r}"
                )
            if boxed != train_answer:
                raise SystemExit(
                    f"Final boxed answer mismatch for {row_id}: boxed {boxed!r} != answer {train_answer!r}"
                )

            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    train_path = ROOT / args.train_csv
    cursor_path = ROOT / args.cursor_cot_csv
    output_path = ROOT / args.output_csv

    if not train_path.exists():
        raise SystemExit(f"Train CSV not found: {train_path}")
    if not cursor_path.exists():
        raise SystemExit(f"Cursor CoT CSV not found: {cursor_path}")

    train_rows = load_train_rows(train_path)
    cursor_rows = load_cursor_rows(cursor_path, train_rows)
    tier_counts: Counter[str] = Counter(row["source_tier"] for row in cursor_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "source_tier",
        "rule",
        "n_lines",
        "n_chars",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in cursor_rows:
            train_row = train_rows[row["id"]]
            writer.writerow(
                {
                    "id": row["id"],
                    "prompt": train_row["prompt"],
                    "answer": train_row["answer"],
                    "generated_cot": row["trace"],
                    "label": "Bit Manipulation",
                    "category": "Bit Manipulation",
                    "source": "cursor_bit_hybrid",
                    "source_tier": row["source_tier"],
                    "rule": row["rule"],
                    "n_lines": row["n_lines"],
                    "n_chars": row["n_chars"],
                }
            )

    print(f"Wrote {len(cursor_rows)} bit-manipulation Phase 2 CoT rows to {output_path}")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count}")


if __name__ == "__main__":
    main()
