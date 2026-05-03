#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import infer_category, load_split_assignments, select_ids_for_splits
from nemotron_baseline.metric import answers_match
from nemotron_baseline.numeric_equation import classify_equation_vs_symbol
from nemotron_baseline.prompts import normalize_generated_cot
from nemotron_baseline.text_cipher import normalize_text_encryption_cot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the phase-2 SFT dataset by anchoring on train.csv, overlaying the main "
            "competition CoT dataset, and replacing selected categories with deterministic "
            "solver traces for the selected split ids."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Full competition train.csv used as the source of truth for prompts and answers.",
    )
    parser.add_argument(
        "--base-cot-csv",
        default=None,
        help=(
            "Optional cleaned competition CoT CSV. If omitted, the script will auto-detect "
            "a likely cleaned file if present."
        ),
    )
    parser.add_argument(
        "--text-cot-csv",
        default="data/trainable/text_cipher_compact_cot.csv",
        help="Deterministic compact Text Cipher CoT dataset.",
    )
    parser.add_argument(
        "--text-cot-column",
        default="generated_cot",
        help="Preferred Text Encryption CoT column from --text-cot-csv.",
    )
    parser.add_argument(
        "--numeric-equation-cot-csv",
        default="data/trainable/numeric_equation_labelled_cot.csv",
        help=(
            "Optional deterministic numeric_equation CoT CSV. If present, Transformation Rules "
            "rows in the numeric_equation subtype will prefer these traces."
        ),
    )
    parser.add_argument(
        "--numeric-equation-deterministic-repeat",
        type=int,
        default=1,
        help=(
            "Sampling repeat count for deterministic numeric_equation CoT rows. "
            "Prefer synthetic augmentation over repeating exact rows."
        ),
    )
    parser.add_argument(
        "--numeric-equation-oracle-repeat",
        type=int,
        default=1,
        help="Sampling repeat count for oracle/speculative numeric_equation CoT rows.",
    )
    parser.add_argument(
        "--numeric-equation-synthetic-cot-csv",
        default="data/trainable/numeric_equation_synthetic_cot.csv",
        help=(
            "Optional synthetic numeric_equation CoT rows. Rows with base_id in the "
            "selected train split are appended after real rows."
        ),
    )
    parser.add_argument(
        "--bit-manipulation-cot-csv",
        default="data/trainable/bit_manipulation_hybrid_cot.csv",
        help=(
            "Optional deterministic Bit Manipulation CoT CSV. If present, Bit Manipulation "
            "rows prefer these cursor/Huikang hybrid traces."
        ),
    )
    parser.add_argument(
        "--unit-conversion-cot-csv",
        default="data/trainable/unit_conversion_cot_method_resolved.csv",
        help=(
            "Optional deterministic Unit Conversion CoT CSV. The default uses rows "
            "solved by the weighted scalar-factor method."
        ),
    )
    parser.add_argument(
        "--gravity-cot-csv",
        default="data/trainable/gravity_cot_method_resolved.csv",
        help=(
            "Optional deterministic Gravity CoT CSV. The default uses rows solved by "
            "the weighted hidden-rate method."
        ),
    )
    parser.add_argument(
        "--numeral-cot-csv",
        default="data/trainable/numeral_cot_method_resolved.csv",
        help=(
            "Optional deterministic Numeral System CoT CSV. The default uses rows "
            "solved by greedy Arabic-to-Roman conversion."
        ),
    )
    parser.add_argument(
        "--symbol-transform-cot-csv",
        default="data/trainable/symbol_transform_phase2_combined.csv",
        help=(
            "Optional deterministic Symbol Transform CoT CSV. Real rows replace "
            "split-selected symbol_transform answer-only rows; synthetic rows are "
            "appended after split selection and do not affect the split."
        ),
    )
    parser.add_argument(
        "--split-csv",
        default="data/splits_75_10_15.config.json",
        help="Split assignment file (CSV or JSON split config).",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["sft_train"],
        help="Split names to include in the phase-2 SFT file.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/train_sft_phase2_75_10_15.csv",
        help="Destination CSV in SFT schema.",
    )
    parser.add_argument(
        "--cot-column",
        default="generated_cot",
        help="CoT column to read from the base CoT CSV.",
    )
    return parser.parse_args()


def resolve_base_cot_path(explicit_path: str | None) -> Path | None:
    if explicit_path:
        path = ROOT / explicit_path
        if not path.exists():
            raise SystemExit(f"Base CoT CSV not found: {path}")
        return path

    candidates = [
        ROOT / "data/trainable/train_cot_gpt_oss_clean.csv",
        ROOT / "data/trainable/train_cot_clean.csv",
        ROOT / "data/trainable/train_cot.csv",
        ROOT / "data/train_cot_gpt_oss_clean.csv",
        ROOT / "data/train_cot_clean.csv",
        ROOT / "data/train_cot.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def wrap_reasoning_as_think_block(reasoning: str | None) -> str:
    normalized = normalize_generated_cot(reasoning)
    if not normalized:
        return ""
    if normalized.startswith("<think>") and "</think>" in normalized:
        return normalized
    return f"<think>\n{normalized}\n</think>"


BOXED_RE = re.compile(r"\\boxed\{[^}]*\}")


def remove_box_mentions_for_fallback(reasoning: str) -> str:
    """Avoid empty boxed mentions when relying on the evaluator's fallback parser."""
    return BOXED_RE.sub("the final answer format", reasoning)


def think_inner(reasoning: str | None) -> str:
    normalized = normalize_generated_cot(reasoning)
    if normalized.startswith("<think>") and normalized.endswith("</think>"):
        return normalized[len("<think>") : -len("</think>")].strip()
    return normalized.strip()


def build_symbol_transform_assistant_content(reasoning: str, answer: str) -> str:
    inner = think_inner(reasoning)
    if "}" in answer:
        # The host extractor stops a boxed answer at the first "}", so these
        # rare symbol answers must use the documented non-boxed fallback path.
        inner = remove_box_mentions_for_fallback(inner)
        if inner:
            return f"<think>\n{inner}\n</think>\nThe final answer is: {answer}"
        return f"The final answer is: {answer}"

    think_block = f"<think>\n{inner}\n</think>" if inner else ""
    if think_block:
        return f"{think_block}\n\\boxed{{{answer}}}"
    return f"\\boxed{{{answer}}}"


def load_symbol_transform_rows(
    csv_path: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    real_rows_by_id: dict[str, dict[str, str]] = {}
    synthetic_rows: list[dict[str, str]] = []
    if not csv_path.exists():
        return real_rows_by_id, synthetic_rows

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"id", "prompt", "answer", "generated_cot", "data_type"}
        missing = required - fieldnames
        if missing:
            raise SystemExit(
                f"Symbol Transform CSV is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            data_type = row.get("data_type", "").strip()
            if data_type == "real":
                real_rows_by_id[row["id"].strip()] = dict(row)
            elif data_type == "synthetic":
                synthetic_rows.append(dict(row))
    return real_rows_by_id, synthetic_rows


def load_synthetic_rows_by_base_id(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_base_id: dict[str, list[dict[str, str]]] = {}
    if not csv_path.exists():
        return rows_by_base_id

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"id", "base_id", "prompt", "answer", "generated_cot"}
        missing = required - fieldnames
        if missing:
            raise SystemExit(
                f"Synthetic CoT CSV is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            base_id = row.get("base_id", "").strip()
            if not base_id:
                raise SystemExit(f"Synthetic row {row.get('id', '')!r} is missing base_id.")
            rows_by_base_id.setdefault(base_id, []).append(dict(row))
    return rows_by_base_id

def load_rows_by_id(csv_path: Path, *, cot_column: str) -> dict[str, dict[str, str]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "id" not in fieldnames:
            raise SystemExit(f"CSV is missing required column 'id': {csv_path}")
        for row in reader:
            row_id = row["id"].strip()
            if not row_id:
                continue
            copied = dict(row)
            if cot_column in fieldnames and cot_column != "generated_cot":
                copied["generated_cot"] = row.get(cot_column, "")
            elif "generated_cot" in fieldnames:
                copied["generated_cot"] = row.get("generated_cot", "")
            rows_by_id[row_id] = copied
    return rows_by_id


def load_train_rows(train_csv: Path) -> dict[str, dict[str, str]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    with train_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"id", "prompt", "answer"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Train CSV is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            rows_by_id[row["id"]] = {
                "id": row["id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
            }
    return rows_by_id


def choose_label(base_row: dict[str, str] | None, category: str, source: str) -> str:
    if source in {"doc2lora_text_cot", "text_cipher_compact_cot"}:
        return "Text Cipher"
    if base_row and base_row.get("label"):
        return base_row["label"]
    return category


def main() -> None:
    args = parse_args()
    train_path = ROOT / args.train_csv
    text_cot_path = ROOT / args.text_cot_csv
    numeric_equation_cot_path = ROOT / args.numeric_equation_cot_csv
    numeric_equation_synthetic_cot_path = ROOT / args.numeric_equation_synthetic_cot_csv
    bit_manipulation_cot_path = ROOT / args.bit_manipulation_cot_csv
    unit_conversion_cot_path = ROOT / args.unit_conversion_cot_csv
    gravity_cot_path = ROOT / args.gravity_cot_csv
    numeral_cot_path = ROOT / args.numeral_cot_csv
    symbol_transform_cot_path = ROOT / args.symbol_transform_cot_csv
    split_path = ROOT / args.split_csv
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_cot_path = resolve_base_cot_path(args.base_cot_csv)
    if base_cot_path is None:
        print("No cleaned base CoT CSV found; using answer-only fallback except for deterministic text rows.")

    train_rows = load_train_rows(train_path)
    split_assignments = load_split_assignments(split_path)
    selected_ids = select_ids_for_splits(split_assignments, args.train_splits)
    text_rows = load_rows_by_id(text_cot_path, cot_column=args.text_cot_column)
    numeric_equation_rows = (
        load_rows_by_id(numeric_equation_cot_path, cot_column="generated_cot")
        if numeric_equation_cot_path.exists()
        else {}
    )
    numeric_equation_synthetic_rows_by_base_id = load_synthetic_rows_by_base_id(
        numeric_equation_synthetic_cot_path
    )
    bit_manipulation_rows = (
        load_rows_by_id(bit_manipulation_cot_path, cot_column="generated_cot")
        if bit_manipulation_cot_path.exists()
        else {}
    )
    unit_conversion_rows = (
        load_rows_by_id(unit_conversion_cot_path, cot_column="generated_cot")
        if unit_conversion_cot_path.exists()
        else {}
    )
    gravity_rows = (
        load_rows_by_id(gravity_cot_path, cot_column="generated_cot")
        if gravity_cot_path.exists()
        else {}
    )
    numeral_rows = (
        load_rows_by_id(numeral_cot_path, cot_column="generated_cot")
        if numeral_cot_path.exists()
        else {}
    )
    symbol_transform_real_rows, symbol_transform_synthetic_rows = load_symbol_transform_rows(
        symbol_transform_cot_path
    )
    base_rows = (
        load_rows_by_id(base_cot_path, cot_column=args.cot_column)
        if base_cot_path is not None
        else {}
    )

    output_rows: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()

    for row_id in sorted(selected_ids):
        if row_id not in train_rows:
            raise SystemExit(f"Split id {row_id!r} does not exist in {train_path}")

        train_row = train_rows[row_id]
        category = infer_category(train_row["prompt"])
        transformation_subtype = (
            classify_equation_vs_symbol(train_row["prompt"])
            if category == "Transformation Rules"
            else None
        )
        base_row = base_rows.get(row_id)
        text_row = text_rows.get(row_id)
        numeric_equation_row = numeric_equation_rows.get(row_id)
        bit_manipulation_row = bit_manipulation_rows.get(row_id)
        unit_conversion_row = unit_conversion_rows.get(row_id)
        gravity_row = gravity_rows.get(row_id)
        numeral_row = numeral_rows.get(row_id)
        symbol_transform_row = symbol_transform_real_rows.get(row_id)

        chosen_cot = ""
        chosen_answer = train_row["answer"]
        assistant_content = ""
        source = "answer_only"
        source_tier = ""
        is_deterministic = ""
        repeat_count = 1
        if base_row and base_row.get("generated_cot", "").strip():
            chosen_cot = base_row["generated_cot"]
            source = "base_cot"
        if (
            category == "Transformation Rules"
            and transformation_subtype == "numeric_equation"
            and numeric_equation_row
            and numeric_equation_row.get("generated_cot", "").strip()
        ):
            if numeric_equation_row.get("answer", "").strip() != train_row["answer"].strip():
                raise SystemExit(
                    f"Answer mismatch for numeric_equation row {row_id}: "
                    f"{numeric_equation_row.get('answer')!r} != {train_row['answer']!r}"
            )
            chosen_cot = normalize_generated_cot(numeric_equation_row["generated_cot"])
            source_tier = numeric_equation_row.get("source_tier", "").strip()
            is_deterministic = numeric_equation_row.get("is_deterministic", "").strip()
            if is_deterministic == "0":
                source = "numeric_equation_extended_oracle_low_weight"
                repeat_count = max(1, args.numeric_equation_oracle_repeat)
            else:
                source = "numeric_equation_extended_deterministic"
                repeat_count = max(1, args.numeric_equation_deterministic_repeat)
        if (
            category == "Bit Manipulation"
            and bit_manipulation_row
            and bit_manipulation_row.get("generated_cot", "").strip()
        ):
            if bit_manipulation_row.get("answer", "").strip() != train_row["answer"].strip():
                raise SystemExit(
                    f"Answer mismatch for bit_manipulation row {row_id}: "
                    f"{bit_manipulation_row.get('answer')!r} != {train_row['answer']!r}"
                )
            chosen_cot = normalize_generated_cot(bit_manipulation_row["generated_cot"])
            source = "bit_manipulation_hybrid_cot"
        if (
            category == "Unit Conversion"
            and unit_conversion_row
            and unit_conversion_row.get("generated_cot", "").strip()
        ):
            unit_answer = unit_conversion_row.get("answer", "").strip()
            if not answers_match(unit_answer, train_row["answer"].strip()):
                raise SystemExit(
                    f"Answer mismatch for unit_conversion row {row_id}: "
                    f"{unit_answer!r} is not accepted for {train_row['answer']!r}"
            )
            chosen_cot = normalize_generated_cot(unit_conversion_row["generated_cot"])
            chosen_answer = unit_answer
            source = "unit_conversion_method_resolved"
        if (
            category == "Gravity"
            and gravity_row
            and gravity_row.get("generated_cot", "").strip()
        ):
            gravity_answer = gravity_row.get("answer", "").strip()
            if not answers_match(gravity_answer, train_row["answer"].strip()):
                raise SystemExit(
                    f"Answer mismatch for gravity row {row_id}: "
                    f"{gravity_answer!r} is not accepted for {train_row['answer']!r}"
                )
            chosen_cot = normalize_generated_cot(gravity_row["generated_cot"])
            chosen_answer = gravity_answer
            source = "gravity_weighted_cot"
        if (
            category == "Numeral System"
            and numeral_row
            and numeral_row.get("generated_cot", "").strip()
        ):
            numeral_answer = numeral_row.get("answer", "").strip()
            if numeral_answer != train_row["answer"].strip():
                raise SystemExit(
                    f"Answer mismatch for numeral row {row_id}: "
                    f"{numeral_answer!r} != {train_row['answer']!r}"
                )
            chosen_cot = normalize_generated_cot(numeral_row["generated_cot"])
            chosen_answer = numeral_answer
            source = "numeral_greedy_roman_cot"
        if (
            category == "Transformation Rules"
            and transformation_subtype == "symbol_transform"
            and symbol_transform_row
            and symbol_transform_row.get("generated_cot", "").strip()
        ):
            symbol_answer = symbol_transform_row.get("answer", "").strip()
            if symbol_answer != train_row["answer"].strip():
                raise SystemExit(
                    f"Answer mismatch for symbol_transform row {row_id}: "
                    f"{symbol_answer!r} != {train_row['answer']!r}"
                )
            chosen_cot = normalize_generated_cot(symbol_transform_row["generated_cot"])
            chosen_answer = symbol_answer
            assistant_content = build_symbol_transform_assistant_content(
                symbol_transform_row["generated_cot"],
                symbol_answer,
            )
            source = "symbol_transform_solver"
        if category == "Text Cipher" and text_row and text_row.get("generated_cot", "").strip():
            if text_row.get("answer", "").strip() != train_row["answer"].strip():
                raise SystemExit(
                    f"Answer mismatch for text row {row_id}: "
                    f"{text_row.get('answer')!r} != {train_row['answer']!r}"
                )
            chosen_cot = normalize_text_encryption_cot(
                train_row["prompt"],
                text_row["generated_cot"],
            )
            source = "text_cipher_compact_cot"

        base_output_row = {
            "id": row_id,
            "prompt": train_row["prompt"],
            "answer": chosen_answer,
            "generated_cot": wrap_reasoning_as_think_block(chosen_cot),
            "label": choose_label(base_row, category, source),
            "category": category,
            "source": source,
            "data_type": "real",
            "assistant_content": assistant_content,
            "source_tier": source_tier,
            "is_deterministic": is_deterministic,
            "repeat_index": "",
            "repeat_count": str(repeat_count),
        }
        for repeat_index in range(repeat_count):
            output_row = dict(base_output_row)
            output_row["repeat_index"] = str(repeat_index + 1)
            if repeat_index > 0:
                output_row["id"] = f"{row_id}__repeat{repeat_index + 1}"
            if not output_row["assistant_content"] and "}" in output_row["answer"]:
                output_row["assistant_content"] = build_symbol_transform_assistant_content(
                    output_row["generated_cot"],
                    output_row["answer"],
                )
            output_rows.append(output_row)
            source_counts[source] += 1

    existing_ids = {row["id"] for row in output_rows}
    for row in symbol_transform_synthetic_rows:
        row_id = row["id"].strip()
        if row_id in existing_ids:
            raise SystemExit(f"Duplicate synthetic Symbol Transform id: {row_id}")
        answer = row.get("answer", "").strip()
        generated_cot = row.get("generated_cot", "").strip()
        output_rows.append(
            {
                "id": row_id,
                "prompt": row["prompt"],
                "answer": answer,
                "generated_cot": wrap_reasoning_as_think_block(generated_cot),
                "label": row.get("label", "Transformation Rules"),
                "category": row.get("category", "Transformation Rules"),
                "source": "symbol_transform_synthetic_cot",
                "data_type": "synthetic",
                "assistant_content": build_symbol_transform_assistant_content(
                    generated_cot,
                    answer,
                ),
                "source_tier": "",
                "is_deterministic": "",
                "repeat_index": "1",
                "repeat_count": "1",
            }
        )
        existing_ids.add(row_id)
        source_counts["symbol_transform_synthetic_cot"] += 1

    existing_ids = {row["id"] for row in output_rows}
    for base_id in sorted(selected_ids):
        for row in numeric_equation_synthetic_rows_by_base_id.get(base_id, []):
            row_id = row["id"].strip()
            if row_id in existing_ids:
                raise SystemExit(f"Duplicate synthetic numeric_equation id: {row_id}")
            answer = row.get("answer", "").strip()
            generated_cot = row.get("generated_cot", "").strip()
            output_rows.append(
                {
                    "id": row_id,
                    "prompt": row["prompt"],
                    "answer": answer,
                    "generated_cot": wrap_reasoning_as_think_block(generated_cot),
                    "label": row.get("label", "Transformation Rules"),
                    "category": row.get("category", "Transformation Rules"),
                    "source": "numeric_equation_synthetic_cot",
                    "data_type": "synthetic",
                    "assistant_content": build_symbol_transform_assistant_content(
                        generated_cot,
                        answer,
                    )
                    if "}" in answer
                    else "",
                    "source_tier": row.get("source_tier", "synthetic_from_deterministic"),
                    "is_deterministic": row.get("is_deterministic", "1"),
                    "repeat_index": "1",
                    "repeat_count": "1",
                }
            )
            existing_ids.add(row_id)
            source_counts["numeric_equation_synthetic_cot"] += 1

    fieldnames = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "data_type",
        "assistant_content",
        "source_tier",
        "is_deterministic",
        "repeat_index",
        "repeat_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} phase-2 SFT rows to {output_path}")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
