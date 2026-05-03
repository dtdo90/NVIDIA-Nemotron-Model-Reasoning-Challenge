#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.symbol_transform import (  # noqa: E402
    DIRECT_TEMPLATE_PRIORITY,
    DIRECT_TEMPLATES,
    parse_symbol_transform_puzzle,
    same_operator_examples,
)


DIRECT_S0 = (
    "S0: Methodology: solve same-operator examples first; test direct templates first; "
    "if they fail, use encrypted digit search with BA_DC|rev or AB_CD|raw; choose the "
    "arithmetic family from same-operator RHS length; keep visible survivor grids; use other "
    "examples only to complete the map; then solve the query."
)

COMPACT_EXPECTATIONS = {
    "motif_drill": {
        "first": "We need to apply motif rule.",
        "required": [
            "Parse ABOCD and apply the requested operand motif.",
            "The final answer is \\boxed{",
        ],
    },
    "operator_family_drill": {
        "first": "We need to apply arithmetic rule.",
        "required": [
            "Form operands, apply the stated rule, then render the output.",
            "The final answer is \\boxed{",
        ],
    },
    "symbol_digit_encode_decode": {
        "first": "We need to convert symbols and digits with the provided map.",
        "required": [
            "Convert each character with the provided map, then join the results.",
            "The final answer is \\boxed{",
        ],
    },
}

COMPACT_FORBIDDEN = [
    "I will put my final answer",
    "I will now return",
    "S0:",
    "S1:",
    "S2:",
    "S3:",
    "S4:",
    "S5:",
]

GLOBAL_FORBIDDEN = [
    "__BS__",
    "__LB__",
    "__RB__",
    "AGREE",
    "BA_DC|rev encrypted digit search",
    "BA_DC|rev and AB_CD|raw",
    "Since a direct template is locked",
    "This is an operator-family drill",
    "This is a symbol-digit conversion drill",
    "This is a methodology card",
    "This is an ambiguity discipline card",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase 1 symbol-equation transformation curriculum rows.")
    parser.add_argument(
        "--csv",
        default="data/trainable/phase1_components/phase1_symbol_transform_direct_curriculum.csv",
    )
    parser.add_argument(
        "--report-json",
        default="data/trainable/phase1_components/phase1_symbol_transform_direct_curriculum.full_audit.json",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def direct_prediction(template_name: str, lhs: str) -> str:
    return "".join(lhs[i] for i in DIRECT_TEMPLATES[template_name])


def source_template(category: str) -> str | None:
    if category.endswith("_0134"):
        return "0134"
    if category.endswith("_3401"):
        return "3401"
    return None


def final_box_line(answer: str) -> str:
    return f"The final answer is \\boxed{{{answer}}}"


def boxed_contains_exact_answer(cot: str, answer: str) -> bool:
    return final_box_line(answer) in cot


def reference_extract_final_answer(text: str | None) -> str:
    """Mirror the local reference/evaluation.py boxed-answer extraction logic."""
    if text is None:
        return "NOT_FOUND"
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return "NOT_FOUND"


def audit_direct_row(row: dict[str, str]) -> list[str]:
    issues: list[str] = []
    template = source_template(row["source_category"])
    if template not in DIRECT_TEMPLATE_PRIORITY:
        return ["direct category does not end in a known template"]

    puzzle = parse_symbol_transform_puzzle(row["prompt"])
    if puzzle is None:
        return ["prompt did not parse as symbol-equation transformation"]

    same = same_operator_examples(puzzle)
    if not same:
        issues.append("no same-operator examples")

    answer = row["answer"]
    predicted = direct_prediction(template, puzzle.query)
    if predicted != answer:
        issues.append(f"query direct prediction {predicted!r} != answer {answer!r}")

    for prior in DIRECT_TEMPLATE_PRIORITY:
        same_preds = [direct_prediction(prior, example.lhs) for example in same]
        passes = [pred == example.rhs for pred, example in zip(same_preds, same)]
        if prior == template:
            if not all(passes):
                issues.append(f"chosen template {template} does not pass all same-op examples")
            break
        if all(passes):
            issues.append(f"higher priority template {prior} also passes before {template}")

    cot = row["generated_cot"]
    required = [
        "We need to deduce the hidden symbol transformation rule by matching the example outputs.",
        "I will put my final answer inside \\boxed{}.",
        DIRECT_S0,
        "S1: Classify this as symbol-equation transformation with fixed shape ABOCD.",
        "S2: Test direct-position templates first.",
        "S3: Apply the locked template to the query.",
        "I will now return the answer in \\boxed{}.",
        final_box_line(answer),
    ]
    for marker in required:
        if marker not in cot:
            issues.append(f"missing direct trace marker: {marker[:80]}")

    if template == "0134" and "Template 3401: SKIP because Template 0134 already LOCKED by priority." not in cot:
        issues.append("0134 row missing 3401 SKIP priority line")
    if template == "3401" and "Template 0134:" not in cot:
        issues.append("3401 row missing 0134 failure attempt")

    if "LOCK Template" not in cot:
        issues.append("missing LOCK wording")
    return issues


def compact_expectation(category: str) -> dict[str, object] | None:
    if category in COMPACT_EXPECTATIONS:
        return COMPACT_EXPECTATIONS[category]
    if category.startswith("rhs_length_family_drill_"):
        return {
            "first_prefix": "Length ",
            "required": ["candidate rules are", "The final answer is \\boxed{"],
        }
    if category.startswith("route_"):
        return {
            "first": "We need to recall one reusable symbol-equation transformation rule.",
            "required": [
                "State the requested symbol-equation transformation rule directly.",
                "The final answer is \\boxed{",
            ],
        }
    return None


def audit_compact_row(row: dict[str, str]) -> list[str]:
    issues: list[str] = []
    category = row["source_category"]
    expectation = compact_expectation(category)
    if expectation is None:
        return [f"unexpected source_category: {category}"]
    cot = row["generated_cot"]
    lines = cot.splitlines()
    if "first" in expectation and (not lines or lines[0] != expectation["first"]):
        issues.append("first line does not match compact trace format")
    if "first_prefix" in expectation and (not lines or not lines[0].startswith(str(expectation["first_prefix"]))):
        issues.append("first line does not match compact trace prefix")
    for marker in expectation["required"]:  # type: ignore[index]
        if marker not in cot:
            issues.append(f"missing compact marker: {marker}")
    for marker in COMPACT_FORBIDDEN:
        if marker in cot:
            issues.append(f"compact trace contains verbose marker: {marker}")
    if not boxed_contains_exact_answer(cot, row["answer"]):
        issues.append("final boxed line does not contain exact answer literal")
    return issues


def audit_answer_alignment(row: dict[str, str]) -> list[str]:
    issues: list[str] = []
    if not boxed_contains_exact_answer(row["generated_cot"], row["answer"]):
        issues.append("expected literal final boxed line not found")
    if row["answer"] not in row["generated_cot"]:
        issues.append("answer string not present in generated_cot")
    return issues


def main() -> None:
    args = parse_args()
    path = ROOT / args.csv
    rows = read_rows(path)

    issues_by_id: dict[str, list[str]] = {}
    warnings_by_id: dict[str, list[str]] = {}
    reference_extract_mismatches: dict[str, dict[str, str]] = {}
    answer_special_char_counts = Counter()
    counts = Counter(row["source_category"] for row in rows)

    duplicate_ids = [row_id for row_id, n in Counter(row["id"] for row in rows).items() if n > 1]

    for row in rows:
        row_issues: list[str] = []
        row_warnings: list[str] = []
        category = row["source_category"]

        for field in ("id", "prompt", "answer", "generated_cot", "source_category"):
            if not row.get(field, "").strip():
                row_issues.append(f"empty field: {field}")

        text = row.get("generated_cot", "")
        for marker in GLOBAL_FORBIDDEN:
            if marker in text:
                row_issues.append(f"forbidden marker: {marker}")

        row_issues.extend(audit_answer_alignment(row))

        if "direct_template" in category:
            row_issues.extend(audit_direct_row(row))
        elif compact_expectation(category) is not None:
            row_issues.extend(audit_compact_row(row))
        else:
            row_issues.append(f"unexpected source_category: {category}")

        if any(ch in row["answer"] for ch in "\\{}"):
            row_warnings.append("answer contains backslash or brace; literal boxed text may stress simple extractors")
        for ch in set(row["answer"]):
            if ch in "\\{}":
                answer_special_char_counts[ch] += 1
        extracted = reference_extract_final_answer(text)
        if extracted != row["answer"]:
            reference_extract_mismatches[row["id"]] = {
                "source_category": category,
                "answer": row["answer"],
                "reference_extracted": extracted,
            }
        if len(text) > 7600:
            row_issues.append("generated_cot exceeds 7600 characters")
        elif len(text) > 5000:
            row_warnings.append("generated_cot is long")

        if row_issues:
            issues_by_id[row["id"]] = row_issues
        if row_warnings:
            warnings_by_id[row["id"]] = row_warnings

    warning_counts = Counter()
    warnings_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    id_to_category = {row["id"]: row["source_category"] for row in rows}
    for row_id, warnings in warnings_by_id.items():
        for warning in warnings:
            warning_counts[warning] += 1
            warnings_by_category[id_to_category[row_id]][warning] += 1

    direct_total = sum(n for key, n in counts.items() if "direct_template" in key)
    compact_total = sum(n for key, n in counts.items() if compact_expectation(key) is not None)

    report = {
        "csv": str(path),
        "total_rows": len(rows),
        "source_category_counts": dict(counts),
        "direct_template_total": direct_total,
        "compact_support_total": compact_total,
        "duplicate_ids": duplicate_ids,
        "issue_count": sum(len(items) for items in issues_by_id.values()),
        "rows_with_issues": len(issues_by_id),
        "issues_sample": dict(list(issues_by_id.items())[:20]),
        "warning_count": sum(warning_counts.values()),
        "rows_with_warnings": len(warnings_by_id),
        "warning_counts": dict(warning_counts),
        "warnings_by_category": {key: dict(value) for key, value in warnings_by_category.items()},
        "warning_sample": dict(list(warnings_by_id.items())[:20]),
        "answer_special_char_counts": dict(answer_special_char_counts),
        "reference_extractor_mismatch_count": len(reference_extract_mismatches),
        "reference_extractor_mismatch_by_category": dict(
            Counter(item["source_category"] for item in reference_extract_mismatches.values())
        ),
        "reference_extractor_mismatch_sample": dict(list(reference_extract_mismatches.items())[:20]),
    }

    report_path = ROOT / args.report_json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if duplicate_ids or issues_by_id:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
