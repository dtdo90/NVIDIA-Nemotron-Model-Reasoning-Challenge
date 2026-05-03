#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/trainable/phase1_components/phase1_numeric_equation_curriculum.csv"
PREFIX = "In Alice's Wonderland numeric-equation transformation rules,"
FINAL_PREFIX = "The final answer is \\boxed{"
FORBIDDEN_TEXT = (
    "numeric-equation solver",
    "the solver",
    "The solver",
    "What should it do?",
    "variant ",
    "Phase 1A",
    "Phase 1B",
    "DSL",
    "dsl",
    "<think>",
    "</think>",
    "candidate rule(s)",
    "example(s)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the active numeric-equation Phase 1 curriculum.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Curriculum CSV to audit.")
    parser.add_argument("--show-examples", action="store_true", help="Print one example per subtype.")
    return parser.parse_args()


def apply_pairing(pairing: str, left: str, right: str) -> tuple[int, int]:
    if pairing == "AB_CD":
        return int(left), int(right)
    if pairing == "AB_DC":
        return int(left), int(right[::-1])
    if pairing == "BA_CD":
        return int(left[::-1]), int(right)
    if pairing == "BA_DC":
        return int(left[::-1]), int(right[::-1])
    raise ValueError(f"unknown pairing {pairing}")


def apply_base_rule(rule: str, x: int, y: int) -> int | str:
    if rule == "x + y":
        return x + y
    if rule == "x + y + 1":
        return x + y + 1
    if rule == "x + y - 1":
        return x + y - 1
    if rule == "x - y":
        return x - y
    if rule == "y - x":
        return y - x
    if rule in {"|x-y|", "|x - y|"}:
        return abs(x - y)
    if rule == "x * y":
        return x * y
    if rule == "x * y + 1":
        return x * y + 1
    if rule == "x * y - 1":
        return x * y - 1
    if rule == "x + y + x*y":
        return x + y + x * y
    if rule == "concat(x, y)":
        return f"{x}{y}"
    if rule == "concat(y, x)":
        return f"{y}{x}"
    if rule == "x // y":
        return x // y
    if rule == "x mod y":
        return x % y
    if rule == "y mod x":
        return y % x
    raise ValueError(f"unknown base rule {rule}")


def render_output(value: int | str, mode: str, operator: str) -> str:
    text = str(value)
    negative = text.startswith("-")
    magnitude = text[1:] if negative else text
    signed_rev = ("-" if negative else "") + magnitude[::-1]
    if mode == "plain":
        return text
    if mode == "rev":
        return signed_rev
    if mode == "neg":
        return f"-{magnitude}"
    if mode == "neg_rev":
        return f"-{magnitude[::-1]}"
    if mode == "abs_rev":
        return magnitude[::-1]
    if mode == "op_prefix":
        return f"{operator}{text}"
    if mode == "op_suffix":
        return f"{text}{operator}"
    if mode == "op_prefix_rev":
        return f"{operator}{signed_rev}"
    if mode == "op_suffix_rev":
        return f"{signed_rev}{operator}"
    if mode == "op_prefix_if_neg":
        return f"{operator}{magnitude}" if negative else text
    if mode == "op_suffix_rev_if_neg":
        return f"{magnitude[::-1]}{operator}" if negative else text
    if mode == "rev_or_op_prefix_rev_if_neg":
        return f"{operator}{magnitude[::-1]}" if negative else magnitude[::-1]
    if mode == "rev_or_op_suffix_if_neg":
        return f"{magnitude}{operator}" if negative else magnitude[::-1]
    if mode == "plain_or_op_prefix_rev_if_neg":
        return f"{operator}{magnitude[::-1]}" if negative else text
    raise ValueError(f"unknown output mode {mode}")


def split_combo(combo: str) -> tuple[str, str, str]:
    pairing, rest = combo.split("|", 1)
    mode_start = rest.rfind("|")
    if mode_start == -1:
        raise ValueError(f"cannot split combo {combo}")
    return pairing, rest[:mode_start], rest[mode_start + 1 :]


def final_line(answer: str) -> str:
    return f"{FINAL_PREFIX}{answer}}}"


def first_alpha_is_upper(text: str) -> bool:
    for char in text.lstrip():
        if char.isalpha():
            return char.isupper()
        if not char.isspace():
            return True
    return True


def audit_row(row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    prompt = row["prompt"]
    answer = row["answer"]
    cot = row["generated_cot"]
    category = row["source_category"]

    if not prompt.startswith(PREFIX):
        errors.append("prompt does not start with Wonderland numeric-equation transformation prefix")
    joined = "\n".join([prompt, answer, cot, category])
    for bad in FORBIDDEN_TEXT:
        if bad in joined:
            errors.append(f"forbidden text remains: {bad}")
    if "abs(x" in joined:
        errors.append("absolute-difference rule should use |x-y| notation, not abs(...)")
    if category == "confidence_gating":
        errors.append("confidence_gating should not be active")
    if not first_alpha_is_upper(cot):
        errors.append("CoT does not start with a capital letter")
    expected_final = final_line(answer)
    if cot.count(expected_final) != 1:
        errors.append("CoT does not contain exactly one final line matching answer")
    if cot.count(FINAL_PREFIX) != 1:
        errors.append("CoT has wrong number of boxed final-answer lines")
    if not cot.rstrip().endswith(expected_final):
        errors.append("final boxed answer is not the last content")
    if any(ch in answer for ch in "\\{}"):
        errors.append("answer contains unsafe boxed literal character")

    if category == "priority_inventory":
        top_match = re.search(r"top(?: (\d+))? scan combination", prompt)
        if top_match:
            expected_count = int(top_match.group(1) or "1")
            if len([part for part in answer.split(";") if part.strip()]) != expected_count:
                errors.append(f"top priority answer does not contain exactly {expected_count} combinations")
            if expected_count == 1 and "top accepted combination" not in cot:
                errors.append("top priority CoT missing top wording")
            if expected_count > 1 and f"top {expected_count} accepted combinations" not in cot:
                errors.append(f"top-{expected_count} priority CoT missing matching wording")
        elif "which combo should be tried first:" in prompt:
            match = re.search(r"which combo should be tried first: (.+) or (.+)\?$", prompt)
            if not match:
                errors.append("pairwise priority prompt is not parseable")
            else:
                choices = {match.group(1), match.group(2)}
                if answer not in choices:
                    errors.append("pairwise priority answer is not one of the two choices")
            if "higher accepted scan priority" not in cot:
                errors.append("pairwise priority CoT missing accurate priority wording")
        elif re.search(r"which scan combination is ranked \d+(?:st|nd|rd|th)\?$", prompt):
            if "accepted scan combination is" not in cot:
                errors.append("rank-specific priority CoT missing rank wording")
            if ";" in answer:
                errors.append("rank-specific priority answer should contain exactly one combination")
        else:
            errors.append("priority_inventory prompt shape is not recognized")

    if category == "pairing_transform":
        match = re.search(r"apply pairing (.+?) to left=(\d+), right=(\d+)\. Return operands x,y\.$", prompt)
        if not match:
            errors.append("pairing prompt is not parseable or lacks operands x,y clarity")
        else:
            x, y = apply_pairing(match.group(1), match.group(2), match.group(3))
            if answer != f"{x},{y}":
                errors.append(f"pairing answer mismatch, expected {x},{y}")

    if category == "base_rule_semantics":
        match = re.search(
            r"using pairing (.+?), compute accepted base rule (.+) for left=(\d+), right=(\d+)\.$",
            prompt,
        )
        if not match:
            errors.append("base-rule prompt is not parseable")
        else:
            x, y = apply_pairing(match.group(1), match.group(3), match.group(4))
            expected = str(apply_base_rule(match.group(2), x, y))
            if answer != expected:
                errors.append(f"base-rule answer mismatch, expected {expected}")

    if category == "output_mode_semantics":
        match = re.search(r"render value (-?\d+) with accepted output mode (.+?) and operator '(.+)'\.$", prompt)
        if not match:
            errors.append("output-mode prompt is not parseable")
        else:
            value, mode, operator = match.groups()
            expected = render_output(int(value), mode, operator)
            if answer != expected:
                errors.append(f"output-mode answer mismatch, expected {expected}")
            if ": value " in cot.lower():
                errors.append("output-mode CoT should split rule and render sentence onto separate lines")
            if "\nValue " not in cot:
                errors.append("output-mode CoT missing separate Value render line")

    if category == "combo_application":
        match = re.search(r"apply combo (.+?) to expression (\d{2})(.)(\d{2})\.$", prompt)
        if not match:
            errors.append("combo prompt is not parseable")
        else:
            combo, left, operator, right = match.groups()
            pairing, base_rule, mode = split_combo(combo)
            x, y = apply_pairing(pairing, left, right)
            expected = render_output(apply_base_rule(base_rule, x, y), mode, operator)
            if answer != expected:
                errors.append(f"combo answer mismatch, expected {expected}")
            if mode == "op_suffix_rev_if_neg" and "do not append suffix" not in cot and "append the operator" not in cot:
                errors.append("op_suffix_rev_if_neg CoT missing explicit sign branch")

    if category == "procedure_card" and "correct high-level order" in prompt:
        if "->" in answer:
            errors.append("procedure high-level answer still uses arrows")
        if answer != "Use same-op evidence first, infer the rule, verify on examples, then apply to the query.":
            errors.append("procedure high-level answer is not the requested sentence")

    return errors


def main() -> None:
    args = parse_args()
    path = Path(args.csv)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    row_errors: list[tuple[str, str, str]] = []
    for row in rows:
        for error in audit_row(row):
            row_errors.append((row["id"], row["source_category"], error))

    duplicate_ids = [key for key, count in Counter(row["id"] for row in rows).items() if count > 1]
    duplicate_prompt_answers = [
        key for key, count in Counter((row["prompt"], row["answer"]) for row in rows).items() if count > 1
    ]
    duplicate_prompts = [key for key, count in Counter(row["prompt"] for row in rows).items() if count > 1]

    print(f"Rows: {len(rows)}")
    print(f"Subtype counts: {dict(Counter(row['source_category'] for row in rows))}")
    print(f"Duplicate ids: {len(duplicate_ids)}")
    print(f"Duplicate prompt+answer pairs: {len(duplicate_prompt_answers)}")
    print(f"Duplicate prompts: {len(duplicate_prompts)}")
    print(f"Audit errors: {len(row_errors)}")
    for row_id, category, error in row_errors[:50]:
        print(f"ERROR {row_id} [{category}]: {error}")

    if args.show_examples:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["source_category"]].append(row)
        for category, category_rows in grouped.items():
            row = category_rows[0]
            print(f"\n### {category} ({len(category_rows)})")
            print(f"Question: {row['prompt']}")
            print("Trace:")
            print(row["generated_cot"])

    if row_errors or duplicate_ids or duplicate_prompt_answers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
