#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*$", re.DOTALL)
WONDERLAND_NUMERIC_PREFIX = "In Alice's Wonderland numeric-equation transformation rules, "

DEFAULT_INPUTS = (
    "data/trainable/phase1_components/phase1a_numeric_equation_knowledge.csv",
    "data/trainable/phase1_components/phase1b_numeric_equation_methodology.csv",
)

OUTPUT_COLUMNS = [
    "id",
    "prompt",
    "answer",
    "generated_cot",
    "label",
    "category",
    "source",
    "source_category",
]

PAIRING_TRANSFORM_TARGETS = {
    "AB_DC": 40,
    "BA_CD": 40,
    "AB_CD": 22,
    "BA_DC": 22,
}

BASE_RULE_SEMANTICS_QUOTAS = {
    "x // y": 6,
    "x + y + x*y": 7,
    "concat(y, x)": 10,
    "x mod y": 12,
    "y mod x": 13,
    "y - x": 24,
    "x + y - 1": 14,
    "x * y + 1": 7,
    "x * y - 1": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge numeric_equation Phase 1A/1B component files into one active "
            "Phase 1 curriculum, deduplicate exact prompt+answer pairs, and "
            "normalize CoT traces to the boxed-final-answer style."
        )
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        default=None,
        help="Input numeric Phase 1 component CSV. Repeat to override defaults.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1_numeric_equation_curriculum.csv",
        help="Merged numeric-equation Phase 1 curriculum CSV.",
    )
    return parser.parse_args()


def strip_think(text: str) -> str:
    match = THINK_RE.match(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def normalize_cot(text: str, answer: str) -> str:
    reasoning = strip_think(text)
    reasoning = capitalize_first(reasoning)
    final_line = f"The final answer is \\boxed{{{answer}}}"
    if not reasoning:
        return final_line
    if final_line in reasoning:
        return reasoning
    return f"{reasoning}\n\n{final_line}"


def clean_common_text(text: str) -> str:
    replacements = (
        ("numeric_equation Phase 1A knowledge", "numeric-equation rule knowledge"),
        ("accepted numeric_equation Phase 1A", "accepted numeric-equation"),
        ("core Phase 1A curriculum", "core rule set"),
        ("Phase 1A curriculum", "core rule set"),
        ("accepted 696-row numeric-equation DSL", "accepted numeric-equation rule set"),
        ("numeric-equation DSL", "numeric-equation rule set"),
        ("DSL boundary", "core rule set boundary"),
        ("DSL", "rule set"),
        ("numeric_equation", "numeric-equation"),
    )
    cleaned = text
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    return cleaned


def normalize_numeric_prompt(prompt: str) -> str:
    """Keep numeric-equation Phase 1 prompts close to the competition wrapper."""
    replacements = (
        ("In Alice's Wonderland numeric-equation methodology, ", WONDERLAND_NUMERIC_PREFIX),
        ("In numeric-equation rule knowledge, ", WONDERLAND_NUMERIC_PREFIX),
        ("For accepted numeric-equation scan priority, ", WONDERLAND_NUMERIC_PREFIX + "for scan priority, "),
        ("In accepted numeric-equation scan priority, ", WONDERLAND_NUMERIC_PREFIX + "for scan priority, "),
        ("Apply accepted numeric-equation pairing ", WONDERLAND_NUMERIC_PREFIX + "apply pairing "),
        ("Apply accepted combo ", WONDERLAND_NUMERIC_PREFIX + "apply combo "),
        ("Using pairing ", WONDERLAND_NUMERIC_PREFIX + "using pairing "),
        ("Render value ", WONDERLAND_NUMERIC_PREFIX + "render value "),
        ("what should the solver do?", "what is the correct action?"),
        ("What should it do?", "What is the correct action?"),
        ("the solver observes:", "we observe:"),
        ("the numeric-equation solver", "Alice's Wonderland numeric-equation transformation rules"),
        ("numeric-equation solver", "Alice's Wonderland numeric-equation transformation rules"),
    )
    cleaned = prompt
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    return cleaned


def remove_frequency_details(text: str) -> str:
    return re.sub(r"\s*\(\d+\)", "", text)


def capitalize_first(text: str) -> str:
    stripped = text.lstrip()
    if not stripped:
        return text
    offset = len(text) - len(stripped)
    return text[:offset] + stripped[0].upper() + stripped[1:]


def top_n_items(answer: str, n: int) -> str:
    return "; ".join(item.strip() for item in answer.split(";")[:n])


def ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def priority_inventory_rows(row: dict[str, str]) -> list[tuple[str, str, str]]:
    answer = clean_common_text(row["answer"].strip())
    combos = [item.strip() for item in answer.split(";") if item.strip()]
    rows: list[tuple[str, str, str]] = []
    for n in range(1, min(10, len(combos)) + 1):
        top_answer = "; ".join(combos[:n])
        if n == 1:
            prompt = WONDERLAND_NUMERIC_PREFIX + "what is the top scan combination to try first?"
            reasoning = f"The top accepted combination is {top_answer}."
        else:
            prompt = WONDERLAND_NUMERIC_PREFIX + f"what are the top {n} scan combinations to try first?"
            reasoning = f"The top {n} accepted combinations are {top_answer}."
        rows.append((prompt, top_answer, reasoning))
    for index, combo in enumerate(combos[:10], start=1):
        rank = ordinal(index)
        prompt = WONDERLAND_NUMERIC_PREFIX + f"which scan combination is ranked {rank}?"
        reasoning = f"The {rank} accepted scan combination is {combo}."
        rows.append((prompt, combo, reasoning))
    return rows


def apply_pairing_text(pairing: str, left: str, right: str) -> tuple[int, int, str]:
    if pairing == "AB_CD":
        return int(left), int(right), "Keep both operands as written"
    if pairing == "AB_DC":
        return int(left), int(right[::-1]), "Keep the left operand as written and reverse the right operand"
    if pairing == "BA_CD":
        return int(left[::-1]), int(right), "Reverse the left operand and keep the right operand as written"
    if pairing == "BA_DC":
        return int(left[::-1]), int(right[::-1]), "Reverse both operands"
    raise ValueError(f"Unknown pairing: {pairing}")


def pairing_transform_rows() -> list[tuple[dict[str, str], str, str, str]]:
    rows: list[tuple[dict[str, str], str, str, str]] = []
    seeds = [
        ("39", "30"),
        ("07", "30"),
        ("72", "10"),
        ("84", "36"),
        ("57", "77"),
        ("21", "28"),
        ("46", "78"),
        ("05", "92"),
        ("63", "40"),
        ("18", "06"),
        ("90", "13"),
        ("34", "85"),
        ("69", "52"),
        ("11", "74"),
        ("48", "09"),
        ("82", "61"),
        ("56", "02"),
        ("75", "62"),
        ("44", "54"),
        ("31", "15"),
        ("64", "25"),
        ("88", "21"),
        ("12", "99"),
        ("03", "47"),
        ("29", "80"),
        ("70", "51"),
        ("91", "16"),
        ("24", "65"),
        ("53", "45"),
        ("80", "32"),
        ("22", "84"),
        ("61", "49"),
        ("38", "90"),
        ("14", "73"),
        ("95", "43"),
        ("06", "67"),
        ("41", "32"),
        ("87", "64"),
        ("50", "49"),
        ("98", "99"),
    ]
    for pairing, target in PAIRING_TRANSFORM_TARGETS.items():
        for index in range(target):
            left, right = seeds[index % len(seeds)]
            x, y, action = apply_pairing_text(pairing, left, right)
            prompt = (
                WONDERLAND_NUMERIC_PREFIX
                + f"apply pairing {pairing} to left={left}, right={right}. Return operands x,y."
            )
            answer = f"{x},{y}"
            reasoning = f"{action}: left={left}, right={right} gives x={x}, y={y}."
            row = {
                "id": f"ne_synth_pairing_transform_{pairing}_{index:03d}",
                "source_category": "pairing_transform",
            }
            rows.append((row, prompt, answer, reasoning))
    return rows


def count_word(value: str) -> str:
    return {
        "0": "no",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
    }.get(value, value)


def singular_or_plural(count: str, singular: str, plural: str) -> str:
    return singular if count == "one" else plural


def remove_variant_marker(text: str) -> str:
    return re.sub(r"\s*\(variant \d+\)", "", text)


def normalize_combo_display(text: str) -> str:
    return (
        text.replace("abs(x - y)", "|x-y|")
        .replace("abs(x-y)", "|x-y|")
        .replace("||x - y||", "||x-y||")
    )


def split_render_sentence(reasoning: str) -> str:
    return re.sub(r": value (.+?) renders as", r".\nValue \1 renders as", reasoning)


def rewrite_op_suffix_rev_if_neg(reasoning: str, answer: str) -> str:
    if "op_suffix_rev_if_neg" not in reasoning and "if negative, reverse the magnitude and append the operator" not in reasoning:
        return reasoning
    matches = re.findall(r"=\s*(-?\d+)", reasoning)
    if not matches:
        return reasoning
    value = int(matches[-1])
    if value < 0:
        magnitude = str(abs(value))
        rendered = magnitude[::-1]
        return (
            re.sub(
                r"\s*Then if negative, reverse the magnitude and append the operator; otherwise write it directly gives .*?\.$",
                "",
                reasoning,
            ).strip()
            + f"\n{value} is negative, so reverse the magnitude {magnitude} -> {rendered} and append the operator."
        )
    return (
        re.sub(
            r"\s*Then if negative, reverse the magnitude and append the operator; otherwise write it directly gives .*?\.$",
            "",
            reasoning,
        ).strip()
        + f"\n{value} is not negative, keep it the same and do not append suffix."
    )


def clean_row(row: dict[str, str]) -> tuple[str, str, str]:
    row_id = row["id"].strip()
    source_category = row.get("source_category", "").strip()
    prompt = clean_common_text(row["prompt"].strip())
    answer = clean_common_text(row["answer"].strip())
    reasoning = clean_common_text(strip_think(row.get("generated_cot", "")))
    reasoning = reasoning.replace("the solver", "the method").replace("The solver", "The method")

    if row_id == "ne_accepteddsl_inventory_pairings":
        prompt = WONDERLAND_NUMERIC_PREFIX + "which operand pairings can appear?"
        answer = "BA_DC, AB_CD, BA_CD, AB_DC"
        reasoning = "The observed pairings are BA_DC, AB_CD, BA_CD, AB_DC."
    elif row_id == "ne_accepteddsl_inventory_base_rules":
        prompt = WONDERLAND_NUMERIC_PREFIX + "which base rules can appear?"
        reasoning = f"The base-rule inventory is {answer}."
    elif row_id == "ne_accepteddsl_inventory_output_modes":
        prompt = WONDERLAND_NUMERIC_PREFIX + "which output modes can appear?"
        reasoning = f"The output-mode inventory is {answer}."
    elif row_id == "ne_accepteddsl_top_priority_combos":
        prompt, answer, reasoning = priority_inventory_rows(row)[3]
    elif row_id.startswith("ne_inventory_pairing_principle_"):
        match = re.search(r"pairing ([A-Z_]+)", prompt)
        pairing_name = match.group(1) if match else "this pairing"
        prompt = WONDERLAND_NUMERIC_PREFIX + f"what does pairing {pairing_name} mean?"
        reasoning = f"Pairing {pairing_name} means {answer}."
    elif source_category == "priority_inventory":
        prompt = prompt.replace(
            "what are the highest-priority accepted rule combinations?",
            "what are the highest-priority numeric-equation rule combinations?",
        )
        if "which combo should be tried first:" in prompt:
            reasoning = f"Among these two options, {answer} has higher accepted scan priority, so try it first."
        else:
            reasoning = f"The top accepted combinations are {answer}."
    elif source_category == "dsl_boundary":
        prompt = prompt.replace("Should accepted numeric-equation include", "Should Alice's Wonderland numeric-equation transformation rules include")
        prompt = prompt.replace(" as core knowledge", "")
        match = re.search(r"include (.+?)\?$", prompt)
        rule_name = match.group(1) if match else "this rule"
        for prefix in ("base rule ", "output mode "):
            if rule_name.startswith(prefix):
                rule_name = rule_name[len(prefix) :]
        prompt = (
            WONDERLAND_NUMERIC_PREFIX
            + f"should {rule_name} belong to the core tier?"
        )
        if answer == "Yes":
            reasoning = (
                f"Yes. {rule_name} appears in accepted labelled numeric-equation rows, "
                "so it belongs to the core tier."
            )
        else:
            reasoning = (
                f"No. {rule_name} is not used by accepted labelled numeric-equation rows, "
                "so it should not be part of the core rule set."
            )
    else:
        reasoning = remove_frequency_details(reasoning)

    if source_category == "pairing_transform":
        prompt = prompt.replace("Return x,y.", "Return operands x,y.")

    if source_category in {"combo_application", "output_mode_semantics"}:
        reasoning = rewrite_op_suffix_rev_if_neg(reasoning, answer)

    if source_category == "output_mode_semantics":
        reasoning = split_render_sentence(reasoning)

    if source_category == "procedure_card" and row_id == "ne_meth_step_order":
        answer = "Use same-op evidence first, infer the rule, verify on examples, then apply to the query."
        reasoning = answer

    if source_category == "same_op_decision_matrix":
        match = re.search(
            r"suppose the query has (\d+) same-operator example\(s\) and (\d+) candidate rule\(s\) fit those examples",
            prompt,
        )
        if match:
            same_op = count_word(match.group(1))
            candidates = count_word(match.group(2))
            prompt = (
                WONDERLAND_NUMERIC_PREFIX
                + f"suppose the query has {same_op} "
                + singular_or_plural(same_op, "same-operator example", "same-operator examples")
                + f" and {candidates} fitting "
                + singular_or_plural(candidates, "candidate rule", "candidate rules")
                + ". Which pathway should be used?"
            )

    if source_category == "ambiguity_handling":
        prompt = remove_variant_marker(prompt)
        prompt, reasoning = {
            "declare ambiguity and apply deterministic tie-break": (
                WONDERLAND_NUMERIC_PREFIX
                + "same-operator evidence has 53&45 = 8. Both x-y and |x-y| fit this example, "
                "but query 38&90 would differ. What is the correct action?",
                "Evidence: 53 - 45 = 8 and |53 - 45| = 8, so the visible positive example cannot distinguish signed subtraction from absolute difference.\n"
                "For query 38&90, x-y gives -52 while |x-y| gives 52.\n"
                "Because the query answer would differ, declare ambiguity and apply a deterministic tie-break.",
            ),
            "prefer rule that matches same-op examples exactly; if tie remains, use declared base priority": (
                WONDERLAND_NUMERIC_PREFIX
                + "same-operator evidence has 22*22 = 2222. Both concat(x, y) and concat(y, x) fit this example, "
                "but query 12*34 would differ. What is the correct action?",
                "Evidence: concat(22, 22) = 2222 and concat(22, 22) = 2222, so the visible same-op example cannot distinguish concat(x, y) from concat(y, x).\n"
                "For query 12*34, concat(x, y) gives 1234 while concat(y, x) gives 3412.\n"
                "First prefer the rule that matches same-op examples exactly; if the tie remains, use declared base priority.",
            ),
            "mark as low-confidence ambiguity": (
                WONDERLAND_NUMERIC_PREFIX
                + "same-operator evidence has 59#62 = 121. Both plain and rev output modes fit because 121 is palindromic, "
                "but query 40#62 would differ. What is the correct action?",
                "Evidence: 59 + 62 = 121, and reversing 121 still gives 121, so plain and rev both fit the same-op example.\n"
                "For query 40#62, plain gives 102 while rev gives 201.\n"
                "Because the output mode is not uniquely identified, mark this as a low-confidence ambiguity.",
            ),
            "use secondary motif ranking policy": (
                WONDERLAND_NUMERIC_PREFIX
                + "one visible operator supports BA_DC|rev and one visible operator supports AB_CD|plain. "
                "The query operator is absent and motif support is tied. What is the correct action?",
                "Evidence: one non-query operator supports BA_DC|rev and one non-query operator supports AB_CD|plain.\n"
                "The query operator has no same-op examples, so neither motif wins by support count.\n"
                "When motif support ties, use the secondary motif ranking policy.",
            ),
        }.get(answer, (prompt, reasoning))

    if source_category == "tiebreak_policy":
        prompt = remove_variant_marker(prompt)
        reasoning = f"Use a deterministic policy: {answer}."

    if source_category == "absent_op_workflow":
        prompt = re.sub(r"query operator is absent", "the query operator does not appear in the examples", prompt)
        prompt = re.sub(
            r"and (\d+) bases are already used by visible operators",
            lambda m: (
                f"and {count_word(m.group(1))} "
                f"{singular_or_plural(count_word(m.group(1)), 'base is', 'bases are')} already used by visible operators"
            ),
            prompt,
        )
        reasoning = (
            "When the query operator is absent, lock the row motif from visible operators, "
            "skip bases already used by visible operator symbols, then test the next base in declared priority order."
        )

    if source_category == "sanity_checks":
        prompt = remove_variant_marker(prompt)
        reasoning = {
            "accept only if width formatting matches examples": (
                "Check that the candidate preserves width behavior such as required leading zeros."
            ),
            "accept only if sign-trigger behavior is consistent": (
                "Check that operator markers appear only in the sign-triggered branch shown by the examples."
            ),
            "downgrade confidence and keep ambiguity": (
                "If the query prediction disagrees with the same-operator pattern, downgrade confidence and keep ambiguity."
            ),
            "reject candidate": (
                "If a candidate fails any same-operator example, reject it."
            ),
            "keep candidate but mark medium confidence": (
                "If same-operator examples pass but the row motif conflicts strongly, keep the candidate but mark medium confidence."
            ),
        }.get(answer, reasoning)

    prompt = clean_common_text(prompt)
    prompt = normalize_numeric_prompt(prompt)
    reasoning = clean_common_text(reasoning)
    prompt = normalize_combo_display(prompt)
    answer = normalize_combo_display(answer)
    reasoning = normalize_combo_display(reasoning)
    return prompt, answer, reasoning


def clean_source_category(row: dict[str, str]) -> str:
    source_category = row.get("source_category", "").strip()
    if row.get("id", "").startswith("ne_inventory_pairing_principle_"):
        return "pairing_principle"
    replacements = {
        "accepted_dsl_inventory": "accepted_rule_inventory",
        "dsl_boundary": "rule_boundary",
    }
    return replacements.get(source_category, source_category)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(OUTPUT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def main() -> None:
    args = parse_args()
    input_paths = [ROOT / item for item in (args.input_csv or DEFAULT_INPUTS)]
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, str]] = []
    seen_prompt_answer: set[tuple[str, str]] = set()
    input_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    unsafe_answer_counts: Counter[str] = Counter()
    selected_base_rule_counts: Counter[str] = Counter()

    def add_output_row(row: dict[str, str], input_path: Path, prompt: str, answer: str, reasoning: str) -> None:
        prompt = normalize_combo_display(normalize_numeric_prompt(clean_common_text(prompt)))
        answer = normalize_combo_display(clean_common_text(answer))
        reasoning = normalize_combo_display(clean_common_text(reasoning))
        key = (prompt, answer)
        if any(ch in answer for ch in "\\{}"):
            unsafe_answer_counts[input_path.name] += 1
            return
        if key in seen_prompt_answer:
            duplicate_counts[input_path.name] += 1
            return
        seen_prompt_answer.add(key)
        source_category = clean_source_category(row)
        source_counts[source_category] += 1
        rows_out.append(
            {
                "id": f"ne_phase1_curriculum_{len(rows_out):04d}",
                "prompt": prompt,
                "answer": answer,
                "generated_cot": normalize_cot(reasoning, answer),
                "label": "Numeric Equation Curriculum",
                "category": "Phase1 Numeric Equation Curriculum",
                "source": "numeric_equation_phase1_curriculum",
                "source_category": source_category,
            }
        )

    for input_path in input_paths:
        if not input_path.exists():
            raise SystemExit(f"Input component not found: {input_path}")
        for row in read_rows(input_path):
            input_counts[input_path.name] += 1
            source_category = row.get("source_category", "").strip()
            if source_category in {"confidence_gating", "ambiguity_handling", "pairing_transform"}:
                continue
            if row.get("id", "").strip() == "ne_accepteddsl_top_priority_combos":
                for prompt, answer, reasoning in priority_inventory_rows(row):
                    add_output_row(row, input_path, prompt, answer, reasoning)
                continue
            prompt, answer, reasoning = clean_row(row)
            if source_category == "base_rule_semantics":
                match = re.search(r"base rule (.+?) for left=", prompt)
                rule_name = match.group(1) if match else ""
                quota = BASE_RULE_SEMANTICS_QUOTAS.get(rule_name, 0)
                if selected_base_rule_counts[rule_name] >= quota:
                    continue
                selected_base_rule_counts[rule_name] += 1
            add_output_row(row, input_path, prompt, answer, reasoning)

    synthetic_input_path = Path("synthetic_numeric_pairing_transform")
    for row, prompt, answer, reasoning in pairing_transform_rows():
        input_counts[synthetic_input_path.name] += 1
        add_output_row(row, synthetic_input_path, prompt, answer, reasoning)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} numeric-equation Phase 1 rows to {output_path}")
    print(f"Input counts: {dict(input_counts)}")
    print(f"Deduplicated rows: {sum(duplicate_counts.values())} ({dict(duplicate_counts)})")
    print(f"Skipped unsafe boxed answers: {sum(unsafe_answer_counts.values())} ({dict(unsafe_answer_counts)})")
    print(f"Source categories: {dict(source_counts)}")


if __name__ == "__main__":
    main()
