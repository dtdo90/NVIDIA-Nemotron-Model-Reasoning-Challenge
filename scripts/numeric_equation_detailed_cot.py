from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HARNESS = ROOT / "reference/cursor/transformation_rules/numeric_equation/harness"
for path in (SRC, HARNESS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extended_dsl import (  # noqa: E402
    Candidate,
    OUTPUT_MODE_BY_NAME,
    PAIRING_NAMES,
    apply_pairing,
    enumerate_matching,
)
from nemotron_baseline.numeric_equation import (  # noqa: E402
    NumericEquationPuzzle,
    ParsedEquation,
    reverse_number_text,
)


OUTPUT_MODES_BY_LENGTH = sorted(OUTPUT_MODE_BY_NAME, key=len, reverse=True)

SCAN_PRIORITY: tuple[tuple[str, str, str], ...] = (
    ("BA_DC", "x + y", "rev"),
    ("BA_DC", "x * y", "rev"),
    ("BA_DC", "x - y", "rev"),
    ("BA_DC", "concat(x, y)", "rev"),
    ("BA_DC", "x * y - 1", "rev"),
    ("AB_CD", "concat(x, y)", "plain"),
    ("BA_DC", "x + y - 1", "rev"),
    ("BA_DC", "x * y + 1", "rev"),
    ("BA_DC", "x + y + 1", "rev"),
    ("AB_CD", "x + y", "plain"),
    ("AB_CD", "x - y", "plain"),
    ("AB_CD", "x * y + 1", "plain"),
    ("AB_CD", "x * y", "plain"),
    ("AB_CD", "x + y + 1", "plain"),
    ("BA_DC", "x - y", "rev_or_op_prefix_rev_if_neg"),
    ("AB_CD", "x - y", "op_prefix_if_neg"),
    ("AB_CD", "|x - y|", "plain"),
    ("AB_CD", "x + y - 1", "plain"),
    ("AB_CD", "x * y - 1", "plain"),
    ("AB_CD", "|x - y|", "op_prefix"),
    ("BA_DC", "|x - y|", "op_prefix_rev"),
    ("BA_DC", "|x - y|", "rev"),
    ("BA_DC", "y - x", "rev"),
    ("AB_CD", "concat(x, y)", "op_prefix_if_neg"),
    ("AB_CD", "concat(y, x)", "plain"),
    ("BA_DC", "y - x", "op_suffix_rev_if_neg"),
    ("BA_DC", "x - y", "neg_rev"),
    ("BA_DC", "x - y", "rev_or_op_suffix_if_neg"),
    ("AB_CD", "y - x", "op_prefix_if_neg"),
    ("AB_CD", "y - x", "plain"),
    ("BA_DC", "concat(y, x)", "rev"),
    ("BA_DC", "y - x", "rev_or_op_prefix_rev_if_neg"),
    ("AB_CD", "x mod y", "op_prefix_if_neg"),
    ("AB_CD", "y mod x", "plain"),
    ("AB_CD", "x // y", "plain"),
    ("BA_DC", "x + y + x*y", "rev"),
)


PAIRING_TEXT = {
    "AB_CD": "use both operands as written",
    "AB_DC": "use the left operand as written and reverse the right operand",
    "BA_CD": "reverse the left operand and use the right operand as written",
    "BA_DC": "reverse both operands",
}

BASE_TEXT = {
    "x + y": "add x and y",
    "x + y + 1": "add x and y, then add 1",
    "x + y - 1": "add x and y, then subtract 1",
    "x - y": "subtract y from x",
    "y - x": "subtract x from y",
    "|x - y|": "take the absolute difference",
    "x * y": "multiply x and y",
    "x * y + 1": "multiply x and y, then add 1",
    "x * y - 1": "multiply x and y, then subtract 1",
    "concat(x, y)": "concatenate x then y",
    "concat(y, x)": "concatenate y then x",
    "x mod y": "take x modulo y",
    "y mod x": "take y modulo x",
    "x // y": "take integer division x // y",
    "x + y + x*y": "compute x + y + x*y",
}

OUT_TEXT = {
    "plain": "write the value directly",
    "rev": "reverse the digits of the value",
    "op_prefix_if_neg": "if the value is negative, replace '-' with the operator prefix; otherwise write it directly",
    "rev_or_op_prefix_rev_if_neg": "normally reverse the digits; if negative, reverse the magnitude and prefix the operator",
    "op_prefix": "prefix the value with the operator",
    "op_prefix_rev": "reverse the value and prefix the operator",
    "neg_rev": "reverse the magnitude and keep a leading '-'",
    "rev_or_op_suffix_if_neg": "normally reverse the digits; if negative, append the operator to the magnitude",
    "op_suffix_rev_if_neg": "if negative, reverse the magnitude and append the operator; otherwise write it directly",
    "neg": "write a leading '-' before the magnitude",
    "op_suffix_rev": "reverse the value and append the operator",
    "plain_or_op_prefix_rev_if_neg": "normally write directly; if negative, reverse the magnitude and prefix the operator",
    "abs_rev": "drop the sign and reverse the magnitude",
    "op_suffix": "append the operator to the value",
}


def parse_rule_label(label: str) -> Candidate:
    width = None
    if "|w=" in label:
        label, width_text = label.rsplit("|w=", 1)
        width = int(width_text)

    pairing = None
    rest = label
    for candidate_pairing in PAIRING_NAMES:
        prefix = f"{candidate_pairing}|"
        if label.startswith(prefix):
            pairing = candidate_pairing
            rest = label[len(prefix):]
            break
    if pairing is None:
        raise ValueError(f"Could not parse pairing from rule_label={label!r}")

    for outmode_name in OUTPUT_MODES_BY_LENGTH:
        suffix = f"|{outmode_name}"
        if rest.endswith(suffix):
            base_name = rest[:-len(suffix)]
            return Candidate(pairing, base_name, outmode_name, width)
    raise ValueError(f"Could not parse output mode from rule_label={label!r}")


def candidate_from_parts(pairing: str, base_name: str, outmode_name: str, width: int | None) -> Candidate:
    return Candidate(pairing, base_name, outmode_name, width)


def compute_base_text(base_name: str, x: int, y: int, value: int | None) -> str:
    if value is None:
        return f"{base_name} is undefined for x={x}, y={y}"
    if base_name == "x + y":
        return f"{x} + {y} = {value}"
    if base_name == "x + y + 1":
        return f"{x} + {y} + 1 = {value}"
    if base_name == "x + y - 1":
        return f"{x} + {y} - 1 = {value}"
    if base_name == "x - y":
        return f"{x} - {y} = {value}"
    if base_name == "y - x":
        return f"{y} - {x} = {value}"
    if base_name == "|x - y|":
        return f"|{x} - {y}| = {value}"
    if base_name == "x * y":
        return f"{x} * {y} = {value}"
    if base_name == "x * y + 1":
        return f"{x} * {y} + 1 = {value}"
    if base_name == "x * y - 1":
        return f"{x} * {y} - 1 = {value}"
    if base_name == "concat(x, y)":
        return f"concat({x}, {y}) = {value}"
    if base_name == "concat(y, x)":
        return f"concat({y}, {x}) = {value}"
    if base_name == "x mod y":
        return f"{x} mod {y} = {value}"
    if base_name == "y mod x":
        return f"{y} mod {x} = {value}"
    if base_name == "x // y":
        return f"{x} // {y} = {value}"
    if base_name == "x + y + x*y":
        return f"{x} + {y} + {x}*{y} = {value}"
    return f"{base_name} on x={x}, y={y} = {value}"


def pairing_step(candidate: Candidate, equation: ParsedEquation) -> tuple[str, int, int]:
    x, y = apply_pairing(equation.left_operand_text, equation.right_operand_text, candidate.pairing)
    if candidate.pairing == "AB_CD":
        text = f"AB_CD keeps both operands: x={x}, y={y}"
    elif candidate.pairing == "AB_DC":
        right_rev = reverse_number_text(equation.right_operand_text)
        text = f"AB_DC keeps left and reverses right: x={x}, {equation.right_operand_text}->{right_rev}, y={y}"
    elif candidate.pairing == "BA_CD":
        left_rev = reverse_number_text(equation.left_operand_text)
        text = f"BA_CD reverses left and keeps right: {equation.left_operand_text}->{left_rev}, x={x}, y={y}"
    else:
        left_rev = reverse_number_text(equation.left_operand_text)
        right_rev = reverse_number_text(equation.right_operand_text)
        text = (
            f"BA_DC reverses both operands: {equation.left_operand_text}->{left_rev}, "
            f"{equation.right_operand_text}->{right_rev}, so x={x}, y={y}"
        )
    return text, x, y


def apply_candidate_explanation(candidate: Candidate, equation: ParsedEquation) -> tuple[str, list[str]]:
    steps: list[str] = []
    pair_text, x, y = pairing_step(candidate, equation)
    steps.append(pair_text)
    base_value = candidate.base_rule.apply(x, y)
    steps.append(compute_base_text(candidate.base_name, x, y, base_value))
    if base_value is None:
        return "", steps
    final = candidate.output_mode.apply(base_value, equation.operator, candidate.width)
    out_desc = OUT_TEXT.get(candidate.outmode_name, candidate.outmode_name)
    steps.append(f"{out_desc}: {base_value} -> {final}")
    return final, steps


def first_mismatch(candidate: Candidate, examples: list[ParsedEquation]) -> tuple[ParsedEquation, str | None] | None:
    for example in examples:
        predicted = candidate.predict(example.left_operand_text, example.right_operand_text, example.operator)
        if predicted != example.rhs_text:
            return example, predicted
    return None


def candidate_label(candidate: Candidate) -> str:
    return candidate.label


def scan_attempts(chosen: Candidate, examples: list[ParsedEquation]) -> list[Candidate]:
    width = chosen.width
    attempts: list[Candidate] = []
    seen: set[str] = set()
    for pairing, base_name, outmode_name in SCAN_PRIORITY:
        candidate = candidate_from_parts(pairing, base_name, outmode_name, width)
        label = candidate_label(candidate)
        if label not in seen:
            attempts.append(candidate)
            seen.add(label)
        if label == chosen.label:
            return attempts

    if chosen.label not in seen:
        attempts.append(chosen)
    return attempts


def motif_from_examples(puzzle: NumericEquationPuzzle, query_op: str) -> tuple[tuple[str, str] | None, tuple[str, ...]]:
    by_operator: dict[str, list[ParsedEquation]] = defaultdict(list)
    for example in puzzle.examples:
        by_operator[example.operator].append(example)

    motif_counts: Counter[tuple[str, str]] = Counter()
    supporters: dict[tuple[str, str], set[str]] = defaultdict(set)
    for operator, examples in by_operator.items():
        if operator == query_op:
            continue
        candidates = enumerate_matching(examples, operator)
        signatures = {candidate.transform_signature for candidate in candidates}
        for signature in signatures:
            motif_counts[signature] += len(examples)
            supporters[signature].add(operator)
    if not motif_counts:
        return None, ()
    best = motif_counts.most_common(1)[0][0]
    return best, tuple(sorted(supporters[best]))


def render_detailed_trace(
    *,
    puzzle: NumericEquationPuzzle,
    candidate: Candidate,
    is_deterministic: bool,
    source_tier: str = "",
) -> str:
    query = puzzle.query
    same_op = [example for example in puzzle.examples if example.operator == query.operator]
    motif, motif_supporters = motif_from_examples(puzzle, query.operator)

    lines: list[str] = []
    lines.append(
        "We need to solve a numeric equation transformation by matching the example outputs."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append(
        "The safest method is to solve the query operator locally: same-operator examples decide the arithmetic rule, while other operators provide row-level motif evidence."
    )
    lines.append("")
    lines.append("Step 1: Parse the row.")
    lines.append(f"- Query expression: {query.lhs_text}")
    lines.append(f"- Query operator: '{query.operator}'")
    if same_op:
        lines.append("- Same-operator examples:")
        for example in same_op:
            lines.append(f"  - {example.lhs_text} = {example.rhs_text}")
    else:
        lines.append("- No example uses the query operator, so the same-operator scan cannot lock a base rule directly.")
    if motif is not None:
        supporters = ", ".join(f"'{operator}'" for operator in motif_supporters) or "the other operators"
        lines.append(f"- Row motif evidence from {supporters}: {motif[0]} pairing with {motif[1]} output mode.")
    else:
        lines.append("- No strong row motif is available from the other operators.")

    lines.append("")
    lines.append("Step 2: Scan candidate rules in priority order.")
    if same_op:
        attempts = scan_attempts(candidate, same_op)
        chosen_seen = False
        for attempt_index, attempt in enumerate(attempts, start=1):
            mismatch = first_mismatch(attempt, same_op)
            is_chosen = attempt.label == candidate.label
            lines.append(f"- Try candidate {attempt_index}: {attempt.label}.")
            if mismatch is not None:
                example, predicted = mismatch
                final, steps = apply_candidate_explanation(attempt, example)
                for step in steps:
                    lines.append(f"  - {step}")
                lines.append(
                    f"  - This gives {predicted}, but {example.lhs_text} has RHS {example.rhs_text}. Reject this candidate."
                )
                continue
            lines.append("  - It matches every same-operator example, so it remains viable.")
            for example in same_op:
                final, steps = apply_candidate_explanation(attempt, example)
                lines.append(f"  - Verify {example.lhs_text} = {example.rhs_text}:")
                for step in steps:
                    lines.append(f"    - {step}")
                lines.append(f"    - Predicted {final}, matching the RHS.")
            if is_chosen:
                chosen_seen = True
                break
        if not chosen_seen:
            lines.append(f"- After continuing the scan, the viable candidate I lock is {candidate.label}.")
    else:
        lines.append("- Step 2 fails intentionally: there is no same-operator example to scan.")
        if motif is not None:
            lines.append(
                f"- I project the row motif {motif[0]}|...|{motif[1]} onto the query operator and choose the accepted base rule {candidate.base_name}."
            )
        else:
            lines.append(f"- I use the accepted rule {candidate.label} for this row.")

    lines.append("")
    lines.append("Step 3: Lock the rule.")
    if is_deterministic:
        lines.append(
            f"- Locked rule: {candidate.label}. This rule is deterministic for this trace because it is selected by same-operator verification, row motif evidence, or the deterministic priority policy."
        )
    else:
        lines.append(
            f"- Plausible accepted rule: {candidate.label}. The visible examples are ambiguous, so this trace teaches the gold-producing rule with lower training weight."
        )
    lines.append(f"- Pairing: {PAIRING_TEXT.get(candidate.pairing, candidate.pairing)}.")
    lines.append(f"- Base rule: {BASE_TEXT.get(candidate.base_name, candidate.base_name)}.")
    lines.append(f"- Output mode: {OUT_TEXT.get(candidate.outmode_name, candidate.outmode_name)}.")

    lines.append("")
    lines.append("Step 4: Apply the locked rule to the query.")
    final, steps = apply_candidate_explanation(candidate, query)
    for step in steps:
        lines.append(f"- {step}")
    lines.append("")
    lines.append(f"The query result is {final}.")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The final answer is \\boxed{{{final}}}")
    return "\n".join(lines)
