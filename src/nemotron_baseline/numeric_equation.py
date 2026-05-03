from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable


_EXAMPLE_RE = re.compile(r"([^\n=]+?)\s*=\s*([^\n]+)")
_QUERY_RE = re.compile(r"determine the result for:\s*([^\n]+)", re.IGNORECASE)
_EQUATION_RE = re.compile(r"(-?\d+)\s*([^\d\s])\s*(-?\d+)")


def classify_equation_vs_symbol(prompt: str) -> str:
    text = prompt.strip()
    match = _QUERY_RE.search(text)
    expr = match.group(1).strip() if match else text

    digit_count = len(re.findall(r"\d", expr))
    alpha_count = len(re.findall(r"[A-Za-z]", expr))
    symbol_count = len(re.findall(r"[^\w\s]", expr))

    if re.search(r"\d+\s*[^0-9\s]\s*\d+", expr):
        return "numeric_equation"

    total_chars = max(1, len(expr))
    if symbol_count / total_chars > 0.5 and digit_count + alpha_count < max(2, symbol_count // 2):
        return "symbol_transform"

    if re.search(r"^\s*\d+[^=\n]*=", text, re.MULTILINE):
        return "numeric_equation"
    if re.search(r"[^\w\s]{2,}", text):
        return "symbol_transform"
    return "unknown"


def reverse_number_text(text: str) -> str:
    sign = ""
    digits = text
    if digits.startswith("-"):
        sign = "-"
        digits = digits[1:]
    return sign + digits[::-1]


def parse_int_from_text(text: str) -> int:
    return int(text)


def format_integer(value: int, width: int | None) -> str:
    if width is None:
        return str(value)
    sign = "-" if value < 0 else ""
    digits = str(abs(value)).zfill(width)
    return sign + digits


def reverse_formatted_integer_text(value: int, width: int | None) -> str:
    formatted = format_integer(value, width)
    if formatted.startswith("-"):
        return "-" + formatted[1:][::-1]
    return formatted[::-1]


@dataclass(frozen=True)
class ParsedEquation:
    lhs_text: str
    rhs_text: str
    left_operand_text: str
    operator: str
    right_operand_text: str

    @property
    def left_operand(self) -> int:
        return parse_int_from_text(self.left_operand_text)

    @property
    def right_operand(self) -> int:
        return parse_int_from_text(self.right_operand_text)


@dataclass(frozen=True)
class NumericEquationPuzzle:
    prompt: str
    examples: tuple[ParsedEquation, ...]
    query: ParsedEquation


@dataclass(frozen=True)
class InputTransform:
    name: str
    apply: Callable[[str], int]
    complexity: int


@dataclass(frozen=True)
class BaseRule:
    name: str
    apply: Callable[[int, int], int]
    complexity: int


@dataclass(frozen=True)
class OutputMode:
    name: str
    apply: Callable[[int, str, int | None], str]
    complexity: int


@dataclass(frozen=True)
class CandidateRule:
    left_transform: InputTransform
    right_transform: InputTransform
    base_rule: BaseRule
    output_mode: OutputMode
    width: int | None

    @property
    def description(self) -> str:
        width_suffix = "" if self.width is None else f", width={self.width}"
        return (
            f"{self.output_mode.name}("
            f"{self.base_rule.name} on {self.left_transform.name},{self.right_transform.name}"
            f"{width_suffix})"
        )

    @property
    def complexity(self) -> int:
        return (
            self.left_transform.complexity
            + self.right_transform.complexity
            + self.base_rule.complexity
            + self.output_mode.complexity
            + (0 if self.width is None else 1)
        )

    @property
    def family_signature(self) -> tuple[str, str, str, int | None]:
        return (
            self.left_transform.name,
            self.right_transform.name,
            self.output_mode.name,
            self.width,
        )

    @property
    def transform_signature(self) -> tuple[str, str, str]:
        return (
            self.left_transform.name,
            self.right_transform.name,
            self.output_mode.name,
        )

    def predict(self, left_operand_text: str, operator: str, right_operand_text: str) -> str:
        left_value = self.left_transform.apply(left_operand_text)
        right_value = self.right_transform.apply(right_operand_text)
        base_value = self.base_rule.apply(left_value, right_value)
        return self.output_mode.apply(base_value, operator, self.width)


@dataclass(frozen=True)
class SolvedOperator:
    operator: str
    candidate: CandidateRule
    example_count: int
    candidate_count: int


@dataclass(frozen=True)
class NumericEquationSolveResult:
    prediction: str | None
    confidence: str
    query_operator: str
    query_example_count: int
    query_candidate_count: int
    query_prediction_variants: tuple[str, ...]
    chosen_candidate: CandidateRule | None
    chosen_description: str | None
    solved_operators: tuple[SolvedOperator, ...]
    candidate_count_by_operator: dict[str, int]
    notes: tuple[str, ...]


def describe_operand_transform(transform: InputTransform, operand_text: str, operand_label: str) -> tuple[str, int]:
    if transform.name == "id":
        return (f"use the {operand_label} operand as written: {operand_text}", parse_int_from_text(operand_text))
    if transform.name == "rev":
        reversed_text = reverse_number_text(operand_text)
        return (
            f"reverse the {operand_label} operand: {operand_text} -> {reversed_text}",
            parse_int_from_text(reversed_text),
        )
    raise ValueError(f"Unsupported input transform: {transform.name}")


def describe_base_rule(base_rule: BaseRule, left_value: int, right_value: int) -> tuple[str, int]:
    result = base_rule.apply(left_value, right_value)
    return (f"apply {base_rule.name}: {base_rule.name.replace('x', str(left_value)).replace('y', str(right_value))} = {result}", result)


def describe_output_mode(
    output_mode: OutputMode,
    base_value: int,
    operator: str,
    width: int | None,
) -> tuple[str, str]:
    final_text = output_mode.apply(base_value, operator, width)
    if output_mode.name == "plain":
        if width is None:
            return (f"write the result directly: {base_value}", final_text)
        padded = format_integer(base_value, width)
        return (f"pad the result to {width} digits: {base_value} -> {padded}", final_text)
    if output_mode.name == "rev":
        reversed_text = reverse_formatted_integer_text(base_value, width)
        return (f"reverse the result text: {format_integer(base_value, width)} -> {reversed_text}", final_text)
    if output_mode.name == "neg":
        return (f"write the negative form of the magnitude: {base_value} -> {final_text}", final_text)
    if output_mode.name == "neg_rev":
        reversed_text = "-" + reverse_formatted_integer_text(abs(base_value), width)
        return (f"reverse the digits, then write the negative form: {base_value} -> {reversed_text}", final_text)
    if output_mode.name == "op_prefix":
        return (f"put the operator in front: {format_integer(base_value, width)} -> {final_text}", final_text)
    if output_mode.name == "op_suffix":
        return (f"put the operator at the end: {format_integer(base_value, width)} -> {final_text}", final_text)
    if output_mode.name == "op_prefix_if_neg":
        if base_value < 0:
            magnitude = str(abs(base_value)).zfill(width or 0)
            return (f"if the result is negative, write the operator then the magnitude: {base_value} -> {operator}{magnitude}", final_text)
        return (f"if the result is nonnegative, write it directly: {base_value} -> {final_text}", final_text)
    if output_mode.name == "op_suffix_rev_if_neg":
        if base_value < 0:
            magnitude = reverse_formatted_integer_text(abs(base_value), width)
            return (
                f"if the result is negative, reverse the magnitude and put the operator at the end: {base_value} -> {magnitude}{operator}",
                final_text,
            )
        return (f"if the result is nonnegative, write it directly: {base_value} -> {final_text}", final_text)
    if output_mode.name == "op_prefix_rev":
        reversed_text = reverse_formatted_integer_text(base_value, width)
        return (f"reverse the digits, then put the operator in front: {reversed_text} -> {final_text}", final_text)
    if output_mode.name == "op_suffix_rev":
        reversed_text = reverse_formatted_integer_text(base_value, width)
        return (f"reverse the digits, then put the operator at the end: {reversed_text} -> {final_text}", final_text)
    raise ValueError(f"Unsupported output mode: {output_mode.name}")


def render_candidate_computation(candidate: CandidateRule, equation: ParsedEquation) -> list[str]:
    left_step, left_value = describe_operand_transform(candidate.left_transform, equation.left_operand_text, "left")
    right_step, right_value = describe_operand_transform(candidate.right_transform, equation.right_operand_text, "right")
    base_step, base_value = describe_base_rule(candidate.base_rule, left_value, right_value)
    output_step, final_text = describe_output_mode(
        candidate.output_mode,
        base_value,
        equation.operator,
        candidate.width,
    )

    steps = []
    if candidate.left_transform.name != "id":
        steps.append(left_step)
    if candidate.right_transform.name != "id":
        steps.append(right_step)
    steps.append(base_step)
    if output_step != f"write the result directly: {base_value}":
        steps.append(output_step)
    else:
        steps.append(output_step)
    steps.append(f"so {equation.lhs_text} becomes {final_text}")
    return steps


def render_numeric_equation_trace(prompt: str, solve_result: NumericEquationSolveResult | None = None) -> str | None:
    puzzle = parse_numeric_equation_puzzle(prompt)
    if puzzle is None:
        return None

    result = solve_result or solve_numeric_equation(prompt)
    candidate = result.chosen_candidate
    if candidate is None or result.prediction is None:
        return None

    same_operator_examples = [example for example in puzzle.examples if example.operator == puzzle.query.operator]
    if not same_operator_examples:
        return None

    lines = []
    lines.append("Each operator can follow its own hidden rule, so I only need the examples with the same operator as the query.")
    lines.append(f"For operator '{puzzle.query.operator}', the matching example(s) are:")
    for example in same_operator_examples:
        lines.append(f"- {example.lhs_text} = {example.rhs_text}")

    lines.append("")
    lines.append("A simple rule that fits these example(s) is:")
    rule_summary = []
    if candidate.left_transform.name == "rev" and candidate.right_transform.name == "rev":
        rule_summary.append("reverse both operands first")
    elif candidate.left_transform.name == "rev":
        rule_summary.append("reverse the left operand first")
    elif candidate.right_transform.name == "rev":
        rule_summary.append("reverse the right operand first")
    rule_summary.append(f"compute {candidate.base_rule.name}")
    if candidate.output_mode.name == "rev":
        rule_summary.append("reverse the result")
    elif candidate.output_mode.name == "neg":
        rule_summary.append("write the negative result")
    elif candidate.output_mode.name == "neg_rev":
        rule_summary.append("reverse the digits and write the negative result")
    elif candidate.output_mode.name == "op_prefix":
        rule_summary.append("put the operator in front of the result")
    elif candidate.output_mode.name == "op_suffix":
        rule_summary.append("put the operator at the end of the result")
    elif candidate.output_mode.name == "op_prefix_if_neg":
        rule_summary.append("if the result is negative, replace the minus sign with the operator")
    elif candidate.output_mode.name == "op_suffix_rev_if_neg":
        rule_summary.append("if the result is negative, reverse the magnitude and put the operator at the end")
    elif candidate.output_mode.name == "op_prefix_rev":
        rule_summary.append("reverse the result and put the operator in front")
    elif candidate.output_mode.name == "op_suffix_rev":
        rule_summary.append("reverse the result and put the operator at the end")
    if candidate.width is not None:
        rule_summary.append(f"keep {candidate.width} digits with leading zeros if needed")
    lines.append(f"- " + "; ".join(rule_summary) + ".")

    for example in same_operator_examples:
        lines.append("")
        lines.append(f"Check {example.lhs_text}:")
        for step in render_candidate_computation(candidate, example):
            lines.append(f"- {step}")

    lines.append("")
    lines.append(f"Apply the same rule to {puzzle.query.lhs_text}:")
    for step in render_candidate_computation(candidate, puzzle.query):
        lines.append(f"- {step}")

    lines.append("")
    lines.append(f"So the result is {result.prediction}.")
    return "\n".join(lines)


INPUT_TRANSFORMS = (
    InputTransform("id", parse_int_from_text, 0),
    InputTransform("rev", lambda text: parse_int_from_text(reverse_number_text(text)), 1),
)


BASE_RULES = (
    BaseRule("x + y", lambda x, y: x + y, 1),
    BaseRule("x + y + 1", lambda x, y: x + y + 1, 1),
    BaseRule("x + y - 1", lambda x, y: x + y - 1, 1),
    BaseRule("x - y", lambda x, y: x - y, 1),
    BaseRule("y - x", lambda x, y: y - x, 1),
    BaseRule("|x - y|", lambda x, y: abs(x - y), 1),
    BaseRule("x * y", lambda x, y: x * y, 2),
    BaseRule("x * y + 1", lambda x, y: x * y + 1, 2),
    BaseRule("x * y - 1", lambda x, y: x * y - 1, 2),
    BaseRule("concat(x, y)", lambda x, y: int(f"{abs(x)}{abs(y)}"), 2),
    BaseRule("concat(y, x)", lambda x, y: int(f"{abs(y)}{abs(x)}"), 2),
    BaseRule("x + y^2", lambda x, y: x + y * y, 3),
    BaseRule("(x + y)^2", lambda x, y: (x + y) ** 2, 3),
    BaseRule("(x - y)^2", lambda x, y: (x - y) ** 2, 3),
    BaseRule(
        "gcd(x, y)",
        lambda x, y: math.gcd(abs(x), abs(y)) if x != 0 and y != 0 else 0,
        3,
    ),
)


ABSENT_OPERATOR_BASE_RULE_PRIORITY = (
    "x + y",
    "x + y + 1",
    "x + y - 1",
    "x * y",
    "x * y + 1",
    "x * y - 1",
    "concat(x, y)",
    "concat(y, x)",
    "x - y",
    "y - x",
    "|x - y|",
    "x + y^2",
    "(x + y)^2",
    "(x - y)^2",
    "gcd(x, y)",
)

SUBTRACTION_FAMILY_RULES = frozenset({"x - y", "y - x", "|x - y|"})
ADDITIVE_TO_MULTIPLICATIVE_PRIORITY = {
    "x + y": ("x * y", "x * y + 1", "x * y - 1"),
    "x + y + 1": ("x * y + 1", "x * y", "x * y - 1"),
    "x + y - 1": ("x * y - 1", "x * y", "x * y + 1"),
}
PREFIX_SUBTRACTION_TRIGGER_RULES = frozenset(
    {
        "x + y",
        "x + y + 1",
        "x + y - 1",
        "x * y",
        "x * y + 1",
        "x * y - 1",
        "concat(x, y)",
        "concat(y, x)",
    }
)


OUTPUT_MODES = (
    OutputMode("plain", lambda value, operator, width: format_integer(value, width), 0),
    OutputMode(
        "rev",
        lambda value, operator, width: reverse_formatted_integer_text(value, width),
        1,
    ),
    OutputMode("neg", lambda value, operator, width: "-" + str(abs(value)).zfill(width or 0), 1),
    OutputMode(
        "neg_rev",
        lambda value, operator, width: "-" + reverse_formatted_integer_text(abs(value), width),
        2,
    ),
    OutputMode("op_prefix", lambda value, operator, width: operator + format_integer(value, width), 1),
    OutputMode("op_suffix", lambda value, operator, width: format_integer(value, width) + operator, 1),
    OutputMode(
        "op_prefix_if_neg",
        lambda value, operator, width: (
            operator + str(abs(value)).zfill(width or 0)
            if value < 0
            else format_integer(value, None)
        ),
        1,
    ),
    OutputMode(
        "op_suffix_rev_if_neg",
        lambda value, operator, width: (
            reverse_formatted_integer_text(abs(value), width) + operator
            if value < 0
            else format_integer(value, None)
        ),
        2,
    ),
    OutputMode(
        "op_prefix_rev",
        lambda value, operator, width: operator + reverse_formatted_integer_text(value, width),
        2,
    ),
    OutputMode(
        "op_suffix_rev",
        lambda value, operator, width: reverse_formatted_integer_text(value, width) + operator,
        2,
    ),
)

INPUT_TRANSFORM_BY_NAME = {transform.name: transform for transform in INPUT_TRANSFORMS}
OUTPUT_MODE_BY_NAME = {output_mode.name: output_mode for output_mode in OUTPUT_MODES}
BASE_RULE_BY_NAME = {base_rule.name: base_rule for base_rule in BASE_RULES}


def parse_numeric_equation_puzzle(prompt: str) -> NumericEquationPuzzle | None:
    example_lines = _EXAMPLE_RE.findall(prompt)
    if not example_lines:
        return None

    parsed_examples: list[ParsedEquation] = []
    for lhs_text, rhs_text in example_lines:
        lhs = lhs_text.strip()
        match = _EQUATION_RE.fullmatch(lhs)
        if match is None:
            return None
        parsed_examples.append(
            ParsedEquation(
                lhs_text=lhs,
                rhs_text=rhs_text.strip(),
                left_operand_text=match.group(1),
                operator=match.group(2),
                right_operand_text=match.group(3),
            )
        )

    query_match = _QUERY_RE.search(prompt)
    if query_match is None:
        return None
    query_text = query_match.group(1).strip()
    match = _EQUATION_RE.fullmatch(query_text)
    if match is None:
        return None

    query = ParsedEquation(
        lhs_text=query_text,
        rhs_text="",
        left_operand_text=match.group(1),
        operator=match.group(2),
        right_operand_text=match.group(3),
    )
    return NumericEquationPuzzle(prompt=prompt, examples=tuple(parsed_examples), query=query)


def candidate_widths_for_examples(examples: list[ParsedEquation], operator: str) -> list[int | None]:
    widths: set[int] = set()
    for example in examples:
        rhs = example.rhs_text.strip()
        if rhs.startswith(operator):
            rhs = rhs[len(operator):]
        if rhs.endswith(operator):
            rhs = rhs[: -len(operator)]
        if rhs.startswith("-"):
            rhs = rhs[1:]
        if rhs.isdigit():
            widths.add(len(rhs))
    ordered = [None]
    ordered.extend(sorted(widths))
    return ordered


def enumerate_matching_candidates(examples: list[ParsedEquation]) -> list[CandidateRule]:
    if not examples:
        return []
    operator = examples[0].operator
    widths = candidate_widths_for_examples(examples, operator)
    matches: list[CandidateRule] = []

    for left_transform in INPUT_TRANSFORMS:
        for right_transform in INPUT_TRANSFORMS:
            for base_rule in BASE_RULES:
                for output_mode in OUTPUT_MODES:
                    for width in widths:
                        candidate = CandidateRule(
                            left_transform=left_transform,
                            right_transform=right_transform,
                            base_rule=base_rule,
                            output_mode=output_mode,
                            width=width,
                        )
                        if all(
                            candidate.predict(
                                example.left_operand_text,
                                example.operator,
                                example.right_operand_text,
                            )
                            == example.rhs_text
                            for example in examples
                        ):
                            matches.append(candidate)

    matches.sort(key=lambda item: (item.complexity, item.description))
    return matches


def choose_candidate_for_operator(
    operator: str,
    examples: list[ParsedEquation],
    candidates: list[CandidateRule],
    signature_counts: Counter[tuple[str, str, str, int | None]],
    transform_signature_counts: Counter[tuple[str, str, str]],
    output_mode_counts: Counter[str],
) -> CandidateRule | None:
    if not candidates:
        return None

    def prefer_absolute_difference(candidate: CandidateRule) -> int:
        if candidate.base_rule.name != "|x - y|":
            return 0
        if candidate.output_mode.name not in {"plain", "op_prefix", "op_prefix_rev", "op_prefix_if_neg"}:
            return 0
        if candidate.output_mode.name == "op_prefix_if_neg":
            if any(
                example.rhs_text.startswith("-") or example.rhs_text.startswith(operator)
                for example in examples
            ):
                return 0
        else:
            if any("-" in example.rhs_text for example in examples):
                return 0
        sibling_exists = any(
            other is not candidate
            and other.left_transform.name == candidate.left_transform.name
            and other.right_transform.name == candidate.right_transform.name
            and other.output_mode.name == candidate.output_mode.name
            and other.width == candidate.width
            and other.base_rule.name in {"x - y", "y - x"}
            for other in candidates
        )
        return 1 if sibling_exists else 0

    ranked = sorted(
        candidates,
        key=lambda item: (
            -prefer_absolute_difference(item),
            -signature_counts[item.family_signature],
            -transform_signature_counts[item.transform_signature],
            -output_mode_counts[item.output_mode.name],
            item.complexity,
            item.description,
        ),
    )
    return ranked[0]


def infer_absent_operator_candidate(
    by_operator: dict[str, list[ParsedEquation]],
    candidates_by_operator: dict[str, list[CandidateRule]],
    query_operator: str,
    chosen_by_operator: dict[str, CandidateRule],
) -> tuple[CandidateRule | None, str | None]:
    if not candidates_by_operator:
        return None, None

    visible_chosen_candidates = [
        candidate
        for operator, candidate in chosen_by_operator.items()
        if operator != query_operator
    ]
    if visible_chosen_candidates:
        visible_transform_signatures = {candidate.transform_signature for candidate in visible_chosen_candidates}
        visible_base_rules = {candidate.base_rule.name for candidate in visible_chosen_candidates}
        if len(visible_transform_signatures) == 1 and visible_base_rules.issubset(PREFIX_SUBTRACTION_TRIGGER_RULES):
            visible_signature = next(iter(visible_transform_signatures))
            if visible_signature == ("id", "id", "plain"):
                return (
                    CandidateRule(
                        left_transform=INPUT_TRANSFORM_BY_NAME["id"],
                        right_transform=INPUT_TRANSFORM_BY_NAME["id"],
                        base_rule=BASE_RULE_BY_NAME["|x - y|"],
                        output_mode=OUTPUT_MODE_BY_NAME["op_prefix"],
                        width=None,
                    ),
                    (
                        "Query operator never appears in the examples; "
                        "visible operators all supported AB_CD|...|plain non-subtraction rules, "
                        "so a subtraction-family operator-prefix rule was preferred."
                    ),
                )
            if visible_signature == ("rev", "rev", "rev"):
                return (
                    CandidateRule(
                        left_transform=INPUT_TRANSFORM_BY_NAME["rev"],
                        right_transform=INPUT_TRANSFORM_BY_NAME["rev"],
                        base_rule=BASE_RULE_BY_NAME["|x - y|"],
                        output_mode=OUTPUT_MODE_BY_NAME["op_prefix_rev"],
                        width=None,
                    ),
                    (
                        "Query operator never appears in the examples; "
                        "visible operators all supported BA_DC|...|rev non-subtraction rules, "
                        "so a subtraction-family prefixed-reverse rule was preferred."
                    ),
                )

    transform_support: Counter[tuple[str, str, str]] = Counter()
    for operator, candidates in candidates_by_operator.items():
        if not candidates:
            continue
        example_count = len(by_operator.get(operator, ()))
        unique_signatures = {candidate.transform_signature for candidate in candidates}
        for signature in unique_signatures:
            transform_support[signature] += max(1, example_count)

    ranked_motifs = transform_support.most_common(2)
    if not ranked_motifs:
        return None, None
    if len(ranked_motifs) >= 2 and ranked_motifs[0][1] == ranked_motifs[1][1]:
        return None, None

    chosen_transform_signature = ranked_motifs[0][0]
    motif_candidates_by_operator = {
        operator: [
            candidate
            for candidate in candidates
            if candidate.transform_signature == chosen_transform_signature
        ]
        for operator, candidates in candidates_by_operator.items()
    }
    motif_candidates_by_operator = {
        operator: candidates
        for operator, candidates in motif_candidates_by_operator.items()
        if candidates
    }
    if not motif_candidates_by_operator:
        return None, None
    if not any(len(by_operator.get(operator, ())) >= 2 for operator in motif_candidates_by_operator):
        return None, None

    motif_candidates = [
        candidate
        for candidates in motif_candidates_by_operator.values()
        for candidate in candidates
    ]
    width_counts: Counter[int | None] = Counter(candidate.width for candidate in motif_candidates)
    chosen_width = None if None in width_counts else width_counts.most_common(1)[0][0]
    left_transform_name, right_transform_name, output_mode_name = chosen_transform_signature
    used_base_rules = {
        candidate.base_rule.name
        for candidate in motif_candidates
    }

    preferred_base_rule_names: list[str] = []
    visible_examples = [
        example
        for operator, examples in by_operator.items()
        if operator != query_operator
        for example in examples
    ]
    visible_outputs_unsigned = bool(visible_examples) and all(
        not example.rhs_text.startswith("-") for example in visible_examples
    )
    if query_operator == "-" and visible_outputs_unsigned:
        preferred_base_rule_names.extend(["|x - y|", "y - x", "x - y"])

    seen_additive_variants = [
        rule_name
        for rule_name in ("x + y + 1", "x + y", "x + y - 1")
        if rule_name in used_base_rules
    ]
    if seen_additive_variants and any(rule_name in SUBTRACTION_FAMILY_RULES for rule_name in used_base_rules):
        for additive_rule_name in seen_additive_variants:
            for candidate_rule_name in ADDITIVE_TO_MULTIPLICATIVE_PRIORITY[additive_rule_name]:
                if candidate_rule_name not in preferred_base_rule_names:
                    preferred_base_rule_names.append(candidate_rule_name)

    priority_order = tuple(
        preferred_base_rule_names
        + [rule_name for rule_name in ABSENT_OPERATOR_BASE_RULE_PRIORITY if rule_name not in preferred_base_rule_names]
    )

    for base_rule_name in priority_order:
        if base_rule_name in used_base_rules:
            continue
        base_rule = BASE_RULE_BY_NAME[base_rule_name]
        candidate = CandidateRule(
            left_transform=INPUT_TRANSFORM_BY_NAME[left_transform_name],
            right_transform=INPUT_TRANSFORM_BY_NAME[right_transform_name],
            base_rule=base_rule,
            output_mode=OUTPUT_MODE_BY_NAME[output_mode_name],
            width=chosen_width,
        )
        note = (
            "Query operator never appears in the examples; "
            f"row-level transform motif {left_transform_name},"
            f"{right_transform_name}->{output_mode_name} "
            f"selected highest-priority unused base rule {base_rule_name}."
        )
        if preferred_base_rule_names and base_rule_name in preferred_base_rule_names:
            if query_operator == "-" and visible_outputs_unsigned:
                note = (
                    "Query operator never appears in the examples; "
                    f"row-level transform motif {left_transform_name},"
                    f"{right_transform_name}->{output_mode_name} "
                    f"and visible unsigned outputs prioritized nonnegative subtraction rule {base_rule_name}."
                )
            else:
                note = (
                    "Query operator never appears in the examples; "
                    f"row-level transform motif {left_transform_name},"
                    f"{right_transform_name}->{output_mode_name} "
                    f"and visible additive-plus-subtraction pattern prioritized multiplicative rule {base_rule_name}."
                )
        return candidate, note

    return None, None


def solve_numeric_equation(prompt: str) -> NumericEquationSolveResult:
    puzzle = parse_numeric_equation_puzzle(prompt)
    if puzzle is None:
        return NumericEquationSolveResult(
            prediction=None,
            confidence="none",
            query_operator="?",
            query_example_count=0,
            query_candidate_count=0,
            query_prediction_variants=(),
            chosen_candidate=None,
            chosen_description=None,
            solved_operators=(),
            candidate_count_by_operator={},
            notes=("Could not parse puzzle into numeric equations.",),
        )

    by_operator: dict[str, list[ParsedEquation]] = defaultdict(list)
    for example in puzzle.examples:
        by_operator[example.operator].append(example)

    candidates_by_operator = {
        operator: enumerate_matching_candidates(operator_examples)
        for operator, operator_examples in by_operator.items()
    }

    signature_counts: Counter[tuple[str, str, str, int | None]] = Counter()
    transform_signature_counts: Counter[tuple[str, str, str]] = Counter()
    output_mode_counts: Counter[str] = Counter()
    for operator, candidates in candidates_by_operator.items():
        if len(by_operator[operator]) < 2 or not candidates:
            continue

        family_signatures = {candidate.family_signature for candidate in candidates}
        if len(family_signatures) == 1:
            signature_counts[next(iter(family_signatures))] += 1

        transform_signatures = {candidate.transform_signature for candidate in candidates}
        if len(transform_signatures) == 1:
            transform_signature_counts[next(iter(transform_signatures))] += 1

        output_modes = {candidate.output_mode.name for candidate in candidates}
        if len(output_modes) == 1:
            output_mode_counts[next(iter(output_modes))] += 1

    solved_operators: list[SolvedOperator] = []
    chosen_by_operator: dict[str, CandidateRule] = {}
    for operator, candidates in candidates_by_operator.items():
        chosen = choose_candidate_for_operator(
            operator,
            by_operator[operator],
            candidates,
            signature_counts,
            transform_signature_counts,
            output_mode_counts,
        )
        if chosen is None:
            continue
        chosen_by_operator[operator] = chosen
        solved_operators.append(
            SolvedOperator(
                operator=operator,
                candidate=chosen,
                example_count=len(by_operator[operator]),
                candidate_count=len(candidates),
            )
        )

    query_operator = puzzle.query.operator
    query_examples = by_operator.get(query_operator, [])
    query_example_count = len(query_examples)
    candidate_count_by_operator = {
        operator: len(candidates)
        for operator, candidates in candidates_by_operator.items()
    }

    notes: list[str] = []
    chosen_candidate = chosen_by_operator.get(query_operator)
    if chosen_candidate is None:
        if query_example_count == 0:
            inferred_candidate, inferred_note = infer_absent_operator_candidate(
                by_operator,
                candidates_by_operator,
                query_operator,
                chosen_by_operator,
            )
            if inferred_candidate is not None:
                prediction = inferred_candidate.predict(
                    puzzle.query.left_operand_text,
                    puzzle.query.operator,
                    puzzle.query.right_operand_text,
                )
                return NumericEquationSolveResult(
                    prediction=prediction,
                    confidence="medium",
                    query_operator=query_operator,
                    query_example_count=query_example_count,
                    query_candidate_count=0,
                    query_prediction_variants=(prediction,),
                    chosen_candidate=inferred_candidate,
                    chosen_description=inferred_candidate.description,
                    solved_operators=tuple(sorted(solved_operators, key=lambda item: item.operator)),
                    candidate_count_by_operator=candidate_count_by_operator,
                    notes=(inferred_note,) if inferred_note else ("Query operator never appears in the examples.",),
                )
            notes.append("Query operator never appears in the examples.")
        else:
            notes.append("No candidate rule matched all examples for the query operator.")
        return NumericEquationSolveResult(
            prediction=None,
            confidence="none" if query_example_count == 0 else "low",
            query_operator=query_operator,
            query_example_count=query_example_count,
            query_candidate_count=candidate_count_by_operator.get(query_operator, 0),
            query_prediction_variants=(),
            chosen_candidate=None,
            chosen_description=None,
            solved_operators=tuple(sorted(solved_operators, key=lambda item: item.operator)),
            candidate_count_by_operator=candidate_count_by_operator,
            notes=tuple(notes),
        )

    query_candidates = candidates_by_operator.get(query_operator, [])
    query_prediction_variants = tuple(
        sorted(
            {
                candidate.predict(
                    puzzle.query.left_operand_text,
                    puzzle.query.operator,
                    puzzle.query.right_operand_text,
                )
                for candidate in query_candidates
            }
        )
    )
    prediction = chosen_candidate.predict(
        puzzle.query.left_operand_text,
        puzzle.query.operator,
        puzzle.query.right_operand_text,
    )

    candidate_count = candidate_count_by_operator.get(query_operator, 0)
    confidence = "low"
    if len(query_prediction_variants) > 1:
        notes.append(
            f"Matching candidates disagree on the query output: {', '.join(query_prediction_variants[:5])}."
        )
        if query_example_count == 1 and signature_counts[chosen_candidate.family_signature] > 0:
            confidence = "high"
            notes.append("Query operator had one example; exact row-level family motif selected among competing candidates.")
        elif query_example_count == 1 and transform_signature_counts[chosen_candidate.transform_signature] > 0:
            confidence = "high"
            notes.append("Query operator had one example; row-level transform motif selected among competing candidates.")
        elif query_example_count == 1 and output_mode_counts[chosen_candidate.output_mode.name] > 0:
            confidence = "medium"
            notes.append("Query operator had one example; row-level output format motif selected among competing candidates.")
    elif query_example_count >= 1:
        confidence = "high"
        if query_example_count == 1 and candidate_count > 1:
            notes.append("Query operator had one example, but all matching candidates agreed on the query output.")
        elif query_example_count >= 2 and candidate_count > 1:
            notes.append("Multiple candidate rules matched the examples, but they all agreed on the query output.")
    elif query_example_count >= 2 and len(query_prediction_variants) == 2:
        confidence = "medium"
    elif query_example_count == 1:
        notes.append("Query operator had one example and multiple matching candidates.")

    return NumericEquationSolveResult(
        prediction=prediction,
        confidence=confidence,
        query_operator=query_operator,
        query_example_count=query_example_count,
        query_candidate_count=candidate_count,
        query_prediction_variants=query_prediction_variants,
        chosen_candidate=chosen_candidate,
        chosen_description=chosen_candidate.description,
        solved_operators=tuple(sorted(solved_operators, key=lambda item: item.operator)),
        candidate_count_by_operator=candidate_count_by_operator,
        notes=tuple(notes),
    )
