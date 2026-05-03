from __future__ import annotations

from dataclasses import dataclass

from .numeric_equation import (
    BASE_RULES,
    INPUT_TRANSFORMS,
    OUTPUT_MODES,
    CandidateRule,
    ParsedEquation,
    candidate_widths_for_examples,
    enumerate_matching_candidates,
    parse_numeric_equation_puzzle,
    solve_numeric_equation,
)


@dataclass(frozen=True)
class ScanCombo:
    pairing: str
    base_rule_name: str
    output_mode_name: str

    @property
    def label(self) -> str:
        return f"{self.pairing}|{self.base_rule_name}|{self.output_mode_name}"


@dataclass(frozen=True)
class DigitTransformSolveResult:
    prediction: str | None
    confidence: str
    query_operator: str
    query_example_count: int
    scan_match_rank: int | None
    chosen_combo: ScanCombo | None
    chosen_width: int | None
    used_fallback: bool
    chosen_description: str | None
    notes: tuple[str, ...]


PAIRING_TO_TRANSFORM_NAMES = {
    "AB_CD": ("id", "id"),
    "AB_DC": ("id", "rev"),
    "BA_CD": ("rev", "id"),
    "BA_DC": ("rev", "rev"),
}

TRANSFORM_NAMES_TO_PAIRING = {
    transform_names: pairing for pairing, transform_names in PAIRING_TO_TRANSFORM_NAMES.items()
}

INPUT_TRANSFORM_BY_NAME = {transform.name: transform for transform in INPUT_TRANSFORMS}
BASE_RULE_BY_NAME = {base_rule.name: base_rule for base_rule in BASE_RULES}
OUTPUT_MODE_BY_NAME = {output_mode.name: output_mode for output_mode in OUTPUT_MODES}
PLAIN_NON_SUBTRACTION_RULES = frozenset(
    {
        "x + y",
        "x + y + 1",
        "x + y - 1",
        "x * y",
        "x * y + 1",
        "x * y - 1",
        "concat(x, y)",
        "concat(y, x)",
        "gcd(x, y)",
    }
)


# Empirical scan order derived from exact high-confidence numeric-equation rows in train.csv.
DEFAULT_SCAN_ORDER = (
    ScanCombo("BA_DC", "x * y", "rev"),
    ScanCombo("AB_CD", "concat(x, y)", "plain"),
    ScanCombo("BA_DC", "x + y", "rev"),
    ScanCombo("BA_DC", "x + y + 1", "rev"),
    ScanCombo("BA_DC", "x + y - 1", "rev"),
    ScanCombo("BA_DC", "x * y - 1", "rev"),
    ScanCombo("BA_DC", "x * y + 1", "rev"),
    ScanCombo("AB_CD", "x * y", "plain"),
    ScanCombo("AB_CD", "concat(y, x)", "plain"),
    ScanCombo("AB_CD", "x * y + 1", "plain"),
    ScanCombo("BA_DC", "x - y", "rev"),
    ScanCombo("BA_DC", "y - x", "op_suffix_rev_if_neg"),
    ScanCombo("AB_CD", "x - y", "plain"),
    ScanCombo("BA_DC", "concat(x, y)", "rev"),
    ScanCombo("AB_CD", "x * y - 1", "plain"),
    ScanCombo("AB_CD", "x + y", "plain"),
    ScanCombo("AB_CD", "x + y + 1", "plain"),
    ScanCombo("AB_CD", "x + y - 1", "plain"),
    ScanCombo("AB_CD", "|x - y|", "plain"),
    ScanCombo("AB_CD", "y - x", "plain"),
    ScanCombo("BA_DC", "y - x", "op_prefix_rev"),
    ScanCombo("BA_DC", "x - y", "neg_rev"),
    ScanCombo("BA_DC", "|x - y|", "rev"),
    ScanCombo("AB_CD", "|x - y|", "op_prefix"),
    ScanCombo("AB_CD", "x - y", "op_prefix"),
)


def pairing_from_candidate(candidate: CandidateRule) -> str | None:
    transform_names = (candidate.left_transform.name, candidate.right_transform.name)
    return TRANSFORM_NAMES_TO_PAIRING.get(transform_names)


def combo_from_candidate(candidate: CandidateRule) -> ScanCombo | None:
    pairing = pairing_from_candidate(candidate)
    if pairing is None:
        return None
    return ScanCombo(pairing, candidate.base_rule.name, candidate.output_mode.name)


def build_candidate(combo: ScanCombo, width: int | None) -> CandidateRule:
    left_transform_name, right_transform_name = PAIRING_TO_TRANSFORM_NAMES[combo.pairing]
    return CandidateRule(
        left_transform=INPUT_TRANSFORM_BY_NAME[left_transform_name],
        right_transform=INPUT_TRANSFORM_BY_NAME[right_transform_name],
        base_rule=BASE_RULE_BY_NAME[combo.base_rule_name],
        output_mode=OUTPUT_MODE_BY_NAME[combo.output_mode_name],
        width=width,
    )


def matches_example(candidate: CandidateRule, example: ParsedEquation) -> bool:
    return (
        candidate.predict(
            example.left_operand_text,
            example.operator,
            example.right_operand_text,
        )
        == example.rhs_text
    )


def scan_same_operator_examples(
    same_operator_examples: list[ParsedEquation],
    query: ParsedEquation,
    scan_order: tuple[ScanCombo, ...] = DEFAULT_SCAN_ORDER,
) -> tuple[CandidateRule | None, ScanCombo | None, int | None]:
    if not same_operator_examples:
        return None, None, None

    widths = candidate_widths_for_examples(same_operator_examples, query.operator)
    rank = 0

    for combo in scan_order:
        for width in widths:
            rank += 1
            candidate = build_candidate(combo, width)
            first_example = same_operator_examples[0]
            if not matches_example(candidate, first_example):
                continue

            if len(same_operator_examples) >= 2 and not matches_example(candidate, same_operator_examples[1]):
                continue

            if all(matches_example(candidate, example) for example in same_operator_examples[2:]):
                return candidate, combo, rank

    return None, None, None


def maybe_apply_safe_prefix_if_neg_override(
    prompt: str,
    same_operator_examples: list[ParsedEquation],
    prediction: str | None,
) -> CandidateRule | None:
    if prediction is None or not prediction.startswith("-") or len(same_operator_examples) != 1:
        return None

    fallback_result = solve_numeric_equation(prompt)
    query_operator = same_operator_examples[0].operator
    visible_candidates = [
        solved_operator.candidate
        for solved_operator in fallback_result.solved_operators
        if solved_operator.operator != query_operator
    ]
    if not visible_candidates:
        return None
    if len({candidate.transform_signature for candidate in visible_candidates}) != 1:
        return None
    if next(iter({candidate.transform_signature for candidate in visible_candidates})) != ("id", "id", "plain"):
        return None
    if not {candidate.base_rule.name for candidate in visible_candidates}.issubset(PLAIN_NON_SUBTRACTION_RULES):
        return None

    matching_candidates = enumerate_matching_candidates(same_operator_examples)
    for candidate in matching_candidates:
        if (
            candidate.left_transform.name == "id"
            and candidate.right_transform.name == "id"
            and candidate.output_mode.name == "op_prefix_if_neg"
            and candidate.base_rule.name in {"x - y", "y - x"}
            and candidate.width is None
        ):
            return candidate
    return None


def solve_digit_transform(
    prompt: str,
    *,
    scan_order: tuple[ScanCombo, ...] = DEFAULT_SCAN_ORDER,
    fallback_to_exhaustive: bool = True,
) -> DigitTransformSolveResult:
    puzzle = parse_numeric_equation_puzzle(prompt)
    if puzzle is None:
        return DigitTransformSolveResult(
            prediction=None,
            confidence="none",
            query_operator="",
            query_example_count=0,
            scan_match_rank=None,
            chosen_combo=None,
            chosen_width=None,
            used_fallback=False,
            chosen_description=None,
            notes=("Prompt is not a numeric-equation style puzzle.",),
        )

    same_operator_examples = [example for example in puzzle.examples if example.operator == puzzle.query.operator]
    if not same_operator_examples:
        if fallback_to_exhaustive:
            fallback_result = solve_numeric_equation(prompt)
            fallback_combo = (
                combo_from_candidate(fallback_result.chosen_candidate)
                if fallback_result.chosen_candidate is not None
                else None
            )
            return DigitTransformSolveResult(
                prediction=fallback_result.prediction,
                confidence=fallback_result.confidence,
                query_operator=fallback_result.query_operator,
                query_example_count=fallback_result.query_example_count,
                scan_match_rank=None,
                chosen_combo=fallback_combo,
                chosen_width=fallback_result.chosen_candidate.width if fallback_result.chosen_candidate is not None else None,
                used_fallback=True,
                chosen_description=fallback_result.chosen_description,
                notes=fallback_result.notes + ("Used exhaustive fallback because the query operator is absent.",),
            )
        return DigitTransformSolveResult(
            prediction=None,
            confidence="none",
            query_operator=puzzle.query.operator,
            query_example_count=0,
            scan_match_rank=None,
            chosen_combo=None,
            chosen_width=None,
            used_fallback=False,
            chosen_description=None,
            notes=("The query operator never appears in the examples.",),
        )

    candidate, combo, rank = scan_same_operator_examples(same_operator_examples, puzzle.query, scan_order)
    if candidate is not None and combo is not None:
        if len(same_operator_examples) == 1 and fallback_to_exhaustive:
            fallback_result = solve_numeric_equation(prompt)
            visible_transform_signatures = {
                solved_operator.candidate.transform_signature
                for solved_operator in fallback_result.solved_operators
                if solved_operator.operator != puzzle.query.operator
            }
            if (
                len(visible_transform_signatures) == 1
                and fallback_result.chosen_candidate is not None
                and fallback_result.chosen_candidate.transform_signature in visible_transform_signatures
                and candidate.transform_signature not in visible_transform_signatures
            ):
                fallback_combo = combo_from_candidate(fallback_result.chosen_candidate)
                if fallback_combo is not None:
                    return DigitTransformSolveResult(
                        prediction=fallback_result.prediction,
                        confidence=fallback_result.confidence,
                        query_operator=fallback_result.query_operator,
                        query_example_count=fallback_result.query_example_count,
                        scan_match_rank=None,
                        chosen_combo=fallback_combo,
                        chosen_width=fallback_result.chosen_candidate.width,
                        used_fallback=True,
                        chosen_description=fallback_result.chosen_description,
                        notes=fallback_result.notes + (
                            "Used exhaustive fallback because the visible operators agreed on a different transform motif.",
                        ),
                    )
        prediction = candidate.predict(
            puzzle.query.left_operand_text,
            puzzle.query.operator,
            puzzle.query.right_operand_text,
        )
        prefix_override_candidate = maybe_apply_safe_prefix_if_neg_override(
            prompt,
            same_operator_examples,
            prediction,
        )
        if prefix_override_candidate is not None:
            prefix_override_combo = combo_from_candidate(prefix_override_candidate)
            if prefix_override_combo is not None:
                return DigitTransformSolveResult(
                    prediction=prefix_override_candidate.predict(
                        puzzle.query.left_operand_text,
                        puzzle.query.operator,
                        puzzle.query.right_operand_text,
                    ),
                    confidence="medium",
                    query_operator=puzzle.query.operator,
                    query_example_count=len(same_operator_examples),
                    scan_match_rank=None,
                    chosen_combo=prefix_override_combo,
                    chosen_width=prefix_override_candidate.width,
                    used_fallback=True,
                    chosen_description=prefix_override_candidate.description,
                    notes=(
                        "Used safe prefix-if-negative override for a one-example subtraction-family row with a strong AB_CD|plain motif.",
                    ),
                )
        confidence = "high" if len(same_operator_examples) >= 2 else "medium"
        return DigitTransformSolveResult(
            prediction=prediction,
            confidence=confidence,
            query_operator=puzzle.query.operator,
            query_example_count=len(same_operator_examples),
            scan_match_rank=rank,
            chosen_combo=combo,
            chosen_width=candidate.width,
            used_fallback=False,
            chosen_description=candidate.description,
            notes=("Solved by scan-order search.",),
        )

    if not fallback_to_exhaustive:
        return DigitTransformSolveResult(
            prediction=None,
            confidence="none",
            query_operator=puzzle.query.operator,
            query_example_count=len(same_operator_examples),
            scan_match_rank=None,
            chosen_combo=None,
            chosen_width=None,
            used_fallback=False,
            chosen_description=None,
            notes=("No scan-order combo fit all examples for the query operator.",),
        )

    fallback_result = solve_numeric_equation(prompt)
    fallback_combo = (
        combo_from_candidate(fallback_result.chosen_candidate)
        if fallback_result.chosen_candidate is not None
        else None
    )
    prefix_override_candidate = maybe_apply_safe_prefix_if_neg_override(
        prompt,
        same_operator_examples,
        fallback_result.prediction,
    )
    if prefix_override_candidate is not None:
        prefix_override_combo = combo_from_candidate(prefix_override_candidate)
        if prefix_override_combo is not None:
            return DigitTransformSolveResult(
                prediction=prefix_override_candidate.predict(
                    puzzle.query.left_operand_text,
                    puzzle.query.operator,
                    puzzle.query.right_operand_text,
                ),
                confidence="medium",
                query_operator=puzzle.query.operator,
                query_example_count=len(same_operator_examples),
                scan_match_rank=None,
                chosen_combo=prefix_override_combo,
                chosen_width=prefix_override_candidate.width,
                used_fallback=True,
                chosen_description=prefix_override_candidate.description,
                notes=(
                    "Used safe prefix-if-negative override for a one-example subtraction-family row with a strong AB_CD|plain motif.",
                ),
            )
    return DigitTransformSolveResult(
        prediction=fallback_result.prediction,
        confidence=fallback_result.confidence,
        query_operator=fallback_result.query_operator,
        query_example_count=fallback_result.query_example_count,
        scan_match_rank=None,
        chosen_combo=fallback_combo,
        chosen_width=fallback_result.chosen_candidate.width if fallback_result.chosen_candidate is not None else None,
        used_fallback=True,
        chosen_description=fallback_result.chosen_description,
        notes=fallback_result.notes + ("Used exhaustive fallback after scan-order miss.",),
    )
