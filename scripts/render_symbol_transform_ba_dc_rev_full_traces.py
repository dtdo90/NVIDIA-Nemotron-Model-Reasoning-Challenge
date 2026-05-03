#!/usr/bin/env python3
"""Render no-map-shortcut BA_DC|rev symbol-transform traces.

This renderer is deliberately stricter than the compact audit renderer:
- It does not print the final symbol-digit map before deriving candidates.
- It scans direct templates, then BA_DC|rev rules in priority order.
- It renders survivor grids/transitions from the examples.

Rows where an earlier-priority rule also leaves same-operator survivors are
skipped for now, because those need an additional case-specific rejection step.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.numeric_equation import classify_equation_vs_symbol  # noqa: E402
from nemotron_baseline.symbol_transform import (  # noqa: E402
    BASE_RULE_BY_NAME,
    DIRECT_TEMPLATES,
    SymbolExample,
    SymbolTransformSolveResult,
    parse_symbol_transform_puzzle,
    same_operator_examples,
    solve_symbol_transform,
)

SOLVER_KWARGS = dict(
    rule_bank="core",
    include_abs_output=True,
    max_states_per_rule=120,
    max_combined_states=120,
    allow_absent_query_operator=False,
    min_query_examples_for_global=2,
    selection="unique",
    adaptive_retry=True,
    adaptive_retry_on="no_rule",
    retry_max_states_per_rule=240,
    max_query_unknowns=1,
)

RULE_PRIORITY = (
    "x * y",
    "x + y",
    "x + y + 1",
    "x + y - 1",
    "x * y + 1",
    "x * y - 1",
    "x - y",
    "y - x",
    "|x - y|",
)

MULTIPLICATION_RULES = ("x * y", "x * y + 1", "x * y - 1")
ADDITION_RULES = ("x + y", "x + y + 1", "x + y - 1")
SUBTRACTION_RULES = ("x - y", "y - x", "|x - y|")
LENGTH3_POPULARITY_RULES = (
    "x + y",
    "x * y",
    "x + y + 1",
)


def _box_payload(answer: str) -> str:
    return answer


def _state_key(state: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(state.items()))


def _dedupe(states: list[dict[str, int]]) -> list[dict[str, int]]:
    return [dict(items) for items in sorted({_state_key(state) for state in states})]


def _merge_pairs(
    state: dict[str, int],
    pairs: list[tuple[str, int]],
) -> dict[str, int] | None:
    local = dict(state)
    used = set(local.values())
    for symbol, digit in pairs:
        existing = local.get(symbol)
        if existing is not None:
            if existing != digit:
                return None
            continue
        if digit in used:
            return None
        local[symbol] = digit
        used.add(digit)
    return local


def _lhs_operand_symbols(lhs: str) -> list[str]:
    out: list[str] = []
    for symbol in (lhs[0], lhs[1], lhs[3], lhs[4]):
        if symbol not in out:
            out.append(symbol)
    return out


def _ba_dc_xy(lhs: str, state: dict[str, int]) -> tuple[int, int]:
    return 10 * state[lhs[1]] + state[lhs[0]], 10 * state[lhs[4]] + state[lhs[3]]


def _rule_value(rule_name: str, x: int, y: int) -> int | None:
    return BASE_RULE_BY_NAME[rule_name].apply(x, y)


def _rule_text(rule_name: str, x: int, y: int, value: int | None) -> str:
    if value is None:
        return f"{rule_name}=undef"
    if rule_name == "x * y":
        return f"{x}*{y}={value}"
    if rule_name == "x + y":
        return f"{x}+{y}={value}"
    if rule_name == "x + y + 1":
        return f"{x}+{y}+1={value}"
    if rule_name == "x + y - 1":
        return f"{x}+{y}-1={value}"
    if rule_name == "x * y + 1":
        return f"{x}*{y}+1={value}"
    if rule_name == "x * y - 1":
        return f"{x}*{y}-1={value}"
    if rule_name == "x - y":
        return f"{x}-{y}={value}"
    if rule_name == "y - x":
        return f"{y}-{x}={value}"
    if rule_name == "|x - y|":
        return f"|{x}-{y}|={value}"
    return f"{rule_name}={value}"


def _render_rev(value: int | None) -> str | None:
    if value is None or value < 0:
        return None
    return str(value)[::-1]


def _extend_one(
    state: dict[str, int],
    example: SymbolExample,
    rule_name: str,
    *,
    fixed: dict[str, int] | None = None,
) -> list[dict[str, int]]:
    fixed = fixed or {}
    seeded = _merge_pairs(state, list(fixed.items()))
    if seeded is None:
        return []

    unknowns = [symbol for symbol in _lhs_operand_symbols(example.lhs) if symbol not in seeded]
    available = [digit for digit in range(10) if digit not in set(seeded.values())]
    out: list[dict[str, int]] = []
    for digits in itertools.permutations(available, len(unknowns)):
        local = dict(seeded)
        local.update(zip(unknowns, digits))
        x, y = _ba_dc_xy(example.lhs, local)
        value = _rule_value(rule_name, x, y)
        rendered = _render_rev(value)
        if rendered is None or len(rendered) != len(example.rhs):
            continue
        merged = _merge_pairs(local, list(zip(example.rhs, map(int, rendered))))
        if merged is not None:
            out.append(merged)
    return _dedupe(out)


def _extend_all(
    states: list[dict[str, int]],
    example: SymbolExample,
    rule_name: str,
) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for state in states:
        out.extend(_extend_one(state, example, rule_name))
    return _dedupe(out)


def _run_examples(
    examples: tuple[SymbolExample, ...],
    rule_name: str,
) -> list[dict[str, int]]:
    states: list[dict[str, int]] = [{}]
    for example in examples:
        states = _extend_all(states, example, rule_name)
        if not states:
            return []
    return states


def _rules_for_rhs_lengths(examples: tuple[SymbolExample, ...]) -> tuple[str, str, tuple[str, ...]]:
    rhs_values = [example.rhs for example in examples]
    rhs_text = ", ".join(rhs_values)
    lengths = [len(rhs) for rhs in rhs_values]
    max_length = max(lengths) if lengths else 0
    if max_length >= 4:
        return (
            "length-4 multiplication family",
            f"The same-operator RHS values are {rhs_text}. At least one RHS has length 4, so use the multiplication family for two-digit by two-digit arithmetic.",
            MULTIPLICATION_RULES,
        )
    if max_length == 3:
        return (
            "length-3 popularity order",
            f"The same-operator RHS values are {rhs_text}. The longest RHS length is 3, so try the capped numeric-symbol BA_DC|rev order: x+y, then x*y, then x+y+1.",
            LENGTH3_POPULARITY_RULES,
        )
    if max_length == 2:
        return (
            "length-2 addition/subtraction family",
            f"The same-operator RHS values are {rhs_text}. Length 2 indicates the +/- family.",
            ADDITION_RULES + SUBTRACTION_RULES,
        )
    if max_length == 1:
        return (
            "subtraction family",
            f"The same-operator RHS values are {rhs_text}. Length 1 indicates a short subtraction-style result.",
            SUBTRACTION_RULES,
        )
    return (
        "short-output addition/subtraction family",
        f"The same-operator RHS values are {rhs_text}. These short outputs suggest addition/subtraction rules before multiplication.",
        ADDITION_RULES + SUBTRACTION_RULES,
    )


def _rules_for_example(example: SymbolExample) -> tuple[str, str, tuple[str, ...]]:
    return _rules_for_rhs_lengths((example,))


def _direct_prediction(template_name: str, lhs: str) -> str:
    return "".join(lhs[index] for index in DIRECT_TEMPLATES[template_name])


def _format_map(state: dict[str, int]) -> str:
    return ",".join(f"{symbol}->{digit}" for symbol, digit in sorted(state.items()))


def _format_compact_map(state: dict[str, int]) -> str:
    return ",".join(f"{symbol}={digit}" for symbol, digit in sorted(state.items()))


def _var_map_for_example(example: SymbolExample) -> dict[str, str]:
    symbols: list[str] = []
    for symbol in (example.lhs[0], example.lhs[1], example.lhs[3], example.lhs[4], *example.rhs):
        if symbol not in symbols:
            symbols.append(symbol)
    letters = "abcdefghijklmnopqrstuvwxyz"
    return {symbol: letters[idx] for idx, symbol in enumerate(symbols)}


def _var_assignments(var_map: dict[str, str]) -> str:
    return ", ".join(f"{symbol}={name}" for symbol, name in var_map.items())


def _two_digit_expr(tens_symbol: str, ones_symbol: str, var_map: dict[str, str]) -> str:
    tens = var_map[tens_symbol]
    ones = var_map[ones_symbol]
    if tens == ones:
        return f"11{tens}"
    return f"10{tens}+{ones}"


def _expanded_digit_expr(symbols: list[str], var_map: dict[str, str]) -> str:
    coeffs: dict[str, int] = {}
    width = len(symbols)
    for idx, symbol in enumerate(symbols):
        coeff = 10 ** (width - idx - 1)
        name = var_map[symbol]
        coeffs[name] = coeffs.get(name, 0) + coeff
    parts: list[str] = []
    for name, coeff in coeffs.items():
        if coeff == 1:
            parts.append(name)
        else:
            parts.append(f"{coeff}{name}")
    return "+".join(parts) if parts else "0"


def _rule_equation_expr(rule_name: str, x_expr: str, y_expr: str) -> str:
    if rule_name == "x * y":
        return f"({x_expr})({y_expr})"
    if rule_name == "x + y":
        return f"({x_expr})+({y_expr})"
    if rule_name == "x + y + 1":
        return f"({x_expr})+({y_expr})+1"
    if rule_name == "x + y - 1":
        return f"({x_expr})+({y_expr})-1"
    if rule_name == "x * y + 1":
        return f"({x_expr})({y_expr})+1"
    if rule_name == "x * y - 1":
        return f"({x_expr})({y_expr})-1"
    if rule_name == "x - y":
        return f"({x_expr})-({y_expr})"
    if rule_name == "y - x":
        return f"({y_expr})-({x_expr})"
    if rule_name == "|x - y|":
        return f"|({x_expr})-({y_expr})|"
    return f"{rule_name}({x_expr},{y_expr})"


def _unit_congruence(example: SymbolExample, rule_name: str, var_map: dict[str, str]) -> str:
    a = var_map[example.lhs[0]]
    c = var_map[example.lhs[3]]
    rhs_unit = var_map[example.rhs[0]]
    if rule_name == "x * y":
        left = f"{a}{c}"
    elif rule_name == "x * y + 1":
        left = f"{a}{c}+1"
    elif rule_name == "x * y - 1":
        left = f"{a}{c}-1"
    elif rule_name == "x + y":
        left = f"{a}+{c}"
    elif rule_name == "x + y + 1":
        left = f"{a}+{c}+1"
    elif rule_name == "x + y - 1":
        left = f"{a}+{c}-1"
    elif rule_name == "x - y":
        left = f"{a}-{c}"
    elif rule_name == "y - x":
        left = f"{c}-{a}"
    elif rule_name == "|x - y|":
        return f"unit condition: either {a}-{c}≡{rhs_unit} or {c}-{a}≡{rhs_unit} mod10"
    else:
        left = f"unit({rule_name})"
    return f"unit condition: {left}≡{rhs_unit} mod10"


def _render_equation_setup(example: SymbolExample, rule_name: str) -> list[str]:
    var_map = _var_map_for_example(example)
    x_expr = _two_digit_expr(example.lhs[1], example.lhs[0], var_map)
    y_expr = _two_digit_expr(example.lhs[4], example.lhs[3], var_map)
    raw_symbols = list(reversed(example.rhs))
    raw_expr = _expanded_digit_expr(raw_symbols, var_map)
    rendered_digits = "".join(var_map[symbol] for symbol in example.rhs)
    raw_digits = "".join(var_map[symbol] for symbol in raw_symbols)
    return [
        f"Use {example.lhs}={example.rhs}. Let {_var_assignments(var_map)}.",
        "Under BA_DC|rev:",
        "```text",
        f"{example.lhs[:2]} -> {example.lhs[1]}{example.lhs[0]} = {x_expr}",
        f"{example.lhs[3:]} -> {example.lhs[4]}{example.lhs[3]} = {y_expr}",
        f"encoded RHS {example.rhs} has digit shape {rendered_digits}; before reverse it is {raw_digits} = {raw_expr}",
        f"{_rule_equation_expr(rule_name, x_expr, y_expr)}={raw_expr}",
        _unit_congruence(example, rule_name, var_map),
        "```",
    ]


def _branch_symbols(example: SymbolExample) -> list[str]:
    # Unit/carry behavior mostly depends on the original ones digits A and C
    # under BA_DC, so branch on them first when possible.
    preferred = [example.lhs[0], example.lhs[3], example.lhs[1], example.lhs[4]]
    out: list[str] = []
    for symbol in preferred:
        if symbol not in out:
            out.append(symbol)
        if len(out) == 2:
            break
    return out


def _label_states(states: list[dict[str, int]], prefix: str = "C") -> dict[tuple[tuple[str, int], ...], str]:
    return {_state_key(state): f"{prefix}{idx}" for idx, state in enumerate(states, start=1)}


def _render_first_grid(
    example: SymbolExample,
    rule_name: str,
    states: list[dict[str, int]],
) -> list[str]:
    labels = _label_states(states, "C")
    branch = _branch_symbols(example)
    lines = _render_equation_setup(example, rule_name)
    lines.append(
        f"Now branch on {','.join(branch)} using the unit condition, then scan the remaining digits and enforce the full equation."
    )
    if len(branch) == 1:
        symbol = branch[0]
        keep: list[str] = []
        for digit in range(10):
            cell_states = _extend_one({}, example, rule_name, fixed={symbol: digit})
            cell_labels = [labels[_state_key(state)] for state in cell_states if _state_key(state) in labels]
            keep.append("|".join(cell_labels) if cell_labels else "x")
        lines.extend(
            [
                "```text",
                f"{symbol}=0123456789",
                "keep=" + ",".join(keep),
                "```",
            ]
        )
        return lines

    row_symbol, col_symbol = branch[:2]
    lines.append("```text")
    lines.append(f"cols {col_symbol}=0123456789")
    for row_digit in range(10):
        cells: list[str] = []
        for col_digit in range(10):
            cell_states = _extend_one(
                {},
                example,
                rule_name,
                fixed={row_symbol: row_digit, col_symbol: col_digit},
            )
            cell_labels = [labels[_state_key(state)] for state in cell_states if _state_key(state) in labels]
            cells.append("|".join(cell_labels) if cell_labels else "x")
        lines.append(f"{row_symbol}={row_digit}:" + ",".join(cells))
    lines.append("```")
    return lines


def _render_transition(
    step_name: str,
    input_states: list[dict[str, int]],
    output_states: list[dict[str, int]],
    example: SymbolExample,
    rule_name: str,
) -> list[str]:
    in_labels = _label_states(input_states, "C")
    out_labels = _label_states(output_states, "C")
    lines = [f"{step_name}: filter with {example.lhs}->{example.rhs} using rule {rule_name}."]
    lines.extend(_render_equation_setup(example, rule_name))
    can_detail = len(input_states) <= 12
    if can_detail:
        for state in input_states:
            unknowns = [
                symbol
                for symbol in _lhs_operand_symbols(example.lhs)
                if symbol not in state
            ]
            if len(unknowns) > 2:
                can_detail = False
                break

    if can_detail:
        for state in input_states:
            label = in_labels[_state_key(state)]
            unknowns = [
                symbol
                for symbol in _lhs_operand_symbols(example.lhs)
                if symbol not in state
            ]
            if not unknowns:
                produced = _extend_one(state, example, rule_name)
                labels = [
                    out_labels[_state_key(item)]
                    for item in produced
                    if _state_key(item) in out_labels
                ]
                lines.extend(
                    [
                        "```text",
                        f"{label}: no new lhs symbols; keep={'|'.join(labels) if labels else 'x'}",
                        "```",
                    ]
                )
            elif len(unknowns) == 1:
                symbol = unknowns[0]
                keep: list[str] = []
                for digit in range(10):
                    produced = _extend_one(
                        state,
                        example,
                        rule_name,
                        fixed={symbol: digit},
                    )
                    labels = [
                        out_labels[_state_key(item)]
                        for item in produced
                        if _state_key(item) in out_labels
                    ]
                    keep.append("|".join(labels) if labels else "x")
                lines.extend(
                    [
                        "```text",
                        f"{label}: scan {symbol}",
                        f"{symbol}=0123456789",
                        "keep=" + ",".join(keep),
                        "```",
                    ]
                )
            else:
                row_symbol, col_symbol = unknowns
                lines.append("```text")
                lines.append(f"{label}: scan {row_symbol},{col_symbol}")
                lines.append(f"cols {col_symbol}=0123456789")
                for row_digit in range(10):
                    cells: list[str] = []
                    for col_digit in range(10):
                        produced = _extend_one(
                            state,
                            example,
                            rule_name,
                            fixed={row_symbol: row_digit, col_symbol: col_digit},
                        )
                        labels = [
                            out_labels[_state_key(item)]
                            for item in produced
                            if _state_key(item) in out_labels
                        ]
                        cells.append("|".join(labels) if labels else "x")
                    lines.append(f"{row_symbol}={row_digit}:" + ",".join(cells))
                lines.append("```")
        return lines

    rows: list[str] = []
    for state in input_states:
        produced = _extend_one(state, example, rule_name)
        labels = [out_labels[_state_key(item)] for item in produced if _state_key(item) in out_labels]
        rows.append(f"{in_labels[_state_key(state)]}->{ '|'.join(labels) if labels else 'x'}")
    lines.extend(["```text", ",".join(rows), "```"])
    return lines


def _encode_digits(rendered: str, state: dict[str, int]) -> str | None:
    digit_to_symbol = {digit: symbol for symbol, digit in state.items()}
    out: list[str] = []
    for char in rendered:
        symbol = digit_to_symbol.get(int(char))
        if symbol is None:
            return None
        out.append(symbol)
    return "".join(out)


def _verify_line(example: SymbolExample, rule_name: str, state: dict[str, int]) -> str:
    x, y = _ba_dc_xy(example.lhs, state)
    value = _rule_value(rule_name, x, y)
    rendered = _render_rev(value)
    encoded = _encode_digits(rendered, state) if rendered is not None else None
    status = "PASS" if encoded == example.rhs else "FAIL"
    return (
        f"- {example.lhs}={example.rhs}: x={x},y={y}; "
        f"{_rule_text(rule_name, x, y, value)}; rev={rendered}; enc={encoded}; {status}."
    )


def _infer_rule_for_example(example: SymbolExample, state: dict[str, int]) -> str | None:
    if any(symbol not in state for symbol in _lhs_operand_symbols(example.lhs)):
        return None
    _, _, rules = _rules_for_example(example)
    for rule_name in rules:
        x, y = _ba_dc_xy(example.lhs, state)
        value = _rule_value(rule_name, x, y)
        rendered = _render_rev(value)
        if rendered is None:
            continue
        encoded = _encode_digits(rendered, state)
        if encoded == example.rhs:
            return rule_name
    return None


def _complete_query_state(
    query: str,
    state: dict[str, int],
    final_map: dict[str, int],
) -> tuple[dict[str, int], list[str]]:
    lines: list[str] = []
    local = dict(state)
    for symbol in (query[0], query[1], query[3], query[4]):
        if symbol in local:
            continue
        digit = final_map[symbol]
        unused = sorted(set(range(10)) - set(local.values()))
        lines.append(f"- Query symbol {symbol} is still unknown; unused digits are {unused}; set {symbol}={digit}.")
        merged = _merge_pairs(local, [(symbol, digit)])
        if merged is None:
            raise ValueError(f"Cannot complete query symbol {symbol}")
        local = merged
    return local, lines


def _is_clean_priority(same: tuple[SymbolExample, ...], chosen_rule: str) -> bool:
    _, _, rules = _rules_for_rhs_lengths(same)
    if chosen_rule not in rules:
        return False
    for rule_name in rules:
        if rule_name == chosen_rule:
            return True
        if _run_examples(same, rule_name):
            return False
    return False


def render_trace(prompt: str, answer: str, result: SymbolTransformSolveResult) -> tuple[str | None, str]:
    puzzle = parse_symbol_transform_puzzle(prompt)
    candidate = result.chosen_candidate
    if puzzle is None or candidate is None or not candidate.digit_map:
        return None, "missing_candidate"
    if candidate.motif.label != "BA_DC|rev":
        return None, "not_ba_dc_rev"

    same = same_operator_examples(puzzle)
    if not same:
        return None, "no_same_operator_examples"

    chosen_rule = candidate.query_rule.name
    if not _is_clean_priority(same, chosen_rule):
        return None, "earlier_rule_has_survivors"

    chosen_rules = dict(candidate.chosen_rules)
    final_map = dict(candidate.digit_map)
    first_states = _run_examples((same[0],), chosen_rule)
    if not first_states:
        return None, "no_first_states"

    lines: list[str] = [
        "We need to deduce the hidden symbol transformation rule by matching the example outputs.",
        "I will put my final answer inside \\boxed{}.",
        "S0: Methodology: solve same-operator examples first; test direct templates first; if they fail, use BA_DC|rev encrypted digit search; choose the arithmetic family from same-operator RHS length; keep visible survivor grids; use other examples only to complete the map; then solve the query.",
        "S1: Classify this as Symbol Transform with fixed shape ABOCD.",
        f"- Query is {puzzle.query}; query operator is {puzzle.query_operator}.",
        f"- Same-operator examples: {'; '.join(f'{ex.lhs}={ex.rhs}' for ex in same)}.",
        "S2: Test direct-position templates first.",
    ]

    for template in ("0134", "3401"):
        checks: list[str] = []
        ok = True
        for ex in same:
            pred = _direct_prediction(template, ex.lhs)
            checks.append(f"{ex.lhs}->{pred} vs {ex.rhs}")
            ok = ok and pred == ex.rhs
        lines.append(f"- Template {template}: {'; '.join(checks)}; {'PASS' if ok else 'FAIL'}.")
        if ok:
            return None, "direct_template_passes"

    family_name, family_reason, family_rules = _rules_for_rhs_lengths(same)
    lines.append("S3: Direct templates fail. Try the BA_DC|rev arithmetic family from the same-operator RHS length.")
    lines.append(f"- {family_reason}")
    lines.append(f"- Therefore try the {family_name}: {', '.join(family_rules)}.")
    for rule_name in family_rules:
        states = _run_examples(same, rule_name)
        if rule_name != chosen_rule:
            lines.append(f"- Rule {rule_name}: survivors after same-operator scan = {len(states)}. FAIL.")
            continue
        lines.append(f"- Rule {rule_name}: LOCK this rule.")
        break

    lines.append("S4: Derive the symbol map for the locked rule without assuming the final map.")
    lines.extend(_render_first_grid(same[0], chosen_rule, first_states))
    lines.append("First-example survivors:")
    lines.append("```text")
    for label, state in zip(_label_states(first_states, "C").values(), first_states):
        lines.append(f"{label}: {_format_compact_map(state)}")
    lines.append("```")

    states = first_states
    for idx, ex in enumerate(same[1:], start=2):
        next_states = _extend_all(states, ex, chosen_rule)
        lines.extend(_render_transition(f"S4.{idx}", states, next_states, ex, chosen_rule))
        states = next_states
        if not states:
            return None, "chosen_rule_lost_states"

    other_examples = [ex for ex in puzzle.examples if ex.operator != puzzle.query_operator]
    if other_examples:
        lines.append("S5: Use other-operator examples only to complete or verify the same digit map.")
    step = 1
    for ex in other_examples:
        rule_name = chosen_rules.get(ex.operator)
        if rule_name is None:
            _, _, example_rules = _rules_for_example(ex)
            for candidate_rule in example_rules:
                maybe_states = _extend_all(states, ex, candidate_rule)
                if maybe_states:
                    rule_name = candidate_rule
                    chosen_rules[ex.operator] = rule_name
                    break
            if rule_name is None:
                lines.append(
                    f"- {ex.lhs}->{ex.rhs}: no consistent BA_DC|rev rule is needed for map completion."
                )
                continue
        next_states = _extend_all(states, ex, rule_name)
        if next_states:
            lines.extend(_render_transition(f"S5.{step}", states, next_states, ex, rule_name))
            states = next_states
            step += 1
        else:
            # Some examples may be pure verification after the final map is known.
            lines.append(f"- {ex.lhs}->{ex.rhs}: no new transition needed under rule {rule_name}.")

    final_key = _state_key(final_map)
    matching = [state for state in states if all(final_map.get(k) == v for k, v in state.items())]
    base_state = matching[0] if matching else states[0]
    completed_state, completion_lines = _complete_query_state(puzzle.query, base_state, final_map)
    if completion_lines:
        lines.append("S6: Complete any query-only symbols by bijection.")
        lines.extend(completion_lines)
    if _state_key(completed_state) != final_key:
        # Include remaining non-query assignments when the solver has a fuller final map.
        for symbol, digit in sorted(final_map.items()):
            if symbol not in completed_state:
                completed_state[symbol] = digit

    lines.append("Final derived map:")
    lines.append("```text")
    lines.append(_format_map(completed_state))
    lines.append("```")

    lines.append("S7: Verify all examples with the derived map and locked operator rules.")
    for ex in puzzle.examples:
        needed_symbols = set(ex.lhs[0] + ex.lhs[1] + ex.lhs[3] + ex.lhs[4] + ex.rhs)
        if not needed_symbols.issubset(completed_state):
            missing = "".join(sorted(needed_symbols - set(completed_state)))
            lines.append(
                f"- {ex.lhs}={ex.rhs}: not used for the query map; missing symbols {missing}."
            )
            continue
        rule_name = chosen_rules.get(ex.operator)
        if rule_name is None:
            rule_name = _infer_rule_for_example(ex, completed_state)
        if rule_name is None:
            lines.append(f"- {ex.lhs}={ex.rhs}: no BA_DC|rev rule needed for the query map.")
            continue
        lines.append(_verify_line(ex, rule_name, completed_state))

    query_rule = chosen_rule
    x, y = _ba_dc_xy(puzzle.query, completed_state)
    value = _rule_value(query_rule, x, y)
    rendered = _render_rev(value)
    encoded = _encode_digits(rendered, completed_state) if rendered is not None else None
    if encoded != answer:
        return None, "query_encode_answer_mismatch"
    lines.extend(
        [
            f"S8: Solve query {puzzle.query} with BA_DC|rev and rule {query_rule}.",
            f"- BA_DC gives x={x},y={y}.",
            f"- {_rule_text(query_rule, x, y, value)}.",
            f"- rev={rendered}; encode={encoded}.",
            f"- answer={answer}.",
            "I will now return the answer in \\boxed{}",
            f"The final answer is \\boxed{{{_box_payload(answer)}}}",
        ]
    )
    return "\n".join(lines), "rendered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument(
        "--analysis-csv",
        default="data/symbol_transform_non_direct_deterministic_exact.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/symbol_transform_ba_dc_rev_full_traces",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_by_id: dict[str, dict[str, str]] = {}
    with (ROOT / args.train_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            train_by_id[row["id"]] = row

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_trace in out_dir.glob("*.md"):
        old_trace.unlink()
    counts: dict[str, int] = {}
    rendered = 0

    with (ROOT / args.analysis_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("is_exact") != "1" or row.get("motif") != "BA_DC|rev":
                continue
            if row.get("confidence") == "low":
                continue
            train_row = train_by_id.get(row["id"])
            if train_row is None:
                counts["missing_train"] = counts.get("missing_train", 0) + 1
                continue
            prompt = train_row["prompt"]
            if classify_equation_vs_symbol(prompt) != "symbol_transform":
                counts["not_symbol_transform"] = counts.get("not_symbol_transform", 0) + 1
                continue
            result = solve_symbol_transform(prompt, **SOLVER_KWARGS)
            if result.prediction != row["answer"]:
                counts["solver_prediction_mismatch"] = counts.get("solver_prediction_mismatch", 0) + 1
                continue
            trace, status = render_trace(prompt, row["answer"], result)
            counts[status] = counts.get(status, 0) + 1
            if trace is None:
                continue
            (out_dir / f"{row['id']}.md").write_text(trace + "\n", encoding="utf-8")
            rendered += 1
            if args.limit and rendered >= args.limit:
                break

    print(f"rendered={rendered}")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
