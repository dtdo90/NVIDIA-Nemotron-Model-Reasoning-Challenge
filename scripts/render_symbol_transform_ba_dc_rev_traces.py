#!/usr/bin/env python3
"""Render compact BA_DC|rev symbol-transform CoT traces for audit.

This is intentionally conservative: it only emits rows where the deterministic
solver produces an exact BA_DC|rev candidate with a concrete symbol-digit map.
The output is meant for human audit before we use the template in phase-2 data.
"""

from __future__ import annotations

import argparse
import csv
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

BA_DC_REV_RULE_PRIORITY = (
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


def _box_payload(answer: str) -> str:
    return answer.replace("\\", "__BS__").replace("{", "__LB__").replace("}", "__RB__")


def _digit_map(candidate) -> dict[str, int]:
    return dict(candidate.digit_map or ())


def _digit_to_symbol(sym_to_digit: dict[str, int]) -> dict[int, str]:
    return {digit: symbol for symbol, digit in sym_to_digit.items()}


def _ba_dc_xy(lhs: str, sym_to_digit: dict[str, int]) -> tuple[int, int]:
    a, b, _, c, d = lhs
    return 10 * sym_to_digit[b] + sym_to_digit[a], 10 * sym_to_digit[d] + sym_to_digit[c]


def _render_rev(value: int | None) -> str | None:
    if value is None or value < 0:
        return None
    return str(value)[::-1]


def _encode(digit_text: str, digit_to_sym: dict[int, str]) -> str | None:
    out: list[str] = []
    for ch in digit_text:
        if not ch.isdigit():
            return None
        symbol = digit_to_sym.get(int(ch))
        if symbol is None:
            return None
        out.append(symbol)
    return "".join(out)


def _compute_rule(rule_name: str, x: int, y: int) -> int | None:
    return BASE_RULE_BY_NAME[rule_name].apply(x, y)


def _compute_text(rule_name: str, x: int, y: int, value: int | None) -> str:
    if value is None:
        return f"{rule_name}: undefined"
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
    return f"{rule_name} gives {value}"


def _prediction_for_example(
    lhs: str,
    rule_name: str,
    sym_to_digit: dict[str, int],
) -> tuple[str | None, int | None, str | None, str | None, int, int]:
    digit_to_sym = _digit_to_symbol(sym_to_digit)
    x, y = _ba_dc_xy(lhs, sym_to_digit)
    value = _compute_rule(rule_name, x, y)
    rendered = _render_rev(value)
    encoded = _encode(rendered, digit_to_sym) if rendered is not None else None
    return encoded, value, rendered, encoded, x, y


def _direct_template_prediction(name: str, lhs: str) -> str:
    return "".join(lhs[index] for index in DIRECT_TEMPLATES[name])


def _map_text(sym_to_digit: dict[str, int]) -> str:
    return ",".join(f"{symbol}->{digit}" for symbol, digit in sorted(sym_to_digit.items()))


def _same_operator_scan_lines(same, chosen_rule: str, sym_to_digit: dict[str, int]) -> list[str]:
    lines: list[str] = []
    attempts: list[str] = []
    for rule_name in BA_DC_REV_RULE_PRIORITY:
        attempts.append(rule_name)
        if rule_name == chosen_rule:
            break

    lines.append("S4: Scan BA_DC|rev arithmetic rules in priority order.")
    for idx, rule_name in enumerate(attempts, start=1):
        lines.append(f"- Try rule {idx}: {rule_name}.")
        first_fail = None
        for ex in same:
            pred, value, rendered, _, x, y = _prediction_for_example(
                ex.lhs, rule_name, sym_to_digit
            )
            if pred != ex.rhs:
                first_fail = (ex, pred, value, rendered, x, y)
                break
        if first_fail is None:
            lines.append(f"  All same-operator examples pass; LOCK rule {rule_name}.")
            break
        ex, pred, value, rendered, x, y = first_fail
        lines.append(
            f"  Check {ex.lhs}={ex.rhs}: BA_DC gives x={x},y={y}; "
            f"{_compute_text(rule_name, x, y, value)}; rev={rendered}; enc={pred}; FAIL."
        )
    return lines


def _verify_lines(examples, chosen_rules: dict[str, str], sym_to_digit: dict[str, int]) -> list[str]:
    lines = ["S5: Verify the locked map and operator rules on all examples."]
    for ex in examples:
        rule_name = chosen_rules.get(ex.operator)
        if rule_name is None:
            lines.append(f"- {ex.lhs}={ex.rhs}: no locked rule recorded for operator {ex.operator}; SKIP.")
            continue
        pred, value, rendered, _, x, y = _prediction_for_example(ex.lhs, rule_name, sym_to_digit)
        status = "PASS" if pred == ex.rhs else "FAIL"
        lines.append(
            f"- {ex.lhs}={ex.rhs}: x={x},y={y}; "
            f"{_compute_text(rule_name, x, y, value)}; rev={rendered}; enc={pred}; {status}."
        )
    return lines


def render_trace(prompt: str, answer: str, result: SymbolTransformSolveResult) -> str | None:
    puzzle = parse_symbol_transform_puzzle(prompt)
    candidate = result.chosen_candidate
    if puzzle is None or candidate is None or not candidate.digit_map:
        return None
    if candidate.motif.label != "BA_DC|rev":
        return None

    sym_to_digit = _digit_map(candidate)
    if not sym_to_digit:
        return None
    query = puzzle.query
    same = same_operator_examples(puzzle)
    chosen_rule = candidate.query_rule.name
    chosen_rules = dict(candidate.chosen_rules)

    lines: list[str] = [
        "We need to deduce the hidden symbol transformation rule by matching the example outputs.",
        "I will put my final answer inside \\boxed{}.",
        "S0: Methodology: solve same-operator examples first; test direct templates first; if they fail, use a bijective symbol-digit map with BA_DC|rev; scan arithmetic rules in priority order; verify the locked rule; then solve the query.",
        "S1: Classify this as Symbol Transform with fixed shape ABOCD.",
        f"- Query is {query}; query operator is {puzzle.query_operator}.",
        f"- Same-operator examples: {'; '.join(f'{ex.lhs}={ex.rhs}' for ex in same)}.",
        "S2: Test direct-position templates first.",
    ]

    for template in ("0134", "3401"):
        checks = []
        for ex in same:
            pred = _direct_template_prediction(template, ex.lhs)
            checks.append(f"{ex.lhs}->{pred} vs {ex.rhs}")
        status = "PASS" if all(part.split(" vs ", 1)[0].rsplit("->", 1)[1] == part.split(" vs ", 1)[1] for part in checks) else "FAIL"
        lines.append(f"- Template {template}: {'; '.join(checks)}; {status}.")
        if status == "PASS":
            lines.append("- Direct template would solve this row, so encrypted search is not needed.")
            return "\n".join(lines)

    lines.extend(
        [
            "S3: Direct templates fail, so use encrypted digit-transform search.",
            "- Use motif BA_DC|rev: read ABOCD as x=BA and y=DC, compute, reverse the numeric result, then encode digits back to symbols.",
            f"- Symbol-digit map: {_map_text(sym_to_digit)}.",
        ]
    )
    if candidate.query_completed_count:
        lines.append(
            f"- The query needed {candidate.query_completed_count} extra operand symbol assignment(s) under the same bijective map."
        )

    lines.extend(_same_operator_scan_lines(same, chosen_rule, sym_to_digit))
    lines.extend(_verify_lines(puzzle.examples, chosen_rules, sym_to_digit))

    pred, value, rendered, encoded, x, y = _prediction_for_example(query, chosen_rule, sym_to_digit)
    lines.extend(
        [
            f"S6: Solve query {query} with BA_DC|rev and rule {chosen_rule}.",
            f"- BA_DC gives x={x},y={y}.",
            f"- {_compute_text(chosen_rule, x, y, value)}.",
            f"- rev={rendered}; encode={encoded}.",
            f"- answer={answer}.",
            "I will now return the answer in \\boxed{}",
            f"The final answer is \\boxed{{{_box_payload(answer)}}}",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument(
        "--analysis-csv",
        default="data/symbol_transform_non_direct_deterministic_exact.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/symbol_transform_ba_dc_rev_traces",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-low-confidence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_by_id: dict[str, dict[str, str]] = {}
    with (ROOT / args.train_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            train_by_id[row["id"]] = row

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    with (ROOT / args.analysis_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("is_exact") != "1" or row.get("motif") != "BA_DC|rev":
                continue
            if not args.include_low_confidence and row.get("confidence") == "low":
                continue
            train_row = train_by_id.get(row["id"])
            if train_row is None:
                skipped += 1
                continue
            prompt = train_row["prompt"]
            if classify_equation_vs_symbol(prompt) != "symbol_transform":
                skipped += 1
                continue
            result = solve_symbol_transform(prompt, **SOLVER_KWARGS)
            if result.prediction != row["answer"]:
                skipped += 1
                continue
            trace = render_trace(prompt, row["answer"], result)
            if trace is None:
                skipped += 1
                continue
            (out_dir / f"{row['id']}.md").write_text(trace + "\n", encoding="utf-8")
            rendered += 1
            if args.limit and rendered >= args.limit:
                break

    print(f"rendered={rendered}")
    print(f"skipped={skipped}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
