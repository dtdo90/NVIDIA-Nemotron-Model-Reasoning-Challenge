#!/usr/bin/env python3
"""Build Phase-2-ready Symbol Transform rows: synthetic + real, with data_type and boxed CoT."""

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
    DIRECT_TEMPLATE_PRIORITY,
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
)
QUERY_COMPLETION_SOLVER_KWARGS = dict(
    SOLVER_KWARGS,
    max_query_unknowns=1,
)

CANONICAL_S0 = (
    "S0: Methodology: solve same-operator examples first; test direct templates first; "
    "if direct templates fail, switch to bijective symbol-digit motif+rule search; reject mismatches; "
    "lock the first consistent rule family; then solve the query."
)
SYMBOL_TRANSFORM_OPENER = (
    "We need to deduce the hidden symbol transformation rule by matching the example outputs."
)
BOXED_INTENT_LINE = "I will put my final answer inside \\boxed{}."
BOXED_RETURN_LINE = "I will now return the answer in \\boxed{}"

PAIRING_DESC = {
    "AB_CD": "keep both two-symbol operands as written",
    "AB_DC": "keep the left operand and reverse the right operand",
    "BA_CD": "reverse the left operand and keep the right operand",
    "BA_DC": "reverse both two-symbol operands",
}

OUTPUT_DESC = {
    "raw": "write the numeric result directly",
    "rev": "reverse the digits of the numeric result",
    "abs": "drop the sign and write the absolute value",
}

CIPHER_SCAN_PRIORITY = (
    ("BA_DC", "rev", "x + y"),
    ("BA_DC", "rev", "x * y"),
    ("BA_DC", "rev", "x + y + 1"),
    ("BA_DC", "rev", "x + y - 1"),
    ("BA_DC", "rev", "x * y + 1"),
    ("BA_DC", "rev", "x * y - 1"),
    ("BA_DC", "rev", "x - y"),
    ("BA_DC", "rev", "y - x"),
    ("BA_DC", "rev", "|x - y|"),
    ("AB_CD", "raw", "x + y"),
    ("AB_CD", "raw", "x * y"),
    ("AB_CD", "raw", "x + y + 1"),
    ("AB_CD", "raw", "x + y - 1"),
    ("AB_CD", "raw", "x * y + 1"),
    ("AB_CD", "raw", "x * y - 1"),
    ("AB_CD", "raw", "x - y"),
    ("AB_CD", "raw", "y - x"),
    ("AB_CD", "raw", "|x - y|"),
    ("AB_CD", "rev", "x + y"),
    ("AB_CD", "rev", "x * y"),
    ("BA_DC", "raw", "x + y"),
    ("BA_DC", "raw", "x * y"),
    ("AB_DC", "raw", "x + y"),
    ("AB_DC", "rev", "x + y"),
    ("BA_CD", "raw", "x + y"),
    ("BA_CD", "rev", "x + y"),
    ("AB_CD", "abs", "x - y"),
    ("BA_DC", "abs", "x - y"),
    ("AB_CD", "abs", "|x - y|"),
    ("BA_DC", "abs", "|x - y|"),
)


def _pair_to_xy(pairing: str, lhs: str, sym_to_digit: dict[str, int]) -> tuple[int, int]:
    a = sym_to_digit[lhs[0]]
    b = sym_to_digit[lhs[1]]
    c = sym_to_digit[lhs[3]]
    d = sym_to_digit[lhs[4]]
    if pairing == "AB_CD":
        return 10 * a + b, 10 * c + d
    if pairing == "AB_DC":
        return 10 * a + b, 10 * d + c
    if pairing == "BA_CD":
        return 10 * b + a, 10 * c + d
    if pairing == "BA_DC":
        return 10 * b + a, 10 * d + c
    raise ValueError(pairing)


def _render_output(value: int | None, output_mode: str) -> str | None:
    if value is None:
        return None
    if output_mode == "abs":
        return str(abs(value))
    if value < 0:
        return None
    text = str(value)
    if output_mode == "raw":
        return text
    if output_mode == "rev":
        return text[::-1]
    return None


def _encode_digits(digit_text: str, digit_to_sym: dict[int, str]) -> str | None:
    out: list[str] = []
    for ch in digit_text:
        if not ch.isdigit():
            return None
        sym = digit_to_sym.get(int(ch))
        if sym is None:
            return None
        out.append(sym)
    return "".join(out)


def _wrap_phase2_inner(cot: str, answer: str) -> str:
    inner = cot.strip()
    payload = _boxed_answer_payload(answer)
    return f"<think>\n{inner}\n</think>\n\n\\boxed{{{payload}}}"


def _boxed_answer_payload(answer: str) -> str:
    return answer.replace("\\", "__BS__").replace("{", "__LB__").replace("}", "__RB__")


def _is_shared_wrapped(cot: str) -> bool:
    lines = [line.strip() for line in cot.strip().splitlines() if line.strip()]
    return (
        len(lines) >= 4
        and lines[0] == SYMBOL_TRANSFORM_OPENER
        and lines[1] == BOXED_INTENT_LINE
        and lines[-2] == BOXED_RETURN_LINE
        and lines[-1].startswith("The final answer is \\boxed{")
    )


def _wrap_symbol_transform_cot(cot: str, answer: str) -> str:
    text = cot.strip()
    if _is_shared_wrapped(text):
        return text
    return "\n".join(
        [
            SYMBOL_TRANSFORM_OPENER,
            BOXED_INTENT_LINE,
            text,
            BOXED_RETURN_LINE,
            f"The final answer is \\boxed{{{answer}}}",
        ]
    ).strip()


def _ensure_methodology_preface(cot: str) -> str:
    text = cot.strip()
    if not text:
        return text
    if _is_shared_wrapped(text):
        lines = text.splitlines()
        inner = "\n".join(lines[2:-2]).strip()
        updated_inner = _ensure_methodology_preface(inner)
        return "\n".join([lines[0], lines[1], updated_inner, lines[-2], lines[-1]])
    lines = text.splitlines()
    if lines and lines[0].startswith("S0: Methodology:"):
        lines[0] = CANONICAL_S0
        return "\n".join(lines)
    return f"{CANONICAL_S0}\n{text}"


def _direct_template_desc(name: str) -> tuple[str, str]:
    if name == "0134":
        return "ABOCD -> ABCD", "0,1,3,4"
    if name == "3401":
        return "ABOCD -> CDAB", "3,4,0,1"
    raise ValueError(name)


def _direct_prediction(name: str, lhs: str) -> str:
    return "".join(lhs[i] for i in DIRECT_TEMPLATES[name])


def _pairing_trace(pairing: str, lhs: str, sym_to_digit: dict[str, int]) -> tuple[str, int, int]:
    a, b, op, c, d = lhs
    x, y = _pair_to_xy(pairing, lhs, sym_to_digit)
    left_raw = f"{sym_to_digit[a]}{sym_to_digit[b]}"
    right_raw = f"{sym_to_digit[c]}{sym_to_digit[d]}"
    if pairing == "AB_CD":
        text = f"{pairing}: {a}{b}->{left_raw}, {c}{d}->{right_raw}; x={x}, y={y}"
    elif pairing == "AB_DC":
        text = (
            f"{pairing}: {a}{b}->{left_raw}, reverse {c}{d} "
            f"from {right_raw} to {right_raw[::-1]}; x={x}, y={y}"
        )
    elif pairing == "BA_CD":
        text = (
            f"{pairing}: reverse {a}{b} from {left_raw} to {left_raw[::-1]}, "
            f"{c}{d}->{right_raw}; x={x}, y={y}"
        )
    else:
        text = (
            f"{pairing}: reverse {a}{b} from {left_raw} to {left_raw[::-1]}, "
            f"reverse {c}{d} from {right_raw} to {right_raw[::-1]}; x={x}, y={y}"
        )
    return text, x, y


def _compute_text(rule_name: str, x: int, y: int, value: int | None) -> str:
    if value is None:
        return f"{rule_name} is undefined for x={x}, y={y}"
    if rule_name == "x + y":
        return f"{x} + {y} = {value}"
    if rule_name == "x + y + 1":
        return f"{x} + {y} + 1 = {value}"
    if rule_name == "x + y - 1":
        return f"{x} + {y} - 1 = {value}"
    if rule_name == "x * y":
        return f"{x} * {y} = {value}"
    if rule_name == "x * y + 1":
        return f"{x} * {y} + 1 = {value}"
    if rule_name == "x * y - 1":
        return f"{x} * {y} - 1 = {value}"
    if rule_name == "x - y":
        return f"{x} - {y} = {value}"
    if rule_name == "y - x":
        return f"{y} - {x} = {value}"
    if rule_name == "|x - y|":
        return f"|{x} - {y}| = {value}"
    if rule_name == "concat(x, y)":
        return f"concat({x}, {y}) = {value}"
    if rule_name == "concat(y, x)":
        return f"concat({y}, {x}) = {value}"
    return f"{rule_name} with x={x}, y={y} gives {value}"


def _render_text(value: int | None, output_mode: str) -> tuple[str | None, str]:
    rendered = _render_output(value, output_mode)
    desc = OUTPUT_DESC.get(output_mode, output_mode)
    if rendered is None:
        return None, f"{desc}: {value} cannot be rendered under this output mode"
    return rendered, f"{desc}: {value} -> {rendered}"


def _cipher_prediction_for_example(
    pairing: str,
    output_mode: str,
    rule_name: str,
    lhs: str,
    sym_to_digit: dict[str, int],
    digit_to_sym: dict[int, str],
) -> tuple[str | None, list[str]]:
    lines: list[str] = []
    pair_text, x, y = _pairing_trace(pairing, lhs, sym_to_digit)
    lines.append(pair_text)
    rule = BASE_RULE_BY_NAME[rule_name]
    value = rule.apply(x, y)
    lines.append(_compute_text(rule_name, x, y, value))
    rendered, render_line = _render_text(value, output_mode)
    lines.append(render_line)
    if rendered is None:
        lines.append("no encoded symbol output is available")
        return None, lines
    encoded = _encode_digits(rendered, digit_to_sym)
    if encoded is None:
        lines.append(
            f"encode digits back through the symbol map: {rendered} cannot be fully encoded"
        )
    else:
        lines.append(f"encode digits back through the symbol map: {rendered} -> {encoded}")
    return encoded, lines


def _first_cipher_mismatch(
    pairing: str,
    output_mode: str,
    rule_name: str,
    examples,
    sym_to_digit: dict[str, int],
) -> tuple[object, str | None, list[str]] | None:
    digit_to_sym = {d: s for s, d in sym_to_digit.items()}
    for example in examples:
        pred, lines = _cipher_prediction_for_example(
            pairing,
            output_mode,
            rule_name,
            example.lhs,
            sym_to_digit,
            digit_to_sym,
        )
        if pred != example.rhs:
            return example, pred, lines
    return None


def _cipher_scan_attempts(pairing: str, output_mode: str, rule_name: str) -> list[tuple[str, str, str]]:
    target = (pairing, output_mode, rule_name)
    attempts: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in CIPHER_SCAN_PRIORITY:
        if candidate in seen:
            continue
        attempts.append(candidate)
        seen.add(candidate)
        if candidate == target:
            return attempts
    if target not in seen:
        attempts.append(target)
    return attempts


def _render_direct_check_lines(
    template_name: str,
    examples,
    *,
    stop_after_first_fail: bool,
) -> list[str]:
    lines: list[str] = []
    for example in examples:
        pred = _direct_prediction(template_name, example.lhs)
        status = "PASS" if pred == example.rhs else "FAIL"
        lines.append(
            f"- {example.lhs} = {example.rhs}: template {template_name} gives {pred}; {status}."
        )
        if stop_after_first_fail and status == "FAIL":
            break
    return lines


def _same_example_summary(same) -> str:
    return "; ".join(f"{ex.lhs} = {ex.rhs}" for ex in same)


def _render_real_direct_cot(puzzle, result: SymbolTransformSolveResult) -> str:
    same = same_operator_examples(puzzle)
    query = puzzle.query
    answer = result.prediction or ""
    method = result.method
    if method.startswith("direct_template_prior_"):
        chosen = method.split("direct_template_prior_", 1)[1]
    elif method.startswith("direct_template_") and not method.startswith("direct_template_prior_"):
        body = method.removeprefix("direct_template_")
        chosen = sorted(body.split(","))[0]
    else:
        return (
            f"S1: Query operator is {query[2]}. Same-operator examples: {len(same)}.\n"
            f"S2: Solver method {method}.\n"
            f"S3: Final answer: {answer}."
        )

    take_desc, take = _direct_template_desc(chosen)
    lines = [
        "S0: Methodology: solve same-operator examples first; test direct position templates in fixed priority order; lock the first template that passes all same-operator checks; then apply it to the query.",
        "S1: Classify the puzzle as Symbol Transform with fixed shape ABOCD.",
        f"S2: Query operator is {query[2]}; ignore other operators first and use these {len(same)} same-operator example(s): {_same_example_summary(same)}.",
        f"S3: Direct-template priority order is {' first, then '.join(DIRECT_TEMPLATE_PRIORITY)}.",
    ]

    step = 4
    for template_name in DIRECT_TEMPLATE_PRIORITY:
        desc, positions = _direct_template_desc(template_name)
        lines.append(
            f"S{step}: Try template {template_name} ({desc}; positions {positions}) and verify it on all same-operator examples."
        )
        lines.extend(
            _render_direct_check_lines(
                template_name,
                same,
                stop_after_first_fail=template_name != chosen,
            )
        )
        if template_name == chosen:
            if method.startswith("direct_template_prior_"):
                lines.append(
                    f"- Template {template_name} is the first priority template that passes; lock it by prior."
                )
            else:
                lines.append(
                    f"- Template {template_name} passes all same-operator checks; lock it."
                )
            break
        lines.append(f"- Template {template_name} fails, so reject it. FAIL.")
        step += 1

    lines.append(f"S{step + 1}: Apply the locked template to query {query}.")
    lines.append(f"- Take positions {take} from {query}; this gives {answer}.")
    lines.append(f"S{step + 2}: Final answer: {answer}.")
    return "\n".join(lines)


def _first_failing_core_rule(
    pairing: str,
    output_mode: str,
    ex_lhs: str,
    ex_rhs: str,
    sym_to_digit: dict[str, int],
    ok_name: str,
) -> tuple[str, int, int, str | None, str | None] | None:
    digit_to_sym = {d: s for s, d in sym_to_digit.items()}
    for fail_name in ("x + y", "x * y", "x - y", "y - x"):
        if fail_name == ok_name:
            continue
        fn = BASE_RULE_BY_NAME[fail_name]
        x, y = _pair_to_xy(pairing, ex_lhs, sym_to_digit)
        digits = _render_output(fn.apply(x, y), output_mode)
        if digits is None:
            continue
        sym = _encode_digits(digits, digit_to_sym)
        if sym is None or sym == ex_rhs:
            continue
        return fail_name, x, y, digits, sym
    return None


def _render_real_cipher_cot(puzzle, result: SymbolTransformSolveResult) -> str:
    cand = result.chosen_candidate
    query = puzzle.query
    answer = result.prediction or ""
    if cand is None or not cand.digit_map:
        return (
            f"S1: Encrypted digit-transform (solver method {result.method}).\n"
            f"S2: Unique prediction agrees with training answer after solver checks.\n"
            f"S3: Final answer: {answer}."
        )
    sym_to_digit = dict(cand.digit_map)
    digit_to_sym = {d: s for s, d in sym_to_digit.items()}
    same = tuple(ex for ex in puzzle.examples if ex.operator == puzzle.query_operator)
    if not same:
        same = puzzle.examples[:1]
    ex0 = same[0]
    ex_lhs, ex_rhs = ex0.lhs, ex0.rhs
    pairing = cand.motif.pairing
    output_mode = cand.motif.output_mode
    ok_name = cand.query_rule.name
    ok_fn = BASE_RULE_BY_NAME[ok_name]

    fail = _first_failing_core_rule(pairing, output_mode, ex_lhs, ex_rhs, sym_to_digit, ok_name)
    mapping_preview = ", ".join(f"{s}->{d}" for s, d in sorted(sym_to_digit.items()))
    q_pred, q_steps = _cipher_prediction_for_example(
        pairing,
        output_mode,
        ok_name,
        query,
        sym_to_digit,
        digit_to_sym,
    )

    lines = [
        "S0: Methodology: solve same-operator examples first; try cheap direct templates; if they fail, switch to bijective symbol-digit mapping and scan motif|rule candidates in priority order; reject each mismatch; lock the first consistent motif+rule; verify it; then solve query by decode -> compute -> render -> encode.",
        "S1: Classify the puzzle as encrypted digit-transform with fixed shape ABOCD.",
        f"- Query is {query}; query operator is {query[2]}.",
        f"- Same-operator examples ({len(same)}): {_same_example_summary(same)}.",
        "- Treat each non-operator symbol as a row-local digit. The map must be bijective: one symbol per digit and one digit per symbol.",
        "S2: First try direct-position templates before doing digit search.",
    ]
    for direct_name in DIRECT_TEMPLATE_PRIORITY:
        desc, positions = _direct_template_desc(direct_name)
        lines.append(f"- Test template {direct_name} ({desc}; positions {positions}).")
        lines.extend(f"  {line}" for line in _render_direct_check_lines(direct_name, same, stop_after_first_fail=True))
    lines.append(
        "S3: Direct templates do not explain the same-operator examples, so switch to encrypted digit-transform search."
    )
    lines.append("- Operand motifs describe how to read the two operands: AB_CD, AB_DC, BA_CD, BA_DC.")
    lines.append("- Output modes describe how to write the arithmetic value: raw, reversed digits, or absolute value.")
    lines.append("- Arithmetic rules are scanned in a fixed priority order under each motif.")
    if result.method.startswith("adaptive_retry_"):
        lines.append("- Adaptive retry widened the symbol-to-digit beam and recovered a unique consistent fit.")
    lines.append(f"S4: Build and use the symbol-digit map for the locked candidate: {mapping_preview}.")
    if cand.query_completed_count:
        lines.append(
            f"- The query has {cand.query_completed_count} operand symbol(s) that were not fixed by the examples; "
            "complete them under the same bijective digit map before computing the query."
        )

    attempts = _cipher_scan_attempts(pairing, output_mode, ok_name)
    lines.append("S5: Scan candidate motif|rule|output triples in priority order.")
    for idx, (try_pairing, try_output, try_rule) in enumerate(attempts, start=1):
        label = f"{try_pairing}|{try_rule}|{try_output}"
        if (try_pairing, try_output, try_rule) == (pairing, output_mode, ok_name):
            lines.append(f"- Try candidate {idx}: {label}. This is the first candidate that survives all checks.")
            break
        mismatch = _first_cipher_mismatch(
            try_pairing,
            try_output,
            try_rule,
            same,
            sym_to_digit,
        )
        lines.append(f"- Try candidate {idx}: {label}.")
        if mismatch is None:
            lines.append("  It does not become the locked candidate after full consistency and agreement checks; continue.")
            continue
        bad_ex, pred, detail_lines = mismatch
        lines.append(f"  Check {bad_ex.lhs} = {bad_ex.rhs}.")
        for detail in detail_lines:
            lines.append(f"  - {detail}.")
        if pred is None:
            lines.append(
                f"  This candidate cannot produce the target symbol string {bad_ex.rhs}; reject candidate {idx}."
            )
        else:
            lines.append(f"  Predicted {pred}, but target is {bad_ex.rhs}; reject candidate {idx}.")

    lines.append(f"S6: Lock motif {pairing}|{output_mode} and arithmetic rule {ok_name}.")
    lines.append(f"- Pairing meaning: {PAIRING_DESC.get(pairing, pairing)}.")
    lines.append(f"- Output meaning: {OUTPUT_DESC.get(output_mode, output_mode)}.")
    lines.append("- Verify the locked motif/rule on every same-operator example:")
    for ex in same:
        v_sym, detail_lines = _cipher_prediction_for_example(
            pairing,
            output_mode,
            ok_name,
            ex.lhs,
            sym_to_digit,
            digit_to_sym,
        )
        status = "PASS" if v_sym == ex.rhs else "FAIL"
        lines.append(f"  Verify {ex.lhs} = {ex.rhs}.")
        for detail in detail_lines:
            lines.append(f"  - {detail}.")
        lines.append(f"  Result is {v_sym}; target is {ex.rhs}; {status}.")
    if result.method == "same_operator_global_consistency_unique":
        lines.append("- The same query output is also unique after row-global motif consistency checks.")
    elif result.method.startswith("encrypted_digit_transform_unique"):
        lines.append("- All fitted encrypted digit-transform candidates agree on this query output.")
    elif result.method.startswith("same_operator_encrypted_digit_transform_unique"):
        lines.append("- Same-operator encrypted digit-transform candidates agree on this query output.")
    lines.append(f"S7: Apply the locked rule to the query {query}.")
    for detail in q_steps:
        lines.append(f"- {detail}.")
    lines.append(f"- Encoded query result is {q_pred}; expected final answer is {answer}.")
    lines.append(f"S8: Final answer: {answer}.")
    return "\n".join(lines)


def render_real_cot(prompt: str, result: SymbolTransformSolveResult) -> str:
    puzzle = parse_symbol_transform_puzzle(prompt)
    if puzzle is None:
        return f"S1: Parse failed.\nS2: Final answer: {result.prediction or ''}."
    if result.method.startswith("direct_template"):
        return _render_real_direct_cot(puzzle, result)
    if (
        "encrypted_digit_transform" in result.method
        or result.method == "same_operator_global_consistency_unique"
    ):
        return _render_real_cipher_cot(puzzle, result)
    return (
        f"S1: Solver method {result.method}.\n"
        f"S2: Final answer: {result.prediction or ''}."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--synthetic-csv",
        default="data/trainable/symbol_transform_synthetic_cot_solver_verified_v2.csv",
    )
    p.add_argument(
        "--real-analysis-csv",
        default="data/symbol_transform_solver_analysis_core_adaptive_safe.csv",
    )
    p.add_argument(
        "--rescue-analysis-csv",
        default="data/symbol_transform_solver_analysis_core_adaptive_query_unknown1.csv",
        help=(
            "Optional exact-analysis CSV for public-gold-verified query-completion rescue rows."
        ),
    )
    p.add_argument("--train-csv", default="data/train.csv")
    p.add_argument(
        "--output-csv",
        default="data/trainable/symbol_transform_phase2_combined.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_path = ROOT / args.train_csv
    synthetic_path = ROOT / args.synthetic_csv
    real_path = ROOT / args.real_analysis_csv
    rescue_path = ROOT / args.rescue_analysis_csv
    out_path = ROOT / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_by_id: dict[str, dict[str, str]] = {}
    with train_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train_by_id[row["id"]] = row

    rows_out: list[dict[str, str]] = []

    with synthetic_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ans = row.get("answer", "").strip()
            cot = _wrap_symbol_transform_cot(
                _ensure_methodology_preface((row.get("generated_cot") or "").strip()),
                ans,
            )
            phase2 = _wrap_phase2_inner(cot, ans)
            rows_out.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "answer": ans,
                    "final_answer_plain": ans,
                    "boxed_answer_payload": _boxed_answer_payload(ans),
                    "generated_cot": cot,
                    "generated_cot_phase2": phase2,
                    "data_type": "synthetic",
                    "label": row.get("label", "Transformation Rules"),
                    "category": row.get("category", "Transformation Rules"),
                    "source": row.get("source", "symbol_transform_synthetic_solver_verified_v2"),
                    "family": row.get("family", ""),
                    "solver_method": row.get("solver_method", ""),
                }
            )

    real_rows_by_id: dict[str, dict[str, str]] = {}
    with real_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("is_exact") != "1":
                continue
            if row.get("method") == "operator_template_prior":
                continue
            real_rows_by_id[row["id"]] = row

    rescue_rows_by_id: dict[str, dict[str, str]] = {}
    if rescue_path.exists():
        with rescue_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("is_exact") != "1":
                    continue
                if row.get("method") == "operator_template_prior":
                    continue
                if row["id"] in real_rows_by_id:
                    continue
                rescue_rows_by_id[row["id"]] = row

    for rid in sorted({*real_rows_by_id, *rescue_rows_by_id}):
        is_rescue = rid in rescue_rows_by_id and rid not in real_rows_by_id
        train_row = train_by_id.get(rid)
        if train_row is None:
            continue
        prompt = train_row["prompt"]
        if classify_equation_vs_symbol(prompt) != "symbol_transform":
            continue
        gold = train_row["answer"].strip()
        res = solve_symbol_transform(
            prompt,
            **(QUERY_COMPLETION_SOLVER_KWARGS if is_rescue else SOLVER_KWARGS),
        )
        if res.prediction != gold:
            continue
        cot = _wrap_symbol_transform_cot(
            _ensure_methodology_preface(render_real_cot(prompt, res)),
            gold,
        )
        rows_out.append(
            {
                "id": rid,
                "prompt": prompt,
                "answer": gold,
                "final_answer_plain": gold,
                "boxed_answer_payload": _boxed_answer_payload(gold),
                "generated_cot": cot,
                "generated_cot_phase2": _wrap_phase2_inner(cot, gold),
                "data_type": "real",
                "label": "Transformation Rules",
                "category": "Transformation Rules",
                "source": (
                    "symbol_transform_train_exact_query_completion_public_rescue"
                    if is_rescue
                    else "symbol_transform_train_exact_core_adaptive_safe"
                ),
                "family": "",
                "solver_method": res.method,
            }
        )

    fieldnames = [
        "id",
        "prompt",
        "answer",
        "final_answer_plain",
        "boxed_answer_payload",
        "generated_cot",
        "generated_cot_phase2",
        "data_type",
        "label",
        "category",
        "source",
        "family",
        "solver_method",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    n_syn = sum(1 for r in rows_out if r["data_type"] == "synthetic")
    n_real = sum(1 for r in rows_out if r["data_type"] == "real")
    print(f"Wrote {len(rows_out)} rows to {out_path}")
    print(f"  synthetic: {n_syn}")
    print(f"  real: {n_real}")


if __name__ == "__main__":
    main()
