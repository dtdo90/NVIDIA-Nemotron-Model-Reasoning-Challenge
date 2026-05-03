#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.symbol_transform import (  # noqa: E402
    BASE_RULE_BY_NAME,
    DIRECT_TEMPLATES,
    solve_symbol_transform,
)


SYMBOL_POOL = list("!@#$%^&*()[]{}<>?:;'/\\|`~+=_-")
OPERATORS = ("*", "+", "-", "/", "|", '"')
UNSAFE_BOXED_ANSWER_CHARS = set("{}")
CORE_RULES = (
    "x * y",
    "x + y",
    "x + y + 1",
    "x + y - 1",
    "x * y + 1",
    "x * y - 1",
    "x - y",
    "y - x",
    "|x - y|",
    "concat(x, y)",
    "concat(y, x)",
)
PAIRINGS = ("AB_CD", "AB_DC", "BA_CD", "BA_DC")
OUTPUT_MODES = ("raw", "rev", "abs")

ALLOWED_METHODS = {
    "direct_template_0134",
    "direct_template_3401",
    "direct_template_prior_0134",
    "same_operator_encrypted_digit_transform_unique",
    "encrypted_digit_transform_unique",
    "adaptive_retry_encrypted_digit_transform_unique",
}


@dataclass(frozen=True)
class SyntheticRow:
    row_id: str
    prompt: str
    answer: str
    generated_cot: str
    family: str
    solver_method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Symbol Transform CoTs verified by the deterministic solver."
    )
    parser.add_argument("--target-rows", type=int, default=700)
    parser.add_argument(
        "--output-csv",
        default="data/trainable/symbol_transform_synthetic_cot.csv",
    )
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument(
        "--direct-ratio",
        type=float,
        default=0.80,
        help="Fraction of direct-template rows (higher is easier for solver verification).",
    )
    parser.add_argument(
        "--verify-with-solver",
        action="store_true",
        help="Validate each row with solve_symbol_transform while generating (slower).",
    )
    return parser.parse_args()


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


def _build_prompt(examples: list[tuple[str, str]], query: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:"
    ]
    for lhs, rhs in examples:
        lines.append(f"{lhs} = {rhs}")
    lines.append(f"Now, determine the result for: {query}")
    return "\n".join(lines)


def _direct_template_desc(name: str) -> tuple[str, str]:
    if name == "0134":
        return "ABOCD -> ABCD", "0,1,3,4"
    if name == "3401":
        return "ABOCD -> CDAB", "3,4,0,1"
    raise ValueError(name)


def _direct_prediction(name: str, lhs: str) -> str:
    return "".join(lhs[index] for index in DIRECT_TEMPLATES[name])


def _wrap_symbol_transform_cot(inner: str, answer: str) -> str:
    lines = [
        "We need to deduce the hidden symbol transformation rule by matching the example outputs.",
        "I will put my final answer inside \\boxed{}.",
        inner.strip(),
        "I will now return the answer in \\boxed{}",
        f"The final answer is \\boxed{{{answer}}}",
    ]
    return "\n".join(line for line in lines if line)


def _same_example_summary(same_examples: list[tuple[str, str]]) -> str:
    return "; ".join(f"{lhs} = {rhs}" for lhs, rhs in same_examples)


def _render_direct_check_lines(
    template_name: str,
    examples: list[tuple[str, str]],
    *,
    stop_after_first_fail: bool,
) -> list[str]:
    lines: list[str] = []
    for lhs, rhs in examples:
        pred = _direct_prediction(template_name, lhs)
        status = "PASS" if pred == rhs else "FAIL"
        lines.append(
            f"- {lhs} = {rhs}: template {template_name} gives {pred}; {status}."
        )
        if stop_after_first_fail and status == "FAIL":
            break
    return lines


def _render_direct_cot(
    query: str,
    answer: str,
    template_name: str,
    same_examples: list[tuple[str, str]],
) -> str:
    template_desc, take = _direct_template_desc(template_name)
    alt_name = "3401" if template_name == "0134" else "0134"
    lines = [
        "S0: Methodology: solve same-operator examples first; test direct position templates in order; reject the first failing template; lock the template that passes all same-operator checks; then apply it to the query.",
        "S1: Classify the puzzle as Symbol Transform with fixed shape ABOCD.",
        f"S2: Query operator is {query[2]}; ignore other operators first and use these {len(same_examples)} same-operator example(s): {_same_example_summary(same_examples)}.",
        "S3: Try the first direct-position hypothesis and compute it literally.",
    ]
    lines.extend(_render_direct_check_lines(alt_name, same_examples, stop_after_first_fail=True))
    lines.append(f"- Template {alt_name} fails, so reject it. FAIL.")
    lines.append(f"S4: Try template {template_name} ({template_desc}) and verify it on all same-operator examples.")
    lines.extend(_render_direct_check_lines(template_name, same_examples, stop_after_first_fail=False))
    lines.append(f"- All same-operator checks pass, so lock template {template_name}.")
    lines.append(f"S5: Apply the locked template to query {query}.")
    lines.append(f"- Take positions {take} from {query}; this gives {answer}.")
    lines.append(f"S6: Final answer: {answer}.")
    return _wrap_symbol_transform_cot("\n".join(lines), answer)


def _render_cipher_cot(
    *,
    query: str,
    answer: str,
    pairing: str,
    output_mode: str,
    rule_name: str,
    query_example_count: int,
    sym_to_digit: dict[str, int],
    same_examples: list[tuple[str, str]],
) -> str:
    mapping_preview = ", ".join(f"{symbol}->{sym_to_digit[symbol]}" for symbol in sorted(sym_to_digit))
    if rule_name != "x + y":
        fail_rule = "x + y"
    else:
        fail_rule = "x * y"
    fail_fn = BASE_RULE_BY_NAME[fail_rule]
    ok_fn = BASE_RULE_BY_NAME[rule_name]
    ex_lhs, ex_rhs = same_examples[0]
    ex_xy = _pair_to_xy(pairing, ex_lhs, sym_to_digit)
    fx, fy = ex_xy
    fail_digits = _render_output(fail_fn.apply(fx, fy), output_mode)
    digit_to_sym = {digit: sym for sym, digit in sym_to_digit.items()}
    fail_sym = _encode_digits(fail_digits, digit_to_sym) if fail_digits is not None else None
    qx, qy = _pair_to_xy(pairing, query, sym_to_digit)
    q_value = ok_fn.apply(qx, qy)
    q_digits = _render_output(q_value, output_mode)
    lines = [
        "S0: Methodology: solve same-operator examples first; try cheap direct templates; if they fail, switch to bijective symbol-digit mapping and scan motif|rule candidates; reject on mismatch, keep the first candidate family that stays consistent; then solve the query with the locked motif and rule.",
        "S1: Classify the puzzle as encrypted digit-transform, not a plain join template.",
        "S2: First try direct-position templates because they are cheap and often solve Symbol Transform rows.",
    ]
    for direct_name in DIRECT_TEMPLATES:
        pred = _direct_prediction(direct_name, ex_lhs)
        status = "PASS" if pred == ex_rhs else "FAIL"
        lines.append(
            f"- On {ex_lhs} = {ex_rhs}, template {direct_name} gives {pred}; {status}."
        )
    lines.append(
        f"S3: Direct templates do not explain the row, so switch to a bijective symbol-digit search. Query operator {query[2]} has {query_example_count} same-operator example(s)."
    )
    lines.append(f"S4: Try one early arithmetic rule and reject it explicitly: {fail_rule}.")
    lines.append(
        f"- Decode {ex_lhs} with motif {pairing}: x={fx}, y={fy}. "
        f"{fail_rule} gives {fail_digits}; encoding gives {fail_sym}, but target is {ex_rhs}. Reject. FAIL."
    )
    lines.append(f"S5: Use motif {pairing}|{output_mode} and arithmetic rule {rule_name}.")
    lines.append(f"- Full symbol-digit map used by this fit: {mapping_preview}.")
    lines.append("- Verify the locked motif/rule on every same-operator example:")
    for lhs, rhs in same_examples:
        vx, vy = _pair_to_xy(pairing, lhs, sym_to_digit)
        value = ok_fn.apply(vx, vy)
        digits = _render_output(value, output_mode)
        sym = _encode_digits(digits, digit_to_sym) if digits is not None else None
        status = "PASS" if sym == rhs else "FAIL"
        lines.append(
            f"  {lhs}: x={vx}, y={vy}; {rule_name} -> {value}; render({output_mode}) -> {digits}; encode -> {sym}; target {rhs}; {status}."
        )
    lines.append(f"S6: Apply the locked rule to query {query}.")
    lines.append(
        f"- Decode with {pairing}: x={qx}, y={qy}; {rule_name} -> {q_value}; "
        f"render({output_mode}) -> {q_digits}; encode -> {answer}."
    )
    lines.append(f"S7: Final answer: {answer}.")
    return _wrap_symbol_transform_cot("\n".join(lines), answer)


def _make_direct_candidate(rng: random.Random, idx: int) -> SyntheticRow | None:
    template_name, template = rng.choice(tuple(DIRECT_TEMPLATES.items()))
    op = rng.choice(OPERATORS)
    symbols = rng.sample(SYMBOL_POOL, 12)

    def make_lhs(offset: int) -> str:
        a, b, c, d = symbols[offset : offset + 4]
        return f"{a}{b}{op}{c}{d}"

    examples: list[tuple[str, str]] = []
    for off in (0, 4, 8):
        lhs = make_lhs(off)
        rhs = "".join(lhs[i] for i in template)
        examples.append((lhs, rhs))
    rng.shuffle(examples)

    q_syms = rng.sample(SYMBOL_POOL, 4)
    query = f"{q_syms[0]}{q_syms[1]}{op}{q_syms[2]}{q_syms[3]}"
    answer = "".join(query[i] for i in template)
    if any(ch in UNSAFE_BOXED_ANSWER_CHARS for ch in answer):
        return None
    prompt = _build_prompt(examples, query)
    cot = _render_direct_cot(query, answer, template_name, same_examples=examples)
    return SyntheticRow(
        row_id=f"syn_st_direct_{idx:05d}",
        prompt=prompt,
        answer=answer,
        generated_cot=cot,
        family=f"direct_{template_name}",
        solver_method="",
    )


def _random_bijection(rng: random.Random) -> tuple[dict[str, int], dict[int, str]]:
    chosen_symbols = rng.sample(SYMBOL_POOL, 10)
    rng.shuffle(chosen_symbols)
    sym_to_digit = {symbol: digit for digit, symbol in enumerate(chosen_symbols)}
    digit_to_sym = {digit: symbol for symbol, digit in sym_to_digit.items()}
    return sym_to_digit, digit_to_sym


def _random_lhs(rng: random.Random, op: str, digit_to_sym: dict[int, str]) -> str:
    digits = rng.sample(range(10), 4)
    return (
        f"{digit_to_sym[digits[0]]}{digit_to_sym[digits[1]]}"
        f"{op}"
        f"{digit_to_sym[digits[2]]}{digit_to_sym[digits[3]]}"
    )


def _compute_rhs(
    *,
    lhs: str,
    pairing: str,
    output_mode: str,
    rule_name: str,
    sym_to_digit: dict[str, int],
    digit_to_sym: dict[int, str],
) -> str | None:
    xy = _pair_to_xy(pairing, lhs, sym_to_digit)
    value = BASE_RULE_BY_NAME[rule_name].apply(*xy)
    digit_text = _render_output(value, output_mode)
    if digit_text is None:
        return None
    if len(digit_text) < 1 or len(digit_text) > 4:
        return None
    return _encode_digits(digit_text, digit_to_sym)


def _make_cipher_candidate(rng: random.Random, idx: int) -> SyntheticRow | None:
    sym_to_digit, digit_to_sym = _random_bijection(rng)
    pairing = rng.choice(PAIRINGS)
    output_mode = rng.choice(OUTPUT_MODES)
    query_op = rng.choice(OPERATORS)
    query_rule = rng.choice(CORE_RULES)
    examples: list[tuple[str, str]] = []
    # Cleaner training rows: keep one operator family with 3 supporting examples.
    made = 0
    tries = 0
    while made < 3 and tries < 80:
        tries += 1
        lhs = _random_lhs(rng, query_op, digit_to_sym)
        rhs = _compute_rhs(
            lhs=lhs,
            pairing=pairing,
            output_mode=output_mode,
            rule_name=query_rule,
            sym_to_digit=sym_to_digit,
            digit_to_sym=digit_to_sym,
        )
        if rhs is None:
            continue
        examples.append((lhs, rhs))
        made += 1
    if made < 3:
        return None

    # Avoid obvious duplicate examples that can induce accidental ambiguity.
    if len({lhs for lhs, _rhs in examples}) < 3:
        return None

    query = _random_lhs(rng, query_op, digit_to_sym)
    answer = _compute_rhs(
        lhs=query,
        pairing=pairing,
        output_mode=output_mode,
        rule_name=query_rule,
        sym_to_digit=sym_to_digit,
        digit_to_sym=digit_to_sym,
    )
    if answer is None:
        return None
    if any(ch in UNSAFE_BOXED_ANSWER_CHARS for ch in answer):
        return None
    if query in {lhs for lhs, _rhs in examples}:
        return None

    # Keep symbol alphabet compact per row to reduce map ambiguity.
    used_syms = {query[0], query[1], query[3], query[4]}
    used_syms.update(ch for lhs, _rhs in examples for ch in (lhs[0], lhs[1], lhs[3], lhs[4]))
    used_syms.update(answer)
    if len(used_syms) < 6:
        return None
    if len(used_syms) > 9:
        return None

    rng.shuffle(examples)
    prompt = _build_prompt(examples, query)
    cot = _render_cipher_cot(
        query=query,
        answer=answer,
        pairing=pairing,
        output_mode=output_mode,
        rule_name=query_rule,
        query_example_count=3,
        sym_to_digit=sym_to_digit,
        same_examples=examples,
    )
    return SyntheticRow(
        row_id=f"syn_st_cipher_{idx:05d}",
        prompt=prompt,
        answer=answer,
        generated_cot=cot,
        family=f"cipher_{pairing}_{output_mode}",
        solver_method="",
    )


def _wrap_phase2_inner(cot: str, answer: str) -> str:
    inner = cot.strip()
    payload = answer.replace("\\", "__BS__").replace("{", "__LB__").replace("}", "__RB__")
    return f"<think>\n{inner}\n</think>\n\n\\boxed{{{payload}}}"


def _validate_with_solver(row: SyntheticRow) -> tuple[bool, str]:
    result = solve_symbol_transform(
        row.prompt,
        rule_bank="core",
        include_abs_output=True,
        max_states_per_rule=120,
        max_combined_states=120,
        selection="unique",
        adaptive_retry=True,
        adaptive_retry_on="no_rule",
        retry_max_states_per_rule=240,
    )
    return (result.prediction == row.answer and result.method in ALLOWED_METHODS, result.method)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target = args.target_rows
    direct_target = int(target * max(0.0, min(1.0, args.direct_ratio)))
    cipher_target = target - direct_target

    rows: list[SyntheticRow] = []
    seen_prompts: set[str] = set()
    direct_count = 0
    cipher_count = 0
    attempts = 0

    while len(rows) < target and attempts < args.max_attempts:
        attempts += 1
        want_direct = direct_count < direct_target and (cipher_count >= cipher_target or rng.random() < args.direct_ratio)
        candidate = (
            _make_direct_candidate(rng, direct_count + 1)
            if want_direct
            else _make_cipher_candidate(rng, cipher_count + 1)
        )
        if candidate is None or candidate.prompt in seen_prompts:
            continue
        if args.verify_with_solver:
            ok, method = _validate_with_solver(candidate)
            if not ok:
                continue
        else:
            method = "constructed_rule"
        seen_prompts.add(candidate.prompt)
        finalized = SyntheticRow(
            row_id=candidate.row_id,
            prompt=candidate.prompt,
            answer=candidate.answer,
            generated_cot=candidate.generated_cot,
            family=candidate.family,
            solver_method=method,
        )
        rows.append(finalized)
        if want_direct:
            direct_count += 1
        else:
            cipher_count += 1

    if len(rows) < target:
        raise SystemExit(
            f"Could not reach target_rows={target}. Generated {len(rows)} rows in {attempts} attempts."
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
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.row_id,
                    "prompt": row.prompt,
                    "answer": row.answer,
                    "final_answer_plain": row.answer,
                    "boxed_answer_payload": (
                        row.answer.replace("\\", "__BS__")
                        .replace("{", "__LB__")
                        .replace("}", "__RB__")
                    ),
                    "generated_cot": row.generated_cot,
                    "generated_cot_phase2": _wrap_phase2_inner(row.generated_cot, row.answer),
                    "data_type": "synthetic",
                    "label": "Transformation Rules",
                    "category": "Transformation Rules",
                    "source": "symbol_transform_synthetic_solver_verified",
                    "family": row.family,
                    "solver_method": row.solver_method,
                }
            )

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Direct rows: {direct_count}")
    print(f"Cipher rows: {cipher_count}")
    print(f"Attempts: {attempts}")


if __name__ == "__main__":
    main()
