#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import string
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.symbol_transform import (  # noqa: E402
    DIRECT_TEMPLATE_PRIORITY,
    DIRECT_TEMPLATES,
    SymbolExample,
    SymbolTransformPuzzle,
    parse_symbol_transform_puzzle,
    same_operator_examples,
)


SYMBOL_ALPHABET = tuple(r"""!@#$%^&*()-_=+[]{}|\:;"'<>,.?/`~""")
SAFE_SYMBOL_ALPHABET = tuple(ch for ch in SYMBOL_ALPHABET if ch not in "\\{}")
LABEL_ALPHABET = tuple(string.ascii_lowercase + string.ascii_uppercase)
SYMBOL_EQUATION_NAME = "symbol-equation transformation rules"
SYMBOL_EQUATION_PREFIX = f"In Alice's Wonderland {SYMBOL_EQUATION_NAME}"

OPENER = "We need to deduce the hidden symbol transformation rule by matching the example outputs."
BOXED_INTENT = "I will put my final answer inside \\boxed{}."
BOXED_RETURN = "I will now return the answer in \\boxed{}."
S0_DIRECT = (
    "S0: Methodology: solve same-operator examples first; test direct templates first; "
    "if they fail, use encrypted digit search with BA_DC|rev or AB_CD|raw; choose the "
    "arithmetic family from same-operator RHS length; keep visible survivor grids; use other "
    "examples only to complete the map; then solve the query."
)
S0_MOTIF = "Parse ABOCD and apply the requested operand motif."
S0_OPERATOR = "Form operands, apply the stated rule, then render the output."
S0_ENCODE_DECODE = "Convert each character with the provided map, then join the results."
S0_ROUTE = "State the requested symbol-equation transformation rule directly."
S0_RHS_LENGTH = "Use same-operator RHS length after direct templates fail to choose candidate arithmetic rules."

RHS_LENGTH_FAMILIES = {
    4: ("multiplication family", ("x*y", "x*y+1", "x*y-1")),
    1: ("subtraction family", ("x-y", "y-x", "|x-y|")),
    2: ("addition or subtraction family", ("x+y", "x+y+1", "x+y-1", "x-y", "y-x", "|x-y|")),
    3: ("addition or multiplication family", ("x+y", "x+y+1", "x+y-1", "x*y", "x*y+1", "x*y-1")),
}


@dataclass(frozen=True)
class Phase1Row:
    id: str
    prompt: str
    answer: str
    generated_cot: str
    label: str
    category: str
    source: str
    source_category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Phase 1 symbol-equation transformation curriculum focused on direct templates, "
            "AB_CD/BA_DC motifs, operator vocabulary, and symbol prediction."
        )
    )
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument(
        "--analysis-csv",
        default="data/symbol_transform_solver_analysis_current.csv",
        help="Solver analysis CSV containing the current direct_template_* solved rows.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1_symbol_transform_direct_curriculum.csv",
    )
    parser.add_argument(
        "--trace-dir",
        default="docs/symbol_transform_phase1_direct_template_traces/authentic",
        help="Plain-text authentic trace export for manual inspection.",
    )
    parser.add_argument(
        "--audit-json",
        default="data/trainable/phase1_components/phase1_symbol_transform_direct_curriculum.audit.json",
    )
    parser.add_argument(
        "--audit-md",
        default="docs/symbol_transform_phase1_direct_template_traces/AUDIT.md",
    )
    parser.add_argument(
        "--validation-csv",
        default="data/symbol_transform_direct_template_validation.csv",
        help="Held-out direct-template validation prompts; not included in Phase 1 training.",
    )
    parser.add_argument(
        "--safe-validation-csv",
        default="data/symbol_transform_direct_template_validation_safe.csv",
        help=(
            "Held-out direct-template validation prompts whose answers avoid backslash/braces; "
            "useful for clean extractor checks."
        ),
    )
    parser.add_argument("--validation-rows", type=int, default=240)
    parser.add_argument("--safe-validation-rows", type=int, default=240)
    parser.add_argument("--real-synthetic-per-row", type=int, default=3)
    parser.add_argument(
        "--direct-template-total",
        type=int,
        default=600,
        help=(
            "Target number of direct-template rows, including authentic rows, real-remaps, "
            "fresh synthetic rows, and contrast rows."
        ),
    )
    parser.add_argument(
        "--fresh-direct",
        type=int,
        default=240,
        help="Preferred number of fresh direct-template rows before filling the remainder as contrast rows.",
    )
    parser.add_argument("--contrast", type=int, default=124)
    parser.add_argument("--motif-drills", type=int, default=100)
    parser.add_argument("--operator-drills", type=int, default=350)
    parser.add_argument("--encode-decode-drills", type=int, default=400)
    parser.add_argument("--route-cards", type=int, default=300)
    parser.add_argument(
        "--rhs-length-family-drills",
        type=int,
        default=160,
        help="Compact RHS-length routing drills, balanced across output lengths 1/2/3/4.",
    )
    parser.add_argument("--ambiguity-cards", type=int, default=0)
    return parser.parse_args()


def direct_prediction(template_name: str, lhs: str) -> str:
    return "".join(lhs[i] for i in DIRECT_TEMPLATES[template_name])


def direct_desc(template_name: str) -> str:
    if template_name == "0134":
        return "ABOCD -> ABCD"
    if template_name == "3401":
        return "ABOCD -> CDAB"
    raise ValueError(template_name)


def normalize_method(method: str) -> str:
    if method.startswith("direct_template_"):
        return method.removeprefix("direct_template_")
    raise ValueError(method)


def collect_symbols(puzzle: SymbolTransformPuzzle, answer: str) -> list[str]:
    ordered: list[str] = []
    for example in puzzle.examples:
        for ch in example.lhs + example.rhs:
            if ch not in ordered:
                ordered.append(ch)
    for ch in puzzle.query + answer:
        if ch not in ordered:
            ordered.append(ch)
    return ordered


def label_lines(symbols: list[str]) -> list[str]:
    if len(symbols) > len(LABEL_ALPHABET):
        raise ValueError("Too many symbols to label.")
    return [f"{sym} = {LABEL_ALPHABET[i]}" for i, sym in enumerate(symbols)]


def same_summary(same: tuple[SymbolExample, ...]) -> str:
    return "; ".join(f"{ex.lhs}={ex.rhs}" for ex in same)


def render_direct_trace(
    puzzle: SymbolTransformPuzzle,
    answer: str,
    chosen_template: str,
    *,
    row_id: str,
) -> str:
    same = same_operator_examples(puzzle)
    same_text = same_summary(same)
    lines: list[str] = [
        OPENER,
        BOXED_INTENT,
        "",
        S0_DIRECT,
        "",
        "S1: Classify this as symbol-equation transformation with fixed shape ABOCD.",
        f"- Query is {puzzle.query}; query operator is {puzzle.query_operator}.",
        f"- Same-operator examples: {same_text}.",
        "",
            "S2: Test direct-position templates first.",
        ]

    for template_name in DIRECT_TEMPLATE_PRIORITY:
        rendered = "; ".join(
            f"{example.lhs} -> {direct_prediction(template_name, example.lhs)} vs {example.rhs}"
            for example in same
        )
        all_pass = all(direct_prediction(template_name, example.lhs) == example.rhs for example in same)
        lines.append(f"- Template {template_name}: {rendered}; {'PASS' if all_pass else 'FAIL'}.")
        if template_name == chosen_template:
            break

    if chosen_template == "0134":
        lines.append("- Template 3401: SKIP because Template 0134 already LOCKED by priority.")

    lines.extend(
        [
            f"- LOCK Template {chosen_template} because every same-operator example matches it.",
            "",
            "S3: Apply the locked template to the query.",
            f"- Query: {puzzle.query}.",
            f"- Template {chosen_template} gives {direct_prediction(chosen_template, puzzle.query)}.",
            "",
            BOXED_RETURN,
            f"The final answer is \\boxed{{{answer}}}",
        ]
    )
    if direct_prediction(chosen_template, puzzle.query) != answer:
        raise ValueError(f"{row_id}: direct trace answer mismatch.")
    return "\n".join(lines)


def make_prompt(examples: list[tuple[str, str]], query: str) -> str:
    body = "\n".join(f"{lhs} = {rhs}" for lhs, rhs in examples)
    return (
        f"{SYMBOL_EQUATION_PREFIX}, below are a few examples:\n"
        f"{body}\n"
        f"Now, determine the result for: {query}"
    )


def remap_text(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def remap_puzzle(
    puzzle: SymbolTransformPuzzle,
    answer: str,
    *,
    rng: random.Random,
) -> tuple[SymbolTransformPuzzle, str]:
    symbols = collect_symbols(puzzle, answer)
    if len(symbols) > len(SYMBOL_ALPHABET):
        raise ValueError("Synthetic alphabet is too small.")
    pool = list(SYMBOL_ALPHABET)
    rng.shuffle(pool)
    mapping = {sym: pool[i] for i, sym in enumerate(symbols)}
    examples = tuple(
        SymbolExample(lhs=remap_text(ex.lhs, mapping), rhs=remap_text(ex.rhs, mapping))
        for ex in puzzle.examples
    )
    query = remap_text(puzzle.query, mapping)
    new_answer = remap_text(answer, mapping)
    return SymbolTransformPuzzle(examples=examples, query=query), new_answer


def random_symbols(
    rng: random.Random,
    n: int,
    *,
    alphabet: tuple[str, ...] = SYMBOL_ALPHABET,
) -> list[str]:
    return rng.sample(alphabet, n)


def random_lhs(rng: random.Random, op: str, symbols: list[str]) -> str:
    a, b, c, d = rng.sample([s for s in symbols if s != op], 4)
    lhs = f"{a}{b}{op}{c}{d}"
    if direct_prediction("0134", lhs) == direct_prediction("3401", lhs):
        return random_lhs(rng, op, symbols)
    return lhs


def synthetic_direct_puzzle(
    *,
    rng: random.Random,
    chosen_template: str,
    same_count: int,
    distractor_count: int,
    alphabet: tuple[str, ...] = SYMBOL_ALPHABET,
) -> tuple[SymbolTransformPuzzle, str]:
    symbols = random_symbols(rng, 14, alphabet=alphabet)
    query_op = symbols[0]
    other_ops = symbols[1:5]
    examples: list[SymbolExample] = []
    for _ in range(same_count):
        lhs = random_lhs(rng, query_op, symbols)
        examples.append(SymbolExample(lhs=lhs, rhs=direct_prediction(chosen_template, lhs)))
    for i in range(distractor_count):
        op = other_ops[i % len(other_ops)]
        lhs = random_lhs(rng, op, symbols)
        template = rng.choice(DIRECT_TEMPLATE_PRIORITY)
        examples.append(SymbolExample(lhs=lhs, rhs=direct_prediction(template, lhs)))
    rng.shuffle(examples)
    query = random_lhs(rng, query_op, symbols)
    answer = direct_prediction(chosen_template, query)
    return SymbolTransformPuzzle(examples=tuple(examples), query=query), answer


def add_row(
    rows: list[Phase1Row],
    *,
    row_id: str,
    prompt: str,
    answer: str,
    cot: str,
    source_category: str,
    label: str = "Symbol-Equation Direct Curriculum",
    category: str = "Phase1 Symbol-Equation Direct Curriculum",
    source: str = "deterministic_symbol_transform_direct_curriculum",
) -> None:
    rows.append(
        Phase1Row(
            id=row_id,
            prompt=prompt,
            answer=answer,
            generated_cot=cot,
            label=label,
            category=category,
            source=source,
            source_category=source_category,
        )
    )


def render_card(reasoning: str, final_answer: str) -> str:
    return "\n".join(
        [
            OPENER,
            BOXED_INTENT,
            "",
            reasoning.strip(),
            "",
            BOXED_RETURN,
            f"The final answer is \\boxed{{{final_answer}}}",
        ]
    )


def render_compact_card(task: str, reasoning: str, final_answer: str) -> str:
    return "\n".join(
        [
            task,
            "",
            reasoning.strip(),
            "",
            f"The final answer is \\boxed{{{final_answer}}}",
        ]
    )


def render_simple_card(reasoning: str, final_answer: str) -> str:
    return "\n".join(
        [
            reasoning.strip(),
            "",
            f"The final answer is \\boxed{{{final_answer}}}",
        ]
    )


def render_route_card(reason: str, final_answer: str) -> str:
    return render_compact_card(
        "We need to recall one reusable symbol-equation transformation rule.",
        "\n".join([S0_ROUTE, reason]),
        final_answer,
    )


def family_answer(length: int) -> str:
    _family_name, rules = RHS_LENGTH_FAMILIES[length]
    return ", ".join(rules)


def add_motif_drills(rows: list[Phase1Row], *, count: int, rng: random.Random) -> None:
    for idx in range(count):
        symbolic = (idx // 2) % 2 == 1
        motif = "AB_CD" if idx % 2 == 0 else "BA_DC"
        if symbolic:
            symbols = random_symbols(rng, 5, alphabet=SAFE_SYMBOL_ALPHABET)
            a, b, op, c, d = symbols
            lhs = f"{a}{b}{op}{c}{d}"
        else:
            a, b, c, d = [str(rng.randrange(10)) for _ in range(4)]
            op = rng.choice(("+", "-", "*", "?", "@", "#"))
            lhs = f"{a}{b}{op}{c}{d}"
        if motif == "AB_CD":
            x, y = f"{a}{b}", f"{c}{d}"
            desc = "keep both operands as written"
        else:
            x, y = f"{b}{a}", f"{d}{c}"
            desc = "reverse both operands"
        answer = f"x={x}, y={y}"
        prompt = (
            f"{SYMBOL_EQUATION_PREFIX}, for input {lhs}, what operands are used by motif {motif}?"
        )
        cot = render_compact_card(
            "We need to apply motif rule.",
            "\n".join(
                [
                    S0_MOTIF,
                    f"{lhs} has A={a}, B={b}, O={op}, C={c}, D={d}.",
                    f"{motif} means {desc}: {answer}.",
                ]
            ),
            answer,
        )
        add_row(
            rows,
            row_id=f"st_phase1_motif_{idx:04d}",
            prompt=prompt,
            answer=answer,
            cot=cot,
            source_category="motif_drill",
        )


def apply_rule(rule: str, x: int, y: int) -> int:
    if rule == "x+y":
        return x + y
    if rule == "x+y+1":
        return x + y + 1
    if rule == "x+y-1":
        return x + y - 1
    if rule == "x*y":
        return x * y
    if rule == "x*y+1":
        return x * y + 1
    if rule == "x*y-1":
        return x * y - 1
    if rule == "x-y":
        return x - y
    if rule == "y-x":
        return y - x
    if rule == "|x-y|":
        return abs(x - y)
    raise ValueError(rule)


def add_operator_drills(rows: list[Phase1Row], *, count: int, rng: random.Random) -> None:
    rules = ("x+y", "x*y", "x+y+1", "x+y-1", "x*y+1", "x*y-1", "x-y", "y-x", "|x-y|")
    motifs = ("AB_CD|raw", "BA_DC|rev")
    for idx in range(count):
        rule = rules[idx % len(rules)]
        motif = motifs[(idx // len(rules)) % len(motifs)]
        for _attempt in range(10_000):
            a, b, c, d = [rng.randrange(10) for _ in range(4)]
            lhs = f"{a}{b}@{c}{d}"
            if motif.startswith("AB_CD"):
                x, y = int(f"{a}{b}"), int(f"{c}{d}")
                motif_text = "AB_CD keeps operands as AB and CD"
            else:
                x, y = int(f"{b}{a}"), int(f"{d}{c}")
                motif_text = "BA_DC reverses both operands into BA and DC"
            value = apply_rule(rule, x, y)
            if value >= 0:
                break
        else:
            raise RuntimeError(f"Could not sample nonnegative output for {motif} {rule}")
        value = apply_rule(rule, x, y)
        output = str(value)
        if motif.endswith("|rev"):
            output = output[::-1]
            render_text = "reverse the output digits"
        else:
            render_text = "write the output directly"
        prompt = (
            f"{SYMBOL_EQUATION_PREFIX}, solve {lhs} under motif {motif} and rule {rule}."
        )
        cot = render_compact_card(
            "We need to apply arithmetic rule.",
            "\n".join(
                [
                    S0_OPERATOR,
                    f"{motif_text}: x={x}, y={y}.",
                    f"{rule} gives {apply_rule(rule, x, y)}; {motif.split('|')[1]} means {render_text}, so output is {output}.",
                ]
            ),
            output,
        )
        add_row(
            rows,
            row_id=f"st_phase1_operator_{idx:04d}",
            prompt=prompt,
            answer=output,
            cot=cot,
            source_category="operator_family_drill",
        )


def add_encode_decode_drills(rows: list[Phase1Row], *, count: int, rng: random.Random) -> None:
    for idx in range(count):
        digits = list(range(10))
        syms = random_symbols(rng, 10, alphabet=SAFE_SYMBOL_ALPHABET)
        digit_to_sym = dict(zip(digits, syms))
        sym_to_digit = {sym: digit for digit, sym in digit_to_sym.items()}
        if idx % 2 == 0:
            number = "".join(str(rng.randrange(10)) for _ in range(rng.choice((2, 3, 4))))
            answer = "".join(digit_to_sym[int(ch)] for ch in number)
            prompt = (
                f"{SYMBOL_EQUATION_PREFIX}, encode digits "
                f"{number} using map "
                + ", ".join(f"{d}->{digit_to_sym[d]}" for d in digits)
                + "."
            )
            steps = [f"{ch}->{digit_to_sym[int(ch)]}" for ch in number]
            action = "Encode each digit to its symbol."
        else:
            encoded = "".join(rng.choice(syms) for _ in range(rng.choice((2, 3, 4))))
            answer = "".join(str(sym_to_digit[ch]) for ch in encoded)
            prompt = (
                f"{SYMBOL_EQUATION_PREFIX}, decode symbols "
                f"{encoded} using map "
                + ", ".join(f"{sym}->{sym_to_digit[sym]}" for sym in syms)
                + "."
            )
            steps = [f"{ch}->{sym_to_digit[ch]}" for ch in encoded]
            action = "Decode each symbol to its digit."
        cot = render_compact_card(
            "We need to convert symbols and digits with the provided map.",
            "\n".join(
                [
                    S0_ENCODE_DECODE,
                    f"{action} " + "; ".join(steps) + f"; output is {answer}.",
                ]
            ),
            answer,
        )
        add_row(
            rows,
            row_id=f"st_phase1_encode_decode_{idx:04d}",
            prompt=prompt,
            answer=answer,
            cot=cot,
            source_category="symbol_digit_encode_decode",
        )


def rhs_family_reason(length: int) -> str:
    family_name, rules = RHS_LENGTH_FAMILIES[length]
    return (
        f"Length {length} points to the {family_name}, so the candidate rules are "
        f"{', '.join(rules)}."
    )


def random_rhs_value(length: int, rng: random.Random) -> str:
    return "".join(random_symbols(rng, length, alphabet=LABEL_ALPHABET))


def add_rhs_length_family_drills(rows: list[Phase1Row], *, count: int, rng: random.Random) -> None:
    lengths = (4, 1, 2, 3)
    for idx in range(count):
        length = lengths[idx % len(lengths)]
        answer = family_answer(length)
        rhs_values = [random_rhs_value(length, rng) for _ in range(2 + (idx % 3))]
        prompt = (
            f"{SYMBOL_EQUATION_PREFIX}, direct templates failed. "
            f"The same-operator RHS values are {', '.join(rhs_values)}. "
            "Which arithmetic candidate rules should be tried?"
        )
        cot = render_simple_card(
            rhs_family_reason(length),
            answer,
        )
        add_row(
            rows,
            row_id=f"st_phase1_rhs_length_family_{idx:04d}",
            prompt=prompt,
            answer=answer,
            cot=cot,
            source_category=f"rhs_length_family_drill_{length}",
        )


def add_route_cards(rows: list[Phase1Row], *, count: int) -> None:
    cards = [
        (
            "route_same_operator_first",
            "same-op first",
            "Use examples with the same operator as the query before using other operators.",
            "same-operator examples first",
        ),
        (
            "route_direct_before_arithmetic",
            "direct before arithmetic",
            "Test 0134 and 3401 before encrypted digit arithmetic.",
            "direct templates first",
        ),
        (
            "route_direct_lock",
            "lock direct",
            "If one direct template matches every same-operator example, lock it and skip digit search.",
            "lock direct template and skip digit search",
        ),
        (
            "route_change_variables_after_direct_fail",
            "change variables after direct failure",
            "Only assign symbol variables after both direct templates fail; then try BA_DC|rev or AB_CD|raw.",
            "change variables after direct templates fail",
        ),
        (
            "route_skip_variables_after_direct_lock",
            "skip variables after direct lock",
            "If 0134 or 3401 locks, no symbol-to-variable map is needed.",
            "skip variable assignment after direct lock",
        ),
        (
            "route_main_encrypted_motifs",
            "main encrypted motifs",
            "After direct templates fail, the two main encrypted digit motifs are BA_DC|rev or AB_CD|raw.",
            "BA_DC|rev or AB_CD|raw",
        ),
        (
            "route_template_0134_meaning",
            "0134 meaning",
            "Template 0134 copies positions 0,1,3,4 from ABOCD.",
            "0134 means ABOCD -> ABCD",
        ),
        (
            "route_template_3401_meaning",
            "3401 meaning",
            "Template 3401 copies positions 3,4,0,1 from ABOCD.",
            "3401 means ABOCD -> CDAB",
        ),
        (
            "route_direct_priority",
            "direct priority",
            "Try 0134 before 3401 because it is the first direct-template priority.",
            "try 0134 before 3401",
        ),
        (
            "route_other_operators_later",
            "other operators later",
            "Other-operator examples are secondary and should not override a locked same-operator direct template.",
            "use other operators only later",
        ),
        (
            "route_rhs_length_4",
            "length 4",
            f"If same-operator RHS length is 4 after direct templates fail, try {family_answer(4)}.",
            family_answer(4),
        ),
        (
            "route_rhs_length_3",
            "length 3",
            f"If same-operator RHS length is 3 after direct templates fail, try {family_answer(3)}.",
            family_answer(3),
        ),
        (
            "route_rhs_length_2",
            "length 2",
            f"If same-operator RHS length is 2 after direct templates fail, try {family_answer(2)}.",
            family_answer(2),
        ),
        (
            "route_rhs_length_1",
            "length 1",
            f"If same-operator RHS length is 1 after direct templates fail, try {family_answer(1)}.",
            family_answer(1),
        ),
        (
            "route_visible_grids",
            "visible grids",
            "When encrypted search is needed, keep visible survivor grids instead of hiding the search.",
            "keep visible survivor grids",
        ),
        (
            "route_final_box",
            "final box",
            "Always end by putting the final literal symbol output inside boxed braces.",
            "final answer inside boxed braces",
        ),
        (
            "route_candidate_agreement",
            "candidate agreement",
            "A cipher prediction is clean only when surviving candidates agree on the query output.",
            "require candidate agreement",
        ),
        (
            "route_ambiguity_policy",
            "ambiguity policy",
            "If valid candidates predict different outputs, do not invent a deterministic tie-break.",
            "mark ambiguous instead of guessing",
        ),
        (
            "route_query_application_order",
            "query application order",
            "After locking a cipher rule, apply decode, compute, render, then encode.",
            "decode -> compute -> render -> encode",
        ),
        (
            "route_global_variables",
            "global variables",
            "After direct templates fail, assign each symbol one global variable and keep using it.",
            "one global variable per symbol",
        ),
        (
            "route_bijective_digits",
            "bijective digits",
            "In encrypted digit search, different symbols must map to different digits.",
            "enforce a bijective symbol-digit map",
        ),
        (
            "route_operator_specific_rules",
            "operator-specific rules",
            "Different operator symbols can have different arithmetic rules in the same puzzle.",
            "operators can have different rules",
        ),
        (
            "route_concat_boundary",
            "concat boundary",
            "Direct templates handle concat-like outputs, so concat should not be reused as a later rescue by default.",
            "direct templates cover concat-like cases",
        ),
        (
            "route_reverse_output",
            "reverse output",
            "For BA_DC|rev, reverse the rendered digit text and preserve zeros before encoding symbols.",
            "reverse digits and preserve zeros",
        ),
        (
            "route_verify_same_operator",
            "same-operator verification",
            "Before applying a locked rule to the query, verify it on every same-operator example.",
            "verify every same-operator example",
        ),
    ]
    for idx in range(count):
        source_category, name, reason, answer = cards[idx % len(cards)]
        prompt = (
            f"{SYMBOL_EQUATION_PREFIX}, "
            f"scenario {idx:03d}: after parsing an ABOCD puzzle, what should the solver remember "
            f"about {name}?"
        )
        cot = render_route_card(reason, answer)
        add_row(
            rows,
            row_id=f"st_phase1_route_{idx:04d}",
            prompt=prompt,
            answer=answer,
            cot=cot,
            source_category=source_category,
        )


def add_ambiguity_cards(rows: list[Phase1Row], *, count: int) -> None:
    cards = [
        (
            "If 0134 and 3401 both fail on same-operator examples, should the solver force a direct-template answer?",
            "no, move to encrypted digit search",
            "Direct-template failure means the answer is not justified by position copying.",
        ),
        (
            "If encrypted digit candidates disagree after direct templates fail, should the solver invent a tie-break answer?",
            "no, mark the case ambiguous",
            "Training should not reward arbitrary longest-output or biggest-value guesses.",
        ),
        (
            "If a direct template locks, should the solver still run BA_DC|rev arithmetic?",
            "no, skip encrypted digit search",
            "The direct template is a high-precision early-exit rule.",
        ),
    ]
    for idx in range(count):
        question, answer, reason = cards[idx % len(cards)]
        prompt = f"{SYMBOL_EQUATION_PREFIX}, {question}"
        cot = render_card(
            "\n".join(
                [
                    S0_ROUTE,
                    f"S1: {reason}",
                    f"S2: Therefore: {answer}.",
                ]
            ),
            answer,
        )
        add_row(
            rows,
            row_id=f"st_phase1_ambiguity_{idx:04d}",
            prompt=prompt,
            answer=answer,
            cot=cot,
            source_category="ambiguity_skip_card",
        )


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def audit_authentic_trace(row: Phase1Row, puzzle: SymbolTransformPuzzle, chosen_template: str) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {"pass1_structure": [], "pass2_logic": [], "pass3_training_readiness": []}
    text = row.generated_cot
    lines = text.splitlines()

    required_snippets = [
        OPENER,
        BOXED_INTENT,
        S0_DIRECT,
        "S1: Classify this as symbol-equation transformation with fixed shape ABOCD.",
        "S2: Test direct-position templates first.",
        "S3: Apply the locked template to the query.",
        BOXED_RETURN,
        f"The final answer is \\boxed{{{row.answer}}}",
    ]
    for snippet in required_snippets:
        if snippet not in text:
            issues["pass1_structure"].append(f"missing snippet: {snippet}")
    if not lines or lines[0] != OPENER:
        issues["pass1_structure"].append("opener is not first line")
    if f"- Query is {puzzle.query}; query operator is {puzzle.query_operator}." not in text:
        issues["pass1_structure"].append("query/operator line missing")
    if same_summary(same_operator_examples(puzzle)) not in text:
        issues["pass1_structure"].append("same-operator summary missing")

    same = same_operator_examples(puzzle)
    if not same:
        issues["pass2_logic"].append("no same-operator examples")
    for prior_template in DIRECT_TEMPLATE_PRIORITY:
        preds = [direct_prediction(prior_template, ex.lhs) for ex in same]
        passes = [pred == ex.rhs for pred, ex in zip(preds, same)]
        if prior_template == chosen_template:
            if not all(passes):
                issues["pass2_logic"].append(f"chosen template {chosen_template} does not pass all same-op examples")
            if direct_prediction(chosen_template, puzzle.query) != row.answer:
                issues["pass2_logic"].append("query direct prediction does not match answer")
            break
        if all(passes):
            issues["pass2_logic"].append(
                f"higher-priority template {prior_template} also passes before chosen {chosen_template}"
            )
    for template_name in DIRECT_TEMPLATE_PRIORITY:
        if f"Template {template_name}" not in text:
            issues["pass2_logic"].append(f"template {template_name} not mentioned")
    if "LOCK Template" not in text:
        issues["pass2_logic"].append("LOCK not present")
    if "Since a direct template is locked" in text:
        issues["pass2_logic"].append("redundant direct-template skip sentence present")
    if "S4: Verify the copied output." in text or "S5: Final answer." in text:
        issues["pass2_logic"].append("redundant S4/S5 direct-template steps present")

    forbidden = ["__BS__", "__LB__", "__RB__", "AGREE", "distinct digits from 0 to 9"]
    for token in forbidden:
        if token in text:
            issues["pass3_training_readiness"].append(f"forbidden token: {token}")
    if "\\boxed{}" not in text:
        issues["pass3_training_readiness"].append("boxed intent missing")
    if "\\boxed{" not in text:
        issues["pass3_training_readiness"].append("boxed final missing")
    long_lines = [
        i + 1
        for i, line in enumerate(lines)
        if len(line) > 220 and line != S0_DIRECT
    ]
    if long_lines:
        issues["pass3_training_readiness"].append(f"overlong lines: {long_lines[:5]}")
    if row.answer not in text:
        issues["pass3_training_readiness"].append("answer literal not present in trace")
    return issues


def write_rows(path: Path, rows: list[Phase1Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "source_category",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def balanced_template_plan(
    *,
    current_counts: Counter[str],
    additional_total: int,
    rng: random.Random,
) -> list[str]:
    """Choose extra direct-template labels so total direct rows are as close to 50/50 as possible."""
    current_0134 = current_counts.get("0134", 0)
    current_3401 = current_counts.get("3401", 0)
    final_total = current_0134 + current_3401 + additional_total
    target_0134 = final_total // 2
    target_3401 = final_total - target_0134
    need_0134 = max(0, target_0134 - current_0134)
    need_3401 = max(0, target_3401 - current_3401)
    plan = ["0134"] * need_0134 + ["3401"] * need_3401
    if len(plan) < additional_total:
        # If current rows already overshoot one side, fill the rest with the smaller side.
        smaller = "0134" if current_0134 + plan.count("0134") <= current_3401 + plan.count("3401") else "3401"
        plan.extend([smaller] * (additional_total - len(plan)))
    if len(plan) > additional_total:
        plan = plan[:additional_total]
    rng.shuffle(plan)
    return plan


def write_validation(
    path: Path,
    *,
    count: int,
    rng: random.Random,
    alphabet: tuple[str, ...] = SYMBOL_ALPHABET,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "prompt", "answer", "template", "same_operator_count", "distractor_count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(count):
            template_name = "0134" if idx % 2 == 0 else "3401"
            same_count = 1 + (idx % 5)
            distractor_count = idx % 4
            puzzle, answer = synthetic_direct_puzzle(
                rng=rng,
                chosen_template=template_name,
                same_count=same_count,
                distractor_count=distractor_count,
                alphabet=alphabet,
            )
            writer.writerow(
                {
                    "id": f"symbol_direct_valid_{idx:04d}",
                    "prompt": make_prompt([(ex.lhs, ex.rhs) for ex in puzzle.examples], puzzle.query),
                    "answer": answer,
                    "template": f"direct_template_{template_name}",
                    "same_operator_count": same_count,
                    "distractor_count": distractor_count,
                }
            )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_rows = read_csv_dicts(ROOT / args.train_csv)
    train_by_id = {row["id"]: row for row in train_rows}
    analysis_rows = read_csv_dicts(ROOT / args.analysis_csv)
    direct_records = [
        row
        for row in analysis_rows
        if truthy(row.get("is_exact"))
        and row.get("method", "").startswith("direct_template_")
    ]
    direct_records.sort(key=lambda row: row["id"])

    rows: list[Phase1Row] = []
    authentic_context: dict[str, tuple[SymbolTransformPuzzle, str]] = {}
    direct_template_counts: Counter[str] = Counter()

    for record in direct_records:
        source_row = train_by_id[record["id"]]
        puzzle = parse_symbol_transform_puzzle(source_row["prompt"])
        if puzzle is None:
            raise SystemExit(f"Could not parse symbol transform prompt for {record['id']}")
        template_name = normalize_method(record["method"])
        answer = source_row["answer"].strip()
        cot = render_direct_trace(puzzle, answer, template_name, row_id=record["id"])
        row_id = f"st_phase1_direct_real_{record['id']}"
        add_row(
            rows,
            row_id=row_id,
            prompt=make_prompt([(ex.lhs, ex.rhs) for ex in puzzle.examples], puzzle.query),
            answer=answer,
            cot=cot,
            source_category=f"authentic_direct_template_{template_name}",
        )
        direct_template_counts[template_name] += 1
        authentic_context[row_id] = (puzzle, template_name)

        for variant_idx in range(args.real_synthetic_per_row):
            synth_puzzle, synth_answer = remap_puzzle(puzzle, answer, rng=rng)
            synth_cot = render_direct_trace(
                synth_puzzle,
                synth_answer,
                template_name,
                row_id=f"{record['id']}_remap_{variant_idx}",
            )
            prompt = make_prompt([(ex.lhs, ex.rhs) for ex in synth_puzzle.examples], synth_puzzle.query)
            add_row(
                rows,
                row_id=f"st_phase1_direct_real_remap_{record['id']}_{variant_idx:02d}",
                prompt=prompt,
                answer=synth_answer,
                cot=synth_cot,
                source_category=f"real_anchored_synthetic_direct_template_{template_name}",
            )
            direct_template_counts[template_name] += 1

    current_direct_total = sum(direct_template_counts.values())
    additional_direct_total = max(0, args.direct_template_total - current_direct_total)
    fresh_direct_count = min(args.fresh_direct, additional_direct_total)
    contrast_count = min(args.contrast, additional_direct_total - fresh_direct_count)
    if fresh_direct_count + contrast_count < additional_direct_total:
        contrast_count = additional_direct_total - fresh_direct_count

    additional_direct_plan = balanced_template_plan(
        current_counts=direct_template_counts,
        additional_total=additional_direct_total,
        rng=rng,
    )

    for idx, template_name in enumerate(additional_direct_plan[:fresh_direct_count]):
        puzzle, answer = synthetic_direct_puzzle(
            rng=rng,
            chosen_template=template_name,
            same_count=2 + (idx % 4),
            distractor_count=2 + (idx % 3),
        )
        cot = render_direct_trace(puzzle, answer, template_name, row_id=f"fresh_{idx}")
        prompt = make_prompt([(ex.lhs, ex.rhs) for ex in puzzle.examples], puzzle.query)
        add_row(
            rows,
            row_id=f"st_phase1_direct_fresh_{idx:04d}",
            prompt=prompt,
            answer=answer,
            cot=cot,
            source_category=f"fresh_balanced_direct_template_{template_name}",
        )
        direct_template_counts[template_name] += 1

    contrast_plan = additional_direct_plan[fresh_direct_count : fresh_direct_count + contrast_count]
    for idx, template_name in enumerate(contrast_plan):
        puzzle, answer = synthetic_direct_puzzle(
            rng=rng,
            chosen_template=template_name,
            same_count=2 + (idx % 3),
            distractor_count=1 + (idx % 2),
        )
        cot = render_direct_trace(puzzle, answer, template_name, row_id=f"contrast_{idx}")
        prompt = make_prompt([(ex.lhs, ex.rhs) for ex in puzzle.examples], puzzle.query)
        add_row(
            rows,
            row_id=f"st_phase1_direct_contrast_{idx:04d}",
            prompt=prompt,
            answer=answer,
            cot=cot,
            source_category=f"direct_template_contrast_{template_name}",
        )
        direct_template_counts[template_name] += 1

    add_motif_drills(rows, count=args.motif_drills, rng=rng)
    add_operator_drills(rows, count=args.operator_drills, rng=rng)
    add_encode_decode_drills(rows, count=args.encode_decode_drills, rng=rng)
    add_rhs_length_family_drills(rows, count=args.rhs_length_family_drills, rng=rng)
    add_route_cards(rows, count=args.route_cards)
    add_ambiguity_cards(rows, count=args.ambiguity_cards)

    output_path = ROOT / args.output_csv
    write_rows(output_path, rows)
    validation_path = ROOT / args.validation_csv
    validation_rng = random.Random(args.seed + 9173)
    write_validation(validation_path, count=args.validation_rows, rng=validation_rng)
    safe_validation_path = ROOT / args.safe_validation_csv
    safe_validation_rng = random.Random(args.seed + 31837)
    safe_alphabet = tuple(ch for ch in SYMBOL_ALPHABET if ch not in "\\{}")
    write_validation(
        safe_validation_path,
        count=args.safe_validation_rows,
        rng=safe_validation_rng,
        alphabet=safe_alphabet,
    )

    trace_dir = ROOT / args.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    audit_records: list[dict[str, object]] = []
    all_authentic_pass = True
    for row in rows:
        if not row.id.startswith("st_phase1_direct_real_") or row.id.startswith(
            "st_phase1_direct_real_remap_"
        ):
            continue
        puzzle, template_name = authentic_context[row.id]
        (trace_dir / f"{row.id.removeprefix('st_phase1_direct_real_')}.txt").write_text(
            row.generated_cot + "\n", encoding="utf-8"
        )
        audit = audit_authentic_trace(row, puzzle, template_name)
        passed = all(not issues for issues in audit.values())
        all_authentic_pass = all_authentic_pass and passed
        audit_records.append(
            {
                "id": row.id,
                "template": template_name,
                "passed": passed,
                "passes": audit,
            }
        )

    duplicate_ids = [item for item, n in Counter(row.id for row in rows).items() if n > 1]
    source_counts = Counter(row.source_category for row in rows)
    summary = {
        "output_csv": args.output_csv,
        "validation_csv": args.validation_csv,
        "safe_validation_csv": args.safe_validation_csv,
        "total_rows": len(rows),
        "direct_template_target": args.direct_template_total,
        "direct_template_after_authentic_and_remap": current_direct_total,
        "fresh_direct_rows": fresh_direct_count,
        "contrast_rows": contrast_count,
        "validation_rows": args.validation_rows,
        "safe_validation_rows": args.safe_validation_rows,
        "direct_template_counts": dict(direct_template_counts),
        "authentic_direct_rows": len(audit_records),
        "all_authentic_passed_three_audit_passes": all_authentic_pass,
        "duplicate_ids": duplicate_ids,
        "source_category_counts": dict(source_counts),
        "audit_passes": [
            "pass1_structure",
            "pass2_logic",
            "pass3_training_readiness",
        ],
        "authentic_audit": audit_records,
    }
    audit_json = ROOT / args.audit_json
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    audit_md = ROOT / args.audit_md
    audit_md.parent.mkdir(parents=True, exist_ok=True)
    failed = [record for record in audit_records if not record["passed"]]
    audit_md.write_text(
        "\n".join(
            [
                "# Phase 1 Symbol-Equation Direct Template Audit",
                "",
                f"Output CSV: `{args.output_csv}`",
                f"Validation CSV: `{args.validation_csv}`",
                f"Safe validation CSV: `{args.safe_validation_csv}`",
                f"Total rows: {len(rows)}",
                f"Validation rows: {args.validation_rows}",
                f"Safe validation rows: {args.safe_validation_rows}",
                f"Direct template counts: {dict(direct_template_counts)}",
                f"Authentic direct rows audited: {len(audit_records)}",
                f"All authentic rows passed three audit passes: {all_authentic_pass}",
                "",
                "## Audit Passes",
                "",
                "1. `pass1_structure`: required opener, S0-S3 sections, same-operator summary, and boxed ending.",
                "2. `pass2_logic`: direct template computations, priority order, LOCK/SKIP, and answer match.",
                "3. `pass3_training_readiness`: no placeholders, no false digit claim, no AGREE wording, manageable lines.",
                "",
                "## Source Counts",
                "",
                *[f"- `{key}`: {value}" for key, value in sorted(source_counts.items())],
                "",
                "## Failed Authentic Rows",
                "",
                *(["None."] if not failed else [f"- `{record['id']}`: {record['passes']}" for record in failed]),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote {args.validation_rows} validation rows to {validation_path}")
    print(f"Wrote {args.safe_validation_rows} safe validation rows to {safe_validation_path}")
    print(f"Authentic direct rows audited: {len(audit_records)}")
    print(f"All authentic rows passed three audit passes: {all_authentic_pass}")
    print(f"Audit JSON: {audit_json}")
    print(f"Audit MD: {audit_md}")
    if duplicate_ids:
        raise SystemExit(f"Duplicate ids found: {duplicate_ids[:5]}")
    if not all_authentic_pass:
        raise SystemExit("Authentic audit failed; inspect audit JSON/MD.")


if __name__ == "__main__":
    main()
