#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAIRINGS = ("AB_CD", "AB_DC", "BA_CD", "BA_DC")
OUTPUT_MODES = ("raw", "rev", "abs")
DIRECT_TEMPLATES = {
    "0134": (0, 1, 3, 4),
    "3401": (3, 4, 0, 1),
}
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
OPERATORS = ("*", "+", "-", "/", "|", '"')


@dataclass(frozen=True)
class KnowledgeRow:
    id: str
    prompt: str
    answer: str
    generated_cot: str
    label: str
    category: str
    source: str
    source_category: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Phase 1A stable knowledge cards for Symbol Transform (scaled, difficulty-weighted)."
    )
    p.add_argument(
        "--target-rows",
        type=int,
        default=900,
        help="Approximate total rows (harder categories get larger share).",
    )
    p.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1a_symbol_transform_knowledge.csv",
        help="Destination CSV in SFT schema.",
    )
    return p.parse_args()


def wrap_think(reasoning: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>"


def add_row(
    rows: list[KnowledgeRow],
    *,
    row_id: str,
    prompt: str,
    answer: str,
    reasoning: str,
    source_category: str,
) -> None:
    rows.append(
        KnowledgeRow(
            id=row_id,
            prompt=prompt,
            answer=answer,
            generated_cot=wrap_think(reasoning),
            label="Symbol Transform Knowledge",
            category="Phase1A Symbol Transform Knowledge",
            source="deterministic_symbol_transform_knowledge",
            source_category=source_category,
        )
    )


def apply_pairing(pairing: str, lhs: str) -> tuple[int, int]:
    a, b, c, d = lhs[0], lhs[1], lhs[3], lhs[4]
    if pairing == "AB_CD":
        x, y = a + b, c + d
    elif pairing == "AB_DC":
        x, y = a + b, d + c
    elif pairing == "BA_CD":
        x, y = b + a, c + d
    elif pairing == "BA_DC":
        x, y = b + a, d + c
    else:
        raise ValueError(pairing)
    return int(x), int(y)


def render_mode(mode: str, value: int) -> str | None:
    if mode == "abs":
        return str(abs(value))
    if value < 0:
        return None
    text = str(value)
    if mode == "raw":
        return text
    if mode == "rev":
        return text[::-1]
    return None


def apply_base_rule(rule: str, x: int, y: int) -> int:
    if rule == "x * y":
        return x * y
    if rule == "x + y":
        return x + y
    if rule == "x + y + 1":
        return x + y + 1
    if rule == "x + y - 1":
        return x + y - 1
    if rule == "x * y + 1":
        return x * y + 1
    if rule == "x * y - 1":
        return x * y - 1
    if rule == "x - y":
        return x - y
    if rule == "y - x":
        return y - x
    if rule == "|x - y|":
        return abs(x - y)
    if rule == "concat(x, y)":
        return int(f"{x}{y}")
    if rule == "concat(y, x)":
        return int(f"{y}{x}")
    raise ValueError(rule)


def lhs_from_digits(a: int, b: int, op: str, c: int, d: int) -> str:
    return f"{a % 10}{b % 10}{op}{c % 10}{d % 10}"


def allocate_counts(target: int) -> dict[str, int]:
    """Difficulty-weighted: harder categories get more rows."""
    # Small fixed buckets first
    fixed = {
        "rule_card": min(16, max(10, target // 60)),
        "sparse_signal_policy": min(120, max(60, target // 8)),
    }
    remaining = target - sum(fixed.values())
    if remaining < 0:
        fixed["sparse_signal_policy"] = max(40, target - fixed["rule_card"])
        remaining = target - sum(fixed.values())

    # Remaining split by weight (sum=1.0)
    weights = {
        "base_rule_semantics": 0.34,
        "digit_symbol_encoding": 0.16,
        "pairing_transform": 0.18,
        "output_mode_semantics": 0.14,
        "direct_template": 0.18,
    }
    wsum = sum(weights.values())
    raw = {k: int(remaining * weights[k] / wsum) for k in weights}
    drift = remaining - sum(raw.values())
    order = sorted(weights, key=lambda k: -weights[k])
    i = 0
    while drift > 0:
        raw[order[i % len(order)]] += 1
        drift -= 1
        i += 1
    out = {**fixed, **raw}
    # Final safety: trim if tiny overshoot
    while sum(out.values()) > target:
        for k in ("direct_template", "output_mode_semantics", "pairing_transform"):
            if out[k] > 20:
                out[k] -= 1
                if sum(out.values()) <= target:
                    break
    while sum(out.values()) < target:
        out["base_rule_semantics"] += 1
    return out


def add_rule_cards(rows: list[KnowledgeRow], *, cap: int) -> None:
    cards = [
        (
            "st_rule_abocd",
            "In Alice's Wonderland Symbol Transform knowledge, what is the fixed query shape used by this solver?",
            "ABOCD (operator at index 2)",
            "The solver assumes 5-char inputs with left pair AB, operator O, and right pair CD.",
        ),
        (
            "st_rule_same_op_first",
            "In Alice's Wonderland Symbol Transform knowledge, what evidence should be used first for query solving?",
            "examples with the same operator as the query",
            "Same-operator examples are primary evidence; other operators are secondary context.",
        ),
        (
            "st_rule_direct_before_cipher",
            "In Alice's Wonderland Symbol Transform knowledge, which route is tested first?",
            "direct position templates before encrypted digit-transform",
            "Direct template checks are cheap and high-precision; cipher route is used when direct checks fail.",
        ),
        (
            "st_rule_cipher_minimum",
            "In Alice's Wonderland Symbol Transform knowledge, what is the minimum same-operator evidence for reliable encrypted digit-transform search?",
            "at least 2 same-operator examples for the query operator",
            "With only one same-op example, SKIP encrypted digit-transform because too many symbol-digit maps can fit; use direct templates if they pass, otherwise abstain.",
        ),
        (
            "st_rule_bijection",
            "In Alice's Wonderland Symbol Transform knowledge, what mapping constraint is enforced in cipher mode?",
            "bijective symbol-to-digit mapping within a row",
            "Each symbol maps to one digit and each digit used in mapping maps back to one symbol for the row.",
        ),
        (
            "st_rule_unique",
            "In Alice's Wonderland Symbol Transform knowledge, when should the solver emit a prediction?",
            "when surviving candidates agree on a unique output",
            "Prediction requires candidate agreement; disagreement indicates ambiguity.",
        ),
        (
            "st_rule_operator_local",
            "In Alice's Wonderland Symbol Transform knowledge, do operator symbols have global meanings across rows?",
            "no",
            "Operator symbols are row-local; meaning is inferred from examples in the same row.",
        ),
        (
            "st_rule_decode_pipeline",
            "In Alice's Wonderland Symbol Transform knowledge, what is the standard cipher solve pipeline?",
            "decode -> compute -> render -> encode",
            "Decode operands to digits, compute, apply output mode, then encode digits back to symbols.",
        ),
        (
            "st_rule_pairing_vocab",
            "In Alice's Wonderland Symbol Transform knowledge, list the four operand pairings used in cipher mode.",
            "AB_CD, AB_DC, BA_CD, BA_DC",
            "These pairings define how two 2-digit operands are formed from the four non-operator symbols.",
        ),
        (
            "st_rule_output_vocab",
            "In Alice's Wonderland Symbol Transform knowledge, list the three output rendering modes used in cipher mode.",
            "raw, rev, abs",
            "raw keeps digit order, rev reverses digit string, abs uses absolute value before encoding.",
        ),
        (
            "st_rule_concat_semantics",
            "In Alice's Wonderland Symbol Transform knowledge, what does concat(x,y) mean numerically?",
            "concatenate decimal representations of x and y as strings, then parse as an integer",
            "Concat treats operands as integers and concatenates their decimal forms.",
        ),
    ]
    extra = [
        (
            f"st_rule_extra_{i}",
            f"In Alice's Wonderland Symbol Transform knowledge (rule fact {i}), should decode happen before applying the locked arithmetic rule?",
            "yes",
            "Operands must be digits before arithmetic; symbols are row-local ciphertext.",
        )
        for i in range(1, max(0, cap - len(cards)) + 1)
    ]
    all_cards = list(cards) + extra
    for rid, prompt, ans, reason in all_cards[:cap]:
        add_row(rows, row_id=rid, prompt=prompt, answer=ans, reasoning=reason, source_category="rule_card")


def add_sparse_signal_cards(rows: list[KnowledgeRow], *, cap: int) -> None:
    bases = [
        (
            "single same-operator example",
            "SKIP encrypted digit-transform; only direct templates can solve safely",
            "A single same-op example is below the minimum for reliable encrypted digit-transform; if direct templates fail, mark the row unresolved instead of using a position-substitution prior.",
        ),
        (
            "two same-operator examples",
            "encrypted digit-transform search reaches the minimum evidence gate",
            "Two same-op examples are the minimum evidence level for motif+rule search to emit a reliable cipher prediction.",
        ),
        (
            "three or more same-operator examples",
            "encrypted digit-transform search with strong consistency checks",
            "More same-op examples increase confidence in motif+rule fits.",
        ),
    ]
    idx = 1
    for variant in range(1, cap + 1):
        same_label, route, reason = bases[variant % len(bases)]
        add_row(
            rows,
            row_id=f"st_sparse_{idx:04d}",
            prompt=(
                "In Alice's Wonderland Symbol Transform knowledge (variant "
                f"{variant}), if the row has {same_label}, which route is preferred before broad cipher enumeration?"
            ),
            answer=route,
            reasoning=reason,
            source_category="sparse_signal_policy",
        )
        idx += 1


def add_template_cards(rows: list[KnowledgeRow], *, cap: int) -> None:
    idx = 1
    for a, b, c, d in itertools.product(range(10), repeat=4):
        if idx > cap:
            return
        for op in OPERATORS:
            if idx > cap:
                return
            query = lhs_from_digits(a, b, op, c, d)
            for name, tpl in DIRECT_TEMPLATES.items():
                if idx > cap:
                    return
                out = "".join(query[i] for i in tpl)
                add_row(
                    rows,
                    row_id=f"st_template_{idx:05d}",
                    prompt=(
                        "In Alice's Wonderland Symbol Transform knowledge, apply direct template "
                        f"{name} to query {query}."
                    ),
                    answer=out,
                    reasoning=f"Template {name} copies indices {tpl}, so {query} -> {out}.",
                    source_category="direct_template",
                )
                idx += 1


def add_pairing_cards(rows: list[KnowledgeRow], *, cap: int) -> None:
    idx = 1
    for pairing in PAIRINGS:
        for a, b, c, d in itertools.product(range(10), repeat=4):
            if idx > cap:
                return
            for op in OPERATORS:
                if idx > cap:
                    return
                lhs = lhs_from_digits(a, b, op, c, d)
                x, y = apply_pairing(pairing, lhs)
                add_row(
                    rows,
                    row_id=f"st_pairing_{idx:05d}",
                    prompt=(
                        "In Alice's Wonderland Symbol Transform knowledge, decode pairing "
                        f"{pairing} on lhs {lhs}. Return x,y."
                    ),
                    answer=f"{x},{y}",
                    reasoning=f"{pairing} maps {lhs} to x={x}, y={y}.",
                    source_category="pairing_transform",
                )
                idx += 1


def add_base_rule_cards(rows: list[KnowledgeRow], *, cap: int) -> None:
    idx = 1
    xy_pairs = [(x, y) for x in range(1, 32) for y in range(1, 32)]
    for x, y in itertools.cycle(xy_pairs):
        if idx > cap:
            return
        for rule in CORE_RULES:
            if idx > cap:
                return
            value = apply_base_rule(rule, x, y)
            add_row(
                rows,
                row_id=f"st_base_{idx:05d}",
                prompt=(
                    "In Alice's Wonderland Symbol Transform knowledge, compute base rule "
                    f"{rule} for x={x}, y={y}."
                ),
                answer=str(value),
                reasoning=f"Apply {rule} directly on x={x}, y={y}: {value}.",
                source_category="base_rule_semantics",
            )
            idx += 1


def add_output_mode_cards(rows: list[KnowledgeRow], *, cap: int) -> None:
    idx = 1
    # Interleave modes so raw/rev/abs all appear in scaled exports.
    per_mode_values: dict[str, list[int]] = {
        "raw": list(range(0, 401)),
        "rev": list(range(0, 401)),
        "abs": list(range(-400, 401)),
    }
    mode_order = itertools.cycle(OUTPUT_MODES)
    mode_iters = {m: itertools.cycle(vs) for m, vs in per_mode_values.items()}

    while idx <= cap:
        mode = next(mode_order)
        value = next(mode_iters[mode])
        rendered = render_mode(mode, value)
        if rendered is None:
            continue
        add_row(
            rows,
            row_id=f"st_outmode_{idx:05d}",
            prompt=(
                "In Alice's Wonderland Symbol Transform knowledge, render value "
                f"{value} using output mode {mode} (row {idx})."
            ),
            answer=rendered,
            reasoning=f"Mode {mode} renders {value} as {rendered}.",
            source_category="output_mode_semantics",
        )
        idx += 1


def add_mapping_cards(rows: list[KnowledgeRow], *, cap: int) -> None:
    """Harder: many small bijections over 0..4 and encode digit strings."""
    idx = 1
    digit_strings = [
        "01234",
        "43210",
        "30142",
        "12403",
        "23401",
        "40321",
        "31420",
        "20134",
        "42301",
        "13024",
    ]
    for perm in itertools.permutations(range(5), 5):
        if idx > cap:
            return
        sym_map = {chr(ord("A") + i): perm[i] for i in range(5)}
        inv = {v: k for k, v in sym_map.items()}
        symbols = ",".join(f"{k}->{v}" for k, v in sorted(sym_map.items()))
        for ds in digit_strings:
            if idx > cap:
                return
            if any(int(ch) not in inv for ch in ds):
                continue
            encoded = "".join(inv[int(ch)] for ch in ds)
            add_row(
                rows,
                row_id=f"st_map_{idx:05d}",
                prompt=(
                    "In Alice's Wonderland Symbol Transform knowledge, encode digit string "
                    f"{ds} using symbol-digit map {symbols}."
                ),
                answer=encoded,
                reasoning=f"Map each digit back to its symbol: {ds} -> {encoded}.",
                source_category="digit_symbol_encoding",
            )
            idx += 1


def build_rows(target: int) -> list[KnowledgeRow]:
    counts = allocate_counts(target)
    rows: list[KnowledgeRow] = []
    add_rule_cards(rows, cap=counts["rule_card"])
    add_sparse_signal_cards(rows, cap=counts["sparse_signal_policy"])
    add_template_cards(rows, cap=counts["direct_template"])
    add_pairing_cards(rows, cap=counts["pairing_transform"])
    add_base_rule_cards(rows, cap=counts["base_rule_semantics"])
    add_output_mode_cards(rows, cap=counts["output_mode_semantics"])
    add_mapping_cards(rows, cap=counts["digit_symbol_encoding"])
    return rows


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.target_rows)
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
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(r.__dict__ for r in rows)
    print(f"Wrote {len(rows)} symbol-transform Phase 1A knowledge rows to {out} (target {args.target_rows})")


if __name__ == "__main__":
    main()
