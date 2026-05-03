#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MethodRow:
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
        description="Generate Phase 1B methodology cards for Symbol Transform (scaled, difficulty-weighted)."
    )
    p.add_argument(
        "--target-rows",
        type=int,
        default=650,
        help="Approximate total rows (harder categories get larger share).",
    )
    p.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1b_symbol_transform_methodology.csv",
        help="Destination CSV in SFT schema.",
    )
    return p.parse_args()


def wrap_think(reasoning: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>"


def add_row(
    rows: list[MethodRow],
    *,
    row_id: str,
    prompt: str,
    answer: str,
    reasoning: str,
    source_category: str,
) -> None:
    rows.append(
        MethodRow(
            id=row_id,
            prompt=prompt,
            answer=answer,
            generated_cot=wrap_think(reasoning),
            label="Symbol Transform Methodology",
            category="Phase1B Symbol Transform Methodology",
            source="deterministic_symbol_transform_methodology",
            source_category=source_category,
        )
    )


def allocate_counts(target: int) -> dict[str, int]:
    """Harder methodology buckets get more rows."""
    fixed = {
        "router_card": min(18, max(14, target // 45)),
    }
    remaining = target - sum(fixed.values())
    if remaining < 0:
        fixed["router_card"] = max(6, target // 10)
        remaining = target - sum(fixed.values())

    weights = {
        "cipher_candidate_decision": 0.30,
        "ambiguity_discipline": 0.17,
        "confidence_gating": 0.17,
        "cipher_transition": 0.12,
        "query_application": 0.07,
        "direct_template_workflow": 0.05,
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
    while sum(out.values()) > target:
        for k in ("query_application", "direct_template_workflow", "cipher_transition"):
            if out[k] > 15:
                out[k] -= 1
                if sum(out.values()) <= target:
                    break
    while sum(out.values()) < target:
        out["cipher_candidate_decision"] += 1
    return out


def add_router_cards(rows: list[MethodRow], *, cap: int) -> None:
    cards = [
        (
            "st_meth_router_order",
            "In Alice's Wonderland Symbol Transform methodology, what is the route order?",
            "same-op direct templates -> same-op encrypted digit-transform -> abstain if evidence is too sparse",
            "Start with cheap direct template checks, then encrypted digit-transform only with enough same-op evidence; do not use position-substitution priors for sparse rows.",
        ),
        (
            "st_meth_same_op",
            "In Alice's Wonderland Symbol Transform methodology, why isolate same-operator examples first?",
            "because operator semantics are row-local and operator-specific",
            "Same-op evidence controls query-operator rule fit; mixing operators too early creates false constraints.",
        ),
        (
            "st_meth_reject_fail",
            "In Alice's Wonderland Symbol Transform methodology, when should a candidate be rejected?",
            "as soon as one same-operator example mismatches",
            "A single mismatch is enough to reject the candidate and move to the next rule/template.",
        ),
        (
            "st_meth_lock_rule",
            "In Alice's Wonderland Symbol Transform methodology, when is motif+rule locked?",
            "when it stays consistent across all same-operator examples",
            "Lock only after full same-op consistency checks pass.",
        ),
        (
            "st_meth_predict_gate",
            "In Alice's Wonderland Symbol Transform methodology, when is final prediction emitted?",
            "when surviving candidates agree on one output",
            "If candidates disagree, mark ambiguous instead of forcing an answer.",
        ),
        (
            "st_meth_adaptive_retry",
            "In Alice's Wonderland Symbol Transform methodology, when should adaptive retry be used?",
            "for no_rule cases only (safe retry policy)",
            "Safe retry widens search for no_rule rows; avoid ambiguous retry by default to protect precision.",
        ),
        (
            "st_meth_decode_first",
            "In Alice's Wonderland Symbol Transform methodology, must operands be decoded before arithmetic?",
            "yes",
            "Cipher mode always decodes symbols to digits before applying the locked arithmetic rule.",
        ),
        (
            "st_meth_encode_last",
            "In Alice's Wonderland Symbol Transform methodology, when is the final symbol string produced?",
            "after render mode is applied and digits are encoded back to symbols",
            "Encode is the last step after compute and output-mode rendering.",
        ),
        (
            "st_meth_no_cross_row",
            "In Alice's Wonderland Symbol Transform methodology, can symbol meanings be copied from another row?",
            "no",
            "Each row is self-contained; cross-row symbol reuse is not assumed.",
        ),
        (
            "st_meth_no_sparse_prior",
            "In Alice's Wonderland Symbol Transform methodology, should a one-example position-substitution prior be used after direct templates fail?",
            "no",
            "One-example position priors can force answers from accidental symbol overlap; mark unresolved instead.",
        ),
        (
            "st_meth_cipher_min_same_op",
            "In Alice's Wonderland Symbol Transform methodology, what is the minimum same-operator evidence for encrypted digit-transform search to emit a final prediction?",
            "at least 2 same-operator examples for the query operator",
            "One same-op example leaves too many symbol-digit maps; SKIP broad cipher search and mark ambiguous unless direct templates already solved it.",
        ),
        (
            "st_meth_one_same_op_no_cipher",
            "In Alice's Wonderland Symbol Transform methodology, if there is only one same-operator example and direct templates fail, should broad encrypted digit-transform be trusted?",
            "no, SKIP encrypted digit-transform",
            "One same-op example is below the reliability gate for broad cipher search; SKIP it and mark the row unresolved rather than using a position-substitution prior.",
        ),
        (
            "st_meth_global_motif_min",
            "In Alice's Wonderland Symbol Transform methodology, what same-operator support is needed before row-global motif consistency can support a cipher prediction?",
            "at least 2 same-operator examples",
            "Global motif consistency is useful only after the query operator itself has enough evidence to avoid one-example hallucinated cipher maps.",
        ),
        (
            "st_meth_example_order",
            "In Alice's Wonderland Symbol Transform methodology, should same-operator examples be checked in a fixed row order?",
            "yes, use the provided example order for deterministic tie handling",
            "Stable ordering makes mismatch rejection and candidate scanning reproducible across runs.",
        ),
    ]
    for rid, prompt, ans, reason in cards[:cap]:
        add_row(rows, row_id=rid, prompt=prompt, answer=ans, reasoning=reason, source_category="router_card")


def add_direct_template_cards(rows: list[MethodRow], *, cap: int) -> None:
    idx = 1
    for same_count, failing_template, locked, check_pass in itertools.product(
        range(1, 16),
        ("0134", "3401"),
        ("0134", "3401"),
        ("first", "second", "third"),
    ):
        if idx > cap:
            return
        if failing_template == locked:
            continue
        add_row(
            rows,
            row_id=f"st_meth_direct_{idx:05d}",
            prompt=(
                "In Alice's Wonderland Symbol Transform methodology, query has "
                f"{same_count} same-operator example(s). Template {failing_template} fails on the {check_pass} check, "
                f"and template {locked} passes all checks. What should the solver do next?"
            ),
            answer=f"lock template {locked} and apply it to query",
            reasoning=(
                f"Reject {failing_template} immediately on mismatch; lock {locked} after all same-op checks pass; then apply to query."
            ),
            source_category="direct_template_workflow",
        )
        idx += 1


def add_cipher_transition_cards(rows: list[MethodRow], *, cap: int) -> None:
    idx = 1
    ops = ("*", "+", "-", "/", "|", '"')
    for op, direct_fail_count, same_count in itertools.product(ops, range(1, 5), range(2, 9)):
        if idx > cap:
            return
        add_row(
            rows,
            row_id=f"st_meth_cipher_trans_{idx:04d}",
            prompt=(
                "In Alice's Wonderland Symbol Transform methodology, for query operator "
                f"{op}, direct templates fail on {direct_fail_count} check(s), and there are "
                f"{same_count} same-operator example(s). Which route should run next?"
            ),
            answer="switch to encrypted digit-transform search",
            reasoning=(
                "After direct template failure, run bijective symbol-digit motif+rule search on same-op evidence."
            ),
            source_category="cipher_transition",
        )
        idx += 1


def add_cipher_decision_cards(rows: list[MethodRow], *, cap: int) -> None:
    idx = 1
    motifs = (
        "AB_CD|raw",
        "AB_CD|rev",
        "BA_DC|raw",
        "BA_DC|rev",
        "AB_DC|abs",
        "BA_CD|raw",
        "AB_DC|raw",
        "BA_CD|rev",
        "AB_CD|abs",
        "BA_DC|abs",
    )
    rules = (
        "x + y",
        "x * y",
        "x - y",
        "|x - y|",
        "x * y - 1",
        "x + y + 1",
        "y - x",
        "concat(x,y)",
    )
    for motif, rule, fail_examples in itertools.product(motifs, rules, (0, 1, 2, 3)):
        if idx > cap:
            return
        if fail_examples == 0:
            ans = "candidate remains viable"
            reason = f"Motif {motif} with rule {rule} matches all checked same-op examples, so keep it."
        else:
            ans = "reject candidate and continue scan"
            reason = f"If {fail_examples} same-op examples fail, reject motif {motif} + rule {rule} immediately."
        add_row(
            rows,
            row_id=f"st_meth_cipher_dec_{idx:05d}",
            prompt=(
                "In Alice's Wonderland Symbol Transform methodology, evaluate candidate "
                f"{motif} with rule {rule}. It fails on {fail_examples} same-operator example(s). "
                "What is the correct action?"
            ),
            answer=ans,
            reasoning=reason,
            source_category="cipher_candidate_decision",
        )
        idx += 1


def add_confidence_cards(rows: list[MethodRow], *, cap: int) -> None:
    idx = 1
    routes = ("direct", "cipher", "prior")
    for same_count, surviving, route in itertools.product(range(0, 7), range(0, 6), routes):
        if idx > cap:
            return
        if surviving == 0:
            ans = "none"
            reason = "No surviving candidate means no prediction."
        elif surviving == 1 and same_count >= 2 and route in {"direct", "cipher"}:
            ans = "high"
            reason = "Unique survivor with sufficient same-op evidence is high confidence."
        elif surviving == 1:
            ans = "medium"
            reason = "Unique survivor with sparse evidence is medium confidence."
        else:
            ans = "ambiguous"
            reason = "Multiple survivors imply ambiguity; do not force deterministic output."
        add_row(
            rows,
            row_id=f"st_meth_conf_{idx:05d}",
            prompt=(
                "In Alice's Wonderland Symbol Transform methodology, assign confidence when "
                f"route={route}, same-op examples={same_count}, surviving candidates={surviving}."
            ),
            answer=ans,
            reasoning=reason,
            source_category="confidence_gating",
        )
        idx += 1


def add_query_application_cards(rows: list[MethodRow], *, cap: int) -> None:
    idx = 1
    scenarios = [
        ("pairing BA_DC, rule x+y, mode rev", "decode -> compute -> reverse digits -> encode"),
        ("pairing AB_CD, rule x*y, mode raw", "decode -> multiply -> keep raw digits -> encode"),
        ("pairing AB_DC, rule x-y, mode abs", "decode -> subtract -> abs digits -> encode"),
        ("pairing BA_CD, rule x*y-1, mode rev", "decode -> multiply minus one -> reverse digits -> encode"),
        ("pairing AB_CD, rule concat(x,y), mode raw", "decode -> concatenate -> raw digits -> encode"),
        ("pairing BA_CD, rule y-x, mode abs", "decode -> swap-subtract -> abs digits -> encode"),
        ("pairing AB_DC, rule x+y+1, mode raw", "decode -> add with offset -> raw digits -> encode"),
        ("pairing BA_DC, rule |x-y|, mode rev", "decode -> absolute difference -> reverse digits -> encode"),
    ]
    for scenario, ans in scenarios:
        for variant in range(1, 200):
            if idx > cap:
                return
            add_row(
                rows,
                row_id=f"st_meth_apply_{idx:05d}",
                prompt=(
                    "In Alice's Wonderland Symbol Transform methodology, after locking "
                    f"{scenario} (variant {variant}), what is the correct query application sequence?"
                ),
                answer=ans,
                reasoning="Always apply locked candidate with decode -> compute -> render -> encode order.",
                source_category="query_application",
            )
            idx += 1


def add_ambiguity_discipline_cards(rows: list[MethodRow], *, cap: int) -> None:
    idx = 1
    cases = [
        "two cipher candidates both fit same-op examples but predict different query outputs",
        "direct route and cipher route both fit but produce different query outputs",
        "adaptive retry adds a new candidate conflicting with previous unique output",
        "same-op count is one and two locally possible position templates produce different substitutions",
        "two motifs with identical same-op loss but different query encodings",
        "rev vs raw rendering ties after decode+compute agree on magnitude only",
    ]
    actions = {
        "abstain": "mark ambiguous and avoid forcing deterministic answer",
        "policy": "apply declared deterministic tie-break policy and lower confidence",
    }
    for case, (action_key, action_text), variant in itertools.product(cases, actions.items(), range(1, 80)):
        if idx > cap:
            return
        add_row(
            rows,
            row_id=f"st_meth_amb_{idx:05d}",
            prompt=(
                "In Alice's Wonderland Symbol Transform methodology, suppose "
                f"{case} (variant {variant}) and policy track is '{action_key}'. What should the solver do?"
            ),
            answer=action_text,
            reasoning=(
                "Ambiguity must be explicit: either abstain or apply a declared deterministic tie-break with reduced confidence."
            ),
            source_category="ambiguity_discipline",
        )
        idx += 1


def build_rows(target: int) -> list[MethodRow]:
    counts = allocate_counts(target)
    rows: list[MethodRow] = []
    add_router_cards(rows, cap=counts["router_card"])
    add_direct_template_cards(rows, cap=counts["direct_template_workflow"])
    add_cipher_transition_cards(rows, cap=counts["cipher_transition"])
    add_cipher_decision_cards(rows, cap=counts["cipher_candidate_decision"])
    add_confidence_cards(rows, cap=counts["confidence_gating"])
    add_ambiguity_discipline_cards(rows, cap=counts["ambiguity_discipline"])
    add_query_application_cards(rows, cap=counts["query_application"])
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
    print(f"Wrote {len(rows)} symbol-transform Phase 1B methodology rows to {out} (target {args.target_rows})")


if __name__ == "__main__":
    main()
