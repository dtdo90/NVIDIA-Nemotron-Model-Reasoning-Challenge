#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Phase 1B methodology cards for numeric_equation. "
            "Cards teach same-operator-first workflow, motif usage, absent-op handling, "
            "and explicit ambiguity discipline."
        )
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1b_numeric_equation_methodology.csv",
        help="Destination CSV in SFT schema.",
    )
    return parser.parse_args()


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
            label="Numeric Equation Methodology",
            category="Phase1B Numeric Equation Methodology",
            source="deterministic_numeric_equation_methodology",
            source_category=source_category,
        )
    )


def add_procedure_cards(rows: list[MethodRow]) -> None:
    cards = [
        (
            "ne_meth_step_order",
            "In Alice's Wonderland numeric_equation methodology, what is the correct high-level order?",
            "same-op examples -> infer pairing/base/outmode/width -> verify -> apply to query",
            "Use same-op evidence first, infer the rule, verify on examples, then apply to the query.",
        ),
        (
            "ne_meth_other_ops_role",
            "In Alice's Wonderland numeric_equation methodology, what role do other operators play when query operator has examples?",
            "row-level motif evidence only",
            "Other operators mainly suggest row motif (pairing/output style), not direct query-operator base rule.",
        ),
        (
            "ne_meth_absent_op",
            "In Alice's Wonderland numeric_equation methodology, what if the query operator has no examples?",
            "infer motif from other operators, then use a stated deterministic tie-break",
            "For absent-op rows, rely on motif and an explicit tie-break such as simplest unused base.",
        ),
        (
            "ne_meth_ambiguity",
            "In Alice's Wonderland numeric_equation methodology, what if several rules fit all visible examples?",
            "treat it as ambiguous and apply a declared tie-break",
            "When multiple rules are equally consistent, do not pretend uniqueness. Use a declared deterministic policy.",
        ),
        (
            "ne_meth_verify",
            "In Alice's Wonderland numeric_equation methodology, when is a candidate rule trusted?",
            "only after it reproduces every relevant example exactly",
            "A candidate must match all same-op training examples before being applied to query.",
        ),
        (
            "ne_meth_width",
            "In Alice's Wonderland numeric_equation methodology, where should width be inferred from?",
            "from visible example formatting",
            "Width should be inferred from the row's visible outputs, not guessed from the query alone.",
        ),
    ]
    for row_id, prompt, answer, reasoning in cards:
        add_row(rows, row_id=row_id, prompt=prompt, answer=answer, reasoning=reasoning, source_category="procedure_card")


def add_same_op_decision_cards(rows: list[MethodRow]) -> None:
    idx = 1
    for n_same in (0, 1, 2, 3):
        for n_fit in (0, 1, 2, 3, 4):
            if n_same == 0:
                answer = "absent-op pathway"
                reasoning = "No same-op evidence exists, so infer motif from other operators and apply absent-op policy."
            elif n_fit == 0:
                answer = "no candidate in current DSL"
                reasoning = "Same-op examples exist but no candidate fits all of them; this row is outside current DSL."
            elif n_fit == 1:
                answer = "unique same-op candidate"
                reasoning = "Exactly one candidate survives same-op checks, so select it and verify formatting details."
            else:
                answer = "ambiguous same-op candidates"
                reasoning = "Several candidates fit same-op examples, so use declared tie-break and report lower confidence."
            add_row(
                rows,
                row_id=f"ne_meth_sameop_matrix_{idx:03d}",
                prompt=(
                    "In Alice's Wonderland numeric_equation methodology, suppose the query has "
                    f"{n_same} same-operator example(s) and {n_fit} candidate rule(s) fit those examples. "
                    "Which pathway should be used?"
                ),
                answer=answer,
                reasoning=reasoning,
                source_category="same_op_decision_matrix",
            )
            idx += 1


def add_motif_cards(rows: list[MethodRow]) -> None:
    motifs = (
        ("AB_CD", "plain"),
        ("AB_CD", "rev"),
        ("BA_DC", "plain"),
        ("BA_DC", "rev"),
    )
    idx = 1
    for pairing, outmode in motifs:
        for supporters in (1, 2, 3, 4, 5):
            answer = f"use motif {pairing}|{outmode} as prior"
            add_row(
                rows,
                row_id=f"ne_meth_motif_{idx:03d}",
                prompt=(
                    "In Alice's Wonderland numeric_equation methodology, if "
                    f"{supporters} non-query operator group(s) support motif {pairing}|{outmode}, "
                    "how should that motif be treated for absent-op inference?"
                ),
                answer=answer,
                reasoning=(
                    "More supporting operators increase motif confidence. Use the strongest motif as the absent-op prior."
                ),
                source_category="motif_evidence",
            )
            idx += 1


def add_ambiguity_cards(rows: list[MethodRow]) -> None:
    scenarios = [
        (
            "|x-y| plain vs x-y with if-neg mode",
            "declare ambiguity and apply deterministic tie-break",
            "Positive-only examples can hide sign-dependent divergence on the query.",
        ),
        (
            "concat(x,y) vs concat(y,x)",
            "prefer rule that matches same-op examples exactly; if tie remains, use declared base priority",
            "Concatenation order is a frequent ambiguity and must be resolved by explicit policy.",
        ),
        (
            "rev vs plain when examples are palindromic outputs",
            "mark as low-confidence ambiguity",
            "Palindromic outputs can make rev and plain indistinguishable from visible examples.",
        ),
        (
            "two motifs tie on support count",
            "use secondary motif ranking policy",
            "When motif support ties, apply deterministic secondary ranking (e.g., scan order).",
        ),
    ]
    idx = 1
    for title, answer, reasoning in scenarios:
        for variant in range(1, 41):
            add_row(
                rows,
                row_id=f"ne_meth_amb_{idx:03d}",
                prompt=(
                    "In Alice's Wonderland numeric_equation methodology, if the row shows ambiguity case "
                    f"'{title}' (variant {variant}), what should the solver do?"
                ),
                answer=answer,
                reasoning=reasoning,
                source_category="ambiguity_handling",
            )
            idx += 1


def add_confidence_gating_cards(rows: list[MethodRow]) -> None:
    """Teach confidence assignment based on evidence strength."""
    idx = 1
    for same_op_count in (0, 1, 2, 3):
        for fitting_count in (0, 1, 2, 3, 4):
            for motif_support in (0, 1, 2, 3):
                if fitting_count == 0:
                    answer = "none"
                    reasoning = "No fitting candidate means abstain."
                elif same_op_count >= 2 and fitting_count == 1:
                    answer = "high"
                    reasoning = "Multiple same-op examples with a unique fit gives high confidence."
                elif same_op_count >= 1 and fitting_count >= 2:
                    answer = "medium"
                    reasoning = "Same-op evidence exists but ambiguity remains, so use medium confidence."
                elif same_op_count == 0 and motif_support >= 2:
                    answer = "medium"
                    reasoning = "Absent-op with strong motif support can be used, but remains weaker than same-op."
                else:
                    answer = "low"
                    reasoning = "Sparse evidence or tie-heavy fit should stay low confidence."
                add_row(
                    rows,
                    row_id=f"ne_meth_conf_{idx:03d}",
                    prompt=(
                        "In Alice's Wonderland numeric_equation methodology, assign confidence when "
                        f"same-op examples={same_op_count}, fitting candidates={fitting_count}, "
                        f"and motif-supporting operators={motif_support}."
                    ),
                    answer=answer,
                    reasoning=reasoning,
                    source_category="confidence_gating",
                )
                idx += 1


def add_tiebreak_policy_cards(rows: list[MethodRow]) -> None:
    """Teach deterministic tie-break ordering across candidate dimensions."""
    idx = 1
    conflict_types = [
        ("pairing differs, base+outmode same", "prefer motif-consistent pairing"),
        ("base differs, pairing+outmode same", "prefer simpler base priority"),
        ("outmode differs, pairing+base same", "prefer row motif outmode family"),
        ("width differs only", "prefer width inferred from examples"),
        ("all fields differ but all fit", "apply global deterministic ranking"),
    ]
    evidence_levels = [
        ("single same-op example", "keep low confidence"),
        ("two same-op examples", "promote deterministic tie-break result"),
        ("three same-op examples", "trust deterministic ranking strongly"),
    ]
    for conflict, answer in conflict_types:
        for evidence_text, confidence_hint in evidence_levels:
            for variant in range(1, 7):
                add_row(
                    rows,
                    row_id=f"ne_meth_tie_{idx:03d}",
                    prompt=(
                        "In Alice's Wonderland numeric_equation methodology, if "
                        f"{conflict} and this is a {evidence_text} case (variant {variant}), "
                        "which tie-break policy should be applied first?"
                    ),
                    answer=answer,
                    reasoning=(
                        f"Use a deterministic policy: {answer}. Then {confidence_hint}."
                    ),
                    source_category="tiebreak_policy",
                )
                idx += 1


def add_absent_op_workflow_cards(rows: list[MethodRow]) -> None:
    """Teach absent-operator workflow with motif and base-priority steps."""
    idx = 1
    motifs = ["AB_CD|plain", "AB_CD|rev", "BA_DC|plain", "BA_DC|rev"]
    base_rank = ["x + y", "x - y", "y - x", "|x - y|", "x * y", "concat(x, y)", "concat(y, x)"]
    for motif in motifs:
        for used_count in (0, 1, 2, 3, 4):
            for rank_idx, base in enumerate(base_rank[:5], start=1):
                answer = f"try base {base} at priority rank {rank_idx}"
                add_row(
                    rows,
                    row_id=f"ne_meth_absent_{idx:03d}",
                    prompt=(
                        "In Alice's Wonderland numeric_equation methodology, query operator is absent. "
                        f"Top motif is {motif}, and {used_count} bases are already used by visible operators. "
                        f"You are currently evaluating priority rank {rank_idx} ({base}). "
                        "What is the deterministic next step?"
                    ),
                    answer=answer,
                    reasoning=(
                        "Absent-op workflow: lock motif, skip used bases, then test the next base in declared priority order."
                    ),
                    source_category="absent_op_workflow",
                )
                idx += 1


def add_sanity_check_cards(rows: list[MethodRow]) -> None:
    """Teach lightweight post-prediction validation checks."""
    idx = 1
    checks = [
        ("final string keeps required leading zeros", "accept only if width formatting matches examples"),
        ("operator marker appears only in negative branch", "accept only if sign-trigger behavior is consistent"),
        ("query prediction disagrees with same-op pattern", "downgrade confidence and keep ambiguity"),
        ("candidate fails one same-op example", "reject candidate"),
        ("all same-op examples pass but motif conflicts strongly", "keep candidate but mark medium confidence"),
    ]
    for check_text, answer in checks:
        for variant in range(1, 15):
            add_row(
                rows,
                row_id=f"ne_meth_sanity_{idx:03d}",
                prompt=(
                    "In Alice's Wonderland numeric_equation methodology, after generating a candidate answer, "
                    f"the solver observes: {check_text} (variant {variant}). What should it do?"
                ),
                answer=answer,
                reasoning="Post-prediction sanity checks prevent brittle over-trust in a single heuristic.",
                source_category="sanity_checks",
            )
            idx += 1


def build_rows() -> list[MethodRow]:
    rows: list[MethodRow] = []
    add_procedure_cards(rows)
    add_same_op_decision_cards(rows)
    add_motif_cards(rows)
    add_ambiguity_cards(rows)
    add_confidence_gating_cards(rows)
    add_tiebreak_policy_cards(rows)
    add_absent_op_workflow_cards(rows)
    add_sanity_check_cards(rows)
    return rows


def main() -> None:
    args = parse_args()
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
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
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)

    print(f"Wrote {len(rows)} numeric-equation Phase 1B methodology rows to {output_path}")


if __name__ == "__main__":
    main()
