#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HARNESS = ROOT / "reference/cursor/transformation_rules/numeric_equation/harness"
for path in (SRC, HARNESS, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extended_dsl import Candidate, apply_pairing  # noqa: E402
from numeric_equation_detailed_cot import (  # noqa: E402
    BASE_TEXT,
    OUT_TEXT,
    PAIRING_TEXT,
    SCAN_PRIORITY,
    compute_base_text,
    parse_rule_label,
)


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
    parser = argparse.ArgumentParser(
        description=(
            "Generate compact Phase 1A knowledge cards for numeric_equation using "
            "only the rule DSL that appears in accepted labelled numeric rows."
        )
    )
    parser.add_argument(
        "--source-csv",
        default="data/trainable/numeric_equation_labelled_cot.csv",
        help="Accepted labelled numeric-equation CoT rows.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1a_numeric_equation_knowledge.csv",
        help="Destination CSV in SFT schema.",
    )
    parser.add_argument("--target-rows", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
            label="Numeric Equation Knowledge",
            category="Phase1A Numeric Equation Knowledge",
            source="accepted_rule_numeric_equation_knowledge",
            source_category=source_category,
        )
    )


def load_accepted_candidates(path: Path) -> list[Candidate]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"rule_label", "id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        return [parse_rule_label(row["rule_label"]) for row in reader]


def weighted_choice(rng: random.Random, counts: Counter[str]) -> str:
    total = sum(counts.values())
    pick = rng.randrange(total)
    running = 0
    for item, count in counts.items():
        running += count
        if pick < running:
            return item
    return next(iter(counts))


def weighted_combo_choice(rng: random.Random, counts: Counter[tuple[str, str, str]]) -> tuple[str, str, str]:
    total = sum(counts.values())
    pick = rng.randrange(total)
    running = 0
    for item, count in counts.items():
        running += count
        if pick < running:
            return item
    return next(iter(counts))


def two_digit(rng: random.Random) -> str:
    return f"{rng.randrange(100):02d}"


def render_base_value(candidate: Candidate, left: str, right: str) -> tuple[int, int, int | None]:
    x, y = apply_pairing(left, right, candidate.pairing)
    return x, y, candidate.base_rule.apply(x, y)


def random_candidate_for_base(
    rng: random.Random,
    base_name: str,
    pairing_counts: Counter[str],
    outmode_counts: Counter[str],
) -> Candidate:
    return Candidate(
        pairing=weighted_choice(rng, pairing_counts),
        base_name=base_name,
        outmode_name=weighted_choice(rng, outmode_counts),
        width=None,
    )


def valid_combo_example(rng: random.Random, candidate: Candidate, operator: str) -> tuple[str, str, str] | None:
    for _ in range(300):
        left = two_digit(rng)
        right = two_digit(rng)
        answer = candidate.predict(left, right, operator)
        if answer is None:
            continue
        if len(answer) > 10:
            continue
        return left, right, answer
    return None


def add_inventory_cards(
    rows: list[KnowledgeRow],
    *,
    rng: random.Random,
    pairing_counts: Counter[str],
    base_counts: Counter[str],
    outmode_counts: Counter[str],
    combo_counts: Counter[tuple[str, str, str]],
) -> None:
    add_row(
        rows,
        row_id="ne_accepteddsl_inventory_pairings",
        prompt="In numeric_equation Phase 1A knowledge, which pairings are used by the accepted labelled rows?",
        answer=", ".join(pairing_counts),
        reasoning=(
            "Use the accepted labelled rows as the DSL boundary. The observed pairings are "
            + ", ".join(f"{name} ({count})" for name, count in pairing_counts.most_common())
            + "."
        ),
        source_category="accepted_dsl_inventory",
    )
    add_row(
        rows,
        row_id="ne_accepteddsl_inventory_base_rules",
        prompt="In numeric_equation Phase 1A knowledge, which base rules should be available?",
        answer=", ".join(base_counts),
        reasoning=(
            "The base-rule inventory is exactly the set used by accepted numeric-equation traces: "
            + ", ".join(f"{name} ({count})" for name, count in base_counts.most_common())
            + "."
        ),
        source_category="accepted_dsl_inventory",
    )
    add_row(
        rows,
        row_id="ne_accepteddsl_inventory_output_modes",
        prompt="In numeric_equation Phase 1A knowledge, which output modes should be available?",
        answer=", ".join(outmode_counts),
        reasoning=(
            "The output-mode inventory is exactly the set used by accepted numeric-equation traces: "
            + ", ".join(f"{name} ({count})" for name, count in outmode_counts.most_common())
            + "."
        ),
        source_category="accepted_dsl_inventory",
    )
    top_combos = combo_counts.most_common(10)
    add_row(
        rows,
        row_id="ne_accepteddsl_top_priority_combos",
        prompt="In numeric_equation Phase 1A knowledge, what are the highest-priority accepted rule combinations?",
        answer="; ".join(f"{p}|{b}|{o}" for (p, b, o), _ in top_combos),
        reasoning=(
            "Scan common accepted combinations first. The top accepted combinations are "
            + "; ".join(f"{p}|{b}|{o} ({count})" for (p, b, o), count in top_combos)
            + "."
        ),
        source_category="priority_inventory",
    )

    accepted_bases = set(base_counts)
    rejected_bases = (
        "x * y + x",
        "x * y + y",
        "x + y^2",
        "(x + y)^2",
        "(x - y)^2",
        "x^2 + y^2",
        "x^2 - y^2",
        "y^2 - x^2",
        "gcd(x, y)",
        "lcm(x, y)",
        "y // x",
        "digit_sum(x) + digit_sum(y)",
        "digit_sum(x) * digit_sum(y)",
        "|digit_sum(x) - digit_sum(y)|",
    )
    accepted_outmodes = set(outmode_counts)
    rejected_outmodes = (
        "abs",
        "rev_or_op_prefix_if_neg",
    )

    base_tiers: dict[str, str] = {}
    for base_name, count in base_counts.items():
        if count >= 50:
            base_tiers[base_name] = "core"
        elif count >= 10:
            base_tiers[base_name] = "common support"
        elif count >= 2:
            base_tiers[base_name] = "rare support"
        else:
            base_tiers[base_name] = "tiny appendix"

    outmode_tiers: dict[str, str] = {}
    for outmode_name, count in outmode_counts.items():
        if count >= 30:
            outmode_tiers[outmode_name] = "core"
        elif count >= 5:
            outmode_tiers[outmode_name] = "support"
        else:
            outmode_tiers[outmode_name] = "tiny appendix"

    idx = 1
    for base_name in sorted(accepted_bases):
        add_row(
            rows,
            row_id=f"ne_inventory_base_accept_{idx:03d}",
            prompt=f"Should accepted numeric_equation Phase 1A include base rule {base_name}?",
            answer="Yes",
            reasoning=(
                f"Yes. {base_name} appears in accepted labelled numeric-equation rows "
                f"{base_counts[base_name]} time(s), so it belongs to the {base_tiers[base_name]} tier."
            ),
            source_category="dsl_boundary",
        )
        idx += 1

    for base_name in rejected_bases:
        add_row(
            rows,
            row_id=f"ne_inventory_base_reject_{idx:03d}",
            prompt=f"Should accepted numeric_equation Phase 1A include base rule {base_name} as core knowledge?",
            answer="No",
            reasoning=(
                f"No. {base_name} is not used by the accepted 696-row numeric-equation DSL, "
                "so it should not be part of the core Phase 1A curriculum."
            ),
            source_category="dsl_boundary",
        )
        idx += 1

    for outmode_name in sorted(accepted_outmodes):
        add_row(
            rows,
            row_id=f"ne_inventory_outmode_accept_{idx:03d}",
            prompt=f"Should accepted numeric_equation Phase 1A include output mode {outmode_name}?",
            answer="Yes",
            reasoning=(
                f"Yes. {outmode_name} appears in accepted labelled numeric-equation rows "
                f"{outmode_counts[outmode_name]} time(s), so it belongs to the {outmode_tiers[outmode_name]} tier."
            ),
            source_category="dsl_boundary",
        )
        idx += 1

    for outmode_name in rejected_outmodes:
        add_row(
            rows,
            row_id=f"ne_inventory_outmode_reject_{idx:03d}",
            prompt=f"Should accepted numeric_equation Phase 1A include output mode {outmode_name} as core knowledge?",
            answer="No",
            reasoning=(
                f"No. {outmode_name} is not used by the accepted 696-row numeric-equation DSL, "
                "so it should not be part of the core Phase 1A curriculum."
            ),
            source_category="dsl_boundary",
        )
        idx += 1

    ordered_combos = combo_counts.most_common()
    combo_rank = {combo: rank for rank, (combo, _) in enumerate(ordered_combos, start=1)}
    top_pool = [combo for combo, _ in ordered_combos[:25]]
    tail_pool = [combo for combo, _ in ordered_combos[25:]] or top_pool
    for _ in range(80):
        left = rng.choice(top_pool)
        right = rng.choice(tail_pool)
        if left == right:
            continue
        if combo_rank[left] > combo_rank[right]:
            left, right = right, left
        left_label = "|".join(left)
        right_label = "|".join(right)
        add_row(
            rows,
            row_id=f"ne_inventory_priority_{idx:03d}",
            prompt=(
                "For accepted numeric_equation scan priority, which combo should be tried first: "
                f"{left_label} or {right_label}?"
            ),
            answer=left_label,
            reasoning=(
                f"{left_label} has accepted-rank {combo_rank[left]}, while {right_label} has "
                f"accepted-rank {combo_rank[right]}. Try {left_label} first."
            ),
            source_category="priority_inventory",
        )
        idx += 1

    pairing_order = [pairing for pairing, _ in pairing_counts.most_common()]
    for _ in range(20):
        pairing = rng.choice(pairing_order)
        add_row(
            rows,
            row_id=f"ne_inventory_pairing_principle_{idx:03d}",
            prompt=f"In accepted numeric_equation solving, what does pairing {pairing} mean and why is it available?",
            answer=PAIRING_TEXT[pairing],
            reasoning=(
                f"{pairing} is available because it appears in accepted labelled rows "
                f"{pairing_counts[pairing]} time(s). It means: {PAIRING_TEXT[pairing]}."
            ),
            source_category="dsl_boundary",
        )
        idx += 1


def add_pairing_cards(
    rows: list[KnowledgeRow],
    *,
    rng: random.Random,
    pairing_counts: Counter[str],
    target: int,
) -> None:
    start = len(rows)
    for pairing in pairing_counts:
        for _ in range(8):
            left = two_digit(rng)
            right = two_digit(rng)
            x, y = apply_pairing(left, right, pairing)
            add_row(
                rows,
                row_id=f"ne_pairing_{len(rows)-start+1:04d}",
                prompt=f"Apply accepted numeric_equation pairing {pairing} to left={left}, right={right}. Return x,y.",
                answer=f"{x},{y}",
                reasoning=f"{PAIRING_TEXT[pairing]}: left={left}, right={right} gives x={x}, y={y}.",
                source_category="pairing_transform",
            )
    while len(rows) - start < target:
        pairing = weighted_choice(rng, pairing_counts)
        left = two_digit(rng)
        right = two_digit(rng)
        x, y = apply_pairing(left, right, pairing)
        add_row(
            rows,
            row_id=f"ne_pairing_{len(rows)-start+1:04d}",
            prompt=f"Apply accepted numeric_equation pairing {pairing} to left={left}, right={right}. Return x,y.",
            answer=f"{x},{y}",
            reasoning=f"{PAIRING_TEXT[pairing]}: left={left}, right={right} gives x={x}, y={y}.",
            source_category="pairing_transform",
        )


def add_base_cards(
    rows: list[KnowledgeRow],
    *,
    rng: random.Random,
    pairing_counts: Counter[str],
    base_counts: Counter[str],
    outmode_counts: Counter[str],
    target: int,
) -> None:
    start = len(rows)
    for base_name in base_counts:
        for _ in range(6):
            candidate = random_candidate_for_base(rng, base_name, pairing_counts, outmode_counts)
            example = valid_combo_example(rng, candidate, "+")
            if example is None:
                continue
            left, right, _ = example
            x, y, value = render_base_value(candidate, left, right)
            add_row(
                rows,
                row_id=f"ne_base_{len(rows)-start+1:04d}",
                prompt=(
                    f"Using pairing {candidate.pairing}, compute accepted base rule {base_name} "
                    f"for left={left}, right={right}."
                ),
                answer=str(value),
                reasoning=(
                    f"{PAIRING_TEXT[candidate.pairing]} gives x={x}, y={y}. "
                    f"{compute_base_text(base_name, x, y, value)}."
                ),
                source_category="base_rule_semantics",
            )
    while len(rows) - start < target:
        base_name = weighted_choice(rng, base_counts)
        candidate = random_candidate_for_base(rng, base_name, pairing_counts, outmode_counts)
        example = valid_combo_example(rng, candidate, "+")
        if example is None:
            continue
        left, right, _ = example
        x, y, value = render_base_value(candidate, left, right)
        add_row(
            rows,
            row_id=f"ne_base_{len(rows)-start+1:04d}",
            prompt=(
                f"Using pairing {candidate.pairing}, compute accepted base rule {base_name} "
                f"for left={left}, right={right}."
            ),
            answer=str(value),
            reasoning=(
                f"{PAIRING_TEXT[candidate.pairing]} gives x={x}, y={y}. "
                f"{compute_base_text(base_name, x, y, value)}."
            ),
            source_category="base_rule_semantics",
        )


def add_output_cards(
    rows: list[KnowledgeRow],
    *,
    rng: random.Random,
    outmode_counts: Counter[str],
    target: int,
) -> None:
    start = len(rows)
    operators = tuple("+-*/?@#$%&!<>[]{}|\\:`'\"^")
    values = (-87, -42, -9, -3, 0, 4, 17, 68, 105, 731)
    for outmode_name in outmode_counts:
        mode = Candidate("AB_CD", "x + y", outmode_name, None).output_mode
        for _ in range(6):
            value = rng.choice(values)
            operator = rng.choice(operators)
            answer = mode.apply(value, operator, None)
            add_row(
                rows,
                row_id=f"ne_outmode_{len(rows)-start+1:04d}",
                prompt=(
                    f"Render value {value} with accepted output mode {outmode_name} "
                    f"and operator '{operator}'."
                ),
                answer=answer,
                reasoning=f"{OUT_TEXT[outmode_name]}: value {value} with operator {operator} renders as {answer}.",
                source_category="output_mode_semantics",
            )
    while len(rows) - start < target:
        outmode_name = weighted_choice(rng, outmode_counts)
        mode = Candidate("AB_CD", "x + y", outmode_name, None).output_mode
        value = rng.choice(values)
        operator = rng.choice(operators)
        answer = mode.apply(value, operator, None)
        add_row(
            rows,
            row_id=f"ne_outmode_{len(rows)-start+1:04d}",
            prompt=(
                f"Render value {value} with accepted output mode {outmode_name} "
                f"and operator '{operator}'."
            ),
            answer=answer,
            reasoning=f"{OUT_TEXT[outmode_name]}: value {value} with operator {operator} renders as {answer}.",
            source_category="output_mode_semantics",
        )


def add_combo_cards(
    rows: list[KnowledgeRow],
    *,
    rng: random.Random,
    combo_counts: Counter[tuple[str, str, str]],
    target: int,
) -> None:
    start = len(rows)
    for pairing, base_name, outmode_name in combo_counts:
        candidate = Candidate(pairing, base_name, outmode_name, None)
        example = valid_combo_example(rng, candidate, rng.choice(tuple("+-*/?@#$%&!<>[]{}|\\:`'\"^")))
        if example is None:
            continue
        left, right, answer = example
        operator = rng.choice(tuple("+-*/?@#$%&!<>[]{}|\\:`'\"^"))
        answer = candidate.predict(left, right, operator)
        if answer is None:
            continue
        x, y, value = render_base_value(candidate, left, right)
        add_row(
            rows,
            row_id=f"ne_combo_{len(rows)-start+1:04d}",
            prompt=(
                f"Apply accepted combo {pairing}|{base_name}|{outmode_name} "
                f"to expression {left}{operator}{right}."
            ),
            answer=answer,
            reasoning=(
                f"{PAIRING_TEXT[pairing]} gives x={x}, y={y}. "
                f"{compute_base_text(base_name, x, y, value)}. "
                f"Then {OUT_TEXT[outmode_name]} gives {answer}."
            ),
            source_category="combo_application",
        )
    while len(rows) - start < target:
        pairing, base_name, outmode_name = weighted_combo_choice(rng, combo_counts)
        candidate = Candidate(pairing, base_name, outmode_name, None)
        operator = rng.choice(tuple("+-*/?@#$%&!<>[]{}|\\:`'\"^"))
        example = valid_combo_example(rng, candidate, operator)
        if example is None:
            continue
        left, right, answer = example
        x, y, value = render_base_value(candidate, left, right)
        add_row(
            rows,
            row_id=f"ne_combo_{len(rows)-start+1:04d}",
            prompt=(
                f"Apply accepted combo {pairing}|{base_name}|{outmode_name} "
                f"to expression {left}{operator}{right}."
            ),
            answer=answer,
            reasoning=(
                f"{PAIRING_TEXT[pairing]} gives x={x}, y={y}. "
                f"{compute_base_text(base_name, x, y, value)}. "
                f"Then {OUT_TEXT[outmode_name]} gives {answer}."
            ),
            source_category="combo_application",
        )


def add_priority_cards(
    rows: list[KnowledgeRow],
    *,
    rng: random.Random,
    combo_counts: Counter[tuple[str, str, str]],
    target: int,
) -> None:
    start = len(rows)
    ordered = [combo for combo, _ in combo_counts.most_common()]
    priority_index = {combo: index for index, combo in enumerate(ordered)}
    while len(rows) - start < target:
        left = rng.choice(ordered[: min(25, len(ordered))])
        right = rng.choice(ordered[: min(25, len(ordered))])
        if left == right:
            continue
        winner = left if priority_index[left] < priority_index[right] else right
        loser = right if winner == left else left
        winner_label = "|".join(winner)
        loser_label = "|".join(loser)
        add_row(
            rows,
            row_id=f"ne_priority_{len(rows)-start+1:04d}",
            prompt=(
                "In accepted numeric_equation scan priority, which combo should be tried first: "
                f"{winner_label} or {loser_label}?"
            ),
            answer=winner_label,
            reasoning=(
                f"{winner_label} appears more often in accepted traces than {loser_label}, "
                "so it should be scanned first when both are plausible."
            ),
            source_category="scan_priority",
        )


def build_rows(
    *,
    rng: random.Random,
    candidates: list[Candidate],
    target_rows: int,
) -> list[KnowledgeRow]:
    pairing_counts = Counter(candidate.pairing for candidate in candidates)
    base_counts = Counter(candidate.base_name for candidate in candidates)
    outmode_counts = Counter(candidate.outmode_name for candidate in candidates)
    combo_counts = Counter(
        (candidate.pairing, candidate.base_name, candidate.outmode_name)
        for candidate in candidates
    )

    rows: list[KnowledgeRow] = []
    add_inventory_cards(
        rows,
        rng=rng,
        pairing_counts=pairing_counts,
        base_counts=base_counts,
        outmode_counts=outmode_counts,
        combo_counts=combo_counts,
    )
    add_pairing_cards(rows, rng=rng, pairing_counts=pairing_counts, target=250)
    add_base_cards(
        rows,
        rng=rng,
        pairing_counts=pairing_counts,
        base_counts=base_counts,
        outmode_counts=outmode_counts,
        target=650,
    )
    add_output_cards(rows, rng=rng, outmode_counts=outmode_counts, target=550)
    add_combo_cards(rows, rng=rng, combo_counts=combo_counts, target=600)
    remaining = max(0, target_rows - len(rows))
    add_priority_cards(rows, rng=rng, combo_counts=combo_counts, target=remaining)
    return rows[:target_rows]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_path = ROOT / args.source_csv
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = load_accepted_candidates(source_path)
    rows = build_rows(rng=rng, candidates=candidates, target_rows=args.target_rows)
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

    print(f"Wrote {len(rows)} accepted-rule numeric-equation Phase 1A rows to {output_path}")
    print("Source category counts:")
    for source_category, count in Counter(row.source_category for row in rows).most_common():
        print(f"  {source_category}: {count}")


if __name__ == "__main__":
    main()
