#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
            "ARCHIVED: generate deterministic Phase 1A knowledge cards for Wonderland gravity. "
            "Current training should not include Gravity Phase 1A; use Phase 2 traces instead. "
            "The cards teach the rate-first d = k*t^2 method, time squaring, "
            "weighted-rate inference, consistency checks, and final two-decimal formatting."
        )
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1a_gravity_knowledge.csv",
        help="Destination CSV in SFT schema.",
    )
    return parser.parse_args()


def dec(value: str | int | float) -> Decimal:
    return Decimal(str(value))


def quantize(value: Decimal, places: str) -> Decimal:
    return value.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def fmt2(value: Decimal) -> str:
    return format(quantize(value, "0.01"), "f")


def fmt4(value: Decimal) -> str:
    return format(quantize(value, "0.0001"), "f")


def fmt_rate(value: Decimal) -> str:
    rounded = quantize(value, "0.0001")
    text = format(rounded.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


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
            label="Gravity Knowledge",
            category="Phase1A Gravity Knowledge",
            source="deterministic_gravity_knowledge",
            source_category=source_category,
        )
    )


def add_rule_cards(rows: list[KnowledgeRow]) -> None:
    cards = [
        (
            "gravity_rule_rewrite",
            "In Alice's Wonderland gravity knowledge, how should I simplify d = 0.5*g*t^2 for solving?",
            "use d = k*t^2",
            "The winner-style method avoids solving for g. Define k as the hidden rate, so the formula becomes d = k*t^2.",
        ),
        (
            "gravity_rule_no_explicit_g",
            "In Alice's Wonderland gravity knowledge, do I need to compute g explicitly?",
            "No",
            "It is enough to infer the hidden rate k directly from the examples. Computing g adds extra arithmetic without helping the answer.",
        ),
        (
            "gravity_rule_k_formula",
            "In Alice's Wonderland gravity knowledge, what rate should be inferred from several observations?",
            "k = sum(distance) / sum(t^2)",
            "From d = k*t^2, combine the rounded examples with k = sum(distance) / sum(t^2).",
        ),
        (
            "gravity_rule_target",
            "In Alice's Wonderland gravity knowledge, after inferring k, how do I answer a query time tq?",
            "distance = k * tq^2",
            "Use the same hidden rate on the query: square the target time, then multiply by k.",
        ),
        (
            "gravity_rule_square_first",
            "In Alice's Wonderland gravity knowledge, should the target time be multiplied by k before or after squaring?",
            "square the time first",
            "The formula is d = k*t^2, so compute t^2 first and then multiply by k.",
        ),
        (
            "gravity_rule_not_linear",
            "In Alice's Wonderland gravity knowledge, is the falling distance linear in t?",
            "No",
            "Distance is proportional to t^2, not t, so doubling time makes distance about four times larger.",
        ),
        (
            "gravity_rule_multiple_examples",
            "In Alice's Wonderland gravity knowledge, why combine several observations?",
            "because displayed distances are rounded",
            "Each displayed distance is rounded, so individual k estimates differ slightly. Combining all observations makes the rate more reliable.",
        ),
        (
            "gravity_rule_weighted",
            "In Alice's Wonderland gravity knowledge, what simple robust k estimate should I use from several observations?",
            "the weighted aggregate k",
            "Use k = sum(distance) / sum(t^2). This gives more weight to observations with larger squared time and is stable under rounding.",
        ),
        (
            "gravity_rule_verification",
            "In Alice's Wonderland gravity knowledge, how can I verify the gravity rate from two examples?",
            "check that their k values are close",
            "Compute k = d/t^2 for two examples. If the values are close, the rate-first gravity rule is consistent.",
        ),
        (
            "gravity_rule_not_earth",
            "In Alice's Wonderland gravity knowledge, should I use Earth's gravitational constant?",
            "No",
            "The Wonderland gravitational constant changes per problem, so infer the hidden rate from the examples.",
        ),
        (
            "gravity_rule_prompt_flavor",
            "In Alice's Wonderland gravity knowledge, what part of the gravity prompt matters most?",
            "the observations and d = k*t^2",
            "The Wonderland wrapper is flavor. Use the observed times and distances to infer the hidden rate.",
        ),
        (
            "gravity_rule_format",
            "In Alice's Wonderland gravity knowledge, how should the final distance answer be formatted?",
            "exactly two decimal places",
            "For Phase 1, teach the model to keep final gravity distances as two-decimal numeric answers.",
        ),
        (
            "gravity_rule_trailing_zero",
            "In Alice's Wonderland gravity knowledge, should a trailing zero be kept in an answer like 45.00?",
            "Yes",
            "The final answer should preserve exactly two decimals, so 45 should be written as 45.00.",
        ),
        (
            "gravity_rule_rate_first",
            "In Alice's Wonderland gravity knowledge, why use the rate-first method?",
            "it reduces arithmetic steps",
            "Using k directly avoids computing g and then multiplying by 0.5 again. Fewer steps means fewer arithmetic errors.",
        ),
    ]
    for row_id, prompt, answer, reasoning in cards:
        add_row(
            rows,
            row_id=row_id,
            prompt=prompt,
            answer=answer,
            reasoning=reasoning,
            source_category="rule_card",
        )


def add_time_square_cards(rows: list[KnowledgeRow]) -> None:
    for index in range(30):
        time = Decimal(100 + ((index * 17) % 401)) / Decimal("100")
        answer = fmt4(time * time)
        time_s = fmt2(time)
        add_row(
            rows,
            row_id=f"gravity_time_square_{index + 1:03d}",
            prompt=(
                "In Alice's Wonderland gravity knowledge, compute t^2 for "
                f"t = {time_s}s. Use four decimal places."
            ),
            answer=answer,
            reasoning=f"Square the time before applying the rate: {time_s}^2 = {answer}.",
            source_category="time_square",
        )


def add_rate_from_observation_cards(rows: list[KnowledgeRow]) -> None:
    for index in range(40):
        time = Decimal(100 + ((index * 23) % 401)) / Decimal("100")
        true_rate = Decimal("2.50") + Decimal((index * 37) % 730) / Decimal("100")
        distance = dec(fmt2(true_rate * time * time))
        inferred = distance / (time * time)
        time_s = fmt2(time)
        distance_s = fmt2(distance)
        answer = fmt4(inferred)
        add_row(
            rows,
            row_id=f"gravity_rate_{index + 1:03d}",
            prompt=(
                "In Alice's Wonderland gravity knowledge, if t = "
                f"{time_s}s and distance = {distance_s} m, estimate k = d/t^2. "
                "Round k to four decimals."
            ),
            answer=answer,
            reasoning=(
                f"Use k = d/t^2. Here t^2 = {time_s}^2 = {fmt4(time * time)}, "
                f"so k = {distance_s}/{fmt4(time * time)} = {answer}."
            ),
            source_category="rate_from_observation",
        )


def add_apply_rate_cards(rows: list[KnowledgeRow]) -> None:
    for index in range(50):
        rate = Decimal("2.50") + Decimal((index * 29) % 731) / Decimal("100")
        time = Decimal(100 + ((index * 31) % 401)) / Decimal("100")
        time_s = fmt2(time)
        rate_s = fmt_rate(rate)
        answer = fmt2(rate * time * time)
        add_row(
            rows,
            row_id=f"gravity_apply_rate_{index + 1:03d}",
            prompt=(
                "In Alice's Wonderland gravity knowledge, using k = "
                f"{rate_s}, determine the falling distance for t = {time_s}s. "
                "Answer with exactly two decimals."
            ),
            answer=answer,
            reasoning=(
                f"Use d = k*t^2. First {time_s}^2 = {fmt4(time * time)}. "
                f"Then d = {rate_s} * {fmt4(time * time)} = {rate * time * time}, "
                f"which rounds to {answer}."
            ),
            source_category="apply_rate",
        )


def add_weighted_rate_cards(rows: list[KnowledgeRow]) -> None:
    for index in range(25):
        true_rate = Decimal("2.503") + Decimal(index) * Decimal("0.091")
        t1 = Decimal(110 + ((index * 19) % 390)) / Decimal("100")
        t2 = Decimal(125 + ((index * 29) % 375)) / Decimal("100")
        t3 = Decimal(140 + ((index * 41) % 360)) / Decimal("100")
        query = Decimal(100 + ((index * 53) % 401)) / Decimal("100")

        pairs = [
            (t1, dec(fmt2(true_rate * t1 * t1))),
            (t2, dec(fmt2(true_rate * t2 * t2))),
            (t3, dec(fmt2(true_rate * t3 * t3))),
        ]
        time_square_sum = sum(time * time for time, _distance in pairs)
        distance_sum = sum(distance for _time, distance in pairs)
        weighted_rate = distance_sum / time_square_sum
        query_s = fmt2(query)
        answer = fmt2(weighted_rate * query * query)
        time_square_text = " + ".join(fmt4(time * time) for time, _distance in pairs)
        distance_text = " + ".join(fmt2(distance) for _time, distance in pairs)
        prompt_pairs = ", ".join(
            f"t={fmt2(time)}s -> {fmt2(distance)}m" for time, distance in pairs
        )
        add_row(
            rows,
            row_id=f"gravity_weighted_{index + 1:03d}",
            prompt=(
                "In Alice's Wonderland gravity knowledge, observations give "
                f"{prompt_pairs}. Using the weighted aggregate k, determine the falling distance "
                f"for t = {query_s}s."
            ),
            answer=answer,
            reasoning=(
                f"Compute k = sum(distance)/sum(t^2). Here sum(distance) = "
                f"{distance_text} = {fmt2(distance_sum)} and sum(t^2) = "
                f"{time_square_text} = {fmt4(time_square_sum)}, so "
                f"k = {fmt2(distance_sum)}/{fmt4(time_square_sum)} = {fmt_rate(weighted_rate)}. "
                f"For the query, {query_s}^2 = {fmt4(query * query)}, so "
                f"distance = weighted k * {fmt4(query * query)}, which rounds to {answer}."
            ),
            source_category="weighted_rate_method",
        )


def add_rate_consistency_cards(rows: list[KnowledgeRow]) -> None:
    for index in range(20):
        true_rate = Decimal("2.70") + Decimal(index) * Decimal("0.18")
        t1 = Decimal(100 + ((index * 37) % 401)) / Decimal("100")
        t2 = Decimal(115 + ((index * 43) % 386)) / Decimal("100")
        d1 = dec(fmt2(true_rate * t1 * t1))
        d2 = dec(fmt2(true_rate * t2 * t2))
        k1 = d1 / (t1 * t1)
        k2 = d2 / (t2 * t2)
        diff = abs(k1 - k2)
        answer = "consistent" if diff < Decimal("0.05") else "not consistent"
        add_row(
            rows,
            row_id=f"gravity_consistency_{index + 1:03d}",
            prompt=(
                "In Alice's Wonderland gravity knowledge, check whether two observations "
                f"are consistent with one gravity rate: t={fmt2(t1)}s -> {fmt2(d1)}m "
                f"and t={fmt2(t2)}s -> {fmt2(d2)}m."
            ),
            answer=answer,
            reasoning=(
                f"Compute k1 = {fmt2(d1)}/{fmt4(t1 * t1)} = {fmt4(k1)} and "
                f"k2 = {fmt2(d2)}/{fmt4(t2 * t2)} = {fmt4(k2)}. "
                f"The difference is {fmt4(diff)}, which is small, so the observations are {answer}."
            ),
            source_category="rate_consistency_check",
        )


def add_rounding_cards(rows: list[KnowledgeRow]) -> None:
    bases = [0, 1, 4, 7, 12, 20, 45, 63, 77, 140]
    suffixes = [".0", ".1", ".004", ".005", ".006", ".234", ".995", ".999"]
    values = [f"{base}{suffix}" for base in bases for suffix in suffixes][:20]
    for index, value_s in enumerate(values, start=1):
        value = dec(value_s)
        answer = fmt2(value)
        add_row(
            rows,
            row_id=f"gravity_rounding_{index:03d}",
            prompt=(
                "In Alice's Wonderland gravity knowledge, format "
                f"{value_s} as a final gravity distance answer with exactly two decimals."
            ),
            answer=answer,
            reasoning=f"Final gravity answers should keep two decimals, so {value_s} becomes {answer}.",
            source_category="two_decimal_formatting",
        )


def build_rows() -> list[KnowledgeRow]:
    rows: list[KnowledgeRow] = []
    add_rule_cards(rows)
    add_time_square_cards(rows)
    add_rate_from_observation_cards(rows)
    add_apply_rate_cards(rows)
    add_weighted_rate_cards(rows)
    add_rate_consistency_cards(rows)
    add_rounding_cards(rows)
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

    print(
        "Archived generator: Gravity Phase 1A is not part of the active training plan. "
        "Use scripts/prepare_gravity_phase2_cot.py for active Gravity data."
    )
    print(f"Wrote {len(rows)} gravity Phase 1A knowledge rows to {output_path}")


if __name__ == "__main__":
    main()
