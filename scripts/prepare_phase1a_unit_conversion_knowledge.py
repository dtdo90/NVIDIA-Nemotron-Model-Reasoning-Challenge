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
            "Generate deterministic Phase 1A knowledge cards for Wonderland unit "
            "conversion. The cards teach scalar-factor inference, rounded examples, "
            "and exact two-decimal output formatting."
        )
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1a_unit_conversion_knowledge.csv",
        help="Destination CSV in SFT schema.",
    )
    return parser.parse_args()


def dec(value: str | int | float) -> Decimal:
    return Decimal(str(value))


def q2(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt2(value: Decimal) -> str:
    return format(q2(value), "f")


def fmt_factor(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def fmt6(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), "f")


def fmt_ratio(value: Decimal) -> str:
    rounded = value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    return fmt_factor(rounded)


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
            label="Unit Conversion Knowledge",
            category="Phase1A Unit Conversion Knowledge",
            source="deterministic_unit_conversion_knowledge",
            source_category=source_category,
        )
    )


def add_rule_cards(rows: list[KnowledgeRow]) -> None:
    cards = [
        (
            "unit_rule_first_try",
            "In Alice's Wonderland unit conversion knowledge, what rule should be tried first for lines like `x m becomes y`?",
            "a single multiplicative scalar factor",
            "Wonderland unit conversion rows use one hidden scalar factor. Start with output = input * factor.",
        ),
        (
            "unit_rule_not_real_world",
            "In Alice's Wonderland unit conversion knowledge, should I use a real-world meters-to-feet conversion table?",
            "No",
            "The unit name is flavor text. Infer the hidden Wonderland scalar from the examples instead.",
        ),
        (
            "unit_rule_ratio",
            "In Alice's Wonderland unit conversion knowledge, how do I estimate the conversion factor from one example?",
            "factor = output / input",
            "A single pair x -> y suggests the scalar factor y / x.",
        ),
        (
            "unit_rule_multiple_examples",
            "In Alice's Wonderland unit conversion knowledge, why should I use several example pairs?",
            "because each displayed output is rounded",
            "Each output is shown to two decimals, so each ratio is approximate. Several pairs make the factor more reliable.",
        ),
        (
            "unit_rule_median",
            "In Alice's Wonderland unit conversion knowledge, what simple robust factor estimate should I use from several ratios?",
            "the median ratio",
            "The median ratio is stable when small two-decimal rounding errors make the individual ratios differ slightly.",
        ),
        (
            "unit_rule_interval",
            "In Alice's Wonderland unit conversion knowledge, what is the robust interval idea for an example x m becomes y?",
            "(y - 0.005) / x <= factor < (y + 0.005) / x",
            "Since y is rounded to two decimals, the true unrounded output lies within half a cent of y.",
        ),
        (
            "unit_rule_apply",
            "In Alice's Wonderland unit conversion knowledge, after I infer the factor, how do I answer the query measurement q?",
            "compute q * factor",
            "The same hidden scalar applies to the query, so multiply the query value by the inferred factor.",
        ),
        (
            "unit_rule_format",
            "In Alice's Wonderland unit conversion knowledge, how should the final numeric answer be formatted?",
            "exactly two decimal places",
            "Unit conversion answers in this task are formatted with exactly two digits after the decimal point.",
        ),
        (
            "unit_rule_trailing_zero",
            "In Alice's Wonderland unit conversion knowledge, should a trailing zero be kept in an answer like 7.50?",
            "Yes",
            "The final answer should keep exactly two decimal places, so 7.5 should be written as 7.50.",
        ),
        (
            "unit_rule_no_addition",
            "In Alice's Wonderland unit conversion knowledge, should I assume an additive offset before testing scalar multiplication?",
            "No",
            "The trusted unit conversion family is a multiplicative scalar-factor task, not an additive-offset task.",
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


def add_ratio_cards(rows: list[KnowledgeRow]) -> None:
    factors = [Decimal("0.50") + Decimal(index) * Decimal("0.02") for index in range(76)]
    factors.extend([Decimal("0.57"), Decimal("0.83"), Decimal("1.17"), Decimal("1.43")])
    factors = sorted(factors)
    examples: list[tuple[str, str]] = []
    for index, factor in enumerate(factors, start=1):
        inp = Decimal(4 + ((index * 7) % 67))
        examples.append((fmt2(inp), fmt2(inp * factor)))

    for index, (inp_s, out_s) in enumerate(examples, start=1):
        inp = dec(inp_s)
        out = dec(out_s)
        factor = out / inp
        answer = fmt_factor(factor)
        add_row(
            rows,
            row_id=f"unit_ratio_{index:03d}",
            prompt=(
                "In Alice's Wonderland unit conversion knowledge, if "
                f"{inp_s} m becomes {out_s}, what scalar factor is suggested?"
            ),
            answer=answer,
            reasoning=f"The factor is output / input = {out_s} / {inp_s} = {answer}.",
            source_category="factor_ratio",
        )


def add_apply_factor_cards(rows: list[KnowledgeRow]) -> None:
    factors = [Decimal("0.50") + Decimal(index) * Decimal("0.03") for index in range(51)]
    examples: list[tuple[str, str]] = []
    for index in range(120):
        factor = factors[(index * 11) % len(factors)]
        query = Decimal(500 + ((index * 137) % 4500)) / Decimal("100")
        examples.append((fmt_factor(factor), fmt2(query)))

    for index, (factor_s, query_s) in enumerate(examples, start=1):
        factor = dec(factor_s)
        query = dec(query_s)
        answer = fmt2(query * factor)
        add_row(
            rows,
            row_id=f"unit_apply_factor_{index:03d}",
            prompt=(
                "In Alice's Wonderland unit conversion knowledge, using scalar "
                f"factor {factor_s}, convert {query_s} m. Answer with exactly two decimals."
            ),
            answer=answer,
            reasoning=(
                f"Apply the scalar factor to the query: {query_s} * {factor_s} = "
                f"{query * factor}. Rounded to exactly two decimals gives {answer}."
            ),
            source_category="apply_factor",
        )


def add_rounding_cards(rows: list[KnowledgeRow]) -> None:
    values = []
    bases = [0, 1, 4, 7, 12, 20, 51, 63, 77, 99]
    suffixes = ["", ".0", ".1", ".004", ".005", ".006", ".234", ".995"]
    for base in bases:
        for suffix in suffixes:
            values.append(f"{base}{suffix}")

    for index, value_s in enumerate(values, start=1):
        value = dec(value_s)
        answer = fmt2(value)
        add_row(
            rows,
            row_id=f"unit_rounding_{index:03d}",
            prompt=(
                "In Alice's Wonderland unit conversion knowledge, format "
                f"{value_s} as a final unit-conversion answer with exactly two decimals."
            ),
            answer=answer,
            reasoning=f"Final unit-conversion answers keep two decimals, so {value_s} becomes {answer}.",
            source_category="two_decimal_formatting",
        )


def add_median_cards(rows: list[KnowledgeRow]) -> None:
    examples: list[tuple[str, str, str, str, str, str, str]] = []
    for index in range(80):
        factor = Decimal("0.503") + Decimal(index) * Decimal("0.0189")
        x1 = Decimal(800 + ((index * 37) % 4200)) / Decimal("100")
        x2 = Decimal(900 + ((index * 53) % 4100)) / Decimal("100")
        x3 = Decimal(1000 + ((index * 71) % 4000)) / Decimal("100")
        query = Decimal(600 + ((index * 73) % 4300)) / Decimal("100")
        examples.append(
            (
                fmt2(x1),
                fmt2(x1 * factor),
                fmt2(x2),
                fmt2(x2 * factor),
                fmt2(x3),
                fmt2(x3 * factor),
                fmt2(query),
            )
        )

    for index, item in enumerate(examples, start=1):
        x1, y1, x2, y2, x3, y3, query_s = item
        pairs = [(dec(x1), dec(y1)), (dec(x2), dec(y2)), (dec(x3), dec(y3))]
        ratios = sorted(y / x for x, y in pairs)
        factor = ratios[1]
        query = dec(query_s)
        answer = fmt2(query * factor)
        ratio_text = ", ".join(fmt_ratio(ratio) for ratio in ratios)
        factor_text = fmt_ratio(factor)
        add_row(
            rows,
            row_id=f"unit_median_{index:03d}",
            prompt=(
                "In Alice's Wonderland unit conversion knowledge, examples give "
                f"{x1} m -> {y1}, {x2} m -> {y2}, and {x3} m -> {y3}. "
                f"Using the median ratio, convert {query_s} m."
            ),
            answer=answer,
            reasoning=(
                f"The three ratios sorted are about {ratio_text}. The median ratio is "
                f"about {factor_text}. Then {query_s} times the median ratio rounds "
                f"to {answer}."
            ),
            source_category="median_ratio_method",
        )


def add_interval_cards(rows: list[KnowledgeRow]) -> None:
    examples: list[tuple[str, str]] = []
    for index in range(40):
        factor = Decimal("0.571") + Decimal(index) * Decimal("0.036")
        inp = Decimal(750 + ((index * 149) % 4250)) / Decimal("100")
        examples.append((fmt2(inp), fmt2(inp * factor)))

    half_cent = Decimal("0.005")
    for index, (inp_s, out_s) in enumerate(examples, start=1):
        inp = dec(inp_s)
        out = dec(out_s)
        lower = (out - half_cent) / inp
        upper = (out + half_cent) / inp
        answer = f"{fmt6(lower)} <= factor < {fmt6(upper)}"
        add_row(
            rows,
            row_id=f"unit_interval_{index:03d}",
            prompt=(
                "In Alice's Wonderland unit conversion knowledge, because "
                f"{inp_s} m becomes {out_s} is rounded to two decimals, what interval "
                "does this imply for the scalar factor? Use six decimal places."
            ),
            answer=answer,
            reasoning=(
                f"The true output is in [{out_s} - 0.005, {out_s} + 0.005). "
                f"Divide by {inp_s}: ({out_s} - 0.005)/{inp_s} <= factor < "
                f"({out_s} + 0.005)/{inp_s}, giving {answer}."
            ),
            source_category="rounded_interval",
        )


def build_rows() -> list[KnowledgeRow]:
    rows: list[KnowledgeRow] = []
    add_rule_cards(rows)
    add_ratio_cards(rows)
    add_apply_factor_cards(rows)
    add_rounding_cards(rows)
    add_median_cards(rows)
    add_interval_cards(rows)
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

    print(f"Wrote {len(rows)} unit conversion Phase 1A knowledge rows to {output_path}")


if __name__ == "__main__":
    main()
