#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ROMAN_VALUES = [
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]

SYMBOL_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
}

ONES = {
    0: "",
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
}

TENS = {
    0: "",
    10: "X",
    20: "XX",
    30: "XXX",
    40: "XL",
    50: "L",
    60: "LX",
    70: "LXX",
    80: "LXXX",
    90: "XC",
    100: "C",
}


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
            "Generate deterministic Phase 1A knowledge cards for the Numeral System "
            "category. The cards teach Roman numeral facts, lookup, decomposition, "
            "and formatting for numbers 1 through 100."
        )
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/phase1_components/phase1a_numeral_knowledge.csv",
        help="Destination CSV in SFT schema.",
    )
    return parser.parse_args()


def roman(n: int) -> str:
    if not 1 <= n <= 100:
        raise ValueError(f"Roman numeral table only supports 1..100, got {n}.")
    remaining = n
    parts: list[str] = []
    for value, symbol in ROMAN_VALUES:
        while remaining >= value:
            parts.append(symbol)
            remaining -= value
    return "".join(parts)


def wrap_think(reasoning: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>"


def split_tens_ones(n: int) -> tuple[int, int]:
    if n == 100:
        return 100, 0
    tens = (n // 10) * 10
    ones = n % 10
    return tens, ones


def decomposition_text(n: int) -> str:
    tens, ones = split_tens_ones(n)
    if n == 100:
        return "100 is C."
    pieces = []
    if tens:
        pieces.append(f"{tens} is {TENS[tens]}")
    if ones:
        pieces.append(f"{ones} is {ONES[ones]}")
    if not pieces:
        pieces.append("0 contributes no Roman symbols")
    return f"{n} = {tens} + {ones}. " + " and ".join(pieces) + "."


def roman_parts_answer(n: int) -> str:
    tens, ones = split_tens_ones(n)
    if n == 100:
        return "C"
    parts = [part for part in (TENS[tens], ONES[ones]) if part]
    return " + ".join(parts) if parts else "empty"


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
            label="Numeral Knowledge",
            category="Phase1A Numeral Knowledge",
            source="deterministic_roman_knowledge",
            source_category=source_category,
        )
    )


def build_rows() -> list[KnowledgeRow]:
    rows: list[KnowledgeRow] = []

    for symbol, value in SYMBOL_VALUES.items():
        add_row(
            rows,
            row_id=f"numeral_symbol_value_{symbol}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, what value does Roman "
                f"symbol {symbol} represent?"
            ),
            answer=str(value),
            reasoning=f"In Roman numerals, {symbol} represents {value}.",
            source_category="symbol_value",
        )
        add_row(
            rows,
            row_id=f"numeral_value_symbol_{value}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, which Roman symbol "
                f"represents {value}?"
            ),
            answer=symbol,
            reasoning=f"The Roman symbol for {value} is {symbol}.",
            source_category="symbol_value_reverse",
        )

    subtractive = [
        (4, "IV", "IIII"),
        (9, "IX", "VIIII"),
        (40, "XL", "XXXX"),
        (90, "XC", "LXXXX"),
    ]
    for value, correct, incorrect in subtractive:
        add_row(
            rows,
            row_id=f"numeral_subtractive_write_{value}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, how is "
                f"{value} written in Roman numerals?"
            ),
            answer=correct,
            reasoning=(
                f"Roman numerals use the subtractive form {correct} for {value}, "
                f"not {incorrect}."
            ),
            source_category="subtractive_form",
        )
        add_row(
            rows,
            row_id=f"numeral_subtractive_value_{correct}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, what value does "
                f"{correct} represent?"
            ),
            answer=str(value),
            reasoning=f"{correct} is the Roman subtractive form for {value}.",
            source_category="subtractive_form_reverse",
        )
        add_row(
            rows,
            row_id=f"numeral_subtractive_choice_{value}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, which form is correct "
                f"for {value}: {incorrect} or {correct}?"
            ),
            answer=correct,
            reasoning=f"The standard Roman numeral for {value} is {correct}.",
            source_category="subtractive_form_choice",
        )

    for value in range(1, 10):
        add_row(
            rows,
            row_id=f"numeral_ones_{value}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, what is the ones-place "
                f"Roman numeral for {value}?"
            ),
            answer=ONES[value],
            reasoning=f"The ones-place Roman table maps {value} to {ONES[value]}.",
            source_category="ones_table",
        )

    for value in range(10, 100, 10):
        add_row(
            rows,
            row_id=f"numeral_tens_{value}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, what is the tens-place "
                f"Roman numeral for {value}?"
            ),
            answer=TENS[value],
            reasoning=f"The tens-place Roman table maps {value} to {TENS[value]}.",
            source_category="tens_table",
        )
    add_row(
        rows,
        row_id="numeral_hundred_100",
        prompt="In Alice's Wonderland numeral knowledge, how is 100 written?",
        answer="C",
        reasoning="The Roman numeral for 100 is C.",
        source_category="hundred_table",
    )

    for value in range(1, 101):
        answer = roman(value)
        add_row(
            rows,
            row_id=f"numeral_lookup_arabic_to_roman_{value:03d}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, write "
                f"{value} in Roman numerals."
            ),
            answer=answer,
            reasoning=f"{decomposition_text(value)} Joining the parts gives {answer}.",
            source_category="arabic_to_roman_lookup",
        )
        add_row(
            rows,
            row_id=f"numeral_lookup_roman_to_arabic_{value:03d}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, what number is represented "
                f"by Roman numeral {answer}?"
            ),
            answer=str(value),
            reasoning=f"The Roman numeral {answer} corresponds to {value}.",
            source_category="roman_to_arabic_lookup",
        )
        add_row(
            rows,
            row_id=f"numeral_parts_{value:03d}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, split "
                f"{value} into Roman numeral parts."
            ),
            answer=roman_parts_answer(value),
            reasoning=f"{decomposition_text(value)} The Roman parts are {roman_parts_answer(value)}.",
            source_category="decomposition",
        )

    formatting_examples = [4, 9, 25, 38, 44, 49, 67, 88, 94, 99, 100]
    for value in formatting_examples:
        answer = roman(value)
        add_row(
            rows,
            row_id=f"numeral_format_uppercase_{value:03d}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, should the answer for "
                f"{value} be lowercase or uppercase?"
            ),
            answer="uppercase",
            reasoning=(
                f"The expected Wonderland numeral output uses uppercase Roman numerals, "
                f"so {value} is written as {answer}."
            ),
            source_category="formatting",
        )
        add_row(
            rows,
            row_id=f"numeral_format_no_spaces_{value:03d}",
            prompt=(
                "In Alice's Wonderland numeral knowledge, write the Roman numeral "
                f"for {value} as one continuous string with no spaces."
            ),
            answer=answer,
            reasoning=f"Roman numeral answers should be one continuous string: {answer}.",
            source_category="formatting",
        )

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

    print(f"Wrote {len(rows)} numeral Phase 1A knowledge rows to {output_path}")


if __name__ == "__main__":
    main()
