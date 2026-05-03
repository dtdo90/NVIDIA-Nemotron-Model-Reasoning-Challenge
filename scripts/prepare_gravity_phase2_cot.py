#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import infer_category


OBSERVATION_RE = re.compile(
    r"For t = ([0-9]+(?:\.[0-9]+)?)s, distance = ([0-9]+(?:\.[0-9]+)?) m"
)
QUERY_RE = re.compile(
    r"falling distance for t = ([0-9]+(?:\.[0-9]+)?)s",
    re.I,
)

FOUR_PLACES = Decimal("0.0001")


@dataclass(frozen=True)
class GravityRow:
    id: str
    prompt: str
    answer: str
    generated_cot: str
    label: str
    category: str
    source: str
    source_mode: str
    example_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Gravity Phase 2 CoT rows from train.csv. "
            "Rows use the weighted hidden-rate method: k = sum(distance) / sum(t^2)."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Competition train.csv containing id, prompt, and answer.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/gravity_cot.csv",
        help="Destination CSV for Gravity CoT rows.",
    )
    parser.add_argument(
        "--method-resolved-output-csv",
        default="data/trainable/gravity_cot_method_resolved.csv",
        help="Destination CSV for rows solved by the weighted hidden-rate method.",
    )
    return parser.parse_args()


def dec(text: str) -> Decimal:
    return Decimal(text)


def q2(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt2(value: Decimal) -> str:
    return format(q2(value), "f")


def fmt4(value: Decimal) -> str:
    return format(value.quantize(FOUR_PLACES, rounding=ROUND_HALF_UP), "f")


def fmt6(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), "f")


def fmt_decimal(value: Decimal) -> str:
    return format(value, "f")


def parse_gravity_prompt(prompt: str) -> tuple[list[tuple[Decimal, Decimal]], Decimal]:
    observations = [(dec(t), dec(d)) for t, d in OBSERVATION_RE.findall(prompt)]
    query_match = QUERY_RE.search(prompt)
    if not observations or query_match is None:
        raise ValueError("Failed to parse gravity prompt.")
    return observations, dec(query_match.group(1))


def weighted_hidden_rate_solution(
    observations: list[tuple[Decimal, Decimal]],
    query_time: Decimal,
) -> tuple[str, Decimal, Decimal, Decimal, Decimal, Decimal]:
    time_square_sum = sum(time * time for time, _distance in observations)
    distance_sum = sum(distance for _time, distance in observations)
    hidden_rate = (distance_sum / time_square_sum).quantize(
        FOUR_PLACES,
        rounding=ROUND_HALF_UP,
    )
    query_time_square = query_time * query_time
    converted = query_time_square * hidden_rate
    answer = fmt2(converted)
    return answer, time_square_sum, distance_sum, hidden_rate, query_time_square, converted


def wrap_think(reasoning: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>"


def render_trace(
    *,
    observations: list[tuple[Decimal, Decimal]],
    query_time: Decimal,
) -> str:
    (
        answer,
        time_square_sum,
        distance_sum,
        hidden_rate,
        query_time_square,
        converted,
    ) = weighted_hidden_rate_solution(observations, query_time)

    pair_text = "; ".join(
        f"t={fmt_decimal(time)}s -> d={fmt_decimal(distance)}m"
        for time, distance in observations
    )
    square_terms = " + ".join(fmt4(time * time) for time, _ in observations)
    distance_terms = " + ".join(fmt_decimal(distance) for _, distance in observations)
    per_example_rates = "; ".join(
        (
            f"{fmt_decimal(distance)} / {fmt4(time * time)}"
            f" = {fmt4(distance / (time * time))}"
        )
        for time, distance in observations
    )
    lines = [
        (
            "We need to determine the falling distance using d = k*t^2. "
            "Let me find k from the examples."
        ),
        "I will put my final answer inside \\boxed{}.",
        "",
        (
            "The displayed distances are rounded, so I will combine all observations "
            "instead of trusting a single k value."
        ),
        "Use k = sum(distance) / sum(t^2).",
        "",
        f"Observations: {pair_text}",
        "",
        "Individual k checks:",
        f"{per_example_rates}.",
        "These should be close, not necessarily identical.",
        "",
        f"sum(t^2) = {square_terms} = {fmt4(time_square_sum)}.",
        f"sum(distance) = {distance_terms} = {fmt_decimal(distance_sum)}.",
        (
            f"k = {fmt_decimal(distance_sum)} / {fmt4(time_square_sum)} = "
            f"{fmt_decimal(hidden_rate)} to four decimal places."
        ),
        "",
        f"For t = {fmt_decimal(query_time)}:",
        f"{fmt_decimal(query_time)}^2 = {fmt4(query_time_square)}.",
        (
            f"d = k*t^2 = {fmt_decimal(hidden_rate)} * {fmt4(query_time_square)} = {fmt6(converted)}."
        ),
        f"Rounding this distance to two decimals gives {answer}.",
        "",
        "I will now return the answer in \\boxed{}",
        f"The final answer is \\boxed{{{answer}}}",
    ]
    return "\n".join(lines)


def build_rows(train_csv: Path) -> tuple[list[GravityRow], Counter[str]]:
    rows: list[GravityRow] = []
    mode_counts: Counter[str] = Counter()
    with train_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            if infer_category(prompt) != "Gravity":
                continue

            observations, query_time = parse_gravity_prompt(prompt)
            answer, *_ = weighted_hidden_rate_solution(observations, query_time)
            mode = "weighted_hidden_rate"
            mode_counts[mode] += 1
            rows.append(
                GravityRow(
                    id=row["id"],
                    prompt=prompt,
                    answer=answer,
                    generated_cot=render_trace(
                        observations=observations,
                        query_time=query_time,
                    ),
                    label="Gravity",
                    category="Gravity",
                    source="deterministic_gravity_cot",
                    source_mode=mode,
                    example_count=len(observations),
                )
            )
    return rows, mode_counts


def main() -> None:
    args = parse_args()
    train_path = ROOT / args.train_csv
    output_path = ROOT / args.output_csv
    resolved_path = ROOT / args.method_resolved_output_csv
    rows, mode_counts = build_rows(train_path)

    fieldnames = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "source_mode",
        "example_count",
    ]
    for path, selected_rows in ((output_path, rows), (resolved_path, rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row.__dict__ for row in selected_rows)

    print(f"Wrote {len(rows)} Gravity CoT rows to {output_path}")
    print(f"  method_resolved: {len(rows)} -> {resolved_path}")
    for key in sorted(mode_counts):
        print(f"  {key}: {mode_counts[key]}")


if __name__ == "__main__":
    main()
