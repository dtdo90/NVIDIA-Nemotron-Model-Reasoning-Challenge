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


PAIR_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*m becomes\s*([0-9]+(?:\.[0-9]+)?)")
QUERY_RE = re.compile(r"measurement:\s*([0-9]+(?:\.[0-9]+)?)\s*m", re.I)


@dataclass(frozen=True)
class UnitRow:
    id: str
    prompt: str
    answer: str
    generated_cot: str
    label: str
    category: str
    source: str
    source_mode: str
    pair_count: int


@dataclass(frozen=True)
class PolicyDecision:
    prediction: str | None
    mode: str
    overlap_margin: Decimal | None


FOUR_PLACES = Decimal("0.0001")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Unit Conversion Phase 2 CoT rows from train.csv. "
            "Rows are solved by a simple weighted scalar-factor method that is accepted "
            "by the competition numeric verifier."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Competition train.csv containing id, prompt, and answer.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/unit_conversion_cot.csv",
        help="Destination CSV for Unit Conversion CoT rows.",
    )
    parser.add_argument(
        "--method-clean-output-csv",
        default="data/trainable/unit_conversion_cot_method_clean.csv",
        help=(
            "Destination CSV for weighted scalar-factor rows. Kept for compatibility "
            "with earlier pipeline names."
        ),
    )
    parser.add_argument(
        "--method-resolved-output-csv",
        default="data/trainable/unit_conversion_cot_method_resolved.csv",
        help=(
            "Destination CSV for rows solved by the weighted scalar-factor method."
        ),
    )
    parser.add_argument(
        "--answer-aligned-output-csv",
        default="data/trainable/unit_conversion_cot_answer_aligned.csv",
        help=(
            "Destination CSV for rows where examples permit multiple rounded outputs. "
            "Some are resolved by a prompt-only tie-breaker; the rest are unresolved."
        ),
    )
    parser.add_argument(
        "--unresolved-output-csv",
        default="data/trainable/unit_conversion_cot_unresolved_ambiguous.csv",
        help=(
            "Destination CSV for ambiguous rows whose supervised answer is not the "
            "weighted scalar-factor prediction under the competition verifier."
        ),
    )
    return parser.parse_args()


def dec(text: str) -> Decimal:
    return Decimal(text)


def q2(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt2(value: Decimal) -> str:
    return format(q2(value), "f")


def fmt6(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), "f")


def fmt_ratio(value: Decimal) -> str:
    rounded = value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    text = format(rounded.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def parse_unit_prompt(prompt: str) -> tuple[list[tuple[Decimal, Decimal]], Decimal]:
    pairs = [(dec(x), dec(y)) for x, y in PAIR_RE.findall(prompt)]
    query_match = QUERY_RE.search(prompt)
    if not pairs or query_match is None:
        raise ValueError("Failed to parse unit-conversion prompt.")
    query = dec(query_match.group(1))
    return pairs, query


def factor_interval(pairs: list[tuple[Decimal, Decimal]]) -> tuple[Decimal, Decimal]:
    half_cent = Decimal("0.005")
    lower = max((y - half_cent) / x for x, y in pairs)
    upper = min((y + half_cent) / x for x, y in pairs)
    return lower, upper


def query_output_candidates(
    *,
    factor_lo: Decimal,
    factor_hi: Decimal,
    query_value: Decimal,
) -> list[str]:
    half_cent = Decimal("0.005")
    q_lo = query_value * factor_lo
    q_hi = query_value * factor_hi
    min_cent = int((q_lo - half_cent) * 100)
    max_cent = int((q_hi + half_cent) * 100) + 1
    cands = set()
    for cent in range(min_cent, max_cent + 1):
        z = Decimal(cent) / Decimal(100)
        z_lo = (z - half_cent) / query_value
        z_hi = (z + half_cent) / query_value
        if max(factor_lo, z_lo) < min(factor_hi, z_hi):
            cands.add(f"{z:.2f}")
    return sorted(cands)


def candidate_factor_interval(candidate: str, query_value: Decimal) -> tuple[Decimal, Decimal]:
    half_cent = Decimal("0.005")
    value = Decimal(candidate)
    return (value - half_cent) / query_value, (value + half_cent) / query_value


def interval_overlap(
    left: tuple[Decimal, Decimal],
    right: tuple[Decimal, Decimal],
) -> Decimal:
    lower = max(left[0], right[0])
    upper = min(left[1], right[1])
    return max(Decimal("0"), upper - lower)


def largest_overlap_candidate(
    *,
    candidates: list[str],
    factor_lo: Decimal,
    factor_hi: Decimal,
    query_value: Decimal,
) -> tuple[str | None, list[tuple[str, Decimal, Decimal, Decimal]]]:
    factor_range = (factor_lo, factor_hi)
    scored: list[tuple[str, Decimal, Decimal, Decimal]] = []
    for candidate in candidates:
        cand_lo, cand_hi = candidate_factor_interval(candidate, query_value)
        scored.append(
            (
                candidate,
                interval_overlap(factor_range, (cand_lo, cand_hi)),
                cand_lo,
                cand_hi,
            )
        )
    scored.sort(key=lambda item: (-item[1], Decimal(item[0])))
    if not scored:
        return None, []
    best_overlap = scored[0][1]
    tied = [item for item in scored if item[1] == best_overlap]
    if len(tied) != 1 or best_overlap <= 0:
        return None, scored
    return scored[0][0], scored


def rounded_ratio_tiebreak_candidate(
    *,
    pairs: list[tuple[Decimal, Decimal]],
    candidates: list[str],
    query_value: Decimal,
) -> tuple[str | None, Decimal | None, list[tuple[str, int]], Decimal | None]:
    rounded_factors = [q2(y / x) for x, y in pairs]
    counts = Counter(rounded_factors)
    if not counts:
        return None, None, [], None
    top_count = max(counts.values())
    winners = sorted(factor for factor, count in counts.items() if count == top_count)
    factor_counts = [(fmt2(factor), counts[factor]) for factor in sorted(counts)]
    if len(winners) != 1:
        return None, None, factor_counts, None
    factor = winners[0]
    target = query_value * factor
    scored = sorted((abs(Decimal(candidate) - target), Decimal(candidate), candidate) for candidate in candidates)
    if not scored:
        return None, factor, factor_counts, target
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None, factor, factor_counts, target
    return scored[0][2], factor, factor_counts, target


def proportional_vote_candidate(
    *,
    pairs: list[tuple[Decimal, Decimal]],
    query_value: Decimal,
) -> tuple[str | None, list[tuple[str, str, str, str]], list[tuple[str, int]]]:
    """Vote using per-example proportions input/output = query/answer."""
    vote_rows: list[tuple[str, str, str, str]] = []
    votes: list[str] = []
    for input_value, output_value in pairs:
        numerator = (output_value * query_value).quantize(
            Decimal("0.0001"),
            rounding=ROUND_HALF_UP,
        )
        candidate = fmt2(numerator / input_value)
        vote_rows.append(
            (
                format(input_value, "f"),
                format(output_value, "f"),
                format(numerator, "f"),
                candidate,
            )
        )
        votes.append(candidate)
    counts = Counter(votes)
    vote_counts = sorted(counts.items(), key=lambda item: (-item[1], Decimal(item[0])))
    if not vote_counts:
        return None, vote_rows, []
    top_count = vote_counts[0][1]
    winners = [candidate for candidate, count in vote_counts if count == top_count]
    if len(winners) == 1 and top_count * 2 >= len(pairs):
        return winners[0], vote_rows, vote_counts
    return None, vote_rows, vote_counts


def proportional_weighted_candidate(
    *,
    pairs: list[tuple[Decimal, Decimal]],
    query_value: Decimal,
) -> tuple[str, Decimal, Decimal]:
    """Fallback estimate: sum(output * query) / sum(input)."""
    numerator_sum = sum(
        (output_value * query_value).quantize(
            Decimal("0.0001"),
            rounding=ROUND_HALF_UP,
        )
        for input_value, output_value in pairs
    )
    denominator_sum = sum(input_value for input_value, output_value in pairs)
    return fmt2(numerator_sum / denominator_sum), numerator_sum, denominator_sum


def weighted_factor_solution(
    pairs: list[tuple[Decimal, Decimal]],
    query_value: Decimal,
) -> tuple[str, Decimal, Decimal, Decimal, Decimal]:
    input_sum = sum(input_value for input_value, _output_value in pairs)
    output_sum = sum(output_value for _input_value, output_value in pairs)
    factor = (output_sum / input_sum).quantize(FOUR_PLACES, rounding=ROUND_HALF_UP)
    converted = query_value * factor
    answer = fmt2(converted)
    return answer, input_sum, output_sum, factor, converted


def interval_overlap_scores(
    *,
    candidates: list[str],
    factor_lo: Decimal,
    factor_hi: Decimal,
    query_value: Decimal,
) -> list[tuple[str, Decimal, Decimal, Decimal]]:
    factor_range = (factor_lo, factor_hi)
    scores: list[tuple[str, Decimal, Decimal, Decimal]] = []
    for candidate in candidates:
        cand_lo, cand_hi = candidate_factor_interval(candidate, query_value)
        scores.append(
            (
                candidate,
                interval_overlap(factor_range, (cand_lo, cand_hi)),
                cand_lo,
                cand_hi,
            )
        )
    return sorted(scores, key=lambda item: (-item[1], Decimal(item[0])))


def overlap_margin(scores: list[tuple[str, Decimal, Decimal, Decimal]]) -> Decimal | None:
    if not scores:
        return None
    total = sum(score[1] for score in scores)
    if total <= 0:
        return None
    best = scores[0][1]
    second = scores[1][1] if len(scores) > 1 else Decimal("0")
    return (best - second) / total


def prompt_only_policy(
    *,
    pairs: list[tuple[Decimal, Decimal]],
    query_value: Decimal,
    low_margin_threshold: Decimal = Decimal("0.10"),
) -> PolicyDecision:
    """Deployable policy: choose exactly one answer from the prompt alone."""
    lo, hi = factor_interval(pairs)
    candidates = query_output_candidates(
        factor_lo=lo,
        factor_hi=hi,
        query_value=query_value,
    )
    if not candidates:
        return PolicyDecision(None, "unresolved_no_interval_candidate", None)
    if len(candidates) == 1:
        return PolicyDecision(candidates[0], "interval_unique", None)

    scores = interval_overlap_scores(
        candidates=candidates,
        factor_lo=lo,
        factor_hi=hi,
        query_value=query_value,
    )
    interval_prediction, _overlap_details = largest_overlap_candidate(
        candidates=candidates,
        factor_lo=lo,
        factor_hi=hi,
        query_value=query_value,
    )
    margin = overlap_margin(scores)
    proportional_prediction, _vote_rows, _vote_counts = proportional_vote_candidate(
        pairs=pairs,
        query_value=query_value,
    )

    if (
        interval_prediction is not None
        and proportional_prediction is not None
        and proportional_prediction != interval_prediction
        and margin is not None
        and margin <= low_margin_threshold
    ):
        return PolicyDecision(
            proportional_prediction,
            "low_margin_proportional_vote",
            margin,
        )

    if interval_prediction is not None:
        return PolicyDecision(interval_prediction, "largest_overlap_interval", margin)

    rounded_prediction, _factor, _factor_counts, _target = rounded_ratio_tiebreak_candidate(
        pairs=pairs,
        candidates=candidates,
        query_value=query_value,
    )
    if rounded_prediction is not None:
        return PolicyDecision(rounded_prediction, "rounded_ratio_interval", margin)

    weighted_prediction, _numerator_sum, _denominator_sum = proportional_weighted_candidate(
        pairs=pairs,
        query_value=query_value,
    )
    return PolicyDecision(weighted_prediction, "proportional_weighted_fallback", margin)


def wrap_think(reasoning: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>"


def render_trace(
    *,
    pairs: list[tuple[Decimal, Decimal]],
    query_value: Decimal,
    answer: str,
    mode: str,
) -> str:
    prediction, input_sum, output_sum, factor, converted = weighted_factor_solution(
        pairs,
        query_value,
    )
    input_terms = " + ".join(format(input_value, "f") for input_value, _ in pairs)
    output_terms = " + ".join(format(output_value, "f") for _, output_value in pairs)
    pair_text = "; ".join(f"{x:.2f}->{y:.2f}" for x, y in pairs)
    lines = [
        (
            "We need to find a conversion rule that maps the inputs to outputs. "
            "Let me estimate the linear factor from the examples."
        ),
        "I will put my final answer inside \\boxed{}.",
        "",
        (
            "The displayed outputs are rounded, so I will combine all examples "
            "instead of trusting a single pair."
        ),
        "Use factor = sum(outputs) / sum(inputs).",
        "",
        f"Example pairs: {pair_text}",
        "",
        f"sum(inputs) = {input_terms} = {format(input_sum, 'f')}.",
        f"sum(outputs) = {output_terms} = {format(output_sum, 'f')}.",
        (
            f"factor = {format(output_sum, 'f')} / {format(input_sum, 'f')} "
            f"= {format(factor, 'f')} to four decimal places."
        ),
        "",
        f"Converting {query_value:.2f}:",
        f"{query_value:.2f} * {format(factor, 'f')} = {fmt6(converted)}.",
        f"Rounding this conversion to two decimals gives {prediction}.",
        "",
        "I will now return the answer in \\boxed{}",
        f"The final answer is \\boxed{{{prediction}}}",
    ]
    return "\n".join(lines)


def build_rows(train_csv: Path) -> tuple[list[UnitRow], Counter[str]]:
    rows: list[UnitRow] = []
    mode_counts: Counter[str] = Counter()
    with train_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            if infer_category(prompt) != "Unit Conversion":
                continue

            pairs, query = parse_unit_prompt(prompt)
            answer, _input_sum, _output_sum, _factor, _converted = weighted_factor_solution(
                pairs,
                query,
            )
            mode = "weighted_scalar_factor"
            mode_counts[mode] += 1
            generated_cot = render_trace(
                pairs=pairs,
                query_value=query,
                answer=answer,
                mode=mode,
            )
            rows.append(
                UnitRow(
                    id=row["id"],
                    prompt=prompt,
                    answer=answer,
                    generated_cot=generated_cot,
                    label="Unit Conversion",
                    category="Unit Conversion",
                    source="deterministic_unit_conversion_cot",
                    source_mode=mode,
                    pair_count=len(pairs),
                )
            )
    return rows, mode_counts


def has_unique_interval_answer(row: UnitRow) -> bool:
    pairs, query = parse_unit_prompt(row.prompt)
    lo, hi = factor_interval(pairs)
    candidates = query_output_candidates(
        factor_lo=lo,
        factor_hi=hi,
        query_value=query,
    )
    return len(candidates) == 1


def main() -> None:
    args = parse_args()
    train_path = ROOT / args.train_csv
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows, mode_counts = build_rows(train_path)
    method_clean_rows = rows
    answer_aligned_rows: list[UnitRow] = []
    method_resolved_rows = rows
    unresolved_rows: list[UnitRow] = []
    fieldnames = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "source_mode",
        "pair_count",
    ]
    outputs = [
        (output_path, rows),
        (
            ROOT / args.method_clean_output_csv,
            method_clean_rows,
        ),
        (
            ROOT / args.method_resolved_output_csv,
            method_resolved_rows,
        ),
        (
            ROOT / args.answer_aligned_output_csv,
            answer_aligned_rows,
        ),
        (
            ROOT / args.unresolved_output_csv,
            unresolved_rows,
        ),
    ]
    for path, selected_rows in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row.__dict__ for row in selected_rows)

    print(f"Wrote {len(rows)} Unit Conversion CoT rows to {output_path}")
    print(
        f"  method_clean: {len(method_clean_rows)} -> "
        f"{ROOT / args.method_clean_output_csv}"
    )
    print(
        "  method_resolved: "
        f"{len(method_resolved_rows)} -> "
        f"{ROOT / args.method_resolved_output_csv}"
    )
    print(
        "  answer_aligned: "
        f"{len(answer_aligned_rows)} -> "
        f"{ROOT / args.answer_aligned_output_csv}"
    )
    print(
        f"  unresolved: {len(unresolved_rows)} -> "
        f"{ROOT / args.unresolved_output_csv}"
    )
    for key in sorted(mode_counts):
        print(f"  {key}: {mode_counts[key]}")


if __name__ == "__main__":
    main()
