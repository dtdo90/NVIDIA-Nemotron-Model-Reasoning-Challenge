#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HARNESS = ROOT / "reference/cursor/transformation_rules/numeric_equation/harness"
for path in (SRC, HARNESS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nemotron_baseline.data import load_split_assignments, select_ids_for_splits  # noqa: E402
from nemotron_baseline.numeric_equation import (  # noqa: E402
    NumericEquationPuzzle,
    ParsedEquation,
    parse_numeric_equation_puzzle,
)
from extended_dsl import (  # noqa: E402
    Candidate,
    OUTPUT_MODE_BY_NAME,
    PAIRING_NAMES,
    enumerate_matching,
)
import numeric_equation_detailed_cot as detailed_cot  # noqa: E402


OPERATORS = tuple("+-*/?@#$%&!<>[]{}|\\:`'\"^")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic numeric_equation CoT rows from deterministic real "
            "numeric_equation rules. Synthetic rows keep the same operator/rule shape "
            "but use fresh two-digit operands."
        )
    )
    parser.add_argument(
        "--source-csv",
        default="data/trainable/numeric_equation_labelled_cot.csv",
        help="Labelled numeric_equation CoT CSV with rule_label and is_deterministic.",
    )
    parser.add_argument(
        "--split-csv",
        default="data/splits_75_10_15.config.json",
        help="Optional split config. Only base ids in --train-splits are augmented.",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["sft_train"],
        help="Split names whose deterministic numeric rows should receive synthetic variants.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/numeric_equation_synthetic_cot.csv",
        help="Destination synthetic CoT CSV.",
    )
    parser.add_argument("--variants-per-row", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-query-examples",
        type=int,
        default=2,
        help=(
            "If the real row has at least one same-operator example, generate at least "
            "this many same-operator examples in the synthetic prompt."
        ),
    )
    return parser.parse_args()


def parse_rule_label(label: str) -> Candidate:
    width = None
    if "|w=" in label:
        label, width_text = label.rsplit("|w=", 1)
        width = int(width_text)

    pairing = None
    rest = label
    for candidate_pairing in PAIRING_NAMES:
        prefix = f"{candidate_pairing}|"
        if label.startswith(prefix):
            pairing = candidate_pairing
            rest = label[len(prefix):]
            break
    if pairing is None:
        raise ValueError(f"Could not parse pairing from rule_label={label!r}")

    for outmode_name in sorted(OUTPUT_MODE_BY_NAME, key=len, reverse=True):
        suffix = f"|{outmode_name}"
        if rest.endswith(suffix):
            base_name = rest[:-len(suffix)]
            return Candidate(pairing, base_name, outmode_name, width)

    raise ValueError(f"Could not parse output mode from rule_label={label!r}")


def two_digit(rng: random.Random) -> str:
    return f"{rng.randrange(100):02d}"


def expression(left: str, operator: str, right: str) -> str:
    return f"{left}{operator}{right}"


def generate_equation(
    rng: random.Random,
    candidate: Candidate,
    operator: str,
    *,
    avoid_answer_brace: bool,
) -> ParsedEquation | None:
    for _ in range(300):
        left = two_digit(rng)
        right = two_digit(rng)
        rhs = candidate.predict(left, right, operator)
        if rhs is None:
            continue
        if len(rhs) > 10:
            continue
        if avoid_answer_brace and "}" in rhs:
            continue
        return ParsedEquation(
            lhs_text=expression(left, operator, right),
            rhs_text=rhs,
            left_operand_text=left,
            operator=operator,
            right_operand_text=right,
        )
    return None


def choose_operator_candidate(
    examples: list[ParsedEquation],
    operator: str,
    query_candidate: Candidate,
) -> Candidate | None:
    candidates = enumerate_matching(examples, operator)
    if not candidates:
        return None
    motif_matches = [
        candidate
        for candidate in candidates
        if candidate.transform_signature == query_candidate.transform_signature
    ]
    pool = motif_matches or candidates
    return sorted(pool, key=lambda candidate: (candidate.complexity, candidate.label))[0]


def build_synthetic_variant(
    *,
    rng: random.Random,
    base_row: dict[str, str],
    variant_index: int,
    min_query_examples: int,
) -> dict[str, str] | None:
    puzzle = parse_numeric_equation_puzzle(base_row["prompt"])
    if puzzle is None:
        return None

    query_candidate = detailed_cot.parse_rule_label(base_row["rule_label"])
    query_operator = puzzle.query.operator
    original_by_operator: dict[str, list[ParsedEquation]] = defaultdict(list)
    for example in puzzle.examples:
        original_by_operator[example.operator].append(example)

    chosen_by_operator: dict[str, Candidate] = {}
    unsupported_operators: set[str] = set()
    for operator, examples in original_by_operator.items():
        if operator == query_operator:
            chosen_by_operator[operator] = query_candidate
            continue
        chosen = choose_operator_candidate(examples, operator, query_candidate)
        if chosen is None:
            unsupported_operators.add(operator)
            continue
        chosen_by_operator[operator] = chosen

    if query_operator not in chosen_by_operator and query_operator in original_by_operator:
        chosen_by_operator[query_operator] = query_candidate

    synthetic_examples: list[ParsedEquation] = []
    same_operator_count = len(original_by_operator.get(query_operator, []))
    for original_example in puzzle.examples:
        operator = original_example.operator
        candidate = chosen_by_operator.get(operator)
        if candidate is None:
            continue
        synthetic_example = generate_equation(
            rng,
            candidate,
            operator,
            avoid_answer_brace=False,
        )
        if synthetic_example is None:
            return None
        synthetic_examples.append(synthetic_example)

    if same_operator_count > 0:
        extra_needed = max(0, min_query_examples - same_operator_count)
        for _ in range(extra_needed):
            synthetic_example = generate_equation(
                rng,
                query_candidate,
                query_operator,
                avoid_answer_brace=False,
            )
            if synthetic_example is None:
                return None
            synthetic_examples.append(synthetic_example)
    if not synthetic_examples:
        for _ in range(max(1, min_query_examples)):
            synthetic_example = generate_equation(
                rng,
                query_candidate,
                query_operator,
                avoid_answer_brace=False,
            )
            if synthetic_example is None:
                return None
            synthetic_examples.append(synthetic_example)

    query_equation = generate_equation(
        rng,
        query_candidate,
        query_operator,
        avoid_answer_brace=False,
    )
    if query_equation is None:
        return None
    query = ParsedEquation(
        lhs_text=query_equation.lhs_text,
        rhs_text="",
        left_operand_text=query_equation.left_operand_text,
        operator=query_equation.operator,
        right_operand_text=query_equation.right_operand_text,
    )
    answer = query_equation.rhs_text

    prompt_lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:",
    ]
    prompt_lines.extend(f"{example.lhs_text} = {example.rhs_text}" for example in synthetic_examples)
    prompt_lines.append(f"Now, determine the result for: {query.lhs_text}")
    prompt = "\n".join(prompt_lines)

    synthetic_puzzle = NumericEquationPuzzle(
        prompt=prompt,
        examples=tuple(synthetic_examples),
        query=query,
    )
    same_op = [example for example in synthetic_examples if example.operator == query_operator]
    note = None
    if not same_op:
        note = (
            "Since the query operator is absent from the examples, I use the row motif "
            "and the selected base rule for this operator."
        )
    generated_cot = detailed_cot.render_detailed_trace(
        puzzle=synthetic_puzzle,
        candidate=query_candidate,
        is_deterministic=True,
        source_tier="synthetic_from_deterministic",
    )

    return {
        "id": f"numeric_equation_synth_{base_row['id']}_{variant_index + 1:02d}",
        "base_id": base_row["id"],
        "prompt": prompt,
        "answer": answer,
        "generated_cot": generated_cot,
        "label": "Transformation Rules",
        "category": "Transformation Rules",
        "source": "numeric_equation_synthetic_cot",
        "rule_label": base_row["rule_label"],
        "source_tier": "synthetic_from_deterministic",
        "is_deterministic": "1",
        "data_type": "synthetic",
    }


def selected_base_ids(split_csv: str, split_names: list[str]) -> set[str]:
    split_path = ROOT / split_csv
    if not split_path.exists():
        return set()
    assignments = load_split_assignments(split_path)
    return select_ids_for_splits(assignments, split_names)


def main() -> None:
    args = parse_args()
    source_path = ROOT / args.source_csv
    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_ids = selected_base_ids(args.split_csv, args.train_splits)
    rng = random.Random(args.seed)

    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "prompt", "answer", "rule_label", "is_deterministic"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{source_path} is missing required columns: {sorted(missing)}")
        base_rows = [
            dict(row)
            for row in reader
            if row.get("is_deterministic", "").strip() == "1"
            and (not selected_ids or row.get("id", "").strip() in selected_ids)
        ]

    synthetic_rows: list[dict[str, str]] = []
    skipped: Counter[str] = Counter()
    for base_row in base_rows:
        for variant_index in range(args.variants_per_row):
            synthetic = build_synthetic_variant(
                rng=rng,
                base_row=base_row,
                variant_index=variant_index,
                min_query_examples=args.min_query_examples,
            )
            if synthetic is None:
                skipped["generation_failed"] += 1
                continue
            synthetic_rows.append(synthetic)

    fieldnames = [
        "id",
        "base_id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "rule_label",
        "source_tier",
        "is_deterministic",
        "data_type",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(synthetic_rows)

    print(f"Base deterministic numeric_equation rows considered: {len(base_rows)}")
    print(f"Wrote {len(synthetic_rows)} synthetic numeric_equation rows -> {output_path}")
    if skipped:
        print(f"Skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
