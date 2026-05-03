#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.numeric_equation import classify_equation_vs_symbol
from nemotron_baseline.symbol_transform import solve_symbol_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the deterministic symbol_transform solver."
    )
    parser.add_argument(
        "--input-csv",
        default="data/train.csv",
        help="Path to the competition training CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/symbol_transform_solver_analysis.csv",
        help="Where to write per-row analysis.",
    )
    parser.add_argument(
        "--rule-bank",
        choices=("core", "asymmetric", "extended"),
        default="asymmetric",
        help="Which operation bank to use for encrypted digit-transform search.",
    )
    parser.add_argument(
        "--selection",
        choices=("unique", "ranked"),
        default="unique",
        help="Whether to require candidate agreement or use ranked tie-breaking.",
    )
    parser.add_argument(
        "--allow-absent-query-operator",
        action="store_true",
        help="Try an aggressive fallback when the query operator is absent from examples.",
    )
    parser.add_argument(
        "--no-abs-output",
        action="store_true",
        help="Disable abs output rendering.",
    )
    parser.add_argument(
        "--output-mode-bank",
        choices=("current", "formats"),
        default="current",
        help=(
            "Output rendering modes to scan. 'current' keeps raw/rev/abs; "
            "'formats' also tries last/last_rev and negative operator marker modes."
        ),
    )
    parser.add_argument(
        "--max-states-per-rule",
        type=int,
        default=120,
        help="Maximum mapping states retained per operator/rule.",
    )
    parser.add_argument(
        "--max-combined-states",
        type=int,
        default=120,
        help="Maximum merged global mapping states retained per motif.",
    )
    parser.add_argument(
        "--min-query-examples-for-global",
        type=int,
        default=2,
        help="Minimum query-operator examples required before global motif fallback can predict.",
    )
    parser.add_argument(
        "--max-query-unknowns",
        type=int,
        default=0,
        help=(
            "Maximum unknown query operand symbols to assign during inference. "
            "0 preserves the original safe behavior."
        ),
    )
    parser.add_argument(
        "--adaptive-retry",
        action="store_true",
        help="Retry no-rule/ambiguous rows with a wider symbol-to-digit state beam.",
    )
    parser.add_argument(
        "--family-rescue",
        action="store_true",
        help=(
            "Enable the low-confidence add/sub/mul family rescue branch for rows "
            "where direct/core search does not give a unique answer."
        ),
    )
    parser.add_argument(
        "--family-rescue-selection",
        choices=("unique", "ranked"),
        default="unique",
        help=(
            "Deprecated compatibility option. Family-rescue ties are no longer "
            "ranked by output length or numeric value; multiple variants remain ambiguous."
        ),
    )
    parser.add_argument(
        "--no-map-completion-rescue",
        action="store_true",
        help=(
            "Disable the rescue that uses other operator equations to complete "
            "missing query symbol mappings after the same-operator rule is locked."
        ),
    )
    parser.add_argument(
        "--no-guarded-min1-rescue",
        action="store_true",
        help=(
            "Disable the conservative branch that allows exactly one query-operator "
            "example only when the row locks a complete 10-symbol map under a dominant motif."
        ),
    )
    parser.add_argument(
        "--adaptive-retry-on",
        choices=("no_rule", "no_rule_or_ambiguous"),
        default="no_rule",
        help="Which failed rows are eligible for --adaptive-retry.",
    )
    parser.add_argument(
        "--retry-max-states-per-rule",
        type=int,
        default=240,
        help="State beam used by --adaptive-retry.",
    )
    parser.add_argument(
        "--retry-max-combined-states",
        type=int,
        default=0,
        help="Optional merged-state beam used by --adaptive-retry; 0 keeps the normal value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for quick experiments.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=8,
        help="How many solved and unsolved examples to print.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    predicted = 0
    exact = 0
    answer_in_variants = 0
    oracle_low_confidence = 0
    method_counts: Counter[str] = Counter()
    method_exact: Counter[str] = Counter()
    method_answer_in_variants: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    motif_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    solved_examples: list[dict[str, str]] = []
    missed_examples: list[dict[str, str]] = []
    records: list[dict[str, str | int]] = []

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            if "transformation rules" not in prompt.lower():
                continue
            if classify_equation_vs_symbol(prompt) != "symbol_transform":
                continue

            total += 1
            result = solve_symbol_transform(
                prompt,
                rule_bank=args.rule_bank,
                include_abs_output=not args.no_abs_output,
                output_mode_bank=args.output_mode_bank,
                max_states_per_rule=args.max_states_per_rule,
                max_combined_states=args.max_combined_states,
                allow_absent_query_operator=args.allow_absent_query_operator,
                min_query_examples_for_global=args.min_query_examples_for_global,
                max_query_unknowns=args.max_query_unknowns,
                selection=args.selection,
                adaptive_retry=args.adaptive_retry,
                adaptive_retry_on=args.adaptive_retry_on,
                retry_max_states_per_rule=args.retry_max_states_per_rule,
                retry_max_combined_states=args.retry_max_combined_states or None,
                enable_family_rescue=args.family_rescue,
                family_rescue_selection=args.family_rescue_selection,
                enable_map_completion_rescue=not args.no_map_completion_rescue,
                enable_guarded_min1_rescue=not args.no_guarded_min1_rescue,
            )

            is_exact = int(result.prediction == row["answer"])
            is_predicted = int(result.prediction is not None)
            variant_set = set(result.prediction_variants)
            variant_count = len(variant_set)
            has_multiple_variants = int(variant_count > 1)
            has_answer_variant = int(row["answer"] in variant_set)
            is_oracle_low_confidence = int(
                not is_predicted and has_multiple_variants and has_answer_variant
            )
            predicted += is_predicted
            exact += is_exact
            answer_in_variants += has_answer_variant
            oracle_low_confidence += is_oracle_low_confidence
            method_counts[result.method] += 1
            method_exact[result.method] += is_exact
            method_answer_in_variants[result.method] += has_answer_variant
            confidence_counts[result.confidence] += 1

            candidate = result.chosen_candidate
            motif = candidate.motif.label if candidate is not None else ""
            rule = candidate.query_rule.name if candidate is not None else ""
            family = candidate.query_rule.family if candidate is not None else ""
            query_completed_count = candidate.query_completed_count if candidate is not None else 0
            if motif:
                motif_counts[motif] += 1
            if rule:
                rule_counts[rule] += 1
            if family:
                family_counts[family] += 1

            record = {
                "id": row["id"],
                "answer": row["answer"],
                "prediction": result.prediction or "",
                "is_exact": is_exact,
                "variant_count": variant_count,
                "multiple_valid_answers": has_multiple_variants,
                "answer_in_variants": has_answer_variant,
                "oracle_low_confidence": is_oracle_low_confidence,
                "method": result.method,
                "confidence": result.confidence,
                "candidate_count": result.candidate_count,
                "prediction_variants": "|".join(result.prediction_variants),
                "prediction_variants_json": json.dumps(result.prediction_variants, ensure_ascii=False),
                "motif": motif,
                "query_rule": rule,
                "query_rule_family": family,
                "query_completed_count": query_completed_count,
                "notes": " ".join(result.notes),
            }
            records.append(record)

            if is_exact and len(solved_examples) < args.show_examples:
                solved_examples.append(record)
            if not is_exact and len(missed_examples) < args.show_examples:
                missed_examples.append(record)

            if args.limit and total >= args.limit:
                break

    fieldnames = [
        "id",
        "answer",
        "prediction",
        "is_exact",
        "variant_count",
        "multiple_valid_answers",
        "answer_in_variants",
        "oracle_low_confidence",
        "method",
        "confidence",
        "candidate_count",
        "prediction_variants",
        "prediction_variants_json",
        "motif",
        "query_rule",
        "query_rule_family",
        "query_completed_count",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Total symbol_transform rows: {total}")
    print(f"Predicted: {predicted}")
    print(f"Exact: {exact}")
    print(f"Answer in variants: {answer_in_variants}")
    print(f"Oracle low-confidence rows: {oracle_low_confidence}")
    print(f"Prediction rate: {predicted / max(1, total):.2%}")
    print(f"Exact rate: {exact / max(1, total):.2%}")
    print(f"Wrote: {output_path}")

    print("\nMethods:")
    for method, count in method_counts.most_common():
        correct = method_exact[method]
        variant_count = method_answer_in_variants[method]
        precision = correct / count if count else 0.0
        print(
            f"  {method}: count={count}, exact={correct}, "
            f"answer_in_variants={variant_count}, precision={precision:.2%}"
        )

    print("\nConfidence:")
    for confidence, count in sorted(confidence_counts.items()):
        print(f"  {confidence}: {count}")

    print("\nTop motifs:")
    for motif, count in motif_counts.most_common(12):
        print(f"  {motif}: {count}")

    print("\nTop query rules:")
    for rule, count in rule_counts.most_common(16):
        print(f"  {rule}: {count}")

    print("\nRule families:")
    for family, count in family_counts.most_common():
        print(f"  {family}: {count}")

    print("\nSolved examples:")
    for record in solved_examples:
        print(
            f"  {record['id']}: answer={record['answer']!r}, method={record['method']}, "
            f"motif={record['motif']}, rule={record['query_rule']}"
        )

    print("\nMissed examples:")
    for record in missed_examples:
        print(
            f"  {record['id']}: gold={record['answer']!r}, pred={record['prediction']!r}, "
            f"method={record['method']}, confidence={record['confidence']}"
        )


if __name__ == "__main__":
    main()
