#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.bit_manipulation import solve_bit_manipulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze deterministic bit-manipulation solver coverage."
    )
    parser.add_argument(
        "--input-csv",
        default="data/train.csv",
        help="Path to a CSV containing id,prompt,answer columns.",
    )
    parser.add_argument(
        "--max-arity",
        type=int,
        default=4,
        choices=(1, 2, 3, 4),
        help="Maximum boolean gate arity to search.",
    )
    parser.add_argument(
        "--include-higher-arity-when-lower-exists",
        action="store_true",
        help="Also scan higher-arity candidates even if lower-arity candidates exist.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=8,
        help="How many solved and unsolved examples to print.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write row-level solver results.",
    )
    return parser


def is_bit_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return "bit manipulation" in lowered or "8-bit binary" in lowered


def candidate_label(result) -> str:
    parts = []
    for bit, candidate in enumerate(result.chosen_candidates):
        if candidate is None:
            parts.append(f"{bit}:DEFAULT0")
        else:
            parts.append(f"{bit}:{candidate.expression}")
    return "; ".join(parts)


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_csv)

    total = 0
    correct = 0
    exact_high = 0
    confidence_counts: Counter[str] = Counter()
    ambiguity_counts: Counter[int] = Counter()
    no_candidate_counts: Counter[int] = Counter()
    gate_counts: Counter[str] = Counter()
    solved_examples: list[tuple[str, str, str, str]] = []
    unsolved_examples: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    output_rows: list[dict[str, str]] = []

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            if not is_bit_prompt(prompt):
                continue

            total += 1
            result = solve_bit_manipulation(
                prompt,
                max_arity=args.max_arity,
                include_higher_arity_when_lower_exists=args.include_higher_arity_when_lower_exists,
            )
            prediction = result.prediction or ""
            is_correct = prediction == row["answer"]
            correct += int(is_correct)
            exact_high += int(is_correct and result.confidence == "high")
            confidence_counts[result.confidence] += 1
            ambiguity_counts[len(result.ambiguous_bits)] += 1
            no_candidate_counts[len(result.no_candidate_bits)] += 1

            for candidate in result.chosen_candidates:
                if candidate is not None:
                    gate_counts[candidate.gate_name] += 1

            if is_correct and len(solved_examples) < args.show_examples:
                solved_examples.append((row["id"], row["answer"], result.confidence, candidate_label(result)))
            elif not is_correct and len(unsolved_examples) < args.show_examples:
                unsolved_examples.append(
                    (row["id"], row["answer"], prediction, result.confidence, result.notes)
                )

            output_rows.append(
                {
                    "id": row["id"],
                    "answer": row["answer"],
                    "prediction": prediction,
                    "correct": str(is_correct),
                    "confidence": result.confidence,
                    "ambiguous_bits": " ".join(str(bit) for bit in result.ambiguous_bits),
                    "no_candidate_bits": " ".join(str(bit) for bit in result.no_candidate_bits),
                    "candidate_counts": " ".join(str(count) for count in result.candidate_count_by_bit),
                    "chosen_candidates": candidate_label(result),
                    "notes": " ".join(result.notes),
                }
            )

    print(f"Total bit-manipulation rows: {total}")
    print(f"Solved exactly: {correct}")
    print(f"Coverage: {correct / max(1, total):.2%}")
    print(f"Correct high-confidence rows: {exact_high}")

    print("\nConfidence counts:")
    for confidence, count in sorted(confidence_counts.items()):
        print(f"  {confidence}: {count}")

    print("\nAmbiguous-bit counts:")
    for count, rows in sorted(ambiguity_counts.items()):
        print(f"  {count}: {rows}")

    print("\nNo-candidate-bit counts:")
    for count, rows in sorted(no_candidate_counts.items()):
        print(f"  {count}: {rows}")

    print("\nChosen gate counts:")
    for gate_name, count in gate_counts.most_common():
        print(f"  {gate_name}: {count}")

    print("\nSolved examples:")
    for row_id, answer, confidence, label in solved_examples:
        print(f"  {row_id}: answer={answer}, confidence={confidence}, rules={label}")

    print("\nUnsolved examples:")
    for row_id, answer, prediction, confidence, notes in unsolved_examples:
        print(
            f"  {row_id}: gold={answer}, prediction={prediction}, "
            f"confidence={confidence}, notes={' '.join(notes) if notes else '-'}"
        )

    if args.output_csv is not None:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "id",
            "answer",
            "prediction",
            "correct",
            "confidence",
            "ambiguous_bits",
            "no_candidate_bits",
            "candidate_counts",
            "chosen_candidates",
            "notes",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)


if __name__ == "__main__":
    main()

