#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WINNER_ROOT = ROOT / "reference/winner-solution/huikang-nemotron-repository-snapshot/nemotron-master"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(WINNER_ROOT) not in sys.path:
    sys.path.insert(0, str(WINNER_ROOT))

from nemotron_baseline.bit_manipulation import solve_bit_manipulation
from reasoning import extract_answer
from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.store_types import Example, Problem
from scripts.render_bit_solver_cot import load_solver_namespace


_EXAMPLE_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
_QUERY_RE = re.compile(r"determine the output for:\s*([01]{8})", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare and ensemble bit-manipulation solvers."
    )
    parser.add_argument(
        "--input-csv",
        default="data/train.csv",
        help="Path to a CSV containing id,prompt,answer columns.",
    )
    parser.add_argument(
        "--notebook-path",
        default="reference/kaggle_notebooks/bit-manipulation-solver-cot-generator.ipynb",
        help="Path to the older notebook solver.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/bit_solver_ensemble_analysis.csv",
        help="Path for row-level predictions.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=8,
        help="Number of disagreement examples to print.",
    )
    return parser


def is_bit_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return "bit manipulation" in lowered or "8-bit binary" in lowered


def winner_predict(row: dict[str, str]) -> str:
    prompt = row["prompt"]
    query_match = _QUERY_RE.search(prompt)
    if query_match is None:
        return ""
    problem = Problem(
        id=row["id"],
        category="bit_manipulation",
        examples=[Example(input_value, output_value) for input_value, output_value in _EXAMPLE_RE.findall(prompt)],
        question=query_match.group(1),
        answer=row["answer"],
        prompt=prompt,
    )
    reasoning_text = reasoning_bit_manipulation(problem)
    if reasoning_text is None:
        return ""
    return extract_answer(reasoning_text)


def notebook_predict(row: dict[str, str], namespace: dict[str, Any]) -> tuple[str, str, str]:
    prediction, method, _details = namespace["solve_puzzle"](row["prompt"])
    confidence = "high" if method == "ctx" or method.startswith("w_") else "low"
    return prediction or "", method, confidence


def choose_policy(
    winner: str,
    notebook: str,
    notebook_confidence: str,
    broad: str,
    broad_confidence: str,
) -> tuple[str, str]:
    """Conservative ensemble policy.

    The Tong/winner solver is the anchor. We only override it when a second
    solver gives a stronger local signal and the older notebook solver is in
    its low-confidence brute-force zone.
    """
    if broad_confidence == "high" and broad and broad != winner:
        return broad, "broad_high_override"
    if notebook_confidence == "low" and broad_confidence == "medium" and broad and broad != winner:
        return broad, "broad_medium_over_notebook_low"
    if notebook_confidence == "high" and notebook and notebook == broad and notebook != winner:
        return notebook, "notebook_broad_agree_override"
    return winner, "winner_default"


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    namespace = load_solver_namespace(Path(args.notebook_path))

    rows_out: list[dict[str, str]] = []
    correct_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    total = 0
    disagreement_examples: list[dict[str, str]] = []

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not is_bit_prompt(row["prompt"]):
                continue

            total += 1
            answer = row["answer"]
            winner = winner_predict(row)
            notebook, notebook_method, notebook_confidence = notebook_predict(row, namespace)
            broad_result = solve_bit_manipulation(row["prompt"])
            broad = broad_result.prediction or ""
            broad_confidence = broad_result.confidence
            ensemble, policy = choose_policy(
                winner,
                notebook,
                notebook_confidence,
                broad,
                broad_confidence,
            )

            predictions = {
                "winner": winner,
                "notebook": notebook,
                "broad": broad,
                "ensemble": ensemble,
            }
            for name, prediction in predictions.items():
                correct_counts[name] += int(prediction == answer)

            oracle = any(prediction == answer for prediction in predictions.values())
            correct_counts["oracle_union"] += int(oracle)
            policy_counts[policy] += 1

            row_out = {
                "id": row["id"],
                "answer": answer,
                "winner": winner,
                "winner_correct": str(winner == answer),
                "notebook": notebook,
                "notebook_correct": str(notebook == answer),
                "notebook_method": notebook_method,
                "notebook_confidence": notebook_confidence,
                "broad": broad,
                "broad_correct": str(broad == answer),
                "broad_confidence": broad_confidence,
                "broad_ambiguous_bits": " ".join(str(bit) for bit in broad_result.ambiguous_bits),
                "ensemble": ensemble,
                "ensemble_correct": str(ensemble == answer),
                "ensemble_policy": policy,
                "oracle_union": str(oracle),
            }
            rows_out.append(row_out)

            if (
                len(disagreement_examples) < args.show_examples
                and len({winner, notebook, broad}) > 1
                and oracle
                and winner != answer
            ):
                disagreement_examples.append(row_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "answer",
        "winner",
        "winner_correct",
        "notebook",
        "notebook_correct",
        "notebook_method",
        "notebook_confidence",
        "broad",
        "broad_correct",
        "broad_confidence",
        "broad_ambiguous_bits",
        "ensemble",
        "ensemble_correct",
        "ensemble_policy",
        "oracle_union",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Total bit-manipulation rows: {total}")
    for name in ("winner", "notebook", "broad", "ensemble", "oracle_union"):
        count = correct_counts[name]
        print(f"{name}: {count} / {total} = {count / max(1, total):.2%}")

    print("\nEnsemble policy counts:")
    for policy, count in policy_counts.most_common():
        print(f"  {policy}: {count}")

    print("\nWinner misses recovered by another solver:")
    recovered = [row for row in rows_out if row["winner_correct"] == "False" and row["oracle_union"] == "True"]
    print(f"  {len(recovered)}")

    print("\nExample recoverable disagreements:")
    for row in disagreement_examples:
        print(
            f"  {row['id']}: gold={row['answer']} winner={row['winner']} "
            f"notebook={row['notebook']}({row['notebook_method']}/{row['notebook_confidence']}) "
            f"broad={row['broad']}({row['broad_confidence']})"
        )


if __name__ == "__main__":
    main()

