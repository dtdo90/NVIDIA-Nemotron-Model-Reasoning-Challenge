from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nemotron_baseline.numeric_equation import classify_equation_vs_symbol


@dataclass(frozen=True)
class ExampleEquation:
    lhs: str
    rhs: str


@dataclass(frozen=True)
class PairRuleSolution:
    family: str
    order: str
    prediction: str
    first_map: tuple[tuple[str, str], ...]
    second_map: tuple[tuple[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for exact-fit pair-based rules on symbol_transform rows with '-' queries."
    )
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--output-csv", default="data/symbol_minus_pair_rule_cases.csv")
    parser.add_argument("--summary-json", default="data/symbol_minus_pair_rule_cases.summary.json")
    return parser.parse_args()


def extract_query(prompt: str) -> str | None:
    marker = "Now, determine the result for: "
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            return stripped.split(marker, 1)[1].strip()
    return None


def parse_examples(prompt: str) -> list[ExampleEquation]:
    examples: list[ExampleEquation] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if " = " not in stripped:
            continue
        lhs, rhs = stripped.split(" = ", 1)
        examples.append(ExampleEquation(lhs=lhs.strip(), rhs=rhs.strip()))
    return examples


def pair_keys(lhs: str, family: str) -> tuple[str, str]:
    if family == "sides":
        return lhs[:2], lhs[3:]
    if family == "mirrored":
        return lhs[0] + lhs[4], lhs[1] + lhs[3]
    raise ValueError(f"Unknown family: {family}")


def search_pair_rule_solutions(examples: list[ExampleEquation], query: str) -> list[PairRuleSolution]:
    solutions: list[PairRuleSolution] = []

    for family in ("sides", "mirrored"):
        query_first, query_second = pair_keys(query, family)

        for order in ("12", "21"):
            partial_solutions: list[tuple[dict[str, str], dict[str, str]]] = [({}, {})]

            for example in examples:
                first_pair, second_pair = pair_keys(example.lhs, family)
                next_partial_solutions: list[tuple[dict[str, str], dict[str, str]]] = []

                for first_map, second_map in partial_solutions:
                    for split_index in range(len(example.rhs) + 1):
                        if order == "12":
                            first_piece = example.rhs[:split_index]
                            second_piece = example.rhs[split_index:]
                        else:
                            second_piece = example.rhs[:split_index]
                            first_piece = example.rhs[split_index:]

                        # Pair-based compression hypothesis:
                        # each pair contributes either nothing or one symbol.
                        if len(first_piece) > 1 or len(second_piece) > 1:
                            continue

                        existing_first = first_map.get(first_pair)
                        existing_second = second_map.get(second_pair)
                        if existing_first is not None and existing_first != first_piece:
                            continue
                        if existing_second is not None and existing_second != second_piece:
                            continue

                        updated_first = dict(first_map)
                        updated_second = dict(second_map)
                        updated_first[first_pair] = first_piece
                        updated_second[second_pair] = second_piece
                        next_partial_solutions.append((updated_first, updated_second))

                partial_solutions = next_partial_solutions
                if not partial_solutions:
                    break

            for first_map, second_map in partial_solutions:
                if query_first not in first_map or query_second not in second_map:
                    continue
                if order == "12":
                    prediction = first_map[query_first] + second_map[query_second]
                else:
                    prediction = second_map[query_second] + first_map[query_first]
                solutions.append(
                    PairRuleSolution(
                        family=family,
                        order=order,
                        prediction=prediction,
                        first_map=tuple(sorted(first_map.items())),
                        second_map=tuple(sorted(second_map.items())),
                    )
                )

    return solutions


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)

    total_rows = 0
    eligible_rows = 0
    rows_with_minus_examples = 0
    rows_with_solutions = 0
    rows_with_unique_prediction = 0
    rows_with_correct_unique_prediction = 0
    rows_with_correct_unique_prediction_and_two_examples = 0
    rows_with_mirrored_solution = 0
    rows_with_correct_unique_mirrored_prediction = 0
    written_rows = 0

    summary_rows: list[dict[str, str | int | bool]] = []

    with train_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            if classify_equation_vs_symbol(prompt) != "symbol_transform":
                continue

            query = extract_query(prompt)
            if query is None or len(query) != 5 or query[2] != "-":
                continue

            total_rows += 1
            examples = parse_examples(prompt)
            minus_examples = [example for example in examples if len(example.lhs) == 5 and example.lhs[2] == "-"]
            if not minus_examples:
                continue

            rows_with_minus_examples += 1
            eligible_rows += 1
            solutions = search_pair_rule_solutions(minus_examples, query)
            if not solutions:
                continue

            rows_with_solutions += 1
            if any(solution.family == "mirrored" for solution in solutions):
                rows_with_mirrored_solution += 1
            unique_predictions = sorted({solution.prediction for solution in solutions})
            is_unique = len(unique_predictions) == 1
            is_correct = is_unique and unique_predictions[0] == row["answer"]
            if is_unique:
                rows_with_unique_prediction += 1
            if is_correct:
                rows_with_correct_unique_prediction += 1
                if len(minus_examples) >= 2:
                    rows_with_correct_unique_prediction_and_two_examples += 1
                if any(solution.family == "mirrored" for solution in solutions):
                    rows_with_correct_unique_mirrored_prediction += 1

            representative = solutions[0]
            summary_rows.append(
                {
                    "id": row["id"],
                    "query": query,
                    "answer": row["answer"],
                    "minus_example_count": len(minus_examples),
                    "solution_count": len(solutions),
                    "unique_prediction": unique_predictions[0] if is_unique else "",
                    "is_unique_prediction": is_unique,
                    "is_correct_unique_prediction": is_correct,
                    "families": ",".join(sorted({solution.family for solution in solutions})),
                    "orders": ",".join(sorted({solution.order for solution in solutions})),
                    "example_equations": " || ".join(f"{example.lhs} = {example.rhs}" for example in minus_examples),
                    "first_map": "; ".join(f"{pair}->{value}" for pair, value in representative.first_map),
                    "second_map": "; ".join(f"{pair}->{value}" for pair, value in representative.second_map),
                }
            )

    with output_csv.open("w", newline="") as handle:
        fieldnames = [
            "id",
            "query",
            "answer",
            "minus_example_count",
            "solution_count",
            "unique_prediction",
            "is_unique_prediction",
            "is_correct_unique_prediction",
            "families",
            "orders",
            "example_equations",
            "first_map",
            "second_map",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary_row in summary_rows:
            writer.writerow(summary_row)
            written_rows += 1

    payload = {
        "total_symbol_minus_rows": total_rows,
        "rows_with_minus_examples": rows_with_minus_examples,
        "rows_with_any_pair_rule_solution": rows_with_solutions,
        "rows_with_unique_prediction": rows_with_unique_prediction,
        "rows_with_correct_unique_prediction": rows_with_correct_unique_prediction,
        "rows_with_correct_unique_prediction_and_two_minus_examples": rows_with_correct_unique_prediction_and_two_examples,
        "rows_with_any_mirrored_solution": rows_with_mirrored_solution,
        "rows_with_correct_unique_mirrored_prediction": rows_with_correct_unique_mirrored_prediction,
        "output_csv": str(output_csv),
    }
    with summary_json.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
