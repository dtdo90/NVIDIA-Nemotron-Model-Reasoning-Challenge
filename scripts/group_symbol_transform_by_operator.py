from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nemotron_baseline.numeric_equation import classify_equation_vs_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group symbol_transform rows by the middle operator in 5-character queries."
    )
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--stats-csv", default="data/symbol_transform_operator_stats.csv")
    parser.add_argument("--grouped-csv", default="data/symbol_transform_operator_groups.csv")
    parser.add_argument("--groups-json", default="data/symbol_transform_operator_groups.json")
    return parser.parse_args()


def extract_query(prompt: str) -> str | None:
    marker = "Now, determine the result for: "
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            return stripped.split(marker, 1)[1].strip()
    return None


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv)
    stats_csv = Path(args.stats_csv)
    grouped_csv = Path(args.grouped_csv)
    groups_json = Path(args.groups_json)

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    operator_counts: Counter[str] = Counter()

    with train_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            if classify_equation_vs_symbol(prompt) != "symbol_transform":
                continue

            query = extract_query(prompt)
            if query is None or len(query) != 5:
                continue

            operator = query[2]
            enriched_row = {
                "id": row["id"],
                "query": query,
                "answer": row["answer"],
                "operator": operator,
                "prompt": prompt,
            }
            grouped_rows[operator].append(enriched_row)
            operator_counts[operator] += 1

    sorted_operators = sorted(operator_counts.items(), key=lambda item: (-item[1], item[0]))

    with stats_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["operator", "count", "example_id", "example_query", "example_answer"],
        )
        writer.writeheader()
        for operator, count in sorted_operators:
            first_example = grouped_rows[operator][0]
            writer.writerow(
                {
                    "operator": operator,
                    "count": count,
                    "example_id": first_example["id"],
                    "example_query": first_example["query"],
                    "example_answer": first_example["answer"],
                }
            )

    with grouped_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["operator", "id", "query", "answer", "prompt"],
        )
        writer.writeheader()
        for operator, _count in sorted_operators:
            for row in grouped_rows[operator]:
                writer.writerow(row)

    json_payload = {
        "group_key": "middle_operator",
        "query_shape": "ABOCD",
        "total_rows": sum(operator_counts.values()),
        "operators": [
            {
                "operator": operator,
                "count": count,
                "ids": [row["id"] for row in grouped_rows[operator]],
            }
            for operator, count in sorted_operators
        ],
    }
    with groups_json.open("w") as handle:
        json.dump(json_payload, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {len(sorted_operators)} operator groups covering {sum(operator_counts.values())} rows.")
    print(f"Stats CSV: {stats_csv}")
    print(f"Grouped CSV: {grouped_csv}")
    print(f"Groups JSON: {groups_json}")
    for operator, count in sorted_operators:
        print(f"{operator!r}: {count}")


if __name__ == "__main__":
    main()
