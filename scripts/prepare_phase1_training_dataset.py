#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMPONENTS = [
    "data/trainable/phase1_components/text_knowledge_phase1.csv",
    "data/trainable/phase1_components/phase1a_bit_manipulation_knowledge.csv",
    "data/trainable/phase1_components/phase1b_bit_manipulation_methodology.csv",
    "data/trainable/phase1_components/phase1_numeric_equation_curriculum.csv",
    "data/trainable/phase1_components/phase1_symbol_transform_direct_curriculum.csv",
]

REQUIRED_COLUMNS = {"id", "prompt", "answer", "generated_cot", "label", "category"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine Phase 1 knowledge and methodology component CSVs into the single "
            "starter-loop Phase 1 training dataset. Component files remain separate "
            "for later Phase 1A/1B ablations."
        )
    )
    parser.add_argument(
        "--component-csv",
        action="append",
        default=None,
        help=(
            "Component CSV to include. Repeat to override the default active component "
            "set. Defaults to text knowledge, bit components, the merged numeric "
            "curriculum, and the symbol-equation direct curriculum."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="data/trainable/phase1_train.csv",
        help="Combined Phase 1 training CSV.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/trainable/phase1_train.summary.json",
        help="Summary of included components and row counts.",
    )
    return parser.parse_args()


def read_component(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def phase1_role(filename: str) -> str:
    if filename == "text_knowledge_phase1.csv":
        return "stable_fact"
    if filename.startswith("phase1a_"):
        return "phase1a_fact"
    if filename.startswith("phase1b_"):
        return "phase1b_method"
    return "phase1_component"


def main() -> None:
    args = parse_args()
    component_paths = [
        ROOT / item for item in (args.component_csv if args.component_csv else DEFAULT_COMPONENTS)
    ]
    output_path = ROOT / args.output_csv
    summary_path = ROOT / args.summary_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    for component_path in component_paths:
        if not component_path.exists():
            raise SystemExit(f"Phase 1 component not found: {component_path}")
        component_name = component_path.name
        role = phase1_role(component_name)
        rows = read_component(component_path)
        for row in rows:
            row_id = row["id"].strip()
            if not row_id:
                raise SystemExit(f"{component_path} contains an empty id.")
            if row_id in seen_ids:
                raise SystemExit(f"Duplicate Phase 1 id {row_id!r} from {component_path}")
            seen_ids.add(row_id)
            copied = dict(row)
            copied["phase1_component"] = component_name
            copied["phase1_role"] = role
            rows_out.append(copied)
            source_counts[component_name] += 1
            role_counts[role] += 1
            category_counts[copied.get("category", "")] += 1

    fieldnames = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "label",
        "category",
        "source",
        "source_category",
        "phase1_component",
        "phase1_role",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "output_csv": args.output_csv,
        "total_rows": len(rows_out),
        "components": dict(source_counts),
        "roles": dict(role_counts),
        "categories": dict(category_counts),
        "excluded_by_default": [
            "phase1a_unit_conversion_knowledge.csv",
            "phase1a_gravity_knowledge.csv",
            "phase1a_numeral_knowledge.csv",
            "phase1a_numeric_equation_knowledge.csv",
            "phase1b_numeric_equation_methodology.csv",
            "phase1a_symbol_transform_knowledge.csv",
            "phase1b_symbol_transform_methodology.csv",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows_out)} Phase 1 training rows to {output_path}")
    for name, count in source_counts.items():
        print(f"  {name}: {count}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
