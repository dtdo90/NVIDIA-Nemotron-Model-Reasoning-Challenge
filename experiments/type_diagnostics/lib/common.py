from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "experiments/type_diagnostics"
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SOURCE_CSV = ROOT / "data/single_phase_training_clean/single_phase_sft.csv"
DATA_DIR = WORKSPACE / "data"
OUTPUT_DIR = WORKSPACE / "outputs"
REPORT_DIR = WORKSPACE / "reports"
SPLIT_NAMES = ("sft_train", "eval_holdout", "grpo_holdout")
TRAIN_ONLY_SOURCE_MODES = {
    "huikang_real_bit_extra_trace",
    "huikang_synthetic_matching",
    "phase1_synthetic_direct_template",
    "single_phase_synthetic_direct_template",
    "single_phase_synthetic_text_cipher_confusion",
    "synthetic",
}

QUESTION_TYPES: dict[str, dict[str, str]] = {
    "bit_manipulation": {
        "category": "Bit Manipulation",
        "display": "Bit Manipulation",
    },
    "gravity": {
        "category": "Gravity",
        "display": "Gravity",
    },
    "unit_conversion": {
        "category": "Unit Conversion",
        "display": "Unit Conversion",
    },
    "text_cipher": {
        "category": "Text Cipher",
        "display": "Text Cipher",
    },
    "numeral_system": {
        "category": "Numeral System",
        "display": "Numeral System",
    },
    "numeric_equation": {
        "category": "Numeric Equation Transformation Rules",
        "display": "Numeric Equation Transformation Rules",
    },
    "symbol_transform": {
        "category": "Symbol Transform",
        "display": "Symbol Transform",
    },
}

CATEGORY_TO_SLUG = {
    payload["category"]: slug for slug, payload in QUESTION_TYPES.items()
}


@dataclass(frozen=True)
class TypePaths:
    slug: str
    data_dir: Path
    train_csv: Path
    split_csv: Path
    summary_json: Path
    output_dir: Path
    report_dir: Path


def type_paths(question_type: str, *, data_dir: Path = DATA_DIR) -> TypePaths:
    slug = normalize_question_type(question_type)
    root = Path(data_dir) / slug
    workspace_root = Path(data_dir).parent if Path(data_dir).name == "data" else WORKSPACE
    return TypePaths(
        slug=slug,
        data_dir=root,
        train_csv=root / f"{slug}.csv",
        split_csv=root / "splits_80_10_10.csv",
        summary_json=root / "dataset_summary.json",
        output_dir=workspace_root / "outputs" / slug,
        report_dir=workspace_root / "reports" / slug,
    )


def normalize_question_type(question_type: str) -> str:
    slug = question_type.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bit": "bit_manipulation",
        "bit_manipulation": "bit_manipulation",
        "unit": "unit_conversion",
        "unit_conversion": "unit_conversion",
        "text": "text_cipher",
        "cipher": "text_cipher",
        "text_cipher": "text_cipher",
        "numeral": "numeral_system",
        "numeral_system": "numeral_system",
        "numeric": "numeric_equation",
        "numeric_equation": "numeric_equation",
        "numeric_equation_transformation_rules": "numeric_equation",
        "symbol": "symbol_transform",
        "symbol_equation": "symbol_transform",
        "symbol_transform": "symbol_transform",
    }
    slug = aliases.get(slug, slug)
    if slug not in QUESTION_TYPES:
        valid = ", ".join(sorted(QUESTION_TYPES))
        raise SystemExit(f"Unknown question type {question_type!r}. Valid choices: {valid}")
    return slug


def read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_csv_rows(path: str | Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_type_rows(question_type: str, source_csv: Path = SOURCE_CSV) -> tuple[list[dict[str, str]], list[str]]:
    slug = normalize_question_type(question_type)
    category = QUESTION_TYPES[slug]["category"]
    rows, fieldnames = read_csv_rows(source_csv)
    type_rows = [row for row in rows if row.get("category") == category]
    if not type_rows:
        raise SystemExit(f"No rows found for {category!r} in {source_csv}")
    return type_rows, fieldnames


def safe_label(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("|", "_").replace("+", "plus").replace("-", "minus")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def source_parts(source: str) -> list[str]:
    normalized = source.replace("\\", "/")
    if "#" in normalized:
        base, *fragments = normalized.split("#")
        return [*base.split("/"), *fragments]
    return normalized.split("/")


def source_key(row: dict[str, str]) -> str:
    source = row.get("source", "")
    parts = source_parts(source)
    for marker in (
        "work_on",
        "synthetic",
        "numeric_equation_transformation",
        "symbol_transform",
    ):
        if marker in parts:
            index = parts.index(marker)
            tail = parts[index + 1 :]
            if tail and "." in tail[-1]:
                tail = tail[:-1]
            return "/".join(tail) or marker
    if "#" in source:
        return source.split("#", 1)[1]
    return source or "unknown"


def template_pass(text: str) -> str | None:
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip().lower().replace(" ", "")
        if line == "template0134passesallexamples":
            return "template0134"
        if line == "template3401passesallexamples":
            return "template3401"
        if line == "trytemplate0134":
            current = "template0134"
            continue
        if line == "trytemplate3401":
            current = "template3401"
            continue
        if line == "pass" and current:
            return current
        if line.startswith("try") and "template" not in line:
            current = None
    return None


def classify_numeric_subtype(row: dict[str, str]) -> str:
    key = source_key(row).lower()
    if key.startswith("synthetic/"):
        key = key.removeprefix("synthetic/")
    text = f"{row.get('generated_cot', '')}\n{row.get('assistant_content', '')}"

    if "direct_template" in key:
        return f"direct_template_{template_pass(text) or 'unknown'}"
    if "operator_absence" in key:
        return "operator_absence"
    if "prefix_postfix" in key:
        parts = key.split("/")
        if len(parts) >= 2 and parts[1]:
            return f"prefix_postfix_{safe_label(parts[1])}"
        return "prefix_postfix"
    if key == "modular" or key.startswith("modular/"):
        return "modular"
    if "low_confidence" in key:
        parts = [safe_label(part) for part in key.split("/") if part]
        if parts and parts[0] == "low_confidence":
            return "_".join(parts[:2]) if len(parts) > 1 else "low_confidence"
        return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]

    top_level_prefixes = (
        "ba_dc_multiplication",
        "ba_dc_addition",
        "ba_dc_subtraction",
        "ba_dc_modular",
        "ab_cd_multiplication",
        "ab_cd_addition",
        "ab_cd_subtraction",
        "ab_cd_modular",
    )
    for prefix in top_level_prefixes:
        if key == prefix or key.startswith(prefix + "/"):
            parts = [safe_label(part) for part in key.split("/") if part]
            return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]

    return safe_label(key)


def classify_symbol_subtype(row: dict[str, str]) -> str:
    key = source_key(row).lower()
    text = f"{row.get('generated_cot', '')}\n{row.get('assistant_content', '')}"
    if "ba_dc" in key:
        return "ba_dc"
    if "direct_template" in key:
        return f"direct_template_{template_pass(text) or 'unknown'}"
    return safe_label(key)


def classify_bit_subtype(row: dict[str, str]) -> str:
    source_mode = row.get("source_mode", "")
    if source_mode == "huikang_real_bit":
        return "huikang_real_rule_found"
    if source_mode == "huikang_real_bit_extra_trace":
        return "huikang_real_extra_trace"
    if source_mode == "huikang_synthetic_matching":
        return "huikang_synthetic_matching"
    if source_mode == "synthetic":
        return "huikang_synthetic_legacy"
    return safe_label(source_mode or "unknown")


def classify_subtype(row: dict[str, str]) -> str:
    category = row.get("category", "")
    if category == "Numeric Equation Transformation Rules":
        return classify_numeric_subtype(row)
    if category == "Symbol Transform":
        return classify_symbol_subtype(row)
    if category == "Bit Manipulation":
        return classify_bit_subtype(row)
    if category in {"Gravity", "Unit Conversion", "Numeral System", "Text Cipher"}:
        return "standard"
    return safe_label(category)


def split_counts(size: int) -> dict[str, int]:
    if size <= 0:
        return {name: 0 for name in SPLIT_NAMES}
    if size == 1:
        return {"sft_train": 1, "eval_holdout": 0, "grpo_holdout": 0}
    if size == 2:
        return {"sft_train": 1, "eval_holdout": 1, "grpo_holdout": 0}
    if size == 3:
        return {"sft_train": 2, "eval_holdout": 1, "grpo_holdout": 0}
    if size < 10:
        return {"sft_train": size - 2, "eval_holdout": 1, "grpo_holdout": 1}

    eval_count = max(1, round(size * 0.10))
    grpo_count = max(1, round(size * 0.10))
    train_count = size - eval_count - grpo_count
    return {
        "sft_train": train_count,
        "eval_holdout": eval_count,
        "grpo_holdout": grpo_count,
    }


def is_eval_eligible(row: dict[str, str]) -> bool:
    if row.get("source_mode", "") in TRAIN_ONLY_SOURCE_MODES:
        return False
    value = (row.get("eval_eligible") or "true").strip().lower()
    return value not in {"0", "false", "no", "n", "off"}


def is_train_only(row: dict[str, str]) -> bool:
    return (
        row.get("source_mode", "") in TRAIN_ONLY_SOURCE_MODES
        or (row.get("split_policy") or "").strip().lower() == "train_only"
        or not is_eval_eligible(row)
    )


def build_stratified_splits(rows: list[dict[str, str]], *, seed: int = 42) -> dict[str, str]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if is_train_only(row):
            continue
        key = f"{row['diagnostic_subtype']}|{row.get('source_mode', 'unknown')}"
        grouped[key].append(row)

    assignments: dict[str, str] = {}
    for row in rows:
        if is_train_only(row):
            assignments[row["id"]] = "sft_train"

    for key in sorted(grouped):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        counts = split_counts(len(bucket))
        offset = 0
        for split_name in SPLIT_NAMES:
            next_offset = offset + counts[split_name]
            for row in bucket[offset:next_offset]:
                assignments[row["id"]] = split_name
            offset = next_offset

    missing = [row["id"] for row in rows if row["id"] not in assignments]
    if missing:
        raise RuntimeError(f"Split assignment missed {len(missing)} ids: {missing[:5]}")
    return assignments


def summarize_rows(rows: list[dict[str, str]], assignments: dict[str, str] | None = None) -> dict[str, object]:
    subtype_counts = Counter(row["diagnostic_subtype"] for row in rows)
    source_counts = Counter(row.get("source_mode", "unknown") for row in rows)
    eval_eligible_counts = Counter("true" if is_eval_eligible(row) else "false" for row in rows)
    split_policy_counts = Counter(row.get("split_policy", "auto") or "auto" for row in rows)
    payload: dict[str, object] = {
        "total": len(rows),
        "by_subtype": dict(sorted(subtype_counts.items())),
        "by_source_mode": dict(sorted(source_counts.items())),
        "by_eval_eligible": dict(sorted(eval_eligible_counts.items())),
        "by_split_policy": dict(sorted(split_policy_counts.items())),
    }
    if assignments:
        split_counts_payload = Counter(assignments[row["id"]] for row in rows)
        by_split_subtype: dict[str, dict[str, int]] = {}
        for split_name in SPLIT_NAMES:
            split_rows = [row for row in rows if assignments[row["id"]] == split_name]
            by_split_subtype[split_name] = dict(
                sorted(Counter(row["diagnostic_subtype"] for row in split_rows).items())
            )
        payload.update(
            {
                "split_counts": dict(sorted(split_counts_payload.items())),
                "by_split_subtype": by_split_subtype,
            }
        )
    return payload


def load_split_assignments(split_csv: str | Path) -> dict[str, str]:
    rows, _ = read_csv_rows(split_csv)
    assignments: dict[str, str] = {}
    for row in rows:
        row_id = row["id"]
        if row_id in assignments:
            raise ValueError(f"Duplicate split assignment for id={row_id}")
        assignments[row_id] = row["split"]
    return assignments


def validate_split_assignments(
    rows: list[dict[str, str]],
    assignments: dict[str, str],
    *,
    split_csv: str | Path,
) -> None:
    row_ids = [row["id"] for row in rows]
    duplicate_ids = [row_id for row_id, count in Counter(row_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"{split_csv} cannot be validated because data has duplicate ids: {duplicate_ids[:5]}")

    row_id_set = set(row_ids)
    assignment_ids = set(assignments)
    missing = sorted(row_id_set - assignment_ids)
    extra = sorted(assignment_ids - row_id_set)
    if missing or extra:
        raise ValueError(
            f"Split assignments in {split_csv} do not match data ids. "
            f"Missing={missing[:5]} extra={extra[:5]}"
        )

    invalid_splits = sorted(set(assignments.values()) - set(SPLIT_NAMES))
    if invalid_splits:
        raise ValueError(f"{split_csv} has invalid split names: {invalid_splits}")


FRESHNESS_COLUMNS = (
    "prompt",
    "answer",
    "generated_cot",
    "assistant_content",
    "label",
    "category",
    "source",
    "source_mode",
    "eval_eligible",
    "split_policy",
    "append_answer_instruction",
)


def type_dataset_freshness_issues(
    question_type: str,
    *,
    type_csv: str | Path,
    source_csv: str | Path = SOURCE_CSV,
) -> list[str]:
    """Return human-readable issues if a cached type CSV differs from source."""
    slug = normalize_question_type(question_type)
    source_rows, _ = load_type_rows(slug, Path(source_csv))
    type_rows, _ = read_csv_rows(type_csv)
    source_by_id = {row["id"]: row for row in source_rows}
    type_by_id = {row["id"]: row for row in type_rows}

    issues: list[str] = []
    missing = sorted(set(source_by_id) - set(type_by_id))
    extra = sorted(set(type_by_id) - set(source_by_id))
    if missing:
        issues.append(f"missing ids from type CSV: {missing[:5]}")
    if extra:
        issues.append(f"extra ids in type CSV: {extra[:5]}")

    changed: list[str] = []
    for row_id in sorted(set(source_by_id) & set(type_by_id)):
        source_row = source_by_id[row_id]
        type_row = type_by_id[row_id]
        for column in FRESHNESS_COLUMNS:
            if source_row.get(column, "") != type_row.get(column, ""):
                changed.append(f"{row_id}:{column}")
                break
        expected_subtype = classify_subtype(source_row)
        if type_row.get("diagnostic_subtype", "") != expected_subtype:
            changed.append(f"{row_id}:diagnostic_subtype")
    if changed:
        issues.append(f"stale row content: {changed[:5]}")
    return issues


def assert_type_dataset_fresh(
    question_type: str,
    *,
    type_csv: str | Path,
    source_csv: str | Path = SOURCE_CSV,
) -> None:
    issues = type_dataset_freshness_issues(
        question_type,
        type_csv=type_csv,
        source_csv=source_csv,
    )
    if not issues:
        return
    issue_text = "\n  - ".join(issues)
    raise SystemExit(
        f"{type_csv} is stale relative to {source_csv}.\n"
        f"  - {issue_text}\n"
        "Regenerate diagnostics with:\n"
        "  python3 experiments/type_diagnostics/prepare_type_datasets.py"
    )


def select_rows_for_splits(
    rows: list[dict[str, str]],
    assignments: dict[str, str],
    split_names: list[str],
) -> list[dict[str, str]]:
    wanted = set(split_names)
    return [row for row in rows if assignments.get(row["id"]) in wanted]
