#!/usr/bin/env python3
"""Build docs-only numeric-equation decision-point curriculum traces.

These rows are auxiliary continuation tasks. They are intentionally kept out of
the SFT CSV until the traces are reviewed.
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data/single_phase_training_clean/single_phase_sft.csv"
OUT_ROOT = ROOT / "docs/numeric_equation_decision_point_curriculum"
MANIFEST = OUT_ROOT / "manifest.csv"

RNG = random.Random(20260603)

TARGETS = {
    "common_intersection": 290,
    "output_format_rendering": 219,
    "literal_minus_rendering_policy": 120,
    "literal_minus_opposite_sign_continuation": 60,
    "low_confidence_branch_discipline": 170,
    "operator_absence_fallback_choice": 41,
}

COMMON_SUBTARGETS = {
    "none": 50,
    "single": 110,
    "double": 80,
    "triple_plus": 50,
}

EQUATION_RE = re.compile(r"^\s*(-?\d+)([^\d\s])(-?\d+)\s*$")
PROMPT_EXAMPLE_RE = re.compile(r"^(.+?)\s+=\s+(.+)$")
OUTPUT_FORMATS = {
    "plain",
    "rev",
    "op_prefix_if_neg",
    "rev_or_op_prefix_rev_if_neg",
    "op_prefix",
    "rev_or_op_suffix_rev_if_neg",
    "op_suffix",
    "op_prefix_rev",
    "abs_rev",
    "abs",
}
MATCH_SECTION_BOUNDARIES = (
    "Example ",
    "Common",
    "Try ",
    "Query",
    "All common",
    "Common output",
    "Candidate ",
    "Use ",
    "Apply ",
    "Answer:",
)


@dataclass(frozen=True)
class SourceRow:
    row_id: str
    prompt: str
    answer: str
    cot: str
    source: str
    source_mode: str


@dataclass(frozen=True)
class Candidate:
    bucket: str
    subbucket: str
    row: SourceRow
    split_pos: int
    pairing: str
    operator: str
    note: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.row.row_id, self.split_pos, self.bucket)

    @property
    def prompt_text(self) -> str:
        prefix = self.row.cot[: self.split_pos].rstrip()
        return f"{self.row.prompt.strip()}\n\nSolution:\n{prefix}"

    @property
    def target_text(self) -> str:
        return self.row.cot[self.split_pos :].lstrip()


def prompt_example_rhs_by_lhs(prompt: str) -> dict[str, str]:
    examples: dict[str, str] = {}
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Now, determine the result for:"):
            break
        match = PROMPT_EXAMPLE_RE.match(stripped)
        if not match:
            continue
        lhs, rhs = match.groups()
        examples[lhs.strip()] = rhs.strip()
    return examples


def normalize_cot_example_headers(prompt: str, cot: str) -> str:
    examples = prompt_example_rhs_by_lhs(prompt)
    if not examples:
        return cot

    fixed_lines: list[str] = []
    for line in cot.splitlines():
        if not line.startswith("Example ") or " = " not in line:
            fixed_lines.append(line)
            continue

        lhs, rhs = line[len("Example ") :].split(" = ", 1)
        lhs = lhs.strip()
        expected_rhs = examples.get(lhs)
        if expected_rhs is not None and rhs.strip() != expected_rhs:
            fixed_lines.append(f"Example {lhs} = {expected_rhs}")
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)


def normalize_cot_match_lists(cot: str) -> str:
    lines = cot.splitlines()
    fixed: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            not line.startswith("Example ")
            or " = " not in line
            or i + 3 >= len(lines)
            or lines[i + 3] != "Match"
        ):
            fixed.append(line)
            i += 1
            continue

        _, rhs = line[len("Example ") :].split(" = ", 1)
        header = lines[i + 1].split()
        values = lines[i + 2].split()
        if len(header) != len(values) or len(header) < 4:
            fixed.append(line)
            i += 1
            continue

        matches = [
            column
            for column, value in zip(header[3:], values[3:])
            if column in OUTPUT_FORMATS and value == rhs.strip()
        ]
        if not matches:
            matches = ["none"]

        fixed.extend([lines[i], lines[i + 1], lines[i + 2], lines[i + 3]])
        fixed.extend(matches)

        i += 4
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].startswith(MATCH_SECTION_BOUNDARIES)
        ):
            i += 1
    return "\n".join(fixed)


def read_sources() -> list[SourceRow]:
    rows: list[SourceRow] = []
    with SOURCE_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["category"] != "Numeric Equation Transformation Rules":
                continue
            if row["source_mode"] != "synthetic":
                continue
            cot = normalize_cot_example_headers(row["prompt"], row["generated_cot"].strip())
            cot = normalize_cot_match_lists(cot)
            if not cot:
                continue
            rows.append(
                SourceRow(
                    row_id=row["id"],
                    prompt=row["prompt"],
                    answer=row["answer"],
                    cot=cot,
                    source=row["source"],
                    source_mode=row["source_mode"],
                )
            )
    return rows


def last_pairing_before(text: str, pos: int) -> str:
    prefix = text[:pos]
    matches = list(re.finditer(r"The current format is (BA_DC|AB_CD)\|", prefix))
    if not matches:
        return "unknown"
    return matches[-1].group(1)


def query_operator_before(text: str, pos: int) -> str:
    prefix = text[:pos]
    operator_matches = list(re.finditer(r"Query operator is ([^\s])", prefix))
    if operator_matches:
        return operator_matches[-1].group(1)
    marker = prefix.rfind("\nQuery\n")
    if marker == -1 and prefix.startswith("Query\n"):
        marker = 0
    if marker == -1:
        marker = prefix.rfind("\nQuery ")
        if marker == -1:
            return "unknown"
        line = prefix[marker:].splitlines()[0]
        match = re.search(r"Query\s+(-?\d+)([^\d\s])(-?\d+)", line)
        return match.group(2) if match else "unknown"
    lines = prefix[marker:].splitlines()
    if len(lines) < 2:
        return "unknown"
    match = EQUATION_RE.match(lines[1].strip())
    return match.group(2) if match else "unknown"


def has_recent_query_block(text: str, pos: int, window: int = 900) -> bool:
    context = text[max(0, pos - window) : pos]
    return "\nQuery\n" in context or context.startswith("Query\n")


def common_subbucket(common_text: str) -> str:
    values = [line.strip() for line in common_text.strip().splitlines() if line.strip()]
    if values == ["none"]:
        return "none"
    if len(values) == 1:
        return "single"
    if len(values) == 2:
        return "double"
    return "triple_plus"


def common_candidates(rows: list[SourceRow]) -> list[Candidate]:
    out: list[Candidate] = []
    pattern = re.compile(r"\nCommon\n(?P<body>.*?)(?=\n\n)", re.DOTALL)
    for row in rows:
        for match in pattern.finditer(row.cot):
            body = match.group("body")
            sub = common_subbucket(body)
            pairing = last_pairing_before(row.cot, match.start())
            if pairing not in {"BA_DC", "AB_CD"}:
                continue
            out.append(
                Candidate(
                    bucket="common_intersection",
                    subbucket=sub,
                    row=row,
                    split_pos=match.start("body"),
                    pairing=pairing,
                    operator=query_operator_before(row.cot, match.start()),
                    note=f"common_{sub}_{pairing}",
                )
            )
    return out


def decision_line_candidates(rows: list[SourceRow], bucket: str) -> list[Candidate]:
    out: list[Candidate] = []
    pattern = re.compile(r"\n(?=(All common output formats|Common output formats) )")
    for row in rows:
        for match in pattern.finditer(row.cot):
            pos = match.start() + 1
            target = row.cot[pos:]
            pairing = last_pairing_before(row.cot, pos)
            operator = query_operator_before(row.cot, pos)
            sub = "agree" if target.startswith("All common output formats agree") else "disagree"
            out.append(
                Candidate(
                    bucket=bucket,
                    subbucket=sub,
                    row=row,
                    split_pos=pos,
                    pairing=pairing,
                    operator=operator,
                    note=f"{sub}_{pairing}_{operator}",
                )
            )
    return out


def output_format_candidates(rows: list[SourceRow]) -> list[Candidate]:
    candidates = decision_line_candidates(rows, "output_format_rendering")
    filtered: list[Candidate] = []
    for cand in candidates:
        if not has_recent_query_block(cand.row.cot, cand.split_pos):
            continue
        context = cand.row.cot[max(0, cand.split_pos - 800) : cand.split_pos]
        if any(mode in context for mode in ("op_prefix", "op_suffix", "rev_or_op", "abs_rev", "rev ")):
            filtered.append(cand)
    return filtered


def literal_minus_subbucket(target: str, pairing: str) -> str:
    if "literal - and x-y negative" in target:
        return f"{pairing}_x_y_negative_literal_priority"
    if "All common output formats agree" in target:
        return f"{pairing}_unanimous_literal_minus_rendering"
    if "x-y gives positive" in target:
        return f"{pairing}_x_y_positive_nonunanimous_policy"
    if "x-y gives negative" in target:
        return f"{pairing}_x_y_negative_nonunanimous_policy"
    return f"{pairing}_literal_minus_nonunanimous_policy"


def literal_minus_candidates(rows: list[SourceRow]) -> list[Candidate]:
    out: list[Candidate] = []
    for cand in decision_line_candidates(rows, "literal_minus_rendering_policy"):
        if not has_recent_query_block(cand.row.cot, cand.split_pos):
            continue
        if cand.operator != "-":
            continue
        context = cand.row.cot[max(0, cand.split_pos - 900) : cand.split_pos]
        target = cand.target_text[:900]
        if any(token in context + target for token in ("op_prefix", "op_suffix", "literal -", "opposite sign")):
            out.append(
                Candidate(
                    bucket=cand.bucket,
                    subbucket=literal_minus_subbucket(cand.target_text[:1400], cand.pairing),
                    row=cand.row,
                    split_pos=cand.split_pos,
                    pairing=cand.pairing,
                    operator=cand.operator,
                    note=cand.note,
                )
            )
    return out


def opposite_sign_candidates(rows: list[SourceRow]) -> list[Candidate]:
    out: list[Candidate] = []
    pattern = re.compile(r"\n(?=The common output formats agree on .+?opposite sign branch)", re.DOTALL)
    for row in rows:
        for match in pattern.finditer(row.cot):
            pos = match.start() + 1
            out.append(
                Candidate(
                    bucket="literal_minus_opposite_sign_continuation",
                    subbucket=f"{last_pairing_before(row.cot, pos)}_literal_minus",
                    row=row,
                    split_pos=pos,
                    pairing=last_pairing_before(row.cot, pos),
                    operator=query_operator_before(row.cot, pos),
                    note="opposite_sign",
                )
            )
    return out


def low_confidence_candidates(rows: list[SourceRow]) -> list[Candidate]:
    out: list[Candidate] = []
    for cand in decision_line_candidates(rows, "low_confidence_branch_discipline"):
        if not has_recent_query_block(cand.row.cot, cand.split_pos):
            continue
        target = cand.target_text[:1400]
        if not target.startswith("All common output formats do not agree"):
            continue
        if (
            "Vote winner" in target
            or "first common format in priority order" in target
            or "For motif BA_DC" in target
            or "For motif AB_CD" in target
        ):
            out.append(
                Candidate(
                    bucket=cand.bucket,
                    subbucket=f"{cand.pairing}_{cand.operator}",
                    row=cand.row,
                    split_pos=cand.split_pos,
                    pairing=cand.pairing,
                    operator=cand.operator,
                    note=cand.note,
                )
            )
    return out


def operator_absence_candidates(rows: list[SourceRow]) -> list[Candidate]:
    out: list[Candidate] = []
    pattern = re.compile(r"Skip operators already used by examples\n.*?\n(?=Choose )", re.DOTALL)
    for row in rows:
        if "Apply the inferred motif and output formats to the absent query operator" not in row.cot:
            continue
        for match in pattern.finditer(row.cot):
            pos = match.end()
            first = row.cot[pos:].lstrip().splitlines()[0] if row.cot[pos:].lstrip() else "Choose unknown"
            choice = first.replace("Choose ", "").strip().replace(" ", "_")
            out.append(
                Candidate(
                    bucket="operator_absence_fallback_choice",
                    subbucket=f"{last_pairing_before(row.cot, pos)}_choose_{choice}",
                    row=row,
                    split_pos=pos,
                    pairing=last_pairing_before(row.cot, pos),
                    operator=query_operator_before(row.cot, pos),
                    note="fallback_choice",
                )
            )
    return out


def sample_with_caps(
    candidates: list[Candidate],
    count: int,
    *,
    used: set[tuple[str, int]],
    cap_by_operator: int | None = None,
    cap_by_source: int | None = None,
) -> list[Candidate]:
    shuffled = candidates[:]
    RNG.shuffle(shuffled)
    selected: list[Candidate] = []
    op_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for cand in shuffled:
        key = (cand.row.row_id, cand.split_pos)
        if key in used:
            continue
        if cap_by_operator is not None and op_counts[cand.operator] >= cap_by_operator:
            continue
        if cap_by_source is not None and source_counts[cand.row.row_id] >= cap_by_source:
            continue
        selected.append(cand)
        used.add(key)
        op_counts[cand.operator] += 1
        source_counts[cand.row.row_id] += 1
        if len(selected) == count:
            return selected
    return selected


def sample_common(candidates: list[Candidate], used: set[tuple[str, int]]) -> list[Candidate]:
    selected: list[Candidate] = []
    for subbucket, total in COMMON_SUBTARGETS.items():
        for pairing, want in (("BA_DC", total // 2), ("AB_CD", total - total // 2)):
            pool = [c for c in candidates if c.subbucket == subbucket and c.pairing == pairing]
            picked = sample_with_caps(pool, want, used=used, cap_by_source=2)
            selected.extend(picked)
    short = TARGETS["common_intersection"] - len(selected)
    if short > 0:
        leftovers = [c for c in candidates if (c.row.row_id, c.split_pos) not in used]
        selected.extend(sample_with_caps(leftovers, short, used=used, cap_by_source=2))
    return selected


def write_candidate(index: int, cand: Candidate) -> dict[str, str]:
    trace_id = f"syn_ne_dp_{cand.bucket}_{index:04d}"
    folder = OUT_ROOT / cand.bucket
    path = folder / f"{trace_id}.txt"
    prompt = cand.prompt_text
    target = cand.target_text
    text = "\n".join(
        [
            f"Problem {trace_id}",
            f"Bucket: {cand.bucket}",
            f"Subbucket: {cand.subbucket}",
            f"Source id: {cand.row.row_id}",
            f"Source mode: {cand.row.source_mode}",
            f"Source path: {cand.row.source}",
            f"Gold answer: {cand.row.answer}",
            f"Pairing: {cand.pairing}",
            f"Operator: {cand.operator}",
            f"Note: {cand.note}",
            "",
            "Training prompt:",
            prompt,
            "",
            "Training target:",
            target,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "id": trace_id,
        "bucket": cand.bucket,
        "subbucket": cand.subbucket,
        "source_id": cand.row.row_id,
        "source_mode": cand.row.source_mode,
        "source_path": cand.row.source,
        "answer": cand.row.answer,
        "pairing": cand.pairing,
        "operator": cand.operator,
        "split_pos": str(cand.split_pos),
        "path": str(path.relative_to(ROOT)),
        "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
        "target_sha1": hashlib.sha1(target.encode("utf-8")).hexdigest(),
    }


def audit_manifest(rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    counts = Counter(row["bucket"] for row in rows)
    for bucket, expected in TARGETS.items():
        if counts[bucket] != expected:
            issues.append(f"{bucket}: expected {expected}, found {counts[bucket]}")
    if len(rows) != sum(TARGETS.values()):
        issues.append(f"total: expected {sum(TARGETS.values())}, found {len(rows)}")
    seen_ids: set[str] = set()
    seen_prompt_targets: set[tuple[str, str]] = set()
    for row in rows:
        path = ROOT / row["path"]
        text = path.read_text(encoding="utf-8")
        if row["id"] in seen_ids:
            issues.append(f"duplicate id: {row['id']}")
        seen_ids.add(row["id"])
        if "Training prompt:\n" not in text or "\n\nTraining target:\n" not in text:
            issues.append(f"missing prompt/target sections: {row['path']}")
        prompt, target = text.split("\n\nTraining target:\n", 1)
        prompt = prompt.split("Training prompt:\n", 1)[1].strip()
        target = target.strip()
        if not target:
            issues.append(f"empty target: {row['path']}")
        prompt_target = (prompt, target)
        if prompt_target in seen_prompt_targets:
            issues.append(f"duplicate prompt+target body: {row['path']}")
        seen_prompt_targets.add(prompt_target)
        if row["source_mode"] != "synthetic":
            issues.append(f"non-synthetic source: {row['path']}")
    return issues


def main() -> None:
    rows = read_sources()
    pools = {
        "common_intersection": common_candidates(rows),
        "output_format_rendering": output_format_candidates(rows),
        "literal_minus_rendering_policy": literal_minus_candidates(rows),
        "literal_minus_opposite_sign_continuation": opposite_sign_candidates(rows),
        "low_confidence_branch_discipline": low_confidence_candidates(rows),
        "operator_absence_fallback_choice": operator_absence_candidates(rows),
    }

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    used: set[tuple[str, int]] = set()
    selected: list[Candidate] = []
    selected.extend(sample_common(pools["common_intersection"], used))
    selected.extend(
        sample_with_caps(
            pools["output_format_rendering"],
            TARGETS["output_format_rendering"],
            used=used,
            cap_by_operator=18,
            cap_by_source=2,
        )
    )
    selected.extend(
        sample_with_caps(
            pools["literal_minus_rendering_policy"],
            TARGETS["literal_minus_rendering_policy"],
            used=used,
            cap_by_source=2,
        )
    )
    selected.extend(
        sample_with_caps(
            pools["literal_minus_opposite_sign_continuation"],
            TARGETS["literal_minus_opposite_sign_continuation"],
            used=used,
            cap_by_source=2,
        )
    )
    selected.extend(
        sample_with_caps(
            pools["low_confidence_branch_discipline"],
            TARGETS["low_confidence_branch_discipline"],
            used=used,
            cap_by_operator=25,
            cap_by_source=2,
        )
    )
    selected.extend(
        sample_with_caps(
            pools["operator_absence_fallback_choice"],
            TARGETS["operator_absence_fallback_choice"],
            used=used,
            cap_by_source=2,
        )
    )

    manifest_rows = [write_candidate(i + 1, cand) for i, cand in enumerate(selected)]
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("pool sizes")
    for bucket, pool in pools.items():
        print(f"{bucket}: {len(pool)}")
    print("\nselected")
    for bucket, count in Counter(row["bucket"] for row in manifest_rows).items():
        print(f"{bucket}: {count}")
    print("\ncommon subbuckets")
    for key, count in Counter(
        row["subbucket"] for row in manifest_rows if row["bucket"] == "common_intersection"
    ).items():
        print(f"{key}: {count}")
    print("\noperator counts for output rendering")
    for key, count in Counter(
        row["operator"] for row in manifest_rows if row["bucket"] == "output_format_rendering"
    ).most_common():
        print(f"{key}: {count}")

    issues = audit_manifest(manifest_rows)
    print("\naudit")
    if issues:
        for issue in issues:
            print(f"ISSUE: {issue}")
        raise SystemExit(1)
    print("ok")


if __name__ == "__main__":
    main()
