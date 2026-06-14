#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HARNESS = ROOT / "reference/cursor/transformation_rules/numeric_equation/harness"
for path in (SRC, HARNESS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extended_dsl import BASE_RULE_BY_NAME, apply_pairing  # noqa: E402
from nemotron_baseline.numeric_equation import (  # noqa: E402
    NumericEquationPuzzle,
    ParsedEquation,
    parse_numeric_equation_puzzle,
)


OUT_DIR = ROOT / "docs/numeric_equation_motif_override_traces"
EXCLUDED_PATTERN_MANIFEST = OUT_DIR / "_excluded_synthetic_opposite_sign_common_agreement.md"
EXCLUDED_UNSTABLE_MANIFEST = OUT_DIR / "_excluded_unstable_motif_drift_ids.md"
GOLDEN_PATHS = [
    ROOT / "data/single_phase_training_clean/single_phase_sft_v2.csv",
    ROOT / "data/single_phase_training_clean/single_phase_sft_v1.csv",
]
V2_PATH = ROOT / "data/single_phase_training_clean/single_phase_sft_v2.csv"
EXCLUDED_TARGET_IDS = {
    # The fewest-example helper group sends this case down a non-gold path.
    # Keep the helper selection rule truthful by excluding it for now.
    "e5956ffa",
    # BA_DC|x-y negative would use reverse magnitude plus operator prefix,
    # but these two gold answers follow the non-deterministic vote result.
    "2049f01d",
    "5c008804",
    "87711597",
    "891942ba",
    "92471ca4",
    "c7a7b13a",
    "db5a5b71",
    "e247d364",
}
UNSTABLE_MOTIF_DRIFT_IDS = {
    # Earlier supported candidates are skipped in v2 with no explicit policy.
    # Leave out until we define a faithful operator-family selection rule.
    "00d8b3db",
    "2beb5851",
    "febd3442",
    # Common-output unanimity would finalize to a non-gold answer under the
    # simplified rule; leave out until we define a faithful continuation rule.
    "12d4a2df",
    "1b6366af",
    "7ac90433",
    # Current renderer exposes earlier AB_CD-supported candidates that do not
    # reach gold; do not keep old copied traces for these.
    "31eb8247",
    "9a5b6b28",
    # With literal '-' op-prefix rendering fixed, an earlier BA_DC x-y
    # candidate becomes supported but does not reach gold.
    "91b34547",
}

BASE_LABEL = {
    "x + y": "x+y",
    "x + y - 1": "x+y-1",
    "x + y + 1": "x+y+1",
    "x - y": "x-y",
    "y - x": "y-x",
    "|x - y|": "abs(x-y)",
    "min(x,y)-max(x,y)": "min(x,y)-max(x,y)",
    "max(x,y)%min(x,y)": "max(x,y)%min(x,y)",
    "x mod y": "x%y",
    "y mod x": "y%x",
    "x * y": "x*y",
    "x * y + 1": "x*y+1",
    "x * y - 1": "x*y-1",
    "concat(x, y)": "template0134",
    "concat(y, x)": "template3401",
}
BASE_FROM_LABEL = {label: base for base, label in BASE_LABEL.items()}

OP_ABSENCE_SYMBOL_ORDER = [
    "+",
    "-",
    "*",
    "/",
    ":",
    "@",
    "[",
    "\\",
    "%",
    "`",
    "!",
    ")",
    "]",
    "^",
    "|",
    "}",
    "#",
    "&",
    "'",
    "(",
    "?",
    "{",
    '"',
    "$",
    "<",
    ">",
]

OP_ABSENCE_SYMBOL_MAPPING = {
    "+": "x + y",
    "-": "x - y",
    "*": "x * y",
    "/": "x * y",
    ":": "concat(x, y)",
    "@": "x + y",
    "[": "x + y",
    "\\": "x - y",
    "%": "x * y",
    "`": "x + y",
    "!": "x + y",
    ")": "x + y",
    "]": "x - y",
    "^": "concat(x, y)",
    "|": "x + y - 1",
    "}": "x - y",
    "#": "concat(x, y)",
    "&": "x + y",
    "'": "x + y",
    "(": "x * y",
    "?": "x - y",
    "{": "x * y - 1",
    '"': "x + y",
    "$": "x - y",
    "<": "x * y",
    ">": "x - y",
}

OP_ABSENCE_SYMBOL_CANDIDATES = {
    "+": ["x + y", "|x - y|"],
    "-": ["x - y"],
    "*": ["x * y", "x + y + 1"],
    "/": ["x * y", "concat(x, y)", "min(x,y)-max(x,y)"],
    ":": ["concat(x, y)", "x - y"],
    "@": ["x + y", "y - x"],
    "[": ["x + y", "x * y"],
    "\\": ["x - y", "x + y", "x * y - 1", "y - x"],
    "%": ["x * y", "x + y"],
    "`": ["x + y"],
    "!": ["x + y", "x * y"],
    ")": ["x + y", "x - y"],
    "]": ["x - y"],
    "^": ["concat(x, y)", "x * y - 1", "y - x"],
    "|": ["x + y - 1"],
    "}": ["x - y", "concat(x, y)"],
    "#": ["concat(x, y)", "x + y"],
    "&": ["x + y"],
    "'": ["x + y"],
    "(": ["x * y", "max(x,y)%min(x,y)"],
    "?": ["x - y", "concat(x, y)"],
    "{": ["x * y - 1"],
    '"': ["x + y"],
    "$": ["x - y"],
    "<": ["x * y"],
    ">": ["x - y"],
}

OPPOSITE_SIGN_BRANCH_BASES = {
    "x - y",
    "y - x",
    "min(x,y)-max(x,y)",
}

RAW_HEADER = {
    ("BA_DC", "x + y"): "BA+DC",
    ("BA_DC", "x + y - 1"): "BA+DC-1",
    ("BA_DC", "x + y + 1"): "BA+DC+1",
    ("BA_DC", "x - y"): "BA-DC",
    ("BA_DC", "y - x"): "DC-BA",
    ("BA_DC", "|x - y|"): "abs(BA-DC)",
    ("BA_DC", "min(x,y)-max(x,y)"): "min(BA,DC)-max(BA,DC)",
    ("BA_DC", "max(x,y)%min(x,y)"): "max(BA,DC)%min(BA,DC)",
    ("BA_DC", "x mod y"): "BA%DC",
    ("BA_DC", "y mod x"): "DC%BA",
    ("BA_DC", "x * y"): "BA*DC",
    ("BA_DC", "x * y + 1"): "BA*DC+1",
    ("BA_DC", "x * y - 1"): "BA*DC-1",
    ("BA_DC", "concat(x, y)"): "template0134(BA,DC)",
    ("BA_DC", "concat(y, x)"): "template3401(BA,DC)",
    ("AB_CD", "x + y"): "AB+CD",
    ("AB_CD", "x + y - 1"): "AB+CD-1",
    ("AB_CD", "x + y + 1"): "AB+CD+1",
    ("AB_CD", "x - y"): "AB-CD",
    ("AB_CD", "y - x"): "CD-AB",
    ("AB_CD", "|x - y|"): "abs(AB-CD)",
    ("AB_CD", "min(x,y)-max(x,y)"): "min(AB,CD)-max(AB,CD)",
    ("AB_CD", "max(x,y)%min(x,y)"): "max(AB,CD)%min(AB,CD)",
    ("AB_CD", "x mod y"): "AB%CD",
    ("AB_CD", "y mod x"): "CD%AB",
    ("AB_CD", "x * y"): "AB*CD",
    ("AB_CD", "x * y + 1"): "AB*CD+1",
    ("AB_CD", "x * y - 1"): "AB*CD-1",
    ("AB_CD", "concat(x, y)"): "template0134(AB,CD)",
    ("AB_CD", "concat(y, x)"): "template3401(AB,CD)",
}

MODES = {
    "BA_DC": [
        "rev",
        "plain",
        "op_prefix_if_neg",
        "rev_or_op_prefix_rev_if_neg",
        "op_prefix",
        "rev_or_op_suffix_rev_if_neg",
        "op_suffix",
        "op_prefix_rev",
        "abs_rev",
        "abs",
    ],
    "AB_CD": [
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
    ],
}

OP_ABSENCE_MODES = {
    pairing: [
        mode
        for mode in [
            *modes[: modes.index("rev_or_op_suffix_rev_if_neg")],
            "rev_or_op_suffix_if_neg",
            *modes[modes.index("rev_or_op_suffix_rev_if_neg") :],
        ]
        if mode not in {"op_prefix", "op_suffix", "op_prefix_rev"}
    ]
    for pairing, modes in MODES.items()
}
OP_ABSENCE_CANDIDATE_OUTPUT_FORMATS = [
    "rev",
    "plain",
    "op_prefix_if_neg",
    "rev_or_op_prefix_rev_if_neg",
    "rev_or_op_suffix_if_neg",
    "rev_or_op_suffix_rev_if_neg",
    "abs_rev",
    "abs",
]

WIDTH_PRESERVING_BASES = {
    "x + y",
    "x + y - 1",
    "x + y + 1",
    "x - y",
    "y - x",
    "|x - y|",
    "min(x,y)-max(x,y)",
}


def has_leading_zero(text: str) -> bool:
    return len(text) > 1 and text.startswith("0")


def format_field_value(value: int, width: int | None) -> str:
    if width is None or value == 0:
        return str(value)
    sign = "-" if value < 0 else ""
    return sign + str(abs(value)).zfill(width)


def reverse_field_value(value: int, width: int | None) -> str:
    text = format_field_value(value, width)
    if text.startswith("-"):
        return "-" + text[1:][::-1]
    return text[::-1]


def magnitude_text(value: int, width: int | None) -> str:
    if value == 0:
        return "0"
    return str(abs(value)).zfill(width or 0)


def reverse_magnitude_text(value: int, width: int | None) -> str:
    return magnitude_text(value, width)[::-1]


def render_mode_value(mode: str, raw: int, operator: str, width: int | None) -> str:
    if raw == 0 and operator == "-" and "op_prefix" in mode:
        return "0"
    if mode == "plain":
        return format_field_value(raw, width)
    if mode == "rev":
        return reverse_field_value(raw, width)
    if mode == "op_prefix_if_neg":
        if raw < 0 and operator == "-":
            return format_field_value(raw, width)
        return operator + magnitude_text(raw, width) if raw < 0 else format_field_value(raw, width)
    if mode == "rev_or_op_prefix_rev_if_neg":
        if raw < 0 and operator == "-":
            return reverse_field_value(raw, width)
        return operator + reverse_magnitude_text(raw, width) if raw < 0 else reverse_field_value(raw, width)
    if mode == "rev_or_op_suffix_if_neg":
        return magnitude_text(raw, width) + operator if raw < 0 else reverse_field_value(raw, width)
    if mode == "op_prefix":
        if raw < 0 and operator == "-":
            return format_field_value(raw, width)
        return operator + (magnitude_text(raw, width) if raw < 0 else format_field_value(raw, width))
    if mode == "rev_or_op_suffix_rev_if_neg":
        return reverse_magnitude_text(raw, width) + operator if raw < 0 else reverse_field_value(raw, width)
    if mode == "op_suffix":
        return (magnitude_text(raw, width) if raw < 0 else format_field_value(raw, width)) + operator
    if mode == "op_prefix_rev":
        if raw < 0 and operator == "-":
            return reverse_field_value(raw, width)
        return operator + (reverse_magnitude_text(raw, width) if raw < 0 else reverse_field_value(raw, width))
    if mode == "abs_rev":
        return reverse_magnitude_text(raw, width)
    if mode == "abs":
        return magnitude_text(raw, width)
    raise ValueError(f"unsupported output mode: {mode}")


def render_width(eq: ParsedEquation, pairing: str, base: str) -> int | None:
    if base not in WIDTH_PRESERVING_BASES:
        return None
    left, right = pair_values(eq, pairing)
    if has_leading_zero(left) and has_leading_zero(right):
        return 2
    return None


def effective_rhs_len(rhs: str, op: str) -> int:
    text = rhs.strip()
    if text.startswith(op):
        text = text[len(op) :]
    if text.endswith(op):
        text = text[: -len(op)]
    if text.startswith("-"):
        text = text[1:]
    return len(text)


def join_values(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    return " and ".join(values)


def count_word(count: int) -> str:
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
    }
    return words.get(count, str(count))


def example_count_phrase(count: int) -> str:
    return f"{count_word(count)} {'example' if count == 1 else 'examples'}"


def same_operator_example_lines(same: list[ParsedEquation]) -> list[str]:
    return [
        "same operator examples",
        *(f"{ex.lhs_text} = {ex.rhs_text}" for ex in same),
        example_count_phrase(len(same)),
    ]


def family_for_examples(examples: list[ParsedEquation], include_direct: bool = False) -> list[str]:
    lengths = {effective_rhs_len(ex.rhs_text, ex.operator) for ex in examples}
    if lengths == {4}:
        out = ["x * y", "x * y + 1", "x * y - 1"]
        return (["template0134", "template3401"] + out) if include_direct else out
    if lengths == {3}:
        return ["x + y", "x + y - 1", "x + y + 1", "x * y", "x * y + 1", "x * y - 1"]
    if lengths == {2}:
        return [
            "x + y",
            "x + y - 1",
            "x + y + 1",
            "x - y",
            "y - x",
            "|x - y|",
            "min(x,y)-max(x,y)",
            "max(x,y)%min(x,y)",
            "x mod y",
            "y mod x",
        ]
    if lengths == {1}:
        return [
            "x - y",
            "y - x",
            "|x - y|",
            "min(x,y)-max(x,y)",
            "max(x,y)%min(x,y)",
            "x mod y",
            "y mod x",
        ]
    if lengths == {2, 3}:
        return ["x + y", "x + y - 1", "x + y + 1"]
    if lengths == {3, 4}:
        return ["x * y", "x * y + 1", "x * y - 1"]
    if lengths == {1, 2}:
        return [
            "x - y",
            "y - x",
            "|x - y|",
            "min(x,y)-max(x,y)",
            "max(x,y)%min(x,y)",
            "x mod y",
            "y mod x",
        ]
    raise ValueError(f"unhandled RHS length mix: {sorted(lengths)}")


def operator_absence_family_for_examples(examples: list[ParsedEquation]) -> list[str]:
    lengths = {effective_rhs_len(ex.rhs_text, ex.operator) for ex in examples}
    if lengths == {4}:
        return ["concat(x, y)", "concat(y, x)", "x * y", "x * y + 1", "x * y - 1"]
    if lengths == {3}:
        return ["x + y", "x + y - 1", "x + y + 1", "x * y", "x * y + 1", "x * y - 1"]
    if lengths == {2}:
        return [
            "x + y",
            "x + y - 1",
            "x + y + 1",
            "x - y",
            "y - x",
            "|x - y|",
            "min(x,y)-max(x,y)",
            "max(x,y)%min(x,y)",
            "x mod y",
            "y mod x",
        ]
    if lengths == {1}:
        return [
            "x - y",
            "y - x",
            "|x - y|",
            "min(x,y)-max(x,y)",
            "max(x,y)%min(x,y)",
            "x mod y",
            "y mod x",
        ]
    if lengths == {2, 3}:
        return ["x + y", "x + y - 1", "x + y + 1"]
    if lengths == {3, 4}:
        return ["x * y", "x * y + 1", "x * y - 1"]
    if lengths == {1, 2}:
        return [
            "x - y",
            "y - x",
            "|x - y|",
            "min(x,y)-max(x,y)",
            "max(x,y)%min(x,y)",
            "x mod y",
            "y mod x",
        ]
    raise ValueError(f"unhandled operator-absence RHS length mix: {sorted(lengths)}")


def route_lines(examples: list[ParsedEquation], prefix: str) -> list[str]:
    op = examples[0].operator
    values = [ex.rhs_text for ex in examples]
    lengths = {effective_rhs_len(v, op) for v in values}
    lines = [f"{prefix} RHS values are {join_values(values)}"]
    if lengths == {4}:
        lines.append("The RHS values have length 4, so use direct templates or multiplication")
        lines.append("Try template0134,template3401,x*y,x*y+1,x*y-1")
    elif lengths == {3}:
        lines.append("The RHS values have length 3, so use addition or multiplication")
        lines.append("Try x+y,x+y-1,x+y+1,x*y,x*y+1,x*y-1")
    elif lengths == {2}:
        lines.append("The RHS values have length 2, so use addition, subtraction, or modular")
        lines.append("Try x+y,x+y-1,x+y+1,x-y,y-x,abs(x-y),min(x,y)-max(x,y),max(x,y)%min(x,y),x%y,y%x")
    elif lengths == {1}:
        lines.append("The RHS values have length 1, so use subtraction or modular")
        lines.append("Try x-y,y-x,abs(x-y),min(x,y)-max(x,y),max(x,y)%min(x,y),x%y,y%x")
    elif lengths == {2, 3}:
        lines.append("The RHS values mix length 2 and 3, so use addition")
        lines.append("Try x+y,x+y-1,x+y+1")
    elif lengths == {3, 4}:
        lines.append("The RHS values mix length 3 and 4, so use multiplication")
        lines.append("Try x*y,x*y+1,x*y-1")
    elif lengths == {1, 2}:
        lines.append("The RHS values mix length 1 and 2, so use subtraction or modular")
        lines.append("Try x-y,y-x,abs(x-y),min(x,y)-max(x,y),max(x,y)%min(x,y),x%y,y%x")
    else:
        raise ValueError(f"unhandled route lengths: {sorted(lengths)}")
    return lines


def pair_values(eq: ParsedEquation, pairing: str) -> tuple[str, str]:
    if pairing == "BA_DC":
        return eq.left_operand_text[::-1], eq.right_operand_text[::-1]
    if pairing == "AB_CD":
        return eq.left_operand_text, eq.right_operand_text
    raise ValueError(pairing)


def raw_value(eq: ParsedEquation, pairing: str, base: str) -> int | None:
    x, y = apply_pairing(eq.left_operand_text, eq.right_operand_text, pairing)
    return BASE_RULE_BY_NAME[base].apply(x, y)


def sign_of_raw(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def rendered_values(
    eq: ParsedEquation,
    pairing: str,
    base: str,
    modes: list[str],
) -> tuple[str, str, str, list[str]]:
    left, right = pair_values(eq, pairing)
    raw = raw_value(eq, pairing, base)
    if raw is None:
        raise ValueError(f"raw value failed for {eq}")
    width = render_width(eq, pairing, base)
    values = [render_mode_value(mode, raw, eq.operator, width) for mode in modes]
    return left, right, format_field_value(raw, width), values


def attempt_block(
    pairing: str,
    base: str,
    op: str,
    examples: list[ParsedEquation],
    modes_override: list[str] | None = None,
    common_suffix: bool = False,
) -> tuple[list[str], list[str]]:
    label = BASE_LABEL[base]
    modes = modes_override if modes_override is not None else MODES[pairing]
    format_label = f"{pairing}|{label}|common" if common_suffix else f"{pairing}|{label}"
    lines = [f"Try {pairing} with {label} for operator {op}", f"The current format is {format_label}", ""]
    common: set[str] | None = None
    for ex in examples:
        left, right, raw, values = rendered_values(ex, pairing, base, modes)
        matched = [mode for mode, value in zip(modes, values) if value == ex.rhs_text]
        if common is None:
            common = set(matched)
        else:
            common &= set(matched)
        lines.extend(
            [
                f"Example {ex.lhs_text} = {ex.rhs_text}",
                f"{'BA DC' if pairing == 'BA_DC' else 'AB CD'} {RAW_HEADER[(pairing, base)]} {' '.join(modes)}",
                f"{left} {right} {raw} {' '.join(values)}",
                "Match",
            ]
        )
        lines.extend(matched if matched else ["none"])
        lines.append("")
    common_list = [mode for mode in modes if common and mode in common]
    lines.extend(["Common", *(common_list if common_list else ["none"]), ""])
    return lines, common_list


def direct_template_match(examples: list[ParsedEquation]) -> str | None:
    for template in ("template0134", "template3401"):
        all_match = True
        for ex in examples:
            value = (
                ex.left_operand_text + ex.right_operand_text
                if template == "template0134"
                else ex.right_operand_text + ex.left_operand_text
            )
            if value != ex.rhs_text:
                all_match = False
                break
        if all_match:
            return template
    return None


def direct_template_query_value(template: str, eq: ParsedEquation) -> str:
    if template == "template0134":
        return eq.left_operand_text + eq.right_operand_text
    if template == "template3401":
        return eq.right_operand_text + eq.left_operand_text
    raise ValueError(template)


def direct_template_block(op: str, examples: list[ParsedEquation], helper_motif_only: bool) -> tuple[list[str], str | None]:
    lines = [] if helper_motif_only else [
        "We try direct templates first. If they fail, we proceed to arithmetic search on motifs BA_DC and AB_CD",
        "",
    ]
    for template in ("template0134", "template3401"):
        lines.extend(
            [
                f"Try {template} for operator {op}",
                f"{template} means {'AB followed by CD' if template == 'template0134' else 'CD followed by AB'}",
                "",
            ]
        )
        all_match = True
        for ex in examples:
            value = direct_template_query_value(template, ex)
            if value != ex.rhs_text:
                all_match = False
            lines.extend(
                [
                    f"Example {ex.lhs_text} = {ex.rhs_text}",
                    f"AB = {ex.left_operand_text}",
                    f"operator = {op}",
                    f"CD = {ex.right_operand_text}",
                    f"{ex.left_operand_text if template == 'template0134' else ex.right_operand_text} followed by {ex.right_operand_text if template == 'template0134' else ex.left_operand_text} gives {value} vs {ex.rhs_text}",
                    "Match" if value == ex.rhs_text else "No match",
                    "",
                ]
            )
        if all_match:
            if not helper_motif_only:
                lines.extend(
                    [
                        f"{template} passes all examples",
                        "For direct templates, apply the passing template to get the answer",
                    ]
                )
            else:
                lines.append(f"{template} supports the helper operator group")
            return lines, template
        lines.extend([f"{template} fails", ""])
    lines.extend(
        ["Direct templates fail", ""]
        if helper_motif_only
        else ["Direct templates fail", "Proceed to arithmetic search on motifs BA_DC and AB_CD", ""]
    )
    return lines, None


def helper_groups(puzzle: NumericEquationPuzzle) -> list[tuple[str, list[ParsedEquation]]]:
    grouped: dict[str, list[ParsedEquation]] = defaultdict(list)
    order: list[str] = []
    for ex in puzzle.examples:
        if ex.operator == puzzle.query.operator:
            continue
        if ex.operator not in grouped:
            order.append(ex.operator)
        grouped[ex.operator].append(ex)
    return [(op, grouped[op]) for op in order]


def is_direct_helper_group(examples: list[ParsedEquation]) -> bool:
    lengths = {effective_rhs_len(ex.rhs_text, ex.operator) for ex in examples}
    return lengths == {4} and direct_template_match(examples) is not None


def ranked_helper_groups(puzzle: NumericEquationPuzzle) -> list[tuple[str, list[ParsedEquation], bool]]:
    groups = helper_groups(puzzle)

    def sort_key(item: tuple[str, list[ParsedEquation]]) -> tuple[int, int, int]:
        op, examples = item
        return (len(examples), int(is_direct_helper_group(examples)), [gop for gop, _ in groups].index(op))

    return [(op, examples, False) for op, examples in sorted(groups, key=sort_key)]


def first_ba_survivor(examples: list[ParsedEquation]) -> tuple[str | None, list[str], list[str]]:
    op = examples[0].operator
    tried: list[str] = []
    for base in family_for_examples(examples, include_direct=False):
        lines, common = attempt_block("BA_DC", base, op, examples)
        tried.extend(lines)
        if common:
            return base, common, tried
        tried.extend([f"{BASE_LABEL[base]} fails under BA_DC", ""])
    return None, [], tried


def deterministic_negative_mode(pairing: str, base: str, raw: int, operator: str) -> str | None:
    if raw >= 0 or operator == "-":
        return None
    if pairing == "AB_CD" and base == "min(x,y)-max(x,y)":
        return "op_prefix_if_neg"
    if base not in {"x - y", "y - x"}:
        return None
    if pairing == "AB_CD" and base == "x - y":
        return "op_prefix_if_neg"
    if pairing == "AB_CD" and base == "y - x":
        return "rev_or_op_suffix_if_neg"
    if pairing == "BA_DC" and base == "x - y":
        return "rev_or_op_prefix_rev_if_neg"
    if pairing == "BA_DC" and base == "y - x":
        return "rev_or_op_suffix_rev_if_neg"
    return None


def query_prediction(
    pairing: str,
    base: str,
    common: list[str],
    puzzle: NumericEquationPuzzle,
    support_examples: list[ParsedEquation],
    gold: str | None = None,
) -> tuple[str, list[str], str]:
    q = puzzle.query
    left, right, raw_text, values = rendered_values(q, pairing, base, common)
    raw = int(raw_text)
    if len(set(values)) == 1:
        return values[0], [left, right, raw_text, *values], "agree"
    if pairing == "BA_DC" and base == "x - y" and raw < 0 and q.operator == "-":
        answer = "-" + raw_text[1:][::-1]
        return answer, [left, right, raw_text, *values], "deterministic:ba_dc_x_y_literal_minus"
    negative_mode = deterministic_negative_mode(pairing, base, raw, q.operator)
    if negative_mode and negative_mode in common:
        answer = values[common.index(negative_mode)]
        return answer, [left, right, raw_text, *values], f"deterministic:{negative_mode}"
    if pairing == "BA_DC" and base == "x - y" and raw < 0:
        raise RuntimeError(
            "BA_DC|x-y negative requires reverse magnitude plus operator prefix; "
            "generic BA_DC voting is not a valid policy for this branch"
        )
    if pairing == "BA_DC":
        counts = Counter(values)
        best_count = max(counts.values())
        for value in values:
            if counts[value] == best_count:
                return value, [left, right, raw_text, *values], "ba_voting"
    return values[0], [left, right, raw_text, *values], f"first:{common[0]}"


def voting_lines(base: str, row: list[str], common: list[str], query_operator: str, answer: str) -> list[str]:
    raw = int(row[2])
    base_label = BASE_LABEL[base]
    if base == "x - y":
        sign = "positive" if raw >= 0 else "negative"
        intro = [
            f"The selected operator x-y gives {sign} value: {row[0]}-{row[1]}={row[2]}",
        ]
        context = "x-y negative and literal -" if raw < 0 and query_operator == "-" else f"x-y {sign}"
    elif base == "y - x":
        sign = "positive" if raw >= 0 else "negative"
        intro = [
            f"The selected operator y-x gives {sign} value: {row[1]}-{row[0]}={row[2]}",
        ]
        context = "y-x negative and literal -" if raw < 0 and query_operator == "-" else f"y-x {sign}"
    else:
        intro = [f"The selected operator {base_label} is neither x-y nor y-x"]
        context = "selected operator neither x-y nor y-x"

    values = row[3:]
    counts = Counter(values)
    ordered_values = sorted(counts, key=lambda value: (-counts[value], values.index(value)))
    best_count = counts[ordered_values[0]]
    lines = [
        *intro,
        f"For motif BA_DC with {context}, use voting. If voting ties, use the first common format in priority order",
        "",
        "Votes",
    ]
    lines.extend(f"{value} has {counts[value]} votes" for value in ordered_values)
    if sum(1 for value in ordered_values if counts[value] == best_count) == 1:
        lines.extend(["", "Vote winner", answer])
    else:
        first_common = common[values.index(answer)]
        lines.extend(["", "Tie", "", "First common format in priority order", first_common, answer])
    return lines


def query_block(
    pairing: str,
    base: str,
    common: list[str],
    puzzle: NumericEquationPuzzle,
    support_examples: list[ParsedEquation],
    gold: str | None = None,
) -> tuple[list[str], str]:
    q = puzzle.query
    label = BASE_LABEL[base]
    if label in {"template0134", "template3401"}:
        value = direct_template_query_value(label, q)
        return [
            f"Apply {label} to the query",
            "",
            "Query",
            q.lhs_text,
            f"{q.left_operand_text} -> {q.left_operand_text}",
            f"{q.right_operand_text} -> {q.right_operand_text}",
            value,
        ], value

    answer, row, reason = query_prediction(pairing, base, common, puzzle, support_examples, gold)
    values = row[3:]
    lines = [
        f"Apply format {pairing}|{label}|common to the query",
        "",
        "Query",
        q.lhs_text,
        f"{'BA DC' if pairing == 'BA_DC' else 'AB CD'} {RAW_HEADER[(pairing, base)]} {' '.join(common)}",
        " ".join(row),
    ]
    if len(set(values)) == 1:
        lines.append(f"All common output formats agree on {values[0]}")
        return lines, answer

    lines.append("All common output formats do not agree")
    raw = int(row[2])
    negative_mode = deterministic_negative_mode(pairing, base, raw, q.operator)
    if reason.startswith("deterministic:") and negative_mode:
        compact_answer = False
        if pairing == "BA_DC" and base == "x - y":
            lines.append("The selected operator x-y gives negative value")
            lines.append(f"{row[0]}-{row[1]}={row[2]}")
            lines.append("For motif BA_DC with x-y negative, reverse the magnitude and use operator prefix")
            compact_answer = True
        elif pairing == "BA_DC" and base == "y - x":
            lines.append("The selected operator y-x gives negative value")
            lines.append(f"{row[1]}-{row[0]}={row[2]}")
            lines.append("For motif BA_DC with y-x negative and nonliteral operator, reverse the magnitude and use operator suffix")
        elif pairing == "AB_CD" and base == "x - y":
            lines.append("The selected operator x-y gives negative value")
            lines.append(f"{row[0]}-{row[1]}={row[2]}")
            lines.append("For motif AB_CD with x-y negative, use operator prefix")
            compact_answer = True
        elif pairing == "AB_CD" and base == "min(x,y)-max(x,y)":
            lines.append("The selected operator min(x,y)-max(x,y) gives negative value")
            lines.append(f"min({row[0]},{row[1]})-max({row[0]},{row[1]})={row[2]}")
            lines.append("For motif AB_CD with min(x,y)-max(x,y) negative, use operator prefix")
            compact_answer = True
        elif pairing == "AB_CD" and base == "y - x":
            lines.append(f"The selected operator y-x gives negative value: {row[1]}-{row[0]}={row[2]}")
            lines.append("For motif AB_CD with y-x negative, use operator suffix")
        lines.extend([answer] if compact_answer else ["", answer])
    elif reason == "deterministic:ba_dc_x_y_literal_minus":
        lines.append("The selected operator x-y gives negative value")
        lines.append(f"{row[0]}-{row[1]}={row[2]}")
        lines.append("For motif BA_DC with x-y negative, reverse the magnitude and use operator prefix")
        lines.append(answer)
    elif pairing == "BA_DC":
        lines.extend(voting_lines(base, row, common, q.operator, answer))
    else:
        lines.append(f"For motif AB_CD, use first common output format {common[0]}")
        lines.append(answer)
    return lines, answer


def final_ba_query_block_from_lines(lines: list[str]) -> tuple[str, list[str], list[str]] | None:
    apply_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("Apply format BA_DC|") and line.endswith("|common to the query"):
            apply_idx = idx
    if apply_idx is None:
        return None

    apply_line = lines[apply_idx]
    format_label = apply_line.removeprefix("Apply format BA_DC|").removesuffix("|common to the query")

    end_idx = len(lines)
    for idx in range(apply_idx + 1, len(lines)):
        if lines[idx].startswith("Answer:"):
            end_idx = idx
            break
        if lines[idx] == "</think>" or lines[idx].startswith("\\boxed{"):
            end_idx = idx
            break
    block = lines[apply_idx:end_idx]
    while block and block[-1] == "":
        block.pop()

    common: list[str] = []
    for idx in range(apply_idx + 1, min(len(lines), apply_idx + 8)):
        if lines[idx].startswith("BA DC "):
            parts = lines[idx].split()
            common = parts[3:]
            break
    if not common:
        return None
    return format_label, common, block


def final_ba_query_block_from_text(text: str) -> tuple[str, list[str], list[str]] | None:
    return final_ba_query_block_from_lines(normalize_operator_format_lines(text.strip().splitlines()))


def final_ba_blocks_from_lines(lines: list[str]) -> tuple[str, list[str], list[str], list[str]] | None:
    query_block = final_ba_query_block_from_lines(lines)
    if query_block is None:
        return None
    format_label, common, query_lines = query_block

    apply_idx = None
    for idx, line in enumerate(lines):
        if line == f"Apply format BA_DC|{format_label}|common to the query":
            apply_idx = idx
    if apply_idx is None:
        return None

    scan_idx = None
    marker = f"Try BA_DC with {format_label}"
    for idx in range(apply_idx - 1, -1, -1):
        if lines[idx].startswith(marker):
            scan_idx = idx
            break
    if scan_idx is None:
        return None

    scan_lines = lines[scan_idx:apply_idx]
    while scan_lines and scan_lines[-1] == "":
        scan_lines.pop()
    return format_label, common, scan_lines, query_lines


def final_ba_blocks_from_text(text: str) -> tuple[str, list[str], list[str], list[str]] | None:
    return final_ba_blocks_from_lines(normalize_operator_format_lines(text.strip().splitlines()))


def final_ab_section_from_lines(lines: list[str]) -> tuple[str, list[str]] | None:
    apply_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("Apply format AB_CD|") and line.endswith("|common to the query"):
            apply_idx = idx
    if apply_idx is None:
        return None

    apply_line = lines[apply_idx]
    format_label = apply_line.removeprefix("Apply format AB_CD|").removesuffix("|common to the query")

    start_idx = None
    for idx in range(apply_idx - 1, -1, -1):
        if lines[idx] == "Try AB_CD":
            start_idx = idx
            break
    if start_idx is None:
        marker = f"Try AB_CD with {format_label}"
        for idx in range(apply_idx - 1, -1, -1):
            if lines[idx].startswith(marker):
                start_idx = idx
                break
    if start_idx is None:
        return None

    end_idx = len(lines)
    for idx in range(apply_idx + 1, len(lines)):
        if lines[idx].startswith("Answer:"):
            end_idx = idx
            break
        if lines[idx] == "</think>" or lines[idx].startswith("\\boxed{"):
            end_idx = idx
            break
    section = lines[start_idx:end_idx]
    while section and section[-1] == "":
        section.pop()
    return format_label, section


def final_ab_section_from_text(text: str) -> tuple[str, list[str]] | None:
    return final_ab_section_from_lines(normalize_operator_format_lines(text.strip().splitlines()))


def normalized_lines(lines: list[str]) -> list[str]:
    out = list(lines)
    while out and out[-1] == "":
        out.pop()
    return out


def equation_operator(text: str) -> str | None:
    lhs = text.split(" = ", 1)[0].strip()
    idx = 0
    while idx < len(lhs) and lhs[idx].isdigit():
        idx += 1
    if idx >= len(lhs):
        return None
    return lhs[idx]


def normalize_operator_format_lines(lines: list[str]) -> list[str]:
    out = [line for line in lines if line != "The output format gives the same negative sign as the same-operator example"]
    current_operator: str | None = None
    expect_query_equation = False
    for idx, line in enumerate(out):
        if line == "Query":
            expect_query_equation = True
            continue
        if expect_query_equation:
            current_operator = equation_operator(line)
            expect_query_equation = False
            continue
        if line.startswith("Example "):
            current_operator = equation_operator(line.removeprefix("Example "))
            continue
        if " for operator " in line:
            current_operator = line.rsplit(" for operator ", 1)[1]
            continue
        if current_operator is None or not (line.startswith("BA DC ") or line.startswith("AB CD ")):
            continue
        if idx + 1 >= len(out):
            continue
        header = line.split()
        values = out[idx + 1].split()
        if len(values) != len(header) or len(values) < 3:
            continue
        raw = values[2]
        if not raw.startswith("-"):
            continue
        raw_value = int(raw)
        width = len(raw[1:]) if len(raw) > 2 else None
        for col, mode in enumerate(header):
            if "op_prefix" not in mode and mode != "op_suffix":
                continue
            values[col] = render_mode_value(mode, raw_value, current_operator, width)
        out[idx + 1] = " ".join(values)
    return out


def parsed_equations_by_lhs(puzzle: NumericEquationPuzzle) -> dict[str, ParsedEquation]:
    return {eq.lhs_text: eq for eq in [*puzzle.examples, puzzle.query]}


def parse_attempt_intro(line: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"Try (BA_DC|AB_CD) with (.+) for operator (.+)", line)
    if not match:
        return None
    pairing, label, op = match.groups()
    base = BASE_FROM_LABEL.get(label)
    if base is None:
        return None
    return pairing, base, op


def refresh_attempt_blocks(lines: list[str], puzzle: NumericEquationPuzzle) -> list[str]:
    eq_by_lhs = parsed_equations_by_lhs(puzzle)
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        parsed = parse_attempt_intro(lines[idx])
        if parsed is None:
            out.append(lines[idx])
            idx += 1
            continue

        pairing, base, op = parsed
        common_idx = None
        for cursor in range(idx + 1, len(lines)):
            if lines[cursor] == "Common":
                common_idx = cursor
                break
            if cursor > idx + 1 and parse_attempt_intro(lines[cursor]) is not None:
                break
        if common_idx is None:
            out.append(lines[idx])
            idx += 1
            continue

        examples: list[ParsedEquation] = []
        for cursor in range(idx + 1, common_idx):
            if not lines[cursor].startswith("Example "):
                continue
            lhs = lines[cursor].removeprefix("Example ").split(" = ", 1)[0].strip()
            if lhs not in eq_by_lhs:
                raise RuntimeError(f"cannot refresh attempt block, unknown example {lhs}")
            examples.append(eq_by_lhs[lhs])
        if not examples:
            out.append(lines[idx])
            idx += 1
            continue

        block, _ = attempt_block(pairing, base, op, examples)
        out.extend(block)

        end_idx = common_idx + 1
        while end_idx < len(lines) and lines[end_idx] != "":
            end_idx += 1
        if end_idx < len(lines) and lines[end_idx] == "":
            end_idx += 1
        idx = end_idx
    return out


def refresh_standalone_arithmetic_rows(lines: list[str], puzzle: NumericEquationPuzzle) -> list[str]:
    eq_by_lhs = parsed_equations_by_lhs(puzzle)
    out = list(lines)
    current_lhs: str | None = None
    expect_query = False
    idx = 0
    while idx < len(out):
        line = out[idx]
        if line == "Query":
            expect_query = True
            idx += 1
            continue
        if expect_query:
            current_lhs = line.strip()
            expect_query = False
            idx += 1
            continue
        if line.startswith("Example "):
            current_lhs = line.removeprefix("Example ").split(" = ", 1)[0].strip()
            idx += 1
            continue
        if not (line.startswith("BA DC ") or line.startswith("AB CD ")):
            idx += 1
            continue
        if idx + 1 >= len(out):
            idx += 1
            continue

        header = line.split()
        pairing = "BA_DC" if line.startswith("BA DC ") else "AB_CD"
        base = None
        for (candidate_pairing, candidate_base), raw_header in RAW_HEADER.items():
            if candidate_pairing == pairing and raw_header == header[2]:
                base = candidate_base
                break
        if base is None:
            idx += 1
            continue
        if current_lhs is None or current_lhs not in eq_by_lhs:
            raise RuntimeError(f"cannot refresh arithmetic row, unknown equation {current_lhs}")

        modes = header[3:]
        left, right, raw, values = rendered_values(eq_by_lhs[current_lhs], pairing, base, modes)
        out[idx + 1] = " ".join([left, right, raw, *values])
        if idx + 2 < len(out) and out[idx + 2] == "Match":
            matched = [mode for mode, value in zip(modes, values) if value == eq_by_lhs[current_lhs].rhs_text]
            end_idx = idx + 3
            while end_idx < len(out) and out[end_idx] != "":
                end_idx += 1
            out[idx + 3 : end_idx] = matched if matched else ["none"]
            idx = idx + 3 + len(matched if matched else ["none"])
            continue
        idx += 1
    return out


def refresh_query_blocks(lines: list[str], puzzle: NumericEquationPuzzle, gold: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = re.fullmatch(r"Apply format (BA_DC|AB_CD)\|(.+)\|common to the query", line)
        if match is None:
            out.append(line)
            idx += 1
            continue

        pairing, label = match.groups()
        base = BASE_FROM_LABEL.get(label)
        if base is None:
            out.append(line)
            idx += 1
            continue

        end_idx = len(lines)
        for cursor in range(idx + 1, len(lines)):
            if lines[cursor].startswith("Answer:") or lines[cursor] == "</think>" or lines[cursor].startswith("\\boxed{"):
                end_idx = cursor
                break

        common: list[str] | None = None
        for cursor in range(idx + 1, end_idx):
            if pairing == "BA_DC" and lines[cursor].startswith("BA DC "):
                common = lines[cursor].split()[3:]
                break
            if pairing == "AB_CD" and lines[cursor].startswith("AB CD "):
                common = lines[cursor].split()[3:]
                break
        if common is None:
            out.extend(lines[idx:end_idx])
            idx = end_idx
            continue

        q_lines, pred = query_block(pairing, base, common, puzzle, [], gold=gold)
        if pred != gold:
            raise RuntimeError(f"refreshed query block predicts {pred}, gold {gold}")
        out.extend(q_lines)
        idx = end_idx
    return out


def refresh_arithmetic_trace(lines: list[str], puzzle: NumericEquationPuzzle, gold: str) -> list[str]:
    refreshed = refresh_attempt_blocks(lines, puzzle)
    refreshed = refresh_standalone_arithmetic_rows(refreshed, puzzle)
    refreshed = refresh_query_blocks(refreshed, puzzle, gold)
    return refreshed


def validated_golden_ba_block(
    golden_ba_block: tuple[str, list[str], list[str], list[str]] | None,
    puzzle: NumericEquationPuzzle,
    same: list[ParsedEquation],
    gold: str,
) -> tuple[str, list[str], list[str], list[str]] | None:
    if golden_ba_block is None or golden_ba_block[0] not in BASE_FROM_LABEL:
        return None
    format_label, common, scan_lines, query_lines = golden_ba_block
    base = BASE_FROM_LABEL[format_label]
    current_scan_lines, current_common = attempt_block("BA_DC", base, puzzle.query.operator, same)
    if current_common != common:
        return None
    current_query_lines, pred = query_block("BA_DC", base, common, puzzle, same, gold=gold)
    if pred != gold:
        return None
    if normalized_lines(current_scan_lines) != normalized_lines(scan_lines):
        return None
    if normalized_lines(current_query_lines) != normalized_lines(query_lines):
        return None
    return golden_ba_block


def return_to_ba_candidate_lines(format_label: str, common: list[str]) -> list[str]:
    return [
        f"Return to the supported same operator format BA_DC|{format_label}|common",
        "Common",
        *common,
        "",
    ]


def ba_gold_scan_lines(
    puzzle: NumericEquationPuzzle,
    same: list[ParsedEquation],
    gold: str,
    initial_base: str,
    initial_common: list[str],
) -> tuple[list[str], str | None]:
    lines: list[str] = []
    started = False
    for base in family_for_examples(same, include_direct=False):
        if base == initial_base:
            started = True
            common = initial_common
        elif not started:
            continue
        else:
            block, common = attempt_block("BA_DC", base, puzzle.query.operator, same)
            lines.extend(block)
            if not common:
                lines.extend([f"{BASE_LABEL[base]} fails under BA_DC", ""])
                continue
        q_lines, pred = query_block("BA_DC", base, common, puzzle, same, gold=gold)
        if pred == gold:
            if base != initial_base:
                lines.append(f"The format BA_DC|{BASE_LABEL[base]}|common supports the single same operator example")
            lines.extend(q_lines)
            return lines, pred
        return lines, None
    return lines, None


def ab_gold_scan_lines(
    puzzle: NumericEquationPuzzle,
    same: list[ParsedEquation],
    gold: str,
) -> tuple[list[str], str | None]:
    lines: list[str] = []
    for base in family_for_examples(same, include_direct=False):
        block, common = attempt_block("AB_CD", base, puzzle.query.operator, same)
        lines.extend(block)
        if not common:
            lines.extend([f"{BASE_LABEL[base]} fails under AB_CD", ""])
            continue
        pred, _, _ = query_prediction("AB_CD", base, common, puzzle, same, gold)
        if pred == gold:
            q_lines, pred = query_block("AB_CD", base, common, puzzle, same, gold=gold)
            lines.extend(q_lines)
            return lines, pred
    return lines, None


def helper_verify_ba_lines(op: str, examples: list[ParsedEquation]) -> tuple[list[str], bool]:
    lines = [f"Same helper operator RHS values are {join_values([ex.rhs_text for ex in examples])}"]
    lines.extend(route_lines(examples, "Same helper operator")[1:])
    lines.append("")
    if {effective_rhs_len(ex.rhs_text, ex.operator) for ex in examples} == {4}:
        dlines, passed = direct_template_block(op, examples, helper_motif_only=True)
        lines.extend(dlines)
        if passed:
            if passed == "template3401":
                lines.extend(
                    [
                        "template3401 supports BA_DC order",
                        "The motif BA_DC is supported by the helper operator group",
                        "So BA_DC is confirmed",
                        "",
                    ]
                )
                return lines, True
            lines.extend(
                [
                    "template0134 supports AB_CD order, not BA_DC",
                    "The motif BA_DC is not supported by the helper operator group",
                    "So BA_DC is rejected",
                    "",
                ]
            )
            return lines, False
    for base in family_for_examples(examples, include_direct=False):
        block, common = attempt_block("BA_DC", base, op, examples)
        lines.extend(block)
        if common:
            lines.extend(
                [
                    f"The format BA_DC|{BASE_LABEL[base]}|common supports the helper operator group",
                    "The motif BA_DC is supported by the helper operator group",
                    "So BA_DC is confirmed",
                    "",
                ]
            )
            return lines, True
        lines.extend([f"{BASE_LABEL[base]} fails under BA_DC", ""])
    lines.extend(["The motif BA_DC is not supported by the helper operator group", "So BA_DC is rejected", ""])
    return lines, False


def common_intro_lines(row: dict[str, str], puzzle: NumericEquationPuzzle, gold: str) -> list[str]:
    lines: list[str] = [
        f"Problem {row['id']}",
        "",
        "Question:",
        row["prompt"],
        "",
        "Gold answer:",
        gold,
        "",
        "Solution:",
        "",
        "We sequentially try direct templates, then motifs BA_DC and AB_CD. For each step, we choose the rule family from same-operator RHS length.",
        "",
        f"Query {puzzle.query.lhs_text}",
        f"Query operator is {puzzle.query.operator}",
        "",
        "Compare example operators",
    ]
    for ex in puzzle.examples:
        lines.extend([f"{ex.lhs_text} = {ex.rhs_text} operator {ex.operator}", "same" if ex.operator == puzzle.query.operator else "different"])
    return lines


def render_direct_same_operator_trace(
    row: dict[str, str],
    puzzle: NumericEquationPuzzle,
    same: list[ParsedEquation],
    template: str,
) -> str:
    gold = row["answer"]
    lines = common_intro_lines(row, puzzle, gold)
    lines.extend(["", *same_operator_example_lines(same), ""])
    lines.extend(route_lines(same, "Same operator"))
    lines.append("")
    dlines, passed = direct_template_block(puzzle.query.operator, same, helper_motif_only=False)
    if passed != template:
        raise RuntimeError(f"{row['id']} expected {template}, got {passed}")
    lines.extend(dlines)
    value = direct_template_query_value(template, puzzle.query)
    if value != gold:
        raise RuntimeError(f"{row['id']} direct template gives {value}, gold {gold}")
    lines.extend(
        [
            f"Apply {template} to the query",
            "",
            "Query",
            puzzle.query.lhs_text,
            f"{puzzle.query.left_operand_text} -> {puzzle.query.left_operand_text}",
            f"{puzzle.query.right_operand_text} -> {puzzle.query.right_operand_text}",
            value,
            "",
            f"Answer: \\boxed{{{gold}}}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def multi_same_candidate_lines(
    pairing: str,
    puzzle: NumericEquationPuzzle,
    same: list[ParsedEquation],
    gold: str,
) -> tuple[list[str], str | None]:
    lines: list[str] = []
    for base in family_for_examples(same, include_direct=False):
        block, common = attempt_block(pairing, base, puzzle.query.operator, same)
        lines.extend(block)
        if not common:
            lines.extend([f"{BASE_LABEL[base]} fails under {pairing}", ""])
            continue

        pred, _, _ = query_prediction(pairing, base, common, puzzle, same, gold)
        if pred != gold:
            lines.extend(
                [
                    f"The format {pairing}|{BASE_LABEL[base]}|common supports all {count_word(len(same))} same operator examples",
                    "More than one same operator row supports this candidate, so finalize",
                ]
            )
            q_lines, _ = query_block(pairing, base, common, puzzle, same, gold=gold)
            lines.extend(q_lines)
            return lines, None

        lines.extend(
            [
                f"The format {pairing}|{BASE_LABEL[base]}|common supports all {count_word(len(same))} same operator examples",
                "More than one same operator row supports this candidate, so finalize",
            ]
        )
        q_lines, pred = query_block(pairing, base, common, puzzle, same, gold=gold)
        lines.extend(q_lines)
        return lines, pred
    return lines, None


def render_multi_same_real_trace(row: dict[str, str]) -> tuple[str, str]:
    puzzle = parse_numeric_equation_puzzle(row["prompt"])
    if puzzle is None:
        raise RuntimeError(row["id"])
    if not is_numeric_puzzle(puzzle):
        raise RuntimeError(f"{row['id']} is not numeric-only")
    gold = row["answer"]
    groups: dict[str, list[ParsedEquation]] = defaultdict(list)
    for ex in puzzle.examples:
        groups[ex.operator].append(ex)
    same = groups[puzzle.query.operator]
    if len(same) < 2:
        raise RuntimeError(f"{row['id']} has fewer than two same-operator examples")

    if {effective_rhs_len(ex.rhs_text, ex.operator) for ex in same} == {4}:
        template = direct_template_match(same)
        if template:
            value = direct_template_query_value(template, puzzle.query)
            if value == gold:
                return render_direct_same_operator_trace(row, puzzle, same, template), "real_multi_same_direct"

    lines = common_intro_lines(row, puzzle, gold)
    lines.extend(["", *same_operator_example_lines(same), ""])
    lines.extend(route_lines(same, "Same operator"))
    lines.append("")

    if {effective_rhs_len(ex.rhs_text, ex.operator) for ex in same} == {4}:
        dlines, passed = direct_template_block(puzzle.query.operator, same, helper_motif_only=False)
        lines.extend(dlines)
        if passed:
            raise RuntimeError(f"{row['id']} direct template passes but query does not reach gold")

    lines.extend(["Try BA_DC first", ""])
    ba_lines, pred = multi_same_candidate_lines("BA_DC", puzzle, same, gold)
    lines.extend(ba_lines)
    if pred == gold:
        lines.extend(["", f"Answer: \\boxed{{{gold}}}"])
        return "\n".join(lines).rstrip() + "\n", "real_multi_same_ba"
    if pred is None and any("More than one same operator row supports this candidate" in line for line in ba_lines):
        raise RuntimeError(f"{row['id']} BA_DC support finalizes to non-gold answer")

    lines.extend([f"No BA_DC candidate supports all {count_word(len(same))} same operator examples"])
    lines.extend(["Try AB_CD", ""])
    ab_lines, pred = multi_same_candidate_lines("AB_CD", puzzle, same, gold)
    lines.extend(ab_lines)
    if pred != gold:
        if pred is None and any("More than one same operator row supports this candidate" in line for line in ab_lines):
            raise RuntimeError(f"{row['id']} AB_CD support finalizes to non-gold answer")
        raise RuntimeError(f"{row['id']} did not reach gold under multi-same policy")
    lines.extend(["", f"Answer: \\boxed{{{gold}}}"])
    return "\n".join(lines).rstrip() + "\n", "real_multi_same_ab"


def operator_absence_group_candidates(
    pairing: str,
    op: str,
    examples: list[ParsedEquation],
    modes: str | list[str],
) -> list[tuple[str, list[str]]]:
    modes_list = [modes] if isinstance(modes, str) else modes
    candidates: list[tuple[str, list[str]]] = []
    for base in operator_absence_family_for_examples(examples):
        _, common = attempt_block(pairing, base, op, examples, modes_list)
        if common:
            candidates.append((base, common))
    return candidates


def operator_absence_query_vote(
    pairing: str,
    base: str,
    common: list[str],
    puzzle: NumericEquationPuzzle,
) -> tuple[str, int, list[str], str]:
    label = BASE_LABEL[base]
    if label in {"template0134", "template3401"}:
        value = direct_template_query_value(label, puzzle.query)
        return value, 1, [value], "direct"
    answer, row, reason = query_prediction(pairing, base, common, puzzle, [], gold=None)
    values = row[3:]
    return answer, max(Counter(values).values()), values, reason


def choose_operator_absence_query_candidate(
    pairing: str,
    candidate_bases: list[str],
    common: list[str],
    puzzle: NumericEquationPuzzle,
    used_bases: list[str],
) -> tuple[str, list[str], str, list[tuple[str, str, int, list[str], str]]] | None:
    scored: list[tuple[str, str, int, list[str], str]] = []
    remaining_bases = [base for base in candidate_bases if base not in used_bases]
    for base in remaining_bases:
        try:
            answer, vote_count, values, reason = operator_absence_query_vote(
                pairing,
                base,
                common,
                puzzle,
            )
        except Exception:
            continue
        scored.append((base, answer, vote_count, values, reason))
    if not scored:
        return None

    order = {base: idx for idx, base in enumerate(candidate_bases)}
    selected = max(scored, key=lambda item: (item[2], -order[item[0]]))
    selected_base, _answer, _vote_count, _values, _reason = selected
    q_lines, pred = query_block(pairing, selected_base, common, puzzle, [], gold=None)
    return selected_base, q_lines, pred, scored


def first_operator_absence_path(
    puzzle: NumericEquationPuzzle,
) -> tuple[str, list[tuple[str, list[ParsedEquation], str, list[str]]], list[str], list[str], str, str, list[str], list[tuple[str, str, int, list[str], str]]] | None:
    candidate_bases = OP_ABSENCE_SYMBOL_CANDIDATES.get(puzzle.query.operator)
    if candidate_bases is None:
        return None
    visible_groups = helper_groups(puzzle)
    if not visible_groups:
        return None

    for pairing in ("BA_DC", "AB_CD"):
        choices: list[tuple[str, list[ParsedEquation], str, list[str]]] = []
        prior_common: list[str] | None = None
        for op, examples in visible_groups:
            modes = OP_ABSENCE_MODES[pairing] if prior_common is None else prior_common
            candidates = operator_absence_group_candidates(pairing, op, examples, modes)
            if not candidates:
                break
            base, common = candidates[0]
            updated_common = common if prior_common is None else [mode for mode in prior_common if mode in common]
            if not updated_common:
                break
            choices.append((op, examples, base, updated_common))
            prior_common = updated_common
        else:
            if prior_common is None:
                continue
            used_bases = [base for _, _, base, _ in choices]
            selected = choose_operator_absence_query_candidate(
                pairing,
                candidate_bases,
                prior_common,
                puzzle,
                used_bases,
            )
            if selected is None:
                continue
            mapped_base, q_lines, pred, scored = selected
            return pairing, choices, prior_common, q_lines, pred, mapped_base, used_bases, scored
    return None


def find_operator_absence_path(
    puzzle: NumericEquationPuzzle,
    gold: str,
) -> tuple[str, list[tuple[str, list[ParsedEquation], str, list[str]]], list[str], list[str], str, str, list[str], list[tuple[str, str, int, list[str], str]]] | None:
    candidate_bases = OP_ABSENCE_SYMBOL_CANDIDATES.get(puzzle.query.operator)
    if candidate_bases is None:
        return None
    visible_groups = helper_groups(puzzle)
    if not visible_groups:
        return None

    for pairing in ("BA_DC", "AB_CD"):
        choices: list[tuple[str, list[ParsedEquation], str, list[str]]] = []
        prior_common: list[str] | None = None
        for op, examples in visible_groups:
            modes = OP_ABSENCE_MODES[pairing] if prior_common is None else prior_common
            candidates = operator_absence_group_candidates(pairing, op, examples, modes)
            if not candidates:
                break
            base, common = candidates[0]
            updated_common = common if prior_common is None else [mode for mode in prior_common if mode in common]
            if not updated_common:
                break
            choices.append((op, examples, base, updated_common))
            prior_common = updated_common
        else:
            if prior_common is None:
                continue
            used_bases = [base for _, _, base, _ in choices]
            selected = choose_operator_absence_query_candidate(
                pairing,
                candidate_bases,
                prior_common,
                puzzle,
                used_bases,
            )
            if selected is None:
                continue

            final_common = prior_common
            mapped_base, q_lines, pred, scored = selected
            if pred == gold:
                q_lines, pred = query_block(pairing, mapped_base, final_common, puzzle, [], gold=gold)
                return pairing, choices, final_common, q_lines, pred, mapped_base, used_bases, scored
    return None


def symbol_mapping_lines(query_operator: str) -> list[str]:
    lines = [f"Use symbol mapping for query operator {query_operator}"]
    for op in OP_ABSENCE_SYMBOL_ORDER:
        bases = OP_ABSENCE_SYMBOL_CANDIDATES[op]
        verdict = "PASS" if op == query_operator else "no"
        lines.append(f"{op} -> {', '.join(BASE_LABEL[base] for base in bases)} {verdict}")
    return lines


def render_operator_absence_group_scan(
    pairing: str,
    op: str,
    examples: list[ParsedEquation],
    selected_base: str,
    selected_common: list[str],
    prior_common: list[str] | None,
) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    if prior_common is None:
        lines.extend(
            [
                f"First try visible operator {op}",
                f"Examples with operator {op}",
                *(f"{ex.lhs_text} = {ex.rhs_text}" for ex in examples),
                "",
                *route_lines(examples, "Visible operator"),
                "",
            ]
        )
        if {effective_rhs_len(ex.rhs_text, ex.operator) for ex in examples} == {4}:
            lines.append("We try direct templates first. If they fail, we proceed to arithmetic search on motifs BA_DC and AB_CD")
            lines.append("")
        if pairing == "BA_DC":
            lines.extend(["Try BA_DC first", ""])
    else:
        lines.extend(
            [
                f"Try visible operator {op} using the surviving motif and common formats",
                "Surviving motif",
                pairing,
                "Surviving common formats",
                *prior_common,
                "",
                f"Examples with operator {op}",
                *(f"{ex.lhs_text} = {ex.rhs_text}" for ex in examples),
                "",
                *route_lines(examples, "Visible operator"),
                "",
            ]
        )

    for base in operator_absence_family_for_examples(examples):
        modes_to_show = prior_common if prior_common is not None else OP_ABSENCE_MODES[pairing]
        block, common = attempt_block(
            pairing,
            base,
            op,
            examples,
            modes_to_show,
            common_suffix=prior_common is not None,
        )
        lines.extend(block)
        updated_common = common if prior_common is None else [mode for mode in prior_common if mode in common]
        if base == selected_base and updated_common:
            if prior_common is not None:
                lines.extend(["Updated surviving common formats", *updated_common, ""])
            return lines, updated_common
        if prior_common is None:
            lines.extend([f"{BASE_LABEL[base]} fails under {pairing}", ""])
        else:
            lines.extend(["Updated surviving common formats", *(updated_common if updated_common else ["none"])])
            lines.extend([f"{BASE_LABEL[base]} does not preserve the surviving common formats", ""])
    raise RuntimeError(f"selected base {selected_base} was not reached for operator {op}")


def render_operator_absence_ba_rejection_scan(puzzle: NumericEquationPuzzle) -> list[str]:
    lines: list[str] = ["Try BA_DC first", ""]
    prior_common: list[str] | None = None

    for op, examples in helper_groups(puzzle):
        if prior_common is None:
            lines.extend(
                [
                    f"First try visible operator {op}",
                    f"Examples with operator {op}",
                    *(f"{ex.lhs_text} = {ex.rhs_text}" for ex in examples),
                    "",
                    *route_lines(examples, "Visible operator"),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"Proceed to operator {op} using the surviving motif and common formats",
                    "Surviving motif",
                    "BA_DC",
                    "Surviving common formats",
                    *prior_common,
                    "",
                    f"Examples with operator {op}",
                    *(f"{ex.lhs_text} = {ex.rhs_text}" for ex in examples),
                    "",
                    *route_lines(examples, "Visible operator"),
                    "",
                ]
            )

        modes_to_show = prior_common if prior_common is not None else OP_ABSENCE_MODES["BA_DC"]
        surviving: list[str] | None = None
        for base in operator_absence_family_for_examples(examples):
            block, common = attempt_block(
                "BA_DC",
                base,
                op,
                examples,
                modes_to_show,
                common_suffix=prior_common is not None,
            )
            lines.extend(block)
            updated_common = common if prior_common is None else [mode for mode in prior_common if mode in common]
            if updated_common:
                lines.extend(
                    [
                        f"Operator {BASE_LABEL[base]} supports BA_DC for all visible examples in operator group {op}",
                        f"Surviving common formats for visible operator {op}",
                        *updated_common,
                        "",
                    ]
                )
                surviving = updated_common
                break
            else:
                if prior_common is None:
                    lines.extend([f"{BASE_LABEL[base]} fails under BA_DC", ""])
                else:
                    lines.extend(["Updated surviving common formats", "none"])
                    lines.extend([f"{BASE_LABEL[base]} does not preserve the surviving common formats", ""])

        if not surviving:
            if prior_common is None:
                lines.extend(
                    [
                        "No operator survives under BA_DC",
                        f"All visible examples in operator group {op} do not support BA_DC",
                    ]
                )
            else:
                lines.extend(
                    [
                        "No operator survives under BA_DC",
                        f"All visible examples in operator group {op} do not preserve the surviving common formats",
                    ]
                )
            lines.extend(["Reject BA_DC", ""])
            return lines

        prior_common = surviving

    raise RuntimeError("BA_DC unexpectedly supports all visible operator groups")


def render_operator_absence_query_candidate_search(
    pairing: str,
    puzzle: NumericEquationPuzzle,
    final_common: list[str],
    candidate_bases: list[str],
    used_bases: list[str],
    mapped_base: str,
    scored: list[tuple[str, str, int, list[str], str]],
) -> list[str]:
    remaining_bases = [base for base in candidate_bases if base not in used_bases]
    scored_by_base = {base: (answer, vote_count, values, reason) for base, answer, vote_count, values, reason in scored}
    lines = [
        f"Search remaining candidate operator families for {puzzle.query.operator}",
        "Remaining candidate operator families",
        *(BASE_LABEL[base] for base in remaining_bases if base in scored_by_base),
        "",
        "Check remaining candidate operator families by common-output agreement",
    ]

    for base in remaining_bases:
        if base not in scored_by_base:
            continue
        label = BASE_LABEL[base]
        answer, vote_count, values, _reason = scored_by_base[base]
        if label in {"template0134", "template3401"}:
            lines.extend(
                [
                    "",
                    f"Candidate {label}",
                    "Query",
                    puzzle.query.lhs_text,
                    f"{puzzle.query.left_operand_text} {puzzle.query.right_operand_text} {label}",
                    answer,
                ]
            )
        else:
            left, right, raw_text, rendered = rendered_values(puzzle.query, pairing, base, final_common)
            if rendered != values:
                raise RuntimeError(f"{puzzle.query.lhs_text} candidate search render mismatch for {label}")
            lines.extend(
                [
                    "",
                    f"Candidate {label}",
                    "Query",
                    puzzle.query.lhs_text,
                    f"{'BA DC' if pairing == 'BA_DC' else 'AB CD'} {RAW_HEADER[(pairing, base)]} {' '.join(final_common)}",
                    f"{left} {right} {raw_text} {' '.join(values)}",
                ]
            )

        counts = Counter(values)
        lines.append("Output votes")
        for value in sorted(counts, key=lambda item: (-counts[item], values.index(item))):
            count = counts[value]
            lines.append(f"{value} has {count} {'vote' if count == 1 else 'votes'}")
        lines.extend(["Highest vote count", str(vote_count)])

    lines.extend(
        [
            "",
            "Choose operator candidate with highest output agreement vote count, if tie choose the first candidate",
            BASE_LABEL[mapped_base],
        ]
    )
    return lines


def render_operator_absence_symbol_mapping_trace(row: dict[str, str]) -> tuple[str, str]:
    puzzle = parse_numeric_equation_puzzle(row["prompt"])
    if puzzle is None:
        raise RuntimeError(row["id"])
    if not is_numeric_puzzle(puzzle):
        raise RuntimeError(f"{row['id']} is not numeric-only")
    groups: dict[str, list[ParsedEquation]] = defaultdict(list)
    for ex in puzzle.examples:
        groups[ex.operator].append(ex)
    if groups[puzzle.query.operator]:
        raise RuntimeError(f"{row['id']} is not an operator-absence row")

    gold = row["answer"]
    path = find_operator_absence_path(puzzle, gold)
    if path is None:
        raise RuntimeError(f"{row['id']} is not solved by operator-absence symbol mapping")
    pairing, choices, final_common, q_lines, pred, mapped_base, used_bases, scored = path
    if pred != gold:
        raise RuntimeError(f"{row['id']} predicts {pred}, gold {gold}")

    lines = common_intro_lines(row, puzzle, gold)
    lines.extend(
        [
            "",
            "same operator examples",
            "none",
            "",
            "For operator absence type, first infer the motif and output formats from all visible operator groups",
            "Candidate output formats",
            *OP_ABSENCE_CANDIDATE_OUTPUT_FORMATS,
            "",
            "Visible operator groups",
        ]
    )
    for op, examples in helper_groups(puzzle):
        lines.append(f"Operator {op}")
        lines.extend(f"{ex.lhs_text} = {ex.rhs_text}" for ex in examples)
    lines.append("")

    if pairing == "AB_CD":
        lines.extend(render_operator_absence_ba_rejection_scan(puzzle))
        lines.extend(["Try AB_CD", ""])

    prior_common: list[str] | None = None
    for op, examples, selected_base, selected_common in choices:
        scan_lines, prior_common = render_operator_absence_group_scan(
            pairing,
            op,
            examples,
            selected_base,
            selected_common,
            prior_common,
        )
        lines.extend(scan_lines)

    if prior_common != final_common:
        raise RuntimeError(f"{row['id']} final common mismatch")

    lines.extend(
        [
            "All visible operator groups support the following motif and common formats",
            "Motif",
            pairing,
            "Common",
            *final_common,
            "",
            *symbol_mapping_lines(puzzle.query.operator),
            "",
            f"Candidate operator families for {puzzle.query.operator}",
            *(BASE_LABEL[base] for base in OP_ABSENCE_SYMBOL_CANDIDATES[puzzle.query.operator]),
            "",
            "Operator families used by the visible examples",
            *(BASE_LABEL[base] for base in used_bases),
            "",
        ]
    )
    lines.extend(
        render_operator_absence_query_candidate_search(
            pairing,
            puzzle,
            final_common,
            OP_ABSENCE_SYMBOL_CANDIDATES[puzzle.query.operator],
            used_bases,
            mapped_base,
            scored,
        )
    )
    lines.extend(q_lines)
    lines.extend(["", f"Answer: \\boxed{{{gold}}}"])
    return "\n".join(lines).rstrip() + "\n", "operator_absence_symbol_mapping"


def choose_viable_helper(
    puzzle: NumericEquationPuzzle,
    same: list[ParsedEquation],
    gold: str,
    ba_base: str,
    ba_common: list[str],
    golden_ab_section: tuple[str, list[str]] | None,
    trusted_golden_ba_block: bool = False,
) -> tuple[str, list[ParsedEquation], list[str], bool, bool, str]:
    first_failure = ""
    for helper_op, helper_examples, used_fallback in ranked_helper_groups(puzzle):
        helper_lines, ba_confirmed = helper_verify_ba_lines(helper_op, helper_examples)
        if ba_confirmed:
            _, pred = ba_gold_scan_lines(puzzle, same, gold, ba_base, ba_common)
            if pred == gold:
                return helper_op, helper_examples, helper_lines, True, used_fallback, ""
            first_failure = (
                first_failure
                or f"{helper_op} confirmed BA_DC but BA_DC continuation did not reach gold"
            )
            continue
        if golden_ab_section is not None:
            return helper_op, helper_examples, helper_lines, False, used_fallback, ""
        _, pred = ab_gold_scan_lines(puzzle, same, gold)
        if pred == gold:
            return helper_op, helper_examples, helper_lines, False, used_fallback, ""
        first_failure = first_failure or f"{helper_op} rejected BA_DC but AB_CD did not reach gold"
    return "", [], [], False, False, first_failure


def render_trace(row: dict[str, str], golden_rows: dict[str, dict[str, str]]) -> tuple[str, str]:
    puzzle = parse_numeric_equation_puzzle(row["prompt"])
    if puzzle is None:
        raise RuntimeError(row["id"])
    gold = row["answer"]
    groups: dict[str, list[ParsedEquation]] = defaultdict(list)
    for ex in puzzle.examples:
        groups[ex.operator].append(ex)
    same = groups[puzzle.query.operator]
    if len(same) != 1:
        raise RuntimeError(f"{row['id']} is not a one-same-operator row")

    if {effective_rhs_len(ex.rhs_text, ex.operator) for ex in same} == {4}:
        template = direct_template_match(same)
        if template and direct_template_query_value(template, puzzle.query) == gold:
            return render_direct_same_operator_trace(row, puzzle, same, template), "same_direct"

    lines = common_intro_lines(row, puzzle, gold)
    lines.extend(["", "same operator examples"])
    lines.extend(f"{ex.lhs_text} = {ex.rhs_text}" for ex in same)
    lines.append("one example")
    lines.append("")
    lines.extend(route_lines(same, "Same operator"))

    if {effective_rhs_len(ex.rhs_text, ex.operator) for ex in same} == {4}:
        lines.append("")
        dlines, passed = direct_template_block(puzzle.query.operator, same, helper_motif_only=False)
        if passed:
            raise RuntimeError(f"{row['id']} direct template passes but does not reach gold")
        lines.extend(dlines)
        lines.extend(["Try BA_DC first", ""])
    else:
        lines.extend(["", "Try BA_DC first", ""])

    golden_ba_block = None
    golden_ab_section = None
    if row["id"] in golden_rows:
        golden_text = (golden_rows[row["id"]].get("generated_cot") or "") + "\n" + (
            golden_rows[row["id"]].get("assistant_content") or ""
        )
        golden_ba_block = final_ba_blocks_from_text(golden_text)
        golden_ab_section = final_ab_section_from_text(golden_text)
    ba_base, ba_common, ba_lines = first_ba_survivor(same)
    lines.extend(ba_lines)
    if ba_base is None:
        lines.extend(["No BA_DC candidate supports the single same operator example"])
        lines.extend(["Try AB_CD", ""])
        ab_lines, pred = ab_gold_scan_lines(puzzle, same, gold)
        if pred != gold:
            raise RuntimeError(f"{row['id']} has no BA_DC support, but AB_CD did not reach gold")
        lines.extend(ab_lines)
        lines.extend(["", f"Answer: \\boxed{{{gold}}}"])
        return "\n".join(lines).rstrip() + "\n", "ba_no_support_ab"

    lines.extend(
        [
            f"The format BA_DC|{BASE_LABEL[ba_base]}|common supports the single same operator example",
            "Only one same operator row supports this candidate, so do not finalize yet",
            "Verify motif BA_DC using an additional helper operator group",
            "",
            "Helper operator groups",
        ]
    )
    for hop, hexamples in helper_groups(puzzle):
        lines.append(f"Operator {hop}")
        lines.extend(f"{ex.lhs_text} = {ex.rhs_text}" for ex in hexamples)

    helper_op, helper_examples, helper_lines, ba_confirmed, used_fallback, failure = choose_viable_helper(
        puzzle, same, gold, ba_base, ba_common, golden_ab_section, golden_ba_block is not None
    )
    if not helper_examples:
        raise RuntimeError(f"{row['id']} no viable helper: {failure}")

    min_all = min(len(examples) for _, examples in helper_groups(puzzle))
    if len(helper_examples) != min_all:
        raise RuntimeError(f"{row['id']} helper choice violates least-example policy")
    min_groups = [(op, examples) for op, examples in helper_groups(puzzle) if len(examples) == min_all]
    tied_non_direct_groups = [(op, examples) for op, examples in min_groups if not is_direct_helper_group(examples)]
    if tied_non_direct_groups and is_direct_helper_group(helper_examples):
        raise RuntimeError(f"{row['id']} helper choice violates non-direct-template tie-break policy")
    choose_line = "Choose the helper operator group with the least number of examples"
    lines.extend(["", choose_line])
    if len(min_groups) > 1 and tied_non_direct_groups:
        lines.append("Tie on number of examples, choose a non-direct-template group")
    lines.extend(f"{ex.lhs_text} = {ex.rhs_text}" for ex in helper_examples)
    lines.append("")
    lines.extend(helper_lines)

    if ba_confirmed:
        q_lines, pred = ba_gold_scan_lines(puzzle, same, gold, ba_base, ba_common)
        if pred != gold:
            raise RuntimeError(
                f"{row['id']} confirms BA_DC but BA_DC continuation did not reach gold"
            )
        lines.extend(return_to_ba_candidate_lines(BASE_LABEL[ba_base], ba_common))
        lines.extend(q_lines)
        kind = "helper_confirms_ba_fallback" if used_fallback else "helper_confirms_ba"
    else:
        lines.extend(["Try AB_CD", ""])
        ab_lines, pred = ab_gold_scan_lines(puzzle, same, gold)
        if pred != gold:
            raise RuntimeError(f"{row['id']} rejects BA_DC but AB_CD did not reach gold")
        lines.extend(ab_lines)
        kind = "helper_rejects_ba_fallback" if used_fallback else "helper_rejects_ba"

    lines.extend(["", f"Answer: \\boxed{{{gold}}}"])
    return "\n".join(lines).rstrip() + "\n", kind


def normalized_direct_template_spacing(lines: list[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line in {"template0134 passes all examples", "template3401 passes all examples"}:
            template = line.split()[0]
            next_idx = idx + 1
            while next_idx < len(lines) and lines[next_idx] == "":
                next_idx += 1
            if (
                next_idx < len(lines)
                and lines[next_idx] == "For direct templates, apply the passing template to get the answer"
            ):
                apply_idx = next_idx + 1
                while apply_idx < len(lines) and lines[apply_idx] == "":
                    apply_idx += 1
                if apply_idx < len(lines) and lines[apply_idx] == f"Apply {template} to the query":
                    out.extend([line, lines[next_idx], lines[apply_idx]])
                    idx = apply_idx + 1
                    continue
        out.append(line)
        idx += 1
    return out


def normalized_direct_template_motif_spacing(lines: list[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line in {"template0134 passes all examples", "template3401 passes all examples"}:
            template = line.split()[0]
            if (
                idx + 2 < len(lines)
                and lines[idx + 1] == ""
                and lines[idx + 2].startswith(f"{template} suggests candidate motif ")
            ):
                out.append(line)
                idx += 2
                continue
        out.append(line)
        idx += 1
    return out


def adapt_real_multi_same_trace_from_v2(row: dict[str, str]) -> tuple[str, str]:
    puzzle = parse_numeric_equation_puzzle(row["prompt"])
    if puzzle is None:
        raise RuntimeError(row["id"])
    if not is_numeric_puzzle(puzzle):
        raise RuntimeError(f"{row['id']} is not numeric-only")
    groups: dict[str, list[ParsedEquation]] = defaultdict(list)
    for ex in puzzle.examples:
        groups[ex.operator].append(ex)
    same = groups[puzzle.query.operator]
    if len(same) < 2:
        raise RuntimeError(f"{row['id']} has fewer than two same-operator examples")

    cot = ((row.get("generated_cot") or "") + "\n" + (row.get("assistant_content") or "")).strip()
    if not cot:
        raise RuntimeError(f"{row['id']} has empty COT")
    lines = cot.splitlines()

    try:
        same_idx = lines.index("Same operator examples")
    except ValueError:
        same_idx = lines.index("same operator examples")
    expected_examples = [f"{ex.lhs_text} = {ex.rhs_text}" for ex in same]
    actual_examples = lines[same_idx + 1 : same_idx + 1 + len(same)]
    if actual_examples != expected_examples:
        raise RuntimeError(f"{row['id']} same-operator example block does not match parsed examples")
    lines[same_idx] = "same operator examples"
    count_idx = same_idx + 1 + len(same)
    if count_idx >= len(lines) or lines[count_idx] != example_count_phrase(len(same)):
        lines.insert(count_idx, example_count_phrase(len(same)))

    apply_indices = [
        idx
        for idx, line in enumerate(lines)
        if line.startswith("Apply format ") and line.endswith("|common to the query")
    ]
    direct_apply_indices = [idx for idx, line in enumerate(lines) if line.startswith("Apply template")]
    if apply_indices:
        if len(apply_indices) != 1:
            raise RuntimeError(f"{row['id']} has {len(apply_indices)} common apply lines")
        apply_idx = apply_indices[0]
        apply_line = lines[apply_idx]
        format_label = apply_line.removeprefix("Apply format ").removesuffix("|common to the query")
        support_line = (
            f"The format {format_label}|common supports all {count_word(len(same))} same operator examples"
        )
        finalize_line = "More than one same operator row supports this candidate, so finalize"
        if support_line not in lines[max(0, apply_idx - 4) : apply_idx]:
            lines[apply_idx:apply_idx] = [support_line, finalize_line]
    elif not direct_apply_indices:
        raise RuntimeError(f"{row['id']} has no query apply line")

    lines = normalized_direct_template_spacing(lines)
    lines = normalized_direct_template_motif_spacing(lines)
    text = "\n".join(lines).rstrip()
    answer_matches = re.findall(r"Answer: \\boxed\{(.+?)\}", text)
    if not answer_matches:
        raise RuntimeError(f"{row['id']} has no boxed answer")
    if answer_matches[-1] != row["answer"]:
        raise RuntimeError(f"{row['id']} boxed answer {answer_matches[-1]} != gold {row['answer']}")

    trace = "\n".join(
        [
            f"Problem {row['id']}",
            "",
            "Question:",
            row["prompt"],
            "",
            "Gold answer:",
            row["answer"],
            "",
            "Solution:",
            "",
            text,
        ]
    ).rstrip() + "\n"
    kind = "real_multi_same_direct" if direct_apply_indices and not apply_indices else "real_multi_same_adapted"
    return trace, kind


def adapt_real_zero_same_trace_from_v2(row: dict[str, str]) -> tuple[str, str]:
    puzzle = parse_numeric_equation_puzzle(row["prompt"])
    if puzzle is None:
        raise RuntimeError(row["id"])
    if not is_numeric_puzzle(puzzle):
        raise RuntimeError(f"{row['id']} is not numeric-only")
    groups: dict[str, list[ParsedEquation]] = defaultdict(list)
    for ex in puzzle.examples:
        groups[ex.operator].append(ex)
    if groups[puzzle.query.operator]:
        raise RuntimeError(f"{row['id']} is not a zero-same-operator row")

    cot = ((row.get("generated_cot") or "") + "\n" + (row.get("assistant_content") or "")).strip()
    if not cot:
        raise RuntimeError(f"{row['id']} has empty COT")
    lines = cot.splitlines()

    try:
        same_idx = lines.index("Same operator examples")
    except ValueError:
        same_idx = lines.index("same operator examples")
    lines[same_idx] = "same operator examples"
    if same_idx + 1 >= len(lines) or lines[same_idx + 1].lower() != "none":
        raise RuntimeError(f"{row['id']} same-operator none block is missing")
    lines[same_idx + 1] = "none"

    lines = normalized_direct_template_spacing(lines)
    lines = normalized_direct_template_motif_spacing(lines)
    lines = refresh_arithmetic_trace(lines, puzzle, row["answer"])
    text = "\n".join(lines).rstrip()
    answer_matches = re.findall(r"Answer: \\boxed\{(.+?)\}", text)
    if not answer_matches:
        raise RuntimeError(f"{row['id']} has no boxed answer")
    if answer_matches[-1] != row["answer"]:
        raise RuntimeError(f"{row['id']} boxed answer {answer_matches[-1]} != gold {row['answer']}")

    trace = "\n".join(
        [
            f"Problem {row['id']}",
            "",
            "Question:",
            row["prompt"],
            "",
            "Gold answer:",
            row["answer"],
            "",
            "Solution:",
            "",
            text,
        ]
    ).rstrip() + "\n"
    return trace, "real_zero_same_operator_absence"


def is_numeric_puzzle(puzzle: NumericEquationPuzzle) -> bool:
    equations = [*puzzle.examples, puzzle.query]
    for eq in equations:
        if not eq.left_operand_text.isdigit() or not eq.right_operand_text.isdigit():
            return False
    return True


def is_synthetic_row(row: dict[str, str]) -> bool:
    return row["id"].startswith("syn_")


def has_opposite_sign_common_agreement_pattern(
    puzzle: NumericEquationPuzzle,
    same: list[ParsedEquation],
) -> bool:
    for base in family_for_examples(same, include_direct=False):
        if base not in OPPOSITE_SIGN_BRANCH_BASES:
            continue
        _, common = attempt_block("BA_DC", base, puzzle.query.operator, same)
        if not common:
            continue
        _, row, _ = query_prediction("BA_DC", base, common, puzzle, same)
        if len(set(row[3:])) != 1:
            continue
        query_raw = raw_value(puzzle.query, "BA_DC", base)
        if query_raw is None:
            continue
        query_sign = sign_of_raw(query_raw)
        if query_sign == 0:
            continue
        support_signs: set[int] = set()
        for ex in same:
            ex_raw = raw_value(ex, "BA_DC", base)
            if ex_raw is None:
                break
            support_signs.add(sign_of_raw(ex_raw))
        else:
            if len(support_signs) == 1 and 0 not in support_signs and query_sign not in support_signs:
                return True
    return False


def target_rows(rows: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    targets: list[dict[str, str]] = []
    excluded_synthetic_pattern: list[dict[str, str]] = []
    for row in rows.values():
        if row["id"] in EXCLUDED_TARGET_IDS or row["id"] in UNSTABLE_MOTIF_DRIFT_IDS:
            continue
        puzzle = parse_numeric_equation_puzzle(row["prompt"])
        if puzzle is None or not is_numeric_puzzle(puzzle):
            continue
        groups: dict[str, list[ParsedEquation]] = defaultdict(list)
        for ex in puzzle.examples:
            groups[ex.operator].append(ex)
        same = groups[puzzle.query.operator]
        if len(same) != 1:
            continue
        if is_synthetic_row(row) and has_opposite_sign_common_agreement_pattern(puzzle, same):
            excluded_synthetic_pattern.append(row)
            continue
        targets.append(row)
    return targets, excluded_synthetic_pattern


def real_v2_multi_same_rows() -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    with V2_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["id"] in UNSTABLE_MOTIF_DRIFT_IDS:
                continue
            if row.get("category") != "Numeric Equation Transformation Rules":
                continue
            if row.get("source_mode") != "real":
                continue
            if not ((row.get("generated_cot") or "") or (row.get("assistant_content") or "")):
                continue
            puzzle = parse_numeric_equation_puzzle(row["prompt"])
            if puzzle is None or not is_numeric_puzzle(puzzle):
                continue
            groups: dict[str, list[ParsedEquation]] = defaultdict(list)
            for ex in puzzle.examples:
                groups[ex.operator].append(ex)
            if len(groups[puzzle.query.operator]) >= 2:
                targets.append(row)
    return targets


def real_v2_zero_same_rows() -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    with V2_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("category") != "Numeric Equation Transformation Rules":
                continue
            if row.get("source_mode") != "real":
                continue
            if not ((row.get("generated_cot") or "") or (row.get("assistant_content") or "")):
                continue
            puzzle = parse_numeric_equation_puzzle(row["prompt"])
            if puzzle is None or not is_numeric_puzzle(puzzle):
                continue
            groups: dict[str, list[ParsedEquation]] = defaultdict(list)
            for ex in puzzle.examples:
                groups[ex.operator].append(ex)
            if len(groups[puzzle.query.operator]) == 0:
                targets.append(row)
    return targets


def operator_absence_symbol_mapping_rows(
    rows: dict[str, dict[str, str]],
    train_ids: set[str],
) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for row in rows.values():
        if row["id"] not in train_ids:
            continue
        puzzle = parse_numeric_equation_puzzle(row["prompt"])
        if puzzle is None or not is_numeric_puzzle(puzzle):
            continue
        groups: dict[str, list[ParsedEquation]] = defaultdict(list)
        for ex in puzzle.examples:
            groups[ex.operator].append(ex)
        if groups[puzzle.query.operator]:
            continue
        if puzzle.query.operator not in OP_ABSENCE_SYMBOL_CANDIDATES:
            continue
        if find_operator_absence_path(puzzle, row["answer"]) is not None:
            targets.append(row)
    return sorted(targets, key=lambda item: item["id"])


def write_excluded_pattern_manifest(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Excluded Synthetic Opposite-Sign Common-Agreement Rows",
        "",
        "These synthetic numeric-equation rows are intentionally excluded from motif-drift trace generation.",
        "They have a BA_DC difference-family candidate where all common output formats agree on the query,",
        "but the single same-operator example and query are on opposite nonzero sign branches.",
        "Do not port these rows into v3 unless we define a new faithful policy for this pattern.",
        "",
        f"Count: {len(rows)}",
        "",
        "IDs:",
    ]
    lines.extend(f"- `{row['id']}`" for row in sorted(rows, key=lambda item: item["id"]))
    EXCLUDED_PATTERN_MANIFEST.write_text("\n".join(lines).rstrip() + "\n")


def write_unstable_manifest() -> None:
    lines = [
        "# Excluded Unstable Motif-Drift IDs",
        "",
        "These rows are intentionally excluded from motif-drift trace generation for now.",
        "Do not port them into v3 until we define a faithful policy for the listed ambiguity.",
        "",
        "Skipped-supported-candidate cases:",
    ]
    skipped_supported = [
        "00d8b3db",
        "2beb5851",
        "febd3442",
        "7ac90433",
    ]
    lines.extend(f"- `{item}`" for item in skipped_supported)
    lines.extend(
        [
            "",
            "Unanimous-common-output continuation cases:",
        ]
    )
    unanimous_continuation = [
        "12d4a2df",
        "1b6366af",
    ]
    lines.extend(f"- `{item}`" for item in unanimous_continuation)
    renderer_policy_regressions = [
        "31eb8247",
        "91b34547",
        "9a5b6b28",
    ]
    lines.extend(
        [
            "",
            "Renderer-policy regression cases:",
        ]
    )
    lines.extend(f"- `{item}`" for item in renderer_policy_regressions)
    lines.extend(
        [
            "",
            f"Count: {len(skipped_supported) + len(unanimous_continuation) + len(renderer_policy_regressions)}",
        ]
    )
    EXCLUDED_UNSTABLE_MANIFEST.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    rows: dict[str, dict[str, str]] = {}
    with (ROOT / "data/train.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["id"]] = row
    train_ids = set(rows)

    golden_rows: dict[str, dict[str, str]] = {}
    for path in GOLDEN_PATHS:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                golden_rows.setdefault(row["id"], row)
                rows.setdefault(row["id"], row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("*.txt"):
        path.unlink()
    stats: Counter[str] = Counter()
    failures: list[str] = []
    targets, excluded_synthetic_pattern = target_rows(rows)
    write_excluded_pattern_manifest(excluded_synthetic_pattern)
    write_unstable_manifest()
    for row in targets:
        try:
            trace, kind = render_trace(row, golden_rows)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{row['id']}: {exc}")
            continue
        (OUT_DIR / f"{row['id']}.txt").write_text(trace)
        stats[kind] += 1

    real_multi_targets = real_v2_multi_same_rows()
    for row in real_multi_targets:
        try:
            trace, kind = render_multi_same_real_trace(row)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{row['id']}: {exc}")
            continue
        (OUT_DIR / f"{row['id']}.txt").write_text(trace)
        stats[kind] += 1

    operator_absence_targets = operator_absence_symbol_mapping_rows(rows, train_ids)
    for row in operator_absence_targets:
        try:
            trace, kind = render_operator_absence_symbol_mapping_trace(row)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{row['id']}: {exc}")
            continue
        (OUT_DIR / f"{row['id']}.txt").write_text(trace)
        stats[kind] += 1

    print(f"target rows: {len(targets)}")
    print(f"real v2 multi-same target rows: {len(real_multi_targets)}")
    print(f"operator-absence symbol-mapping target rows: {len(operator_absence_targets)}")
    print(f"excluded synthetic opposite-sign common-agreement rows: {len(excluded_synthetic_pattern)}")
    print(f"wrote traces: {sum(stats.values())}")
    for kind, count in sorted(stats.items()):
        print(f"{kind}: {count}")
    if failures:
        print("skipped failures:")
        for failure in failures:
            print(failure)


if __name__ == "__main__":
    main()
