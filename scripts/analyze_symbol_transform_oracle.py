#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.numeric_equation import classify_equation_vs_symbol
from nemotron_baseline.symbol_transform import (
    BASE_RULES,
    CORE_RULE_NAMES,
    MOTIF_SCAN_ORDER,
    BaseRule,
    Motif,
    SymbolExample,
    SymbolTransformPuzzle,
    _merge_maps,
    _pair_to_xy,
    parse_symbol_transform_puzzle,
    same_operator_examples,
)


OP_TOKEN = "__OP__"
CURRENT_OUTPUT_MODES = frozenset({"raw", "rev", "abs"})


EXTRA_BASE_RULES = (
    BaseRule("x * y + x", lambda x, y: x * y + x, 30, "mul_linear", True),
    BaseRule("x * y + y", lambda x, y: x * y + y, 31, "mul_linear", True),
    BaseRule("x * y - x", lambda x, y: x * y - x, 32, "mul_linear", True),
    BaseRule("x * y - y", lambda x, y: x * y - y, 33, "mul_linear", True),
    BaseRule("x^2 + y", lambda x, y: x * x + y, 34, "quadratic", True),
    BaseRule("y^2 + x", lambda x, y: y * y + x, 35, "quadratic", True),
)


OUTPUT_MODE_BANKS = {
    "current": ("raw", "rev", "abs"),
    "formats": (
        "raw",
        "rev",
        "abs",
        "zpad",
        "zpad_rev",
        "op_prefix",
        "op_suffix",
        "op_prefix_rev",
        "op_suffix_rev",
        "op_prefix_if_neg",
        "op_suffix_rev_if_neg",
    ),
    "all": (
        "raw",
        "rev",
        "abs",
        "zpad",
        "zpad_rev",
        "last",
        "last_rev",
        "op_prefix",
        "op_suffix",
        "op_prefix_rev",
        "op_suffix_rev",
        "op_prefix_if_neg",
        "op_suffix_rev_if_neg",
    ),
}


@dataclass(frozen=True)
class OracleCandidate:
    stage: str
    motif: Motif
    query_rule: BaseRule
    output_mode: str
    map_size: int
    example_count: int
    chosen_rules: tuple[tuple[str, str], ...]

    @property
    def novelty(self) -> str:
        has_extra_rule = self.query_rule.name not in CORE_RULE_NAMES
        has_extra_output = self.output_mode not in CURRENT_OUTPUT_MODES
        if has_extra_rule and has_extra_output:
            return "operation_and_output_extension"
        if has_extra_rule:
            return "operation_extension"
        if has_extra_output:
            return "output_extension"
        return "current_core_space"

    @property
    def label(self) -> str:
        return f"{self.motif.pairing}|{self.output_mode}|{self.query_rule.name}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Oracle-diagnose missed symbol_transform rows using the gold answer as a constraint."
    )
    parser.add_argument("--input-csv", default="data/train.csv")
    parser.add_argument(
        "--current-analysis-csv",
        default="data/symbol_transform_solver_analysis_core.csv",
        help="Analysis CSV from scripts/analyze_symbol_transform_solver.py.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/symbol_transform_oracle_analysis.csv",
        help="Where to write per-row oracle diagnostics.",
    )
    parser.add_argument(
        "--target",
        choices=("missed", "no_rule", "wrong", "ambiguous", "no_rule_wrong"),
        default="missed",
        help="Which current rows to oracle-diagnose.",
    )
    parser.add_argument(
        "--rule-bank",
        choices=("core", "asymmetric", "extended"),
        default="extended",
    )
    parser.add_argument(
        "--output-mode-bank",
        choices=tuple(OUTPUT_MODE_BANKS),
        default="all",
    )
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Also run slower row-global motif oracle after same-operator oracle.",
    )
    parser.add_argument(
        "--no-tiered",
        action="store_true",
        help="Disable tiered search and try the requested full rule/output bank at once.",
    )
    parser.add_argument("--max-states-per-rule", type=int, default=80)
    parser.add_argument("--max-combined-states", type=int, default=120)
    parser.add_argument("--max-candidates-per-row", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def _rules_for_bank(name: str) -> tuple[BaseRule, ...]:
    if name == "core":
        return tuple(rule for rule in BASE_RULES if rule.name in CORE_RULE_NAMES)
    if name == "asymmetric":
        return tuple(rule for rule in BASE_RULES if rule.name in CORE_RULE_NAMES or rule.is_asymmetric)
    if name == "extended":
        return tuple(BASE_RULES) + EXTRA_BASE_RULES
    raise ValueError(f"Unknown rule bank: {name}")


def _tier_specs(
    rule_bank: str,
    output_mode_bank: str,
) -> tuple[tuple[str, tuple[BaseRule, ...], tuple[str, ...]], ...]:
    core_rules = _rules_for_bank("core")
    requested_rules = _rules_for_bank(rule_bank)
    current_modes = OUTPUT_MODE_BANKS["current"]
    requested_modes = OUTPUT_MODE_BANKS[output_mode_bank]

    specs: list[tuple[str, tuple[BaseRule, ...], tuple[str, ...]]] = [
        ("current_core_space", core_rules, current_modes),
    ]
    if any(mode not in CURRENT_OUTPUT_MODES for mode in requested_modes):
        specs.append(("output_extension", core_rules, requested_modes))
    if any(rule.name not in CORE_RULE_NAMES for rule in requested_rules):
        specs.append(("operation_extension", requested_rules, current_modes))
    if (
        any(mode not in CURRENT_OUTPUT_MODES for mode in requested_modes)
        and any(rule.name not in CORE_RULE_NAMES for rule in requested_rules)
    ):
        specs.append(("operation_and_output_extension", requested_rules, requested_modes))
    return tuple(specs)


def _digit_tokens(text: str) -> tuple[int | str, ...] | None:
    if not text.isdigit():
        return None
    return tuple(int(char) for char in text)


def _render_tokens(
    value: int | None,
    output_mode: str,
    target_len: int,
) -> tuple[int | str, ...] | None:
    if value is None:
        return None

    if output_mode == "abs":
        return _digit_tokens(str(abs(value))) if len(str(abs(value))) == target_len else None

    if output_mode == "op_prefix_if_neg":
        text = str(abs(value)) if value < 0 else str(value)
        tokens = _digit_tokens(text)
        if tokens is None:
            return None
        rendered = (OP_TOKEN, *tokens) if value < 0 else tokens
        return rendered if len(rendered) == target_len else None

    if output_mode == "op_suffix_rev_if_neg":
        text = str(abs(value))[::-1] if value < 0 else str(value)
        tokens = _digit_tokens(text)
        if tokens is None:
            return None
        rendered = (*tokens, OP_TOKEN) if value < 0 else tokens
        return rendered if len(rendered) == target_len else None

    if value < 0:
        return None

    text = str(value)
    if output_mode == "raw":
        rendered = text
    elif output_mode == "rev":
        rendered = text[::-1]
    elif output_mode == "zpad":
        if len(text) > target_len:
            return None
        rendered = text.zfill(target_len)
    elif output_mode == "zpad_rev":
        if len(text) > target_len:
            return None
        rendered = text.zfill(target_len)[::-1]
    elif output_mode == "last":
        rendered = str(value % (10**target_len)).zfill(target_len)
    elif output_mode == "last_rev":
        rendered = str(value % (10**target_len)).zfill(target_len)[::-1]
    elif output_mode == "op_prefix":
        tokens = _digit_tokens(text)
        if tokens is None:
            return None
        rendered_tokens = (OP_TOKEN, *tokens)
        return rendered_tokens if len(rendered_tokens) == target_len else None
    elif output_mode == "op_suffix":
        tokens = _digit_tokens(text)
        if tokens is None:
            return None
        rendered_tokens = (*tokens, OP_TOKEN)
        return rendered_tokens if len(rendered_tokens) == target_len else None
    elif output_mode == "op_prefix_rev":
        tokens = _digit_tokens(text[::-1])
        if tokens is None:
            return None
        rendered_tokens = (OP_TOKEN, *tokens)
        return rendered_tokens if len(rendered_tokens) == target_len else None
    elif output_mode == "op_suffix_rev":
        tokens = _digit_tokens(text[::-1])
        if tokens is None:
            return None
        rendered_tokens = (*tokens, OP_TOKEN)
        return rendered_tokens if len(rendered_tokens) == target_len else None
    else:
        return None

    tokens = _digit_tokens(rendered)
    return tokens if tokens is not None and len(tokens) == target_len else None


def _extend_state_for_example(
    example: SymbolExample,
    motif: Motif,
    base_rule: BaseRule,
    output_mode: str,
    state: dict[str, int],
) -> list[dict[str, int]]:
    lhs_symbols = (example.lhs[0], example.lhs[1], example.lhs[3], example.lhs[4])
    unknown_lhs_symbols = sorted({symbol for symbol in lhs_symbols if symbol not in state})
    used_digits = set(state.values())
    available_digits = [digit for digit in range(10) if digit not in used_digits]
    out: list[dict[str, int]] = []

    for digits in itertools.permutations(available_digits, len(unknown_lhs_symbols)):
        local = dict(state)
        local_used = set(used_digits)
        for symbol, digit in zip(unknown_lhs_symbols, digits):
            local[symbol] = digit
            local_used.add(digit)

        xy = _pair_to_xy(motif.pairing, example.lhs, local)
        if xy is None:
            continue
        tokens = _render_tokens(base_rule.apply(*xy), output_mode, len(example.rhs))
        if tokens is None:
            continue

        ok = True
        for symbol, token in zip(example.rhs, tokens):
            if token == OP_TOKEN:
                if symbol != example.operator or symbol in local:
                    ok = False
                    break
                continue

            digit = int(token)
            existing = local.get(symbol)
            if existing is not None:
                if existing != digit:
                    ok = False
                    break
            else:
                if digit in local_used:
                    ok = False
                    break
                local[symbol] = digit
                local_used.add(digit)
        if ok:
            out.append(local)
    return out


def _fit_examples(
    examples: tuple[SymbolExample, ...],
    motif: Motif,
    base_rule: BaseRule,
    output_mode: str,
    *,
    max_states: int,
) -> tuple[dict[str, int], ...]:
    states: list[dict[str, int]] = [{}]
    for example in examples:
        next_states: list[dict[str, int]] = []
        for state in states:
            next_states.extend(
                _extend_state_for_example(example, motif, base_rule, output_mode, state)
            )
        if not next_states:
            return ()
        next_states.sort(key=lambda item: (-len(item), tuple(sorted(item.items()))))
        states = next_states[:max_states]
    return tuple(states)


def _candidate_sort_key(candidate: OracleCandidate) -> tuple[int, int, int, int, str]:
    novelty_rank = {
        "current_core_space": 0,
        "output_extension": 1,
        "operation_extension": 2,
        "operation_and_output_extension": 3,
    }[candidate.novelty]
    stage_rank = 0 if candidate.stage == "same_operator_oracle" else 1
    motif_rank = MOTIF_SCAN_ORDER.index(candidate.motif) if candidate.motif in MOTIF_SCAN_ORDER else 999
    return (novelty_rank, stage_rank, motif_rank, candidate.query_rule.rank, candidate.label)


def _same_operator_oracle(
    puzzle: SymbolTransformPuzzle,
    answer: str,
    rules: tuple[BaseRule, ...],
    output_modes: tuple[str, ...],
    *,
    max_states_per_rule: int,
    max_candidates: int,
) -> tuple[OracleCandidate, ...]:
    same = same_operator_examples(puzzle)
    if not same:
        return ()

    examples = (*same, SymbolExample(lhs=puzzle.query, rhs=answer))
    candidates: list[OracleCandidate] = []
    for motif in MOTIF_SCAN_ORDER:
        for rule in rules:
            for output_mode in output_modes:
                states = _fit_examples(
                    examples,
                    motif,
                    rule,
                    output_mode,
                    max_states=max_states_per_rule,
                )
                if not states:
                    continue
                candidates.append(
                    OracleCandidate(
                        stage="same_operator_oracle",
                        motif=Motif(motif.pairing, output_mode),
                        query_rule=rule,
                        output_mode=output_mode,
                        map_size=max(len(state) for state in states),
                        example_count=len(examples),
                        chosen_rules=((puzzle.query_operator, rule.name),),
                    )
                )
                if len(candidates) >= max_candidates:
                    return tuple(sorted(candidates, key=_candidate_sort_key))
    return tuple(sorted(candidates, key=_candidate_sort_key))


def _group_examples_by_operator(
    examples: Iterable[SymbolExample],
) -> dict[str, tuple[SymbolExample, ...]]:
    groups: dict[str, list[SymbolExample]] = {}
    for example in examples:
        groups.setdefault(example.operator, []).append(example)
    return {operator: tuple(items) for operator, items in groups.items()}


def _global_oracle(
    puzzle: SymbolTransformPuzzle,
    answer: str,
    rules: tuple[BaseRule, ...],
    output_modes: tuple[str, ...],
    *,
    max_states_per_rule: int,
    max_combined_states: int,
    max_candidates: int,
) -> tuple[OracleCandidate, ...]:
    examples = (*puzzle.examples, SymbolExample(lhs=puzzle.query, rhs=answer))
    groups = _group_examples_by_operator(examples)
    candidates: list[OracleCandidate] = []

    for base_motif in MOTIF_SCAN_ORDER:
        for output_mode in output_modes:
            motif = Motif(base_motif.pairing, output_mode)
            fits_by_operator: dict[str, list[tuple[BaseRule, tuple[dict[str, int], ...]]]] = {}
            for operator, operator_examples in groups.items():
                operator_fits: list[tuple[BaseRule, tuple[dict[str, int], ...]]] = []
                for rule in rules:
                    states = _fit_examples(
                        operator_examples,
                        motif,
                        rule,
                        output_mode,
                        max_states=max_states_per_rule,
                    )
                    if states:
                        operator_fits.append((rule, states))
                if not operator_fits:
                    fits_by_operator = {}
                    break
                fits_by_operator[operator] = operator_fits
            if not fits_by_operator:
                continue

            partials: list[tuple[dict[str, int], dict[str, BaseRule]]] = [({}, {})]
            for operator in sorted(groups, key=lambda item: (item != puzzle.query_operator, item)):
                next_partials: list[tuple[dict[str, int], dict[str, BaseRule]]] = []
                for current_map, chosen_rules in partials:
                    for rule, states in fits_by_operator[operator]:
                        for state in states:
                            merged = _merge_maps(current_map, state)
                            if merged is None:
                                continue
                            next_rules = dict(chosen_rules)
                            next_rules[operator] = rule
                            next_partials.append((merged, next_rules))
                if not next_partials:
                    partials = []
                    break
                next_partials.sort(
                    key=lambda item: (
                        sum(rule.rank for rule in item[1].values()),
                        -len(item[0]),
                        tuple(sorted((op, rule.name) for op, rule in item[1].items())),
                    )
                )
                partials = next_partials[:max_combined_states]
            if not partials:
                continue

            for sym_to_digit, chosen_rules in partials[:3]:
                query_rule = chosen_rules[puzzle.query_operator]
                candidates.append(
                    OracleCandidate(
                        stage="global_motif_oracle",
                        motif=motif,
                        query_rule=query_rule,
                        output_mode=output_mode,
                        map_size=len(sym_to_digit),
                        example_count=len(examples),
                        chosen_rules=tuple(
                            sorted((operator, rule.name) for operator, rule in chosen_rules.items())
                        ),
                    )
                )
                if len(candidates) >= max_candidates:
                    return tuple(sorted(candidates, key=_candidate_sort_key))
    return tuple(sorted(candidates, key=_candidate_sort_key))


def _load_current_analysis(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def _is_target(row: dict[str, str], current: dict[str, str], target: str) -> bool:
    is_exact = current.get("is_exact") == "1"
    prediction = current.get("prediction", "")
    method = current.get("method", "")
    confidence = current.get("confidence", "")
    is_wrong_prediction = bool(prediction) and not is_exact

    if target == "missed":
        return not is_exact
    if target == "no_rule":
        return method == "no_rule"
    if target == "wrong":
        return is_wrong_prediction
    if target == "ambiguous":
        return confidence == "ambiguous"
    if target == "no_rule_wrong":
        return method == "no_rule" or is_wrong_prediction
    raise ValueError(target)


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_csv)
    current_analysis_path = Path(args.current_analysis_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current_by_id = _load_current_analysis(current_analysis_path)
    rules = _rules_for_bank(args.rule_bank)
    output_modes = OUTPUT_MODE_BANKS[args.output_mode_bank]
    tier_specs = (
        (("full_requested_space", rules, output_modes),)
        if args.no_tiered
        else _tier_specs(args.rule_bank, args.output_mode_bank)
    )

    records: list[dict[str, str | int]] = []
    summary: Counter[str] = Counter()
    novelty_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    output_mode_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    unseen_answer_counts: Counter[str] = Counter()
    target_count = 0

    with input_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompt = row["prompt"]
            if "transformation rules" not in prompt.lower():
                continue
            if classify_equation_vs_symbol(prompt) != "symbol_transform":
                continue

            current = current_by_id.get(row["id"])
            if current is None or not _is_target(row, current, args.target):
                continue

            target_count += 1
            puzzle = parse_symbol_transform_puzzle(prompt)
            if puzzle is None:
                continue

            candidates: tuple[OracleCandidate, ...] = ()
            for _tier_name, tier_rules, tier_output_modes in tier_specs:
                same_candidates = _same_operator_oracle(
                    puzzle,
                    row["answer"],
                    tier_rules,
                    tier_output_modes,
                    max_states_per_rule=args.max_states_per_rule,
                    max_candidates=args.max_candidates_per_row,
                )
                global_candidates: tuple[OracleCandidate, ...] = ()
                if args.include_global and not same_candidates:
                    global_candidates = _global_oracle(
                        puzzle,
                        row["answer"],
                        tier_rules,
                        tier_output_modes,
                        max_states_per_rule=args.max_states_per_rule,
                        max_combined_states=args.max_combined_states,
                        max_candidates=args.max_candidates_per_row,
                    )
                candidates = tuple(
                    sorted((*same_candidates, *global_candidates), key=_candidate_sort_key)
                )
                if candidates:
                    break

            best = candidates[0] if candidates else None
            status = best.novelty if best is not None else "unexplained"
            seen_symbols = set(puzzle.query)
            for example in puzzle.examples:
                seen_symbols.update(example.lhs)
                seen_symbols.update(example.rhs)
            answer_unseen_symbols = "".join(sorted(set(row["answer"]) - seen_symbols))
            summary[status] += 1
            if answer_unseen_symbols:
                unseen_answer_counts[status] += 1
            if best is not None:
                novelty_counts[best.novelty] += 1
                stage_counts[best.stage] += 1
                output_mode_counts[best.output_mode] += 1
                rule_counts[best.query_rule.name] += 1
                family_counts[best.query_rule.family] += 1

            records.append(
                {
                    "id": row["id"],
                    "answer": row["answer"],
                    "current_method": current.get("method", ""),
                    "current_prediction": current.get("prediction", ""),
                    "current_confidence": current.get("confidence", ""),
                    "oracle_status": status,
                    "answer_unseen_symbols": answer_unseen_symbols,
                    "candidate_count": len(candidates),
                    "best_stage": best.stage if best else "",
                    "best_motif": best.motif.label if best else "",
                    "best_rule": best.query_rule.name if best else "",
                    "best_rule_family": best.query_rule.family if best else "",
                    "best_output_mode": best.output_mode if best else "",
                    "best_map_size": best.map_size if best else 0,
                    "best_label": best.label if best else "",
                    "candidate_labels": " || ".join(candidate.label for candidate in candidates[:12]),
                }
            )

            if args.limit and target_count >= args.limit:
                break

    fieldnames = [
        "id",
        "answer",
        "current_method",
        "current_prediction",
        "current_confidence",
        "oracle_status",
        "answer_unseen_symbols",
        "candidate_count",
        "best_stage",
        "best_motif",
        "best_rule",
        "best_rule_family",
        "best_output_mode",
        "best_map_size",
        "best_label",
        "candidate_labels",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Target rows: {target_count}")
    print(f"Explained by oracle: {target_count - summary['unexplained']}")
    print(f"Unexplained: {summary['unexplained']}")
    print(f"Wrote: {output_path}")

    print("\nOracle status:")
    for key, count in summary.most_common():
        print(f"  {key}: {count}")

    print("\nStages:")
    for key, count in stage_counts.most_common():
        print(f"  {key}: {count}")

    print("\nOutput modes:")
    for key, count in output_mode_counts.most_common(16):
        print(f"  {key}: {count}")

    print("\nRules:")
    for key, count in rule_counts.most_common(20):
        print(f"  {key}: {count}")

    print("\nRule families:")
    for key, count in family_counts.most_common():
        print(f"  {key}: {count}")

    print("\nRows with answer symbols unseen in prompt:")
    for key, count in unseen_answer_counts.most_common():
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
