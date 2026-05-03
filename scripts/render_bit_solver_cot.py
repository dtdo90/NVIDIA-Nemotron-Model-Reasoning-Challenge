#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BIT_LABELS = {
    "Bit Manipulation",
    "bitwise and binary transformation tasks",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render deterministic, cleaner bit-manipulation reasoning traces from the "
            "notebook solver for high-confidence rows only."
        )
    )
    parser.add_argument("--input-csv", default="data/trainable/train_cot_gpt_oss_failed.csv")
    parser.add_argument(
        "--output-csv",
        default="data/trainable/train_cot_gpt_oss_failed_bit_solver_high_conf.csv",
    )
    parser.add_argument(
        "--skipped-csv",
        default="data/trainable/train_cot_gpt_oss_failed_bit_solver_high_conf_skipped.csv",
    )
    parser.add_argument(
        "--notebook-path",
        default="kaggle_notebooks/bit-manipulation-solver-cot-generator.ipynb",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of input rows processed.",
    )
    parser.add_argument(
        "--keep-original-column",
        action="store_true",
        default=True,
        help="Preserve any existing generated_cot in generated_cot_original.",
    )
    parser.add_argument(
        "--no-keep-original-column",
        dest="keep_original_column",
        action="store_false",
        help="Do not preserve an existing generated_cot column separately.",
    )
    return parser.parse_args()


def is_bit_row(row: dict[str, str]) -> bool:
    label = (row.get("label") or "").strip()
    row_type = (row.get("type") or "").strip()
    prompt = row.get("prompt") or ""
    return (
        label in BIT_LABELS
        or row_type in BIT_LABELS
        or "bit manipulation" in prompt.lower()
        or "8-bit binary" in prompt.lower()
    )


def load_solver_namespace(notebook_path: Path) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    solver_cell_source = "".join(notebook["cells"][5]["source"])
    solver_cell_source = solver_cell_source.split('print("Solver loaded successfully.")')[0]
    namespace: dict[str, Any] = {
        "re": re,
        "itertools": itertools,
        "Counter": Counter,
        "defaultdict": defaultdict,
    }
    exec(solver_cell_source, namespace)
    return namespace


def format_bits(bits: list[int]) -> str:
    return "".join(str(bit) for bit in bits)


def get_function_maps(namespace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    unary_functions = {name: func for name, func in namespace["FUNC_1"]}
    symmetric_functions = dict(namespace["FUNC_2_ORIG"])
    asymmetric_functions = {name: func for name, func in namespace["FUNC_2_ASYM"]}
    ternary_functions = {name: func for name, func in namespace["FUNC_3"]}
    binary_functions = {}
    binary_functions.update(symmetric_functions)
    binary_functions.update(asymmetric_functions)
    return unary_functions, binary_functions, ternary_functions


def evaluate_candidate(candidate: tuple[str, tuple[int, ...], int, int], query_bits: list[int], namespace: dict[str, Any]) -> int:
    unary_functions, binary_functions, ternary_functions = get_function_maps(namespace)
    name, inputs, predicted_value, _complexity = candidate
    if name == "C0":
        return 0
    if name == "C1":
        return 1
    if len(inputs) == 1 and name in unary_functions:
        return unary_functions[name](query_bits[inputs[0]])
    if len(inputs) == 2 and name in binary_functions:
        return binary_functions[name](query_bits[inputs[0]], query_bits[inputs[1]])
    if len(inputs) == 3 and name in ternary_functions:
        return ternary_functions[name](
            query_bits[inputs[0]],
            query_bits[inputs[1]],
            query_bits[inputs[2]],
        )
    composite_match = re.fullmatch(r"([A-Z]+)\(([A-Z]+)\)", name)
    if composite_match and len(inputs) == 3:
        outer_name, inner_name = composite_match.groups()
        inner_value = binary_functions[inner_name](query_bits[inputs[0]], query_bits[inputs[1]])
        return binary_functions[outer_name](inner_value, query_bits[inputs[2]])
    return predicted_value


def describe_candidate(candidate: tuple[str, tuple[int, ...], int, int], query_bits: list[int], output_bit: int, namespace: dict[str, Any]) -> str:
    name, inputs, _predicted_value, _complexity = candidate
    actual_value = evaluate_candidate(candidate, query_bits, namespace)
    if name == "C0":
        return f"bit {output_bit} is always 0 in the examples, so it stays 0"
    if name == "C1":
        return f"bit {output_bit} is always 1 in the examples, so it stays 1"

    if len(inputs) == 1 and name == "ID":
        return f"bit {output_bit} = in[{inputs[0]}] = {query_bits[inputs[0]]}"
    if len(inputs) == 1 and name == "NOT":
        return f"bit {output_bit} = NOT(in[{inputs[0]}]) = NOT({query_bits[inputs[0]]}) = {actual_value}"

    if len(inputs) == 2:
        left = query_bits[inputs[0]]
        right = query_bits[inputs[1]]
        return (
            f"bit {output_bit} = {name}(in[{inputs[0]}], in[{inputs[1]}]) = "
            f"{name}({left}, {right}) = {actual_value}"
        )

    if len(inputs) == 3 and re.fullmatch(r"[A-Z]+", name):
        first = query_bits[inputs[0]]
        second = query_bits[inputs[1]]
        third = query_bits[inputs[2]]
        return (
            f"bit {output_bit} = {name}(in[{inputs[0]}], in[{inputs[1]}], in[{inputs[2]}]) = "
            f"{name}({first}, {second}, {third}) = {actual_value}"
        )

    composite_match = re.fullmatch(r"([A-Z]+)\(([A-Z]+)\)", name)
    if composite_match and len(inputs) == 3:
        outer_name, inner_name = composite_match.groups()
        first = query_bits[inputs[0]]
        second = query_bits[inputs[1]]
        third = query_bits[inputs[2]]
        inner_value = evaluate_candidate((inner_name, (inputs[0], inputs[1]), 0, 3), query_bits, namespace)
        return (
            f"bit {output_bit} = {outer_name}({inner_name}(in[{inputs[0]}], in[{inputs[1]}]), in[{inputs[2]}]) = "
            f"{outer_name}({inner_name}({first}, {second})={inner_value}, {third}) = {actual_value}"
        )

    return f"bit {output_bit} = {actual_value}"


def detect_xor_rule(examples: list[tuple[str, str]], namespace: dict[str, Any]) -> str | None:
    bits = namespace["bits"]
    xor_mask = [a ^ b for a, b in zip(bits(examples[0][0]), bits(examples[0][1]))]
    if all([a ^ b for a, b in zip(bits(input_bits), bits(output_bits))] == xor_mask for input_bits, output_bits in examples[1:]):
        return format_bits(xor_mask)
    return None


def detect_rotation_rule(examples: list[tuple[str, str]], namespace: dict[str, Any]) -> tuple[str, int] | None:
    bits = namespace["bits"]
    for shift in range(1, 8):
        if all(bits(input_bits)[shift:] + bits(input_bits)[:shift] == bits(output_bits) for input_bits, output_bits in examples):
            return ("left", shift)
        if all(bits(input_bits)[-shift:] + bits(input_bits)[:-shift] == bits(output_bits) for input_bits, output_bits in examples):
            return ("right", shift)
    return None


def detect_rotxor_rule(examples: list[tuple[str, str]], namespace: dict[str, Any]) -> tuple[int, str] | None:
    bits = namespace["bits"]
    for shift in range(1, 8):
        rotated_first = bits(examples[0][0])[shift:] + bits(examples[0][0])[:shift]
        xor_mask = [a ^ b for a, b in zip(rotated_first, bits(examples[0][1]))]
        if all(
            [a ^ b for a, b in zip(bits(input_bits)[shift:] + bits(input_bits)[:shift], xor_mask)] == bits(output_bits)
            for input_bits, output_bits in examples[1:]
        ):
            return (shift, format_bits(xor_mask))
    return None


def detect_revxor_rule(examples: list[tuple[str, str]], namespace: dict[str, Any]) -> str | None:
    bits = namespace["bits"]
    reversed_first = bits(examples[0][0])[::-1]
    xor_mask = [a ^ b for a, b in zip(reversed_first, bits(examples[0][1]))]
    if all([a ^ b for a, b in zip(bits(input_bits)[::-1], xor_mask)] == bits(output_bits) for input_bits, output_bits in examples[1:]):
        return format_bits(xor_mask)
    return None


def detect_permutation_rule(examples: list[tuple[str, str]], namespace: dict[str, Any]) -> list[tuple[int, bool]] | None:
    bits = namespace["bits"]
    input_rows = [bits(input_bits) for input_bits, _ in examples]
    output_rows = [bits(output_bits) for _, output_bits in examples]
    mapping: list[tuple[int, bool]] = []
    for output_bit in range(8):
        target = [output_rows[row_index][output_bit] for row_index in range(len(examples))]
        matched = None
        for input_bit in range(8):
            source = [input_rows[row_index][input_bit] for row_index in range(len(examples))]
            if source == target:
                matched = (input_bit, False)
                break
            if [1 - bit for bit in source] == target:
                matched = (input_bit, True)
                break
        if matched is None:
            return None
        mapping.append(matched)
    return mapping


def detect_uniform_two_input_rule(examples: list[tuple[str, str]], namespace: dict[str, Any]) -> tuple[str, int, int] | None:
    bits = namespace["bits"]
    input_rows = [bits(input_bits) for input_bits, _ in examples]
    output_rows = [bits(output_bits) for _, output_bits in examples]
    for function_name, function in namespace["FUNC_2_ORIG"]:
        for first_shift in range(8):
            for second_shift in range(first_shift + 1, 8):
                matches_all = True
                for row_index in range(len(examples)):
                    for output_bit in range(8):
                        computed_value = function(
                            input_rows[row_index][(output_bit + first_shift) % 8],
                            input_rows[row_index][(output_bit + second_shift) % 8],
                        )
                        if computed_value != output_rows[row_index][output_bit]:
                            matches_all = False
                            break
                    if not matches_all:
                        break
                if matches_all:
                    return (function_name, first_shift, second_shift)
    return None


def detect_mixed_rule(examples: list[tuple[str, str]], namespace: dict[str, Any], query_bits: list[int]) -> dict[str, Any] | None:
    bits = namespace["bits"]
    input_rows = [bits(input_bits) for input_bits, _ in examples]
    output_rows = [bits(output_bits) for _, output_bits in examples]
    for identity_shift in range(8):
        fixed_rules: dict[int, tuple[str, int]] = {}
        for output_bit in range(8):
            source_index = (output_bit + identity_shift) % 8
            target = [output_rows[row_index][output_bit] for row_index in range(len(examples))]
            source = [input_rows[row_index][source_index] for row_index in range(len(examples))]
            if source == target:
                fixed_rules[output_bit] = ("ID", source_index)
            elif [1 - bit for bit in source] == target:
                fixed_rules[output_bit] = ("NOT", source_index)

        if len(fixed_rules) < 3 or len(fixed_rules) == 8:
            continue

        remaining_bits = [output_bit for output_bit in range(8) if output_bit not in fixed_rules]
        if len(remaining_bits) > 5:
            continue

        for function_name, function in namespace["FUNC_2_ORIG"]:
            for first_shift in range(8):
                for second_shift in range(first_shift + 1, 8):
                    matches_all = True
                    for row_index in range(len(examples)):
                        for output_bit in remaining_bits:
                            computed_value = function(
                                input_rows[row_index][(output_bit + first_shift) % 8],
                                input_rows[row_index][(output_bit + second_shift) % 8],
                            )
                            if computed_value != output_rows[row_index][output_bit]:
                                matches_all = False
                                break
                        if not matches_all:
                            break
                    if matches_all:
                        return {
                            "identity_shift": identity_shift,
                            "fixed_rules": fixed_rules,
                            "shared_function": function_name,
                            "shared_shifts": (first_shift, second_shift),
                        }
    return None


def find_best_supporting_candidate(
    output_bit: int,
    value: int,
    confidence: int,
    all_candidates: list[list[tuple[str, tuple[int, ...], int, int]]],
    permutation_map: dict[int, int],
    dominant_shift: int | None,
    dominant_pair: tuple[tuple[int, int, str], int] | None,
    namespace: dict[str, Any],
) -> tuple[str, tuple[int, ...], int, int] | None:
    operation_prior = namespace["OP_PRIOR"]
    supporting_candidates = [candidate for candidate in all_candidates[output_bit] if candidate[2] == value]
    if not supporting_candidates:
        return None

    best_candidate = None
    best_key = None
    for candidate in supporting_candidates:
        name, inputs, _predicted_value, complexity = candidate
        preference = 0.0

        if output_bit in permutation_map and name in ("ID", "NOT") and len(inputs) == 1 and inputs[0] == permutation_map[output_bit]:
            preference += 100.0
        if confidence == 2 and output_bit in permutation_map and name in ("ID", "NOT") and len(inputs) == 1 and inputs[0] == permutation_map[output_bit]:
            preference += 50.0

        if dominant_pair and len(inputs) == 2 and name == dominant_pair[0][2]:
            first_shift = (inputs[0] - output_bit) % 8
            second_shift = (inputs[1] - output_bit) % 8
            if tuple(sorted((first_shift, second_shift))) == (dominant_pair[0][0], dominant_pair[0][1]):
                preference += 40.0

        if dominant_shift is not None and name in ("ID", "NOT") and len(inputs) == 1 and (inputs[0] - output_bit) % 8 == dominant_shift:
            preference += 25.0

        if dominant_shift is not None and len(inputs) == 2:
            shifts = {(inputs[0] - output_bit) % 8, (inputs[1] - output_bit) % 8}
            if dominant_shift in shifts:
                preference += 6.0

        key = (
            preference,
            operation_prior.get(name, 0.0),
            -complexity,
            -len(inputs),
            tuple(-position for position in inputs),
            name,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_candidate = candidate

    return best_candidate


def render_whole_byte_xor(mask: str, query: str, answer: str) -> str:
    lines = [
        "<think>",
        f"The examples all use the same XOR mask: input XOR {mask} gives the output each time.",
        f"Applying that mask to the query gives {query} XOR {mask} = {answer}.",
        "</think>",
        f"\\boxed{{{answer}}}",
    ]
    return "\n".join(lines)


def render_whole_byte_rotation(direction: str, shift: int, query: str, answer: str) -> str:
    lines = [
        "<think>",
        f"Each example is the input rotated {direction} by {shift} positions to produce the output.",
        f"Rotating the query {query} {direction} by {shift} positions gives {answer}.",
        "</think>",
        f"\\boxed{{{answer}}}",
    ]
    return "\n".join(lines)


def render_whole_byte_rotxor(shift: int, mask: str, query: str, answer: str) -> str:
    lines = [
        "<think>",
        f"The examples fit a two-step rule: rotate the input left by {shift}, then XOR with the fixed mask {mask}.",
        f"Applying the same rotation and XOR to {query} gives {answer}.",
        "</think>",
        f"\\boxed{{{answer}}}",
    ]
    return "\n".join(lines)


def render_whole_byte_revxor(mask: str, query: str, answer: str) -> str:
    lines = [
        "<think>",
        f"The outputs are formed by reversing the input bits and then XORing with the fixed mask {mask}.",
        f"Reversing {query} and applying the same XOR mask gives {answer}.",
        "</think>",
        f"\\boxed{{{answer}}}",
    ]
    return "\n".join(lines)


def render_whole_byte_permutation(mapping: list[tuple[int, bool]], query_bits: list[int], answer: str) -> str:
    lines = [
        "<think>",
        "Each output bit consistently copies one input bit, with some positions optionally inverted.",
    ]
    computed_bits: list[str] = []
    for output_bit, (source_bit, inverted) in enumerate(mapping):
        source_value = query_bits[source_bit]
        output_value = 1 - source_value if inverted else source_value
        if inverted:
            lines.append(
                f"- bit {output_bit} = NOT(in[{source_bit}]) = NOT({source_value}) = {output_value}"
            )
        else:
            lines.append(f"- bit {output_bit} = in[{source_bit}] = {output_value}")
        computed_bits.append(str(output_value))
    lines.append(f"Putting the bits together gives {''.join(computed_bits)}.")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def render_whole_byte_uniform_two_input(function_name: str, first_shift: int, second_shift: int, query_bits: list[int], answer: str, namespace: dict[str, Any]) -> str:
    _unary_functions, binary_functions, _ternary_functions = get_function_maps(namespace)
    function = binary_functions[function_name]
    lines = [
        "<think>",
        (
            f"Every output bit uses the same two-input rule: bit i = {function_name}"
            f"(in[(i+{first_shift}) mod 8], in[(i+{second_shift}) mod 8])."
        ),
    ]
    rendered_bits: list[str] = []
    for output_bit in range(8):
        left_index = (output_bit + first_shift) % 8
        right_index = (output_bit + second_shift) % 8
        left_value = query_bits[left_index]
        right_value = query_bits[right_index]
        output_value = function(left_value, right_value)
        lines.append(
            f"- bit {output_bit} = {function_name}(in[{left_index}], in[{right_index}]) = "
            f"{function_name}({left_value}, {right_value}) = {output_value}"
        )
        rendered_bits.append(str(output_value))
    lines.append(f"Putting the bits together gives {''.join(rendered_bits)}.")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def render_whole_byte_mixed_rule(rule_details: dict[str, Any], query_bits: list[int], answer: str, namespace: dict[str, Any]) -> str:
    _unary_functions, binary_functions, _ternary_functions = get_function_maps(namespace)
    function_name = rule_details["shared_function"]
    first_shift, second_shift = rule_details["shared_shifts"]
    function = binary_functions[function_name]
    lines = [
        "<think>",
        (
            f"The examples show a mixed structure: several output bits are shifted copies or inversions "
            f"with shift {rule_details['identity_shift']}, and the remaining bits all use "
            f"{function_name}(in[(i+{first_shift}) mod 8], in[(i+{second_shift}) mod 8])."
        ),
    ]
    rendered_bits: list[str] = []
    for output_bit in range(8):
        if output_bit in rule_details["fixed_rules"]:
            rule_name, source_bit = rule_details["fixed_rules"][output_bit]
            source_value = query_bits[source_bit]
            output_value = source_value if rule_name == "ID" else 1 - source_value
            if rule_name == "ID":
                lines.append(f"- bit {output_bit} = in[{source_bit}] = {output_value}")
            else:
                lines.append(
                    f"- bit {output_bit} = NOT(in[{source_bit}]) = NOT({source_value}) = {output_value}"
                )
        else:
            left_index = (output_bit + first_shift) % 8
            right_index = (output_bit + second_shift) % 8
            left_value = query_bits[left_index]
            right_value = query_bits[right_index]
            output_value = function(left_value, right_value)
            lines.append(
                f"- bit {output_bit} = {function_name}(in[{left_index}], in[{right_index}]) = "
                f"{function_name}({left_value}, {right_value}) = {output_value}"
            )
        rendered_bits.append(str(output_value))
    lines.append(f"Putting the bits together gives {''.join(rendered_bits)}.")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def render_context_solution(details: dict[str, Any], query: str, answer: str, namespace: dict[str, Any]) -> str:
    query_bits = namespace["bits"](query)
    all_candidates = details["all_cands"]
    resolved_bits = details["result_phase12"]
    confidence = details["conf"]
    shift_evidence = details["shift_ev"]
    pair_evidence = details["pair_ev"]
    permutation_map = details["perm"]

    dominant_shift = shift_evidence.most_common(1)[0][0] if shift_evidence else None
    dominant_pair = pair_evidence.most_common(1)[0] if pair_evidence else None

    lines = [
        "<think>",
        "No single whole-byte rule fits every example, so solve the output bit by bit using the consistent local patterns from the examples.",
    ]

    if dominant_shift is not None and shift_evidence.most_common(1)[0][1] >= 5:
        lines.append(
            f"A strong shared pattern appears: many confirmed bits use source positions with shift {dominant_shift}."
        )

    rendered_bits: list[str] = []
    for output_bit, value in enumerate(resolved_bits):
        candidate = find_best_supporting_candidate(
            output_bit=output_bit,
            value=value,
            confidence=confidence[output_bit],
            all_candidates=all_candidates,
            permutation_map=permutation_map,
            dominant_shift=dominant_shift,
            dominant_pair=dominant_pair,
            namespace=namespace,
        )
        if candidate is None:
            lines.append(f"- bit {output_bit} = {value}")
            rendered_bits.append(str(value))
            continue
        lines.append(f"- {describe_candidate(candidate, query_bits, output_bit, namespace)}")
        rendered_bits.append(str(evaluate_candidate(candidate, query_bits, namespace)))

    lines.append(f"Putting the eight output bits together gives {''.join(rendered_bits)}.")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def render_solution(prompt: str, answer: str, method: str, details: dict[str, Any], namespace: dict[str, Any]) -> tuple[str, str]:
    examples = details["examples"]
    query = details["query"]
    query_bits = namespace["bits"](query)

    if method == "ctx":
        return render_context_solution(details, query, answer, namespace), "rendered_ctx"

    if method == "w_xor":
        xor_mask = detect_xor_rule(examples, namespace)
        if xor_mask is None:
            raise ValueError("Could not reconstruct XOR mask for w_xor row.")
        return render_whole_byte_xor(xor_mask, query, answer), "rendered_w_xor"

    if method == "w_rot":
        rotation = detect_rotation_rule(examples, namespace)
        if rotation is None:
            raise ValueError("Could not reconstruct rotation rule for w_rot row.")
        direction, shift = rotation
        return render_whole_byte_rotation(direction, shift, query, answer), "rendered_w_rot"

    if method == "w_rotxor":
        rule_details = detect_rotxor_rule(examples, namespace)
        if rule_details is None:
            raise ValueError("Could not reconstruct rotation+XOR rule for w_rotxor row.")
        shift, xor_mask = rule_details
        return render_whole_byte_rotxor(shift, xor_mask, query, answer), "rendered_w_rotxor"

    if method == "w_revxor":
        xor_mask = detect_revxor_rule(examples, namespace)
        if xor_mask is None:
            raise ValueError("Could not reconstruct reverse+XOR rule for w_revxor row.")
        return render_whole_byte_revxor(xor_mask, query, answer), "rendered_w_revxor"

    if method == "w_perm":
        mapping = detect_permutation_rule(examples, namespace)
        if mapping is None:
            raise ValueError("Could not reconstruct permutation rule for w_perm row.")
        return render_whole_byte_permutation(mapping, query_bits, answer), "rendered_w_perm"

    if method == "w_uni2":
        rule_details = detect_uniform_two_input_rule(examples, namespace)
        if rule_details is None:
            raise ValueError("Could not reconstruct uniform two-input rule for w_uni2 row.")
        function_name, first_shift, second_shift = rule_details
        return (
            render_whole_byte_uniform_two_input(function_name, first_shift, second_shift, query_bits, answer, namespace),
            "rendered_w_uni2",
        )

    if method == "w_mix":
        rule_details = detect_mixed_rule(examples, namespace, query_bits)
        if rule_details is None:
            raise ValueError("Could not reconstruct mixed rule for w_mix row.")
        return render_whole_byte_mixed_rule(rule_details, query_bits, answer, namespace), "rendered_w_mix"

    raise ValueError(f"Unsupported high-confidence method: {method}")


def is_high_confidence(method: str, prediction: str | None, answer: str) -> tuple[bool, str]:
    if prediction is None:
        return False, "solver_error"
    if prediction != answer.strip():
        return False, "solver_incorrect"
    if method == "ctx" or method.startswith("w_"):
        return True, "high_confidence"
    return False, "solver_low_confidence"


def build_output_fieldnames(input_fieldnames: list[str], keep_original_column: bool) -> list[str]:
    fieldnames = list(input_fieldnames)
    if keep_original_column and "generated_cot" in fieldnames and "generated_cot_original" not in fieldnames:
        fieldnames.append("generated_cot_original")
    for extra_field in [
        "generated_cot",
        "solver_prediction",
        "solver_method",
        "solver_confidence",
        "render_notes",
        "skip_reason",
    ]:
        if extra_field not in fieldnames:
            fieldnames.append(extra_field)
    return fieldnames


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    skipped_path = Path(args.skipped_csv)
    notebook_path = Path(args.notebook_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)

    solver_namespace = load_solver_namespace(notebook_path)

    rendered_count = 0
    skipped_count = 0
    examined_bit_rows = 0
    skip_reasons: Counter[str] = Counter()

    with input_path.open(newline="", encoding="utf-8") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{input_path} is missing a CSV header.")
        fieldnames = build_output_fieldnames(reader.fieldnames, args.keep_original_column)

        with output_path.open("w", newline="", encoding="utf-8") as output_handle, skipped_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as skipped_handle:
            rendered_writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            skipped_writer = csv.DictWriter(skipped_handle, fieldnames=fieldnames)
            rendered_writer.writeheader()
            skipped_writer.writeheader()

            for row_index, row in enumerate(reader):
                if args.max_rows is not None and row_index >= args.max_rows:
                    break

                if not is_bit_row(row):
                    continue

                examined_bit_rows += 1
                original_generated_cot = row.get("generated_cot", "")
                if args.keep_original_column and "generated_cot" in row:
                    row["generated_cot_original"] = original_generated_cot

                prediction, method, details = solver_namespace["solve_puzzle"](row["prompt"])
                row["solver_prediction"] = prediction or ""
                row["solver_method"] = method

                keep_row, confidence_reason = is_high_confidence(method, prediction, row["answer"])
                row["solver_confidence"] = "high" if keep_row else ("low" if prediction is not None else "error")

                if not keep_row:
                    row["render_notes"] = ""
                    row["skip_reason"] = confidence_reason
                    skipped_writer.writerow(row)
                    skipped_count += 1
                    skip_reasons[confidence_reason] += 1
                    continue

                try:
                    rendered_cot, render_note = render_solution(row["prompt"], row["answer"].strip(), method, details, solver_namespace)
                except Exception as exc:  # pragma: no cover - diagnostic path
                    row["render_notes"] = ""
                    row["skip_reason"] = f"render_error:{exc}"
                    skipped_writer.writerow(row)
                    skipped_count += 1
                    skip_reasons["render_error"] += 1
                    continue

                row["generated_cot"] = rendered_cot
                row["render_notes"] = render_note
                row["skip_reason"] = ""
                rendered_writer.writerow(row)
                rendered_count += 1

    summary = {
        "input_csv": str(input_path.resolve()),
        "output_csv": str(output_path.resolve()),
        "skipped_csv": str(skipped_path.resolve()),
        "examined_bit_rows": examined_bit_rows,
        "rendered_rows": rendered_count,
        "skipped_rows": skipped_count,
        "skip_reasons": dict(skip_reasons.most_common()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
