from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable


N_BITS = 8

_EXAMPLE_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
_QUERY_RE = re.compile(r"determine the output for:\s*([01]{8})", re.IGNORECASE)


BoolFunc = Callable[..., int]


@dataclass(frozen=True)
class BitPuzzle:
    prompt: str
    examples: tuple[tuple[str, str], ...]
    query: str


@dataclass(frozen=True)
class GateSpec:
    name: str
    arity: int
    apply: BoolFunc
    prior: float
    ordered: bool = False


@dataclass(frozen=True)
class BitCandidate:
    output_bit: int
    gate_name: str
    input_indices: tuple[int, ...]
    query_value: int
    arity: int
    prior: float

    @property
    def expression(self) -> str:
        if self.gate_name in {"C0", "C1"}:
            return self.gate_name
        args = ", ".join(f"in[{idx}]" for idx in self.input_indices)
        return f"{self.gate_name}({args})"

    @property
    def shifts(self) -> tuple[int, ...]:
        return tuple((idx - self.output_bit) % N_BITS for idx in self.input_indices)


@dataclass(frozen=True)
class BitSolveResult:
    prediction: str | None
    confidence: str
    method: str
    ambiguous_bits: tuple[int, ...]
    no_candidate_bits: tuple[int, ...]
    chosen_candidates: tuple[BitCandidate | None, ...]
    candidate_count_by_bit: tuple[int, ...]
    value_scores_by_bit: tuple[tuple[float, float], ...]
    notes: tuple[str, ...]


def parse_bit_manipulation_puzzle(prompt: str) -> BitPuzzle | None:
    examples = tuple(_EXAMPLE_RE.findall(prompt))
    query_match = _QUERY_RE.search(prompt)
    if not examples or query_match is None:
        return None
    return BitPuzzle(prompt=prompt, examples=examples, query=query_match.group(1))


def bits(value: str) -> tuple[int, ...]:
    return tuple(int(bit) for bit in value)


def bit_not(value: int) -> int:
    return 1 - value


UNARY_GATES = (
    GateSpec("ID", 1, lambda a: a, 40.2),
    GateSpec("NOT", 1, lambda a: 1 - a, 5.0),
)


BINARY_GATES = (
    GateSpec("AND", 2, lambda a, b: a & b, 14.7),
    GateSpec("XOR", 2, lambda a, b: a ^ b, 8.1),
    GateSpec("XNOR", 2, lambda a, b: 1 - (a ^ b), 7.3),
    GateSpec("OR", 2, lambda a, b: a | b, 7.3),
    GateSpec("NOR", 2, lambda a, b: 1 - (a | b), 5.8),
    GateSpec("NAND", 2, lambda a, b: 1 - (a & b), 3.4),
    GateSpec("INHIB", 2, lambda a, b: a & (1 - b), 2.0, ordered=True),
    GateSpec("IMPL", 2, lambda a, b: (1 - a) | b, 1.5, ordered=True),
)


TERNARY_GATES = (
    GateSpec("MAJ", 3, lambda a, b, c: 1 if a + b + c >= 2 else 0, 0.8),
    GateSpec("PAR3", 3, lambda a, b, c: a ^ b ^ c, 0.5),
    GateSpec("CHO", 3, lambda a, b, c: (a & b) | ((1 - a) & c), 0.5, ordered=True),
    GateSpec("AO", 3, lambda a, b, c: (a & b) | c, 0.25, ordered=True),
    GateSpec("OA", 3, lambda a, b, c: (a | b) & c, 0.25, ordered=True),
    GateSpec("AX", 3, lambda a, b, c: (a & b) ^ c, 0.25, ordered=True),
    GateSpec("OX", 3, lambda a, b, c: (a | b) ^ c, 0.25, ordered=True),
    GateSpec("XA", 3, lambda a, b, c: (a ^ b) & c, 0.25, ordered=True),
    GateSpec("XO", 3, lambda a, b, c: (a ^ b) | c, 0.25, ordered=True),
)


QUATERNARY_GATES = (
    GateSpec("AOA", 4, lambda a, b, c, d: (a & b) | (c & d), 0.1, ordered=True),
    GateSpec("OAO", 4, lambda a, b, c, d: (a | b) & (c | d), 0.1, ordered=True),
    GateSpec("PAR4", 4, lambda a, b, c, d: a ^ b ^ c ^ d, 0.1),
    GateSpec("XX", 4, lambda a, b, c, d: (a ^ b) ^ (c ^ d), 0.1, ordered=True),
    GateSpec("AXA", 4, lambda a, b, c, d: (a & b) ^ (c & d), 0.1, ordered=True),
)


def _index_tuples(arity: int, *, ordered: bool) -> tuple[tuple[int, ...], ...]:
    if arity == 0:
        return ((),)
    if ordered:
        return tuple(itertools.permutations(range(N_BITS), arity))
    return tuple(itertools.combinations(range(N_BITS), arity))


def _candidate_matches(
    gate: GateSpec,
    input_indices: tuple[int, ...],
    input_rows: tuple[tuple[int, ...], ...],
    target_column: tuple[int, ...],
) -> bool:
    for row, target_value in zip(input_rows, target_column):
        values = tuple(row[idx] for idx in input_indices)
        if gate.apply(*values) != target_value:
            return False
    return True


def enumerate_bit_candidates(
    puzzle: BitPuzzle,
    output_bit: int,
    *,
    max_arity: int = 4,
    include_higher_arity_when_lower_exists: bool = False,
) -> tuple[BitCandidate, ...]:
    input_rows = tuple(bits(input_value) for input_value, _ in puzzle.examples)
    output_rows = tuple(bits(output_value) for _, output_value in puzzle.examples)
    query_bits = bits(puzzle.query)
    target_column = tuple(row[output_bit] for row in output_rows)

    candidates: list[BitCandidate] = []

    if all(value == 0 for value in target_column):
        candidates.append(BitCandidate(output_bit, "C0", (), 0, 0, 0.3))
    if all(value == 1 for value in target_column):
        candidates.append(BitCandidate(output_bit, "C1", (), 1, 0, 0.3))

    gate_groups: tuple[tuple[GateSpec, ...], ...] = (
        UNARY_GATES,
        BINARY_GATES,
        TERNARY_GATES,
        QUATERNARY_GATES,
    )

    for arity, gate_group in enumerate(gate_groups, start=1):
        if arity > max_arity:
            break
        found_before_group = bool(candidates)
        for gate in gate_group:
            for input_indices in _index_tuples(arity, ordered=gate.ordered):
                if not _candidate_matches(gate, input_indices, input_rows, target_column):
                    continue
                query_values = tuple(query_bits[idx] for idx in input_indices)
                candidates.append(
                    BitCandidate(
                        output_bit=output_bit,
                        gate_name=gate.name,
                        input_indices=input_indices,
                        query_value=gate.apply(*query_values),
                        arity=arity,
                        prior=gate.prior,
                    )
                )
        if candidates and found_before_group and not include_higher_arity_when_lower_exists:
            # Once a simpler family already exists, avoid exploding into thousands
            # of accidental higher-arity explanations for the same short columns.
            break

    return tuple(candidates)


def _base_candidate_score(candidate: BitCandidate, candidates_for_bit: tuple[BitCandidate, ...]) -> float:
    score = candidate.prior
    score -= 0.4 * candidate.arity
    score -= 0.05 * len(candidate.input_indices)
    if candidate.gate_name in {"C0", "C1"} and len(candidates_for_bit) > 1:
        score -= 8.0
    if candidate.arity >= 3:
        score -= 1.5 * (candidate.arity - 2)
    return score


def _dominant_shift_signatures(
    all_candidates: tuple[tuple[BitCandidate, ...], ...],
) -> tuple[tuple[int, int] | None, tuple[str, tuple[int, ...]] | None]:
    unary_counter: Counter[int] = Counter()
    pair_counter: Counter[tuple[str, tuple[int, ...]]] = Counter()

    for candidates in all_candidates:
        for candidate in candidates:
            weight = max(1, round(candidate.prior))
            if candidate.arity == 1 and candidate.input_indices:
                unary_counter[candidate.shifts[0]] += weight
            elif candidate.arity == 2:
                shifts = candidate.shifts
                if candidate.gate_name in {"AND", "OR", "XOR", "XNOR", "NAND", "NOR"}:
                    shifts = tuple(sorted(shifts))
                pair_counter[(candidate.gate_name, shifts)] += weight

    dominant_unary = unary_counter.most_common(1)[0] if unary_counter else None
    dominant_pair = pair_counter.most_common(1)[0][0] if pair_counter else None
    return dominant_unary, dominant_pair


def _candidate_score(
    candidate: BitCandidate,
    candidates_for_bit: tuple[BitCandidate, ...],
    dominant_unary: tuple[int, int] | None,
    dominant_pair: tuple[str, tuple[int, ...]] | None,
) -> float:
    score = _base_candidate_score(candidate, candidates_for_bit)

    if dominant_unary is not None and candidate.arity == 1:
        shift, count = dominant_unary
        if candidate.shifts[0] == shift and count >= 12:
            score += min(10.0, count / 4)

    if dominant_pair is not None and candidate.arity == 2:
        gate_name, shifts = dominant_pair
        candidate_shifts = candidate.shifts
        if candidate.gate_name in {"AND", "OR", "XOR", "XNOR", "NAND", "NOR"}:
            candidate_shifts = tuple(sorted(candidate_shifts))
        if candidate.gate_name == gate_name and candidate_shifts == shifts:
            score += 6.0

    return score


def _choose_candidate_for_bit(
    candidates_for_bit: tuple[BitCandidate, ...],
    dominant_unary: tuple[int, int] | None,
    dominant_pair: tuple[str, tuple[int, ...]] | None,
) -> tuple[int, BitCandidate | None, tuple[float, float], bool]:
    if not candidates_for_bit:
        return 0, None, (0.0, 0.0), True

    value_scores = [0.0, 0.0]
    best_by_value: dict[int, tuple[float, BitCandidate]] = {}

    for candidate in candidates_for_bit:
        score = _candidate_score(candidate, candidates_for_bit, dominant_unary, dominant_pair)
        value_scores[candidate.query_value] += max(0.05, score)
        current = best_by_value.get(candidate.query_value)
        if current is None or score > current[0]:
            best_by_value[candidate.query_value] = (score, candidate)

    if value_scores[1] > value_scores[0]:
        chosen_value = 1
    elif value_scores[0] > value_scores[1]:
        chosen_value = 0
    else:
        best_candidates = sorted(
            ((score, candidate) for score, candidate in best_by_value.values()),
            key=lambda item: item[0],
            reverse=True,
        )
        chosen_value = best_candidates[0][1].query_value

    chosen_candidate = best_by_value[chosen_value][1]
    ambiguous = bool(value_scores[0] and value_scores[1])
    return chosen_value, chosen_candidate, (value_scores[0], value_scores[1]), ambiguous


def solve_bit_manipulation(
    prompt: str,
    *,
    max_arity: int = 4,
    include_higher_arity_when_lower_exists: bool = False,
) -> BitSolveResult:
    puzzle = parse_bit_manipulation_puzzle(prompt)
    if puzzle is None:
        return BitSolveResult(
            prediction=None,
            confidence="none",
            method="parse_failed",
            ambiguous_bits=(),
            no_candidate_bits=(),
            chosen_candidates=(),
            candidate_count_by_bit=(),
            value_scores_by_bit=(),
            notes=("Prompt is not a bit-manipulation puzzle.",),
        )

    all_candidates = tuple(
        enumerate_bit_candidates(
            puzzle,
            output_bit,
            max_arity=max_arity,
            include_higher_arity_when_lower_exists=include_higher_arity_when_lower_exists,
        )
        for output_bit in range(N_BITS)
    )
    dominant_unary, dominant_pair = _dominant_shift_signatures(all_candidates)

    output_bits: list[str] = []
    chosen_candidates: list[BitCandidate | None] = []
    value_scores_by_bit: list[tuple[float, float]] = []
    ambiguous_bits: list[int] = []
    no_candidate_bits: list[int] = []

    for output_bit, candidates_for_bit in enumerate(all_candidates):
        chosen_value, chosen_candidate, value_scores, ambiguous = _choose_candidate_for_bit(
            candidates_for_bit,
            dominant_unary,
            dominant_pair,
        )
        output_bits.append(str(chosen_value))
        chosen_candidates.append(chosen_candidate)
        value_scores_by_bit.append(value_scores)
        if ambiguous:
            ambiguous_bits.append(output_bit)
        if chosen_candidate is None:
            no_candidate_bits.append(output_bit)

    prediction = "".join(output_bits)

    if no_candidate_bits:
        confidence = "low"
    elif not ambiguous_bits:
        confidence = "high"
    elif len(ambiguous_bits) <= 2:
        confidence = "medium"
    else:
        confidence = "low"

    notes = []
    if dominant_unary is not None:
        notes.append(f"Dominant unary shift={dominant_unary[0]} evidence={dominant_unary[1]}.")
    if dominant_pair is not None:
        gate_name, shifts = dominant_pair
        notes.append(f"Dominant pair motif={gate_name}{shifts}.")
    if ambiguous_bits:
        notes.append(f"Ambiguous query bits: {', '.join(str(bit) for bit in ambiguous_bits)}.")
    if no_candidate_bits:
        notes.append(f"No candidate bits defaulted to 0: {', '.join(str(bit) for bit in no_candidate_bits)}.")

    return BitSolveResult(
        prediction=prediction,
        confidence=confidence,
        method="per_bit_boolean_scan",
        ambiguous_bits=tuple(ambiguous_bits),
        no_candidate_bits=tuple(no_candidate_bits),
        chosen_candidates=tuple(chosen_candidates),
        candidate_count_by_bit=tuple(len(candidates) for candidates in all_candidates),
        value_scores_by_bit=tuple(value_scores_by_bit),
        notes=tuple(notes),
    )

