from __future__ import annotations

import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable


_QUERY_RE = re.compile(r"determine the result for:\s*([^\n]+)", re.IGNORECASE)


@dataclass(frozen=True)
class SymbolExample:
    lhs: str
    rhs: str

    @property
    def operator(self) -> str:
        return self.lhs[2]


@dataclass(frozen=True)
class SymbolTransformPuzzle:
    examples: tuple[SymbolExample, ...]
    query: str

    @property
    def query_operator(self) -> str:
        return self.query[2]


@dataclass(frozen=True)
class BaseRule:
    name: str
    apply: Callable[[int, int], int | None]
    rank: int
    family: str
    is_asymmetric: bool


@dataclass(frozen=True)
class Motif:
    pairing: str
    output_mode: str

    @property
    def label(self) -> str:
        return f"{self.pairing}|{self.output_mode}"


@dataclass(frozen=True)
class OperatorFit:
    operator: str
    base_rule: BaseRule
    states: tuple[dict[str, int], ...]


@dataclass(frozen=True)
class SymbolTransformCandidate:
    prediction: str
    motif: Motif
    query_rule: BaseRule
    query_operator_seen: bool
    query_example_count: int
    total_examples_supported: int
    mapped_symbol_count: int
    chosen_rules: tuple[tuple[str, str], ...]
    digit_map: tuple[tuple[str, int], ...] | None = None
    query_completed_count: int = 0

    @property
    def label(self) -> str:
        seen = "seen" if self.query_operator_seen else "absent"
        return f"{self.motif.label}|{self.query_rule.name}|{seen}"


@dataclass(frozen=True)
class SymbolTransformSolveResult:
    prediction: str | None
    confidence: str
    method: str
    candidate_count: int
    prediction_variants: tuple[str, ...]
    chosen_candidate: SymbolTransformCandidate | None
    notes: tuple[str, ...]


PAIRINGS = ("AB_CD", "AB_DC", "BA_CD", "BA_DC")
OP_TOKEN = "__OP__"
CURRENT_OUTPUT_MODES = ("raw", "rev", "abs")
FORMAT_OUTPUT_MODES = (
    "raw",
    "rev",
    "abs",
    "last",
    "last_rev",
    "op_prefix_if_neg",
    "op_suffix_rev_if_neg",
)
OUTPUT_MODE_BANKS = {
    "current": CURRENT_OUTPUT_MODES,
    "formats": FORMAT_OUTPUT_MODES,
}

CURRENT_MOTIF_SCAN_ORDER = (
    Motif("BA_DC", "rev"),
    Motif("AB_CD", "raw"),
    Motif("AB_CD", "rev"),
    Motif("BA_DC", "raw"),
    Motif("AB_DC", "raw"),
    Motif("AB_DC", "rev"),
    Motif("BA_CD", "raw"),
    Motif("BA_CD", "rev"),
    Motif("AB_CD", "abs"),
    Motif("BA_DC", "abs"),
    Motif("AB_DC", "abs"),
    Motif("BA_CD", "abs"),
)
EXTENDED_MOTIF_SCAN_ORDER = (
    *CURRENT_MOTIF_SCAN_ORDER,
    *(Motif(pairing, mode) for mode in ("last", "last_rev") for pairing in PAIRINGS),
    *(
        Motif(pairing, mode)
        for mode in ("op_prefix_if_neg", "op_suffix_rev_if_neg")
        for pairing in PAIRINGS
    ),
)
MOTIF_SCAN_ORDER = CURRENT_MOTIF_SCAN_ORDER
MOTIF_RANK = {motif: rank for rank, motif in enumerate(MOTIF_SCAN_ORDER)}
EXTENDED_MOTIF_RANK = {motif: rank for rank, motif in enumerate(EXTENDED_MOTIF_SCAN_ORDER)}


def _safe_mod(x: int, y: int) -> int | None:
    if y == 0:
        return None
    return x % y


def _safe_floordiv(x: int, y: int) -> int | None:
    if y == 0:
        return None
    return x // y


BASE_RULES = (
    BaseRule("x * y", lambda x, y: x * y, 0, "mul", False),
    BaseRule("x + y", lambda x, y: x + y, 1, "add", False),
    BaseRule("x + y + 1", lambda x, y: x + y + 1, 2, "add", False),
    BaseRule("x + y - 1", lambda x, y: x + y - 1, 3, "add", False),
    BaseRule("x * y + 1", lambda x, y: x * y + 1, 4, "mul", False),
    BaseRule("x * y - 1", lambda x, y: x * y - 1, 5, "mul", False),
    BaseRule("x - y", lambda x, y: x - y, 6, "sub", True),
    BaseRule("y - x", lambda x, y: y - x, 7, "sub", True),
    BaseRule("|x - y|", lambda x, y: abs(x - y), 8, "sub", False),
    BaseRule("concat(x, y)", lambda x, y: int(f"{abs(x)}{abs(y)}"), 9, "concat", True),
    BaseRule("concat(y, x)", lambda x, y: int(f"{abs(y)}{abs(x)}"), 10, "concat", True),
    BaseRule("max - min", lambda x, y: max(x, y) - min(x, y), 11, "sub", False),
    BaseRule("x % y", _safe_mod, 12, "mod", True),
    BaseRule("y % x", lambda x, y: _safe_mod(y, x), 13, "mod", True),
    BaseRule("max % min", lambda x, y: _safe_mod(max(x, y), min(x, y)), 14, "mod", False),
    BaseRule("min % max", lambda x, y: _safe_mod(min(x, y), max(x, y)), 15, "mod", False),
    BaseRule("x // y", _safe_floordiv, 16, "div", True),
    BaseRule("y // x", lambda x, y: _safe_floordiv(y, x), 17, "div", True),
    BaseRule("x + y^2", lambda x, y: x + y * y, 18, "quadratic", True),
    BaseRule("y + x^2", lambda x, y: y + x * x, 19, "quadratic", True),
    BaseRule("x^2 - y", lambda x, y: x * x - y, 20, "quadratic", True),
    BaseRule("y^2 - x", lambda x, y: y * y - x, 21, "quadratic", True),
)
BASE_RULE_BY_NAME = {rule.name: rule for rule in BASE_RULES}


CORE_RULE_NAMES = frozenset(
    {
        "x * y",
        "x + y",
        "x + y + 1",
        "x + y - 1",
        "x * y + 1",
        "x * y - 1",
        "x - y",
        "y - x",
        "|x - y|",
        "concat(x, y)",
        "concat(y, x)",
    }
)


DIRECT_TEMPLATES = {
    "0134": (0, 1, 3, 4),
    "3401": (3, 4, 0, 1),
}
DIRECT_TEMPLATE_PRIORITY = ("0134", "3401")
UNIT_CONSTRAINT_MUL_MOTIFS = (
    # Match the strongest numeric-equation motif first.
    Motif("BA_DC", "rev"),
    Motif("AB_CD", "raw"),
)
UNIT_CONSTRAINT_MUL_RULE_NAMES = (
    "x * y",
    "x * y + 1",
    "x * y - 1",
)
FAMILY_RESCUE_MOTIFS = (
    Motif("BA_DC", "rev"),
    Motif("AB_CD", "raw"),
)
GUARDED_MIN1_GLOBAL_MOTIFS = frozenset(
    {
        Motif("BA_DC", "rev"),
        Motif("AB_CD", "raw"),
    }
)
GUARDED_MIN1_MIN_MAPPED_SYMBOLS = 10
GUARDED_MIN1_MIN_SUPPORTED_EXAMPLES = 4
ADD_RULE_NAMES = ("x + y", "x + y + 1", "x + y - 1")
SUB_RULE_NAMES = ("x - y", "y - x", "|x - y|")
MUL_RULE_NAMES = ("x * y", "x * y + 1", "x * y - 1")
LENGTH1_SUBTRACTION_RESCUE_MOTIFS = (
    Motif("AB_CD", "raw"),
    Motif("BA_DC", "rev"),
)
LENGTH1_RESCUE_LEN3_RULE_NAMES = (
    "x + y",
    "x * y",
    "x + y + 1",
    "x + y - 1",
    "x * y + 1",
    "x * y - 1",
)
SIGNED_MARKER_RESCUE_MOTIFS = (
    Motif("AB_CD", "op_prefix_if_neg"),
)
SIGNED_MARKER_RESCUE_MAX_STATES = 80
LAST_DIGIT_RESCUE_MOTIFS = (
    Motif("AB_CD", "last"),
)
LAST_DIGIT_RESCUE_MAX_STATES = 80
BA_DC_REV_RESCUE_MOTIF = Motif("BA_DC", "rev")
BA_DC_REV_RESCUE_MAX_STATES = 80

def parse_symbol_transform_puzzle(prompt: str) -> SymbolTransformPuzzle | None:
    examples: list[SymbolExample] = []
    for line in prompt.splitlines():
        text = line.strip()
        if " = " not in text:
            continue
        lhs, rhs = text.split(" = ", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if len(lhs) == 5:
            examples.append(SymbolExample(lhs=lhs, rhs=rhs))

    query_match = _QUERY_RE.search(prompt)
    if query_match is None:
        return None
    query = query_match.group(1).strip()
    if len(query) != 5 or not examples:
        return None
    return SymbolTransformPuzzle(examples=tuple(examples), query=query)


def same_operator_examples(puzzle: SymbolTransformPuzzle) -> tuple[SymbolExample, ...]:
    op = puzzle.query_operator
    return tuple(example for example in puzzle.examples if example.operator == op)


def _group_examples_by_operator(
    examples: Iterable[SymbolExample],
) -> dict[str, tuple[SymbolExample, ...]]:
    groups: dict[str, list[SymbolExample]] = defaultdict(list)
    for example in examples:
        groups[example.operator].append(example)
    return {operator: tuple(items) for operator, items in groups.items()}


def _pair_to_xy(pairing: str, lhs: str, sym_to_digit: dict[str, int]) -> tuple[int, int] | None:
    needed = (lhs[0], lhs[1], lhs[3], lhs[4])
    if any(symbol not in sym_to_digit for symbol in needed):
        return None
    a = sym_to_digit[lhs[0]]
    b = sym_to_digit[lhs[1]]
    c = sym_to_digit[lhs[3]]
    d = sym_to_digit[lhs[4]]
    if pairing == "AB_CD":
        return 10 * a + b, 10 * c + d
    if pairing == "AB_DC":
        return 10 * a + b, 10 * d + c
    if pairing == "BA_CD":
        return 10 * b + a, 10 * c + d
    if pairing == "BA_DC":
        return 10 * b + a, 10 * d + c
    return None


def _digit_tokens(text: str) -> tuple[int | str, ...] | None:
    if not text.isdigit():
        return None
    return tuple(int(char) for char in text)


def _render_output_tokens(
    value: int | None,
    output_mode: str,
    target_len: int,
) -> tuple[int | str, ...] | None:
    if value is None:
        return None

    if output_mode == "abs":
        text = str(abs(value))
        return _digit_tokens(text) if len(text) == target_len else None

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
    elif output_mode == "last":
        rendered = str(value % (10**target_len)).zfill(target_len)
    elif output_mode == "last_rev":
        rendered = str(value % (10**target_len)).zfill(target_len)[::-1]
    else:
        return None

    tokens = _digit_tokens(rendered)
    return tokens if tokens is not None and len(tokens) == target_len else None


def _render_query_token_variants(
    value: int | None,
    output_mode: str,
    candidate_lengths: tuple[int, ...],
) -> tuple[tuple[int | str, ...], ...]:
    if value is None:
        return ()

    if output_mode in {"raw", "rev", "abs", "op_prefix_if_neg", "op_suffix_rev_if_neg"}:
        target_len = len(str(abs(value))) + (
            1 if value < 0 and output_mode in {"op_prefix_if_neg", "op_suffix_rev_if_neg"} else 0
        )
        tokens = _render_output_tokens(value, output_mode, target_len)
        return (tokens,) if tokens is not None else ()

    if value < 0 or output_mode not in {"last", "last_rev"}:
        return ()

    lengths = {
        length
        for length in candidate_lengths
        if 1 <= length <= 4
    }
    lengths.add(len(str(value)))
    variants: set[tuple[int | str, ...]] = set()
    for target_len in sorted(lengths):
        tokens = _render_output_tokens(value, output_mode, target_len)
        if tokens is not None:
            variants.add(tokens)
    return tuple(sorted(variants, key=lambda item: (len(item), item)))


def _encode_tokens(
    tokens: tuple[int | str, ...],
    sym_to_digit: dict[str, int],
    *,
    operator: str,
) -> str | None:
    if OP_TOKEN in tokens and operator in sym_to_digit:
        return None

    digit_to_sym = {digit: symbol for symbol, digit in sym_to_digit.items()}
    out: list[str] = []
    for token in tokens:
        if token == OP_TOKEN:
            out.append(operator)
            continue
        symbol = digit_to_sym.get(int(token))
        if symbol is None:
            return None
        out.append(symbol)
    return "".join(out)


def _render_output(value: int | None, output_mode: str) -> str | None:
    if value is None:
        return None
    if output_mode in {"last", "last_rev", "op_prefix_if_neg", "op_suffix_rev_if_neg"}:
        return None
    tokens = _render_output_tokens(value, output_mode, len(str(abs(value))))
    if tokens is None or any(token == OP_TOKEN for token in tokens):
        return None
    text = "".join(str(int(token)) for token in tokens)
    if output_mode in {"raw", "rev", "abs"}:
        return text
    return None


def _encode_digits(digit_text: str, sym_to_digit: dict[str, int]) -> str | None:
    digit_to_sym = {digit: symbol for symbol, digit in sym_to_digit.items()}
    out: list[str] = []
    for char in digit_text:
        if not char.isdigit():
            return None
        symbol = digit_to_sym.get(int(char))
        if symbol is None:
            return None
        out.append(symbol)
    return "".join(out)


def _merge_maps(left: dict[str, int], right: dict[str, int]) -> dict[str, int] | None:
    merged = dict(left)
    used = set(merged.values())
    for symbol, digit in right.items():
        existing = merged.get(symbol)
        if existing is not None:
            if existing != digit:
                return None
            continue
        if digit in used:
            return None
        merged[symbol] = digit
        used.add(digit)
    return merged


def _merge_symbol_digit_pairs(
    state: dict[str, int],
    pairs: Iterable[tuple[str, int]],
) -> dict[str, int] | None:
    local = dict(state)
    used = set(local.values())
    for symbol, digit in pairs:
        existing = local.get(symbol)
        if existing is not None:
            if existing != digit:
                return None
            continue
        if digit in used:
            return None
        local[symbol] = digit
        used.add(digit)
    return local


def _two_digit_tokens(value: int) -> tuple[int, int]:
    return value // 10, value % 10


def _lhs_digit_pairs_for_xy(
    pairing: str,
    lhs: str,
    x: int,
    y: int,
) -> tuple[tuple[str, int], ...] | None:
    x_tens, x_ones = _two_digit_tokens(x)
    y_tens, y_ones = _two_digit_tokens(y)
    if pairing == "AB_CD":
        return ((lhs[0], x_tens), (lhs[1], x_ones), (lhs[3], y_tens), (lhs[4], y_ones))
    if pairing == "AB_DC":
        return ((lhs[0], x_tens), (lhs[1], x_ones), (lhs[4], y_tens), (lhs[3], y_ones))
    if pairing == "BA_CD":
        return ((lhs[1], x_tens), (lhs[0], x_ones), (lhs[3], y_tens), (lhs[4], y_ones))
    if pairing == "BA_DC":
        return ((lhs[1], x_tens), (lhs[0], x_ones), (lhs[4], y_tens), (lhs[3], y_ones))
    return None


def _extend_state_for_mul_unit_constraint(
    example: SymbolExample,
    motif: Motif,
    base_rule: BaseRule,
    state: dict[str, int],
    *,
    max_results: int,
) -> list[dict[str, int]]:
    """Extend one map using a multiplication motif.

    The implementation checks the product's unit digit first when possible.
    This mirrors the hand trace: use modulo-10 constraints before accepting a
    full digit-shape match.
    """

    lhs_symbols = (example.lhs[0], example.lhs[1], example.lhs[3], example.lhs[4])
    unknown_lhs_symbols = sorted({symbol for symbol in lhs_symbols if symbol not in state})
    used_digits = set(state.values())
    available_digits = [digit for digit in range(10) if digit not in used_digits]
    unit_symbol = example.rhs[-1] if motif.output_mode == "raw" else example.rhs[0]
    known_unit = state.get(unit_symbol)
    out: dict[tuple[tuple[str, int], ...], dict[str, int]] = {}

    for digits in itertools.permutations(available_digits, len(unknown_lhs_symbols)):
        local = dict(state)
        for symbol, digit in zip(unknown_lhs_symbols, digits):
            local[symbol] = digit

        xy = _pair_to_xy(motif.pairing, example.lhs, local)
        if xy is None:
            continue

        value = base_rule.apply(*xy)
        if value is None:
            continue
        if known_unit is not None and value % 10 != known_unit:
            continue

        tokens = _render_output_tokens(value, motif.output_mode, len(example.rhs))
        if tokens is None:
            continue

        # Unit-output pair is placed before the rest of the RHS assignments so
        # collisions fail as early as possible.
        if motif.output_mode == "raw":
            unit_pair = ((example.rhs[-1], int(tokens[-1])),)
            other_output_pairs = tuple(
                (symbol, int(token))
                for symbol, token in zip(example.rhs[:-1], tokens[:-1])
            )
        else:
            unit_pair = ((example.rhs[0], int(tokens[0])),)
            other_output_pairs = tuple(
                (symbol, int(token))
                for symbol, token in zip(example.rhs[1:], tokens[1:])
            )

        merged = _merge_symbol_digit_pairs(
            local,
            (*unit_pair, *other_output_pairs),
        )
        if merged is None:
            continue
        out[tuple(sorted(merged.items()))] = merged
        if len(out) > max_results:
            return []

    return list(out.values())


def _search_same_operator_mul_unit_constraint(
    puzzle: SymbolTransformPuzzle,
    *,
    max_query_unknowns: int,
    max_states: int = 5000,
) -> tuple[SymbolTransformCandidate, ...]:
    same = same_operator_examples(puzzle)
    if len(same) < 2:
        return ()
    if not any(len(example.rhs) == 4 for example in same):
        return ()
    same = tuple(
        sorted(
            same,
            key=lambda example: (
                -len(example.rhs),
                len(set(example.lhs + example.rhs)),
                example.lhs,
                example.rhs,
            ),
        )
    )

    candidates: list[SymbolTransformCandidate] = []
    query_lengths = tuple(sorted({len(example.rhs) for example in same}))

    for motif in UNIT_CONSTRAINT_MUL_MOTIFS:
        for rule_name in UNIT_CONSTRAINT_MUL_RULE_NAMES:
            query_rule = BASE_RULE_BY_NAME[rule_name]
            states: tuple[dict[str, int], ...] = ({},)
            for example in same:
                next_states: list[dict[str, int]] = []
                for state in states:
                    next_states.extend(
                        _extend_state_for_mul_unit_constraint(
                            example,
                            motif,
                            query_rule,
                            state,
                            max_results=max_states,
                        )
                    )
                if len(next_states) > max_states * 2:
                    next_states = []
                    break
                if not next_states:
                    states = ()
                    break
                deduped = {
                    tuple(sorted(state.items())): state
                    for state in next_states
                }
                sorted_states = sorted(
                    deduped.values(),
                    key=lambda item: (-len(item), tuple(sorted(item.items()))),
                )
                if len(sorted_states) > max_states:
                    states = ()
                    break
                states = tuple(sorted_states)
            if not states:
                continue

            for sym_to_digit in states:
                for query_map in _complete_query_operand_maps(
                    puzzle.query,
                    sym_to_digit,
                    max_unknowns=max_query_unknowns,
                ):
                    xy = _pair_to_xy(motif.pairing, puzzle.query, query_map)
                    if xy is None:
                        continue
                    token_variants = _render_query_token_variants(
                        query_rule.apply(*xy),
                        motif.output_mode,
                        query_lengths,
                    )
                    for tokens in token_variants:
                        prediction = _encode_tokens(tokens, query_map, operator=puzzle.query_operator)
                        if prediction is None:
                            continue
                        candidates.append(
                            SymbolTransformCandidate(
                                prediction=prediction,
                                motif=motif,
                                query_rule=query_rule,
                                query_operator_seen=True,
                                query_example_count=len(same),
                                total_examples_supported=len(same),
                                mapped_symbol_count=len(query_map),
                                chosen_rules=((puzzle.query_operator, query_rule.name),),
                                digit_map=tuple(sorted(query_map.items())),
                                query_completed_count=len(query_map) - len(sym_to_digit),
                            )
                        )

    return _dedupe_candidates(candidates)


def _complete_query_operand_maps(
    query: str,
    sym_to_digit: dict[str, int],
    *,
    max_unknowns: int,
) -> tuple[dict[str, int], ...]:
    operand_symbols = (query[0], query[1], query[3], query[4])
    unknown_symbols = sorted({symbol for symbol in operand_symbols if symbol not in sym_to_digit})
    if not unknown_symbols:
        return (sym_to_digit,)
    if len(unknown_symbols) > max_unknowns:
        return ()

    used_digits = set(sym_to_digit.values())
    available_digits = [digit for digit in range(10) if digit not in used_digits]
    out: list[dict[str, int]] = []
    for digits in itertools.permutations(available_digits, len(unknown_symbols)):
        local = dict(sym_to_digit)
        for symbol, digit in zip(unknown_symbols, digits):
            local[symbol] = digit
        out.append(local)
    return tuple(out)


def _extend_state_for_example(
    example: SymbolExample,
    motif: Motif,
    base_rule: BaseRule,
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
        tokens = _render_output_tokens(base_rule.apply(*xy), motif.output_mode, len(example.rhs))
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


def _fit_operator(
    operator: str,
    examples: tuple[SymbolExample, ...],
    motif: Motif,
    base_rules: tuple[BaseRule, ...],
    *,
    max_states_per_rule: int,
    priority_symbols: frozenset[str] = frozenset(),
) -> tuple[OperatorFit, ...]:
    fits: list[OperatorFit] = []
    for base_rule in base_rules:
        states: list[dict[str, int]] = [{}]
        for example in examples:
            next_states: list[dict[str, int]] = []
            for state in states:
                next_states.extend(_extend_state_for_example(example, motif, base_rule, state))
            if not next_states:
                states = []
                break
            next_states.sort(
                key=lambda item: (
                    -len(priority_symbols.intersection(item)),
                    -len(item),
                    tuple(sorted(item.items())),
                )
            )
            states = next_states[:max_states_per_rule]
        if states:
            fits.append(OperatorFit(operator=operator, base_rule=base_rule, states=tuple(states)))
    return tuple(fits)


def _try_direct_templates(puzzle: SymbolTransformPuzzle) -> SymbolTransformSolveResult | None:
    same = same_operator_examples(puzzle)
    if not same:
        return None

    for name in DIRECT_TEMPLATE_PRIORITY:
        template = DIRECT_TEMPLATES[name]
        if all("".join(example.lhs[index] for index in template) == example.rhs for example in same):
            prediction = "".join(puzzle.query[index] for index in template)
            return SymbolTransformSolveResult(
                prediction=prediction,
                confidence="high" if len(same) >= 2 else "medium",
                method=f"direct_template_{name}",
                candidate_count=1,
                prediction_variants=(prediction,),
                chosen_candidate=None,
                notes=(
                    f"Direct template {name} is the first priority template that fits all same-operator examples.",
                ),
            )
    return None


def _candidate_sort_key(
    candidate: SymbolTransformCandidate,
) -> tuple[int, int, int, int, int, int, int, str]:
    motif_rank = EXTENDED_MOTIF_RANK.get(candidate.motif, len(EXTENDED_MOTIF_RANK))
    seen_penalty = 0 if candidate.query_operator_seen else 1
    asymmetric_bonus = 0 if candidate.query_rule.is_asymmetric else 1
    return (
        seen_penalty,
        -candidate.query_example_count,
        motif_rank,
        candidate.query_rule.rank,
        asymmetric_bonus,
        candidate.query_completed_count,
        -candidate.mapped_symbol_count,
        candidate.prediction,
    )


def _dedupe_candidates(
    candidates: Iterable[SymbolTransformCandidate],
) -> tuple[SymbolTransformCandidate, ...]:
    best_by_key: dict[tuple[str, str, str, str], SymbolTransformCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.prediction,
            candidate.motif.pairing,
            candidate.motif.output_mode,
            candidate.query_rule.name,
        )
        existing = best_by_key.get(key)
        if existing is None or _candidate_sort_key(candidate) < _candidate_sort_key(existing):
            best_by_key[key] = candidate
    return tuple(sorted(best_by_key.values(), key=_candidate_sort_key))


def _dedupe_partials(
    partials: Iterable[tuple[dict[str, int], dict[str, BaseRule]]],
) -> list[tuple[dict[str, int], dict[str, BaseRule]]]:
    best: dict[
        tuple[tuple[tuple[str, int], ...], tuple[tuple[str, str], ...]],
        tuple[dict[str, int], dict[str, BaseRule]],
    ] = {}
    for sym_to_digit, chosen_rules in partials:
        key = (
            tuple(sorted(sym_to_digit.items())),
            tuple(sorted((operator, rule.name) for operator, rule in chosen_rules.items())),
        )
        best[key] = (sym_to_digit, chosen_rules)
    return list(best.values())


def _family_rule_names_for_examples(
    motif: Motif,
    examples: tuple[SymbolExample, ...],
) -> tuple[str, ...]:
    max_len = max(len(example.rhs) for example in examples)
    if max_len >= 4:
        names = MUL_RULE_NAMES
    elif max_len == 1:
        names = SUB_RULE_NAMES
    elif motif == Motif("BA_DC", "rev"):
        names = (
            "x + y",
            "x - y",
            "x + y - 1",
            "x + y + 1",
            "|x - y|",
            "y - x",
        )
    elif motif == Motif("AB_CD", "raw"):
        names = (
            "x - y",
            "|x - y|",
            "x + y",
            "x + y + 1",
            "x + y - 1",
            "y - x",
        )
    else:
        names = (*ADD_RULE_NAMES, *SUB_RULE_NAMES)

    # Length-3 outputs can come from addition with carry or multiplication.
    if max_len == 3:
        names = (*ADD_RULE_NAMES, *MUL_RULE_NAMES, *SUB_RULE_NAMES)

    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return tuple(out)


def _length1_rescue_rule_names_for_examples(
    examples: tuple[SymbolExample, ...],
) -> tuple[str, ...]:
    max_len = max(len(example.rhs) for example in examples)
    if max_len >= 4:
        return MUL_RULE_NAMES
    if max_len == 3:
        return LENGTH1_RESCUE_LEN3_RULE_NAMES
    if max_len == 1:
        return SUB_RULE_NAMES
    return (*ADD_RULE_NAMES, *SUB_RULE_NAMES)


def _query_operand_unknown_count(query: str, sym_to_digit: dict[str, int]) -> int:
    operand_symbols = (query[0], query[1], query[3], query[4])
    return len({symbol for symbol in operand_symbols if symbol not in sym_to_digit})


def _query_predictions_from_state(
    puzzle: SymbolTransformPuzzle,
    motif: Motif,
    query_rule: BaseRule,
    sym_to_digit: dict[str, int],
    *,
    query_lengths: tuple[int, ...],
) -> tuple[str, ...]:
    if _query_operand_unknown_count(puzzle.query, sym_to_digit) != 0:
        return ()

    xy = _pair_to_xy(motif.pairing, puzzle.query, sym_to_digit)
    if xy is None:
        return ()

    predictions: set[str] = set()
    for tokens in _render_query_token_variants(
        query_rule.apply(*xy),
        motif.output_mode,
        query_lengths,
    ):
        prediction = _encode_tokens(
            tokens,
            sym_to_digit,
            operator=puzzle.query_operator,
        )
        if prediction is not None:
            predictions.add(prediction)
    return tuple(sorted(predictions))


def _query_missing_output_digits_from_state(
    puzzle: SymbolTransformPuzzle,
    motif: Motif,
    query_rule: BaseRule,
    sym_to_digit: dict[str, int],
    *,
    query_lengths: tuple[int, ...],
) -> frozenset[int]:
    """Return output digits needed to encode the query but not yet mapped.

    The hand solution sometimes has the query operands already determined, but
    still needs another row to learn which symbol represents a result digit.
    This helper detects that inverse-map gap so map completion can continue.
    """

    if _query_operand_unknown_count(puzzle.query, sym_to_digit) != 0:
        return frozenset()

    xy = _pair_to_xy(motif.pairing, puzzle.query, sym_to_digit)
    if xy is None:
        return frozenset()

    digit_to_sym = {digit: symbol for symbol, digit in sym_to_digit.items()}
    missing: set[int] = set()
    for tokens in _render_query_token_variants(
        query_rule.apply(*xy),
        motif.output_mode,
        query_lengths,
    ):
        for token in tokens:
            if token == OP_TOKEN:
                continue
            digit = int(token)
            if digit not in digit_to_sym:
                missing.add(digit)
    return frozenset(missing)


def _search_same_operator_map_completion_candidates(
    puzzle: SymbolTransformPuzzle,
    *,
    rule_bank: str,
    include_abs_output: bool,
    output_mode_bank: str,
    max_states_per_rule: int,
    max_completion_states: int,
) -> tuple[SymbolTransformCandidate, ...]:
    """Use other operators only to complete the map after same-operator lock.

    This differs from the row-global solver: other operator groups are optional.
    A group that does not fit the current motif is skipped rather than causing
    the whole candidate to fail. This matches the hand method: lock the query
    operator first, then use whichever remaining equations determine missing
    symbols needed by the query.
    """

    same = same_operator_examples(puzzle)
    if not same:
        return ()

    groups = _group_examples_by_operator(puzzle.examples)
    other_operators = tuple(operator for operator in groups if operator != puzzle.query_operator)
    if not other_operators:
        return ()

    other_text_by_operator = {
        operator: "".join(example.lhs + example.rhs for example in groups[operator])
        for operator in other_operators
    }
    other_symbols = set("".join(other_text_by_operator.values()))
    query_operand_symbols = (puzzle.query[0], puzzle.query[1], puzzle.query[3], puzzle.query[4])
    allowed_rule_names = {rule.name for rule in _rule_bank(rule_bank)}
    query_lengths = tuple(sorted({len(example.rhs) for example in same}))
    candidates: list[SymbolTransformCandidate] = []

    allowed_motifs = {
        motif
        for motif in _motif_bank(include_abs_output, output_mode_bank)
        if motif in FAMILY_RESCUE_MOTIFS
    }
    for motif in FAMILY_RESCUE_MOTIFS:
        if motif not in allowed_motifs:
            continue
        query_rule_names = tuple(
            name
            for name in _family_rule_names_for_examples(motif, same)
            if name in allowed_rule_names
        )
        if not query_rule_names:
            continue
        base_rules = tuple(BASE_RULE_BY_NAME[name] for name in query_rule_names)
        query_fits = _fit_operator(
            puzzle.query_operator,
            same,
            motif,
            base_rules,
            max_states_per_rule=max_states_per_rule,
            priority_symbols=frozenset(query_operand_symbols),
        )
        if not query_fits:
            continue

        for fit in query_fits:
            seed_partials: list[tuple[dict[str, int], dict[str, BaseRule]]] = []
            for state in fit.states[:max_completion_states]:
                missing_query_symbols = {
                    symbol for symbol in query_operand_symbols if symbol not in state
                }
                missing_output_digits = _query_missing_output_digits_from_state(
                    puzzle,
                    motif,
                    fit.base_rule,
                    state,
                    query_lengths=query_lengths,
                )
                if not missing_query_symbols and not missing_output_digits:
                    continue
                if (
                    missing_output_digits
                    and not missing_query_symbols
                    and fit.base_rule.family != "mul"
                ):
                    continue
                if len(missing_query_symbols) > 2:
                    continue
                if not missing_query_symbols.issubset(other_symbols):
                    continue
                if missing_output_digits:
                    unmapped_other_symbols = {
                        symbol for symbol in other_symbols if symbol not in state
                    }
                    if len(unmapped_other_symbols) < len(missing_output_digits):
                        continue
                seed_partials.append((state, {puzzle.query_operator: fit.base_rule}))
            if not seed_partials:
                continue
            partials: tuple[tuple[dict[str, int], dict[str, BaseRule]], ...] = tuple(seed_partials)

            group_order = tuple(
                sorted(
                    other_operators,
                    key=lambda operator: (
                        0
                        if any(
                            symbol in other_text_by_operator[operator]
                            for symbol in query_operand_symbols
                        )
                        else 1,
                        -len(groups[operator]),
                        -max(len(example.rhs) for example in groups[operator]),
                        operator,
                    ),
                )
            )

            for operator in group_order:
                rule_names = _family_rule_names_for_examples(motif, groups[operator])
                rules = tuple(BASE_RULE_BY_NAME[name] for name in rule_names)
                extended = _fit_family_group_from_partials(
                    partials,
                    operator,
                    groups[operator],
                    motif,
                    rules,
                    max_states=max_completion_states,
                )
                if not extended:
                    continue

                # Keep the option of skipping this group. Some rows have useful
                # map-completion examples mixed with incompatible distractors.
                partials = tuple(
                    sorted(
                        _dedupe_partials((*partials, *extended)),
                        key=lambda item: (
                            _query_operand_unknown_count(puzzle.query, item[0]),
                            -len(item[0]),
                            sum(rule.rank for rule in item[1].values()),
                            tuple(sorted((op, rule.name) for op, rule in item[1].items())),
                            tuple(sorted(item[0].items())),
                        ),
                    )[:max_completion_states]
                )

                for sym_to_digit, chosen_rules in partials:
                    predictions = _query_predictions_from_state(
                        puzzle,
                        motif,
                        fit.base_rule,
                        sym_to_digit,
                        query_lengths=query_lengths,
                    )
                    for prediction in predictions:
                        candidates.append(
                            SymbolTransformCandidate(
                                prediction=prediction,
                                motif=motif,
                                query_rule=fit.base_rule,
                                query_operator_seen=True,
                                query_example_count=len(same),
                                total_examples_supported=sum(
                                    len(groups[op]) for op in chosen_rules
                                ),
                                mapped_symbol_count=len(sym_to_digit),
                                chosen_rules=tuple(
                                    sorted(
                                        (op, rule.name)
                                        for op, rule in chosen_rules.items()
                                    )
                                ),
                                digit_map=tuple(sorted(sym_to_digit.items())),
                                query_completed_count=0,
                            )
                        )

    return _dedupe_candidates(candidates)


def _fit_family_group_from_partials(
    partials: tuple[tuple[dict[str, int], dict[str, BaseRule]], ...],
    operator: str,
    examples: tuple[SymbolExample, ...],
    motif: Motif,
    base_rules: tuple[BaseRule, ...],
    *,
    max_states: int,
) -> tuple[tuple[dict[str, int], dict[str, BaseRule]], ...]:
    next_partials: list[tuple[dict[str, int], dict[str, BaseRule]]] = []
    for current_map, chosen_rules in partials:
        for base_rule in base_rules:
            states: tuple[dict[str, int], ...] = (current_map,)
            for example in examples:
                next_states: list[dict[str, int]] = []
                for state in states:
                    next_states.extend(_extend_state_for_example(example, motif, base_rule, state))
                if not next_states:
                    states = ()
                    break
                deduped = {
                    tuple(sorted(state.items())): state
                    for state in next_states
                }
                states = tuple(
                    sorted(
                        deduped.values(),
                        key=lambda item: (-len(item), tuple(sorted(item.items()))),
                    )[:max_states]
                )
            if not states:
                continue
            for state in states:
                next_rules = dict(chosen_rules)
                next_rules[operator] = base_rule
                next_partials.append((state, next_rules))

    deduped_partials = _dedupe_partials(next_partials)
    deduped_partials.sort(
        key=lambda item: (
            -len(item[0]),
            sum(rule.rank for rule in item[1].values()),
            tuple(sorted((operator, rule.name) for operator, rule in item[1].items())),
            tuple(sorted(item[0].items())),
        )
    )
    return tuple(deduped_partials[:max_states])


def _search_family_rescue_candidates(
    puzzle: SymbolTransformPuzzle,
    *,
    max_states: int = 80,
    max_query_unknowns: int = 2,
    max_prediction_variants: int = 12,
) -> tuple[tuple[tuple[int, str], SymbolTransformCandidate], ...]:
    same = same_operator_examples(puzzle)
    if len(same) < 2:
        return ()
    if not any(len(example.rhs) <= 2 for example in same):
        return ()

    groups = _group_examples_by_operator(puzzle.examples)
    query_operator = puzzle.query_operator
    if len(groups) < 2:
        return ()
    scored: list[tuple[tuple[int, str], SymbolTransformCandidate]] = []

    for motif in FAMILY_RESCUE_MOTIFS:
        query_rule_names = _family_rule_names_for_examples(motif, same)
        query_rules = tuple(BASE_RULE_BY_NAME[name] for name in query_rule_names)

        for query_rule in query_rules:
            partials = _fit_family_group_from_partials(
                (({}, {}),),
                query_operator,
                same,
                motif,
                (query_rule,),
                max_states=max_states,
            )
            if not partials:
                continue

            other_operators = tuple(
                sorted(
                    (operator for operator in groups if operator != query_operator),
                    key=lambda operator: (
                        -len(groups[operator]),
                        -max(len(example.rhs) for example in groups[operator]),
                        operator,
                    ),
                )
            )
            for operator in other_operators:
                rule_names = _family_rule_names_for_examples(motif, groups[operator])
                rules = tuple(BASE_RULE_BY_NAME[name] for name in rule_names)
                partials = _fit_family_group_from_partials(
                    partials,
                    operator,
                    groups[operator],
                    motif,
                    rules,
                    max_states=max_states,
                )
                if not partials:
                    break
            if not partials:
                continue

            query_lengths = tuple(sorted({len(example.rhs) for example in same}))
            variants_seen: set[str] = set()
            local_scored: list[tuple[tuple[int, str], SymbolTransformCandidate]] = []
            for sym_to_digit, chosen_rules in partials:
                for query_map in _complete_query_operand_maps(
                    puzzle.query,
                    sym_to_digit,
                    max_unknowns=max_query_unknowns,
                ):
                    xy = _pair_to_xy(motif.pairing, puzzle.query, query_map)
                    if xy is None:
                        continue
                    value = query_rule.apply(*xy)
                    token_variants = _render_query_token_variants(
                        value,
                        motif.output_mode,
                        query_lengths,
                    )
                    for tokens in token_variants:
                        if any(token == OP_TOKEN for token in tokens):
                            continue
                        prediction = _encode_tokens(
                            tokens,
                            query_map,
                            operator=query_operator,
                        )
                        if prediction is None:
                            continue
                        variants_seen.add(prediction)
                        candidate = SymbolTransformCandidate(
                            prediction=prediction,
                            motif=motif,
                            query_rule=query_rule,
                            query_operator_seen=True,
                            query_example_count=len(same),
                            total_examples_supported=len(puzzle.examples),
                            mapped_symbol_count=len(query_map),
                            chosen_rules=tuple(
                                sorted(
                                    (operator, rule.name)
                                    for operator, rule in chosen_rules.items()
                                )
                            ),
                            digit_map=tuple(sorted(query_map.items())),
                            query_completed_count=len(query_map) - len(sym_to_digit),
                        )
                        # This score is only for stable display order and
                        # representative metadata. It must not encode an answer
                        # choice heuristic such as longest output or largest
                        # numeric value.
                        score = (query_rule.rank, prediction)
                        local_scored.append((score, candidate))
                        if len(variants_seen) > max_prediction_variants:
                            local_scored = []
                            break
                    if len(variants_seen) > max_prediction_variants:
                        break
                if len(variants_seen) > max_prediction_variants:
                    break
            scored.extend(local_scored)

    best_by_prediction: dict[str, tuple[tuple[int, str], SymbolTransformCandidate]] = {}
    for score, candidate in scored:
        existing = best_by_prediction.get(candidate.prediction)
        if existing is None or score < existing[0]:
            best_by_prediction[candidate.prediction] = (score, candidate)

    return tuple(sorted(best_by_prediction.values(), key=lambda item: item[0]))


def _family_rescue_result(
    puzzle: SymbolTransformPuzzle,
    *,
    max_query_unknowns: int,
    selection: str = "unique",
) -> SymbolTransformSolveResult | None:
    scored_candidates = _search_family_rescue_candidates(
        puzzle,
        max_query_unknowns=max(2, max_query_unknowns),
    )
    if not scored_candidates:
        return None

    variants = tuple(sorted({candidate.prediction for _, candidate in scored_candidates}))
    chosen = scored_candidates[0][1]
    if len(variants) != 1:
        return SymbolTransformSolveResult(
            prediction=None,
            confidence="ambiguous",
            method="family_rescue_ambiguous",
            candidate_count=len(scored_candidates),
            prediction_variants=variants,
            chosen_candidate=chosen,
            notes=(
                "Heuristic rescue found multiple outputs. Do not rank these for deterministic data; "
                "the gold answer is often present but the tie-break is not reliable.",
            ),
        )

    return SymbolTransformSolveResult(
        prediction=chosen.prediction,
        confidence="low",
        method="family_rescue_unique",
        candidate_count=len(scored_candidates),
        prediction_variants=variants,
        chosen_candidate=chosen,
        notes=(
            "Heuristic rescue: fixed a dominant motif and fit add/sub/mul families across operators. "
            "A prediction is emitted only when every valid candidate gives the same output.",
        ),
    )


def _is_safe_mul_unit_candidate(candidate: SymbolTransformCandidate) -> bool:
    """Conservative promotion rule for the unit-first multiplication search.

    The broader unit-constraint search is useful for analysis, but public-train
    checks showed BA_DC|rev locks can be coincidental. Promote only the stable
    AB_CD/raw branch; for plain multiplication, require the row to determine all
    ten digit symbols before trusting the query completion.
    """

    if candidate.motif != Motif("AB_CD", "raw"):
        return False
    if candidate.query_rule.name == "x * y" and candidate.mapped_symbol_count < 10:
        return False
    return True


def _rule_bank(rule_bank: str) -> tuple[BaseRule, ...]:
    if rule_bank == "core":
        return tuple(rule for rule in BASE_RULES if rule.name in CORE_RULE_NAMES)
    if rule_bank == "asymmetric":
        return tuple(rule for rule in BASE_RULES if rule.is_asymmetric or rule.name in CORE_RULE_NAMES)
    if rule_bank == "extended":
        return BASE_RULES
    raise ValueError(f"Unknown rule bank: {rule_bank}")


def _motif_bank(include_abs: bool, output_mode_bank: str) -> tuple[Motif, ...]:
    modes = OUTPUT_MODE_BANKS[output_mode_bank]
    scan_order = EXTENDED_MOTIF_SCAN_ORDER if output_mode_bank != "current" else CURRENT_MOTIF_SCAN_ORDER
    if include_abs:
        return tuple(motif for motif in scan_order if motif.output_mode in modes)
    return tuple(motif for motif in scan_order if motif.output_mode in modes and motif.output_mode != "abs")


def _search_encrypted_digit_transform(
    puzzle: SymbolTransformPuzzle,
    *,
    rule_bank: str,
    include_abs_output: bool,
    output_mode_bank: str,
    max_states_per_rule: int,
    max_combined_states: int,
    allow_absent_query_operator: bool,
    min_query_examples: int,
    max_query_unknowns: int,
    motif_filter: frozenset[Motif] | None = None,
) -> tuple[SymbolTransformCandidate, ...]:
    groups = _group_examples_by_operator(puzzle.examples)
    query_operator = puzzle.query_operator
    if query_operator not in groups and not allow_absent_query_operator:
        return ()
    if query_operator in groups and len(groups[query_operator]) < min_query_examples:
        return ()

    base_rules = _rule_bank(rule_bank)
    candidates: list[SymbolTransformCandidate] = []

    for motif in _motif_bank(include_abs_output, output_mode_bank):
        if motif_filter is not None and motif not in motif_filter:
            continue
        fits_by_operator: dict[str, tuple[OperatorFit, ...]] = {}
        for operator, examples in groups.items():
            fits = _fit_operator(
                operator,
                examples,
                motif,
                base_rules,
                max_states_per_rule=max_states_per_rule,
            )
            if not fits:
                fits_by_operator = {}
                break
            fits_by_operator[operator] = fits
        if not fits_by_operator:
            continue

        group_order = sorted(
            groups,
            key=lambda operator: (
                0 if operator == query_operator else 1,
                sum(len(fit.states) for fit in fits_by_operator[operator]),
                operator,
            ),
        )

        partials: list[tuple[dict[str, int], dict[str, BaseRule]]] = [({}, {})]
        for operator in group_order:
            next_partials: list[tuple[dict[str, int], dict[str, BaseRule]]] = []
            for current_map, chosen_rules in partials:
                for fit in fits_by_operator[operator]:
                    for state in fit.states:
                        merged = _merge_maps(current_map, state)
                        if merged is None:
                            continue
                        next_rules = dict(chosen_rules)
                        next_rules[operator] = fit.base_rule
                        next_partials.append((merged, next_rules))
            if not next_partials:
                partials = []
                break
            next_partials.sort(
                key=lambda item: (
                    len(set(rule.family for rule in item[1].values())),
                    sum(rule.rank for rule in item[1].values()),
                    -len(item[0]),
                    tuple(sorted((op, rule.name) for op, rule in item[1].items())),
                )
            )
            partials = next_partials[:max_combined_states]
        if not partials:
            continue

        for sym_to_digit, chosen_rules in partials:
            query_rule = chosen_rules.get(query_operator)
            query_seen = query_rule is not None
            if query_rule is None:
                used_names = {rule.name for rule in chosen_rules.values()}
                query_rule = next(
                    (
                        rule
                        for rule in base_rules
                        if rule.name not in used_names and rule.is_asymmetric
                    ),
                    None,
                )
                if query_rule is None:
                    continue

            query_lengths = tuple(
                sorted({len(example.rhs) for example in groups.get(query_operator, puzzle.examples)})
            )
            for query_map in _complete_query_operand_maps(
                puzzle.query,
                sym_to_digit,
                max_unknowns=max_query_unknowns,
            ):
                query_completed_count = len(query_map) - len(sym_to_digit)
                xy = _pair_to_xy(motif.pairing, puzzle.query, query_map)
                if xy is None:
                    continue
                token_variants = _render_query_token_variants(
                    query_rule.apply(*xy),
                    motif.output_mode,
                    query_lengths,
                )
                for tokens in token_variants:
                    prediction = _encode_tokens(tokens, query_map, operator=query_operator)
                    if prediction is None:
                        continue

                    candidates.append(
                        SymbolTransformCandidate(
                            prediction=prediction,
                            motif=motif,
                            query_rule=query_rule,
                            query_operator_seen=query_seen,
                            query_example_count=len(groups.get(query_operator, ())),
                            total_examples_supported=len(puzzle.examples),
                            mapped_symbol_count=len(query_map),
                            chosen_rules=tuple(
                                sorted((operator, rule.name) for operator, rule in chosen_rules.items())
                            ),
                            digit_map=tuple(sorted(query_map.items())),
                            query_completed_count=query_completed_count,
                        ),
                    )

    return _dedupe_candidates(candidates)


def _search_same_operator_encrypted_digit_transform(
    puzzle: SymbolTransformPuzzle,
    *,
    rule_bank: str,
    include_abs_output: bool,
    output_mode_bank: str,
    max_states_per_rule: int,
    max_query_unknowns: int,
) -> tuple[SymbolTransformCandidate, ...]:
    same = same_operator_examples(puzzle)
    if len(same) < 2:
        return ()

    base_rules = _rule_bank(rule_bank)
    candidates: list[SymbolTransformCandidate] = []
    query_lengths = tuple(sorted({len(example.rhs) for example in same}))
    for motif in _motif_bank(include_abs_output, output_mode_bank):
        fits = _fit_operator(
            puzzle.query_operator,
            same,
            motif,
            base_rules,
            max_states_per_rule=max_states_per_rule,
        )
        for fit in fits:
            for sym_to_digit in fit.states:
                for query_map in _complete_query_operand_maps(
                    puzzle.query,
                    sym_to_digit,
                    max_unknowns=max_query_unknowns,
                ):
                    query_completed_count = len(query_map) - len(sym_to_digit)
                    xy = _pair_to_xy(motif.pairing, puzzle.query, query_map)
                    if xy is None:
                        continue
                    token_variants = _render_query_token_variants(
                        fit.base_rule.apply(*xy),
                        motif.output_mode,
                        query_lengths,
                    )
                    for tokens in token_variants:
                        prediction = _encode_tokens(tokens, query_map, operator=puzzle.query_operator)
                        if prediction is None:
                            continue
                        candidates.append(
                            SymbolTransformCandidate(
                                prediction=prediction,
                                motif=motif,
                                query_rule=fit.base_rule,
                                query_operator_seen=True,
                                query_example_count=len(same),
                                total_examples_supported=len(same),
                                mapped_symbol_count=len(query_map),
                                chosen_rules=((puzzle.query_operator, fit.base_rule.name),),
                                digit_map=tuple(sorted(query_map.items())),
                                query_completed_count=query_completed_count,
                            )
                        )
    return _dedupe_candidates(candidates)


def _safe_mul_unit_constraint_result(
    puzzle: SymbolTransformPuzzle,
    *,
    max_query_unknowns: int,
) -> SymbolTransformSolveResult | None:
    candidates = tuple(
        candidate
        for candidate in _search_same_operator_mul_unit_constraint(
            puzzle,
            max_query_unknowns=max_query_unknowns,
        )
        if _is_safe_mul_unit_candidate(candidate)
    )
    variants = tuple(sorted({candidate.prediction for candidate in candidates}))
    if len(variants) != 1:
        return None

    return SymbolTransformSolveResult(
        prediction=variants[0],
        confidence="high",
        method="same_operator_mul_unit_constraint_unique",
        candidate_count=len(candidates),
        prediction_variants=variants,
        chosen_candidate=candidates[0],
        notes=(
            "Rescue: same-operator multiplication digit constraints give a unique query output.",
        ),
    )


def _same_operator_map_completion_result(
    puzzle: SymbolTransformPuzzle,
    *,
    rule_bank: str,
    include_abs_output: bool,
    output_mode_bank: str,
    max_states_per_rule: int,
    max_completion_states: int,
) -> SymbolTransformSolveResult | None:
    candidates = _search_same_operator_map_completion_candidates(
        puzzle,
        rule_bank=rule_bank,
        include_abs_output=include_abs_output,
        output_mode_bank=output_mode_bank,
        max_states_per_rule=max_states_per_rule,
        max_completion_states=max_completion_states,
    )
    variants = tuple(sorted({candidate.prediction for candidate in candidates}))
    if len(variants) != 1:
        return None

    chosen = candidates[0]
    confidence = "high" if chosen.query_example_count >= 2 else "medium"
    return SymbolTransformSolveResult(
        prediction=variants[0],
        confidence=confidence,
        method="same_operator_map_completion_unique",
        candidate_count=len(candidates),
        prediction_variants=variants,
        chosen_candidate=chosen,
        notes=(
            "Same-operator examples lock the query rule; other equations complete the missing symbol map.",
        ),
    )


def _guarded_min1_global_result(
    puzzle: SymbolTransformPuzzle,
    *,
    rule_bank: str,
    include_abs_output: bool,
    output_mode_bank: str,
    max_states_per_rule: int,
    max_combined_states: int,
    max_query_unknowns: int,
) -> SymbolTransformSolveResult | None:
    """Conservative rescue for exactly one same-operator example.

    The normal global solver requires at least two examples of the query
    operator because one row is often coincidental. This branch only relaxes
    that threshold when the full row still locks a complete digit map under one
    of the two dominant motifs. It never ranks multiple possible outputs.
    """

    same = same_operator_examples(puzzle)
    if len(same) != 1 or len(puzzle.examples) < GUARDED_MIN1_MIN_SUPPORTED_EXAMPLES:
        return None

    groups = _group_examples_by_operator(puzzle.examples)
    allowed_rule_names = {rule.name for rule in _rule_bank(rule_bank)}
    allowed_motifs = {
        motif
        for motif in _motif_bank(include_abs_output, output_mode_bank)
        if motif in GUARDED_MIN1_GLOBAL_MOTIFS
    }
    group_order = tuple(
        sorted(
            groups,
            key=lambda operator: (
                -len(groups[operator]),
                -max(len(example.rhs) for example in groups[operator]),
                0 if operator == puzzle.query_operator else 1,
                operator,
            ),
        )
    )
    candidates: list[SymbolTransformCandidate] = []

    for motif in sorted(allowed_motifs, key=lambda item: MOTIF_RANK.get(item, 99)):
        partials: tuple[tuple[dict[str, int], dict[str, BaseRule]], ...] = (({}, {}),)
        for operator in group_order:
            rule_names = tuple(
                name
                for name in _family_rule_names_for_examples(motif, groups[operator])
                if name in allowed_rule_names
            )
            if not rule_names:
                partials = ()
                break
            rules = tuple(BASE_RULE_BY_NAME[name] for name in rule_names)
            partials = _fit_family_group_from_partials(
                partials,
                operator,
                groups[operator],
                motif,
                rules,
                max_states=max(max_states_per_rule, max_combined_states),
            )
            if not partials:
                break
            partials = partials[:max_combined_states]
        if not partials:
            continue

        query_lengths = tuple(sorted({len(example.rhs) for example in same}))
        for sym_to_digit, chosen_rules in partials:
            query_rule = chosen_rules.get(puzzle.query_operator)
            if query_rule is None:
                continue
            for query_map in _complete_query_operand_maps(
                puzzle.query,
                sym_to_digit,
                max_unknowns=max_query_unknowns,
            ):
                if len(query_map) < GUARDED_MIN1_MIN_MAPPED_SYMBOLS:
                    continue
                predictions = _query_predictions_from_state(
                    puzzle,
                    motif,
                    query_rule,
                    query_map,
                    query_lengths=query_lengths,
                )
                for prediction in predictions:
                    candidates.append(
                        SymbolTransformCandidate(
                            prediction=prediction,
                            motif=motif,
                            query_rule=query_rule,
                            query_operator_seen=True,
                            query_example_count=1,
                            total_examples_supported=len(puzzle.examples),
                            mapped_symbol_count=len(query_map),
                            chosen_rules=tuple(
                                sorted(
                                    (operator, rule.name)
                                    for operator, rule in chosen_rules.items()
                                )
                            ),
                            digit_map=tuple(sorted(query_map.items())),
                            query_completed_count=len(query_map) - len(sym_to_digit),
                        )
                    )

    candidates = _dedupe_candidates(candidates)
    variants = tuple(sorted({candidate.prediction for candidate in candidates}))
    if len(variants) != 1:
        return None

    return SymbolTransformSolveResult(
        prediction=variants[0],
        confidence="medium",
        method="guarded_min1_global_consistency_unique",
        candidate_count=len(candidates),
        prediction_variants=variants,
        chosen_candidate=candidates[0],
        notes=(
            "Guarded rescue: exactly one same-operator example is allowed only "
            "when all examples fit a dominant motif and determine a complete "
            "10-symbol digit map with a unique query output.",
        ),
    )


def _length1_subtraction_rescue_result(
    puzzle: SymbolTransformPuzzle,
    *,
    max_states_per_rule: int,
    max_combined_states: int,
    max_query_unknowns: int,
) -> SymbolTransformSolveResult | None:
    """Rescue rows where a one-symbol RHS indicates subtraction family.

    This intentionally uses a narrow rule set:
    - the query operator must have same-operator examples with one-symbol RHS
    - the query operator is fit only with x-y, y-x, or |x-y|
    - helper rows with length-3 RHS try x+y before x+y+1
    """

    same = same_operator_examples(puzzle)
    if not same or max(len(example.rhs) for example in same) != 1:
        return None

    groups = _group_examples_by_operator(puzzle.examples)
    other_operators = tuple(operator for operator in groups if operator != puzzle.query_operator)
    if not other_operators:
        return None

    candidates: list[SymbolTransformCandidate] = []
    # One-symbol subtraction rows are intentionally underconstrained. Keep a
    # wider local beam until helper rows such as multiplication/addition have a
    # chance to collapse the candidate set.
    state_cap = max(max_states_per_rule, max_combined_states, 5000)
    combined_cap = max(max_combined_states, 5000)

    for motif in LENGTH1_SUBTRACTION_RESCUE_MOTIFS:
        partials: tuple[tuple[dict[str, int], dict[str, BaseRule]], ...] = ()
        for rule_name in SUB_RULE_NAMES:
            fitted = _fit_family_group_from_partials(
                (({}, {}),),
                puzzle.query_operator,
                same,
                motif,
                (BASE_RULE_BY_NAME[rule_name],),
                max_states=state_cap,
            )
            if fitted:
                partials = (*partials, *fitted)

        partials = tuple(_dedupe_partials(partials))[:combined_cap]
        if not partials:
            continue

        group_order = tuple(
            sorted(
                other_operators,
                key=lambda operator: (
                    -max(len(example.rhs) for example in groups[operator]),
                    -len(groups[operator]),
                    operator,
                ),
            )
        )
        supported_operators = {puzzle.query_operator}
        used_len3_base_add_helper = False

        for operator in group_order:
            selected: tuple[tuple[dict[str, int], dict[str, BaseRule]], ...] = ()
            selected_rule_name = ""
            for rule_name in _length1_rescue_rule_names_for_examples(groups[operator]):
                fitted = _fit_family_group_from_partials(
                    partials,
                    operator,
                    groups[operator],
                    motif,
                    (BASE_RULE_BY_NAME[rule_name],),
                    max_states=state_cap,
                )
                if fitted:
                    selected = tuple(fitted[:combined_cap])
                    selected_rule_name = rule_name
                    break
            if selected:
                partials = selected
                supported_operators.add(operator)
                if (
                    max(len(example.rhs) for example in groups[operator]) == 3
                    and selected_rule_name == "x + y"
                ):
                    used_len3_base_add_helper = True

        if len(supported_operators) < 2 or not used_len3_base_add_helper:
            continue

        query_lengths = tuple(sorted({len(example.rhs) for example in same}))
        for sym_to_digit, chosen_rules in partials:
            query_rule = chosen_rules.get(puzzle.query_operator)
            if query_rule is None:
                continue
            for query_map in _complete_query_operand_maps(
                puzzle.query,
                sym_to_digit,
                max_unknowns=max_query_unknowns,
            ):
                query_completed_count = len(query_map) - len(sym_to_digit)
                predictions = _query_predictions_from_state(
                    puzzle,
                    motif,
                    query_rule,
                    query_map,
                    query_lengths=query_lengths,
                )
                for prediction in predictions:
                    candidates.append(
                        SymbolTransformCandidate(
                            prediction=prediction,
                            motif=motif,
                            query_rule=query_rule,
                            query_operator_seen=True,
                            query_example_count=len(same),
                            total_examples_supported=sum(
                                len(groups[operator]) for operator in chosen_rules
                            ),
                            mapped_symbol_count=len(query_map),
                            chosen_rules=tuple(
                                sorted(
                                    (operator, rule.name)
                                    for operator, rule in chosen_rules.items()
                                )
                            ),
                            digit_map=tuple(sorted(query_map.items())),
                            query_completed_count=query_completed_count,
                        )
                    )

    candidates = _dedupe_candidates(candidates)
    variants = tuple(sorted({candidate.prediction for candidate in candidates}))
    if len(variants) != 1:
        return None

    return SymbolTransformSolveResult(
        prediction=variants[0],
        confidence="medium",
        method="length1_subtraction_rescue_unique",
        candidate_count=len(candidates),
        prediction_variants=variants,
        chosen_candidate=candidates[0],
        notes=(
            "Length-1 same-operator RHS rescue: query operator is restricted "
            "to subtraction family, while length-3 helper rows try x+y before "
            "x+y+1 or multiplication variants.",
        ),
    )


def _signed_operator_marker_rescue_result(
    puzzle: SymbolTransformPuzzle,
    *,
    max_query_unknowns: int,
) -> SymbolTransformSolveResult | None:
    """Conservative rescue for rows that visibly use the operator as a sign.

    Some symbol-equation rows render a negative value by prefixing the operator
    character instead of using a minus sign. The full `formats` motif bank is
    too broad and creates many spurious matches, so this branch only tests the
    stable observed submotif:

    - operands use AB_CD
    - negative outputs use operator-prefix rendering
    - at least one example visibly prefixes its RHS with its own operator
    - concat helper rules are forbidden because direct templates already cover
      concat-like behavior
    - pure add-family rows are rejected; they were the observed false-positive
      shape in public-train diagnostics
    """

    if not any(example.rhs.startswith(example.operator) for example in puzzle.examples):
        return None

    candidates = _search_encrypted_digit_transform(
        puzzle,
        rule_bank="core",
        include_abs_output=True,
        output_mode_bank="formats",
        max_states_per_rule=SIGNED_MARKER_RESCUE_MAX_STATES,
        max_combined_states=SIGNED_MARKER_RESCUE_MAX_STATES,
        allow_absent_query_operator=False,
        min_query_examples=1,
        max_query_unknowns=max_query_unknowns,
        motif_filter=frozenset(SIGNED_MARKER_RESCUE_MOTIFS),
    )

    trusted: list[SymbolTransformCandidate] = []
    for candidate in candidates:
        chosen_rule_names = tuple(rule_name for _, rule_name in candidate.chosen_rules)
        if any(rule_name.startswith("concat(") for rule_name in chosen_rule_names):
            continue
        if {
            BASE_RULE_BY_NAME[rule_name].family
            for rule_name in chosen_rule_names
        } == {"add"}:
            continue
        if candidate.mapped_symbol_count < 9:
            continue
        if candidate.query_rule.name == "x * y - 1" and candidate.mapped_symbol_count < 10:
            continue
        trusted.append(candidate)

    trusted = list(_dedupe_candidates(trusted))
    variants = tuple(sorted({candidate.prediction for candidate in trusted}))
    if len(variants) != 1:
        return None

    chosen = trusted[0]
    return SymbolTransformSolveResult(
        prediction=variants[0],
        confidence="high" if chosen.mapped_symbol_count == 10 else "medium",
        method="signed_operator_marker_global_unique",
        candidate_count=len(trusted),
        prediction_variants=variants,
        chosen_candidate=chosen,
        notes=(
            "Signed-marker rescue: examples use the operator as a negative-prefix marker, "
            "and a guarded AB_CD|op_prefix_if_neg global fit gives a unique output.",
        ),
    )


def _last_digit_rescue_result(
    puzzle: SymbolTransformPuzzle,
    *,
    max_query_unknowns: int,
) -> SymbolTransformSolveResult | None:
    """Late fallback for rows that use the last digits of a raw result.

    This is intentionally narrower than the full `last/last_rev` format bank.
    Public-train diagnostics showed `last_rev` and subtraction variants were
    noisy, while `AB_CD|last` with cross-family evidence recovered a small set
    of otherwise no-rule rows without disturbing existing exact solves.
    """

    same = same_operator_examples(puzzle)
    if len(same) != 1:
        return None
    helper_examples = tuple(
        example for example in puzzle.examples if example.operator != puzzle.query_operator
    )
    if not helper_examples or max(len(example.rhs) for example in helper_examples) < 3:
        return None

    candidates = _search_encrypted_digit_transform(
        puzzle,
        rule_bank="core",
        include_abs_output=True,
        output_mode_bank="formats",
        max_states_per_rule=LAST_DIGIT_RESCUE_MAX_STATES,
        max_combined_states=LAST_DIGIT_RESCUE_MAX_STATES,
        allow_absent_query_operator=False,
        min_query_examples=1,
        max_query_unknowns=max_query_unknowns,
        motif_filter=frozenset(LAST_DIGIT_RESCUE_MOTIFS),
    )

    trusted: list[SymbolTransformCandidate] = []
    for candidate in candidates:
        if candidate.query_rule.family not in {"add", "mul"}:
            continue
        chosen_rule_names = tuple(rule_name for _, rule_name in candidate.chosen_rules)
        if any(rule_name.startswith("concat(") for rule_name in chosen_rule_names):
            continue
        if candidate.mapped_symbol_count < 9:
            continue
        if {
            BASE_RULE_BY_NAME[rule_name].family
            for rule_name in chosen_rule_names
        } in ({"add"}, {"mul"}):
            continue
        trusted.append(candidate)

    trusted = list(_dedupe_candidates(trusted))
    variants = tuple(sorted({candidate.prediction for candidate in trusted}))
    if len(variants) != 1:
        return None

    chosen = trusted[0]
    return SymbolTransformSolveResult(
        prediction=variants[0],
        confidence="medium",
        method="last_digit_global_unique",
        candidate_count=len(trusted),
        prediction_variants=variants,
        chosen_candidate=chosen,
        notes=(
            "Late last-digit rescue: AB_CD|last with add/mul query rule and cross-family helper evidence "
            "gives a unique output after the standard solver found no deterministic rule.",
        ),
    )


def _ba_dc_rev_guarded_rescue_result(
    puzzle: SymbolTransformPuzzle,
    *,
    max_query_unknowns: int,
) -> SymbolTransformSolveResult | None:
    """Late rescue for tightly guarded BA_DC|rev rows.

    This keeps the original dominant motif but avoids broad BA_DC guessing. Two
    public-train-safe shapes are allowed:

    - length-3 same-operator evidence with query rule x+y and a unique global
      BA_DC|rev fit
    - exactly one same-operator row, where length 4 requires a complete-map
      multiplication-family fit and length 3 requires x+y with at least 9
      mapped symbols
    """

    same = same_operator_examples(puzzle)
    if not same:
        return None
    same_rhs_lengths = tuple(sorted(len(example.rhs) for example in same))
    max_same_len = max(same_rhs_lengths)
    if max_same_len not in {3, 4}:
        return None

    candidates = _search_encrypted_digit_transform(
        puzzle,
        rule_bank="core",
        include_abs_output=True,
        output_mode_bank="current",
        max_states_per_rule=BA_DC_REV_RESCUE_MAX_STATES,
        max_combined_states=BA_DC_REV_RESCUE_MAX_STATES,
        allow_absent_query_operator=False,
        min_query_examples=1,
        max_query_unknowns=max_query_unknowns,
        motif_filter=frozenset((BA_DC_REV_RESCUE_MOTIF,)),
    )

    trusted: list[SymbolTransformCandidate] = []
    for candidate in candidates:
        if candidate.motif != BA_DC_REV_RESCUE_MOTIF:
            continue

        chosen_rule_names = tuple(rule_name for _, rule_name in candidate.chosen_rules)

        if len(same) == 1:
            if any(rule_name.startswith("concat(") for rule_name in chosen_rule_names):
                continue
            if max_same_len == 4:
                if candidate.query_rule.family != "mul":
                    continue
                if candidate.mapped_symbol_count < 10:
                    continue
            elif max_same_len == 3:
                if candidate.query_rule.name != "x + y":
                    continue
                if candidate.mapped_symbol_count < 9:
                    continue
            trusted.append(candidate)
            continue

        if max_same_len == 3:
            if candidate.query_rule.name != "x + y":
                continue
            if candidate.mapped_symbol_count < 9:
                continue
            trusted.append(candidate)

    trusted = list(_dedupe_candidates(trusted))
    variants = tuple(sorted({candidate.prediction for candidate in trusted}))
    if len(variants) != 1:
        return None

    chosen = trusted[0]
    confidence = "high" if chosen.mapped_symbol_count == 10 else "medium"
    return SymbolTransformSolveResult(
        prediction=variants[0],
        confidence=confidence,
        method="ba_dc_rev_guarded_global_unique",
        candidate_count=len(trusted),
        prediction_variants=variants,
        chosen_candidate=chosen,
        notes=(
            "Late BA_DC|rev rescue: a guarded dominant-motif global fit gives a unique output.",
        ),
    )


def _solve_encrypted_digit_transform_pipeline(
    puzzle: SymbolTransformPuzzle,
    *,
    rule_bank: str = "asymmetric",
    include_abs_output: bool = True,
    output_mode_bank: str = "current",
    max_states_per_rule: int = 120,
    max_combined_states: int = 200,
    allow_absent_query_operator: bool = False,
    min_query_examples_for_global: int = 2,
    max_query_unknowns: int = 0,
    selection: str = "unique",
    enable_family_rescue: bool = False,
    family_rescue_selection: str = "unique",
    enable_map_completion_rescue: bool = True,
    enable_guarded_min1_rescue: bool = True,
) -> SymbolTransformSolveResult:
    same_op_candidates = _search_same_operator_encrypted_digit_transform(
        puzzle,
        rule_bank=rule_bank,
        include_abs_output=include_abs_output,
        output_mode_bank=output_mode_bank,
        max_states_per_rule=max_states_per_rule,
        max_query_unknowns=max_query_unknowns,
    )
    same_op_variants = tuple(sorted({candidate.prediction for candidate in same_op_candidates}))
    if len(same_op_variants) == 1:
        global_candidates = _search_encrypted_digit_transform(
            puzzle,
            rule_bank=rule_bank,
            include_abs_output=include_abs_output,
            output_mode_bank=output_mode_bank,
            max_states_per_rule=max_states_per_rule,
            max_combined_states=max_combined_states,
            allow_absent_query_operator=allow_absent_query_operator,
            min_query_examples=min_query_examples_for_global,
            max_query_unknowns=max_query_unknowns,
        )
        global_variants = tuple(sorted({candidate.prediction for candidate in global_candidates}))
        if len(global_variants) == 1:
            chosen = global_candidates[0]
            method = "same_operator_global_consistency_unique"
            notes = (
                "Same-operator candidates agree, and row-global motif consistency gives a unique query output.",
            )
            return SymbolTransformSolveResult(
                prediction=global_variants[0],
                confidence="high",
                method=method,
                candidate_count=len(global_candidates),
                prediction_variants=global_variants,
                chosen_candidate=chosen,
                notes=notes,
            )

        if max_query_unknowns > 0 and any(
            candidate.query_completed_count for candidate in same_op_candidates
        ):
            if enable_map_completion_rescue:
                map_completion = _same_operator_map_completion_result(
                    puzzle,
                    rule_bank=rule_bank,
                    include_abs_output=include_abs_output,
                    output_mode_bank=output_mode_bank,
                    max_states_per_rule=max_states_per_rule,
                    max_completion_states=max_combined_states,
                )
                if map_completion is not None:
                    return map_completion
            unit_rescue = _safe_mul_unit_constraint_result(
                puzzle,
                max_query_unknowns=max_query_unknowns,
            )
            if unit_rescue is not None:
                return unit_rescue
            if enable_family_rescue:
                family_rescue = _family_rescue_result(
                    puzzle,
                    max_query_unknowns=max_query_unknowns,
                    selection=family_rescue_selection,
                )
                if family_rescue is not None:
                    return family_rescue
            return SymbolTransformSolveResult(
                prediction=None,
                confidence="ambiguous",
                method="same_operator_query_completion_requires_global_consistency",
                candidate_count=len(same_op_candidates),
                prediction_variants=same_op_variants,
                chosen_candidate=same_op_candidates[0],
                notes=(
                    "Same-operator query-completion candidates agree, but row-global consistency did not produce a unique output.",
                ),
            )

        chosen = same_op_candidates[0]
        return SymbolTransformSolveResult(
            prediction=same_op_variants[0],
            confidence="high",
            method="same_operator_encrypted_digit_transform_unique",
            candidate_count=len(same_op_candidates),
            prediction_variants=same_op_variants,
            chosen_candidate=chosen,
            notes=("Same-operator encrypted digit-transform candidates agree on the query output.",),
        )
    if selection == "ranked" and same_op_candidates:
        chosen = same_op_candidates[0]
        return SymbolTransformSolveResult(
            prediction=chosen.prediction,
            confidence="medium",
            method="same_operator_encrypted_digit_transform_ranked",
            candidate_count=len(same_op_candidates),
            prediction_variants=same_op_variants,
            chosen_candidate=chosen,
            notes=("Same-operator candidates disagreed; selected the best ranked motif/rule candidate.",),
        )

    if enable_map_completion_rescue:
        map_completion = _same_operator_map_completion_result(
            puzzle,
            rule_bank=rule_bank,
            include_abs_output=include_abs_output,
            output_mode_bank=output_mode_bank,
            max_states_per_rule=max_states_per_rule,
            max_completion_states=max_combined_states,
        )
        if map_completion is not None:
            return map_completion

    global_candidates = _search_encrypted_digit_transform(
        puzzle,
        rule_bank=rule_bank,
        include_abs_output=include_abs_output,
        output_mode_bank=output_mode_bank,
        max_states_per_rule=max_states_per_rule,
        max_combined_states=max_combined_states,
        allow_absent_query_operator=allow_absent_query_operator,
        min_query_examples=min_query_examples_for_global,
        max_query_unknowns=max_query_unknowns,
    )
    variants = tuple(sorted({candidate.prediction for candidate in global_candidates}))
    if not global_candidates:
        if enable_map_completion_rescue:
            map_completion = _same_operator_map_completion_result(
                puzzle,
                rule_bank=rule_bank,
                include_abs_output=include_abs_output,
                output_mode_bank=output_mode_bank,
                max_states_per_rule=max_states_per_rule,
                max_completion_states=max_combined_states,
            )
            if map_completion is not None:
                return map_completion
        if enable_guarded_min1_rescue:
            length1_rescue = _length1_subtraction_rescue_result(
                puzzle,
                max_states_per_rule=max_states_per_rule,
                max_combined_states=max_combined_states,
                max_query_unknowns=max_query_unknowns,
            )
            if length1_rescue is not None:
                return length1_rescue
            guarded_min1 = _guarded_min1_global_result(
                puzzle,
                rule_bank=rule_bank,
                include_abs_output=include_abs_output,
                output_mode_bank=output_mode_bank,
                max_states_per_rule=max_states_per_rule,
                max_combined_states=max_combined_states,
                max_query_unknowns=max_query_unknowns,
            )
            if guarded_min1 is not None:
                return guarded_min1
        unit_rescue = _safe_mul_unit_constraint_result(
            puzzle,
            max_query_unknowns=max_query_unknowns,
        )
        if unit_rescue is not None:
            return unit_rescue
        if enable_family_rescue:
            family_rescue = _family_rescue_result(
                puzzle,
                max_query_unknowns=max_query_unknowns,
                selection=family_rescue_selection,
            )
            if family_rescue is not None:
                return family_rescue
        if same_op_candidates:
            return SymbolTransformSolveResult(
                prediction=None,
                confidence="ambiguous",
                method="same_operator_encrypted_digit_transform_ambiguous",
                candidate_count=len(same_op_candidates),
                prediction_variants=same_op_variants,
                chosen_candidate=same_op_candidates[0],
                notes=(
                    "Same-operator encrypted digit-transform candidates fit but disagree on the query output.",
                ),
            )
        return SymbolTransformSolveResult(
            prediction=None,
            confidence="none",
            method="no_rule",
            candidate_count=0,
            prediction_variants=(),
            chosen_candidate=None,
            notes=("No direct template or encrypted digit-transform candidate fit.",),
        )

    if len(variants) == 1:
        chosen = global_candidates[0]
        return SymbolTransformSolveResult(
            prediction=variants[0],
            confidence="high" if chosen.query_example_count >= 2 else "medium",
            method="encrypted_digit_transform_unique",
            candidate_count=len(global_candidates),
            prediction_variants=variants,
            chosen_candidate=chosen,
            notes=("All fitted encrypted digit-transform candidates agree on the query output.",),
        )

    if selection == "ranked":
        chosen = global_candidates[0]
        return SymbolTransformSolveResult(
            prediction=chosen.prediction,
            confidence="medium" if chosen.query_example_count >= 1 else "low",
            method="encrypted_digit_transform_ranked",
            candidate_count=len(global_candidates),
            prediction_variants=variants,
            chosen_candidate=chosen,
            notes=("Multiple candidates fit; selected the best ranked motif/rule candidate.",),
        )

    unit_rescue = _safe_mul_unit_constraint_result(
        puzzle,
        max_query_unknowns=max_query_unknowns,
    )
    if unit_rescue is not None:
        return unit_rescue
    if enable_guarded_min1_rescue:
        length1_rescue = _length1_subtraction_rescue_result(
            puzzle,
            max_states_per_rule=max_states_per_rule,
            max_combined_states=max_combined_states,
            max_query_unknowns=max_query_unknowns,
        )
        if length1_rescue is not None:
            return length1_rescue
        guarded_min1 = _guarded_min1_global_result(
            puzzle,
            rule_bank=rule_bank,
            include_abs_output=include_abs_output,
            output_mode_bank=output_mode_bank,
            max_states_per_rule=max_states_per_rule,
            max_combined_states=max_combined_states,
            max_query_unknowns=max_query_unknowns,
        )
        if guarded_min1 is not None:
            return guarded_min1
    if enable_family_rescue:
        family_rescue = _family_rescue_result(
            puzzle,
            max_query_unknowns=max_query_unknowns,
            selection=family_rescue_selection,
        )
        if family_rescue is not None:
            return family_rescue

    return SymbolTransformSolveResult(
        prediction=None,
        confidence="ambiguous",
        method="encrypted_digit_transform_ambiguous",
        candidate_count=len(global_candidates),
        prediction_variants=variants,
        chosen_candidate=global_candidates[0],
        notes=("Multiple fitted encrypted digit-transform candidates disagree on the query output.",),
    )


def solve_symbol_transform(
    prompt: str,
    *,
    rule_bank: str = "asymmetric",
    include_abs_output: bool = True,
    output_mode_bank: str = "current",
    max_states_per_rule: int = 120,
    max_combined_states: int = 200,
    allow_absent_query_operator: bool = False,
    min_query_examples_for_global: int = 2,
    max_query_unknowns: int = 0,
    selection: str = "unique",
    adaptive_retry: bool = False,
    adaptive_retry_on: str = "no_rule",
    retry_max_states_per_rule: int = 240,
    retry_max_combined_states: int | None = None,
    enable_family_rescue: bool = False,
    family_rescue_selection: str = "unique",
    enable_map_completion_rescue: bool = True,
    enable_guarded_min1_rescue: bool = True,
    enable_signed_marker_rescue: bool = True,
    enable_last_digit_rescue: bool = True,
    enable_ba_dc_rev_rescue: bool = True,
) -> SymbolTransformSolveResult:
    puzzle = parse_symbol_transform_puzzle(prompt)
    if puzzle is None:
        return SymbolTransformSolveResult(
            prediction=None,
            confidence="none",
            method="parse_fail",
            candidate_count=0,
            prediction_variants=(),
            chosen_candidate=None,
            notes=("Prompt is not a length-5 symbol_transform puzzle.",),
        )

    direct = _try_direct_templates(puzzle)
    if direct is not None:
        return direct

    if enable_signed_marker_rescue:
        signed_marker = _signed_operator_marker_rescue_result(
            puzzle,
            max_query_unknowns=max_query_unknowns,
        )
        if signed_marker is not None:
            return signed_marker

    result = _solve_encrypted_digit_transform_pipeline(
        puzzle,
        rule_bank=rule_bank,
        include_abs_output=include_abs_output,
        output_mode_bank=output_mode_bank,
        max_states_per_rule=max_states_per_rule,
        max_combined_states=max_combined_states,
        allow_absent_query_operator=allow_absent_query_operator,
        min_query_examples_for_global=min_query_examples_for_global,
        max_query_unknowns=max_query_unknowns,
        selection=selection,
        enable_family_rescue=enable_family_rescue,
        family_rescue_selection=family_rescue_selection,
        enable_map_completion_rescue=enable_map_completion_rescue,
        enable_guarded_min1_rescue=enable_guarded_min1_rescue,
    )
    retry_allowed = result.method == "no_rule" or (
        adaptive_retry_on == "no_rule_or_ambiguous" and result.confidence == "ambiguous"
    )
    if not adaptive_retry or not retry_allowed or retry_max_states_per_rule <= max_states_per_rule:
        if enable_ba_dc_rev_rescue and result.prediction is None:
            ba_dc_rev = _ba_dc_rev_guarded_rescue_result(
                puzzle,
                max_query_unknowns=max_query_unknowns,
            )
            if ba_dc_rev is not None:
                return ba_dc_rev
        if enable_last_digit_rescue and result.method == "no_rule":
            last_digit = _last_digit_rescue_result(
                puzzle,
                max_query_unknowns=max_query_unknowns,
            )
            if last_digit is not None:
                return last_digit
        return result

    retry_result = _solve_encrypted_digit_transform_pipeline(
        puzzle,
        rule_bank=rule_bank,
        include_abs_output=include_abs_output,
        output_mode_bank=output_mode_bank,
        max_states_per_rule=retry_max_states_per_rule,
        max_combined_states=retry_max_combined_states or max_combined_states,
        allow_absent_query_operator=allow_absent_query_operator,
        min_query_examples_for_global=min_query_examples_for_global,
        max_query_unknowns=max_query_unknowns,
        selection=selection,
        enable_family_rescue=enable_family_rescue,
        family_rescue_selection=family_rescue_selection,
        enable_map_completion_rescue=enable_map_completion_rescue,
        enable_guarded_min1_rescue=enable_guarded_min1_rescue,
    )
    if retry_result.prediction is None:
        if enable_ba_dc_rev_rescue:
            ba_dc_rev = _ba_dc_rev_guarded_rescue_result(
                puzzle,
                max_query_unknowns=max_query_unknowns,
            )
            if ba_dc_rev is not None:
                return ba_dc_rev
        if enable_last_digit_rescue and retry_result.method == "no_rule":
            last_digit = _last_digit_rescue_result(
                puzzle,
                max_query_unknowns=max_query_unknowns,
            )
            if last_digit is not None:
                return last_digit
        return result

    return SymbolTransformSolveResult(
        prediction=retry_result.prediction,
        confidence=retry_result.confidence,
        method=f"adaptive_retry_{retry_result.method}",
        candidate_count=retry_result.candidate_count,
        prediction_variants=retry_result.prediction_variants,
        chosen_candidate=retry_result.chosen_candidate,
        notes=(
            *retry_result.notes,
            f"Recovered after widening max_states_per_rule from {max_states_per_rule} to {retry_max_states_per_rule}.",
        ),
    )
