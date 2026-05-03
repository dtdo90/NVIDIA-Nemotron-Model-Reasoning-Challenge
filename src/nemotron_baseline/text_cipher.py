from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


TEXT_ENCRYPTION_MAPPING_PREFIX = "From the examples, I extract character mappings:"

DEFAULT_WONDERLAND_VOCABULARY: tuple[str, ...] = (
    "above",
    "alice",
    "ancient",
    "around",
    "beyond",
    "bird",
    "book",
    "bright",
    "castle",
    "cat",
    "cave",
    "chases",
    "clever",
    "colorful",
    "creates",
    "crystal",
    "curious",
    "dark",
    "discovers",
    "door",
    "dragon",
    "draws",
    "dreams",
    "explores",
    "follows",
    "forest",
    "found",
    "garden",
    "golden",
    "hatter",
    "hidden",
    "imagines",
    "in",
    "inside",
    "island",
    "key",
    "king",
    "knight",
    "library",
    "magical",
    "map",
    "message",
    "mirror",
    "mountain",
    "mouse",
    "mysterious",
    "near",
    "ocean",
    "palace",
    "potion",
    "princess",
    "puzzle",
    "queen",
    "rabbit",
    "reads",
    "school",
    "secret",
    "sees",
    "silver",
    "story",
    "strange",
    "student",
    "studies",
    "teacher",
    "the",
    "through",
    "tower",
    "treasure",
    "turtle",
    "under",
    "valley",
    "village",
    "watches",
    "wise",
    "wizard",
    "wonderland",
    "writes",
)


@dataclass(frozen=True)
class TextCipherWordSolution:
    cipher_word: str
    partial: str
    pattern_candidates: tuple[str, ...]
    bijective_candidates: tuple[str, ...]
    chosen: str
    added_mappings: tuple[tuple[str, str], ...]
    fully_mapped: bool


@dataclass(frozen=True)
class TextCipherSolution:
    query: str
    initial_mapping: dict[str, str]
    final_mapping: dict[str, str]
    word_solutions: tuple[TextCipherWordSolution, ...]
    decoded_phrase: str
    unique: bool


def extract_text_encryption_examples(prompt: str) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line or "->" not in line:
            continue
        if line.lower().startswith("now, decrypt"):
            continue
        left, right = line.split("->", 1)
        cipher_text = left.strip()
        plain_text = right.strip()
        if cipher_text and plain_text:
            examples.append((cipher_text, plain_text))
    return examples


def extract_text_encryption_query(prompt: str) -> str:
    match = re.search(
        r"Now,\s*decrypt\s+the\s+following\s+text:\s*(.+?)\s*$",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError("Failed to extract Text Encryption target query from prompt.")
    return match.group(1).strip()


def derive_text_encryption_mapping(prompt: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cipher_text, plain_text in extract_text_encryption_examples(prompt):
        if len(cipher_text) != len(plain_text):
            raise ValueError(
                "Text Encryption example has mismatched lengths and cannot be aligned: "
                f"{cipher_text!r} -> {plain_text!r}"
            )
        for cipher_char, plain_char in zip(cipher_text, plain_text):
            if cipher_char == " " and plain_char == " ":
                continue
            if not cipher_char.isalpha() or not plain_char.isalpha():
                continue
            existing = mapping.get(cipher_char)
            if existing is not None and existing != plain_char:
                raise ValueError(
                    "Conflicting Text Encryption mapping derived from examples: "
                    f"{cipher_char!r} -> {existing!r} and {plain_char!r}"
                )
            mapping[cipher_char] = plain_char
    if not mapping:
        raise ValueError("Failed to derive any Text Encryption character mapping from prompt examples.")
    return dict(sorted(mapping.items()))


def expand_text_encryption_mapping_line(prompt: str) -> str:
    mapping = derive_text_encryption_mapping(prompt)
    mapping_items = ", ".join(f"{cipher}→{plain}" for cipher, plain in mapping.items())
    return f"{TEXT_ENCRYPTION_MAPPING_PREFIX} {mapping_items}"


def word_pattern(word: str) -> tuple[int, ...]:
    seen: dict[str, int] = {}
    pattern: list[int] = []
    for char in word:
        if char not in seen:
            seen[char] = len(seen)
        pattern.append(seen[char])
    return tuple(pattern)


def candidate_words_for_partial(
    *,
    partial: str,
    cipher_word: str,
    mapping: dict[str, str],
    vocabulary: Sequence[str] = DEFAULT_WONDERLAND_VOCABULARY,
    enforce_bijection: bool,
) -> tuple[str, ...]:
    plain_to_cipher = {plain: cipher for cipher, plain in mapping.items()}
    cipher_pattern = word_pattern(cipher_word)
    candidates: list[str] = []

    for word in sorted(vocabulary):
        if len(word) != len(cipher_word):
            continue
        if word_pattern(word) != cipher_pattern:
            continue
        if any(known != "?" and known != plain for known, plain in zip(partial, word)):
            continue

        local_cipher_to_plain: dict[str, str] = {}
        local_plain_to_cipher: dict[str, str] = {}
        valid = True
        for cipher_char, plain_char in zip(cipher_word, word):
            mapped_plain = mapping.get(cipher_char)
            if mapped_plain is not None:
                if mapped_plain != plain_char:
                    valid = False
                    break
                continue

            if enforce_bijection and plain_char in plain_to_cipher:
                valid = False
                break
            if local_cipher_to_plain.get(cipher_char, plain_char) != plain_char:
                valid = False
                break
            if local_plain_to_cipher.get(plain_char, cipher_char) != cipher_char:
                valid = False
                break
            local_cipher_to_plain[cipher_char] = plain_char
            local_plain_to_cipher[plain_char] = cipher_char

        if valid:
            candidates.append(word)

    return tuple(candidates)


def solve_text_cipher(
    prompt: str,
    *,
    vocabulary: Sequence[str] = DEFAULT_WONDERLAND_VOCABULARY,
) -> TextCipherSolution:
    initial_mapping = derive_text_encryption_mapping(prompt)
    working_mapping = dict(initial_mapping)
    query = extract_text_encryption_query(prompt)
    word_solutions: list[TextCipherWordSolution] = []
    decoded_words: list[str] = []
    unique = True

    for cipher_word in query.split():
        partial = "".join(working_mapping.get(char, "?") for char in cipher_word)
        fully_mapped = "?" not in partial
        if fully_mapped:
            pattern_candidates = (partial,) if partial in vocabulary else ()
            bijective_candidates = pattern_candidates
            chosen = partial
        else:
            pattern_candidates = candidate_words_for_partial(
                partial=partial,
                cipher_word=cipher_word,
                mapping=working_mapping,
                vocabulary=vocabulary,
                enforce_bijection=False,
            )
            bijective_candidates = candidate_words_for_partial(
                partial=partial,
                cipher_word=cipher_word,
                mapping=working_mapping,
                vocabulary=vocabulary,
                enforce_bijection=True,
            )
            if len(bijective_candidates) != 1:
                unique = False
            chosen = bijective_candidates[0] if bijective_candidates else partial

        added_mappings: list[tuple[str, str]] = []
        for cipher_char, plain_char in zip(cipher_word, chosen):
            if cipher_char not in working_mapping and plain_char != "?":
                working_mapping[cipher_char] = plain_char
                added_mappings.append((cipher_char, plain_char))

        decoded_words.append(chosen)
        word_solutions.append(
            TextCipherWordSolution(
                cipher_word=cipher_word,
                partial=partial,
                pattern_candidates=pattern_candidates,
                bijective_candidates=bijective_candidates,
                chosen=chosen,
                added_mappings=tuple(added_mappings),
                fully_mapped=fully_mapped,
            )
        )

    return TextCipherSolution(
        query=query,
        initial_mapping=initial_mapping,
        final_mapping=dict(sorted(working_mapping.items())),
        word_solutions=tuple(word_solutions),
        decoded_phrase=" ".join(decoded_words),
        unique=unique,
    )


def _format_candidates(candidates: Sequence[str], *, max_items: int = 12) -> str:
    if not candidates:
        return "none"
    if len(candidates) <= max_items:
        return ", ".join(candidates)
    shown = ", ".join(candidates[:max_items])
    return f"{shown}, ... ({len(candidates)} total)"


def _format_added_mappings(added_mappings: Sequence[tuple[str, str]]) -> str:
    if not added_mappings:
        return "none"
    return ", ".join(f"{cipher}→{plain}" for cipher, plain in added_mappings)


def render_text_cipher_compact_cot(
    prompt: str,
    *,
    vocabulary: Sequence[str] = DEFAULT_WONDERLAND_VOCABULARY,
) -> str:
    solution = solve_text_cipher(prompt, vocabulary=vocabulary)
    lines = [
        "We need to deduce the hidden text cipher by matching the example plaintext outputs.",
        "I will put my final answer inside \\boxed{}.",
        expand_text_encryption_mapping_line(prompt),
        "S1: This is a bijective letter-substitution cipher over the 77-word Wonderland vocabulary.",
        "S2: Decode with the example mapping first; use vocabulary matching only for unresolved target words.",
        f'S3: Target text: "{solution.query}"',
        "S4: Decode target words:",
    ]

    for word in solution.word_solutions:
        if word.fully_mapped:
            lines.append(f'- "{word.cipher_word}" -> "{word.chosen}" (fully mapped)')
            continue

        lines.append(
            f'- "{word.cipher_word}" -> "{word.partial}"; '
            f"vocab matches: {_format_candidates(word.pattern_candidates)}; "
            f"bijective matches: {_format_candidates(word.bijective_candidates)}; "
            f'choose "{word.chosen}"; add {_format_added_mappings(word.added_mappings)}'
        )

    lines.append(f"S5: Decoded phrase = {solution.decoded_phrase}")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The final answer is \\boxed{{{solution.decoded_phrase}}}")
    return "\n".join(lines)


def normalize_text_encryption_cot(prompt: str, reasoning: str | None) -> str:
    if not reasoning:
        return ""
    text = reasoning.strip()
    if not text:
        return ""
    lines = text.splitlines()
    expanded_mapping_line = expand_text_encryption_mapping_line(prompt)
    mapping_line_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith(TEXT_ENCRYPTION_MAPPING_PREFIX)
        ),
        None,
    )
    if mapping_line_index is not None:
        lines[mapping_line_index] = expanded_mapping_line
    else:
        lines.insert(0, expanded_mapping_line)
    return "\n".join(lines).strip()
