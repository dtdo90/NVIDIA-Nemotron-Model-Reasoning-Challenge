from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable


NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
PHRASE_PATTERNS = [
    re.compile(
        r"(?is)(?:final answer|answer)\s*(?:is|=|:)\s*(.+?)\s*$"
    ),
    re.compile(r"(?is)(?:therefore|thus|so)\s*,?\s*(.+?)\s*$"),
]


@dataclass(frozen=True)
class PredictionResult:
    example_id: str
    category: str
    gold_answer: str
    raw_prediction: str
    extracted_prediction: str
    correct: bool


def _extract_balanced(text: str, start_index: int) -> str | None:
    depth = 1
    index = start_index
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index:index]
        index += 1
    return None


def _extract_literal_boxed_from_line(line: str) -> str | None:
    """Extract the last boxed payload on one line as literal text.

    Symbol-transform answers can contain literal braces, backslashes, or dollar
    signs. For local audits we therefore treat the final boxed line as a
    delimiter convention: content starts after the last ``\boxed{`` on the line
    and ends at the line's final ``}``.
    """

    marker = r"\boxed{"
    marker_index = line.rfind(marker)
    if marker_index < 0:
        return None
    stripped = line.strip().rstrip(" .,;")
    if not stripped.endswith("}"):
        return stripped[marker_index + len(marker):].strip()
    return stripped[marker_index + len(marker):-1].strip()


def extract_boxed_answer(text: str) -> str | None:
    marker = r"\boxed{"

    # Prefer final-line literal extraction. This preserves challenge symbols
    # such as $, {, }, and \ that are valid answer characters.
    for line in reversed(text.splitlines()):
        content = _extract_literal_boxed_from_line(line)
        if content is not None:
            return content

    found: list[str] = []
    search_start = 0
    while True:
        marker_index = text.find(marker, search_start)
        if marker_index < 0:
            break
        content_start = marker_index + len(marker)
        content = _extract_balanced(text, content_start)
        if content is not None:
            found.append(content.strip())
        search_start = content_start
    return found[-1] if found else None


def _strip_wrappers(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.strip("$")
    cleaned = cleaned.strip()
    if cleaned.startswith("{") and cleaned.endswith("}") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.rstrip(" .,;")
    cleaned = cleaned.strip()
    return cleaned


def extract_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed.strip()

    for pattern in PHRASE_PATTERNS:
        match = pattern.search(text)
        if match:
            return _strip_wrappers(match.group(1))

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty_lines:
        line = non_empty_lines[-1]
        line = re.sub(r"^[-*]\s*", "", line)
        return _strip_wrappers(line)

    numeric_matches = NUMERIC_PATTERN.findall(text)
    if numeric_matches:
        return numeric_matches[-1]

    return ""


def _canonical_text(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.strip())


def _parse_decimal(answer: str) -> Decimal | None:
    normalized = _strip_wrappers(answer)
    if not re.fullmatch(NUMERIC_PATTERN, normalized):
        return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def answers_match(predicted: str, gold: str, relative_tolerance: float = 1e-2) -> bool:
    predicted_text = _canonical_text(predicted)
    gold_text = _canonical_text(gold)
    if predicted_text == gold_text:
        return True

    predicted_number = _parse_decimal(predicted_text)
    gold_number = _parse_decimal(gold_text)
    if predicted_number is None or gold_number is None:
        return False

    if gold_number == 0:
        return predicted_number == 0

    difference = abs(predicted_number - gold_number)
    threshold = abs(gold_number) * Decimal(str(relative_tolerance))
    return difference <= threshold


def score_prediction(
    *,
    example_id: str,
    category: str,
    raw_prediction: str,
    gold_answer: str,
) -> PredictionResult:
    extracted_prediction = extract_answer(raw_prediction)
    return PredictionResult(
        example_id=example_id,
        category=category,
        gold_answer=gold_answer,
        raw_prediction=raw_prediction,
        extracted_prediction=extracted_prediction,
        correct=answers_match(extracted_prediction, gold_answer),
    )


def summarize_results(results: Iterable[PredictionResult]) -> dict[str, object]:
    result_list = list(results)
    total = len(result_list)
    correct = sum(1 for result in result_list if result.correct)

    by_category: dict[str, dict[str, int | float]] = {}
    for result in result_list:
        stats = by_category.setdefault(
            result.category,
            {"total": 0, "correct": 0, "accuracy": 0.0},
        )
        stats["total"] += 1
        stats["correct"] += 1 if result.correct else 0

    for stats in by_category.values():
        stats["accuracy"] = (
            float(stats["correct"]) / float(stats["total"]) if stats["total"] else 0.0
        )

    return {
        "total": total,
        "correct": correct,
        "accuracy": float(correct) / float(total) if total else 0.0,
        "by_category": by_category,
    }


def result_to_json(result: PredictionResult) -> str:
    return json.dumps(
        {
            "id": result.example_id,
            "category": result.category,
            "gold_answer": result.gold_answer,
            "raw_prediction": result.raw_prediction,
            "extracted_prediction": result.extracted_prediction,
            "correct": result.correct,
        },
        ensure_ascii=False,
    )
