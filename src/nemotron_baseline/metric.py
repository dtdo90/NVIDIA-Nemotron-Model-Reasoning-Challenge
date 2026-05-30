from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Iterable


NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
PHRASE_PATTERNS = [
    re.compile(r"The final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
]


@dataclass(frozen=True)
class PredictionResult:
    example_id: str
    category: str
    gold_answer: str
    raw_prediction: str
    extracted_prediction: str
    correct: bool


def extract_boxed_answer(text: str) -> str | None:
    """Extract boxed answers using the competition metric convention.

    For each ``\boxed{`` occurrence, the reference evaluator reads up to the
    last ``}`` before the next boxed answer. This preserves literal answer
    characters such as ``}`` while still handling nested LaTeX-like payloads.
    """

    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for index, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[index + 1].start() if index + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if not matches:
        return None
    non_empty = [match.strip() for match in matches if match.strip()]
    if non_empty:
        return non_empty[-1]
    return matches[-1].strip()


def extract_answer(text: str | None) -> str:
    if text is None:
        return "NOT_FOUND"

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed.strip()

    for pattern in PHRASE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()

    numeric_matches = NUMERIC_PATTERN.findall(text)
    if numeric_matches:
        return numeric_matches[-1]

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty_lines:
        return non_empty_lines[-1]

    return "NOT_FOUND"


def answers_match(predicted: str, gold: str, relative_tolerance: float = 1e-2) -> bool:
    """Match answers using the reference competition metric.

    Binary strings are compared strictly as strings. Other numeric answers use
    the documented relative tolerance of 1e-2 plus the reference absolute
    tolerance of 1e-5. Non-numeric answers compare case-insensitively.
    """

    predicted_text = predicted.strip()
    gold_text = gold.strip()

    if re.fullmatch(r"[01]+", gold_text):
        return predicted_text.lower() == gold_text.lower()

    try:
        predicted_number = float(predicted_text)
        gold_number = float(gold_text)
        return math.isclose(
            gold_number,
            predicted_number,
            rel_tol=relative_tolerance,
            abs_tol=1e-5,
        )
    except Exception:
        return predicted_text.lower() == gold_text.lower()


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
