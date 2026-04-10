from __future__ import annotations

import re
from typing import Any

from .metric import answers_match, extract_answer


FINAL_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*\\boxed\{.*\}\s*$"
)
BOXED_PATTERN = re.compile(r"\\boxed\{")


def completion_to_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
        return completion_to_text(content)
    if isinstance(completion, list):
        parts = [completion_to_text(item) for item in completion]
        return "".join(part for part in parts if part).strip()
    return str(completion)


def has_exact_final_answer_line(text: str) -> bool:
    return bool(FINAL_ANSWER_LINE_RE.search(text))


def has_single_boxed_answer(text: str) -> bool:
    return len(BOXED_PATTERN.findall(text)) == 1


def accuracy_reward(
    prompts,
    completions,
    answer,
    **kwargs,
) -> list[float]:
    del prompts, kwargs
    rewards: list[float] = []
    for completion, gold_answer in zip(completions, answer):
        text = completion_to_text(completion)
        predicted = extract_answer(text)
        rewards.append(1.0 if answers_match(predicted, gold_answer) else 0.0)
    return rewards


def final_line_reward(
    prompts,
    completions,
    answer,
    **kwargs,
) -> list[float]:
    del prompts, answer, kwargs
    return [
        0.02 if has_exact_final_answer_line(completion_to_text(completion)) else 0.0
        for completion in completions
    ]


def single_box_reward(
    prompts,
    completions,
    answer,
    **kwargs,
) -> list[float]:
    del prompts, answer, kwargs
    return [
        0.08 if has_single_boxed_answer(completion_to_text(completion)) else 0.0
        for completion in completions
    ]


def competition_reward(
    prompts,
    completions,
    answer,
    **kwargs,
) -> list[float]:
    accuracy = accuracy_reward(prompts, completions, answer, **kwargs)
    single_box = single_box_reward(prompts, completions, answer, **kwargs)
    final_line = final_line_reward(prompts, completions, answer, **kwargs)
    return [
        accuracy_score + single_box_score + final_line_score
        for accuracy_score, single_box_score, final_line_score in zip(
            accuracy,
            single_box,
            final_line,
        )
    ]
