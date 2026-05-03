from __future__ import annotations

import re

BOXED_ANSWER_INSTRUCTION = (
    "Please put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
WONDERLAND_PREFIX_RE = re.compile(r"^\s*In Alice['’]s Wonderland,\s*", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"(?is)<think>\s*(.*?)\s*</think>")


def strip_wonderland_prefix(prompt: str) -> str:
    return WONDERLAND_PREFIX_RE.sub("", prompt, count=1).strip()


def build_user_message(prompt: str) -> str:
    cleaned_prompt = prompt.strip()
    return f"{cleaned_prompt}\n{BOXED_ANSWER_INSTRUCTION}"


def build_assistant_message(answer: str) -> str:
    return f"\\boxed{{{answer}}}"


def normalize_generated_cot(generated_cot: str | None) -> str:
    if not generated_cot:
        return ""

    text = generated_cot.strip()
    if not text:
        return ""

    think_match = THINK_BLOCK_RE.search(text)
    if think_match:
        think_content = think_match.group(1).strip()
        if think_content:
            return f"<think>\n{think_content}\n</think>"
        return "<think>\n\n</think>"

    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    while lines:
        tail = lines[-1].strip()
        has_boxed_answer = "\\boxed{" in tail or "$\\boxed{" in tail
        has_final_answer_prefix = bool(
            re.search(r"(?i)^\s*(final answer|answer)\s*:", tail)
        )
        has_final_answer_phrase = bool(
            re.search(r"(?i)\b(final answer|answer)\b", tail)
        )
        if has_boxed_answer or has_final_answer_prefix or has_final_answer_phrase:
            lines.pop()
            while lines and not lines[-1].strip():
                lines.pop()
            continue
        break

    text = "\n".join(lines).strip()
    if not text:
        return ""
    return text


def build_messages(prompt: str, answer: str | None = None) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": build_user_message(prompt)}]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": build_assistant_message(answer)}
        )
    return messages


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"<|im_start|>{message['role']}")
            rendered.append(message["content"])
            rendered.append("<|im_end|>")
        if add_generation_prompt:
            rendered.append("<|im_start|>assistant")
        return "\n".join(rendered)


def build_assistant_content(answer: str, generated_cot: str | None = None) -> str:
    cot = normalize_generated_cot(generated_cot)
    if cot:
        if cot.lstrip().startswith("<think>") and "</think>" in cot:
            return f"{cot}\n\\boxed{{{answer}}}"
        return f"{cot}\n\n\\boxed{{{answer}}}"
    return build_assistant_message(answer)


def build_training_text(
    tokenizer,
    prompt: str,
    answer: str,
    generated_cot: str | None = None,
) -> str:
    assistant_content = build_assistant_content(answer, generated_cot)
    return apply_chat_template(
        tokenizer,
        [{"role": "user", "content": build_user_message(prompt)}, {"role": "assistant", "content": assistant_content}],
        add_generation_prompt=False,
    )


def build_generation_prompt(tokenizer, prompt: str) -> str:
    return apply_chat_template(
        tokenizer,
        build_messages(prompt, None),
        add_generation_prompt=True,
    )
