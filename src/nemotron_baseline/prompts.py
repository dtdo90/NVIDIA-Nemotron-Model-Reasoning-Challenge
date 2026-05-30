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


def build_user_message(prompt: str, *, append_answer_instruction: bool = True) -> str:
    cleaned_prompt = prompt.strip()
    if not append_answer_instruction:
        return cleaned_prompt
    return f"{cleaned_prompt}\n{BOXED_ANSWER_INSTRUCTION}"


def build_assistant_message(answer: str, *, answer_style: str = "boxed") -> str:
    if answer_style == "plain":
        return f"Answer: {answer}"
    if answer_style != "boxed":
        raise ValueError(f"Unsupported answer_style: {answer_style!r}")
    return f"Answer: \\boxed{{{answer}}}"


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


def build_messages(
    prompt: str,
    answer: str | None = None,
    *,
    append_answer_instruction: bool = True,
    answer_style: str = "boxed",
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "user",
            "content": build_user_message(
                prompt,
                append_answer_instruction=append_answer_instruction,
            ),
        }
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": build_assistant_message(answer, answer_style=answer_style),
            }
        )
    return messages


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> str:
    try:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"<|im_start|>{message['role']}")
            rendered.append(message["content"])
            rendered.append("<|im_end|>")
        if add_generation_prompt:
            rendered.append("<|im_start|>assistant")
        return "\n".join(rendered)


def build_assistant_content(
    answer: str,
    generated_cot: str | None = None,
    *,
    answer_style: str = "boxed",
) -> str:
    cot = normalize_generated_cot(generated_cot)
    if cot:
        if answer_style == "plain":
            return f"{cot}\n\nAnswer: {answer}"
        if answer_style != "boxed":
            raise ValueError(f"Unsupported answer_style: {answer_style!r}")
        if cot.lstrip().startswith("<think>") and "</think>" in cot:
            return f"{cot}\nAnswer: \\boxed{{{answer}}}"
        return f"{cot}\n\nAnswer: \\boxed{{{answer}}}"
    return build_assistant_message(answer, answer_style=answer_style)


def build_training_text(
    tokenizer,
    prompt: str,
    answer: str,
    generated_cot: str | None = None,
    *,
    append_answer_instruction: bool = True,
    answer_style: str = "boxed",
) -> str:
    assistant_content = build_assistant_content(
        answer,
        generated_cot,
        answer_style=answer_style,
    )
    return apply_chat_template(
        tokenizer,
        [
            {
                "role": "user",
                "content": build_user_message(
                    prompt,
                    append_answer_instruction=append_answer_instruction,
                ),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        add_generation_prompt=False,
    )


def build_generation_prompt(tokenizer, prompt: str) -> str:
    return apply_chat_template(
        tokenizer,
        build_messages(prompt, None),
        add_generation_prompt=True,
        enable_thinking=True,
    )


def build_assistant_trace_content(
    answer: str,
    *,
    generated_cot: str | None = None,
    assistant_content: str | None = None,
) -> str:
    """Build the assistant completion scored by SFT.

    The completion follows the competition-style thinking format:
    <think> ... Answer: \boxed{...} </think> \boxed{...}
    """
    if assistant_content and assistant_content.strip():
        return assistant_content.strip()

    cot = normalize_generated_cot(generated_cot)
    if cot.lstrip().startswith("<think>") and "</think>" in cot:
        think_match = THINK_BLOCK_RE.search(cot)
        inner = think_match.group(1).strip() if think_match else cot.strip()
    else:
        inner = cot.strip()

    if inner:
        lines = inner.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines or not re.search(r"(?i)^\s*Answer\s*:", lines[-1]) or "\\boxed{" not in lines[-1]:
            lines.extend(["", f"Answer: \\boxed{{{answer}}}"])
        inner = "\n".join(lines).strip()
    else:
        inner = f"Answer: \\boxed{{{answer}}}"

    return f"<think>\n{inner}\n</think>\n\\boxed{{{answer}}}"


def build_competition_prompt(
    tokenizer,
    prompt: str,
    *,
    append_answer_instruction: bool = True,
) -> str:
    """Render the exact user-side prompt style used by the competition metric."""
    return apply_chat_template(
        tokenizer,
        [
            {
                "role": "user",
                "content": build_user_message(
                    prompt,
                    append_answer_instruction=append_answer_instruction,
                ),
            }
        ],
        add_generation_prompt=True,
        enable_thinking=True,
    )
