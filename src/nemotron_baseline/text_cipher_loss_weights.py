"""Token-level loss weighting for Text Cipher SFT traces.

The v5 Text Cipher traces already encode most of the repair behavior in data
itself, so weighting should be narrow: emphasize the tokens that decide the
branch or protect against the observed failure modes, and leave routine copies,
the vocabulary dump, and ordinary `no` scan lines flat.

Weight-2 spans:
  - process-example word alignments (`cipher -> plain`) and their `letters ...`
    source-letter anchors
  - all rows in `Summary character mappings`
  - `letters ...` anchors after target/candidate/re-read source lines
  - per-character decode outputs in target words
  - assembled target pattern
  - `match` scan verdicts and candidate-list lines
  - candidate verification outputs that can flip a verdict: `new`, `conflicts`,
    `already exists`, `already used`, `PASS`, `FAIL`, `PASS confirm`,
    `FAIL confirm, continue scanning`, and `add ...`
  - the chosen word and decoded phrase. `Answer: \\boxed{...}` is an echo for
    Text Cipher and stays weight 1.0, including the appended boxed echo after
    `</think>`.

Routine process-example mappings, `agrees` confirmations, vocab `no` lines, and
fixed prose scaffolding stay at weight 1.0.
"""
from __future__ import annotations

import re

HIGH = 2.0
BASE = 1.0

_VERDICT_LINES = {"PASS", "FAIL", "PASS confirm", "FAIL confirm, continue scanning"}
_ARROW_RE = re.compile(r"^(\s*)(\S+) -> (.+?)\s*$")
_SCAN_VERDICT_RE = re.compile(r"^(.+) (no|match)$")
_CANDIDATE_COUNT_RE = re.compile(r"^(no|one|two|three|four|five|six|seven|eight|nine|ten|\d+) candidates?$")
_MULTI_CANDIDATE_COUNT_RE = re.compile(
    r"^(two|three|four|five|six|seven|eight|nine|ten|\d+) candidates$"
)
_CHECK_OPENERS = (
    "for no candidate,",
    "for one candidate,",
    "for multiple candidates,",
    "for more than one candidate,",
)


def build_char_weights(text: str, *, high: float = HIGH, base: float = BASE) -> list[float]:
    """Per-character weight array over a completion string (len == len(text))."""
    weights = [base] * len(text)

    # line table with absolute char offsets (newline-preserving)
    lines: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        lines.append((offset, line))
        offset += len(line) + 1

    def mark(start: int, end: int) -> None:
        for k in range(max(start, 0), min(end, len(weights))):
            weights[k] = high

    def lead_len(line: str) -> int:
        return len(line) - len(line.lstrip())

    n = len(lines)
    promote_next = False
    in_process_examples = False
    in_summary_mappings = False
    in_target_decode = False
    in_candidate_check = False
    reread_skip_target = False
    reread_mark_word = False
    for i, (start, line) in enumerate(lines):
        stripped = line.strip()

        if stripped == "Process examples":
            in_process_examples = True
            in_target_decode = False
            in_candidate_check = False
            continue
        if stripped == "Summary character mappings":
            in_process_examples = False
            in_summary_mappings = True
            continue
        if stripped == "Vocab library":
            in_summary_mappings = False
            continue
        if stripped == "Decode target words from character mappings":
            in_process_examples = False
            in_target_decode = True
            in_candidate_check = False
            continue
        if stripped == "Decoded phrase":
            in_target_decode = False
            in_candidate_check = False
            promote_next = True
            continue
        if stripped == "Check scan candidates against summary mappings":
            in_candidate_check = True
            in_target_decode = False
            continue
        if stripped.startswith("Vocab pattern scan"):
            in_candidate_check = False
            continue
        if stripped.startswith("Scan candidates for "):
            in_candidate_check = False
            promote_next = True
            continue

        if stripped == "re-read source word from input query":
            reread_skip_target = True
            continue

        if promote_next:
            if stripped:
                lead = lead_len(line)
                mark(start + lead, start + lead + len(stripped))
                promote_next = False
            continue

        if not stripped:
            continue

        if reread_skip_target:
            reread_skip_target = False
            reread_mark_word = True
            continue

        if reread_mark_word:
            lead = lead_len(line)
            mark(start + lead, start + len(line.rstrip()))
            reread_mark_word = False
            continue

        if stripped in _VERDICT_LINES:
            lead = lead_len(line)
            mark(start + lead, start + lead + len(stripped))
            continue

        if _CANDIDATE_COUNT_RE.match(stripped):
            if _MULTI_CANDIDATE_COUNT_RE.match(stripped):
                lead = lead_len(line)
                mark(start + lead, start + lead + len(stripped))
            continue

        if stripped.startswith(_CHECK_OPENERS):
            if stripped.startswith("for no candidate,"):
                in_target_decode = True
                in_candidate_check = False
            elif stripped.startswith(
                (
                    "for one candidate,",
                    "for multiple candidates,",
                    "for more than one candidate,",
                )
            ):
                in_target_decode = False
                in_candidate_check = True
                if not stripped.startswith("for one candidate,"):
                    lead = lead_len(line)
                    mark(start + lead, start + lead + len(stripped))
            continue

        if stripped.startswith("same length and known letters match "):
            idx = line.find("same length and known letters match ") + len(
                "same length and known letters match "
            )
            mark(start + idx, start + len(line.rstrip()))
            continue

        if stripped.startswith("letters ") and (
            in_process_examples or in_target_decode or in_candidate_check
        ):
            idx = line.find("letters ") + len("letters ")
            mark(start + idx, start + len(line.rstrip()))
            continue

        if stripped.startswith("choose "):
            idx = line.find("choose ") + len("choose ")
            mark(start + idx, start + len(line.rstrip()))
            continue

        if stripped.startswith("add ") and " -> " in stripped:
            lead = lead_len(line)
            mark(start + lead, start + lead + len(stripped))
            continue

        if "\\boxed{" in line:
            continue

        m = _ARROW_RE.match(line)
        if m:
            lead = len(m.group(1))
            lhs, rhs = m.group(2), m.group(3)
            rhs_start = start + lead + len(lhs) + len(" -> ")
            rhs_end = rhs_start + len(rhs)

            if in_summary_mappings:
                mark(start + lead, start + len(line.rstrip()))
                continue

            if in_process_examples and " " not in lhs and len(lhs) > 1:
                mark(start + lead, start + len(line.rstrip()))
                continue

            # single-letter LHS -> mapping / decode / check line
            if len(lhs) == 1 and lhs.isalpha():
                if in_target_decode:
                    mark(rhs_start, rhs_end)
                elif in_candidate_check and not rhs.endswith(" agrees"):
                    mark(rhs_start, rhs_end)
                continue

            # word LHS -> scan verdict, partial, or echo
            if in_candidate_check and " " not in lhs:
                mark(start + lead, start + len(line.rstrip()))
                continue

            verdict = _SCAN_VERDICT_RE.match(rhs)
            if verdict:
                if verdict.group(2) == "match":
                    vstart = rhs_start + len(verdict.group(1)) + 1
                    mark(vstart, rhs_end)
                continue

            nxt = ""
            for j in range(i + 1, n):
                if lines[j][1].strip():
                    nxt = lines[j][1].strip()
                    break
            if nxt in ("fully mapped", "not fully mapped"):
                mark(rhs_start, rhs_end)  # the assembled pattern
            elif in_candidate_check:
                mark(rhs_start, rhs_end)
            continue

    return weights


def token_weights_from_offsets(
    offset_mapping,
    char_weights: list[float],
    *,
    char_base: int = 0,
    prompt_token_count: int = 0,
    base: float = BASE,
) -> list[float]:
    """Aggregate per-character weights to per-token weights (max over span).

    offset_mapping: list of (start, end) char spans in the SAME string the
      char_weights were built over (pass char_base if char_weights covers only a
      substring whose first char sits at absolute index char_base).
    prompt_token_count: leading tokens to force to 0.0 (masked prompt).
    """
    out: list[float] = []
    L = len(char_weights)
    for idx, (s, e) in enumerate(offset_mapping):
        if idx < prompt_token_count:
            out.append(0.0)
            continue
        cs, ce = s - char_base, e - char_base
        if ce <= cs or cs >= L or ce <= 0:
            out.append(base)
            continue
        seg = char_weights[max(cs, 0):min(ce, L)]
        out.append(max(seg) if seg else base)
    return out


def completion_label_weights(
    tokenizer,
    prompt_text: str,
    completion_text: str,
    *,
    high: float = HIGH,
    base: float = BASE,
) -> list[float]:
    """Weights aligned 1:1 with the joint tokenization of prompt+completion.

    Matches train_sft_single_phase.tokenize_masked_example: full_text =
    prompt_text + completion_text, prompt tokens are an exact token prefix.
    Prompt-region tokens get 0.0 (they are -100 in labels anyway).
    """
    full_text = prompt_text + completion_text
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    prompt_token_count = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    char_weights = [base] * len(full_text)
    comp = build_char_weights(completion_text, high=high, base=base)
    pbase = len(prompt_text)
    for i, w in enumerate(comp):
        char_weights[pbase + i] = w

    return token_weights_from_offsets(
        enc["offset_mapping"],
        char_weights,
        prompt_token_count=prompt_token_count,
        base=base,
    )


def _main(argv: list[str] | None = None) -> int:
    """Inspect the weighting on a completion: prints weight-2 tokens.

    Usage:
        PYTHONPATH=src python3 -m nemotron_baseline.text_cipher_loss_weights [completion.txt] [--tokenizer PATH]
    With no file argument a built-in sample completion is used.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("completion", nargs="?", help="path to a completion/target text file")
    parser.add_argument(
        "--tokenizer",
        default=str(
            Path(__file__).resolve().parents[2]
            / "reference/winner-solution/nemotron-master-huikang/tokenizer.json"
        ),
        help="path to a tokenizer.json (HuggingFace tokenizers format)",
    )
    args = parser.parse_args(argv)

    if args.completion:
        text = Path(args.completion).read_text(encoding="utf-8")
    else:
        text = (
            "aivvjc\n"
            "a -> p\ni -> ? unknown\nv -> ? unknown\nv -> ? unknown\nj -> l\nc -> e\n"
            "aivvjc -> p???le\nnot fully mapped\n\n"
            "Vocab pattern scan\nsame length and known letters match p???le\n"
            "palace -> p???le no\npuzzle -> p???le match\n\n"
            "Scan candidates for p???le\npuzzle\n\n"
            "Check scan candidates against summary mappings\n\n"
            "aivvjc -> puzzle\na -> p agrees\ni -> u new\nv -> z new\nv -> z agrees\n"
            "j -> l agrees\nc -> e agrees\nPASS\nadd i -> u\nadd v -> z\n\n"
            "choose puzzle\n\nDecoded phrase\npuzzle\n\nAnswer: \\boxed{puzzle}"
        )

    char_weights = build_char_weights(text)

    try:
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(args.tokenizer)
        enc = tok.encode(text, add_special_tokens=False)
        weights = token_weights_from_offsets(enc.offsets, char_weights)
        n2 = sum(1 for w in weights if w >= HIGH)
        print(f"tokens={len(weights)} weight2={n2} weight1={len(weights) - n2}")
        print("weight-2 tokens:")
        print("  " + " ".join(repr(t) for t, w in zip(enc.tokens, weights) if w >= HIGH))
    except Exception as exc:  # tokenizer unavailable -> fall back to char spans
        print(f"(tokenizer unavailable: {exc}; showing weight-2 character spans)")
        spans, i = [], 0
        while i < len(text):
            if char_weights[i] >= HIGH:
                j = i
                while j < len(text) and char_weights[j] >= HIGH:
                    j += 1
                spans.append(text[i:j])
                i = j
            else:
                i += 1
        print("  " + " | ".join(s for s in spans if s.strip()))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
