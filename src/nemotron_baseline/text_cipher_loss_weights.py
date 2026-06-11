"""Token-level loss weighting for Text Cipher SFT traces.

Decision points and empirically failure-prone spans get weight 2.0; every other
token (boilerplate, echoes, the 77-word vocab dump, the routine `no` scan lines,
and `agrees` confirmations) stays at weight 1.0. No down-weighting.

The weight is placed on the *predicted* right-hand-side / verdict tokens of each
line, not on echoed left-hand sides, so the gradient lands on the actual
decision rather than on copies of the cipher word.

Weight-2 line patterns (see audit):
  P1  `<c> -> <p> new`                  bind a mapping (alignment failures)
  P2  `<c> -> <p>` under Summary ...     authoritative map
  P5  `<c> -> <p>` / `<c> -> ? unknown`  per-char decode (slot drops)
      `<word> -> <partial>` (before fully/not fully mapped)   the pattern
      `fully mapped` / `not fully mapped`                     branch
  P6  `<word> -> <partial> match`        survives pattern filter
      line after `Scan candidates for ...`                    candidate set
  P7  `<c> -> <p> conflicts` / `<c> -> <p> new` / `... already exists/used`
      `PASS` / `FAIL`                                          verdict
      `add <c> -> <p>`                                         commit mapping
  P8  `choose <word>` / line after `Decoded phrase` / `\\boxed{...}`

Everything else (including `<c> -> <p> agrees` and `<word> -> <partial> no`)
stays weight 1.0.
"""
from __future__ import annotations

import re

HIGH = 2.0
BASE = 1.0

_VERDICT_LINES = {"fully mapped", "not fully mapped", "PASS", "FAIL"}
_PROMOTE_NEXT_PREFIXES = ("Scan candidates for ",)
_PROMOTE_NEXT_EXACT = {"Decoded phrase"}
_ARROW_RE = re.compile(r"^(\s*)(\S+) -> (.+?)\s*$")
_SCAN_VERDICT_RE = re.compile(r"^(.+) (no|match)$")


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
    for i, (start, line) in enumerate(lines):
        stripped = line.strip()

        if promote_next:
            if stripped:
                lead = lead_len(line)
                mark(start + lead, start + lead + len(stripped))
                promote_next = False
            continue

        if not stripped:
            continue

        if stripped in _PROMOTE_NEXT_EXACT or stripped.startswith(_PROMOTE_NEXT_PREFIXES):
            promote_next = True
            continue

        if stripped in _VERDICT_LINES:
            lead = lead_len(line)
            mark(start + lead, start + lead + len(stripped))
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
            idx = line.find("\\boxed{")
            mark(start + idx, start + len(line.rstrip()))
            continue

        m = _ARROW_RE.match(line)
        if m:
            lead = len(m.group(1))
            lhs, rhs = m.group(2), m.group(3)
            rhs_start = start + lead + len(lhs) + len(" -> ")
            rhs_end = rhs_start + len(rhs)

            # single-letter LHS -> mapping / decode / check line
            if len(lhs) == 1 and lhs.isalpha():
                if not rhs.endswith(" agrees"):  # 'agrees' stays weight 1
                    mark(rhs_start, rhs_end)
                continue

            # word LHS -> scan verdict, partial, or echo
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
