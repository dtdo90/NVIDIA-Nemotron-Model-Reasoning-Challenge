"""Token-level loss weighting for Symbol Transform SFT traces.

Same idea as text_cipher_loss_weights: decision points and failure-prone spans
get weight 2.0, routine boilerplate/echoes/reject entries stay 1.0. The weight
lands on the predicted decision tokens, never on echoed left-hand sides.

Symbol Transform traces use two solve paths that share a routing skeleton:
  - direct template (template0134 / template3401)
  - encrypted digit search (BA_DC|rev, AB_CD|plain) with per-row coefficient
    expansions, mod-10 unit constraints, and helper scan tables.

Weight-2 spans:
  operator-compare verdicts        `same` / `different`
  family routing                   `The RHS values have length ...`, `RHS length N means ...`,
                                   `Try template0134,...` / `Try x*y,...`
  template/format selection        `Try template0134`, `Try BA_DC|rev with x*y ...`,
                                   `The current format is ...`, `Apply template... to the query`,
                                   `Use helper row ...`, `The helper operator ... so try ...`
  template/query computation       the produced value after `gives` (before ` vs`)
  match verdicts                   `Match` / `No match` / `... passes all examples` / `... fails`
  digit derivations                `<sym> -> <reversed/coeff form>`, lines with `mod 10`,
                                   formula lines like `i=(19h+b)/9`
  scan survivors                   `C<k>` / `T<k>` candidate labels (not the `x` rejects)
  candidate reasoning              `First helper candidates ...`, `Only C4 can pass ...`,
                                   `For C4, b=8,h=1,i=3. ...`, `FAIL` / `PASS`
  final map + answer               `<sym> = <letter> = <digit>`, `\\boxed{...}`

Everything else (preamble, `Query ...`, `Compare example operators`, echoed
example lines, `AB = ...`/`operator = ...`/`CD = ...` breakdowns, `Assign global
variables`, `? = a` naming, section headers, and the `x` reject entries / numeric
scan values) stays weight 1.0.
"""
from __future__ import annotations

import re

from .text_cipher_loss_weights import token_weights_from_offsets, HIGH, BASE

_WHOLE_LINE_RES = [
    re.compile(r"^(same|different|Match|No match|FAIL|PASS)$"),
    re.compile(r"passes all examples$"),
    re.compile(r"^The RHS values have length"),
    re.compile(r"^RHS length \d"),
    re.compile(r"^Try (template|BA_DC|AB_CD|x\*y|x\+y|x-y|y-x)"),
    re.compile(r"^Apply (template|format)"),
    re.compile(r"^The current format is"),
    re.compile(r"^Use helper row"),
    re.compile(r"^The helper operator .* so try"),
    re.compile(r"^First helper candidates"),
    re.compile(r"^Only C\d+"),
    re.compile(r"^For C\d+,"),
    re.compile(r"mod 10$"),                       # unit-digit constraints
    re.compile(r"^[a-z]\w*\s*=\s*\(.*\)\s*/\s*\d+$"),  # formula lines i=(19h+b)/9
    # operator-absence routing (cautious default; intentionally may not hit gold)
    re.compile(r"^None$"),
    re.compile(r"no same query operator example"),
    re.compile(r"use direct template matching"),
]
_SHORT_FAIL_RE = re.compile(r"\bfails?\b")
_GIVES_RE = re.compile(r" gives (.+?)(?: vs .+)?$")
_ARROW_RE = re.compile(r"^\s*\S+ -> (.+?)\s*$")
_MAP_RE = re.compile(r"^\s*\S+ = [a-z] = (\d+)\s*$")
_CAND_RE = re.compile(r"(?<![A-Za-z0-9])[CT]\d+(?![A-Za-z0-9])")
_BOXED = "\\boxed{"


def build_char_weights(text: str, *, high: float = HIGH, base: float = BASE) -> list[float]:
    weights = [base] * len(text)

    lines: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        lines.append((offset, line))
        offset += len(line) + 1

    def mark(start: int, end: int) -> None:
        for k in range(max(start, 0), min(end, len(weights))):
            weights[k] = high

    for start, line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lead = len(line) - len(line.lstrip())
        body_start = start + lead
        body_end = start + len(line.rstrip())

        # whole-line decision/verdict/routing patterns
        if any(rx.search(stripped) for rx in _WHOLE_LINE_RES):
            mark(body_start, body_end)
            continue
        if len(stripped) <= 45 and _SHORT_FAIL_RE.search(stripped):
            mark(body_start, body_end)
            continue

        # boxed answer
        if _BOXED in line:
            mark(start + line.find(_BOXED), body_end)
            continue

        # final map line: weight the digit only
        m = _MAP_RE.match(line)
        if m:
            mark(start + m.start(1), start + m.end(1))
            continue

        # "... gives <produced>[ vs <target>]" -> weight the produced value
        g = _GIVES_RE.search(line)
        if g:
            mark(start + g.start(1), start + g.end(1))
            # also mark inline candidate labels if any, then done
            for cm in _CAND_RE.finditer(line):
                mark(start + cm.start(), start + cm.end())
            continue

        # "<sym> -> <reversed / coefficient form>" -> weight the right side
        a = _ARROW_RE.match(line)
        if a:
            mark(start + a.start(1), start + a.end(1))
            continue

        # otherwise: weight only inline candidate/survivor labels (scan tables, prose)
        for cm in _CAND_RE.finditer(line):
            mark(start + cm.start(), start + cm.end())

    return weights


def completion_label_weights(
    tokenizer,
    prompt_text: str,
    completion_text: str,
    *,
    high: float = HIGH,
    base: float = BASE,
) -> list[float]:
    """Weights aligned 1:1 with the joint tokenization of prompt+completion.
    Prompt-region tokens get 0.0 (masked by -100 in labels)."""
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
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Inspect Symbol Transform decision weighting.")
    parser.add_argument("completion", help="path to a completion/target text file")
    parser.add_argument(
        "--tokenizer",
        default=str(
            Path(__file__).resolve().parents[2]
            / "reference/winner-solution/nemotron-master-huikang/tokenizer.json"
        ),
    )
    parser.add_argument("--show-text", action="store_true", help="print trace with weight-2 spans bracketed")
    args = parser.parse_args(argv)
    text = Path(args.completion).read_text(encoding="utf-8")
    cw = build_char_weights(text)
    if args.show_text:
        out, i, n = [], 0, len(text)
        while i < n:
            if cw[i] >= HIGH:
                j = i
                while j < n and cw[j] >= HIGH:
                    j += 1
                out.append("⟦" + text[i:j] + "⟧")
                i = j
            else:
                out.append(text[i])
                i += 1
        print("".join(out))
        return 0
    try:
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(args.tokenizer)
        enc = tok.encode(text, add_special_tokens=False)
        weights = token_weights_from_offsets(enc.offsets, cw)
        n2 = sum(1 for w in weights if w >= HIGH)
        print(f"tokens={len(weights)} weight2={n2} weight1={len(weights) - n2}")
    except Exception as exc:
        print(f"(tokenizer unavailable: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
