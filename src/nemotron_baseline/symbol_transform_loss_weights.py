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
  direct-template routing payload  query value/operator and compared operator symbols
  direct-template computation      the right-hand side of `AB = ...` / `CD = ...`,
                                   the operator symbol, and the concrete
                                   `followed by ... gives ... vs ...` line
  operator absence                 `None` and the fallback `template0134` token only
  encrypted-digit routing          `Try BA_DC...` / `Try AB_CD...` / `Try x*y...`,
                                   `The current format is ...`, `Apply format...`,
                                   `Use helper row ...`, `The helper operator ... so try ...`
  arithmetic/query computation     the produced value after `gives` (before ` vs`)
  match verdicts                   `Match` / `No match` / `... passes all examples` / `... fails`
  digit derivations                `<sym> -> <reversed/coeff form>`, lines with `mod 10`,
                                   formula lines like `i=(19h+b)/9`
  scan survivors                   `C<k>` / `T<k>` candidate labels (not the `x` rejects)
  candidate reasoning              `First helper candidates ...`, `Only C4 can pass ...`,
                                   `For C4, b=8,h=1,i=3. ...`, `FAIL` / `PASS`
  final map + answer               `<sym> = <letter> = <digit>`, the boxed
                                   value on the in-think `Answer:` line

Everything else (preamble, `Query ...`, `Compare example operators`, echoed
example lines, template routing boilerplate such as `The RHS values have length`
and `Try template0134`, `Assign global variables`, `? = a` naming, section
headers, and the `x` reject entries / numeric scan values) stays weight 1.0.
"""
from __future__ import annotations

import re

from .text_cipher_loss_weights import token_weights_from_offsets, HIGH, BASE

_WHOLE_LINE_RES = [
    re.compile(r"^(same|different|Match|No match|FAIL|PASS)$"),
    re.compile(r"^Try (BA_DC|AB_CD|x\*y|x\+y|x-y|y-x)"),
    re.compile(r"^Apply format"),
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
]
_SHORT_FAIL_RE = re.compile(r"\bfails?\b")
_GIVES_RE = re.compile(r" gives (.+?)(?: vs .+)?$")
_QUERY_RE = re.compile(r"^Query (.+)$")
_QUERY_OPERATOR_RE = re.compile(r"^Query operator is (.+)$")
_COMPARE_OPERATOR_RE = re.compile(r"^.+ = .+ operator (\S+)$")
_DIRECT_TEMPLATE_ASSIGN_RE = re.compile(r"^(AB|CD) = (.+)$")
_OPERATOR_ASSIGN_RE = re.compile(r"^operator = (.+)$")
_FOLLOWED_GIVES_RE = re.compile(r" followed by .+ gives .+")
_OP_ABSENCE_TEMPLATE_RE = re.compile(
    r"^For symbol equation transformation rules with no same query operator example, "
    r"use direct template matching with (template0134)$"
)
_ARROW_RE = re.compile(r"^\s*\S+ -> (.+?)\s*$")
_MAP_RE = re.compile(r"^\s*\S+ = [a-z] = (\d+)\s*$")
_CAND_RE = re.compile(r"(?<![A-Za-z0-9])[CT]\d+(?![A-Za-z0-9])")
_ANSWER_BOXED_RE = re.compile(r"^Answer:\s*(\\boxed\{.*)$")


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

        # Box only the in-think answer line. The bare boxed echo appended after
        # </think> stays base-weight.
        answer_box = _ANSWER_BOXED_RE.match(stripped)
        if answer_box:
            mark(body_start + answer_box.start(1), body_start + answer_box.end(1))
            continue
        if "\\boxed{" in line:
            continue

        # final map line: weight the digit only
        m = _MAP_RE.match(line)
        if m:
            mark(start + m.start(1), start + m.end(1))
            continue

        # Direct-template routing payload: query value/operator and the
        # operator symbols being compared. Keep the surrounding prose flat.
        qo = _QUERY_OPERATOR_RE.match(stripped)
        if qo:
            mark(body_start + qo.start(1), body_start + qo.end(1))
            continue
        q = _QUERY_RE.match(stripped)
        if q:
            mark(body_start + q.start(1), body_start + q.end(1))
            continue
        co = _COMPARE_OPERATOR_RE.search(stripped)
        if co:
            mark(body_start + co.start(1), body_start + co.end(1))
            continue

        # Direct-template parse payload. Weight only the AB/CD values, and only
        # the operator symbol on the operator line.
        dt = _DIRECT_TEMPLATE_ASSIGN_RE.match(stripped)
        if dt:
            mark(body_start + dt.start(2), body_start + dt.end(2))
            continue
        op = _OPERATOR_ASSIGN_RE.match(stripped)
        if op:
            mark(body_start + op.start(1), body_start + op.end(1))
            continue

        # Direct-template computation line. This is the actual rule application,
        # unlike the generic `Try template...` scaffolding above it.
        if _FOLLOWED_GIVES_RE.search(stripped):
            mark(body_start, body_end)
            continue

        op_abs = _OP_ABSENCE_TEMPLATE_RE.match(stripped)
        if op_abs:
            mark(body_start + op_abs.start(1), body_start + op_abs.end(1))
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
