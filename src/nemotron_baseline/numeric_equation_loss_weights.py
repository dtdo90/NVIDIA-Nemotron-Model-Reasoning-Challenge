"""Token-level loss weighting for Numeric Equation Transformation traces.

Numeric Equation traces are long because they replay candidate families across
example rows. The high-weight spans should therefore land on decisions and
fragile policy outputs, not on every repeated arithmetic cell.

Weight-2 spans:
  - operator comparison verdicts: ``same`` / ``different``
  - same-operator branch counts: ``none``, ``one example``, ``four examples``
  - RHS-length routing and candidate-family lists
  - computed numeric rows under BA/DC or AB/CD output-format tables
  - the concrete family/format in supported and selected formats
  - surviving common formats, ``none`` failures, and failure summaries
  - motif-drift decisions: single-example guard, helper choice, confirm/reject,
    return to the supported same-operator format
  - direct-template produced value after ``gives`` and template pass/apply lines
  - operator-absence policy: query-operator PASS row, candidate families,
    output vote counts, tie rule, and chosen candidate
  - final query value rows, common-output/voting/prefix/suffix policy lines,
    and the boxed answer

Routine boilerplate, echoed examples, AB/CD breakdown lines, and repeated
scaffold lines such as ``Try BA_DC first``, ``The current format is ...``,
``Match``, and ``Common`` stay at weight 1.0. When a scaffold line contains a
decision format such as ``BA_DC|x-y|common``, only that concrete format is
promoted.
"""
from __future__ import annotations

import re

from .text_cipher_loss_weights import token_weights_from_offsets, HIGH, BASE


_COUNT_LINE_RE = re.compile(
    r"^(one|two|three|four|five|six|seven|eight|nine|ten|\d+) examples?$"
)
_WHOLE_LINE_RES = [
    re.compile(r"^(same|different|PASS|FAIL|none)$"),
    _COUNT_LINE_RE,
    re.compile(r"^The RHS values (have|mix)"),
    re.compile(r"^RHS length \d"),
    re.compile(r"^For direct templates,"),
    re.compile(r"^Direct templates fail"),
    re.compile(r"^We try direct templates first"),
    re.compile(r"^(template0134|template3401) (passes|fails|supports|does not preserve)"),
    re.compile(r"^(x\+y|x\+y-1|x\+y\+1|x-y|y-x|abs\(x-y\)|min\(x,y\)-max\(x,y\)|max\(x,y\)%min\(x,y\)|x%y|y%x|x\*y|x\*y\+1|x\*y-1) (fails|does not preserve)"),
    re.compile(r"^No (BA_DC|AB_CD) candidate supports"),
    re.compile(r"^Only one same operator row supports"),
    re.compile(r"^Verify motif BA_DC"),
    re.compile(r"^Helper operator groups$"),
    re.compile(r"^Choose the helper operator group"),
    re.compile(r"^The motif BA_DC is (supported|not supported)"),
    re.compile(r"^So BA_DC is (confirmed|rejected)"),
    re.compile(r"^Return to the supported same operator format"),
    re.compile(r"^All visible operator groups support"),
    re.compile(r"^Use symbol mapping for query operator"),
    re.compile(r"^Candidate operator families for "),
    re.compile(r"^Operator families used by the visible examples"),
    re.compile(r"^Search remaining candidate operator families"),
    re.compile(r"^Remaining candidate operator families$"),
    re.compile(r"^Check remaining candidate operator families"),
    re.compile(r"^Candidate [^ ]+$"),
    re.compile(r"^.+ has \d+ votes$"),
    re.compile(r"^For motif (BA_DC|AB_CD)"),
    re.compile(r"^The selected operator"),
]
_GIVES_RE = re.compile(r" gives (.+?)(?: vs .+)?$")
_BOXED = "\\boxed{"
_QUERY_OPERATOR_RE = re.compile(r"^Use symbol mapping for query operator (.+)$")
_MAPPING_ROW_RE = re.compile(r"^(.+?) -> .+ (PASS|no)$")
_FORMAT_NAME_RE = re.compile(
    r"^(rev|plain|op_prefix_if_neg|rev_or_op_prefix_rev_if_neg|"
    r"rev_or_op_suffix_if_neg|rev_or_op_suffix_rev_if_neg|op_prefix|op_suffix|"
    r"op_prefix_rev|abs_rev|abs|BA_DC|AB_CD|"
    r"template0134|template3401|x\+y|x\+y-1|x\+y\+1|x-y|y-x|"
    r"abs\(x-y\)|min\(x,y\)-max\(x,y\)|max\(x,y\)%min\(x,y\)|"
    r"x%y|y%x|x\*y|x\*y\+1|x\*y-1)$"
)
_QUERY_VALUE_SKIP_RE = re.compile(
    r"^(BA|AB|CD|DC|operator\b|plain\b|rev\b|template|abs\(|max\(|min\(|"
    r"[A-Z]{2}\b)"
)
_FORMAT_TABLE_HEADER_RE = re.compile(r"^(BA DC|AB CD) .*(rev|plain|op_prefix|abs|template)")


def build_char_weights(text: str, *, high: float = HIGH, base: float = BASE) -> list[float]:
    weights = [base] * len(text)
    rare_policy_trace = bool(
        re.search(
            r"For motif (?:"
            r"BA_DC with (?:x-y negative|x-y positive|y-x negative|selected operator neither x-y nor y-x)|"
            r"AB_CD with (?:x-y negative|y-x negative|min\(x,y\)-max\(x,y\) negative)|"
            r"AB_CD, use first common output format"
            r")",
            text,
        )
    )

    lines: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        lines.append((offset, line))
        offset += len(line) + 1

    def mark(start: int, end: int) -> None:
        for k in range(max(start, 0), min(end, len(weights))):
            weights[k] = high

    def mark_line(start: int, line: str) -> None:
        lead = len(line) - len(line.lstrip())
        mark(start + lead, start + len(line.rstrip()))

    promote_list = False
    promote_next_value = False
    query_operator_for_mapping: str | None = None
    in_symbol_mapping = False
    query_block_remaining = 0
    direct_query_block = False
    promote_table_row = False

    for start, line in lines:
        stripped = line.strip()
        if not stripped:
            promote_list = False
            promote_next_value = False
            query_block_remaining = 0
            direct_query_block = False
            promote_table_row = False
            if in_symbol_mapping:
                in_symbol_mapping = False
            continue

        # Final answer.
        if _BOXED in line:
            mark(start + line.find(_BOXED), start + len(line.rstrip()))
            continue

        if promote_next_value:
            mark_line(start, line)
            promote_next_value = False
            continue

        if promote_table_row:
            mark_line(start, line)
            promote_table_row = False
            continue

        if _FORMAT_TABLE_HEADER_RE.match(stripped):
            promote_table_row = True
            continue

        # Query blocks are the single most answer-sensitive table. Weight the
        # computed output row(s), but skip the echoed query expression and table
        # header lines.
        if stripped == "Query":
            query_block_remaining = 8
            direct_query_block = False
            continue
        if query_block_remaining:
            query_block_remaining -= 1
            if " -> " in stripped:
                rhs_start = line.find(" -> ") + len(" -> ")
                mark(start + rhs_start, start + len(line.rstrip()))
                direct_query_block = True
                continue
            if direct_query_block:
                mark_line(start, line)
                continue
            if _QUERY_VALUE_SKIP_RE.search(stripped):
                continue
            if re.search(r"\d", stripped):
                mark_line(start, line)
                continue

        # Repeated scaffold phrases are easy and stay base-weight.
        if stripped in {"Match", "No match", "Try BA_DC first", "Try AB_CD"}:
            continue
        if stripped.startswith(("Try BA_DC with ", "Try AB_CD with ", "Try visible operator ")):
            continue
        if stripped.startswith("Try "):
            idx = line.find("Try ") + len("Try ")
            mark(start + idx, start + len(line.rstrip()))
            continue
        if stripped.startswith("First try visible operator "):
            idx = line.find("First try visible operator ") + len("First try visible operator ")
            mark(start + idx, start + len(line.rstrip()))
            continue
        if stripped.startswith("The current format is "):
            continue
        if stripped.startswith("Apply format "):
            idx = line.find("Apply format ") + len("Apply format ")
            end = line.find(" to the query")
            mark(start + idx, start + (end if end != -1 else len(line.rstrip())))
            continue
        if stripped.startswith("Apply template"):
            idx = line.find("Apply ") + len("Apply ")
            end = line.find(" to the query")
            mark(start + idx, start + (end if end != -1 else len(line.rstrip())))
            continue
        if stripped.startswith("Candidate ") and len(stripped.split()) == 2:
            idx = line.find("Candidate ") + len("Candidate ")
            mark(start + idx, start + len(line.rstrip()))
            continue
        if stripped.startswith("All common output formats agree on "):
            idx = line.find("All common output formats agree on ") + len("All common output formats agree on ")
            mark(start + idx, start + len(line.rstrip()))
            continue
        if stripped == "All common output formats do not agree":
            idx = line.find("do not agree")
            mark(start + idx, start + len(line.rstrip()))
            continue

        if stripped in {"Highest vote count", "Choose operator candidate with highest output agreement vote count, if tie choose the first candidate"}:
            promote_next_value = True
            continue

        # Section-list values after these headers are decisions: surviving
        # common formats, candidate families, and output formats. The header
        # itself stays base weight.
        if promote_list and _FORMAT_NAME_RE.match(stripped):
            mark_line(start, line)
            continue

        if stripped in {
            "Common",
            "Updated surviving common formats",
            "Surviving common formats",
            "Candidate output formats",
            "Motif",
            "Remaining candidate operator families",
            "Operator families used by the visible examples",
            "Output votes",
        } or stripped.startswith("Candidate operator families for "):
            promote_list = True
            continue

        # Operator-absence symbol mapping: only the query operator row is the
        # decision; the other 25 rows are reference context.
        opm = _QUERY_OPERATOR_RE.match(stripped)
        if opm:
            query_operator_for_mapping = opm.group(1)
            in_symbol_mapping = True
            mark_line(start, line)
            continue
        if in_symbol_mapping:
            mm = _MAPPING_ROW_RE.match(stripped)
            if mm and query_operator_for_mapping is not None and mm.group(1) == query_operator_for_mapping:
                mark_line(start, line)
            continue

        # Direct template: weight the produced value, not the echoed operands.
        g = _GIVES_RE.search(line)
        if g:
            mark(start + g.start(1), start + g.end(1))
            continue

        if stripped.startswith("The format ") and " supports" in stripped:
            idx = line.find("The format ") + len("The format ")
            end = line.find(" supports", idx)
            mark(start + idx, start + (end if end != -1 else len(line.rstrip())))
            continue

        if any(rx.search(stripped) for rx in _WHOLE_LINE_RES):
            mark_line(start, line)
            continue

    return weights


def completion_label_weights(
    tokenizer,
    prompt_text: str,
    completion_text: str,
    *,
    high: float = HIGH,
    base: float = BASE,
) -> list[float]:
    """Weights aligned 1:1 with prompt+completion tokenization."""
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

    parser = argparse.ArgumentParser(description="Inspect Numeric Equation decision weighting.")
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
