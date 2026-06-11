"""Token-level loss weighting for Numeric Equation Transformation traces.

Numeric Equation traces are long because they replay candidate families across
example rows. The high-weight spans should therefore land on decisions and
fragile policy outputs, not on every repeated arithmetic cell.

Important spans:
  - operator comparison verdicts: ``same`` / ``different``
  - same-operator branch counts: ``none``, ``one example``, ``four examples``
  - RHS-length routing payloads that select the candidate arithmetic families
  - the concrete family/format in supported and selected formats
  - surviving common formats, ``none`` failures, and failure summaries
  - motif-drift decisions: single-example guard, helper choice, confirm/reject,
    return to the supported same-operator format
  - direct-template produced value after ``gives`` and template pass/apply lines
  - operator-absence policy: query-operator PASS row, candidate families,
    output vote counts, tie rule, and chosen candidate
  - successful decision table rows that feed surviving output formats
  - final query value rows, common-output/voting/prefix/suffix policy lines,
    and answer values

Routine boilerplate, echoed examples, AB/CD breakdown lines, family-list
scaffolding, failing-candidate rows, and repeated scaffold lines such as
``Try BA_DC first``, ``The current format is ...``, ``Match``, and ``Common``
stay at weight 1.0. When a scaffold line contains a decision payload such as
``BA_DC|x-y|common`` or ``mix length 1 and 2, so use subtraction or modular``,
only that payload is promoted.

Tiering: ``high`` marks genuine turning points. Rare answer-critical outputs
use ``base + 2 * (high - base)``; with the default high=2.0 this gives weight
3.0 for final query output rows, rare policy lines, and vote counts/winners.
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
    re.compile(r"^No (BA_DC|AB_CD) candidate supports"),
    re.compile(r"^All visible operator groups support"),
    re.compile(r"^Search remaining candidate operator families"),
    re.compile(r"^Check remaining candidate operator families"),
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
_VOTE_LINE_RE = re.compile(r"^.+ has \d+ votes$")
_TEMPLATE_RESULT_RE = re.compile(r"^(template0134|template3401) (passes|supports)")
_TABLE_HEADER_RE = re.compile(r"^(BA DC|AB CD)\b")
_FORMAT_SPAN_RE = re.compile(r"(BA_DC|AB_CD)\|[^ ]+\|common")
_HELPER_EQUATION_RE = re.compile(r"^\d{2}[^ \d]\d{2} = .+$")


def build_char_weights(text: str, *, high: float = HIGH, base: float = BASE) -> list[float]:
    weights = [base] * len(text)
    critical = base + 2 * (high - base)

    lines: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        lines.append((offset, line))
        offset += len(line) + 1

    def mark(start: int, end: int, weight: float = high) -> None:
        for k in range(max(start, 0), min(end, len(weights))):
            weights[k] = max(weights[k], weight)

    def mark_line(start: int, line: str, weight: float = high) -> None:
        lead = len(line) - len(line.lstrip())
        mark(start + lead, start + len(line.rstrip()), weight)

    promote_list = False
    promote_next_value = False
    promote_next_weight = high
    query_operator_for_mapping: str | None = None
    in_symbol_mapping = False
    query_block_remaining = 0
    query_expression_seen = False
    direct_query_block = False
    last_table_row: tuple[int, str] | None = None
    match_table_row: tuple[int, str] | None = None
    in_match_format_list = False
    expect_table_row = False
    promote_helper_equations = False
    next_query_weight = critical
    active_query_weight = critical

    for start, line in lines:
        stripped = line.strip()
        if not stripped:
            promote_list = False
            promote_next_value = False
            promote_next_weight = high
            query_block_remaining = 0
            query_expression_seen = False
            direct_query_block = False
            last_table_row = None
            match_table_row = None
            in_match_format_list = False
            expect_table_row = False
            promote_helper_equations = False
            next_query_weight = critical
            if in_symbol_mapping:
                in_symbol_mapping = False
            continue

        # Final answer.
        if _BOXED in line:
            continue

        if promote_next_value:
            mark_line(start, line, promote_next_weight)
            promote_next_value = False
            promote_next_weight = high
            continue

        if expect_table_row:
            last_table_row = (start, line)
            expect_table_row = False
            continue

        # Query blocks are the single most answer-sensitive table. Weight the
        # computed output row(s), but skip the echoed query expression and table
        # header lines.
        if stripped == "Query":
            query_block_remaining = 8
            query_expression_seen = False
            direct_query_block = False
            active_query_weight = next_query_weight
            next_query_weight = critical
            continue
        if query_block_remaining:
            query_block_remaining -= 1
            if " -> " in stripped:
                rhs_start = line.find(" -> ") + len(" -> ")
                mark(start + rhs_start, start + len(line.rstrip()), active_query_weight)
                direct_query_block = True
                continue
            if direct_query_block:
                mark_line(start, line, active_query_weight)
                continue
            if _QUERY_VALUE_SKIP_RE.search(stripped):
                continue
            if not query_expression_seen and re.search(r"\d", stripped):
                query_expression_seen = True
                continue
            if re.search(r"\d", stripped):
                mark_line(start, line, active_query_weight)
                query_block_remaining = 0
                continue

        # Repeated scaffold phrases are easy and stay base-weight.
        if stripped == "Match":
            match_table_row = last_table_row
            in_match_format_list = True
            continue
        if stripped in {"No match", "Try BA_DC first", "Try AB_CD"}:
            continue
        if _TABLE_HEADER_RE.match(stripped):
            expect_table_row = True
            continue
        if stripped.startswith(("Try BA_DC with ", "Try AB_CD with ", "Try visible operator ")):
            continue
        if stripped.startswith("Try "):
            continue
        if stripped.startswith("First try visible operator "):
            continue
        if stripped.startswith("The RHS values "):
            idx = line.find("The RHS values ") + len("The RHS values ")
            mark(start + idx, start + len(line.rstrip()))
            continue
        if stripped.startswith("RHS length "):
            mark_line(start, line)
            continue
        if stripped.startswith("The current format is "):
            continue
        if stripped.startswith("Apply format "):
            idx = line.find("Apply format ") + len("Apply format ")
            end = line.find(" to the query")
            mark(start + idx, start + (end if end != -1 else len(line.rstrip())))
            next_query_weight = critical
            continue
        if stripped.startswith("Apply template"):
            idx = line.find("Apply ") + len("Apply ")
            end = line.find(" to the query")
            mark(start + idx, start + (end if end != -1 else len(line.rstrip())))
            next_query_weight = critical
            continue
        if stripped.startswith("Candidate ") and len(stripped.split()) == 2:
            idx = line.find("Candidate ") + len("Candidate ")
            mark(start + idx, start + len(line.rstrip()))
            next_query_weight = high
            continue
        if stripped.startswith("Return to the supported same operator format "):
            fm = _FORMAT_SPAN_RE.search(line)
            if fm:
                mark(start + fm.start(), start + fm.end())
            continue
        if stripped.startswith("The motif BA_DC is "):
            if "not supported" in stripped:
                idx = line.find("not supported")
                mark(start + idx, start + idx + len("not supported"))
            elif "supported" in stripped:
                idx = line.find("supported")
                mark(start + idx, start + idx + len("supported"))
            continue
        if stripped.startswith("So BA_DC is "):
            if "confirmed" in stripped:
                idx = line.find("confirmed")
                mark(start + idx, start + idx + len("confirmed"))
            elif "rejected" in stripped:
                idx = line.find("rejected")
                mark(start + idx, start + idx + len("rejected"))
            continue
        if stripped.startswith("Only one same operator row supports"):
            continue
        if stripped.startswith("Verify motif BA_DC"):
            continue
        if stripped.startswith("Choose the helper operator group"):
            promote_helper_equations = True
            continue
        if stripped.startswith("All common output formats agree on "):
            idx = line.find("All common output formats agree on ") + len("All common output formats agree on ")
            mark(start + idx, start + len(line.rstrip()), critical)
            continue
        if stripped == "All common output formats do not agree":
            idx = line.find("do not agree")
            mark(start + idx, start + len(line.rstrip()))
            continue
        if stripped in {"For direct templates, apply the passing template to get the answer"}:
            continue

        if stripped in {"Highest vote count", "Vote winner"}:
            promote_next_value = True
            promote_next_weight = critical
            continue
        if stripped == "Choose operator candidate with highest output agreement vote count, if tie choose the first candidate":
            promote_next_value = True
            promote_next_weight = high
            continue

        # Section-list values after these headers are decisions: surviving
        # common formats, candidate families, and output formats. The header
        # itself stays base weight.
        if promote_list and _FORMAT_NAME_RE.match(stripped):
            mark_line(start, line)
            continue

        if promote_helper_equations:
            if _HELPER_EQUATION_RE.match(stripped):
                mark_line(start, line)
                continue
            promote_helper_equations = False

        if in_match_format_list and stripped == "Common":
            if match_table_row is not None:
                mark_line(match_table_row[0], match_table_row[1])
            in_match_format_list = False
            match_table_row = None

        if in_match_format_list:
            if stripped == "none":
                mark_line(start, line)
                in_match_format_list = False
                match_table_row = None
                continue
            if _FORMAT_NAME_RE.match(stripped):
                if match_table_row is not None:
                    mark_line(match_table_row[0], match_table_row[1])
                    match_table_row = None
                mark_line(start, line)
                continue
            in_match_format_list = False
            match_table_row = None

        if stripped in {
            "Common",
            "Updated surviving common formats",
            "Surviving common formats",
            "Candidate output formats",
            "Motif",
            "Remaining candidate operator families",
            "Operator families used by the visible examples",
        } or stripped.startswith("Candidate operator families for "):
            promote_list = True
            continue

        # Operator-absence symbol mapping: only the query operator row is the
        # decision; the other 25 rows are reference context.
        opm = _QUERY_OPERATOR_RE.match(stripped)
        if opm:
            query_operator_for_mapping = opm.group(1)
            in_symbol_mapping = True
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

        if _VOTE_LINE_RE.match(stripped):
            mark_line(start, line, critical)
            continue
        tm = _TEMPLATE_RESULT_RE.match(stripped)
        if tm:
            mark(start + line.find(tm.group(1)), start + line.find(tm.group(1)) + len(tm.group(1)))
            continue
        if stripped.startswith("The format ") and " supports" in stripped:
            idx = line.find("The format ") + len("The format ")
            end = line.find(" supports", idx)
            mark(start + idx, start + (end if end != -1 else len(line.rstrip())))
            continue

        if any(rx.search(stripped) for rx in _WHOLE_LINE_RES):
            weight = critical if stripped.startswith("For motif ") else high
            mark_line(start, line, weight)
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
