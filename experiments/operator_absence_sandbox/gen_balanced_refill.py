#!/usr/bin/env python3
"""Generate deployed-format AGREE-terminal numeric-equation traces for
under-represented operators, to rebalance the synthetic operator distribution
toward `real`.

Panels come verbatim from the verified `attempt_lines`/`query_lines` builders;
the surrounding wrappers are reconstructed to match the DEPLOYED corpus format
(confirmed against syn_ne_ba_dc_mul_xmuly_0001 and syn_ne_ab_cd_add_len2_0001).

Restricted to first-arithmetic-base families so no failed-earlier-base blocks
are needed:  len1->x-y, len2->x+y, len3->x+y, len4->x*y.
AB_CD limited to len2/len3 (no template section). Random sampling + skeleton
cap so the refill itself stays diverse.
"""
from __future__ import annotations
import csv, re, sys, hashlib, random, collections, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts", "src", "reference/cursor/transformation_rules/numeric_equation/harness"):
    sys.path.insert(0, str(ROOT / p))
spec = importlib.util.spec_from_file_location(
    "rare_gen", ROOT / "scripts/generate_numeric_equation_rare_synthetic_traces.py")
G = importlib.util.module_from_spec(spec); sys.modules["rare_gen"] = G
spec.loader.exec_module(G)
from extended_dsl import apply_pairing  # noqa

HEADER = "We sequentially try direct templates, then motifs BA_DC and AB_CD. For each step, we choose the rule family from same-operator RHS length."

def rhs_len(eq):
    return len("".join(c for c in eq.rhs if c.isdigit()))

def route_lines(examples):
    lengths = sorted({rhs_len(eq) for eq in examples})
    rhs_values = " and ".join(eq.rhs for eq in examples)
    out = [f"Same operator RHS values are {rhs_values}"]
    if lengths == [1]:
        out += ["The RHS values have length 1, so use subtraction or modular",
                "Try x-y,y-x,abs(x-y),min(x,y)-max(x,y),max(x,y)%min(x,y),x%y,y%x"]
    elif lengths in ([2], [1, 2]):
        out += ["The RHS values have length 2, so use addition, subtraction, or modular",
                "Try x+y,x+y-1,x+y+1,x-y,y-x,abs(x-y),min(x,y)-max(x,y),max(x,y)%min(x,y),x%y,y%x"]
    elif lengths == [3, 4]:
        out += ["The RHS values mix length 3 and 4, so use multiplication",
                "Try x*y,x*y+1,x*y-1"]
    elif lengths == [4]:
        out += ["The RHS values have length 4, so use direct templates or multiplication",
                "Try template0134,template3401,x*y,x*y+1,x*y-1"]
    else:
        out += ["The RHS values have length 3, so use addition or multiplication",
                "Try x+y,x+y-1,x+y+1,x*y,x*y+1,x*y-1"]
    return out, lengths

def direct_template_section(examples, op):
    ex = examples[0]; ab, cd = ex.left, ex.right
    L = ["We try direct templates first. If they fail, we proceed to arithmetic search on motifs BA_DC and AB_CD", "",
         f"Try template0134 for operator {op}", "template0134 means AB followed by CD", "",
         f"Example {ex.lhs} = {ex.rhs}", f"AB = {ab}", f"operator = {op}", f"CD = {cd}",
         f"{ab} followed by {cd} gives {ab}{cd} vs {ex.rhs}", "No match", "", "template0134 fails", "",
         f"Try template3401 for operator {op}", "template3401 means CD followed by AB", "",
         f"Example {ex.lhs} = {ex.rhs}", f"AB = {ab}", f"operator = {op}", f"CD = {cd}",
         f"{cd} followed by {ab} gives {cd}{ab} vs {ex.rhs}", "No match", "", "template3401 fails", "",
         "Direct templates fail", "Proceed to arithmetic search on motifs BA_DC and AB_CD", ""]
    return L

def compare_block(query, all_examples, same_examples):
    L = [f"Query {query.lhs}", f"Query operator is {query.op}", "", "Compare example operators"]
    for eq in all_examples:
        L += [f"{eq.lhs} = {eq.rhs} operator {eq.op}", "same" if eq.op == query.op else "different"]
    # list same-op examples in PANEL order (matches route_lines / motif panels)
    L += ["", "Same operator examples", *[f"{eq.lhs} = {eq.rhs}" for eq in same_examples], ""]
    return L

# helper-operator pool (single-char operators, picked != query op)
HELPER_OPS = list("+-*@#&^%<>?!/:$[]{}().'`\"|\\")

# arithmetic-base order per RHS length (templates handled separately for len4)
ARITH_ORDER = {
    1: ['x - y', 'y - x', '|x - y|', 'min(x,y)-max(x,y)', 'max(x,y)%min(x,y)', 'x mod y', 'y mod x'],
    2: ['x + y', 'x + y - 1', 'x + y + 1', 'x - y', 'y - x', '|x - y|',
        'min(x,y)-max(x,y)', 'max(x,y)%min(x,y)', 'x mod y', 'y mod x'],
    3: ['x + y', 'x + y - 1', 'x + y + 1', 'x * y', 'x * y + 1', 'x * y - 1'],
    "3+4": ['x * y', 'x * y + 1', 'x * y - 1'],
    4: ['x * y', 'x * y + 1', 'x * y - 1'],
}

def _panels_up_to(motif, target_base, length, examples):
    """Chain attempt_lines for bases up to target; earlier must fail, target must survive."""
    order = ARITH_ORDER[length]
    if target_base not in order:
        return None
    lines, common = [], None
    for b in order[:order.index(target_base) + 1]:
        panel, c = G.attempt_lines(motif, b, examples)
        if b != target_base and c:
            return None  # an earlier base unexpectedly survives -> trace would be wrong
        lines += panel
        if b == target_base:
            common = c
    if not common:
        return None
    return lines, common

def motif_section(pairing, base, examples, length):
    if pairing == "BA_DC":
        res = _panels_up_to("BA_DC", base, length, examples)
        if not res:
            return None
        lines, common = res
        return ["Try BA_DC first", "", *lines], common
    # AB_CD: BA_DC must fail across the whole arithmetic order first
    lines = ["Try BA_DC first", ""]
    for b in ARITH_ORDER[length]:
        panel, c = G.attempt_lines("BA_DC", b, examples)
        if c:
            return None
        lines += panel
    lines += ["BA_DC has no common output format for this family", "Try AB_CD", ""]
    res = _panels_up_to("AB_CD", base, length, examples)
    if not res:
        return None
    ab_lines, common = res
    return lines + ab_lines, common

def build(pairing, length, base, op, rng):
    mode = rng.choice(list(G.OUTPUT_MODES[pairing]))
    nums = [f"{i:02d}" for i in range(10, 99)]
    # example-structure sampled to match real: 3-5 total, 2-3 same-op, >=1 helper
    total = rng.choice([3, 4, 5])
    n_same = 3 if length == "3+4" else min(rng.choice([2, 3]), total - 1)
    if n_same >= total:
        total = n_same + 1
    n_helper = total - n_same
    # same-operator examples (drive the panels/answer)
    same_seeds = [(rng.choice(nums), op, rng.choice(nums)) for _ in range(n_same)]
    try:
        same_examples = [G.make_eq(l, o, r, pairing, base, mode) for l, o, r in same_seeds]
    except Exception:
        return None
    # all same-op rhs must have the length pattern this family routes on
    expected_lengths = [3, 4] if length == "3+4" else [length]
    if sorted({rhs_len(e) for e in same_examples}) != expected_lengths:
        return None
    ms = motif_section(pairing, base, same_examples, length)
    if not ms:
        return None
    motif_lines, common = ms
    # helper examples: different operators, same motif/base/mode -> valid puzzle rows
    helper_ops = [s for s in HELPER_OPS if s != op]
    helper_examples = []
    for _ in range(n_helper):
        hop = rng.choice(helper_ops)
        try:
            helper_examples.append(G.make_eq(rng.choice(nums), hop, rng.choice(nums), pairing, base, mode))
        except Exception:
            return None
    all_examples = same_examples + helper_examples
    rng.shuffle(all_examples)
    # query
    ql, qr = rng.choice(nums), rng.choice(nums)
    query = G.Eq(ql, op, qr, "")
    if (ql, op, qr) in same_seeds:
        return None
    try:
        qlines, values = G.query_lines(pairing, base, query, common)
    except Exception:
        return None
    if len(set(values.values())) != 1:
        return None
    answer = next(iter(values.values()))
    if not re.fullmatch(r"\d+", answer or ""):
        return None  # keep clean digit-string answers (agree-terminal corpus)
    lengths = sorted({rhs_len(e) for e in same_examples})
    sol = [HEADER, ""] + compare_block(query, all_examples, same_examples)
    rl, _ = route_lines(same_examples)
    sol += rl + [""]
    if lengths == [4]:
        sol += direct_template_section(same_examples, op)
    sol += motif_lines
    sol += [f"Apply format {pairing}|{G.BASE_LABEL[base]}|common to the query", "", *qlines,
            f"All common output formats agree on {answer}"]
    gcot = "\n".join(sol) + f"\n\nAnswer: \\boxed{{{answer}}}"
    prompt = G.question_text(all_examples, query)
    return prompt, gcot, answer

# base sampling weights ~ real base distribution; eligible RHS lengths per base
REAL_BASE_W = {'x - y': 25.8, 'x + y': 20.3, 'x * y': 16.9, 'x * y - 1': 7.4, 'x + y - 1': 7.0,
               'x * y + 1': 6.8, 'x + y + 1': 6.1, '|x - y|': 3.5, 'max(x,y)%min(x,y)': 3.1,
               'y - x': 2.9, 'min(x,y)-max(x,y)': 0.2}
BASE_LENGTHS = {'x + y': [2, 3], 'x + y - 1': [2, 3], 'x + y + 1': [2, 3],
                'x * y': [3, "3+4", 4], 'x * y + 1': [3, "3+4", 4], 'x * y - 1': [3, "3+4", 4],
                'x - y': [1, 2], 'y - x': [1, 2], '|x - y|': [1, 2],
                'min(x,y)-max(x,y)': [1, 2], 'max(x,y)%min(x,y)': [1, 2]}
_BASES = list(REAL_BASE_W); _BW = [REAL_BASE_W[b] for b in _BASES]

def sample_family(rng):
    base = rng.choices(_BASES, weights=_BW)[0]
    length = rng.choice(BASE_LENGTHS[base])
    # AB_CD only for len 2/3 (no template enumeration); else BA_DC
    pairing = "AB_CD" if (length in (2, 3) and rng.random() < 0.28) else "BA_DC"
    return pairing, length, base

def skel(t):
    return hashlib.md5(re.sub(r"[ \t]+", " ", re.sub(r"\d", "#", t)).encode()).hexdigest()

def generate(quota_by_op, seed=7, cap=5, init_skel=None, used_prompts=None, rhs34_quota_by_op=None):
    rng = random.Random(seed)
    rows = []; skc = collections.Counter(init_skel or {}); usedq = set(used_prompts or set())
    rhs34_quota_by_op = rhs34_quota_by_op or {}
    for op, quota in quota_by_op.items():
        made = 0; att = 0; made_rhs34 = 0
        target_rhs34 = rhs34_quota_by_op.get(op)
        while made < quota and att < quota * 600 + 6000:
            att += 1
            pairing, length, base = sample_family(rng)
            remaining = quota - made
            need_rhs34 = 0 if target_rhs34 is None else target_rhs34 - made_rhs34
            if target_rhs34 is not None and length != "3+4" and remaining <= need_rhs34:
                continue
            try:
                built = build(pairing, length, base, op, rng)
            except Exception:
                continue
            if not built:
                continue
            if target_rhs34 is not None:
                if length == "3+4" and made_rhs34 >= target_rhs34:
                    continue
                if length != "3+4" and remaining <= need_rhs34:
                    continue
            prompt, gcot, answer = built
            if prompt in usedq:
                continue
            sk = skel(gcot)
            if skc[sk] >= cap:
                continue
            skc[sk] += 1; usedq.add(prompt)
            rows.append({"op": op, "pairing": pairing, "length": length, "base": base,
                         "prompt": prompt, "gcot": gcot, "answer": answer})
            made += 1
            if length == "3+4":
                made_rhs34 += 1
        if target_rhs34 is not None and made_rhs34 != target_rhs34:
            raise RuntimeError(f"operator {op} generated {made_rhs34} mixed 3+4 rows, expected {target_rhs34}")
    return rows

if __name__ == "__main__":
    test = generate({"-": 4, "*": 4, "+": 4}, cap=5)
    print("generated:", len(test), "by op:", collections.Counter(r["op"] for r in test))
    print("distinct skeletons:", len({skel(r["gcot"]) for r in test}))
    s = test[0]
    print("==== SAMPLE op=%s %s len=%d ====" % (s["op"], s["pairing"], s["length"]))
    print("PROMPT:", s["prompt"])
    print(s["gcot"])
