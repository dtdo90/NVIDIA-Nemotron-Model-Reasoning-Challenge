#!/usr/bin/env python3
"""Independent mechanical verifier for generated refill traces.
Recomputes every panel value via the harness, checks Common=intersection of
matches, query agree=unanimous, boxed==answer. Mirrors deep_audit_numeric_900.
"""
import sys, re, collections, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference/cursor/transformation_rules/numeric_equation/harness"))
sys.path.insert(0, str(ROOT / "src"))
from extended_dsl import apply_pairing, BASE_RULE_BY_NAME, OUTPUT_MODE_BY_NAME
from nemotron_baseline.prompts import build_assistant_trace_content

BASE_LBL = {'x+y':'x + y','x+y-1':'x + y - 1','x+y+1':'x + y + 1','x-y':'x - y','y-x':'y - x',
            'abs(x-y)':'|x - y|','min(x,y)-max(x,y)':'min(x,y)-max(x,y)','max(x,y)%min(x,y)':'max(x,y)%min(x,y)',
            'x%y':'x mod y','y%x':'y mod x','x*y':'x * y','x*y+1':'x * y + 1','x*y-1':'x * y - 1'}
OPFAM = {'op_prefix','op_suffix','op_prefix_rev','op_suffix_rev'}

def verify_rows(rows):
    iss = collections.defaultdict(list); cells = 0
    for I, prompt, gcot, answer, op in rows:
        t = gcot
        # boxed
        bx = re.search(r'\\boxed\{(.*)\}\s*$', t, re.S)
        if not bx or bx.group(1) != answer:
            iss['boxed_neq_answer'].append(I); continue
        # wrap
        c = build_assistant_trace_content(answer, generated_cot=gcot, assistant_content="")
        if not (c.lstrip().startswith('<think>') and re.search(r'</think>\s*\\boxed\{.*\}\s*$', c, re.S)):
            iss['bad_wrap'].append(I)
        # per Try-panel arithmetic + match list + Common
        for mt in re.finditer(r'(?m)^Try (BA_DC|AB_CD) with (.+?) for operator (.+)$', t):
            motif, bl, pop = mt.groups(); bn = BASE_LBL.get(bl)
            if not bn: iss['unknown_base_label'].append((I, bl)); continue
            seg = t[mt.end():]; nt = seg.find('\nTry '); seg = seg[:nt] if nt >= 0 else seg
            ap = seg.find('\nApply '); seg = seg[:ap] if ap >= 0 else seg
            ex_matches = []
            for em in re.finditer(r'(?m)^Example (\d+)(.)(\d+) = (.+)$', seg):
                L, o, R, rhs = em.group(1), em.group(2), em.group(3), em.group(4); sub = seg[em.end():]
                hl = re.match(r'\n((?:BA DC|AB CD) .+)\n(.+)', sub)
                if not hl: continue
                modes = hl.group(1).split()[3:]; vals = hl.group(2).split()
                if len(vals) < 3 + len(modes): iss['short_value_row'].append((I, L+o+R)); continue
                pvals = vals[3:3+len(modes)]
                x, y = apply_pairing(L, R, motif); bv = BASE_RULE_BY_NAME[bn].apply(x, y)
                if bv is not None and vals[2] != str(bv): iss['base_value_wrong'].append((I, L+o+R, vals[2], bv))
                for mn, gv in zip(modes, pvals):
                    om = OUTPUT_MODE_BY_NAME.get(mn)
                    if not om or mn in OPFAM: continue
                    cells += 1
                    try: ev = om.apply(bv, o, None)
                    except Exception: ev = None
                    if ev is not None and ev != gv:
                        iss['panel_value_wrong'].append((I, L+o+R, mn, gv, ev))
                true = set(mn for mn, v in zip(modes, pvals) if v == rhs)
                after = sub[hl.end():].split("\n")
                mi = next((j for j, l in enumerate(after) if l.strip() == 'Match'), None)
                printed = set()
                if mi is not None:
                    for l in after[mi+1:]:
                        s2 = l.strip()
                        if s2 in modes: printed.add(s2)
                        else: break
                    if printed != true: iss['match_list_wrong'].append((I, L+o+R, sorted(printed ^ true)))
                ex_matches.append(printed)
            mc = re.search(r'(?m)^Common\n((?:.*\n?)*)', seg)
            if mc and ex_matches:
                pc = []
                for l in mc.group(1).split("\n"):
                    s2 = l.strip()
                    if s2 == '': continue
                    if s2 in OUTPUT_MODE_BY_NAME: pc.append(s2)
                    else: break
                inter = set.intersection(*ex_matches) if ex_matches else set()
                if set(pc) != inter: iss['common_wrong'].append((I, bl, sorted(set(pc)), sorted(inter)))
        # consistency: "Same operator examples" list == first-panel Example order == route order
        som = re.search(r'(?m)^Same operator examples\n((?:\d+.\d+ = \S+\n)+)', t)
        if som:
            listed = [ln.replace(' = ', '=') for ln in som.group(1).strip().split('\n')]
            first_seg = t[re.search(r'(?m)^Try (?:BA_DC|AB_CD) with', t).start():]
            nxt = first_seg.find('\nTry ', 3); first_seg = first_seg[:nxt] if nxt >= 0 else first_seg
            panel_ex = [f"{m.group(1)}{m.group(2)}{m.group(3)}={m.group(4)}"
                        for m in re.finditer(r'(?m)^Example (\d+)(.)(\d+) = (.+)$', first_seg)]
            if listed != panel_ex:
                iss['sameop_order_mismatch'].append((I, listed, panel_ex))
            rline = re.search(r'Same operator RHS values are ([^\n]+)', t)
            if rline:
                route_rhs = rline.group(1).split(' and ')
                if route_rhs != [e.split('=')[1] for e in listed]:
                    iss['route_order_mismatch'].append((I, route_rhs))
            # helpers must NOT appear inside any panel
            qop = re.search(r'Query operator is (.+)', t).group(1)
            for m in re.finditer(r'(?m)^Example (\d+)(.)(\d+) = ', t):
                if m.group(2) != qop:
                    iss['helper_in_panel'].append((I, m.group(0).strip()))
        # query agree: recompute query panel
        qm = re.search(r'Apply format (BA_DC|AB_CD)\|([^|]+)\|common to the query\n\nQuery\n(\d+)(.)(\d+)\n([^\n]+)\n([^\n]+)\nAll common output formats agree on (\S+)', t)
        if not qm: iss['no_query_block'].append(I); continue
        pairing, bl, ql, qop, qr, qhdr, qval, agree = qm.groups()
        bn = BASE_LBL.get(bl)
        modes = qhdr.split()[3:]; vals = qval.split()[3:]
        x, y = apply_pairing(ql, qr, pairing); bv = BASE_RULE_BY_NAME[bn].apply(x, y)
        recomputed = set()
        for mn, gv in zip(modes, vals):
            om = OUTPUT_MODE_BY_NAME.get(mn)
            try: ev = om.apply(bv, qop, None)
            except Exception: ev = None
            if ev is not None and ev != gv: iss['query_value_wrong'].append((I, mn, gv, ev))
            recomputed.add(gv)
        if len(recomputed) != 1: iss['query_not_unanimous'].append((I, sorted(recomputed)))
        if agree != answer: iss['agree_neq_answer'].append((I, agree, answer))
    return iss, cells

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments/operator_absence_sandbox"))
    import gen_balanced_refill as B
    test = B.generate({'-':80,'*':80,'+':80,'%':20,'&':20,'#':20,'^':20}, cap=5)
    rows = [(f"gen_{i:04d}", r['prompt'], r['gcot'], r['answer'], r['op']) for i, r in enumerate(test)]
    iss, cells = verify_rows(rows)
    print("verified %d traces, %d panel cells" % (len(rows), cells))
    print("distinct skeletons:", len({B.skel(r['gcot']) for r in test}))
    print("by op:", dict(collections.Counter(r['op'] for r in test)))
    if not iss: print("ISSUES: none")
    for k in sorted(iss):
        v = iss[k]; print("  %s: %d  e.g. %s" % (k, len(set(x[0] if isinstance(x, tuple) else x for x in v)), v[:2]))
