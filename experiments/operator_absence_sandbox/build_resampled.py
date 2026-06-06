#!/usr/bin/env python3
"""Assemble the operator-rebalanced + deduped synthetic block and write
single_phase_sft_v2_resampled.csv. v2 itself is left untouched.

Pipeline (numeric synthetic only; real/curriculum/eval/text-cipher untouched):
  1. drop synth-only operators (~ ;)
  2. cap each digit-blanked skeleton cluster at 5
  3. downsample operators over real-target; generate agree-traces for those under
  4. verify generated traces mechanically; write final CSV; report distribution
"""
import sys, csv, re, collections, random, hashlib
from pathlib import Path
csv.field_size_limit(10**8)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/operator_absence_sandbox"))
import gen_balanced_refill as B
from verify_refill import verify_rows

SRC = ROOT / "data/single_phase_training_clean/single_phase_sft_v2.csv"
OUT = ROOT / "data/single_phase_training_clean/single_phase_sft_v2_resampled.csv"
EXOTIC = {'~', ';'}
CAP = 5
TARGET_TOTAL = 3187  # keep numeric-synthetic size exactly constant

def proportional_targets(counter, total):
    """Largest-remainder allocation so rounded targets sum exactly to total."""
    denom = sum(counter.values())
    raw = {k: counter[k] / denom * total for k in counter}
    out = {k: int(raw[k]) for k in raw}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda x: (raw[x] - out[x], counter[x], x), reverse=True)[:left]:
        out[k] += 1
    return out

def body(r): return (r['assistant_content'] or '').strip() or (r['generated_cot'] or '').strip()
def op_of(r):
    m = re.search(r'result for[:\s]+(\d+)(\D)(\d+)', r['prompt']); return m.group(2) if m else None
def skel(t): return hashlib.md5(re.sub(r"[ \t]+", " ", re.sub(r"\d", "#", t)).encode()).hexdigest()

def rhs_pattern_of(r):
    op = op_of(r)
    if not op:
        return "none"
    vals = []
    for line in r['prompt'].splitlines():
        m = re.match(r'(-?\d+)(\D)(-?\d+)\s*=\s*(\S+)', line.strip())
        if m and m.group(2) == op:
            vals.append(m.group(4))
    if not vals:
        return "none"
    lengths = sorted({len("".join(ch for ch in v if ch.isdigit())) for v in vals})
    return "+".join(map(str, lengths))

def normalize_apply_format(text):
    return re.sub(r'(?m)^Apply ((?:AB_CD|BA_DC)\|[^\n]+ to the query)$', r'Apply format \1', text or '')

def normalize_row(row):
    out = dict(row)
    out['generated_cot'] = normalize_apply_format(out.get('generated_cot', ''))
    out['assistant_content'] = normalize_apply_format(out.get('assistant_content', ''))
    return out

def main():
    rows = list(csv.DictReader(open(SRC)))
    cols = rows[0].keys()
    is_num_syn = lambda r: r['source_mode'] == 'synthetic' and r['category'] == 'Numeric Equation Transformation Rules'
    syn = [r for r in rows if is_num_syn(r)]
    other = [r for r in rows if not is_num_syn(r)]
    real = [r for r in rows if r['source_mode'] == 'real' and r['category'] == 'Numeric Equation Transformation Rules']

    # 1. drop exotic
    syn1 = [r for r in syn if op_of(r) not in EXOTIC]
    # 2. cap skeleton clusters
    clusters = collections.defaultdict(list)
    for r in sorted(syn1, key=lambda x: x['id']): clusters[skel(body(r))].append(r)
    kept = []
    for c, m in clusters.items(): kept.extend(m[:CAP])

    # 3. target distribution = real shares * TARGET_TOTAL
    realc = collections.Counter(filter(None, (op_of(r) for r in real))); rt = sum(realc.values())
    target = proportional_targets(realc, TARGET_TOTAL)

    rng = random.Random(20240607)
    # downsample over-target ops
    by_op = collections.defaultdict(list)
    for r in kept: by_op[op_of(r)].append(r)
    final_kept = []
    quota_gen = {}
    for o, tgt in sorted(target.items()):
        rs = by_op.get(o, [])
        if len(rs) > tgt:
            final_kept.extend(rng.sample(rs, tgt))
        else:
            final_kept.extend(rs)
            quota_gen[o] = tgt - len(rs)
    # clean: ensure quota only positive
    quota_gen = {o: n for o, n in quota_gen.items() if n > 0}

    real_rhs = collections.Counter(rhs_pattern_of(r) for r in real)
    target_rhs34 = round(real_rhs["3+4"] / len(real) * TARGET_TOTAL)
    kept_rhs34 = sum(1 for r in final_kept if rhs_pattern_of(r) == "3+4")
    rhs34_to_generate = max(0, target_rhs34 - kept_rhs34)
    rhs34_quota_by_op = proportional_targets(collections.Counter(quota_gen), rhs34_to_generate) if rhs34_to_generate else {}

    print("kept after dedup:", len(kept), "| ops to generate:", sum(quota_gen.values()))
    print("quota_gen:", dict(sorted(quota_gen.items(), key=lambda x: -x[1])))
    print("rhs 3+4 target:", target_rhs34, "| kept:", kept_rhs34, "| generate:", rhs34_to_generate)

    # 4. generate (skeleton-aware against the kept set so combined honors cap)
    kept_skels = collections.Counter(skel(body(r)) for r in final_kept)
    kept_prompts = {r['prompt'] for r in final_kept}
    gen = B.generate(
        quota_gen,
        seed=7,
        cap=CAP,
        init_skel=kept_skels,
        used_prompts=kept_prompts,
        rhs34_quota_by_op=rhs34_quota_by_op,
    )
    if len(gen) != sum(quota_gen.values()):
        raise RuntimeError(f"generated {len(gen)} rows, expected {sum(quota_gen.values())}")
    gen_by_op = collections.Counter(g['op'] for g in gen)
    print("generated:", len(gen), "by op:", dict(gen_by_op))
    # verify generated
    vrows = [(f"gen_{i}", g['prompt'], g['gcot'], g['answer'], g['op']) for i, g in enumerate(gen)]
    iss, cells = verify_rows(vrows)
    print("verify generated: %d cells; issues: %s" % (cells, {k: len(v) for k, v in iss.items()} or "none"))
    if iss:
        print("ABORT: generated traces failed verification"); return

    # build new synthetic rows
    new_syn = [normalize_row(r) for r in final_kept]
    base_row = syn[0]
    for i, g in enumerate(gen):
        nr = {c: '' for c in cols}
        nr.update({
            'id': f"syn_ne_oprebal_{i:05d}",
            'prompt': g['prompt'], 'answer': g['answer'],
            'generated_cot': normalize_apply_format(g['gcot']), 'assistant_content': '',
            'label': 'Numeric Equation Transformation Rules',
            'category': 'Numeric Equation Transformation Rules',
            'source': f"synthetic/operator_rebalance/syn_ne_oprebal_{i:05d}.txt",
            'source_mode': 'synthetic', 'eval_eligible': 'false', 'split_policy': 'train_only',
            'append_answer_instruction': 'true', 'official_answer': g['answer'],
            'huikang_status': '', 'prompt_format': 'competition_chat_template',
        })
        new_syn.append(nr)

    out_rows = other + new_syn
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols), lineterminator='\n'); w.writeheader(); w.writerows(out_rows)

    # 5. report
    finalc = collections.Counter(filter(None, (op_of(r) for r in new_syn)))
    ft = sum(finalc.values())
    print("\nwrote %s" % OUT.name)
    print("rows: total %d (was %d) | numeric-synthetic %d (was %d)" % (len(out_rows), len(rows), len(new_syn), len(syn)))
    # skeleton cap check
    sc = collections.Counter(skel(body(r)) for r in new_syn)
    print("max skeleton cluster in new synthetic:", max(sc.values()))
    print("\nop   real%%   new-syn%%   (target_n / actual_n)")
    for o in sorted(set(realc) | set(finalc), key=lambda o: -realc[o]):
        print("  %-3s %5.1f%%  %6.1f%%   (%d / %d)" % (o, 100*realc[o]/rt, 100*finalc[o]/ft, target.get(o, 0), finalc[o]))

if __name__ == "__main__":
    main()
