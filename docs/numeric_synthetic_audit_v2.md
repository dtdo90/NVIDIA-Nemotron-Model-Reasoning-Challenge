# Numeric-Equation Synthetic-Trace Audit (v2)

**Question:** why would adding synthetic numeric-equation traces *reduce* model
performance? **Scope:** all numeric-equation rows in
`single_phase_sft_v2.csv` — 552 real, 2,859 synthetic, 828 decision-point
curriculum, 22 held-out eval.

**Headline:** the synthetic traces are mechanically correct (arithmetic,
rendering, boxed==answer all verified clean in prior rounds). The risk is not
errors — it is **distribution mismatch, heavy templating, and over-representation
of ambiguous / policy-defined labels**. All three pull the model's prior away
from the real/eval distribution, which is the classic way "more data" hurts.

The held-out **eval** set is the tell: on every axis it tracks **real**, not
synthetic. So synthetic ≠ test distribution.

---

## Finding A — Operator distribution is badly skewed (high severity)

Query-operator share by group:

| operator | real | eval | synth | curric |
|---|---|---|---|---|
| `-` | 18.5% | 18.2% | **6.2%** | 25.7% |
| `+` | 13.9% | 9.1% | 8.8% | 6.3% |
| `*` | 12.9% | 9.1% | **3.2%** | 1.3% |
| `%` | 2.4% | 13.6% | 10.0% | 6.3% |
| `&` `#` `^` `~` | ~2% each | low | 6–7% each | elevated |

Real and eval concentrate on the natural arithmetic operators (`-`,`+`,`*` ≈ 45%
of real). Synthetic **flattens** the distribution across the symbol alphabet:
it starves `-` (~3×) and `*` (~4×) and floods exotic symbols. Worse, `~` and `;`
appear **174 / several** times in synthetic but **never** in real or eval — the
generator invented operators the test-like data does not contain.

*Effect:* the model spends capacity learning a symbol world the test set
doesn't share, and sees the dominant `-`/`*` cases under-weighted.

## Finding B — ~40% of synthetic is near-duplicate scaffolding (high severity)

Digit-blanked structural skeletons, duplication within group:

| group | distinct skeletons | largest identical cluster | rows in duplicate clusters |
|---|---|---|---|
| real | 100% unique | 2 | 0% |
| synth | 69% unique | **43** | **40%** |
| curric | 75% unique | 18 | 34% |

Real traces are essentially all structurally unique. **40% of synthetic rows
share a skeleton with another row**, the largest cluster being 43 identical-shape
traces. Every synthetic trace opens with the same boilerplate sentence (2,859/2,859),
and the verbose mode-panel header repeats 1,400+ times.

*Effect:* the model is shown the same long scaffold over and over, encouraging it
to memorize the template rather than the reasoning — and inflates the synthetic
share without adding diversity.

## Finding C — Synthetic/curriculum over-represent ambiguous, policy-defined labels (high severity)

Brute-forcing every `{AB_CD, BA_DC} × base × output-mode` rule consistent with
each problem's shown examples (relative comparison; absolute rates are
conservative because the search omits width/padding):

| group | ambiguous (≥2 distinct query answers) | label not reproducible by any consistent rule |
|---|---|---|
| real | 0.7% | 2.4% |
| eval | 4.5% | (small n) |
| synth | **22.8%** | **10.0%** |
| curric | **42.0%** | **8.8%** |

Even normalized to problems solvable within the search universe, synthetic and
curriculum are far more ambiguous than real. Many of these problems are
**under-determined** — the displayed examples fit multiple DSL rules that
disagree on the query — so the "correct" answer is fixed by the resolution
*policy* (voting / priority / op-fallback), not forced by the examples. About
10% of synthetic labels aren't reproducible by *any* example-consistent rule
(operator-absence "fallback" families especially): the label is a heuristic
target.

This is partly **by design** — the curriculum exists to teach disagreement
resolution. But it means a large fraction of training signal depends on a policy
the competition grader may not share. If the grader resolves these cases
differently, the model has been confidently trained toward the wrong answer.

Terminal-type mix confirms the shift: real is **88.9% simple "agree"**; synthetic
is 78.8% and curriculum only 65.3%, with 3–5× the rate of voting / priority /
op-fallback endings.

## Finding D — Length & coverage drift (medium)

- Synthetic traces run ~30% longer than real (median 3,182 vs 2,439 chars).
- Synthetic has **zero** length-`3+4` mixed-RHS problems, which are **7.2% of
  real** — a real pattern the synthetic set never covers. It over-weights
  length-1 (10.9% vs 3.8%).
- Curriculum over-weights the `abs(x-y)` base (16.8% vs real 3.5%) and uses
  modular bases `x%y` / `y%x` that are **0% in real**.

## Finding E — Motif drift (low)

Final chosen motif: real `BA_DC` 72.4% vs synth 65.2%, curric 59.6%. Minor.

---

## Recommendations (in priority order)

1. **Resample synthetic operators to match real/eval.** Boost `-`, `+`, `*`;
   cut the exotic-symbol mass; drop or heavily down-weight `~` and `;` (absent
   from test-like data).
2. **Deduplicate scaffolding.** Cap near-identical skeleton clusters (e.g. ≤5
   per skeleton). Trimming the 40% structural filler shrinks synthetic without
   losing coverage and reduces template overfit.
3. **Rebalance terminal types toward "agree."** Bring the voting/priority/
   fallback share down toward the ~11% real rate, or confirm the curriculum's
   resolution policy matches the grader before training on it at volume.
4. **Audit operator-absence "fallback" labels.** ~10% of synthetic answers are
   heuristic, not example-forced; verify the fallback choice matches the intended
   target, and down-weight if uncertain.
5. **Close coverage gaps / fix base skew.** Add length-`3+4` mixed cases; reduce
   the `abs`/modular over-representation; match trace length to real.

## How to confirm

The decisive test is an **ablation**: train with vs without the synthetic block
(and separately with vs without curriculum) and compare eval. The distributional
evidence above predicts that the raw synthetic block, as currently weighted,
degrades real/eval performance; an operator-rebalanced + deduplicated subset
should help instead of hurt.
