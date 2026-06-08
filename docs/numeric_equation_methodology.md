# Numeric Equation Transformation Methodology

This note records the methodology we want to teach for the
`Transformation Rules / numeric_equation` subtype.

The goal is to translate numeric-equation solving into reproducible CoT traces,
not just solver summaries. The trace should teach the model to search a small
structured DSL:

```text
motif | operation | output_mode
```

where:

```text
motif       in {BA_DC, AB_CD}
operation   is local to the visible operator
output_mode is usually row-level, such as plain, rev, op_prefix, op_suffix, or negative-only variants
```

## 1. Problem Shape

Each row contains examples of the form:

```text
AB op CD = RHS
```

where `AB` and `CD` are two-digit operands and `op` is a visible operator
character. The query has the same shape:

```text
AB op CD
```

The model should infer how to transform the query expression into the answer.

Important assumptions:

1. The operand motif and output mode are usually shared across the row.
2. The arithmetic operation can differ by visible operator.
3. Same-operator examples are the strongest evidence for the query operation.
4. Other-operator examples can provide row motif and output-mode evidence.
5. Query operators absent from examples require cautious motif projection, not
   a free guess.

## 2. Parse And Group By Operator

Parse every example into:

```text
left operand AB
operator op
right operand CD
RHS text
```

Then identify:

```text
query operator = op in the query
same-operator examples = examples whose operator equals the query operator
helper examples = all other examples
```

The trace should start with the same-operator examples when they exist.

Use helper examples later for:

1. motif evidence
2. output-mode evidence
3. resolving ambiguity between multiple same-operator candidates
4. absent-query-operator projection

## 3. Direct Template Or Concat-Like Check

Before arithmetic, check whether a simple direct positional template explains
the same-operator rows.

For numeric equations, use the same direct-template terminology as the
symbol-transform traces. In the fixed shape `ABOCD`, where `O` is the visible
operator:

```text
template0134 -> ABCD
template3401 -> CDAB
```

This is analogous to direct template matching in symbol-transform rows. If a
direct template explains every same-operator example, use it immediately and
do not invent arithmetic.

For the numeric DSL, these direct templates are implemented by concat-style
rules. For example, `0134` is equivalent to writing `AB` then `CD`, while
`3401` is equivalent to writing `CD` then `AB`. Some concat candidates can be
algebraically equivalent after `rev`, but the trace should still call the
visible direct template `0134` or `3401`.

If direct template matching fails, move to arithmetic search.

## 4. Motif Search

Use only the two active motifs:

```text
AB_CD: x = AB, y = CD
BA_DC: x = BA, y = DC
```

Try `BA_DC` first. If it fails for the same-operator examples, then try
`AB_CD`.

For a trace, write the motif conversion before computing:

```text
82/15 under BA_DC:
82 -> 28, 15 -> 51
x=28, y=51
```

The core motifs are:

```text
BA_DC|rev
AB_CD|plain
```

Do not introduce `AB_DC` or `BA_CD` in training traces unless we explicitly
open a later phase for rare motifs.

## 5. Use RHS Length To Choose Operation Family

RHS length is a strong routing signal, copied from the symbol-transform method:

```text
RHS length 4 -> multiplication family
RHS length 3 -> addition or multiplication family
RHS length 2 -> addition or subtraction family
RHS length 1 -> subtraction family
```

When same-operator RHS lengths are mixed, use the intersection of possible
families:

```text
lengths 1 and 2 -> subtraction or modular family only
lengths 2 and 3 -> addition family only
```

Family candidates:

```text
multiplication family: x*y, x*y+1, x*y-1
addition family:       x+y, x+y+1, x+y-1
subtraction family:    x-y, y-x, abs(x-y), max(x,y)%min(x,y), x%y, y%x
```

For length 3, use the capped learned order:

```text
x+y, x*y, x+y+1
```

For length 2, keep the modular variants at the end after addition and
subtraction candidates.

The trace should not say "try everything" without showing why the current
family is being tried.

## 6. Output Modes

After computing the raw value, test output rendering.

Core output modes:

```text
plain: write the value directly
rev: reverse the value text
op_prefix: prefix the rendered value with the visible operator
op_suffix: append the visible operator after the rendered value
```

Negative-sensitive output modes:

```text
op_prefix_if_neg
op_suffix_if_neg
op_suffix_rev_if_neg
rev_or_op_prefix_if_neg
rev_or_op_suffix_if_neg
rev_or_op_prefix_rev_if_neg
neg
neg_rev
abs_rev
op_prefix_rev
```

Trace rule:

1. If examples show an operator prefix or suffix, state that evidence before
   applying it to the query.
2. If examples are all positive but the query computation becomes negative,
   consider negative-sensitive output modes only when helper rows support that
   row-level output style.
3. Do not silently drop signs. Explain the rendering.

### 6.1 Disagreement Resolution Policy

When the common output formats disagree on the query rendering, the trace must
resolve the disagreement with the following policy. The policy has two layers:
a small set of **deterministic operator-rendering rules** for the clean
subtraction base with a negative value, and a **motif-specific tiebreaker** for
every other case.

**Layer 1 — deterministic operator rendering.** This applies only when the
selected base is plain subtraction (`x-y` or `y-x`), the query value is
negative, and the query operator is not the literal `-`. Take the magnitude
(the positive part), reverse it only under `BA_DC`, and attach the operator on
the side determined by the base:

| Motif | Base | Value | Operator | Rule |
|---|---|---|---|---|
| `AB_CD` | `x-y` | negative | not `-` | magnitude as-is, operator **prefix** |
| `BA_DC` | `x-y` | negative | not `-` | **reverse** magnitude, operator **prefix** |
| `AB_CD` | `y-x` | negative | not `-` | magnitude as-is, operator **suffix** |
| `BA_DC` | `y-x` | negative | not `-` | **reverse** magnitude, operator **suffix** |

So `x-y` selects prefix, `y-x` selects suffix; `BA_DC` reverses the magnitude
first, `AB_CD` does not. Worked examples:

```text
AB_CD|x-y  10&32 -> 10-32=-22 -> magnitude 22 -> prefix & -> &22
BA_DC|x-y  10$32 -> 01-23=-22 -> magnitude 22, reverse 22 -> prefix $ -> $22
AB_CD|y-x  32&10 -> 10-32=-22 -> magnitude 22 -> suffix & -> 22&
BA_DC|y-x  32$10 -> 01-23=-22 -> magnitude 22, reverse 22 -> suffix $ -> 22$
```

**Layer 2 — motif tiebreaker.** Every case not covered by Layer 1 (positive
value, non-subtraction base such as `abs`, modular, or min/max, or a literal
`-` operator) is resolved by a fixed tiebreaker that depends only on the motif:

| Motif | Tiebreaker |
|---|---|
| `AB_CD` | use the **first common format in priority order** |
| `BA_DC` | use **voting** across the common formats; on a tie, fall back to the first common format in priority order |

This is why the corpus uses voting at all: voting is `BA_DC`'s tiebreaker, not a
contradiction of the deterministic rules. `AB_CD` never votes — it always falls
back to the priority-ordered first common format.

**The literal `-` carve-out.** When the query operator is the literal `-`,
operator-prefix/suffix rendering collapses into an ordinary negative sign and
becomes a weak teaching signal, so the deterministic Layer-1 rules are skipped
and the case drops to the Layer-2 tiebreaker instead:

- `AB_CD`, `x-y` negative, literal `-` -> first common format in priority order
- `BA_DC`, `y-x` negative, literal `-` -> voting

**Complete decision table** (motif, base/sign condition -> resolution):

```text
AB_CD | x-y negative, operator not -            -> operator prefix (magnitude as-is)
AB_CD | x-y negative, literal -                 -> first common format in priority order
AB_CD | x-y positive                            -> first common format in priority order
AB_CD | base is not x-y (incl. y-x, abs, mod)   -> first common format in priority order
BA_DC | x-y negative                            -> reverse magnitude, operator prefix
BA_DC | x-y positive                            -> voting
BA_DC | y-x negative, operator not -            -> reverse magnitude, operator suffix
BA_DC | y-x negative, literal -                 -> voting
BA_DC | base is neither x-y nor y-x (abs, mod)  -> voting
```

Note on `AB_CD|y-x`: the deterministic suffix rule in Layer 1 is the intended
rendering, but in the current corpus `AB_CD|y-x` negative cases are reached
through the generic `AB_CD` tiebreaker (first common format in priority order),
which already yields the agreed answer, so no dedicated `AB_CD|y-x` policy
samples are generated.

Example:

```text
If negative, reverse the magnitude and append the operator; otherwise write it directly.
47-89=-42
reverse magnitude 42 -> 24
append - -> 24-
```

## 7. Candidate Verification Flow

For each candidate:

```text
motif | operation | output_mode
```

verify on same-operator examples first.

After choosing a motif and operation, test the active output modes with a
visible table. The table should compute the raw value once, then show how each
output mode renders it.

Example table shape:

```text
Try BA_DC with x*y for operator /.

82/15 = 8241
82 -> 28, 15 -> 51
28*51=1428

output_mode: plain,rev,op_prefix,op_suffix,op_prefix_if_neg,op_suffix_rev_if_neg,op_prefix_rev,rev_or_op_prefix_rev_if_neg,rev_or_op_suffix_if_neg,neg,neg_rev,abs_rev
result: 1428,8241,/1428,1428/,1428,1428,/8241,8241,8241,-1428,-8241,8241
match: x,ok,x,x,x,x,x,ok,ok,x,x,ok
```

If multiple output modes match positive examples in the same way, use helper
rows or query-sign evidence to decide. If they still produce the same query
answer, the trace can proceed by agreement. If they produce different answers,
do not mark the trace deterministic.

A compact row-by-row pattern is:

```text
Try BA_DC|x*y|rev for operator /.

82/15 = 8241
82 -> 28, 15 -> 51
28*51=1428
rev(1428)=8241
PASS
```

If it fails:

```text
Try BA_DC|x+y|rev.
28+51=79
rev(79)=97, not 8241
FAIL
```

After same-operator verification:

1. If one candidate remains, use it for the query.
2. If multiple candidates remain but all give the same query answer, the trace
   may mark the answer deterministic by agreement.
3. If multiple candidates give different query answers, use helper rows for
   motif/output-mode evidence.
4. If ambiguity remains, do not call the trace deterministic.

Single-row motif-override work to resume:

```text
Rewrite the one-same-operator motif-conflict rows by listing the same-operator
examples first, then giving the count after the list:

same operator examples
17*71 = 87
one example

After the BA_DC Common block, use:

The format BA_DC|x+y-1|common supports the single same operator example
Only one same operator row supports this candidate, so do not finalize yet
Verify motif BA_DC using an additional helper operator group

Then choose the helper operator group with the least number of examples, verify
only motif BA_DC on that helper group. Do not repeat `Try BA_DC first` inside
the helper verification because this branch is already verifying BA_DC.

If the helper group supports motif BA_DC, write:

The motif BA_DC is supported by the helper operator group
So BA_DC is confirmed

Apply format BA_DC|x+y-1|common to the query

If the helper group does not support motif BA_DC, write:

So BA_DC is rejected

Try AB_CD

For the normal multi-row case, keep the change minimal:

The format BA_DC|x+y-1|common supports all 2 same operator examples
More than one same operator rows support this candidate, so finalize
Apply format BA_DC|x+y-1|common to the query

Use the exact count in the sentence, for example 2 or 3.
```

Rows to revisit with this pattern:

```text
4dcc1844
b9bf883d
c5b058d6
fc759a1a
6cdc3a9f
95afbb5f
0c8a8a16
16699d43
b7b1d1a8
d22f2d08
da3f727d
e8de8b47
```

## 8. Helper Rows

Helper rows should not override same-operator evidence. They are used to infer
the row-level pieces:

```text
motif
output_mode
negative rendering
```

Example:

```text
The helper `-` examples support BA_DC|rev because:
64-65:
64 -> 46, 65 -> 56
|46-56|=10
rev(10)=01, padded/rendered as 201 under the detected output mode
```

If a helper row uses a different visible operator, its arithmetic rule may
differ. Only reuse its motif and output mode unless the trace explicitly
justifies sharing the base operation.

## 9. Absent Query Operator

If the query operator does not appear in the examples:

1. infer the row motif and output mode from visible operators
2. choose a conservative base operation using RHS length and visible operator
   patterns
3. avoid exotic operations
4. mark low confidence if multiple outputs remain possible

A good trace should say:

```text
No example uses query operator +.
Visible rows consistently use BA_DC|rev.
For absent +, try the common addition-family rule x+y first.
```

## 10. Preferred Trace Skeleton

Use this compact flow for training traces:

```text
Apply numeric-equation transformation search.
- Query is 85/77; query operator is /.
- Same-operator examples: 82/15=8241.

Test direct templates first.
- template0134: 82/15 -> 8215 vs 8241. FAIL.
- template3401: 82/15 -> 1582 vs 8241. FAIL.

Try BA_DC first.
- Same-operator RHS is 8241, length 4, so use multiplication family.
- Try x*y, then x*y+1, then x*y-1.

Try BA_DC with x*y for operator /.
82 -> 28, 15 -> 51
28*51=1428

Output-mode table:
- plain: 1428, no
- rev: 8241, yes
- op_prefix: /1428, no
- op_suffix: 1428/, no
- op_prefix_if_neg: 1428, no
- op_suffix_rev_if_neg: 1428, no
- op_prefix_rev: /8241, no
- rev_or_op_prefix_rev_if_neg: 8241, yes
- rev_or_op_suffix_if_neg: 8241, yes
- neg: -1428, no
- neg_rev: -8241, no
- abs_rev: 8241, yes

Use helper/query evidence if needed. Here rev gives the same query rendering as the other positive-only rev variants.
rev(1428)=8241
PASS

All same-operator examples pass, so use BA_DC|x*y|rev.

Query:
85/77
85 -> 58, 77 -> 77
58*77=4466
rev(4466)=6644

Answer: \boxed{6644}
```

## 11. What To Avoid

Avoid traces that:

1. jump directly to the final rule without showing motif and operation tests
2. list many irrelevant rare operations before the RHS-length family is used
3. use helper rows to assert a query operation without same-operator evidence
4. ignore operator prefix/suffix evidence
5. drop negative signs without explaining output mode
6. say "locked" before the candidate passes same-operator checks
