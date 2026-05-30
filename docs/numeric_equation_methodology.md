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

When common output formats disagree because the query computation is negative,
do not decide this by voting. Use the motif/base fallback policy:

- `AB_CD|x-y`: use operator prefix on the magnitude, e.g. `10&32` gives
  `10-32=-22`, so render `&22`.
- `AB_CD|y-x`: use operator suffix on the magnitude, e.g. `32&10` gives
  `10-32=-22`, so render `22&`.
- `BA_DC|x-y`: reverse the magnitude and use operator prefix, e.g. `10$32`
  gives `01-23=-22`, reverse `22 -> 22`, then render `$22`.
- `BA_DC|y-x`: reverse the magnitude and use operator suffix, e.g. `32$10`
  gives `01-23=-22`, reverse `22 -> 22`, then render `22$`.

For training traces, prefer a non-minus query operator in these fallback cases.
If the query operator is `-`, operator-prefix rendering can collapse into a
normal negative sign and becomes a weak teaching signal.

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
