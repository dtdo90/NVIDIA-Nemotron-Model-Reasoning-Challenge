# Symbol Transform Methodology

This note records the detailed methodology we want to teach the model for the
`Transformation Rules / symbol_transform` subtype.

The goal is not only to solve more rows programmatically. The goal is to teach
the model a reusable reasoning habit:

1. recognize direct symbol templates first
2. otherwise treat the row as encrypted digit equations
3. lock a small motif and operator rule from same-operator examples
4. use remaining examples only when the query still has unmapped symbols
5. solve digit constraints systematically using unit congruences and bijection

## 1. Problem Shape

Each row contains examples of the form:

```text
ABOCD = RHS
```

where:

1. `A` and `B` are the two symbols in the left operand
2. `O` is the operator character
3. `C` and `D` are the two symbols in the right operand
4. `RHS` is the transformed output

The query is another expression of the same shape:

```text
ABOCD
```

The central task is to determine the output symbol string for the query.

Important assumptions for the encrypted numeric branch:

1. digit symbols are bijective: one symbol maps to one digit, and one digit maps
   to one symbol
2. different operator characters can mean different arithmetic rules
3. the same row often shares the same operand motif and output rendering motif
4. the arithmetic operation is usually local to the operator character, not
   global to the whole row

The most important correction is:

```text
same motif across operators does not imply same arithmetic operation
```

For example, one row may use:

```text
* -> BA_DC | x*y | rev
+ -> BA_DC | x+y+1 | rev
- -> BA_DC | x-y | rev
```

The shared object is `BA_DC | ... | rev`; the base rule changes by operator.

## 2. Parse And Group By Operator

First parse every example into:

```text
left operand symbols: A B
operator symbol: O
right operand symbols: C D
RHS text
```

Then identify the query operator:

```text
query operator = query[2]
```

Collect same-operator examples:

```text
same_operator_examples = all examples whose operator equals query[2]
```

These are the strongest evidence for the query rule.

Use other-operator examples later only for:

1. row motif evidence
2. completing missing symbol-to-digit assignments
3. checking consistency

Do not start with unrelated operators if same-operator examples exist.

## 3. Stage 1: Direct Template Search

Before doing any numeric reasoning, test direct positional templates. These are
pure symbol rearrangements, not arithmetic.

The trusted direct templates are:

```text
0134: ABOCD -> ABCD
3401: ABOCD -> CDAB
```

For `0134`, take positions `0,1,3,4` from the input:

```text
ABOCD -> ABCD
```

For `3401`, take positions `3,4,0,1`:

```text
ABOCD -> CDAB
```

Procedure:

1. take same-operator examples
2. test `0134`
3. if all same-operator examples match, lock `0134` and solve the query
4. otherwise test `3401`
5. if all same-operator examples match, lock `3401` and solve the query
6. otherwise move to encrypted numeric reasoning

This stage absorbs most concat-like behavior. Therefore the later fallback
should not use `concat(x,y)` or `concat(y,x)` as a default rescue. If direct
templates failed, concat is more likely to create spurious numeric explanations.

## 4. Stage 2: Encrypted Numeric Motif Search

If direct templates fail, treat symbols as encrypted digits.

The two main motifs to try first are:

```text
BA_DC | operation | rev
AB_CD | operation | raw
```

Here `raw` means writing the numeric result directly, and `rev` means reversing
the numeric result text before encoding it as symbols.

### 4.1 BA_DC

For an input:

```text
ABOCD
```

`BA_DC` means:

```text
left numeric operand  = BA
right numeric operand = DC
```

Example:

```text
34?25
```

under `BA_DC` means:

```text
x = 43
y = 52
```

### 4.2 AB_CD

For an input:

```text
ABOCD
```

`AB_CD` means:

```text
left numeric operand  = AB
right numeric operand = CD
```

Example:

```text
34?25
```

under `AB_CD` means:

```text
x = 34
y = 25
```

### 4.3 Rev Output

If raw numeric result is:

```text
1050
```

then `rev` output is:

```text
0501
```

Leading zero is meaningful. Do not drop it. It must be encoded back to its
symbol.

## 5. Use RHS Length To Choose Operation Family

The RHS length is a strong operation-family clue.

Use these rules as the first branching guide:

```text
RHS length 4 -> multiplication family
RHS length 2 -> addition or subtraction family
RHS length 1 -> subtraction family
```

More explicitly:

### 5.1 RHS Length 4

Two 2-digit operands producing 4 output characters strongly suggests:

```text
x*y
x*y+1
x*y-1
```

Try the exact multiplication rule first. If it fails, try the `+1` and `-1`
variants.

### 5.2 RHS Length 2

Two 2-digit operands producing 2 output characters usually suggests addition or
subtraction:

```text
x+y
x+y+1
x+y-1
x-y
y-x
|x-y|
```

Use the visible operator character as a weak hint, but keep the full
addition/subtraction family available:

1. if the operator is `+`, try `x+y`, `x+y+1`, `x+y-1` first, then `x-y`, `y-x`, `|x-y|`
2. if the operator is `-`, try subtraction rules early, but still keep the addition variants available
3. stop as soon as a rule locks against the same-operator examples

### 5.3 RHS Length 1

One output character usually suggests subtraction family:

```text
x-y
y-x
|x-y|
```

This happens when the difference is a one-digit number.

For the narrow length-1 rescue, keep the query operator restricted to this
subtraction family. Then use helper rows to complete the map. If a helper row
has length-3 RHS, try the clean rule before variants:

```text
x+y
x*y
x+y+1
x+y-1
x*y+1
x*y-1
```

The trusted rescue requires at least one length-3 helper row to lock `x+y`.
This avoids noisy cases where a one-symbol subtraction row plus only length-2
or length-4 helpers creates a unique but wrong answer.

### 5.4 Different Characters Mean Different Operations

Never assume the visible character has its normal meaning. Instead, use it as a
priority hint.

For example:

```text
+ may mean x+y+1
- may mean x-y
* may mean x*y
```

The important point is that the row can share the motif while each operator has
its own base rule.

## 6. Fallback Operation Priority

After direct templates fail, exclude concat from the arithmetic rescue branch.

The practical rescue operation families are:

### 6.1 Under BA_DC | ... | rev

Use this priority for same-operator examples or missing-symbol rescue:

```text
x+y
x*y
x-y
x*y-1
x+y-1
x+y+1
x*y+1
|x-y|
y-x
```

This is based on the numeric-equation statistics, with concat removed because
template behavior is already checked by `0134` and `3401`.

Keep `|x-y|` as a real subtraction-family rule. It is not always replaceable by
one fixed direction: some rows require the direction to flip across examples,
while the visible operator still behaves like absolute difference. Treat this
as a genuine rule, but keep it behind the directed subtraction rules so we only
use it when `x-y` and `y-x` cannot explain the row cleanly.

When the visible operator suggests a family, use the family-aware priority
inside the motif:

```text
visible *: x*y, x*y+1, x*y-1
visible +: x+y, x+y+1, x+y-1
visible -: x-y, y-x, |x-y|
```

### 6.2 Under AB_CD | ... | raw

Use this priority after excluding concat:

```text
x-y
|x-y|
x*y
x+y
x*y+1
x+y+1
x+y-1
x*y-1
y-x
```

Again, the visible operator can reorder the local family:

```text
visible *: x*y, x*y+1, x*y-1
visible +: x+y, x+y+1, x+y-1
visible -: x-y, y-x, |x-y|
```

## 7. Fit Same-Operator Examples

For a candidate motif and operation:

1. translate every same-operator example into numeric form
2. enforce symbol-to-digit bijection
3. compute the numeric result
4. render the result as `rev` or `raw`
5. compare the rendered digit shape against RHS symbols
6. reject on the first mismatch

Keep a candidate only if it reproduces all same-operator examples.

A candidate fails immediately if:

1. one symbol would need two different digits
2. two symbols would need the same digit
3. the numeric result has the wrong output length
4. the rendered output does not match RHS symbols

### 7.1 Human-Faithful Trace Standard

For training data, solver correctness is not enough. The trace must be
human-method faithful.

That means a failed candidate must receive the same kind of work as the
candidate that eventually locks. Do not write only:

```text
Rule x+y: survivors=0. FAIL.
```

and do not replace the reasoning with candidate counts such as:

```text
row1 leaves 24 candidates; row2 leaves 0
```

unless the trace has already shown the actual equation setup, unit congruence,
branch grid, filtering step, and contradiction that makes the branch fail.

For every candidate tried in S3:

1. write the same variable assignment style used for the successful branch
2. write the numeric equation under the fixed motif
3. write the unit congruence
4. branch with the same grid/table style used for successful branches
5. filter against the next same-operator example when needed
6. show the exact contradiction or empty grid
7. only then write `FAIL`

The successful branch should not be the only branch with full derivation. If
`x*y` is tried before `x*y+1`, the `x*y` failure must be shown with comparable
detail before `x*y+1` is locked. If `x+y` fails before `x+y+1`, the `x+y`
failure must be worked through in the same table/grid style.

The same rule applies in missing-symbol rescue. If an other-operator row is
used with `x-y`, but `x+y` and `x+y+1` would be tried first by the priority
order, the trace must show those earlier rules failing before using `x-y`.

## 8. Unit-Congruence Search

When the digit map is unknown or partially known, solve the equations by
starting from the units digit. This is the main search-space reduction.

Do not present the trace as magical brute force. The model should learn this
sequence:

1. write symbol variables
2. write the numeric equation under a fixed motif
3. use the unit digit congruence
4. branch only on values satisfying the congruence
5. substitute each branch back into the full equation
6. simplify, often by subtracting known terms and dividing by 10
7. use the next units digit or digit-shape condition
8. continue until the assignment is locked or rejected

### 8.1 Multiplication Under BA_DC | rev

For:

```text
AB*CD = WXYZ
```

under:

```text
BA_DC | x*y | rev
```

we have:

```text
x = BA
y = DC
raw product = ZYXW
```

So the first output symbol `W` is the units digit of the raw product.

If:

```text
A=a, B=b, C=c, D=d, W=w
```

then:

```text
x = 10b + a
y = 10d + c
unit condition: a*c = w mod 10
```

For `x*y+1`:

```text
a*c + 1 = w mod 10
```

For `x*y-1`:

```text
a*c - 1 = w mod 10
```

This unit condition should be applied before trying full assignments.

### 8.2 Multiplication Under AB_CD | raw

For:

```text
AB*CD = WXYZ
```

under:

```text
AB_CD | x*y | raw
```

we have:

```text
x = AB
y = CD
raw product = WXYZ
```

The final output symbol `Z` is the units digit of the product.

If:

```text
A=a, B=b, C=c, D=d, Z=z
```

then:

```text
x = 10a + b
y = 10c + d
unit condition: b*d = z mod 10
```

For `x*y+1`:

```text
b*d + 1 = z mod 10
```

For `x*y-1`:

```text
b*d - 1 = z mod 10
```

### 8.3 Addition Under BA_DC | rev

For:

```text
AB+CD = WXYZ
```

under:

```text
BA_DC | x+y | rev
```

we have:

```text
x = BA
y = DC
raw sum = ZYXW
```

The first output symbol `W` is the units digit of the raw sum:

```text
a+c = w mod 10
```

For `x+y+1`:

```text
a+c+1 = w mod 10
```

For `x+y-1`:

```text
a+c-1 = w mod 10
```

After the unit digit, carry behavior matters. Substitute the possible unit
assignments and simplify the remaining tens/hundreds equation.

### 8.4 Subtraction Under BA_DC | rev

For:

```text
AB-CD = RHS
```

under:

```text
BA_DC | x-y | rev
```

we have:

```text
x = BA
y = DC
raw difference = reverse(RHS)
```

If the RHS is a normal digit string, use the unit of the raw difference:

```text
a-c = unit(raw difference) mod 10
```

For subtraction, negative outputs can appear. If the RHS contains the operator
symbol, it may indicate a special negative renderer such as:

```text
operator-prefix negative
operator-suffix reversed negative
```

In the missing-symbol fallback, do not overuse negative renderers. First try the
plain subtraction family. Use special negative formats only when the RHS visibly
contains the operator symbol or the row motif already supports that behavior.

## 9. Missing-Symbol Rescue

Sometimes same-operator examples lock the motif and most of the digit map, but
the query still cannot be encoded. There are two versions of this problem:
the query operands may contain unmapped symbols, or the query operands may be
known but the numeric result needs a digit whose output symbol is still unknown.
In either case, do not guess the missing symbols. Use the remaining examples.

Procedure:

1. keep the locked motif fixed
2. keep the locked query-operator rule fixed
3. identify unknown query operand symbols and unknown output digits
4. assign variables to unknown symbols
5. use other-operator examples to solve for those variables
6. try exact-looking operation first for each other operator
7. if exact fails, try `+1` or `-1`
8. reject any assignment that violates bijection
9. once missing symbols are known, return to the query and solve it

This step comes before arbitrary query completion. If same-operator examples
leave a query operand symbol or required output digit unknown, do not pick a
remaining digit just because it makes an answer. First ask whether another
equation in the same row determines that symbol.

Other equations are optional map-completion constraints, not an all-or-nothing
global fit. If one other-operator group determines the needed symbol but a
different group does not fit the same motif cleanly, keep the useful group and
do not discard the same-operator lock.

This is especially important in multiplication-shaped rows. A same-operator
`*` family can determine the query product, while another equation is needed
only to learn which symbol encodes one digit of that product. For now, treat
this output-digit completion as safe only for multiplication-family locks; the
same idea produced false unique answers when applied broadly to add/sub rows.

### 9.1 Worked Pattern

Suppose same-operator `*` examples lock:

```text
# = 4
$ = 8
% = 7
& = 2
? = 6
] = 1
```

and the row has unknown symbols:

```text
" = a
\ = b
@ = c
( = d
```

with remaining digits:

```text
{a,b,c,d} = {0,3,5,9}
```

If the locked motif is:

```text
BA_DC | ... | rev
```

then an example:

```text
]%-"] = &@
```

becomes:

```text
]% -> 71
"] -> 1a
&@ reversed -> c2
```

So:

```text
71 - 1a = c2
```

Try `-` as addition:

```text
71 + 10 = 81
71 + 13 = 84
71 + 15 = 86
71 + 19 = 90
```

No result has form `c2`, so reject addition for this operator.

Try `-` as subtraction:

```text
a=0: 71 - 10 = 61, not c2
a=3: 71 - 13 = 58, not c2
a=5: 71 - 15 = 56, not c2
a=9: 71 - 19 = 52, so c=5
```

So:

```text
a = 9
c = 5
```

Another example:

```text
$%+"\ = $]]
```

becomes:

```text
$% -> 78
"\ -> ba
$]] reversed -> 118
```

So:

```text
78 + ba = 118
```

Try exact addition:

```text
ba = 118 - 78 = 40
```

This is impossible because `4` is already used by `#`, and it also contradicts
`a=9`.

Try `x+y+1`:

```text
78 + ba + 1 = 118
ba = 39
```

So:

```text
b = 3
a = 9
```

Now the remaining digit is:

```text
d = 0
```

Thus:

```text
" = 9
\ = 3
@ = 5
( = 0
```

If the query is:

```text
(@*]&
```

under `BA_DC`:

```text
(@ -> @(
]& -> &]
```

So:

```text
@( = 50
&] = 21
```

Apply the locked query rule:

```text
50 * 21 = 1050
rev(1050) = 0501
```

Encode back:

```text
0 -> (
5 -> @
0 -> (
1 -> ]
```

Final:

```text
(@(]
```

This is the canonical missing-symbol rescue pattern.

### 9.2 Guarded One-Example Global Rescue

The normal encrypted numeric search should require at least two examples of
the query operator. A single same-operator row is often coincidental. However,
there is one conservative exception.

Use a one-example global rescue only when all of the following are true:

1. direct templates `0134` and `3401` already failed
2. the query operator appears in exactly one example
3. all examples fit one of the dominant motifs: `BA_DC|rev` or `AB_CD|raw`
4. the fitted row determines a complete 10-symbol digit map
5. at least four examples support the same global fit
6. every surviving candidate gives the same query output

If any of these conditions fails, do not choose by longest output, largest
value, or public-answer agreement. Mark the row as unresolved or oracle-low
confidence instead.

## 10. CoT Trace Style To Teach

The CoT should be detailed enough for the model to learn the method, not just
the answer.

Use this structure:

```text
We need to deduce the hidden symbol transformation rule by matching the example outputs.
I will put my final answer inside \boxed{}.

Step 1: Parse query operator and collect same-operator examples.
Step 2: Try direct templates 0134 and 3401.
Step 3: Since direct templates fail, treat the row as encrypted numeric equations.
Step 4: Try BA_DC|...|rev first.
Step 5: Use RHS length to choose the operation family.
Step 6: Use unit congruence to reduce digit assignments.
Step 7: Lock the motif/rule if all same-operator examples agree.
Step 8: If query has unmapped symbols, solve them from remaining examples.
Step 9: Apply the locked query rule and encode the result.
The final answer is \boxed{...}
```

When showing branch search, use compact repeated lines rather than prose-only
claims:

```text
Try a=0: ...
FAIL because ...
Try a=3: ...
FAIL because ...
Try a=9: ...
PASS, so ...
```

Do not write:

```text
All other cases fail.
```

unless the trace has already shown a clear unit-congruence reason that removes
the other cases.

## 11. Confidence Policy

Use high-confidence rows when:

1. direct template is verified on all same-operator examples
2. encrypted numeric motif/rule is verified on all same-operator examples and
   gives a unique query output
3. missing-symbol rescue uses remaining examples to determine the query symbols
   uniquely

Use lower-confidence or oracle-derived labels when:

1. multiple digit maps satisfy all examples but give different query outputs
2. a required output symbol never appears in the prompt
3. the rule requires choosing among multiple equally valid assignments using the
   public gold answer
4. the row needs an output format not yet established by examples

Rows with ambiguity should not be used as deterministic high-confidence traces.

Family-rescue ranking should not be used. In several real rows, the gold answer
appears inside the candidate set, but simple tie-breaks such as longest output
or largest numeric value choose the wrong candidate. Therefore:

1. use `family_rescue_unique` only when every rescue candidate gives the same
   query output
2. keep `family_rescue_ambiguous` as an unresolved multi-answer set
3. if the public gold answer is inside that set, mark the row as
   low-confidence/oracle-aligned for training analysis
4. do not generate deterministic CoT from `family_rescue_ambiguous` rows unless
   we manually add a new rule that removes the ambiguity

## 12. What Not To Teach As A Core Rule

Avoid teaching:

1. arbitrary positional templates beyond `0134` and `3401`
2. relaxed non-bijective digit maps
3. concat as a late fallback after direct templates fail
4. BA_DC|rev guesses that only fit one example and do not determine the query
5. answer-symbol assignments that cannot be inferred from prompt examples
6. compressed CoT that says "solve by brute force" without showing unit
   constraints or branch rejection

The model should learn a disciplined search:

```text
direct template -> motif -> operation family -> unit constraints -> bijective map -> query
```

## 13. Representative Failure Modes

These are the main gaps found after inspecting unsolved or wrongly solved rows.

### 13.1 Single Same-Operator Example

Example id: `00457d26`

The query operator has only one matching example. A gold-aware oracle can fit
`BA_DC | y-x | rev`, but the same single example admits many digit maps. The
other operator examples do not share a compatible global motif, so there is no
safe way to select the query output without using the public answer.

Policy: keep this unsolved unless another same-operator example or a stronger
row-global motif determines the map.

### 13.2 Unseen Answer Symbol

Example id: `00c032a8`

The same-operator multiplication examples do not fully solve the row by
themselves, but the other equations matter. Under `BA_DC | ... | rev`, the
second equation fixes the subtraction-family branch and gives `\ = 9`; the
first equation is consistent with the addition branch. This lets us infer the
query's numeric result as:

```text
))!\) -> 11 * 19 = 209 -> rev = 902
```

The remaining issue is narrower: digit `0` has no observed symbol in the local
prompt. The gold answer uses `^`, but `^` never appears in the examples or
query, so the local puzzle determines the digit sequence `902` but not the
specific unseen symbol used for `0`.

Policy: use other equations to complete the numeric map whenever possible, but
do not force an unseen answer-symbol assignment unless a separate problem-level
alphabet prior is introduced.

### 13.3 Gold Inside Candidate Set But Tie-Break Is Unsafe

Examples: `02664ad5`, `26a2a1b8`, `7b3d06f7`

Family rescue can fit a dominant motif and produce a candidate set containing
the gold answer. However, simple tie-breaks such as longest output or largest
numeric value solve some rows and fail others.

Policy: `family_rescue_unique` is acceptable. If multiple variants remain, do
not rank them by length or numeric value. Mark the row as
low-confidence/oracle-aligned only when the public gold answer is one of the
variants.

### 13.4 Missing Output Format

Examples: `7db5c1af`, `fdbdf50c`

The current core output modes can produce a unique but wrong answer when the
true row uses an output renderer such as `last_rev` or
`op_prefix_if_neg`. Broadening to the format bank often exposes the ambiguity
but does not always give a unique answer.

Policy: use broader output-format checks as a veto or analysis tool, not as a
default high-confidence prediction rule until the format is inferred from the
examples.
