# Symbol Transform Methodology

This note records the detailed methodology we want to teach the model for the
`Transformation Rules / symbol_transform` subtype.

The goal is not only to solve more rows programmatically. The goal is to teach
the model a reusable reasoning habit:

1. recognize direct symbol templates first
2. otherwise treat the row as encrypted digit equations
3. determine a small motif and operator rule from same-operator examples
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

Nuance from the audited hard traces: "same-operator first" means the trace
should inspect and write the same-operator equations first. If a same-operator
equation has too many active variables for a useful visible grid, do not hide a
brute-force solution. State why the row is deferred, use other operators to
reduce the map, and then return to the same-operator equation once fewer
variables remain.

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
3. if all same-operator examples match, use `0134` and solve the query
4. otherwise test `3401`
5. if all same-operator examples match, use `3401` and solve the query
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
RHS length 3 -> addition or multiplication family
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

### 5.2 RHS Length 3

Three output characters suggest either addition family or multiplication
family:

```text
x+y
x+y-1
x+y+1
x*y-1
```

Use the numeric-equation success order unless the visible operator gives
stronger evidence. For the main `BA_DC|rev` route, try:

```text
x+y
x+y-1
x+y+1
x*y-1
```

This order differs from the older symbol-only heuristic because numeric-equation
rows give stronger solved evidence for length-3 RHS routing.

If the branch is addition-shaped, such as:

```text
ab+cd=efg
```

then the leading RHS digit can only be `1` or `0` for ordinary two-digit
addition. In the trace, try the leading digit `1` first:

```text
f=1
```

If that branch has no candidates, try:

```text
f=0
```

After choosing the leading digit, continue with the unit digit and derive the
next variable from the full equation. A compact reusable pattern is:

```text
unit condition -> one derived digit -> grid table -> survivor summary
```

Hard rule: when the equation is already split into LHS operands and RHS output
digits, try to solve LHS-side variables in terms of RHS-side variables whenever
the algebra allows it. Do not default to scanning LHS variables and computing
RHS digits if the unit condition can be inverted or branched from a RHS digit.
If the inverse is not unique, list the possible branches from the RHS digit and
then compute the remaining LHS variable.

Make this step explicit before deferring a row as too large. For example, if a
same-operator row gives:

```text
(10f+c)+(10a+e)=100e+10b+d
```

then record the length-3 addition-family fact:

```text
Because the RHS has length 3 and this is addition family, the leading digit value must be 1, so e=1.
```

Then solve with `e=1`:

```text
d=(c+1) mod 10
b=(10f+c+10a+1-100-d)/10
```

If this branch leaves many candidates, carry `e=1` forward into the helper
rows and keep filtering until the `e=1` branch is fully handled. Do not mention
or start the `e=0` branch while the `e=1` branch is still active.

Only if the `e=1` branch has no viable candidates after the available
same-operator and helper-row filtering should the trace start the `e=0` branch:

```text
d=c
```

If `d=c`, reject that branch by distinctness. This carry-plus-unit reduction
must be written down even if the remaining table is deferred to helper rows.

For example, if a same-operator row under `BA_DC|rev` gives:

```text
10b+10c+2a=100+10e+d
```

then:

```text
d=2a mod 10
```

do not immediately scan `a,c` and compute RHS digits. Since `d` is an RHS
digit, branch from `d` back to the possible `a` values:

```text
d=0 -> a=0 or 5
d=2 -> a=1 or 6
d=4 -> a=2 or 7
d=6 -> a=3 or 8
d=8 -> a=4 or 9
```

Then compute the remaining LHS variable, for example:

```text
c=(90+10e+d-2a)/10
```

The trace should show a visible grid over RHS-side `d,e` plus the branched
`a` values, with staged entries such as `[x]`, `[c,x]`, `[c,Ck]`. Do not hide
this as a black-box brute force, and do not use the older pattern that scans
only LHS variables first.

### 5.3 RHS Length 2

Two 2-digit operands producing 2 output characters usually suggests addition or
subtraction:

```text
x+y
x+y+1
x+y-1
x-y
y-x
abs(x-y)
```

For the `BA_DC|rev` route, use the accepted numeric-equation statistics for
length-2 addition/subtraction rows. The current priority is:

```text
x+y
x-y
abs(x-y)
x+y+1
y-x
x+y-1
```

Use the visible operator character as a local ordering hint only inside this
length-2 family. If the visible operator is `-`, try `x-y` first, then continue
through the remaining length-2 family rules if `x-y` fails. This exception is
especially useful for helper rows such as `AB-CD = EF`, where a visible minus
row often supplies the missing digit map. The trace should state the RHS
length, state the visible-operator ordering, and then try rules sequentially
until a rule leaves viable assignments.

### 5.4 RHS Length 1

One output character usually suggests subtraction family:

```text
x-y
y-x
abs(x-y)
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

The trusted rescue requires at least one length-3 helper row to determine `x+y`.
This avoids noisy cases where a one-symbol subtraction row plus only length-2
or length-4 helpers creates a unique but wrong answer.

### 5.5 Different Characters Mean Different Operations

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

Also enforce operator distinctness inside a row. Different visible operator
symbols should normally correspond to different arithmetic base rules. If one
operator has already been determined as a rule in a family, do not immediately
reuse the same rule for another operator symbol. Instead, move to the next
candidate in the family priority order.

For example, if `%']/% = ?)` has already established that operator `]` uses
`x-y`, then in `@"!") = ?`, operator `!` should not try `x-y` first. It should
start from the next subtraction-family candidate according to the learned
priority, such as `y-x` or `abs(x-y)`.

## 6. Fallback Operation Priority

After direct templates fail, exclude concat from the arithmetic rescue branch.

The practical rescue operation families are:

### 6.1 Under BA_DC | ... | rev

Use RHS length first, then use the matching family priority. The broad
`BA_DC|rev` popularity order is:

```text
x+y
x*y
x-y
x*y-1
x+y-1
x+y+1
x*y+1
abs(x-y)
y-x
```

This broad list is based on the numeric-equation statistics, with concat
removed because template behavior is already checked by `0134` and `3401`.
When the RHS length gives a narrower family, prefer the narrower family order.
For length-2 helper rows, use:

```text
x+y
x-y
abs(x-y)
x+y+1
y-x
x+y-1
```

Exception: for length-2 helper rows whose visible operator is `-`, try `x-y`
first. If it fails, continue sequentially through the remaining length-2 rules.
If multiple symbol-digit maps survive, keep the row only when they agree on the
exact final query output needed for the boxed answer.

Keep `abs(x-y)` as a real subtraction-family rule. It is not always replaceable by
one fixed direction: some rows require the direction to flip across examples,
while the visible operator still behaves like absolute difference. Treat this
as a genuine rule, but keep it behind the directed subtraction rules so we only
use it when `x-y` and `y-x` cannot explain the row cleanly.

There is one narrow tie-break for query rendering. If `x-y` fits the same
operator evidence and determines a valid map, but the query value under `x-y`
would be negative, do not invent a sign-dropping render. Under `raw` or `rev`,
negative query values are unsupported unless the examples explicitly show a
signed format. If `abs(x-y)` fits the same evidence with the same valid map, move
to `abs(x-y)` for the query and render the nonnegative value.

When the visible operator suggests a family, use the family-aware priority
inside the motif:

```text
visible *: x*y, x*y+1, x*y-1
visible +: x+y, x+y+1, x+y-1
visible -: x-y, y-x, abs(x-y)
```

### 6.2 Under AB_CD | ... | raw

Use this priority after excluding concat:

```text
x-y
abs(x-y)
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
visible -: x-y, y-x, abs(x-y)
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
candidate that eventually works. Do not write only:

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
detail before `x*y+1` is accepted. If `x+y` fails before `x+y+1`, the `x+y`
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
8. continue until the assignment is determined or rejected

For the revised CoT traces, make this order explicit in the wording:

```text
state current motif branch -> write symbolic equation -> use modulo 10 first
-> derive one digit or carry condition -> simplify or make a grid if needed
```

If one same-operator row leaves exactly one survivor under the current branch,
carry those values forward inside that branch. Do not phrase this as proof that
the whole operator rule is globally confirmed until the remaining same-operator or
helper evidence has also been checked.

When helper rows complete missing symbols, first write the conversion in labels
such as `10g+e`, then substitute known digits such as `e=9`. This keeps the
trace reusable instead of looking like a numerical shortcut.

### 8.1 Variable-Count Grid Grammar

All symbol-equation variables represent distinct digits from `0` to `9`.
Always start with the modulo-10 condition before deciding the table format.
Then choose the smallest visible grid that shows the actual derived digits and
the distinctness check.

For every table:

1. write the equation being solved
2. write the modulo-10 condition first
3. say which variable is derived from which free variables
4. list real computed values in the table body
5. use `x` only as the status marker for rejected branches
6. use `Ck` for first-stage survivors and `ok` for later filter survivors
7. do not use `n` as shorthand for "not an integer"; show the exact computed
   value, including fractions such as `46/11`, and mark the status as `x`

After a first-stage table creates candidates `C1`, `C2`, ..., later filter
tables may use those candidate labels as row names. This is a good compact
indexing pattern when the candidate already represents several fixed variable
values. The later table must still show the real derived values, for example
`[i,h,status]`, rather than replacing the row with a bare `x`.

When a later table uses candidate labels as rows, explicitly say what each
candidate fixes. For example:

```text
Each row C_i is a choice of a,c,e,f,h,i,j that passes row 1.
Each entry is [b,d,status]. x means reject; ok means this candidate passes row 2.
```

If the previous candidate row already fixes every variable except one scanned
digit, and the new row only tests whether the original equation holds, do not
introduce a residual helper variable such as `r`. Scan the remaining digit and
write the table as `x` or `ok`.

Example:

```text
Each row Ck gives fixed b,c,d,f,g,h. Scan a.
a: 0,1,2,3,4,5,6,7,8,9
C1: x,x,x,x,x,x,x,x,x,x
C2: x,ok,x,x,x,x,x,x,x,x
```

#### Equation In 2 Variables

If the equation has two active variables, derive one from the other:

```text
b = f(a)
```

Use one row of free values and one row of derived values:

```text
a: 0,1,2,3,4,5,6,7,8,9
b: ...
status: ...
```

Or, if candidate labels are needed:

```text
a: 0,1,2,3,4,5,6,7,8,9
entry: [b,status],...
```

#### Equation In 3 Variables

Most commonly, one variable is expressible in terms of the other two:

```text
c = f(a,b)
```

Use a two-dimensional table:

```text
a: 0,1,2,3,4,5,6,7,8,9
b=0: [c,status],...
b=1: [c,status],...
...
b=9: [c,status],...
```

If two variables are both determined by one variable:

```text
b = f(a)
c = g(a)
```

use aligned rows:

```text
a: 0,1,2,3,4,5,6,7,8,9
b: ...
c: ...
status: ...
```

#### Equation In 4 Variables

Most commonly, derive one variable from the other three:

```text
d = f(a,b,c)
```

Use two header rows and one body dimension:

```text
a: 0,0,...,0,1,1,...,1,...,9,9,...,9
b: 0,1,...,9,0,1,...,9,...,0,1,...,9
c=0: [d,status],...
c=1: [d,status],...
...
c=9: [d,status],...
```

If one variable is not computed by a formula and is simply a free scan digit,
say so explicitly before the table. For example, in
`(10b+a)(10b+c)=...`, `b` may be the scanned column rather than a derived
value. The trace should say:

```text
Here b is not computed by a formula. It is the free digit that we scan.
Each table column is one scanned value of b.
```

Then each cell computes the derived RHS/place-value digits in order.

If two variables are expressible in terms of two free variables:

```text
c = f(a,b)
d = g(a,b)
```

use:

```text
a: 0,1,2,3,4,5,6,7,8,9
b=0: [c,d,status],...
b=1: [c,d,status],...
...
b=9: [c,d,status],...
```

If three variables are expressible in terms of one variable:

```text
b = f(a)
c = g(a)
d = h(a)
```

use:

```text
a: 0,1,2,3,4,5,6,7,8,9
b: ...
c: ...
d: ...
status: ...
```

#### Equation In 5 Variables

If only one variable is expressible in terms of the other four:

```text
e = f(a,b,c,d)
```

do not immediately make a four-free-variable brute-force table. First use the
modulo-10 condition.

If modulo 10 uniquely determines one variable in terms of another, for example:

```text
b = 2a mod 10
```

then reduce the problem to:

```text
e = f(a,c,d)
```

and use:

```text
a: 0,0,...,0,1,1,...,1,...,9,9,...,9
b: b(a),b(a),...,b(a),...
c: 0,1,...,9,0,1,...,9,...,0,1,...,9
d=0: [e,status],...
d=1: [e,status],...
...
d=9: [e,status],...
```

If modulo 10 does not uniquely determine one variable, for example:

```text
a(b-1)=0 mod 10
```

then split into cases such as `a=0` and `a!=0`, or enumerate the allowed
`(a,b)` pairs. Continue from the reduced case. Do not introduce an extra
helper variable if it only hides the case logic.

If two variables are expressible in terms of three free variables:

```text
d = f(a,b,c)
e = g(a,b,c)
```

use:

```text
a: 0,0,...,0,1,1,...,1,...,9,9,...,9
b: 0,1,...,9,0,1,...,9,...,0,1,...,9
c=0: [d,e,status],...
c=1: [d,e,status],...
...
c=9: [d,e,status],...
```

If three variables are expressible in terms of two free variables:

```text
c = f(a,b)
d = g(a,b)
e = h(a,b)
```

use:

```text
a: 0,1,2,3,4,5,6,7,8,9
b=0: [c,d,e,status],...
b=1: [c,d,e,status],...
...
b=9: [c,d,e,status],...
```

If four variables are expressible in terms of one free variable, use aligned
rows as in the lower-variable cases.

#### Large But Printable Versus Too Large

A `10^3` scan is acceptable when it can be shown as a structured table, for
example:

```text
a: 0,...,0,...,9,...,9
b: 0,...,9,...,0,...,9
c=0: ...
...
c=9: ...
```

Use this style often; many hard rows need it. The table is large, but it still
teaches the reusable method and keeps the search visible.

Do not print a blind `10^4` or larger scan unless the unit condition and digit
extraction reduce it to a compact table. If the equation remains too large,
write the equation, write the unit condition, count the remaining independent
variables after reduction, and then defer it explicitly.

Use wording like:

```text
After unit and place-value reduction, the equation still has four independent variables c,d,e,f, so the 10^4 scan is too large to print.
Skip this row for now and use other operator rows to reduce the map first.
```

This is a deferral, not a rejection. The trace should later return to the
deferred same-operator row after helper equations determine enough variables.

If a six-variable equation is actually reducible by RHS digit extraction, show
that extraction instead of skipping. For example, when the RHS variables do not
appear on the LHS, compute the RHS digits directly from the raw LHS value and
table the derived tuple.

### 8.2 RHS Digit Extraction Without Helper Variables

When an equation has RHS variables that do not appear on the LHS, compute the
RHS digits directly from the raw LHS value. This is often clearer than creating
temporary helper variables.

General rule:

```text
raw = numeric value from the LHS after choosing the motif and arithmetic rule
RHS shape = digit variables written in place-value order before output rendering
```

Use modulo for the units digit, floor for leading digits, and subtraction for
middle digits.

For a 3-digit RHS:

```text
raw = 100d + 10e + f
f = raw mod 10
d = floor(raw / 100)
e = (raw - 100d - f) / 10
```

Example:

```text
ab * ac = def
raw = (10a+b)(10a+c)
f = raw mod 10
d = floor(raw / 100)
e = (raw - 100d - f) / 10
```

The table entries should follow the computation order:

```text
[f,d,e,status] if the unit digit is computed first
[d,f,e,status] if the trace first fixes a leading/output digit by shape
```

Choose the order that matches the written derivation. Do not hide how a digit
is obtained.

For a 2-digit RHS:

```text
raw = 10d + e
e = raw mod 10
d = floor(raw / 10)
```

For a 4-digit RHS:

```text
raw = 1000d + 100e + 10f + g
g = raw mod 10
d = floor(raw / 1000)
e = floor((raw - 1000d) / 100)
f = (raw - 1000d - 100e - g) / 10
```

For a 1-digit RHS:

```text
raw = d
d = raw
```

Always check after extraction:

1. every extracted value is an integer digit from `0` to `9`
2. extracted digits respect symbol-digit distinctness
3. recomposing the RHS shape gives the original raw value
4. the rendered output under `raw` or `rev` matches the RHS symbols

This rule is especially useful when the RHS symbols are new variables and the
LHS already determines the numeric value. It turns a vague "scan RHS symbols"
step into a visible table such as:

```text
a: 0,1,2,3,4,5,6,7,8,9
c=0: [f,d,e,status],...
c=1: [f,d,e,status],...
...
```

### 8.3 Preferred Wide-Grid Table Style

When one equation has at most about five unknown digit variables, prefer a
single visible wide-grid table instead of a hidden brute-force statement. The
pattern is:

1. write the exact numeric equation
2. use the unit congruence to compute one output variable
3. algebraically isolate another output variable from the full equation
4. enumerate the remaining free variables in a compact grid
5. put each computed entry as `[derived_digit,status]`

Use this style whenever the equation shape lets the trace derive enough
variables algebraically. It is especially useful for multiplication rows and
for length-3 addition rows after the carry digit is branched.

Use:

```text
x means reject
Ck means survivor candidate k
```

Reject immediately if:

1. the derived value is not an integer
2. the derived value is outside `0..9`
3. any symbols forced by the branch reuse the same digit
4. the full equation does not match the RHS digit shape

Canonical example from `0e009c6d`:

```text
>@'>> = )?>
```

Under `BA_DC|rev` and rule `x*y`, with:

```text
> = a
@ = b
) = c
? = d
```

the row becomes:

```text
>@ -> @> = 10b+a
>> -> >> = 11a
)?> has digit shape cda; before reverse it is adc = 100a+10d+c

(10b+a)(11a)=100a+10d+c
```

The unit digit gives:

```text
a*a ≡ c mod10
c = a^2 mod10
```

Then the full equation gives:

```text
110ab+11a^2=100a+10d+c
d=(110ab+11a^2-100a-c)/10
```

So the table should scan `a,b`, compute `c` from the unit condition, compute
`d` from the full equation, and mark whether `a,b,c,d` are distinct digits:

```text
a: 0,1,2,3,4,5,6,7,8,9
c: 0,1,4,9,6,5,6,9,4,1
b=0: [0,x],[-9,x],[-16,x],[-21,x],[-23,x],[-22,x],[-19,x],[-17,x],[-10,x],[-1,x]
b=1: [0,x],[2,x],[6,C1],[12,x],[21,x],[33,x],[47,x],[60,x],[78,x],[98,x]
...
b=9: [0,x],[10,x],[182,x],[276,x],[373,x],[472,x],[569,x],[676,x],[782,x],[890,x]
```

Only `C` rows are carried forward into the next same-operator equation.

For a more general shape such as:

```text
ab * ac = bcde
```

the unit digit gives:

```text
b*c ≡ e mod10
e = b*c mod10
```

Then derive `d` from the full product after `a,b,c,e` are fixed. A compact grid
can use the first rows as branch headers and each body entry as `[d,status]`:

```text
b: 0,1,2,3,4,5,6,7,8,9
c: 0,1,2,3,4,5,6,7,8,9
e: 0,1,4,9,6,5,6,9,4,1
a=0: [d,status],...
a=1: [d,status],...
...
a=9: [d,status],...
```

This is still a grid search, but it is not a blind `10!` search. The trace
shows the reusable reduction: unit congruence first, derived digit second,
bijection check third, survivor carry-forward last.

For length-3 addition, use the same idea. For example:

```text
ab+cd=efg
```

First branch on the carry digit. For ordinary two-digit addition, the leading
digit can only be `1` or `0`, so try:

```text
e=1, then e=0 if e=1 has no candidates
```

Then use the unit digit:

```text
b+d ≡ g mod10
g = (b+d) mod10
```

and derive the tens digit from the full equation. A compact grid can list the
free digit choices and use entries like `[f,g,status]` or `[f,status]`,
depending on which digit is derived first. The important point is the same:
carry branch, unit congruence, derived digit, bijection check, survivor list.

### 8.4 Multiplication Under BA_DC | rev

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

### 8.5 Multiplication Under AB_CD | raw

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

### 8.6 Addition Under BA_DC | rev

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

### 8.7 Subtraction Under BA_DC | rev

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

Sometimes same-operator examples determine the motif and most of the digit map, but
the query still cannot be encoded. There are two versions of this problem:
the query operands may contain unmapped symbols, or the query operands may be
known but the numeric result needs a digit whose output symbol is still unknown.
In either case, do not guess the missing symbols. Use the remaining examples.

Procedure:

1. keep the determined motif fixed
2. keep the determined query-operator rule fixed
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

After the equations determine all but one symbol-digit assignment, using the
last unused digit for the last unused symbol is allowed. This is not guessing;
it follows from the bijection constraint. State it plainly:

```text
Only one symbol and one digit remain, so assign the unused digit by bijection.
```

Other equations are optional map-completion constraints, not an all-or-nothing
global fit. If one other-operator group determines the needed symbol but a
different group does not fit the same motif cleanly, keep the useful group and
do not discard the same-operator evidence.

This is especially important in multiplication-shaped rows. A same-operator
`*` family can determine the query product, while another equation is needed
only to learn which symbol encodes one digit of that product. For now, treat
this output-digit completion as safe only for multiplication-family determinations; the
same idea produced false unique answers when applied broadly to add/sub rows.

### 9.1 Worked Pattern

Suppose same-operator `*` examples determine:

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

If the determined motif is:

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

Apply the determined query rule:

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
3. all examples fit one of the dominant motifs: `BA_DC|rev` or `AB_CD|plain`
4. the fitted row determines a complete 10-symbol digit map
5. at least four examples support the same global fit
6. every surviving candidate gives the same query output

If any of these conditions fails, do not choose by longest output, largest
value, or public-answer agreement. Mark the row as unresolved or oracle-low
confidence instead.

### 9.3 Absent Query-Operator Hypothesis

Sometimes the query operator does not appear in any example. In this case
there are no same-operator examples for the query itself.

Do not force the normal same-operator flow. Instead:

1. test direct templates first as usual
2. if direct templates fail, choose a motif route, starting with `BA_DC|rev`
   and then `AB_CD|plain`
3. solve the visible example operators under that motif
4. assign different arithmetic rules to different visible operator symbols
5. treat the query operator as an unseen operator whose rule must be different
   from the visible operator rules

For each visible example operator, use its RHS lengths to choose the family:

```text
RHS length 4 -> multiplication family, prefer x*y
RHS length 3 -> addition or multiplication family, try x+y, x+y-1, x+y+1, then x*y-1
RHS length 2 -> addition or subtraction family, prefer x+y unless the visible operator is -, then try x-y first
RHS length 1 -> subtraction family, prefer x-y
```

After the visible operators are aligned, the absent query operator may be tried
with the most common still-unused rule. This is only a hypothesis. Public-train
experiments showed that a simple unique absent-operator guess is noisy: it
often gives a unique but wrong answer even when the digit map is complete.

Therefore, use absent-query-operator reasoning only as:

1. a low-confidence candidate generator
2. a manual analysis route when the visible examples determine nearly the whole
   map
3. a deterministic solver branch only if additional constraints make every
   surviving valid map give the same query output

Do not make a high-confidence trace from this branch merely because the guessed
rule is the most common unused rule.

## 10. CoT Trace Style To Teach

The CoT should be detailed enough for the model to learn the method, not just
the answer.

Use this structure:

```text
We need to deduce the hidden symbol transformation rule by matching the example outputs.

S0: Methodology: solve same-operator examples first; test direct templates first; if they fail, use encrypted digit search with BA_DC|rev or AB_CD|plain; choose the arithmetic family from same-operator RHS length; keep visible survivor grids; use other examples only to complete the map; use guarded output-format rescues only when their evidence is explicit; then solve the query.
S1: Classify this as Symbol Transform with fixed shape ABOCD.
S2: Test direct-position templates first.
S2.1: Direct templates failed, so assign global variables once.
S3: Try BA_DC|rev or AB_CD|plain from the same-operator RHS length.
S4: Try the selected arithmetic rule and solve same-operator equations with modulo-10-first tables.
S5: If needed, use other operators only to reduce or complete the map.
S6: Apply the query operator in digits.
All symbols in the query are now known.
S7: Decode.
Answer: \boxed{...}
```

The query-handoff sentence should appear once, directly under the step that
applies the query operator in digits. Do not repeat it elsewhere.

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

### 13.4 Guarded Output-Format Rescues

The full output-format bank is too noisy to use as a default. However, two
narrow submotifs have been useful when heavily guarded.

#### Signed Operator Marker

Use this only after direct templates fail.

```text
AB_CD | operation | op_prefix_if_neg
```

Meaning:

```text
if the numeric result is negative, prefix the visible operator character;
otherwise write the value directly.
```

Activation rule:

1. at least one example visibly uses its own operator as the first RHS symbol
2. the global fit is unique under `AB_CD|op_prefix_if_neg`
3. no helper operation is concat
4. the chosen helper rules are not all addition-family rules
5. at least 9 digit symbols are mapped
6. if the query rule is `x*y-1`, require all 10 digit symbols to be mapped

Public-train check after adding this guard:

```text
signed_operator_marker_global_unique: 27 rows, 27 exact
```

This branch recovered 18 rows that were previously missed or uniquely wrong.

#### Last-Digit Raw Output

Use this only as a late fallback when the standard solver returns `no_rule`.

```text
AB_CD | operation | last
```

Meaning:

```text
write the last k digits of the raw numeric result, where k is the RHS length
observed for that operator.
```

Activation rule:

1. exactly one same-operator example exists
2. at least one helper row has RHS length 3 or 4
3. query rule is addition-family or multiplication-family
4. no helper operation is concat
5. at least 9 digit symbols are mapped
6. helper rules include cross-family evidence, not only add or only mul

Public-train check after adding this guard:

```text
last_digit_global_unique: 3 rows, 3 exact
```

Do not enable broad `last_rev`, `op_suffix_rev_if_neg`, or all-format scans by
default. They produce many plausible-looking but wrong unique answers.

### 13.5 Guarded BA_DC Rev Global Rescue

The dominant `BA_DC|rev` motif can still recover additional rows, but only with
strict guards. Broad BA_DC guesses are unsafe because one same-operator example
can fit many digit maps.

Use this branch only as a late rescue after the standard solver has no
deterministic prediction.

```text
BA_DC | operation | rev
```

Activation rule:

1. same-operator RHS length is 3 or 4
2. the global fit is unique under `BA_DC|rev`
3. for exactly one same-operator row with RHS length 4, require multiplication
   family and all 10 digit symbols mapped
4. for exactly one same-operator row with RHS length 3, require `x+y` and at
   least 9 digit symbols mapped
5. for two or more same-operator rows with max RHS length 3, require `x+y` and
   at least 9 digit symbols mapped
6. no concat helper rule is allowed in the one-row branch

Public-train check after adding this guard:

```text
ba_dc_rev_guarded_global_unique: 5 rows, 5 exact
```

The recovered rows were:

```text
0e2d6796, 6c7231ac, b4b73143, cca882b8, ea04ed35
```
