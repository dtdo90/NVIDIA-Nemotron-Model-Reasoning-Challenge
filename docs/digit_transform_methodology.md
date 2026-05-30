# Digit Transform Methodology

This note records the current logical method for the `Transformation Rules / numeric_equation` subtype.

The goal is not only to solve rows programmatically, but also to capture a clean reasoning procedure that we can later teach to the model during SFT or Doc2LoRA-style knowledge injection.

Current stable solver status:

- dataset slice: `732` numeric-equation rows from [train.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
- current stable score: `550 / 732`
- current stable accuracy: `75.14%`

## Problem Type

Each row is a small operator language over 2-digit strings:

```text
AB ⊕ CD = OUTPUT
```

The hidden rule has three parts:

1. operand pairing
2. numeric base rule
3. output rendering format

The same row may contain several operators, and each operator may have its own base rule. What often stays shared inside the row is the transform motif:

```text
pairing + output format
```

Examples:

- `AB_CD | ... | plain`
- `BA_DC | ... | rev`

## Canonical Reasoning Procedure

### 1. Parse the Row as a Mini DSL

For every example and the query:

1. keep the left 2-character operand string
2. keep the operator character
3. keep the right 2-character operand string
4. keep the output as text, not just as an integer

This matters because:

- leading zeros matter
- reverse-text output matters
- operator-prefixed or operator-suffixed outputs matter

### 2. Group by Operator

Solve operator-by-operator.

Do not assume one arithmetic rule for the whole row.

The row-level information is still useful, but mainly as a motif prior:

- shared pairing
- shared output format
- sometimes shared arithmetic family

### 3. Enumerate the Three Hidden Dimensions

#### A. Pairing

We currently use four pairings:

1. `AB_CD`
2. `AB_DC`
3. `BA_CD`
4. `BA_DC`

Examples:

- `AB_CD`: `34 ? 25` means `34` and `25`
- `BA_DC`: `34 ? 25` means `43` and `52`

#### B. Base Rule

Current trusted base-rule family:

1. `x + y`
2. `x + y + 1`
3. `x + y - 1`
4. `x - y`
5. `y - x`
6. `|x - y|`
7. `x * y`
8. `x * y + 1`
9. `x * y - 1`
10. `concat(x, y)`
11. `concat(y, x)`
12. `x + y^2`
13. `(x + y)^2`
14. `(x - y)^2`
15. `gcd(x, y)`

In practice, the heavy hitters are much smaller:

1. `x * y`
2. `x + y`
3. `x + y + 1`
4. `x + y - 1`
5. `x * y + 1`
6. `x * y - 1`
7. `concat(x, y)`
8. `concat(y, x)`
9. subtraction family

#### C. Output Format

Current trusted output formats:

1. `plain`
2. `rev`
3. `neg`
4. `neg_rev`
5. `op_prefix`
6. `op_suffix`
7. `op_prefix_if_neg`
8. `op_prefix_rev`
9. `op_suffix_rev`
10. `op_suffix_rev_if_neg`

Important meanings:

- `rev`: reverse the output digit string, preserving zeros
- `op_prefix_if_neg`: positive outputs stay plain; negative outputs become `operator + magnitude`
- `op_suffix_rev_if_neg`: positive outputs stay plain; negative outputs become `reversed_magnitude + operator`

## Search Strategy

### 4. Use Same-Operator Examples First

If the query operator appears in the examples:

1. keep only examples with that operator
2. test candidate `(pairing | base rule | format)` combos on those examples
3. reject a combo as soon as it fails an example
4. keep only exact-fit candidates

This is the single most important discipline. We do not use unrelated operators first.

### 5. Scan Common Combos Before Exhaustive Search

The fast path is a frequency-ordered scan over common combos.

Examples near the head of the scan:

1. `BA_DC | x * y | rev`
2. `AB_CD | concat(x, y) | plain`
3. `BA_DC | x + y | rev`
4. `BA_DC | x + y + 1 | rev`
5. `BA_DC | x + y - 1 | rev`

This mirrors the reference `symbol-digit` strategy:

- check the highest-frequency combos first
- once a combo fits `EX1`, verify it on `EX2`
- only then keep it

If the fast scan misses, fall back to full exhaustive enumeration.

## Selection Logic

### 6. If Many Candidates Fit, Choose by Query-Output Agreement First

After fitting the operator examples, multiple candidates may still survive.

The strongest signal is:

1. compute the query output from every surviving candidate
2. if they all agree, the row is usually usable
3. if they disagree, the row is ambiguous unless we have additional motif evidence

This is more important than picking the “simplest-looking” rule.

### 7. Use Row-Level Motif Evidence as Tie-Break, Not as the First Move

When the query operator has only one example, row-level motif evidence becomes valuable.

We look at better-supported operators in the same row and ask:

- do they all suggest the same pairing?
- do they all suggest the same output format family?

Strong examples:

- `AB_CD | ... | plain`
- `BA_DC | ... | rev`

Then we prefer query-operator candidates that preserve that motif.

Example:

- [026106f5](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
- `{` examples support `BA_DC | ... | rev`
- the one-example `*` operator is then better explained by `BA_DC | x + y | rev`

### 8. For Absent Query Operators, Infer the Missing Rule from the Row

If the query operator never appears in the examples:

1. infer the dominant row motif from the visible operators
2. keep that motif fixed
3. choose the base rule by a structured prior, not by random guessing

Important heuristics that worked well:

#### A. Additive-plus-subtraction rows often imply a multiplicative unseen rule

If visible operators show:

- an additive variant such as `x + y + 1`
- plus a subtraction-family rule

then the unseen operator often belongs to the analogous multiplicative family.

Example:

- [078df00e](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
- visible motif: `BA_DC | ... | rev`
- visible `+`: `x + y + 1`
- visible `-`: subtraction family
- unseen `*` is best solved as `BA_DC | x * y + 1 | rev`

#### B. Unsigned rows prefer unsigned subtraction

If the visible outputs are all unsigned and the unseen operator is subtraction-like, prefer:

1. `|x - y|`
2. then `y - x`
3. only then signed `x - y`

Example:

- [094bf548](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
- `:` is best interpreted as `AB_CD | |x - y| | plain`

#### C. Prefix subtraction family

If the visible operators all support:

```text
AB_CD | ... | plain
```

and they are non-subtraction rules, then an unseen subtraction-family operator may take operator-prefix form.

Example:

- [38489191](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
- visible:
  - `% -> AB_CD | x + y | plain`
  - `( -> AB_CD | x * y + 1 | plain`
- unseen `/` is best read as:
  - `AB_CD | |x - y| | op_prefix`

#### D. Conditional prefix subtraction

Some rows need:

```text
AB_CD | x - y | op_prefix_if_neg
```

Meaning:

- positive result: plain
- negative result: operator-prefixed magnitude

Example:

- [118f8c86](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
- `18}50 = }32`
- `65}48 = 17`
- query `50}49 = 1`

#### E. Conditional postfix-reverse subtraction

Some rows need:

```text
BA_DC | y - x | op_suffix_rev_if_neg
```

Meaning:

- positive result: plain
- negative result: reverse the magnitude and append the operator

Example:

- [00d8b3db](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
- `/` examples force the subtraction family
- `|` and `\` suggest the row motif `BA_DC | ... | rev`
- query `69/52 -> 17/`

## Output-Format Lessons

### 9. Reverse Must Be Text-Based, Not Integer-Based

This was a major correction.

`rev` must mean:

1. format the result as text
2. reverse the text

not:

1. reverse the numeric value as an integer

Example:

- `160`
- integer-style reverse would incorrectly become `61`
- text-style reverse correctly becomes `061`

This fix unlocked rows like:

- [0819520a](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)

### 10. Distinguish Absolute Difference from Signed Difference

Rows like:

- [5787c3d0](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)

show that `op_prefix` on a positive example often means:

```text
|x - y| + prefix
```

not signed subtraction.

This matters especially for:

- `plain`
- `op_prefix`
- `op_prefix_if_neg`
- `op_prefix_rev`

## What the LLM Should Learn

For future training, the model should not memorize answers. It should internalize the procedure:

1. isolate the query operator
2. fit candidate rules only on same-operator examples
3. search pairing, base rule, and output format jointly
4. reject a candidate immediately when an example fails
5. if many candidates survive, compare their query outputs
6. use row-level motif evidence only as a tie-break or absent-operator prior
7. respect output text details:
   - reversal
   - leading zeros
   - conditional prefix
   - conditional postfix

## Good Teaching Examples

These rows are good training exemplars because each highlights one logical step clearly:

1. [144a3e31](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
   - `BA_DC | x * y | rev`
2. [094bf548](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
   - prefer `|x - y|` over signed subtraction
3. [118f8c86](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
   - `op_prefix_if_neg`
4. [078df00e](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
   - absent-operator inference via row motif + additive-to-multiplicative analogy
5. [00d8b3db](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
   - `op_suffix_rev_if_neg`
6. [0bcffccd](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv)
   - one-example operator fixed by row-level motif against scan-order bias

## Recommended Training Decomposition

If we later convert this into knowledge-injection data, I would split it into:

1. combo-prior knowledge
   - common pairings
   - common base rules
   - common output formats

2. method exemplars
   - same-operator solving
   - motif tie-breaks
   - absent-operator heuristics
   - conditional prefix/postfix formatting

3. hard-negative exemplars
   - wrong combo fits `EX1` but fails `EX2`
   - signed subtraction vs `abs(x-y)`
   - integer reverse vs text reverse

This is the part we want the model to internalize:

```text
same-operator solve first
-> verify on all visible examples
-> only then use row motif
-> only then extrapolate to unseen operators
```
