# Solver Method Record

This document records the current deterministic or semi-deterministic solving methods we trust enough to reuse later.

The main purpose is to preserve reusable knowledge for a future Doc2LoRA-style knowledge-injection stage. We are not using this file directly for training yet. It is a method inventory and design reference.

## Current Scope

We currently have reusable method knowledge for:

1. `Text Encryption`
2. `Bit Manipulation`
3. `Unit Conversion`
4. `Gravity`
5. `Numeral System`
6. `Transformation Rules / numeric_equation`
7. `Transformation Rules / symbol_transform`

`symbol_transform` remains the hardest and least complete solver family, but it
now has a current reusable procedure: test direct templates first, then use
encrypted digit search over `BA_DC|rev` or `AB_CD|raw`, with same-operator
examples as the primary evidence and other examples used only to complete the
symbol-digit map.

## How To Use This Record Later

There are two different kinds of knowledge to inject:

1. `Stable factual knowledge`
   Example: the fixed 77-word Wonderland vocabulary for text encryption.

2. `Reusable procedural knowledge`
   Example: "group examples by operator, fit a candidate rule per operator, then apply it to the query."

For a future Doc2LoRA-style phase, we should keep those two knowledge types separate:

1. `knowledge facts`
   Short QA or lookup tasks.

2. `method traces`
   Short worked examples that teach a reusable procedure.

The sections below are written in that form on purpose.

## Phase 1B Decision Matrix

Phase 1B should be used only when compact facts are not enough and the model needs a bridge into reusable procedures before full task traces.

For the starter training loop, Phase 1A and Phase 1B are combined into one
trainable Phase 1 dataset:

1. [phase1_train.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/phase1_train.csv)
2. [phase1_train.summary.json](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/phase1_train.summary.json)

The separate component files remain in:

1. [phase1_components](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/phase1_components)

This lets us train with only two active stages, `Phase 1 -> Phase 2`, while
preserving the old `1A` vs `1B` split for later ablation.

Active SFT schedule:

1. `Phase 1`: train `configs/phase1_training.json` for `1` LoRA epoch at learning rate `1e-4`.
2. `Phase 2`: train `configs/cot_training_phase2_75_10_15.json` for `1` LoRA epoch at learning rate `5e-5`.
3. Phase 2 starts from `outputs/phase1_training/adapter`, so it continues the LoRA weights learned in Phase 1.

After regenerating data, run the train-script preflight checks before launching
GPU training:

1. `python3 train_sft.py --config configs/phase1_training.json --validate-only`
2. `python3 train_sft.py --config configs/cot_training_phase2_75_10_15.json --validate-only`
3. `python3 train_grpo.py --config configs/grpo_stage2.json --validate-only`

Current decision:

1. `Text Encryption`
   Phase 1B is optional and small. Phase 1A handles the 77-word dictionary; Phase 2 traces should teach most of the mapping-and-completion behavior.

2. `Numeral System`
   Phase 1B is probably unnecessary. Roman conversion is deterministic, and Phase 2 traces can teach full task behavior.

3. `Unit Conversion`
   Phase 1A and Phase 1B are excluded from the default starter loop. The weighted scalar method is simple enough to teach through deterministic Phase 2 traces.

4. `Gravity`
   Phase 1A and Phase 1B are excluded from the default starter loop. The method is a simple scalar law over examples, and deterministic Phase 2 traces are sufficient for now.

5. `Bit Manipulation`
   Phase 1B is useful. The model benefits from short procedure cards for rule-family names, bitwise operations, and high-confidence solver templates before long traces.

6. `Transformation Rules / numeric_equation`
   The active starter-loop Phase 1 uses a single merged Numeric Equation Curriculum. It is built from the older Phase 1A facts and Phase 1B methodology cards, deduplicated, stripped of `<think>` wrappers, and normalized to boxed final-answer style. The archived 1A/1B files remain useful for ablation.

7. `Transformation Rules / symbol-equation`
   The active starter-loop Phase 1 uses only the Symbol-Equation Direct Curriculum. The older symbol-transform Phase 1A/1B files are retained for ablation, but they overlap heavily with the direct curriculum and are excluded from the default training CSV. The active direct curriculum includes direct-template rows, motif drills, operator-family drills, symbol-digit conversion drills, RHS-length family drills, and compact route cards.

## Numeral System

### Phase Decision

Numeral System should be treated as Phase 2-only for now.

Phase 1A is not necessary because the required knowledge is tiny and stable:
the task is standard Arabic-to-Roman conversion. A full worked CoT trace teaches
the needed method more efficiently than a separate knowledge-injection dataset.

### Canonical Procedure

The reusable procedure is Huikang-style greedy Roman conversion:

1. Read the target Arabic number.
2. Use the standard Roman table in descending order:
   `1000=M, 900=CM, 500=D, 400=CD, 100=C, 90=XC, 50=L, 40=XL, 10=X, 9=IX, 5=V, 4=IV, 1=I`.
3. Repeatedly subtract the largest value not exceeding the remaining number.
4. Append the corresponding Roman symbol at each subtraction step.
5. Join the symbols without spaces and output the final Roman numeral.

### Current Implementation

Relevant files:

1. [prepare_numeral_phase2_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_numeral_phase2_cot.py)
2. [prepare_phase2_sft_dataset.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase2_sft_dataset.py)

The CoT wording is adapted from Huikang's numeral reasoner but normalized to
our format:

1. begin with a short statement that this is Arabic-to-Roman conversion
2. show the greedy subtraction steps
3. show the joined result
4. end with `The final answer is \boxed{...}` in generated exports

## Text Encryption

### What Is Stable Knowledge

The core stable knowledge is the fixed Wonderland plaintext vocabulary.

The current intended source is:

1. [knowledge_qa.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/knowledge_qa.csv)

This is the part that should be treated as memorization knowledge. The model should internalize the word list and simple pattern matching over that word list.

### What Is Procedural Knowledge

The reusable procedure is:

1. Read the example encrypted words and plaintext words.
2. Extract character-to-character mappings from the examples.
3. Apply those mappings to the target encrypted text.
4. If a target word is only partially decoded, match the pattern against the memorized Wonderland vocabulary.
5. Use one consistent substitution mapping across the whole prompt.
6. Produce the final plaintext answer.

### Why This Split Matters

The model should not have to rebuild a giant visible dictionary during every solution.

The intended behavior is:

1. The vocabulary lives in the adapter weights.
2. The reasoning trace only teaches how to use the extracted cipher mapping and vocabulary completion.

We now prefer a deterministic compact CoT renderer over the earlier
`encryption_new_cot.csv` traces. The compact renderer covers all public
Text Cipher rows and makes the bijection/vocabulary-fill rule explicit without
copying Huikang's very long per-word dictionary scans.

### Current Implementation

Relevant files:

1. [text_cipher.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/src/nemotron_baseline/text_cipher.py)
2. [prepare_text_cipher_compact_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_text_cipher_compact_cot.py)
3. [prepare_text_knowledge_phase1.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_text_knowledge_phase1.py)
4. [prepare_phase2_sft_dataset.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase2_sft_dataset.py)

### Phase 1A Dataset Balance Note

The converted dictionary source is:

1. [text_knowledge_phase1.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/phase1_components/text_knowledge_phase1.csv)

It contains 1,553 rows, while the current numeral knowledge source contains 363 rows.

The text dictionary file is intentionally kept as the full source pool, but it should not automatically dominate a balanced Phase 1 run. For a balanced Phase 1A mixture, prefer:

1. keep all non-pattern dictionary rows
2. keep all ambiguous pattern rows
3. downsample ordinary `pattern_match` rows to roughly 350-500 examples

This gives the 77-word dictionary about 500-650 training rows, enough to preserve vocabulary lookup and pattern matching without swamping smaller knowledge categories.

### Canonical Procedure

The canonical procedure we want the model to learn is:

1. `Mapping extraction`
   Compare example encrypted and plaintext words letter by letter and derive the full cipher-to-plaintext mapping visible in the prompt.

2. `Direct decode`
   Use known mappings to decode as much of each target word as possible.

3. `Vocabulary completion`
   If a decoded target word still has unknown letters, match the partial pattern against the Wonderland vocabulary.

4. `Consistency check`
   Make sure every guessed word is compatible with the same substitution mapping.

5. `Final answer`
   Output the fully decrypted phrase.

### What We Should Teach

We should teach:

1. how to derive letter mappings from examples
2. how to decode with partial mappings
3. how to use the Wonderland vocabulary to finish partial words
4. how to keep one mapping consistent across the whole prompt

We should avoid teaching:

1. verbose per-puzzle reconstructed dictionaries
2. long manual lookup tables in visible reasoning

### Recommended Future Knowledge Injection

For later Doc2LoRA-style use:

1. `facts dataset`
   Use [knowledge_qa.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/knowledge_qa.csv) for vocabulary memorization.

2. `method dataset`
   Use the compact deterministic traces from [text_cipher_compact_cot.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/text_cipher_compact_cot.csv). These traces teach mapping extraction, partial decoding, Wonderland vocabulary matching, and bijective filtering.

## Unit Conversion

### What Is Stable Knowledge

This category has very little fixed factual knowledge. The reusable knowledge is mostly procedural:

1. Wonderland unit conversion prompts use one hidden scalar factor per row.
2. The examples are rounded to two decimal places.
3. The final answer should be formatted with exactly two decimal places.
4. The real-world unit name should not be used as a physical conversion table.

The training set currently has 1,594 unit-conversion rows. A factor-interval check confirms that every row is consistent with a single hidden multiplicative factor.

### Canonical Procedure

The canonical procedure we want the model to learn is:

1. `Type detection`
   Identify the prompt as unit conversion from lines like `x m becomes y`.

2. `Pair extraction`
   Extract all example pairs `(input, output)` and the query measurement.

3. `Factor inference`
   Estimate one scalar factor with the weighted aggregate `factor = sum(outputs) / sum(inputs)`. This is simple, stable, and passes the host numeric verifier on all current unit-conversion rows.

4. `Apply factor`
   Multiply the query value by the inferred factor.

5. `Formatting`
   Round the result to exactly two decimal places and preserve trailing zeros.

### Weighted Scalar Method

The recommended method is:

```text
factor = sum(example outputs) / sum(example inputs)
answer = target * factor
```

For the written CoT, round the factor to four decimal places before applying it:

```text
factor ~= 0.6636
answer ~= target * 0.6636
```

This is a weighted average of the per-pair ratios:

```text
sum(outputs) / sum(inputs)
= sum(input_i * (output_i / input_i)) / sum(input_i)
```

It is better than the simple average `average(output / input)` because small input values create noisier ratios after the displayed output is rounded to two decimals. Weighting by input size reduces that noise.

The host verifier in `reference/evaluation.py` uses numeric tolerance `rel_tol=1e-2`, so the weighted method does not need to reproduce the CSV label to the exact cent. It only needs to output a numerically close value inside `\boxed{}`.

### What We Should Teach

We should teach:

1. unit conversion is hidden scalar multiplication
2. example outputs are rounded, so ratios are approximate
3. use all examples together instead of trusting one pair
4. estimate `factor = sum(outputs) / sum(inputs)`
5. show the factor to four decimal places in the trace
6. output a numeric value, usually rounded to two decimals

We should avoid teaching:

1. real-world physical conversion constants
2. additive or affine rules unless a prompt family later proves they exist
3. long arithmetic traces in Phase 1A
4. exact-cent overfitting, because the host verifier accepts close numeric answers

### Recommended Future Knowledge Injection

Phase 1A is probably unnecessary for unit conversion. The method is short enough to teach directly through Phase 2 CoT traces. If we keep any Phase 1A rows, they should be low-weight auxiliary cards only, not a major knowledge-injection block.

### Current Implementation

Relevant files:

1. [prepare_unit_conversion_phase2_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_unit_conversion_phase2_cot.py)
2. [unit_conversion_cot_method_resolved.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/unit_conversion_cot_method_resolved.csv)
3. [prepare_phase2_sft_dataset.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase2_sft_dataset.py)

The current generated Phase 2 files contain:

1. `unit_conversion_cot.csv`: 1,594 total unit-conversion rows
2. `unit_conversion_cot_method_clean.csv`: 1,594 weighted-scalar rows
3. `unit_conversion_cot_method_resolved.csv`: 1,594 weighted-scalar rows
4. `unit_conversion_cot_answer_aligned.csv`: 0 rows
5. `unit_conversion_cot_unresolved_ambiguous.csv`: 0 rows

`unit_conversion_cot_method_resolved.csv` is used by default in the Phase 2 SFT builder. It contains:

1. `weighted_scalar_factor`: 1,594 rows

The weighted answers match the CSV labels exactly on 1,399 rows and pass the host-style numeric verifier on all 1,594 rows. The Phase 2 builder now allows unit-conversion answers that differ from the CSV label when `answers_match` accepts them, and uses the weighted answer as the supervised final boxed answer for this category. The current 75/10/15 Phase 2 train split receives 1,196 unit-conversion traces.

Audit notes:

1. all unit Phase 2 traces are wrapped in `<think>...</think>`
2. no unit Phase 2 trace contains an internal `\boxed{...}`
3. every unit-conversion answer in the generated file passes `answers_match` against the CSV label
4. the final boxed answer is added later by the SFT prompt builder
5. the CoT uses the weighted scalar-factor method consistently across all unit-conversion rows

## Gravity

### What Is Stable Knowledge

This category is almost the same shape as Unit Conversion: every prompt contains several rounded observations generated from one hidden scalar. The prompt mentions the physical formula `d = 0.5*g*t^2`, but the simplest reusable solving trick is to collapse it into a hidden rate:

```text
d = k*t^2
k = sum(distance) / sum(t^2)
```

The key is not to compute Earth gravity, and not even to compute `g` explicitly. Use `k = 0.5*g` as the problem-local rate. Because displayed distances are rounded, the most stable deployable estimate is the weighted aggregate rate:

1. square the time first
2. sum all squared times
3. sum all observed distances
4. compute `k = sum(distance) / sum(t^2)`, rounded to four decimals for the trace
5. apply `distance = k * target_t^2`
6. round the final distance to two decimals

The training set currently has 1,597 gravity rows. This weighted-rate rule is accepted by the host-style verifier on all 1,597 rows.

### Current Solver Result

Using `answer = round_2(target_t^2 * round_4(sum(distance) / sum(t^2)))`:

1. exact label match: 1,243 / 1,597
2. accepted by `answers_match` / host-style relative tolerance: 1,597 / 1,597
3. generated source mode: `weighted_hidden_rate`

For comparison, the older median-rate method is also accepted on all 1,597 rows, but it is less exact against the released labels and requires teaching an extra selection step. The weighted method is therefore preferable for Phase 2 traces.

### Canonical Procedure

The canonical procedure we want the model to learn is:

1. `Type detection`
   Identify the prompt as Wonderland gravity from the changed-gravitational-constant wrapper and observations like `For t = ...s, distance = ... m`.

2. `Formula collapse`
   Rewrite the prompt formula as `d = k*t^2`, where `k = 0.5*g`, and avoid explicitly computing `g`.

3. `Weighted rate inference`
   Compute `sum(t^2)` and `sum(distance)`, then use `k = sum(distance) / sum(t^2)`.

4. `Apply rate`
   Square the query time and compute `distance = k * query_t^2`.

5. `Formatting`
   Round the final answer to two decimals and preserve trailing zeros when possible.

### What We Should Teach

We should teach:

1. `d = 0.5*g*t^2` can be solved as `d = k*t^2`
2. `k = sum(distance) / sum(t^2)` is the preferred robust estimate
3. time must be squared before summing or multiplying
4. rounded observations make per-example `k` estimates approximate
5. final answers should use two-decimal numeric formatting

We should avoid teaching:

1. explicit `g = 2d/t^2` unless a prompt asks for `g`
2. Earth's gravitational constant
3. linear-in-time shortcuts
4. median-rate selection as the primary method
5. long manual arithmetic traces in Phase 1A

### Phase 1 / Phase 2 Recommendation

Gravity is simple enough that we should not spend Phase 1A capacity on it. The weighted-rate procedure is short, fully covered by Phase 2 traces, and accepted by the host-style verifier on every released Gravity row.

Recommended handling:

1. do not include Gravity in Phase 1A knowledge injection
2. teach Gravity through deterministic Phase 2 CoT only
3. keep the old Phase 1A generator/data only as an archived artifact, not as an active training input

### Current Implementation

Relevant files:

1. [prepare_gravity_phase2_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_gravity_phase2_cot.py)
2. [gravity_cot_method_resolved.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/gravity_cot_method_resolved.csv)

The current generated Phase 2 file contains all 1,597 Gravity rows. The archived `phase1a_gravity_knowledge.csv` should not be mixed into active Phase 1 training.

## Bit Manipulation

### What Is Stable Knowledge

This category is mostly procedural, not factual.

There is no equivalent of the 77-word dictionary here. The reusable knowledge is the solving algorithm itself.

### Core Method

The current trusted method comes from the bit-manipulation notebook solver, but we only trust the `high-confidence` subset.

High-confidence means:

1. the puzzle is solved without the notebook's final brute-force tie-break phase
2. the result is determined by whole-byte rules or by per-bit rules plus structural propagation

We do **not** treat brute-force-scored low-confidence traces as faithful reasoning traces.

### Canonical Procedure

#### Phase 0: Whole-Byte Search

Try simple whole-byte transformations first:

1. XOR with a fixed mask
2. left or right rotation
3. permutation with optional inversion
4. rotate then XOR
5. reverse then XOR
6. uniform 2-input bit rule across all output positions
7. mixed structured rule

If one of these fits all examples, solve the query with that whole-byte rule.

#### Phase 1: Per-Bit Candidate Enumeration

If no whole-byte rule works, treat each output bit independently and enumerate simple candidate rules that fit all examples for that output bit.

Candidate families:

1. constants `0` and `1`
2. one-input rules `ID` and `NOT`
3. symmetric 2-input boolean rules:
   `AND`, `OR`, `XOR`, `NAND`, `NOR`, `XNOR`
4. asymmetric 2-input rules:
   `INHIB`, `IMPL`
5. only if needed:
   `MAJ`, `CH`, `XOR3`
6. only if still needed:
   small composite two-layer boolean functions

#### Phase 2: Structural Propagation

After per-bit candidates are collected:

1. resolve bits where all surviving candidates agree
2. detect a dominant shift pattern
3. detect repeated source-pair patterns
4. detect partial permutation structure
5. propagate those structures to resolve the remaining bits

#### Confidence Rule

Keep only rows where all bits are resolved before the notebook's brute-force ambiguity resolution.

### Current Implementation

Relevant files:

1. [prepare_phase1a_bit_manipulation_knowledge.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase1a_bit_manipulation_knowledge.py)
2. [prepare_phase1b_bit_manipulation_methodology.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase1b_bit_manipulation_methodology.py)
3. [prepare_bit_manipulation_phase2_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_bit_manipulation_phase2_cot.py)
4. [bit_manipulation_hybrid_cot.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/bit_manipulation_hybrid_cot.csv)

### Solver Sanity Check

We tested two deterministic routes on all `1602` bit-manipulation rows in `data/train.csv`.

1. Notebook-derived solver:
   `1208 / 1602 = 75.41%`
2. Pure broad per-bit boolean scan:
   `1001 / 1602 = 62.48%`
3. Oracle union of both predictions:
   `1273 / 1602 = 79.46%`

Conclusion: simply widening the candidate gate library is not enough. The difficult part is ambiguity resolution when many candidate functions fit the short example columns. The notebook solver wins because it uses whole-byte detection, dominant shift evidence, pair-pattern propagation, and brute-force scoring. The pure broad solver is still useful as an analysis harness because it exposes candidate ambiguity and finds some rows the notebook misses, but it should not replace the notebook-derived high-confidence trace source yet.

### Cursor Hybrid Strategy

`reference/cursor/bit_manipulation` contains a stronger sandbox strategy for bit manipulation.

Measured coverage on all `1602` public train bit rows:

1. Huikang reasoner, `hu_reas`:
   `1364 / 1602 = 85.14%`
2. Huikang investigator, `hu_inv`:
   `1428 / 1602 = 89.14%`
3. Extended investigator, `hu_inv + ext`:
   `1477 / 1602 = 92.20%`
4. Offline labelled set, `hu_reas + hu_inv + ext`, gold-filtered:
   `1510 / 1602 = 94.26%`
5. With notebook oracle added:
   `1528 / 1602 = 95.38%`
6. All five local solvers as an oracle union:
   `1532 / 1602 = 95.63%`

The key extension is `reference/cursor/bit_manipulation/harness/extended_investigator.py`.
It only runs after Huikang's original investigator misses. It searches depth-4 byte expressions:

```text
R(v) = atom_a(v) OP_top atom_b(v)
```

where each atom is a depth-1 or depth-2 expression over:

```text
I, NOT, REV, NOT REV, ROT(k), SHL(k), SHR(k), NOT ROT(k), NOT SHL(k), NOT SHR(k)
```

and the top operators are:

```text
XOR, AND, OR
```

The generated CoT file is:

```text
reference/cursor/bit_manipulation/results/labelled_cot.csv
```

It contains `1510` oracle-filtered traces:

1. `tier1_hu_reas`: `1364` rows, Huikang reasoner trace verbatim
2. `tier2_rule`: `146` rows, Huikang search prefix spliced with a depth-4 rule derivation

Validation performed:

1. every emitted trace has a final `\boxed{...}`
2. every final boxed value matches the CSV `answer`
3. no missing boxed answers were found

Important integration caveat:

The cursor Phase 2 CSV uses columns:

```text
id, source_tier, rule, trace, prediction, answer, n_lines, n_chars
```

Our current SFT builders expect:

```text
id, prompt, answer, generated_cot, label, category, source
```

So we need a small adapter before using it in Phase 2. The adapter should join `labelled_cot.csv` back to `data/train.csv` by `id`, rename `trace` to `generated_cot`, keep `answer`, and set:

```text
label = Bit Manipulation
category = Bit Manipulation
source = cursor_bit_hybrid
```

Phase 1A and Phase 1B bit datasets also exist in the cursor sandbox:

1. `phase1a_bit_manipulation_knowledge.csv`: `993` rows
2. `phase1b_bit_manipulation_methodology.csv`: `252` rows

The Phase 1A data is fact/primitives-heavy. The Phase 1B data teaches the escalation procedure and compact worked examples.

### What We Should Teach

We should teach:

1. try easy whole-byte rules first
2. if that fails, solve output bits one by one
3. use simple boolean functions before complex ones
4. reuse shift and permutation structure across bit positions

We should avoid teaching:

1. raw solver-log dumps
2. exhaustive candidate lists for every bit
3. heuristic brute-force scoring as if it were a proof

### Recommended Future Knowledge Injection

For later method injection:

1. build short method examples from the deterministic renderer
2. restrict to high-confidence rows only
3. teach the hierarchy:
   whole-byte search -> per-bit search -> structural propagation

## Transformation Rules / numeric_equation

### What Is Stable Knowledge

This category is also mostly procedural.

The reusable knowledge is not a global meaning for the operator symbols. The same operator symbol can mean different things in different rows.

So the method must be taught as:

1. row-local rule induction
2. operator-specific solving within one prompt

### Important Observation

Inside `Transformation Rules`, the numeric subset and the symbol-only subset behave very differently.

Current split on [train.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv):

1. `732` numeric-equation rows
2. `823` symbol-transform rows

The numeric solver should be treated as a method for the `numeric_equation` subtype only.

### Canonical Procedure

#### Step 1: Parse the Row as a Mini DSL

For each example and the query:

1. parse `left_operand`
2. parse `operator_symbol`
3. parse `right_operand`

Preserve the original operand strings, not just their integer values. This matters for:

1. leading zeros
2. reverse-digit rules
3. zero-padded outputs

#### Step 2: Group Examples by Operator

Within the same row, each operator can have its own hidden rule.

So we solve operator-by-operator, not row-wide with one universal formula.

#### Step 3: Enumerate Candidate Rule Families

For each operator, test candidate rules over three dimensions.

##### Input Transforms

1. use left operand as written
2. reverse left operand
3. use right operand as written
4. reverse right operand

##### Base Numeric Rules

Current trusted rule set in [numeric_equation.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/src/nemotron_baseline/numeric_equation.py):

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

##### Output Rendering Modes

After computing the numeric result, render it in one of these forms:

1. plain digits
2. reversed digits
3. negative form
4. negative reversed form
5. operator prefix
6. operator suffix
7. operator prefix after reversal
8. operator suffix after reversal

Also preserve output width when the examples imply leading zeros.

#### Step 4: Keep Only Rules That Match All Examples for That Operator

For a given operator inside a row:

1. test all candidate combinations
2. keep only candidates that reproduce every example for that operator

#### Step 5: Predict the Query Operator

If the query operator is present in the examples:

1. use the surviving candidates for that operator
2. check how many different query outputs those candidates produce
3. if the query operator has only one example and multiple candidates remain, use row-level motif evidence from operators with `2+` examples:
   - exact family signature consensus when available
   - otherwise transform-signature consensus
   - otherwise output-format consensus as the weakest tie-break

Confidence logic:

1. if all matching candidates agree on the query output, the row is usable
2. if matching candidates disagree on the query output, the row is ambiguous
3. if the query operator never appears in the examples, do not trust the row

### Why This Confidence Rule Matters

This selector works better than the notebook's original numeric trace logic.

Historical local-analysis stats from the older numeric-equation solver:

1. total numeric-equation rows: `732`
2. exact solved overall: `510`
3. exported exact high-confidence rows: `384`
4. previous `75/10/15` SFT split received `288` numeric-equation solver traces

On the current solver:

1. high-confidence exactness is `384 / 392 = 97.96%`
2. medium-confidence exactness is `30 / 111 = 27.03%`
3. low-confidence exactness is `96 / 204 = 47.06%`

So the key reusable rule is:

1. trust only rows where all surviving candidates agree on the query output

For one-example operators, a useful secondary heuristic is:

1. infer the row-level transform motif from better-supported operators
2. prefer candidates for the query operator that preserve that motif

Example:

1. in [train.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv) row `026106f5`, the `{` operator is consistently solved by `rev(|x-y| on rev,rev)`
2. that suggests a row-level `rev,rev -> rev` motif
3. so the one-example `*` operator is better explained by `rev(x+y on rev,rev)` than by `plain(x+y on id,id)`

### Extended Numeric Equation Export

The stronger cursor sandbox in [reference/cursor/transformation_rules/numeric_equation](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/reference/cursor/transformation_rules/numeric_equation) should now be treated as the active Phase 2 source for this subtype.

Verified counts:

1. extended labelled CoT rows: `696 / 732`
2. deterministic rows: `572`
3. oracle/speculative rows: `124`
4. deterministic rows selected by the current `sft_train` split: `435`
5. oracle rows selected by the current `sft_train` split: `87`

The active files are:

1. [prepare_numeric_equation_phase2_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_numeric_equation_phase2_cot.py)
2. [numeric_equation_labelled_cot.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/numeric_equation_labelled_cot.csv)
3. [numeric_equation_deterministic_cot.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/numeric_equation_deterministic_cot.csv)
4. [prepare_numeric_equation_synthetic_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_numeric_equation_synthetic_cot.py)
5. [numeric_equation_synthetic_cot.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/numeric_equation_synthetic_cot.csv)

Training policy:

1. include all selected deterministic numeric rows once
2. include all selected oracle/speculative numeric rows once as low-mass exposure
3. do not repeat exact rows
4. generate `2x` synthetic variants from selected deterministic rows
5. keep oracle rows untouched, with no synthetic expansion

Current Phase 2 contribution after this policy:

1. real deterministic numeric rows: `435`
2. real oracle numeric rows: `87`
3. synthetic deterministic numeric rows: `870`
4. total numeric-equation Phase 2 exposures: `1392`

This gives the model a strong deterministic rule-learning signal without letting numeric-equation traces dominate Phase 2.

Detailed CoT template update:

1. [numeric_equation_detailed_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/numeric_equation_detailed_cot.py) is the active renderer for real and synthetic numeric-equation Phase 2 traces.
2. The template is parse -> scan candidates in taught priority order -> reject every failed candidate sequentially until the first matching candidate -> verify matching candidate -> lock rule -> apply to query.
3. Same-operator examples are treated as primary evidence; other operators only provide row-level motif evidence.
4. Oracle/speculative rows use weaker wording: "plausible accepted rule" and "lower training weight."
5. The raw traces use the shared CoT wrapper: "I will put my final answer inside `\boxed{}`" near the opening and "The final answer is `\boxed{...}`" at the end. The Phase 2 builder normalizes those final lines into a clean `<think>...</think>` block and then appends the supervised boxed answer, with a safe fallback for rare answers containing `}`.

Current detailed CoT length audit:

1. real numeric rows: median `2633` chars, max `11668` chars
2. split synthetic numeric rows: median `2833` chars, max `11583` chars
3. all-deterministic synthetic numeric rows: median `3033` chars, max `12519` chars
4. the longest traces now show up to `37` candidate attempts, which is intentional for rules appended after the core `36`-candidate priority list
5. by character length, these traces remain reasonable for the `7600` output-token cap

For final-submission training, we also keep a separate full synthetic file:

1. [numeric_equation_synthetic_cot_all_deterministic.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/numeric_equation_synthetic_cot_all_deterministic.csv)
2. base rows: all `572` deterministic numeric-equation rows
3. synthetic rows: `1716 = 572 x 3`
4. this remains a heavier ablation/final-training option, not the active Phase 2 default
4. intended use: final training run after local split-based evaluation is no longer needed

Phase 1 numeric-equation curriculum:

1. [prepare_phase1a_numeric_equation_knowledge.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase1a_numeric_equation_knowledge.py) reads the accepted numeric-equation rule labels.
2. [prepare_phase1b_numeric_equation_methodology.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase1b_numeric_equation_methodology.py) creates compact method cards.
3. [prepare_phase1_numeric_equation_curriculum.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase1_numeric_equation_curriculum.py) merges those two files into the active training component.
4. The merged file deduplicates exact prompt+answer pairs, removes `<think>` wrappers, skips unsafe boxed-answer literals, and appends `The final answer is \boxed{...}`.

Active merged numeric-equation card mix:

1. total rows: `1662`
2. base-rule semantics: `100`
3. combo application: `594`
4. output-mode semantics: `418`
5. pairing transforms: `124`
6. scan priority: `114`
7. priority inventory: `95`
8. rule-boundary cards: `45`
9. pairing-principle cards: `3`
10. accepted rule inventory: `3`
11. methodology cards: `166` across procedure, same-op decision, motif evidence, tie-break, absent-op, and sanity-check subtypes
12. confidence-gating and ambiguity-handling cards are excluded from the active curriculum because they teach abstention or hesitation rather than a final competition answer

### Frequency-Ordered Digit-Transform Scan

The reference `symbol-digit` method suggests a slightly different way to search this same subtype:

1. treat each row as a hidden combination of:
   `pairing | base rule | output rendering`
2. order those combinations by how often they appear in train data
3. scan the most frequent combinations first
4. as soon as one combination matches `EX1`, verify it immediately on `EX2`
5. only then continue checking the rest of the same-operator examples

This method is now implemented in:

1. [digit_transform.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/src/nemotron_baseline/digit_transform.py)
2. [solve_digit_transform.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/solve_digit_transform.py)

The scan uses the empirical ordering derived from the exact high-confidence numeric rows. The strongest combinations currently start with:

1. `BA_DC | x * y | rev`
2. `AB_CD | concat(x, y) | plain`
3. `BA_DC | x + y | rev`
4. `BA_DC | x + y + 1 | rev`
5. `BA_DC | x + y - 1 | rev`

Older measured coverage on the `732` `Transformation Rules / numeric_equation` rows in [train.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/train.csv):

1. exact solver predictions: `510 / 732 = 69.67%`
2. high-confidence exact exportable rows: `384 / 732 = 52.46%`
3. no-prediction rows: `52`

This scan is useful because it captures the notebook-style `digit_transform` intuition directly:

1. many rows are generated from a small menu of pairing and arithmetic templates
2. checking the most common templates first solves a large portion of the subtype quickly
3. verifying on `EX2` right after `EX1` prevents many accidental matches

### Current Implementation

Relevant files:

1. [numeric_equation.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/src/nemotron_baseline/numeric_equation.py)
2. [numeric_equation_detailed_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/numeric_equation_detailed_cot.py)
3. [prepare_numeric_equation_phase2_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_numeric_equation_phase2_cot.py)
4. [prepare_numeric_equation_synthetic_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_numeric_equation_synthetic_cot.py)
5. [numeric_equation_labelled_cot.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/numeric_equation_labelled_cot.csv)
6. [numeric_equation_synthetic_cot.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/trainable/numeric_equation_synthetic_cot.csv)

### What We Should Teach

We should teach:

1. operator-specific reasoning within one row
2. operand reversal as a first-class hypothesis
3. output formatting as part of the hidden rule
4. leading-zero preservation
5. candidate agreement as a confidence signal

We should avoid teaching:

1. any global semantics for symbols like `*`, `-`, `/`, `@`, `#`
2. the assumption that one arithmetic formula explains the whole row

### Recommended Future Knowledge Injection

For later method injection:

1. create short examples that teach:
   parse by operator -> test candidate families -> keep consistent ones -> apply to query
2. use only the exported high-confidence subset
3. keep operator symbols row-local in the training examples so the model does not memorize false global meanings

## Transformation Rules / symbol_transform Exploratory Notes

### Current Reference Method

Inside the `symbol_transform` subtype, grouping by the middle operator of the 5-character query shape `ABOCD` is useful.

Current operator statistics are materialized in:

1. [symbol_transform_operator_stats.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/symbol_transform_operator_stats.csv)
2. [symbol_transform_operator_groups.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/symbol_transform_operator_groups.csv)
3. [symbol_transform_operator_groups.json](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/symbol_transform_operator_groups.json)

The current reference implementation is:

1. [symbol_transform_solver.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/reference/cursor/transformation_rules/symbol_transform/harness/symbol_transform_solver.py)
2. [evaluate_router.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/reference/cursor/transformation_rules/symbol_transform/harness/evaluate_router.py)
3. [README.md](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/reference/cursor/transformation_rules/symbol_transform/README.md)

The reference router was re-run and its headline result is verified:

1. subset size: `823`
2. predicted rows: `166`
3. exact rows: `88`
4. exact rate: `10.69%`
5. stage3 cipher-numeric precision: `29 / 34`

### What Is Actually Reliable

The router contains several experimental stages, but precision is concentrated in only a few places:

1. `stage1_star_join`: `22 / 23`
2. `stage2_direct_permutation`: `35 / 57`
3. `stage3_cipher_numeric_same_op`: `29 / 34`
4. `stage4_operator_template_prior`: retracted from training despite `2 / 2` public-train matches

The `stage4_operator_template_prior` family looked precise on public rows, but we removed it after inspecting row `52be4988`. It can force an answer from accidental overlap between the single same-operator example and the query, rather than deriving a stable transformation rule.

The later fallback stages are not currently usable for training traces:

1. `stage5_shared_map_templates`: `0 / 11`
2. `stage7_perm_subst`: `0 / 12`
3. `stage8_template_map_same_op`: `0 / 23`
4. `stage9_template_map_absent_op`: `0 / 4`

So the first deterministic solver should not teach these fallback stages yet.

### Direct Symbol Join / Permutation Subcase

Our earlier `*`-operator observation generalizes cleanly.

The two trusted direct templates are:

1. `0134`: remove the operator and keep the four operand symbols, `ABOCD -> ABCD`
2. `3401`: swap the two operand pairs, `ABOCD -> CDAB`

On the current router output:

1. `stage1_star_join` with templates `0134` or `3401`: `22 / 22`
2. `stage2_direct_permutation` with templates `0134` or `3401`: `35 / 35`

Other direct-permutation templates are not trusted yet. In the current run they contribute errors, not additional exact solves.

### Cipher-Numeric Same-Operator Subcase

The most important nontrivial method treats the symbol puzzle as an encrypted digit-transform puzzle:

1. each non-operator symbol is a row-local encrypted digit
2. the symbol-to-digit map is bijective within the row
3. examples with the same middle operator as the query are used to infer both the digit map and the numeric rule
4. after computing the numeric result, encode the result digits back into symbols

Current stage3 candidates search over:

1. pairings like `AB_CD`, `AB_DC`, `BA_CD`, `BA_DC`
2. base rules like `x + y`, `x + y +/- 1`, `x - y`, `|x-y|`, `x*y`, `x*y +/- 1`
3. output modes `raw`, `rev`, and `abs`

The current `wide` mode is better than `strict` or `wide2`:

1. `strict`: `23 / 29`
2. `wide`: `29 / 34`
3. `wide2`: `10 / 17`

The gated `wide2` fallback should stay disabled for training data. It did not add coverage and creates ambiguity.

### Precision-First Selection For CoT Data

For Phase 2 CoT generation, precision matters more than raw prediction count.

Recommended export filter:

1. keep `stage1_star_join` only for templates `0134` and `3401`
2. keep `stage2_direct_permutation` only for templates `0134` and `3401`
3. do not keep `stage4_operator_template_prior`; one-example position-substitution priors are too brittle
4. keep `stage3_cipher_numeric_same_op` only when the fit is unique, bijective, and the generated answer matches the public training answer

Measured public-train options:

1. direct templates plus stage4 only: old measurement `59 / 59`, now retracted because stage4 is not logically stable
2. direct templates plus stage4 plus all stage3-wide rows: old measurement `88 / 93`, now retracted for training policy
3. direct templates plus stage4 plus a stage3 template whitelist: old measurement `88 / 89`, now retracted for training policy

For SFT data, use gold verification on public train and export only exact solved rows. For future private-style inference, treat stage3 as high-confidence but not proof.

### Next Improvement Direction

The reference stage3 currently solves only from examples with the same operator as the query. This is safe, but it leaves information unused.

The implemented exploratory solver combines the cipher-numeric idea with our `digit_transform` motif method:

1. infer one row-local symbol-to-digit bijection from all examples
2. infer a row-level operand pairing and output rendering motif when possible
3. allow each operator symbol to have its own base arithmetic rule
4. use same-operator examples first when the query operator appears
5. when the query operator is absent, use the numeric-equation heuristic:
   prefer frequent unused base rules that preserve the row motif

Current implementation:

1. [symbol_transform.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/src/nemotron_baseline/symbol_transform.py)
2. [prepare_phase1_symbol_transform_direct_curriculum.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase1_symbol_transform_direct_curriculum.py)
3. [prepare_phase1a_symbol_transform_knowledge.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase1a_symbol_transform_knowledge.py) archived for ablation
4. [prepare_phase1b_symbol_transform_methodology.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_phase1b_symbol_transform_methodology.py) archived for ablation
5. [prepare_symbol_transform_synthetic_cot.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_symbol_transform_synthetic_cot.py)
6. [prepare_symbol_transform_phase2_export.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/prepare_symbol_transform_phase2_export.py)

Current full-train exploratory results on the `823` symbol-transform rows:

1. conservative core unique mode before the row-global consistency patch: `130` exact from `138` predictions
2. safe adaptive retry on `no_rule` only before the row-global consistency patch: `134` exact from `142` predictions
3. current safe default, using core unique mode, same-operator row-global consistency, and safe adaptive retry on `no_rule` only: `136` exact from `142` predictions
4. current safe default precision on public train: `95.77%`
5. bounded query-operand completion with `max_query_unknowns=1`, if used directly as an inference solver: `142` exact from `151` predictions
6. bounded query-operand completion direct precision on public train: `94.04%`
7. public-gold-verified query-completion rescue for Phase 2 export only: `7` additional real rows beyond the safe `136`
8. broad asymmetric mode, measured before the final direct-template prior patch: `110` exact from `123` predictions
9. ranked tie-break mode, measured before the final direct-template prior patch: `112` exact from `183` predictions

So the best current default is `core` + unique agreement, plus the same-operator row-global consistency check, plus optional safe adaptive retry for `no_rule` rows. The former tiny operator-template prior has been removed from the solver and trainable data because it teaches brittle one-example position matching rather than a real numeric/symbol rule.

The query-completion branch should not replace the private-style default yet. It recovers `7` additional public-train rows, but it also introduces `3` new wrong predictions if used directly. For training data, we use it only as a public-gold-verified rescue pass: safe exact rows remain, and query-completion rows are appended only when the generated prediction equals the known public training answer.

Adaptive retry notes:

1. retrying only `no_rule` rows at `max_states_per_rule=240` gives `4 / 4` additional exact predictions
2. retrying both `no_rule` and ambiguous rows gives `6 / 8` additional exact predictions
3. ambiguous retry is not trusted by default because it adds wrong predictions
4. the current safe adaptive result is `136 / 823`

The broad asymmetric bank is not ready as a default. It found a few new correct rows, including a modulo row, but it also displaced more correct core solves than it added. For now, asymmetric operations should be explored as gated plugins, not merged broadly into the main operation bank.

Important current gates:

1. keep strict bijective symbol-to-digit maps
2. require candidate agreement before predicting
3. when a same-operator encrypted digit-transform prediction is available, first check whether full-row global consistency produces exactly one candidate
4. require at least two query-operator examples before the broad global motif fallback predicts
5. keep one-example broad global motif rows as research-only because they produced confident false positives
6. keep query-operand completion as a Phase 2 public-rescue tool, not as a promoted inference default

Later guarded-min1 update:

1. [symbol_transform_guarded_min1_additional_exact.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/symbol_transform_guarded_min1_additional_exact.csv) records `7` additional train-exact rows.
2. [symbol_transform_solver_analysis_current.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/symbol_transform_solver_analysis_current.csv) has been updated from `198` to `205` exact rows.
3. These rows use `guarded_min1_global_consistency_unique`, not the broad one-example fallback.
4. The guard requires direct templates to fail, exactly one same-operator example, motif `BA_DC|rev` or `AB_CD|raw`, a complete 10-symbol digit map, at least four supported examples, and one agreed query output.

Later length-1 RHS update:

1. [symbol_transform_length1_rhs_additional_exact.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/symbol_transform_length1_rhs_additional_exact.csv) records `3` additional train-exact rows found while studying one-symbol RHS subtraction cases.
2. [symbol_transform_solver_analysis_current.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/symbol_transform_solver_analysis_current.csv) has been updated from `205` to `208` exact rows.
3. The new trusted branch is `length1_subtraction_rescue_unique`.
4. The guard requires the query same-operator RHS length to be `1`, restricts the query operator to `x-y`, `y-x`, or `|x-y|`, and requires a length-3 helper row to lock `x+y` before variants such as `x+y+1`.
5. A broader version of this branch produced wrong unique predictions, so rows without the length-3 `x+y` helper remain unresolved.

### Oracle Diagnostic Direction

The next research tool is:

1. [analyze_symbol_transform_oracle.py](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/scripts/analyze_symbol_transform_oracle.py)

This script uses the public gold answer as an extra constraint. It is not an inference solver. Its purpose is to classify missed rows into:

1. `current_core_space`
   The existing core rule space can explain the gold answer once the answer constrains missing digit-symbol assignments. These rows point to ranking, beam, or incomplete symbol-to-digit-map issues.

2. `output_extension`
   Core arithmetic works, but the output renderer needs a new format such as `last`, `last_rev`, or operator-prefix/suffix handling.

3. `operation_extension`
   A non-core operation is needed with current output rendering.

4. `operation_and_output_extension`
   Both operation and rendering need expansion.

5. `unexplained`
   The row is not explained by the tested encrypted digit-transform hypothesis.

Current full oracle results after the safe solver patch:

1. target rows missed by the active solver: `687`
2. gold-explainable by the oracle search: `507`
3. still unexplained by the tested encrypted digit-transform hypothesis: `180`
4. `current_core_space`: `450`
5. `output_extension`: `53`
6. `operation_and_output_extension`: `4`

The current-core diagnostic table is:

1. [symbol_transform_core_gap_diagnostics.csv](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/data/symbol_transform_core_gap_diagnostics.csv)

It splits the `450` current-core misses into action buckets:

1. `beam_or_map_completion_large_core`: `200`
2. `beam_or_map_completion_small_core`: `141`
3. `map_completion_unseen_answer_symbol`: `69`
4. `ranking_small_ambiguous_core`: `20`
5. `ranking_large_ambiguous_core`: `19`
6. `wrong_unique_candidate_core`: `1`

The most common explainable motifs are `BA_DC|rev` and `AB_CD|raw`; the most common rules are `x + y`, `x * y`, `x - y`, and the `+/- 1` variants. Same-operator-count `1` is the largest gap, with `299` of the `450` current-core misses.

The main signal is not yet "add many asymmetric operations." The stronger signal is:

1. improve symbol-to-digit-map completion and ranking inside the current core space
2. add gated output renderers, especially `last` and `last_rev`
3. only then test asymmetric operation plugins one family at a time

### Canonical CoT Templates To Teach

For direct symbol templates:

1. identify the query operator
2. collect examples with the same operator
3. test whether those examples all follow `ABOCD -> ABCD` or `ABOCD -> CDAB`
4. apply the same position template to the query
5. box the resulting symbol string

For cipher-numeric rows:

1. identify the row as encrypted digit-transform
2. use same-operator examples to build a bijective symbol-to-digit table
3. test candidate `pairing | base rule | output mode` triples
4. keep only triples that reproduce the same-operator examples
5. decode the query operands, compute the numeric result, render it, then encode digits back to symbols
6. box the resulting symbol string

Detailed methodology for teaching this subtype is now recorded in:

1. [symbol_transform_methodology.md](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/docs/symbol_transform_methodology.md)

### Unit-First Digit Constraint Trace

For difficult encrypted digit-transform rows, especially when a same-operator example has a 4-symbol output, use a reusable digit-constraint trace instead of unstructured brute force.

The preferred trace policy is:

1. choose a small numeric motif first, such as `BA_DC | x*y | rev`, when two 2-symbol operands produce a 3- or 4-symbol output
2. write each symbol as a digit variable and translate the example into a digit-shape equation
3. start with the units digit, for example `a(c-1) = 0 mod 10` or `eg = f mod 10`
4. branch on the unit condition before enumerating any values
5. inside every branch, simplify the equation and again use the new units digit or digit-shape condition before trial enumeration
6. when enumeration is still needed, list the reduced branch table explicitly: candidate variable value, computed shape, and `PASS` or exact rejection reason
7. intersect candidate tables from multiple same-operator equations
8. only then lock the digit map and verify every same-operator example

This method should be rendered in the Huikang bit-trace style: short scan blocks, repeated `PAIR`, `PASS`, `FAIL`, `Best`, `LOCK`, `Selected`, and `Applying` lines. Avoid Markdown table separators because they are token overhead and force the model to learn unnecessary formatting. The trace should be systematic enough that the model learns the algorithm, not only the answer.

We should avoid teaching:

1. low-precision template-map fallback stages
2. relaxed non-bijective digit mappings
3. arbitrary direct permutations outside `0134` and `3401`

### Historical `*` Operator Observation

Within the `*` group:

1. total rows: `184`
2. rows with 4-character outputs: `147`
3. rows solvable by exact concatenation / permutation of the four non-operator symbols: `30`

Trusted recurring join patterns:

1. `AB*CD -> ABCD`
2. `AB*CD -> CDAB`

Observed counts:

1. `ABCD`: `22`
2. `CDAB`: `7`
3. other exact permutations: only isolated or ambiguity-driven one-offs, not yet trusted as reusable families

So the current high-precision `*` subcase is:

1. `star_join_identity`
2. `star_join_swap_halves`

Everything else in the `*` group should currently remain in `star_other`.

## Commands To Regenerate Artifacts

These are the active deterministic build commands. Older external-model
generation steps, notebook teacher traces, and exploratory diagnostics are not
part of the current training-data path.

### Text Encryption

```bash
python3 scripts/prepare_text_knowledge_phase1.py
python3 scripts/prepare_text_cipher_compact_cot.py
```

### Numeral System Phase 2

Numeral System does not need Phase 1A. Regenerate only the deterministic Phase 2 traces:

```bash
python3 scripts/prepare_numeral_phase2_cot.py
```

### Bit Manipulation

```bash
python3 scripts/prepare_phase1a_bit_manipulation_knowledge.py
python3 scripts/prepare_phase1b_bit_manipulation_methodology.py
python3 scripts/prepare_bit_manipulation_phase2_cot.py
```

The Phase 2 builder consumes `data/trainable/bit_manipulation_hybrid_cot.csv`
when constructing the combined SFT file.

### Numeric Equation

Phase 1:

```bash
python3 scripts/prepare_phase1a_numeric_equation_knowledge.py
python3 scripts/prepare_phase1b_numeric_equation_methodology.py
python3 scripts/prepare_phase1_numeric_equation_curriculum.py
```

Phase 2:

```bash
python3 scripts/prepare_numeric_equation_phase2_cot.py
python3 scripts/prepare_numeric_equation_synthetic_cot.py --variants-per-row 2
```

Optional heavier final-training ablation:

```bash
python3 scripts/prepare_numeric_equation_synthetic_cot.py --split-csv DOES_NOT_EXIST.json --variants-per-row 3 --output-csv data/trainable/numeric_equation_synthetic_cot_all_deterministic.csv
```

### Combined Phase 1 Training Dataset

After refreshing any desired Phase 1 component files, combine the active
starter-loop components into one training CSV:

```bash
python3 scripts/prepare_phase1_training_dataset.py
```

Unit conversion, gravity, and numeral knowledge-card component files may exist
locally for ablation, but they are excluded from the default Phase 1 mixture.

### Gravity Phase 2

Gravity does not need Phase 1A. Regenerate only the deterministic Phase 2 traces:

```bash
python3 scripts/prepare_gravity_phase2_cot.py
```

### Unit Conversion Phase 2

```bash
python3 scripts/prepare_unit_conversion_phase2_cot.py
```

### Symbol-Equation Phase 1

```bash
python3 scripts/prepare_phase1_symbol_transform_direct_curriculum.py
```

For the starter loop, do not include
`phase1a_symbol_transform_knowledge.csv` or
`phase1b_symbol_transform_methodology.csv` in
`phase1_train.csv`. They remain useful as local ablation/reference pools, but
the active symbol-equation Phase 1 component is the direct curriculum only.

The direct-template curriculum currently contains `1910` rows:

1. `600` direct-template rows
2. `100` motif drills
3. `350` operator-family drills
4. `400` symbol-digit encode/decode drills
5. `160` RHS-length family drills
6. `300` compact route cards

### Symbol-Equation Phase 2 CoT Export

```bash
python3 scripts/prepare_symbol_transform_synthetic_cot.py --target-rows 700 --direct-ratio 0.85 --output-csv data/trainable/symbol_transform_synthetic_cot_solver_verified_v2.csv --verify-with-solver
python3 scripts/prepare_symbol_transform_phase2_export.py --output-csv data/trainable/symbol_transform_phase2_combined.csv
```

Current exported file:

```text
data/trainable/symbol_transform_phase2_combined.csv
```

Final audit after expanding the CoT renderer:

```text
rows: 843
synthetic: 700
real train rows: 143
safe real rows: 136
query-completion public-rescue real rows: 7
final-answer plain mismatches: 0
phase-2 boxed decode mismatches: 0
phase-2 train symbol assistant extraction mismatches: 0
```

The `generated_cot` template now teaches the solver procedure explicitly:

1. classify the row as symbol-equation transformation with `ABOCD` structure
2. isolate same-operator examples first
3. try direct-position templates and reject the failed one
4. if direct templates fail, switch to encrypted digit-transform
5. state the bijective symbol-to-digit map used by the locked fit
6. scan candidate `motif|rule|output` triples in priority order
7. show failed candidate arithmetic, rendering, encoding, and rejection
8. lock the first consistent candidate
9. verify the locked rule on every same-operator example
10. apply the locked rule to the query and box the final answer

The real-row traces are intentionally more detailed than before. Current real
trace length statistics are roughly:

1. mean: `3445` characters
2. median: `3208` characters
3. max: `8778` characters
4. fallback solver traces: `0`
5. placeholder `None` text in real traces: `0`

The Phase 2 SFT builder now consumes this file directly:

1. real `symbol_transform` rows replace answer-only rows when their ids are in the active split
2. synthetic rows are appended after split selection, so they do not affect the split
3. symbol answers containing `}` use a pre-rendered `assistant_content` fallback line instead of malformed `\boxed{...}`

For the current `75/10/15` split, this contributes:

1. `104` real solved `symbol_transform` rows
2. `700` synthetic `symbol_transform` rows

Synthetic data recommendation:

1. keep the current 700 rows as a clean Phase 2 starter set
2. add another targeted synthetic batch later for encrypted digit-transform only
3. avoid using unverified random cipher rows, because many constructed rows are not uniquely recovered by the current solver
4. include more rows with distractor operators once the global-motif route is more stable
5. keep direct-template rows, but do not let them dominate future additions

### Combined Phase 2 Training Dataset

After refreshing the deterministic source files above, rebuild the active
Phase 2 SFT CSV:

```bash
python3 scripts/prepare_phase2_sft_dataset.py \
  --train-csv data/train.csv \
  --split-csv data/splits_75_10_15.config.json \
  --train-splits sft_train \
  --output-csv data/trainable/train_sft_phase2_75_10_15.csv
```

The active combined file is:

```text
data/trainable/train_sft_phase2_75_10_15.csv
```

## Practical Next Step

### Next Session Handoff

Start with `Transformation Rules`.

First task:

1. generate synthetic `numeric_equation` questions that strengthen the already solved rule families
2. keep synthetic rows outside the real train/GRPO/eval split, appending them only to Phase 2 training data
3. verify every synthetic row with the deterministic solver before adding it to training
4. render CoT in the same explicit solver-template style as the current high-confidence numeric-equation rows

Then:

1. continue improving the `numeric_equation` and `symbol_transform` solvers for higher real-row solve rate
2. prioritize rules that explain multiple real unsolved rows, not one-off hacks
3. update this method record and regenerate Phase 2 exports after each stable solver improvement

Once we figure out the remaining solver methods, we can convert this record into explicit knowledge datasets:

1. `facts`
2. `method exemplars`
3. `confidence-filtered worked traces`

At that point, the model can be injected not just with the Wonderland dictionary, but also with reusable solving procedures for the hard categories.
