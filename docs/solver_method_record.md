# Solver Method Record

This is the compact method inventory for the cleaned training repository.

The full exploratory notebooks, renderer scripts, draft traces, and solver
experiments are intentionally not part of the GitHub-ready tree. The training
CSV files already contain the final traces used by SFT.

## Active Data Flow

1. Phase 1 teaches compact reusable knowledge and method cards from
   `data/training_ready_clean/phase1_train.csv`
2. Phase 2 teaches competition-style reasoning traces from
   `data/training_ready_clean/phase2_sft.csv`
3. `data/training_ready_clean/phase2_splits_80_10_10.csv` defines:
   `sft_train`, `eval_holdout`, and `grpo_holdout`
4. Default SFT trains on all Phase 1 rows plus Phase 2 `sft_train`
5. Optional GRPO trains on `eval_holdout`
6. Final local evaluation should use `grpo_holdout`

## Trusted Problem-Type Methods

### Text Cipher

Extract character mappings from examples, decode target words, use the fixed
Wonderland vocabulary when needed, and keep bijective consistency.

### Unit Conversion

Use a single linear conversion factor:

```text
factor = sum(outputs) / sum(inputs)
```

Round the computed target conversion to three decimals before reporting two
decimals.

### Gravity

Use the quadratic falling-distance form:

```text
d = k * t^2
k = sum(distance) / sum(t^2)
```

Compute the target distance and round to two decimals.

### Numeral System

Convert Arabic numbers to Roman numerals with the standard greedy table,
including subtractive forms such as `IX`, `XL`, `XC`, `CD`, and `CM`.

### Bit Manipulation

Use the original per-bit matching traces for the trusted Tier 1 set. These
trace rows are embedded in `phase2_sft.csv`; the experimental OP3 and deeper
search drafts are excluded from the cleaned project.

### Numeric Equation Transformation Rules

Use the numeric-equation methodology in
`docs/numeric_equation_methodology.md`.

Core workflow:

1. Group examples by visible operator
2. Try motifs `BA_DC` then `AB_CD`
3. Choose operator families from RHS length
4. Try templates and arithmetic rules in fixed priority order
5. Evaluate output formats in fixed priority order
6. Stop at the first answer-safe rule
7. Use deterministic low-confidence tie rules only where documented

### Symbol Transform

Use the symbol-transform methodology in
`docs/symbol_transform_methodology.md`.

Core workflow:

1. Group same-operator examples
2. Use RHS length to route direct templates or arithmetic families
3. Try direct templates first when length allows
4. If templates fail, assign global digit variables once
5. Try `BA_DC|rev` before `AB_CD|plain`
6. Keep explicit staged-rejection tables
7. Use helper rows only to complete the symbol-digit map
8. Decode the final boxed answer from the solved map

## Active Scripts

1. `train_sft.py`: minimal portable SFT trainer for Colab, Lightning AI, or local GPUs
2. `train_sft_kaggle.py`: Kaggle-oriented SFT trainer
3. `train_grpo.py`: optional GRPO trainer
4. `infer_eval.py`: local evaluation

## Competition Constraints

The active scripts enforce or follow the competition policy:

1. LoRA rank at most `32`
2. max model length `8192`
3. generation cap `7680`
4. final answers inside `\boxed{}`
