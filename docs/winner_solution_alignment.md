# Winner Solution Alignment Notes

This note summarizes the parts of the midpoint winner/reference materials that are most relevant to our repo.

Sources reviewed:

- [solution.md](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/reference/winner-solution/solution.md)
- [nemotron-tong-style-cot-sft-updated-v2.ipynb](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/reference/winner-reference/nemotron-tong-style-cot-sft-updated-v2.ipynb)
- [end-to-end-finetuning-for-lb-0-83.ipynb](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/reference/winner-solution/end-to-end-finetuning-for-lb-0-83.ipynb)

## Core Philosophy

The winner's approach is not centered on RL or teacher distillation.

The core bet is:

1. generate deterministic code-written traces
2. make each reasoning step mechanically easy
3. train with SFT so the model follows the deterministic policy
4. inspect low-logprob tokens and revise the trace template

Important quote-level ideas from the solution:

- temperature is `0.0`, so deterministic next-token accuracy matters
- optimize for minimum logprob, not just average loss
- make CoT steps simple and tokenization-aware
- do not aim for diversity
- do not rely on RL if the optimal policy is already known

This strongly supports our current direction of building deterministic solvers and trace renderers.

## Category Targets from Winner

The winner's stated target solve rates:

- Numeral: `1576 / 1576`
- Unit conversion: `1594 / 1594`
- Gravity: `1597 / 1597`
- Cipher/Text decryption: `1576 / 1576`
- Bit manipulation: `1364 / 1602`
- Equation numeric deduce: `540 / 596`
- Equation numeric guess: `21 / 136`
- Cryptarithm deduce: `54 / 659`
- Cryptarithm guess: `11 / 164`

Overall target:

- `8333 / 9500`
- `87.7%`

## Training Recipe

The winner-style notebook uses:

- LoRA rank: `32`
- LoRA alpha: `32`
- dropout: `0.0`
- max sequence length: `8192`
- learning rate: `2e-4`
- one epoch in the Tong-style reproduction notebook
- target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `in_proj`
  - `out_proj`
  - `up_proj`
  - `down_proj`
  - `lm_head`

The exact prompt suffix is:

```text
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

This matches the suffix we already switched to.

## Data Mix from Tong-Style Notebook

The local reference data includes these generated CoT CSVs:

- `numeral_system.csv`: `1497`
- `gravity_physics.csv`: `1516`
- `unit_conversion.csv`: `1513`
- `text_decryption.csv`: `1492`
- `bit_manipulation_including_wrong.csv`: `1508`
- `equation_numeric.csv`: `535`
- `equation_numeric_guess.csv`: `17`
- `cryptarithm.csv`: `69`
- `cryptarithm_guess.csv`: `13`

The notebook variants downsample and upweight categories. A representative high-performing mix is:

- Numeral: `600`
- Gravity: `1200`
- Unit conversion: `1150`
- Text decryption: `1492`
- Bit manipulation including wrong: `1508`
- Bit manipulation synthetic including wrong: `500`
- Equation numeric: `535`
- Cryptarithm: `69`

Important difference from our earlier setup:

- They train on intentionally generated deterministic traces, including some "including_wrong" traces for categories where failures or scan limits are part of the policy.
- They also duplicate priority low-logprob examples in some variants.

## Equation Numeric Alignment

This is the closest match to our current `digit_transform` work.

Winner stated target:

- deduce: `540 / 596`
- guess: `21 / 136`
- total: `561 / 732`

Our current stable numeric-equation solver:

- `550 / 732`

So we are close, but not identical.

Winner's trace structure is more verbose than our solver trace:

1. list examples
2. normalize operator-prefix outputs into negative values
3. identify query operator
4. if query operator appears, solve only that operator
5. scan common operations first
6. verify against all examples for that operator
7. apply to query
8. reapply prefix/suffix formatting if needed

This validates our method note in [digit_transform_methodology.md](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/docs/digit_transform_methodology.md), especially:

- same-operator solving first
- frequency-ordered scan
- text-preserving reverse
- prefix/suffix output handling

Potential improvement:

- render our numeric-equation CoT closer to their deterministic template, with explicit scan lines and "wrong/match/correct" labels.

## Cryptarithm / Symbol Transform Alignment

This was the biggest strategic correction from the reference solution.

The winner solution did **not** attempt a full encrypted `digit_transform`
solver for every cryptarithm row. Their cryptarithm strategy was intentionally
narrow:

1. check whether the query operator is concatenation
2. check whether the query operator is reverse concatenation
3. if the query operator is absent, guess concatenation

Winner target:

- `54 / 659` cryptarithm deduce
- `11 / 164` cryptarithm guess
- `65 / 823` total

The sample traces in `cryptarithm.csv` show exactly this style:

- split input into left/operator/right
- test concatenation
- test reverse concatenation
- mark operator as known or unknown
- solve query if its operator is known

Current repo decision:

1. keep the direct-template idea as the first and most important phase-1 skill
2. teach `0134` and `3401` with authentic and synthetic direct-template traces
3. add compact drills for `AB_CD` and `BA_DC` motifs
4. use encrypted digit search only after direct templates fail
5. restrict the hard branch to the two currently useful motifs: `BA_DC|rev` and `AB_CD|raw`

So we are now slightly more ambitious than the winner's public cryptarithm
branch, but the curriculum still starts with the proven direct-template habit
instead of throwing the model immediately into full cryptarithm search.

## Bit Manipulation Alignment

The winner identifies bit manipulation as the main differentiator.

Their traces are highly deterministic and verbose:

1. list every output bit by position
2. list output bit columns and bitsum hash
3. list every input bit by position
4. match per-bit boolean rules
5. compute target bit by bit

This confirms our earlier instinct:

- bit manipulation should not be concise
- per-bit serial reasoning is the point
- synthetic bit data may be valuable

Potential action:

- compare our bit-solver CoT with `bit_manipulation_including_wrong.csv`
- adopt their column/bitsum layout if it is easier for Nemotron to learn

## Text Decryption Alignment

Winner expects `100%` on text decryption.

Their philosophy is:

1. compute the current character mapping
2. collate unmapped characters
3. assign unmapped characters

This partly overlaps with our 77-word-memory idea, but their solution emphasizes deterministic character mapping rather than relying on hidden memory.

Potential action:

- keep Phase 1 Wonderland vocabulary injection
- but make Phase 2 Text Cipher traces more explicit and deterministic, closer to their char-by-char table style

## Training Infrastructure Alignment

The end-to-end notebook differs from our current repo in several important ways:

1. It trains from tokenized corpora, not plain text records.
2. Each example has:
   - `tokens`
   - `targets`
   - `weights`
3. Loss is masked/weighted over completion tokens.
4. LoRA parameters are cast to `fp32`.
5. LoRA includes `lm_head`.
6. MoE expert LoRA has Tinker-style tied weights.

This is more advanced than our current SFT stack.

Near-term practical takeaway:

- do not rewrite our trainer yet
- first align trace generation and data mix
- then decide whether token-level preprocessing / weighted loss is worth copying

## Recommended Alignment Plan

### Phase 1: Trace Alignment

1. Use winner-style deterministic traces for easy categories:
   - numeral
   - gravity
   - unit conversion
   - text decryption

2. Replace generic CoT with code-generated traces wherever possible.

3. Make numeric-equation traces more winner-like:
   - scan common operations
   - mark wrong/match/correct
   - verify explicitly

### Phase 2: Symbol Transform Curriculum

1. Keep direct templates as the starter skill:
   - `0134`: `ABOCD -> ABCD`
   - `3401`: `ABOCD -> CDAB`

2. Teach motif recognition before hard encrypted arithmetic:
   - `AB_CD|raw`
   - `BA_DC|rev`

3. Use the encrypted digit-transform branch only after direct templates fail.

4. Keep hard real traces solver-backed and detailed; use Phase 1 synthetic
   curriculum to make the direct-template behavior easy for the model first.

### Phase 3: Bit Manipulation Upgrade

1. Compare our bit traces against `bit_manipulation_including_wrong.csv`.
2. Add their output-column and bitsum sections if useful.
3. Add synthetic bit traces only after verifying our deterministic solver coverage.

### Phase 4: Training Mix

Use their category proportions as the first target mix:

- fewer easy traces than full data
- all text decryption
- all bit manipulation traces
- numeric-equation solved traces
- narrow cryptarithm traces only

Then add priority upweighting based on:

- held-out failures
- low minimum logprob
- formatting mistakes

## Bottom Line

The biggest alignment changes for us are:

1. prioritize deterministic code-generated traces over generated teacher CoT
2. treat `digit_transform` as almost solved enough; focus on trace quality
3. start `symbol_transform` with direct templates and compact motif drills
4. invest heavily in bit-manipulation trace quality
5. keep training simple until trace quality is aligned
