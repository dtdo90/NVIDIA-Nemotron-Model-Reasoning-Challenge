# Project Handoff: Current Data, Training, and Evaluation State

This file is the quick-start state record for continuing the project in a fresh
chat. It intentionally points to the existing source-of-truth docs, manifests,
and scripts instead of trying to duplicate every trace-design detail.

## Current Goal

The active regime is single-phase SFT on a curated reasoning-trace corpus, with
optional GRPO later. The old two-phase SFT pipeline is legacy and lives under
`legacy/two_phase/`.

Main working questions:

1. Keep the easy types at 100%: Gravity, Unit Conversion, Numeral System.
2. Improve Text Cipher without hurting its current strong baseline.
3. Improve Numeric Equation and Symbol Transform trace quality.
4. Be very cautious with curriculum rows: recent evidence suggests some
   curriculum-style rows may hurt if their format drifts from normal examples.

## Active Files

Use these as the active source of truth:

- `data/single_phase_training_clean/single_phase_sft_v2.csv`
- `data/single_phase_training_clean/single_phase_splits_80_10_10.csv`
- `data/single_phase_training_clean/manifest.json`
- `experiments/type_diagnostics/data/all_types_summary.json`
- `train_sft_single_phase.py`
- `infer_eval.py`
- `experiments/type_diagnostics/prepare_type_datasets.py`
- `experiments/type_diagnostics/README.md`

Important: `README.md` currently contains stale validated counts. The current
counts below come from the live CSV/split files and `manifest.json`.

## Current Live Counts

Computed from `single_phase_sft_v2.csv` joined with
`single_phase_splits_80_10_10.csv`:

- Total rows: `18477`
- `sft_train`: `16680`
- `eval_holdout`: `901`
- `grpo_holdout`: `896`

Category totals:

- Bit Manipulation: `6117`
- Numeric Equation Transformation Rules: `4733`
- Text Cipher: `1976`
- Gravity: `1597`
- Unit Conversion: `1594`
- Numeral System: `1576`
- Symbol Transform: `884`

Category split counts:

- Bit Manipulation: `5797` train, `160` eval, `160` grpo
- Gravity: `1277` train, `160` eval, `160` grpo
- Numeral System: `1260` train, `158` eval, `158` grpo
- Numeric Equation: `4594` train, `72` eval, `67` grpo
- Symbol Transform: `816` train, `34` eval, `34` grpo
- Text Cipher: `1660` train, `158` eval, `158` grpo
- Unit Conversion: `1276` train, `159` eval, `159` grpo

Prompt formats:

- All `18477` active rows currently use `competition_chat_template`.
- No active rows currently use `decision_point_chat_template`.
- Earlier decision-point curriculum has been converted into normal full
  question + full CoT samples.

Evaluation eligibility:

- `eval_eligible=false`: `9543`
- `eval_eligible=true`: `8934`
- `split_policy=train_only`: `9543`
- `split_policy=auto`: `8792`
- `split_policy=eval_only`: `142`

## Data Creation and Documentation Map

General data build:

- `scripts/build_single_phase_training_clean.py`
- `data/single_phase_training_clean/manifest.json`
- `scripts/rebuild_honest_single_phase_splits.py`
- `experiments/type_diagnostics/prepare_type_datasets.py`

Numeric Equation:

- Methodology: `docs/numeric_equation_methodology.md`
- Decision-point curriculum docs: `docs/numeric_equation_decision_point_curriculum/`
- Decision-point manifest: `docs/numeric_equation_decision_point_curriculum/manifest.csv`
- Decision-point generator: `scripts/generate_numeric_equation_decision_point_curriculum.py`
- CSV add script: `scripts/add_numeric_equation_decision_point_curriculum_rows.py`
- Synthetic/render scripts include:
  - `scripts/generate_numeric_equation_direct_template_synthetic.py`
  - `scripts/generate_numeric_equation_ba_dc_subtraction_synthetic.py`
  - `scripts/generate_numeric_equation_ab_cd_addition_synthetic.py`
  - `scripts/generate_numeric_equation_ab_cd_multiplication_synthetic.py`
  - `scripts/generate_numeric_equation_ab_cd_subtraction_synthetic.py`
  - `scripts/generate_numeric_equation_rare_synthetic_traces.py`
  - `scripts/normalize_numeric_equation_work_on_traces.py`

Text Cipher:

- Decision-point curriculum docs: `docs/text_cipher_decision_point_curriculum/`
- Decision-point manifest: `docs/text_cipher_decision_point_curriculum/manifest.csv`
- Generator: `scripts/generate_text_cipher_decision_point_curriculum.py`
- Confusion synthetic generator: `scripts/generate_text_cipher_confusion_synthetic.py`
- Format normalization/audit:
  - `scripts/normalize_text_cipher_cot_format.py`
  - `scripts/apply_text_cipher_exhaustive_vocab_scan.py`
  - `scripts/audit_text_cipher_exhaustive_vocab_scan.py`
  - `scripts/regenerate_text_cipher_traces.py`

Bit Manipulation:

- HuiKang alignment doc: `docs/winner_solution_alignment.md`
- Winner methodology: `reference/winner-solution/bit_methodology.md`
- Winner solution summary: `reference/winner-solution/solution.md`
- Import script: `scripts/add_huikang_bit_methodology_rows.py`
- Current bit source modes:
  - `huikang_real_bit`: `1364`
  - `huikang_real_bit_extra_trace`: `238`
  - `huikang_synthetic_matching`: `4515`

Symbol Transform:

- Methodology: `docs/symbol_transform_methodology.md`
- Direct-template audit docs:
  - `docs/symbol_transform_phase1_direct_template_traces/AUDIT.md`
  - `docs/symbol_transform_phase1_direct_template_traces/AUDIT_safe_v2.md`
- BA/DC scratch traces are archived in `docs/symbol_transform_ba_dc_rev_traces/`
  and related folders, but weak BA/DC symbol traces should not be assumed safe.

Gravity, Unit Conversion, Numeral System:

- Normalization scripts:
  - `scripts/normalize_gravity_cot_format.py`
  - `scripts/normalize_unit_conversion_cot_format.py`
  - `scripts/normalize_numeral_cot_format.py`
- These trace families were audited for arithmetic/rounding. Gravity, Unit
  Conversion, and Numeral System reached 100% in the last reported full run.

## Current Type-Diagnostic Data

Regenerate all per-type datasets and sync the root split with:

```bash
python3 experiments/type_diagnostics/prepare_type_datasets.py
```

Per-type data lives under:

```text
experiments/type_diagnostics/data/{type}/
```

Each type has:

- `{type}.csv`
- `splits_80_10_10.csv`
- `dataset_summary.json`

Important diagnostic subtype/source summaries:

Numeric Equation:

- Total: `4733`
- Split: `4594` train, `72` eval, `67` grpo
- Source modes:
  - `real`: `624`
  - `synthetic`: `3187`
  - `numeric_equation_decision_point_curriculum`: `900`
  - `numeric_equation_untrained_eval_only`: `22`

Numeric decision-point curriculum buckets:

- `common_intersection`: `290`
- `output_format_rendering`: `219`
- `low_confidence_branch_discipline`: `170`
- `literal_minus_rendering_policy`: `120`
- `literal_minus_opposite_sign_continuation`: `60`
- `operator_absence_fallback_choice`: `41`

Text Cipher:

- Total: `1976`
- Split: `1660` train, `158` eval, `158` grpo
- Source modes:
  - `real`: `1576`
  - `single_phase_synthetic_text_cipher_confusion`: `200`
  - `text_cipher_decision_point_curriculum`: `200`
- Active text-cipher decision-point/full-synthetic subtypes:
  - `decision_point_phrase_alignment`: `47`
  - `decision_point_phrase_copy_alignment`: `45`
  - `decision_point_phrase_reject_failed_candidate`: `57`
  - `decision_point_phrase_repeated_letter`: `51`

Symbol Transform:

- Total: `884`
- Split: `816` train, `34` eval, `34` grpo
- Subtypes:
  - `direct_template_template0134`: `400`
  - `direct_template_template3401`: `200`
  - `operator_absence_template0134`: `164`
  - `untrained_eval_only`: `120`
- Warning: current split still has `96` `untrained_eval_only` symbol rows in
  `sft_train` (`87` from `symbol_transform_untrained_eval_only` plus `9` real
  untrained rows). This conflicts with the earlier intent to train only direct
  templates plus operator-absence guesses. Verify this before another full run.

Numeric warning:

- Current split has `18` `numeric_equation_untrained_eval_only` rows in
  `sft_train`, with `2` in each holdout. Verify whether that is intentional.

## Training Script Design

Main script: `train_sft_single_phase.py`

Defaults:

- Train CSV: `data/single_phase_training_clean/single_phase_sft_v2.csv`
- Split CSV: `data/single_phase_training_clean/single_phase_splits_80_10_10.csv`
- Train split: `sft_train`
- Output dir: `outputs/sft_single_phase`
- Model path:
  - Kaggle path if present:
    `/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`
  - Otherwise: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Override via `--model-path`, `MODEL_PATH`, or `BASE_MODEL_PATH`.

LoRA:

- Rank: `32`
- Alpha: `32`
- Dropout: `0.0`
- Bias: `none`
- Task type: causal LM
- Target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `in_proj`
  - `out_proj`
  - `up_proj`
  - `down_proj`
  - `lm_head`

Trainer:

- Epochs: `1.0`
- LR: `2e-4`
- Minimum LR floor: `2e-6`
- LR schedule: cosine with warmup ratio `0.05`
- Optimizer: `adamw_torch`
- Batch size: default `1`
- Gradient accumulation: default `8`
- Max sequence length: `8192`
- bf16: on
- TF32: on
- Max grad norm: `1.0`
- Checkpoint saving: `save_strategy="no"`
- Loss masking: assistant-only
- `PYTORCH_CUDA_ALLOC_CONF` defaults to `expandable_segments:True`

Recommended command:

```bash
python3 train_sft_single_phase.py \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --balanced-accumulation
```

Validation commands:

```bash
python3 train_sft_single_phase.py --validate-only
python3 train_sft_single_phase.py --validate-tokenization
```

Colab/Drive save behavior:

- The script avoids putting Trainer scratch state directly on Google Drive when
  possible.
- `--trainer-state-dir` can override scratch state.
- `--mirror-output-dir` can mirror final adapter artifacts back to Drive.
- Final adapter should contain:
  - `adapter_config.json`
  - `adapter_model.safetensors`

## Balanced Accumulation

`--balanced-accumulation` does not concatenate examples. It keeps one puzzle per
sequence.

Main single-phase training balances by question `category` within each effective
gradient-accumulation window.

Type-diagnostic training balances by `diagnostic_subtype` within each effective
window.

If `--balanced-accumulation` is not used, Trainer uses normal sampling and
`group_by_length=True`.

## Chat Template and Loss Masking

The training script calls the Nemotron competition chat template with
`enable_thinking=True`, which opens the assistant turn with `<think>`.

For normal `competition_chat_template` rows:

1. Prompt side is rendered with the user prompt plus the boxed-answer
   instruction.
2. The chat template opens the assistant generation prompt.
3. If the stored assistant trace starts with `<think>`, the training script
   strips that duplicated opening tag from the scored completion.
4. The completion includes the reasoning, then:

```text
Answer: \boxed{...}
</think>
\boxed{...}
<|im_end|>
```

The exact end token comes from the tokenizer/template helper.

The tokenizer masking code asserts:

```python
input_ids[: len(prompt_ids)] == prompt_ids
```

If tokenization merges across the prompt/completion boundary, training fails
loudly instead of silently shifting labels.

All prompt tokens are masked with `-100`; only assistant completion tokens are
scored.

## Type-Diagnostic Training and Ablations

Prepare data:

```bash
python3 experiments/type_diagnostics/prepare_type_datasets.py
```

Train one type:

```bash
python3 experiments/type_diagnostics/scripts/train_numeric_equation.py \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --balanced-accumulation
```

Suspicious curriculum ablation wrappers:

```bash
python3 experiments/type_diagnostics/scripts/train_numeric_equation_with_curriculum.py
python3 experiments/type_diagnostics/scripts/train_numeric_equation_without_curriculum.py
python3 experiments/type_diagnostics/scripts/train_text_cipher_with_curriculum.py
python3 experiments/type_diagnostics/scripts/train_text_cipher_without_curriculum.py
```

Before expensive runs, use:

```bash
python3 experiments/type_diagnostics/scripts/train_numeric_equation_with_curriculum.py --validate-tokenization
python3 experiments/type_diagnostics/scripts/train_text_cipher_with_curriculum.py --validate-tokenization
```

## Evaluation

Main evaluator: `infer_eval.py`

Evaluate final holdout:

```bash
python3 infer_eval.py \
  --train-csv data/single_phase_training_clean/single_phase_sft_v2.csv \
  --adapter-dir outputs/sft_single_phase/adapter \
  --split-csv data/single_phase_training_clean/single_phase_splits_80_10_10.csv \
  --eval-splits eval_holdout
```

Evaluate both holdouts:

```bash
python3 infer_eval.py \
  --train-csv data/single_phase_training_clean/single_phase_sft_v2.csv \
  --adapter-dir outputs/sft_single_phase/adapter \
  --split-csv data/single_phase_training_clean/single_phase_splits_80_10_10.csv \
  --eval-splits grpo_holdout eval_holdout
```

Notes:

- vLLM evaluation should use `VLLM_USE_V1=0` for this Nemotron LoRA setup.
- `infer_eval.py` follows `reference/evaluation.py` style extraction and
  numeric tolerance from `description.md`.
- Failed generations are grouped by type/subtype; up to three are printed and
  saved for non-100% subtypes.

Last reported full-run result from the user:

- Overall: `1700/1912 = 0.8891`
- Bit Manipulation: `284/320 = 0.8875`
- Gravity: `320/320 = 1.0`
- Unit Conversion: `318/318 = 1.0`
- Text Cipher: `302/316 = 0.9557`
- Numeral System: `316/316 = 1.0`
- Numeric Equation: `123/157 = 0.7834`
- Symbol Transform: `37/165 = 0.2242`

Interpretation at that time:

- Gravity, Unit Conversion, Numeral System trace fixes worked.
- Numeric Equation and Symbol Transform still need careful trace/data work.
- Text Cipher is strong but curriculum additions must be tested carefully.

## Recent Audit Findings To Continue From

Numeric decision-point curriculum, 900 active rows:

- No issues found in prompt parsing, compare-operator blocks,
  same-operator example selection, RHS-length routing, table width, operand
  rendering, `Match` lists, or true `Common` intersections after respecting the
  numeric leading-zero convention.
- Hard formatting risk: `syn_ne_dp_common_intersection_0230` has answer `}52`,
  and final line `Answer: \boxed{}52}` is structurally ambiguous because the
  answer begins with `}`.
- Fixed: `28` rows with `56` `Common` / `none` failure-bridge instances now
  include a blank separator before `x-y fails under BA_DC` or
  `y-x fails under BA_DC`.
- Fixed: decision-point and non-curriculum numeric traces now use
  `Apply format AB_CD|...|common to the query` /
  `Apply format BA_DC|...|common to the query` wording.

Text Cipher current conventions:

- Target decoding should be strict left-to-right.
- Vocab scan candidates now use explicit headers like `cipherword -> candidate`.
- Vocab scan order should preserve first-seen order.
- Synthetic text-cipher curriculum should use normal full question style, not
  `Solution:` partial-trace style.
- One-word synthetic curriculum examples were removed; current promoted
  curriculum focuses on 2-5 word phrase cases.

Symbol Transform warnings:

- Intended training focus was direct-template rows and operator-absence
  template0134 guess rows.
- Current split still places `untrained_eval_only` symbol rows in `sft_train`.
  Re-check this before another full SFT run.

Numeric Equation warnings:

- Some `numeric_equation_untrained_eval_only` rows are also in `sft_train`.
  Re-check whether this matches the desired split policy.
- Direct-template matching must preserve leading zeros.
- Arithmetic BA/DC and AB/CD table values often display numerically, so `01`
  may appear as `1` in arithmetic panels; do not confuse this with direct
  template leading-zero loss.

## Git/Workspace Caution

At the time this handoff was written, the workspace was dirty with many data and
doc changes, plus untracked backup CSVs and newly generated text-cipher docs.
Before committing or pushing, run:

```bash
git status --short
```

Do not revert user/data changes casually. The current data changes are likely
part of the active experiment history.

## Recommended Next Steps

1. Decide whether to fix or remove the ambiguous `}52` numeric curriculum row.
2. Fix the 28 numeric decision-point rows with contaminated `Common / none`
   blocks.
3. Decide whether to globally standardize `Apply format ...` wording.
4. Re-check split intent for `symbol_transform_untrained_eval_only` and
   `numeric_equation_untrained_eval_only`.
5. Run:

```bash
python3 experiments/type_diagnostics/prepare_type_datasets.py
python3 train_sft_single_phase.py --validate-only
python3 train_sft_single_phase.py --validate-tokenization
```

6. Use type diagnostics before any full 18-hour single-phase rerun.
