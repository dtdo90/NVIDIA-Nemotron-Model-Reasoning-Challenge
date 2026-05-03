# Nemotron Reasoning Challenge Training Pipeline

This repository uses a deterministic, solver-written data pipeline for LoRA
training. The current strategy is:

1. `Phase 1`: inject compact knowledge and reusable methodology cards.
2. `Phase 2`: continue SFT on deterministic full task traces plus verified
   synthetic rows for the hard families.
3. `Optional Phase 3`: run GRPO from the Phase 2 adapter.

We no longer use external teacher-model or notebook-generated traces as part of
the active training-data build. CoT rows used for training should come from the
deterministic renderers and solver-backed CSV builders in `scripts/`.

## Core Files

Training and evaluation:

1. `train_sft.py`: trains answer-only or CoT-supervised LoRA adapters.
2. `infer_eval.py`: evaluates adapters on local held-out splits.
3. `train_grpo.py`: optional GRPO update after SFT.

Active trainable datasets:

1. `data/trainable/phase1_train.csv`
2. `data/trainable/train_sft_phase2_75_10_15.csv`

Important method records:

1. [docs/solver_method_record.md](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/docs/solver_method_record.md)
2. [docs/digit_transform_methodology.md](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/docs/digit_transform_methodology.md)
3. [docs/symbol_transform_methodology.md](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/docs/symbol_transform_methodology.md)
4. [docs/winner_solution_alignment.md](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/docs/winner_solution_alignment.md)

## Active Training Schedule

The starter training loop is intentionally two SFT stages:

1. `Phase 1`: `configs/phase1_training.json`
2. `Phase 2`: `configs/cot_training_phase2_75_10_15.json`

Exact schedule:

1. Phase 1 trains a LoRA adapter for `1` epoch with learning rate `1e-4`.
2. Phase 1 writes the adapter to `outputs/phase1_training/adapter`.
3. Phase 2 initializes from `outputs/phase1_training/adapter`.
4. Phase 2 continues LoRA training for `1` epoch with learning rate `5e-5`.

Both active configs use LoRA rank `32`, LoRA alpha `32`, and gradient
accumulation `4`.

## Persistent Splits

Create the fixed `75/10/15` split used by SFT, GRPO, and local eval:

```bash
python3 scripts/make_splits.py \
  --input-csv data/train.csv \
  --output-csv data/splits_75_10_15.csv
```

This writes both:

1. `data/splits_75_10_15.csv`
2. `data/splits_75_10_15.config.json`

The split names are:

1. `sft_train`
2. `grpo_train`
3. `eval`

The split is stratified by category, useful subcategory, and deterministic
solve source. Synthetic rows are appended only after split selection and should
not be counted when auditing the real train/GRPO/eval split.

## Phase 1 Dataset

Phase 1 combines stable facts and short methodology cards into one CSV:

```bash
python3 scripts/prepare_text_knowledge_phase1.py
python3 scripts/prepare_phase1a_bit_manipulation_knowledge.py
python3 scripts/prepare_phase1b_bit_manipulation_methodology.py
python3 scripts/prepare_phase1a_numeric_equation_knowledge.py
python3 scripts/prepare_phase1b_numeric_equation_methodology.py
python3 scripts/prepare_phase1a_symbol_transform_knowledge.py
python3 scripts/prepare_phase1b_symbol_transform_methodology.py
python3 scripts/prepare_phase1_symbol_transform_direct_curriculum.py
python3 scripts/prepare_phase1_training_dataset.py
```

The current combined file is:

```text
data/trainable/phase1_train.csv
```

Current Phase 1 size:

```text
9244 rows
```

Phase 1 intentionally excludes unit conversion, gravity, and numeral-system
knowledge cards from the default mixture because those problem types are simple
enough to learn from deterministic Phase 2 traces.

Train Phase 1:

```bash
python3 train_sft.py \
  --config configs/phase1_training.json
```

Preflight regenerated Phase 1 data without loading the model:

```bash
python3 train_sft.py \
  --config configs/phase1_training.json \
  --validate-only
```

Config check:

1. `num_epochs`: `1.0`
2. `learning_rate`: `0.0001`
3. `output_dir`: `outputs/phase1_training`

## Phase 2 Dataset

Phase 2 is the main task SFT dataset. It uses the `sft_train` split of
`data/train.csv`, replaces supported rows with deterministic CoT traces, keeps
unsupported rows answer-only, then appends verified synthetic data.

Build deterministic source files:

```bash
python3 scripts/prepare_text_cipher_compact_cot.py
python3 scripts/prepare_bit_manipulation_phase2_cot.py
python3 scripts/prepare_unit_conversion_phase2_cot.py
python3 scripts/prepare_gravity_phase2_cot.py
python3 scripts/prepare_numeral_phase2_cot.py
python3 scripts/prepare_numeric_equation_phase2_cot.py
python3 scripts/prepare_numeric_equation_synthetic_cot.py --variants-per-row 2
python3 scripts/prepare_symbol_transform_synthetic_cot.py --target-rows 700 --direct-ratio 0.85 --output-csv data/trainable/symbol_transform_synthetic_cot_solver_verified_v2.csv --verify-with-solver
python3 scripts/prepare_symbol_transform_phase2_export.py --output-csv data/trainable/symbol_transform_phase2_combined.csv
```

Build the split-aware Phase 2 training CSV:

```bash
python3 scripts/prepare_phase2_sft_dataset.py \
  --train-csv data/train.csv \
  --split-csv data/splits_75_10_15.config.json \
  --train-splits sft_train \
  --output-csv data/trainable/train_sft_phase2_75_10_15.csv
```

The current Phase 2 file has `8695` rows. Its largest source buckets are:

1. gravity deterministic traces: `1197`
2. unit-conversion deterministic traces: `1196`
3. text-cipher compact traces: `1182`
4. numeral greedy-Roman traces: `1182`
5. bit-manipulation hybrid traces: `1132`
6. numeric-equation synthetic traces: `870`
7. symbol-transform synthetic traces: `700`
8. answer-only fallback rows: `610`
9. numeric-equation real deterministic/oracle rows: `522`
10. symbol-transform real solver rows: `104`

Train Phase 2 from the Phase 1 adapter:

```bash
python3 train_sft.py \
  --config configs/cot_training_phase2_75_10_15.json
```

Preflight regenerated Phase 2 data without loading the model:

```bash
python3 train_sft.py \
  --config configs/cot_training_phase2_75_10_15.json \
  --validate-only
```

`train_sft.py` supports `--init-adapter-dir`, and the Phase 2 config uses it
to continue training the same LoRA adapter learned in Phase 1.

Config check:

1. `init_adapter_dir`: `outputs/phase1_training/adapter`
2. `num_epochs`: `1.0`
3. `learning_rate`: `0.00005`
4. `output_dir`: `outputs/cot_training_phase2_75_10_15`

## Local Evaluation

Evaluate the Phase 2 adapter on the fixed held-out `eval` split:

```bash
python3 infer_eval.py \
  --run-config outputs/cot_training_phase2_75_10_15/run_config.json \
  --adapter-dir outputs/cot_training_phase2_75_10_15/adapter \
  --split-csv data/splits_75_10_15.csv \
  --eval-splits eval
```

The evaluation prompt suffix is:

```text
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

Training traces should therefore end with:

```text
The final answer is \boxed{...}
```

## Optional GRPO Stage

After Phase 2 SFT, run GRPO from the Phase 2 adapter:

```bash
python3 train_grpo.py \
  --config configs/grpo_stage2.json
```

Preflight GRPO split wiring and batch divisibility without loading the model:

```bash
python3 train_grpo.py \
  --config configs/grpo_stage2.json \
  --validate-only
```

Then compare SFT-only and GRPO adapters on the same held-out `eval` split with
`infer_eval.py`.

## Current Method Principles

The active pipeline follows these rules:

1. Prefer deterministic solvers and rendered CoT templates over external model
   generation.
2. Keep Phase 1 compact and reusable: facts, rule names, route cards, and small
   canonical worked examples.
3. Keep Phase 2 task-like: full deterministic traces in the same prompt format
   as competition evaluation.
4. Use the 77-word Wonderland vocabulary for text cipher knowledge, but teach
   the solving behavior through compact mapping/vocabulary-completion traces.
5. Treat unit conversion, gravity, and numeral system as Phase 2-only simple
   deterministic families.
6. Give bit manipulation, numeric equation, and symbol transform explicit
   Phase 1 methodology support.
7. For symbol transform, emphasize direct template matching first, then
   encrypted digit search with `BA_DC|rev` or `AB_CD|raw` only when needed.
