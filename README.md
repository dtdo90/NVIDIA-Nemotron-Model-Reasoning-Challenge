# NVIDIA Nemotron Model Reasoning Challenge

This repository contains a deterministic LoRA fine-tuning pipeline for the
Nemotron reasoning challenge.

The current method uses two active training stages:

1. `Phase 1`: compact knowledge injection and reusable methodology cards.
2. `Phase 2`: SFT on deterministic solver-written CoT traces plus verified
   synthetic rows for the hard categories.

We do not use external teacher-model CoT generation in the active training
pipeline. Training traces should come from the deterministic scripts in
`scripts/` and the curated CSVs in `data/trainable/`.

## Active Training Data

Main trainable files:

1. `data/trainable/phase1_train.csv`
2. `data/trainable/train_sft_phase2_75_10_15.csv`

Persistent split files:

1. `data/splits_75_10_15.csv`
2. `data/splits_75_10_15.config.json`

Synthetic rows are appended after split selection, so they should not be used
when auditing the real train/GRPO/eval split.

## Build Datasets

Regenerate the current deterministic data path:

```bash
python3 scripts/make_splits.py --input-csv data/train.csv --output-csv data/splits_75_10_15.csv
python3 scripts/prepare_phase1_training_dataset.py
python3 scripts/prepare_phase2_sft_dataset.py --train-csv data/train.csv --split-csv data/splits_75_10_15.config.json --train-splits sft_train --output-csv data/trainable/train_sft_phase2_75_10_15.csv
```

When source components need refreshing, use the deterministic builders listed
in [README_baseline.md](README_baseline.md).

## Train

Train Phase 1:

```bash
python3 train_sft.py --config configs/phase1_training.json
```

Train Phase 2 from the Phase 1 adapter:

```bash
python3 train_sft.py --config configs/cot_training_phase2_75_10_15.json
```

Optional GRPO:

```bash
python3 train_grpo.py --config configs/grpo_stage2.json --sft-adapter-dir outputs/cot_training_phase2_75_10_15/adapter --output-dir outputs/grpo_stage2
```

## Method Notes

Current method records:

1. [docs/solver_method_record.md](docs/solver_method_record.md)
2. [docs/digit_transform_methodology.md](docs/digit_transform_methodology.md)
3. [docs/symbol_transform_methodology.md](docs/symbol_transform_methodology.md)
4. [docs/winner_solution_alignment.md](docs/winner_solution_alignment.md)

The important high-level rule is simple: teach stable reusable knowledge in
Phase 1, then teach exact competition-style task behavior with deterministic
Phase 2 traces.
