# NVIDIA Nemotron Model Reasoning Challenge

This repository contains a deterministic LoRA fine-tuning pipeline for the
Nemotron reasoning challenge.

The current method uses two active training stages:

1. `Phase 1`: one LoRA epoch for compact knowledge injection and reusable
   methodology cards.
2. `Phase 2`: one LoRA epoch for SFT on deterministic solver-written CoT traces plus verified
   synthetic rows for the hard categories.

We do not use external teacher-model CoT generation in the active training
pipeline. Training traces should come from the deterministic scripts in
`scripts/` and the curated CSVs in `data/trainable/`.

## Active Training Schedule

The intended starter schedule is exactly:

1. Phase 1 uses `configs/phase1_training.json`: `1` epoch, LoRA training,
   learning rate `1e-4`, output adapter `outputs/phase1_training/adapter`.
2. Phase 2 uses `configs/cot_training_phase2_75_10_15.json`: `1` epoch, LoRA
   training, learning rate `5e-5`, initialized from
   `outputs/phase1_training/adapter`.

Both stages use LoRA rank `32` and LoRA alpha `32`.

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

Validate regenerated datasets before launching model training:

```bash
python3 train_sft.py --config configs/phase1_training.json --validate-only
python3 train_sft.py --config configs/cot_training_phase2_75_10_15.json --validate-only
python3 train_grpo.py --config configs/grpo_stage2.json --validate-only
```

These checks do not load the base model. They verify required CSV columns,
row counts, split wiring, duplicate ids, CoT/answer-only coverage, and GRPO
batch divisibility.

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
python3 train_grpo.py --config configs/grpo_stage2.json
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
