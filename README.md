# NVIDIA Nemotron Model Reasoning Challenge

Clean LoRA fine-tuning project for the NVIDIA Nemotron reasoning challenge.

The repository keeps only the training-ready data, active training/evaluation
scripts, local helper package, and concise methodology notes. Experimental
solver drafts and intermediate trace folders are intentionally excluded.

## Data

Core files:

1. `data/train.csv`: original competition train set
2. `data/test.csv`: original competition test set
3. `data/training_ready_clean/phase1_train.csv`: compact methodology and rule curriculum
4. `data/training_ready_clean/phase2_sft.csv`: competition-style SFT data with `generated_cot`
5. `data/training_ready_clean/phase2_splits_80_10_10.csv`: `sft_train`, `eval_holdout`, and `grpo_holdout`

Default SFT uses all Phase 1 rows plus Phase 2 `sft_train`.

Current validated counts:

1. Phase 1: `5077` rows
2. Phase 2 `sft_train`: `7348` rows
3. Combined default SFT: `12425` rows
4. Phase 2 holdout: `1836` rows
5. GRPO train bucket: `919` rows
6. Final local eval bucket: `917` rows

## Install

Use an H100 or L40S machine with a recent CUDA PyTorch environment.

```bash
pip install -r requirements.txt
```

## Train SFT

Set `MODEL_PATH` to the local Nemotron base model path or a Hugging Face model id.

Default combined Phase 1 + Phase 2 SFT:

```bash
MODEL_PATH=/path/to/Nemotron-3-Nano-30B python3 train_sft.py
```

Phase 1 only:

```bash
MODEL_PATH=/path/to/Nemotron-3-Nano-30B python3 train_sft.py --phase1-only
```

Validate data wiring without loading the model:

```bash
python3 train_sft.py --validate-only
python3 train_sft.py --phase1-only --validate-only
```

The minimal trainer uses:

1. LoRA rank `32`
2. sequence length `8192`
3. bf16 + TF32
4. batch size `1`, gradient accumulation `4`
5. no gradient checkpointing
6. no QLoRA

Outputs are written to `outputs/sft_combined_h100/` by default.

## Optional GRPO

After SFT, GRPO can train on the 919-row `eval_holdout` bucket while reserving
`grpo_holdout` for final local evaluation.

```bash
python3 train_grpo.py --config configs/grpo_stage2.json --validate-only
python3 train_grpo.py --config configs/grpo_stage2.json
```

## Local Evaluation

```bash
python3 infer_eval.py \
  --model-path /path/to/Nemotron-3-Nano-30B \
  --train-csv data/training_ready_clean/phase2_sft.csv \
  --adapter-dir outputs/sft_combined_h100/adapter \
  --split-csv data/training_ready_clean/phase2_splits_80_10_10.csv \
  --eval-splits grpo_holdout
```

The competition metric expects the final answer in `\boxed{...}`.

## Methodology

Core method notes live in:

1. `docs/solver_method_record.md`
2. `docs/digit_transform_methodology.md`
3. `docs/numeric_equation_methodology.md`
4. `docs/symbol_transform_methodology.md`
5. `docs/winner_solution_alignment.md`
