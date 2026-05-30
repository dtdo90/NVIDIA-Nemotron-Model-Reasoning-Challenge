# NVIDIA Nemotron Model Reasoning Challenge

Clean LoRA fine-tuning project for the NVIDIA Nemotron reasoning challenge.

The main training path is now single-phase SFT with optional GRPO. The old
two-phase SFT curriculum is preserved under `legacy/two_phase/` only for
reproducibility.

## Data

Core files:

1. `data/train.csv`: original competition train set
2. `data/test.csv`: original competition test set
3. `data/single_phase_training_clean/single_phase_sft.csv`: active SFT corpus
4. `data/single_phase_training_clean/single_phase_splits_80_10_10.csv`: optional GRPO/eval split
5. `data/single_phase_training_clean/manifest.json`: source counts and split metadata

Current validated single-phase counts:

1. SFT corpus: `12491` rows
2. Optional GRPO train bucket, named `eval_holdout`: `1261` rows
3. Final local eval bucket, named `grpo_holdout`: `1237` rows
4. SFT split bucket, named `sft_train`: `9993` rows

The single-phase corpus contains real traces plus selected synthetic curriculum
rows. Bit manipulation uses HuiKang-style traces, while transformation-rule and
text-cipher traces use the cleaned methodology formats.

## Install

Use an H100/H200-class machine when possible. RTX6000/L40S can work with smaller
micro-batches and gradient checkpointing.

```bash
pip install -r requirements.txt
pip uninstall -y torchvision
pip install --no-build-isolation --no-deps -r requirements-nemotron.txt
```

For vLLM evaluation:

```bash
pip install "numpy<2" vllm
```

The scripts use Kaggle's mounted model at
`/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`
when that path exists. Otherwise they use
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. Override with `--model-path`,
`MODEL_PATH`, or `BASE_MODEL_PATH`.

## Train SFT

Recommended single-phase SFT:

```bash
python3 train_sft_single_phase.py \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

This trains fresh LoRA weights on
`data/single_phase_training_clean/single_phase_sft.csv` for one epoch at
learning rate `2e-4` and writes:

1. final adapter: `outputs/sft_single_phase_h200/adapter`
2. submission zip: `outputs/sft_single_phase_h200/submission.zip`
3. run metadata: `outputs/sft_single_phase_h200/run_config.json`

Validate data wiring without loading the model:

```bash
python3 train_sft_single_phase.py --validate-only
```

Default trainer settings:

1. LoRA rank `32`
2. sequence length `8192`
3. bf16 + TF32
4. cosine LR schedule with warmup ratio `0.05`
5. minimum learning rate floor `2e-6`
6. optimizer `adamw_torch_fused`
7. LoRA dropout `0.05`
8. assistant-only loss masking
9. competition chat-template prompt format

If memory is tight:

```bash
python3 train_sft_single_phase.py \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing
```

## Optional GRPO

GRPO remains an optional second step for both regimes. In the active single-phase
regime, it starts from `outputs/sft_single_phase_h200/adapter` and trains on the
10% bucket named `eval_holdout`.

Smoke-check the wiring:

```bash
python3 train_grpo.py --config configs/grpo_stage2.json --validate-only
```

Run GRPO:

```bash
python3 train_grpo.py --config configs/grpo_stage2.json
```

The active config is `configs/grpo_stage2.json`. To run GRPO on the archived
two-phase SFT adapter instead, pass
`--config legacy/two_phase/configs/grpo_stage2.json`.

## Local Evaluation

Evaluate the single-phase SFT adapter on the final held-out bucket:

```bash
python3 infer_eval.py \
  --train-csv data/single_phase_training_clean/single_phase_sft.csv \
  --adapter-dir outputs/sft_single_phase_h200/adapter \
  --split-csv data/single_phase_training_clean/single_phase_splits_80_10_10.csv \
  --eval-splits eval_holdout grpo_holdout \
  --backend vllm \
  --max-model-len 8192 \
  --max-new-tokens 7680
```

Evaluate the optional GRPO training bucket if needed:

```bash
python3 infer_eval.py \
  --train-csv data/single_phase_training_clean/single_phase_sft.csv \
  --adapter-dir outputs/sft_single_phase_h200/adapter \
  --split-csv data/single_phase_training_clean/single_phase_splits_80_10_10.csv \
  --eval-splits eval_holdout grpo_holdout \
  --backend vllm \
  --max-model-len 8192 \
  --max-new-tokens 7680
```

For a smoke test, add `--max-eval-samples 20`. If vLLM is unavailable, pass
`--backend transformers`.

The competition metric expects the final answer in `\boxed{...}`.

## Legacy

Archived two-phase SFT files live in `legacy/two_phase/`:

1. `legacy/two_phase/train_sft.py`
2. `legacy/two_phase/train_sft_kaggle.py`
3. `legacy/two_phase/data/training_ready_clean/`
4. `legacy/two_phase/configs/`

These files are retained to reproduce older runs, but they are not the default
training path.

## Methodology

Core method notes live in:

1. `docs/solver_method_record.md`
2. `docs/digit_transform_methodology.md`
3. `docs/numeric_equation_methodology.md`
4. `docs/symbol_transform_methodology.md`
5. `docs/winner_solution_alignment.md`
