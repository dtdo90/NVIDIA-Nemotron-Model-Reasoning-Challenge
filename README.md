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
4. `data/single_phase_training_clean/single_phase_splits_80_10_10.csv`: canonical SFT/GRPO/eval split
5. `data/single_phase_training_clean/manifest.json`: source counts and split metadata
6. `experiments/type_diagnostics/data/global_splits_80_10_10.csv`: same split assignment, generated from the per-type diagnostics

Current validated single-phase counts:

1. Full SFT corpus: `12487` rows
2. SFT training bucket, named `sft_train`: `9982` rows
3. Optional GRPO train bucket, named `eval_holdout`: `1255` rows
4. Final local eval bucket, named `grpo_holdout`: `1250` rows

The single-phase corpus contains real traces plus selected synthetic curriculum
rows. Bit manipulation uses HuiKang-style traces, while transformation-rule and
text-cipher traces use the cleaned methodology formats.

The split ratios are approximate. The important invariant is that the full
single-phase run and the per-question-type diagnostic runs use the same row-level
split assignment. Regenerate and sync the split with:

```bash
python3 experiments/type_diagnostics/prepare_type_datasets.py
```

This writes the seven per-type split files and copies their union to
`data/single_phase_training_clean/single_phase_splits_80_10_10.csv`.

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
pip uninstall -y vllm opencv-python-headless
pip install "vllm==0.18.0"
pip install -U "scipy>=1.14" "pandas>=2.2.3" "scikit-learn>=1.5" "matplotlib>=3.9"
```

Use `VLLM_USE_V1=0` for Nemotron LoRA inference. Newer/default vLLM V1 paths
can try to register Nemotron `mixer.conv1d` as a LoRA layer and fail with
`BaseLayerWithLoRA` assertions.

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

This trains fresh LoRA weights for one epoch at learning rate `2e-4`. By
default it uses only the `sft_train` rows from
`data/single_phase_training_clean/single_phase_splits_80_10_10.csv`, which is
the same split assignment used by the type-diagnostic experiments.

1. final adapter: `outputs/sft_single_phase_h200/adapter`
2. submission zip: `outputs/sft_single_phase_h200/submission.zip`
3. run metadata: `outputs/sft_single_phase_h200/run_config.json`

Validate data wiring without loading the model:

```bash
python3 train_sft_single_phase.py --validate-only
```

To intentionally train on every row, bypassing holdouts:

```bash
python3 train_sft_single_phase.py --train-all
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
  --eval-splits grpo_holdout \
  --backend vllm \
  --max-model-len 8192 \
  --max-new-tokens 7680
```

If your local vLLM build hits a Nemotron LoRA `mixer.conv1d` assertion, force
the classic vLLM engine:

```bash
VLLM_USE_V1=0 python3 infer_eval.py \
  --train-csv data/single_phase_training_clean/single_phase_sft.csv \
  --adapter-dir outputs/sft_single_phase_h200/adapter \
  --split-csv data/single_phase_training_clean/single_phase_splits_80_10_10.csv \
  --eval-splits grpo_holdout \
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
  --eval-splits eval_holdout \
  --backend vllm \
  --max-model-len 8192 \
  --max-new-tokens 7680
```

For a smoke test, add `--max-eval-samples 20`. If vLLM is unavailable, pass
`--backend transformers`.

The competition metric expects the final answer in `\boxed{...}`.

## Type Diagnostics

Use these experiments to isolate whether each question type's traces are
learnable without mixed-task interference.

Prepare all seven diagnostic datasets and sync the root single-phase split:

```bash
python3 experiments/type_diagnostics/prepare_type_datasets.py
```

Train one question type:

```bash
python3 experiments/type_diagnostics/scripts/train_numeric_equation.py \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

Evaluate one question type on its held-out `eval_holdout` split:

```bash
python3 experiments/type_diagnostics/scripts/infer_numeric_equation.py \
  --adapter-dir experiments/type_diagnostics/outputs/numeric_equation/adapter \
  --backend vllm \
  --max-model-len 8192 \
  --max-new-tokens 7680
```

Evaluate both held-out diagnostic splits:

```bash
python3 experiments/type_diagnostics/scripts/infer_numeric_equation.py \
  --adapter-dir experiments/type_diagnostics/outputs/numeric_equation/adapter \
  --eval-splits eval_holdout grpo_holdout \
  --backend vllm \
  --max-model-len 8192 \
  --max-new-tokens 7680
```

Type-specific train/infer wrappers exist for:

1. `bit_manipulation`
2. `gravity`
3. `unit_conversion`
4. `text_cipher`
5. `numeral_system`
6. `numeric_equation`
7. `symbol_transform`

Each diagnostic report writes subtype accuracy to
`experiments/type_diagnostics/reports/{type}/metrics.json` and saves up to
three failed model generations for every non-100% subtype under
`experiments/type_diagnostics/reports/{type}/failed_traces/`.

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
