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
6. `data/single_phase_training_clean/single_phase_sft.csv`: single-phase SFT mix with Phase 2 plus selected synthetic direct-template rows

Default SFT uses all Phase 1 rows plus Phase 2 `sft_train`.

Current validated counts:

1. Phase 1: `5077` rows
2. Phase 2 `sft_train`: `7348` rows
3. Combined default SFT: `12425` rows
4. Phase 2 holdout: `1836` rows
5. GRPO train bucket: `919` rows
6. Final local eval bucket: `917` rows
7. Single-phase SFT: `9425` rows

## Install

Use an H100 or L40S machine with a recent CUDA PyTorch environment.

```bash
pip install -r requirements.txt
pip uninstall -y torchvision
pip install --no-build-isolation --no-deps -r requirements-nemotron.txt
```

## Train SFT

By default, the portable scripts use Kaggle's mounted model at
`/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`
when that path exists, otherwise they use the Hugging Face model
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. Override with `--model-path`,
`MODEL_PATH`, or `BASE_MODEL_PATH`.

Default two-stage SFT:

```bash
python3 train_sft.py
```

This first trains Phase 1 for one epoch at learning rate `1e-4`, saves
`outputs/sft_two_stage_h200/phase1/adapter`, then continues on Phase 2
`sft_train` for one epoch at learning rate `5e-5` and saves the final adapter to
`outputs/sft_two_stage_h200/adapter`.

Recommended explicit two-step SFT:

```bash
python3 train_sft.py --phase1-only \
  --phase1-learning-rate 1e-4 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 2
```

Then continue Phase 2 from the saved Phase 1 adapter:

```bash
python3 train_sft.py --phase2 \
  --phase1-adapter-dir outputs/sft_phase1_h200/adapter \
  --phase2-learning-rate 5e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

Single-phase Phase 2 training from fresh LoRA weights at learning rate `2e-4`:

```bash
python3 train_sft_single_phase.py \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

This skips Phase 1 entirely and writes the final adapter to
`outputs/sft_single_phase_h200/adapter`.

The first command assumes an H200-class GPU. On a 96GB RTX6000, use the same
safer micro-batch setting for Phase 1:

```bash
python3 train_sft.py --phase1-only \
  --phase1-learning-rate 1e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing
```

Validate data wiring without loading the model:

```bash
python3 train_sft.py --validate-only
python3 train_sft.py --phase1-only --validate-only
python3 train_sft.py --phase2 --validate-only
```

The minimal trainer uses:

1. LoRA rank `32`
2. sequence length `8192`
3. bf16 + TF32
4. default H200 batch size `8`, gradient accumulation `2`
5. cosine LR schedule with warmup ratio `0.05`
6. minimum learning rate floor `2e-6`
7. gradient checkpointing disabled
8. dataloader workers `4`

Outputs are written to `outputs/sft_two_stage_h200/` by default.

## Optional GRPO

After SFT, GRPO can train on the 919-row `eval_holdout` bucket while reserving
`grpo_holdout` for final local evaluation.

```bash
python3 train_grpo.py --config configs/grpo_stage2.json --validate-only
python3 train_grpo.py --config configs/grpo_stage2.json
```

## Local Evaluation

The default evaluator uses the vLLM backend for faster batched generation with
the LoRA adapter. Install vLLM in the same environment after the base training
dependencies are installed:

```bash
pip install vllm
```

If vLLM is unavailable on the machine, pass `--backend transformers` to use the
slower Transformers fallback.

```bash
python3 infer_eval.py \
  --train-csv data/training_ready_clean/phase2_sft.csv \
  --adapter-dir outputs/sft_two_stage_h200/adapter \
  --split-csv data/training_ready_clean/phase2_splits_80_10_10.csv \
  --eval-splits grpo_holdout \
  --backend vllm \
  --max-model-len 8192 \
  --max-new-tokens 7680
```

For a quick smoke test, add `--max-eval-samples 20`. The script writes summary
and prediction files next to the adapter directory.

The competition metric expects the final answer in `\boxed{...}`.

## Methodology

Core method notes live in:

1. `docs/solver_method_record.md`
2. `docs/digit_transform_methodology.md`
3. `docs/numeric_equation_methodology.md`
4. `docs/symbol_transform_methodology.md`
5. `docs/winner_solution_alignment.md`
