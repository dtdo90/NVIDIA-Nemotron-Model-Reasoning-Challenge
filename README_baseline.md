# First Baseline

This repository now includes a script-based version of the first baseline:

- `train_sft.py` trains either answer-only or CoT-supervised LoRA adapters.
- `infer_eval.py` runs local held-out evaluation with the same stratified split.
- `--supervision-format auto` uses CoT when the CSV has a `generated_cot` column, otherwise answer-only.

## Training

```bash
python3 train_sft.py \
  --config configs/baseline_answer_only.json \
  --output-dir outputs/baseline_answer_only \
  --save-tokenizer
```

If you already have the Nemotron base model on disk, pass `--model-path /path/to/model`.

## Persistent Splits

To create a fixed `70/15/15` split for SFT, GRPO, and final evaluation:

```bash
python3 scripts/make_splits.py \
  --input-csv data/train.csv \
  --output-csv data/splits_70_15_15.csv
```

This writes `id,category,split` rows with the split names:

- `sft_train`
- `grpo_train`
- `eval`

It also writes `data/splits_70_15_15.config.json`, which stores the exact ids for each split and can be used anywhere the scripts accept `--split-csv`.

## CoT Training

```bash
python3 train_sft.py \
  --config configs/cot_training.json \
  --output-dir outputs/cot_training
```

This uses `data/train_cot.csv` and trains on `<generated_cot> + \boxed{answer}` when the CSV contains a `generated_cot` column.

To train on only the fixed `sft_train` split, use:

```bash
python3 train_sft.py \
  --config configs/cot_training_70_15_15.json \
  --train-csv data/train_cot_gpt_oss_clean.csv \
  --output-dir outputs/cot_training_70_15_15
```

## Two-Stage Text Knowledge + Task SFT

To inject the Wonderland text-encryption vocabulary first, convert the doc2lora knowledge QA file into the repo's training schema:

```bash
python3 scripts/prepare_text_knowledge_phase1.py \
  --input-csv data/knowledge_qa.csv \
  --output-csv data/text_knowledge_phase1.csv
```

Then train the phase-1 knowledge adapter:

```bash
python3 train_sft.py \
  --config configs/text_knowledge_phase1.json
```

For the main task SFT stage, build a split-aware phase-2 dataset that:

- anchors on `data/train.csv`
- overlays your cleaned competition CoT file
- replaces `Text Cipher` rows with doc2lora's shorter `encryption_new_cot.csv` traces for the selected split ids

```bash
python3 scripts/prepare_phase2_sft_dataset.py \
  --train-csv data/train.csv \
  --base-cot-csv data/train_cot_gpt_oss_clean.csv \
  --text-cot-csv data/encryption_new_cot.csv \
  --split-csv data/splits_70_15_15.config.json \
  --train-splits sft_train \
  --output-csv data/train_sft_phase2_70_15_15.csv
```

Then continue SFT from the phase-1 adapter:

```bash
python3 train_sft.py \
  --config configs/cot_training_phase2_70_15_15.json
```

`train_sft.py` now supports `--init-adapter-dir`, so phase 2 can keep training the LoRA weights learned in phase 1 instead of starting from a fresh adapter.

## CoT Generation

```bash
export GEMINI_API_KEY=your_api_key_here

python3 scripts/generate_gemini_cot.py \
  --input-csv data/train.csv \
  --output-csv data/train_cot_gemini.csv \
  --model gemini-3-flash
```

The generator writes `id,prompt,answer,generated_cot,label` rows, supports resume-by-id, and only keeps generations whose extracted final answer matches the gold answer unless `--allow-unverified` is set.

## Local GPT-OSS CoT Generation

```bash
python3 scripts/generate_gpt_oss_cot.py \
  --input-csv data/train.csv \
  --output-csv data/train_cot_gpt_oss.csv \
  --model-path /kaggle/input/gpt-oss-120b/transformers/default/1
```

This follows the same `train_cot.csv` schema, but serves a local `gpt-oss-120b` model through vLLM and renders prompts with `openai_harmony`, based on the approach in [aimo-3-gpt-oss-120b-with-tools-and-revision.ipynb](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/kaggle_notebooks/aimo-3-gpt-oss-120b-with-tools-and-revision.ipynb). By default it starts a local vLLM OpenAI-compatible server, verifies the boxed final answer against the gold label, and resumes by `id`.

The generator now shows a progress bar by default. If you want to see tokens as they are produced, add `--stream-output`. With multiple attempts, only attempt 1 is shown live. If you prefer plain logging only, add `--no-progress`.

By default the generator now uses `4` attempts per row with early stop at `2` agreeing answers.

For notebook-style weighted selection with multiple attempts per row, use:

```bash
python3 scripts/generate_gpt_oss_cot.py \
  --input-csv data/train.csv \
  --output-csv data/train_cot_gpt_oss.csv \
  --model-path /kaggle/input/gpt-oss-120b/transformers/default/1 \
  --attempts 4 \
  --attempt-workers 4 \
  --early-stop-votes 2
```

This runs 4 generations for each row, aggregates extracted answers with notebook-style entropy-based scoring, and keeps the highest-scoring trace from the winning answer.

## Cleaning And Approval

```bash
python3 clean_reasoning_trace.py \
  --input-csv data/train_cot_gpt_oss.csv \
  --output-csv data/train_cot_gpt_oss_clean.csv
```

This runs a second-pass cleaner plus a strict approver on each raw trace. The output CSV keeps:

- `generated_cot_raw`: the original raw trace
- `generated_cot_clean`: the cleaned trace before approval filtering
- `generated_cot`: the approved training trace, or an empty string if the row failed approval

Rows with empty `generated_cot` naturally fall back to answer-only supervision in `train_sft.py`.

To reuse an already-running vLLM server:

```bash
python3 clean_reasoning_trace.py \
  --no-start-server \
  --port 8000 \
  --input-csv data/train_cot_gpt_oss.csv \
  --output-csv data/train_cot_gpt_oss_clean.csv
```

For a fast prompt sanity-check, use:

```bash
python3 scripts/generate_gpt_oss_cot.py \
  --input-csv data/train.csv \
  --output-csv data/train_cot_gpt_oss_debug.csv \
  --model-path /kaggle/input/gpt-oss-120b/transformers/default/1 \
  --debug \
  --stream-output
```

`--debug` selects at most one unseen sample per category type.

To debug a single category, add `--debug-type` with one of:
`bit`, `gravity`, `conversion`, `cipher`, `transformation`, `numeral`

## Deterministic Bit Renderer

For Bit Manipulation rows that `gpt-oss` failed but the notebook solver can solve with high confidence, you can render cleaner deterministic CoT traces with:

```bash
python3 scripts/render_bit_solver_cot.py \
  --input-csv data/train_cot_gpt_oss_failed.csv \
  --output-csv data/train_cot_gpt_oss_failed_bit_solver_high_conf.csv
```

This reruns the bit solver from [bit-manipulation-solver-cot-generator.ipynb](/Users/taido/Desktop/Tai/NVIDIA%20Nemotron%20Model%20Reasoning/kaggle_notebooks/bit-manipulation-solver-cot-generator.ipynb), keeps only solver-correct high-confidence rows (`w_*` and `ctx` methods), and writes clean traces in the format:

```text
<think>
...
</think>
\boxed{answer}
```

Rows that are solver-incorrect or solver-low-confidence are written to the skipped CSV instead.

## Evaluation

```bash
python3 infer_eval.py \
  --run-config outputs/baseline_answer_only/run_config.json \
  --adapter-dir outputs/baseline_answer_only/adapter \
  --max-eval-samples 256 \
  --batch-size 1
```

This recreates the same validation split from `data/train.csv` using:

- `seed=42`
- `val_fraction=0.2`

To evaluate on the fixed held-out `eval` split instead:

```bash
python3 infer_eval.py \
  --run-config outputs/cot_training_70_15_15/run_config.json \
  --adapter-dir outputs/cot_training_70_15_15/adapter \
  --split-csv data/splits_70_15_15.csv \
  --eval-splits eval
```

## GRPO Stage

After SFT, you can run a second-stage GRPO update starting from the SFT adapter:

```bash
python3 train_grpo.py \
  --config configs/grpo_stage2.json \
  --sft-adapter-dir outputs/cot_training_70_15_15/adapter \
  --output-dir outputs/grpo_stage2
```

This:

- trains on the `grpo_train` split from `data/splits_70_15_15.csv`
- starts from the SFT LoRA adapter
- uses Nemotron's default chat-template thinking mode
- rewards exact final-answer correctness with small bonuses for a single boxed final line

Then compare the SFT-only adapter and the GRPO adapter on the same held-out `eval` split with `infer_eval.py`.

Artifacts are written under the chosen output directory:

- `adapter/`
- `submission.zip`
- `run_config.json`
- `dataset_summary.json`
- `val_summary.json`
- `val_predictions.jsonl`
