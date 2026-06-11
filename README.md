# NVIDIA Nemotron Model Reasoning Challenge

Clean LoRA fine-tuning project for the NVIDIA Nemotron reasoning challenge.

The main training path is now single-phase SFT with optional GRPO. The old
two-phase SFT curriculum is preserved under `legacy/two_phase/` only for
reproducibility.

## Data

Core files:

1. `data/train.csv`: original competition train set
2. `data/test.csv`: original competition test set
3. `data/single_phase_training_clean/single_phase_sft_v3.csv`: active SFT corpus
4. `data/single_phase_training_clean/single_phase_splits_80_10_10.csv`: canonical SFT/GRPO/eval split
5. `data/single_phase_training_clean/manifest.json`: source counts and split metadata
6. `experiments/type_diagnostics/data/global_splits_80_10_10.csv`: same split assignment, generated from the per-type diagnostics

`single_phase_sft_v1.csv` and `single_phase_sft_v2.csv` are retained as legacy
snapshots. The active training and type-diagnostic defaults use v3.

Current validated single-phase counts:

1. Full SFT corpus: `18565` rows
2. SFT training bucket, named `sft_train`: `16929` rows
3. Optional GRPO train bucket, named `eval_holdout`: `818` rows
4. Final local eval bucket, named `grpo_holdout`: `818` rows

The single-phase corpus contains real traces plus selected synthetic curriculum
rows. Synthetic curriculum rows are also train-only. The two holdout buckets are drawn
only from eval-eligible real/current-evaluation rows.

The split ratios are approximate. The important invariant is that the full
single-phase run and the per-question-type diagnostic runs use the same row-level
split assignment. Regenerate and sync the split with:

```bash
python3 experiments/type_diagnostics/prepare_type_datasets.py
```

This writes the seven per-type split files and copies their union to
`data/single_phase_training_clean/single_phase_splits_80_10_10.csv`.
Type-diagnostic train/infer scripts check freshness against the root SFT CSV
and stop if the cached per-type files are stale.

## Install

Use an RTX6000/H100/H200 machine when possible. RTX6000/H100 would need gradient checkpointing. 

```bash
pip install -r requirements.txt
pip uninstall -y torchvision
pip install -U pip setuptools wheel packaging ninja einops
pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm
```

Install the Nemotron CUDA extensions after PyTorch is installed. If imports fail
with an undefined-symbol error, rebuild them against the active PyTorch install:

```bash
pip uninstall -y causal-conv1d mamba-ssm
MAX_JOBS=8 pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall --no-binary=causal-conv1d causal-conv1d
MAX_JOBS=8 pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall --no-binary=mamba-ssm mamba-ssm
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
If your hardware supports, train with batch size 2, or skip gradient-checkpointing for faster speed.
```bash
python3 train_sft_single_phase.py \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --balanced-accumulation
```
This trains fresh LoRA weights for one epoch at learning rate `2e-4`. By
default it uses only the `sft_train` rows from
`data/single_phase_training_clean/single_phase_splits_80_10_10.csv`, which is
the same split assignment used by the type-diagnostic experiments.
`--balanced-accumulation` keeps one puzzle per sequence but spreads question
types across each gradient-accumulation update.

1. final adapter: `outputs/sft_single_phase/adapter`
2. submission zip: `outputs/sft_single_phase/submission.zip`
3. run metadata: `outputs/sft_single_phase/run_config.json`

Validate data wiring without loading the model:

```bash
python3 train_sft_single_phase.py --validate-only
```

Validate exact tokenizer masking and token caps without loading the model:

```bash
python3 train_sft_single_phase.py --validate-tokenization
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
6. optimizer `adamw_torch`
7. LoRA dropout `0.0`
8. assistant-only loss masking
9. competition chat-template prompt format

```

## Text Cipher decision-weighted loss

Token-level loss weighting for Text Cipher traces lives in
`src/nemotron_baseline/text_cipher_loss_weights.py`. It assigns weight `2.0` to
decision points and empirically failure-prone spans (per-character map binds,
the assembled decode pattern, `fully`/`not fully mapped`, scan `match` verdicts,
the candidate set, `conflicts`/`new` mapping checks, `PASS`/`FAIL`, `add`,
`choose`, and the boxed answer) and leaves everything else at `1.0` (boilerplate,
echoes, the 77-word vocab dump, `agrees` lines, and the routine `no` scan lines).

Symbol Transform traces are weighted the same way by
`src/nemotron_baseline/symbol_transform_loss_weights.py`: weight `2.0` lands on
the routing/method decisions (operator-compare `same`/`different`, RHS-length
family routing, template/format selection, `Match`/`No match`/`pass`/`fail`
verdicts, the produced value after `gives`, the reversed/coefficient digit
derivations, `mod 10` constraints, the `Ck`/`Tk` scan survivors, and the
operator-absence default `None` / `use direct template matching`), and `1.0` on
echoes, the `AB = ...`/`operator = ...` breakdowns, variable naming, and the `x`
reject entries. `tokenize_masked_example` dispatches by `category` (Text Cipher,
Symbol Transform, or Numeric Equation Transformation Rules); other categories
get flat weights. Note: many Symbol
Transform rows are operator-absence cases whose final answer intentionally does
not match gold, so the weighting reinforces the cautious routing/method rather
than gold-hitting.

Numeric Equation traces are weighted by
`src/nemotron_baseline/numeric_equation_loss_weights.py`: weight `2.0` lands on
RHS-length routing, candidate-family lists, rendered output-format table rows,
surviving common-format names, motif-confirm/reject policy, direct-template
produced values, operator-absence candidate/vote decisions, final query output
rows, and boxed answers. Repeated scaffold such as `Try BA_DC first`, `The
current format is ...`, `Match`, and `Common` stays at `1.0`; when a policy line
contains a concrete format such as `BA_DC|x-y|common`, only that format span is
promoted.

Wire it into `tokenize_masked_example` with
`completion_label_weights(tokenizer, prompt_text, completion_text)`, store the
result next to `labels` as `label_weights`, pad it in the collator with `0.0`,
and apply a weighted cross-entropy.

Inspect/verify the weighting on a trace before training (prints each weight-2
token so you can confirm the high weight lands on decisions, not boilerplate):

```bash
PYTHONPATH=src python3 -m nemotron_baseline.text_cipher_loss_weights path/to/completion.txt
```

Run with no file argument to check a built-in sample, and pass `--tokenizer
PATH` to point at a different `tokenizer.json`.

## Optional GRPO

GRPO remains an optional second step for both regimes. In the active single-phase
regime, it starts from `outputs/sft_single_phase/adapter` and trains on the
10% bucket named `eval_holdout`.

Smoke-check the wiring:

```bash
python3 train_grpo.py --config configs/grpo_stage2.json --validate-only
```

Run GRPO:

```bash
python3 train_grpo.py --config configs/grpo_stage2.json
```

## Local Evaluation

Evaluate the eval_holdout bucket (10% data):

```bash
python3 infer_eval.py \
  --train-csv data/single_phase_training_clean/single_phase_sft_v3.csv \
  --adapter-dir outputs/sft_single_phase/adapter \
  --split-csv data/single_phase_training_clean/single_phase_splits_80_10_10.csv \
  --eval-splits eval_holdout
```

Evaluate both held-out buckets (20% data):

```bash
python3 infer_eval.py \
  --train-csv data/single_phase_training_clean/single_phase_sft_v3.csv \
  --adapter-dir outputs/sft_single_phase/adapter \
  --split-csv data/single_phase_training_clean/single_phase_splits_80_10_10.csv \
  --eval-splits grpo_holdout eval_holdout
```

For a smoke test, add `--max-eval-samples 20`. If vLLM is unavailable, pass
`--backend transformers`.

`infer_eval.py` reports accuracy by question type and diagnostic subtype under
`by_type`. For every subtype that is not 100% correct, it writes and prints up
to three failed generations by default. Add `--no-print-failed-traces` to keep
stdout compact. Failed samples are saved under
`{adapter_parent}/{split}_failed_traces/{type}/{subtype}/`, or under
`--report-dir` if provided.

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
  --gradient-accumulation-steps 8 \
  --balanced-accumulation
```

The plain type-diagnostic wrappers use the active v3 single-phase corpus. The
old curriculum ablation wrappers are retained only for legacy/manual comparison
and are no longer the default training path.

Text Cipher, Symbol Transform, and Numeric Equation diagnostics enable the
decision-weighted loss by default (`--decision-weight 2.0`). The full
`train_sft_single_phase.py` path still defaults to `1.0` (off), so opt in when
training the full corpus. At `1.0` the loss is byte-identical to the standard
mean loss. Disable or tune it per run:

```bash
python3 experiments/type_diagnostics/scripts/train_text_cipher.py --decision-weight 1.0
python3 train_sft_single_phase.py --decision-weight 2.0   # opt in for the full v3 corpus
```

Before training a type diagnostic, dry-run the exact tokenizer boundary and
token cap check:

```bash
python3 experiments/type_diagnostics/scripts/train_numeric_equation.py --validate-tokenization
python3 experiments/type_diagnostics/scripts/train_text_cipher.py --validate-tokenization
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
