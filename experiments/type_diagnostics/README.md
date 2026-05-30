# Type Diagnostics

This workspace isolates dataset quality by training one LoRA adapter per question
type, then evaluating each adapter on held-out rows from the same type.

## Prepare Data

Build all seven per-type datasets and 80/10/10 diagnostic splits:

```bash
python3 experiments/type_diagnostics/prepare_type_datasets.py
```

Each type is written under `experiments/type_diagnostics/data/{type}/`:

- `{type}.csv`: rows copied from `data/single_phase_training_clean/single_phase_sft.csv`
- `splits_80_10_10.csv`: `sft_train`, `eval_holdout`, and `grpo_holdout`
- `dataset_summary.json`: subtype/source/split counts

The split is stratified by `diagnostic_subtype` and `source_mode` whenever the
bucket is large enough. Very small buckets keep as much train coverage as
possible while still adding holdout rows when feasible.

## Train One Type

Generic form:

```bash
python3 experiments/type_diagnostics/train_type.py --question-type numeric_equation
```

Type-specific wrappers:

```bash
python3 experiments/type_diagnostics/scripts/train_bit_manipulation.py
python3 experiments/type_diagnostics/scripts/train_gravity.py
python3 experiments/type_diagnostics/scripts/train_unit_conversion.py
python3 experiments/type_diagnostics/scripts/train_text_cipher.py
python3 experiments/type_diagnostics/scripts/train_numeral_system.py
python3 experiments/type_diagnostics/scripts/train_numeric_equation.py
python3 experiments/type_diagnostics/scripts/train_symbol_transform.py
```

Default adapters are saved to:

```text
experiments/type_diagnostics/outputs/{type}/adapter
```

Use `--validate-only` before training to inspect rows and subtype splits without
loading the model.

## Evaluate One Type

Generic form:

```bash
python3 experiments/type_diagnostics/infer_type.py \
  --question-type numeric_equation \
  --adapter-dir experiments/type_diagnostics/outputs/numeric_equation/adapter
```

Type-specific wrappers:

```bash
python3 experiments/type_diagnostics/scripts/infer_numeric_equation.py \
  --adapter-dir experiments/type_diagnostics/outputs/numeric_equation/adapter
```

By default inference evaluates `eval_holdout`. To evaluate both held-out splits:

```bash
python3 experiments/type_diagnostics/infer_type.py \
  --question-type numeric_equation \
  --adapter-dir experiments/type_diagnostics/outputs/numeric_equation/adapter \
  --eval-splits eval_holdout grpo_holdout
```

Reports are saved to:

```text
experiments/type_diagnostics/reports/{type}/metrics.json
experiments/type_diagnostics/reports/{type}/predictions.jsonl
experiments/type_diagnostics/reports/{type}/failed_traces/{subtype}/
```

For each subtype that is not 100% correct, the evaluator records up to three
failed model generations for review.
