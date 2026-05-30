# Legacy Two-Phase Training

This folder preserves the earlier two-phase SFT workflow for reproducibility.
It is no longer the main training path.

Contents:

1. `train_sft.py`: legacy Phase 1 synthetic curriculum followed by Phase 2 real traced SFT
2. `train_sft_kaggle.py`: older Kaggle-oriented SFT runner
3. `data/training_ready_clean/`: archived Phase 1 and Phase 2 training bundle
4. `configs/`: legacy configs with paths updated to this folder

The active path is now root-level `train_sft_single_phase.py`, followed
optionally by root-level `train_grpo.py`.

Legacy smoke checks:

```bash
python3 legacy/two_phase/train_sft.py --validate-only
python3 legacy/two_phase/train_sft_kaggle.py \
  --config legacy/two_phase/configs/phase2_training_ready_clean_kaggle.json \
  --validate-only
python3 train_grpo.py \
  --config legacy/two_phase/configs/grpo_stage2.json \
  --validate-only
```
