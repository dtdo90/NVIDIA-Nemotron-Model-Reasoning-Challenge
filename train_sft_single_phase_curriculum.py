#!/usr/bin/env python3
"""Single-phase SFT entry point for the v1 Text Cipher curriculum mix."""
from __future__ import annotations

import sys

import train_sft_single_phase as trainer


trainer.SINGLE_PHASE_CSV = (
    trainer.ROOT / "data/single_phase_training_clean/single_phase_sft_v1.csv"
)
trainer.SINGLE_PHASE_SPLIT_CSV = (
    trainer.ROOT / "data/single_phase_training_clean/single_phase_splits_80_10_10_v1.csv"
)

_parse_args = trainer.parse_args


def parse_args():
    args = _parse_args()
    if "--output-dir" not in sys.argv:
        args.output_dir = "outputs/sft_single_phase_curriculum"
    return args


trainer.parse_args = parse_args


if __name__ == "__main__":
    trainer.main()
