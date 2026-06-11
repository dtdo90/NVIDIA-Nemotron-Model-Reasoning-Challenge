#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import DATA_DIR_V1, SOURCE_CSV_V1
from lib.train import main


if __name__ == "__main__":
    main(
        "text_cipher",
        default_data_dir=DATA_DIR_V1,
        default_source_csv=SOURCE_CSV_V1,
        default_output_suffix="with_curriculum",
        default_decision_weight=1.0,
    )
