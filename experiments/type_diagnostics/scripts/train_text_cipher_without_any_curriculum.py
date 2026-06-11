#!/usr/bin/env python3
"""Train Text Cipher type diagnostic with v2 data but no curriculum rows."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import DATA_DIR_V2, SOURCE_CSV_V2
from lib.train import main


if __name__ == "__main__":
    main(
        "text_cipher",
        default_data_dir=DATA_DIR_V2,
        default_source_csv=SOURCE_CSV_V2,
        default_exclude_source_modes=[
            "text_cipher_decision_point_curriculum",
        ],
        default_output_suffix="without_any_curriculum",
        default_decision_weight=1.0,
    )
