#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.train import main


if __name__ == "__main__":
    main(
        "numeric_equation",
        default_exclude_source_modes=["numeric_equation_decision_point_curriculum"],
        default_output_suffix="without_curriculum",
    )
