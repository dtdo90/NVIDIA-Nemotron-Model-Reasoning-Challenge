#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def decode_payload(text: str) -> str:
    return text.replace("__BS__", "\\").replace("__LB__", "{").replace("__RB__", "}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate boxed payload decoding for symbol_transform phase2 export.")
    p.add_argument(
        "--input-csv",
        default="data/trainable/symbol_transform_phase2_combined.csv",
    )
    p.add_argument(
        "--mismatch-csv",
        default="data/symbol_transform_phase2_boxed_decode_mismatch.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_csv)
    mismatch_path = Path(args.mismatch_csv)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    mismatches: list[dict[str, str]] = []
    for row in rows:
        phase2 = row.get("generated_cot_phase2", "")
        matches = BOX_RE.findall(phase2)
        # The trace intentionally contains an earlier ``\boxed{}`` reminder.
        # Evaluation uses the final box, so validate the last boxed payload.
        extracted = matches[-1] if matches else ""
        decoded = decode_payload(extracted)
        answer = row.get("final_answer_plain") or row.get("answer", "")
        if decoded != answer:
            mismatches.append(
                {
                    "id": row.get("id", ""),
                    "data_type": row.get("data_type", ""),
                    "answer": answer,
                    "boxed_extracted": extracted,
                    "boxed_decoded": decoded,
                }
            )

    mismatch_path.parent.mkdir(parents=True, exist_ok=True)
    with mismatch_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "data_type", "answer", "boxed_extracted", "boxed_decoded"],
        )
        w.writeheader()
        w.writerows(mismatches)

    print(f"rows: {len(rows)}")
    print(f"mismatches: {len(mismatches)}")
    print(f"wrote: {mismatch_path}")


if __name__ == "__main__":
    main()
