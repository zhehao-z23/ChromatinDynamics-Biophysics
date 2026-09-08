#!/usr/bin/env python
"""Build the flat no-analysis-QC v5.2.1 bundle-frame cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsb_states.unfiltered_cache import build_unfiltered_cache, write_unfiltered_cache
from dsb_states.v521_intake import load_intake_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    result = build_unfiltered_cache(
        load_intake_table(snapshot, "trajectory_points"),
        load_intake_table(snapshot, "bp1_allele_frames"),
        load_intake_table(snapshot, "allele_index"),
        load_intake_table(snapshot, "crop_index"),
    )
    outputs = write_unfiltered_cache(result, args.output_dir, source_snapshot=snapshot)
    print(
        json.dumps(
            {"status": "complete", "census": result.census, "outputs": {k: str(v) for k, v in outputs.items()}},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
