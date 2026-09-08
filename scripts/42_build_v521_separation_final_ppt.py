#!/usr/bin/env python
"""Build the final slide-ready Site1/Site2 separation figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dsb_states.v521_separation_final_ppt import write_final_separation_ppt_result
from dsb_states.v521_shared_frame_separation import build_shared_frame_separation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-nm", type=float, default=500.0)
    parser.add_argument(
        "--multiple-testing",
        choices=("none", "holm", "bonferroni"),
        default="holm",
    )
    args = parser.parse_args()

    cache = args.cache.resolve()
    frame_path = cache / "tables" / "bundle_frame_master.parquet"
    cache_contract = cache / "CACHE_CONTRACT.json"
    for source in (frame_path, cache_contract):
        if not source.is_file():
            raise FileNotFoundError(source)

    columns = [
        "bundle_id",
        "nd2_id",
        "crop_id",
        "hour_post_delivery",
        "frame",
        "site1_valid",
        "site2_valid",
        "site1_x_um",
        "site1_y_um",
        "site2_x_um",
        "site2_y_um",
        "site1_site2_separation_nm",
    ]
    result = build_shared_frame_separation(
        pd.read_parquet(frame_path, columns=columns),
        threshold_nm=args.threshold_nm,
        bin_width_nm=50.0,
    )
    project = Path(__file__).resolve().parents[1]
    paths = write_final_separation_ppt_result(
        result,
        args.output_dir,
        source_paths=(
            frame_path,
            cache_contract,
            Path(__file__).resolve(),
            project / "src" / "dsb_states" / "v521_separation_final_ppt.py",
            project / "src" / "dsb_states" / "v521_shared_frame_separation.py",
            project / "src" / "dsb_states" / "plots.py",
        ),
        multiple_testing=args.multiple_testing,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "shared_frames": len(result.frame_distances),
                "outputs": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
