#!/usr/bin/env python
"""Build the overall unfiltered v5.2.1 shared-frame separation histogram."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dsb_states.v521_shared_frame_separation import (
    build_shared_frame_separation,
    write_shared_frame_separation_result,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-nm", type=float, default=550.0)
    parser.add_argument("--bin-width-nm", type=float, default=50.0)
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
        bin_width_nm=args.bin_width_nm,
    )
    output_paths = write_shared_frame_separation_result(
        result,
        args.output_dir,
        source_paths=(
            frame_path,
            cache_contract,
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1]
            / "src"
            / "dsb_states"
            / "v521_shared_frame_separation.py",
            Path(__file__).resolve().parents[1] / "src" / "dsb_states" / "plots.py",
        ),
    )
    record = result.summary.iloc[0]
    print(
        json.dumps(
            {
                "status": "complete",
                "shared_frames": int(record["shared_frame_count"]),
                "threshold_nm": float(record["threshold_nm"]),
                "n_above_threshold": int(record["n_above_threshold"]),
                "percent_above_threshold": float(record["percent_above_threshold"]),
                "outputs": {key: str(value) for key, value in output_paths.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
