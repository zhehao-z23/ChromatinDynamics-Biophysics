#!/usr/bin/env python
"""Select complete paired-trajectory cases from the unfiltered v5.2.1 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dsb_states.v521_case_selection import (
    select_complete_pair_cases,
    write_complete_pair_case_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-extreme", type=int, default=5)
    parser.add_argument("--display-min-frames", type=int, default=20)
    args = parser.parse_args()

    cache = args.cache.resolve()
    bundle_index_path = cache / "tables" / "bundle_index.parquet"
    frame_path = cache / "tables" / "bundle_frame_master.parquet"
    cache_contract_path = cache / "CACHE_CONTRACT.json"
    for source in (bundle_index_path, frame_path, cache_contract_path):
        if not source.is_file():
            raise FileNotFoundError(source)

    index_columns = [
        "bundle_id",
        "nd2_id",
        "crop_id",
        "fov_id",
        "hour_post_delivery",
        "allele_index",
        "site1_points",
        "site2_points",
        "shared_site_frames",
        "movie_frames",
        "frame_interval_s_production",
    ]
    frame_columns = [
        "bundle_id",
        "frame",
        "time_s",
        "site1_valid",
        "site2_valid",
        "shared_site_frame",
        "site1_x_nm",
        "site1_y_nm",
        "site2_x_nm",
        "site2_y_nm",
        "site1_site2_separation_nm",
    ]
    result = select_complete_pair_cases(
        pd.read_parquet(bundle_index_path, columns=index_columns),
        pd.read_parquet(frame_path, columns=frame_columns),
        n_extreme=args.n_extreme,
        display_min_frames=args.display_min_frames,
    )
    output_paths = write_complete_pair_case_selection(
        result,
        args.output_dir,
        source_paths=(
            bundle_index_path,
            frame_path,
            cache_contract_path,
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1]
            / "src"
            / "dsb_states"
            / "v521_case_selection.py",
        ),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "complete_pair_count": len(result.trajectory_summary),
                "complete_pair_frame_count": int(result.trajectory_summary["movie_frames"].sum()),
                "outputs": {key: str(path) for key, path in output_paths.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
