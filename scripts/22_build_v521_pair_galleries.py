#!/usr/bin/env python
"""Apply the frozen prior T3/T4 rules and build v5.2.1 review galleries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dsb_states.v521_pair_galleries import build_pair_gallery_result, write_pair_gallery_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache.resolve()
    frames_path = cache / "tables" / "bundle_frame_master.parquet"
    bundle_path = cache / "tables" / "bundle_index.parquet"
    result = build_pair_gallery_result(
        pd.read_parquet(frames_path),
        pd.read_parquet(bundle_path),
    )
    outputs = write_pair_gallery_result(
        result,
        args.output_dir,
        source_files=(frames_path, bundle_path, cache / "CACHE_CONTRACT.json"),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "t3_bundles": int(result.membership.t3_included.sum()),
                "t4_bundles": int(result.membership.t4_included.sum()),
                "shared_half_width_um": result.shared_half_width_um,
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
