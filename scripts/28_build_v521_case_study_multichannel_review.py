#!/usr/bin/env python
"""Build the two final shared-scale paired-distance poster demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsb_states.v521_case_study_multichannel_review import (
    configure_case_style,
    load_multichannel_review_cases,
    write_distance_demo_variants,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fullrun-root", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-exe", type=Path, required=True)
    parser.add_argument("--reference-script-dir", type=Path, required=True)
    parser.add_argument("--original-matlab", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    selection_csv = args.selection_csv.resolve()
    reference_dir = args.reference_script_dir.resolve()
    cases, config = load_multichannel_review_cases(
        config_path=config_path,
        fullrun_root=args.fullrun_root.resolve(),
        selection_csv=selection_csv,
    )
    configure_case_style()
    result = write_distance_demo_variants(
        cases=cases,
        config=config,
        config_path=config_path,
        output_dir=args.output_dir,
        ffmpeg_exe=args.ffmpeg_exe.resolve(),
        extra_source_paths=(
            selection_csv,
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1]
            / "src"
            / "dsb_states"
            / "v521_case_study_multichannel_review.py",
            reference_dir / "README.md",
            reference_dir / "animation_core.py",
            reference_dir / "build_demo_videos.py",
            args.original_matlab.resolve(),
        ),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "case_count": len(cases),
                "products": result["products"],
                "path_table": str(result["path_table"]),
                "manifest": str(result["manifest"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
