#!/usr/bin/env python
"""Select and render the top fitted Site1-alpha cases at 3 h."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from dsb_states.v521_case_study_multichannel_review import (
    configure_case_style,
    load_multichannel_review_cases,
    write_multichannel_case_review,
)
from dsb_states.v521_msd_final_visualization import select_top_alpha_cases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-table", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fullrun-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-exe", type=Path, required=True)
    parser.add_argument("--reference-script-dir", type=Path, required=True)
    parser.add_argument("--original-matlab", type=Path, required=True)
    args = parser.parse_args()

    alpha_path = args.alpha_table.resolve()
    selection_path = args.selection_csv.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection_contract = config["selection"]
    alpha_table = pd.read_csv(alpha_path)
    selected = select_top_alpha_cases(
        alpha_table,
        primary_hour=float(selection_contract["hour_post_delivery"]),
        site=str(selection_contract["site"]),
        top_n=int(selection_contract["top_n"]),
        minimum_r2=float(selection_contract["minimum_r2"]),
    )
    configured_ids = [str(item["bundle_id"]) for item in config["cases"]]
    selected_ids = selected["bundle_id"].astype(str).tolist()
    if configured_ids != selected_ids:
        raise ValueError("Frozen case configuration does not match the recomputed alpha ranking")

    counterpart = alpha_table.loc[
        alpha_table["site"].astype(str).eq("site2"),
        ["bundle_id", "alpha", "r2_log", "n_fit_lags", "fit_lag_min_s", "fit_lag_max_s"],
    ].rename(
        columns={
            "alpha": "paired_site2_alpha",
            "r2_log": "paired_site2_r2_log",
            "n_fit_lags": "paired_site2_n_fit_lags",
            "fit_lag_min_s": "paired_site2_fit_lag_min_s",
            "fit_lag_max_s": "paired_site2_fit_lag_max_s",
        }
    )
    ranked = selected.merge(counterpart, on="bundle_id", how="left", validate="one_to_one")
    ranked["selection_role"] = (
        "descending Site1 raw-MSD alpha at 3 h among the frozen complete-coverage cohort; "
        "R2 support only"
    )

    cases, review_config = load_multichannel_review_cases(
        config_path=config_path,
        fullrun_root=args.fullrun_root.resolve(),
        selection_csv=selection_path,
    )
    configure_case_style()
    reference_dir = args.reference_script_dir.resolve()
    script_path = Path(__file__).resolve()
    case_module = (
        script_path.parents[1] / "src" / "dsb_states" / "v521_case_study_multichannel_review.py"
    )
    msd_module = script_path.parents[1] / "src" / "dsb_states" / "v521_msd_final_visualization.py"
    case_result = write_multichannel_case_review(
        cases=cases,
        config=review_config,
        config_path=config_path,
        output_dir=output / "case_studies",
        ffmpeg_exe=args.ffmpeg_exe.resolve(),
        extra_source_paths=(
            alpha_path,
            selection_path,
            script_path,
            case_module,
            msd_module,
            reference_dir / "README.md",
            reference_dir / "animation_core.py",
            reference_dir / "build_demo_videos.py",
            args.original_matlab.resolve(),
        ),
    )

    table_dir = output / "tables"
    table_dir.mkdir()
    ranking_path = table_dir / "3h_site1_alpha_top5.csv"
    ranked.to_csv(ranking_path, index=False)
    contract = {
        "analysis_unit": "one Site1 trajectory with its paired Site2 shown for context",
        "source_cohort": (
            "previously frozen 100%-coverage Site1/Site2 cohort after excluding acquisitions "
            "with only 5 or 10 movie frames"
        ),
        "hour_post_delivery": float(selection_contract["hour_post_delivery"]),
        "ranking_site": str(selection_contract["site"]),
        "ranking": "descending fitted raw-MSD alpha",
        "top_n": int(selection_contract["top_n"]),
        "minimum_r2": float(selection_contract["minimum_r2"]),
        "fit_window_s": selection_contract["fit_window_s"],
        "excluded_selection_variables": ["appearance", "distance", "motion amplitude", "53BP1"],
        "paired_site2_role": "display and audit only; did not define rank",
        "physical_interpretation": "descriptive case screen, not validated discrete states",
    }
    contract_path = output / "METHOD_CONTRACT.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# 3 h complete-coverage Site1 alpha top five",
        "",
        (
            "Cases are ranked only by descending Site1 raw-MSD alpha among the frozen 3 h "
            "complete-coverage cohort, with the original R2 support threshold."
        ),
        "",
    ]
    for row in ranked.itertuples(index=False):
        report_lines.append(
            f"- Top {row.alpha_rank}: `{row.bundle_id}`; Site1 alpha={row.alpha:.3f}, "
            f"R2={row.r2_log:.3f}; paired Site2 alpha={row.paired_site2_alpha:.3f}, "
            f"R2={row.paired_site2_r2_log:.3f}."
        )
    report_lines.extend(
        [
            "",
            (
                "The videos reuse the frozen multichannel layout. Alpha remains an uncorrected "
                "10--50 s raw-MSD scaling exponent and is not a state label."
            ),
        ]
    )
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    sources = [
        alpha_path,
        selection_path,
        config_path,
        script_path,
        case_module,
        msd_module,
        reference_dir / "README.md",
        reference_dir / "animation_core.py",
        reference_dir / "build_demo_videos.py",
        args.original_matlab.resolve(),
    ]
    artifacts = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "OUTPUT_MANIFEST.json"
    )
    manifest_path = output / "OUTPUT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                    for path in sources
                ],
                "artifacts": [
                    {
                        "path": path.relative_to(output).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in artifacts
                ],
                "products": case_result["products"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_cases": len(ranked),
                "ranking_table": str(ranking_path),
                "products": case_result["products"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
