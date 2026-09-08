#!/usr/bin/env python
"""Select and render the largest supported near-10-s paired MSCD cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dsb_states.v521_case_study_multichannel_review import (
    configure_case_style,
    load_multichannel_review_cases,
    write_multichannel_case_review,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_near_10s_mscd(
    unit_curves: pd.DataFrame,
    eligible: pd.DataFrame,
    contract: dict[str, object],
) -> tuple[pd.DataFrame, int]:
    target = float(contract["target_lag_s"])
    lag_low, lag_high = (float(value) for value in contract["accepted_lag_window_s"])
    minimum_pairs = int(contract["minimum_endpoint_pairs"])
    top_n = int(contract["top_n"])
    curves = unit_curves.loc[
        unit_curves["metric"].astype(str).eq("mscd")
        & unit_curves["site"].astype(str).eq("pair")
        & unit_curves["bundle_id"].astype(str).isin(eligible["bundle_id"].astype(str))
    ].copy()
    for column in ("lag_median_s", "value", "pair_count", "lag_frames"):
        curves[column] = pd.to_numeric(curves[column], errors="coerce")
    curves = curves.loc[
        curves["lag_median_s"].between(lag_low, lag_high, inclusive="both")
        & np.isfinite(curves["value"])
        & (curves["value"] >= 0.0)
        & (curves["pair_count"] >= minimum_pairs)
    ].copy()
    curves["absolute_lag_error_s"] = (curves["lag_median_s"] - target).abs()
    nearest = (
        curves.sort_values(
            ["bundle_id", "absolute_lag_error_s", "pair_count", "lag_frames"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .drop_duplicates("bundle_id", keep="first")
        .merge(
            eligible,
            on="bundle_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_curve", "_eligible"),
        )
    )
    ranked = nearest.sort_values(
        ["value", "pair_count", "absolute_lag_error_s", "bundle_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).head(top_n)
    if len(ranked) != top_n:
        raise ValueError(f"Only {len(ranked)} rankable bundles; expected {top_n}")
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "mscd10_rank", np.arange(1, len(ranked) + 1, dtype=int))
    ranked.insert(1, "case_id", [f"mscd10_top{rank:02d}" for rank in ranked["mscd10_rank"]])
    ranked = ranked.rename(
        columns={
            "value": "observed_mscd_um2",
            "endpoint_pair_sd": "endpoint_pair_sd_um2",
            "lag_median_s": "observed_lag_s",
            "lag_frames": "observed_lag_frames",
            "pair_count": "endpoint_pair_count",
        }
    )
    ranked["relative_rms_change_nm"] = np.sqrt(ranked["observed_mscd_um2"]) * 1000.0
    ranked["selection_role"] = (
        "descending directly observed relative-vector MSCD at the supported lag nearest 10 s"
    )
    return ranked, len(nearest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-curves", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fullrun-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-exe", type=Path, required=True)
    parser.add_argument("--reference-script-dir", type=Path, required=True)
    parser.add_argument("--original-matlab", type=Path, required=True)
    args = parser.parse_args()

    curves_path = args.unit_curves.resolve()
    selection_path = args.selection_csv.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    eligible = pd.read_csv(selection_path)
    ranked, rankable_count = _rank_near_10s_mscd(
        pd.read_parquet(curves_path), eligible, config["selection"]
    )
    configured_ids = [str(item["bundle_id"]) for item in config["cases"]]
    selected_ids = ranked["bundle_id"].astype(str).tolist()
    if configured_ids != selected_ids:
        raise ValueError("Frozen case configuration does not match the recomputed MSCD ranking")

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
    case_result = write_multichannel_case_review(
        cases=cases,
        config=review_config,
        config_path=config_path,
        output_dir=output / "case_studies",
        ffmpeg_exe=args.ffmpeg_exe.resolve(),
        extra_source_paths=(
            curves_path,
            selection_path,
            script_path,
            case_module,
            reference_dir / "README.md",
            reference_dir / "animation_core.py",
            reference_dir / "build_demo_videos.py",
            args.original_matlab.resolve(),
        ),
    )

    table_dir = output / "tables"
    table_dir.mkdir()
    ranking_path = table_dir / "complete_coverage_mscd10_top4.csv"
    ranked.to_csv(ranking_path, index=False)
    selection_contract = config["selection"]
    contract = {
        "analysis_unit": "one complete-coverage Site1/Site2 bundle",
        "source_cohort": (
            "previously frozen 100%-coverage paired cohort after excluding acquisitions "
            "with only 5 or 10 movie frames"
        ),
        "eligible_bundle_count": len(eligible),
        "rankable_near_10s_bundle_count": rankable_count,
        "ranking": selection_contract["ranking"],
        "target_lag_s": float(selection_contract["target_lag_s"]),
        "accepted_lag_window_s": selection_contract["accepted_lag_window_s"],
        "minimum_endpoint_pairs": int(selection_contract["minimum_endpoint_pairs"]),
        "top_n": int(selection_contract["top_n"]),
        "excluded_selection_variables": [
            "appearance",
            "separation",
            "53BP1",
            "folder-hour",
            "acquisition identity",
            "learned state",
        ],
        "important_boundary": (
            "The rank uses a directly observed acquisition-supported lag, not an interpolated "
            "or fitted exact-10-s value. Exact observed lag is retained per case."
        ),
        "physical_interpretation": "descriptive high-relative-motion case screen, not a state label",
    }
    contract_path = output / "METHOD_CONTRACT.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# Complete-coverage paired MSCD top four",
        "",
        (
            f"Four cases are ranked from {rankable_count}/{len(eligible)} frozen eligible bundles "
            "by descending directly observed relative-vector MSCD at the supported lag nearest 10 s."
        ),
        "",
    ]
    for row in ranked.itertuples(index=False):
        report_lines.append(
            f"- Top {row.mscd10_rank}: `{row.bundle_id}`; {row.hour_post_delivery_eligible:g} h; "
            f"MSCD={row.observed_mscd_um2:.5f} um2 at {row.observed_lag_s:.2f} s "
            f"({int(row.observed_lag_frames)} frame(s), {int(row.endpoint_pair_count)} endpoint pairs); "
            f"relative RMS change={row.relative_rms_change_nm:.1f} nm."
        )
    report_lines.extend(
        [
            "",
            (
                "Videos reuse the frozen multichannel case-review layout. The ranking is a visual "
                "case screen and does not establish a discrete repair or motion state."
            ),
        ]
    )
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    sources = [
        curves_path,
        selection_path,
        config_path,
        script_path,
        case_module,
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
                "rankable_bundles": rankable_count,
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
