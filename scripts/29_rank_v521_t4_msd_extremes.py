"""Rank Site1/Site2 T4 trajectories by matched approximately-10-s raw MSD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SITES = ("site1", "site2")
TARGET_BIN_S = 10.0
MINIMUM_ENDPOINT_PAIRS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    resolved = path.resolve()
    stored = resolved if relative_to is None else resolved.relative_to(relative_to.resolve())
    return {"path": str(stored), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_ranking_census(
    membership: pd.DataFrame, unit_bins: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the T4 census, matched-10-s scores and site-specific extremes."""

    membership_columns = {
        "bundle_id",
        "cohort",
        "nd2_id",
        "crop_id",
        "fov_id",
        "hour_post_delivery",
        "allele_index",
        "site1_frames",
        "site2_frames",
        "paired_frames",
        "longest_paired_run",
        "paired_coverage",
        "movie_frames",
        "t4_included",
    }
    bin_columns = {
        "metric",
        "unit_id",
        "bundle_id",
        "site",
        "lag_bin_center_s",
        "lag_median_s",
        "value",
        "endpoint_pair_sd",
        "pair_count",
    }
    _require_columns(membership, membership_columns, name="membership")
    _require_columns(unit_bins, bin_columns, name="unit_bins")

    t4 = membership.loc[membership["t4_included"].astype(bool)].copy()
    if t4.empty or t4["bundle_id"].duplicated().any():
        raise ValueError("T4 membership must be nonempty and unique by bundle_id")
    t4_ids = set(t4["bundle_id"].astype(str))
    matched = unit_bins.loc[
        unit_bins["metric"].eq("msd")
        & unit_bins["site"].isin(SITES)
        & unit_bins["bundle_id"].astype(str).isin(t4_ids)
        & unit_bins["lag_bin_center_s"].eq(TARGET_BIN_S)
    ].copy()
    if matched.duplicated(["bundle_id", "site"]).any():
        raise ValueError("matched-10-s MSD table is not unique by bundle/site")
    lookup = {
        (str(row.bundle_id), str(row.site)): row for row in matched.itertuples(index=False)
    }

    metadata = [
        "bundle_id",
        "cohort",
        "nd2_id",
        "crop_id",
        "fov_id",
        "hour_post_delivery",
        "allele_index",
        "site1_frames",
        "site2_frames",
        "paired_frames",
        "longest_paired_run",
        "paired_coverage",
        "movie_frames",
    ]
    rows: list[dict[str, object]] = []
    for record in t4[metadata].itertuples(index=False):
        base = {column: getattr(record, column) for column in metadata}
        for site in SITES:
            source = lookup.get((str(record.bundle_id), site))
            eligible = bool(
                source is not None
                and pd.notna(source.value)
                and float(source.value) > 0.0
                and int(source.pair_count) >= MINIMUM_ENDPOINT_PAIRS
            )
            row: dict[str, object] = {
                **base,
                "unit_id": str(record.bundle_id),
                "site": site,
                "target_lag_bin_s": TARGET_BIN_S,
                "ranking_eligible": eligible,
                "exclusion_reason": "" if eligible else "no_supported_matched_10s_msd",
                "actual_lag_median_s": float(source.lag_median_s)
                if source is not None
                else pd.NA,
                "matched_10s_msd_um2": float(source.value) if eligible else pd.NA,
                "endpoint_pair_sd_um2": float(source.endpoint_pair_sd)
                if eligible
                else pd.NA,
                "endpoint_pair_count": int(source.pair_count) if source is not None else pd.NA,
            }
            rows.append(row)

    census = pd.DataFrame(rows).sort_values(["site", "bundle_id"], kind="stable")
    for column in (
        "actual_lag_median_s",
        "matched_10s_msd_um2",
        "endpoint_pair_sd_um2",
        "endpoint_pair_count",
    ):
        census[column] = pd.to_numeric(census[column], errors="coerce")
    eligible = census.loc[census["ranking_eligible"]].copy()
    if eligible.groupby("site")["bundle_id"].nunique().lt(20).any():
        raise ValueError("fewer than 20 matched-10-s T4 curves at one or more sites")

    eligible = eligible.sort_values(
        ["site", "matched_10s_msd_um2", "bundle_id"], kind="stable"
    ).reset_index(drop=True)
    eligible["rank_low_to_high"] = eligible.groupby("site").cumcount() + 1
    eligible["rank_high_to_low"] = (
        eligible.groupby("site")["matched_10s_msd_um2"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    extremes: list[pd.DataFrame] = []
    for _, site_rows in eligible.groupby("site", sort=True):
        low = site_rows.nsmallest(10, "matched_10s_msd_um2").copy()
        low["extreme"] = "lowest"
        low["extreme_rank"] = range(1, len(low) + 1)
        high = site_rows.nlargest(10, "matched_10s_msd_um2").copy()
        high["extreme"] = "highest"
        high["extreme_rank"] = range(1, len(high) + 1)
        extremes.extend([low, high])
    extreme_table = pd.concat(extremes, ignore_index=True)
    extreme_table["extreme_order"] = extreme_table["extreme"].map(
        {"lowest": 0, "highest": 1}
    )
    extreme_table = extreme_table.sort_values(
        ["site", "extreme_order", "extreme_rank"], kind="stable"
    ).drop(columns="extreme_order")
    return census.reset_index(drop=True), eligible, extreme_table.reset_index(drop=True)


def _support_summary(census: pd.DataFrame) -> pd.DataFrame:
    summary = (
        census.groupby(["site", "hour_post_delivery"], as_index=False)
        .agg(t4_units=("bundle_id", "size"), ranking_eligible=("ranking_eligible", "sum"))
        .sort_values(["site", "hour_post_delivery"], kind="stable")
    )
    summary["excluded_without_matched_10s_support"] = (
        summary["t4_units"] - summary["ranking_eligible"]
    )
    summary["eligible_fraction"] = summary["ranking_eligible"] / summary["t4_units"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc-membership", type=Path, required=True)
    parser.add_argument("--unit-lag-bins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    tables = output / "tables"
    tables.mkdir(parents=True)

    membership = pd.read_parquet(args.qc_membership)
    unit_bins = pd.read_parquet(args.unit_lag_bins)
    census, eligible, extremes = build_ranking_census(membership, unit_bins)
    support = _support_summary(census)

    paths = {
        "census": tables / "t4_matched_10s_msd_ranking_census.csv",
        "eligible": tables / "eligible_t4_matched_10s_msd_scores.csv",
        "extremes": tables / "site_specific_matched_10s_msd_top_bottom10.csv",
        "support": tables / "ranking_support_by_hour.csv",
    }
    census.to_csv(paths["census"], index=False)
    eligible.to_csv(paths["eligible"], index=False)
    extremes.to_csv(paths["extremes"], index=False)
    support.to_csv(paths["support"], index=False)

    contract = {
        "analysis": "v5.2.1_T4_site_specific_matched_10s_raw_MSD_extreme_selection",
        "source_cohort": "T4 strict paired trajectories from the prior full-field QC gallery",
        "t4_rule": {
            "site1_frames_min": 20,
            "site2_frames_min": 20,
            "paired_frames_min": 20,
            "longest_paired_run_min": 10,
            "paired_coverage_min": 0.5,
        },
        "msd_definition": "mean of squared 2D endpoint displacement at the matched lag, in um^2",
        "metric_support_inherited": {
            "observed_finite_endpoints_only": True,
            "coordinate_interpolation": False,
            "missing_frames_compressed": False,
            "minimum_endpoint_pairs_per_unit_lag": MINIMUM_ENDPOINT_PAIRS,
            "maximum_lag_frames": "min(50, floor(movie_frames/4))",
        },
        "ranking": {
            "performed_separately_for": list(SITES),
            "target_lag_bin_s": TARGET_BIN_S,
            "score": "trajectory-time-averaged raw MSD in the nominal 10-s physical-time bin",
            "actual_lag_stored_per_trajectory": True,
            "role": "case selection only; not a new biological endpoint",
            "why_not_full_10_50s_curve": (
                "requiring complete 10-50 s support retained only 1/131 T4 bundles at 2 h "
                "and 2/20 at 3.5 h, creating acquisition-duration/cadence selection bias"
            ),
        },
        "oligo_livefish_alignment": {
            "equation_4_msd_definition": "matched",
            "within_trajectory_then_equal_trajectory_average": "matched in population summaries",
            "equation_5_error_aware_fbm_fit": "not reproduced",
            "reason": "exposure-time, localization-error and motion-blur calibration are unavailable",
        },
        "visualization_created": False,
    }
    contract_path = output / "METHOD_CONTRACT.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    report = f"""# T4 site-specific matched-10-s MSD extremes (selection only)

T4 contains {census.bundle_id.nunique():,} paired bundles. Site1 and Site2 were ranked
independently. A trajectory is eligible only when its nominal 10-s bin contains at least eight
observed endpoint pairs. Coordinates are not interpolated and missing frames are not compressed.

The ranking value is each trajectory's time-averaged raw MSD in the matched approximately-10-s
physical-time bin. Actual median lags are stored per trajectory. This is a reproducible case-selection
rule, not an Oligo-LiveFISH Eq. 5 error-aware fit and not a complete-curve biological endpoint.

Eligible curves: {eligible.site.eq('site1').sum():,} Site1 and
{eligible.site.eq('site2').sum():,} Site2. No visualization was generated.
"""
    report_path = output / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    source_paths = [args.qc_membership.resolve(), args.unit_lag_bins.resolve(), Path(__file__).resolve()]
    artifact_paths = [*paths.values(), contract_path, report_path]
    manifest = {
        "sources": [_file_record(path) for path in source_paths],
        "artifacts": [_file_record(path, relative_to=output) for path in artifact_paths],
        "counts": {
            "t4_bundles": int(census.bundle_id.nunique()),
            "eligible_site1": int(eligible.site.eq("site1").sum()),
            "eligible_site2": int(eligible.site.eq("site2").sum()),
            "extreme_rows": len(extremes),
        },
    }
    (output / "OUTPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
