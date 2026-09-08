"""Select complete paired-trajectory candidates for v5.2.1 case studies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompletePairCaseSelection:
    """Complete-pair trajectory summaries and declared extreme candidates."""

    trajectory_summary: pd.DataFrame
    hour_census: pd.DataFrame
    global_candidates: pd.DataFrame
    display_candidates: pd.DataFrame
    method_contract: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_extremes(summary: pd.DataFrame, n_extreme: int) -> pd.DataFrame:
    shortest = summary.sort_values(
        ["mean_separation_nm", "bundle_id"], ascending=[True, True]
    ).head(n_extreme)
    longest = summary.sort_values(
        ["mean_separation_nm", "bundle_id"], ascending=[False, True]
    ).head(n_extreme)
    pieces: list[pd.DataFrame] = []
    for label, frame in (("shortest", shortest), ("longest", longest)):
        selected = frame.copy()
        selected.insert(0, "extreme_class", label)
        selected.insert(1, "rank_within_class", np.arange(1, len(selected) + 1))
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def select_complete_pair_cases(
    bundle_index: pd.DataFrame,
    bundle_frames: pd.DataFrame,
    *,
    n_extreme: int = 5,
    display_min_frames: int = 20,
) -> CompletePairCaseSelection:
    """Rank fully observed paired trajectories by their mean 2D separation.

    The primary complete-pair predicate is exactly
    ``site1_points == site2_points == shared_site_frames == movie_frames``.
    No motion, distance, hour, acquisition, 53BP1 or historical T3/T4 rule is used.
    """

    index_columns = {
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
    }
    frame_columns = {
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
    }
    missing_index = sorted(index_columns.difference(bundle_index.columns))
    missing_frames = sorted(frame_columns.difference(bundle_frames.columns))
    if missing_index or missing_frames:
        raise ValueError(
            f"Missing required columns: bundle_index={missing_index}, bundle_frames={missing_frames}"
        )
    if n_extreme < 1 or display_min_frames < 1:
        raise ValueError("n_extreme and display_min_frames must be positive")
    if bundle_index["bundle_id"].duplicated().any():
        raise ValueError("bundle_index must be unique by bundle_id")
    if bundle_frames.duplicated(["bundle_id", "frame"]).any():
        raise ValueError("bundle_frames must be unique by bundle_id x frame")

    full_mask = (
        (bundle_index["movie_frames"] > 0)
        & (bundle_index["site1_points"] == bundle_index["movie_frames"])
        & (bundle_index["site2_points"] == bundle_index["movie_frames"])
        & (bundle_index["shared_site_frames"] == bundle_index["movie_frames"])
    )
    complete_index = bundle_index.loc[full_mask, sorted(index_columns)].copy()
    if complete_index.empty:
        raise ValueError("No complete paired trajectories satisfy the declared predicate")

    frames = bundle_frames.loc[
        bundle_frames["bundle_id"].isin(complete_index["bundle_id"]), sorted(frame_columns)
    ].copy()
    if not frames["site1_valid"].fillna(False).all():
        raise ValueError("Complete-pair cohort contains an invalid Site1 frame")
    if not frames["site2_valid"].fillna(False).all():
        raise ValueError("Complete-pair cohort contains an invalid Site2 frame")
    if not frames["shared_site_frame"].fillna(False).all():
        raise ValueError("Complete-pair cohort contains a non-shared frame")
    coordinate_columns = ["site1_x_nm", "site1_y_nm", "site2_x_nm", "site2_y_nm"]
    if not np.isfinite(frames[coordinate_columns].to_numpy(dtype=float)).all():
        raise ValueError("Complete-pair cohort contains non-finite coordinates")

    frames["distance_recomputed_nm"] = np.hypot(
        frames["site2_x_nm"] - frames["site1_x_nm"],
        frames["site2_y_nm"] - frames["site1_y_nm"],
    )
    cached_distance = frames["site1_site2_separation_nm"].to_numpy(dtype=float)
    recomputed_distance = frames["distance_recomputed_nm"].to_numpy(dtype=float)
    if not np.allclose(cached_distance, recomputed_distance, rtol=0.0, atol=1e-9):
        raise ValueError("Cached separation does not reconcile with recorded coordinates")

    grouped = frames.groupby("bundle_id", sort=False, observed=True)
    trajectory = grouped.agg(
        observed_frames=("frame", "size"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        first_time_s=("time_s", "min"),
        last_time_s=("time_s", "max"),
        mean_separation_nm=("distance_recomputed_nm", "mean"),
        sd_separation_nm=("distance_recomputed_nm", "std"),
        median_separation_nm=("distance_recomputed_nm", "median"),
        min_separation_nm=("distance_recomputed_nm", "min"),
        max_separation_nm=("distance_recomputed_nm", "max"),
    ).reset_index()
    summary = complete_index.merge(trajectory, on="bundle_id", how="left", validate="one_to_one")
    if not (summary["observed_frames"] == summary["movie_frames"]).all():
        raise ValueError("Frame-master row counts do not match movie_frames")
    summary["site1_coverage"] = summary["site1_points"] / summary["movie_frames"]
    summary["site2_coverage"] = summary["site2_points"] / summary["movie_frames"]
    summary["paired_coverage"] = summary["shared_site_frames"] / summary["movie_frames"]
    summary["recorded_duration_s"] = summary["last_time_s"] - summary["first_time_s"]
    summary["display_min_frames_eligible"] = summary["movie_frames"] >= display_min_frames
    summary["global_shortest_rank"] = summary["mean_separation_nm"].rank(
        method="first", ascending=True
    ).astype(int)
    summary["global_longest_rank"] = summary["mean_separation_nm"].rank(
        method="first", ascending=False
    ).astype(int)
    summary = summary.sort_values(["mean_separation_nm", "bundle_id"]).reset_index(drop=True)

    all_hour = (
        bundle_index.groupby("hour_post_delivery", as_index=False, observed=True)
        .agg(all_bundle_count=("bundle_id", "size"))
        .sort_values("hour_post_delivery")
    )
    complete_hour = complete_index.groupby(
        "hour_post_delivery", as_index=False, observed=True
    ).agg(
        complete_pair_count=("bundle_id", "size"),
        complete_pair_frame_count=("movie_frames", "sum"),
    )
    hour_census = all_hour.merge(complete_hour, on="hour_post_delivery", how="left")
    hour_census[["complete_pair_count", "complete_pair_frame_count"]] = hour_census[
        ["complete_pair_count", "complete_pair_frame_count"]
    ].fillna(0).astype(int)
    hour_census["complete_pair_percent"] = (
        100.0 * hour_census["complete_pair_count"] / hour_census["all_bundle_count"]
    )

    global_candidates = _pick_extremes(summary, n_extreme)
    display_pool = summary.loc[summary["display_min_frames_eligible"]].copy()
    display_candidates = _pick_extremes(display_pool, n_extreme)
    method_contract: dict[str, Any] = {
        "schema_version": 1,
        "source_cohort": "v5.2.1_unfiltered_cache",
        "analysis_unit": "one_site1_site2_bundle_trajectory",
        "complete_pair_predicate": (
            "movie_frames>0 AND site1_points=movie_frames AND "
            "site2_points=movie_frames AND shared_site_frames=movie_frames"
        ),
        "coverage_definition": "valid_recorded_frames/movie_frames",
        "distance_definition": "hypot(site2_x_nm-site1_x_nm, site2_y_nm-site1_y_nm)",
        "trajectory_statistic": "arithmetic_mean_over_all_nominal_movie_frames",
        "distance_unit": "nm",
        "coordinates": "recorded_coordinates_only",
        "interpolation": "prohibited",
        "missing_frame_compression": "prohibited",
        "primary_candidate_rule": f"global_bottom_{n_extreme}_and_top_{n_extreme}_by_mean",
        "primary_candidate_extra_filters": [],
        "display_sensitivity_rule": (
            f"same_ranking_restricted_to_movie_frames_at_least_{display_min_frames}"
        ),
        "display_sensitivity_is_primary": False,
        "53bp1_filter": None,
        "historical_t3_t4_filter": None,
        "inference": "descriptive_case_selection_only",
    }
    return CompletePairCaseSelection(
        trajectory_summary=summary,
        hour_census=hour_census,
        global_candidates=global_candidates,
        display_candidates=display_candidates,
        method_contract=method_contract,
    )


def write_complete_pair_case_selection(
    result: CompletePairCaseSelection,
    output_dir: Path,
    *,
    source_paths: Iterable[Path],
) -> dict[str, Path]:
    """Write immutable tables, contract, report and hash manifest."""

    output_dir = output_dir.resolve()
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "summary_csv": tables_dir / "complete_pair_trajectory_summary.csv",
        "summary_parquet": tables_dir / "complete_pair_trajectory_summary.parquet",
        "hour_census": tables_dir / "complete_pair_hour_census.csv",
        "global_candidates": tables_dir / "global_extreme_candidates.csv",
        "display_candidates": tables_dir / "display_preferred_extreme_candidates.csv",
        "contract": output_dir / "SELECTION_CONTRACT.json",
        "report": output_dir / "REPORT.md",
        "manifest": output_dir / "OUTPUT_MANIFEST.json",
    }
    result.trajectory_summary.to_csv(paths["summary_csv"], index=False)
    result.trajectory_summary.to_parquet(paths["summary_parquet"], index=False)
    result.hour_census.to_csv(paths["hour_census"], index=False)
    result.global_candidates.to_csv(paths["global_candidates"], index=False)
    result.display_candidates.to_csv(paths["display_candidates"], index=False)
    paths["contract"].write_text(
        json.dumps(result.method_contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    complete_count = len(result.trajectory_summary)
    complete_frames = int(result.trajectory_summary["movie_frames"].sum())
    report = (
        "# v5.2.1 complete-pair trajectory case selection\n\n"
        f"- Complete paired trajectories: **{complete_count:,}**.\n"
        f"- Fully paired recorded frames: **{complete_frames:,}**.\n"
        "- Primary cases are the five smallest and five largest trajectory-level mean "
        "Site1/Site2 2D separations, pooled across all folder-hours.\n"
        "- No motion, separation, hour, acquisition, 53BP1 or T3/T4 filter was applied.\n"
        "- A separate display-preferred sensitivity list requires at least 20 movie frames; "
        "it does not replace the primary global ranking.\n"
        "- This is descriptive case selection, not population inference. No figure was made.\n"
    )
    paths["report"].write_text(report, encoding="utf-8")

    sources = [Path(path).resolve() for path in source_paths]
    artifacts = [path for key, path in paths.items() if key != "manifest"]
    manifest = {
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sources
        ],
        "artifacts": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths
