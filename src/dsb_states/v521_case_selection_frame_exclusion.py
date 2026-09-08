"""Re-rank complete v5.2.1 pairs after declared acquisition-length exclusions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v521_case_selection import CompletePairCaseSelection


@dataclass(frozen=True)
class FrameExcludedCaseSelection:
    """Eligible, excluded and extreme trajectory tables after frame-count exclusion."""

    eligible_summary: pd.DataFrame
    excluded_summary: pd.DataFrame
    hour_census: pd.DataFrame
    candidates: pd.DataFrame
    method_contract: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_candidates(summary: pd.DataFrame, n_extreme: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    rules = (
        ("shortest", True),
        ("longest", False),
    )
    for label, ascending in rules:
        ranked = summary.sort_values(
            ["mean_separation_nm", "bundle_id"],
            ascending=[ascending, True],
        ).head(n_extreme)
        ranked = ranked.copy()
        ranked.insert(0, "extreme_class", label)
        ranked.insert(1, "rank_within_class", np.arange(1, len(ranked) + 1))
        pieces.append(ranked)
    return pd.concat(pieces, ignore_index=True)


def exclude_acquisition_frame_counts_and_rerank(
    base: CompletePairCaseSelection,
    *,
    excluded_movie_frames: Iterable[int] = (5, 10),
    n_extreme: int = 5,
) -> FrameExcludedCaseSelection:
    """Exclude declared total acquisition frame counts, then re-rank trajectory means."""

    excluded_counts = tuple(sorted({int(value) for value in excluded_movie_frames}))
    if not excluded_counts or any(value < 1 for value in excluded_counts):
        raise ValueError("excluded_movie_frames must contain positive integers")
    if n_extreme < 1:
        raise ValueError("n_extreme must be positive")

    complete = base.trajectory_summary.copy()
    excluded_mask = complete["movie_frames"].isin(excluded_counts)
    excluded = complete.loc[excluded_mask].copy()
    eligible = complete.loc[~excluded_mask].copy()
    if eligible.empty:
        raise ValueError("Acquisition-frame exclusion removed every complete pair")

    for column in ("global_shortest_rank", "global_longest_rank"):
        if column in eligible.columns:
            eligible = eligible.rename(columns={column: f"pre_exclusion_{column}"})
            excluded = excluded.rename(columns={column: f"pre_exclusion_{column}"})
    eligible["eligible_shortest_rank"] = eligible["mean_separation_nm"].rank(
        method="first", ascending=True
    ).astype(int)
    eligible["eligible_longest_rank"] = eligible["mean_separation_nm"].rank(
        method="first", ascending=False
    ).astype(int)
    eligible = eligible.sort_values(["mean_separation_nm", "bundle_id"]).reset_index(drop=True)
    excluded = excluded.sort_values(
        ["movie_frames", "hour_post_delivery", "mean_separation_nm", "bundle_id"]
    ).reset_index(drop=True)

    before = complete.groupby("hour_post_delivery", as_index=False, observed=True).agg(
        complete_pair_count_before_exclusion=("bundle_id", "size")
    )
    removed = excluded.groupby("hour_post_delivery", as_index=False, observed=True).agg(
        excluded_t5_t10_pair_count=("bundle_id", "size")
    )
    after = eligible.groupby("hour_post_delivery", as_index=False, observed=True).agg(
        eligible_complete_pair_count=("bundle_id", "size"),
        eligible_complete_pair_frame_count=("movie_frames", "sum"),
    )
    census = base.hour_census[["hour_post_delivery", "all_bundle_count"]].copy()
    census = census.merge(before, on="hour_post_delivery", how="left")
    census = census.merge(removed, on="hour_post_delivery", how="left")
    census = census.merge(after, on="hour_post_delivery", how="left")
    count_columns = [
        "complete_pair_count_before_exclusion",
        "excluded_t5_t10_pair_count",
        "eligible_complete_pair_count",
        "eligible_complete_pair_frame_count",
    ]
    census[count_columns] = census[count_columns].fillna(0).astype(int)
    census["eligible_complete_pair_percent_of_all_bundles"] = (
        100.0 * census["eligible_complete_pair_count"] / census["all_bundle_count"]
    )

    candidates = _rank_candidates(eligible, n_extreme)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "source_cohort": "v5.2.1_unfiltered_cache",
        "analysis_unit": "one_site1_site2_bundle_trajectory",
        "complete_pair_predicate": (
            "movie_frames>0 AND site1_points=movie_frames AND "
            "site2_points=movie_frames AND shared_site_frames=movie_frames"
        ),
        "additional_primary_exclusion": {
            "field": "movie_frames",
            "excluded_exact_values": list(excluded_counts),
            "semantics": "exclude_acquisitions_whose_total_time_axis_has_T_equal_to_5_or_10",
        },
        "distance_definition": "hypot(site2_x_nm-site1_x_nm, site2_y_nm-site1_y_nm)",
        "trajectory_statistic": "arithmetic_mean_over_all_frames_of_the_acquisition",
        "candidate_rule": f"global_bottom_{n_extreme}_and_top_{n_extreme}_after_exclusion",
        "other_filters": [],
        "53bp1_filter": None,
        "historical_t3_t4_filter": None,
        "interpolation": "prohibited",
        "inference": "descriptive_case_selection_only",
    }
    return FrameExcludedCaseSelection(
        eligible_summary=eligible,
        excluded_summary=excluded,
        hour_census=census,
        candidates=candidates,
        method_contract=contract,
    )


def write_frame_excluded_case_selection(
    result: FrameExcludedCaseSelection,
    output_dir: Path,
    *,
    source_paths: Iterable[Path],
) -> dict[str, Path]:
    """Write tables, method contract, concise report and hash manifest."""

    output_dir = output_dir.resolve()
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "eligible_csv": tables_dir / "eligible_complete_pair_trajectory_summary.csv",
        "eligible_parquet": tables_dir / "eligible_complete_pair_trajectory_summary.parquet",
        "excluded_csv": tables_dir / "excluded_t5_t10_complete_pair_trajectories.csv",
        "hour_census": tables_dir / "complete_pair_hour_census_after_t5_t10_exclusion.csv",
        "candidates": tables_dir / "extreme_candidates_after_t5_t10_exclusion.csv",
        "contract": output_dir / "SELECTION_CONTRACT.json",
        "report": output_dir / "REPORT.md",
        "manifest": output_dir / "OUTPUT_MANIFEST.json",
    }
    result.eligible_summary.to_csv(paths["eligible_csv"], index=False)
    result.eligible_summary.to_parquet(paths["eligible_parquet"], index=False)
    result.excluded_summary.to_csv(paths["excluded_csv"], index=False)
    result.hour_census.to_csv(paths["hour_census"], index=False)
    result.candidates.to_csv(paths["candidates"], index=False)
    paths["contract"].write_text(
        json.dumps(result.method_contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = (
        "# v5.2.1 complete-pair case selection after T=5/10 exclusion\n\n"
        f"- Complete pairs before exclusion: **{len(result.eligible_summary) + len(result.excluded_summary):,}**.\n"
        f"- Excluded complete pairs from T=5/10 acquisitions: **{len(result.excluded_summary):,}**.\n"
        f"- Eligible complete pairs after exclusion: **{len(result.eligible_summary):,}**.\n"
        "- Both sites still cover every frame of their own acquisition.\n"
        "- The only added rule is exclusion of acquisitions with exactly 5 or 10 total frames.\n"
        "- Candidates are the five smallest and five largest trajectory-level mean 2D separations.\n"
        "- No figure was made and no biological-state interpretation was assigned.\n"
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
