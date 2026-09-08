"""Build neutral Site1/Site2 frame and bundle tables from a frozen snapshot."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import load_json, parse_trajectory_filename, read_trajectory_csv, sha256_file


class TableBuildError(RuntimeError):
    """Raised when snapshot contents cannot be reconciled into analysis tables."""


class OutputScopeError(ValueError):
    """Raised before a writer targets a path outside the analysis output trees."""


@dataclass(frozen=True)
class AnalysisTablesResult:
    frame_table: pd.DataFrame
    bundle_metadata: pd.DataFrame
    reconciliation: dict[str, Any]


def canonical_output_roots(project_root: str | Path) -> tuple[Path, Path]:
    """Return the only two project trees in which analysis writes are allowed."""

    root = Path(project_root).resolve()
    return (root / "data_snapshot").resolve(), (root / "results").resolve()


def _resolve_from_project(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _is_within_or_equal(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_analysis_output_dir(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    allowed_roots: Iterable[str | Path],
    forbidden_roots: Iterable[str | Path] = (),
) -> Path:
    """Resolve and validate an output directory before any filesystem mutation.

    Supplied allowlist entries must themselves be within the canonical analysis
    ``data_snapshot`` or ``results`` trees.  This prevents a caller from
    bypassing the policy by declaring the source archive as an allowed root.
    Existing symlink/reparse-point parents are resolved by :meth:`Path.resolve`.
    """

    project = Path(project_root).resolve()
    canonical = canonical_output_roots(project)
    allowed = tuple(_resolve_from_project(value, project) for value in allowed_roots)
    if not allowed:
        raise OutputScopeError("At least one explicit analysis output root is required")
    invalid_allowed = [
        path
        for path in allowed
        if not any(_is_within_or_equal(path, canonical_root) for canonical_root in canonical)
    ]
    if invalid_allowed:
        raise OutputScopeError(
            "Allowed output roots must be within the analysis project's data_snapshot/results "
            f"trees: {[str(path) for path in invalid_allowed]}"
        )

    candidate = _resolve_from_project(output_dir, project)
    if not any(_is_within_or_equal(candidate, root) for root in allowed):
        raise OutputScopeError(
            f"Output directory is outside the explicit analysis allowlist: {candidate}"
        )

    forbidden = tuple(_resolve_from_project(value, project) for value in forbidden_roots)
    overlaps = [
        root
        for root in forbidden
        if _is_within_or_equal(candidate, root) or _is_within_or_equal(root, candidate)
    ]
    if overlaps:
        raise OutputScopeError(
            "Output directory overlaps a source/archive/production root: "
            f"{candidate}; forbidden={[str(path) for path in overlaps]}"
        )
    return candidate


def longest_true_run(values: pd.Series | np.ndarray | list[bool]) -> int:
    array = np.asarray(values, dtype=bool)
    longest = current = 0
    for value in array:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _read_snapshot_table(root: Path, filename: str) -> pd.DataFrame:
    path = root / filename
    if not path.exists():
        raise TableBuildError(f"Snapshot table is missing: {path}")
    return pd.read_parquet(path)


def _trajectory_map(manifest: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    trajectories = manifest.loc[manifest["record_type"].eq("formal_final_trajectory")].copy()
    if trajectories.duplicated(["bundle_id", "site_id"]).any():
        duplicate = trajectories.loc[
            trajectories.duplicated(["bundle_id", "site_id"], keep=False),
            ["bundle_id", "site_id"],
        ]
        raise TableBuildError(
            "Snapshot contains duplicate formal trajectory files for a bundle/site: "
            f"{duplicate.head().to_dict(orient='records')}"
        )
    return {
        (str(row.bundle_id), str(row.site_id)): row for row in trajectories.itertuples(index=False)
    }


def _optional_manifest_row(
    mapping: dict[tuple[str, str], Any], bundle_id: str, site_id: str
) -> Any | None:
    return mapping.get((bundle_id, site_id))


def _load_one_trajectory(snapshot_root: Path, manifest_row: Any | None) -> pd.DataFrame | None:
    if manifest_row is None:
        return None
    relative = Path(str(manifest_row.snapshot_relative_path))
    path = (snapshot_root / relative).resolve()
    try:
        path.relative_to(snapshot_root.resolve())
    except ValueError as exc:
        raise TableBuildError(
            f"Snapshot-relative trajectory path escaped root: {relative}"
        ) from exc
    if not path.exists():
        raise TableBuildError(f"Snapshotted trajectory file is missing: {path}")
    if sha256_file(path) != str(manifest_row.sha256):
        raise TableBuildError(f"Snapshot trajectory checksum mismatch: {path}")
    allele, site = parse_trajectory_filename(path)
    if allele != int(manifest_row.allele_index) or site != str(manifest_row.site_id):
        raise TableBuildError(f"Manifest/file identity mismatch: {path}")
    table = read_trajectory_csv(path)
    if table["frame"].duplicated().any():
        raise TableBuildError(f"Duplicate frame rows in trajectory: {path}")
    return table


def _assign_site(frame_table: pd.DataFrame, trajectory: pd.DataFrame | None, site_id: str) -> None:
    x_column = f"{site_id}_x_nm"
    y_column = f"{site_id}_y_nm"
    valid_column = f"{site_id}_valid"
    if trajectory is None or trajectory.empty:
        frame_table[x_column] = np.nan
        frame_table[y_column] = np.nan
        frame_table[valid_column] = False
        return
    if not trajectory["frame"].between(1, len(frame_table), inclusive="both").all():
        raise TableBuildError(f"{site_id} trajectory contains a frame outside the nominal movie")
    indexed = trajectory.set_index("frame")
    frame_table[x_column] = frame_table["frame"].map(indexed["x_nm"])
    frame_table[y_column] = frame_table["frame"].map(indexed["y_nm"])
    frame_table[valid_column] = frame_table[x_column].notna() & frame_table[y_column].notna()


def _metadata_lookup(metadata: pd.DataFrame) -> dict[tuple[str, str], Any]:
    if metadata.duplicated(["nd2_id", "cell_id"]).any():
        raise TableBuildError("crop_metadata is not unique by acquisition/cell")
    return {(str(row.nd2_id), str(row.cell_id)): row for row in metadata.itertuples(index=False)}


def _exact_times(snapshot_root: Path, metadata_row: Any, nominal: int) -> np.ndarray:
    relative = Path(str(metadata_row.exact_metadata_snapshot_relative_path))
    path = (snapshot_root / relative).resolve()
    if sha256_file(path) != str(metadata_row.exact_metadata_sha256):
        raise TableBuildError(f"Exact metadata checksum mismatch: {path}")
    metadata = load_json(path)
    values = metadata.get("time", {}).get("relative_time_s")
    if values is None:
        raise TableBuildError(f"Exact relative_time_s is absent: {path}")
    times = np.asarray(values, dtype=float)
    if len(times) != nominal or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise TableBuildError(f"Invalid exact time vector: {path}")
    return times


def _interval_per_frame(times: np.ndarray, representative_interval: float) -> np.ndarray:
    if len(times) <= 1:
        return np.full(len(times), representative_interval, dtype=float)
    intervals = np.empty(len(times), dtype=float)
    # Row i stores the interval from row i to i+1, matching the schema
    # validator.  The final row has no successor and repeats the last observed
    # interval solely to keep the required positive field populated.
    intervals[:-1] = np.diff(times)
    intervals[-1] = intervals[-2]
    return intervals


def _nullable_string(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    return str(value)


def build_analysis_tables(snapshot_root: str | Path) -> AnalysisTablesResult:
    """Build full nominal-frame rows for every authoritative bundle.

    ``micro_time_s`` comes from exact ``relative_time_s``.  The production
    uniform interval remains explicitly present as provenance but is not used
    to replace the real clock.
    """

    root = Path(snapshot_root).resolve()
    manifest = _read_snapshot_table(root, "source_manifest.parquet")
    bundles = _read_snapshot_table(root, "bundle_index_formal_usable.parquet")
    metadata = _read_snapshot_table(root, "crop_metadata.parquet")
    trajectories = _trajectory_map(manifest)
    crop_metadata = _metadata_lookup(metadata)
    frame_parts: list[pd.DataFrame] = []

    if bundles["bundle_id"].duplicated().any():
        raise TableBuildError("Snapshot bundle index contains duplicate bundle IDs")

    for bundle in bundles.itertuples(index=False):
        bundle_id = str(bundle.bundle_id)
        acquisition_id = str(bundle.nd2_id)
        cell_id = str(bundle.crop_id)
        nominal = int(bundle.total_frames)
        if nominal <= 0:
            raise TableBuildError(f"Bundle {bundle_id} has a non-positive frame count")
        metadata_row = crop_metadata.get((acquisition_id, cell_id))
        if metadata_row is None:
            raise TableBuildError(f"Bundle {bundle_id} has no crop metadata")
        times = _exact_times(root, metadata_row, nominal)
        representative_exact_dt = float(metadata_row.exact_interval_median_s)
        production_uniform_dt = float(metadata_row.production_uniform_interval_s)
        frames = pd.DataFrame(
            {
                "biological_experiment_id": pd.Series(
                    [_nullable_string(bundle.biological_experiment_id)] * nominal,
                    dtype="string",
                ),
                "acquisition_id": acquisition_id,
                "fov_id": str(bundle.fov),
                "cell_id": cell_id,
                "bundle_id": bundle_id,
                "allele_index": int(bundle.allele_index),
                "anchor_locus": bundle.anchor_locus,
                "hour_post_delivery": bundle.hour_post_delivery,
                "hour_mapping_source": _nullable_string(bundle.hour_mapping_source),
                "hour_mapping_status": str(bundle.hour_mapping_status),
                "acquisition_date_observed": _nullable_string(
                    getattr(metadata_row, "acquisition_date_observed", None)
                ),
                "acquisition_date_semantics": _nullable_string(
                    getattr(metadata_row, "acquisition_date_semantics", None)
                ),
                "frame": np.arange(1, nominal + 1, dtype=np.int64),
                "micro_time_s": times,
                "frame_interval_s": _interval_per_frame(times, representative_exact_dt),
                "exact_interval_median_s": representative_exact_dt,
                "production_uniform_frame_interval_s": production_uniform_dt,
                "production_timing_policy": _nullable_string(
                    metadata_row.production_uniform_policy
                ),
                "timing_axis_primary": "exact_relative_time_s",
                "timing_provenance": "exact clock; extraction used preserved uniform sidecar",
                "flank_mapping_status": "unresolved",
            }
        )
        source_paths: dict[str, Any] = {}
        for site_id in ("site1", "site2", "53bp1"):
            manifest_row = _optional_manifest_row(trajectories, bundle_id, site_id)
            trajectory = _load_one_trajectory(root, manifest_row)
            if site_id == "53bp1":
                _assign_site(frames, trajectory, "bp1")
                source_paths[site_id] = (
                    pd.NA if manifest_row is None else manifest_row.snapshot_relative_path
                )
            else:
                _assign_site(frames, trajectory, site_id)
                source_paths[site_id] = (
                    pd.NA if manifest_row is None else manifest_row.snapshot_relative_path
                )
        frames["paired_valid"] = frames["site1_valid"] & frames["site2_valid"]
        frames["site1_source_file"] = source_paths["site1"]
        frames["site2_source_file"] = source_paths["site2"]
        frames["bp1_source_file"] = source_paths["53bp1"]
        frames["source_file"] = "snapshot source_manifest.parquet"
        frame_parts.append(frames)

    frame_table = pd.concat(frame_parts, ignore_index=True)
    frame_table = frame_table.sort_values(["bundle_id", "frame"], kind="stable").reset_index(
        drop=True
    )
    bundle_metadata = summarize_bundles(frame_table, bundles)

    trajectory_manifest = manifest.loc[manifest["record_type"].eq("formal_final_trajectory")]
    expected_localizations = int(
        pd.to_numeric(trajectory_manifest["localization_rows"], errors="raise").sum()
    )
    observed_localizations = int(
        frame_table[["site1_valid", "site2_valid", "bp1_valid"]].sum().sum()
    )
    reconciliation = {
        "snapshot_bundle_count": len(bundles),
        "bundle_metadata_count": len(bundle_metadata),
        "nominal_frame_rows_expected": int(
            pd.to_numeric(bundles["total_frames"], errors="raise").sum()
        ),
        "frame_table_rows": len(frame_table),
        "trajectory_files_expected": len(trajectory_manifest),
        "trajectory_files_represented": int(
            frame_table.groupby("bundle_id")[["site1_valid", "site2_valid", "bp1_valid"]]
            .any()
            .sum()
            .sum()
        ),
        "localizations_expected": expected_localizations,
        "localizations_represented": observed_localizations,
    }
    reconciliation.update(
        {
            "bundle_count_reconciled": reconciliation["snapshot_bundle_count"]
            == reconciliation["bundle_metadata_count"],
            "nominal_frame_count_reconciled": reconciliation["nominal_frame_rows_expected"]
            == reconciliation["frame_table_rows"],
            "trajectory_file_count_reconciled": reconciliation["trajectory_files_expected"]
            == reconciliation["trajectory_files_represented"],
            "localization_count_reconciled": reconciliation["localizations_expected"]
            == reconciliation["localizations_represented"],
        }
    )
    if not all(
        reconciliation[key]
        for key in (
            "bundle_count_reconciled",
            "nominal_frame_count_reconciled",
            "trajectory_file_count_reconciled",
            "localization_count_reconciled",
        )
    ):
        raise TableBuildError(f"Table reconciliation failed: {json.dumps(reconciliation)}")
    return AnalysisTablesResult(frame_table, bundle_metadata, reconciliation)


def summarize_bundles(frame_table: pd.DataFrame, bundle_index: pd.DataFrame) -> pd.DataFrame:
    """Summarize nominal-frame coverage without interpreting missing channels biologically."""

    index = bundle_index.set_index("bundle_id", drop=False)
    rows: list[dict[str, Any]] = []
    for bundle_id, group in frame_table.groupby("bundle_id", sort=False):
        ordered = group.sort_values("frame", kind="stable")
        source = index.loc[bundle_id]
        nominal = len(ordered)
        site1 = ordered["site1_valid"].to_numpy(bool)
        site2 = ordered["site2_valid"].to_numpy(bool)
        paired = site1 & site2
        bp1 = ordered["bp1_valid"].to_numpy(bool)
        site1_count = int(site1.sum())
        site2_count = int(site2.sum())
        paired_count = int(paired.sum())
        rows.append(
            {
                "bundle_id": bundle_id,
                "biological_experiment_id": _nullable_string(source.biological_experiment_id),
                "acquisition_id": str(source.nd2_id),
                "fov_id": str(source.fov),
                "cell_id": str(source.crop_id),
                "allele_index": int(source.allele_index),
                "anchor_locus": source.anchor_locus,
                "hour_post_delivery": source.hour_post_delivery,
                "hour_mapping_source": _nullable_string(source.hour_mapping_source),
                "hour_mapping_status": str(source.hour_mapping_status),
                "acquisition_date_observed": _nullable_string(
                    ordered["acquisition_date_observed"].iloc[0]
                    if "acquisition_date_observed" in ordered
                    else None
                ),
                "acquisition_date_semantics": _nullable_string(
                    ordered["acquisition_date_semantics"].iloc[0]
                    if "acquisition_date_semantics" in ordered
                    else None
                ),
                "nominal_frame_count": nominal,
                "paired_valid_count": paired_count,
                "paired_coverage": paired_count / nominal,
                "longest_paired_run": longest_true_run(paired),
                # Neutral fields are primary while flank mapping remains unresolved.
                "site1_valid_count": site1_count,
                "site2_valid_count": site2_count,
                "site1_coverage": site1_count / nominal,
                "site2_coverage": site2_count / nominal,
                "longest_site1_run": longest_true_run(site1),
                "longest_site2_run": longest_true_run(site2),
                "median_site_intensity": np.nan,
                "bp1_valid_count": int(bp1.sum()),
                "bp1_detection_fraction": float(bp1.sum() / nominal),
                "manual_or_pipeline_qc_flags": str(source.run_manifest_status),
                "trajectory_usability_class": str(source.trajectory_usability_class),
                "flank_mapping_status": "unresolved",
                "exact_interval_median_s": float(ordered["exact_interval_median_s"].iloc[0]),
                "production_uniform_frame_interval_s": float(
                    ordered["production_uniform_frame_interval_s"].iloc[0]
                ),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values("bundle_id", kind="stable").reset_index(drop=True)


def write_analysis_tables(
    result: AnalysisTablesResult,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    allowed_roots: Iterable[str | Path],
    forbidden_roots: Iterable[str | Path] = (),
) -> dict[str, Path]:
    output = validate_analysis_output_dir(
        output_dir,
        project_root=project_root,
        allowed_roots=allowed_roots,
        forbidden_roots=forbidden_roots,
    )
    output.mkdir(parents=True, exist_ok=True)
    frame_path = output / "frame_table.parquet"
    bundle_path = output / "bundle_metadata.parquet"
    reconciliation_path = output / "table_reconciliation.json"
    result.frame_table.to_parquet(frame_path, index=False)
    result.bundle_metadata.to_parquet(bundle_path, index=False)
    reconciliation_path.write_text(
        json.dumps(result.reconciliation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "frame_table": frame_path,
        "bundle_metadata": bundle_path,
        "table_reconciliation": reconciliation_path,
    }
