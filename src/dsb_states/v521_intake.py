"""Build a compact, analysis-ready view of the frozen v5.2.1 DSB full run.

The production run is strictly read-only.  This module materializes normalized
Parquet tables plus provenance indexes beneath ``data_snapshot``.  Availability
is assessed separately for Site1, Site2, paired frames, locus-linked 53BP1 and
global 53BP1 objects; the task-level status is never treated as a substitute
for those asset-specific checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

SCHEMA_VERSION = "v5.2.1-analysis-intake-v1"
CHANNEL_TO_SITE = {"R": "site1", "P": "site2", "G": "53bp1_spt"}
BP1_FILES = (
    "bp1_focus_labels.tif",
    "bp1_frame_segmentation.csv",
    "bp1_focus_objects.csv",
    "bp1_allele_frame_metrics.csv",
    "bp1_allele_summary.csv",
    "bp1_analysis_manifest.json",
)
INDEX_TABLES = (
    "acquisition_index",
    "crop_index",
    "trajectory_index",
    "allele_index",
    "unavailable_crops",
)
PARTITIONED_TABLES = (
    "trajectory_points",
    "bp1_allele_frames",
    "bp1_allele_summary",
    "bp1_frame_segmentation",
    "bp1_focus_objects",
)


class IntakeError(RuntimeError):
    """Raised when the frozen source or generated intake violates its contract."""


@dataclass(frozen=True)
class IntakeBuildResult:
    root: Path
    summary: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, *, sep: str = ",") -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=sep, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _normalize_nd2(value: str) -> str:
    value = str(value).strip()
    return value[:-4] if value.lower().endswith(".nd2") else value


def _localize(path_value: str, source_root: Path) -> Path:
    """Map a frozen Sherlock path to the copied Windows run root."""

    text = str(path_value).replace("\\", "/")
    marker = f"/{source_root.name}/"
    if marker in text:
        relative = text.split(marker, 1)[1]
        return source_root.joinpath(*relative.split("/"))
    candidate = Path(path_value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    raise IntakeError(f"Path is outside the frozen run contract: {path_value}")


def _relative(path: Path, source_root: Path) -> str:
    return path.resolve().relative_to(source_root.resolve()).as_posix()


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().isin({"1", "true", "yes"})


def _source_identity(
    source_root: Path, project_root: Path, config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    source_files = {
        "run_contract": source_root / "00_provenance" / "run_contract.json",
        "new_10h_acquisitions": source_root / "00_provenance" / "new_10h_acquisitions.tsv",
        "acquisition_inventory": source_root / "01_inventory" / "acquisition_inventory.tsv",
        "crop_final_audit": source_root / "06_reports" / "crop_final_audit.tsv",
        "final_audit_summary": source_root / "06_reports" / "final_audit_summary.json",
        "posthoc_summary": source_root / "06_reports" / "posthoc_reclassification_summary.json",
    }
    missing = [str(path) for path in source_files.values() if not path.exists()]
    if missing:
        raise IntakeError(f"Frozen source lacks required controls: {missing}")
    records = {
        role: {
            "relative_path": _relative(path, source_root),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in source_files.items()
    }
    m0_raw = Path(str(config["metadata"]["existing_m0_path"]))
    m0_path = m0_raw if m0_raw.is_absolute() else project_root / m0_raw
    if not m0_path.exists():
        raise IntakeError(f"M0 acquisition metadata is missing: {m0_path}")
    records["m0_acquisition_metadata"] = {
        "project_relative_path": m0_path.resolve().relative_to(project_root.resolve()).as_posix(),
        "bytes": m0_path.stat().st_size,
        "sha256": _sha256(m0_path),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_root_name": source_root.name,
        "source_controls": records,
        "metadata_contract": config.get("metadata", {}),
        "validation_contract": config.get("validation", {}),
        "materialization_contract": config.get("materialize", {}),
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["identity_sha256"] = identity
    return identity, payload


def _load_hour_map(project_root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = Path(str(config["metadata"]["existing_m0_path"]))
    path = raw if raw.is_absolute() else project_root / raw
    table = pd.read_csv(path)
    required = {"nd2_id", "fov_id", "hour_post_delivery", "hour_mapping_evidence"}
    if missing := required.difference(table.columns):
        raise IntakeError(f"M0 metadata lacks columns: {sorted(missing)}")
    out: dict[str, dict[str, Any]] = {}
    for row in table.itertuples(index=False):
        out[_normalize_nd2(row.nd2_id)] = {
            "fov_id": str(row.fov_id),
            "hour_post_delivery": float(row.hour_post_delivery),
            "hour_mapping_evidence": str(row.hour_mapping_evidence),
        }
    return out


def _add_common_columns(
    frame: pd.DataFrame,
    *,
    cohort: str,
    nd2_id: str,
    crop_id: str,
    hour: float,
    fov_id: str,
    allele_rows: bool = False,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "crop_id", crop_id)
    frame.insert(0, "nd2_id", nd2_id)
    frame.insert(0, "cohort", cohort)
    frame.insert(3, "fov_id", fov_id)
    frame.insert(4, "hour_post_delivery", hour)
    if allele_rows and "allele_index" in frame:
        allele = pd.to_numeric(frame["allele_index"], errors="raise").astype(int)
        frame["allele_index"] = allele
        frame.insert(
            5,
            "bundle_id",
            [f"{nd2_id}|{crop_id}|a{value:03d}" for value in allele],
        )
    return frame


def _write_partition(
    frames: list[pd.DataFrame],
    root: Path,
    table_name: str,
    ordinal: int,
    nd2_id: str,
    compression: str,
) -> tuple[int, Path | None]:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return 0, None
    table = _parquet_safe(pd.concat(nonempty, ignore_index=True))
    target = root / "tables" / table_name / f"part_{ordinal:03d}_{_safe_slug(nd2_id)}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(target, index=False, compression=compression)
    return len(table), target


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed identifier/list columns without changing numeric measurements."""

    frame = frame.copy()
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].astype("string")
    return frame


def _trajectory_validation(
    manifest_row: Any,
    trajectory_path: Path,
    exact_times: list[float],
    minimum_points: int,
) -> tuple[pd.DataFrame, str]:
    if not trajectory_path.exists():
        return pd.DataFrame(), "BASELINE_CSV_MISSING"
    points = _read_csv(trajectory_path)
    required = {"frame", "x_nm", "y_nm"}
    if missing := required.difference(points.columns):
        return pd.DataFrame(), f"BASELINE_COLUMNS_MISSING:{','.join(sorted(missing))}"
    points = points.loc[:, ["frame", "x_nm", "y_nm"]].copy()
    for column in points.columns:
        points[column] = pd.to_numeric(points[column], errors="coerce")
    if points.isna().any().any():
        return pd.DataFrame(), "BASELINE_NONNUMERIC_OR_MISSING"
    points["frame"] = points["frame"].astype(int)
    if len(points) < minimum_points:
        return pd.DataFrame(), "BASELINE_BELOW_MINIMUM_POINTS"
    if points["frame"].duplicated().any() or not points["frame"].is_monotonic_increasing:
        return pd.DataFrame(), "BASELINE_FRAME_ORDER_OR_DUPLICATE"
    if points["frame"].min() < 1 or points["frame"].max() > len(exact_times):
        return pd.DataFrame(), "BASELINE_FRAME_OUTSIDE_EXACT_TIMING"
    if int(manifest_row.points) != len(points):
        return pd.DataFrame(), "BASELINE_POINT_COUNT_MISMATCH"
    points["time_s"] = [float(exact_times[value - 1]) for value in points["frame"]]
    points["uniform_time_s"] = (points["frame"] - 1) * float(manifest_row.frame_interval_s)
    points["x_um"] = points["x_nm"] / 1000.0
    points["y_um"] = points["y_nm"] / 1000.0
    return points, "VALID"


def classify_availability(
    *,
    site1: bool,
    site2: bool,
    paired_tracks: bool,
    paired_frames: bool,
    bp1_global: bool,
    bp1_locus: bool,
    bp1_spt: bool,
) -> str:
    """Return a descriptive asset class; this is not a biological QC tier."""

    if paired_frames and bp1_locus:
        return "PAIRED_FRAMES_PLUS_LOCUS_BP1"
    if paired_tracks and bp1_locus:
        return "PAIRED_TRACKS_NO_SHARED_FRAMES_PLUS_LOCUS_BP1"
    if site1 and site2 and bp1_locus:
        return "SITE1_SITE2_UNPAIRED_PLUS_LOCUS_BP1"
    if site2 and bp1_locus:
        return "SITE2_PLUS_LOCUS_BP1"
    if site1 and bp1_global:
        return "SITE1_PLUS_GLOBAL_BP1"
    if bp1_global or bp1_spt:
        return "BP1_ONLY"
    if site1 or site2:
        return "PARTIAL_SITE_TRAJECTORY"
    return "NO_USABLE_CURRENT_OUTPUT"


def _failure_reason(task_status: str, result_dir: Path) -> str:
    if task_status == "CORE_COMPLETE_POSTPROCESS_FAILED":
        return "EMPTY_BASELINE;GLOBAL_BP1_ONLY;POSTPROCESS_FAILED"
    if task_status != "TECHNICAL_FAILED":
        return ""
    log_path = result_dir / "log_anchor_roi_v4.txt"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if "No P anchor trajectories found" in text:
        return "NO_FORMAL_SITE2_P_ANCHOR"
    if "Input/output error" in text or "OSError: [Errno 5]" in text:
        return "PIPELINE_IO_ERROR"
    return "PIPELINE_INTERRUPTED_OR_UNCLASSIFIED"


def _availability_summary(crops: pd.DataFrame, alleles: pd.DataFrame) -> dict[str, int]:
    columns = (
        "site1_usable",
        "site2_usable",
        "bp1_spt_usable",
        "paired_tracks_usable",
        "paired_frames_usable",
        "bp1_global_usable",
        "bp1_locus_usable",
        "bp1_continuous_intensity_usable",
        "bp1_component_association_observed",
    )
    result = {f"crops_{column}": int(crops[column].sum()) for column in columns}
    if not alleles.empty:
        for column in (
            "site1_usable",
            "site2_usable",
            "paired_tracks_usable",
            "paired_frames_usable",
            "bp1_locus_usable",
        ):
            result[f"alleles_{column}"] = int(alleles[column].sum())
    return result


def build_intake(config_path: Path) -> IntakeBuildResult:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise IntakeError("Intake configuration schema does not match the builder")
    source_root = Path(str(config["source"]["root"])).resolve()
    contract = _json(source_root / "00_provenance" / "run_contract.json")
    repository_contract = contract.get("repository", {})
    if str(repository_contract.get("version")) != str(config["source"]["expected_version"]):
        raise IntakeError("Unexpected production version")
    if str(repository_contract.get("commit")) != str(config["source"]["expected_commit"]):
        raise IntakeError("Unexpected production commit")

    identity, source_identity = _source_identity(source_root, project_root, config)
    output_parent_raw = Path(str(config["output"]["parent"]))
    output_parent = (
        output_parent_raw if output_parent_raw.is_absolute() else project_root / output_parent_raw
    ).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / f"{config['output']['prefix']}_{identity[:12]}"
    if output.exists():
        manifest = output / "build_manifest.json"
        if manifest.exists() and _json(manifest).get("status") == "COMPLETE":
            return IntakeBuildResult(output, _json(manifest)["summary"])
        raise IntakeError(f"Refusing to overwrite incomplete intake: {output}")
    staging = output_parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False)

    compression = str(config["output"].get("compression", "zstd"))
    minimum_points = int(config["validation"]["minimum_trajectory_points"])
    hour_map = _load_hour_map(project_root, config)
    acquisition = _read_csv(source_root / "01_inventory" / "acquisition_inventory.tsv", sep="\t")
    audit = _read_csv(source_root / "06_reports" / "crop_final_audit.tsv", sep="\t")
    audit = audit.sort_values(["nd2_id", "crop_index"], kind="stable").reset_index(drop=True)

    acquisition_records: list[dict[str, Any]] = []
    acquisition_meta: dict[str, dict[str, Any]] = {}
    for row in acquisition.itertuples(index=False):
        nd2_id = _normalize_nd2(row.nd2_id)
        if row.cohort == "new_10h":
            hour = float(config["metadata"]["new_10h_hour_post_delivery"])
            evidence = str(config["metadata"]["new_10h_hour_evidence"])
            fov_id = str(row.fov_label)
        else:
            mapped = hour_map.get(nd2_id)
            hour = np.nan if mapped is None else float(mapped["hour_post_delivery"])
            evidence = "" if mapped is None else str(mapped["hour_mapping_evidence"])
            source_text = str(row.source_nd2).replace("\\", "/")
            fov_id = Path(source_text).parent.name if mapped is None else str(mapped["fov_id"])
        acquisition_meta[nd2_id] = {
            "cohort": str(row.cohort),
            "fov_id": fov_id,
            "hour_post_delivery": hour,
            "hour_mapping_evidence": evidence,
        }
        acquisition_records.append(
            {
                "cohort": row.cohort,
                "nd2_id": nd2_id,
                "fov_id": fov_id,
                "hour_post_delivery": hour,
                "hour_mapping_evidence": evidence,
                "macro_time_semantics": config["metadata"]["macro_time_semantics"],
                "acquisition_role": config["metadata"]["acquisition_role"],
                "source_nd2": row.source_nd2,
                "movie_frames": row.frames,
                "z_planes": row.z_planes,
                "median_interval_s": row.median_interval_s,
                "trajectory_eligible": bool(int(row.trajectory_eligible)),
                "acquisition_exclusion_reason": row.exclusion_reason,
                "crop_policy": row.crop_policy,
            }
        )

    crop_records: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    allele_records: list[dict[str, Any]] = []
    table_rows: Counter[str] = Counter()

    try:
        for ordinal, (nd2_id_raw, acquisition_crops) in enumerate(audit.groupby("nd2_id"), 1):
            nd2_id = _normalize_nd2(nd2_id_raw)
            if nd2_id not in acquisition_meta:
                raise IntakeError(f"Crop audit acquisition is absent from inventory: {nd2_id}")
            meta_acq = acquisition_meta[nd2_id]
            partition_frames: dict[str, list[pd.DataFrame]] = {
                name: [] for name in PARTITIONED_TABLES
            }

            for row in acquisition_crops.itertuples(index=False):
                result_dir = _localize(row.result_dir, source_root)
                metadata_path = _localize(row.metadata_sidecar, source_root)
                metadata = _json(metadata_path)
                exact_times = [float(value) for value in metadata["time"]["relative_time_s"]]
                if len(exact_times) != int(row.frames):
                    raise IntakeError(f"Exact timing length mismatch: {row.crop_id}")
                hour = float(meta_acq["hour_post_delivery"])
                fov_id = str(meta_acq["fov_id"])
                cohort = str(row.cohort)

                baseline_path = result_dir / "baseline_longest" / "baseline_manifest.csv"
                baseline = _read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
                valid_by_allele: dict[int, dict[str, dict[str, Any]]] = {}
                point_partition: list[pd.DataFrame] = []
                if not baseline.empty:
                    for base in baseline.itertuples(index=False):
                        channel = str(base.channel)
                        if channel not in CHANNEL_TO_SITE:
                            raise IntakeError(f"Unknown baseline channel {channel}: {row.crop_id}")
                        allele_index = int(base.allele_index)
                        site_id = CHANNEL_TO_SITE[channel]
                        trajectory_path = _localize(base.baseline_csv, source_root)
                        points, validation = _trajectory_validation(
                            base, trajectory_path, exact_times, minimum_points
                        )
                        usable = validation == "VALID"
                        trajectory_id = f"{nd2_id}|{row.crop_id}|a{allele_index:03d}|{site_id}"
                        bundle_id = f"{nd2_id}|{row.crop_id}|a{allele_index:03d}"
                        trajectory_records.append(
                            {
                                "trajectory_id": trajectory_id,
                                "bundle_id": bundle_id,
                                "cohort": cohort,
                                "nd2_id": nd2_id,
                                "crop_id": row.crop_id,
                                "fov_id": fov_id,
                                "hour_post_delivery": hour,
                                "allele_index": allele_index,
                                "site_id": site_id,
                                "source_channel": channel,
                                "usable": usable,
                                "validation_reason": validation,
                                "points": int(base.points),
                                "first_frame": int(base.first_frame),
                                "last_frame": int(base.last_frame),
                                "frame_span": int(base.frame_span),
                                "temporal_coverage_fraction": float(
                                    base.temporal_coverage_fraction
                                ),
                                "maximum_missing_frames_between_points": int(
                                    base.maximum_missing_frames_between_points
                                ),
                                "median_step_px": float(base.median_step_px),
                                "p95_step_px": float(base.p95_step_px),
                                "frame_interval_s_production": float(base.frame_interval_s),
                                "movie_frame_count": int(base.movie_frame_count),
                                "pixel_size_nm_per_px": float(base.pixel_size_nm_per_px),
                                "source_relative_path": _relative(trajectory_path, source_root),
                            }
                        )
                        if usable:
                            valid_by_allele.setdefault(allele_index, {})[site_id] = {
                                "points": len(points),
                                "path": _relative(trajectory_path, source_root),
                            }
                            points.insert(0, "trajectory_id", trajectory_id)
                            points.insert(1, "bundle_id", bundle_id)
                            points.insert(2, "cohort", cohort)
                            points.insert(3, "nd2_id", nd2_id)
                            points.insert(4, "crop_id", row.crop_id)
                            points.insert(5, "fov_id", fov_id)
                            points.insert(6, "hour_post_delivery", hour)
                            points.insert(7, "allele_index", allele_index)
                            points.insert(8, "site_id", site_id)
                            point_partition.append(points)
                partition_frames["trajectory_points"].extend(point_partition)

                bp1_dir = result_dir / "53bp1_metrics"
                bp_paths = {name: bp1_dir / name for name in BP1_FILES}
                bp_manifest: dict[str, Any] = {}
                if bp_paths["bp1_analysis_manifest.json"].exists():
                    bp_manifest = _json(bp_paths["bp1_analysis_manifest.json"])
                bp1_global_usable = (
                    all(path.exists() for path in bp_paths.values())
                    and bp_manifest.get("status") == "COMPLETE"
                    and bp_manifest.get("schema_version")
                    == config["validation"]["require_bp1_sidecar_schema"]
                )

                bp_summary = (
                    _read_csv(bp_paths["bp1_allele_summary.csv"])
                    if bp_paths["bp1_allele_summary.csv"].exists()
                    else pd.DataFrame()
                )
                summary_by_allele: dict[int, Any] = {}
                if not bp_summary.empty:
                    bp_summary = _add_common_columns(
                        bp_summary,
                        cohort=cohort,
                        nd2_id=nd2_id,
                        crop_id=row.crop_id,
                        hour=hour,
                        fov_id=fov_id,
                        allele_rows=True,
                    )
                    partition_frames["bp1_allele_summary"].append(bp_summary)
                    summary_by_allele = {
                        int(value.allele_index): value
                        for value in bp_summary.itertuples(index=False)
                    }

                source_to_output = {
                    "bp1_allele_frame_metrics.csv": "bp1_allele_frames",
                    "bp1_frame_segmentation.csv": "bp1_frame_segmentation",
                    "bp1_focus_objects.csv": "bp1_focus_objects",
                }
                for source_name, output_name in source_to_output.items():
                    if not config["materialize"].get(output_name, True):
                        continue
                    path = bp_paths[source_name]
                    if not path.exists():
                        continue
                    frame = _read_csv(path)
                    if frame.empty:
                        continue
                    frame = _add_common_columns(
                        frame,
                        cohort=cohort,
                        nd2_id=nd2_id,
                        crop_id=row.crop_id,
                        hour=hour,
                        fov_id=fov_id,
                        allele_rows=source_name == "bp1_allele_frame_metrics.csv",
                    )
                    partition_frames[output_name].append(frame)

                allele_indices = sorted(set(valid_by_allele).union(summary_by_allele))
                allele_rows_for_crop: list[dict[str, Any]] = []
                for allele_index in allele_indices:
                    available = valid_by_allele.get(allele_index, {})
                    summary = summary_by_allele.get(allele_index)
                    site1 = "site1" in available
                    site2 = "site2" in available
                    bp1_spt = "53bp1_spt" in available
                    shared_frames = 0 if summary is None else int(summary.shared_site_frames)
                    intensity_frames = 0 if summary is None else int(summary.intensity_valid_frames)
                    component_frames = (
                        0 if summary is None else int(summary.associated_component_frames)
                    )
                    record = {
                        "bundle_id": f"{nd2_id}|{row.crop_id}|a{allele_index:03d}",
                        "cohort": cohort,
                        "nd2_id": nd2_id,
                        "crop_id": row.crop_id,
                        "fov_id": fov_id,
                        "hour_post_delivery": hour,
                        "allele_index": allele_index,
                        "site1_usable": site1,
                        "site2_usable": site2,
                        "bp1_spt_usable": bp1_spt,
                        "site1_points": int(available.get("site1", {}).get("points", 0)),
                        "site2_points": int(available.get("site2", {}).get("points", 0)),
                        "bp1_spt_points": int(available.get("53bp1_spt", {}).get("points", 0)),
                        "paired_tracks_usable": site1 and site2,
                        "shared_site_frames": shared_frames,
                        "paired_frames_usable": site1 and site2 and shared_frames > 0,
                        "bp1_locus_usable": site2 and summary is not None,
                        "bp1_continuous_intensity_valid_frames": intensity_frames,
                        "bp1_continuous_intensity_usable": site2 and intensity_frames > 0,
                        "bp1_component_associated_frames": component_frames,
                        "bp1_component_association_observed": component_frames > 0,
                    }
                    allele_records.append(record)
                    allele_rows_for_crop.append(record)

                site1_usable = any(value["site1_usable"] for value in allele_rows_for_crop)
                site2_usable = any(value["site2_usable"] for value in allele_rows_for_crop)
                bp1_spt_usable = any(value["bp1_spt_usable"] for value in allele_rows_for_crop)
                paired_tracks = any(value["paired_tracks_usable"] for value in allele_rows_for_crop)
                paired_frames = any(value["paired_frames_usable"] for value in allele_rows_for_crop)
                bp1_locus = any(value["bp1_locus_usable"] for value in allele_rows_for_crop)
                bp1_intensity = any(
                    value["bp1_continuous_intensity_usable"] for value in allele_rows_for_crop
                )
                bp1_component = any(
                    value["bp1_component_association_observed"] for value in allele_rows_for_crop
                )
                crop_records.append(
                    {
                        "cohort": cohort,
                        "nd2_id": nd2_id,
                        "crop_id": row.crop_id,
                        "crop_index": int(row.crop_index),
                        "fov_id": fov_id,
                        "hour_post_delivery": hour,
                        "movie_frames": int(row.frames),
                        "exact_timing_points": len(exact_times),
                        "frame_interval_s_production": float(metadata["time"]["finterval_s"]),
                        "task_status": row.task_status,
                        "runner_exit_code": int(row.runner_exit_code),
                        "site1_usable": site1_usable,
                        "site2_usable": site2_usable,
                        "bp1_spt_usable": bp1_spt_usable,
                        "paired_tracks_usable": paired_tracks,
                        "paired_frames_usable": paired_frames,
                        "bp1_global_usable": bp1_global_usable,
                        "bp1_locus_usable": bp1_locus,
                        "bp1_continuous_intensity_usable": bp1_intensity,
                        "bp1_component_association_observed": bp1_component,
                        "availability_class": classify_availability(
                            site1=site1_usable,
                            site2=site2_usable,
                            paired_tracks=paired_tracks,
                            paired_frames=paired_frames,
                            bp1_global=bp1_global_usable,
                            bp1_locus=bp1_locus,
                            bp1_spt=bp1_spt_usable,
                        ),
                        "unavailable_reason": _failure_reason(row.task_status, result_dir),
                        "crop_tif_source_relative_path": _relative(
                            _localize(row.crop_tif, source_root), source_root
                        ),
                        "metadata_source_relative_path": _relative(metadata_path, source_root),
                        "result_source_relative_path": _relative(result_dir, source_root),
                        "bp1_labels_source_relative_path": (
                            _relative(bp_paths["bp1_focus_labels.tif"], source_root)
                            if bp_paths["bp1_focus_labels.tif"].exists()
                            else ""
                        ),
                    }
                )

            for table_name in PARTITIONED_TABLES:
                if not config["materialize"].get(table_name, True):
                    continue
                rows, _ = _write_partition(
                    partition_frames[table_name],
                    staging,
                    table_name,
                    ordinal,
                    nd2_id,
                    compression,
                )
                table_rows[table_name] += rows

        crops = pd.DataFrame(crop_records).sort_values(["nd2_id", "crop_index"])
        trajectories = pd.DataFrame(trajectory_records).sort_values(
            ["nd2_id", "crop_id", "allele_index", "site_id"]
        )
        alleles = pd.DataFrame(allele_records).sort_values(["nd2_id", "crop_id", "allele_index"])
        acquisitions = pd.DataFrame(acquisition_records).sort_values(["cohort", "nd2_id"])

        crop_counts = crops.groupby("nd2_id").size().rename("candidate_crops")
        for flag in (
            "site1_usable",
            "site2_usable",
            "paired_frames_usable",
            "bp1_global_usable",
            "bp1_locus_usable",
        ):
            count = crops.loc[crops[flag]].groupby("nd2_id").size().rename(f"crops_{flag}")
            acquisitions = acquisitions.merge(count, on="nd2_id", how="left")
        acquisitions = acquisitions.merge(crop_counts, on="nd2_id", how="left")
        count_columns = [column for column in acquisitions if column.startswith("crops_")]
        acquisitions[count_columns + ["candidate_crops"]] = (
            acquisitions[count_columns + ["candidate_crops"]].fillna(0).astype(int)
        )

        unavailable = crops.loc[
            ~(crops["site1_usable"] | crops["site2_usable"] | crops["bp1_global_usable"])
        ].copy()
        index_frames = {
            "acquisition_index": acquisitions,
            "crop_index": crops,
            "trajectory_index": trajectories,
            "allele_index": alleles,
            "unavailable_crops": unavailable,
        }
        for name, frame in index_frames.items():
            frame = _parquet_safe(frame)
            index_frames[name] = frame
            parquet_path = staging / "tables" / f"{name}.parquet"
            csv_path = staging / "tables" / f"{name}.csv"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(parquet_path, index=False, compression=compression)
            frame.to_csv(csv_path, index=False)
            table_rows[name] = len(frame)

        availability = _availability_summary(crops, alleles)
        summary = {
            "source_crops": len(crops),
            "source_acquisitions": len(acquisitions),
            "task_status_counts": {
                str(key): int(value) for key, value in crops["task_status"].value_counts().items()
            },
            "availability": availability,
            "trajectory_rows_by_site": {
                str(key): int(value)
                for key, value in trajectories.loc[trajectories["usable"], "site_id"]
                .value_counts()
                .items()
            },
            "trajectory_points_by_site": {
                str(key): int(value)
                for key, value in trajectories.loc[trajectories["usable"]]
                .groupby("site_id")["points"]
                .sum()
                .items()
            },
            "table_rows": {str(key): int(value) for key, value in sorted(table_rows.items())},
        }
        (staging / "source_identity.json").write_text(
            json.dumps(source_identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        registry: dict[str, Any] = {}
        for name in INDEX_TABLES:
            path = staging / "tables" / f"{name}.parquet"
            frame = index_frames[name]
            registry[name] = {
                "layout": "single_parquet_plus_csv",
                "relative_path": f"tables/{name}.parquet",
                "rows": len(frame),
                "columns": list(frame.columns),
            }
        for name in PARTITIONED_TABLES:
            path = staging / "tables" / name
            files = sorted(path.glob("*.parquet")) if path.exists() else []
            columns: list[str] = []
            if files:
                columns = list(pd.read_parquet(files[0]).columns)
            registry[name] = {
                "layout": "parquet_dataset_partitioned_by_acquisition_file",
                "relative_path": f"tables/{name}",
                "rows": int(table_rows[name]),
                "partitions": len(files),
                "columns": columns,
            }
        (staging / "table_registry.json").write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _write_documentation(staging, summary, registry, source_root)

        file_records = []
        for path in sorted(value for value in staging.rglob("*") if value.is_file()):
            file_records.append(
                {
                    "relative_path": path.relative_to(staging).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "built_at": datetime.now(UTC).isoformat(),
            "identity_sha256": identity,
            "source_root": str(source_root),
            "source_commit": repository_contract.get("commit"),
            "source_version": repository_contract.get("version"),
            "source_read_only": True,
            "coordinates": "corrected 2D; x/y stored in nm and um; no interpolation",
            "micro_time": "exact crop metadata relative_time_s; production uniform time retained",
            "summary": summary,
            "files": file_records,
        }
        (staging / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(output)
        return IntakeBuildResult(output, summary)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_documentation(
    root: Path,
    summary: dict[str, Any],
    registry: dict[str, Any],
    source_root: Path,
) -> None:
    availability = summary["availability"]
    readme = f"""# Frozen v5.2.1 analysis intake

This is a compact, analysis-derived view of the read-only production run at
`{source_root}`. It does not copy or modify ND2/TIFF production assets.

## What is usable

- Site1 trajectories: {availability["crops_site1_usable"]} crops / {availability["alleles_site1_usable"]} alleles
- Site2 trajectories: {availability["crops_site2_usable"]} crops / {availability["alleles_site2_usable"]} alleles
- Both tracks in one allele: {availability["crops_paired_tracks_usable"]} crops / {availability["alleles_paired_tracks_usable"]} alleles
- At least one shared Site1/Site2 frame: {availability["crops_paired_frames_usable"]} crops / {availability["alleles_paired_frames_usable"]} alleles
- Global 53BP1 segmentation/object dynamics: {availability["crops_bp1_global_usable"]} crops
- Site2-linked 53BP1 metrics: {availability["crops_bp1_locus_usable"]} crops / {availability["alleles_bp1_locus_usable"]} alleles

These are availability layers, not QC tiers. Apply T1-T4 or other scientific QC after loading.

## Core rules

- `COMPLETE` is not assumed to mean all channels are present.
- Site1 and Site2 are retained independently.
- Pair analyses require both tracks in the same allele; frame geometry additionally requires shared frames.
- Exact `time_s` comes from each crop metadata sidecar. `uniform_time_s` only records the production interval convention.
- Coordinates are corrected 2D coordinates in nm and um. No missing position is interpolated.
- Global 53BP1 objects and Site2-linked 53BP1 measurements are separate assets.
- Acquisition is a technical grouping variable, not a biological replicate.

See `QUICKSTART.md`, `table_registry.json`, and `build_manifest.json`.
"""
    quickstart = """# Quick start

```python
from pathlib import Path
from dsb_states.v521_intake import load_intake_table

snapshot = Path(r"PATH_TO_THIS_SNAPSHOT")

# Small indexes
crops = load_intake_table(snapshot, "crop_index")
alleles = load_intake_table(snapshot, "allele_index")

# Site trajectories; filter pushdown avoids loading unrelated rows.
site1 = load_intake_table(
    snapshot, "trajectory_points", filters=[("site_id", "=", "site1")]
)
site2_10h = load_intake_table(
    snapshot,
    "trajectory_points",
    filters=[("site_id", "=", "site2"), ("hour_post_delivery", "=", 10.0)],
)

# Dense Site2-linked 53BP1 and paired-site frame measurements.
allele_frames = load_intake_table(
    snapshot,
    "bp1_allele_frames",
    columns=[
        "bundle_id", "frame", "time_s", "site1_valid", "site2_valid",
        "site1_x_nm", "site1_y_nm", "site2_x_nm", "site2_y_nm",
        "site2_bp1_continuous_intensity_score", "assignment_status",
        "assigned_focus_track_id", "site1_site2_separation_nm",
    ],
)

# Global 53BP1 component dynamics, independent of Site2 availability.
objects = load_intake_table(
    snapshot,
    "bp1_focus_objects",
    columns=["nd2_id", "crop_id", "frame", "time_s", "focus_track_id", "area_um2"],
)
```

Do not infer repair, break completion, biological replication, or missing-as-zero from these tables.
"""
    dictionary_lines = ["# Table dictionary", ""]
    for name, item in registry.items():
        dictionary_lines.extend(
            [
                f"## `{name}`",
                "",
                f"- Layout: `{item['layout']}`",
                f"- Rows: {item['rows']}",
                f"- Path: `{item['relative_path']}`",
                f"- Columns: {', '.join(f'`{value}`' for value in item['columns'])}",
                "",
            ]
        )
    (root / "README.md").write_text(readme, encoding="utf-8")
    (root / "QUICKSTART.md").write_text(quickstart, encoding="utf-8")
    (root / "DATA_DICTIONARY.md").write_text("\n".join(dictionary_lines) + "\n", encoding="utf-8")


def load_intake_table(
    snapshot_root: Path | str,
    table_name: str,
    *,
    columns: list[str] | None = None,
    filters: Any = None,
) -> pd.DataFrame:
    """Load an index or partitioned table with optional Arrow filter pushdown."""

    root = Path(snapshot_root)
    registry = _json(root / "table_registry.json")
    if table_name not in registry:
        raise KeyError(f"Unknown intake table: {table_name}")
    path = root / registry[table_name]["relative_path"]
    if path.is_file():
        return pd.read_parquet(path, columns=columns, filters=filters)
    dataset = ds.dataset(path, format="parquet")
    expression = None
    if filters:
        for column, operator, value in filters:
            field = ds.field(column)
            condition = {
                "=": field == value,
                "==": field == value,
                "!=": field != value,
                ">": field > value,
                ">=": field >= value,
                "<": field < value,
                "<=": field <= value,
            }.get(operator)
            if condition is None:
                raise ValueError(f"Unsupported filter operator: {operator}")
            expression = condition if expression is None else expression & condition
    return dataset.to_table(columns=columns, filter=expression).to_pandas()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/v521_intake.yaml"))
    args = parser.parse_args(argv)
    result = build_intake(args.config)
    print(json.dumps({"root": str(result.root), "summary": result.summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
