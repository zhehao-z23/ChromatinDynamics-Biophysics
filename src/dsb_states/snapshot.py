"""Create a read-only, trajectory-only snapshot of the v5.1 DSB archive.

Only production ``longest_spt_cleaned`` trajectories from crops explicitly
marked usable in ``cell_index.tsv`` are copied.  Review trajectories are never
consulted.  Exact and production-uniform metadata sidecars are retained side
by side so downstream tables can use the exact time axis without losing the
timing provenance of the extraction run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import load_authoritative_m0_metadata
from .io import (
    SourceLayout,
    archive_crop_path,
    exact_micro_times,
    find_formal_trajectory_files,
    load_cell_index,
    load_json,
    locate_crop_metadata,
    parse_trajectory_filename,
    read_trajectory_csv,
    read_tsv,
    sha256_file,
    snapshot_copy,
    validate_archive_layout,
)

PIPELINE_VERSION = "v5.1.0"
PIPELINE_COMMIT = "6316f9e164c1d21f65cce56780f66d94cdcfc839"
SNAPSHOT_SCHEMA_VERSION = "2.0"
TRUE_TOKENS = {"1", "true", "yes", "y"}
SNAPSHOT_CONTROL_ARTIFACTS = (
    "source_manifest.parquet",
    "source_manifest.csv",
    "cell_index_formal_usable.parquet",
    "bundle_index_formal_usable.parquet",
    "crop_metadata.parquet",
    "snapshot_reconciliation.json",
    "snapshot_reconciliation.csv",
)


class SnapshotError(RuntimeError):
    """Raised when a snapshot would violate the frozen source contract."""


@dataclass(frozen=True)
class SnapshotBuildResult:
    root: Path
    source_manifest_path: Path
    source_manifest_csv_path: Path
    source_hashes_path: Path
    cell_index_path: Path
    bundle_index_path: Path
    crop_metadata_path: Path
    reconciliation_path: Path
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "source_manifest": str(self.source_manifest_path),
            "source_manifest_csv": str(self.source_manifest_csv_path),
            "source_hashes": str(self.source_hashes_path),
            "cell_index": str(self.cell_index_path),
            "bundle_index": str(self.bundle_index_path),
            "crop_metadata": str(self.crop_metadata_path),
            "reconciliation": str(self.reconciliation_path),
            "summary": dict(self.summary),
        }


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().isin(TRUE_TOKENS)


def _project_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config.get("_project_root", Path.cwd()))).resolve()


def _resolve_project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _project_root(config) / path
    return path.resolve()


def default_snapshot_root(config: Mapping[str, Any]) -> Path:
    configured = config.get("project", {}).get("snapshot_root", "data_snapshot")
    identity = snapshot_identity(config)
    return _resolve_project_path(config, configured) / f"v5_1_formal_{identity[:12]}"


def _enforce_output_scope(config: Mapping[str, Any], output: Path, archive_root: Path) -> None:
    configured = config.get("project", {}).get("snapshot_root", "data_snapshot")
    allowed = _resolve_project_path(config, configured)
    canonical = (_project_root(config) / "data_snapshot").resolve()
    try:
        allowed.relative_to(canonical)
    except ValueError as exc:
        raise SnapshotError(
            f"Configured snapshot root must remain beneath {canonical}: {allowed}"
        ) from exc
    try:
        output.resolve().relative_to(allowed)
    except ValueError as exc:
        raise SnapshotError(f"Snapshot output must remain beneath {allowed}: {output}") from exc

    source = archive_root.resolve()
    destination = output.resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise SnapshotError("Snapshot output and source archive must not overlap")


def _load_bundle_index(
    config: Mapping[str, Any], cell_index: pd.DataFrame
) -> tuple[pd.DataFrame, Path]:
    raw = config.get("data", {}).get("bundle_inventory")
    if not raw:
        raise SnapshotError(
            "data.bundle_inventory is required to preserve authoritative bundle IDs"
        )
    path = _resolve_project_path(config, raw)
    if not path.exists():
        raise SnapshotError(f"Bundle inventory does not exist: {path}")
    bundles = read_tsv(path)
    required = {
        "bundle_id",
        "nd2_id",
        "crop_id",
        "allele_index",
        "anchor_locus",
        "total_frames",
        "trajectory_artifacts_usable",
    }
    missing = required.difference(bundles.columns)
    if missing:
        raise SnapshotError(f"Bundle inventory lacks columns: {sorted(missing)}")
    bundles = bundles.loc[_truthy(bundles["trajectory_artifacts_usable"])].copy()
    usable_keys = cell_index[["nd2_id", "crop_id"]].drop_duplicates()
    bundles = bundles.merge(
        usable_keys, on=["nd2_id", "crop_id"], how="inner", validate="many_to_one"
    )
    if bundles["bundle_id"].isna().any() or bundles["bundle_id"].duplicated().any():
        raise SnapshotError("Authoritative usable bundle IDs must be nonmissing and unique")
    key = ["nd2_id", "crop_id", "allele_index"]
    if bundles.duplicated(key).any():
        raise SnapshotError(f"Bundle inventory is not unique by {key}")
    return bundles.reset_index(drop=True), path


def _load_hour_map(config: Mapping[str, Any]) -> tuple[pd.DataFrame, Path | None]:
    """Load only explicit mappings from the frozen ND2 inventory.

    No filename/folder parsing is performed here.  Missing inventory rows stay
    missing in the snapshot and are reported as unmapped.
    """

    raw = config.get("data", {}).get("nd2_inventory")
    if not raw:
        return pd.DataFrame(columns=["nd2_id", "hour_post_delivery", "hour_mapping_source"]), None
    path = _resolve_project_path(config, raw)
    if not path.exists():
        return pd.DataFrame(columns=["nd2_id", "hour_post_delivery", "hour_mapping_source"]), path
    inventory = read_tsv(path)
    required = {"file_name", "timepoint_h", "timepoint_source"}
    missing = required.difference(inventory.columns)
    if missing:
        raise SnapshotError(f"ND2 inventory lacks hour-mapping columns: {sorted(missing)}")
    if "retained_after_dedup" in inventory:
        inventory = inventory.loc[_truthy(inventory["retained_after_dedup"])].copy()
    mapping = inventory.loc[:, ["file_name", "timepoint_h", "timepoint_source"]].copy()
    mapping["timepoint_h"] = pd.to_numeric(mapping["timepoint_h"], errors="coerce")
    inferred = mapping["timepoint_source"].astype("string").str.lower().str.startswith("inferred:")
    # The master plan forbids reconstructing missing macro-time labels.  Keep
    # the upstream provenance text but do not promote an inferred inventory
    # label into an analysis hour.
    mapping.loc[inferred, "timepoint_h"] = np.nan
    conflicts = (
        mapping.dropna(subset=["timepoint_h"])
        .groupby("file_name", dropna=False)["timepoint_h"]
        .nunique(dropna=True)
    )
    if (conflicts > 1).any():
        raise SnapshotError("Frozen ND2 inventory contains conflicting hour mappings")
    mapping = mapping.drop_duplicates("file_name", keep="first").rename(
        columns={
            "file_name": "nd2_id",
            "timepoint_h": "hour_post_delivery",
            "timepoint_source": "hour_mapping_source",
        }
    )
    return mapping.reset_index(drop=True), path


def _manifest_record(
    copied: Mapping[str, Any],
    *,
    snapshot_root: Path,
    snapshot_relative_root: Path | None = None,
    archive_root: Path,
    record_type: str,
    row: pd.Series,
    allele_index: int | None = None,
    site_id: str | None = None,
    bundle_id: str | None = None,
    localization_rows: int | None = None,
) -> dict[str, Any]:
    source = Path(str(copied["source"]))
    snapshot = Path(str(copied["snapshot"]))
    relative_root = snapshot_relative_root or snapshot_root
    return {
        "record_type": record_type,
        "source_path": str(source),
        "source_archive_relative_path": source.relative_to(archive_root).as_posix(),
        "snapshot_relative_path": snapshot.relative_to(relative_root).as_posix(),
        "bytes": int(copied["bytes"]),
        "sha256": copied["sha256"],
        "dsb_id": row.get("dsb_id"),
        "nd2_id": row["nd2_id"],
        "fov_id": row["fov"],
        "cell_id": row["crop_id"],
        "allele_index": allele_index,
        "site_id": site_id,
        "bundle_id": bundle_id,
        "localization_rows": localization_rows,
        "formal_usable": True,
        "review_included": False,
    }


def _source_index_hash(path: Path | None, role: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"role": role, "path": None if path is None else str(path), "status": "missing"}
    return {
        "role": role,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "status": "hashed",
    }


def _configured_snapshot_source_indexes(
    config: Mapping[str, Any], layout: SourceLayout
) -> list[dict[str, Any]]:
    bundle_value = config.get("data", {}).get("bundle_inventory")
    bundle_path = _resolve_project_path(config, bundle_value) if bundle_value else None
    m0_value = config.get("hierarchy", {}).get("m0_metadata_path")
    if m0_value is not None:
        hour_path = _resolve_project_path(config, m0_value)
        hour_role = "authoritative_m0_metadata"
    else:
        hour_value = config.get("data", {}).get("nd2_inventory")
        hour_path = _resolve_project_path(config, hour_value) if hour_value else None
        hour_role = "nd2_inventory"
    return [
        _source_index_hash(layout.archive_manifest, "archive_manifest"),
        _source_index_hash(layout.archive_summary, "archive_summary"),
        _source_index_hash(layout.cell_index, "cell_index"),
        _source_index_hash(bundle_path, "bundle_inventory"),
        _source_index_hash(hour_path, hour_role),
    ]


def snapshot_identity(config: Mapping[str, Any]) -> str:
    """Return a content-bound identity for the inputs that determine a snapshot.

    The identity changes when an authoritative archive index, cohort mapping,
    selection policy, or production pipeline version changes.  Consequently a
    new input contract gets a new directory instead of silently rewriting an
    earlier run's reusable snapshot.
    """

    archive_value = config.get("data", {}).get("archive_root_resolved")
    if not archive_value:
        raise SnapshotError("Cannot derive snapshot identity before archive root resolution")
    layout = SourceLayout(Path(str(archive_value)).resolve())
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_commit": PIPELINE_COMMIT,
        "selection": {
            "formal_usable_only": config.get("data", {}).get("formal_usable_only"),
            "review_data_policy": config.get("data", {}).get("review_data_policy"),
            "exact_relative_time_primary": config.get("data", {}).get("use_exact_timestamps"),
        },
        "source_indexes": _configured_snapshot_source_indexes(config, layout),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _snapshot_staging_root(final_root: Path) -> Path:
    """Return a same-parent staging path suitable for atomic directory rename."""

    return final_root.parent / f".{final_root.name}.building-{os.getpid()}-{uuid.uuid4().hex}"


def snapshot_control_artifact_hashes(snapshot_root: str | Path) -> list[dict[str, Any]]:
    """Hash every snapshot-side control artifact consumed by table construction."""

    root = Path(snapshot_root).resolve()
    records: list[dict[str, Any]] = []
    for relative in SNAPSHOT_CONTROL_ARTIFACTS:
        path = root / relative
        if not path.is_file():
            raise SnapshotError(f"Snapshot control artifact is missing: {path}")
        records.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return records


def _normalized_source_index(record: Mapping[str, Any]) -> dict[str, Any]:
    path_value = record.get("path")
    normalized_path = None if path_value is None else str(Path(str(path_value)).resolve())
    return {
        "role": record.get("role"),
        "path": normalized_path,
        "bytes": record.get("bytes"),
        "sha256": record.get("sha256"),
        "status": record.get("status"),
    }


def _verify_existing_snapshot(
    root: Path,
    *,
    layout: SourceLayout,
    expected_source_indexes: list[dict[str, Any]],
    expected_identity: str,
) -> SnapshotBuildResult:
    """Verify an existing immutable snapshot without modifying any file."""

    required = (*SNAPSHOT_CONTROL_ARTIFACTS, "source_hashes.json")
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise SnapshotError(
            "Existing snapshot is incomplete and will not be overwritten; "
            f"missing={missing}, root={root}"
        )

    source_hashes_path = root / "source_hashes.json"
    try:
        source_payload = json.loads(source_hashes_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SnapshotError(
            f"Existing snapshot provenance is unreadable: {source_hashes_path}"
        ) from exc

    if source_payload.get("pipeline_version") != PIPELINE_VERSION:
        raise SnapshotError("Existing snapshot pipeline version differs from the active contract")
    if source_payload.get("pipeline_commit") != PIPELINE_COMMIT:
        raise SnapshotError("Existing snapshot pipeline commit differs from the active contract")
    expected_selection = {
        "formal_usable_only": True,
        "review_data_policy": "exclude",
        "exact_relative_time_primary": True,
    }
    if source_payload.get("selection") != expected_selection:
        raise SnapshotError("Existing snapshot selection policy differs from the active contract")
    recorded_identity = source_payload.get("snapshot_identity")
    if recorded_identity is not None and recorded_identity != expected_identity:
        raise SnapshotError("Existing snapshot identity differs from the current source contract")

    recorded_indexes = {
        str(record.get("role")): _normalized_source_index(record)
        for record in source_payload.get("source_indexes", [])
        if isinstance(record, Mapping)
    }
    expected_indexes = {
        str(record.get("role")): _normalized_source_index(record)
        for record in expected_source_indexes
    }
    schema_version = str(source_payload.get("schema_version", "legacy"))
    if schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotError(
            "Existing snapshot predates immutable provenance schema 2.0 and will not be "
            "modified or reused; choose the content-addressed default snapshot path"
        )
    required_roles = set(expected_indexes)
    if set(recorded_indexes) != required_roles:
        raise SnapshotError(
            "Existing snapshot source-index roles differ from the current contract; "
            f"recorded={sorted(recorded_indexes)}, expected={sorted(required_roles)}"
        )
    for role in sorted(required_roles):
        if recorded_indexes[role] != expected_indexes[role]:
            raise SnapshotError(f"Existing snapshot source index changed: {role}")
    archive_manifest = expected_indexes["archive_manifest"]
    if archive_manifest.get("status") != "hashed":
        raise SnapshotError("Archive manifest must exist and be SHA-256 bound before reuse")

    try:
        manifest = pd.read_parquet(root / "source_manifest.parquet")
    except Exception as exc:  # pragma: no cover - backend-specific parse failures
        raise SnapshotError("Existing snapshot manifest is unreadable") from exc
    copied_columns = [
        "record_type",
        "source_path",
        "snapshot_relative_path",
        "bytes",
        "sha256",
    ]
    if not set(copied_columns).issubset(manifest.columns):
        raise SnapshotError("Existing snapshot manifest lacks provenance columns")
    recorded_copies = source_payload.get("copied_files")
    if not isinstance(recorded_copies, list) or not recorded_copies:
        raise SnapshotError("Existing snapshot copied-file hash ledger is empty")
    manifest_copies = manifest[copied_columns].to_dict(orient="records")
    if manifest_copies != recorded_copies:
        raise SnapshotError("Existing snapshot manifest disagrees with its copied-file ledger")

    expected_snapshot_files: set[Path] = set()
    archive_root = layout.archive_root.resolve()
    for record in recorded_copies:
        relative = Path(str(record["snapshot_relative_path"]))
        snapshot_path = (root / relative).resolve()
        source_path = Path(str(record["source_path"])).resolve()
        try:
            snapshot_path.relative_to(root)
            source_path.relative_to(archive_root)
        except ValueError as exc:
            raise SnapshotError("Snapshot ledger contains an out-of-scope path") from exc
        expected_snapshot_files.add(snapshot_path)
        expected_bytes = int(record["bytes"])
        expected_hash = str(record["sha256"])
        for role, path in (("source", source_path), ("snapshot", snapshot_path)):
            if not path.is_file():
                raise SnapshotError(f"Existing snapshot {role} file is missing: {path}")
            if path.stat().st_size != expected_bytes or sha256_file(path) != expected_hash:
                raise SnapshotError(f"Existing snapshot {role} file hash changed: {path}")

    source_root = root / "source_files"
    actual_snapshot_files = {path.resolve() for path in source_root.rglob("*") if path.is_file()}
    if actual_snapshot_files != expected_snapshot_files:
        raise SnapshotError("Existing snapshot source_files tree has missing or untracked files")

    recorded_artifacts = source_payload.get("snapshot_artifacts")
    actual_artifacts = snapshot_control_artifact_hashes(root)
    if recorded_artifacts != actual_artifacts:
        raise SnapshotError("Existing snapshot control-artifact hashes changed")

    try:
        reconciliation = json.loads(
            (root / "snapshot_reconciliation.json").read_text(encoding="utf-8")
        )
        cells = pd.read_parquet(root / "cell_index_formal_usable.parquet")
        bundles = pd.read_parquet(root / "bundle_index_formal_usable.parquet")
        crop_metadata = pd.read_parquet(root / "crop_metadata.parquet")
        manifest_csv = pd.read_csv(root / "source_manifest.csv", low_memory=False)
    except Exception as exc:  # pragma: no cover - backend-specific parse failures
        raise SnapshotError("Existing snapshot control artifacts are unreadable") from exc
    critical = (
        reconciliation.get("trajectory_file_count_reconciled") is True
        and reconciliation.get("localization_count_reconciled") is True
        and reconciliation.get("review_exclusion_reconciled") is True
        and len(cells) == int(reconciliation.get("formal_usable_crop_count", -1))
        and len(bundles) == int(reconciliation.get("authoritative_bundle_count", -1))
        and len(crop_metadata) == len(cells)
        and len(manifest_csv) == len(manifest)
    )
    if not critical:
        raise SnapshotError("Existing snapshot reconciliation/control-table checks failed")

    summary = dict(reconciliation)
    summary.update(
        {
            "snapshot_access": "verified_reuse",
            "snapshot_schema_version": schema_version,
            "snapshot_identity": expected_identity,
            "archive_manifest": archive_manifest,
            "snapshot_artifacts": actual_artifacts,
        }
    )
    return SnapshotBuildResult(
        root=root,
        source_manifest_path=root / "source_manifest.parquet",
        source_manifest_csv_path=root / "source_manifest.csv",
        source_hashes_path=source_hashes_path,
        cell_index_path=root / "cell_index_formal_usable.parquet",
        bundle_index_path=root / "bundle_index_formal_usable.parquet",
        crop_metadata_path=root / "crop_metadata.parquet",
        reconciliation_path=root / "snapshot_reconciliation.json",
        summary=summary,
    )


def build_trajectory_snapshot(
    config: Mapping[str, Any], snapshot_root: str | Path | None = None
) -> SnapshotBuildResult:
    """Copy the formal usable trajectory cohort into ``data_snapshot``.

    The production archive is opened only for reads.  The returned manifests
    are the sole discovery interface for later table construction.
    """

    data_config = config.get("data", {})
    if data_config.get("formal_usable_only") is not True:
        raise SnapshotError("Snapshot policy requires data.formal_usable_only=true")
    if str(data_config.get("review_data_policy", "")).lower() != "exclude":
        raise SnapshotError("Snapshot policy requires data.review_data_policy=exclude")

    archive_value = data_config.get("archive_root_resolved")
    if not archive_value:
        raise SnapshotError(
            "Load the configuration before snapshotting; archive root is unresolved"
        )
    layout, archive_issues = validate_archive_layout(archive_value)
    blocking = [issue for issue in archive_issues if issue.get("severity") == "error"]
    if blocking:
        raise SnapshotError(f"Archive validation failed: {json.dumps(blocking, default=str)}")

    final_root = Path(snapshot_root).resolve() if snapshot_root else default_snapshot_root(config)
    _enforce_output_scope(config, final_root, layout.archive_root)
    identity = snapshot_identity(config)
    source_indexes = _configured_snapshot_source_indexes(config, layout)
    if final_root.exists():
        return _verify_existing_snapshot(
            final_root,
            layout=layout,
            expected_source_indexes=source_indexes,
            expected_identity=identity,
        )
    root = _snapshot_staging_root(final_root)
    _enforce_output_scope(config, root, layout.archive_root)
    source_root = root / "source_files"
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir(exist_ok=False)
    except FileExistsError:
        raise SnapshotError(f"Unexpected snapshot staging collision: {root}")

    cells = load_cell_index(layout, formal_usable_only=True)
    if "review_scope" in cells:
        cells = cells.loc[~_truthy(cells["review_scope"])].copy()
    if not _truthy(cells["trajectory_artifacts_usable"]).all():
        raise SnapshotError("Non-usable crops survived the formal cohort filter")
    cells = cells.reset_index(drop=True)

    bundles, bundle_source_path = _load_bundle_index(config, cells)
    configured_m0_path = config.get("hierarchy", {}).get("m0_metadata_path")
    if configured_m0_path is not None:
        try:
            m0_metadata, hour_source_path = load_authoritative_m0_metadata(config, cells)
        except ValueError as exc:
            raise SnapshotError(str(exc)) from exc
        m0_metadata = m0_metadata.rename(
            columns={
                "fov_id": "fov",
                "hour_mapping_evidence": "hour_mapping_source",
            }
        )
        cells = cells.merge(
            m0_metadata,
            on=["nd2_id", "fov"],
            how="left",
            validate="many_to_one",
        )
        cells["hour_mapping_status"] = "mapped_from_authoritative_m0_metadata"
        hour_source_role = "authoritative_m0_metadata"
    else:
        hour_map, hour_source_path = _load_hour_map(config)
        cells = cells.merge(hour_map, on="nd2_id", how="left", validate="many_to_one")
        cells["hour_mapping_status"] = np.select(
            [
                cells["hour_post_delivery"].notna(),
                cells["hour_mapping_source"]
                .astype("string")
                .str.lower()
                .str.startswith("inferred:"),
            ],
            ["mapped_from_frozen_inventory", "inferred_inventory_label_not_used"],
            default="unmapped",
        )
        cells["biological_experiment_id"] = pd.Series(pd.NA, index=cells.index, dtype="string")
        hour_source_role = "nd2_inventory"

    bundle_lookup = {
        (str(row.nd2_id), str(row.crop_id), int(row.allele_index)): str(row.bundle_id)
        for row in bundles.itertuples(index=False)
    }
    manifest_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    observed_trajectory_files = 0
    observed_localizations = 0

    use_exact = data_config.get("use_exact_timestamps") is True
    for _, row in cells.iterrows():
        crop_path = archive_crop_path(layout, row)
        exact_source = locate_crop_metadata(crop_path, str(row["crop_id"]), exact=True)
        uniform_source = locate_crop_metadata(crop_path, str(row["crop_id"]), exact=False)
        exact_metadata = load_json(exact_source)
        uniform_metadata = load_json(uniform_source)
        relative = exact_metadata.get("time", {}).get("relative_time_s")
        if use_exact and relative is None:
            raise SnapshotError(f"Exact relative_time_s is missing: {exact_source}")
        exact_times = exact_micro_times(exact_metadata)
        nominal = int(row["total_frames"])
        if len(exact_times) != nominal:
            raise SnapshotError(
                f"Exact timestamp count differs from cell_index total_frames for {row['crop_id']}"
            )
        uniform_dt = float(uniform_metadata.get("time", {}).get("finterval_s", np.nan))
        if not np.isfinite(uniform_dt) or uniform_dt <= 0:
            raise SnapshotError(f"Invalid production uniform interval: {uniform_source}")

        relative_crop = Path(str(row["archive_relative_path"]))
        destination_crop = source_root / relative_crop
        exact_copy = snapshot_copy(exact_source, destination_crop / exact_source.name)
        uniform_copy = snapshot_copy(uniform_source, destination_crop / uniform_source.name)
        manifest_rows.append(
            _manifest_record(
                exact_copy,
                snapshot_root=root,
                snapshot_relative_root=root,
                archive_root=layout.archive_root,
                record_type="exact_metadata",
                row=row,
            )
        )
        manifest_rows.append(
            _manifest_record(
                uniform_copy,
                snapshot_root=root,
                snapshot_relative_root=root,
                archive_root=layout.archive_root,
                record_type="production_uniform_metadata",
                row=row,
            )
        )

        differences = np.diff(exact_times)
        override = uniform_metadata.get("uniform_timing_override", {})
        metadata_rows.append(
            {
                "nd2_id": row["nd2_id"],
                "fov_id": row["fov"],
                "cell_id": row["crop_id"],
                "acquisition_date_observed": exact_metadata.get("acquisition_date"),
                "acquisition_date_semantics": (
                    "instrument timestamp; not a biological experiment or replicate label"
                ),
                "nominal_frame_count": nominal,
                "hour_post_delivery": row["hour_post_delivery"],
                "hour_mapping_source": row.get("hour_mapping_source"),
                "hour_mapping_status": row["hour_mapping_status"],
                "biological_experiment_id": row["biological_experiment_id"],
                "exact_time_source": exact_metadata.get("time", {}).get("finterval_source"),
                "exact_relative_time_s_present": relative is not None,
                "exact_interval_median_s": (
                    float(np.median(differences)) if differences.size else np.nan
                ),
                "exact_interval_min_s": float(np.min(differences)) if differences.size else np.nan,
                "exact_interval_max_s": float(np.max(differences)) if differences.size else np.nan,
                "production_uniform_interval_s": uniform_dt,
                "production_uniform_policy": override.get("policy"),
                "production_original_sidecar_sha256": override.get("original_sidecar_sha256"),
                "exact_metadata_snapshot_relative_path": Path(exact_copy["snapshot"])
                .relative_to(root)
                .as_posix(),
                "uniform_metadata_snapshot_relative_path": Path(uniform_copy["snapshot"])
                .relative_to(root)
                .as_posix(),
                "exact_metadata_sha256": exact_copy["sha256"],
                "uniform_metadata_sha256": uniform_copy["sha256"],
            }
        )

        trajectory_files = find_formal_trajectory_files(crop_path)
        expected_files = int(row["formal_final_trajectory_count"])
        if len(trajectory_files) != expected_files:
            raise SnapshotError(
                f"Formal trajectory count mismatch for {row['crop_id']}: "
                f"cell_index={expected_files}, discovered={len(trajectory_files)}"
            )
        for source in trajectory_files:
            allele_index, site_id = parse_trajectory_filename(source)
            bundle_id = bundle_lookup.get((str(row["nd2_id"]), str(row["crop_id"]), allele_index))
            if bundle_id is None:
                raise SnapshotError(
                    "Trajectory has no authoritative bundle ID: "
                    f"{row['nd2_id']} | {row['crop_id']} | allele {allele_index}"
                )
            trajectory = read_trajectory_csv(source)
            destination = destination_crop / "spt_results" / "final_trajectories" / source.name
            copied = snapshot_copy(source, destination)
            manifest_rows.append(
                _manifest_record(
                    copied,
                    snapshot_root=root,
                    snapshot_relative_root=root,
                    archive_root=layout.archive_root,
                    record_type="formal_final_trajectory",
                    row=row,
                    allele_index=allele_index,
                    site_id=site_id,
                    bundle_id=bundle_id,
                    localization_rows=len(trajectory),
                )
            )
            observed_trajectory_files += 1
            observed_localizations += len(trajectory)

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["record_type", "nd2_id", "cell_id", "allele_index", "site_id"],
        kind="stable",
        na_position="first",
    )
    metadata_table = pd.DataFrame(metadata_rows)

    cell_columns = [
        "task_index",
        "dsb_id",
        "nd2_id",
        "fov",
        "crop_id",
        "total_frames",
        "analysis_scope",
        "trajectory_usability_class",
        "trajectory_artifacts_usable",
        "formal_final_trajectory_count",
        "formal_final_localization_count",
        "archive_relative_path",
        "hour_post_delivery",
        "hour_mapping_source",
        "hour_mapping_status",
        "biological_experiment_id",
        "biological_experiment_evidence",
        "biological_experiment_confirmed",
        "imaging_day",
        "biological_replicate",
    ]
    cells_snapshot = cells.loc[:, [column for column in cell_columns if column in cells]].copy()
    bundle_cells = cells[
        [
            "nd2_id",
            "crop_id",
            "fov",
            "hour_post_delivery",
            "hour_mapping_source",
            "hour_mapping_status",
            "biological_experiment_id",
            *[
                column
                for column in (
                    "biological_experiment_evidence",
                    "biological_experiment_confirmed",
                    "imaging_day",
                    "biological_replicate",
                )
                if column in cells
            ],
            "trajectory_usability_class",
        ]
    ]
    bundles_snapshot = bundles.merge(
        bundle_cells,
        on=["nd2_id", "crop_id"],
        how="left",
        validate="many_to_one",
    )
    if bundles_snapshot["fov"].isna().any():
        raise SnapshotError("One or more bundles could not be mapped back to cell_index.fov")

    source_manifest_path = root / "source_manifest.parquet"
    source_manifest_csv_path = root / "source_manifest.csv"
    cell_index_path = root / "cell_index_formal_usable.parquet"
    bundle_index_path = root / "bundle_index_formal_usable.parquet"
    crop_metadata_path = root / "crop_metadata.parquet"
    manifest.to_parquet(source_manifest_path, index=False)
    manifest.to_csv(source_manifest_csv_path, index=False)
    cells_snapshot.to_parquet(cell_index_path, index=False)
    bundles_snapshot.to_parquet(bundle_index_path, index=False)
    metadata_table.to_parquet(crop_metadata_path, index=False)

    expected_files = int(
        pd.to_numeric(cells["formal_final_trajectory_count"], errors="raise").sum()
    )
    expected_localizations = int(
        pd.to_numeric(cells["formal_final_localization_count"], errors="raise").sum()
    )
    checks = {
        "formal_usable_crop_count": len(cells),
        "authoritative_bundle_count": len(bundles_snapshot),
        "expected_trajectory_files_from_cell_index": expected_files,
        "observed_trajectory_files": int(observed_trajectory_files),
        "expected_localizations_from_cell_index": expected_localizations,
        "observed_localizations": int(observed_localizations),
        "review_files_included": int(manifest["review_included"].fillna(False).sum()),
        "mapped_acquisitions": int(
            cells.loc[cells["hour_post_delivery"].notna(), "nd2_id"].nunique()
        ),
        "unmapped_acquisitions": int(
            cells.loc[cells["hour_post_delivery"].isna(), "nd2_id"].nunique()
        ),
        "observed_acquisition_timestamp_count": int(
            metadata_table["acquisition_date_observed"].nunique(dropna=True)
        ),
        "observed_calendar_date_tokens": sorted(
            metadata_table["acquisition_date_observed"]
            .dropna()
            .astype(str)
            .str.extract(r"^(\S+)", expand=False)
            .dropna()
            .unique()
            .tolist()
        ),
        "acquisition_date_interpretation": (
            "instrument provenance only; not promoted to biological_experiment_id"
        ),
    }
    checks["trajectory_file_count_reconciled"] = expected_files == observed_trajectory_files
    checks["localization_count_reconciled"] = expected_localizations == observed_localizations
    checks["review_exclusion_reconciled"] = checks["review_files_included"] == 0
    reconciliation_path = root / "snapshot_reconciliation.json"
    _write_json(checks, reconciliation_path)
    pd.DataFrame([{"check": key, "value": value} for key, value in checks.items()]).to_csv(
        root / "snapshot_reconciliation.csv", index=False
    )

    built_source_indexes = [
        _source_index_hash(layout.archive_manifest, "archive_manifest"),
        _source_index_hash(layout.archive_summary, "archive_summary"),
        _source_index_hash(layout.cell_index, "cell_index"),
        _source_index_hash(bundle_source_path, "bundle_inventory"),
        _source_index_hash(hour_source_path, hour_source_role),
    ]
    if built_source_indexes != source_indexes:
        raise SnapshotError(
            "Snapshot source indexes changed while the immutable snapshot was built"
        )
    artifact_hashes = snapshot_control_artifact_hashes(root)
    source_hashes = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_identity": identity,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_commit": PIPELINE_COMMIT,
        "selection": {
            "formal_usable_only": True,
            "review_data_policy": "exclude",
            "exact_relative_time_primary": True,
        },
        "source_indexes": built_source_indexes,
        "snapshot_artifacts": artifact_hashes,
        "copied_files": manifest[
            [
                "record_type",
                "source_path",
                "snapshot_relative_path",
                "bytes",
                "sha256",
            ]
        ].to_dict(orient="records"),
    }
    source_hashes_path = root / "source_hashes.json"
    _write_json(source_hashes, source_hashes_path)

    critical = (
        checks["trajectory_file_count_reconciled"]
        and checks["localization_count_reconciled"]
        and checks["review_exclusion_reconciled"]
    )
    if not critical:
        raise SnapshotError(f"Snapshot count reconciliation failed; see {reconciliation_path}")

    checks.update(
        {
            "snapshot_access": "created",
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_identity": identity,
            "archive_manifest": next(
                record for record in built_source_indexes if record["role"] == "archive_manifest"
            ),
            "snapshot_artifacts": artifact_hashes,
        }
    )

    try:
        root.rename(final_root)
    except FileExistsError:
        shutil.rmtree(root)
        return _verify_existing_snapshot(
            final_root,
            layout=layout,
            expected_source_indexes=source_indexes,
            expected_identity=identity,
        )
    return SnapshotBuildResult(
        root=final_root,
        source_manifest_path=final_root / source_manifest_path.relative_to(root),
        source_manifest_csv_path=final_root / source_manifest_csv_path.relative_to(root),
        source_hashes_path=final_root / source_hashes_path.relative_to(root),
        cell_index_path=final_root / cell_index_path.relative_to(root),
        bundle_index_path=final_root / bundle_index_path.relative_to(root),
        crop_metadata_path=final_root / crop_metadata_path.relative_to(root),
        reconciliation_path=final_root / reconciliation_path.relative_to(root),
        summary=checks,
    )


# Discoverable alias for callers that use "create" rather than "build".
create_trajectory_snapshot = build_trajectory_snapshot
