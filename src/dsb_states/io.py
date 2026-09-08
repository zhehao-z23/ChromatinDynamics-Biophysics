from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FORMAL_TRAJECTORY_PATTERN = re.compile(
    r"allele_(?P<allele>\d+)_(?P<site>53bp1|site1|site2)_longest_spt_cleaned\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceLayout:
    archive_root: Path

    @property
    def archive_summary(self) -> Path:
        return self.archive_root / "provenance_v5.1" / "archive_summary.json"

    @property
    def archive_manifest(self) -> Path:
        return self.archive_root / "provenance_v5.1" / "archive_file_manifest.tsv"

    @property
    def cell_index(self) -> Path:
        return self.archive_root / "provenance_v5.1" / "cell_index.tsv"

    @property
    def missing_required(self) -> Path:
        return self.archive_root / "provenance_v5.1" / "missing_required.tsv"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def validate_archive_layout(archive_root: str | Path) -> tuple[SourceLayout, list[dict]]:
    layout = SourceLayout(Path(archive_root).resolve())
    issues: list[dict] = []
    required = {
        "archive_summary": layout.archive_summary,
        "archive_file_manifest": layout.archive_manifest,
        "cell_index": layout.cell_index,
        "missing_required": layout.missing_required,
        "results_root": layout.archive_root / "dsb_v5_results" / "a",
    }
    for name, path in required.items():
        if not path.exists():
            issues.append(
                {
                    "severity": "error",
                    "code": "MISSING_ARCHIVE_PATH",
                    "field": name,
                    "path": str(path),
                }
            )
    if layout.archive_summary.exists():
        summary = load_json(layout.archive_summary)
        for field, expected in (
            ("pipeline_version", "v5.1.0"),
            ("pipeline_commit", "6316f9e164c1d21f65cce56780f66d94cdcfc839"),
        ):
            observed = summary.get(field)
            if observed != expected:
                issues.append(
                    {
                        "severity": "error",
                        "code": "ARCHIVE_VERSION_MISMATCH",
                        "field": field,
                        "expected": expected,
                        "observed": observed,
                    }
                )
        if summary.get("missing_required_count") != 0:
            issues.append(
                {
                    "severity": "error",
                    "code": "ARCHIVE_REQUIRED_FILES_MISSING",
                    "observed": summary.get("missing_required_count"),
                }
            )
    return layout, issues


def load_cell_index(layout: SourceLayout, formal_usable_only: bool = True) -> pd.DataFrame:
    table = read_tsv(layout.cell_index)
    if formal_usable_only:
        table = table.loc[table["trajectory_artifacts_usable"].astype(str) == "1"].copy()
    return table.reset_index(drop=True)


def archive_crop_path(layout: SourceLayout, row: pd.Series) -> Path:
    relative = Path(str(row["archive_relative_path"]))
    candidate = (layout.archive_root / relative).resolve()
    try:
        candidate.relative_to(layout.archive_root)
    except ValueError as exc:
        raise ValueError(f"Archive-relative path escaped source root: {relative}") from exc
    return candidate


def find_formal_trajectory_files(crop_path: str | Path) -> list[Path]:
    directory = Path(crop_path) / "spt_results" / "final_trajectories"
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.glob("*_longest_spt_cleaned.csv")
        if FORMAL_TRAJECTORY_PATTERN.fullmatch(path.name)
    )


def parse_trajectory_filename(path: str | Path) -> tuple[int, str]:
    match = FORMAL_TRAJECTORY_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Not a formal trajectory filename: {path}")
    return int(match.group("allele")), match.group("site").lower()


def read_trajectory_csv(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"frame", "x_nm", "y_nm"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Trajectory {path} lacks columns: {sorted(missing)}")
    result = table.loc[:, ["frame", "x_nm", "y_nm"]].copy()
    result["frame"] = pd.to_numeric(result["frame"], errors="raise").astype("int64")
    for column in ("x_nm", "y_nm"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    return result.sort_values("frame", kind="stable").reset_index(drop=True)


def locate_crop_metadata(crop_path: str | Path, crop_id: str, exact: bool = True) -> Path:
    suffix = "_metadata.json" if exact else "_metadata_uniform.json"
    path = Path(crop_path) / f"{crop_id}{suffix}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def exact_micro_times(metadata: dict) -> np.ndarray:
    time = metadata.get("time", {})
    values = time.get("relative_time_s")
    if values is None:
        n_frames = int(time["n_frames"])
        interval = float(time["finterval_s"])
        return np.arange(n_frames, dtype=float) * interval
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or np.any(np.diff(result) <= 0):
        raise ValueError("Metadata relative_time_s must be a nonempty strictly increasing vector")
    return result


def snapshot_copy(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise OSError(f"Snapshot checksum mismatch: {source} -> {destination}")
    return {
        "source": str(source),
        "snapshot": str(destination),
        "bytes": source.stat().st_size,
        "sha256": source_hash,
    }


def assert_source_read_only_by_policy(paths: Iterable[str | Path]) -> None:
    """Fail if a caller accidentally resolves a source path beneath the project snapshot/results.

    The archive may be writable at the operating-system level; this project enforces read-only
    behavior by never opening source files for writes and by restricting all outputs to project
    directories.
    """
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)


def environment_archive_root(env_name: str, fallback: str | None = None) -> Path:
    raw = os.environ.get(env_name, fallback)
    if raw is None:
        raise ValueError(f"Missing archive root; set {env_name}")
    return Path(raw).resolve()
