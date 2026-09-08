"""Build a flat, no-analysis-QC cache for the frozen v5.2.1 intake.

The cache has one primary table with one row per observed/assayed bundle-frame.
Site1, Site2 and the trajectory-extracted 53BP1 position are stored side by
side.  The locus-linked 53BP1 measurements are left in their native columns.
Rows are never removed because of track length, pairing, coverage, motion,
separation, 53BP1 association, hour or acquisition.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class UnfilteredCacheError(ValueError):
    """Raised when source tables cannot be reconciled losslessly."""


KEY = ["bundle_id", "frame"]
IDENTITY = [
    "cohort",
    "nd2_id",
    "crop_id",
    "fov_id",
    "hour_post_delivery",
    "allele_index",
]
SITES = ("site1", "site2", "53bp1_spt")


@dataclass(frozen=True)
class UnfilteredCacheResult:
    """Flat frame cache plus lightweight entity indexes."""

    bundle_frames: pd.DataFrame
    bundle_index: pd.DataFrame
    crop_availability: pd.DataFrame
    census: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_unique(table: pd.DataFrame, key: list[str], name: str) -> None:
    if table.duplicated(key).any():
        examples = table.loc[table.duplicated(key, keep=False), key].head().to_dict("records")
        raise UnfilteredCacheError(f"{name} is not unique by {key}: {examples}")


def _numeric_agreement(left: pd.Series, right: pd.Series, name: str) -> None:
    both = left.notna() & right.notna()
    if both.any() and not np.allclose(
        pd.to_numeric(left[both], errors="raise"),
        pd.to_numeric(right[both], errors="raise"),
        rtol=0,
        atol=1e-9,
    ):
        raise UnfilteredCacheError(f"conflicting values for {name}")


def _text_agreement(left: pd.Series, right: pd.Series, name: str) -> None:
    both = left.notna() & right.notna()
    if both.any() and not left[both].astype(str).equals(right[both].astype(str)):
        raise UnfilteredCacheError(f"conflicting values for {name}")


def _observed_wide(trajectory_points: pd.DataFrame) -> pd.DataFrame:
    required = set(KEY + IDENTITY + ["site_id", "x_nm", "y_nm", "time_s", "uniform_time_s"])
    missing = sorted(required.difference(trajectory_points.columns))
    if missing:
        raise UnfilteredCacheError(f"trajectory_points missing columns: {missing}")
    if trajectory_points.empty:
        raise UnfilteredCacheError("trajectory_points is empty")
    unexpected = sorted(set(trajectory_points.site_id.dropna().astype(str)).difference(SITES))
    if unexpected:
        raise UnfilteredCacheError(f"unexpected trajectory site IDs: {unexpected}")
    _require_unique(trajectory_points, KEY + ["site_id"], "trajectory_points")

    merged: pd.DataFrame | None = None
    for site in SITES:
        part = trajectory_points.loc[trajectory_points.site_id.eq(site)].copy()
        rename = {
            "x_nm": f"{site}_x_nm_observed",
            "y_nm": f"{site}_y_nm_observed",
            "time_s": f"{site}_time_s_observed",
            "uniform_time_s": f"{site}_uniform_time_s_observed",
        }
        part = part[KEY + IDENTITY + list(rename)].rename(columns=rename)
        part[f"{site}_position_observed"] = True
        if merged is None:
            merged = part
            continue
        merged = merged.merge(part, on=KEY, how="outer", suffixes=("", f"_{site}"), validate="1:1")
        for column in IDENTITY:
            candidate = f"{column}_{site}"
            if candidate not in merged:
                continue
            if column == "hour_post_delivery" or column == "allele_index":
                _numeric_agreement(merged[column], merged[candidate], column)
            else:
                _text_agreement(merged[column], merged[candidate], column)
            merged[column] = merged[column].combine_first(merged[candidate])
            merged = merged.drop(columns=candidate)
    assert merged is not None
    for site in SITES:
        observed = f"{site}_position_observed"
        if observed not in merged:
            merged[observed] = False
        merged[observed] = merged[observed].fillna(False).astype(bool)
    return merged


def build_unfiltered_cache(
    trajectory_points: pd.DataFrame,
    bp1_allele_frames: pd.DataFrame,
    allele_index: pd.DataFrame,
    crop_index: pd.DataFrame,
) -> UnfilteredCacheResult:
    """Create one losslessly reconciled bundle-frame table without analysis QC."""

    observed = _observed_wide(trajectory_points)
    required_locus = set(KEY + IDENTITY + ["time_s", "site1_valid", "site2_valid"])
    missing_locus = sorted(required_locus.difference(bp1_allele_frames.columns))
    if missing_locus:
        raise UnfilteredCacheError(f"bp1_allele_frames missing columns: {missing_locus}")
    _require_unique(bp1_allele_frames, KEY, "bp1_allele_frames")

    locus = bp1_allele_frames.copy()
    locus["locus_bp1_frame_available"] = True
    master = locus.merge(observed, on=KEY, how="outer", suffixes=("", "_trajectory"), validate="1:1")

    for column in IDENTITY:
        candidate = f"{column}_trajectory"
        if candidate not in master:
            continue
        if column in {"hour_post_delivery", "allele_index"}:
            _numeric_agreement(master[column], master[candidate], column)
        else:
            _text_agreement(master[column], master[candidate], column)
        master[column] = master[column].combine_first(master[candidate])
        master = master.drop(columns=candidate)

    observed_times = [f"{site}_time_s_observed" for site in SITES]
    for left_index, left in enumerate(observed_times):
        for right in observed_times[left_index + 1 :]:
            _numeric_agreement(master[left], master[right], "trajectory time_s")
    trajectory_time = master[observed_times].bfill(axis=1).iloc[:, 0]
    _numeric_agreement(master["time_s"], trajectory_time, "locus/trajectory time_s")
    master["time_s"] = master["time_s"].combine_first(trajectory_time)

    for site in ("site1", "site2"):
        observed_flag = master[f"{site}_position_observed"].fillna(False).astype(bool)
        locus_flag = master[f"{site}_valid"].fillna(False).astype(bool)
        locus_x = master[f"{site}_x_nm"]
        locus_y = master[f"{site}_y_nm"]
        observed_x = master[f"{site}_x_nm_observed"]
        observed_y = master[f"{site}_y_nm_observed"]
        _numeric_agreement(locus_x, observed_x, f"{site}_x_nm")
        _numeric_agreement(locus_y, observed_y, f"{site}_y_nm")
        if not observed_flag.equals(locus_flag | observed_flag):
            # Any locus-valid coordinate must be represented in trajectory_points.
            unexpected = locus_flag & ~observed_flag
            if unexpected.any():
                raise UnfilteredCacheError(f"{site} locus-valid rows missing from trajectory_points")
        master[f"{site}_x_nm"] = locus_x.combine_first(observed_x)
        master[f"{site}_y_nm"] = locus_y.combine_first(observed_y)
        master[f"{site}_position_observed"] = observed_flag
        master = master.drop(columns=[f"{site}_x_nm_observed", f"{site}_y_nm_observed"])

    master["53bp1_spt_position_observed"] = master[
        "53bp1_spt_position_observed"
    ].fillna(False).astype(bool)
    master = master.rename(
        columns={
            "53bp1_spt_x_nm_observed": "53bp1_spt_x_nm",
            "53bp1_spt_y_nm_observed": "53bp1_spt_y_nm",
        }
    )
    master["locus_bp1_frame_available"] = master["locus_bp1_frame_available"].fillna(False).astype(bool)
    master["shared_site_frame"] = (
        master["site1_position_observed"] & master["site2_position_observed"]
    )
    master["site1_x_um"] = master["site1_x_nm"] / 1000.0
    master["site1_y_um"] = master["site1_y_nm"] / 1000.0
    master["site2_x_um"] = master["site2_x_nm"] / 1000.0
    master["site2_y_um"] = master["site2_y_nm"] / 1000.0
    master["53bp1_spt_x_um"] = master["53bp1_spt_x_nm"] / 1000.0
    master["53bp1_spt_y_um"] = master["53bp1_spt_y_nm"] / 1000.0
    recomputed_separation = np.hypot(
        master["site2_x_nm"] - master["site1_x_nm"],
        master["site2_y_nm"] - master["site1_y_nm"],
    )
    if "site1_site2_separation_nm" in master:
        _numeric_agreement(
            master["site1_site2_separation_nm"],
            recomputed_separation.where(master["shared_site_frame"]),
            "site1_site2_separation_nm",
        )
    master["site1_site2_separation_nm"] = recomputed_separation.where(
        master["shared_site_frame"]
    )

    drop_helpers = observed_times + [f"{site}_uniform_time_s_observed" for site in SITES]
    master = master.drop(columns=[column for column in drop_helpers if column in master])
    master = master.sort_values(
        ["hour_post_delivery", "nd2_id", "crop_id", "allele_index", "frame"], kind="stable"
    ).reset_index(drop=True)
    _require_unique(master, KEY, "bundle_frame_master")

    bundle_index = allele_index.copy().merge(
        crop_index[
            [
                "nd2_id",
                "crop_id",
                "movie_frames",
                "exact_timing_points",
                "frame_interval_s_production",
                "task_status",
                "availability_class",
                "unavailable_reason",
            ]
        ],
        on=["nd2_id", "crop_id"],
        how="left",
        validate="many_to_one",
    )
    _require_unique(bundle_index, ["bundle_id"], "bundle_index")

    source_counts = trajectory_points.groupby("site_id", sort=False).size().to_dict()
    expected_counts = {site: int(source_counts.get(site, 0)) for site in SITES}
    observed_counts = {
        site: int(master[f"{site}_position_observed"].sum()) for site in SITES
    }
    if expected_counts != observed_counts:
        raise UnfilteredCacheError(
            f"trajectory point reconciliation failed: expected={expected_counts}, observed={observed_counts}"
        )
    if int(master["locus_bp1_frame_available"].sum()) != len(bp1_allele_frames):
        raise UnfilteredCacheError("locus-frame reconciliation failed")
    census = {
        "analysis_filtering_applied": False,
        "rows": len(master),
        "bundles": int(master.bundle_id.nunique()),
        "crops_with_rows": int(master.crop_id.nunique()),
        "hours": sorted(pd.to_numeric(master.hour_post_delivery).dropna().unique().tolist()),
        "trajectory_points_by_site": observed_counts,
        "shared_site_frames": int(master.shared_site_frame.sum()),
        "locus_bp1_frames": int(master.locus_bp1_frame_available.sum()),
        "crop_registry_rows": len(crop_index),
        "bundle_registry_rows": len(bundle_index),
        "rules_not_applied": [
            "minimum track length beyond production availability",
            "paired coverage",
            "longest continuous run",
            "motion or separation threshold",
            "hour or acquisition selection",
            "53BP1 detection, association or intensity threshold",
        ],
    }
    return UnfilteredCacheResult(master, bundle_index, crop_index.copy(), census)


def write_unfiltered_cache(
    result: UnfilteredCacheResult,
    output_dir: str | Path,
    *,
    source_snapshot: str | Path,
) -> dict[str, Path]:
    """Write an immutable, compact cache and its provenance."""

    output = Path(output_dir).resolve()
    source = Path(source_snapshot).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {output}")
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        tables = staging / "tables"
        tables.mkdir()
        paths = {
            "bundle_frames": tables / "bundle_frame_master.parquet",
            "bundle_index": tables / "bundle_index.parquet",
            "crop_availability": tables / "crop_availability.parquet",
            "census": staging / "CACHE_CENSUS.json",
            "contract": staging / "CACHE_CONTRACT.json",
            "quickstart": staging / "QUICKSTART.md",
        }
        result.bundle_frames.to_parquet(
            paths["bundle_frames"], index=False, compression="zstd", row_group_size=100_000
        )
        result.bundle_index.to_parquet(paths["bundle_index"], index=False, compression="zstd")
        result.crop_availability.to_parquet(
            paths["crop_availability"], index=False, compression="zstd"
        )
        paths["census"].write_text(
            json.dumps(result.census, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        contract = {
            "schema_version": "v5.2.1-unfiltered-bundle-frame-cache-v1",
            "source_snapshot": str(source),
            "primary_key": ["bundle_id", "frame"],
            "primary_table": "tables/bundle_frame_master.parquet",
            "observation_policy": "retain every available source row; no analysis QC or interpolation",
            "missingness_policy": "missing stays missing; 53BP1 nondetection is not an exclusion",
            "global_bp1_objects": str(source / "tables" / "bp1_focus_objects"),
            "global_bp1_segmentation": str(source / "tables" / "bp1_frame_segmentation"),
            "note": (
                "Global 53BP1 objects remain separate because they are one-to-many per crop-frame; "
                "joining them into the bundle-frame table would create a Cartesian expansion."
            ),
        }
        paths["contract"].write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        paths["quickstart"].write_text(
            "# Quick start\n\n"
            "```python\n"
            "import pandas as pd\n"
            "from pathlib import Path\n\n"
            "cache = Path(r\"PATH_TO_THIS_CACHE\")\n"
            "table = cache / \"tables\" / \"bundle_frame_master.parquet\"\n\n"
            "# One acquisition/hour/site-pair slice with Parquet predicate pushdown.\n"
            "frames = pd.read_parquet(\n"
            "    table,\n"
            "    filters=[(\"hour_post_delivery\", \"=\", 10.0)],\n"
            "    columns=[\"bundle_id\", \"frame\", \"time_s\",\n"
            "             \"site1_position_observed\", \"site1_x_um\", \"site1_y_um\",\n"
            "             \"site2_position_observed\", \"site2_x_um\", \"site2_y_um\",\n"
            "             \"53bp1_spt_position_observed\", \"53bp1_spt_x_um\", \"53bp1_spt_y_um\",\n"
            "             \"assignment_status\", \"site2_bp1_continuous_intensity_score\"],\n"
            ")\n"
            "```\n\n"
            "The table is unfiltered at the analysis level. Position flags distinguish observed "
            "coordinates from missing coordinates. `assignment_status` retains valid no-association "
            "states; it must not be converted into an eligibility filter.\n",
            encoding="utf-8",
        )
        manifest_path = staging / "BUILD_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "source_snapshot": str(source),
                    "artifacts": [
                        {
                            "role": role,
                            "path": path.relative_to(staging).as_posix(),
                            "bytes": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                        for role, path in sorted(paths.items())
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {role: output / path.relative_to(staging) for role, path in paths.items()} | {
        "manifest": output / "BUILD_MANIFEST.json"
    }
