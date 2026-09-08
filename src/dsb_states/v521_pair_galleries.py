"""Reproduce the prior T3/T4 paired-trajectory galleries on v5.2.1 data."""

from __future__ import annotations

import json
import shutil
import textwrap
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

SITE1_COLOR = "#2878B5"
SITE2_COLOR = "#F28E2B"
T3_THRESHOLDS = {
    "site1_frames": 10,
    "site2_frames": 10,
    "paired_frames": 10,
    "longest_paired_run": 5,
}
T4_THRESHOLDS = {
    "site1_frames": 20,
    "site2_frames": 20,
    "paired_frames": 20,
    "longest_paired_run": 10,
    "paired_coverage": 0.50,
}


class PairGalleryError(ValueError):
    """Raised when a gallery source or cohort violates the frozen contract."""


@dataclass(frozen=True)
class PairGalleryResult:
    """Frozen memberships, aligned shared frames and review order."""

    membership: pd.DataFrame
    t3_aligned: pd.DataFrame
    t4_aligned: pd.DataFrame
    t3_order: pd.DataFrame
    t4_order: pd.DataFrame
    census_by_hour: pd.DataFrame
    shared_half_width_um: float
    contract: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _longest_consecutive_run(frames: pd.Series) -> int:
    values = np.sort(pd.to_numeric(frames, errors="raise").to_numpy(dtype=int))
    if len(values) == 0:
        return 0
    if len(np.unique(values)) != len(values):
        raise PairGalleryError("duplicate frame within bundle")
    boundaries = np.flatnonzero(np.diff(values) != 1) + 1
    return int(max(len(part) for part in np.split(values, boundaries)))


def _membership(bundle_frames: pd.DataFrame, bundle_index: pd.DataFrame) -> pd.DataFrame:
    required_frames = {
        "bundle_id",
        "frame",
        "site1_position_observed",
        "site2_position_observed",
    }
    required_index = {
        "bundle_id",
        "nd2_id",
        "crop_id",
        "fov_id",
        "hour_post_delivery",
        "allele_index",
        "movie_frames",
    }
    missing = sorted(required_frames.difference(bundle_frames.columns))
    missing_index = sorted(required_index.difference(bundle_index.columns))
    if missing or missing_index:
        raise PairGalleryError(f"missing frame={missing}; index={missing_index}")
    if bundle_frames.duplicated(["bundle_id", "frame"]).any():
        raise PairGalleryError("bundle frame table is not unique by bundle_id/frame")
    if bundle_index.bundle_id.duplicated().any():
        raise PairGalleryError("bundle index is not unique by bundle_id")

    rows: list[dict[str, Any]] = []
    for bundle_id, group in bundle_frames.groupby("bundle_id", sort=False):
        site1 = group.site1_position_observed.astype(bool)
        site2 = group.site2_position_observed.astype(bool)
        paired = site1 & site2
        rows.append(
            {
                "bundle_id": str(bundle_id),
                "site1_frames": int(site1.sum()),
                "site2_frames": int(site2.sum()),
                "paired_frames": int(paired.sum()),
                "longest_paired_run": _longest_consecutive_run(group.loc[paired, "frame"]),
            }
        )
    result = bundle_index.copy()
    result["bundle_id"] = result.bundle_id.astype(str)
    counts = pd.DataFrame(rows)
    result = result.merge(counts, on="bundle_id", how="left", validate="1:1")
    for column in ("site1_frames", "site2_frames", "paired_frames", "longest_paired_run"):
        result[column] = result[column].fillna(0).astype(int)
    movie_frames = pd.to_numeric(result.movie_frames, errors="coerce")
    if movie_frames.isna().any() or (movie_frames <= 0).any():
        raise PairGalleryError("movie_frames must be finite and positive")
    result["paired_coverage"] = result.paired_frames / movie_frames
    result["t3_included"] = (
        (result.site1_frames >= T3_THRESHOLDS["site1_frames"])
        & (result.site2_frames >= T3_THRESHOLDS["site2_frames"])
        & (result.paired_frames >= T3_THRESHOLDS["paired_frames"])
        & (result.longest_paired_run >= T3_THRESHOLDS["longest_paired_run"])
    )
    result["t4_included"] = (
        (result.site1_frames >= T4_THRESHOLDS["site1_frames"])
        & (result.site2_frames >= T4_THRESHOLDS["site2_frames"])
        & (result.paired_frames >= T4_THRESHOLDS["paired_frames"])
        & (result.longest_paired_run >= T4_THRESHOLDS["longest_paired_run"])
        & (result.paired_coverage >= T4_THRESHOLDS["paired_coverage"])
    )
    if not result.loc[result.t4_included, "t3_included"].all():
        raise PairGalleryError("T4 is not nested inside T3")
    failure_columns = {
        "site1_frames": "site1_frames",
        "site2_frames": "site2_frames",
        "paired_frames": "paired_frames",
        "longest_paired_run": "longest_paired_run",
    }
    result["t3_exclusion_reasons"] = [
        "|".join(
            name
            for name, column in failure_columns.items()
            if int(row[column]) < int(T3_THRESHOLDS[name])
        )
        for _, row in result.iterrows()
    ]
    result["t4_exclusion_reasons"] = [
        "|".join(
            [
                *[
                    name
                    for name, column in failure_columns.items()
                    if int(row[column]) < int(T4_THRESHOLDS[name])
                ],
                *(
                    ["paired_coverage"]
                    if float(row.paired_coverage) < T4_THRESHOLDS["paired_coverage"]
                    else []
                ),
            ]
        )
        for _, row in result.iterrows()
    ]
    return result.sort_values(["hour_post_delivery", "nd2_id", "crop_id", "allele_index"])


def _aligned_shared_frames(
    bundle_frames: pd.DataFrame,
    membership: pd.DataFrame,
    inclusion_column: str,
    tier: str,
) -> pd.DataFrame:
    included = set(membership.loc[membership[inclusion_column], "bundle_id"].astype(str))
    selected = bundle_frames.loc[
        bundle_frames.bundle_id.astype(str).isin(included)
        & bundle_frames.site1_position_observed.astype(bool)
        & bundle_frames.site2_position_observed.astype(bool)
    ].copy()
    selected = selected.merge(
        membership[["bundle_id", "nd2_id", "crop_id", "hour_post_delivery", "allele_index"]].rename(
            columns={
                "nd2_id": "acquisition_id",
                "crop_id": "cell_id",
                "hour_post_delivery": "hour_post_delivery_membership",
                "allele_index": "allele_index_membership",
            }
        ),
        on="bundle_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[pd.DataFrame] = []
    for bundle_id, raw_group in selected.groupby("bundle_id", sort=True):
        group = raw_group.sort_values("frame").copy()
        site1 = group[["site1_x_um", "site1_y_um"]].to_numpy(float).copy()
        site2 = group[["site2_x_um", "site2_y_um"]].to_numpy(float).copy()
        if not np.isfinite(np.vstack([site1, site2])).all():
            raise PairGalleryError(f"non-finite paired coordinates in {bundle_id}")
        pooled_centroid = np.vstack([site1, site2]).mean(axis=0)
        site1 -= pooled_centroid
        site2 -= pooled_centroid
        out = group[
            [
                "bundle_id",
                "acquisition_id",
                "cell_id",
                "hour_post_delivery_membership",
                "allele_index_membership",
                "frame",
                "time_s",
            ]
        ].rename(
            columns={
                "hour_post_delivery_membership": "hour_post_delivery",
                "allele_index_membership": "allele_index",
            }
        )
        out["tier"] = tier
        out["site1_x_aligned_um"] = site1[:, 0]
        out["site1_y_aligned_um"] = site1[:, 1]
        out["site2_x_aligned_um"] = site2[:, 0]
        out["site2_y_aligned_um"] = site2[:, 1]
        rows.append(out)
    if not rows:
        raise PairGalleryError(f"{tier} has no shared-frame observations")
    return pd.concat(rows, ignore_index=True).sort_values(["bundle_id", "frame"], kind="stable")


def gallery_order(aligned: pd.DataFrame, *, within_hour: bool) -> pd.DataFrame:
    """Sort pairs by RMS of conventional Site1/Site2 radii of gyration."""

    rows: list[dict[str, Any]] = []
    for bundle_id, group in aligned.groupby("bundle_id", sort=True):
        site1 = group[["site1_x_aligned_um", "site1_y_aligned_um"]].to_numpy(float)
        site2 = group[["site2_x_aligned_um", "site2_y_aligned_um"]].to_numpy(float)
        site1_centered = site1 - site1.mean(axis=0)
        site2_centered = site2 - site2.mean(axis=0)
        site1_rg = float(np.sqrt(np.mean(np.sum(site1_centered**2, axis=1))))
        site2_rg = float(np.sqrt(np.mean(np.sum(site2_centered**2, axis=1))))
        rows.append(
            {
                "bundle_id": str(bundle_id),
                "acquisition_id": str(group.acquisition_id.iloc[0]),
                "cell_id": str(group.cell_id.iloc[0]),
                "hour_post_delivery": float(group.hour_post_delivery.iloc[0]),
                "paired_frames": len(group),
                "site1_rg_um": site1_rg,
                "site2_rg_um": site2_rg,
                "paired_site_rg_rms_um": float(np.sqrt((site1_rg**2 + site2_rg**2) / 2.0)),
            }
        )
    order = pd.DataFrame(rows)
    sort_columns = (
        ["hour_post_delivery", "paired_site_rg_rms_um", "bundle_id"]
        if within_hour
        else [
            "paired_site_rg_rms_um",
            "bundle_id",
        ]
    )
    order = order.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if within_hour:
        order["gallery_rank"] = order.groupby("hour_post_delivery").cumcount() + 1
    else:
        order["gallery_rank"] = np.arange(1, len(order) + 1)
    return order


def build_pair_gallery_result(
    bundle_frames: pd.DataFrame,
    bundle_index: pd.DataFrame,
) -> PairGalleryResult:
    membership = _membership(bundle_frames, bundle_index)
    t3 = _aligned_shared_frames(bundle_frames, membership, "t3_included", "T3_relaxed")
    t4 = _aligned_shared_frames(bundle_frames, membership, "t4_included", "T4_strict")
    t3_order = gallery_order(t3, within_hour=True)
    t4_order = gallery_order(t4, within_hour=False)
    maximum = float(
        np.abs(
            t3[
                [
                    "site1_x_aligned_um",
                    "site1_y_aligned_um",
                    "site2_x_aligned_um",
                    "site2_y_aligned_um",
                ]
            ].to_numpy(float)
        ).max()
    )
    half_width = float(np.ceil(maximum * 4.0) / 4.0)
    hours = sorted(pd.to_numeric(membership.hour_post_delivery).dropna().unique())
    census_rows = []
    for hour in hours:
        group = membership.loc[np.isclose(membership.hour_post_delivery, hour)]
        for tier, column in (("T3_relaxed", "t3_included"), ("T4_strict", "t4_included")):
            included = group.loc[group[column]]
            census_rows.append(
                {
                    "hour_post_delivery": float(hour),
                    "tier": tier,
                    "bundles": len(included),
                    "acquisitions": included.nd2_id.nunique(),
                    "cells": included.crop_id.nunique(),
                    "paired_frames": int(included.paired_frames.sum()),
                }
            )
    contract = {
        "analysis": "v5.2.1_pair_trajectory_manual_review_galleries",
        "source": "v5.2.1 unfiltered bundle-frame cache",
        "t3_rule": T3_THRESHOLDS,
        "t4_rule": T4_THRESHOLDS,
        "53bp1_filter": "none",
        "pairing": "same bundle and same recorded frame; both Site1 and Site2 observed",
        "alignment": "subtract pooled Site1/Site2 centroid per bundle; preserve native orientation",
        "gap_display": "connect recorded paired positions in frame order across gaps; no interpolation",
        "gallery_order": "ascending sqrt((Rg_site1^2 + Rg_site2^2)/2)",
        "scale": f"all galleries use fixed +/-{half_width:.2f} um tile coordinates",
        "time_semantics": "cross-sectional hour after Cas9 delivery; not time since cutting",
        "inference": "manual descriptive review only",
    }
    return PairGalleryResult(
        membership=membership.reset_index(drop=True),
        t3_aligned=t3.reset_index(drop=True),
        t4_aligned=t4.reset_index(drop=True),
        t3_order=t3_order,
        t4_order=t4_order,
        census_by_hour=pd.DataFrame(census_rows),
        shared_half_width_um=half_width,
        contract=contract,
    )


def _theme() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _gallery_figure(
    aligned: pd.DataFrame,
    order: pd.DataFrame,
    *,
    half_width_um: float,
    columns: int,
    title: str,
    caption: str,
) -> Figure:
    _theme()
    if columns <= 0 or order.empty:
        raise PairGalleryError("gallery needs positive columns and at least one pair")
    columns = min(columns, len(order))
    rows = int(np.ceil(len(order) / columns))
    pitch = 2.0 * half_width_um + 0.45
    lines_by_site: dict[str, list[np.ndarray]] = {"site1": [], "site2": []}
    rank = order.set_index("bundle_id").gallery_rank.to_dict()
    for bundle_id, group in aligned.groupby("bundle_id", sort=False):
        index = int(rank[str(bundle_id)]) - 1
        row = index // columns
        column = index % columns
        offset = np.array([column * pitch, (rows - 1 - row) * pitch])
        sorted_group = group.sort_values("frame")
        for site in ("site1", "site2"):
            points = sorted_group[[f"{site}_x_aligned_um", f"{site}_y_aligned_um"]].to_numpy(float)
            # Missing frames intentionally do not split the visible line. No point is added.
            lines_by_site[site].append(points + offset)

    figure_width = min(14.5, max(7.0, 14.5 * columns / 20.0))
    figure_height = max(2.8, figure_width * rows / columns + 1.25)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    for site, color in (("site1", SITE1_COLOR), ("site2", SITE2_COLOR)):
        axis.add_collection(
            LineCollection(
                lines_by_site[site],
                colors=[color],
                linewidths=0.55,
                alpha=0.88,
                rasterized=True,
            )
        )
    x_min = -half_width_um
    x_max = (columns - 1) * pitch + half_width_um
    y_min = -half_width_um
    y_max = (rows - 1) * pitch + half_width_um
    footer = 0.60 * pitch
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min - footer, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)
    scale_x = x_min + 0.2 * pitch
    scale_y = y_min - 0.26 * pitch
    axis.plot([scale_x, scale_x + 1.0], [scale_y, scale_y], color="#222222", linewidth=2.2)
    axis.text(scale_x + 0.5, scale_y - 0.10 * pitch, "1 µm", ha="center", va="top")
    axis.legend(
        handles=[
            Line2D([0], [0], color=SITE1_COLOR, linewidth=2, label="Site1"),
            Line2D([0], [0], color=SITE2_COLOR, linewidth=2, label="Site2"),
        ],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
    )
    wrap_width = max(48, int(9 * figure_width))
    axis.set_title(textwrap.fill(title, width=wrap_width), loc="left", fontweight="bold")
    figure.subplots_adjust(top=0.965, bottom=0.045, left=0.015, right=0.995)
    figure.text(
        0.5,
        0.012,
        textwrap.fill(caption, width=wrap_width),
        ha="center",
        color="#42526A",
        fontsize=8.5,
    )
    return figure


def _write_figure(figure: Figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix in ("png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        options: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix == "png":
            options["dpi"] = 300
        figure.savefig(path, **options)
        outputs.append(path)
    plt.close(figure)
    return outputs


def write_pair_gallery_result(
    result: PairGalleryResult,
    output_dir: str | Path,
    *,
    source_files: tuple[str | Path, ...],
) -> dict[str, Path]:
    """Write immutable review galleries, tables, method contract and manifest."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite gallery result: {output}")
    sources = tuple(Path(path).resolve() for path in source_files)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        figures = staging / "figures"
        tables = staging / "tables"
        figures.mkdir()
        tables.mkdir()
        paths: dict[str, Path] = {}
        table_specs = (
            ("membership", result.membership, "qc_membership.parquet"),
            ("t3_aligned", result.t3_aligned, "t3_aligned_shared_frames.parquet"),
            ("t4_aligned", result.t4_aligned, "t4_aligned_shared_frames.parquet"),
            ("t3_order", result.t3_order, "t3_hour_gallery_order.csv"),
            ("t4_order", result.t4_order, "t4_gallery_order.csv"),
            ("census", result.census_by_hour, "census_by_hour.csv"),
        )
        for role, table, filename in table_specs:
            path = tables / filename
            if path.suffix == ".parquet":
                table.to_parquet(path, index=False, compression="zstd")
            else:
                table.to_csv(path, index=False, lineterminator="\n")
            paths[role] = path
        contract_path = staging / "METHOD_CONTRACT.json"
        contract_path.write_text(
            json.dumps(result.contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        paths["contract"] = contract_path

        for path in _write_figure(
            _gallery_figure(
                result.t4_aligned,
                result.t4_order,
                half_width_um=result.shared_half_width_um,
                columns=25,
                title=f"T4 strict paired-trajectory gallery | n={len(result.t4_order):,}",
                caption=(
                    "One pair per tile, ordered by increasing paired-site radius of gyration. "
                    "Fixed physical scale; recorded shared-frame positions are connected across gaps without interpolation."
                ),
            ),
            figures / "fig_qc_t4_strict_pair_gallery",
        ):
            paths[f"t4_gallery_{path.suffix[1:]}"] = path

        for hour in sorted(result.t3_order.hour_post_delivery.unique()):
            order = result.t3_order.loc[np.isclose(result.t3_order.hour_post_delivery, hour)].copy()
            aligned = result.t3_aligned.loc[np.isclose(result.t3_aligned.hour_post_delivery, hour)]
            acquisitions = order.acquisition_id.nunique()
            tag = f"{hour:g}".replace(".", "p") + "h"
            for path in _write_figure(
                _gallery_figure(
                    aligned,
                    order,
                    half_width_um=result.shared_half_width_um,
                    columns=20,
                    title=(
                        f"{hour:g} h | T3 relaxed paired trajectories | "
                        f"n={len(order):,}; acquisitions={acquisitions}"
                    ),
                    caption=(
                        "Within-hour radius-of-gyration order for visual review only. "
                        f"All hour galleries use the same ±{result.shared_half_width_um:.2f} µm tile scale; "
                        "hour is after Cas9 delivery, not verified time since cutting."
                    ),
                ),
                figures / f"fig_qc_t3_pair_gallery_{tag}",
            ):
                paths[f"t3_gallery_{tag}_{path.suffix[1:]}"] = path

        report_path = staging / "REPORT.md"
        t3_count = int(result.membership.t3_included.sum())
        t4_count = int(result.membership.t4_included.sum())
        report_path.write_text(
            "\n".join(
                [
                    "# v5.2.1 paired-trajectory QC review galleries",
                    "",
                    f"- T3 relaxed: {t3_count:,} bundles.",
                    f"- T4 strict: {t4_count:,} bundles.",
                    f"- Shared gallery tile range: ±{result.shared_half_width_um:.2f} µm.",
                    "- T3: Site1/Site2/shared >=10 frames and longest shared run >=5; no coverage rule.",
                    "- T4: Site1/Site2/shared >=20, longest shared run >=10 and shared coverage >=0.50.",
                    "- 53BP1 detection, association and intensity are not filters.",
                    "- Only recorded same-frame Site1/Site2 positions are drawn. Gaps are visually connected; no coordinate is interpolated.",
                    "- Each pair is centered on its pooled Site1/Site2 centroid, retains native image orientation, and is not individually rescaled.",
                    "- Ordering is a manual-review aid, not a cluster, state or temporal sequence.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        paths["report"] = report_path
        manifest_path = staging / "OUTPUT_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "sources": [
                        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                        for path in sources
                    ],
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
        "manifest": output / "OUTPUT_MANIFEST.json"
    }
