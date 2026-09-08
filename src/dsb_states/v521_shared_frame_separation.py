"""Overall v5.2.1 shared-frame Site1/Site2 separation distribution.

The analysis unit is one recorded frame with finite Site1 and Site2 coordinates.
No trajectory-length, paired-coverage, motion, separation, 53BP1, hour, acquisition,
or learned-state filter is applied.  Frames remain repeated observations and are not
promoted to biological replicates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .plots import COLORS, configure_publication_style, save_figure

_BELOW_THRESHOLD_COLOR = "#9EB4C2"
_ABOVE_THRESHOLD_COLOR = "#D4573B"
_CANDIDATE_COLOR = "#2F6F8F"

_REQUIRED_COLUMNS = {
    "bundle_id",
    "nd2_id",
    "crop_id",
    "hour_post_delivery",
    "frame",
    "site1_valid",
    "site2_valid",
    "site1_x_um",
    "site1_y_um",
    "site2_x_um",
    "site2_y_um",
    "site1_site2_separation_nm",
}


@dataclass(frozen=True)
class SharedFrameSeparationResult:
    """Exact shared-frame distances and their fixed-bin distribution."""

    frame_distances: pd.DataFrame
    histogram: pd.DataFrame
    summary: pd.DataFrame
    hour_histogram: pd.DataFrame
    hour_summary: pd.DataFrame
    method_contract: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_shared_frame_separation(
    bundle_frames: pd.DataFrame,
    *,
    threshold_nm: float = 500.0,
    bin_width_nm: float = 50.0,
) -> SharedFrameSeparationResult:
    """Retain every recorded shared frame and build an overall histogram."""

    missing = sorted(_REQUIRED_COLUMNS.difference(bundle_frames.columns))
    if missing:
        raise ValueError(f"bundle_frames is missing required columns: {missing}")
    if bundle_frames.empty:
        raise ValueError("bundle_frames must not be empty")
    if bundle_frames.duplicated(["bundle_id", "frame"]).any():
        raise ValueError("bundle_frames must be unique by bundle_id x frame")
    threshold = float(threshold_nm)
    width = float(bin_width_nm)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_nm must be finite and positive")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bin_width_nm must be finite and positive")
    if not np.isclose(threshold / width, round(threshold / width), atol=1e-12):
        raise ValueError("threshold_nm must lie exactly on a histogram bin edge")
    # Null flags occur only on registry-preserving rows without an available
    # localization output.  They cannot be shared observations and are kept as
    # explicit non-support rather than converted into a biological negative.
    shared_mask = bundle_frames["site1_valid"].fillna(False).astype(bool) & bundle_frames[
        "site2_valid"
    ].fillna(False).astype(bool)
    columns = [
        "bundle_id",
        "nd2_id",
        "crop_id",
        "hour_post_delivery",
        "frame",
        "site1_x_um",
        "site1_y_um",
        "site2_x_um",
        "site2_y_um",
        "site1_site2_separation_nm",
    ]
    frames = bundle_frames.loc[shared_mask, columns].copy()
    if frames.empty:
        raise ValueError("no recorded shared Site1/Site2 frames are available")
    coordinate_columns = ["site1_x_um", "site1_y_um", "site2_x_um", "site2_y_um"]
    coordinates = frames[coordinate_columns].apply(pd.to_numeric, errors="coerce")
    cached = pd.to_numeric(frames["site1_site2_separation_nm"], errors="coerce")
    if not np.isfinite(coordinates.to_numpy(float)).all() or not np.isfinite(cached).all():
        raise ValueError("shared frames contain non-finite coordinates or cached distances")
    calculated = (
        np.hypot(
            coordinates["site2_x_um"] - coordinates["site1_x_um"],
            coordinates["site2_y_um"] - coordinates["site1_y_um"],
        )
        * 1000.0
    )
    if not np.allclose(cached.to_numpy(float), calculated.to_numpy(float), atol=1e-8, rtol=1e-10):
        raise ValueError("cached Site1/Site2 separation does not match recorded coordinates")
    frames[coordinate_columns] = coordinates
    frames["separation_nm"] = calculated.to_numpy(float)
    frames["above_threshold"] = frames["separation_nm"].gt(threshold)

    distances = frames["separation_nm"].to_numpy(float)
    upper = math.ceil(float(np.max(distances)) / width) * width
    if upper <= float(np.max(distances)):
        upper += width
    edges = np.arange(0.0, upper + 0.5 * width, width)
    counts, observed_edges = np.histogram(distances, bins=edges)
    if counts.sum() != len(frames):
        raise AssertionError("histogram did not retain every shared frame")
    probabilities = counts.astype(float) / len(frames)
    histogram = pd.DataFrame(
        {
            "bin_left_nm": observed_edges[:-1],
            "bin_right_nm": observed_edges[1:],
            "bin_center_nm": (observed_edges[:-1] + observed_edges[1:]) / 2.0,
            "frame_count": counts,
            "frame_probability": probabilities,
            "frame_percent": 100.0 * probabilities,
            "threshold_group": np.where(
                observed_edges[:-1] >= threshold,
                f">{threshold:g} nm",
                f"≤{threshold:g} nm",
            ),
        }
    )
    n_above = int(np.count_nonzero(distances > threshold))
    n_at_or_below = int(len(distances) - n_above)
    summary = pd.DataFrame(
        [
            {
                "threshold_nm": threshold,
                "shared_frame_count": len(frames),
                "bundle_count": int(frames["bundle_id"].nunique()),
                "crop_count": int(frames["crop_id"].nunique()),
                "acquisition_count": int(frames["nd2_id"].nunique()),
                "n_at_or_below_threshold": n_at_or_below,
                "n_above_threshold": n_above,
                "fraction_above_threshold": n_above / len(frames),
                "percent_above_threshold": 100.0 * n_above / len(frames),
                "median_nm": float(np.median(distances)),
                "q25_nm": float(np.quantile(distances, 0.25)),
                "q75_nm": float(np.quantile(distances, 0.75)),
                "q90_nm": float(np.quantile(distances, 0.90)),
                "maximum_nm": float(np.max(distances)),
            }
        ]
    )
    hour_histogram_parts: list[pd.DataFrame] = []
    hour_summary_rows: list[dict[str, Any]] = []
    for hour, group in frames.groupby("hour_post_delivery", sort=True):
        hour_distances = group["separation_nm"].to_numpy(float)
        hour_counts, _ = np.histogram(hour_distances, bins=edges)
        hour_probabilities = hour_counts.astype(float) / len(group)
        hour_histogram_parts.append(
            pd.DataFrame(
                {
                    "hour_post_delivery": float(hour),
                    "bin_left_nm": observed_edges[:-1],
                    "bin_right_nm": observed_edges[1:],
                    "bin_center_nm": (observed_edges[:-1] + observed_edges[1:]) / 2.0,
                    "frame_count": hour_counts,
                    "frame_probability": hour_probabilities,
                    "frame_percent": 100.0 * hour_probabilities,
                    "threshold_group": np.where(
                        observed_edges[:-1] >= threshold,
                        f">{threshold:g} nm",
                        f"≤{threshold:g} nm",
                    ),
                }
            )
        )
        hour_n_above = int(np.count_nonzero(hour_distances > threshold))
        hour_summary_rows.append(
            {
                "hour_post_delivery": float(hour),
                "shared_frame_count": len(group),
                "bundle_count": int(group["bundle_id"].nunique()),
                "crop_count": int(group["crop_id"].nunique()),
                "acquisition_count": int(group["nd2_id"].nunique()),
                "n_above_threshold": hour_n_above,
                "percent_above_threshold": 100.0 * hour_n_above / len(group),
                "median_nm": float(np.median(hour_distances)),
                "q25_nm": float(np.quantile(hour_distances, 0.25)),
                "q75_nm": float(np.quantile(hour_distances, 0.75)),
                "maximum_nm": float(np.max(hour_distances)),
            }
        )
    hour_histogram = pd.concat(hour_histogram_parts, ignore_index=True)
    hour_summary = pd.DataFrame(hour_summary_rows)
    if not np.allclose(
        hour_histogram.groupby("hour_post_delivery")["frame_probability"].sum(),
        1.0,
        atol=1e-12,
    ):
        raise AssertionError("hour-specific histogram probabilities do not sum to one")
    contract = {
        "schema_version": 1,
        "source_cohort": "v5.2.1_unfiltered_cache",
        "analysis_unit": "recorded_frame_with_finite_site1_and_site2_coordinates",
        "distance_definition": "1000*hypot(site2_x_um-site1_x_um, site2_y_um-site1_y_um)",
        "distance_unit": "nm",
        "threshold_nm": threshold,
        "threshold_definition": "strictly_greater_than",
        "bin_width_nm": width,
        "histogram_weighting": "each_recorded_shared_frame_equal_weight",
        "coordinates": "recorded_coordinates_only",
        "interpolation": "prohibited",
        "missing_frame_compression": "prohibited",
        "scientific_filters": [],
        "excluded_by_analysis_unit": "frames_without_both_recorded_sites",
        "null_validity_flag_semantics": "unavailable_localization_not_shared_frame",
        "frame_independent_sample": False,
        "acquisitions_same_batch_assumption": True,
        "batch_effect_model": None,
        "inference": "descriptive_distribution_only",
        "folder_hours": [float(value) for value in hour_summary["hour_post_delivery"]],
        "hour_histogram_axes": "common_bins_common_x_and_y_limits",
    }
    return SharedFrameSeparationResult(
        frame_distances=frames.reset_index(drop=True),
        histogram=histogram,
        summary=summary,
        hour_histogram=hour_histogram,
        hour_summary=hour_summary,
        method_contract=contract,
    )


def build_shared_frame_separation_figure(
    result: SharedFrameSeparationResult,
    output_stem: str | Path,
) -> list[Path]:
    """Draw a compact, single-column separation histogram."""

    histogram = result.histogram
    record = result.summary.iloc[0]
    threshold = float(record["threshold_nm"])
    configure_publication_style()
    figure, axis = plt.subplots(figsize=(3.55, 2.85))
    colors = np.where(
        histogram["bin_left_nm"].to_numpy(float) >= threshold,
        _ABOVE_THRESHOLD_COLOR,
        _BELOW_THRESHOLD_COLOR,
    )
    axis.bar(
        histogram["bin_left_nm"],
        histogram["frame_percent"],
        width=histogram["bin_right_nm"] - histogram["bin_left_nm"],
        align="edge",
        color=colors,
        edgecolor="none",
        linewidth=0.0,
    )
    axis.axvline(
        threshold,
        color="#20252A",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=4,
    )
    y_upper = float(histogram["frame_percent"].max()) * 1.17
    axis.text(
        threshold,
        y_upper * 0.98,
        f"{threshold:g} nm",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#20252A",
    )
    axis.text(
        0.98,
        0.92,
        f">{threshold:g} nm  {record['percent_above_threshold']:.1f}%\n"
        f"{int(record['n_above_threshold']):,} / {int(record['shared_frame_count']):,} frames",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color=_ABOVE_THRESHOLD_COLOR,
        linespacing=1.35,
    )
    axis.set_xlim(0.0, float(histogram["bin_right_nm"].max()))
    axis.set_ylim(0.0, y_upper)
    axis.set_xlabel("Observed 2D Site1–Site2 separation (nm)")
    axis.set_ylabel("Shared frames per 50-nm bin (%)")
    axis.set_title("Site1–Site2 separation", loc="left", weight="bold", pad=7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.8)
    axis.xaxis.set_major_locator(plt.MultipleLocator(1000.0))
    axis.xaxis.set_minor_locator(plt.MultipleLocator(500.0))
    axis.legend(
        handles=[
            Patch(
                facecolor=_BELOW_THRESHOLD_COLOR,
                edgecolor="none",
                label=f"≤{threshold:g} nm",
            ),
            Patch(
                facecolor=_ABOVE_THRESHOLD_COLOR,
                edgecolor="none",
                label=f">{threshold:g} nm",
            ),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.73),
        handlelength=1.2,
        borderaxespad=0.0,
    )
    figure.text(
        0.16,
        0.005,
        "All recorded shared frames; no interpolation or trajectory-level QC.\n"
        "Frames are repeated observations, not biological replicates.",
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=COLORS["neutral"],
    )
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.25, top=0.90)
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def build_shared_frame_separation_by_hour_figure(
    result: SharedFrameSeparationResult,
    output_stem: str | Path,
) -> list[Path]:
    """Draw all folder-hours with common bins and axes."""

    histograms = result.hour_histogram
    summaries = result.hour_summary
    hours = sorted(summaries["hour_post_delivery"].astype(float).unique())
    if len(hours) > 9:
        raise ValueError("the fixed 3 x 3 layout supports at most nine folder-hours")
    threshold = float(result.summary.iloc[0]["threshold_nm"])
    configure_publication_style()
    figure, axes = plt.subplots(3, 3, figsize=(7.15, 6.15), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    y_upper = float(histograms["frame_percent"].max()) * 1.20
    x_upper = float(result.histogram["bin_right_nm"].max())
    for index, hour in enumerate(hours):
        axis = flat_axes[index]
        histogram = histograms.loc[
            np.isclose(histograms["hour_post_delivery"].astype(float), hour)
        ].sort_values("bin_left_nm")
        record = summaries.loc[
            np.isclose(summaries["hour_post_delivery"].astype(float), hour)
        ].iloc[0]
        colors = np.where(
            histogram["bin_left_nm"].to_numpy(float) >= threshold,
            _ABOVE_THRESHOLD_COLOR,
            _BELOW_THRESHOLD_COLOR,
        )
        axis.bar(
            histogram["bin_left_nm"],
            histogram["frame_percent"],
            width=histogram["bin_right_nm"] - histogram["bin_left_nm"],
            align="edge",
            color=colors,
            edgecolor="none",
            linewidth=0.0,
        )
        axis.axvline(
            threshold,
            color="#20252A",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            zorder=4,
        )
        axis.set_title(f"{chr(65 + index)}  {hour:g} h", loc="left", weight="bold", pad=4)
        axis.text(
            0.97,
            0.92,
            f">{threshold:g} nm  {record['percent_above_threshold']:.1f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.8,
            color=_ABOVE_THRESHOLD_COLOR,
        )
        axis.text(
            0.97,
            0.80,
            f"n={int(record['shared_frame_count']):,}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
            color=COLORS["neutral"],
        )
        axis.set_xlim(0.0, x_upper)
        axis.set_ylim(0.0, y_upper)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=2.8, width=0.75)
        axis.xaxis.set_major_locator(plt.MultipleLocator(2000.0))
        axis.xaxis.set_minor_locator(plt.MultipleLocator(1000.0))
    for axis in flat_axes[len(hours) :]:
        axis.set_visible(False)
    figure.supxlabel("Observed 2D Site1–Site2 separation (nm)", y=0.050, fontsize=9)
    figure.supylabel("Shared frames per 50-nm bin (%)", x=0.020, fontsize=9)
    figure.suptitle(
        "Site1–Site2 separation by time after Cas9 delivery",
        x=0.08,
        ha="left",
        fontsize=11,
        weight="bold",
    )
    figure.legend(
        handles=[
            Patch(
                facecolor=_BELOW_THRESHOLD_COLOR,
                edgecolor="none",
                label=f"≤{threshold:g} nm",
            ),
            Patch(
                facecolor=_ABOVE_THRESHOLD_COLOR,
                edgecolor="none",
                label=f">{threshold:g} nm",
            ),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        ncol=2,
        handlelength=1.1,
        columnspacing=1.2,
    )
    figure.text(
        0.08,
        0.012,
        "Common bins and axes; all recorded shared frames; no interpolation or trajectory-level "
        "QC. Folder hour is not synchronized cut time.",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=COLORS["neutral"],
    )
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.11, top=0.90, wspace=0.23, hspace=0.28)
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def build_shared_frame_separation_ppt_figure(
    result: SharedFrameSeparationResult,
    output_stem: str | Path,
    *,
    display_hours: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5, 10.0),
    display_max_nm: float = 2500.0,
) -> list[Path]:
    """Draw a slide-ready five-histogram panel plus the complete hour trend.

    The last histogram bar is an explicitly pooled overflow bin, so enlarging
    the informative 0--2.5-um range does not silently discard the long tail.
    """

    threshold = float(result.summary.iloc[0]["threshold_nm"])
    width = float(result.method_contract["bin_width_nm"])
    if display_max_nm <= threshold or not np.isclose(
        display_max_nm / width, round(display_max_nm / width), atol=1e-12
    ):
        raise ValueError("display_max_nm must exceed the threshold and lie on a bin edge")
    available_hours = set(result.hour_summary["hour_post_delivery"].astype(float))
    missing_hours = [hour for hour in display_hours if float(hour) not in available_hours]
    if missing_hours:
        raise ValueError(f"requested display hours are unavailable: {missing_hours}")

    regular_edges = np.arange(0.0, display_max_nm + 0.5 * width, width)
    per_hour: dict[float, np.ndarray] = {}
    maximum_percent = 0.0
    for hour in display_hours:
        values = result.frame_distances.loc[
            np.isclose(result.frame_distances["hour_post_delivery"].astype(float), hour),
            "separation_nm",
        ].to_numpy(float)
        counts, _ = np.histogram(values[values < display_max_nm], bins=regular_edges)
        overflow = int(np.count_nonzero(values >= display_max_nm))
        percentages = 100.0 * np.append(counts, overflow) / len(values)
        per_hour[float(hour)] = percentages
        maximum_percent = max(maximum_percent, float(percentages.max()))

    configure_publication_style()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 17,
            "axes.labelsize": 17,
            "axes.titlesize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "axes.linewidth": 1.2,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(13.333, 7.50))
    grid = figure.add_gridspec(
        2,
        5,
        height_ratios=(1.34, 0.82),
        left=0.055,
        right=0.99,
        bottom=0.105,
        top=0.96,
        wspace=0.62,
        hspace=0.57,
    )
    y_upper = math.ceil(maximum_percent * 1.13)
    bar_left = np.append(regular_edges[:-1], display_max_nm)
    bar_width = np.full(len(bar_left), width)
    colors = np.where(bar_left >= threshold, _ABOVE_THRESHOLD_COLOR, _BELOW_THRESHOLD_COLOR)
    overflow_center = display_max_nm + width / 2.0
    x_upper = display_max_nm + width

    for index, hour in enumerate(display_hours):
        axis = figure.add_subplot(grid[0, index])
        percentages = per_hour[float(hour)]
        axis.bar(
            bar_left,
            percentages,
            width=bar_width,
            align="edge",
            color=colors,
            edgecolor="none",
            linewidth=0.0,
        )
        axis.axvline(
            threshold,
            color="#20252A",
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            zorder=4,
        )
        record = result.hour_summary.loc[
            np.isclose(result.hour_summary["hour_post_delivery"].astype(float), hour)
        ].iloc[0]
        axis.set_title(
            f"{chr(65 + index)}  {hour:g} h  n={int(record['shared_frame_count']):,}",
            loc="left",
            weight="bold",
            pad=9,
            fontsize=17,
        )
        axis.text(
            0.98,
            0.64,
            f"{record['percent_above_threshold']:.1f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=17,
            color="#20252A",
            linespacing=1.25,
        )
        axis.legend(
            handles=[
                Patch(
                    facecolor=_BELOW_THRESHOLD_COLOR,
                    edgecolor="none",
                    label=f"≤{threshold:g} nm",
                ),
                Patch(
                    facecolor=_ABOVE_THRESHOLD_COLOR,
                    edgecolor="none",
                    label=f">{threshold:g} nm",
                ),
            ],
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.0, 0.98),
            fontsize=16.5,
            handlelength=0.65,
            handletextpad=0.28,
            labelspacing=0.18,
            borderaxespad=0.0,
        )
        axis.set_xlim(0.0, x_upper)
        axis.set_ylim(0.0, y_upper)
        axis.set_xticks(
            [0.0, 1000.0, overflow_center],
            ["0", "1k", "≥2.5k"],
        )
        axis.set_yticks(np.arange(0.0, y_upper + 0.1, 5.0))
        axis.set_xlabel("Separation (nm)", labelpad=6)
        axis.set_ylabel("Frames / 50 nm (%)", labelpad=6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=4.0, width=1.1)

    trend_axis = figure.add_subplot(grid[1, 0:3])
    trend = result.hour_summary.loc[
        ~np.isclose(result.hour_summary["hour_post_delivery"].astype(float), 5.0)
    ].sort_values("hour_post_delivery", kind="stable")
    trend_hours = trend["hour_post_delivery"].to_numpy(float)
    trend_percent = trend["percent_above_threshold"].to_numpy(float)
    positions = np.arange(len(trend))
    trend_axis.plot(
        positions,
        trend_percent,
        color=_ABOVE_THRESHOLD_COLOR,
        linewidth=3.0,
        zorder=2,
    )
    selected_mask = np.array(
        [any(np.isclose(hour, chosen) for chosen in display_hours) for hour in trend_hours]
    )
    trend_axis.scatter(
        positions[selected_mask],
        trend_percent[selected_mask],
        color=_ABOVE_THRESHOLD_COLOR,
        edgecolor="white",
        linewidth=1.2,
        s=88,
        zorder=4,
        label="Histogram panels",
    )
    trend_axis.scatter(
        positions[~selected_mask],
        trend_percent[~selected_mask],
        facecolor="white",
        edgecolor=_ABOVE_THRESHOLD_COLOR,
        linewidth=1.8,
        s=76,
        zorder=4,
        label="Other hours",
    )
    for position, hour, value in zip(positions, trend_hours, trend_percent, strict=True):
        trend_axis.text(
            position,
            value + 1.25,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=17,
            color="#20252A",
        )
    trend_axis.set_xlim(-0.35, len(trend) - 0.65)
    trend_axis.set_ylim(20.0, 46.5)
    trend_axis.set_xticks(positions, [f"{hour:g}" for hour in trend_hours])
    trend_axis.set_yticks(np.arange(20.0, 46.0, 5.0))
    trend_axis.set_xlabel("Time after Cas9 delivery (h)")
    trend_axis.set_ylabel(f"Shared frames >{threshold:g} nm (%)")
    trend_axis.spines[["top", "right"]].set_visible(False)
    trend_axis.tick_params(direction="out", length=4.0, width=1.1)
    trend_axis.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=2,
        fontsize=17,
        borderaxespad=0.0,
    )
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    configure_publication_style()
    return outputs


def build_threshold_sensitivity(
    frame_distances: pd.DataFrame,
    *,
    threshold_min_nm: float = 400.0,
    threshold_max_nm: float = 1000.0,
    threshold_step_nm: float = 25.0,
    candidate_rounding_nm: float = 50.0,
    expected_peak_hour: float = 4.5,
    late_hours: tuple[float, float] = (5.0, 10.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explore threshold sensitivity under an explicit rise-then-fall shape rule.

    This is deliberately post-hoc.  The selected candidate is not a confirmatory
    threshold and is kept separate from the user-prespecified primary threshold.
    """

    required = {"hour_post_delivery", "separation_nm"}
    if missing := required.difference(frame_distances.columns):
        raise ValueError(f"frame_distances is missing columns: {sorted(missing)}")
    lower = float(threshold_min_nm)
    upper = float(threshold_max_nm)
    step = float(threshold_step_nm)
    rounding = float(candidate_rounding_nm)
    if lower <= 0.0 or upper <= lower or step <= 0.0 or rounding <= 0.0:
        raise ValueError("threshold sensitivity limits must be positive and increasing")
    thresholds = np.arange(lower, upper + 0.5 * step, step)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for hour, group in frame_distances.groupby("hour_post_delivery", sort=True):
            values = group["separation_nm"].to_numpy(float)
            n_above = int(np.count_nonzero(values > threshold))
            rows.append(
                {
                    "threshold_nm": float(threshold),
                    "hour_post_delivery": float(hour),
                    "shared_frame_count": len(values),
                    "n_above_threshold": n_above,
                    "percent_above_threshold": 100.0 * n_above / len(values),
                }
            )
    sweep = pd.DataFrame(rows)
    pivot = sweep.pivot(
        index="threshold_nm",
        columns="hour_post_delivery",
        values="percent_above_threshold",
    ).sort_index()
    required_hours = [
        *sorted(hour for hour in pivot.columns if hour <= expected_peak_hour),
        *late_hours,
    ]
    if (
        expected_peak_hour not in pivot.columns
        or any(hour not in pivot.columns for hour in late_hours)
        or len(required_hours) < 5
    ):
        selection = pd.DataFrame(
            [
                {
                    "selection_status": "gated_required_hours_unavailable",
                    "candidate_threshold_nm": np.nan,
                }
            ]
        )
        return sweep, selection

    pre_hours = sorted(hour for hour in pivot.columns if hour <= expected_peak_hour)
    candidate_rows: list[dict[str, Any]] = []
    for threshold, values in pivot.iterrows():
        if not np.isclose(threshold / rounding, round(threshold / rounding), atol=1e-12):
            continue
        pre = values.loc[pre_hours].to_numpy(float)
        peak = float(values.loc[expected_peak_hour])
        late = values.loc[list(late_hours)].to_numpy(float)
        shape_ok = bool(np.all(np.diff(pre) > 0.0) and np.all(late < peak))
        candidate_rows.append(
            {
                "threshold_nm": float(threshold),
                "shape_ok": shape_ok,
                "rise_1p5_to_peak_pp": peak - float(pre[0]),
                "drop_peak_to_5h_pp": peak - float(values.loc[late_hours[0]]),
                "drop_peak_to_10h_pp": peak - float(values.loc[late_hours[1]]),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    supported = candidates.loc[candidates["shape_ok"]].copy()
    if supported.empty:
        selection = pd.DataFrame(
            [
                {
                    "selection_status": "no_rounded_threshold_satisfied_shape",
                    "candidate_threshold_nm": np.nan,
                }
            ]
        )
        return sweep, selection
    chosen = supported.sort_values(
        ["drop_peak_to_10h_pp", "rise_1p5_to_peak_pp", "threshold_nm"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    selection = pd.DataFrame(
        [
            {
                "selection_status": "posthoc_visual_candidate_only",
                "candidate_threshold_nm": float(chosen["threshold_nm"]),
                "scan_min_nm": lower,
                "scan_max_nm": upper,
                "scan_step_nm": step,
                "candidate_rounding_nm": rounding,
                "expected_peak_hour": expected_peak_hour,
                "shape_rule": (
                    "strict_increase_1p5_to_4p5_and_both_5h_10h_below_4p5"
                ),
                "selection_rule": (
                    "among_50nm_aligned_shape_matches_maximize_4p5_minus_10h_drop"
                ),
                "rise_1p5_to_peak_pp": float(chosen["rise_1p5_to_peak_pp"]),
                "drop_peak_to_5h_pp": float(chosen["drop_peak_to_5h_pp"]),
                "drop_peak_to_10h_pp": float(chosen["drop_peak_to_10h_pp"]),
                "inference_status": "exploratory_posthoc_not_confirmatory",
            }
        ]
    )
    return sweep, selection


def build_threshold_sensitivity_figure(
    sensitivity: pd.DataFrame,
    selection: pd.DataFrame,
    output_stem: str | Path,
    *,
    primary_threshold_nm: float,
) -> list[Path]:
    """Show the full threshold scan and primary/candidate temporal curves."""

    if selection.empty or not np.isfinite(selection.iloc[0]["candidate_threshold_nm"]):
        raise ValueError("threshold sensitivity figure requires a supported candidate")
    candidate = float(selection.iloc[0]["candidate_threshold_nm"])
    pivot = sensitivity.pivot(
        index="threshold_nm",
        columns="hour_post_delivery",
        values="percent_above_threshold",
    ).sort_index()
    thresholds = pivot.index.to_numpy(float)
    hours = pivot.columns.to_numpy(float)
    configure_publication_style()
    figure, (heatmap_axis, curve_axis) = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.0),
        gridspec_kw={"width_ratios": (1.28, 1.0)},
    )
    image = heatmap_axis.imshow(
        pivot.to_numpy(float),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        extent=(-0.5, len(hours) - 0.5, thresholds[0] - 12.5, thresholds[-1] + 12.5),
    )
    heatmap_axis.axhline(
        primary_threshold_nm,
        color=_ABOVE_THRESHOLD_COLOR,
        linewidth=1.4,
        label=f"Primary {primary_threshold_nm:g} nm",
    )
    heatmap_axis.axhline(
        candidate,
        color="white",
        linewidth=1.1,
        linestyle=(0, (3, 2)),
        label=f"Candidate {candidate:g} nm",
    )
    heatmap_axis.set_xticks(np.arange(len(hours)), [f"{hour:g}" for hour in hours])
    heatmap_axis.set_xlabel("Folder-hour after delivery")
    heatmap_axis.set_ylabel("Distance threshold (nm)")
    heatmap_axis.set_title("A  Threshold sensitivity", loc="left", weight="bold")
    heatmap_axis.legend(frameon=False, loc="upper left", fontsize=7)
    colorbar = figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.025)
    colorbar.set_label("Frames above threshold (%)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    for threshold, color, linestyle, label in (
        (primary_threshold_nm, _ABOVE_THRESHOLD_COLOR, "-", f"{primary_threshold_nm:g} nm primary"),
        (candidate, _CANDIDATE_COLOR, (0, (3, 2)), f"{candidate:g} nm candidate"),
    ):
        curve = sensitivity.loc[np.isclose(sensitivity["threshold_nm"], threshold)].sort_values(
            "hour_post_delivery"
        )
        x = curve["hour_post_delivery"].to_numpy(float)
        y = curve["percent_above_threshold"].to_numpy(float)
        curve_axis.plot(x, y, color=color, linestyle=linestyle, linewidth=1.6, label=label)
        ordinary = ~np.isclose(x, 5.0)
        curve_axis.scatter(x[ordinary], y[ordinary], color=color, s=21, zorder=3)
        if np.any(~ordinary):
            curve_axis.scatter(
                x[~ordinary],
                y[~ordinary],
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                s=28,
                zorder=4,
            )
    curve_axis.set_xlabel("Folder-hour after delivery")
    curve_axis.set_ylabel("Frames above threshold (%)")
    curve_axis.set_title("B  Selected temporal profiles", loc="left", weight="bold")
    curve_axis.spines[["top", "right"]].set_visible(False)
    curve_axis.tick_params(direction="out")
    curve_axis.legend(frameon=False, loc="best", fontsize=7)
    curve_axis.annotate(
        "5 h: one acquisition",
        xy=(5.0, float(curve_axis.get_ylim()[0])),
        xytext=(5.25, float(curve_axis.get_ylim()[0]) + 1.2),
        fontsize=6.8,
        color=COLORS["neutral"],
    )
    figure.text(
        0.01,
        0.005,
        "Exploratory post-hoc scan (400–1000 nm); candidate selection is not confirmatory. "
        f"The {primary_threshold_nm:g}-nm threshold remains the prespecified main display.",
        fontsize=6.7,
        color=COLORS["neutral"],
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.91, wspace=0.34)
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def write_shared_frame_separation_result(
    result: SharedFrameSeparationResult,
    output_dir: str | Path,
    *,
    source_paths: tuple[str | Path, ...],
) -> dict[str, Path]:
    """Write an immutable result with source data and hashes."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {output}")
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    tables = staging / "tables"
    figures = staging / "figures"
    tables.mkdir()
    figures.mkdir()
    result.frame_distances.to_parquet(
        tables / "shared_frame_distances.parquet", index=False, compression="zstd"
    )
    result.histogram.to_csv(tables / "histogram_bins.csv", index=False, lineterminator="\n")
    result.summary.to_csv(tables / "threshold_summary.csv", index=False, lineterminator="\n")
    result.hour_histogram.to_csv(
        tables / "hour_histogram_bins.csv", index=False, lineterminator="\n"
    )
    result.hour_summary.to_csv(
        tables / "hour_threshold_summary.csv", index=False, lineterminator="\n"
    )
    sensitivity, selection = build_threshold_sensitivity(result.frame_distances)
    sensitivity.to_csv(
        tables / "threshold_sensitivity.csv", index=False, lineterminator="\n"
    )
    selection.to_csv(
        tables / "threshold_selection.csv", index=False, lineterminator="\n"
    )
    (staging / "METHOD_CONTRACT.json").write_text(
        json.dumps(result.method_contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    record = result.summary.iloc[0]
    threshold = float(record["threshold_nm"])
    hour_lines = "\n".join(
        f"- {row.hour_post_delivery:g} h: {int(row.n_above_threshold):,}/"
        f"{int(row.shared_frame_count):,} ({row.percent_above_threshold:.1f}%) "
        f">{threshold:g} nm; "
        f"median {row.median_nm:.1f} nm."
        for row in result.hour_summary.itertuples(index=False)
    )
    selected_threshold = float(selection.iloc[0]["candidate_threshold_nm"])
    (staging / "REPORT.md").write_text(
        "# Overall shared-frame Site1/Site2 separation\n\n"
        f"All {int(record['shared_frame_count']):,} recorded frames with valid Site1 and Site2 "
        "coordinates were retained. No trajectory-level QC, motion, separation, 53BP1, hour, "
        "acquisition or learned-state filter was applied.\n\n"
        f"The median observed 2D separation is {record['median_nm']:.1f} nm. "
        f"{int(record['n_above_threshold']):,}/{int(record['shared_frame_count']):,} shared "
        f"frames ({record['percent_above_threshold']:.1f}%) have separation strictly greater "
        f"than {record['threshold_nm']:.0f} nm. Frames are repeated observations, not "
        "independent biological replicates.\n\n"
        "## Folder-hour summaries\n\n"
        f"{hour_lines}\n\n"
        "## Exploratory threshold sensitivity\n\n"
        "A post-hoc 400--1000 nm scan was kept separate from the primary display. Among "
        "50-nm-aligned thresholds satisfying strict increase from 1.5 through 4.5 h and both "
        f"5/10 h below 4.5 h, the visual candidate is {selected_threshold:.0f} nm. It is not a "
        "confirmatory cutoff.\n",
        encoding="utf-8",
    )
    threshold_token = f"{threshold:g}".replace(".", "p")
    figure_paths = build_shared_frame_separation_figure(
        result,
        figures / f"fig_shared_frame_site1_site2_separation_{threshold_token}nm",
    )
    hour_figure_paths = build_shared_frame_separation_by_hour_figure(
        result,
        figures / f"fig_shared_frame_site1_site2_separation_by_hour_{threshold_token}nm",
    )
    sensitivity_figure_paths = build_threshold_sensitivity_figure(
        sensitivity,
        selection,
        figures / "fig_shared_frame_site1_site2_threshold_sensitivity",
        primary_threshold_nm=threshold,
    )
    ppt_summary = result.hour_summary.loc[
        result.hour_summary["hour_post_delivery"].isin((1.5, 2.5, 3.5, 4.5, 10.0))
    ].copy()
    ppt_summary.to_csv(
        tables / "ppt_panel_summary.csv", index=False, lineterminator="\n"
    )
    far_tail_summary = (
        result.frame_distances.groupby("hour_post_delivery", sort=True)["separation_nm"]
        .agg(
            shared_frame_count="size",
            n_above_2500nm=lambda values: int(np.count_nonzero(values.to_numpy(float) > 2500.0)),
            maximum_separation_nm="max",
        )
        .reset_index()
    )
    far_tail_summary["percent_above_2500nm"] = (
        100.0
        * far_tail_summary["n_above_2500nm"]
        / far_tail_summary["shared_frame_count"]
    )
    far_tail_summary.to_csv(
        tables / "separation_above_2500nm_by_hour.csv",
        index=False,
        lineterminator="\n",
    )
    farthest_indices = result.frame_distances.groupby(
        "hour_post_delivery", sort=True
    )["separation_nm"].idxmax()
    farthest_frames = result.frame_distances.loc[
        farthest_indices,
        [
            "hour_post_delivery",
            "nd2_id",
            "crop_id",
            "bundle_id",
            "frame",
            "separation_nm",
            "site1_x_um",
            "site1_y_um",
            "site2_x_um",
            "site2_y_um",
        ],
    ].sort_values("hour_post_delivery", kind="stable")
    farthest_frames.to_csv(
        tables / "farthest_shared_frame_by_hour.csv",
        index=False,
        lineterminator="\n",
    )
    ppt_figure_paths = build_shared_frame_separation_ppt_figure(
        result,
        figures / f"fig_shared_frame_site1_site2_separation_{threshold_token}nm_ppt",
    )
    sources = [Path(path).resolve() for path in source_paths]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    artifacts = sorted(path for path in staging.rglob("*") if path.is_file())
    (staging / "OUTPUT_MANIFEST.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in sources
                ],
                "artifacts": [
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in artifacts
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    staging.rename(output)
    return {
        "output_dir": output,
        "png": output / "figures" / hour_figure_paths[0].name,
        "pdf": output / "figures" / hour_figure_paths[1].name,
        "svg": output / "figures" / hour_figure_paths[2].name,
        "overall_png": output / "figures" / figure_paths[0].name,
        "sensitivity_png": output / "figures" / sensitivity_figure_paths[0].name,
        "ppt_png": output / "figures" / ppt_figure_paths[0].name,
        "ppt_pdf": output / "figures" / ppt_figure_paths[1].name,
        "ppt_svg": output / "figures" / ppt_figure_paths[2].name,
        "summary": output / "tables" / "threshold_summary.csv",
        "hour_summary": output / "tables" / "hour_threshold_summary.csv",
        "threshold_selection": output / "tables" / "threshold_selection.csv",
    }
