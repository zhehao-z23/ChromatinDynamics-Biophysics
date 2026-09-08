from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "site1": "#D99400",
    "site2": "#6F49A8",
    "bp1": "#168A5B",
    "neutral": "#435365",
    "light": "#E8EDF2",
    "warning": "#B64A3A",
}


def configure_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    figure: mpl.figure.Figure,
    output_stem: str | Path,
    formats: Iterable[str] = ("png", "pdf", "svg"),
) -> list[Path]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        figure.savefig(path, facecolor="white")
        outputs.append(path)
    return outputs


def build_analysis_design_figure(output_stem: str | Path) -> list[Path]:
    """Render the two-time-axis and pair-decomposition design without biological claims."""

    configure_publication_style()
    figure = plt.figure(figsize=(10.2, 5.2), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.25, 1), height_ratios=(1, 1))
    modality = figure.add_subplot(grid[:, 0])
    macro = figure.add_subplot(grid[0, 1])
    micro = figure.add_subplot(grid[1, 1])

    modality.set_xlim(0, 10)
    modality.set_ylim(0, 10)
    modality.axis("off")
    modality.set_title(
        "A  Paired-locus measurement and neutral decomposition", loc="left", weight="bold"
    )
    modality.add_patch(plt.Circle((3.0, 5.7), 0.33, color=COLORS["site1"], zorder=3))
    modality.add_patch(plt.Circle((7.0, 5.1), 0.33, color=COLORS["site2"], zorder=3))
    modality.plot([3.0, 4.9, 7.0], [5.7, 4.8, 5.1], color=COLORS["neutral"], lw=3)
    modality.scatter([5.0], [7.2], s=190, marker="*", color=COLORS["bp1"], zorder=4)
    modality.text(3.0, 6.35, "Site1", ha="center", color=COLORS["site1"], weight="bold")
    modality.text(7.0, 5.75, "Site2 (anchor)", ha="center", color=COLORS["site2"], weight="bold")
    modality.text(5.0, 7.85, "53BP1 point trajectory", ha="center", color=COLORS["bp1"])
    modality.annotate(
        "Site1–Site2 separation\n(local chromatin geometry)",
        xy=(5.0, 5.35),
        xytext=(5.0, 3.55),
        arrowprops={"arrowstyle": "->", "color": COLORS["neutral"]},
        ha="center",
    )
    modality.text(
        5,
        1.55,
        "common mode = pair midpoint\nrelative mode = Site2 − Site1\ncoupling = normalized velocity correlation",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.5", "fc": COLORS["light"], "ec": "none"},
    )
    modality.text(
        5,
        0.35,
        "Flank orientation and probe-to-cut distances unresolved: no broken-end language",
        ha="center",
        color=COLORS["warning"],
        fontsize=8,
    )

    macro.set_title("B  Macro-time: cross-sectional acquisitions", loc="left", weight="bold")
    macro.set_xlim(1, 10)
    macro.set_ylim(-0.25, 1.2)
    macro.set_yticks([])
    macro.set_xlabel("Hours after Cas9 delivery (not verified cut onset)")
    macro.spines[["left", "right", "top"]].set_visible(False)
    for hour in (1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5):
        macro.scatter(hour, 0.45, color=COLORS["neutral"], s=32)
    macro.annotate("different cells / acquisitions", (4.2, 0.76), ha="center")
    macro.text(
        5.5,
        1.05,
        "Population-level ordering only",
        ha="center",
        color=COLORS["warning"],
        fontsize=8,
    )

    micro.set_title("C  Micro-time: within one movie", loc="left", weight="bold")
    micro.set_xlim(0, 10)
    micro.set_ylim(-0.2, 1.15)
    micro.set_yticks([])
    micro.set_xlabel("Seconds within acquisition (exact timestamps)")
    micro.spines[["left", "right", "top"]].set_visible(False)
    micro.plot(
        [0.5, 2, 3.2, 4.8, 6.4, 8.0, 9.5],
        [0.3, 0.45, 0.38, 0.73, 0.62, 0.85, 0.72],
        color=COLORS["site1"],
        lw=2,
    )
    micro.plot(
        [0.5, 2, 3.2, 4.8, 6.4, 8.0, 9.5],
        [0.55, 0.62, 0.5, 0.79, 0.68, 0.75, 0.58],
        color=COLORS["site2"],
        lw=2,
    )
    micro.text(1.0, 0.94, "MSD / MSCD / VAC / VCC / DCD / state transitions", fontsize=8)
    micro.text(
        1.0, 0.04, "No interpolation across missing frames", color=COLORS["warning"], fontsize=8
    )
    return save_figure(figure, output_stem)


def _annotate_technical_scope(figure: mpl.figure.Figure, text: str) -> None:
    figure.text(
        0.01,
        0.005,
        text,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=COLORS["neutral"],
    )


def build_qc_cohort_flow_figure(source: pd.DataFrame, output_stem: str | Path) -> list[Path]:
    """Plot the configured bundle-level QC flow from an explicit source table."""

    required = {"display_order", "stage_label", "bundle_count"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Cohort-flow figure source lacks columns: {sorted(missing)}")
    ordered = source.sort_values("display_order", kind="stable").reset_index(drop=True)
    if ordered.empty or (pd.to_numeric(ordered["bundle_count"], errors="coerce") < 0).any():
        raise ValueError("Cohort-flow figure source must contain nonnegative bundle counts")

    configure_publication_style()
    figure, axis = plt.subplots(figsize=(6.8, 3.7))
    figure.subplots_adjust(left=0.25, right=0.96, top=0.84, bottom=0.24)
    counts = pd.to_numeric(ordered["bundle_count"], errors="raise").to_numpy(dtype=float)
    colors = [
        COLORS.get(str(value), COLORS["neutral"])
        for value in ordered.get("color_key", pd.Series(["neutral"] * len(ordered)))
    ]
    bars = axis.barh(np.arange(len(ordered)), counts, color=colors, height=0.62)
    axis.set_yticks(np.arange(len(ordered)), ordered["stage_label"])
    axis.invert_yaxis()
    axis.set_xlabel("Bundles")
    axis.set_title("T03A  Configured technical-QC cohort flow", loc="left", weight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    maximum = max(float(counts.max(initial=0)), 1.0)
    axis.set_xlim(0, maximum * 1.18)
    eligible = counts[0] if counts.size else np.nan
    for bar, count in zip(bars, counts, strict=True):
        fraction = count / eligible if eligible > 0 else np.nan
        label = f"{int(count):,}"
        if np.isfinite(fraction):
            label += f"  ({fraction:.1%})"
        axis.text(
            bar.get_width() + maximum * 0.015,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
        )
    _annotate_technical_scope(
        figure,
        "Technical inclusion only; counts are bundles, not independent biological replicates. "
        "Biological-experiment IDs are unavailable (M0 gated).",
    )
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def _coverage_group_summary(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "acquisition_id",
        "fov_id",
        "bundle_id",
        "cell_id",
        "site1_coverage",
        "site2_coverage",
        "paired_coverage",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Coverage figure source lacks columns: {sorted(missing)}")
    grouped = source.groupby(["acquisition_id", "fov_id"], dropna=False, sort=True)
    summary = grouped.agg(
        bundle_count=("bundle_id", "nunique"),
        cell_count=("cell_id", "nunique"),
        paired_median=("paired_coverage", "median"),
        paired_q1=("paired_coverage", lambda value: value.quantile(0.25)),
        paired_q3=("paired_coverage", lambda value: value.quantile(0.75)),
    ).reset_index()
    return summary.sort_values("paired_median", kind="stable").reset_index(drop=True)


def build_qc_coverage_hierarchy_figure(source: pd.DataFrame, output_stem: str | Path) -> list[Path]:
    """Plot bundle coverage distributions and acquisition/FOV technical heterogeneity."""

    group_summary = _coverage_group_summary(source)
    configure_publication_style()
    figure, (distribution, hierarchy) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.5),
        gridspec_kw={"width_ratios": (0.82, 1.45)},
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.23, wspace=0.25)

    coverage_columns = ("site1_coverage", "site2_coverage", "paired_coverage")
    coverage_labels = ("Site1", "Site2", "Paired")
    coverage_colors = (COLORS["site1"], COLORS["site2"], COLORS["neutral"])
    values = [
        pd.to_numeric(source[column], errors="coerce").dropna().to_numpy()
        for column in coverage_columns
    ]
    boxes = distribution.boxplot(
        values,
        tick_labels=coverage_labels,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        whiskerprops={"color": COLORS["neutral"]},
        capprops={"color": COLORS["neutral"]},
    )
    for patch, color in zip(boxes["boxes"], coverage_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.9)
    distribution.set_ylim(-0.03, 1.03)
    distribution.set_ylabel("Valid-frame fraction per bundle")
    distribution.set_title("A  Bundle coverage", loc="left", weight="bold")
    distribution.spines[["top", "right"]].set_visible(False)
    distribution.text(
        0.02,
        0.98,
        f"n = {source['bundle_id'].nunique():,} bundles",
        transform=distribution.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    rank = np.arange(1, len(group_summary) + 1)
    median = group_summary["paired_median"].to_numpy(float)
    sizes = 18 + 42 * np.sqrt(
        group_summary["cell_count"].to_numpy(float)
        / max(float(group_summary["cell_count"].max()), 1.0)
    )
    hierarchy.vlines(rank, group_summary["paired_q1"], group_summary["paired_q3"], color="#AAB4BE")
    hierarchy.scatter(rank, median, s=sizes, color=COLORS["neutral"], zorder=3)
    hierarchy.axhline(
        float(pd.to_numeric(source["paired_coverage"], errors="coerce").median()),
        color=COLORS["warning"],
        linestyle="--",
        linewidth=1,
        label="overall bundle median",
    )
    hierarchy.set_xlim(0.25, len(group_summary) + 0.75)
    hierarchy.set_ylim(-0.03, 1.03)
    hierarchy.set_xlabel("Ranked acquisition/FOV technical group")
    hierarchy.set_ylabel("Median paired coverage (IQR)")
    hierarchy.set_title("B  Technical-group heterogeneity", loc="left", weight="bold")
    hierarchy.spines[["top", "right"]].set_visible(False)
    hierarchy.legend(frameon=False, loc="upper left")
    hierarchy.text(
        0.99,
        0.03,
        (
            f"{len(group_summary)} acquisition/FOV pairs\n"
            f"{source['cell_id'].nunique():,} cells; point size tracks cell count"
        ),
        transform=hierarchy.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    _annotate_technical_scope(
        figure,
        "Coverage is a tracking-quality endpoint. Current acquisition and FOV IDs are paired "
        "one-to-one, so their contributions cannot be separated.",
    )
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def build_qc_missingness_asymmetry_figure(
    source: pd.DataFrame, output_stem: str | Path
) -> list[Path]:
    """Plot neutral channel-observation patterns overall and by acquisition."""

    required = {
        "level",
        "nominal_frame_rows",
        "site1_only_frames",
        "site2_only_frames",
        "both_missing_frames",
        "site1_minus_site2_valid_fraction",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missingness-asymmetry figure source lacks columns: {sorted(missing)}")
    overall_rows = source.loc[source["level"].eq("overall")]
    acquisition = source.loc[source["level"].eq("acquisition")].copy()
    if len(overall_rows) != 1 or acquisition.empty:
        raise ValueError("Missingness-asymmetry source needs one overall row and acquisition rows")
    overall = overall_rows.iloc[0]
    both_valid = int(overall["nominal_frame_rows"]) - sum(
        int(overall[column])
        for column in ("site1_only_frames", "site2_only_frames", "both_missing_frames")
    )
    pattern_counts = np.array(
        [
            both_valid,
            int(overall["site1_only_frames"]),
            int(overall["site2_only_frames"]),
            int(overall["both_missing_frames"]),
        ],
        dtype=float,
    )
    acquisition = acquisition.sort_values(
        "site1_minus_site2_valid_fraction", kind="stable"
    ).reset_index(drop=True)

    configure_publication_style()
    figure, (patterns, asymmetry) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.5),
        gridspec_kw={"width_ratios": (0.9, 1.35)},
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.23, wspace=0.25)
    labels = ("Both valid", "Site1 only", "Site2 only", "Both missing")
    colors = (COLORS["neutral"], COLORS["site1"], COLORS["site2"], COLORS["light"])
    bars = patterns.bar(np.arange(4), pattern_counts, color=colors, edgecolor=COLORS["neutral"])
    patterns.set_xticks(np.arange(4), labels, rotation=20, ha="right")
    patterns.set_ylabel("Nominal frame rows")
    patterns.set_title("A  Neutral observation pattern", loc="left", weight="bold")
    patterns.spines[["top", "right"]].set_visible(False)
    maximum = max(float(pattern_counts.max(initial=0)), 1.0)
    patterns.set_ylim(0, maximum * 1.18)
    for bar, count in zip(bars, pattern_counts, strict=True):
        patterns.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + maximum * 0.025,
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    difference = pd.to_numeric(
        acquisition["site1_minus_site2_valid_fraction"], errors="raise"
    ).to_numpy()
    rank = np.arange(1, len(acquisition) + 1)
    asymmetry.vlines(rank, 0, difference, color="#AAB4BE", linewidth=1)
    asymmetry.scatter(
        rank,
        difference,
        c=np.where(difference >= 0, COLORS["site1"], COLORS["site2"]),
        s=28,
        zorder=3,
    )
    asymmetry.axhline(0, color=COLORS["neutral"], linewidth=0.8)
    asymmetry.set_xlim(0.25, len(acquisition) + 0.75)
    asymmetry.set_xlabel("Ranked acquisition")
    asymmetry.set_ylabel("Site1 − Site2 valid-frame fraction")
    asymmetry.set_title("B  Acquisition-level asymmetry", loc="left", weight="bold")
    asymmetry.spines[["top", "right"]].set_visible(False)
    asymmetry.text(
        0.99,
        0.03,
        f"n = {len(acquisition)} acquisitions",
        transform=asymmetry.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    _annotate_technical_scope(
        figure,
        "Observation/missingness only: Site1/Site2 differences may reflect channel and tracking "
        "behavior and are not interpreted as locus biology.",
    )
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def build_qc_timing_heterogeneity_figure(
    source: pd.DataFrame, output_stem: str | Path
) -> list[Path]:
    """Plot acquisition-level exact frame-interval range and heterogeneity."""

    required = {
        "acquisition_id",
        "fov_id",
        "cell_count",
        "exact_interval_median_s",
        "exact_interval_min_s",
        "exact_interval_max_s",
        "exact_interval_max_min_ratio",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Timing figure source lacks columns: {sorted(missing)}")
    minimum_values = pd.to_numeric(source["exact_interval_min_s"], errors="coerce")
    if source.empty or minimum_values.isna().any() or (minimum_values <= 0).any():
        raise ValueError("Timing figure source requires finite positive exact intervals")

    interval = source.sort_values("exact_interval_median_s", kind="stable").reset_index(drop=True)
    ratio = source.sort_values("exact_interval_max_min_ratio", kind="stable").reset_index(drop=True)
    configure_publication_style()
    figure, (time_axis, ratio_axis) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.3),
        gridspec_kw={"width_ratios": (1.35, 1)},
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.24, wspace=0.27)

    rank = np.arange(1, len(interval) + 1)
    medians = pd.to_numeric(interval["exact_interval_median_s"], errors="raise").to_numpy()
    minimum = pd.to_numeric(interval["exact_interval_min_s"], errors="raise").to_numpy()
    maximum = pd.to_numeric(interval["exact_interval_max_s"], errors="raise").to_numpy()
    ratios = pd.to_numeric(interval["exact_interval_max_min_ratio"], errors="raise").to_numpy()
    point_colors = np.where(ratios > 1.5, COLORS["warning"], COLORS["neutral"])
    time_axis.vlines(rank, minimum, maximum, color="#AAB4BE", linewidth=1.2)
    time_axis.scatter(rank, medians, c=point_colors, s=30, zorder=3)
    time_axis.set_yscale("log")
    time_axis.set_xlim(0.25, len(interval) + 0.75)
    time_axis.set_xlabel("Acquisition ranked by median interval")
    time_axis.set_ylabel("Exact frame interval (s; log scale)")
    time_axis.set_title("A  Exact interval median and range", loc="left", weight="bold")
    time_axis.spines[["top", "right"]].set_visible(False)
    time_axis.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    time_axis.text(
        0.02,
        0.97,
        f"n = {len(interval)} acquisitions; vertical line = min–max",
        transform=time_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    ratio_rank = np.arange(1, len(ratio) + 1)
    ratio_values = pd.to_numeric(ratio["exact_interval_max_min_ratio"], errors="raise").to_numpy()
    ratio_axis.vlines(ratio_rank, 1, ratio_values, color="#AAB4BE", linewidth=1.2)
    ratio_axis.scatter(
        ratio_rank,
        ratio_values,
        c=np.where(ratio_values > 1.5, COLORS["warning"], COLORS["neutral"]),
        s=30,
        zorder=3,
    )
    ratio_axis.axhline(1.5, color=COLORS["site1"], linestyle="--", linewidth=1, label="ratio 1.5")
    ratio_axis.axhline(2.0, color=COLORS["warning"], linestyle=":", linewidth=1, label="ratio 2")
    ratio_axis.set_xlim(0.25, len(ratio) + 0.75)
    ratio_axis.set_ylim(0.9, max(2.15, float(ratio_values.max()) * 1.08))
    ratio_axis.set_xlabel("Acquisition ranked by max/min ratio")
    ratio_axis.set_ylabel("Exact interval max/min")
    ratio_axis.set_title("B  Within-acquisition heterogeneity", loc="left", weight="bold")
    ratio_axis.spines[["top", "right"]].set_visible(False)
    ratio_axis.legend(frameon=False, loc="upper left")
    ratio_axis.text(
        0.98,
        0.97,
        f">1.5: {(ratio_values > 1.5).sum()}/{len(ratio_values)}\n"
        f">2: {(ratio_values > 2).sum()}/{len(ratio_values)}",
        transform=ratio_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    _annotate_technical_scope(
        figure,
        "Acquisitions are the displayed units; frame and crop rows are not replicates. Exact "
        "timestamps are retained for downstream lag and velocity calculations.",
    )
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def build_qc_sensitivity_figure(source: pd.DataFrame, output_stem: str | Path) -> list[Path]:
    """Plot the configured three-way technical cohort-selection sensitivity grid."""

    required = {
        "min_paired_coverage",
        "min_paired_valid_frames",
        "min_longest_paired_run",
        "included_bundles",
        "inclusion_fraction",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"QC-sensitivity figure source lacks columns: {sorted(missing)}")
    coverage = sorted(pd.to_numeric(source["min_paired_coverage"], errors="raise").unique())
    paired = sorted(pd.to_numeric(source["min_paired_valid_frames"], errors="raise").unique())
    runs = sorted(pd.to_numeric(source["min_longest_paired_run"], errors="raise").unique())
    if not coverage or not paired or not runs:
        raise ValueError("QC-sensitivity figure source is empty")

    configure_publication_style()
    figure, axes = plt.subplots(1, len(runs), figsize=(10.2, 3.8), squeeze=False)
    figure.subplots_adjust(left=0.07, right=0.91, top=0.83, bottom=0.25, wspace=0.30)
    axes_row = axes[0]
    image = None
    for axis, run in zip(axes_row, runs, strict=True):
        block = source.loc[source["min_longest_paired_run"].eq(run)]
        fraction = (
            block.pivot(
                index="min_paired_valid_frames",
                columns="min_paired_coverage",
                values="inclusion_fraction",
            )
            .reindex(index=paired, columns=coverage)
            .to_numpy(float)
        )
        counts = (
            block.pivot(
                index="min_paired_valid_frames",
                columns="min_paired_coverage",
                values="included_bundles",
            )
            .reindex(index=paired, columns=coverage)
            .to_numpy(float)
        )
        image = axis.imshow(
            fraction,
            vmin=0,
            vmax=max(float(source["inclusion_fraction"].max()), 0.01),
            cmap="viridis",
            aspect="auto",
        )
        axis.set_xticks(np.arange(len(coverage)), [f"{value:.1f}" for value in coverage])
        axis.set_yticks(np.arange(len(paired)), [f"{int(value)}" for value in paired])
        axis.set_xlabel("Minimum paired coverage")
        axis.set_title(f"Contiguous run ≥ {int(run)} frames", weight="bold")
        for row_index in range(len(paired)):
            for column_index in range(len(coverage)):
                axis.text(
                    column_index,
                    row_index,
                    f"{int(counts[row_index, column_index]):,}\n"
                    f"{fraction[row_index, column_index]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if fraction[row_index, column_index] < 0.10 else "black",
                    fontsize=7.5,
                )
    axes_row[0].set_ylabel("Minimum paired valid frames")
    if image is not None:
        colorbar_axis = figure.add_axes((0.93, 0.25, 0.015, 0.58))
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Included fraction")
    figure.suptitle("T03E  Technical cohort-selection sensitivity", weight="bold", y=0.96)
    _annotate_technical_scope(
        figure,
        "Cells in the grid show included bundles and eligible-bundle fraction. Site1/Site2 "
        "minimum valid frames remain fixed at 20; this is not a biological robustness test.",
    )
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def build_qc_motion_dropout_figure(
    motion_source: pd.DataFrame,
    dropout_source: pd.DataFrame,
    output_stem: str | Path,
) -> list[Path]:
    """Plot acquisition-level motion/coverage and next-frame loss diagnostics."""

    motion_required = {"motion_metric", "spearman_rho", "supported", "acquisition_id"}
    dropout_required = {
        "analysis",
        "channel",
        "cell_mean_risk_difference",
        "supported",
        "acquisition_id",
    }
    if not motion_required.issubset(motion_source.columns):
        raise ValueError(
            "Motion diagnostic figure source lacks columns: "
            f"{sorted(motion_required.difference(motion_source.columns))}"
        )
    if not dropout_required.issubset(dropout_source.columns):
        raise ValueError(
            "Dropout diagnostic figure source lacks columns: "
            f"{sorted(dropout_required.difference(dropout_source.columns))}"
        )
    motion = motion_source.loc[
        motion_source["supported"].eq(True) & motion_source["spearman_rho"].notna()
    ].copy()
    dropout = dropout_source.loc[
        dropout_source["supported"].eq(True) & dropout_source["cell_mean_risk_difference"].notna()
    ].copy()
    if motion.empty and dropout.empty:
        raise ValueError("Motion/dropout diagnostic figure has no supported acquisition estimates")

    configure_publication_style()
    figure, (motion_axis, loss_axis) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    figure.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.25, wspace=0.28)

    metric_order = [
        metric
        for metric in ("site1", "site2", "common_mode")
        if metric in set(motion["motion_metric"])
    ]
    metric_labels = {"site1": "Site1", "site2": "Site2", "common_mode": "Common mode"}
    metric_colors = {
        "site1": COLORS["site1"],
        "site2": COLORS["site2"],
        "common_mode": COLORS["neutral"],
    }
    for index, metric in enumerate(metric_order):
        values = pd.to_numeric(
            motion.loc[motion["motion_metric"].eq(metric), "spearman_rho"], errors="raise"
        ).to_numpy()
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
        motion_axis.scatter(
            index + jitter,
            values,
            s=25,
            color=metric_colors[metric],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        motion_axis.plot(
            [index - 0.22, index + 0.22],
            [np.median(values), np.median(values)],
            color="black",
            linewidth=1.4,
            zorder=4,
        )
        motion_axis.text(
            index,
            0.04,
            f"n={len(values)} acq.",
            ha="center",
            va="bottom",
            fontsize=7.5,
            transform=motion_axis.get_xaxis_transform(),
        )
    motion_axis.axhline(0, color="#AAB4BE", linewidth=0.8)
    motion_axis.set_xticks(range(len(metric_order)), [metric_labels[item] for item in metric_order])
    motion_axis.set_xlim(-0.5, max(len(metric_order) - 0.5, 0.5))
    motion_axis.set_ylim(-1.12, 1.05)
    motion_axis.set_ylabel("Within-acquisition Spearman rho")
    motion_axis.set_title("A  Coverage vs exact-time speed", loc="left", weight="bold")
    motion_axis.spines[["top", "right"]].set_visible(False)

    dropout_keys = [
        ("next_frame_channel_loss_after_high_speed", "site1"),
        ("next_frame_channel_loss_after_high_speed", "site2"),
        ("next_frame_pair_loss_after_large_separation", "paired"),
    ]
    dropout_keys = [
        key
        for key in dropout_keys
        if ((dropout["analysis"] == key[0]) & (dropout["channel"] == key[1])).any()
    ]
    dropout_labels = {
        ("next_frame_channel_loss_after_high_speed", "site1"): "Site1\nhigh speed",
        ("next_frame_channel_loss_after_high_speed", "site2"): "Site2\nhigh speed",
        ("next_frame_pair_loss_after_large_separation", "paired"): "Pair\nlarge separation",
    }
    dropout_colors = {
        ("next_frame_channel_loss_after_high_speed", "site1"): COLORS["site1"],
        ("next_frame_channel_loss_after_high_speed", "site2"): COLORS["site2"],
        ("next_frame_pair_loss_after_large_separation", "paired"): COLORS["neutral"],
    }
    all_loss_values: list[float] = []
    for index, key in enumerate(dropout_keys):
        selected = dropout.loc[
            dropout["analysis"].eq(key[0]) & dropout["channel"].eq(key[1]),
            "cell_mean_risk_difference",
        ]
        values = pd.to_numeric(selected, errors="raise").to_numpy()
        all_loss_values.extend(values.tolist())
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
        loss_axis.scatter(
            index + jitter,
            values,
            s=25,
            color=dropout_colors[key],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        loss_axis.plot(
            [index - 0.22, index + 0.22],
            [np.median(values), np.median(values)],
            color="black",
            linewidth=1.4,
            zorder=4,
        )
        loss_axis.text(
            index,
            0.04,
            f"n={len(values)} acq.",
            ha="center",
            va="bottom",
            fontsize=7.5,
            transform=loss_axis.get_xaxis_transform(),
        )
    loss_axis.axhline(0, color="#AAB4BE", linewidth=0.8)
    loss_axis.set_xticks(range(len(dropout_keys)), [dropout_labels[item] for item in dropout_keys])
    loss_axis.set_xlim(-0.5, max(len(dropout_keys) - 0.5, 0.5))
    if all_loss_values:
        bound = max(0.08, max(abs(value) for value in all_loss_values) * 1.22)
        loss_axis.set_ylim(-bound, bound)
    loss_axis.set_ylabel("Cell-balanced loss-risk difference\n(high tail − reference)")
    loss_axis.set_title("B  Next-frame tracking loss", loc="left", weight="bold")
    loss_axis.spines[["top", "right"]].set_visible(False)

    _annotate_technical_scope(
        figure,
        "Each point is one supported acquisition; black line is the median. Predictors are "
        "defined within acquisition. This is a technical missingness screen, not biological or "
        "causal evidence.",
    )
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def write_initial_qc_figures(
    *,
    cohort_flow_source: pd.DataFrame,
    coverage_source: pd.DataFrame,
    missingness_asymmetry_source: pd.DataFrame,
    timing_hierarchy_source: pd.DataFrame,
    qc_sensitivity_source: pd.DataFrame,
    motion_coverage_source: pd.DataFrame,
    dropout_risk_source: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the complete T03 figure set and exact source CSVs.

    The API intentionally accepts only already-derived analysis tables. It never
    discovers or opens production/archive files.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "cohort_flow_source": output / "fig_t03a_cohort_flow_source.csv",
        "coverage_hierarchy_source": output / "fig_t03b_coverage_hierarchy_source.csv",
        "missingness_asymmetry_source": output / "fig_t03c_missingness_asymmetry_source.csv",
        "timing_heterogeneity_source": output / "fig_t03d_timing_heterogeneity_source.csv",
        "qc_sensitivity_source": output / "fig_t03e_qc_sensitivity_source.csv",
        "motion_coverage_source": output / "fig_t03f_motion_coverage_source.csv",
        "dropout_risk_source": output / "fig_t03f_dropout_risk_source.csv",
    }
    cohort_flow_source.to_csv(sources["cohort_flow_source"], index=False)
    coverage_source.to_csv(sources["coverage_hierarchy_source"], index=False)
    missingness_asymmetry_source.to_csv(sources["missingness_asymmetry_source"], index=False)
    timing_hierarchy_source.to_csv(sources["timing_heterogeneity_source"], index=False)
    qc_sensitivity_source.to_csv(sources["qc_sensitivity_source"], index=False)
    motion_coverage_source.to_csv(sources["motion_coverage_source"], index=False)
    dropout_risk_source.to_csv(sources["dropout_risk_source"], index=False)

    figure_sets = {
        "cohort_flow": build_qc_cohort_flow_figure(
            cohort_flow_source, output / "fig_t03a_cohort_flow"
        ),
        "coverage_hierarchy": build_qc_coverage_hierarchy_figure(
            coverage_source, output / "fig_t03b_coverage_hierarchy"
        ),
        "missingness_asymmetry": build_qc_missingness_asymmetry_figure(
            missingness_asymmetry_source, output / "fig_t03c_missingness_asymmetry"
        ),
        "qc_sensitivity": build_qc_sensitivity_figure(
            qc_sensitivity_source, output / "fig_t03e_qc_sensitivity"
        ),
    }
    if not timing_hierarchy_source.empty:
        figure_sets["timing_heterogeneity"] = build_qc_timing_heterogeneity_figure(
            timing_hierarchy_source, output / "fig_t03d_timing_heterogeneity"
        )
    supported_motion = bool(
        not motion_coverage_source.empty and motion_coverage_source["supported"].eq(True).any()
    )
    supported_dropout = bool(
        not dropout_risk_source.empty and dropout_risk_source["supported"].eq(True).any()
    )
    if supported_motion or supported_dropout:
        figure_sets["motion_dropout_diagnostics"] = build_qc_motion_dropout_figure(
            motion_coverage_source,
            dropout_risk_source,
            output / "fig_t03f_motion_dropout_diagnostics",
        )
    paths: dict[str, Path] = dict(sources)
    for figure_name, outputs in figure_sets.items():
        for path in outputs:
            paths[f"{figure_name}_{path.suffix.lstrip('.')}"] = path
    return paths
