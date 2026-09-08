"""Final slide-ready shared-frame Site1/Site2 separation figure.

The displayed means and standard deviations are frame-weighted descriptive
summaries.  Hour contrasts use a crop-clustered OLS covariance with Holm
adjustment; they are technical, exploratory comparisons rather than biological-
replicate inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

from .plots import configure_publication_style, save_figure
from .v521_shared_frame_separation import SharedFrameSeparationResult

_BELOW_COLOR = "#9EB4C2"
_ABOVE_COLOR = "#D4573B"
_TEXT_COLOR = "#20252A"
_NEUTRAL_COLOR = "#435365"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_hour_mean_statistics(
    frame_distances: pd.DataFrame,
    *,
    display_hours: tuple[float, ...],
    reference_hour: float = 1.5,
    multiple_testing: str = "holm",
) -> pd.DataFrame:
    """Return frame summaries and crop-clustered hour contrasts."""

    required = {"hour_post_delivery", "separation_nm", "nd2_id", "crop_id"}
    if missing := required.difference(frame_distances.columns):
        raise ValueError(f"frame_distances is missing columns: {sorted(missing)}")
    hours = tuple(float(hour) for hour in display_hours)
    reference = float(reference_hour)
    correction = str(multiple_testing).lower()
    if correction not in {"none", "holm", "bonferroni"}:
        raise ValueError("multiple_testing must be one of: none, holm, bonferroni")
    if reference not in hours or len(set(hours)) != len(hours):
        raise ValueError("reference must occur once in unique display_hours")
    selected = frame_distances.loc[
        frame_distances["hour_post_delivery"].astype(float).isin(hours)
    ].copy()
    if selected.empty:
        raise ValueError("no frames are available for requested display hours")
    selected["separation_nm"] = pd.to_numeric(selected["separation_nm"], errors="coerce")
    if not np.isfinite(selected["separation_nm"].to_numpy(float)).all():
        raise ValueError("selected separations contain non-finite values")

    summaries = (
        selected.groupby("hour_post_delivery", sort=True)
        .agg(
            shared_frame_count=("separation_nm", "size"),
            mean_separation_nm=("separation_nm", "mean"),
            sd_separation_nm=("separation_nm", "std"),
            median_separation_nm=("separation_nm", "median"),
            crop_count=("crop_id", "nunique"),
            acquisition_count=("nd2_id", "nunique"),
        )
        .reindex(hours)
        .reset_index()
    )
    if summaries["shared_frame_count"].isna().any():
        unavailable = summaries.loc[
            summaries["shared_frame_count"].isna(), "hour_post_delivery"
        ].tolist()
        raise ValueError(f"requested hours are unavailable: {unavailable}")

    design = pd.DataFrame({"intercept": 1.0}, index=selected.index)
    comparison_hours = [hour for hour in hours if not np.isclose(hour, reference)]
    design_columns: list[str] = []
    for hour in comparison_hours:
        name = f"hour_{hour:g}"
        design[name] = np.isclose(
            selected["hour_post_delivery"].to_numpy(float), hour
        ).astype(float)
        design_columns.append(name)
    cluster = selected["nd2_id"].astype(str) + "|" + selected["crop_id"].astype(str)
    fit = sm.OLS(
        selected["separation_nm"].to_numpy(float),
        design[["intercept", *design_columns]].to_numpy(float),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster.to_numpy(), "use_correction": True},
    )
    raw_p = np.asarray(fit.pvalues[1:], dtype=float)
    adjusted_p = (
        raw_p.copy()
        if correction == "none"
        else multipletests(raw_p, method=correction)[1]
    )
    parameter_covariance = np.asarray(fit.cov_params(), dtype=float)
    mean_cluster_se: dict[float, float] = {}
    for hour in hours:
        contrast_vector = np.zeros(len(design_columns) + 1, dtype=float)
        contrast_vector[0] = 1.0
        if not np.isclose(hour, reference):
            contrast_vector[comparison_hours.index(hour) + 1] = 1.0
        variance = float(contrast_vector @ parameter_covariance @ contrast_vector)
        mean_cluster_se[hour] = math.sqrt(max(variance, 0.0))
    contrast = {
        hour: {
            "difference_vs_reference_nm": float(fit.params[index + 1]),
            "cluster_robust_se_nm": float(fit.bse[index + 1]),
            "mean_cluster_robust_se_nm": mean_cluster_se[hour],
            "p_raw": float(raw_p[index]),
            "p_adjusted": float(adjusted_p[index]),
        }
        for index, hour in enumerate(comparison_hours)
    }

    output_rows: list[dict[str, Any]] = []
    for record in summaries.to_dict(orient="records"):
        hour = float(record["hour_post_delivery"])
        if np.isclose(hour, reference):
            extra = {
                "reference_hour": reference,
                "difference_vs_reference_nm": 0.0,
                "cluster_robust_se_nm": np.nan,
                "mean_cluster_robust_se_nm": mean_cluster_se[hour],
                "p_raw": np.nan,
                "p_adjusted": np.nan,
                "multiple_testing": correction,
                "significance": "ref",
            }
        else:
            extra = {"reference_hour": reference, **contrast[hour]}
            extra["multiple_testing"] = correction
            p_value = float(extra["p_adjusted"])
            extra["significance"] = (
                "***"
                if p_value < 0.001
                else "**"
                if p_value < 0.01
                else "*"
                if p_value < 0.05
                else "ns"
            )
        output_rows.append({**record, **extra})
    return pd.DataFrame(output_rows)


def build_final_separation_ppt_figure(
    result: SharedFrameSeparationResult,
    statistics: pd.DataFrame,
    output_stem: str | Path,
    *,
    histogram_hours: tuple[float, float] = (1.5, 3.5),
    display_hours: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 10.0),
    display_max_nm: float = 2500.0,
) -> list[Path]:
    """Draw two distributions, the threshold trend and mean plus/minus frame SD."""

    threshold = float(result.summary.iloc[0]["threshold_nm"])
    width = float(result.method_contract["bin_width_nm"])
    if display_max_nm <= threshold or not np.isclose(
        display_max_nm / width, round(display_max_nm / width), atol=1e-12
    ):
        raise ValueError("display_max_nm must exceed threshold and lie on a bin edge")
    available = set(result.hour_summary["hour_post_delivery"].astype(float))
    requested = {*histogram_hours, *display_hours}
    if missing := sorted(requested.difference(available)):
        raise ValueError(f"requested display hours are unavailable: {missing}")
    corrections = statistics["multiple_testing"].astype(str).unique().tolist()
    if len(corrections) != 1 or corrections[0] not in {"none", "holm", "bonferroni"}:
        raise ValueError("statistics must contain one supported multiple-testing method")
    correction = corrections[0]
    correction_phrase = {
        "none": "unadjusted P values",
        "holm": "Holm correction",
        "bonferroni": "Bonferroni correction",
    }[correction]

    regular_edges = np.arange(0.0, display_max_nm + 0.5 * width, width)
    histogram_percent: dict[float, np.ndarray] = {}
    maximum_percent = 0.0
    for hour in histogram_hours:
        values = result.frame_distances.loc[
            np.isclose(result.frame_distances["hour_post_delivery"].astype(float), hour),
            "separation_nm",
        ].to_numpy(float)
        counts, _ = np.histogram(values[values < display_max_nm], bins=regular_edges)
        overflow = int(np.count_nonzero(values >= display_max_nm))
        percentages = 100.0 * np.append(counts, overflow) / len(values)
        histogram_percent[hour] = percentages
        maximum_percent = max(maximum_percent, float(percentages.max()))

    configure_publication_style()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 19,
            "xtick.labelsize": 15.5,
            "ytick.labelsize": 15.5,
            "legend.fontsize": 16,
            "axes.linewidth": 1.3,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(13.333, 7.50))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.06, 0.94),
        left=0.075,
        right=0.985,
        bottom=0.175,
        top=0.93,
        wspace=0.29,
        hspace=0.52,
    )

    bar_left = np.append(regular_edges[:-1], display_max_nm)
    colors = np.where(bar_left >= threshold, _ABOVE_COLOR, _BELOW_COLOR)
    y_upper = max(17.0, math.ceil(maximum_percent * 1.14))
    overflow_center = display_max_nm + width / 2.0
    x_upper = display_max_nm + width
    for index, hour in enumerate(histogram_hours):
        axis = figure.add_subplot(grid[0, index])
        axis.bar(
            bar_left,
            histogram_percent[hour],
            width=width,
            align="edge",
            color=colors,
            edgecolor="none",
            linewidth=0.0,
        )
        axis.axvline(
            threshold,
            color=_TEXT_COLOR,
            linewidth=1.2,
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
            pad=7,
        )
        axis.text(
            0.975,
            0.66,
            f"{record['percent_above_threshold']:.1f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=18,
            color=_TEXT_COLOR,
        )
        axis.legend(
            handles=[
                Patch(
                    facecolor=_BELOW_COLOR,
                    edgecolor="none",
                    label=f"\N{LESS-THAN OR EQUAL TO}{threshold:g} nm",
                ),
                Patch(
                    facecolor=_ABOVE_COLOR,
                    edgecolor="none",
                    label=f">{threshold:g} nm",
                ),
            ],
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.98),
            ncol=2,
            handlelength=0.82,
            handletextpad=0.38,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
        axis.set_xlim(0.0, x_upper)
        axis.set_ylim(0.0, y_upper)
        axis.set_xticks(
            [0.0, 500.0, 1000.0, 1500.0, 2000.0, overflow_center],
            ["0", "0.5k", "1k", "1.5k", "2k", ">=2.5k"],
        )
        axis.set_yticks(np.arange(0.0, y_upper + 0.1, 5.0))
        axis.set_xlabel("Separation (nm)", labelpad=5)
        axis.set_ylabel("Shared frames (%)", labelpad=5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=4.5, width=1.2)

    trend_axis = figure.add_subplot(grid[1, 0])
    trend = result.hour_summary.set_index("hour_post_delivery").loc[list(display_hours)].reset_index()
    positions = np.arange(len(trend))
    trend_values = trend["percent_above_threshold"].to_numpy(float)
    trend_axis.plot(
        positions,
        trend_values,
        color=_ABOVE_COLOR,
        linewidth=3.0,
        marker="o",
        markersize=7.5,
        markerfacecolor=_ABOVE_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.1,
        zorder=3,
    )
    label_offsets = np.array([1.25, 1.55, 1.25, 1.55, 1.25, 1.55, 1.25, 1.25])
    for position, value, offset in zip(
        positions, trend_values, label_offsets, strict=True
    ):
        trend_axis.text(
            position,
            value + offset,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14.5,
            color=_TEXT_COLOR,
        )
    trend_axis.set_title(
        f"C  Fraction above {threshold:g} nm", loc="left", weight="bold", pad=6
    )
    trend_axis.set_xlim(-0.38, len(trend) - 0.62)
    trend_axis.set_ylim(20.0, 47.0)
    trend_axis.set_xticks(positions, [f"{hour:g}" for hour in display_hours])
    trend_axis.set_yticks(np.arange(20.0, 46.0, 5.0))
    trend_axis.set_xlabel("Time after Cas9 delivery (h)")
    trend_axis.set_ylabel(f"Shared frames >{threshold:g} nm (%)")
    trend_axis.spines[["top", "right"]].set_visible(False)
    trend_axis.tick_params(direction="out", length=4.5, width=1.2)

    mean_axis = figure.add_subplot(grid[1, 1])
    stats = statistics.set_index("hour_post_delivery").loc[list(display_hours)].reset_index()
    mean_values = stats["mean_separation_nm"].to_numpy(float)
    sem_values = stats["mean_cluster_robust_se_nm"].to_numpy(float)
    mean_axis.bar(
        positions,
        mean_values,
        width=0.70,
        color=_ABOVE_COLOR,
        alpha=0.92,
        edgecolor="none",
        zorder=2,
    )
    mean_axis.errorbar(
        positions,
        mean_values,
        yerr=sem_values,
        fmt="none",
        ecolor=_TEXT_COLOR,
        elinewidth=1.5,
        capsize=3.5,
        capthick=1.5,
        zorder=3,
    )
    annotation_y = mean_values + sem_values + 24.0
    for position, label, y_value in zip(
        positions, stats["significance"].astype(str), annotation_y, strict=True
    ):
        if label == "ref":
            continue
        mean_axis.text(
            position,
            y_value,
            label,
            ha="center",
            va="bottom",
            fontsize=15,
            color=_TEXT_COLOR,
        )
    mean_axis.set_title(
        "D  Mean separation \N{PLUS-MINUS SIGN} cluster SEM",
        loc="left",
        weight="bold",
        pad=6,
    )
    mean_axis.set_xlim(-0.55, len(stats) - 0.45)
    mean_axis.set_ylim(0.0, math.ceil(float(annotation_y.max() + 70.0) / 100.0) * 100.0)
    mean_axis.set_xticks(positions, [f"{hour:g}" for hour in display_hours])
    mean_axis.set_xlabel("Time after Cas9 delivery (h)")
    mean_axis.set_ylabel("Separation (nm)")
    mean_axis.spines[["top", "right"]].set_visible(False)
    mean_axis.tick_params(direction="out", length=4.5, width=1.2)

    figure.text(
        0.075,
        0.022,
        "50-nm bins; \N{GREATER-THAN OR EQUAL TO}2.5 \N{MICRO SIGN}m values pooled. "
        "Bars: frame-weighted mean \N{PLUS-MINUS SIGN} ND2 x crop-clustered SEM.\n"
        f"Tests vs 1.5 h: crop-clustered OLS with {correction_phrase} "
        "(ns, P\N{GREATER-THAN OR EQUAL TO}0.05; **, P<0.01; ***, P<0.001).",
        ha="left",
        va="bottom",
        fontsize=16,
        color=_NEUTRAL_COLOR,
    )
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    configure_publication_style()
    return outputs


def write_final_separation_ppt_result(
    result: SharedFrameSeparationResult,
    output_dir: str | Path,
    *,
    source_paths: tuple[str | Path, ...],
    histogram_hours: tuple[float, float] = (1.5, 3.5),
    display_hours: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 10.0),
    multiple_testing: str = "holm",
) -> dict[str, Path]:
    """Write the final slide figure, statistics and audit metadata immutably."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {output}")
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    figures = staging / "figures"
    tables = staging / "tables"
    figures.mkdir()
    tables.mkdir()

    statistics = build_hour_mean_statistics(
        result.frame_distances,
        display_hours=display_hours,
        reference_hour=1.5,
        multiple_testing=multiple_testing,
    )
    stats_path = tables / "hour_mean_sd_significance_vs_1p5h.csv"
    statistics.to_csv(stats_path, index=False, lineterminator="\n")
    threshold_path = tables / "hour_threshold_summary.csv"
    result.hour_summary.to_csv(threshold_path, index=False, lineterminator="\n")
    figure_paths = build_final_separation_ppt_figure(
        result,
        statistics,
        figures / "fig_site1_site2_separation_final_ppt",
        histogram_hours=histogram_hours,
        display_hours=display_hours,
    )

    contract = {
        **result.method_contract,
        "schema_version": 2,
        "final_histogram_hours": list(histogram_hours),
        "final_trend_and_mean_hours": list(display_hours),
        "omitted_final_display_hour": 5.0,
        "mean_summary": "all recorded shared frames equal weight within folder-hour",
        "uncertainty": "one cluster-robust SEM for each hour mean; clusters are ND2 x crop",
        "hour_contrast_model": "OLS separation_nm ~ categorical folder-hour",
        "hour_contrast_reference": 1.5,
        "hour_contrast_covariance": "cluster robust by ND2 x crop identity",
        "multiple_testing": (
            "none; raw two-sided p values"
            if multiple_testing == "none"
            else f"{multiple_testing} across seven displayed non-reference hours"
        ),
        "inference_limit": "exploratory technical comparison; not biological-replicate inference",
    }
    contract_path = staging / "METHOD_CONTRACT.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path = staging / "REPORT.md"
    significance_lines = "\n".join(
        f"- {row.hour_post_delivery:g} h vs 1.5 h: mean {row.mean_separation_nm:.1f} nm, "
        f"SD {row.sd_separation_nm:.1f} nm, displayed P="
        f"{row.p_adjusted:.4g} ({row.significance})."
        for row in statistics.itertuples(index=False)
        if not np.isclose(row.hour_post_delivery, 1.5)
    )
    report_path.write_text(
        "# Final PPT separation figure\n\n"
        "The two distribution panels retain every recorded shared Site1/Site2 frame at 1.5 and "
        "3.5 h. The threshold trend and frame-weighted mean plus/minus cluster-robust SEM retain "
        "the "
        "previously "
        "frozen display hours and omit 5 h. All trend points use one marker style.\n\n"
        "## Exploratory comparisons to 1.5 h\n\n"
        f"{significance_lines}\n\n"
        f"P values use a crop-clustered covariance and multiple-testing method "
        f"'{multiple_testing}'. They quantify technical "
        "hour contrasts only; acquisitions are not biological replicates and folder-hour is not "
        "verified time since cutting.\n",
        encoding="utf-8",
    )

    sources = [Path(path).resolve() for path in source_paths]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    artifacts = sorted(path for path in staging.rglob("*") if path.is_file())
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
        "figure_png": output / figure_paths[0].relative_to(staging),
        "figure_pdf": output / figure_paths[1].relative_to(staging),
        "figure_svg": output / figure_paths[2].relative_to(staging),
        "statistics": output / stats_path.relative_to(staging),
        "threshold_summary": output / threshold_path.relative_to(staging),
        "method_contract": output / contract_path.relative_to(staging),
        "report": output / report_path.relative_to(staging),
        "manifest": output / manifest_path.relative_to(staging),
    }
