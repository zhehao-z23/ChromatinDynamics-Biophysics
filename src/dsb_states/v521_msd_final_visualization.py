"""Compact final MSD summary and complete-coverage alpha case selection.

The population display reuses the frozen v5.2.1 time-resolved MSD tables.  The
single-trajectory alpha screen is deliberately limited to the previously frozen
100%-coverage paired cohort and uses the same raw 10--50 s power-law convention.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SITE_COLORS = {"site1": "#F4B400", "site2": "#7A3DB8"}
SITE_LABELS = {"site1": "Site1", "site2": "Site2"}


def configure_style() -> None:
    """Apply a compact Arial/Science-like style suitable for direct PPT use."""

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.5,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def fit_trajectory_power_law(
    curve: pd.DataFrame,
    *,
    lower_s: float = 10.0,
    upper_s: float = 50.0,
    minimum_pairs: int = 8,
    minimum_points: int = 5,
    minimum_span_fold: float = 4.0,
) -> dict[str, Any] | None:
    """Fit raw single-trajectory MSD = A*tau**alpha in log space.

    Endpoint-pair counts are regression weights.  No coordinate interpolation,
    localization offset, or motion-blur term is introduced.
    """

    lag = pd.to_numeric(curve["lag_median_s"], errors="coerce").to_numpy(float)
    value = pd.to_numeric(curve["value"], errors="coerce").to_numpy(float)
    pairs = pd.to_numeric(curve["pair_count"], errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(lag)
        & np.isfinite(value)
        & np.isfinite(pairs)
        & (lag >= float(lower_s))
        & (lag <= float(upper_s))
        & (value > 0.0)
        & (pairs >= int(minimum_pairs))
    )
    if np.count_nonzero(keep) < int(minimum_points):
        return None
    x_lag = lag[keep]
    span = float(np.max(x_lag) / np.min(x_lag))
    if span < float(minimum_span_fold):
        return None
    x = np.log10(x_lag)
    y = np.log10(value[keep])
    weights = pairs[keep]
    design = np.column_stack([np.ones(len(x)), x])
    root_weight = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(
        design * root_weight[:, np.newaxis], y * root_weight, rcond=None
    )
    prediction = design @ coefficients
    weighted_mean = float(np.average(y, weights=weights))
    denominator = float(np.sum(weights * (y - weighted_mean) ** 2))
    numerator = float(np.sum(weights * (y - prediction) ** 2))
    r2 = 1.0 - numerator / denominator if denominator > 0.0 else np.nan
    return {
        "alpha": float(coefficients[1]),
        "amplitude_at_1s_um2": float(10.0 ** coefficients[0]),
        "r2_log": r2,
        "n_fit_lags": len(x),
        "fit_lag_min_s": float(np.min(x_lag)),
        "fit_lag_max_s": float(np.max(x_lag)),
        "fit_span_fold": span,
        "fit_endpoint_pairs": int(np.sum(weights)),
    }


def build_complete_coverage_alpha_table(
    unit_curves: pd.DataFrame,
    eligible_complete_pairs: pd.DataFrame,
    *,
    lower_s: float = 10.0,
    upper_s: float = 50.0,
    minimum_pairs: int = 8,
    minimum_points: int = 5,
    minimum_span_fold: float = 4.0,
) -> pd.DataFrame:
    """Fit Site1 and Site2 separately within the frozen complete-pair cohort."""

    required = {
        "bundle_id",
        "metric",
        "site",
        "lag_median_s",
        "value",
        "pair_count",
    }
    missing = sorted(required.difference(unit_curves.columns))
    if missing:
        raise ValueError(f"unit_curves missing columns: {missing}")
    metadata_columns = [
        "bundle_id",
        "nd2_id",
        "crop_id",
        "allele_index",
        "fov_id",
        "hour_post_delivery",
        "movie_frames",
        "frame_interval_s_production",
        "mean_separation_nm",
        "site1_coverage",
        "site2_coverage",
        "paired_coverage",
    ]
    missing_metadata = sorted(set(metadata_columns).difference(eligible_complete_pairs.columns))
    if missing_metadata:
        raise ValueError(f"eligible_complete_pairs missing columns: {missing_metadata}")
    metadata = eligible_complete_pairs[metadata_columns].copy()
    if not (
        metadata[["site1_coverage", "site2_coverage", "paired_coverage"]]
        .astype(float)
        .eq(1.0)
        .all(axis=None)
    ):
        raise ValueError("The supplied eligible cohort is not exactly 100% covered")
    eligible_ids = set(metadata["bundle_id"].astype(str))
    source = unit_curves.loc[
        unit_curves["metric"].astype(str).eq("msd")
        & unit_curves["bundle_id"].astype(str).isin(eligible_ids)
        & unit_curves["site"].astype(str).isin(["site1", "site2"])
    ].copy()
    rows: list[dict[str, Any]] = []
    for (bundle_id, site), curve in source.groupby(["bundle_id", "site"], sort=True):
        fit = fit_trajectory_power_law(
            curve,
            lower_s=lower_s,
            upper_s=upper_s,
            minimum_pairs=minimum_pairs,
            minimum_points=minimum_points,
            minimum_span_fold=minimum_span_fold,
        )
        if fit is not None:
            rows.append({"bundle_id": str(bundle_id), "site": str(site), **fit})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.merge(metadata, on="bundle_id", how="left", validate="many_to_one")
    result["distance_to_alpha_1"] = (result["alpha"] - 1.0).abs()
    result["distance_to_alpha_0_5"] = (result["alpha"] - 0.5).abs()
    return result.sort_values(
        ["hour_post_delivery", "bundle_id", "site"], kind="stable"
    ).reset_index(drop=True)


def select_alpha_cases(
    alpha_table: pd.DataFrame,
    *,
    primary_hour: float = 3.0,
    targets: Iterable[tuple[str, float]] = (("alpha1", 1.0), ("alpha05", 0.5)),
    minimum_r2: float = 0.9,
) -> pd.DataFrame:
    """Select one unique complete-pair bundle per target alpha."""

    eligible = alpha_table.loc[
        np.isclose(alpha_table["hour_post_delivery"].astype(float), float(primary_hour))
        & alpha_table["r2_log"].astype(float).ge(float(minimum_r2))
    ].copy()
    if eligible.empty:
        raise ValueError(f"No supported alpha candidates at {primary_hour:g} h")
    used: set[str] = set()
    selected: list[pd.Series] = []
    for case_id, target in targets:
        ranked = eligible.loc[~eligible["bundle_id"].astype(str).isin(used)].copy()
        ranked["target_alpha"] = float(target)
        ranked["distance_to_target"] = (ranked["alpha"] - float(target)).abs()
        ranked = ranked.sort_values(
            ["distance_to_target", "r2_log", "fit_endpoint_pairs"],
            ascending=[True, False, False],
            kind="stable",
        )
        if ranked.empty:
            raise ValueError(f"No unused candidate for {case_id}")
        row = ranked.iloc[0].copy()
        row["case_id"] = str(case_id)
        row["display_label"] = f"alpha about {target:g} ({SITE_LABELS[str(row['site'])]})"
        selected.append(row)
        used.add(str(row["bundle_id"]))
    return pd.DataFrame(selected).reset_index(drop=True)


def select_top_alpha_cases(
    alpha_table: pd.DataFrame,
    *,
    primary_hour: float = 3.0,
    site: str = "site1",
    top_n: int = 5,
    minimum_r2: float = 0.9,
) -> pd.DataFrame:
    """Rank supported complete-coverage trajectories by descending fitted alpha."""

    required = {"bundle_id", "site", "hour_post_delivery", "alpha", "r2_log"}
    missing = sorted(required.difference(alpha_table.columns))
    if missing:
        raise ValueError(f"alpha table missing columns: {missing}")
    if int(top_n) < 1:
        raise ValueError("top_n must be positive")
    eligible = alpha_table.loc[
        alpha_table["site"].astype(str).eq(str(site))
        & np.isclose(alpha_table["hour_post_delivery"].astype(float), float(primary_hour))
        & (pd.to_numeric(alpha_table["r2_log"], errors="coerce") >= float(minimum_r2))
        & np.isfinite(pd.to_numeric(alpha_table["alpha"], errors="coerce"))
    ].copy()
    if len(eligible) < int(top_n):
        raise ValueError(
            f"Only {len(eligible)} supported {site} cases at {primary_hour:g} h; "
            f"cannot select top {top_n}"
        )
    sort_columns = ["alpha", "r2_log"]
    ascending = [False, False]
    if "fit_endpoint_pairs" in eligible.columns:
        sort_columns.append("fit_endpoint_pairs")
        ascending.append(False)
    selected = eligible.sort_values(
        sort_columns,
        ascending=ascending,
        kind="stable",
    ).head(int(top_n))
    selected = selected.reset_index(drop=True)
    selected.insert(0, "alpha_rank", np.arange(1, len(selected) + 1, dtype=int))
    selected.insert(1, "case_id", [f"alpha_top{rank:02d}" for rank in selected["alpha_rank"]])
    return selected


def fitted_log_window_mean(
    amplitude_at_1s: float,
    alpha: float,
    lower_s: float,
    upper_s: float,
) -> float:
    """Mean fitted MSD over a log-lag window, retaining units of square microns."""

    amplitude = float(amplitude_at_1s)
    exponent = float(alpha)
    lower = float(lower_s)
    upper = float(upper_s)
    if amplitude <= 0.0 or lower <= 0.0 or upper <= lower:
        raise ValueError("Invalid fitted log-window arguments")
    if abs(exponent) < 1e-12:
        return amplitude
    return float(
        amplitude
        * (upper**exponent - lower**exponent)
        / (exponent * np.log(upper / lower))
    )


def _population_fit(curve: pd.DataFrame) -> dict[str, float] | None:
    lag = pd.to_numeric(curve["lag"], errors="coerce").to_numpy(float)
    value = pd.to_numeric(curve["mean"], errors="coerce").to_numpy(float)
    count = pd.to_numeric(curve["n"], errors="coerce").to_numpy(float)
    keep = np.isfinite(lag) & np.isfinite(value) & np.isfinite(count) & (value > 0) & (count > 0)
    if np.count_nonzero(keep) < 5:
        return None
    x = np.log(lag[keep])
    y = np.log(value[keep])
    root_weight = np.sqrt(count[keep])
    slope, intercept = np.polyfit(x, y, 1, w=root_weight)
    return {"alpha": float(slope), "amplitude_at_1s": float(np.exp(intercept))}


def build_window_mean_summary(
    fits: pd.DataFrame,
    primary_unit_lag_bins: pd.DataFrame,
    *,
    lower_s: float = 10.0,
    upper_s: float = 50.0,
    bootstrap_iterations: int = 500,
    bootstrap_seed: int = 20260823,
) -> pd.DataFrame:
    """Summarize the complete fitted 10--50 s MSD curve by folder-hour.

    The point estimate is the log-lag mean of the frozen population fit.  CIs
    resample crop clusters and jointly refit amplitude and alpha.
    """

    msd_fits = fits.loc[fits["metric"].astype(str).eq("msd")].copy()
    pivot = msd_fits.pivot_table(
        index=["site", "hour_post_delivery"],
        columns="estimate_name",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    required = {"site", "hour_post_delivery", "amplitude_at_1s", "exponent"}
    missing = sorted(required.difference(pivot.columns))
    if missing:
        raise ValueError(f"fits missing population parameters: {missing}")
    source = primary_unit_lag_bins.loc[
        primary_unit_lag_bins["metric"].astype(str).eq("msd")
        & primary_unit_lag_bins["site"].astype(str).isin(["site1", "site2"])
        & primary_unit_lag_bins["lag_bin_center_s"].astype(float).between(
            float(lower_s), float(upper_s), inclusive="both"
        )
    ].copy()
    rows: list[dict[str, Any]] = []
    for record in pivot.itertuples(index=False):
        site = str(record.site)
        hour = float(record.hour_post_delivery)
        group = source.loc[
            source["site"].astype(str).eq(site)
            & np.isclose(source["hour_post_delivery"].astype(float), hour)
        ].copy()
        clusters = sorted(group["crop_id"].astype(str).unique())
        bootstrap: list[float] = []
        if len(clusters) >= 2 and int(bootstrap_iterations) > 0:
            by_crop = {
                crop: group.loc[group["crop_id"].astype(str).eq(crop)] for crop in clusters
            }
            seed = int(
                (int(bootstrap_seed) + zlib.crc32(f"msd-window|{site}|{hour:g}".encode()))
                % (2**32 - 1)
            )
            generator = np.random.default_rng(seed)
            for _ in range(int(bootstrap_iterations)):
                sampled = generator.choice(clusters, size=len(clusters), replace=True)
                replicate = pd.concat([by_crop[crop] for crop in sampled], ignore_index=True)
                curve = (
                    replicate.groupby("lag_bin_center_s", as_index=False)
                    .agg(mean=("value", "mean"), n=("value", "size"))
                    .rename(columns={"lag_bin_center_s": "lag"})
                )
                fitted = _population_fit(curve)
                if fitted is not None:
                    bootstrap.append(
                        fitted_log_window_mean(
                            fitted["amplitude_at_1s"],
                            fitted["alpha"],
                            lower_s,
                            upper_s,
                        )
                    )
        finite = np.asarray(bootstrap, dtype=float)
        finite = finite[np.isfinite(finite)]
        low, high = (np.nan, np.nan)
        if finite.size >= 20:
            low, high = np.quantile(finite, [0.025, 0.975])
        rows.append(
            {
                "site": site,
                "hour_post_delivery": hour,
                "window_lower_s": float(lower_s),
                "window_upper_s": float(upper_s),
                "fitted_log_window_mean_msd_um2": fitted_log_window_mean(
                    float(record.amplitude_at_1s),
                    float(record.exponent),
                    lower_s,
                    upper_s,
                ),
                "ci_low": float(low),
                "ci_high": float(high),
                "n_units": int(group["unit_id"].nunique()),
                "n_crops": int(group["crop_id"].nunique()),
                "n_acquisitions": int(group["nd2_id"].nunique()),
                "bootstrap_successes": int(finite.size),
                "summary_definition": "mean fitted MSD over d(log lag) from 10 to 50 s",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["site", "hour_post_delivery"], kind="stable"
    ).reset_index(drop=True)


def _fit_record(fits: pd.DataFrame, hour: float, site: str, name: str) -> pd.Series:
    selected = fits.loc[
        fits["metric"].astype(str).eq("msd")
        & fits["site"].astype(str).eq(site)
        & np.isclose(fits["hour_post_delivery"].astype(float), float(hour))
        & fits["estimate_name"].astype(str).eq(name)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one {name} fit for {site}, {hour:g} h")
    return selected.iloc[0]


def build_final_msd_figure(
    hour_curves: pd.DataFrame,
    fits: pd.DataFrame,
    predicted_curves: pd.DataFrame,
    window_summary: pd.DataFrame,
    *,
    representative_hours: Iterable[float] = (1.5, 3.0, 10.0),
    lower_s: float = 10.0,
    upper_s: float = 50.0,
) -> mpl.figure.Figure:
    """Build three representative raw-fit panels plus one whole-window trend panel."""

    configure_style()
    hours = tuple(float(value) for value in representative_hours)
    if len(hours) != 3:
        raise ValueError("The compact final layout requires exactly three representative hours")
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 5.4), constrained_layout=False)
    figure.subplots_adjust(left=0.062, right=0.99, bottom=0.23, top=0.90, wspace=0.34)
    panel_letters = "ABCD"
    for panel_index, (axis, hour) in enumerate(zip(axes[:3], hours, strict=True)):
        for site in ("site1", "site2"):
            curve = hour_curves.loc[
                hour_curves["metric"].astype(str).eq("msd")
                & hour_curves["site"].astype(str).eq(site)
                & np.isclose(hour_curves["hour_post_delivery"].astype(float), hour)
                & hour_curves["lag_bin_center_s"].astype(float).between(
                    float(lower_s), float(upper_s), inclusive="both"
                )
            ].sort_values("lag_s_mean")
            x = curve["lag_s_mean"].to_numpy(float)
            y = curve["mean"].to_numpy(float)
            sd = curve["sd"].to_numpy(float)
            color = SITE_COLORS[site]
            lower = np.maximum(y - sd, 0.004)
            upper = y + sd
            axis.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
            axis.plot(
                x,
                y,
                color=color,
                marker="o",
                markersize=5.5,
                linestyle="none",
                label=SITE_LABELS[site],
                zorder=3,
            )
            prediction = predicted_curves.loc[
                predicted_curves["metric"].astype(str).eq("msd")
                & predicted_curves["site"].astype(str).eq(site)
                & np.isclose(predicted_curves["hour_post_delivery"].astype(float), hour)
                & predicted_curves["x"].astype(float).between(
                    float(lower_s), float(upper_s), inclusive="both"
                )
            ].sort_values("x")
            axis.plot(
                prediction["x"],
                prediction["y"],
                color=color,
                linestyle="--",
                linewidth=2.0,
                zorder=4,
            )
            alpha = _fit_record(fits, hour, site, "exponent")
            axis.text(
                0.05,
                0.93 if site == "site1" else 0.83,
                rf"{SITE_LABELS[site]}  $\alpha$={float(alpha['estimate']):.2f}",
                color=color,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(9.2, 54.5)
        axis.set_ylim(0.008, 1.6)
        axis.set_xticks([10, 20, 50], labels=["10", "20", "50"])
        axis.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
        axis.set_title(f"{hour:g} h", pad=8)
        axis.set_xlabel("Lag time (s)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.18,
            1.08,
            panel_letters[panel_index],
            transform=axis.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
        )
    axes[0].set_ylabel(r"MSD ($\mu$m$^2$)")
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.7)

    trend = axes[3]
    for site in ("site1", "site2"):
        summary = window_summary.loc[window_summary["site"].astype(str).eq(site)].sort_values(
            "hour_post_delivery"
        )
        x = summary["hour_post_delivery"].to_numpy(float)
        y = summary["fitted_log_window_mean_msd_um2"].to_numpy(float)
        low = summary["ci_low"].to_numpy(float)
        high = summary["ci_high"].to_numpy(float)
        error = np.vstack([y - low, high - y])
        color = SITE_COLORS[site]
        trend.plot(x, y, color=color, linewidth=2.2, zorder=2)
        trend.errorbar(
            x,
            y,
            yerr=error,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3.0,
            markersize=6.0,
            label=SITE_LABELS[site],
            zorder=3,
        )
        sparse = np.isclose(x, 5.0)
        if np.any(sparse):
            trend.scatter(
                x[sparse],
                y[sparse],
                s=72,
                facecolor="white",
                edgecolor=color,
                linewidth=2.0,
                zorder=4,
            )
    trend.set_title("Whole-window trend", pad=8)
    trend.set_xlabel("Hour after delivery")
    trend.set_ylabel("10–50 s fitted window-mean\n" + r"MSD ($\mu$m$^2$)")
    trend.set_xticks([1.5, 2.5, 3.5, 4.5, 5, 10])
    trend.set_xticklabels(["1.5", "2.5", "3.5", "4.5", "5", "10"], rotation=35)
    trend.set_ylim(bottom=0.0)
    trend.spines[["top", "right"]].set_visible(False)
    trend.legend(frameon=False, loc="upper right", handlelength=1.7)
    trend.text(
        -0.18,
        1.08,
        panel_letters[3],
        transform=trend.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    trend.text(
        0.98,
        0.03,
        "Open marker: sparse 5 h Site1 support",
        transform=trend.transAxes,
        ha="right",
        va="bottom",
        fontsize=11.5,
        color="#555555",
    )
    figure.text(
        0.5,
        0.045,
        "Points/lines: equal-trajectory mean; shading: between-trajectory SD; dashed: raw 10–50 s power-law fit. No localization-error or motion-blur correction.",
        ha="center",
        va="bottom",
        fontsize=13.5,
        color="#3F4C5C",
    )
    return figure


def save_figure_set(figure: mpl.figure.Figure, stem: Path, *, dpi: int = 300) -> list[Path]:
    """Save one figure as PNG, PDF and editable-text SVG."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        figure.savefig(path, dpi=dpi if suffix == ".png" else None, facecolor="white")
        paths.append(path)
    return paths
