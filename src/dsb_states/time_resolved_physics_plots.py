"""Publication figures for the time-resolved MSD, MSCD, VAC and VCC tables.

The functions in this module are intentionally presentation-only.  They do
not pool trajectories, fit models, or silently replace missing time points.
Every plotted line is supplied by an upstream tidy table whose ``mean`` and
``sd`` columns describe equal-weight trajectory (or bundle) summaries.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from .plots import COLORS, configure_publication_style, save_figure

_CURVE_REQUIRED = {
    "metric",
    "site",
    "hour_post_delivery",
    "lag_bin_center_s",
    "lag_s_mean",
    "scaled_lag_mean",
    "mean",
    "sd",
    "q25",
    "q75",
    "n_units",
    "total_pairs",
}
_FIT_REQUIRED = {
    "metric",
    "site",
    "hour_post_delivery",
    "estimate_name",
    "estimate",
    "ci_low",
    "ci_high",
    "fit_status",
}
_PREDICTION_REQUIRED = {"metric", "site", "hour_post_delivery", "x", "y"}
_HOUR_ORDER = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0)
_SITE_COLORS = {
    "site1": COLORS["site1"],
    "site2": COLORS["site2"],
    "pair": COLORS["neutral"],
}


def _require_columns(table: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _metric_rows(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    return table.loc[table["metric"].astype(str).str.casefold().eq(metric.casefold())].copy()


def _ordered_hours(source: pd.DataFrame) -> list[float]:
    observed = set(
        pd.to_numeric(source["hour_post_delivery"], errors="coerce").dropna().astype(float)
    )
    ordered = [hour for hour in _HOUR_ORDER if hour in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    # Preserve the standard 3 x 3 macro-time layout when this is the known DSB
    # experiment; genuinely absent hours are shown explicitly as unsupported.
    if observed and observed.issubset(set(_HOUR_ORDER)):
        return list(_HOUR_ORDER)
    return ordered[:9]


def _hour_label(hour: float) -> str:
    return f"{hour:g} h"


def _site_label(site: str) -> str:
    return {"site1": "Site1", "site2": "Site2", "pair": "Site1/Site2"}.get(
        str(site).casefold(), str(site)
    )


def _site_color(site: str) -> str:
    return _SITE_COLORS.get(str(site).casefold(), COLORS["neutral"])


def _prediction_rows(
    predicted_curves: pd.DataFrame | None,
    *,
    metric: str,
    site: str,
    hour: float,
) -> pd.DataFrame:
    if predicted_curves is None or predicted_curves.empty:
        return pd.DataFrame(columns=sorted(_PREDICTION_REQUIRED))
    _require_columns(predicted_curves, _PREDICTION_REQUIRED, name="predicted_curves")
    return predicted_curves.loc[
        predicted_curves["metric"].astype(str).str.casefold().eq(metric.casefold())
        & predicted_curves["site"].astype(str).str.casefold().eq(str(site).casefold())
        & np.isclose(
            pd.to_numeric(predicted_curves["hour_post_delivery"], errors="coerce"), hour
        )
    ].sort_values("x", kind="stable")


def _numeric_xy(source: pd.DataFrame, x_column: str) -> pd.DataFrame:
    result = source.copy()
    for column in (x_column, "mean", "sd", "n_units", "total_pairs"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.loc[np.isfinite(result[x_column]) & np.isfinite(result["mean"])].sort_values(
        x_column, kind="stable"
    )


def _plot_mean_sd(
    axis: plt.Axes,
    source: pd.DataFrame,
    *,
    x_column: str,
    color: str,
    label: str | None,
    log_y: bool,
) -> None:
    data = _numeric_xy(source, x_column)
    if data.empty:
        return
    x = data[x_column].to_numpy(float)
    mean = data["mean"].to_numpy(float)
    sd = data["sd"].fillna(0.0).to_numpy(float)
    if log_y:
        positive = mean[np.isfinite(mean) & (mean > 0)]
        if not positive.size:
            return
        floor = max(float(np.min(positive)) * 0.25, np.finfo(float).tiny)
        lower = np.maximum(mean - sd, floor)
        upper = mean + sd
    else:
        lower = mean - sd
        upper = mean + sd
    axis.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)
    axis.plot(x, mean, color=color, linewidth=1.7, marker="o", markersize=2.8, label=label)


def _tail_annotation(axis: plt.Axes, source: pd.DataFrame, *, x_column: str) -> None:
    data = _numeric_xy(source, x_column)
    if data.empty:
        return
    tail = data.iloc[-1]
    n_units = int(tail["n_units"]) if np.isfinite(tail["n_units"]) else 0
    pairs = int(tail["total_pairs"]) if np.isfinite(tail["total_pairs"]) else 0
    axis.text(
        0.97,
        0.04,
        f"tail: n={n_units}; pairs={pairs:,}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=COLORS["neutral"],
    )


def _empty_panel(axis: plt.Axes) -> None:
    axis.text(
        0.5,
        0.5,
        "insufficient support",
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=8,
        color="#8A95A0",
    )


def _small_multiple_axes(hours: list[float], *, figsize: tuple[float, float] = (9.8, 8.0)):
    configure_publication_style()
    figure, axes = plt.subplots(3, 3, figsize=figsize, sharex=False, sharey=False)
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.ravel()[len(hours) :]:
        axis.axis("off")
    return figure, axes.ravel()


def _build_squared_displacement_curves(
    hour_curves: pd.DataFrame,
    *,
    metric: str,
    output_stem: str | Path,
    predicted_curves: pd.DataFrame | None,
    fit_window_s: tuple[float, float],
) -> list[Path]:
    _require_columns(hour_curves, _CURVE_REQUIRED, name="hour_curves")
    curves = _metric_rows(hour_curves, metric)
    hours = _ordered_hours(curves)
    if not hours:
        raise ValueError(f"hour_curves contains no {metric!r} rows")
    figure, axes = _small_multiple_axes(hours)
    sites = ["site1", "site2"] if metric == "msd" else ["pair"]

    for index, hour in enumerate(hours):
        axis = axes[index]
        axis.set_title(f"{chr(65 + index)}  {_hour_label(hour)}", loc="left", weight="bold")
        axis.axvspan(*fit_window_s, color="#DCE3E9", alpha=0.42, zorder=0)
        displayed = False
        tail_source = pd.DataFrame()
        for site in sites:
            selected = curves.loc[
                curves["site"].astype(str).str.casefold().eq(site)
                & np.isclose(
                    pd.to_numeric(curves["hour_post_delivery"], errors="coerce"), hour
                )
            ]
            if selected.empty:
                continue
            displayed = True
            tail_source = selected if tail_source.empty else tail_source
            _plot_mean_sd(
                axis,
                selected,
                x_column="lag_s_mean",
                color=_site_color(site),
                label=_site_label(site),
                log_y=True,
            )
            prediction = _prediction_rows(
                predicted_curves, metric=metric, site=site, hour=hour
            )
            if not prediction.empty:
                axis.plot(
                    pd.to_numeric(prediction["x"], errors="coerce"),
                    pd.to_numeric(prediction["y"], errors="coerce"),
                    color=_site_color(site),
                    linestyle="--",
                    linewidth=1.15,
                )
        if not displayed:
            _empty_panel(axis)
        else:
            _tail_annotation(axis, tail_source, x_column="lag_s_mean")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 3.0), numticks=5))
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
        axis.set_xlabel("Lag time, τ (s)")
        axis.set_ylabel(f"{metric.upper()} (µm²)")
        if index == 0 and displayed:
            axis.legend(frameon=False, loc="upper left")

    title_metric = "MSD: Site-specific motion" if metric == "msd" else "MSCD: relative-vector motion"
    figure.suptitle(
        f"{title_metric} by folder-hour",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.01,
        0.004,
        "Line = equal-weight unit mean; band = between-unit SD (display-clipped at zero on "
        f"log axes); grey = {fit_window_s[0]:g}-{fit_window_s[1]:g} s descriptive fit window.",
        fontsize=7.2,
        color=COLORS["neutral"],
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.965))
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def build_msd_curves_figure(
    hour_curves: pd.DataFrame,
    output_stem: str | Path,
    *,
    predicted_curves: pd.DataFrame | None = None,
    fit_window_s: tuple[float, float] = (10.0, 50.0),
) -> list[Path]:
    """Plot Site1/Site2 MSD small multiples with the descriptive fit window."""

    return _build_squared_displacement_curves(
        hour_curves,
        metric="msd",
        output_stem=output_stem,
        predicted_curves=predicted_curves,
        fit_window_s=fit_window_s,
    )


def build_mscd_curves_figure(
    hour_curves: pd.DataFrame,
    output_stem: str | Path,
    *,
    predicted_curves: pd.DataFrame | None = None,
    fit_window_s: tuple[float, float] = (10.0, 50.0),
) -> list[Path]:
    """Plot Site1/Site2 relative-vector MSCD small multiples."""

    return _build_squared_displacement_curves(
        hour_curves,
        metric="mscd",
        output_stem=output_stem,
        predicted_curves=predicted_curves,
        fit_window_s=fit_window_s,
    )


def _select_estimate(fits: pd.DataFrame, metric: str, estimate_token: str) -> pd.DataFrame:
    _require_columns(fits, _FIT_REQUIRED, name="fits")
    selected = _metric_rows(fits, metric)
    selected = selected.loc[
        selected["estimate_name"].astype(str).str.casefold().str.contains(estimate_token.casefold())
    ].copy()
    for column in ("hour_post_delivery", "estimate", "ci_low", "ci_high"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected.loc[np.isfinite(selected["estimate"])]


def _build_exponent_figure(
    fits: pd.DataFrame,
    *,
    metric: str,
    estimate_token: str,
    symbol: str,
    output_stem: str | Path,
) -> list[Path]:
    selected = _select_estimate(fits, metric, estimate_token)
    if selected.empty:
        raise ValueError(f"fits contains no supported {metric} {estimate_token} estimates")
    configure_publication_style()
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for site, group in selected.groupby("site", sort=False):
        group = group.sort_values("hour_post_delivery", kind="stable")
        estimate = group["estimate"].to_numpy(float)
        low = group["ci_low"].to_numpy(float)
        high = group["ci_high"].to_numpy(float)
        finite_ci = np.isfinite(low) & np.isfinite(high) & (low <= estimate) & (high >= estimate)
        errors = None
        if np.all(finite_ci):
            errors = np.vstack((estimate - low, high - estimate))
        axis.errorbar(
            group["hour_post_delivery"],
            estimate,
            yerr=errors,
            color=_site_color(str(site)),
            marker="o",
            markersize=5,
            linewidth=1.5,
            capsize=2.5,
            label=_site_label(str(site)),
        )
    axis.set_xlabel("Folder-hour after delivery")
    axis.set_ylabel(f"Fitted {symbol}")
    axis.set_title(f"{metric.upper()} fitted exponent across macro-time", loc="left", weight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    axis.text(
        0.99,
        0.02,
        "Points are descriptive hour-level fits; bars are supplied CIs.",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=COLORS["neutral"],
    )
    figure.tight_layout()
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def build_msd_alpha_figure(fits: pd.DataFrame, output_stem: str | Path) -> list[Path]:
    """Plot the descriptive MSD power-law alpha estimate by hour."""

    return _build_exponent_figure(
        fits,
        metric="msd",
        estimate_token="exponent",
        symbol="α",
        output_stem=output_stem,
    )


def build_mscd_beta_figure(fits: pd.DataFrame, output_stem: str | Path) -> list[Path]:
    """Plot the descriptive MSCD power-law beta estimate by hour."""

    return _build_exponent_figure(
        fits,
        metric="mscd",
        estimate_token="exponent",
        symbol="β",
        output_stem=output_stem,
    )


def _primary_matched_rows(source: pd.DataFrame) -> pd.DataFrame:
    if "is_primary_matched_10s" not in source.columns:
        return source
    mask = source["is_primary_matched_10s"].fillna(False).astype(bool)
    return source.loc[mask]


def build_vac_curves_figures(
    hour_curves: pd.DataFrame,
    output_dir: str | Path,
    *,
    predicted_curves: pd.DataFrame | None = None,
) -> dict[str, list[Path]]:
    """Write one 3 x 3 matched-delta VAC figure per site."""

    _require_columns(hour_curves, _CURVE_REQUIRED, name="hour_curves")
    curves = _primary_matched_rows(_metric_rows(hour_curves, "vac"))
    if curves.empty:
        raise ValueError("hour_curves contains no primary matched VAC rows")
    destination = Path(output_dir)
    outputs: dict[str, list[Path]] = {}
    for site in ("site1", "site2"):
        site_curves = curves.loc[curves["site"].astype(str).str.casefold().eq(site)]
        if site_curves.empty:
            continue
        hours = _ordered_hours(site_curves)
        figure, axes = _small_multiple_axes(hours)
        for index, hour in enumerate(hours):
            axis = axes[index]
            selected = site_curves.loc[
                np.isclose(
                    pd.to_numeric(site_curves["hour_post_delivery"], errors="coerce"), hour
                )
            ].copy()
            selected = selected.loc[
                pd.to_numeric(selected["scaled_lag_mean"], errors="coerce").between(
                    0.0, 5.1, inclusive="both"
                )
            ]
            axis.set_title(f"{chr(65 + index)}  {_hour_label(hour)}", loc="left", weight="bold")
            axis.axhline(0.0, color="#7E8994", linewidth=0.8)
            axis.axvline(1.0, color="#7E8994", linestyle=":", linewidth=1.0)
            if selected.empty:
                _empty_panel(axis)
            else:
                _plot_mean_sd(
                    axis,
                    selected,
                    x_column="scaled_lag_mean",
                    color=_site_color(site),
                    label=None,
                    log_y=False,
                )
                _tail_annotation(axis, selected, x_column="scaled_lag_mean")
                prediction = _prediction_rows(
                    predicted_curves, metric="vac", site=site, hour=hour
                )
                if not prediction.empty:
                    axis.plot(
                        pd.to_numeric(prediction["x"], errors="coerce"),
                        pd.to_numeric(prediction["y"], errors="coerce"),
                        color="#15191D",
                        linestyle="--",
                        linewidth=1.2,
                        label="fBM fit",
                    )
                    if index == 0:
                        axis.legend(frameon=False, loc="best")
            axis.set_xlabel("Scaled lag, τ/δ")
            axis.set_ylabel("Normalized VAC")
            axis.set_xlim(-0.05, 5.1)
        figure.suptitle(
            f"{_site_label(site)} VAC at acquisition-matched δ ≈ 10 s",
            fontsize=14,
            weight="bold",
            y=0.995,
        )
        figure.text(
            0.01,
            0.004,
            "Line = equal-weight trajectory mean; band = between-trajectory SD; dotted "
            "vertical line marks τ/δ = 1. fBM overlay is descriptive when supplied.",
            fontsize=7.2,
            color=COLORS["neutral"],
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.965))
        site_outputs = save_figure(figure, destination / f"vac_{site}_matched_10s")
        plt.close(figure)
        outputs[site] = site_outputs
    return outputs


def build_vac_alpha_figure(fits: pd.DataFrame, output_stem: str | Path) -> list[Path]:
    """Plot fitted fBM alpha for Site1 and Site2 VAC curves."""

    return _build_exponent_figure(
        fits,
        metric="vac",
        estimate_token="exponent",
        symbol="fBM α",
        output_stem=output_stem,
    )


def build_vcc_curves_figure(
    hour_curves: pd.DataFrame,
    output_stem: str | Path,
    *,
    predicted_curves: pd.DataFrame | None = None,
) -> list[Path]:
    """Plot the symmetric, rotation-invariant normalized VCC trace by hour."""

    _require_columns(hour_curves, _CURVE_REQUIRED, name="hour_curves")
    curves = _primary_matched_rows(_metric_rows(hour_curves, "vcc"))
    if curves.empty:
        raise ValueError("hour_curves contains no primary matched VCC rows")
    hours = _ordered_hours(curves)
    figure, axes = _small_multiple_axes(hours)
    for index, hour in enumerate(hours):
        axis = axes[index]
        selected = curves.loc[
            np.isclose(pd.to_numeric(curves["hour_post_delivery"], errors="coerce"), hour)
        ].copy()
        selected = selected.loc[
            pd.to_numeric(selected["scaled_lag_mean"], errors="coerce").between(
                0.0, 5.1, inclusive="both"
            )
        ]
        axis.set_title(f"{chr(65 + index)}  {_hour_label(hour)}", loc="left", weight="bold")
        axis.axhline(0.0, color="#7E8994", linewidth=0.8)
        axis.axvline(1.0, color="#7E8994", linestyle=":", linewidth=1.0)
        if selected.empty:
            _empty_panel(axis)
        else:
            _plot_mean_sd(
                axis,
                selected,
                x_column="scaled_lag_mean",
                color=COLORS["neutral"],
                label=None,
                log_y=False,
            )
            _tail_annotation(axis, selected, x_column="scaled_lag_mean")
            prediction = _prediction_rows(
                predicted_curves, metric="vcc", site="pair", hour=hour
            )
            if not prediction.empty:
                axis.plot(
                    pd.to_numeric(prediction["x"], errors="coerce"),
                    pd.to_numeric(prediction["y"], errors="coerce"),
                    color=COLORS["warning"],
                    linestyle="--",
                    linewidth=1.2,
                    label="smooth guide",
                )
                if index == 0:
                    axis.legend(frameon=False, loc="best")
        axis.set_xlabel("Scaled lag, τ/δ")
        axis.set_ylabel("Symmetric normalized VCC trace")
        axis.set_xlim(-0.05, 5.1)
    figure.suptitle(
        "Site1/Site2 velocity cross-correlation by folder-hour",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.01,
        0.004,
        "Display is the symmetric rotation-invariant trace summary; the full 2 x 2 VCC "
        "matrix remains the primary stored object. Any dashed curve is nonmechanistic.",
        fontsize=7.2,
        color=COLORS["neutral"],
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.965))
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def build_vcc_endpoints_figure(fits: pd.DataFrame, output_stem: str | Path) -> list[Path]:
    """Plot available descriptive VCC endpoints without forcing a mechanistic fit."""

    _require_columns(fits, _FIT_REQUIRED, name="fits")
    selected = _metric_rows(fits, "vcc")
    selected = selected.loc[
        selected["fit_status"].astype(str).str.casefold().str.startswith("descriptive")
    ].copy()
    selected["estimate"] = pd.to_numeric(selected["estimate"], errors="coerce")
    selected["hour_post_delivery"] = pd.to_numeric(
        selected["hour_post_delivery"], errors="coerce"
    )
    selected = selected.loc[np.isfinite(selected["estimate"])]
    if selected.empty:
        raise ValueError("fits contains no supported descriptive VCC endpoints")

    endpoint_names = list(dict.fromkeys(selected["estimate_name"].astype(str)))
    configure_publication_style()
    figure, axes = plt.subplots(
        1,
        len(endpoint_names),
        figsize=(max(4.0, 3.4 * len(endpoint_names)), 3.8),
        squeeze=False,
    )
    for axis, endpoint in zip(axes[0], endpoint_names, strict=True):
        block = selected.loc[selected["estimate_name"].astype(str).eq(endpoint)].sort_values(
            "hour_post_delivery", kind="stable"
        )
        estimate = block["estimate"].to_numpy(float)
        low = pd.to_numeric(block["ci_low"], errors="coerce").to_numpy(float)
        high = pd.to_numeric(block["ci_high"], errors="coerce").to_numpy(float)
        error = None
        if np.all(np.isfinite(low) & np.isfinite(high) & (low <= estimate) & (high >= estimate)):
            error = np.vstack((estimate - low, high - estimate))
        axis.errorbar(
            block["hour_post_delivery"],
            estimate,
            yerr=error,
            marker="o",
            color=COLORS["neutral"],
            linewidth=1.4,
            capsize=2.5,
        )
        axis.axhline(0.0, color="#AAB4BE", linewidth=0.8)
        axis.set_xlabel("Folder-hour after delivery")
        axis.set_ylabel(endpoint.replace("_", " "))
        axis.set_title(endpoint.replace("_", " "), loc="left", weight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("VCC descriptive endpoints", weight="bold", y=1.01)
    figure.text(
        0.01,
        0.005,
        "Descriptive summaries of the supplied VCC curves; no mechanistic relaxation model is implied.",
        fontsize=7.2,
        color=COLORS["neutral"],
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def write_time_resolved_physics_figures(
    *,
    hour_curves: pd.DataFrame,
    fits: pd.DataFrame,
    output_dir: str | Path,
    predicted_curves: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write the complete, separate-figure time-resolved physics set."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_sets: dict[str, list[Path]] = {
        "msd_curves": build_msd_curves_figure(
            hour_curves, output / "fig_msd_curves_by_hour", predicted_curves=predicted_curves
        ),
        "msd_alpha": build_msd_alpha_figure(fits, output / "fig_msd_alpha_by_hour"),
        "mscd_curves": build_mscd_curves_figure(
            hour_curves, output / "fig_mscd_curves_by_hour", predicted_curves=predicted_curves
        ),
        "mscd_beta": build_mscd_beta_figure(fits, output / "fig_mscd_beta_by_hour"),
        "vac_alpha": build_vac_alpha_figure(fits, output / "fig_vac_alpha_by_hour"),
        "vcc_curves": build_vcc_curves_figure(
            hour_curves, output / "fig_vcc_curves_by_hour", predicted_curves=predicted_curves
        ),
        "vcc_endpoints": build_vcc_endpoints_figure(
            fits, output / "fig_vcc_descriptive_endpoints_by_hour"
        ),
    }
    for site, paths in build_vac_curves_figures(
        hour_curves, output, predicted_curves=predicted_curves
    ).items():
        figure_sets[f"vac_curves_{site}"] = paths

    flattened: dict[str, Path] = {}
    for name, paths in figure_sets.items():
        for path in paths:
            flattened[f"{name}_{path.suffix.lstrip('.')}"] = path
    return flattened
