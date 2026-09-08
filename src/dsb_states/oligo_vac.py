"""Oligo-LiveFISH-style multi-delta velocity autocorrelation analysis.

The implementation follows Oligo-LiveFISH equations 7--9 for single-locus
VAC.  It deliberately does not use the Rouse numerical-table WLS code in the
local VVCF folder: that code estimates two-locus VCC communication time.
Coordinates remain on the exact acquisition frame schedule and are never
interpolated or compressed across missing observations.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.optimize import minimize_scalar

from .physical_metrics import velocity_autocorrelation_by_frame_steps
from .time_resolved_physics import (
    _base_identity,
    _build_acquisition_schedules,
    _dense_coordinates,
    _matched_measurement_delta,
    _median_velocity_duration,
)
from .time_resolved_physics_summary import fbm_normalized_vac


@dataclass(frozen=True)
class OligoVacResult:
    """Per-unit curves, summaries, fits, and explicit support accounting."""

    unit_curves: pd.DataFrame
    support_census: pd.DataFrame
    raw_hour_curves: pd.DataFrame
    scaled_delta_hour_curves: pd.DataFrame
    collapsed_unit_curves: pd.DataFrame
    collapsed_hour_curves: pd.DataFrame
    fits: pd.DataFrame


def _positive_targets(values: tuple[float, ...]) -> tuple[float, ...]:
    targets = tuple(sorted({float(value) for value in values}))
    if not targets or not np.all(np.isfinite(targets)) or min(targets) <= 0.0:
        raise ValueError("target delta seconds must be finite and positive")
    return targets


def _centers_with_zero(values: tuple[float, ...], *, name: str) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if (
        centers.ndim != 1
        or centers.size < 2
        or not np.all(np.isfinite(centers))
        or not np.isclose(centers[0], 0.0)
        or np.any(np.diff(centers) <= 0.0)
    ):
        raise ValueError(f"{name} must start at zero and increase strictly")
    return centers


def _nearest_center(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distance = np.abs(values[:, None] - centers[None, :])
    return centers[np.argmin(distance, axis=1)]


def _nearest_unit_bins(
    unit_curves: pd.DataFrame,
    *,
    value_column: str,
    centers: np.ndarray,
    bin_column: str,
) -> pd.DataFrame:
    selected = unit_curves.loc[
        np.isfinite(pd.to_numeric(unit_curves[value_column], errors="coerce"))
    ].copy()
    values = pd.to_numeric(selected[value_column], errors="coerce").to_numpy(float)
    selected[bin_column] = _nearest_center(values, centers)
    selected["bin_distance"] = np.abs(values - selected[bin_column].to_numpy(float))
    keys = [
        "hour_post_delivery",
        "site",
        "unit_id",
        "target_delta_s",
        bin_column,
    ]
    return (
        selected.sort_values(
            keys + ["bin_distance", "pair_count", "lag_frames"],
            ascending=[True] * len(keys) + [True, False, True],
            kind="stable",
        )
        .drop_duplicates(keys, keep="first")
        .drop(columns="bin_distance")
        .reset_index(drop=True)
    )


def _sample_sd(values: pd.Series) -> float:
    return float(np.std(values.to_numpy(float), ddof=1)) if len(values) > 1 else float("nan")


def _aggregate_delta_curves(source: pd.DataFrame, *, bin_column: str) -> pd.DataFrame:
    group = ["hour_post_delivery", "site", "target_delta_s", bin_column]
    return (
        source.groupby(group, as_index=False, dropna=False)
        .agg(
            x_mean=(
                "lag_median_s" if bin_column == "raw_lag_bin_s" else "scaled_lag",
                "mean",
            ),
            mean=("normalized_vac", "mean"),
            sd=("normalized_vac", _sample_sd),
            median=("normalized_vac", "median"),
            n_units=("unit_id", "nunique"),
            n_crops=("crop_id", "nunique"),
            n_acquisitions=("nd2_id", "nunique"),
            total_velocity_pairs=("pair_count", "sum"),
            actual_delta_s_mean=("actual_delta_s", "mean"),
            actual_delta_s_min=("actual_delta_s", "min"),
            actual_delta_s_max=("actual_delta_s", "max"),
        )
        .sort_values(group, kind="stable")
        .reset_index(drop=True)
    )


def _fit_alpha(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    bounds: tuple[float, float],
) -> tuple[float, float] | None:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (x > 0.0) & (weights > 0.0)
    if np.count_nonzero(valid) < 3:
        return None
    x_fit = x[valid]
    y_fit = y[valid]
    weight = weights[valid] / np.sum(weights[valid])

    def objective(alpha: float) -> float:
        residual = y_fit - fbm_normalized_vac(x_fit, alpha)
        return float(np.sum(weight * residual**2))

    fit = minimize_scalar(objective, method="bounded", bounds=bounds)
    if not fit.success:
        return None
    alpha = float(fit.x)
    prediction = fbm_normalized_vac(x_fit, alpha)
    rmse = float(np.sqrt(np.average((y_fit - prediction) ** 2, weights=weight)))
    return alpha, rmse


def _bootstrap_alpha(
    source: pd.DataFrame,
    *,
    bounds: tuple[float, float],
    iterations: int,
    seed: int,
) -> np.ndarray:
    crops = sorted(source["crop_id"].astype(str).unique())
    if len(crops) < 2 or iterations < 1:
        return np.array([], dtype=float)
    by_crop = {crop: source.loc[source["crop_id"].astype(str).eq(crop)] for crop in crops}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        pieces: list[pd.DataFrame] = []
        for replicate_index, crop in enumerate(rng.choice(crops, size=len(crops), replace=True)):
            piece = by_crop[str(crop)].copy()
            piece["bootstrap_unit"] = (
                piece["unit_id"].astype(str) + f"|bootstrap{replicate_index:05d}"
            )
            pieces.append(piece)
        replicate = pd.concat(pieces, ignore_index=True)
        unit = replicate.groupby(
            ["bootstrap_unit", "scaled_lag_bin", "crop_id"], as_index=False
        ).agg(normalized_vac=("normalized_vac", "mean"))
        curve = (
            unit.groupby("scaled_lag_bin", as_index=False)
            .agg(mean=("normalized_vac", "mean"), n=("bootstrap_unit", "nunique"))
            .sort_values("scaled_lag_bin", kind="stable")
        )
        fitted = _fit_alpha(
            curve["scaled_lag_bin"].to_numpy(float),
            curve["mean"].to_numpy(float),
            curve["n"].to_numpy(float),
            bounds=bounds,
        )
        if fitted is not None:
            estimates.append(fitted[0])
    return np.asarray(estimates, dtype=float)


def compute_oligo_vac(
    bundle_frames: pd.DataFrame,
    bundle_index: pd.DataFrame,
    *,
    target_delta_s: tuple[float, ...] = (10.0, 20.0, 40.0),
    delta_relative_tolerance: float = 0.25,
    maximum_scaled_lag: float = 2.5,
    minimum_velocity_pairs: int = 8,
    raw_lag_bin_centers_s: tuple[float, ...] = (
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        80.0,
        100.0,
    ),
    scaled_lag_bin_centers: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
    ),
    alpha_bounds: tuple[float, float] = (0.25, 1.0),
    bootstrap_iterations: int = 300,
    bootstrap_seed: int = 20260824,
) -> OligoVacResult:
    """Calculate multi-delta normalized VAC and fit the Eq. 9 collapse by hour."""

    targets = _positive_targets(target_delta_s)
    raw_centers = _centers_with_zero(raw_lag_bin_centers_s, name="raw lag centers")
    scaled_centers = _centers_with_zero(scaled_lag_bin_centers, name="scaled lag centers")
    tolerance = float(delta_relative_tolerance)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("delta_relative_tolerance must lie in [0, 1)")
    if maximum_scaled_lag <= 0.0:
        raise ValueError("maximum_scaled_lag must be positive")
    if minimum_velocity_pairs < 1:
        raise ValueError("minimum_velocity_pairs must be positive")
    lower_alpha, upper_alpha = (float(value) for value in alpha_bounds)
    if not 0.0 < lower_alpha < upper_alpha <= 2.0:
        raise ValueError("alpha bounds must be positive and increasing")

    schedules = _build_acquisition_schedules(
        bundle_frames,
        bundle_index,
        primary_delta_target_s=targets[0],
        primary_delta_acceptable_s=(
            targets[0] * (1.0 - tolerance),
            targets[0] * (1.0 + tolerance),
        ),
    )
    acquisition_delta: dict[tuple[str, float], tuple[int | None, float]] = {}
    for nd2_id, schedule in schedules.items():
        for target in targets:
            acquisition_delta[(nd2_id, target)] = _matched_measurement_delta(
                schedule.times_s,
                target_s=target,
                acceptable_s=(target * (1.0 - tolerance), target * (1.0 + tolerance)),
            )

    frame_groups = bundle_frames.groupby(bundle_frames["bundle_id"].astype(str), sort=False)
    unit_rows: list[dict[str, Any]] = []
    census_rows: list[dict[str, Any]] = []
    for record in bundle_index.sort_values("bundle_id", kind="stable").itertuples(index=False):
        bundle_id = str(record.bundle_id)
        nd2_id = str(record.nd2_id)
        schedule = schedules[nd2_id]
        rows = (
            frame_groups.get_group(bundle_id)
            if bundle_id in frame_groups.groups
            else bundle_frames.iloc[0:0]
        )
        site1, site2 = _dense_coordinates(rows, schedule)
        identity = _base_identity(record)
        for site, coordinates in (("site1", site1), ("site2", site2)):
            observed = int(np.count_nonzero(np.all(np.isfinite(coordinates), axis=1)))
            for target in targets:
                delta_frames, schedule_delta_s = acquisition_delta[(nd2_id, target)]
                if delta_frames is None:
                    census_rows.append(
                        {
                            **identity,
                            "site": site,
                            "target_delta_s": target,
                            "delta_frames": pd.NA,
                            "schedule_delta_s": schedule_delta_s,
                            "actual_delta_s": np.nan,
                            "observed_frames": observed,
                            "maximum_pair_count": 0,
                            "emitted_lag_count": 0,
                            "unit_included": False,
                            "exclusion_reason": "no_integer_delta_within_tolerance",
                        }
                    )
                    continue
                max_lag_frames = min(
                    schedule.frames.size - 1,
                    int(np.floor(maximum_scaled_lag * delta_frames + 1e-12)),
                )
                lag_frames = np.arange(max_lag_frames + 1, dtype=int)
                vac = velocity_autocorrelation_by_frame_steps(
                    coordinates,
                    times=schedule.times_s,
                    measurement_delta_frames=delta_frames,
                    lag_frames=lag_frames,
                    min_pairs=minimum_velocity_pairs,
                )
                actual_delta_s = _median_velocity_duration(
                    coordinates, schedule.times_s, delta_frames
                )
                finite = np.isfinite(vac.values) & np.isfinite(vac.start_time_lag_median)
                for index in np.flatnonzero(finite):
                    lag_s = float(vac.start_time_lag_median[index])
                    unit_rows.append(
                        {
                            **identity,
                            "site": site,
                            "target_delta_s": target,
                            "delta_frames": int(delta_frames),
                            "schedule_delta_s": float(schedule_delta_s),
                            "actual_delta_s": float(actual_delta_s),
                            "lag_frames": int(vac.lag_frames[index]),
                            "lag_median_s": lag_s,
                            "scaled_lag": lag_s / actual_delta_s,
                            "normalized_vac": float(vac.values[index]),
                            "raw_vac_um2_per_s2": float(vac.raw_values[index]),
                            "zero_lag_energy_um2_per_s2": float(vac.normalization),
                            "pair_count": int(vac.counts[index]),
                        }
                    )
                maximum_pairs = int(np.max(vac.counts)) if vac.counts.size else 0
                if np.any(finite):
                    reason = "included"
                elif maximum_pairs < minimum_velocity_pairs:
                    reason = "insufficient_velocity_pairs_all_lags"
                else:
                    reason = "nonpositive_or_unavailable_zero_lag_energy"
                census_rows.append(
                    {
                        **identity,
                        "site": site,
                        "target_delta_s": target,
                        "delta_frames": int(delta_frames),
                        "schedule_delta_s": float(schedule_delta_s),
                        "actual_delta_s": float(actual_delta_s),
                        "observed_frames": observed,
                        "maximum_pair_count": maximum_pairs,
                        "emitted_lag_count": int(np.count_nonzero(finite)),
                        "unit_included": bool(np.any(finite)),
                        "exclusion_reason": reason,
                    }
                )

    unit_curves = pd.DataFrame(unit_rows)
    support_census = pd.DataFrame(census_rows)
    if unit_curves.empty:
        raise ValueError("No supported VAC curve points were produced")
    raw_unit = _nearest_unit_bins(
        unit_curves,
        value_column="lag_median_s",
        centers=raw_centers,
        bin_column="raw_lag_bin_s",
    )
    scaled_unit = _nearest_unit_bins(
        unit_curves,
        value_column="scaled_lag",
        centers=scaled_centers,
        bin_column="scaled_lag_bin",
    )
    raw_hour = _aggregate_delta_curves(raw_unit, bin_column="raw_lag_bin_s")
    scaled_delta_hour = _aggregate_delta_curves(scaled_unit, bin_column="scaled_lag_bin")
    collapsed_unit = (
        scaled_unit.groupby(
            [
                "hour_post_delivery",
                "site",
                "unit_id",
                "bundle_id",
                "nd2_id",
                "crop_id",
                "fov_id",
                "scaled_lag_bin",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            normalized_vac=("normalized_vac", "mean"),
            n_supported_deltas=("target_delta_s", "nunique"),
            total_velocity_pairs=("pair_count", "sum"),
        )
        .sort_values(
            ["hour_post_delivery", "site", "unit_id", "scaled_lag_bin"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    collapsed_hour = (
        collapsed_unit.groupby(
            ["hour_post_delivery", "site", "scaled_lag_bin"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean=("normalized_vac", "mean"),
            sd=("normalized_vac", _sample_sd),
            median=("normalized_vac", "median"),
            n_units=("unit_id", "nunique"),
            n_crops=("crop_id", "nunique"),
            n_acquisitions=("nd2_id", "nunique"),
            mean_supported_deltas=("n_supported_deltas", "mean"),
        )
        .sort_values(["hour_post_delivery", "site", "scaled_lag_bin"], kind="stable")
        .reset_index(drop=True)
    )

    fit_rows: list[dict[str, Any]] = []
    for (hour, site), curve in collapsed_hour.groupby(["hour_post_delivery", "site"], sort=True):
        fitted = _fit_alpha(
            curve["scaled_lag_bin"].to_numpy(float),
            curve["mean"].to_numpy(float),
            curve["n_units"].to_numpy(float),
            bounds=(lower_alpha, upper_alpha),
        )
        if fitted is None:
            continue
        units = collapsed_unit.loc[
            np.isclose(collapsed_unit["hour_post_delivery"], float(hour))
            & collapsed_unit["site"].eq(site)
        ]
        seed = int((bootstrap_seed + zlib.crc32(f"{hour:g}|{site}".encode())) % (2**32 - 1))
        bootstrap = _bootstrap_alpha(
            units,
            bounds=(lower_alpha, upper_alpha),
            iterations=bootstrap_iterations,
            seed=seed,
        )
        low = float(np.quantile(bootstrap, 0.025)) if bootstrap.size >= 20 else np.nan
        high = float(np.quantile(bootstrap, 0.975)) if bootstrap.size >= 20 else np.nan
        alpha, rmse = fitted
        fit_rows.append(
            {
                "hour_post_delivery": float(hour),
                "site": site,
                "alpha": alpha,
                "alpha_ci_low": low,
                "alpha_ci_high": high,
                "rmse": rmse,
                "n_fit_bins": int(np.count_nonzero(curve["scaled_lag_bin"].gt(0.0))),
                "n_units": int(units["unit_id"].nunique()),
                "n_crops": int(units["crop_id"].nunique()),
                "n_acquisitions": int(units["nd2_id"].nunique()),
                "bootstrap_successes": int(bootstrap.size),
                "fit_model": "Oligo-LiveFISH_Eq9_joint_multi_delta_collapse",
                "fit_bounds": f"[{lower_alpha:g},{upper_alpha:g}]",
                "fit_status": (
                    "descriptive_boundary_solution"
                    if alpha <= lower_alpha + 0.005 or alpha >= upper_alpha - 0.005
                    else "descriptive_interior_solution"
                ),
            }
        )
    fits = pd.DataFrame(fit_rows).sort_values(["hour_post_delivery", "site"], kind="stable")
    return OligoVacResult(
        unit_curves=unit_curves,
        support_census=support_census,
        raw_hour_curves=raw_hour,
        scaled_delta_hour_curves=scaled_delta_hour,
        collapsed_unit_curves=collapsed_unit,
        collapsed_hour_curves=collapsed_hour,
        fits=fits,
    )


def configure_oligo_vac_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "axes.linewidth": 2.0,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.fontsize": 13,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _site_label(site: str) -> str:
    return "Site 1" if site == "site1" else "Site 2"


def _site_color(site: str) -> str:
    return "#E600A9" if site == "site1" else "#15803D"


def _save_figure(figure: mpl.figure.Figure, stem: Path, *, dpi: int = 300) -> list[Path]:
    # ``Path.with_suffix`` treats the decimal part of an hour such as the
    # ``.5h_oligo_style`` in ``vac_1.5h_oligo_style`` as a suffix and silently
    # truncates the intended stem.  Appending extensions preserves decimal
    # folder-hour labels and keeps the manifest consistent with the filenames.
    outputs = [Path(f"{stem}{extension}") for extension in (".png", ".pdf", ".svg")]
    figure.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    figure.savefig(outputs[2], bbox_inches="tight")
    return outputs


def build_oligo_vac_hour_figures(
    result: OligoVacResult,
    msd_alphas: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_delta_s: tuple[float, ...],
    maximum_scaled_lag: float,
    raw_lag_limit_s: float,
    minimum_hour_bin_units: int = 8,
) -> dict[float, list[Path]]:
    """Write one Oligo-LiveFISH-style raw-lag/collapse figure per hour.

    The blue Eq. 9 curve uses the independently estimated MSD exponent, as in
    the source paper.  The VAC-only exponent in ``result.fits`` is retained as
    a model-consistency diagnostic and is deliberately not used for the
    primary theoretical overlay.
    """

    configure_oligo_vac_style()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    targets = _positive_targets(target_delta_s)
    required_alpha_columns = {"hour_post_delivery", "site", "alpha_msd"}
    missing_alpha_columns = sorted(required_alpha_columns.difference(msd_alphas.columns))
    if missing_alpha_columns:
        raise ValueError(f"msd_alphas is missing required columns: {missing_alpha_columns}")
    if minimum_hour_bin_units < 1:
        raise ValueError("minimum_hour_bin_units must be positive")
    color_map = mpl.colormaps["turbo"]
    normalization = Normalize(vmin=min(targets), vmax=max(targets))
    outputs: dict[float, list[Path]] = {}
    hours = sorted(result.raw_hour_curves["hour_post_delivery"].unique())
    for hour in hours:
        figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), sharey=True)
        for row, site in enumerate(("site1", "site2")):
            raw_axis, scaled_axis = axes[row]
            for axis in (raw_axis, scaled_axis):
                axis.axhline(0.0, color="#8A929A", linewidth=1.0, zorder=0)
                axis.set_ylim(-0.5, 1.05)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.tick_params(direction="out")
            raw = result.raw_hour_curves.loc[
                np.isclose(result.raw_hour_curves["hour_post_delivery"], hour)
                & result.raw_hour_curves["site"].eq(site)
            ]
            scaled = result.scaled_delta_hour_curves.loc[
                np.isclose(result.scaled_delta_hour_curves["hour_post_delivery"], hour)
                & result.scaled_delta_hour_curves["site"].eq(site)
            ]
            for target in targets:
                color = color_map(normalization(target))
                raw_curve = raw.loc[np.isclose(raw["target_delta_s"], target)].sort_values(
                    "raw_lag_bin_s", kind="stable"
                )
                scaled_curve = scaled.loc[np.isclose(scaled["target_delta_s"], target)].sort_values(
                    "scaled_lag_bin", kind="stable"
                )
                # Do not connect a group mean supported by only a handful of
                # trajectories at the long-lag tail.  This is display-only;
                # the full support tables and VAC-only diagnostic fit remain
                # unchanged.
                raw_curve = raw_curve.loc[raw_curve["n_units"].ge(minimum_hour_bin_units)]
                scaled_curve = scaled_curve.loc[
                    scaled_curve["n_units"].ge(minimum_hour_bin_units)
                ]
                raw_axis.plot(
                    raw_curve["x_mean"],
                    raw_curve["mean"],
                    color=color,
                    marker="o",
                    markersize=4.0,
                    linewidth=2.0,
                )
                scaled_axis.plot(
                    scaled_curve["x_mean"],
                    scaled_curve["mean"],
                    color=color,
                    marker="o",
                    markersize=4.0,
                    linewidth=2.0,
                )
            reference = msd_alphas.loc[
                np.isclose(msd_alphas["hour_post_delivery"], hour)
                & msd_alphas["site"].eq(site)
            ]
            if len(reference) == 1:
                reference_row = reference.iloc[0]
                alpha_msd = float(reference_row["alpha_msd"])
                grid = np.linspace(0.0, maximum_scaled_lag, 301)
                scaled_axis.plot(
                    grid,
                    fbm_normalized_vac(grid, alpha_msd),
                    color="#003BFF",
                    linewidth=3.0,
                    label="Eq. 9 from MSD",
                    zorder=10,
                )
                interval = ""
                if np.isfinite(reference_row.get("alpha_msd_ci_low", np.nan)) and np.isfinite(
                    reference_row.get("alpha_msd_ci_high", np.nan)
                ):
                    interval = (
                        "\n95% CI "
                        f"{reference_row['alpha_msd_ci_low']:.2f}-"
                        f"{reference_row['alpha_msd_ci_high']:.2f}"
                    )
                scaled_axis.text(
                    0.97,
                    0.94,
                    f"MSD alpha = {alpha_msd:.2f}{interval}",
                    transform=scaled_axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=14,
                )
            scaled_axis.axvline(1.0, color="#8A929A", linestyle=":", linewidth=1.3)
            raw_axis.set_xlim(0.0, raw_lag_limit_s)
            scaled_axis.set_xlim(0.0, maximum_scaled_lag)
            raw_axis.set_title(
                f"{_site_label(site)} VAC", color=_site_color(site), fontweight="bold"
            )
            scaled_axis.set_title(
                f"{_site_label(site)} VAC - rescaled",
                color=_site_color(site),
                fontweight="bold",
            )
            raw_axis.set_xlabel("Lag time, τ (s)")
            scaled_axis.set_xlabel("Rescaled lag, τ/δ")
            raw_axis.set_ylabel(r"$C_v^\delta(\tau)\,/\,C_v^\delta(0)$")
            if row == 0:
                scaled_axis.legend(frameon=False, loc="lower right")
        figure.suptitle(f"{hour:g} h after delivery", fontsize=21, fontweight="bold", y=0.995)
        figure.subplots_adjust(
            left=0.09, right=0.88, bottom=0.09, top=0.92, wspace=0.24, hspace=0.34
        )
        colorbar_axis = figure.add_axes((0.91, 0.20, 0.022, 0.62))
        scalar = mpl.cm.ScalarMappable(norm=normalization, cmap=color_map)
        colorbar = figure.colorbar(scalar, cax=colorbar_axis, ticks=list(targets))
        colorbar.set_label("δ (s)", fontsize=17)
        colorbar.ax.tick_params(labelsize=14, width=1.5, length=5)
        stem = destination / f"vac_{hour:g}h_oligo_style"
        outputs[float(hour)] = _save_figure(figure, stem)
        plt.close(figure)
    return outputs


def build_vac_msd_alpha_consistency_figure(
    vac_fits: pd.DataFrame,
    msd_alphas: pd.DataFrame,
    output_stem: str | Path,
) -> list[Path]:
    """Compare the primary MSD exponent with the VAC-only diagnostic fit."""

    configure_oligo_vac_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharex=True, sharey=True)
    for axis, site in zip(axes, ("site1", "site2"), strict=True):
        msd = msd_alphas.loc[msd_alphas["site"].eq(site)].sort_values(
            "hour_post_delivery", kind="stable"
        )
        vac = vac_fits.loc[vac_fits["site"].eq(site)].sort_values(
            "hour_post_delivery", kind="stable"
        )
        x_msd = msd["hour_post_delivery"].to_numpy(float)
        y_msd = msd["alpha_msd"].to_numpy(float)
        msd_low = msd.get("alpha_msd_ci_low", pd.Series(np.nan, index=msd.index)).to_numpy(float)
        msd_high = msd.get(
            "alpha_msd_ci_high", pd.Series(np.nan, index=msd.index)
        ).to_numpy(float)
        axis.errorbar(
            x_msd,
            y_msd,
            yerr=np.vstack((y_msd - msd_low, msd_high - y_msd)),
            color=_site_color(site),
            marker="o",
            markersize=6.5,
            linewidth=2.0,
            capsize=3,
            label="MSD exponent (primary)",
        )
        x_vac = vac["hour_post_delivery"].to_numpy(float)
        y_vac = vac["alpha"].to_numpy(float)
        vac_low = vac["alpha_ci_low"].to_numpy(float)
        vac_high = vac["alpha_ci_high"].to_numpy(float)
        axis.errorbar(
            x_vac,
            y_vac,
            yerr=np.vstack((y_vac - vac_low, vac_high - y_vac)),
            color="#184E77",
            marker="s",
            markerfacecolor="white",
            markeredgewidth=1.5,
            markersize=6.5,
            linestyle="--",
            linewidth=1.8,
            capsize=3,
            label="VAC-only Eq. 9 fit (diagnostic)",
        )
        axis.axhline(1.0, color="#8A929A", linewidth=1.0, linestyle=":")
        axis.set_title(_site_label(site), color=_site_color(site), fontweight="bold")
        axis.set_xlabel("Folder-hour after delivery")
        axis.set_ylim(0.2, 1.3)
        axis.set_xticks(sorted(msd_alphas["hour_post_delivery"].unique()))
        axis.tick_params(axis="x", labelrotation=45)
        for tick_label in axis.get_xticklabels():
            tick_label.set_horizontalalignment("right")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out")
    axes[0].set_ylabel("Exponent, α")
    axes[0].legend(frameon=False, loc="best", fontsize=11)
    figure.tight_layout()
    outputs = _save_figure(figure, Path(output_stem))
    plt.close(figure)
    return outputs


def build_oligo_vac_alpha_figure(
    fits: pd.DataFrame,
    output_stem: str | Path,
) -> list[Path]:
    """Plot the joint multi-delta Eq. 9 shape parameter by folder-hour."""

    configure_oligo_vac_style()
    figure, axis = plt.subplots(figsize=(7.8, 5.4))
    for site, marker in (("site1", "o"), ("site2", "s")):
        selected = fits.loc[fits["site"].eq(site)].sort_values("hour_post_delivery", kind="stable")
        x = selected["hour_post_delivery"].to_numpy(float)
        y = selected["alpha"].to_numpy(float)
        low = selected["alpha_ci_low"].to_numpy(float)
        high = selected["alpha_ci_high"].to_numpy(float)
        yerr = np.vstack((y - low, high - y))
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            color=_site_color(site),
            marker=marker,
            markersize=7,
            linewidth=2.2,
            capsize=3,
            label=_site_label(site),
        )
    axis.set_xlabel("Folder-hour after delivery")
    axis.set_ylabel("fBM alpha (multi-delta VAC)")
    axis.set_ylim(0.2, 1.05)
    axis.set_xticks(sorted(fits["hour_post_delivery"].unique()))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")
    axis.legend(frameon=False, ncol=2, loc="best")
    figure.tight_layout()
    outputs = _save_figure(figure, Path(output_stem))
    plt.close(figure)
    return outputs
