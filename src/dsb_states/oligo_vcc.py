"""Oligo-LiveFISH-style two-locus velocity cross-correlation analysis.

This module keeps the empirical VCC as a lag-indexed 2 x 2 tensor and uses
the symmetric trace only for the Rouse communication-time fit.  Coordinates
remain on the exact acquisition frame schedule; missing observations are never
zero-filled, interpolated, or compressed.
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
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from .physical_metrics import velocity_cross_correlation_matrix_by_frame_steps
from .time_resolved_physics import (
    _base_identity,
    _build_acquisition_schedules,
    _dense_coordinates,
    _matched_measurement_delta,
    _paired_median_velocity_duration,
)

_DIRECTIONS = ("site1_later_vs_site2", "site2_later_vs_site1")
_COMPONENTS = ("x", "y")


@dataclass(frozen=True)
class OligoVccResult:
    """Per-bundle VCC tensors, equal-bundle summaries, and Rouse fits."""

    directional_unit_curves: pd.DataFrame
    directional_unit_matrices: pd.DataFrame
    symmetric_unit_curves: pd.DataFrame
    symmetric_unit_matrices: pd.DataFrame
    support_census: pd.DataFrame
    binned_unit_curves: pd.DataFrame
    hour_curves: pd.DataFrame
    hour_matrix_curves: pd.DataFrame
    communication_time_fits: pd.DataFrame
    fit_predictions: pd.DataFrame
    alpha_sensitivity: pd.DataFrame


class RouseVccReference:
    """Linear interpolator for the corrected local numerical VVCF tables."""

    def __init__(
        self,
        alpha_grid: np.ndarray,
        log10_delta_over_tau_grid: np.ndarray,
        scaled_lag_grid: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.alpha_grid = np.asarray(alpha_grid, dtype=float)
        self.log10_delta_over_tau_grid = np.asarray(
            log10_delta_over_tau_grid, dtype=float
        )
        self.scaled_lag_grid = np.asarray(scaled_lag_grid, dtype=float)
        self.values = np.asarray(values, dtype=float)
        expected = (
            self.alpha_grid.size,
            self.log10_delta_over_tau_grid.size,
            self.scaled_lag_grid.size,
        )
        if self.values.shape != expected:
            raise ValueError(f"Rouse table shape {self.values.shape} != {expected}")
        self._interpolator = RegularGridInterpolator(
            (
                self.alpha_grid,
                self.log10_delta_over_tau_grid,
                self.scaled_lag_grid,
            ),
            self.values,
            method="linear",
            bounds_error=True,
        )

    @classmethod
    def from_directory(cls, tables_dir: str | Path) -> RouseVccReference:
        root = Path(tables_dir)
        alpha_grid = np.round(np.arange(0.25, 1.0001, 0.025), 3)
        blocks: list[np.ndarray] = []
        for alpha in alpha_grid:
            path = root / f"table{round(alpha * 1000):04d}.mat"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = loadmat(path)
            if "VAC_Table" not in payload:
                raise ValueError(f"{path} does not contain VAC_Table")
            table = np.asarray(payload["VAC_Table"], dtype=float)
            if table.shape != (25, 501) or not np.all(np.isfinite(table)):
                raise ValueError(f"Unexpected or non-finite numerical table: {path}")
            blocks.append(table)
        return cls(
            alpha_grid,
            np.linspace(-3.0, 3.0, 25),
            np.linspace(0.0, 5.0, 501),
            np.stack(blocks, axis=0),
        )

    def evaluate(
        self,
        alpha: float | np.ndarray,
        delta_over_tau: float | np.ndarray,
        scaled_lag: float | np.ndarray,
    ) -> np.ndarray:
        """Evaluate the corrected trilinear interpolation on broadcast inputs."""

        alpha_values, ratios, lags = np.broadcast_arrays(
            np.asarray(alpha, dtype=float),
            np.asarray(delta_over_tau, dtype=float),
            np.asarray(scaled_lag, dtype=float),
        )
        if (
            not np.all(np.isfinite(alpha_values))
            or not np.all(np.isfinite(ratios))
            or not np.all(np.isfinite(lags))
            or np.any(ratios <= 0.0)
        ):
            raise ValueError("Rouse interpolation inputs must be finite and delta/tau positive")
        points = np.column_stack(
            (
                alpha_values.ravel(),
                np.log10(ratios).ravel(),
                lags.ravel(),
            )
        )
        return self._interpolator(points).reshape(alpha_values.shape)


def _sample_sd(values: pd.Series) -> float:
    array = values.to_numpy(float)
    return float(np.std(array, ddof=1)) if array.size > 1 else float("nan")


def _positive_targets(values: tuple[float, ...]) -> tuple[float, ...]:
    targets = tuple(sorted({float(value) for value in values}))
    if not targets or not np.all(np.isfinite(targets)) or min(targets) <= 0.0:
        raise ValueError("target delta seconds must be finite and positive")
    return targets


def _scaled_centers(values: tuple[float, ...]) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if (
        centers.ndim != 1
        or centers.size < 2
        or not np.all(np.isfinite(centers))
        or not np.isclose(centers[0], 0.0)
        or np.any(np.diff(centers) <= 0.0)
    ):
        raise ValueError("scaled-lag centers must start at zero and increase strictly")
    return centers


def _nearest_binned_curve(source: pd.DataFrame, centers: np.ndarray) -> pd.DataFrame:
    selected = source.loc[
        np.isfinite(pd.to_numeric(source["scaled_lag"], errors="coerce"))
        & np.isfinite(pd.to_numeric(source["normalized_vcc"], errors="coerce"))
    ].copy()
    values = selected["scaled_lag"].to_numpy(float)
    distance = np.abs(values[:, None] - centers[None, :])
    selected["scaled_lag_bin"] = centers[np.argmin(distance, axis=1)]
    selected["bin_distance"] = np.min(distance, axis=1)
    keys = ["hour_post_delivery", "unit_id", "target_delta_s", "scaled_lag_bin"]
    return (
        selected.sort_values(
            keys + ["bin_distance", "pair_count_min", "lag_frames"],
            ascending=[True] * len(keys) + [True, False, True],
            kind="stable",
        )
        .drop_duplicates(keys, keep="first")
        .drop(columns="bin_distance")
        .reset_index(drop=True)
    )


def _aggregate_hour_curves(source: pd.DataFrame) -> pd.DataFrame:
    group = ["hour_post_delivery", "target_delta_s", "scaled_lag_bin"]
    return (
        source.groupby(group, as_index=False, dropna=False)
        .agg(
            scaled_lag_mean=("scaled_lag", "mean"),
            mean=("normalized_vcc", "mean"),
            sd=("normalized_vcc", _sample_sd),
            median=("normalized_vcc", "median"),
            n_units=("unit_id", "nunique"),
            n_crops=("crop_id", "nunique"),
            n_acquisitions=("nd2_id", "nunique"),
            total_velocity_pairs=("pair_count_min", "sum"),
            actual_delta_s_mean=("actual_delta_s", "mean"),
            actual_delta_s_min=("actual_delta_s", "min"),
            actual_delta_s_max=("actual_delta_s", "max"),
        )
        .sort_values(group, kind="stable")
        .reset_index(drop=True)
    )


def _symmetrize_curves(source: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "unit_id",
        "bundle_id",
        "cohort",
        "nd2_id",
        "crop_id",
        "fov_id",
        "hour_post_delivery",
        "target_delta_s",
        "delta_frames",
        "schedule_delta_s",
        "actual_delta_s",
        "lag_frames",
    ]
    counts = source.groupby(keys, dropna=False)["direction"].nunique()
    complete = counts.loc[counts.eq(2)].index
    selected = source.set_index(keys).loc[complete].reset_index()
    return (
        selected.groupby(keys, as_index=False, dropna=False)
        .agg(
            lag_median_s=("lag_median_s", "mean"),
            scaled_lag=("scaled_lag", "mean"),
            normalized_vcc=("normalized_vcc", "mean"),
            raw_vcc_um2_per_s2=("raw_vcc_um2_per_s2", "mean"),
            zero_lag_energy_um2_per_s2=("zero_lag_energy_um2_per_s2", "mean"),
            pair_count_min=("pair_count", "min"),
            direction_count=("direction", "nunique"),
        )
        .sort_values(["hour_post_delivery", "unit_id", "target_delta_s", "lag_frames"])
        .reset_index(drop=True)
    )


def _symmetrize_matrices(source: pd.DataFrame) -> pd.DataFrame:
    data = source.copy()
    reverse = data["direction"].eq("site2_later_vs_site1")
    old_row = data.loc[reverse, "row_component"].copy()
    data.loc[reverse, "row_component"] = data.loc[reverse, "column_component"].to_numpy()
    data.loc[reverse, "column_component"] = old_row.to_numpy()
    keys = [
        "unit_id",
        "bundle_id",
        "cohort",
        "nd2_id",
        "crop_id",
        "fov_id",
        "hour_post_delivery",
        "target_delta_s",
        "delta_frames",
        "schedule_delta_s",
        "actual_delta_s",
        "lag_frames",
        "row_component",
        "column_component",
    ]
    counts = data.groupby(keys, dropna=False)["direction"].nunique()
    complete = counts.loc[counts.eq(2)].index
    selected = data.set_index(keys).loc[complete].reset_index()
    return (
        selected.groupby(keys, as_index=False, dropna=False)
        .agg(
            lag_median_s=("lag_median_s", "mean"),
            scaled_lag=("scaled_lag", "mean"),
            normalized_value=("normalized_value", "mean"),
            raw_value_um2_per_s2=("raw_value_um2_per_s2", "mean"),
            pair_count_min=("pair_count", "min"),
            direction_count=("direction", "nunique"),
        )
        .sort_values(
            [
                "hour_post_delivery",
                "unit_id",
                "target_delta_s",
                "lag_frames",
                "row_component",
                "column_component",
            ]
        )
        .reset_index(drop=True)
    )


def _bin_and_aggregate_matrices(source: pd.DataFrame, centers: np.ndarray) -> pd.DataFrame:
    data = source.copy()
    scaled = data["scaled_lag"].to_numpy(float)
    distance = np.abs(scaled[:, None] - centers[None, :])
    data["scaled_lag_bin"] = centers[np.argmin(distance, axis=1)]
    data["bin_distance"] = np.min(distance, axis=1)
    keys = [
        "hour_post_delivery",
        "unit_id",
        "target_delta_s",
        "scaled_lag_bin",
        "row_component",
        "column_component",
    ]
    binned = (
        data.sort_values(
            keys + ["bin_distance", "pair_count_min", "lag_frames"],
            ascending=[True] * len(keys) + [True, False, True],
            kind="stable",
        )
        .drop_duplicates(keys, keep="first")
        .drop(columns="bin_distance")
    )
    return (
        binned.groupby(
            [
                "hour_post_delivery",
                "target_delta_s",
                "scaled_lag_bin",
                "row_component",
                "column_component",
            ],
            as_index=False,
        )
        .agg(
            mean=("normalized_value", "mean"),
            sd=("normalized_value", _sample_sd),
            n_units=("unit_id", "nunique"),
            n_crops=("crop_id", "nunique"),
            n_acquisitions=("nd2_id", "nunique"),
        )
        .sort_values(
            [
                "hour_post_delivery",
                "target_delta_s",
                "scaled_lag_bin",
                "row_component",
                "column_component",
            ]
        )
        .reset_index(drop=True)
    )


def _fit_tau_grid(
    curve: pd.DataFrame,
    reference: RouseVccReference,
    *,
    alpha: float,
    tau_grid_s: np.ndarray,
    minimum_units_per_bin: int,
) -> dict[str, Any] | None:
    selected = curve.loc[
        curve["n_units"].ge(minimum_units_per_bin)
        & curve["scaled_lag_bin"].between(0.0, 2.5)
        & np.isfinite(curve["mean"])
        & np.isfinite(curve["actual_delta_s_mean"])
    ].copy()
    supported_targets = selected.groupby("target_delta_s").size()
    targets = supported_targets.loc[supported_targets.ge(3)].index.to_numpy(float)
    selected = selected.loc[selected["target_delta_s"].isin(targets)].copy()
    if targets.size < 2 or len(selected) < 6:
        return None
    x = selected["scaled_lag_bin"].to_numpy(float)
    delta = selected["actual_delta_s_mean"].to_numpy(float)
    observed = selected["mean"].to_numpy(float)
    predictions = reference.evaluate(
        float(alpha),
        delta[:, None] / tau_grid_s[None, :],
        x[:, None],
    )
    squared = (observed[:, None] - predictions) ** 2
    target_mse = []
    target_rmse_at_best: dict[float, float] = {}
    for target in targets:
        mask = np.isclose(selected["target_delta_s"].to_numpy(float), target)
        target_mse.append(np.mean(squared[mask], axis=0))
    objective = np.mean(np.vstack(target_mse), axis=0)
    best_index = int(np.argmin(objective))
    fitted = predictions[:, best_index]
    residual_sum = float(np.sum((observed - fitted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    for target in targets:
        mask = np.isclose(selected["target_delta_s"].to_numpy(float), target)
        target_rmse_at_best[float(target)] = float(
            np.sqrt(np.mean((observed[mask] - fitted[mask]) ** 2))
        )
    return {
        "tau_s": float(tau_grid_s[best_index]),
        "wmse": float(objective[best_index]),
        "rmse": float(np.sqrt(objective[best_index])),
        "r_squared": float(1.0 - residual_sum / total_sum) if total_sum > 0.0 else np.nan,
        "n_fit_points": len(selected),
        "n_target_deltas": int(targets.size),
        "fit_targets": ",".join(f"{value:g}" for value in targets),
        "boundary_solution": bool(best_index in (0, tau_grid_s.size - 1)),
        "selected": selected,
        "predicted": fitted,
        "target_rmse": target_rmse_at_best,
    }


def _bootstrap_tau(
    source: pd.DataFrame,
    reference: RouseVccReference,
    *,
    alpha: float,
    tau_grid_s: np.ndarray,
    minimum_units_per_bin: int,
    iterations: int,
    seed: int,
) -> np.ndarray:
    crops = sorted(source["crop_id"].astype(str).unique())
    if len(crops) < 2 or iterations < 1:
        return np.empty(0, dtype=float)
    groups = {crop: source.loc[source["crop_id"].astype(str).eq(crop)] for crop in crops}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        pieces: list[pd.DataFrame] = []
        for replicate, crop in enumerate(rng.choice(crops, size=len(crops), replace=True)):
            piece = groups[str(crop)].copy()
            piece["bootstrap_unit"] = piece["unit_id"].astype(str) + f"|b{replicate:05d}"
            pieces.append(piece)
        sampled = pd.concat(pieces, ignore_index=True)
        sampled = sampled.rename(columns={"unit_id": "original_unit_id"})
        sampled = sampled.rename(columns={"bootstrap_unit": "unit_id"})
        curve = _aggregate_hour_curves(sampled)
        fitted = _fit_tau_grid(
            curve,
            reference,
            alpha=alpha,
            tau_grid_s=tau_grid_s,
            minimum_units_per_bin=minimum_units_per_bin,
        )
        if fitted is not None:
            estimates.append(float(fitted["tau_s"]))
    return np.asarray(estimates, dtype=float)


def _fit_hours(
    binned: pd.DataFrame,
    hour_curves: pd.DataFrame,
    reference: RouseVccReference,
    *,
    alpha: float,
    alpha_sensitivity: tuple[float, ...],
    tau_grid_s: np.ndarray,
    minimum_units_per_bin: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for hour, curve in hour_curves.groupby("hour_post_delivery", sort=True):
        fitted = _fit_tau_grid(
            curve,
            reference,
            alpha=alpha,
            tau_grid_s=tau_grid_s,
            minimum_units_per_bin=minimum_units_per_bin,
        )
        if fitted is None:
            continue
        units = binned.loc[np.isclose(binned["hour_post_delivery"], float(hour))]
        seed = int((bootstrap_seed + zlib.crc32(f"vcc|{hour:g}".encode())) % (2**32 - 1))
        bootstrap = _bootstrap_tau(
            units,
            reference,
            alpha=alpha,
            tau_grid_s=tau_grid_s,
            minimum_units_per_bin=minimum_units_per_bin,
            iterations=bootstrap_iterations,
            seed=seed,
        )
        ci_low = float(np.quantile(bootstrap, 0.025)) if bootstrap.size >= 20 else np.nan
        ci_high = float(np.quantile(bootstrap, 0.975)) if bootstrap.size >= 20 else np.nan
        fit_rows.append(
            {
                "hour_post_delivery": float(hour),
                "rouse_alpha_fixed": float(alpha),
                "communication_time_s": fitted["tau_s"],
                "communication_time_ci_low_s": ci_low,
                "communication_time_ci_high_s": ci_high,
                "wmse": fitted["wmse"],
                "rmse": fitted["rmse"],
                "r_squared": fitted["r_squared"],
                "n_fit_points": fitted["n_fit_points"],
                "n_target_deltas": fitted["n_target_deltas"],
                "fit_targets_delta_s": fitted["fit_targets"],
                "n_units": int(units["unit_id"].nunique()),
                "n_crops": int(units["crop_id"].nunique()),
                "n_acquisitions": int(units["nd2_id"].nunique()),
                "bootstrap_successes": int(bootstrap.size),
                "fit_bounds_s": f"[{tau_grid_s[0]:g},{tau_grid_s[-1]:g}]",
                "fit_status": (
                    "descriptive_boundary_solution"
                    if fitted["boundary_solution"]
                    else "descriptive_interior_solution"
                ),
            }
        )
        selected = fitted["selected"].reset_index(drop=True)
        for index, row in selected.iterrows():
            prediction_rows.append(
                {
                    "hour_post_delivery": float(hour),
                    "target_delta_s": float(row["target_delta_s"]),
                    "scaled_lag_bin": float(row["scaled_lag_bin"]),
                    "actual_delta_s_mean": float(row["actual_delta_s_mean"]),
                    "observed_mean": float(row["mean"]),
                    "observed_sd": float(row["sd"]),
                    "n_units": int(row["n_units"]),
                    "predicted_vcc": float(fitted["predicted"][index]),
                    "communication_time_s": fitted["tau_s"],
                    "rouse_alpha_fixed": float(alpha),
                }
            )
        for candidate_alpha in sorted({float(value) for value in alpha_sensitivity}):
            candidate = _fit_tau_grid(
                curve,
                reference,
                alpha=candidate_alpha,
                tau_grid_s=tau_grid_s,
                minimum_units_per_bin=minimum_units_per_bin,
            )
            if candidate is None:
                continue
            sensitivity_rows.append(
                {
                    "hour_post_delivery": float(hour),
                    "rouse_alpha": candidate_alpha,
                    "communication_time_s": candidate["tau_s"],
                    "rmse": candidate["rmse"],
                    "r_squared": candidate["r_squared"],
                    "boundary_solution": candidate["boundary_solution"],
                }
            )
    return (
        pd.DataFrame(fit_rows).sort_values("hour_post_delivery").reset_index(drop=True),
        pd.DataFrame(prediction_rows).sort_values(
            ["hour_post_delivery", "target_delta_s", "scaled_lag_bin"]
        ),
        pd.DataFrame(sensitivity_rows).sort_values(
            ["hour_post_delivery", "rouse_alpha"]
        ),
    )


def compute_oligo_vcc(
    bundle_frames: pd.DataFrame,
    bundle_index: pd.DataFrame,
    reference: RouseVccReference,
    *,
    target_delta_s: tuple[float, ...] = (10.0, 20.0, 40.0),
    delta_relative_tolerance: float = 0.25,
    maximum_scaled_lag: float = 2.5,
    minimum_velocity_pairs: int = 8,
    scaled_lag_bin_centers: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
    ),
    rouse_alpha_fixed: float = 0.9,
    rouse_alpha_sensitivity: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0),
    communication_time_bounds_s: tuple[int, int] = (1, 1000),
    minimum_hour_bin_units_for_fit: int = 8,
    bootstrap_iterations: int = 200,
    bootstrap_seed: int = 20260824,
) -> OligoVccResult:
    """Compute multi-delta pair VCC and fit Rouse communication time by hour."""

    targets = _positive_targets(target_delta_s)
    centers = _scaled_centers(scaled_lag_bin_centers)
    tolerance = float(delta_relative_tolerance)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("delta_relative_tolerance must lie in [0,1)")
    if maximum_scaled_lag <= 0.0:
        raise ValueError("maximum_scaled_lag must be positive")
    if minimum_velocity_pairs < 1:
        raise ValueError("minimum_velocity_pairs must be positive")
    tau_start, tau_end = (int(value) for value in communication_time_bounds_s)
    if tau_start < 1 or tau_end <= tau_start:
        raise ValueError("communication-time bounds must be increasing positive integers")
    tau_grid = np.arange(tau_start, tau_end + 1, dtype=float)

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
    curve_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    census_rows: list[dict[str, Any]] = []
    for record in bundle_index.sort_values("bundle_id", kind="stable").itertuples(index=False):
        unit_id = str(record.bundle_id)
        nd2_id = str(record.nd2_id)
        schedule = schedules[nd2_id]
        rows = (
            frame_groups.get_group(unit_id)
            if unit_id in frame_groups.groups
            else bundle_frames.iloc[0:0]
        )
        site1, site2 = _dense_coordinates(rows, schedule)
        identity = _base_identity(record)
        observed1 = int(np.count_nonzero(np.all(np.isfinite(site1), axis=1)))
        observed2 = int(np.count_nonzero(np.all(np.isfinite(site2), axis=1)))
        shared = int(
            np.count_nonzero(
                np.all(np.isfinite(site1), axis=1) & np.all(np.isfinite(site2), axis=1)
            )
        )
        for target in targets:
            delta_frames, schedule_delta_s = acquisition_delta[(nd2_id, target)]
            if delta_frames is None:
                census_rows.append(
                    {
                        **identity,
                        "target_delta_s": target,
                        "delta_frames": pd.NA,
                        "schedule_delta_s": schedule_delta_s,
                        "actual_delta_s": np.nan,
                        "observed_site1_frames": observed1,
                        "observed_site2_frames": observed2,
                        "shared_site_frames": shared,
                        "maximum_pair_count": 0,
                        "emitted_symmetric_lag_count": 0,
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
            actual_delta_s = _paired_median_velocity_duration(
                site1, site2, schedule.times_s, delta_frames
            )
            direction_finite: list[np.ndarray] = []
            direction_counts: list[np.ndarray] = []
            for direction, later_site, earlier_site in (
                (_DIRECTIONS[0], site1, site2),
                (_DIRECTIONS[1], site2, site1),
            ):
                vcc = velocity_cross_correlation_matrix_by_frame_steps(
                    later_site,
                    earlier_site,
                    times=schedule.times_s,
                    measurement_delta_frames=delta_frames,
                    lag_frames=lag_frames,
                    min_pairs=minimum_velocity_pairs,
                )
                finite = np.isfinite(vcc.trace_values) & np.isfinite(
                    vcc.start_time_lag_median
                )
                direction_finite.append(finite)
                direction_counts.append(vcc.counts)
                for index in np.flatnonzero(finite):
                    lag_s = float(vcc.start_time_lag_median[index])
                    common = {
                        **identity,
                        "target_delta_s": target,
                        "delta_frames": int(delta_frames),
                        "schedule_delta_s": float(schedule_delta_s),
                        "actual_delta_s": float(actual_delta_s),
                        "direction": direction,
                        "lag_frames": int(vcc.lag_frames[index]),
                        "lag_median_s": lag_s,
                        "scaled_lag": lag_s / actual_delta_s,
                        "pair_count": int(vcc.counts[index]),
                        "zero_lag_energy_um2_per_s2": float(vcc.normalization),
                    }
                    curve_rows.append(
                        {
                            **common,
                            "normalized_vcc": float(vcc.trace_values[index]),
                            "raw_vcc_um2_per_s2": float(vcc.raw_trace_values[index]),
                        }
                    )
                    for row_index, row_component in enumerate(_COMPONENTS):
                        for column_index, column_component in enumerate(_COMPONENTS):
                            matrix_rows.append(
                                {
                                    **common,
                                    "row_component": row_component,
                                    "column_component": column_component,
                                    "normalized_value": float(
                                        vcc.matrices[index, row_index, column_index]
                                    ),
                                    "raw_value_um2_per_s2": float(
                                        vcc.raw_matrices[index, row_index, column_index]
                                    ),
                                }
                            )
            if len(direction_finite) == 2:
                symmetric = direction_finite[0] & direction_finite[1]
                maximum_pairs = int(
                    max(np.max(values) if values.size else 0 for values in direction_counts)
                )
            else:
                symmetric = np.zeros(lag_frames.shape, dtype=bool)
                maximum_pairs = 0
            if np.any(symmetric):
                reason = "included"
            elif maximum_pairs < minimum_velocity_pairs:
                reason = "insufficient_velocity_pairs_all_lags"
            else:
                reason = "missing_bidirectional_or_nonpositive_energy"
            census_rows.append(
                {
                    **identity,
                    "target_delta_s": target,
                    "delta_frames": int(delta_frames),
                    "schedule_delta_s": float(schedule_delta_s),
                    "actual_delta_s": float(actual_delta_s),
                    "observed_site1_frames": observed1,
                    "observed_site2_frames": observed2,
                    "shared_site_frames": shared,
                    "maximum_pair_count": maximum_pairs,
                    "emitted_symmetric_lag_count": int(np.count_nonzero(symmetric)),
                    "unit_included": bool(np.any(symmetric)),
                    "exclusion_reason": reason,
                }
            )

    directional_curves = pd.DataFrame(curve_rows)
    directional_matrices = pd.DataFrame(matrix_rows)
    census = pd.DataFrame(census_rows)
    if directional_curves.empty or directional_matrices.empty:
        raise ValueError("No supported pair VCC curve points were produced")
    symmetric_curves = _symmetrize_curves(directional_curves)
    symmetric_matrices = _symmetrize_matrices(directional_matrices)
    binned = _nearest_binned_curve(symmetric_curves, centers)
    hour_curves = _aggregate_hour_curves(binned)
    hour_matrices = _bin_and_aggregate_matrices(symmetric_matrices, centers)
    fits, predictions, sensitivity = _fit_hours(
        binned,
        hour_curves,
        reference,
        alpha=float(rouse_alpha_fixed),
        alpha_sensitivity=rouse_alpha_sensitivity,
        tau_grid_s=tau_grid,
        minimum_units_per_bin=int(minimum_hour_bin_units_for_fit),
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    return OligoVccResult(
        directional_unit_curves=directional_curves,
        directional_unit_matrices=directional_matrices,
        symmetric_unit_curves=symmetric_curves,
        symmetric_unit_matrices=symmetric_matrices,
        support_census=census,
        binned_unit_curves=binned,
        hour_curves=hour_curves,
        hour_matrix_curves=hour_matrices,
        communication_time_fits=fits,
        fit_predictions=predictions,
        alpha_sensitivity=sensitivity,
    )


def configure_vcc_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.linewidth": 1.9,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.fontsize": 14,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(figure: mpl.figure.Figure, stem: Path) -> list[Path]:
    outputs = [Path(f"{stem}{extension}") for extension in (".png", ".pdf", ".svg")]
    figure.savefig(outputs[0], dpi=300, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    figure.savefig(outputs[2], bbox_inches="tight")
    plt.close(figure)
    return outputs


def _delta_colors(targets: list[float]) -> dict[float, str]:
    # MATLAB R2014b+ ``lines`` palette, matching the source figure language.
    palette = ("#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30")
    return {target: palette[index % len(palette)] for index, target in enumerate(targets)}


def _style_axis(axis: mpl.axes.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.9)
    axis.tick_params(
        direction="out",
        top=True,
        right=True,
        width=1.8,
        length=6,
        color="black",
    )
    axis.axhline(0.0, color="#707070", lw=0.75, zorder=0)


def _plot_hour_fit(
    axis: mpl.axes.Axes,
    hour: float,
    predictions: pd.DataFrame,
    fit: pd.Series,
    colors: dict[float, str],
    *,
    show_legend: bool,
) -> None:
    panel = predictions.loc[np.isclose(predictions["hour_post_delivery"], hour)]
    for target, curve in panel.groupby("target_delta_s", sort=True):
        curve = curve.sort_values("scaled_lag_bin")
        color = colors[float(target)]
        x = curve["scaled_lag_bin"].to_numpy(float)
        mean = curve["observed_mean"].to_numpy(float)
        axis.plot(
            x,
            mean,
            lw=2.15,
            color=color,
            label=rf"$\delta={target:g}$ s",
        )
        axis.plot(
            x,
            curve["predicted_vcc"],
            ls="--",
            lw=1.9,
            color=color,
            label="_nolegend_",
        )
    status = "boundary" if str(fit["fit_status"]).endswith("boundary_solution") else ""
    axis.text(
        0.04 if show_legend else 0.96,
        0.95,
        rf"$\tau_{{\Delta n}}$={fit['communication_time_s']:.0f} s{('*' if status else '')}"
        "\n"
        rf"RMSE={fit['rmse']:.3f}; n={int(fit['n_units'])}",
        transform=axis.transAxes,
        ha="left" if show_legend else "right",
        va="top",
        fontsize=12,
    )
    axis.set_title(
        f"Experimental VCC (sites 1&2) · {hour:g} h" if show_legend else f"{hour:g} h",
        loc="center" if show_legend else "left",
        weight="normal" if show_legend else "bold",
    )
    axis.set_xlim(0.0, 2.5)
    axis.set_ylim(-0.5, 1.0)
    axis.set_xticks(np.arange(0.0, 2.51, 0.5))
    axis.set_yticks([-0.5, 0.0, 0.5, 1.0])
    _style_axis(axis)
    if show_legend:
        handles, labels = axis.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="black", ls="--", lw=1.9))
        labels.append("Rouse prediction")
        axis.legend(handles, labels, frameon=False, loc="upper right")


def build_vcc_fit_figures(result: OligoVccResult, output_dir: str | Path) -> dict[str, Any]:
    """Write all-hour and per-hour empirical-plus-Rouse VCC figures."""

    configure_vcc_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fits = result.communication_time_fits
    predictions = result.fit_predictions
    hours = sorted(float(value) for value in fits["hour_post_delivery"].unique())
    targets = sorted(float(value) for value in predictions["target_delta_s"].unique())
    colors = _delta_colors(targets)

    figure, axes = plt.subplots(3, 3, figsize=(15.5, 12.0), sharex=True, sharey=True)
    flat = axes.ravel()
    for index, hour in enumerate(hours):
        fit = fits.loc[np.isclose(fits["hour_post_delivery"], hour)].iloc[0]
        _plot_hour_fit(flat[index], hour, predictions, fit, colors, show_legend=False)
    for axis in flat[len(hours) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel(r"Scaled lag, $\tau/\delta$")
    for axis in axes[:, 0]:
        if axis.get_visible():
            axis.set_ylabel(r"Normalized VCC trace")
    legend_handles = [
        Line2D([0], [0], color=colors[target], lw=2.15, label=rf"$\delta={target:g}$ s")
        for target in targets
    ]
    legend_handles.append(
        Line2D([0], [0], color="black", ls="--", lw=1.9, label="Rouse prediction")
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), w_pad=1.3, h_pad=1.4)
    atlas = _save_figure(figure, output / "fig_vcc_rouse_fits_by_hour")

    per_hour: dict[float, list[Path]] = {}
    for hour in hours:
        figure, axis = plt.subplots(figsize=(6.6, 5.2))
        fit = fits.loc[np.isclose(fits["hour_post_delivery"], hour)].iloc[0]
        _plot_hour_fit(axis, hour, predictions, fit, colors, show_legend=True)
        axis.set_xlabel(r"Rescaled lag time $\tau/\delta$")
        axis.set_ylabel(r"$C_{v_1v_2}(\tau)/C_{v_1v_2}(0)$")
        figure.tight_layout()
        per_hour[hour] = _save_figure(figure, output / f"vcc_{hour:g}h_rouse_fit")

    figure, axis = plt.subplots(figsize=(7.3, 5.3))
    x = fits["hour_post_delivery"].to_numpy(float)
    y = fits["communication_time_s"].to_numpy(float)
    low = fits["communication_time_ci_low_s"].to_numpy(float)
    high = fits["communication_time_ci_high_s"].to_numpy(float)
    yerr = np.vstack((y - low, high - y))
    valid_ci = np.all(np.isfinite(yerr), axis=0) & np.all(yerr >= 0.0, axis=0)
    axis.plot(x, y, color="#0072BD", lw=2.0)
    axis.scatter(x, y, color="#0072BD", s=55, zorder=3)
    if np.any(valid_ci):
        axis.errorbar(
            x[valid_ci],
            y[valid_ci],
            yerr=yerr[:, valid_ci],
            fmt="none",
            ecolor="#0072BD",
            elinewidth=1.7,
            capsize=4,
        )
    boundary = fits["fit_status"].astype(str).str.contains("boundary").to_numpy()
    if np.any(boundary):
        axis.scatter(
            x[boundary], y[boundary], marker="^", facecolor="white", edgecolor="#A2142F", s=75
        )
    upper = max(125.0, 25.0 * np.ceil(float(np.nanmax(high)) / 25.0))
    axis.set_ylim(0.0, upper)
    axis.set_yticks(np.arange(0.0, upper + 0.1, 25.0))
    axis.set_xticks(x)
    axis.set_xticklabels([f"{value:g}" for value in x], rotation=45, ha="right")
    axis.set_xlabel("Time after Cas9 delivery (h)")
    axis.set_ylabel(r"Communication time, $\tau_{\Delta n}$ (s)")
    _style_axis(axis)
    figure.tight_layout()
    trend = _save_figure(figure, output / "fig_vcc_communication_time_by_hour")
    return {"atlas": atlas, "per_hour": per_hour, "trend": trend}


def build_vcc_matrix_figure(
    result: OligoVccResult,
    output_stem: str | Path,
    *,
    target_delta_s: float = 10.0,
) -> list[Path]:
    """Plot all four symmetrized VCC tensor components by folder-hour."""

    configure_vcc_style()
    source = result.hour_matrix_curves.loc[
        np.isclose(result.hour_matrix_curves["target_delta_s"], target_delta_s)
    ]
    hours = sorted(float(value) for value in source["hour_post_delivery"].unique())
    styles = {
        ("x", "x"): ("Cxx", "#0072BD", "-"),
        ("x", "y"): ("Cxy", "#D95319", "--"),
        ("y", "x"): ("Cyx", "#7E2F8E", "--"),
        ("y", "y"): ("Cyy", "#77AC30", "-"),
    }
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 12.0), sharex=True, sharey=True)
    flat = axes.ravel()
    for panel_index, hour in enumerate(hours):
        axis = flat[panel_index]
        panel = source.loc[np.isclose(source["hour_post_delivery"], hour)]
        for components, (label, color, line_style) in styles.items():
            curve = panel.loc[
                panel["row_component"].eq(components[0])
                & panel["column_component"].eq(components[1])
            ].sort_values("scaled_lag_bin")
            axis.plot(
                curve["scaled_lag_bin"],
                curve["mean"],
                color=color,
                ls=line_style,
                lw=2.0,
                label=label,
            )
        axis.set_title(f"{hour:g} h", loc="left", weight="bold")
        axis.set_xlim(0.0, 2.5)
        axis.set_ylim(-0.11, 0.27)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
        _style_axis(axis)
        if panel_index == 0:
            axis.legend(frameon=False, ncol=2)
    for axis in flat[len(hours) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel(r"Scaled lag, $\tau/\delta$")
    for axis in axes[:, 0]:
        if axis.get_visible():
            axis.set_ylabel("Normalized VCC component")
    figure.tight_layout(w_pad=1.3, h_pad=1.4)
    return _save_figure(figure, Path(output_stem))
