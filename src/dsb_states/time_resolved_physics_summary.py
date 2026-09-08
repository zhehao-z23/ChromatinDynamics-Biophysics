"""Hour-resolved summaries and deliberately limited fits for physical curves.

The input is the metric-support-only per-unit output from
``time_resolved_physics``.  Acquisition is retained as provenance but is not a
statistical stratum: the user confirmed that all acquisitions belong to one
experimental batch.  Different frame cadences are nevertheless respected by
binning curve points on their observed physical lag in seconds.

MSD/MSCD use descriptive power laws over the common 10--50 s window.  VAC is
fit to the normalized fractional-Brownian-motion expression used by
Oligo-LiveFISH.  VCC receives only a non-mechanistic smoothing spline; a Rouse
communication-time fit is intentionally not invented without the full model
implementation and its additional assumptions.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import minimize_scalar

_REQUIRED_COLUMNS = {
    "metric",
    "unit_id",
    "bundle_id",
    "nd2_id",
    "crop_id",
    "hour_post_delivery",
    "site",
    "direction",
    "is_primary_matched_10s",
    "delta_median_s",
    "lag_frames",
    "lag_median_s",
    "scaled_lag_tau_over_delta",
    "value",
    "endpoint_pair_sd",
    "pair_count",
}
_REQUIRED_MATRIX_COLUMNS = {
    "unit_id",
    "nd2_id",
    "crop_id",
    "hour_post_delivery",
    "direction",
    "is_primary_matched_10s",
    "lag_frames",
    "lag_median_s",
    "scaled_lag_tau_over_delta",
    "row_component",
    "column_component",
    "normalized_value",
    "raw_value",
    "pair_count",
}


@dataclass(frozen=True)
class TimeResolvedPhysicsSummary:
    """Equal-unit hour curves, fitted endpoints, predictions and support."""

    unit_bins: pd.DataFrame
    hour_curves: pd.DataFrame
    fits: pd.DataFrame
    predicted_curves: pd.DataFrame
    support_by_hour: pd.DataFrame
    method_contract: dict[str, Any]


def fbm_normalized_vac(scaled_lag: np.ndarray | float, alpha: float) -> np.ndarray:
    """Equation 9 from Zhu et al. Oligo-LiveFISH (Cell, 2025)."""

    x = np.asarray(scaled_lag, dtype=float)
    exponent = float(alpha)
    return (
        np.abs(x - 1.0) ** exponent
        + np.abs(x + 1.0) ** exponent
        - 2.0 * np.abs(x) ** exponent
    ) / 2.0


def _require_columns(table: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise ValueError(f"unit_curves is missing required columns: {missing}")
    if table.empty:
        raise ValueError("unit_curves is empty")


def _validate_centers(values: tuple[float, ...]) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if (
        centers.ndim != 1
        or centers.size < 2
        or not np.all(np.isfinite(centers))
        or np.any(centers <= 0.0)
        or np.any(np.diff(centers) <= 0.0)
    ):
        raise ValueError("physical lag centers must be finite, positive and increasing")
    return centers


def _bin_edges(centers: np.ndarray) -> np.ndarray:
    interior = np.sqrt(centers[:-1] * centers[1:])
    first = centers[0] ** 2 / interior[0]
    last = centers[-1] ** 2 / interior[-1]
    return np.concatenate(([first], interior, [last]))


def _assign_lag_bins(lag: np.ndarray, centers: np.ndarray) -> np.ndarray:
    edges = _bin_edges(centers)
    zero = np.isclose(lag, 0.0, atol=1e-12)
    positive = np.isfinite(lag) & (lag > 0.0)
    assigned = np.full(lag.shape, np.nan, dtype=float)
    positive_index = np.flatnonzero(positive)
    bin_index = np.searchsorted(edges, lag[positive], side="right") - 1
    in_range = (bin_index >= 0) & (bin_index < centers.size)
    assigned[positive_index[in_range]] = centers[bin_index[in_range]]
    assigned[zero] = 0.0
    return assigned


def _select_primary_points(unit_curves: pd.DataFrame) -> pd.DataFrame:
    displacement = unit_curves["metric"].isin(["msd", "mscd"])
    velocity = unit_curves["metric"].isin(["vac", "vcc"]) & unit_curves[
        "is_primary_matched_10s"
    ].astype(bool)
    selected = unit_curves.loc[displacement | velocity].copy()
    if selected.empty:
        raise ValueError("no displacement or primary matched-10-s velocity curves are available")
    return selected


def _nearest_unit_bins(unit_curves: pd.DataFrame, centers: np.ndarray) -> pd.DataFrame:
    selected = _select_primary_points(unit_curves)
    lag = pd.to_numeric(selected["lag_median_s"], errors="coerce").to_numpy(float)
    selected["lag_bin_center_s"] = _assign_lag_bins(lag, centers)
    selected = selected.loc[np.isfinite(selected["lag_bin_center_s"])].copy()
    selected["lag_distance_log"] = 0.0
    positive_selected = selected["lag_bin_center_s"].gt(0.0)
    selected.loc[positive_selected, "lag_distance_log"] = np.abs(
        np.log(
            selected.loc[positive_selected, "lag_median_s"].to_numpy(float)
            / selected.loc[positive_selected, "lag_bin_center_s"].to_numpy(float)
        )
    )

    keys = [
        "metric",
        "unit_id",
        "hour_post_delivery",
        "site",
        "direction",
        "lag_bin_center_s",
    ]
    chosen = (
        selected.sort_values(
            keys + ["lag_distance_log", "pair_count", "lag_frames"],
            ascending=[True] * len(keys) + [True, False, True],
            kind="stable",
        )
        .drop_duplicates(keys, keep="first")
        .drop(columns="lag_distance_log")
        .reset_index(drop=True)
    )

    # For the VCC headline, remove the arbitrary lead/lag convention by
    # requiring both directions and averaging their rotation-invariant traces.
    vcc = chosen.loc[chosen["metric"].eq("vcc")].copy()
    non_vcc = chosen.loc[~chosen["metric"].eq("vcc")].copy()
    if not vcc.empty:
        identity = [
            "metric",
            "unit_id",
            "bundle_id",
            "nd2_id",
            "crop_id",
            "hour_post_delivery",
            "site",
            "lag_bin_center_s",
        ]
        direction_counts = vcc.groupby(identity, dropna=False)["direction"].nunique()
        valid_index = direction_counts.loc[direction_counts.eq(2)].index
        if len(valid_index):
            indexed = vcc.set_index(identity)
            vcc = indexed.loc[indexed.index.isin(valid_index)].reset_index()
            numeric = vcc.groupby(identity, as_index=False, dropna=False).agg(
                lag_median_s=("lag_median_s", "mean"),
                scaled_lag_tau_over_delta=("scaled_lag_tau_over_delta", "mean"),
                delta_median_s=("delta_median_s", "mean"),
                value=("value", "mean"),
                endpoint_pair_sd=("endpoint_pair_sd", "mean"),
                pair_count=("pair_count", "min"),
                lag_frames=("lag_frames", "min"),
            )
            numeric["direction"] = "symmetric_trace_mean"
            numeric["is_primary_matched_10s"] = True
            numeric["raw_value"] = np.nan
            numeric["normalization"] = np.nan
            numeric["delta_frames"] = pd.NA
            numeric["delta_role"] = "primary_matched_10s"
            numeric["value_unit"] = "dimensionless"
            numeric["raw_value_unit"] = "um^2/s^2"
            numeric["endpoint_pair_sd_unit"] = None
            vcc = numeric
        else:
            vcc = vcc.iloc[0:0]
    combined = pd.concat([non_vcc, vcc], ignore_index=True, sort=False)
    return combined.sort_values(
        ["metric", "site", "hour_post_delivery", "unit_id", "lag_bin_center_s"],
        kind="stable",
    ).reset_index(drop=True)


def build_vcc_matrix_hour_summary(
    vcc_matrices: pd.DataFrame,
    *,
    lag_bin_centers_s: tuple[float, ...] = (
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        7.5,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        75.0,
        100.0,
        150.0,
        200.0,
        300.0,
        500.0,
    ),
) -> pd.DataFrame:
    """Equal-bundle hour summary of the primary bidirectional VCC matrices.

    Reverse-direction matrices are transposed before averaging so every row is
    expressed as Site1-component by Site2-component.  The full directional
    unit table remains the authoritative primary object.
    """

    missing = sorted(_REQUIRED_MATRIX_COLUMNS.difference(vcc_matrices.columns))
    if missing:
        raise ValueError(f"vcc_matrices is missing required columns: {missing}")
    centers = _validate_centers(lag_bin_centers_s)
    source = vcc_matrices.loc[vcc_matrices["is_primary_matched_10s"].astype(bool)].copy()
    if source.empty:
        return pd.DataFrame()
    lag = pd.to_numeric(source["lag_median_s"], errors="coerce").to_numpy(float)
    source["lag_bin_center_s"] = _assign_lag_bins(lag, centers)
    source = source.loc[np.isfinite(source["lag_bin_center_s"])].copy()
    reverse = source["direction"].astype(str).eq("site2_later_vs_site1")
    reverse_rows = source.loc[reverse, "row_component"].copy()
    source.loc[reverse, "row_component"] = source.loc[reverse, "column_component"].to_numpy()
    source.loc[reverse, "column_component"] = reverse_rows.to_numpy()
    source["lag_distance_log"] = 0.0
    positive = source["lag_bin_center_s"].gt(0.0)
    source.loc[positive, "lag_distance_log"] = np.abs(
        np.log(
            source.loc[positive, "lag_median_s"].to_numpy(float)
            / source.loc[positive, "lag_bin_center_s"].to_numpy(float)
        )
    )
    keys = [
        "unit_id",
        "hour_post_delivery",
        "direction",
        "row_component",
        "column_component",
        "lag_bin_center_s",
    ]
    source = (
        source.sort_values(
            keys + ["lag_distance_log", "pair_count"],
            ascending=[True] * len(keys) + [True, False],
            kind="stable",
        )
        .drop_duplicates(keys, keep="first")
        .drop(columns="lag_distance_log")
    )
    identity = [
        "unit_id",
        "nd2_id",
        "crop_id",
        "hour_post_delivery",
        "row_component",
        "column_component",
        "lag_bin_center_s",
    ]
    direction_count = source.groupby(identity, dropna=False)["direction"].nunique()
    supported = direction_count.loc[direction_count.eq(2)].index
    source = source.set_index(identity)
    source = source.loc[source.index.isin(supported)].reset_index()
    if source.empty:
        return pd.DataFrame()
    unit = source.groupby(identity, as_index=False, dropna=False).agg(
        lag_s_mean=("lag_median_s", "mean"),
        scaled_lag_mean=("scaled_lag_tau_over_delta", "mean"),
        normalized_value=("normalized_value", "mean"),
        raw_value=("raw_value", "mean"),
        pair_count=("pair_count", "min"),
    )
    group = [
        "hour_post_delivery",
        "row_component",
        "column_component",
        "lag_bin_center_s",
    ]
    result = unit.groupby(group, as_index=False, dropna=False).agg(
        lag_s_mean=("lag_s_mean", "mean"),
        scaled_lag_mean=("scaled_lag_mean", "mean"),
        normalized_mean=("normalized_value", "mean"),
        normalized_sd=(
            "normalized_value",
            lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        ),
        raw_mean_um2_per_s2=("raw_value", "mean"),
        raw_sd_um2_per_s2=(
            "raw_value",
            lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        ),
        n_bundles=("unit_id", "nunique"),
        n_crops=("crop_id", "nunique"),
        n_acquisitions=("nd2_id", "nunique"),
        total_pairs=("pair_count", "sum"),
    )
    result["matrix_orientation"] = "Site1 row component x Site2 column component"
    result["direction_reduction"] = "mean(C12(+tau), transpose(C21(+tau)))"
    return result.sort_values(group, kind="stable").reset_index(drop=True)


def _aggregate_hour_curves(unit_bins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    denominators = (
        unit_bins.groupby(["metric", "site", "hour_post_delivery"], as_index=False)
        .agg(
            available_units=("unit_id", "nunique"),
            available_crops=("crop_id", "nunique"),
            source_acquisitions=("nd2_id", "nunique"),
        )
        .sort_values(["metric", "site", "hour_post_delivery"], kind="stable")
    )
    group_fields = ["metric", "site", "hour_post_delivery", "lag_bin_center_s"]
    grouped = unit_bins.groupby(group_fields, as_index=False, dropna=False)
    curves = grouped.agg(
        lag_s_mean=("lag_median_s", "mean"),
        lag_s_min=("lag_median_s", "min"),
        lag_s_max=("lag_median_s", "max"),
        scaled_lag_mean=("scaled_lag_tau_over_delta", "mean"),
        delta_s_mean=("delta_median_s", "mean"),
        mean=("value", "mean"),
        sd=("value", lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else np.nan),
        median=("value", "median"),
        q25=("value", lambda values: float(np.quantile(values, 0.25))),
        q75=("value", lambda values: float(np.quantile(values, 0.75))),
        mean_within_unit_endpoint_pair_sd=("endpoint_pair_sd", "mean"),
        n_units=("unit_id", "nunique"),
        n_crops=("crop_id", "nunique"),
        n_acquisitions=("nd2_id", "nunique"),
        total_pairs=("pair_count", "sum"),
    )
    curves = curves.merge(denominators, on=["metric", "site", "hour_post_delivery"], how="left")
    curves["retained_unit_fraction"] = curves["n_units"] / curves["available_units"]
    curves["sd_definition"] = "between_trajectory_or_bundle_sample_SD"
    curves["mean_weighting"] = "equal_trajectory_or_bundle"
    return (
        curves.sort_values(group_fields, kind="stable").reset_index(drop=True),
        denominators.reset_index(drop=True),
    )


def _power_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict[str, float] | None:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (x > 0) & (y > 0) & (weights > 0)
    if np.count_nonzero(valid) < 2:
        return None
    lx = np.log(x[valid])
    ly = np.log(y[valid])
    weight = np.sqrt(weights[valid])
    slope, intercept = np.polyfit(lx, ly, 1, w=weight)
    fitted = intercept + slope * lx
    residual = np.sum(weight * (ly - fitted) ** 2)
    centered = np.sum(weight * (ly - np.average(ly, weights=weight)) ** 2)
    return {
        "exponent": float(slope),
        "amplitude_at_1s": float(np.exp(intercept)),
        "value_at_10s": float(np.exp(intercept) * 10.0**slope),
        "r2_log": float(1.0 - residual / centered) if centered > 0 else float("nan"),
    }


def _fit_fbm_vac(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict[str, float] | None:
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(weights)
        & (x > 0.0)
        # The target display/fit window is tau/delta <= 5.  Acquisition-specific
        # physical-time binning can move the final nominal point a few percent
        # above five, so retain a 2% numerical/cadence tolerance here.
        & (x <= 5.1)
        & (weights > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        return None
    x_fit = x[valid]
    y_fit = y[valid]
    weight = weights[valid] / np.sum(weights[valid])

    def objective(alpha: float) -> float:
        residual = y_fit - fbm_normalized_vac(x_fit, alpha)
        return float(np.sum(weight * residual**2))

    result = minimize_scalar(objective, method="bounded", bounds=(0.05, 1.95))
    if not result.success:
        return None
    alpha = float(result.x)
    prediction = fbm_normalized_vac(x_fit, alpha)
    return {
        "exponent": alpha,
        "rmse": float(np.sqrt(np.average((y_fit - prediction) ** 2, weights=weight))),
        "at_boundary": bool(alpha < 0.055 or alpha > 1.945),
    }


def _group_seed(base_seed: int, label: str) -> int:
    return int((int(base_seed) + zlib.crc32(label.encode("utf-8"))) % (2**32 - 1))


def _bootstrap_group_means(
    source: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> list[pd.DataFrame]:
    clusters = sorted(source["crop_id"].astype(str).unique())
    if len(clusters) < 2 or iterations < 1:
        return []
    by_cluster = {cluster: source.loc[source["crop_id"].astype(str).eq(cluster)] for cluster in clusters}
    rng = np.random.default_rng(seed)
    results: list[pd.DataFrame] = []
    for _ in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        replicate = pd.concat([by_cluster[cluster] for cluster in sampled], ignore_index=True)
        means = (
            replicate.groupby("lag_bin_center_s", as_index=False)
            .agg(mean=("value", "mean"), n=("value", "size"), scaled=("scaled_lag_tau_over_delta", "mean"))
            .sort_values("lag_bin_center_s", kind="stable")
        )
        results.append(means)
    return results


def _ci(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 20:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _append_fit_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    site: str,
    hour: float,
    model: str,
    status: str,
    estimates: dict[str, float | bool],
    bootstrap: dict[str, list[float]],
    n_points: int,
    n_units: int,
    n_crops: int,
    fit_min: float,
    fit_max: float,
) -> None:
    for name, value in estimates.items():
        if isinstance(value, bool):
            continue
        low, high = _ci(bootstrap.get(name, []))
        rows.append(
            {
                "metric": metric,
                "site": site,
                "hour_post_delivery": hour,
                "model": model,
                "estimate_name": name,
                "estimate": float(value),
                "ci_low": low,
                "ci_high": high,
                "fit_status": status,
                "n_fit_points": n_points,
                "n_units": n_units,
                "n_crops": n_crops,
                "fit_lag_min": fit_min,
                "fit_lag_max": fit_max,
                "fit_time_span_fold": fit_max / fit_min if fit_min > 0 else np.nan,
            }
        )


def _fit_displacement_and_vac(
    unit_bins: pd.DataFrame,
    hour_curves: pd.DataFrame,
    *,
    common_window_s: tuple[float, float],
    minimum_fit_points: int,
    minimum_span_fold: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for (metric, site, hour), curve in hour_curves.loc[
        hour_curves["metric"].isin(["msd", "mscd", "vac"])
    ].groupby(["metric", "site", "hour_post_delivery"], sort=True):
        units = unit_bins.loc[
            unit_bins["metric"].eq(metric)
            & unit_bins["site"].eq(site)
            & np.isclose(unit_bins["hour_post_delivery"], hour)
        ].copy()
        if metric in {"msd", "mscd"}:
            lower, upper = common_window_s
            fit_curve = curve.loc[
                curve["lag_bin_center_s"].between(lower, upper, inclusive="both")
                & curve["mean"].gt(0.0)
            ].copy()
            x = fit_curve["lag_s_mean"].to_numpy(float)
            y = fit_curve["mean"].to_numpy(float)
            weights = fit_curve["n_units"].to_numpy(float)
            fitted = _power_fit(x, y, weights)
            span = float(np.max(x) / np.min(x)) if x.size and np.min(x) > 0 else 0.0
            enough = len(fit_curve) >= minimum_fit_points and span >= minimum_span_fold
            if fitted is None or not enough:
                continue
            bootstrap: dict[str, list[float]] = {name: [] for name in fitted}
            replicates = _bootstrap_group_means(
                units.loc[units["lag_bin_center_s"].between(lower, upper, inclusive="both")],
                iterations=bootstrap_iterations,
                seed=_group_seed(bootstrap_seed, f"{metric}|{site}|{hour:g}"),
            )
            for replicate in replicates:
                estimate = _power_fit(
                    replicate["lag_bin_center_s"].to_numpy(float),
                    replicate["mean"].to_numpy(float),
                    replicate["n"].to_numpy(float),
                )
                if estimate is not None and len(replicate) >= minimum_fit_points:
                    for name, value in estimate.items():
                        bootstrap[name].append(value)
            status = "descriptive_common_window_weak_identification" if span < 10.0 else "descriptive_common_window"
            _append_fit_rows(
                fit_rows,
                metric=metric,
                site=site,
                hour=float(hour),
                model="power_law_without_localization_offset",
                status=status,
                estimates=fitted,
                bootstrap=bootstrap,
                n_points=len(fit_curve),
                n_units=int(units["unit_id"].nunique()),
                n_crops=int(units["crop_id"].nunique()),
                fit_min=float(np.min(x)),
                fit_max=float(np.max(x)),
            )
            grid = np.geomspace(lower, upper, 120)
            prediction = fitted["amplitude_at_1s"] * grid ** fitted["exponent"]
            prediction_rows.extend(
                {
                    "metric": metric,
                    "site": site,
                    "hour_post_delivery": float(hour),
                    "model": "power_law_without_localization_offset",
                    "x": float(x_value),
                    "y": float(y_value),
                    "x_axis": "lag_s",
                }
                for x_value, y_value in zip(grid, prediction, strict=True)
            )
        else:
            fit_curve = curve.loc[
                curve["scaled_lag_mean"].gt(0.0) & curve["scaled_lag_mean"].le(5.1)
            ].copy()
            x = fit_curve["scaled_lag_mean"].to_numpy(float)
            y = fit_curve["mean"].to_numpy(float)
            weights = fit_curve["n_units"].to_numpy(float)
            fitted = _fit_fbm_vac(x, y, weights)
            if fitted is None or len(fit_curve) < minimum_fit_points:
                continue
            bootstrap = {"exponent": [], "rmse": []}
            replicates = _bootstrap_group_means(
                units,
                iterations=bootstrap_iterations,
                seed=_group_seed(bootstrap_seed, f"vac|{site}|{hour:g}"),
            )
            for replicate in replicates:
                estimate = _fit_fbm_vac(
                    replicate["scaled"].to_numpy(float),
                    replicate["mean"].to_numpy(float),
                    replicate["n"].to_numpy(float),
                )
                if estimate is not None and len(replicate) >= minimum_fit_points:
                    bootstrap["exponent"].append(estimate["exponent"])
                    bootstrap["rmse"].append(estimate["rmse"])
            status = "fbm_equation9_descriptive"
            if fitted["at_boundary"]:
                status += "_boundary_solution"
            _append_fit_rows(
                fit_rows,
                metric="vac",
                site=site,
                hour=float(hour),
                model="oligo_livefish_fbm_equation9",
                status=status,
                estimates=fitted,
                bootstrap=bootstrap,
                n_points=len(fit_curve),
                n_units=int(units["unit_id"].nunique()),
                n_crops=int(units["crop_id"].nunique()),
                fit_min=float(np.min(x)),
                fit_max=float(np.max(x)),
            )
            grid = np.linspace(0.0, 5.0, 201)
            prediction = fbm_normalized_vac(grid, fitted["exponent"])
            prediction_rows.extend(
                {
                    "metric": "vac",
                    "site": site,
                    "hour_post_delivery": float(hour),
                    "model": "oligo_livefish_fbm_equation9",
                    "x": float(x_value),
                    "y": float(y_value),
                    "x_axis": "tau_over_delta",
                }
                for x_value, y_value in zip(grid, prediction, strict=True)
            )
    return fit_rows, prediction_rows


def _first_zero_crossing(x: np.ndarray, y: np.ndarray) -> float:
    for index in range(1, len(x)):
        if y[index - 1] == 0.0:
            return float(x[index - 1])
        if y[index - 1] * y[index] < 0.0:
            fraction = -y[index - 1] / (y[index] - y[index - 1])
            return float(x[index - 1] + fraction * (x[index] - x[index - 1]))
    return float("nan")


def _summarize_vcc(
    unit_bins: pd.DataFrame,
    hour_curves: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fits: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    vcc_units = unit_bins.loc[unit_bins["metric"].eq("vcc")]
    for hour, curve in hour_curves.loc[hour_curves["metric"].eq("vcc")].groupby(
        "hour_post_delivery", sort=True
    ):
        ordered = curve.sort_values("scaled_lag_mean", kind="stable")
        x = ordered["scaled_lag_mean"].to_numpy(float)
        y = ordered["mean"].to_numpy(float)
        n = ordered["n_units"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= 5.0)
        x, y, n = x[valid], y[valid], n[valid]
        if x.size < 4 or np.any(np.diff(x) <= 0.0):
            continue
        try:
            spline = make_smoothing_spline(x, y, w=np.sqrt(np.maximum(n, 1.0)))
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 201)
            smooth = np.asarray(spline(grid), dtype=float)
            model = "gcv_cubic_smoothing_spline_nonmechanistic"
        except (ValueError, np.linalg.LinAlgError):
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 201)
            smooth = np.interp(grid, x, y)
            model = "linear_interpolation_nonmechanistic_fallback"
        predictions.extend(
            {
                "metric": "vcc",
                "site": "pair",
                "hour_post_delivery": float(hour),
                "model": model,
                "x": float(x_value),
                "y": float(y_value),
                "x_axis": "tau_over_delta",
            }
            for x_value, y_value in zip(grid, smooth, strict=True)
        )

        source = vcc_units.loc[np.isclose(vcc_units["hour_post_delivery"], hour)]
        unit_endpoints: dict[str, list[dict[str, Any]]] = {
            "zero_lag_trace": [],
            "positive_auc_0_5delta": [],
            "first_zero_crossing_tau_over_delta": [],
            "late_mean_3_5delta": [],
        }
        for _, unit in source.groupby("unit_id", sort=False):
            crop_id = str(unit["crop_id"].iloc[0])
            unit = unit.sort_values("scaled_lag_tau_over_delta", kind="stable")
            ux = unit["scaled_lag_tau_over_delta"].to_numpy(float)
            uy = unit["value"].to_numpy(float)
            finite = np.isfinite(ux) & np.isfinite(uy) & (ux >= 0.0) & (ux <= 5.0)
            ux, uy = ux[finite], uy[finite]
            if ux.size == 0:
                continue
            zero_index = int(np.argmin(np.abs(ux)))
            if abs(ux[zero_index]) < 1e-6:
                unit_endpoints["zero_lag_trace"].append(
                    {"crop_id": crop_id, "value": float(uy[zero_index])}
                )
            if ux.size >= 3 and ux[-1] - ux[0] >= 3.0:
                unit_endpoints["positive_auc_0_5delta"].append(
                    {
                        "crop_id": crop_id,
                        "value": float(np.trapezoid(np.maximum(uy, 0.0), ux)),
                    }
                )
                unit_endpoints["first_zero_crossing_tau_over_delta"].append(
                    {"crop_id": crop_id, "value": _first_zero_crossing(ux, uy)}
                )
            late = uy[(ux >= 3.0) & (ux <= 5.0)]
            if late.size:
                unit_endpoints["late_mean_3_5delta"].append(
                    {"crop_id": crop_id, "value": float(np.mean(late))}
                )
        for name, rows in unit_endpoints.items():
            endpoint_table = pd.DataFrame(rows)
            if endpoint_table.empty:
                continue
            endpoint_table["value"] = pd.to_numeric(
                endpoint_table["value"], errors="coerce"
            )
            endpoint_table = endpoint_table.loc[np.isfinite(endpoint_table["value"])].copy()
            if endpoint_table.empty:
                continue
            clusters = sorted(endpoint_table["crop_id"].astype(str).unique())
            bootstrap_means: list[float] = []
            if len(clusters) >= 2 and bootstrap_iterations > 0:
                by_cluster = {
                    cluster: endpoint_table.loc[
                        endpoint_table["crop_id"].astype(str).eq(cluster), "value"
                    ].to_numpy(float)
                    for cluster in clusters
                }
                rng = np.random.default_rng(
                    _group_seed(bootstrap_seed, f"vcc|pair|{hour:g}|{name}")
                )
                for _ in range(bootstrap_iterations):
                    sampled = rng.choice(clusters, size=len(clusters), replace=True)
                    replicate = np.concatenate([by_cluster[cluster] for cluster in sampled])
                    bootstrap_means.append(float(np.mean(replicate)))
            low, high = _ci(bootstrap_means)
            fits.append(
                {
                    "metric": "vcc",
                    "site": "pair",
                    "hour_post_delivery": float(hour),
                    "model": model,
                    "estimate_name": name,
                    "estimate": float(endpoint_table["value"].mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "fit_status": "descriptive_nonmechanistic",
                    "n_fit_points": len(x),
                    "n_units": len(endpoint_table),
                    "n_crops": int(endpoint_table["crop_id"].nunique()),
                    "fit_lag_min": float(np.min(x)),
                    "fit_lag_max": float(np.max(x)),
                    "fit_time_span_fold": np.nan,
                }
            )
    return fits, predictions


def build_time_resolved_physics_summary(
    unit_curves: pd.DataFrame,
    *,
    lag_bin_centers_s: tuple[float, ...] = (
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        7.5,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        75.0,
        100.0,
        150.0,
        200.0,
        300.0,
        500.0,
    ),
    common_fit_window_s: tuple[float, float] = (10.0, 50.0),
    minimum_fit_points: int = 5,
    minimum_fit_span_fold: float = 4.0,
    bootstrap_iterations: int = 500,
    bootstrap_seed: int = 20260823,
) -> TimeResolvedPhysicsSummary:
    """Build pooled hour curves while keeping every calculation unit equal."""

    _require_columns(unit_curves)
    centers = _validate_centers(lag_bin_centers_s)
    lower, upper = (float(common_fit_window_s[0]), float(common_fit_window_s[1]))
    if lower <= 0.0 or upper <= lower:
        raise ValueError("common_fit_window_s must be positive and increasing")
    if minimum_fit_points < 2 or minimum_fit_span_fold <= 1.0:
        raise ValueError("fit support settings are invalid")
    unit_bins = _nearest_unit_bins(unit_curves, centers)
    hour_curves, support = _aggregate_hour_curves(unit_bins)
    fit_rows, prediction_rows = _fit_displacement_and_vac(
        unit_bins,
        hour_curves,
        common_window_s=(lower, upper),
        minimum_fit_points=int(minimum_fit_points),
        minimum_span_fold=float(minimum_fit_span_fold),
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    vcc_fits, vcc_predictions = _summarize_vcc(
        unit_bins,
        hour_curves,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    fit_rows.extend(vcc_fits)
    prediction_rows.extend(vcc_predictions)
    fits = pd.DataFrame(fit_rows)
    predictions = pd.DataFrame(prediction_rows)
    contract = {
        "experimental_batch_assumption": {
            "same_batch": True,
            "batch_effect_model": None,
            "evidence": "user confirmation 2026-08-23",
            "acquisition_role": "provenance and cadence/support diagnostics only",
        },
        "pooling": {
            "macro_time": "folder hour after Cas9 delivery; not cut onset",
            "primary_unit": {
                "msd_vac": "trajectory",
                "mscd_vcc": "Site1/Site2 bundle",
            },
            "within_hour_weight": "equal unit after within-unit time averaging",
            "uncertainty": "between-unit sample SD; crop-cluster bootstrap 95% CI for model fits",
        },
        "support_only_filter": {
            "coordinates": "finite observed endpoints on exact acquisition schedule",
            "minimum_pairs_per_unit_lag": 8,
            "coordinate_interpolation": False,
            "excluded_filters": [
                "T3/T4 global paired QC",
                "motion magnitude",
                "separation",
                "53BP1 detection or recruitment",
                "hour",
                "acquisition identity",
                "learned state or cluster",
            ],
        },
        "physical_time_harmonization": {
            "coordinate_interpolation": False,
            "lag_binning_only": True,
            "lag_bin_centers_s": centers.tolist(),
            "representative_per_unit_bin": "supported curve point nearest bin center",
        },
        "models": {
            "msd": {
                "formula": "MSD(tau)=A*tau^alpha",
                "fit_window_s": [lower, upper],
                "status": "descriptive raw MSD; localization/motion-blur correction unavailable",
            },
            "mscd": {
                "formula": "MSCD(tau)=A*tau^beta",
                "fit_window_s": [lower, upper],
                "status": "descriptive relative-vector change",
            },
            "vac": {
                "formula": "(|x-1|^alpha+|x+1|^alpha-2|x|^alpha)/2; x=tau/delta",
                "source": "Oligo-LiveFISH Eq. 9",
                "primary_delta": "acquisition-specific integer frame offset nearest 10 s, constrained to 7.5-12.5 s",
            },
            "vcc": {
                "primary_object": "lag-indexed normalized 2x2 matrix in source table",
                "headline": "mean of bidirectional rotation-invariant traces",
                "fit": "GCV smoothing spline for display only; no Rouse communication-time claim",
            },
        },
        "warnings": [
            "10-50 s spans only five-fold, so power-law exponents are weakly identified.",
            "5 h has only about 10 Site1 trajectories and 9 paired bundles.",
            "Acquisitions are pooled biologically but retain unequal cadence and movie length.",
            "Tracking missingness can remain outcome-dependent at long lags.",
        ],
    }
    # Round-trip here catches accidental non-JSON scalar types before writing.
    json.dumps(contract)
    return TimeResolvedPhysicsSummary(
        unit_bins=unit_bins,
        hour_curves=hour_curves,
        fits=fits,
        predicted_curves=predictions,
        support_by_hour=support,
        method_contract=contract,
    )
