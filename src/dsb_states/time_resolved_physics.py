"""Metric-specific, full-lag physical curves from the unfiltered v5.2.1 cache.

This module is deliberately computation-only.  It aligns every bundle to the
complete, one-based frame schedule of its acquisition and leaves unobserved
coordinates as ``NaN``.  Coordinates are never interpolated and omitted rows
are therefore never compressed into apparently adjacent frames.

The only eligibility rule is metric support: at least ``min_pairs`` directly
observed endpoint or velocity pairs are required for an emitted unit-lag
value.  No T3/T4, motion, separation, hour, acquisition, 53BP1, or state filter
is applied here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .physical_metrics import (
    mean_squared_change_in_distance_by_frame_steps,
    mean_squared_displacement_by_frame_steps,
    velocity_autocorrelation_by_frame_steps,
    velocity_cross_correlation_matrix_by_frame_steps,
    velocity_series_by_frame_steps,
)

_FRAME_COLUMNS = {
    "bundle_id",
    "nd2_id",
    "frame",
    "time_s",
    "site1_x_um",
    "site1_y_um",
    "site2_x_um",
    "site2_y_um",
}
_INDEX_COLUMNS = {
    "bundle_id",
    "cohort",
    "nd2_id",
    "crop_id",
    "fov_id",
    "hour_post_delivery",
    "movie_frames",
}
_UNIT_CURVE_COLUMNS = [
    "metric",
    "unit_id",
    "bundle_id",
    "cohort",
    "nd2_id",
    "crop_id",
    "fov_id",
    "hour_post_delivery",
    "site",
    "direction",
    "delta_frames",
    "delta_role",
    "is_primary_matched_10s",
    "delta_median_s",
    "lag_frames",
    "lag_median_s",
    "scaled_lag_tau_over_delta",
    "value",
    "raw_value",
    "endpoint_pair_sd",
    "pair_count",
    "normalization",
    "value_unit",
    "raw_value_unit",
    "endpoint_pair_sd_unit",
]
_VCC_MATRIX_COLUMNS = [
    "metric",
    "unit_id",
    "bundle_id",
    "cohort",
    "nd2_id",
    "crop_id",
    "fov_id",
    "hour_post_delivery",
    "site",
    "direction",
    "delta_frames",
    "delta_role",
    "is_primary_matched_10s",
    "delta_median_s",
    "lag_frames",
    "lag_median_s",
    "scaled_lag_tau_over_delta",
    "row_component",
    "column_component",
    "normalized_value",
    "raw_value",
    "pair_count",
    "normalization",
    "normalized_value_unit",
    "raw_value_unit",
]
_CENSUS_COLUMNS = [
    "metric",
    "unit_id",
    "bundle_id",
    "cohort",
    "nd2_id",
    "crop_id",
    "fov_id",
    "hour_post_delivery",
    "site",
    "direction",
    "delta_frames",
    "delta_role",
    "is_primary_matched_10s",
    "primary_matched_delta_frames",
    "primary_matched_delta_median_s",
    "movie_frames",
    "max_lag_frames",
    "observed_site1_frames",
    "observed_site2_frames",
    "shared_site_frames",
    "attempted_lag_count",
    "emitted_lag_count",
    "lag_count_below_min_pairs",
    "lag_count_invalid_after_pair_threshold",
    "maximum_pair_count",
    "min_pairs_required",
    "unit_included",
    "exclusion_reason",
]


@dataclass(frozen=True)
class TimeResolvedPhysicsUnits:
    """Tidy per-unit curves, full VCC matrices, and support accounting."""

    unit_curves: pd.DataFrame
    vcc_matrices: pd.DataFrame
    support_census: pd.DataFrame


@dataclass(frozen=True)
class _AcquisitionSchedule:
    nd2_id: str
    frames: np.ndarray
    times_s: np.ndarray
    matched_delta_frames: int | None
    matched_delta_median_s: float


def _require_columns(table: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 1 or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _matched_measurement_delta(
    times_s: np.ndarray,
    *,
    target_s: float,
    acceptable_s: tuple[float, float],
) -> tuple[int | None, float]:
    """Choose the frame offset whose acquisition-median duration is nearest target."""

    lower, upper = acceptable_s
    candidates: list[tuple[float, int, float]] = []
    for delta in range(1, times_s.size):
        median_s = float(np.median(times_s[delta:] - times_s[:-delta]))
        candidates.append((abs(median_s - target_s), delta, median_s))
    if not candidates:
        return None, float("nan")
    _, delta, median_s = min(candidates, key=lambda item: (item[0], item[1]))
    if median_s < lower or median_s > upper:
        return None, median_s
    return delta, median_s


def _build_acquisition_schedules(
    bundle_frames: pd.DataFrame,
    bundle_index: pd.DataFrame,
    *,
    primary_delta_target_s: float,
    primary_delta_acceptable_s: tuple[float, float],
) -> dict[str, _AcquisitionSchedule]:
    schedules: dict[str, _AcquisitionSchedule] = {}
    index_nd2 = bundle_index["nd2_id"].astype(str)
    frame_nd2 = bundle_frames["nd2_id"].astype(str)

    for nd2_id in sorted(index_nd2.unique()):
        movie_values = pd.to_numeric(
            bundle_index.loc[index_nd2.eq(nd2_id), "movie_frames"], errors="raise"
        ).unique()
        if movie_values.size != 1:
            raise ValueError(f"acquisition {nd2_id!r} has inconsistent movie_frames")
        movie_frames = _positive_integer(movie_values[0], name="movie_frames")
        acquisition = bundle_frames.loc[
            frame_nd2.eq(nd2_id), ["frame", "time_s"]
        ].copy()
        if acquisition.empty:
            raise ValueError(f"acquisition {nd2_id!r} has no exact frame/time schedule")
        acquisition["frame"] = pd.to_numeric(acquisition["frame"], errors="raise")
        acquisition["time_s"] = pd.to_numeric(acquisition["time_s"], errors="raise")
        if not np.all(np.isfinite(acquisition[["frame", "time_s"]].to_numpy(float))):
            raise ValueError(f"acquisition {nd2_id!r} has non-finite frame/time values")
        if not np.all(acquisition["frame"].eq(np.floor(acquisition["frame"]))):
            raise ValueError(f"acquisition {nd2_id!r} contains non-integer frames")

        timing = acquisition.groupby("frame", sort=True)["time_s"].agg(["min", "median", "max"])
        expected = np.arange(1, movie_frames + 1, dtype=np.int64)
        observed = timing.index.to_numpy(dtype=np.int64)
        if not np.array_equal(observed, expected):
            missing = sorted(set(expected).difference(observed))
            raise ValueError(
                f"acquisition {nd2_id!r} does not expose the complete one-based schedule; "
                f"missing frames: {missing[:10]}"
            )
        medians = timing["median"].to_numpy(float)
        timing_span = (timing["max"] - timing["min"]).to_numpy(float)
        tolerance = 1e-9 * np.maximum(1.0, np.abs(medians))
        if np.any(timing_span > tolerance):
            raise ValueError(f"acquisition {nd2_id!r} has conflicting exact times for a frame")
        if np.any(np.diff(medians) <= 0.0):
            raise ValueError(f"acquisition {nd2_id!r} times are not strictly increasing")
        matched_delta, matched_seconds = _matched_measurement_delta(
            medians,
            target_s=primary_delta_target_s,
            acceptable_s=primary_delta_acceptable_s,
        )
        schedules[nd2_id] = _AcquisitionSchedule(
            nd2_id=nd2_id,
            frames=expected,
            times_s=medians,
            matched_delta_frames=matched_delta,
            matched_delta_median_s=matched_seconds,
        )
    return schedules


def _dense_coordinates(
    bundle_rows: pd.DataFrame,
    schedule: _AcquisitionSchedule,
) -> tuple[np.ndarray, np.ndarray]:
    site1 = np.full((schedule.frames.size, 2), np.nan, dtype=float)
    site2 = np.full((schedule.frames.size, 2), np.nan, dtype=float)
    if bundle_rows.empty:
        return site1, site2

    frames_numeric = pd.to_numeric(bundle_rows["frame"], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(frames_numeric)) or not np.all(frames_numeric == np.floor(frames_numeric)):
        raise ValueError("bundle frame values must be finite integers")
    frames = frames_numeric.astype(np.int64)
    if np.any(frames < 1) or np.any(frames > schedule.frames.size):
        raise ValueError("bundle frame lies outside its acquisition schedule")
    times = pd.to_numeric(bundle_rows["time_s"], errors="raise").to_numpy(float)
    expected_times = schedule.times_s[frames - 1]
    if not np.allclose(times, expected_times, rtol=1e-12, atol=1e-9, equal_nan=False):
        raise ValueError("bundle time_s conflicts with its acquisition schedule")

    indices = frames - 1
    site1[indices] = bundle_rows[["site1_x_um", "site1_y_um"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    site2[indices] = bundle_rows[["site2_x_um", "site2_y_um"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    return site1, site2


def _base_identity(record: Any) -> dict[str, Any]:
    return {
        "unit_id": str(record.bundle_id),
        "bundle_id": str(record.bundle_id),
        "cohort": record.cohort,
        "nd2_id": str(record.nd2_id),
        "crop_id": record.crop_id,
        "fov_id": record.fov_id,
        "hour_post_delivery": float(record.hour_post_delivery),
    }


def _delta_specifications(
    schedule: _AcquisitionSchedule,
    paper_delta_frames: tuple[int, ...],
) -> list[tuple[int, str, bool]]:
    specifications: list[tuple[int, str, bool]] = []
    matched = schedule.matched_delta_frames
    for delta in sorted(set(paper_delta_frames).union(() if matched is None else (matched,))):
        in_paper_family = delta in paper_delta_frames
        specifications.append(
            (
                delta,
                "paper_family" if in_paper_family else "primary_matched_10s",
                delta == matched,
            )
        )
    return specifications


def _median_velocity_duration(
    coordinates: np.ndarray,
    times_s: np.ndarray,
    delta_frames: int,
) -> float:
    series = velocity_series_by_frame_steps(
        coordinates,
        times=times_s,
        measurement_delta_frames=delta_frames,
    )
    if series.n_velocities == 0:
        return float("nan")
    return float(np.median(series.durations))


def _paired_median_velocity_duration(
    site1: np.ndarray,
    site2: np.ndarray,
    times_s: np.ndarray,
    delta_frames: int,
) -> float:
    durations = []
    for coordinates in (site1, site2):
        series = velocity_series_by_frame_steps(
            coordinates,
            times=times_s,
            measurement_delta_frames=delta_frames,
        )
        if series.n_velocities:
            durations.append(series.durations)
    if not durations:
        return float("nan")
    return float(np.median(np.concatenate(durations)))


def _append_census(
    rows: list[dict[str, Any]],
    *,
    identity: dict[str, Any],
    metric: str,
    site: str,
    delta_frames: int | None,
    delta_role: str | None,
    is_primary: bool,
    schedule: _AcquisitionSchedule,
    max_lag_frames: int,
    observed_counts: tuple[int, int, int],
    counts: np.ndarray,
    finite_values: np.ndarray,
    min_pairs: int,
    direction: str | None = None,
) -> None:
    pair_supported = counts >= min_pairs
    emitted = pair_supported & finite_values
    if np.any(emitted):
        reason = "included"
    elif not np.any(pair_supported):
        reason = "insufficient_valid_pairs_all_lags"
    elif np.any(pair_supported & ~finite_values):
        reason = "nonpositive_or_unavailable_normalization"
    else:
        reason = "no_supported_lags"
    rows.append(
        {
            **identity,
            "metric": metric,
            "site": site,
            "direction": direction,
            "delta_frames": delta_frames,
            "delta_role": delta_role,
            "is_primary_matched_10s": bool(is_primary),
            "primary_matched_delta_frames": schedule.matched_delta_frames,
            "primary_matched_delta_median_s": schedule.matched_delta_median_s,
            "movie_frames": int(schedule.frames.size),
            "max_lag_frames": max_lag_frames,
            "observed_site1_frames": observed_counts[0],
            "observed_site2_frames": observed_counts[1],
            "shared_site_frames": observed_counts[2],
            "attempted_lag_count": int(counts.size),
            "emitted_lag_count": int(np.count_nonzero(emitted)),
            "lag_count_below_min_pairs": int(np.count_nonzero(~pair_supported)),
            "lag_count_invalid_after_pair_threshold": int(
                np.count_nonzero(pair_supported & ~finite_values)
            ),
            "maximum_pair_count": int(np.max(counts)) if counts.size else 0,
            "min_pairs_required": min_pairs,
            "unit_included": bool(np.any(emitted)),
            "exclusion_reason": reason,
        }
    )


def _empty_or_ordered(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=columns)
    for column in ("delta_frames", "lag_frames"):
        if column in result:
            result[column] = pd.array(result[column], dtype="Int64")
    return result


def compute_time_resolved_physics(
    bundle_frames: pd.DataFrame,
    bundle_index: pd.DataFrame,
    *,
    max_lag_cap: int = 50,
    max_lag_fraction: float = 0.25,
    paper_delta_frames: tuple[int, ...] = (1, 2, 4, 8),
    min_pairs: int = 8,
    primary_delta_target_s: float = 10.0,
    primary_delta_acceptable_s: tuple[float, float] = (7.5, 12.5),
) -> TimeResolvedPhysicsUnits:
    """Compute unfiltered, metric-support-only MSD/VAC/MSCD/VCC unit curves.

    MSD and MSCD values (and endpoint-pair sample SDs) are in micrometres
    squared.  VAC and VCC values are dimensionless zero-lag-energy-normalized
    correlations; their raw values and normalization scales are in
    micrometres squared per seconds squared.
    """

    _require_columns(bundle_frames, _FRAME_COLUMNS, name="bundle_frames")
    _require_columns(bundle_index, _INDEX_COLUMNS, name="bundle_index")
    cap = _positive_integer(max_lag_cap, name="max_lag_cap")
    minimum = _positive_integer(min_pairs, name="min_pairs")
    fraction = float(max_lag_fraction)
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError("max_lag_fraction must be finite and in (0, 1]")
    deltas = tuple(_positive_integer(value, name="paper_delta_frames") for value in paper_delta_frames)
    if len(set(deltas)) != len(deltas):
        raise ValueError("paper_delta_frames must not contain duplicates")
    target = float(primary_delta_target_s)
    acceptable = tuple(float(value) for value in primary_delta_acceptable_s)
    if (
        not np.isfinite(target)
        or target <= 0.0
        or len(acceptable) != 2
        or not np.all(np.isfinite(acceptable))
        or acceptable[0] <= 0.0
        or acceptable[0] > acceptable[1]
    ):
        raise ValueError("primary matched-delta target/tolerance is invalid")

    if bundle_index["bundle_id"].duplicated().any():
        raise ValueError("bundle_index contains duplicate bundle_id values")
    if bundle_frames.duplicated(["bundle_id", "frame"]).any():
        raise ValueError("bundle_frames contains duplicate bundle_id/frame rows")
    known_bundles = set(bundle_index["bundle_id"].astype(str))
    unknown_bundles = set(bundle_frames["bundle_id"].astype(str)).difference(known_bundles)
    if unknown_bundles:
        raise ValueError(f"bundle_frames contains bundle IDs absent from bundle_index: {sorted(unknown_bundles)[:5]}")

    schedules = _build_acquisition_schedules(
        bundle_frames,
        bundle_index,
        primary_delta_target_s=target,
        primary_delta_acceptable_s=(acceptable[0], acceptable[1]),
    )
    grouped = bundle_frames.groupby(bundle_frames["bundle_id"].astype(str), sort=False)
    group_indices = grouped.indices
    unit_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    census_rows: list[dict[str, Any]] = []

    for record in bundle_index.itertuples(index=False):
        identity = _base_identity(record)
        schedule = schedules[identity["nd2_id"]]
        indices = group_indices.get(identity["bundle_id"])
        rows = bundle_frames.iloc[indices] if indices is not None else bundle_frames.iloc[0:0]
        if not rows.empty and not rows["nd2_id"].astype(str).eq(identity["nd2_id"]).all():
            raise ValueError(f"bundle {identity['bundle_id']!r} has inconsistent nd2_id metadata")
        site1, site2 = _dense_coordinates(rows, schedule)
        site1_observed = np.all(np.isfinite(site1), axis=1)
        site2_observed = np.all(np.isfinite(site2), axis=1)
        observed_counts = (
            int(np.count_nonzero(site1_observed)),
            int(np.count_nonzero(site2_observed)),
            int(np.count_nonzero(site1_observed & site2_observed)),
        )
        max_lag = min(cap, int(np.floor(fraction * schedule.frames.size)))
        displacement_lags = np.arange(1, max_lag + 1, dtype=np.int64)
        correlation_lags = np.arange(0, max_lag + 1, dtype=np.int64)

        for site_name, coordinates in (("site1", site1), ("site2", site2)):
            msd = mean_squared_displacement_by_frame_steps(
                coordinates,
                times=schedule.times_s,
                lag_steps=displacement_lags,
                min_pairs=minimum,
            )
            finite = np.isfinite(msd.values)
            for index in np.flatnonzero(finite):
                unit_rows.append(
                    {
                        **identity,
                        "metric": "msd",
                        "site": site_name,
                        "delta_frames": None,
                        "delta_role": None,
                        "is_primary_matched_10s": False,
                        "delta_median_s": np.nan,
                        "lag_frames": int(msd.lag_steps[index]),
                        "lag_median_s": float(msd.lag_time_median[index]),
                        "scaled_lag_tau_over_delta": np.nan,
                        "value": float(msd.values[index]),
                        "raw_value": float(msd.values[index]),
                        "endpoint_pair_sd": float(msd.standard_deviations[index]),
                        "pair_count": int(msd.counts[index]),
                        "normalization": np.nan,
                        "value_unit": "um^2",
                        "raw_value_unit": "um^2",
                        "endpoint_pair_sd_unit": "um^2",
                    }
                )
            _append_census(
                census_rows,
                identity=identity,
                metric="msd",
                site=site_name,
                delta_frames=None,
                delta_role=None,
                is_primary=False,
                schedule=schedule,
                max_lag_frames=max_lag,
                observed_counts=observed_counts,
                counts=msd.counts,
                finite_values=finite,
                min_pairs=minimum,
            )

            for delta, delta_role, is_primary in _delta_specifications(schedule, deltas):
                vac = velocity_autocorrelation_by_frame_steps(
                    coordinates,
                    times=schedule.times_s,
                    measurement_delta_frames=delta,
                    lag_frames=correlation_lags,
                    min_pairs=minimum,
                )
                delta_median_s = _median_velocity_duration(
                    coordinates, schedule.times_s, delta
                )
                finite = np.isfinite(vac.values)
                for index in np.flatnonzero(finite):
                    lag_seconds = float(vac.start_time_lag_median[index])
                    unit_rows.append(
                        {
                            **identity,
                            "metric": "vac",
                            "site": site_name,
                            "delta_frames": delta,
                            "delta_role": delta_role,
                            "is_primary_matched_10s": is_primary,
                            "delta_median_s": delta_median_s,
                            "lag_frames": int(vac.lag_frames[index]),
                            "lag_median_s": lag_seconds,
                            "scaled_lag_tau_over_delta": lag_seconds / delta_median_s,
                            "value": float(vac.values[index]),
                            "raw_value": float(vac.raw_values[index]),
                            "endpoint_pair_sd": np.nan,
                            "pair_count": int(vac.counts[index]),
                            "normalization": float(vac.normalization),
                            "value_unit": "dimensionless",
                            "raw_value_unit": "um^2/s^2",
                            "endpoint_pair_sd_unit": None,
                        }
                    )
                _append_census(
                    census_rows,
                    identity=identity,
                    metric="vac",
                    site=site_name,
                    delta_frames=delta,
                    delta_role=delta_role,
                    is_primary=is_primary,
                    schedule=schedule,
                    max_lag_frames=max_lag,
                    observed_counts=observed_counts,
                    counts=vac.counts,
                    finite_values=finite,
                    min_pairs=minimum,
                )

        mscd = mean_squared_change_in_distance_by_frame_steps(
            site1,
            site2,
            times=schedule.times_s,
            lag_steps=displacement_lags,
            min_pairs=minimum,
        )
        finite_mscd = np.isfinite(mscd.values)
        for index in np.flatnonzero(finite_mscd):
            unit_rows.append(
                {
                    **identity,
                    "metric": "mscd",
                    "site": "pair",
                    "delta_frames": None,
                    "delta_role": None,
                    "is_primary_matched_10s": False,
                    "delta_median_s": np.nan,
                    "lag_frames": int(mscd.lag_steps[index]),
                    "lag_median_s": float(mscd.lag_time_median[index]),
                    "scaled_lag_tau_over_delta": np.nan,
                    "value": float(mscd.values[index]),
                    "raw_value": float(mscd.values[index]),
                    "endpoint_pair_sd": float(mscd.standard_deviations[index]),
                    "pair_count": int(mscd.counts[index]),
                    "normalization": np.nan,
                    "value_unit": "um^2",
                    "raw_value_unit": "um^2",
                    "endpoint_pair_sd_unit": "um^2",
                }
            )
        _append_census(
            census_rows,
            identity=identity,
            metric="mscd",
            site="pair",
            delta_frames=None,
            delta_role=None,
            is_primary=False,
            schedule=schedule,
            max_lag_frames=max_lag,
            observed_counts=observed_counts,
            counts=mscd.counts,
            finite_values=finite_mscd,
            min_pairs=minimum,
        )

        for delta, delta_role, is_primary in _delta_specifications(schedule, deltas):
            delta_median_s = _paired_median_velocity_duration(
                site1, site2, schedule.times_s, delta
            )
            directions = (
                ("site1_later_vs_site2", site1, site2),
                ("site2_later_vs_site1", site2, site1),
            )
            for direction, later_site, earlier_site in directions:
                vcc = velocity_cross_correlation_matrix_by_frame_steps(
                    later_site,
                    earlier_site,
                    times=schedule.times_s,
                    measurement_delta_frames=delta,
                    lag_frames=correlation_lags,
                    min_pairs=minimum,
                )
                trace_values = vcc.trace_values
                raw_trace_values = vcc.raw_trace_values
                finite_vcc = np.isfinite(trace_values)
                for index in np.flatnonzero(finite_vcc):
                    lag_seconds = float(vcc.start_time_lag_median[index])
                    common = {
                        **identity,
                        "metric": "vcc",
                        "site": "pair",
                        "direction": direction,
                        "delta_frames": delta,
                        "delta_role": delta_role,
                        "is_primary_matched_10s": is_primary,
                        "delta_median_s": delta_median_s,
                        "lag_frames": int(vcc.lag_frames[index]),
                        "lag_median_s": lag_seconds,
                        "scaled_lag_tau_over_delta": lag_seconds / delta_median_s,
                        "pair_count": int(vcc.counts[index]),
                        "normalization": float(vcc.normalization),
                    }
                    unit_rows.append(
                        {
                            **common,
                            "value": float(trace_values[index]),
                            "raw_value": float(raw_trace_values[index]),
                            "endpoint_pair_sd": np.nan,
                            "value_unit": "dimensionless",
                            "raw_value_unit": "um^2/s^2",
                            "endpoint_pair_sd_unit": None,
                        }
                    )
                    for row_component, row_index in (("x", 0), ("y", 1)):
                        for column_component, column_index in (("x", 0), ("y", 1)):
                            matrix_rows.append(
                                {
                                    **common,
                                    "row_component": row_component,
                                    "column_component": column_component,
                                    "normalized_value": float(
                                        vcc.matrices[index, row_index, column_index]
                                    ),
                                    "raw_value": float(
                                        vcc.raw_matrices[index, row_index, column_index]
                                    ),
                                    "normalized_value_unit": "dimensionless",
                                    "raw_value_unit": "um^2/s^2",
                                }
                            )
                _append_census(
                    census_rows,
                    identity=identity,
                    metric="vcc",
                    site="pair",
                    delta_frames=delta,
                    delta_role=delta_role,
                    is_primary=is_primary,
                    schedule=schedule,
                    max_lag_frames=max_lag,
                    observed_counts=observed_counts,
                    counts=vcc.counts,
                    finite_values=finite_vcc,
                    min_pairs=minimum,
                    direction=direction,
                )

    return TimeResolvedPhysicsUnits(
        unit_curves=_empty_or_ordered(unit_rows, _UNIT_CURVE_COLUMNS),
        vcc_matrices=_empty_or_ordered(matrix_rows, _VCC_MATRIX_COLUMNS),
        support_census=_empty_or_ordered(census_rows, _CENSUS_COLUMNS),
    )
