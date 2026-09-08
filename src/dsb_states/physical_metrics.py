"""Non-parametric physical metrics for trajectories on their true time support.

Exact-time APIs match observed timestamps directly.  Separate primary
frame-step APIs pair rows ``i`` and ``i + k`` and report the empirical
elapsed-time distribution for each ``k``; they remain well-defined for jittered
or nonuniform timestamps.  An exact-time API called with an implicit lag grid
fails closed when that grid has no support, rather than returning a silently
empty curve.
Coordinates are never interpolated, and every curve carries valid-pair counts.
Lags, measurement intervals, and timestamps use the same arbitrary time unit
(frames when ``times`` is omitted).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pair_modes import relative_mode

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class LagCurve:
    """A lagged scalar metric with explicit valid-pair counts."""

    lags: FloatArray
    values: FloatArray
    counts: IntArray

    def __iter__(self) -> Iterator[FloatArray | IntArray]:
        """Allow explicit tuple unpacking as ``lags, values, counts``."""

        yield self.lags
        yield self.values
        yield self.counts

    @property
    def valid(self) -> NDArray[np.bool_]:
        return np.isfinite(self.values) & (self.counts > 0)


@dataclass(frozen=True)
class SquaredDisplacementLagCurve(LagCurve):
    """MSD/MSCD means with endpoint-pair sample SD at every lag.

    ``standard_deviations`` is the sample SD (``ddof=1``) of the individual
    squared-displacement observations entering ``values``.  It describes the
    endpoint-pair distribution and is not a standard error or confidence
    interval for the mean.
    """

    standard_deviations: FloatArray


@dataclass(frozen=True)
class FrameStepLagCurve:
    """Squared-displacement curve indexed by observed-row offset.

    ``lag_steps`` is the row offset ``k``.  The four ``lag_time_*`` arrays
    describe the true elapsed times among the same finite-endpoint pairs used
    for each metric value.  This makes irregular acquisition timing visible
    rather than silently treating every row step as an identical duration.
    """

    lag_steps: IntArray
    values: FloatArray
    counts: IntArray
    lag_time_mean: FloatArray
    lag_time_median: FloatArray
    lag_time_min: FloatArray
    lag_time_max: FloatArray
    standard_deviations: FloatArray

    def __iter__(self) -> Iterator[FloatArray | IntArray]:
        """Allow explicit tuple unpacking as ``steps, values, counts``."""

        yield self.lag_steps
        yield self.values
        yield self.counts

    @property
    def valid(self) -> NDArray[np.bool_]:
        return np.isfinite(self.values) & (self.counts > 0)

    @property
    def lag_time_count(self) -> IntArray:
        """Counts underlying the elapsed-time summaries (same metric support)."""

        return self.counts


@dataclass(frozen=True)
class FrameStepCrossCorrelationMatrixCurve:
    """Lag-indexed 2-D cross-component VCC matrices.

    For each lag ``tau``, ``raw_matrices`` stores
    ``<v_site1(t+tau) v_site2(t)^T>`` with shape ``(n_lags, 2, 2)``.
    ``matrices`` divides every component by the fixed, rotation-invariant
    zero-lag energy scale
    ``sqrt(<|v_site1|^2><|v_site2|^2>)``.  Matrix components depend on the
    image coordinate basis; their trace is rotation invariant and exactly
    reproduces the historical scalar VCC compatibility endpoint.
    """

    lag_frames: IntArray
    matrices: FloatArray
    counts: IntArray
    start_time_lag_mean: FloatArray
    start_time_lag_median: FloatArray
    start_time_lag_min: FloatArray
    start_time_lag_max: FloatArray
    raw_matrices: FloatArray
    normalization: float
    lag_specific_normalizations: FloatArray
    measurement_delta_frames: int

    @property
    def valid(self) -> NDArray[np.bool_]:
        return np.all(np.isfinite(self.matrices), axis=(1, 2)) & (self.counts > 0)

    @property
    def trace_values(self) -> FloatArray:
        """Rotation-invariant scalar compatibility reduction."""

        return np.trace(self.matrices, axis1=1, axis2=2)

    @property
    def raw_trace_values(self) -> FloatArray:
        """Unnormalized mean velocity dot products at each lag."""

        return np.trace(self.raw_matrices, axis1=1, axis2=2)


@dataclass(frozen=True)
class FrameStepCorrelationCurve:
    """Velocity correlation indexed by original velocity-start row offset.

    ``values`` contains the normalized correlation and ``raw_values`` the mean
    dot product.  ``normalizations`` contains the elementwise denominators that
    were actually applied.  For VCC, the master-plan primary normalization is
    the fixed zero-lag energy scale in ``normalization`` at every lag.  A
    lag-specific bounded sensitivity is exposed only through fields whose names
    explicitly include ``secondary``.  The
    ``start_time_lag_*`` arrays describe the true elapsed time between the
    velocity starts contributing at each ``lag_frames`` value.
    """

    lag_frames: IntArray
    values: FloatArray
    counts: IntArray
    start_time_lag_mean: FloatArray
    start_time_lag_median: FloatArray
    start_time_lag_min: FloatArray
    start_time_lag_max: FloatArray
    raw_values: FloatArray
    normalization: float
    normalizations: FloatArray | None = None
    secondary_lag_specific_values: FloatArray | None = None
    secondary_lag_specific_normalizations: FloatArray | None = None

    def __iter__(self) -> Iterator[FloatArray | IntArray]:
        """Allow explicit tuple unpacking as ``lags, values, counts``."""

        yield self.lag_frames
        yield self.values
        yield self.counts

    @property
    def valid(self) -> NDArray[np.bool_]:
        return np.isfinite(self.values) & (self.counts > 0)

    @property
    def start_time_lag_count(self) -> IntArray:
        """Counts underlying the start-time-lag summaries."""

        return self.counts


@dataclass(frozen=True)
class CorrelationCurve(LagCurve):
    """Raw and normalized velocity correlation on a common lag grid.

    ``values`` contains the normalized correlation and ``raw_values`` contains
    the mean dot product.  ``normalizations`` contains the elementwise
    denominators that were actually applied.  For VCC, ``normalization`` is the
    master-plan fixed zero-lag energy scale applied at every lag.  The optional
    lag-specific bounded sensitivity is explicitly marked ``secondary``.
    """

    raw_values: FloatArray
    normalization: float
    normalizations: FloatArray | None = None
    secondary_lag_specific_values: FloatArray | None = None
    secondary_lag_specific_normalizations: FloatArray | None = None


@dataclass(frozen=True)
class CurveIntegral:
    """Guarded trapezoidal integral over directly supported adjacent lags.

    ``supported_duration`` is the sum of elapsed intervals actually integrated;
    unsupported internal lags are never bridged.  An invalid result carries a
    ``NaN`` area and a machine-readable reason instead of becoming a feature.
    """

    area: float
    valid: bool
    reason: str | None
    supported_lag_count: int
    supported_interval_count: int
    supported_duration: float
    minimum_duration: float
    min_pairs_per_lag: int
    positive_only: bool
    maximum_lag_time: float | None


class ExactTimeSupportError(ValueError):
    """Raised when an implicit exact-time lag grid has no supported pairs."""


@dataclass(frozen=True)
class VelocitySeries:
    """Observed finite-difference velocities retaining original endpoint rows."""

    start_times: FloatArray
    end_times: FloatArray
    durations: FloatArray
    vectors: FloatArray
    start_indices: IntArray
    end_indices: IntArray

    @property
    def n_velocities(self) -> int:
        return int(self.start_times.size)


@dataclass(frozen=True)
class FrameStepVelocitySeries(VelocitySeries):
    """Finite-endpoint velocities retaining their original trajectory rows."""

    measurement_delta_frames: int


@dataclass(frozen=True)
class AngularDistribution:
    """Directional-change histogram over angles in radians from zero to pi."""

    bin_edges: FloatArray
    counts: IntArray
    probabilities: FloatArray
    n_angles: int


@dataclass(frozen=True)
class TurningPersistence:
    """Near-zero and near-pi turning-angle summaries."""

    n_angles: int
    threshold_radians: float
    near_zero_probability: float
    near_pi_probability: float
    persistence_score: float
    null_near_zero_probability: float
    null_near_pi_probability: float
    near_zero_excess: float
    near_pi_excess: float
    null_corrected_score: float
    n_shuffles: int


@dataclass(frozen=True)
class FrameStepTurningAngles:
    """Turning angles with original-row identity and true cadence metadata."""

    measurement_delta_frames: int
    angles: FloatArray
    first_step_start_indices: IntArray
    vertex_indices: IntArray
    second_step_end_indices: IntArray
    vertex_times: FloatArray
    first_step_durations: FloatArray
    second_step_durations: FloatArray
    full_turn_durations: FloatArray

    @property
    def n_angles(self) -> int:
        return int(self.angles.size)


@dataclass(frozen=True)
class FrameStepAngularDistribution(AngularDistribution):
    """DCD histogram plus the exact row/time support used to construct it."""

    turns: FrameStepTurningAngles


@dataclass(frozen=True)
class FrameStepTurningPersistence(TurningPersistence):
    """Persistence summary plus the exact row/time support behind it."""

    turns: FrameStepTurningAngles


@dataclass(frozen=True)
class DirectedBurstCandidate:
    """Descriptive run of consecutive near-zero frame-step turning angles.

    This is a screening candidate, not a fitted state or biological label.
    Thresholds are supplied explicitly by the caller and should be included in
    downstream sensitivity analyses.
    """

    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration: float
    measurement_delta_frames: int
    n_consecutive_turns: int
    mean_angle_radians: float
    maximum_angle_radians: float


def _as_coordinates(values: ArrayLike, *, name: str = "positions") -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D coordinate array")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one coordinate dimension")
    return np.array(array, dtype=float, copy=True)


def _as_times(times: ArrayLike | None, n_observations: int) -> FloatArray:
    if times is None:
        result = np.arange(n_observations, dtype=float)
    else:
        result = np.asarray(times, dtype=float)
        if result.ndim != 1 or result.size != n_observations:
            raise ValueError("times must be one-dimensional and match the coordinate rows")
        result = np.array(result, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("times must contain only finite values")
    if result.size > 1 and np.any(np.diff(result) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return result


def _as_lags(lags: ArrayLike) -> FloatArray:
    result = np.asarray(lags, dtype=float)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("lags must be a finite one-dimensional array")
    if np.any(result < 0.0):
        raise ValueError("lags must be non-negative")
    return np.array(result, dtype=float, copy=True)


def _as_lag_steps(lag_steps: ArrayLike) -> IntArray:
    result = np.asarray(lag_steps)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1:
        raise ValueError("lag_steps must be a one-dimensional integer array")
    try:
        numeric = np.asarray(result, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("lag_steps must be a one-dimensional integer array") from error
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError("lag_steps must contain only non-negative integers")
    if np.any(numeric < 0.0):
        raise ValueError("lag_steps must contain only non-negative integers")
    if np.any(numeric > np.iinfo(np.int64).max):
        raise ValueError("lag_steps values exceed the supported integer range")
    return numeric.astype(np.int64)


def _positive_frame_offset(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if not np.isfinite(numeric) or numeric != np.floor(numeric) or numeric < 1.0:
        raise ValueError(f"{name} must be a positive integer")
    if numeric > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds the supported integer range")
    return int(numeric)


def _match_frame_shift(
    later_start_indices: IntArray,
    earlier_start_indices: IntArray,
    lag_frames: int,
) -> tuple[IntArray, IntArray]:
    """Match velocity starts separated by an exact original-row offset."""

    if later_start_indices.size == 0 or earlier_start_indices.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    if lag_frames > int(later_start_indices[-1]) - int(earlier_start_indices[0]):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    desired = earlier_start_indices + int(lag_frames)
    later_positions = np.searchsorted(later_start_indices, desired, side="left")
    earlier_positions = np.arange(earlier_start_indices.size, dtype=np.int64)
    in_range = later_positions < later_start_indices.size
    matched = np.zeros(in_range.shape, dtype=bool)
    matched[in_range] = later_start_indices[later_positions[in_range]] == desired[in_range]
    keep = in_range & matched
    return later_positions[keep].astype(np.int64), earlier_positions[keep]


def _validate_min_pairs(min_pairs: int) -> int:
    if isinstance(min_pairs, (bool, np.bool_)) or int(min_pairs) != min_pairs:
        raise ValueError("min_pairs must be a positive integer")
    minimum = int(min_pairs)
    if minimum < 1:
        raise ValueError("min_pairs must be a positive integer")
    return minimum


def _require_implicit_exact_time_support(
    *,
    metric: str,
    implicit_grid: bool,
    counts: IntArray,
    min_pairs: int,
) -> None:
    """Reject a silently empty automatically generated exact-time curve.

    Explicit lags remain strict and may intentionally have zero matches.  The
    fail-closed rule applies only when this module selected the lag grid for the
    caller, because nominal-median lags are commonly absent from jittered clock
    data.  The primary frame-step APIs retain row-offset support while reporting
    the true elapsed-time distribution and never interpolate coordinates.
    """

    if implicit_grid and counts.size and not np.any(counts >= min_pairs):
        raise ExactTimeSupportError(
            f"{metric} implicit exact-time lag grid has no lag with at least "
            f"{min_pairs} valid pair(s). Use the corresponding primary frame-step "
            "API to retain row-offset pairs and report their true elapsed times, "
            "or supply explicit physical lags/time_tolerance for an intentional "
            "exact-time analysis. Coordinates are not interpolated."
        )


def _time_tolerance(
    requested: float | None,
    *time_arrays: FloatArray,
    lag: float = 0.0,
) -> float:
    if requested is not None:
        tolerance = float(requested)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("time_tolerance must be finite and non-negative")
        return tolerance

    populated = [values for values in time_arrays if values.size]
    if not populated:
        return 64.0 * np.finfo(float).eps * max(1.0, abs(float(lag)))

    # Absolute epoch timestamps can be many orders of magnitude larger than a
    # requested lag.  Scaling a relative tolerance by that epoch would turn an
    # exact-lag matcher into a broad physical lag bin (for example, about 0.014
    # at 1e12).  Center the numerical scale on the first observed epoch while
    # retaining one input ULP so decimal intervals stored beside a large epoch
    # remain matchable at the precision the input can actually represent.
    origin = min(float(values[0]) for values in populated)
    scale = max(1.0, abs(float(lag)))
    input_ulp = 0.0
    for values in populated:
        centered = values - origin
        scale = max(scale, float(np.max(np.abs(centered))))
        input_ulp = max(input_ulp, float(np.max(np.abs(np.spacing(values)))))
    centered_roundoff = 64.0 * np.finfo(float).eps * scale
    return max(centered_roundoff, input_ulp)


def _match_time_shift(
    later_times: FloatArray,
    earlier_times: FloatArray,
    lag: float,
    *,
    time_tolerance: float | None,
) -> tuple[IntArray, IntArray]:
    """Match ``later_time ~= earlier_time + lag`` without interpolation."""

    if later_times.size == 0 or earlier_times.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    tolerance = _time_tolerance(
        time_tolerance,
        later_times,
        earlier_times,
        lag=lag,
    )
    # Search and compare after subtracting a shared origin.  This avoids adding
    # a small lag directly to a large epoch and keeps the default tolerance tied
    # to elapsed-time precision rather than epoch magnitude.
    origin = min(float(later_times[0]), float(earlier_times[0]))
    later_centered = later_times - origin
    earlier_centered = earlier_times - origin
    desired = earlier_centered + lag
    insertion = np.searchsorted(later_centered, desired, side="left")
    candidates = np.stack(
        (
            np.clip(insertion - 1, 0, later_times.size - 1),
            np.clip(insertion, 0, later_times.size - 1),
        ),
        axis=0,
    )
    candidate_times = later_centered[candidates]
    errors = np.abs(candidate_times - desired[np.newaxis, :])
    if lag > 0.0:
        errors[candidate_times <= earlier_centered[np.newaxis, :]] = np.inf
    else:
        errors[candidate_times < earlier_centered[np.newaxis, :]] = np.inf

    best_row = np.argmin(errors, axis=0)
    earlier_indices = np.arange(earlier_times.size, dtype=np.int64)
    later_indices = candidates[best_row, earlier_indices]
    best_error = errors[best_row, earlier_indices]
    keep = np.isfinite(best_error) & (best_error <= tolerance)
    return later_indices[keep].astype(np.int64), earlier_indices[keep]


def nominal_time_step(times: ArrayLike) -> float:
    """Return the median positive observation interval."""

    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("at least two finite, one-dimensional times are required")
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError("times must be strictly increasing")
    return float(np.median(differences))


def default_lags(
    times: ArrayLike,
    *,
    max_lag_steps: int | None = None,
    include_zero: bool = False,
) -> FloatArray:
    """Build the prespecified default lag grid in true time units.

    By default the largest lag is ``min(50, floor(0.25 * n_times))`` nominal
    intervals, as specified in the project master plan.
    """

    values = np.asarray(times, dtype=float)
    if values.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if values.size < 2:
        return np.array([0.0], dtype=float) if include_zero else np.empty(0, dtype=float)
    step = nominal_time_step(values)
    if max_lag_steps is None:
        maximum = min(50, int(np.floor(0.25 * values.size)))
    else:
        if isinstance(max_lag_steps, (bool, np.bool_)) or int(max_lag_steps) != max_lag_steps:
            raise ValueError("max_lag_steps must be a non-negative integer")
        maximum = int(max_lag_steps)
        if maximum < 0:
            raise ValueError("max_lag_steps must be a non-negative integer")
    positive = step * np.arange(1, maximum + 1, dtype=float)
    if include_zero:
        return np.concatenate((np.array([0.0]), positive))
    return positive


def default_frame_lag_steps(
    n_observations: int,
    *,
    max_lag_steps: int | None = None,
    include_zero: bool = False,
) -> IntArray:
    """Build the prespecified row-offset grid for frame-step metrics."""

    if (
        isinstance(n_observations, (bool, np.bool_))
        or int(n_observations) != n_observations
        or int(n_observations) < 0
    ):
        raise ValueError("n_observations must be a non-negative integer")
    observation_count = int(n_observations)
    if max_lag_steps is None:
        maximum = min(50, int(np.floor(0.25 * observation_count)))
    else:
        if isinstance(max_lag_steps, (bool, np.bool_)) or int(max_lag_steps) != max_lag_steps:
            raise ValueError("max_lag_steps must be a non-negative integer")
        maximum = int(max_lag_steps)
        if maximum < 0:
            raise ValueError("max_lag_steps must be a non-negative integer")
    start = 0 if include_zero else 1
    return np.arange(start, maximum + 1, dtype=np.int64)


def mean_squared_displacement_by_frame_steps(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    lag_steps: ArrayLike | None = None,
    max_lag_steps: int | None = None,
    min_pairs: int = 1,
) -> FrameStepLagCurve:
    """Calculate squared displacement for observed-row offsets ``i`` to ``i+k``.

    This API is robust to timestamp jitter because matching is by row offset,
    not by an assumed exact elapsed time.  It does *not* regularize time: the
    empirical elapsed-time mean, median, minimum, and maximum are reported for
    every ``k``.  Only finite endpoint coordinates contribute; intermediate
    rows may be missing because no coordinate is interpolated or otherwise
    invented.
    """

    coordinates = _as_coordinates(positions)
    time_values = _as_times(times, coordinates.shape[0])
    steps = (
        default_frame_lag_steps(
            coordinates.shape[0],
            max_lag_steps=max_lag_steps,
        )
        if lag_steps is None
        else _as_lag_steps(lag_steps)
    )
    minimum = _validate_min_pairs(min_pairs)
    values = np.full(steps.shape, np.nan, dtype=float)
    standard_deviations = np.full(steps.shape, np.nan, dtype=float)
    counts = np.zeros(steps.shape, dtype=np.int64)
    lag_time_mean = np.full(steps.shape, np.nan, dtype=float)
    lag_time_median = np.full(steps.shape, np.nan, dtype=float)
    lag_time_min = np.full(steps.shape, np.nan, dtype=float)
    lag_time_max = np.full(steps.shape, np.nan, dtype=float)
    finite_rows = np.all(np.isfinite(coordinates), axis=1)
    n_observations = coordinates.shape[0]

    for index, step in enumerate(steps):
        offset = int(step)
        if offset >= n_observations:
            continue
        earlier = np.arange(n_observations - offset, dtype=np.int64)
        later = earlier + offset
        supported = finite_rows[earlier] & finite_rows[later]
        earlier = earlier[supported]
        later = later[supported]
        counts[index] = earlier.size
        if earlier.size == 0:
            continue

        elapsed = time_values[later] - time_values[earlier]
        lag_time_mean[index] = float(np.mean(elapsed))
        lag_time_median[index] = float(np.median(elapsed))
        lag_time_min[index] = float(np.min(elapsed))
        lag_time_max[index] = float(np.max(elapsed))
        if earlier.size >= minimum:
            displacements = coordinates[later] - coordinates[earlier]
            squared = np.einsum("ij,ij->i", displacements, displacements)
            values[index] = float(np.mean(squared))
            if squared.size >= 2:
                standard_deviations[index] = float(np.std(squared, ddof=1))

    return FrameStepLagCurve(
        lag_steps=steps,
        values=values,
        counts=counts,
        lag_time_mean=lag_time_mean,
        lag_time_median=lag_time_median,
        lag_time_min=lag_time_min,
        lag_time_max=lag_time_max,
        standard_deviations=standard_deviations,
    )


def mean_squared_displacement(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    lags: ArrayLike | None = None,
    max_lag_steps: int | None = None,
    min_pairs: int = 1,
    time_tolerance: float | None = None,
) -> SquaredDisplacementLagCurve:
    """Calculate MSD using observed endpoint pairs at each requested time lag.

    A missing intermediate frame is never filled.  A pair contributes when its
    two endpoints are observed, finite, and separated by the requested true
    time lag.  Thus coarse lags may still be estimable when an intermediate
    coordinate is absent, without inventing that coordinate.
    """

    coordinates = _as_coordinates(positions)
    time_values = _as_times(times, coordinates.shape[0])
    implicit_grid = lags is None
    lag_values = (
        default_lags(time_values, max_lag_steps=max_lag_steps) if lags is None else _as_lags(lags)
    )
    minimum = _validate_min_pairs(min_pairs)
    values = np.full(lag_values.shape, np.nan, dtype=float)
    standard_deviations = np.full(lag_values.shape, np.nan, dtype=float)
    counts = np.zeros(lag_values.shape, dtype=np.int64)
    finite_rows = np.all(np.isfinite(coordinates), axis=1)

    for index, lag in enumerate(lag_values):
        later, earlier = _match_time_shift(
            time_values,
            time_values,
            float(lag),
            time_tolerance=time_tolerance,
        )
        supported = finite_rows[later] & finite_rows[earlier]
        later = later[supported]
        earlier = earlier[supported]
        counts[index] = later.size
        if later.size >= minimum:
            displacements = coordinates[later] - coordinates[earlier]
            squared = np.einsum("ij,ij->i", displacements, displacements)
            values[index] = float(np.mean(squared))
            if squared.size >= 2:
                standard_deviations[index] = float(np.std(squared, ddof=1))

    _require_implicit_exact_time_support(
        metric="MSD",
        implicit_grid=implicit_grid,
        counts=counts,
        min_pairs=minimum,
    )
    return SquaredDisplacementLagCurve(
        lags=lag_values,
        values=values,
        counts=counts,
        standard_deviations=standard_deviations,
    )


def mean_squared_change_in_distance(
    site1: ArrayLike,
    site2: ArrayLike,
    *,
    times: ArrayLike | None = None,
    lags: ArrayLike | None = None,
    max_lag_steps: int | None = None,
    min_pairs: int = 1,
    time_tolerance: float | None = None,
) -> SquaredDisplacementLagCurve:
    """Calculate two-point MSCD from the paired relative-vector trajectory.

    Despite its historical name, this is the mean squared *vector* change
    ``||(site2-site1)(t+lag) - (site2-site1)(t)||^2``; it is not the squared
    change in scalar separation and is not assumed to equal the sum of the two
    site-specific MSD curves.
    """

    relative = relative_mode(site1, site2)
    return mean_squared_displacement(
        relative,
        times=times,
        lags=lags,
        max_lag_steps=max_lag_steps,
        min_pairs=min_pairs,
        time_tolerance=time_tolerance,
    )


def mean_squared_change_in_distance_by_frame_steps(
    site1: ArrayLike,
    site2: ArrayLike,
    *,
    times: ArrayLike | None = None,
    lag_steps: ArrayLike | None = None,
    max_lag_steps: int | None = None,
    min_pairs: int = 1,
) -> FrameStepLagCurve:
    """Calculate relative-vector MSCD by observed-row offsets.

    The elapsed-time summaries refer to the finite paired-site endpoints that
    contribute to each row-offset MSCD value.
    """

    relative = relative_mode(site1, site2)
    return mean_squared_displacement_by_frame_steps(
        relative,
        times=times,
        lag_steps=lag_steps,
        max_lag_steps=max_lag_steps,
        min_pairs=min_pairs,
    )


def velocity_series_by_frame_steps(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    measurement_delta_frames: int = 1,
) -> FrameStepVelocitySeries:
    """Return velocities between rows ``i`` and ``i + measurement_delta_frames``.

    Every vector is divided by its own true elapsed time.  Rows with a
    non-finite coordinate at either endpoint are omitted, while original start
    and end indices are retained so later correlation code cannot compress a
    missing velocity into an apparently adjacent one.
    """

    coordinates = _as_coordinates(positions)
    time_values = _as_times(times, coordinates.shape[0])
    delta_frames = _positive_frame_offset(
        measurement_delta_frames,
        name="measurement_delta_frames",
    )
    n_candidates = max(coordinates.shape[0] - delta_frames, 0)
    start_indices = np.arange(n_candidates, dtype=np.int64)
    end_indices = start_indices + delta_frames
    finite_rows = np.all(np.isfinite(coordinates), axis=1)
    supported = finite_rows[start_indices] & finite_rows[end_indices]
    start_indices = start_indices[supported]
    end_indices = end_indices[supported]
    durations = time_values[end_indices] - time_values[start_indices]
    vectors = (coordinates[end_indices] - coordinates[start_indices]) / durations[:, np.newaxis]
    return FrameStepVelocitySeries(
        start_times=time_values[start_indices],
        end_times=time_values[end_indices],
        durations=durations,
        vectors=vectors,
        start_indices=start_indices,
        end_indices=end_indices,
        measurement_delta_frames=delta_frames,
    )


def velocity_series(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    delta: float = 1.0,
    time_tolerance: float | None = None,
) -> VelocitySeries:
    """Return velocities whose two endpoints are observed ``delta`` apart."""

    coordinates = _as_coordinates(positions)
    time_values = _as_times(times, coordinates.shape[0])
    interval = float(delta)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("delta must be finite and strictly positive")

    later, earlier = _match_time_shift(
        time_values,
        time_values,
        interval,
        time_tolerance=time_tolerance,
    )
    finite_rows = np.all(np.isfinite(coordinates), axis=1)
    supported = finite_rows[later] & finite_rows[earlier]
    later = later[supported]
    earlier = earlier[supported]
    durations = time_values[later] - time_values[earlier]
    vectors = (coordinates[later] - coordinates[earlier]) / durations[:, np.newaxis]
    return VelocitySeries(
        start_times=time_values[earlier],
        end_times=time_values[later],
        durations=durations,
        vectors=vectors,
        start_indices=earlier,
        end_indices=later,
    )


def _correlation_lags(
    supplied: ArrayLike | None,
    times: FloatArray,
    max_lag_steps: int | None,
) -> FloatArray:
    if supplied is not None:
        return _as_lags(supplied)
    if times.size < 2:
        return np.array([0.0], dtype=float)
    return default_lags(times, max_lag_steps=max_lag_steps, include_zero=True)


def _frame_correlation_lags(
    supplied: ArrayLike | None,
    n_observations: int,
    max_lag_frames: int | None,
) -> IntArray:
    if supplied is not None:
        return _as_lag_steps(supplied)
    return default_frame_lag_steps(
        n_observations,
        max_lag_steps=max_lag_frames,
        include_zero=True,
    )


def _empty_start_time_lag_summaries(
    shape: tuple[int, ...],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    return (
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
    )


def _record_start_time_lag_summary(
    elapsed: FloatArray,
    index: int,
    mean: FloatArray,
    median: FloatArray,
    minimum: FloatArray,
    maximum: FloatArray,
) -> None:
    if elapsed.size == 0:
        return
    mean[index] = float(np.mean(elapsed))
    median[index] = float(np.median(elapsed))
    minimum[index] = float(np.min(elapsed))
    maximum[index] = float(np.max(elapsed))


def velocity_autocorrelation_by_frame_steps(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    measurement_delta_frames: int = 1,
    lag_frames: ArrayLike | None = None,
    max_lag_frames: int | None = None,
    min_pairs: int = 1,
) -> FrameStepCorrelationCurve:
    """Calculate VAC from frame-step velocities and start-row lag offsets.

    A pair at ``lag_frames=k`` contains velocities whose original start rows
    differ by exactly ``k``.  The vectors may have unequal measurement
    durations because each is divided by its own observed elapsed time.
    """

    coordinates = _as_coordinates(positions)
    time_values = _as_times(times, coordinates.shape[0])
    series = velocity_series_by_frame_steps(
        coordinates,
        times=time_values,
        measurement_delta_frames=measurement_delta_frames,
    )
    frame_lags = _frame_correlation_lags(
        lag_frames,
        coordinates.shape[0],
        max_lag_frames,
    )
    minimum_pairs = _validate_min_pairs(min_pairs)
    raw = np.full(frame_lags.shape, np.nan, dtype=float)
    counts = np.zeros(frame_lags.shape, dtype=np.int64)
    lag_mean, lag_median, lag_min, lag_max = _empty_start_time_lag_summaries(frame_lags.shape)

    if series.n_velocities:
        squared_speeds = np.einsum("ij,ij->i", series.vectors, series.vectors)
        normalization = float(np.mean(squared_speeds))
    else:
        normalization = float("nan")

    for index, lag in enumerate(frame_lags):
        later, earlier = _match_frame_shift(
            series.start_indices,
            series.start_indices,
            int(lag),
        )
        counts[index] = later.size
        elapsed = series.start_times[later] - series.start_times[earlier]
        _record_start_time_lag_summary(
            elapsed,
            index,
            lag_mean,
            lag_median,
            lag_min,
            lag_max,
        )
        if later.size >= minimum_pairs:
            products = np.einsum(
                "ij,ij->i",
                series.vectors[later],
                series.vectors[earlier],
            )
            raw[index] = float(np.mean(products))

    normalized = np.full(raw.shape, np.nan, dtype=float)
    if np.isfinite(normalization) and normalization > 0.0:
        normalized = raw / normalization
    return FrameStepCorrelationCurve(
        lag_frames=frame_lags,
        values=normalized,
        counts=counts,
        start_time_lag_mean=lag_mean,
        start_time_lag_median=lag_median,
        start_time_lag_min=lag_min,
        start_time_lag_max=lag_max,
        raw_values=raw,
        normalization=normalization,
        normalizations=np.full(raw.shape, normalization, dtype=float),
    )


def velocity_cross_correlation_by_frame_steps(
    site1: ArrayLike,
    site2: ArrayLike,
    *,
    times: ArrayLike | None = None,
    measurement_delta_frames: int = 1,
    lag_frames: ArrayLike | None = None,
    max_lag_frames: int | None = None,
    min_pairs: int = 1,
) -> FrameStepCorrelationCurve:
    """Calculate normalized paired-site VCC using original frame offsets.

    At ``lag_frames=k`` the raw statistic is the mean dot product between a
    site1 velocity starting at row ``i+k`` and a site2 velocity starting at row
    ``i``.  In accordance with the master plan, every primary ``values`` entry
    is divided by the same zero-lag denominator
    ``sqrt(C_site1,site1(0) * C_site2,site2(0))``.  Each auto-energy uses all
    valid velocities for that site; it is not restricted to complete-case
    cross-site frames.  Primary values are deliberately not clipped: differing
    missingness support can make them exceed one.  A bounded lag-specific
    sensitivity is retained only in the explicitly secondary fields.
    """

    first_coordinates = _as_coordinates(site1, name="site1")
    second_coordinates = _as_coordinates(site2, name="site2")
    if first_coordinates.shape != second_coordinates.shape:
        raise ValueError("site1 and site2 must have identical coordinate shapes")
    time_values = _as_times(times, first_coordinates.shape[0])
    first = velocity_series_by_frame_steps(
        first_coordinates,
        times=time_values,
        measurement_delta_frames=measurement_delta_frames,
    )
    second = velocity_series_by_frame_steps(
        second_coordinates,
        times=time_values,
        measurement_delta_frames=measurement_delta_frames,
    )
    frame_lags = _frame_correlation_lags(
        lag_frames,
        first_coordinates.shape[0],
        max_lag_frames,
    )
    minimum_pairs = _validate_min_pairs(min_pairs)
    raw = np.full(frame_lags.shape, np.nan, dtype=float)
    lag_specific_normalizations = np.full(frame_lags.shape, np.nan, dtype=float)
    counts = np.zeros(frame_lags.shape, dtype=np.int64)
    lag_mean, lag_median, lag_min, lag_max = _empty_start_time_lag_summaries(frame_lags.shape)

    for index, lag in enumerate(frame_lags):
        later, earlier = _match_frame_shift(
            first.start_indices,
            second.start_indices,
            int(lag),
        )
        counts[index] = later.size
        elapsed = first.start_times[later] - second.start_times[earlier]
        _record_start_time_lag_summary(
            elapsed,
            index,
            lag_mean,
            lag_median,
            lag_min,
            lag_max,
        )
        if later.size >= minimum_pairs:
            first_vectors = first.vectors[later]
            second_vectors = second.vectors[earlier]
            products = np.einsum(
                "ij,ij->i",
                first_vectors,
                second_vectors,
            )
            raw[index] = float(np.mean(products))
            first_energy = float(np.mean(np.einsum("ij,ij->i", first_vectors, first_vectors)))
            second_energy = float(np.mean(np.einsum("ij,ij->i", second_vectors, second_vectors)))
            lag_specific_normalizations[index] = float(np.sqrt(first_energy * second_energy))

    if first.n_velocities >= minimum_pairs and second.n_velocities >= minimum_pairs:
        first_energy = float(np.mean(np.einsum("ij,ij->i", first.vectors, first.vectors)))
        second_energy = float(np.mean(np.einsum("ij,ij->i", second.vectors, second.vectors)))
        normalization = float(np.sqrt(first_energy * second_energy))
    else:
        normalization = float("nan")

    normalized = np.full(raw.shape, np.nan, dtype=float)
    applied_normalizations = np.full(raw.shape, normalization, dtype=float)
    supported = np.isfinite(raw) & np.isfinite(normalization) & (normalization > 0.0)
    normalized[supported] = raw[supported] / normalization

    secondary = np.full(raw.shape, np.nan, dtype=float)
    secondary_supported = (
        np.isfinite(raw)
        & np.isfinite(lag_specific_normalizations)
        & (lag_specific_normalizations > 0.0)
    )
    secondary[secondary_supported] = np.clip(
        raw[secondary_supported] / lag_specific_normalizations[secondary_supported],
        -1.0,
        1.0,
    )
    return FrameStepCorrelationCurve(
        lag_frames=frame_lags,
        values=normalized,
        counts=counts,
        start_time_lag_mean=lag_mean,
        start_time_lag_median=lag_median,
        start_time_lag_min=lag_min,
        start_time_lag_max=lag_max,
        raw_values=raw,
        normalization=normalization,
        normalizations=applied_normalizations,
        secondary_lag_specific_values=secondary,
        secondary_lag_specific_normalizations=lag_specific_normalizations,
    )


def velocity_cross_correlation_matrix_by_frame_steps(
    site1: ArrayLike,
    site2: ArrayLike,
    *,
    times: ArrayLike | None = None,
    measurement_delta_frames: int = 1,
    lag_frames: ArrayLike | None = None,
    max_lag_frames: int | None = None,
    min_pairs: int = 1,
) -> FrameStepCrossCorrelationMatrixCurve:
    """Calculate the full lag-indexed Site1/Site2 VCC matrix curve.

    The returned tensor is
    ``C_LR(tau) = <v_L(t+tau) v_R(t)^T>``.  It retains all four component
    cross-products at every requested start-time lag.  Normalization uses one
    fixed zero-lag rotation-invariant energy scale for every component and lag.
    Use :attr:`FrameStepCrossCorrelationMatrixCurve.trace_values` only when a
    scalar, rotation-invariant compatibility summary is explicitly required.
    """

    first_coordinates = _as_coordinates(site1, name="site1")
    second_coordinates = _as_coordinates(site2, name="site2")
    if first_coordinates.shape != second_coordinates.shape:
        raise ValueError("site1 and site2 must have identical coordinate shapes")
    if first_coordinates.shape[1] != 2:
        raise ValueError("VCC matrix output requires two-dimensional x/y coordinates")
    time_values = _as_times(times, first_coordinates.shape[0])
    first = velocity_series_by_frame_steps(
        first_coordinates,
        times=time_values,
        measurement_delta_frames=measurement_delta_frames,
    )
    second = velocity_series_by_frame_steps(
        second_coordinates,
        times=time_values,
        measurement_delta_frames=measurement_delta_frames,
    )
    frame_lags = _frame_correlation_lags(
        lag_frames,
        first_coordinates.shape[0],
        max_lag_frames,
    )
    minimum_pairs = _validate_min_pairs(min_pairs)
    dimensions = first_coordinates.shape[1]
    raw_matrices = np.full((frame_lags.size, dimensions, dimensions), np.nan, dtype=float)
    lag_specific_normalizations = np.full(frame_lags.shape, np.nan, dtype=float)
    counts = np.zeros(frame_lags.shape, dtype=np.int64)
    lag_mean, lag_median, lag_min, lag_max = _empty_start_time_lag_summaries(frame_lags.shape)

    for index, lag in enumerate(frame_lags):
        later, earlier = _match_frame_shift(
            first.start_indices,
            second.start_indices,
            int(lag),
        )
        counts[index] = later.size
        elapsed = first.start_times[later] - second.start_times[earlier]
        _record_start_time_lag_summary(
            elapsed,
            index,
            lag_mean,
            lag_median,
            lag_min,
            lag_max,
        )
        if later.size >= minimum_pairs:
            first_vectors = first.vectors[later]
            second_vectors = second.vectors[earlier]
            raw_matrices[index] = np.einsum("ni,nj->ij", first_vectors, second_vectors) / later.size
            first_energy = float(np.mean(np.einsum("ij,ij->i", first_vectors, first_vectors)))
            second_energy = float(np.mean(np.einsum("ij,ij->i", second_vectors, second_vectors)))
            lag_specific_normalizations[index] = float(np.sqrt(first_energy * second_energy))

    if first.n_velocities >= minimum_pairs and second.n_velocities >= minimum_pairs:
        first_energy = float(np.mean(np.einsum("ij,ij->i", first.vectors, first.vectors)))
        second_energy = float(np.mean(np.einsum("ij,ij->i", second.vectors, second.vectors)))
        normalization = float(np.sqrt(first_energy * second_energy))
    else:
        normalization = float("nan")
    matrices = np.full(raw_matrices.shape, np.nan, dtype=float)
    if np.isfinite(normalization) and normalization > 0.0:
        matrices = raw_matrices / normalization
    return FrameStepCrossCorrelationMatrixCurve(
        lag_frames=frame_lags,
        matrices=matrices,
        counts=counts,
        start_time_lag_mean=lag_mean,
        start_time_lag_median=lag_median,
        start_time_lag_min=lag_min,
        start_time_lag_max=lag_max,
        raw_matrices=raw_matrices,
        normalization=normalization,
        lag_specific_normalizations=lag_specific_normalizations,
        measurement_delta_frames=int(measurement_delta_frames),
    )


def velocity_autocorrelation(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    delta: float = 1.0,
    lags: ArrayLike | None = None,
    max_lag_steps: int | None = None,
    min_pairs: int = 1,
    time_tolerance: float | None = None,
) -> CorrelationCurve:
    """Calculate raw and zero-lag-normalized velocity autocorrelation."""

    series = velocity_series(
        positions,
        times=times,
        delta=delta,
        time_tolerance=time_tolerance,
    )
    implicit_grid = lags is None
    lag_values = _correlation_lags(lags, series.start_times, max_lag_steps)
    minimum = _validate_min_pairs(min_pairs)
    raw = np.full(lag_values.shape, np.nan, dtype=float)
    counts = np.zeros(lag_values.shape, dtype=np.int64)

    if series.n_velocities:
        squared_speeds = np.einsum("ij,ij->i", series.vectors, series.vectors)
        normalization = float(np.mean(squared_speeds))
    else:
        normalization = float("nan")

    for index, lag in enumerate(lag_values):
        later, earlier = _match_time_shift(
            series.start_times,
            series.start_times,
            float(lag),
            time_tolerance=time_tolerance,
        )
        counts[index] = later.size
        if later.size >= minimum:
            products = np.einsum(
                "ij,ij->i",
                series.vectors[later],
                series.vectors[earlier],
            )
            raw[index] = float(np.mean(products))

    normalized = np.full(raw.shape, np.nan, dtype=float)
    if np.isfinite(normalization) and normalization > 0.0:
        normalized = raw / normalization
    _require_implicit_exact_time_support(
        metric="VAC",
        implicit_grid=implicit_grid,
        counts=counts,
        min_pairs=minimum,
    )
    return CorrelationCurve(
        lags=lag_values,
        values=normalized,
        counts=counts,
        raw_values=raw,
        normalization=normalization,
        normalizations=np.full(raw.shape, normalization, dtype=float),
    )


def velocity_cross_correlation(
    site1: ArrayLike,
    site2: ArrayLike,
    *,
    times: ArrayLike | None = None,
    delta: float = 1.0,
    lags: ArrayLike | None = None,
    max_lag_steps: int | None = None,
    min_pairs: int = 1,
    time_tolerance: float | None = None,
) -> CorrelationCurve:
    """Calculate normalized paired-site velocity cross-correlation.

    At lag ``tau`` the raw statistic is
    ``mean(v_site1(t + tau) dot v_site2(t))``.  Every primary ``values`` entry
    uses the master-plan fixed zero-lag denominator
    ``sqrt(C_site1,site1(0) * C_site2,site2(0))``.  Each zero-lag auto-energy
    uses that site's full valid velocity support rather than complete-case
    cross-site frames.  It is not clipped.  The explicitly secondary fields
    retain a bounded lag-specific-support sensitivity analysis.
    """

    first_coordinates = _as_coordinates(site1, name="site1")
    second_coordinates = _as_coordinates(site2, name="site2")
    if first_coordinates.shape != second_coordinates.shape:
        raise ValueError("site1 and site2 must have identical coordinate shapes")
    time_values = _as_times(times, first_coordinates.shape[0])
    first = velocity_series(
        first_coordinates,
        times=time_values,
        delta=delta,
        time_tolerance=time_tolerance,
    )
    second = velocity_series(
        second_coordinates,
        times=time_values,
        delta=delta,
        time_tolerance=time_tolerance,
    )
    implicit_grid = lags is None
    lag_values = _correlation_lags(lags, time_values, max_lag_steps)
    minimum = _validate_min_pairs(min_pairs)

    raw = np.full(lag_values.shape, np.nan, dtype=float)
    lag_specific_normalizations = np.full(lag_values.shape, np.nan, dtype=float)
    counts = np.zeros(lag_values.shape, dtype=np.int64)
    for index, lag in enumerate(lag_values):
        later, earlier = _match_time_shift(
            first.start_times,
            second.start_times,
            float(lag),
            time_tolerance=time_tolerance,
        )
        counts[index] = later.size
        if later.size >= minimum:
            first_vectors = first.vectors[later]
            second_vectors = second.vectors[earlier]
            products = np.einsum(
                "ij,ij->i",
                first_vectors,
                second_vectors,
            )
            raw[index] = float(np.mean(products))
            first_energy = float(np.mean(np.einsum("ij,ij->i", first_vectors, first_vectors)))
            second_energy = float(np.mean(np.einsum("ij,ij->i", second_vectors, second_vectors)))
            lag_specific_normalizations[index] = float(np.sqrt(first_energy * second_energy))

    if first.n_velocities >= minimum and second.n_velocities >= minimum:
        first_energy = float(np.mean(np.einsum("ij,ij->i", first.vectors, first.vectors)))
        second_energy = float(np.mean(np.einsum("ij,ij->i", second.vectors, second.vectors)))
        normalization = float(np.sqrt(first_energy * second_energy))
    else:
        normalization = float("nan")

    normalized = np.full(raw.shape, np.nan, dtype=float)
    applied_normalizations = np.full(raw.shape, normalization, dtype=float)
    supported = np.isfinite(raw) & np.isfinite(normalization) & (normalization > 0.0)
    normalized[supported] = raw[supported] / normalization

    secondary = np.full(raw.shape, np.nan, dtype=float)
    secondary_supported = (
        np.isfinite(raw)
        & np.isfinite(lag_specific_normalizations)
        & (lag_specific_normalizations > 0.0)
    )
    secondary[secondary_supported] = np.clip(
        raw[secondary_supported] / lag_specific_normalizations[secondary_supported],
        -1.0,
        1.0,
    )
    _require_implicit_exact_time_support(
        metric="VCC",
        implicit_grid=implicit_grid,
        counts=counts,
        min_pairs=minimum,
    )
    return CorrelationCurve(
        lags=lag_values,
        values=normalized,
        counts=counts,
        raw_values=raw,
        normalization=normalization,
        normalizations=applied_normalizations,
        secondary_lag_specific_values=secondary,
        secondary_lag_specific_normalizations=lag_specific_normalizations,
    )


def _curve_time_axis(
    curve: LagCurve | FrameStepLagCurve | FrameStepCorrelationCurve,
) -> FloatArray:
    if isinstance(curve, FrameStepCorrelationCurve):
        return np.asarray(curve.start_time_lag_median, dtype=float)
    if isinstance(curve, FrameStepLagCurve):
        return np.asarray(curve.lag_time_median, dtype=float)
    if isinstance(curve, LagCurve):
        return np.asarray(curve.lags, dtype=float)
    raise TypeError("curve must be a supported lag-curve result")


def _invalid_curve_integral(
    *,
    reason: str,
    supported_lag_count: int,
    supported_interval_count: int,
    supported_duration: float,
    minimum_duration: float,
    min_pairs_per_lag: int,
    positive_only: bool,
    maximum_lag_time: float | None,
) -> CurveIntegral:
    return CurveIntegral(
        area=float("nan"),
        valid=False,
        reason=reason,
        supported_lag_count=supported_lag_count,
        supported_interval_count=supported_interval_count,
        supported_duration=supported_duration,
        minimum_duration=minimum_duration,
        min_pairs_per_lag=min_pairs_per_lag,
        positive_only=positive_only,
        maximum_lag_time=maximum_lag_time,
    )


def _positive_linear_interval_area(first: float, second: float, width: float) -> float:
    """Integrate the positive part of a linear segment exactly."""

    if first >= 0.0 and second >= 0.0:
        return 0.5 * (first + second) * width
    if first <= 0.0 and second <= 0.0:
        return 0.0
    if first > 0.0:
        positive_width = width * first / (first - second)
        return 0.5 * first * positive_width
    positive_width = width * second / (second - first)
    return 0.5 * second * positive_width


def integrate_supported_curve(
    curve: LagCurve | FrameStepLagCurve | FrameStepCorrelationCurve,
    *,
    min_pairs_per_lag: int,
    minimum_duration: float,
    maximum_lag_time: float | None = None,
    positive_only: bool = False,
) -> CurveIntegral:
    """Integrate only adjacent, count-supported curve intervals.

    Frame-step results use their empirical median elapsed time, never a nominal
    frame duration.  Both endpoints of an interval must meet the pair-count
    threshold, so an unsupported internal lag is not bridged.  ``minimum_duration``
    is checked against the sum of intervals actually integrated.  This makes an
    under-supported curve unavailable instead of silently emitting an AUC.

    Set ``positive_only=True`` for the prespecified positive short-lag VCC area;
    zero crossings are handled by exact linear interpolation within each
    supported interval.
    """

    minimum_pairs = _validate_min_pairs(min_pairs_per_lag)
    required_duration = float(minimum_duration)
    if not np.isfinite(required_duration) or required_duration < 0.0:
        raise ValueError("minimum_duration must be finite and non-negative")
    if not isinstance(positive_only, (bool, np.bool_)):
        raise TypeError("positive_only must be boolean")

    cutoff: float | None
    if maximum_lag_time is None:
        cutoff = None
    else:
        cutoff = float(maximum_lag_time)
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("maximum_lag_time must be finite and non-negative")

    times = _curve_time_axis(curve)
    values = np.asarray(curve.values, dtype=float)
    counts = np.asarray(curve.counts, dtype=np.int64)
    if times.shape != values.shape or counts.shape != values.shape:
        raise ValueError("curve time, value, and count arrays must align")

    supported = np.isfinite(times) & np.isfinite(values) & (counts >= minimum_pairs)
    if cutoff is not None:
        supported &= times <= cutoff
    supported_lags = int(np.count_nonzero(supported))
    if supported_lags < 2:
        return _invalid_curve_integral(
            reason="fewer_than_two_supported_lags",
            supported_lag_count=supported_lags,
            supported_interval_count=0,
            supported_duration=0.0,
            minimum_duration=required_duration,
            min_pairs_per_lag=minimum_pairs,
            positive_only=bool(positive_only),
            maximum_lag_time=cutoff,
        )

    area = 0.0
    duration = 0.0
    interval_count = 0
    for index in range(values.size - 1):
        if not supported[index] or not supported[index + 1]:
            continue
        width = float(times[index + 1] - times[index])
        if not np.isfinite(width) or width <= 0.0:
            return _invalid_curve_integral(
                reason="non_increasing_supported_lag_times",
                supported_lag_count=supported_lags,
                supported_interval_count=interval_count,
                supported_duration=duration,
                minimum_duration=required_duration,
                min_pairs_per_lag=minimum_pairs,
                positive_only=bool(positive_only),
                maximum_lag_time=cutoff,
            )
        first = float(values[index])
        second = float(values[index + 1])
        if positive_only:
            area += _positive_linear_interval_area(first, second, width)
        else:
            area += 0.5 * (first + second) * width
        duration += width
        interval_count += 1

    if interval_count == 0:
        return _invalid_curve_integral(
            reason="no_adjacent_supported_lags",
            supported_lag_count=supported_lags,
            supported_interval_count=0,
            supported_duration=0.0,
            minimum_duration=required_duration,
            min_pairs_per_lag=minimum_pairs,
            positive_only=bool(positive_only),
            maximum_lag_time=cutoff,
        )
    if duration < required_duration:
        return _invalid_curve_integral(
            reason="insufficient_supported_duration",
            supported_lag_count=supported_lags,
            supported_interval_count=interval_count,
            supported_duration=duration,
            minimum_duration=required_duration,
            min_pairs_per_lag=minimum_pairs,
            positive_only=bool(positive_only),
            maximum_lag_time=cutoff,
        )
    return CurveIntegral(
        area=float(area),
        valid=True,
        reason=None,
        supported_lag_count=supported_lags,
        supported_interval_count=interval_count,
        supported_duration=float(duration),
        minimum_duration=required_duration,
        min_pairs_per_lag=minimum_pairs,
        positive_only=bool(positive_only),
        maximum_lag_time=cutoff,
    )


def positive_short_lag_integral(
    curve: LagCurve | FrameStepLagCurve | FrameStepCorrelationCurve,
    *,
    maximum_lag_time: float,
    min_pairs_per_lag: int,
    minimum_duration: float,
) -> CurveIntegral:
    """Return the guarded positive-area endpoint used for short-lag VCC."""

    return integrate_supported_curve(
        curve,
        min_pairs_per_lag=min_pairs_per_lag,
        minimum_duration=minimum_duration,
        maximum_lag_time=maximum_lag_time,
        positive_only=True,
    )


def _angles_between(first: FloatArray, second: FloatArray) -> FloatArray:
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    supported = (
        np.all(np.isfinite(first), axis=1)
        & np.all(np.isfinite(second), axis=1)
        & (first_norm > 0.0)
        & (second_norm > 0.0)
    )
    if not np.any(supported):
        return np.empty(0, dtype=float)
    cosines = np.einsum("ij,ij->i", first[supported], second[supported])
    cosines /= first_norm[supported] * second_norm[supported]
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def _successive_velocity_pairs(
    series: VelocitySeries,
) -> tuple[IntArray, IntArray]:
    """Pair velocities only when they share the same observed vertex row."""

    if series.n_velocities == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    later = np.searchsorted(series.start_indices, series.end_indices, side="left")
    earlier = np.arange(series.n_velocities, dtype=np.int64)
    in_range = later < series.n_velocities
    matched = np.zeros(in_range.shape, dtype=bool)
    matched[in_range] = series.start_indices[later[in_range]] == series.end_indices[in_range]
    keep = in_range & matched
    return later[keep].astype(np.int64), earlier[keep]


def _supported_turning_pairs(
    series: VelocitySeries,
) -> tuple[IntArray, IntArray, FloatArray]:
    """Return shared-vertex velocity pairs with finite nonzero directions."""

    later, earlier = _successive_velocity_pairs(series)
    if later.size == 0:
        empty_indices = np.empty(0, dtype=np.int64)
        return empty_indices, empty_indices.copy(), np.empty(0, dtype=float)
    first_vectors = series.vectors[earlier]
    second_vectors = series.vectors[later]
    first_norm = np.linalg.norm(first_vectors, axis=1)
    second_norm = np.linalg.norm(second_vectors, axis=1)
    supported = (
        np.all(np.isfinite(first_vectors), axis=1)
        & np.all(np.isfinite(second_vectors), axis=1)
        & (first_norm > 0.0)
        & (second_norm > 0.0)
    )
    later = later[supported]
    earlier = earlier[supported]
    if later.size == 0:
        return later, earlier, np.empty(0, dtype=float)
    products = np.einsum(
        "ij,ij->i",
        series.vectors[earlier],
        series.vectors[later],
    )
    cosines = products / (first_norm[supported] * second_norm[supported])
    return later, earlier, np.arccos(np.clip(cosines, -1.0, 1.0))


def _resolved_exact_turning_series(
    positions: ArrayLike,
    *,
    times: ArrayLike | None,
    delta: float | None,
    time_tolerance: float | None,
) -> tuple[VelocitySeries, bool, int]:
    coordinates = _as_coordinates(positions)
    time_values = _as_times(times, coordinates.shape[0])
    implicit_delta = delta is None
    if delta is None:
        interval = nominal_time_step(time_values) if time_values.size >= 2 else 1.0
    else:
        interval = float(delta)
    series = velocity_series(
        coordinates,
        times=time_values,
        delta=interval,
        time_tolerance=time_tolerance,
    )
    return series, implicit_delta, coordinates.shape[0]


def _require_implicit_exact_turn_support(
    *,
    metric: str,
    implicit_delta: bool,
    n_observations: int,
    n_angles: int,
) -> None:
    if implicit_delta and n_observations >= 3 and n_angles == 0:
        raise ExactTimeSupportError(
            f"{metric} implicit exact-time measurement interval has no supported "
            "shared-vertex turn. Use the corresponding primary frame-step API "
            "to preserve original row offsets and report true step durations, or "
            "supply an explicit physical delta/time_tolerance. Coordinates are "
            "not interpolated."
        )


def _frame_step_turns_from_series(
    series: FrameStepVelocitySeries,
) -> FrameStepTurningAngles:
    later, earlier, angles = _supported_turning_pairs(series)
    first_starts = series.start_indices[earlier]
    vertices = series.end_indices[earlier]
    second_ends = series.end_indices[later]
    return FrameStepTurningAngles(
        measurement_delta_frames=series.measurement_delta_frames,
        angles=angles,
        first_step_start_indices=first_starts,
        vertex_indices=vertices,
        second_step_end_indices=second_ends,
        vertex_times=series.end_times[earlier],
        first_step_durations=series.durations[earlier],
        second_step_durations=series.durations[later],
        full_turn_durations=series.end_times[later] - series.start_times[earlier],
    )


def turning_angles_by_frame_steps(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    measurement_delta_frames: int = 1,
) -> FrameStepTurningAngles:
    """Return primary DCD angles indexed by original row offsets.

    A turn uses endpoints ``i``, ``i+k``, and ``i+2k`` only when all three are
    observed and the two steps share the exact middle row.  Missing rows are
    never compressed into adjacency and coordinates are never interpolated.
    Each contributing step retains its actual elapsed duration, making clock
    jitter and cadence heterogeneity explicit.
    """

    series = velocity_series_by_frame_steps(
        positions,
        times=times,
        measurement_delta_frames=measurement_delta_frames,
    )
    return _frame_step_turns_from_series(series)


def turning_angles(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    delta: float | None = None,
    time_tolerance: float | None = None,
    return_times: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    """Return angles between successive supported coarse-grained steps.

    Zero radians denotes continued motion in the same direction and pi denotes
    reversal.  An angle requires two velocities whose original trajectory rows
    share the same endpoint/startpoint vertex.  This index-level condition also
    applies when ``time_tolerance`` is explicit, so nearby but distinct samples
    cannot create an artificial turn.
    """

    series, implicit_delta, n_observations = _resolved_exact_turning_series(
        positions,
        times=times,
        delta=delta,
        time_tolerance=time_tolerance,
    )
    later, _, angles = _supported_turning_pairs(series)
    vertex_times = series.start_times[later]
    _require_implicit_exact_turn_support(
        metric="turning-angle/DCD",
        implicit_delta=implicit_delta,
        n_observations=n_observations,
        n_angles=angles.size,
    )
    if return_times:
        return vertex_times, angles
    return angles


def directional_change_distribution(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    delta: float | None = None,
    bins: int | ArrayLike = 18,
    time_tolerance: float | None = None,
) -> AngularDistribution:
    """Calculate a normalized DCD histogram from turning angles."""

    angles = turning_angles(
        positions,
        times=times,
        delta=delta,
        time_tolerance=time_tolerance,
    )
    return _angular_distribution(angles, bins)


def _angular_distribution(
    angles: FloatArray,
    bins: int | ArrayLike,
) -> AngularDistribution:
    if np.isscalar(bins):
        n_bins = int(bins)
        if isinstance(bins, (bool, np.bool_)) or n_bins != bins or n_bins < 1:
            raise ValueError("bins must be a positive integer or valid bin edges")
        edges = np.linspace(0.0, np.pi, n_bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
            or edges[0] > 0.0
            or edges[-1] < np.pi
        ):
            raise ValueError("bin edges must increase and span the interval [0, pi]")
    counts, edges = np.histogram(angles, bins=edges)
    probabilities = np.full(counts.shape, np.nan, dtype=float)
    if angles.size:
        probabilities = counts.astype(float) / float(angles.size)
    return AngularDistribution(
        bin_edges=edges.astype(float),
        counts=counts.astype(np.int64),
        probabilities=probabilities,
        n_angles=int(angles.size),
    )


def directional_change_distribution_by_frame_steps(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    measurement_delta_frames: int = 1,
    bins: int | ArrayLike = 18,
) -> FrameStepAngularDistribution:
    """Calculate the primary DCD on row-offset turns with true cadence metadata."""

    turns = turning_angles_by_frame_steps(
        positions,
        times=times,
        measurement_delta_frames=measurement_delta_frames,
    )
    distribution = _angular_distribution(turns.angles, bins)
    return FrameStepAngularDistribution(
        bin_edges=distribution.bin_edges,
        counts=distribution.counts,
        probabilities=distribution.probabilities,
        n_angles=distribution.n_angles,
        turns=turns,
    )


def _angle_probabilities(angles: FloatArray, threshold: float) -> tuple[float, float]:
    if angles.size == 0:
        return float("nan"), float("nan")
    near_zero = float(np.mean(angles <= threshold))
    near_pi = float(np.mean(angles >= (np.pi - threshold)))
    return near_zero, near_pi


def _validated_persistence_inputs(
    threshold_degrees: float,
    n_shuffles: int,
) -> tuple[float, int]:
    threshold = np.deg2rad(float(threshold_degrees))
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold > np.pi / 2.0:
        raise ValueError("threshold_degrees must be in the interval (0, 90]")
    if isinstance(n_shuffles, (bool, np.bool_)) or int(n_shuffles) != n_shuffles:
        raise ValueError("n_shuffles must be a non-negative integer")
    shuffle_count = int(n_shuffles)
    if shuffle_count < 0:
        raise ValueError("n_shuffles must be a non-negative integer")
    return float(threshold), shuffle_count


def _turning_persistence_fields(
    series: VelocitySeries,
    *,
    later: IntArray,
    earlier: IntArray,
    observed_angles: FloatArray,
    threshold: float,
    shuffle_count: int,
    random_state: int | np.random.Generator | None,
) -> dict[str, float | int]:
    near_zero, near_pi = _angle_probabilities(observed_angles, threshold)
    persistence = near_zero - near_pi

    null_zero = float("nan")
    null_pi = float("nan")
    if shuffle_count > 0 and series.n_velocities > 1 and later.size:
        generator = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        zero_probabilities = np.full(shuffle_count, np.nan, dtype=float)
        pi_probabilities = np.full(shuffle_count, np.nan, dtype=float)
        for index in range(shuffle_count):
            shuffled = series.vectors[generator.permutation(series.n_velocities)]
            shuffled_angles = _angles_between(shuffled[earlier], shuffled[later])
            zero_probabilities[index], pi_probabilities[index] = _angle_probabilities(
                shuffled_angles,
                threshold,
            )
        if np.any(np.isfinite(zero_probabilities)):
            null_zero = float(np.nanmean(zero_probabilities))
        if np.any(np.isfinite(pi_probabilities)):
            null_pi = float(np.nanmean(pi_probabilities))

    return {
        "n_angles": int(observed_angles.size),
        "threshold_radians": float(threshold),
        "near_zero_probability": near_zero,
        "near_pi_probability": near_pi,
        "persistence_score": persistence,
        "null_near_zero_probability": null_zero,
        "null_near_pi_probability": null_pi,
        "near_zero_excess": near_zero - null_zero,
        "near_pi_excess": near_pi - null_pi,
        "null_corrected_score": persistence - (null_zero - null_pi),
        "n_shuffles": shuffle_count,
    }


def turning_angle_persistence(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    delta: float | None = None,
    threshold_degrees: float = 30.0,
    n_shuffles: int = 0,
    random_state: int | np.random.Generator | None = 0,
    time_tolerance: float | None = None,
) -> TurningPersistence:
    """Summarize exact-time persistence and optional shuffled-null excess.

    ``persistence_score`` is ``P(angle near 0) - P(angle near pi)``.  When
    ``n_shuffles`` is positive, velocity vectors are permuted across their
    observed within-trajectory start times; the original shared-vertex support
    is retained. An implicit measurement interval fails closed if it produces
    no turns on a trajectory long enough to contain one.
    """

    threshold, shuffle_count = _validated_persistence_inputs(
        threshold_degrees,
        n_shuffles,
    )

    series, implicit_delta, n_observations = _resolved_exact_turning_series(
        positions,
        times=times,
        delta=delta,
        time_tolerance=time_tolerance,
    )
    later, earlier, observed_angles = _supported_turning_pairs(series)
    _require_implicit_exact_turn_support(
        metric="turning-angle persistence",
        implicit_delta=implicit_delta,
        n_observations=n_observations,
        n_angles=observed_angles.size,
    )
    fields = _turning_persistence_fields(
        series,
        later=later,
        earlier=earlier,
        observed_angles=observed_angles,
        threshold=threshold,
        shuffle_count=shuffle_count,
        random_state=random_state,
    )
    return TurningPersistence(**fields)


def turning_angle_persistence_by_frame_steps(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    measurement_delta_frames: int = 1,
    threshold_degrees: float = 30.0,
    n_shuffles: int = 0,
    random_state: int | np.random.Generator | None = 0,
) -> FrameStepTurningPersistence:
    """Summarize primary frame-step DCD persistence with cadence provenance."""

    threshold, shuffle_count = _validated_persistence_inputs(
        threshold_degrees,
        n_shuffles,
    )
    series = velocity_series_by_frame_steps(
        positions,
        times=times,
        measurement_delta_frames=measurement_delta_frames,
    )
    later, earlier, observed_angles = _supported_turning_pairs(series)
    turns = _frame_step_turns_from_series(series)
    fields = _turning_persistence_fields(
        series,
        later=later,
        earlier=earlier,
        observed_angles=observed_angles,
        threshold=threshold,
        shuffle_count=shuffle_count,
        random_state=random_state,
    )
    return FrameStepTurningPersistence(**fields, turns=turns)


def directed_persistence_score(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    delta: float | None = None,
    threshold_degrees: float = 30.0,
    time_tolerance: float | None = None,
) -> float:
    """Return the uncorrected near-zero minus near-pi DCD score."""

    return turning_angle_persistence(
        positions,
        times=times,
        delta=delta,
        threshold_degrees=threshold_degrees,
        time_tolerance=time_tolerance,
    ).persistence_score


def directed_persistence_score_by_frame_steps(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    measurement_delta_frames: int = 1,
    threshold_degrees: float = 30.0,
) -> float:
    """Return the primary row-offset DCD persistence score."""

    return turning_angle_persistence_by_frame_steps(
        positions,
        times=times,
        measurement_delta_frames=measurement_delta_frames,
        threshold_degrees=threshold_degrees,
    ).persistence_score


def directed_burst_candidates_by_frame_steps(
    positions: ArrayLike,
    *,
    times: ArrayLike | None = None,
    measurement_delta_frames: int = 1,
    threshold_degrees: float,
    min_consecutive_turns: int,
    minimum_duration: float,
) -> tuple[DirectedBurstCandidate, ...]:
    """Screen deterministic near-zero-turn runs as directed-burst candidates.

    This implements the master-plan B8 candidate-segment output without
    inventing a state model.  All scientific thresholds are mandatory caller
    inputs.  A run continues only when consecutive turns share their original
    row-offset step, so a missing endpoint, zero-length step, or coverage gap
    splits candidates.  Returned labels remain descriptive candidates.
    """

    threshold = np.deg2rad(float(threshold_degrees))
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold > np.pi / 2.0:
        raise ValueError("threshold_degrees must be in the interval (0, 90]")
    if (
        isinstance(min_consecutive_turns, (bool, np.bool_))
        or int(min_consecutive_turns) != min_consecutive_turns
        or int(min_consecutive_turns) < 1
    ):
        raise ValueError("min_consecutive_turns must be a positive integer")
    minimum_turns = int(min_consecutive_turns)
    required_duration = float(minimum_duration)
    if not np.isfinite(required_duration) or required_duration < 0.0:
        raise ValueError("minimum_duration must be finite and non-negative")

    turns = turning_angles_by_frame_steps(
        positions,
        times=times,
        measurement_delta_frames=measurement_delta_frames,
    )
    qualifying = np.flatnonzero(turns.angles <= threshold)
    if qualifying.size == 0:
        return ()

    boundaries: list[int] = [0]
    for offset in range(1, qualifying.size):
        previous = int(qualifying[offset - 1])
        current = int(qualifying[offset])
        continuous = (
            current == previous + 1
            and turns.first_step_start_indices[current] == turns.vertex_indices[previous]
        )
        if not continuous:
            boundaries.append(offset)
    boundaries.append(qualifying.size)

    candidates: list[DirectedBurstCandidate] = []
    for start_offset, stop_offset in pairwise(boundaries):
        run = qualifying[start_offset:stop_offset]
        if run.size < minimum_turns:
            continue
        first = int(run[0])
        last = int(run[-1])
        start_time = float(turns.vertex_times[first] - turns.first_step_durations[first])
        end_time = float(turns.vertex_times[last] + turns.second_step_durations[last])
        duration = end_time - start_time
        if duration < required_duration:
            continue
        candidates.append(
            DirectedBurstCandidate(
                start_index=int(turns.first_step_start_indices[first]),
                end_index=int(turns.second_step_end_indices[last]),
                start_time=start_time,
                end_time=end_time,
                duration=float(duration),
                measurement_delta_frames=turns.measurement_delta_frames,
                n_consecutive_turns=int(run.size),
                mean_angle_radians=float(np.mean(turns.angles[run])),
                maximum_angle_radians=float(np.max(turns.angles[run])),
            )
        )
    return tuple(candidates)


# Concise aliases matching the notation used in the analysis plan.  The
# ``primary_*`` names make the jitter-safe row-offset/true-elapsed-time contract
# explicit; exact-time aliases remain available for prespecified physical lags.
msd = mean_squared_displacement
mscd = mean_squared_change_in_distance
frame_step_squared_displacement = mean_squared_displacement_by_frame_steps
frame_step_msd = mean_squared_displacement_by_frame_steps
frame_step_mscd = mean_squared_change_in_distance_by_frame_steps
frame_step_velocity = velocity_series_by_frame_steps
frame_step_vac = velocity_autocorrelation_by_frame_steps
# Primary VCC representation: lag x Site1-component x Site2-component.
frame_step_vcc_matrix = velocity_cross_correlation_matrix_by_frame_steps
# Scalar trace reduction retained for backward-compatible endpoints only.
frame_step_vcc = velocity_cross_correlation_by_frame_steps
frame_step_turning_angles = turning_angles_by_frame_steps
frame_step_dcd = directional_change_distribution_by_frame_steps
frame_step_turning_persistence = turning_angle_persistence_by_frame_steps
primary_msd = mean_squared_displacement_by_frame_steps
primary_mscd = mean_squared_change_in_distance_by_frame_steps
primary_vac = velocity_autocorrelation_by_frame_steps
primary_vcc = velocity_cross_correlation_matrix_by_frame_steps
primary_vcc_scalar_legacy = velocity_cross_correlation_by_frame_steps
primary_dcd = directional_change_distribution_by_frame_steps
vac = velocity_autocorrelation
vcc = velocity_cross_correlation
dcd = directional_change_distribution
