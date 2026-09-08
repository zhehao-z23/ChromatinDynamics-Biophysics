from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import toeplitz

from dsb_states.physical_metrics import (
    ExactTimeSupportError,
    LagCurve,
    directed_burst_candidates_by_frame_steps,
    directional_change_distribution,
    directional_change_distribution_by_frame_steps,
    integrate_supported_curve,
    mean_squared_change_in_distance,
    mean_squared_change_in_distance_by_frame_steps,
    mean_squared_displacement,
    mean_squared_displacement_by_frame_steps,
    positive_short_lag_integral,
    primary_msd,
    turning_angle_persistence,
    turning_angle_persistence_by_frame_steps,
    turning_angles,
    turning_angles_by_frame_steps,
    velocity_autocorrelation,
    velocity_autocorrelation_by_frame_steps,
    velocity_cross_correlation,
    velocity_cross_correlation_by_frame_steps,
    velocity_cross_correlation_matrix_by_frame_steps,
    velocity_series_by_frame_steps,
)


def _random_walk(seed: int, n_steps: int, dimensions: int = 2) -> np.ndarray:
    generator = np.random.default_rng(seed)
    increments = generator.normal(size=(n_steps, dimensions))
    return np.vstack((np.zeros((1, dimensions)), np.cumsum(increments, axis=0)))


def _fractional_brownian_motion(seed: int, n_steps: int, hurst: float) -> np.ndarray:
    """Small exact-covariance fBM fixture; not a production simulator."""

    offsets = np.arange(n_steps, dtype=float)
    covariance = 0.5 * (
        np.abs(offsets + 1.0) ** (2.0 * hurst)
        - 2.0 * np.abs(offsets) ** (2.0 * hurst)
        + np.abs(offsets - 1.0) ** (2.0 * hurst)
    )
    generator = np.random.default_rng(seed)
    increments = generator.multivariate_normal(
        np.zeros(n_steps),
        toeplitz(covariance),
        method="cholesky",
    )
    return np.r_[0.0, np.cumsum(increments)]


def test_identical_pair_has_zero_mscd_and_unit_zero_lag_vcc() -> None:
    trajectory = _random_walk(11, 4000)
    lag_grid = np.array([1.0, 2.0, 5.0, 10.0])
    curve = mean_squared_change_in_distance(trajectory, trajectory, lags=lag_grid)
    np.testing.assert_allclose(curve.values, 0.0)
    np.testing.assert_array_equal(curve.counts, trajectory.shape[0] - lag_grid.astype(int))

    coupling = velocity_cross_correlation(trajectory, trajectory, lags=[0.0, 1.0])
    assert np.isclose(coupling.values[0], 1.0)
    assert coupling.counts[0] == trajectory.shape[0] - 1


def test_opposite_motion_has_minus_one_zero_lag_vcc() -> None:
    trajectory = _random_walk(12, 2000)
    coupling = velocity_cross_correlation(trajectory, -trajectory, lags=[0.0])

    assert np.isclose(coupling.values[0], -1.0)


def test_independent_brownian_pair_has_near_zero_vcc() -> None:
    site1 = _random_walk(21, 30_000)
    site2 = _random_walk(22, 30_000)
    coupling = velocity_cross_correlation(site1, site2, lags=[0.0])

    assert abs(coupling.values[0]) < 0.025


def test_brownian_msd_is_linear_with_lag() -> None:
    trajectory = _random_walk(30, 40_000)
    lags = np.arange(1.0, 31.0)
    curve = mean_squared_displacement(trajectory, lags=lags)
    slope, intercept = np.polyfit(curve.lags, curve.values, deg=1)

    # Unit-variance increments in two dimensions have E[MSD(tau)] = 2*tau.
    assert 1.85 < slope < 2.15
    assert abs(intercept) < 1.0
    np.testing.assert_array_equal(curve.counts, trajectory.shape[0] - lags.astype(int))


def test_subdiffusive_fbm_has_negative_short_lag_vac() -> None:
    trajectory = _fractional_brownian_motion(seed=103, n_steps=384, hurst=0.2)
    curve = velocity_autocorrelation(trajectory, lags=[0.0, 1.0, 2.0])

    # Fractional Gaussian noise has gamma(1) = 0.5*(2**(2H)-2), negative for H<0.5.
    assert curve.values[1] < -0.2
    assert curve.counts[1] == trajectory.size - 2


def test_localization_noise_produces_identifiable_msd_intercept() -> None:
    true_trajectory = _random_walk(104, 40_000)
    sigma_nm = 2.0
    generator = np.random.default_rng(105)
    observed = true_trajectory + generator.normal(scale=sigma_nm, size=true_trajectory.shape)
    lags = np.arange(1.0, 31.0)

    clean_curve = mean_squared_displacement(true_trajectory, lags=lags)
    noisy_curve = mean_squared_displacement(observed, lags=lags)
    _, clean_intercept = np.polyfit(clean_curve.lags, clean_curve.values, deg=1)
    noisy_slope, noisy_intercept = np.polyfit(noisy_curve.lags, noisy_curve.values, deg=1)

    # In two dimensions, independent localization error adds 4*sigma**2 to MSD.
    assert abs(clean_intercept) < 1.0
    assert 1.8 < noisy_slope < 2.2
    assert noisy_intercept == pytest.approx(4.0 * sigma_nm**2, abs=2.0)


def test_vac_is_normalized_by_observed_zero_lag_energy() -> None:
    trajectory = _random_walk(31, 5000)
    curve = velocity_autocorrelation(trajectory, lags=[0.0, 1.0, 2.0])

    assert np.isclose(curve.values[0], 1.0)
    assert np.isclose(curve.raw_values[0], curve.normalization)
    assert abs(curve.values[1]) < 0.06


def test_directed_motion_enriches_near_zero_turns() -> None:
    times = np.arange(101, dtype=float) * 0.25
    trajectory = np.column_stack((times, 0.2 * times))
    angles = turning_angles(trajectory, times=times, delta=0.25)
    summary = turning_angle_persistence(
        trajectory,
        times=times,
        delta=0.25,
        threshold_degrees=20.0,
        n_shuffles=25,
        random_state=7,
    )

    np.testing.assert_allclose(angles, 0.0, atol=1e-8)
    assert summary.near_zero_probability == 1.0
    assert summary.near_pi_probability == 0.0
    assert summary.persistence_score == 1.0


def test_reversal_motion_enriches_near_pi_turns() -> None:
    trajectory = np.zeros((101, 2), dtype=float)
    trajectory[:, 0] = np.arange(101) % 2
    angles = turning_angles(trajectory)
    summary = turning_angle_persistence(trajectory, threshold_degrees=20.0)

    np.testing.assert_allclose(angles, np.pi)
    assert summary.near_zero_probability == 0.0
    assert summary.near_pi_probability == 1.0
    assert summary.persistence_score == -1.0


def test_primary_frame_step_dcd_recovers_directed_and_reversal_signatures() -> None:
    times = np.array([0.0, 1.1, 2.0, 3.2, 4.0])
    directed = np.column_stack((times, 2.0 * times))
    reversal = (np.arange(times.size) % 2).astype(float)

    directed_turns = turning_angles_by_frame_steps(directed, times=times)
    directed_dcd = directional_change_distribution_by_frame_steps(directed, times=times)
    directed_summary = turning_angle_persistence_by_frame_steps(
        directed,
        times=times,
        threshold_degrees=20.0,
    )
    reversal_summary = turning_angle_persistence_by_frame_steps(
        reversal,
        times=times,
        threshold_degrees=20.0,
    )

    np.testing.assert_allclose(directed_turns.angles, 0.0, atol=5e-8)
    np.testing.assert_array_equal(directed_turns.first_step_start_indices, [0, 1, 2])
    np.testing.assert_array_equal(directed_turns.vertex_indices, [1, 2, 3])
    np.testing.assert_array_equal(directed_turns.second_step_end_indices, [2, 3, 4])
    np.testing.assert_allclose(directed_turns.first_step_durations, [1.1, 0.9, 1.2])
    np.testing.assert_allclose(directed_turns.second_step_durations, [0.9, 1.2, 0.8])
    np.testing.assert_allclose(directed_turns.full_turn_durations, [2.0, 2.1, 2.0])
    assert directed_dcd.n_angles == 3
    assert directed_dcd.turns is directed_turns or np.array_equal(
        directed_dcd.turns.angles,
        directed_turns.angles,
    )
    assert directed_summary.persistence_score == 1.0
    assert reversal_summary.persistence_score == -1.0


def test_implicit_exact_time_dcd_fails_closed_on_jitter() -> None:
    times = np.array([0.0, 1.1, 2.0, 3.2, 4.0])
    positions = np.column_stack((times, 2.0 * times))

    with pytest.raises(ExactTimeSupportError, match="primary frame-step"):
        turning_angles(positions, times=times)
    with pytest.raises(ExactTimeSupportError, match="primary frame-step"):
        directional_change_distribution(positions, times=times)
    with pytest.raises(ExactTimeSupportError, match="primary frame-step"):
        turning_angle_persistence(positions, times=times)


def test_missing_frames_use_only_valid_endpoint_pairs_without_interpolation() -> None:
    # The missing coordinate at t=1 removes both unit-lag pairs touching it.
    # Lag-two endpoints remain directly observable and therefore valid.
    times = np.arange(5, dtype=float)
    positions = np.array([[0.0], [np.nan], [2.0], [3.0], [4.0]])
    curve = mean_squared_displacement(positions, times=times, lags=[1.0, 2.0])

    np.testing.assert_array_equal(curve.counts, [2, 2])
    np.testing.assert_allclose(curve.values, [1.0, 4.0])

    # With the time row absent entirely, the time jump is not mistaken for one
    # frame; the two supported unit-lag pairs are (0,1) and (3,4).
    omitted_times = np.array([0.0, 1.0, 3.0, 4.0])
    omitted_positions = omitted_times[:, np.newaxis]
    omitted = mean_squared_displacement(
        omitted_positions,
        times=omitted_times,
        lags=[1.0, 2.0],
    )
    np.testing.assert_array_equal(omitted.counts, [2, 1])
    np.testing.assert_allclose(omitted.values, [1.0, 4.0])


def test_exact_time_default_tolerance_is_epoch_safe_without_becoming_a_lag_bin() -> None:
    origin = 1.0e12
    times = origin + np.array([0.0, 0.1, 0.2, 0.3])
    positions = np.arange(times.size, dtype=float)

    representable = mean_squared_displacement(
        positions,
        times=times,
        lags=[0.1],
    )
    wrong_lag = mean_squared_displacement(
        positions,
        times=times,
        lags=[0.101],
    )

    # Decimal tenths beside this epoch are quantized at roughly 1.22e-4 s, so
    # the default must absorb one input ULP but reject a physical 1 ms mismatch.
    np.testing.assert_array_equal(representable.counts, [3])
    np.testing.assert_allclose(representable.values, [1.0])
    np.testing.assert_array_equal(wrong_lag.counts, [0])
    assert np.isnan(wrong_lag.values[0])


def test_missing_frame_does_not_create_a_turn_or_velocity_across_unit_gap() -> None:
    positions = np.array([[0.0], [1.0], [np.nan], [3.0], [4.0]])
    angles = turning_angles(positions, delta=1.0)
    vac = velocity_autocorrelation(positions, delta=1.0, lags=[0.0, 1.0])

    assert angles.size == 0
    np.testing.assert_array_equal(vac.counts, [2, 0])
    assert np.isclose(vac.values[0], 1.0)
    assert np.isnan(vac.values[1])

    frame_turns = turning_angles_by_frame_steps(positions)
    frame_dcd = directional_change_distribution_by_frame_steps(positions)
    assert frame_turns.n_angles == 0
    assert frame_dcd.n_angles == 0
    assert np.isnan(frame_dcd.probabilities).all()


def test_directed_burst_candidates_are_split_by_original_row_gaps() -> None:
    times = np.array([0.0, 1.0, 2.1, 3.0, 4.2, 5.0, 6.2, 7.0, 8.1, 9.0, 10.2])
    positions = times.copy()
    positions[5] = np.nan

    candidates = directed_burst_candidates_by_frame_steps(
        positions,
        times=times,
        threshold_degrees=10.0,
        min_consecutive_turns=3,
        minimum_duration=3.0,
    )

    assert len(candidates) == 2
    assert [(item.start_index, item.end_index) for item in candidates] == [(0, 4), (6, 10)]
    np.testing.assert_allclose([item.duration for item in candidates], [4.2, 4.0])
    assert all(item.n_consecutive_turns == 3 for item in candidates)
    assert all(item.maximum_angle_radians < 5e-8 for item in candidates)


def test_turning_angle_tolerance_cannot_join_distinct_physical_vertices() -> None:
    times = np.array([0.0, 1.0, 1.1, 2.0, 2.1])
    positions = np.array([0.0, 1.0, 10.0, np.nan, 11.0])

    # The supported velocities use rows 0->1 and 2->4.  Their start times are
    # within the explicit tolerance of one delta apart, but row 1 is not row 2:
    # there is no shared observed vertex and therefore no physical turn.
    angles = turning_angles(
        positions,
        times=times,
        delta=1.0,
        time_tolerance=0.15,
    )

    assert angles.size == 0


def test_turning_angle_tolerance_keeps_a_true_shared_vertex() -> None:
    times = np.array([0.0, 1.1, 2.2])
    positions = np.array([0.0, 1.0, 2.0])

    angles = turning_angles(
        positions,
        times=times,
        delta=1.0,
        time_tolerance=0.15,
    )

    np.testing.assert_allclose(angles, [0.0])


def test_min_pairs_masks_value_but_preserves_pair_count() -> None:
    trajectory = np.arange(5, dtype=float)
    curve = mean_squared_displacement(trajectory, lags=[1.0, 4.0], min_pairs=2)

    np.testing.assert_array_equal(curve.counts, [4, 1])
    assert np.isclose(curve.values[0], 1.0)
    assert np.isnan(curve.values[1])


def test_frame_step_msd_retains_pairs_under_small_timestamp_jitter() -> None:
    times = np.array([0.0, 1.01, 2.0, 3.02, 4.0])
    positions = np.arange(times.size, dtype=float)

    exact_time = mean_squared_displacement(positions, times=times, lags=[1.0])
    frame_step = mean_squared_displacement_by_frame_steps(
        positions,
        times=times,
        lag_steps=[1, 2],
    )

    # No elapsed time is exactly one, so the original exact-time API remains
    # intentionally strict.  Row offsets retain all finite endpoint pairs.
    np.testing.assert_array_equal(exact_time.counts, [0])
    np.testing.assert_array_equal(frame_step.lag_steps, [1, 2])
    np.testing.assert_array_equal(frame_step.counts, [4, 3])
    np.testing.assert_allclose(frame_step.values, [1.0, 4.0])
    np.testing.assert_allclose(frame_step.standard_deviations, [0.0, 0.0])
    np.testing.assert_allclose(frame_step.lag_time_mean, [1.0, 2.0033333333333334])
    np.testing.assert_allclose(frame_step.lag_time_median, [1.0, 2.0])
    np.testing.assert_allclose(frame_step.lag_time_min, [0.98, 2.0])
    np.testing.assert_allclose(frame_step.lag_time_max, [1.02, 2.01])
    np.testing.assert_array_equal(frame_step.lag_time_count, [4, 3])

    # The frame-step MSCD wrapper uses the same support and time summaries.
    zeros = np.zeros((times.size, 1))
    site2 = positions[:, np.newaxis]
    mscd_curve = mean_squared_change_in_distance_by_frame_steps(
        zeros,
        site2,
        times=times,
        lag_steps=[1, 2],
    )
    np.testing.assert_allclose(mscd_curve.values, [1.0, 4.0])
    np.testing.assert_allclose(mscd_curve.standard_deviations, [0.0, 0.0])
    np.testing.assert_allclose(mscd_curve.lag_time_median, [1.0, 2.0])


def test_implicit_exact_time_curves_fail_closed_on_jitter_but_primary_retains_support() -> None:
    times = np.array([0.0, 1.01, 2.0, 3.02, 4.0])
    positions = np.arange(times.size, dtype=float)

    with pytest.raises(ExactTimeSupportError, match="primary frame-step"):
        mean_squared_displacement(positions, times=times)
    with pytest.raises(ExactTimeSupportError, match="primary frame-step"):
        velocity_autocorrelation(positions, times=times)
    with pytest.raises(ExactTimeSupportError, match="primary frame-step"):
        velocity_cross_correlation(positions, positions, times=times)

    primary = primary_msd(positions, times=times, lag_steps=[1, 2])
    np.testing.assert_array_equal(primary.counts, [4, 3])
    np.testing.assert_allclose(primary.lag_time_median, [1.0, 2.0])


def test_frame_step_curve_exposes_burst_and_long_gap_elapsed_times() -> None:
    times = np.array([0.0, 0.1, 0.2, 5.2, 5.3, 5.4])
    positions = np.arange(times.size, dtype=float)
    curve = mean_squared_displacement_by_frame_steps(
        positions,
        times=times,
        lag_steps=[1, 2],
    )

    np.testing.assert_array_equal(curve.counts, [5, 4])
    np.testing.assert_allclose(curve.values, [1.0, 4.0])
    np.testing.assert_allclose(curve.lag_time_mean, [1.08, 2.65])
    np.testing.assert_allclose(curve.lag_time_median, [0.1, 2.65])
    np.testing.assert_allclose(curve.lag_time_min, [0.1, 0.2])
    np.testing.assert_allclose(curve.lag_time_max, [5.0, 5.1])


def test_frame_step_elapsed_time_summary_uses_valid_metric_endpoints() -> None:
    times = np.array([0.0, 1.0, 10.0, 11.0])
    positions = np.array([0.0, np.nan, 2.0, 3.0])
    curve = mean_squared_displacement_by_frame_steps(
        positions,
        times=times,
        lag_steps=[1],
    )

    # Only rows 2 -> 3 have two finite coordinate endpoints.
    np.testing.assert_array_equal(curve.counts, [1])
    np.testing.assert_allclose(curve.values, [1.0])
    np.testing.assert_allclose(curve.lag_time_mean, [1.0])
    np.testing.assert_allclose(curve.lag_time_median, [1.0])
    np.testing.assert_allclose(curve.lag_time_min, [1.0])
    np.testing.assert_allclose(curve.lag_time_max, [1.0])


def test_frame_step_velocity_and_vac_handle_jittered_timestamps() -> None:
    times = np.array([0.0, 1.1, 2.0, 3.2, 4.0])
    trajectory = np.column_stack((times, 2.0 * times))

    velocities = velocity_series_by_frame_steps(
        trajectory,
        times=times,
        measurement_delta_frames=1,
    )
    coarse_velocities = velocity_series_by_frame_steps(
        trajectory,
        times=times,
        measurement_delta_frames=2,
    )
    curve = velocity_autocorrelation_by_frame_steps(
        trajectory,
        times=times,
        measurement_delta_frames=1,
        lag_frames=[0, 1, 2],
    )

    np.testing.assert_array_equal(velocities.start_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(velocities.end_indices, [1, 2, 3, 4])
    np.testing.assert_allclose(velocities.durations, [1.1, 0.9, 1.2, 0.8])
    np.testing.assert_allclose(velocities.vectors, [[1.0, 2.0]] * 4)
    np.testing.assert_array_equal(coarse_velocities.start_indices, [0, 1, 2])
    np.testing.assert_array_equal(coarse_velocities.end_indices, [2, 3, 4])
    np.testing.assert_allclose(coarse_velocities.durations, [2.0, 2.1, 2.0])
    np.testing.assert_allclose(coarse_velocities.vectors, [[1.0, 2.0]] * 3)
    np.testing.assert_array_equal(curve.counts, [4, 3, 2])
    np.testing.assert_allclose(curve.raw_values, [5.0, 5.0, 5.0])
    np.testing.assert_allclose(curve.values, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(curve.start_time_lag_mean, [0.0, 3.2 / 3.0, 2.05])
    np.testing.assert_allclose(curve.start_time_lag_median, [0.0, 1.1, 2.05])
    np.testing.assert_allclose(curve.start_time_lag_min, [0.0, 0.9, 2.0])
    np.testing.assert_allclose(curve.start_time_lag_max, [0.0, 1.2, 2.1])
    np.testing.assert_array_equal(curve.start_time_lag_count, [4, 3, 2])


def test_frame_step_vcc_is_unit_for_identical_and_opposite_motion() -> None:
    times = np.array([0.0, 0.8, 1.9, 3.3, 4.0, 5.2])
    site1 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [0.5, 2.0],
            [2.5, 1.0],
            [3.0, 3.0],
            [5.0, 2.5],
        ]
    )
    identical = velocity_cross_correlation_by_frame_steps(
        site1,
        site1,
        times=times,
        measurement_delta_frames=1,
        lag_frames=[0],
    )
    opposite = velocity_cross_correlation_by_frame_steps(
        site1,
        -site1,
        times=times,
        measurement_delta_frames=1,
        lag_frames=[0],
    )

    np.testing.assert_allclose(identical.values, [1.0])
    np.testing.assert_allclose(opposite.values, [-1.0])
    np.testing.assert_array_equal(identical.counts, [5])
    np.testing.assert_allclose(identical.start_time_lag_mean, [0.0])


def test_msd_reports_endpoint_pair_sample_sd_not_standard_error() -> None:
    positions = np.array([0.0, 1.0, 3.0, 6.0])
    exact = mean_squared_displacement(positions, lags=[1.0])
    frame_step = mean_squared_displacement_by_frame_steps(positions, lag_steps=[1])
    squared = np.array([1.0, 4.0, 9.0])

    for curve in (exact, frame_step):
        np.testing.assert_allclose(curve.values, [np.mean(squared)])
        np.testing.assert_allclose(curve.standard_deviations, [np.std(squared, ddof=1)])
        assert curve.counts[0] == squared.size


def test_vcc_matrix_retains_cross_components_and_scalar_is_its_trace() -> None:
    times = np.arange(3, dtype=float)
    site1 = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    site2 = np.array([[0.0, 0.0], [0.0, 3.0], [0.0, 7.0]])
    matrix = velocity_cross_correlation_matrix_by_frame_steps(
        site1,
        site2,
        times=times,
        measurement_delta_frames=1,
        lag_frames=[0],
    )
    scalar = velocity_cross_correlation_by_frame_steps(
        site1,
        site2,
        times=times,
        measurement_delta_frames=1,
        lag_frames=[0],
    )

    np.testing.assert_allclose(matrix.raw_matrices[0], [[0.0, 5.5], [0.0, 0.0]])
    assert matrix.counts[0] == 2
    assert matrix.normalization == pytest.approx(np.sqrt(2.5 * 12.5))
    np.testing.assert_allclose(matrix.trace_values, scalar.values)
    # Scalar zero hides a real off-diagonal component; this is why the matrix is primary.
    assert matrix.trace_values[0] == 0.0
    assert matrix.matrices[0, 0, 1] > 0.0


def test_vcc_primary_uses_fixed_zero_lag_energy_under_asymmetric_missingness() -> None:
    times = np.arange(6, dtype=float)
    site1 = np.array([np.nan, 0.0, 100.0, np.nan, 0.0, 1.0])
    site2 = np.array([0.0, 100.0, np.nan, np.nan, 0.0, 1.0])

    exact = velocity_cross_correlation(
        site1,
        site2,
        times=times,
        lags=[0.0, 1.0],
    )
    frame_step = velocity_cross_correlation_by_frame_steps(
        site1,
        site2,
        times=times,
        lag_frames=[0, 1],
    )

    # Zero-lag cross support contains only the unit-speed pair at row 4.  Each
    # site's own auto-energy support contains speeds 100 and 1, so the fixed
    # master-plan denominator is 5000.5. Lag-one cross support contains only the
    # 100-speed pair. The bounded paired-support ratio remains available only as
    # an explicitly secondary sensitivity.
    for curve in (exact, frame_step):
        np.testing.assert_array_equal(curve.counts, [1, 1])
        np.testing.assert_allclose(curve.raw_values, [1.0, 10_000.0])
        assert curve.normalization == 5000.5
        assert curve.normalizations is not None
        np.testing.assert_allclose(curve.normalizations, [5000.5, 5000.5])
        np.testing.assert_allclose(curve.values, [1.0 / 5000.5, 10_000.0 / 5000.5])
        assert curve.secondary_lag_specific_normalizations is not None
        assert curve.secondary_lag_specific_values is not None
        np.testing.assert_allclose(
            curve.secondary_lag_specific_normalizations,
            [1.0, 10_000.0],
        )
        np.testing.assert_allclose(curve.secondary_lag_specific_values, [1.0, 1.0])


def test_guarded_curve_integral_does_not_bridge_unsupported_lags() -> None:
    curve = LagCurve(
        lags=np.array([0.0, 1.0, 2.0, 3.0]),
        values=np.array([0.0, 2.0, -2.0, 2.0]),
        counts=np.array([10, 1, 10, 10]),
    )

    result = integrate_supported_curve(
        curve,
        min_pairs_per_lag=5,
        minimum_duration=2.0,
    )

    # Only 2 -> 3 is adjacent and supported.  The missing lag at one is not
    # bridged, so the one-second support fails the prespecified duration rule.
    assert not result.valid
    assert result.reason == "insufficient_supported_duration"
    assert result.supported_interval_count == 1
    assert result.supported_duration == 1.0
    assert np.isnan(result.area)


def test_positive_short_lag_integral_handles_zero_crossings_exactly() -> None:
    curve = LagCurve(
        lags=np.array([0.0, 1.0, 2.0, 3.0]),
        values=np.array([0.0, 2.0, -2.0, 2.0]),
        counts=np.array([10, 10, 10, 10]),
    )

    result = positive_short_lag_integral(
        curve,
        maximum_lag_time=3.0,
        min_pairs_per_lag=5,
        minimum_duration=3.0,
    )

    assert result.valid
    assert result.supported_duration == 3.0
    assert result.area == pytest.approx(2.0)


def test_curve_auc_uses_frame_step_empirical_elapsed_time() -> None:
    times = np.array([0.0, 1.0, 3.0, 6.0])
    curve = mean_squared_displacement_by_frame_steps(
        np.arange(4, dtype=float),
        times=times,
        lag_steps=[1, 2, 3],
    )
    result = integrate_supported_curve(
        curve,
        min_pairs_per_lag=1,
        minimum_duration=4.0,
    )

    np.testing.assert_allclose(curve.lag_time_median, [2.0, 4.0, 6.0])
    assert result.valid
    assert result.supported_duration == 4.0
    assert result.area == pytest.approx(18.0)


def test_frame_step_velocity_correlations_do_not_compress_missing_endpoints() -> None:
    times = np.array([0.0, 1.1, 2.0, 3.2, 4.0])
    complete = times[:, np.newaxis]
    missing = complete.copy()
    missing[2] = np.nan

    velocities = velocity_series_by_frame_steps(
        missing,
        times=times,
        measurement_delta_frames=1,
    )
    vac_curve = velocity_autocorrelation_by_frame_steps(
        missing,
        times=times,
        measurement_delta_frames=1,
        lag_frames=[0, 1, 3],
    )
    vcc_curve = velocity_cross_correlation_by_frame_steps(
        missing,
        complete,
        times=times,
        measurement_delta_frames=1,
        lag_frames=[0, 1],
    )

    # Starts 1 and 2 touch the missing row and cannot define velocities.  The
    # remaining starts keep indices 0 and 3 rather than becoming adjacent.
    np.testing.assert_array_equal(velocities.start_indices, [0, 3])
    np.testing.assert_array_equal(velocities.end_indices, [1, 4])
    np.testing.assert_allclose(velocities.durations, [1.1, 0.8])
    np.testing.assert_array_equal(vac_curve.counts, [2, 0, 1])
    assert np.isnan(vac_curve.values[1])
    assert np.isnan(vac_curve.start_time_lag_mean[1])
    np.testing.assert_allclose(vac_curve.start_time_lag_mean[[0, 2]], [0.0, 3.2])
    np.testing.assert_array_equal(vcc_curve.counts, [2, 1])
    np.testing.assert_allclose(vcc_curve.start_time_lag_mean, [0.0, 1.2])
