from __future__ import annotations

import numpy as np
import pandas as pd

from dsb_states.oligo_vcc import (
    RouseVccReference,
    _fit_tau_grid,
    _symmetrize_curves,
    _symmetrize_matrices,
)


def _linear_reference() -> RouseVccReference:
    alpha = np.array([0.25, 1.0])
    log_ratio = np.array([-3.0, 3.0])
    lag = np.array([0.0, 5.0])
    values = (
        0.1 * alpha[:, None, None]
        + 0.2 * log_ratio[None, :, None]
        + 0.3 * lag[None, None, :]
    )
    return RouseVccReference(alpha, log_ratio, lag, values)


def _identity(direction: str) -> dict[str, object]:
    return {
        "unit_id": "u1",
        "bundle_id": "u1",
        "cohort": "test",
        "nd2_id": "a1",
        "crop_id": "c1",
        "fov_id": "f1",
        "hour_post_delivery": 2.0,
        "target_delta_s": 10.0,
        "delta_frames": 2,
        "schedule_delta_s": 10.0,
        "actual_delta_s": 10.0,
        "direction": direction,
        "lag_frames": 0,
        "lag_median_s": 0.0,
        "scaled_lag": 0.0,
        "pair_count": 8,
    }


def test_reference_interpolation_is_trilinear() -> None:
    reference = _linear_reference()
    observed = reference.evaluate(0.625, 1.0, 2.5)
    assert np.isclose(observed, 0.1 * 0.625 + 0.3 * 2.5)


def test_tau_grid_recovers_synthetic_communication_time() -> None:
    reference = _linear_reference()
    rows = []
    true_tau = 100.0
    for target in (10.0, 20.0):
        for scaled_lag in (0.0, 0.5, 1.0):
            predicted = reference.evaluate(0.9, target / true_tau, scaled_lag)
            rows.append(
                {
                    "target_delta_s": target,
                    "scaled_lag_bin": scaled_lag,
                    "mean": float(predicted),
                    "n_units": 12,
                    "actual_delta_s_mean": target,
                }
            )
    fit = _fit_tau_grid(
        pd.DataFrame(rows),
        reference,
        alpha=0.9,
        tau_grid_s=np.arange(1.0, 201.0),
        minimum_units_per_bin=8,
    )
    assert fit is not None
    assert fit["tau_s"] == true_tau
    assert fit["rmse"] < 1e-12


def test_bidirectional_trace_is_equal_weighted() -> None:
    rows = []
    for direction, value in (
        ("site1_later_vs_site2", 0.2),
        ("site2_later_vs_site1", 0.6),
    ):
        rows.append(
            {
                **_identity(direction),
                "normalized_vcc": value,
                "raw_vcc_um2_per_s2": value * 2,
                "zero_lag_energy_um2_per_s2": 2.0,
            }
        )
    symmetric = _symmetrize_curves(pd.DataFrame(rows))
    assert len(symmetric) == 1
    assert np.isclose(symmetric.iloc[0]["normalized_vcc"], 0.4)
    assert symmetric.iloc[0]["direction_count"] == 2


def test_reverse_direction_matrix_is_transposed_before_mean() -> None:
    rows = []
    forward = np.array([[1.0, 2.0], [3.0, 4.0]])
    reverse = np.array([[5.0, 6.0], [7.0, 8.0]])
    for direction, matrix in (
        ("site1_later_vs_site2", forward),
        ("site2_later_vs_site1", reverse),
    ):
        for row_index, row_component in enumerate(("x", "y")):
            for column_index, column_component in enumerate(("x", "y")):
                rows.append(
                    {
                        **_identity(direction),
                        "row_component": row_component,
                        "column_component": column_component,
                        "normalized_value": matrix[row_index, column_index],
                        "raw_value_um2_per_s2": 2 * matrix[row_index, column_index],
                    }
                )
    symmetric = _symmetrize_matrices(pd.DataFrame(rows))
    pivot = symmetric.pivot(
        index="row_component", columns="column_component", values="normalized_value"
    ).loc[["x", "y"], ["x", "y"]]
    np.testing.assert_allclose(pivot.to_numpy(), (forward + reverse.T) / 2)
