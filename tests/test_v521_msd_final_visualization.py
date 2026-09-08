from __future__ import annotations

import numpy as np
import pandas as pd

from dsb_states.v521_msd_final_visualization import (
    fit_trajectory_power_law,
    fitted_log_window_mean,
    select_alpha_cases,
    select_top_alpha_cases,
)


def test_trajectory_power_fit_recovers_exact_alpha() -> None:
    lag = np.linspace(10.0, 50.0, 20)
    curve = pd.DataFrame(
        {
            "lag_median_s": lag,
            "value": 0.0025 * lag**0.5,
            "pair_count": np.arange(60, 40, -1),
        }
    )
    fit = fit_trajectory_power_law(curve)
    assert fit is not None
    assert np.isclose(fit["alpha"], 0.5, atol=1e-10)
    assert np.isclose(fit["r2_log"], 1.0, atol=1e-10)


def test_log_window_mean_matches_constant_curve() -> None:
    assert np.isclose(fitted_log_window_mean(0.04, 0.0, 10.0, 50.0), 0.04)


def test_case_selection_uses_closest_supported_unique_bundles() -> None:
    table = pd.DataFrame(
        {
            "bundle_id": ["a", "b", "c", "d"],
            "site": ["site1", "site2", "site1", "site2"],
            "hour_post_delivery": [3.0, 3.0, 3.0, 3.0],
            "alpha": [0.997, 0.55, 0.2, 1.01],
            "r2_log": [0.99, 0.98, 0.99, 0.5],
            "fit_endpoint_pairs": [100, 100, 100, 100],
        }
    )
    selected = select_alpha_cases(table, minimum_r2=0.9)
    assert selected["bundle_id"].tolist() == ["a", "b"]


def test_top_alpha_selection_is_descending_and_support_guarded() -> None:
    table = pd.DataFrame(
        {
            "bundle_id": ["a", "b", "c", "d"],
            "site": ["site1", "site1", "site1", "site2"],
            "hour_post_delivery": [3.0, 3.0, 3.0, 3.0],
            "alpha": [0.8, 1.2, 1.5, 2.0],
            "r2_log": [0.99, 0.98, 0.5, 0.99],
            "fit_endpoint_pairs": [100, 90, 100, 100],
        }
    )
    selected = select_top_alpha_cases(table, top_n=2, minimum_r2=0.9)
    assert selected["bundle_id"].tolist() == ["b", "a"]
    assert selected["alpha_rank"].tolist() == [1, 2]
    assert selected["case_id"].tolist() == ["alpha_top01", "alpha_top02"]
