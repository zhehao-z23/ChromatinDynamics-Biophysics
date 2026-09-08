from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsb_states.time_resolved_physics_summary import (
    build_time_resolved_physics_summary,
    build_vcc_matrix_hour_summary,
    fbm_normalized_vac,
)


def _row(
    *,
    unit: int,
    metric: str,
    site: str,
    lag_s: float,
    value: float,
    direction: str | None = None,
    delta_s: float = np.nan,
    primary: bool = False,
) -> dict[str, object]:
    return {
        "metric": metric,
        "unit_id": f"u{unit}",
        "bundle_id": f"u{unit}",
        "nd2_id": "a1",
        "crop_id": f"c{unit // 2}",
        "hour_post_delivery": 2.0,
        "site": site,
        "direction": direction,
        "is_primary_matched_10s": primary,
        "delta_median_s": delta_s,
        "lag_frames": round(lag_s),
        "lag_median_s": lag_s,
        "scaled_lag_tau_over_delta": lag_s / delta_s if np.isfinite(delta_s) else np.nan,
        "value": value,
        "raw_value": value,
        "endpoint_pair_sd": 0.1 * abs(value) if metric in {"msd", "mscd"} else np.nan,
        "pair_count": 20,
        "normalization": 1.0 if metric in {"vac", "vcc"} else np.nan,
    }


def _synthetic_curves() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    displacement_lags = (10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
    scaled_lags = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
    for unit in range(24):
        scale = 0.9 + 0.2 * unit / 23.0
        for lag in displacement_lags:
            rows.append(
                _row(
                    unit=unit,
                    metric="msd",
                    site="site1",
                    lag_s=lag,
                    value=scale * 0.01 * lag**0.6,
                )
            )
            rows.append(
                _row(
                    unit=unit,
                    metric="mscd",
                    site="pair",
                    lag_s=lag,
                    value=scale * 0.02 * lag**0.4,
                )
            )
        for scaled in scaled_lags:
            lag = 10.0 * scaled
            rows.append(
                _row(
                    unit=unit,
                    metric="vac",
                    site="site1",
                    lag_s=lag,
                    delta_s=10.0,
                    primary=True,
                    value=scale * float(fbm_normalized_vac(scaled, 0.7)),
                )
            )
            for direction in ("site1_later_vs_site2", "site2_later_vs_site1"):
                rows.append(
                    _row(
                        unit=unit,
                        metric="vcc",
                        site="pair",
                        direction=direction,
                        lag_s=lag,
                        delta_s=10.0,
                        primary=True,
                        value=scale * 0.3 * np.exp(-scaled),
                    )
                )
    return pd.DataFrame(rows)


def test_summary_recovers_known_descriptive_exponents_and_symmetric_vcc() -> None:
    result = build_time_resolved_physics_summary(
        _synthetic_curves(),
        bootstrap_iterations=30,
    )

    msd_alpha = result.fits.loc[
        result.fits["metric"].eq("msd") & result.fits["estimate_name"].eq("exponent"),
        "estimate",
    ].iloc[0]
    mscd_beta = result.fits.loc[
        result.fits["metric"].eq("mscd") & result.fits["estimate_name"].eq("exponent"),
        "estimate",
    ].iloc[0]
    vac_alpha = result.fits.loc[
        result.fits["metric"].eq("vac") & result.fits["estimate_name"].eq("exponent"),
        "estimate",
    ].iloc[0]
    assert msd_alpha == pytest.approx(0.6, abs=1e-8)
    assert mscd_beta == pytest.approx(0.4, abs=1e-8)
    assert vac_alpha == pytest.approx(0.7, abs=0.03)
    assert result.hour_curves.loc[result.hour_curves["metric"].eq("msd"), "sd"].gt(0).all()
    assert set(result.unit_bins.loc[result.unit_bins["metric"].eq("vcc"), "direction"]) == {
        "symmetric_trace_mean"
    }
    vcc_zero = result.fits.loc[
        result.fits["metric"].eq("vcc")
        & result.fits["estimate_name"].eq("zero_lag_trace")
    ].iloc[0]
    assert np.isfinite(vcc_zero.ci_low)
    assert np.isfinite(vcc_zero.ci_high)
    assert vcc_zero.ci_high - vcc_zero.ci_low < 0.05
    assert result.method_contract["experimental_batch_assumption"]["same_batch"] is True


def test_non_primary_velocity_family_is_not_used_for_cross_hour_summary() -> None:
    source = _synthetic_curves()
    extra = source.loc[source["metric"].eq("vac")].copy()
    extra["is_primary_matched_10s"] = False
    extra["value"] = 1000.0
    result = build_time_resolved_physics_summary(
        pd.concat([source, extra], ignore_index=True),
        bootstrap_iterations=0,
    )
    assert result.hour_curves.loc[result.hour_curves["metric"].eq("vac"), "mean"].abs().max() < 2


def test_vcc_matrix_summary_transposes_reverse_direction_before_mean() -> None:
    rows = []
    for direction, values in (
        ("site1_later_vs_site2", {("x", "y"): 2.0}),
        ("site2_later_vs_site1", {("y", "x"): 4.0}),
    ):
        for (row, column), value in values.items():
            rows.append(
                {
                    "unit_id": "u1",
                    "nd2_id": "a1",
                    "crop_id": "c1",
                    "hour_post_delivery": 2.0,
                    "direction": direction,
                    "is_primary_matched_10s": True,
                    "lag_frames": 1,
                    "lag_median_s": 10.0,
                    "scaled_lag_tau_over_delta": 1.0,
                    "row_component": row,
                    "column_component": column,
                    "normalized_value": value,
                    "raw_value": 10.0 * value,
                    "pair_count": 12,
                }
            )
    summary = build_vcc_matrix_hour_summary(pd.DataFrame(rows))
    assert len(summary) == 1
    assert summary.iloc[0].row_component == "x"
    assert summary.iloc[0].column_component == "y"
    assert summary.iloc[0].normalized_mean == pytest.approx(3.0)
