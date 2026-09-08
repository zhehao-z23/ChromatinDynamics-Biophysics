from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsb_states.time_resolved_physics_plots import (
    build_msd_curves_figure,
    build_vac_curves_figures,
    write_time_resolved_physics_figures,
)

HOURS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0)


def _curves(*, omit_five_hour_vac: bool = False) -> pd.DataFrame:
    rows: list[dict] = []
    for hour in HOURS:
        for site, multiplier in (("site1", 1.0), ("site2", 0.82)):
            for lag in (5.0, 15.0, 40.0, 80.0):
                rows.append(
                    {
                        "metric": "msd",
                        "site": site,
                        "hour_post_delivery": hour,
                        "lag_bin_center_s": lag,
                        "lag_s_mean": lag,
                        "scaled_lag_mean": np.nan,
                        "mean": multiplier * 0.002 * lag**0.55,
                        "sd": multiplier * 0.0004 * lag**0.55,
                        "q25": multiplier * 0.0017 * lag**0.55,
                        "q75": multiplier * 0.0023 * lag**0.55,
                        "n_units": 30,
                        "total_pairs": 450,
                        "is_primary_matched_10s": False,
                    }
                )
        for lag in (5.0, 15.0, 40.0, 80.0):
            rows.append(
                {
                    "metric": "mscd",
                    "site": "pair",
                    "hour_post_delivery": hour,
                    "lag_bin_center_s": lag,
                    "lag_s_mean": lag,
                    "scaled_lag_mean": np.nan,
                    "mean": 0.0015 * lag**0.62,
                    "sd": 0.00035 * lag**0.62,
                    "q25": 0.0012 * lag**0.62,
                    "q75": 0.0018 * lag**0.62,
                    "n_units": 24,
                    "total_pairs": 360,
                    "is_primary_matched_10s": False,
                }
            )
        if not (omit_five_hour_vac and hour == 5.0):
            for site, shift in (("site1", 0.0), ("site2", 0.03)):
                for scaled_lag, value in ((0.0, 1.0), (0.5, 0.18), (1.0, -0.23), (2.0, -0.03)):
                    rows.append(
                        {
                            "metric": "vac",
                            "site": site,
                            "hour_post_delivery": hour,
                            "lag_bin_center_s": scaled_lag * 10.0,
                            "lag_s_mean": scaled_lag * 10.0,
                            "scaled_lag_mean": scaled_lag,
                            "mean": value + shift,
                            "sd": 0.08,
                            "q25": value - 0.04 + shift,
                            "q75": value + 0.04 + shift,
                            "n_units": 18,
                            "total_pairs": 180,
                            "is_primary_matched_10s": True,
                        }
                    )
        for scaled_lag, value in ((0.0, 0.25), (0.5, 0.14), (1.0, 0.06), (2.0, 0.0)):
            rows.append(
                {
                    "metric": "vcc",
                    "site": "pair",
                    "hour_post_delivery": hour,
                    "lag_bin_center_s": scaled_lag * 10.0,
                    "lag_s_mean": scaled_lag * 10.0,
                    "scaled_lag_mean": scaled_lag,
                    "mean": value,
                    "sd": 0.06,
                    "q25": value - 0.03,
                    "q75": value + 0.03,
                    "n_units": 16,
                    "total_pairs": 160,
                    "is_primary_matched_10s": True,
                }
            )
    return pd.DataFrame(rows)


def _fits() -> pd.DataFrame:
    rows: list[dict] = []
    for hour in HOURS:
        for site, estimate in (("site1", 0.58), ("site2", 0.52)):
            rows.extend(
                [
                    {
                        "metric": "msd",
                        "site": site,
                        "hour_post_delivery": hour,
                        "estimate_name": "exponent",
                        "estimate": estimate,
                        "ci_low": estimate - 0.07,
                        "ci_high": estimate + 0.07,
                        "fit_status": "ok",
                    },
                    {
                        "metric": "vac",
                        "site": site,
                        "hour_post_delivery": hour,
                        "estimate_name": "exponent",
                        "estimate": estimate,
                        "ci_low": estimate - 0.08,
                        "ci_high": estimate + 0.08,
                        "fit_status": "ok",
                    },
                ]
            )
        rows.append(
            {
                "metric": "mscd",
                "site": "pair",
                "hour_post_delivery": hour,
                "estimate_name": "exponent",
                "estimate": 0.64,
                "ci_low": 0.55,
                "ci_high": 0.73,
                "fit_status": "ok",
            }
        )
        for endpoint, estimate in (
            ("zero_lag_trace", 0.25),
            ("positive_short_lag_area", 0.17),
            ("zero_crossing_tau_over_delta", 2.0),
        ):
            rows.append(
                {
                    "metric": "vcc",
                    "site": "pair",
                    "hour_post_delivery": hour,
                    "estimate_name": endpoint,
                    "estimate": estimate,
                    "ci_low": estimate - 0.04,
                    "ci_high": estimate + 0.04,
                    "fit_status": "descriptive",
                }
            )
    return pd.DataFrame(rows)


def _predictions() -> pd.DataFrame:
    rows: list[dict] = []
    for hour in HOURS:
        for site in ("site1", "site2"):
            for x in np.linspace(0.0, 2.0, 12):
                rows.append(
                    {
                        "metric": "vac",
                        "site": site,
                        "hour_post_delivery": hour,
                        "x": x,
                        "y": np.exp(-x) * np.cos(np.pi * x / 2.0),
                    }
                )
        for x in np.linspace(0.0, 2.0, 12):
            rows.append(
                {
                    "metric": "vcc",
                    "site": "pair",
                    "hour_post_delivery": hour,
                    "x": x,
                    "y": 0.25 * np.exp(-1.5 * x),
                }
            )
        for metric, sites, exponent in (
            ("msd", ("site1", "site2"), 0.55),
            ("mscd", ("pair",), 0.62),
        ):
            for site in sites:
                for x in np.geomspace(5.0, 80.0, 12):
                    rows.append(
                        {
                            "metric": metric,
                            "site": site,
                            "hour_post_delivery": hour,
                            "x": x,
                            "y": 0.0017 * x**exponent,
                        }
                    )
    return pd.DataFrame(rows)


def test_writer_runs_all_separate_figures_and_emits_nonempty_formats(tmp_path: Path) -> None:
    paths = write_time_resolved_physics_figures(
        hour_curves=_curves(),
        fits=_fits(),
        predicted_curves=_predictions(),
        output_dir=tmp_path,
    )

    expected_groups = {
        "msd_curves",
        "msd_alpha",
        "mscd_curves",
        "mscd_beta",
        "vac_curves_site1",
        "vac_curves_site2",
        "vac_alpha",
        "vcc_curves",
        "vcc_endpoints",
    }
    assert expected_groups.issubset({key.rsplit("_", 1)[0] for key in paths})
    assert len(paths) == len(expected_groups) * 3
    assert {path.suffix for path in paths.values()} == {".png", ".pdf", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 500 for path in paths.values())


def test_missing_five_hour_vac_is_an_explicit_empty_panel_not_a_failure(tmp_path: Path) -> None:
    outputs = build_vac_curves_figures(_curves(omit_five_hour_vac=True), tmp_path)
    assert set(outputs) == {"site1", "site2"}
    assert all(path.is_file() and path.stat().st_size > 500 for paths in outputs.values() for path in paths)


def test_curve_schema_is_validated_before_plotting(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="total_pairs"):
        build_msd_curves_figure(_curves().drop(columns="total_pairs"), tmp_path / "invalid")
