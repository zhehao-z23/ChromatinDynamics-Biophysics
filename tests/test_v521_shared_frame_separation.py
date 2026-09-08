from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsb_states.v521_separation_final_ppt import (
    build_final_separation_ppt_figure,
    build_hour_mean_statistics,
)
from dsb_states.v521_shared_frame_separation import (
    build_shared_frame_separation,
    build_shared_frame_separation_by_hour_figure,
    build_shared_frame_separation_figure,
    build_shared_frame_separation_ppt_figure,
)


def _frames() -> pd.DataFrame:
    rows = []
    for index, separation in enumerate((100.0, 250.0, 499.0, 501.0, 750.0, 1200.0)):
        rows.append(
            {
                "bundle_id": f"b{index // 2}",
                "nd2_id": "a1",
                "crop_id": f"c{index // 2}",
                "hour_post_delivery": 2.0,
                "frame": index + 1,
                "site1_valid": True,
                "site2_valid": True,
                "site1_x_um": 0.0,
                "site1_y_um": 0.0,
                "site2_x_um": separation / 1000.0,
                "site2_y_um": 0.0,
                "site1_site2_separation_nm": separation,
            }
        )
    rows.append(
        {
            "bundle_id": "missing",
            "nd2_id": "a1",
            "crop_id": "missing",
            "hour_post_delivery": 2.0,
            "frame": 1,
            "site1_valid": True,
            "site2_valid": False,
            "site1_x_um": 0.0,
            "site1_y_um": 0.0,
            "site2_x_um": np.nan,
            "site2_y_um": np.nan,
            "site1_site2_separation_nm": np.nan,
        }
    )
    return pd.DataFrame(rows)


def test_shared_frame_histogram_retains_all_and_marks_threshold() -> None:
    result = build_shared_frame_separation(_frames())
    summary = result.summary.iloc[0]
    assert len(result.frame_distances) == 6
    assert result.histogram["frame_count"].sum() == 6
    assert result.histogram["frame_probability"].sum() == pytest.approx(1.0)
    assert summary.n_above_threshold == 3
    assert summary.percent_above_threshold == 50.0
    assert set(result.histogram["threshold_group"]) == {"≤500 nm", ">500 nm"}
    assert result.hour_histogram["frame_count"].sum() == 6
    assert result.hour_summary.iloc[0].percent_above_threshold == 50.0
    assert result.method_contract["scientific_filters"] == []


def test_shared_frame_histogram_writes_three_figure_formats(tmp_path: Path) -> None:
    result = build_shared_frame_separation(_frames())
    outputs = build_shared_frame_separation_figure(result, tmp_path / "separation")
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 500 for path in outputs)
    hour_outputs = build_shared_frame_separation_by_hour_figure(result, tmp_path / "by_hour")
    assert {path.suffix for path in hour_outputs} == {".png", ".pdf", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 500 for path in hour_outputs)


def test_slide_figure_writes_five_histograms_and_complete_trend(tmp_path: Path) -> None:
    frames = pd.concat(
        [
            _frames().assign(hour_post_delivery=hour)
            for hour in (1.5, 2.5, 3.5, 4.5, 10.0)
        ],
        ignore_index=True,
    )
    frames["bundle_id"] = frames["bundle_id"].astype(str) + "_" + frames[
        "hour_post_delivery"
    ].astype(str)
    result = build_shared_frame_separation(frames, threshold_nm=550.0)
    outputs = build_shared_frame_separation_ppt_figure(result, tmp_path / "slide")
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 500 for path in outputs)


def test_final_ppt_figure_writes_two_histograms_and_hour_statistics(tmp_path: Path) -> None:
    hours = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 10.0)
    rows = []
    for hour_index, hour in enumerate(hours):
        for crop_index in range(3):
            for frame_index in range(5):
                separation = 180.0 + 30.0 * hour_index + 9.0 * crop_index + frame_index
                rows.append(
                    {
                        "bundle_id": f"b_{hour_index}_{crop_index}",
                        "nd2_id": f"a_{hour_index}_{crop_index}",
                        "crop_id": f"c_{crop_index}",
                        "hour_post_delivery": hour,
                        "frame": frame_index + 1,
                        "site1_valid": True,
                        "site2_valid": True,
                        "site1_x_um": 0.0,
                        "site1_y_um": 0.0,
                        "site2_x_um": separation / 1000.0,
                        "site2_y_um": 0.0,
                        "site1_site2_separation_nm": separation,
                    }
                )
    result = build_shared_frame_separation(pd.DataFrame(rows), threshold_nm=550.0)
    statistics = build_hour_mean_statistics(
        result.frame_distances,
        display_hours=hours,
    )
    assert statistics["hour_post_delivery"].tolist() == list(hours)
    assert statistics.loc[0, "significance"] == "ref"
    assert statistics.loc[1:, "p_adjusted"].between(0.0, 1.0).all()
    assert statistics["mean_cluster_robust_se_nm"].gt(0.0).all()
    assert set(statistics["multiple_testing"]) == {"holm"}
    unadjusted = build_hour_mean_statistics(
        result.frame_distances,
        display_hours=hours,
        multiple_testing="none",
    )
    bonferroni = build_hour_mean_statistics(
        result.frame_distances,
        display_hours=hours,
        multiple_testing="bonferroni",
    )
    np.testing.assert_allclose(
        unadjusted.loc[1:, "p_adjusted"],
        unadjusted.loc[1:, "p_raw"],
    )
    np.testing.assert_allclose(
        bonferroni.loc[1:, "p_adjusted"],
        np.minimum(bonferroni.loc[1:, "p_raw"].to_numpy(float) * 7.0, 1.0),
    )

    outputs = build_final_separation_ppt_figure(
        result,
        statistics,
        tmp_path / "final_ppt",
    )
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 500 for path in outputs)
