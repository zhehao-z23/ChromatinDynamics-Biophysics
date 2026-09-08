from __future__ import annotations

import pandas as pd

from dsb_states.v521_case_selection import select_complete_pair_cases


def test_complete_pair_selection_uses_trajectory_mean_and_declared_coverage() -> None:
    index = pd.DataFrame(
        {
            "bundle_id": ["short", "long", "incomplete"],
            "nd2_id": ["a", "a", "a"],
            "crop_id": ["c1", "c2", "c3"],
            "fov_id": ["f", "f", "f"],
            "hour_post_delivery": [1.5, 2.5, 3.5],
            "allele_index": [1, 1, 1],
            "site1_points": [2, 2, 1],
            "site2_points": [2, 2, 2],
            "shared_site_frames": [2, 2, 1],
            "movie_frames": [2, 2, 2],
            "frame_interval_s_production": [1.0, 1.0, 1.0],
        }
    )
    frames = pd.DataFrame(
        {
            "bundle_id": ["short", "short", "long", "long", "incomplete", "incomplete"],
            "frame": [0, 1, 0, 1, 0, 1],
            "time_s": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "site1_valid": [True, True, True, True, True, False],
            "site2_valid": [True, True, True, True, True, True],
            "shared_site_frame": [True, True, True, True, True, False],
            "site1_x_nm": [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
            "site1_y_nm": [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
            "site2_x_nm": [10.0, 30.0, 100.0, 300.0, 20.0, 20.0],
            "site2_y_nm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "site1_site2_separation_nm": [10.0, 30.0, 100.0, 300.0, 20.0, float("nan")],
        }
    )
    result = select_complete_pair_cases(
        index, frames, n_extreme=1, display_min_frames=2
    )
    assert set(result.trajectory_summary["bundle_id"]) == {"short", "long"}
    assert result.global_candidates.loc[0, "bundle_id"] == "short"
    assert result.global_candidates.loc[1, "bundle_id"] == "long"
    assert result.trajectory_summary.set_index("bundle_id").loc["short", "mean_separation_nm"] == 20
