from __future__ import annotations

import numpy as np
import pandas as pd

from dsb_states.unfiltered_cache import build_unfiltered_cache


def test_unfiltered_cache_preserves_site_rows_and_no_association() -> None:
    trajectories = pd.DataFrame(
        [
            {"bundle_id": "b1", "frame": 0, "cohort": "c", "nd2_id": "a", "crop_id": "x", "fov_id": "f", "hour_post_delivery": 2.0, "allele_index": 1, "site_id": "site1", "x_nm": 0.0, "y_nm": 0.0, "time_s": 0.0, "uniform_time_s": 0.0},
            {"bundle_id": "b1", "frame": 0, "cohort": "c", "nd2_id": "a", "crop_id": "x", "fov_id": "f", "hour_post_delivery": 2.0, "allele_index": 1, "site_id": "site2", "x_nm": 300.0, "y_nm": 400.0, "time_s": 0.0, "uniform_time_s": 0.0},
            {"bundle_id": "b1", "frame": 1, "cohort": "c", "nd2_id": "a", "crop_id": "x", "fov_id": "f", "hour_post_delivery": 2.0, "allele_index": 1, "site_id": "site1", "x_nm": 10.0, "y_nm": 0.0, "time_s": 1.0, "uniform_time_s": 1.0},
        ]
    )
    locus = pd.DataFrame(
        [
            {"bundle_id": "b1", "frame": 0, "cohort": "c", "nd2_id": "a", "crop_id": "x", "fov_id": "f", "hour_post_delivery": 2.0, "allele_index": 1, "time_s": 0.0, "site1_valid": True, "site2_valid": True, "site1_x_nm": 0.0, "site1_y_nm": 0.0, "site2_x_nm": 300.0, "site2_y_nm": 400.0, "assignment_status": "NO_ASSOCIATED_FOCUS", "site1_site2_separation_nm": 500.0},
            {"bundle_id": "b1", "frame": 1, "cohort": "c", "nd2_id": "a", "crop_id": "x", "fov_id": "f", "hour_post_delivery": 2.0, "allele_index": 1, "time_s": 1.0, "site1_valid": True, "site2_valid": False, "site1_x_nm": 10.0, "site1_y_nm": 0.0, "site2_x_nm": np.nan, "site2_y_nm": np.nan, "assignment_status": "DNA_ANCHOR_MISSING", "site1_site2_separation_nm": np.nan},
        ]
    )
    allele = pd.DataFrame([{"bundle_id": "b1", "nd2_id": "a", "crop_id": "x"}])
    crop = pd.DataFrame([{"nd2_id": "a", "crop_id": "x", "movie_frames": 2, "exact_timing_points": 2, "frame_interval_s_production": 1.0, "task_status": "COMPLETE", "availability_class": "COMPLETE", "unavailable_reason": pd.NA}])
    result = build_unfiltered_cache(trajectories, locus, allele, crop)
    assert len(result.bundle_frames) == 2
    assert result.census["analysis_filtering_applied"] is False
    assert result.bundle_frames.site1_position_observed.sum() == 2
    assert result.bundle_frames.site2_position_observed.sum() == 1
    assert result.bundle_frames.shared_site_frame.sum() == 1
    assert result.bundle_frames.loc[0, "assignment_status"] == "NO_ASSOCIATED_FOCUS"
    assert result.bundle_frames.loc[0, "site1_site2_separation_nm"] == 500.0
