from __future__ import annotations

import numpy as np
import pandas as pd

from dsb_states.v521_pair_galleries import build_pair_gallery_result


def test_frozen_t3_t4_rules_and_radius_order() -> None:
    rows = []
    bundles = []
    for bundle_id, frames, amplitude in (("t3", 10, 1.0), ("t4", 20, 2.0), ("out", 9, 0.5)):
        for frame in range(20):
            observed = frame < frames
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "frame": frame,
                    "time_s": float(frame),
                    "site1_position_observed": observed,
                    "site2_position_observed": observed,
                    "site1_x_um": amplitude * frame / max(frames, 1) if observed else np.nan,
                    "site1_y_um": 0.0 if observed else np.nan,
                    "site2_x_um": amplitude * frame / max(frames, 1) if observed else np.nan,
                    "site2_y_um": 0.2 if observed else np.nan,
                }
            )
        bundles.append(
            {
                "bundle_id": bundle_id,
                "nd2_id": "a",
                "crop_id": bundle_id,
                "fov_id": "f",
                "hour_post_delivery": 2.0,
                "allele_index": 1,
                "movie_frames": 20,
            }
        )
    result = build_pair_gallery_result(pd.DataFrame(rows), pd.DataFrame(bundles))
    membership = result.membership.set_index("bundle_id")
    assert membership.loc["t3", "t3_included"]
    assert not membership.loc["t3", "t4_included"]
    assert membership.loc["t4", "t4_included"]
    assert not membership.loc["out", "t3_included"]
    assert result.t4_order.bundle_id.tolist() == ["t4"]
    assert result.t3_order.paired_site_rg_rms_um.is_monotonic_increasing
    assert result.contract["53bp1_filter"] == "none"
