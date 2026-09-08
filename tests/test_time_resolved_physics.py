from __future__ import annotations

import numpy as np
import pandas as pd

from dsb_states.time_resolved_physics import compute_time_resolved_physics


def _synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_rows: list[dict[str, object]] = []

    def add_bundle(
        bundle_id: str,
        *,
        frames: range | list[int],
        site1: bool,
        site2: bool,
    ) -> None:
        for frame in frames:
            time_s = float(frame - 1)
            frame_rows.append(
                {
                    "bundle_id": bundle_id,
                    "nd2_id": "acq",
                    "frame": frame,
                    "time_s": time_s,
                    "site1_x_um": time_s if site1 else np.nan,
                    "site1_y_um": 0.0 if site1 else np.nan,
                    "site2_x_um": time_s if site2 else np.nan,
                    "site2_y_um": 0.0 if site2 else np.nan,
                }
            )

    # This bundle exposes the complete exact acquisition schedule and supports
    # every paired metric.
    add_bundle("complete", frames=range(1, 41), site1=True, site2=True)
    # Frame 2 is absent as a row, not merely set to NaN.  The computation must
    # align to the acquisition schedule rather than compressing frames 1 and 3.
    add_bundle("gap", frames=[1, *range(3, 41)], site1=True, site2=False)
    add_bundle("site1_only", frames=range(1, 41), site1=True, site2=False)

    index = pd.DataFrame(
        [
            {
                "bundle_id": bundle_id,
                "cohort": "synthetic",
                "nd2_id": "acq",
                "crop_id": f"crop_{bundle_id}",
                "fov_id": "fov",
                "hour_post_delivery": 2.0,
                "movie_frames": 40,
            }
            for bundle_id in ("complete", "gap", "site1_only")
        ]
    )
    return pd.DataFrame(frame_rows), index


def test_absent_frame_is_not_compressed_and_site_metrics_are_independent() -> None:
    frames, index = _synthetic_tables()
    result = compute_time_resolved_physics(frames, index)

    gap_msd = result.unit_curves.loc[
        result.unit_curves.metric.eq("msd")
        & result.unit_curves.bundle_id.eq("gap")
        & result.unit_curves.site.eq("site1")
        & result.unit_curves.lag_frames.eq(1)
    ].iloc[0]
    assert gap_msd.pair_count == 37
    assert gap_msd.value == 1.0
    assert gap_msd.lag_median_s == 1.0

    site1_only = result.unit_curves.loc[result.unit_curves.bundle_id.eq("site1_only")]
    assert not site1_only.loc[site1_only.site.eq("site1") & site1_only.metric.eq("msd")].empty
    assert not site1_only.loc[site1_only.site.eq("site1") & site1_only.metric.eq("vac")].empty
    assert site1_only.loc[site1_only.site.eq("site2")].empty
    assert site1_only.loc[site1_only.site.eq("pair")].empty

    excluded = result.support_census.loc[
        result.support_census.bundle_id.eq("site1_only")
        & result.support_census.metric.eq("mscd")
    ].iloc[0]
    assert not excluded.unit_included
    assert excluded.exclusion_reason == "insufficient_valid_pairs_all_lags"
    assert excluded.observed_site1_frames == 40
    assert excluded.observed_site2_frames == 0
    assert excluded.shared_site_frames == 0


def test_units_normalization_sd_matched_delta_and_bidirectional_vcc() -> None:
    frames, index = _synthetic_tables()
    result = compute_time_resolved_physics(frames, index)
    complete = result.unit_curves.loc[result.unit_curves.bundle_id.eq("complete")]

    msd = complete.loc[
        complete.metric.eq("msd") & complete.site.eq("site1") & complete.lag_frames.eq(1)
    ].iloc[0]
    assert msd.value == 1.0
    assert msd.raw_value == 1.0
    assert msd.endpoint_pair_sd == 0.0
    assert msd.value_unit == "um^2"
    assert msd.endpoint_pair_sd_unit == "um^2"

    mscd = complete.loc[complete.metric.eq("mscd") & complete.lag_frames.eq(1)].iloc[0]
    assert mscd.value == 0.0
    assert mscd.endpoint_pair_sd == 0.0
    assert mscd.value_unit == "um^2"

    primary_vac = complete.loc[
        complete.metric.eq("vac")
        & complete.site.eq("site1")
        & complete.is_primary_matched_10s
        & complete.lag_frames.eq(0)
    ]
    assert primary_vac.delta_frames.tolist() == [10]
    assert primary_vac.delta_role.tolist() == ["primary_matched_10s"]
    assert primary_vac.iloc[0].delta_median_s == 10.0
    assert primary_vac.iloc[0].value == 1.0
    assert primary_vac.iloc[0].value_unit == "dimensionless"
    assert primary_vac.iloc[0].raw_value_unit == "um^2/s^2"
    assert primary_vac.iloc[0].normalization == 1.0

    paper_deltas = set(
        complete.loc[
            complete.metric.eq("vac")
            & complete.site.eq("site1")
            & complete.delta_role.eq("paper_family"),
            "delta_frames",
        ].dropna()
    )
    assert paper_deltas == {1, 2, 4, 8}

    primary_vcc = complete.loc[
        complete.metric.eq("vcc")
        & complete.is_primary_matched_10s
        & complete.lag_frames.eq(0)
    ]
    assert set(primary_vcc.direction) == {
        "site1_later_vs_site2",
        "site2_later_vs_site1",
    }
    np.testing.assert_allclose(primary_vcc.value, 1.0)
    assert (primary_vcc.value_unit == "dimensionless").all()
    assert (primary_vcc.normalization == 1.0).all()

    matrices = result.vcc_matrices.loc[
        result.vcc_matrices.bundle_id.eq("complete")
        & result.vcc_matrices.is_primary_matched_10s
        & result.vcc_matrices.lag_frames.eq(0)
    ]
    assert len(matrices) == 8  # four matrix components for each direction
    assert set(matrices.direction) == set(primary_vcc.direction)
    assert set(zip(matrices.row_component, matrices.column_component, strict=True)) == {
        ("x", "x"),
        ("x", "y"),
        ("y", "x"),
        ("y", "y"),
    }
    xx = matrices.loc[
        matrices.row_component.eq("x") & matrices.column_component.eq("x")
    ]
    np.testing.assert_allclose(xx.normalized_value, 1.0)


def test_primary_matched_delta_can_overlap_paper_family_without_recalculation_rows() -> None:
    frames, index = _synthetic_tables()
    frames = frames.copy()
    frames["time_s"] *= 5.0
    result = compute_time_resolved_physics(frames, index)

    selected = result.unit_curves.loc[
        result.unit_curves.bundle_id.eq("complete")
        & result.unit_curves.metric.eq("vac")
        & result.unit_curves.site.eq("site1")
        & result.unit_curves.delta_frames.eq(2)
        & result.unit_curves.lag_frames.eq(0)
    ]
    assert len(selected) == 1
    assert selected.iloc[0].delta_role == "paper_family"
    assert selected.iloc[0].is_primary_matched_10s
    assert selected.iloc[0].delta_median_s == 10.0
