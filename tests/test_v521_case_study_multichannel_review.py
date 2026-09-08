from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dsb_states.v521_case_study_multichannel_review import (
    CaseData,
    CropBounds,
    _hour_slug,
    _limits,
    _normalize,
    _reference_axis_limit_nm,
    _reference_shifted_nm,
    _shared_reference_axis_limit_nm,
    _shared_trace_limit_nm,
    _validated_nucleus_mask,
    make_distance_demo_variant_figure,
)


def _synthetic_case() -> CaseData:
    site1 = pd.DataFrame(
        {
            "frame": [1, 2],
            "x_nm": [100.0, 120.0],
            "y_nm": [200.0, 210.0],
            "x_px": [0.0, 0.2],
            "y_px": [1.0, 1.1],
        }
    )
    site2 = pd.DataFrame(
        {
            "frame": [1, 2],
            "x_nm": [300.0, 320.0],
            "y_nm": [400.0, 410.0],
            "x_px": [2.0, 2.2],
            "y_px": [3.0, 3.1],
        }
    )
    stack = np.ones((2, 5, 5), dtype=float)
    mask = np.zeros((2, 5, 5), dtype=bool)
    mask[:, 1:4, 1:4] = True
    return CaseData(
        case_id="case",
        display_label="Case",
        bundle_id="bundle",
        nd2_id="nd2",
        crop_id="crop",
        allele_index=1,
        hour_post_delivery=2.0,
        mean_separation_nm=1.0,
        sd_separation_nm=1.0,
        pixel_size_nm=100.0,
        times_s=np.array([0.0, 1.0]),
        site1_stack=stack,
        site2_stack=stack,
        nucleus_mask_stack=mask,
        nucleus_mask_path=Path("mask.tif"),
        site1_track=site1,
        site2_track=site2,
        raw_bounds=CropBounds(0, 5, 0, 5),
        site1_limits=(0.0, 2.0),
        site2_limits=(0.0, 2.0),
        spatial_limits_um=(-1.0, 1.0, -1.0, 1.0),
        trace_limit_nm=100.0,
        source_paths=(),
    )


def test_hour_normalization_and_reference_shift_contract() -> None:
    assert _hour_slug(2.0) == "2h"
    assert _hour_slug(1.5) == "1p5h"
    stack = np.arange(100, dtype=float).reshape(2, 5, 10)
    limits = _limits(stack, (5.0, 95.0))
    image = _normalize(stack[0], limits, 1.0)
    assert limits[0] < limits[1]
    assert np.all((image >= 0) & (image <= 1))
    case = _synthetic_case()
    x1, y1 = _reference_shifted_nm(case, case.site1_track)
    x2, y2 = _reference_shifted_nm(case, case.site2_track)
    assert np.isclose(np.min(np.concatenate([x1, x2])), 50.0)
    assert np.isclose(np.min(np.concatenate([y1, y2])), 50.0)
    assert _reference_axis_limit_nm(case) == 600.0
    assert _shared_reference_axis_limit_nm([case, case]) == 600.0
    assert _shared_trace_limit_nm([case, case]) == 100.0
    np.testing.assert_allclose(x2 - x1, [200.0, 200.0])
    np.testing.assert_allclose(y2 - y1, [200.0, 200.0])


def test_distance_demo_variants_have_expected_panel_counts() -> None:
    case = _synthetic_case()
    case.site1_stack[:] = np.arange(50, dtype=float).reshape(2, 5, 5)
    display = {
        "site_percentiles": [0.0, 100.0],
        "bp1_percentiles": [0.0, 100.0],
        "site_gamma": 1.0,
        "bp1_gamma": 1.0,
        "site1_color": "#F4B400",
        "site2_color": "#7A3DB8",
        "bp1_color": "#00B050",
        "site1_weight": 0.85,
        "site2_weight": 0.85,
        "bp1_weight": 0.60,
        "track_linewidth": 2.4,
        "track_halo_linewidth": 4.8,
        "pixel_scale_bar_linewidth": 5.0,
        "pixel_scale_bar_halo_linewidth": 8.5,
        "pixel_scale_bar_fontsize": 18.0,
        "pixel_label_halo_linewidth": 3.2,
        "full_cell_time_fontsize": 18.0,
        "time_colormap": "jet",
        "figure_size_inches": [15.8, 7.6],
    }
    compact = make_distance_demo_variant_figure(
        case,
        case.site1_stack,
        display,
        variant="compact_no_xy",
        shared_axis_limit_nm=600.0,
    )
    common = make_distance_demo_variant_figure(
        case,
        case.site1_stack,
        display,
        variant="shared_joint_origin_xy",
        shared_axis_limit_nm=600.0,
    )
    assert len(compact.axes) == 5
    assert len(common.axes) == 7
    compact_labels = [text.get_text() for axis in compact.axes for text in axis.texts]
    assert "1 µm" in compact_labels
    assert compact_labels.count("0.5 µm") == 2
    assert "t = 1.0 s\nframe 2/2" in compact_labels


def test_nucleus_mask_validation_rejects_scaffolds() -> None:
    valid = np.zeros((2, 5, 5), dtype=np.uint8)
    valid[:, 1:4, 1:4] = 255
    checked = _validated_nucleus_mask(valid, valid.shape)
    assert checked.dtype == np.bool_
    assert int(checked.sum()) == 18
    with np.testing.assert_raises(ValueError):
        _validated_nucleus_mask(np.ones((2, 5, 5), dtype=np.uint8), valid.shape)
