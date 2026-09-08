import numpy as np
import pandas as pd

from dsb_states.bp1 import (
    component_quantification_gate,
    quantify_local_component,
    summarize_point_bp1,
)


def test_point_bp1_geometry_and_persistence() -> None:
    frame = pd.DataFrame(
        {
            "frame": [1, 2, 3, 4],
            "site1_x_nm": [0, 0, 0, 0],
            "site1_y_nm": [0, 0, 0, 0],
            "site1_valid": [1, 1, 1, 1],
            "site2_x_nm": [2, 2, 2, 2],
            "site2_y_nm": [0, 0, 0, 0],
            "site2_valid": [1, 1, 1, 1],
            "bp1_x_nm": [1, 1, np.nan, 20],
            "bp1_y_nm": [0, 0, np.nan, 0],
            "bp1_valid": [1, 1, 0, 1],
        }
    )
    result = summarize_point_bp1(frame, colocalization_radius_nm=3)
    assert result.bp1_present_bundle
    # Without explicit status metadata, only detected points are known to be
    # evaluable. The nominal tracking density remains explicitly technical.
    assert result.bp1_detection_fraction == 1.0
    assert result.bp1_detection_fraction_nominal_technical == 0.75
    assert result.bp1_colocalized_fraction == 2 / 3
    assert result.bp1_colocalized_fraction_detected_paired == 2 / 3
    assert result.longest_bp1_run == 2
    assert result.longest_colocalized_run == 2


def test_component_gate_is_explicit() -> None:
    assert component_quantification_gate()["status"] == "gated"


def test_synthetic_component_quantification() -> None:
    image = np.full((31, 31), 10.0)
    yy, xx = np.ogrid[:31, :31]
    component = (xx - 15) ** 2 + (yy - 14) ** 2 <= 3**2
    image[component] = 30.0
    result = quantify_local_component(
        image,
        pair_proxy_x_px=15,
        pair_proxy_y_px=14,
        pixel_size_nm=100,
        threshold_k=0.5,
    )
    assert result is not None
    assert result.area_px2 == int(component.sum())
    assert result.pair_proxy_inside_component
    assert result.centroid_distance_to_pair_nm < 1e-6
    assert result.background_corrected_intensity > 0
