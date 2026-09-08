import numpy as np
import pandas as pd

from dsb_states.missingness import (
    apply_empirical_dropout,
    bundle_dropout_asymmetry,
    channel_missingness_table,
    missingness_model_gate,
)


def test_missingness_long_table_and_asymmetry() -> None:
    frames = pd.DataFrame(
        {
            "bundle_id": ["b", "b"],
            "frame": [1, 2],
            "site1_valid": [1, 0],
            "site2_valid": [1, 1],
            "bp1_valid": [0, 1],
        }
    )
    long = channel_missingness_table(frames)
    assert len(long) == 6
    bundle = bundle_dropout_asymmetry(
        pd.DataFrame({"site1_coverage": [0.5], "site2_coverage": [1.0]})
    )
    assert bundle["site_coverage_difference"].iloc[0] == -0.5


def test_empirical_dropout_is_reproducible_and_no_interpolation() -> None:
    positions = np.column_stack([np.arange(10), np.zeros(10)])
    result = apply_empirical_dropout(
        positions, np.ones(10, dtype=bool), dropout_fraction=0.3, random_state=7
    )
    repeated = apply_empirical_dropout(
        positions, np.ones(10, dtype=bool), dropout_fraction=0.3, random_state=7
    )
    assert np.array_equal(result.dropped_frames, repeated.dropped_frames)
    assert result.retained_valid_count == 7
    assert np.isnan(result.positions_with_dropout[result.dropped_frames]).all()


def test_missingness_gate_reports_absent_intensity() -> None:
    report = missingness_model_gate(pd.DataFrame({"acquisition_id": ["a"]}))
    assert report["status"] == "partial"
    assert not report["availability"]["53bp1_intensity"]
