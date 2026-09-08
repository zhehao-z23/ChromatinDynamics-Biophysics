from pathlib import Path

import pandas as pd

from dsb_states.plots import build_analysis_design_figure, build_qc_motion_dropout_figure


def test_design_figure_writes_all_formats(tmp_path: Path) -> None:
    outputs = build_analysis_design_figure(tmp_path / "design")
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg"}
    assert all(path.stat().st_size > 0 for path in outputs)


def test_motion_dropout_figure_writes_supported_acquisition_estimates(tmp_path: Path) -> None:
    motion = pd.DataFrame(
        {
            "motion_metric": ["site1", "site1", "site2", "common_mode"],
            "spearman_rho": [-0.4, -0.2, 0.1, 0.3],
            "supported": [True] * 4,
            "acquisition_id": ["a1", "a2", "a1", "a1"],
        }
    )
    dropout = pd.DataFrame(
        {
            "analysis": [
                "next_frame_channel_loss_after_high_speed",
                "next_frame_channel_loss_after_high_speed",
                "next_frame_pair_loss_after_large_separation",
            ],
            "channel": ["site1", "site2", "paired"],
            "cell_mean_risk_difference": [0.08, -0.03, 0.12],
            "supported": [True] * 3,
            "acquisition_id": ["a1", "a1", "a1"],
        }
    )
    outputs = build_qc_motion_dropout_figure(motion, dropout, tmp_path / "motion_dropout")
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg"}
    assert all(path.stat().st_size > 0 for path in outputs)
