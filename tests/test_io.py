from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dsb_states.io import exact_micro_times, parse_trajectory_filename, read_trajectory_csv


def test_parse_formal_trajectory_filename() -> None:
    assert parse_trajectory_filename("allele_012_site1_longest_spt_cleaned.csv") == (12, "site1")
    assert parse_trajectory_filename("allele_002_53bp1_longest_spt_cleaned.csv") == (2, "53bp1")


def test_read_trajectory_contract(tmp_path: Path) -> None:
    path = tmp_path / "traj.csv"
    pd.DataFrame({"frame": [3, 1], "x_nm": [2, 1], "y_nm": [4, 3]}).to_csv(path, index=False)
    table = read_trajectory_csv(path)
    assert table["frame"].tolist() == [1, 3]
    assert table["x_nm"].dtype == np.float64


def test_exact_micro_times_does_not_replace_nonuniform_axis() -> None:
    metadata = {"time": {"n_frames": 3, "finterval_s": 1.0, "relative_time_s": [0, 1, 3]}}
    assert exact_micro_times(metadata).tolist() == [0.0, 1.0, 3.0]
