from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dsb_states.oligo_vac import _fit_alpha, _save_figure, compute_oligo_vac
from dsb_states.time_resolved_physics_summary import fbm_normalized_vac


def _synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(5)
    n_frames = 120
    times = np.arange(n_frames, dtype=float)
    steps = rng.normal(size=(n_frames - 1, 2))
    coordinates = np.vstack((np.zeros((1, 2)), np.cumsum(steps, axis=0)))
    frames = pd.DataFrame(
        {
            "bundle_id": "bundle",
            "nd2_id": "acq",
            "frame": np.arange(1, n_frames + 1),
            "time_s": times,
            "site1_x_um": coordinates[:, 0],
            "site1_y_um": coordinates[:, 1],
            "site2_x_um": coordinates[:, 0] + 0.5,
            "site2_y_um": coordinates[:, 1] - 0.5,
        }
    )
    index = pd.DataFrame(
        {
            "bundle_id": ["bundle"],
            "cohort": ["test"],
            "nd2_id": ["acq"],
            "crop_id": ["crop"],
            "fov_id": ["fov"],
            "hour_post_delivery": [3.0],
            "movie_frames": [n_frames],
        }
    )
    return frames, index


def test_eq9_contract_and_alpha_recovery() -> None:
    x = np.linspace(0.0, 2.5, 11)
    alpha = 0.55
    y = fbm_normalized_vac(x, alpha)
    assert np.isclose(y[0], 1.0)
    assert y[np.argmin(np.abs(x - 1.0))] < 0.0
    fitted = _fit_alpha(x, y, np.ones_like(x), bounds=(0.25, 1.0))
    assert fitted is not None
    assert np.isclose(fitted[0], alpha, atol=1e-5)


def test_multi_delta_vac_uses_only_metric_support() -> None:
    frames, index = _synthetic_tables()
    result = compute_oligo_vac(
        frames,
        index,
        target_delta_s=(5.0, 10.0, 20.0),
        minimum_velocity_pairs=8,
        bootstrap_iterations=0,
    )
    assert set(result.unit_curves["site"]) == {"site1", "site2"}
    assert set(result.unit_curves["target_delta_s"]) == {5.0, 10.0, 20.0}
    zero = result.unit_curves.loc[result.unit_curves["lag_frames"].eq(0)]
    np.testing.assert_allclose(zero["normalized_vac"], 1.0)
    assert result.support_census["unit_included"].all()


def test_save_figure_preserves_decimal_hour_stem(tmp_path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0])
    stem = tmp_path / "vac_1.5h_oligo_style"
    outputs = _save_figure(figure, stem, dpi=72)
    plt.close(figure)
    assert {path.name for path in outputs} == {
        "vac_1.5h_oligo_style.png",
        "vac_1.5h_oligo_style.pdf",
        "vac_1.5h_oligo_style.svg",
    }
    assert all(path.is_file() for path in outputs)
