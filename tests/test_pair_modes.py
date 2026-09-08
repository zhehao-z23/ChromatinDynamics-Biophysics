from __future__ import annotations

import numpy as np

from dsb_states.pair_modes import common_differential_energy, decompose_pair_modes


def test_identical_pair_has_zero_relative_mode_and_differential_energy() -> None:
    trajectory = np.array(
        [
            [0.0, 0.0],
            [1.0, -0.5],
            [1.5, 2.0],
            [3.0, 2.5],
        ]
    )
    modes = decompose_pair_modes(trajectory, trajectory)

    np.testing.assert_allclose(modes.common, trajectory)
    np.testing.assert_allclose(modes.relative, 0.0)
    np.testing.assert_allclose(modes.separation, 0.0)
    assert modes.n_valid == trajectory.shape[0]

    steps = np.diff(trajectory, axis=0)
    energy = common_differential_energy(steps, steps)
    np.testing.assert_allclose(energy.differential, 0.0)
    np.testing.assert_allclose(energy.coherent_fraction, 1.0)


def test_opposite_steps_have_only_differential_energy() -> None:
    site1_steps = np.array([[1.0, 2.0], [-3.0, 0.5], [0.25, -2.0]])
    energy = common_differential_energy(site1_steps, -site1_steps)

    np.testing.assert_allclose(energy.common, 0.0)
    np.testing.assert_allclose(energy.coherent_fraction, 0.0)
    np.testing.assert_allclose(energy.identity_residual, 0.0, atol=1e-14)


def test_common_differential_parallelogram_identity_for_general_motion() -> None:
    generator = np.random.default_rng(1047)
    site1_steps = generator.normal(size=(1000, 3))
    site2_steps = generator.normal(size=(1000, 3))
    energy = common_differential_energy(site1_steps, site2_steps)

    expected_total = np.sum(site1_steps**2, axis=1) + np.sum(site2_steps**2, axis=1)
    np.testing.assert_allclose(energy.common + energy.differential, expected_total)
    np.testing.assert_allclose(energy.total, expected_total)
    np.testing.assert_allclose(energy.identity_residual, 0.0, atol=5e-14)
    assert np.all((energy.coherent_fraction >= 0.0) & (energy.coherent_fraction <= 1.0))


def test_pair_modes_retain_missing_rows_without_partial_coordinate_use() -> None:
    site1 = np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 2.0]])
    site2 = np.array([[2.0, 0.0], [3.0, 1.0], [4.0, 2.0]])
    modes = decompose_pair_modes(site1, site2)

    np.testing.assert_array_equal(modes.valid, [True, False, True])
    assert np.all(np.isnan(modes.common[1]))
    assert np.all(np.isnan(modes.relative[1]))
    assert np.isnan(modes.separation[1])
    np.testing.assert_allclose(modes.common[[0, 2]], [[1.0, 0.0], [3.0, 2.0]])


def test_zero_motion_has_undefined_coherent_fraction() -> None:
    zeros = np.zeros((4, 2))
    energy = common_differential_energy(zeros, zeros)

    assert np.all(np.isnan(energy.coherent_fraction))
    assert np.isnan(energy.aggregate_coherent_fraction)
