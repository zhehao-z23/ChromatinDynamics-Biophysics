"""Coordinate and energy decompositions for paired trajectories.

The functions in this module deliberately use the neutral names ``site1`` and
``site2``.  The two tracked probes are local flanking-locus measurements; the
decomposition does not assign an orientation or claim that either probe is a
physical DNA end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _as_coordinates(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a copied ``(n_observations, n_dimensions)`` float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D coordinate array")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one coordinate dimension")
    return np.array(array, dtype=float, copy=True)


def _paired_coordinates(site1: ArrayLike, site2: ArrayLike) -> tuple[FloatArray, FloatArray]:
    first = _as_coordinates(site1, name="site1")
    second = _as_coordinates(site2, name="site2")
    if first.shape != second.shape:
        raise ValueError("site1 and site2 must have the same observation and coordinate dimensions")
    return first, second


@dataclass(frozen=True)
class PairModes:
    """Frame-aligned common and relative coordinates for a paired trajectory.

    A row is valid only when every coordinate is finite at both sites.  Invalid
    rows are retained in place and represented by ``NaN`` in all derived
    coordinates.  Retaining frame alignment makes missingness explicit and
    prevents accidental interpolation.
    """

    common: FloatArray
    relative: FloatArray
    separation: FloatArray
    valid: BoolArray

    @property
    def n_observations(self) -> int:
        return int(self.valid.size)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))


@dataclass(frozen=True)
class MotionEnergy:
    """Per-observation common/differential displacement-energy partition."""

    common: FloatArray
    differential: FloatArray
    site1: FloatArray
    site2: FloatArray
    total: FloatArray
    coherent_fraction: FloatArray
    identity_residual: FloatArray
    valid: BoolArray

    @property
    def aggregate_coherent_fraction(self) -> float:
        """Ratio of summed common energy to summed decomposed energy."""

        denominator = float(np.nansum(self.total))
        if not np.isfinite(denominator) or denominator <= 0.0:
            return float("nan")
        return float(np.nansum(self.common) / denominator)


def decompose_pair_modes(site1: ArrayLike, site2: ArrayLike) -> PairModes:
    """Decompose paired positions into common and relative coordinates.

    The definitions are

    ``common = (site1 + site2) / 2`` and ``relative = site2 - site1``.

    No rows are dropped or filled.  A frame with a non-finite coordinate at
    either site is invalid for every paired-mode quantity.
    """

    first, second = _paired_coordinates(site1, site2)
    valid = np.all(np.isfinite(first), axis=1) & np.all(np.isfinite(second), axis=1)

    common = (first + second) / 2.0
    relative = second - first
    common[~valid] = np.nan
    relative[~valid] = np.nan
    separation = np.linalg.norm(relative, axis=1)
    separation[~valid] = np.nan

    return PairModes(
        common=common,
        relative=relative,
        separation=separation,
        valid=valid,
    )


def common_mode(site1: ArrayLike, site2: ArrayLike) -> FloatArray:
    """Return the paired common-mode (midpoint) trajectory."""

    return decompose_pair_modes(site1, site2).common


def relative_mode(site1: ArrayLike, site2: ArrayLike) -> FloatArray:
    """Return the relative vector ``site2 - site1`` at each frame."""

    return decompose_pair_modes(site1, site2).relative


def pair_separation(site1: ArrayLike, site2: ArrayLike) -> FloatArray:
    """Return the Euclidean site1--site2 separation at each frame."""

    return decompose_pair_modes(site1, site2).separation


def common_differential_energy(
    displacement_site1: ArrayLike,
    displacement_site2: ArrayLike,
) -> MotionEnergy:
    """Partition paired displacement energy into common and differential modes.

    For displacement vectors ``a`` and ``b`` the per-row definitions are

    ``E_common = 0.5 * ||a + b||**2``

    ``E_differential = 0.5 * ||a - b||**2``.

    Consequently, the parallelogram identity gives
    ``E_common + E_differential = ||a||**2 + ||b||**2``.  The returned
    ``identity_residual`` records the floating-point residual of that identity.
    ``coherent_fraction`` is undefined (``NaN``) when both displacements have
    zero total energy.
    """

    first, second = _paired_coordinates(displacement_site1, displacement_site2)
    valid = np.all(np.isfinite(first), axis=1) & np.all(np.isfinite(second), axis=1)

    common = 0.5 * np.einsum("ij,ij->i", first + second, first + second)
    differential = 0.5 * np.einsum("ij,ij->i", first - second, first - second)
    site1_energy = np.einsum("ij,ij->i", first, first)
    site2_energy = np.einsum("ij,ij->i", second, second)
    total = common + differential

    coherent_fraction = np.full(total.shape, np.nan, dtype=float)
    positive_total = valid & (total > 0.0)
    coherent_fraction[positive_total] = common[positive_total] / total[positive_total]

    identity_residual = total - (site1_energy + site2_energy)
    for output in (
        common,
        differential,
        site1_energy,
        site2_energy,
        total,
        identity_residual,
    ):
        output[~valid] = np.nan

    return MotionEnergy(
        common=common,
        differential=differential,
        site1=site1_energy,
        site2=site2_energy,
        total=total,
        coherent_fraction=coherent_fraction,
        identity_residual=identity_residual,
        valid=valid,
    )


# A short alias is useful in feature-building code while retaining the explicit
# public name above in reports and method documentation.
energy_partition = common_differential_energy
