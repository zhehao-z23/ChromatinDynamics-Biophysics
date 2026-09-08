from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DropoutStressResult:
    requested_dropout_fraction: float
    observed_dropout_fraction: float
    original_valid_count: int
    retained_valid_count: int
    dropped_frames: np.ndarray
    positions_with_dropout: np.ndarray


def channel_missingness_table(frame_table: pd.DataFrame) -> pd.DataFrame:
    """Return one row per frame and channel without treating frames as replicates."""

    required = {"bundle_id", "frame", "site1_valid", "site2_valid", "bp1_valid"}
    missing = required.difference(frame_table.columns)
    if missing:
        raise ValueError(f"Missing columns for missingness audit: {sorted(missing)}")
    id_columns = [
        column
        for column in (
            "biological_experiment_id",
            "acquisition_id",
            "fov_id",
            "cell_id",
            "bundle_id",
            "hour_post_delivery",
            "frame",
            "micro_time_s",
        )
        if column in frame_table.columns
    ]
    rows = []
    for channel in ("site1", "site2", "bp1"):
        block = frame_table[id_columns].copy()
        block["channel"] = channel
        block["observed"] = frame_table[f"{channel}_valid"].astype(bool).to_numpy()
        block["missing"] = ~block["observed"]
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def bundle_dropout_asymmetry(bundle_table: pd.DataFrame) -> pd.DataFrame:
    """Quantify neutral Site1/Site2 coverage asymmetry by bundle."""

    result = bundle_table.copy()
    if {"site1_coverage", "site2_coverage"}.issubset(result.columns):
        first, second = "site1_coverage", "site2_coverage"
    elif {"left_coverage", "right_coverage"}.issubset(result.columns):
        first, second = "left_coverage", "right_coverage"
    else:
        raise ValueError("Bundle table lacks neutral or mapped pair coverage columns")
    result["site_coverage_difference"] = result[first] - result[second]
    result["site_coverage_absolute_difference"] = result["site_coverage_difference"].abs()
    return result


def apply_empirical_dropout(
    positions: np.ndarray,
    valid: np.ndarray,
    *,
    dropout_fraction: float,
    random_state: int | np.random.Generator = 0,
    eligible: np.ndarray | None = None,
) -> DropoutStressResult:
    """Remove observed frames without filling them for sensitivity experiments."""

    coordinates = np.asarray(positions, dtype=float)
    valid_mask = np.asarray(valid, dtype=bool)
    if coordinates.ndim != 2 or coordinates.shape[0] != valid_mask.size:
        raise ValueError("positions and valid mask must align")
    if not 0 <= dropout_fraction < 1:
        raise ValueError("dropout_fraction must lie in [0, 1)")
    selectable = valid_mask.copy()
    if eligible is not None:
        eligible_mask = np.asarray(eligible, dtype=bool)
        if eligible_mask.shape != valid_mask.shape:
            raise ValueError("eligible mask must align with valid")
        selectable &= eligible_mask
    candidates = np.flatnonzero(selectable)
    n_drop = int(np.floor(dropout_fraction * candidates.size))
    generator = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    dropped = (
        np.sort(generator.choice(candidates, size=n_drop, replace=False))
        if n_drop
        else np.array([], dtype=int)
    )
    stressed = coordinates.copy()
    stressed[dropped] = np.nan
    retained = valid_mask.copy()
    retained[dropped] = False
    original_count = int(valid_mask.sum())
    retained_count = int(retained.sum())
    return DropoutStressResult(
        requested_dropout_fraction=float(dropout_fraction),
        observed_dropout_fraction=(
            (original_count - retained_count) / original_count if original_count else np.nan
        ),
        original_valid_count=original_count,
        retained_valid_count=retained_count,
        dropped_frames=dropped,
        positions_with_dropout=stressed,
    )


def missingness_model_gate(frame_table: pd.DataFrame) -> dict:
    """Report which prespecified A3 predictors are actually available."""

    requested = {
        "hour": "hour_post_delivery",
        "53bp1_intensity": "bp1_intensity_bg_corrected",
        "site_intensity": "median_site_intensity",
        "fov": "fov_id",
        "acquisition": "acquisition_id",
    }
    availability = {name: column in frame_table.columns for name, column in requested.items()}
    return {
        "requested_predictors": requested,
        "availability": availability,
        "status": "ready" if all(availability.values()) else "partial",
        "rule": "Unavailable predictors are reported, never synthesized or silently imputed.",
    }
