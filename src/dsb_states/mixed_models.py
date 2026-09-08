from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class ClusterBootstrapEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    cluster_count: int
    iterations: int


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1) | ~np.isfinite(values)):
        raise ValueError("p-values must be a finite vector in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * values.size / np.arange(1, values.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def cluster_bootstrap(
    table: pd.DataFrame,
    *,
    cluster_column: str,
    statistic: Callable[[pd.DataFrame], float],
    iterations: int = 2000,
    confidence: float = 0.95,
    random_state: int = 20260811,
) -> ClusterBootstrapEstimate:
    """Resample independent clusters and retain all rows within sampled clusters."""

    if cluster_column not in table:
        raise ValueError(f"Missing cluster column: {cluster_column}")
    if iterations < 100:
        raise ValueError("Use at least 100 cluster-bootstrap iterations")
    clusters = pd.unique(table[cluster_column].dropna())
    if len(clusters) < 2:
        raise ValueError("Cluster bootstrap requires at least two clusters")
    estimate = float(statistic(table))
    generator = np.random.default_rng(random_state)
    samples = np.full(iterations, np.nan)
    grouped = {cluster: block for cluster, block in table.groupby(cluster_column, sort=False)}
    for iteration in range(iterations):
        selected = generator.choice(clusters, size=len(clusters), replace=True)
        resampled = pd.concat(
            [
                grouped[cluster].assign(_bootstrap_cluster=index)
                for index, cluster in enumerate(selected)
            ],
            ignore_index=True,
        )
        samples[iteration] = float(statistic(resampled))
    alpha = 1 - confidence
    low, high = np.nanquantile(samples, [alpha / 2, 1 - alpha / 2])
    return ClusterBootstrapEstimate(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        cluster_count=len(clusters),
        iterations=iterations,
    )


def normal_two_sided_pvalue(estimate: float, standard_error: float) -> float:
    if not np.isfinite(standard_error) or standard_error <= 0:
        return np.nan
    return float(2 * norm.sf(abs(estimate / standard_error)))


def macro_time_model_contract(table: pd.DataFrame) -> dict:
    """Check semantic prerequisites before a mixed macro-time analysis."""

    required = {"hour_post_delivery", "acquisition_id", "fov_id", "bundle_id"}
    missing = sorted(required.difference(table.columns))
    hour_missing = (
        True
        if "hour_post_delivery" not in table
        else bool(table["hour_post_delivery"].isna().any())
    )
    return {
        "status": "ready" if not missing and not hour_missing else "gated",
        "missing_columns": missing,
        "hour_has_missing_or_unmapped_values": hour_missing,
        "primary_hour_encoding": "categorical",
        "minimum_uncertainty_cluster": "acquisition_id/FOV",
        "causal_language": "association only",
    }
