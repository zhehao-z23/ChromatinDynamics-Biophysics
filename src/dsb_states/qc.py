"""Initial cohort QC and missingness census for neutral Site1/Site2 tables."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

from .plots import write_initial_qc_figures
from .tables import validate_analysis_output_dir

QC_THRESHOLDS = (
    # Legacy left/right keys remain accepted only as config aliases. Outputs
    # and active configs use neutral Site1/Site2 terminology.
    (
        "min_site1_valid_frames",
        "min_site1_valid_frames",
        "min_left_valid_frames",
        "site1_valid_count",
    ),
    (
        "min_site2_valid_frames",
        "min_site2_valid_frames",
        "min_right_valid_frames",
        "site2_valid_count",
    ),
    ("min_paired_valid_frames", "min_paired_valid_frames", None, "paired_valid_count"),
    ("min_longest_paired_run", "min_longest_paired_run", None, "longest_paired_run"),
    ("min_paired_coverage", "min_paired_coverage", None, "paired_coverage"),
)


@dataclass(frozen=True)
class InitialQcResult:
    frame_table_qc: pd.DataFrame
    bundle_metadata_evaluated: pd.DataFrame
    bundle_metadata_qc: pd.DataFrame
    bundle_exclusions: pd.DataFrame
    cohort_flowchart: pd.DataFrame
    missingness_census: pd.DataFrame
    missingness_asymmetry: pd.DataFrame
    technical_hierarchy: pd.DataFrame
    timing_hierarchy: pd.DataFrame
    qc_sensitivity: pd.DataFrame
    coverage_associations: pd.DataFrame
    motion_coverage_bundle_summary: pd.DataFrame
    motion_coverage_associations: pd.DataFrame
    dropout_risk_contrasts: pd.DataFrame
    missingness_gates: pd.DataFrame
    summary: dict[str, Any]


def _resolved_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    qc = config.get("qc", {})
    missing = [
        primary_key
        for _, primary_key, legacy_key, _ in QC_THRESHOLDS
        if primary_key not in qc and (legacy_key is None or legacy_key not in qc)
    ]
    if missing:
        raise ValueError(f"QC configuration lacks thresholds: {missing}")
    thresholds = {
        reason: float(qc[primary_key] if primary_key in qc else qc[legacy_key])
        for reason, primary_key, legacy_key, _ in QC_THRESHOLDS
    }
    invalid: list[str] = []
    count_thresholds = {
        "min_site1_valid_frames",
        "min_site2_valid_frames",
        "min_paired_valid_frames",
        "min_longest_paired_run",
    }
    for name, value in thresholds.items():
        if not np.isfinite(value):
            invalid.append(f"{name}=nonfinite")
        elif name in count_thresholds and (value < 0 or not float(value).is_integer()):
            invalid.append(f"{name}=must be a nonnegative integer")
        elif name == "min_paired_coverage" and not 0 <= value <= 1:
            invalid.append(f"{name}=must lie in [0, 1]")
    if invalid:
        raise ValueError("Invalid QC thresholds: " + "; ".join(invalid))
    return thresholds


def _rule_failures(
    bundle_metadata: pd.DataFrame, thresholds: Mapping[str, float]
) -> dict[str, pd.Series]:
    failures: dict[str, pd.Series] = {}
    for threshold_key, _, _, column in QC_THRESHOLDS:
        if column not in bundle_metadata:
            raise ValueError(f"bundle_metadata lacks QC field: {column}")
        numeric = pd.to_numeric(bundle_metadata[column], errors="coerce")
        failures[threshold_key] = numeric.isna() | (numeric < thresholds[threshold_key])
    return failures


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment that preserves missing entries and row order."""

    numeric = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric)
    if not valid.any():
        return result
    ordered = numeric.loc[valid].sort_values(kind="stable")
    count = len(ordered)
    adjusted = ordered.to_numpy(dtype=float) * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.clip(adjusted, 0.0, 1.0)
    return result


def _acquisition_fov_one_to_one(bundle_metadata: pd.DataFrame) -> bool:
    pairs = bundle_metadata[["acquisition_id", "fov_id"]].drop_duplicates()
    return bool(
        not pairs.empty
        and pairs.groupby("acquisition_id", dropna=False)["fov_id"].nunique(dropna=False).max() == 1
        and pairs.groupby("fov_id", dropna=False)["acquisition_id"].nunique(dropna=False).max() == 1
    )


def build_technical_hierarchy(bundle_metadata: pd.DataFrame) -> pd.DataFrame:
    """Summarize tracking coverage at the acquisition/FOV technical-group level."""

    required = {
        "acquisition_id",
        "fov_id",
        "cell_id",
        "bundle_id",
        "site1_valid_count",
        "site2_valid_count",
        "paired_valid_count",
        "site1_coverage",
        "site2_coverage",
        "paired_coverage",
        "qc_included",
    }
    missing = required.difference(bundle_metadata.columns)
    if missing:
        raise ValueError(f"Bundle metadata lacks technical-hierarchy fields: {sorted(missing)}")
    source = bundle_metadata.assign(
        both_sites_observed=(bundle_metadata["site1_valid_count"] > 0)
        & (bundle_metadata["site2_valid_count"] > 0),
        any_paired_frame=bundle_metadata["paired_valid_count"] > 0,
    )
    grouped = source.groupby(["acquisition_id", "fov_id"], dropna=False, sort=True)
    result = grouped.agg(
        bundle_count=("bundle_id", "nunique"),
        cell_count=("cell_id", "nunique"),
        site1_coverage_median=("site1_coverage", "median"),
        site1_coverage_q1=("site1_coverage", lambda value: value.quantile(0.25)),
        site1_coverage_q3=("site1_coverage", lambda value: value.quantile(0.75)),
        site2_coverage_median=("site2_coverage", "median"),
        site2_coverage_q1=("site2_coverage", lambda value: value.quantile(0.25)),
        site2_coverage_q3=("site2_coverage", lambda value: value.quantile(0.75)),
        paired_coverage_median=("paired_coverage", "median"),
        paired_coverage_q1=("paired_coverage", lambda value: value.quantile(0.25)),
        paired_coverage_q3=("paired_coverage", lambda value: value.quantile(0.75)),
        both_sites_observed_fraction=("both_sites_observed", "mean"),
        any_paired_frame_fraction=("any_paired_frame", "mean"),
        primary_qc_inclusion_fraction=("qc_included", "mean"),
    ).reset_index()
    result["acquisition_fov_one_to_one"] = _acquisition_fov_one_to_one(bundle_metadata)
    return result


def build_timing_hierarchy(frame_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact frame-interval heterogeneity with acquisitions as units.

    Frame rows are first collapsed within each cell. The output contains one row
    per acquisition/FOV pair, so neither frames nor repeated bundle copies of a
    crop time axis are treated as independent observations.
    """

    required = {
        "acquisition_id",
        "fov_id",
        "cell_id",
        "frame_interval_s",
        "exact_interval_median_s",
    }
    missing = required.difference(frame_table.columns)
    if missing:
        return pd.DataFrame(
            columns=[
                "acquisition_id",
                "fov_id",
                "cell_count",
                "exact_interval_median_s",
                "exact_interval_min_s",
                "exact_interval_max_s",
                "exact_interval_max_min_ratio",
                "ratio_gt_1_5",
                "ratio_gt_2",
                "analysis_unit",
                "timing_axis",
            ]
        )
    cell = (
        frame_table.groupby(["acquisition_id", "fov_id", "cell_id"], dropna=False, sort=True)
        .agg(
            exact_interval_median_s=("exact_interval_median_s", "median"),
            exact_interval_min_s=("frame_interval_s", "min"),
            exact_interval_max_s=("frame_interval_s", "max"),
        )
        .reset_index()
    )
    acquisition = (
        cell.groupby(["acquisition_id", "fov_id"], dropna=False, sort=True)
        .agg(
            cell_count=("cell_id", "nunique"),
            exact_interval_median_s=("exact_interval_median_s", "median"),
            exact_interval_min_s=("exact_interval_min_s", "min"),
            exact_interval_max_s=("exact_interval_max_s", "max"),
        )
        .reset_index()
    )
    acquisition["exact_interval_max_min_ratio"] = (
        acquisition["exact_interval_max_s"] / acquisition["exact_interval_min_s"]
    )
    acquisition["ratio_gt_1_5"] = acquisition["exact_interval_max_min_ratio"] > 1.5
    acquisition["ratio_gt_2"] = acquisition["exact_interval_max_min_ratio"] > 2.0
    acquisition["analysis_unit"] = "acquisition"
    acquisition["timing_axis"] = "exact_relative_time_s"
    return acquisition


def build_qc_sensitivity(bundle_metadata: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Evaluate the configured 3-way technical inclusion-threshold grid."""

    required = {
        "site1_valid_count",
        "site2_valid_count",
        "paired_valid_count",
        "longest_paired_run",
        "paired_coverage",
    }
    missing = required.difference(bundle_metadata.columns)
    if missing:
        raise ValueError(f"Bundle metadata lacks QC-sensitivity fields: {sorted(missing)}")
    thresholds = _resolved_thresholds(config)
    qc = config.get("qc", {})
    coverage_values = tuple(
        float(value) for value in qc.get("coverage_sensitivity", (0.4, 0.6, 0.8))
    )
    paired_values = tuple(int(value) for value in qc.get("paired_frames_sensitivity", (10, 20, 30)))
    run_values = tuple(int(value) for value in qc.get("contiguous_run_sensitivity", (5, 10, 20)))
    if not coverage_values or not paired_values or not run_values:
        raise ValueError("QC sensitivity grids must be nonempty")
    if any(not 0 <= value <= 1 for value in coverage_values):
        raise ValueError("Coverage sensitivity values must lie in [0, 1]")
    if any(value < 0 for value in (*paired_values, *run_values)):
        raise ValueError("Frame/run sensitivity values must be nonnegative")

    base = (bundle_metadata["site1_valid_count"] >= thresholds["min_site1_valid_frames"]) & (
        bundle_metadata["site2_valid_count"] >= thresholds["min_site2_valid_frames"]
    )
    rows = []
    eligible = len(bundle_metadata)
    for coverage, paired_frames, contiguous_run in product(
        coverage_values, paired_values, run_values
    ):
        include = (
            base
            & (bundle_metadata["paired_coverage"] >= coverage)
            & (bundle_metadata["paired_valid_count"] >= paired_frames)
            & (bundle_metadata["longest_paired_run"] >= contiguous_run)
        )
        rows.append(
            {
                "min_site1_valid_frames": thresholds["min_site1_valid_frames"],
                "min_site2_valid_frames": thresholds["min_site2_valid_frames"],
                "min_paired_coverage": coverage,
                "min_paired_valid_frames": paired_frames,
                "min_longest_paired_run": contiguous_run,
                "eligible_bundles": eligible,
                "included_bundles": int(include.sum()),
                "inclusion_fraction": float(include.mean()) if eligible else np.nan,
                "evidence_scope": "technical_selection_sensitivity",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["min_longest_paired_run", "min_paired_valid_frames", "min_paired_coverage"],
        kind="stable",
    )


def build_coverage_associations(bundle_metadata: pd.DataFrame) -> pd.DataFrame:
    """Test descriptive coverage heterogeneity using cells as the analysis unit.

    This is a technical association screen, not biological inference. Coverage is
    first averaged across bundles within a cell so individual frame rows are never
    treated as independent observations.
    """

    required = {
        "acquisition_id",
        "fov_id",
        "cell_id",
        "site1_coverage",
        "site2_coverage",
        "paired_coverage",
    }
    missing = required.difference(bundle_metadata.columns)
    if missing:
        raise ValueError(f"Bundle metadata lacks coverage-association fields: {sorted(missing)}")
    cell = (
        bundle_metadata.groupby(["acquisition_id", "fov_id", "cell_id"], dropna=False, sort=True)[
            ["site1_coverage", "site2_coverage", "paired_coverage"]
        ]
        .mean()
        .reset_index()
    )
    one_to_one = _acquisition_fov_one_to_one(bundle_metadata)
    grouping_fields = ("acquisition_id",) if one_to_one else ("acquisition_id", "fov_id")
    rows: list[dict[str, Any]] = []
    for grouping in grouping_fields:
        for outcome in ("site1_coverage", "site2_coverage", "paired_coverage"):
            groups = [
                block[outcome].dropna().to_numpy(dtype=float)
                for _, block in cell.groupby(grouping, dropna=False, sort=True)
            ]
            groups = [values for values in groups if len(values)]
            n_units = int(sum(len(values) for values in groups))
            statistic = np.nan
            p_value = np.nan
            if len(groups) >= 2 and n_units > len(groups):
                try:
                    statistic, p_value = (float(value) for value in kruskal(*groups))
                except ValueError:
                    # scipy rejects the uninformative all-identical case.
                    pass
            denominator = n_units - len(groups)
            epsilon_squared = (
                max(0.0, (statistic - len(groups) + 1) / denominator)
                if denominator > 0 and np.isfinite(statistic)
                else np.nan
            )
            rows.append(
                {
                    "analysis_unit": "cell_mean_across_bundles",
                    "grouping_field": grouping,
                    "outcome": outcome,
                    "n_cells": n_units,
                    "n_groups": len(groups),
                    "kruskal_h": statistic,
                    "p_value": p_value,
                    "epsilon_squared": epsilon_squared,
                    "acquisition_fov_one_to_one": one_to_one,
                    "evidence_level": "descriptive_technical",
                    "interpretation_scope": (
                        "acquisition/FOV not separable"
                        if one_to_one
                        else "technical grouping association"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["p_value_bh"] = _bh_adjust(result["p_value"])
    return result


def _motion_missingness_settings(config: Mapping[str, Any]) -> tuple[float, int, int]:
    """Resolve predeclared technical-screen settings without inspecting outcomes."""

    qc = config.get("qc", {})
    high_quantile = float(qc.get("motion_missingness_high_quantile", 0.75))
    raw_steps = qc.get("motion_missingness_min_valid_steps_per_bundle", 3)
    raw_cells = qc.get("motion_missingness_min_cells_per_acquisition", 5)
    if (
        isinstance(raw_steps, bool)
        or not np.isfinite(float(raw_steps))
        or float(raw_steps) < 1
        or not float(raw_steps).is_integer()
    ):
        raise ValueError("motion_missingness_min_valid_steps_per_bundle must be a positive integer")
    if (
        isinstance(raw_cells, bool)
        or not np.isfinite(float(raw_cells))
        or float(raw_cells) < 2
        or not float(raw_cells).is_integer()
    ):
        raise ValueError("motion_missingness_min_cells_per_acquisition must be an integer >= 2")
    if not np.isfinite(high_quantile) or not 0 < high_quantile < 1:
        raise ValueError("motion_missingness_high_quantile must lie strictly between 0 and 1")
    return high_quantile, int(raw_steps), int(raw_cells)


MOTION_COVERAGE_BUNDLE_COLUMNS = (
    "bundle_id",
    "acquisition_id",
    "fov_id",
    "cell_id",
    "motion_metric",
    "coverage_metric",
    "coverage",
    "valid_exact_speed_steps",
    "median_exact_speed_nm_s",
    "speed_definition",
)

MOTION_COVERAGE_ASSOCIATION_COLUMNS = (
    "acquisition_id",
    "fov_id",
    "motion_metric",
    "coverage_metric",
    "n_cells",
    "n_bundles",
    "spearman_rho",
    "p_value_descriptive",
    "p_value_bh",
    "supported",
    "status_reason",
    "analysis_unit",
    "timing_axis",
    "minimum_valid_steps_per_bundle",
    "minimum_cells_per_acquisition",
)

DROPOUT_RISK_CONTRAST_COLUMNS = (
    "acquisition_id",
    "fov_id",
    "analysis",
    "channel",
    "predictor",
    "predictor_unit",
    "high_quantile",
    "within_acquisition_threshold",
    "n_cells_with_any_eligible_transition",
    "n_cells_with_both_strata",
    "eligible_transitions_high_all_cells",
    "loss_events_high_all_cells",
    "eligible_transitions_reference_all_cells",
    "loss_events_reference_all_cells",
    "eligible_transitions_high_effect_cells",
    "loss_events_high_effect_cells",
    "eligible_transitions_reference_effect_cells",
    "loss_events_reference_effect_cells",
    "cell_mean_loss_risk_high",
    "cell_mean_loss_risk_reference",
    "cell_mean_risk_difference",
    "cell_median_risk_difference",
    "supported",
    "status_reason",
    "effect_analysis_unit",
    "event_denominator",
    "timing_axis",
    "minimum_cells_per_acquisition",
)


def _empty_table(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _neutral_motion_frame_table(frame_table: pd.DataFrame) -> pd.DataFrame | None:
    required = {
        "bundle_id",
        "acquisition_id",
        "fov_id",
        "cell_id",
        "frame",
        "micro_time_s",
        "site1_x_nm",
        "site1_y_nm",
        "site1_valid",
        "site2_x_nm",
        "site2_y_nm",
        "site2_valid",
    }
    if not required.issubset(frame_table.columns):
        return None
    ordered = frame_table.loc[:, sorted(required)].copy()
    for column in (
        "frame",
        "micro_time_s",
        "site1_x_nm",
        "site1_y_nm",
        "site2_x_nm",
        "site2_y_nm",
    ):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    for column in ("site1_valid", "site2_valid"):
        ordered[column] = ordered[column].eq(True).fillna(False)
    ordered = ordered.sort_values(["bundle_id", "frame"], kind="stable").reset_index(drop=True)
    return ordered


def build_motion_coverage_diagnostics(
    frame_table: pd.DataFrame,
    bundle_metadata: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relate neutral tracking coverage to observed exact-time motion.

    Motion is summarized within bundle, then collapsed to cells before an
    acquisition-specific Spearman screen. No frame or bundle is used as an
    independent cross-acquisition replicate.
    """

    _, minimum_steps, minimum_cells = _motion_missingness_settings(config)
    ordered = _neutral_motion_frame_table(frame_table)
    metadata_required = {
        "bundle_id",
        "acquisition_id",
        "fov_id",
        "cell_id",
        "site1_coverage",
        "site2_coverage",
        "paired_coverage",
    }
    if ordered is None or not metadata_required.issubset(bundle_metadata.columns):
        return (
            _empty_table(MOTION_COVERAGE_BUNDLE_COLUMNS),
            _empty_table(MOTION_COVERAGE_ASSOCIATION_COLUMNS),
        )
    metadata = bundle_metadata.loc[:, sorted(metadata_required)].copy()
    if metadata["bundle_id"].duplicated().any():
        raise ValueError("Motion/coverage audit requires one metadata row per bundle")

    grouped = ordered.groupby("bundle_id", sort=False, dropna=False)
    previous_frame = grouped["frame"].shift(1)
    previous_time = grouped["micro_time_s"].shift(1)
    elapsed = ordered["micro_time_s"] - previous_time
    consecutive = ordered["frame"].sub(previous_frame).eq(1) & elapsed.gt(0) & np.isfinite(elapsed)

    speed_specs: list[tuple[str, str, pd.Series, str]] = []
    site_positions: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    for site in ("site1", "site2"):
        x = ordered[f"{site}_x_nm"]
        y = ordered[f"{site}_y_nm"]
        valid = ordered[f"{site}_valid"] & np.isfinite(x) & np.isfinite(y)
        previous_x = grouped[f"{site}_x_nm"].shift(1)
        previous_y = grouped[f"{site}_y_nm"].shift(1)
        previous_valid = grouped[f"{site}_valid"].shift(1).eq(True)
        step_valid = (
            consecutive & valid & previous_valid & np.isfinite(previous_x) & np.isfinite(previous_y)
        )
        speed = pd.Series(np.nan, index=ordered.index, dtype=float)
        speed.loc[step_valid] = (
            np.hypot(
                x.loc[step_valid] - previous_x.loc[step_valid],
                y.loc[step_valid] - previous_y.loc[step_valid],
            )
            / elapsed.loc[step_valid]
        )
        speed_specs.append(
            (
                site,
                f"{site}_coverage",
                speed,
                f"consecutive-frame {site} displacement divided by exact elapsed seconds",
            )
        )
        site_positions[site] = (x, y, valid)

    midpoint_x = (site_positions["site1"][0] + site_positions["site2"][0]) / 2.0
    midpoint_y = (site_positions["site1"][1] + site_positions["site2"][1]) / 2.0
    paired_valid = site_positions["site1"][2] & site_positions["site2"][2]
    bundle_groups = ordered["bundle_id"]
    previous_midpoint_x = midpoint_x.groupby(bundle_groups, sort=False, dropna=False).shift(1)
    previous_midpoint_y = midpoint_y.groupby(bundle_groups, sort=False, dropna=False).shift(1)
    previous_paired = (
        paired_valid.groupby(bundle_groups, sort=False, dropna=False).shift(1).eq(True)
    )
    common_valid = (
        consecutive
        & paired_valid
        & previous_paired
        & np.isfinite(previous_midpoint_x)
        & np.isfinite(previous_midpoint_y)
    )
    common_speed = pd.Series(np.nan, index=ordered.index, dtype=float)
    common_speed.loc[common_valid] = (
        np.hypot(
            midpoint_x.loc[common_valid] - previous_midpoint_x.loc[common_valid],
            midpoint_y.loc[common_valid] - previous_midpoint_y.loc[common_valid],
        )
        / elapsed.loc[common_valid]
    )
    speed_specs.append(
        (
            "common_mode",
            "paired_coverage",
            common_speed,
            "consecutive paired-midpoint displacement divided by exact elapsed seconds",
        )
    )

    bundle_rows: list[pd.DataFrame] = []
    for metric, coverage_metric, speed, definition in speed_specs:
        source = ordered.loc[:, ["bundle_id"]].assign(_speed=speed)
        summary = (
            source.groupby("bundle_id", dropna=False, sort=True)["_speed"]
            .agg(valid_exact_speed_steps="count", median_exact_speed_nm_s="median")
            .reset_index()
            .merge(metadata, on="bundle_id", how="left", validate="one_to_one")
        )
        summary["motion_metric"] = metric
        summary["coverage_metric"] = coverage_metric
        summary["coverage"] = pd.to_numeric(summary[coverage_metric], errors="coerce")
        summary["speed_definition"] = definition
        bundle_rows.append(summary.loc[:, list(MOTION_COVERAGE_BUNDLE_COLUMNS)])
    bundle_summary = pd.concat(bundle_rows, ignore_index=True)

    eligible = bundle_summary.loc[
        bundle_summary["valid_exact_speed_steps"].ge(minimum_steps)
        & np.isfinite(bundle_summary["coverage"])
        & np.isfinite(bundle_summary["median_exact_speed_nm_s"])
    ].copy()
    cell = (
        eligible.groupby(
            ["acquisition_id", "fov_id", "cell_id", "motion_metric", "coverage_metric"],
            dropna=False,
            sort=True,
        )
        .agg(
            coverage=("coverage", "mean"),
            median_exact_speed_nm_s=("median_exact_speed_nm_s", "median"),
            n_bundles=("bundle_id", "nunique"),
        )
        .reset_index()
    )
    association_rows: list[dict[str, Any]] = []
    for keys, block in cell.groupby(
        ["acquisition_id", "fov_id", "motion_metric", "coverage_metric"],
        dropna=False,
        sort=True,
    ):
        acquisition_id, fov_id, metric, coverage_metric = keys
        n_cells = int(block["cell_id"].nunique())
        enough_cells = n_cells >= minimum_cells
        variable_coverage = block["coverage"].nunique(dropna=True) > 1
        variable_speed = block["median_exact_speed_nm_s"].nunique(dropna=True) > 1
        supported = enough_cells and variable_coverage and variable_speed
        rho = np.nan
        p_value = np.nan
        if supported:
            statistic = spearmanr(
                block["coverage"].to_numpy(float),
                block["median_exact_speed_nm_s"].to_numpy(float),
            )
            rho = float(statistic.statistic)
            p_value = float(statistic.pvalue)
        reasons = []
        if not enough_cells:
            reasons.append(f"fewer_than_{minimum_cells}_cells")
        if not variable_coverage:
            reasons.append("coverage_constant")
        if not variable_speed:
            reasons.append("speed_constant")
        association_rows.append(
            {
                "acquisition_id": acquisition_id,
                "fov_id": fov_id,
                "motion_metric": metric,
                "coverage_metric": coverage_metric,
                "n_cells": n_cells,
                "n_bundles": int(block["n_bundles"].sum()),
                "spearman_rho": rho,
                "p_value_descriptive": p_value,
                "p_value_bh": np.nan,
                "supported": supported,
                "status_reason": "supported" if supported else "|".join(reasons),
                "analysis_unit": "cell_within_acquisition",
                "timing_axis": "exact_micro_time_s_no_interpolation",
                "minimum_valid_steps_per_bundle": minimum_steps,
                "minimum_cells_per_acquisition": minimum_cells,
            }
        )
    associations = pd.DataFrame(association_rows, columns=MOTION_COVERAGE_ASSOCIATION_COLUMNS)
    if not associations.empty:
        associations["p_value_bh"] = _bh_adjust(associations["p_value_descriptive"])
    return bundle_summary, associations


def _acquisition_dropout_contrasts(
    events: pd.DataFrame,
    *,
    analysis: str,
    channel: str,
    predictor: str,
    predictor_unit: str,
    high_quantile: float,
    minimum_cells: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (acquisition_id, fov_id), block in events.groupby(
        ["acquisition_id", "fov_id"], dropna=False, sort=True
    ):
        threshold = float(block["predictor_value"].quantile(high_quantile))
        prepared = block.assign(
            stratum=np.where(block["predictor_value"].ge(threshold), "high", "reference")
        )
        totals = prepared.groupby("stratum", observed=False)["loss_event"].agg(["size", "sum"])
        cells = (
            prepared.groupby(["cell_id", "stratum"], observed=False, sort=True)["loss_event"]
            .agg(eligible_transitions="size", loss_events="sum", loss_risk="mean")
            .reset_index()
        )
        cell_risk = cells.pivot(index="cell_id", columns="stratum", values="loss_risk")
        both_cells = (
            cell_risk.dropna(subset=["high", "reference"], how="any")
            if {
                "high",
                "reference",
            }.issubset(cell_risk.columns)
            else pd.DataFrame()
        )
        both_ids = set(both_cells.index) if not both_cells.empty else set()
        effect_events = cells.loc[cells["cell_id"].isin(both_ids)]
        effect_totals = effect_events.groupby("stratum", observed=False)[
            ["eligible_transitions", "loss_events"]
        ].sum()

        def _value(table: pd.DataFrame, row: str, column: str, default: int = 0) -> int:
            if row not in table.index or column not in table.columns:
                return default
            return int(table.loc[row, column])

        n_both = len(both_cells)
        supported = n_both >= minimum_cells
        if supported:
            high_risk = float(both_cells["high"].mean())
            reference_risk = float(both_cells["reference"].mean())
            differences = both_cells["high"] - both_cells["reference"]
            mean_difference = float(differences.mean())
            median_difference = float(differences.median())
            reason = "supported"
        else:
            high_risk = reference_risk = mean_difference = median_difference = np.nan
            reason = f"fewer_than_{minimum_cells}_cells_with_both_strata"
        rows.append(
            {
                "acquisition_id": acquisition_id,
                "fov_id": fov_id,
                "analysis": analysis,
                "channel": channel,
                "predictor": predictor,
                "predictor_unit": predictor_unit,
                "high_quantile": high_quantile,
                "within_acquisition_threshold": threshold,
                "n_cells_with_any_eligible_transition": int(prepared["cell_id"].nunique()),
                "n_cells_with_both_strata": n_both,
                "eligible_transitions_high_all_cells": _value(totals, "high", "size"),
                "loss_events_high_all_cells": _value(totals, "high", "sum"),
                "eligible_transitions_reference_all_cells": _value(totals, "reference", "size"),
                "loss_events_reference_all_cells": _value(totals, "reference", "sum"),
                "eligible_transitions_high_effect_cells": _value(
                    effect_totals, "high", "eligible_transitions"
                ),
                "loss_events_high_effect_cells": _value(effect_totals, "high", "loss_events"),
                "eligible_transitions_reference_effect_cells": _value(
                    effect_totals, "reference", "eligible_transitions"
                ),
                "loss_events_reference_effect_cells": _value(
                    effect_totals, "reference", "loss_events"
                ),
                "cell_mean_loss_risk_high": high_risk,
                "cell_mean_loss_risk_reference": reference_risk,
                "cell_mean_risk_difference": mean_difference,
                "cell_median_risk_difference": median_difference,
                "supported": supported,
                "status_reason": reason,
                "effect_analysis_unit": "acquisition_mean_of_within_cell_risk_differences",
                "event_denominator": "observed_nominal_next_frame_transitions",
                "timing_axis": "exact_micro_time_s_no_interpolation",
                "minimum_cells_per_acquisition": minimum_cells,
            }
        )
    return pd.DataFrame(rows, columns=DROPOUT_RISK_CONTRAST_COLUMNS)


def build_track_loss_diagnostics(
    frame_table: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Screen next-frame tracking loss after high speed or large separation.

    High predictors are defined within acquisition. Effects are computed as
    within-cell risk differences and then summarized per acquisition, preventing
    nominal frame rows from becoming cross-acquisition replicates.
    """

    high_quantile, _, minimum_cells = _motion_missingness_settings(config)
    ordered = _neutral_motion_frame_table(frame_table)
    if ordered is None:
        return _empty_table(DROPOUT_RISK_CONTRAST_COLUMNS)
    grouped = ordered.groupby("bundle_id", sort=False, dropna=False)
    previous_frame = grouped["frame"].shift(1)
    next_frame = grouped["frame"].shift(-1)
    previous_time = grouped["micro_time_s"].shift(1)
    elapsed = ordered["micro_time_s"] - previous_time
    previous_consecutive = ordered["frame"].sub(previous_frame).eq(1)
    next_exists = next_frame.sub(ordered["frame"]).eq(1)

    contrast_tables: list[pd.DataFrame] = []
    for site in ("site1", "site2"):
        x = ordered[f"{site}_x_nm"]
        y = ordered[f"{site}_y_nm"]
        valid = ordered[f"{site}_valid"] & np.isfinite(x) & np.isfinite(y)
        previous_x = grouped[f"{site}_x_nm"].shift(1)
        previous_y = grouped[f"{site}_y_nm"].shift(1)
        previous_valid = grouped[f"{site}_valid"].shift(1).eq(True)
        next_valid = grouped[f"{site}_valid"].shift(-1).eq(True)
        eligible = (
            previous_consecutive
            & next_exists
            & elapsed.gt(0)
            & np.isfinite(elapsed)
            & valid
            & previous_valid
            & np.isfinite(previous_x)
            & np.isfinite(previous_y)
        )
        speed = np.hypot(x - previous_x, y - previous_y) / elapsed
        events = ordered.loc[eligible, ["acquisition_id", "fov_id", "cell_id", "bundle_id"]].copy()
        events["predictor_value"] = speed.loc[eligible].to_numpy(float)
        events["loss_event"] = (~next_valid.loc[eligible]).to_numpy(bool)
        if not events.empty:
            contrast_tables.append(
                _acquisition_dropout_contrasts(
                    events,
                    analysis="next_frame_channel_loss_after_high_speed",
                    channel=site,
                    predictor="previous_to_current_exact_speed",
                    predictor_unit="nm_per_s",
                    high_quantile=high_quantile,
                    minimum_cells=minimum_cells,
                )
            )

    site1_valid = (
        ordered["site1_valid"]
        & np.isfinite(ordered["site1_x_nm"])
        & np.isfinite(ordered["site1_y_nm"])
    )
    site2_valid = (
        ordered["site2_valid"]
        & np.isfinite(ordered["site2_x_nm"])
        & np.isfinite(ordered["site2_y_nm"])
    )
    paired_valid = site1_valid & site2_valid
    next_paired_valid = grouped["site1_valid"].shift(-1).eq(True) & grouped["site2_valid"].shift(
        -1
    ).eq(True)
    separation = np.hypot(
        ordered["site2_x_nm"] - ordered["site1_x_nm"],
        ordered["site2_y_nm"] - ordered["site1_y_nm"],
    )
    eligible_pair = next_exists & paired_valid & np.isfinite(separation)
    pair_events = ordered.loc[
        eligible_pair, ["acquisition_id", "fov_id", "cell_id", "bundle_id"]
    ].copy()
    pair_events["predictor_value"] = separation.loc[eligible_pair].to_numpy(float)
    pair_events["loss_event"] = (~next_paired_valid.loc[eligible_pair]).to_numpy(bool)
    if not pair_events.empty:
        contrast_tables.append(
            _acquisition_dropout_contrasts(
                pair_events,
                analysis="next_frame_pair_loss_after_large_separation",
                channel="paired",
                predictor="current_site1_site2_separation",
                predictor_unit="nm",
                high_quantile=high_quantile,
                minimum_cells=minimum_cells,
            )
        )
    if not contrast_tables:
        return _empty_table(DROPOUT_RISK_CONTRAST_COLUMNS)
    return pd.concat(contrast_tables, ignore_index=True).loc[:, list(DROPOUT_RISK_CONTRAST_COLUMNS)]


def _column_has_observed_values(table: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    present = [column for column in columns if column in table.columns]
    return bool(present and table[present].notna().any().any())


def build_missingness_analysis_gates(
    frame_table: pd.DataFrame,
    bundle_metadata: pd.DataFrame,
    *,
    motion_coverage_associations: pd.DataFrame | None = None,
    dropout_risk_contrasts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Declare which prespecified T03 missingness analyses are currently supported."""

    checks = (
        (
            "acquisition_fov_coverage",
            bundle_metadata,
            ("acquisition_id", "fov_id", "paired_coverage"),
            "supported_descriptive",
            "Coverage and channel-validity fields support technical-group summaries.",
        ),
        (
            "site_intensity",
            bundle_metadata,
            ("median_site_intensity", "site1_intensity", "site2_intensity"),
            "supported_descriptive",
            "Requires observed Site1/Site2 intensity values.",
        ),
        (
            "localization_confidence",
            frame_table,
            (
                "site1_localization_sigma_nm",
                "site2_localization_sigma_nm",
                "extraction_confidence",
            ),
            "supported_descriptive",
            "Requires observed localization uncertainty or extraction-confidence values.",
        ),
        (
            "bp1_intensity",
            frame_table,
            ("bp1_intensity_raw", "bp1_intensity_bg_corrected"),
            "supported_descriptive",
            "Requires validated image-derived 53BP1 intensity values.",
        ),
        (
            "bp1_sbr",
            frame_table,
            ("bp1_sbr",),
            "supported_descriptive",
            "Requires validated image-derived 53BP1 signal-to-background values.",
        ),
        (
            "bp1_component_metrics",
            frame_table,
            ("bp1_component_area_px2", "bp1_component_equivalent_radius_nm"),
            "supported_descriptive",
            "Requires a validated image-component workflow; point trajectories are insufficient.",
        ),
    )
    rows = []
    for analysis, table, columns, ready_status, requirement in checks:
        available = _column_has_observed_values(table, columns)
        rows.append(
            {
                "analysis": analysis,
                "status": ready_status if available else "gated_unavailable",
                "required_columns": "|".join(columns),
                "observed_values_available": available,
                "reason": requirement if not available else "Observed values are available.",
                "evidence_scope": "technical/descriptive only",
            }
        )
    motion_required = (
        "micro_time_s",
        "site1_x_nm",
        "site1_y_nm",
        "site2_x_nm",
        "site2_y_nm",
    )
    coordinate_time_available = set(motion_required).issubset(frame_table.columns) and bool(
        frame_table[list(motion_required)].notna().any().all()
    )
    motion_supported = bool(
        motion_coverage_associations is not None
        and not motion_coverage_associations.empty
        and motion_coverage_associations["supported"].eq(True).any()
    )
    dropout = dropout_risk_contrasts if dropout_risk_contrasts is not None else pd.DataFrame()
    speed_supported = bool(
        not dropout.empty
        and dropout.loc[
            dropout["analysis"].eq("next_frame_channel_loss_after_high_speed"), "supported"
        ]
        .eq(True)
        .any()
    )
    separation_supported = bool(
        not dropout.empty
        and dropout.loc[
            dropout["analysis"].eq("next_frame_pair_loss_after_large_separation"), "supported"
        ]
        .eq(True)
        .any()
    )
    for analysis, supported, requirement in (
        (
            "coverage_vs_motion_magnitude",
            motion_supported,
            "Requires exact-time neutral-coordinate steps in enough cells per acquisition.",
        ),
        (
            "track_loss_after_high_speed",
            speed_supported,
            "Requires exact-time consecutive neutral-site steps followed by an observed next frame.",
        ),
        (
            "track_loss_after_large_separation",
            separation_supported,
            "Requires paired neutral-site coordinates followed by an observed next frame.",
        ),
    ):
        if supported:
            status = "supported_descriptive"
            reason = "Acquisition-stratified, cell-balanced technical diagnostics are available."
        elif coordinate_time_available:
            status = "gated_insufficient_support"
            reason = requirement
        else:
            status = "gated_unavailable"
            reason = requirement
        rows.append(
            {
                "analysis": analysis,
                "status": status,
                "required_columns": "|".join(motion_required),
                "observed_values_available": coordinate_time_available,
                "reason": reason,
                "evidence_scope": "technical/descriptive only",
            }
        )
    rows.append(
        {
            "analysis": "macro_time_missingness",
            "status": "gated_m0_authoritative_mapping",
            "required_columns": "hour_post_delivery|biological_experiment_id",
            "observed_values_available": bool(bundle_metadata["hour_post_delivery"].notna().any()),
            "reason": (
                "Folder-label hours are preserved as provenance, but the authoritative macro-time "
                "and biological-experiment mapping remains unresolved."
            ),
            "evidence_scope": "provenance counts only",
        }
    )
    return pd.DataFrame(rows)


def _motion_missingness_summary(
    associations: pd.DataFrame, contrasts: pd.DataFrame
) -> dict[str, Any]:
    correlation_rows: list[dict[str, Any]] = []
    if not associations.empty:
        for metric, block in associations.groupby("motion_metric", sort=True):
            supported = block.loc[block["supported"].eq(True) & block["spearman_rho"].notna()]
            values = supported["spearman_rho"].to_numpy(float)
            correlation_rows.append(
                {
                    "motion_metric": str(metric),
                    "supported_acquisitions": len(supported),
                    "assessed_acquisitions": len(block),
                    "median_within_acquisition_spearman_rho": (
                        float(np.median(values)) if len(values) else None
                    ),
                    "minimum_rho": float(np.min(values)) if len(values) else None,
                    "maximum_rho": float(np.max(values)) if len(values) else None,
                }
            )
    dropout_rows: list[dict[str, Any]] = []
    if not contrasts.empty:
        for (analysis, channel), block in contrasts.groupby(["analysis", "channel"], sort=True):
            supported = block.loc[
                block["supported"].eq(True) & block["cell_mean_risk_difference"].notna()
            ]
            values = supported["cell_mean_risk_difference"].to_numpy(float)
            dropout_rows.append(
                {
                    "analysis": str(analysis),
                    "channel": str(channel),
                    "supported_acquisitions": len(supported),
                    "assessed_acquisitions": len(block),
                    "median_acquisition_cell_mean_risk_difference": (
                        float(np.median(values)) if len(values) else None
                    ),
                    "minimum_risk_difference": float(np.min(values)) if len(values) else None,
                    "maximum_risk_difference": float(np.max(values)) if len(values) else None,
                }
            )
    return {
        "status": (
            "supported_descriptive"
            if any(row["supported_acquisitions"] for row in (*correlation_rows, *dropout_rows))
            else "gated_insufficient_or_unavailable"
        ),
        "coverage_vs_motion": correlation_rows,
        "next_frame_track_loss": dropout_rows,
        "inference_scope": (
            "technical screen; within-acquisition cell units and acquisition-level summaries"
        ),
        "causal_or_biological_interpretation": "prohibited",
    }


def apply_primary_qc(
    frame_table: pd.DataFrame,
    bundle_metadata: pd.DataFrame,
    config: Mapping[str, Any],
) -> InitialQcResult:
    """Apply all configured primary thresholds and record every reason.

    Exclusions are bundle-level.  Rows retain recorded hour/acquisition IDs;
    missing hour values remain missing rather than being inferred.
    """

    thresholds = _resolved_thresholds(config)
    failures = _rule_failures(bundle_metadata, thresholds)
    failed_matrix = pd.DataFrame(failures, index=bundle_metadata.index)
    include = ~failed_matrix.any(axis=1)
    evaluated = bundle_metadata.copy()
    evaluated["qc_included"] = include
    evaluated["qc_exclusion_reasons"] = [
        "|".join(key for key in failed_matrix.columns if bool(failed_matrix.loc[index, key]))
        for index in failed_matrix.index
    ]
    excluded = evaluated.loc[~include].copy()
    included = evaluated.loc[include].copy().reset_index(drop=True)
    included_ids = set(included["bundle_id"].astype(str))
    frames_qc = frame_table.loc[frame_table["bundle_id"].astype(str).isin(included_ids)].copy()
    frames_qc = frames_qc.reset_index(drop=True)

    flow_rows = [
        {
            "flow_type": "overall",
            "stage": "eligible_formal_bundles",
            "reason": "formal_usable_snapshot",
            "hour_post_delivery": pd.NA,
            "acquisition_id": pd.NA,
            "bundle_count": len(evaluated),
        },
        {
            "flow_type": "overall",
            "stage": "included_primary_qc",
            "reason": "all_configured_thresholds_passed",
            "hour_post_delivery": pd.NA,
            "acquisition_id": pd.NA,
            "bundle_count": int(include.sum()),
        },
        {
            "flow_type": "overall",
            "stage": "excluded_primary_qc",
            "reason": "one_or_more_thresholds_failed",
            "hour_post_delivery": pd.NA,
            "acquisition_id": pd.NA,
            "bundle_count": int((~include).sum()),
        },
    ]
    for reason, failed in failures.items():
        threshold = thresholds[reason]
        selected = bundle_metadata.loc[failed]
        flow_rows.append(
            {
                "flow_type": "reason_total_nonexclusive",
                "stage": "excluded_primary_qc",
                "reason": reason,
                "threshold": threshold,
                "hour_post_delivery": pd.NA,
                "acquisition_id": pd.NA,
                "bundle_count": int(failed.sum()),
            }
        )
        grouped = (
            selected.groupby(["hour_post_delivery", "acquisition_id"], dropna=False)
            .size()
            .rename("bundle_count")
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            flow_rows.append(
                {
                    "flow_type": "reason_by_hour_acquisition_nonexclusive",
                    "stage": "excluded_primary_qc",
                    "reason": reason,
                    "threshold": threshold,
                    "hour_post_delivery": row.hour_post_delivery,
                    "acquisition_id": row.acquisition_id,
                    "bundle_count": int(row.bundle_count),
                }
            )
    cohort_flow = pd.DataFrame(flow_rows)
    census = build_missingness_census(frame_table, bundle_metadata)
    asymmetry = build_missingness_asymmetry(frame_table, bundle_metadata)
    hierarchy = build_technical_hierarchy(evaluated)
    timing_hierarchy = build_timing_hierarchy(frame_table)
    qc_sensitivity = build_qc_sensitivity(evaluated, config)
    associations = build_coverage_associations(evaluated)
    motion_bundle, motion_associations = build_motion_coverage_diagnostics(
        frame_table, evaluated, config
    )
    dropout_contrasts = build_track_loss_diagnostics(frame_table, config)
    missingness_gates = build_missingness_analysis_gates(
        frame_table,
        evaluated,
        motion_coverage_associations=motion_associations,
        dropout_risk_contrasts=dropout_contrasts,
    )

    both_sites_observed = (evaluated["site1_valid_count"] > 0) & (
        evaluated["site2_valid_count"] > 0
    )
    any_paired_frame = evaluated["paired_valid_count"] > 0
    acquisition_columns = [
        column
        for column in (
            "acquisition_id",
            "hour_post_delivery",
            "hour_mapping_source",
            "hour_mapping_status",
        )
        if column in evaluated.columns
    ]
    acquisition_hours = evaluated[acquisition_columns].drop_duplicates("acquisition_id")
    source = acquisition_hours.get(
        "hour_mapping_source", pd.Series(pd.NA, index=acquisition_hours.index, dtype="string")
    ).astype("string")
    status = acquisition_hours.get(
        "hour_mapping_status", pd.Series(pd.NA, index=acquisition_hours.index, dtype="string")
    ).astype("string")
    hour = acquisition_hours.get(
        "hour_post_delivery", pd.Series(np.nan, index=acquisition_hours.index, dtype=float)
    )
    folder_label = source.str.strip().str.lower().eq("folder label") & pd.notna(hour)
    inferred_not_used = status.str.contains("inferred", case=False, na=False) & pd.isna(hour)

    summary = {
        "eligible_bundles": len(evaluated),
        "included_bundles": len(included),
        "excluded_bundles": len(excluded),
        "eligible_frame_rows": len(frame_table),
        "included_frame_rows": len(frames_qc),
        "thresholds": thresholds,
        "exclusion_counts_nonexclusive": {key: int(mask.sum()) for key, mask in failures.items()},
        "mapped_hours_preserved": int(
            bundle_metadata.loc[
                bundle_metadata["hour_post_delivery"].notna(), "acquisition_id"
            ].nunique()
        ),
        "unmapped_hours_not_inferred": int(
            bundle_metadata.loc[
                bundle_metadata["hour_post_delivery"].isna(), "acquisition_id"
            ].nunique()
        ),
        "technical_hierarchy": {
            "acquisition_count": int(evaluated["acquisition_id"].nunique()),
            "fov_count": int(evaluated["fov_id"].nunique()),
            "cell_count": int(evaluated["cell_id"].nunique()),
            "acquisition_fov_one_to_one": _acquisition_fov_one_to_one(evaluated),
            "biological_experiment_id_available": bool(
                "biological_experiment_id" in evaluated
                and evaluated["biological_experiment_id"].notna().any()
            ),
        },
        "hour_label_provenance": {
            "folder_label_acquisitions": int(folder_label.sum()),
            "inferred_inventory_label_left_unmapped_acquisitions": int(inferred_not_used.sum()),
            "all_unmapped_acquisitions": int(pd.isna(hour).sum()),
            "macro_time_biological_inference": "gated_m0_authoritative_mapping",
        },
        "bundle_pairing_success": {
            "definition": "at least one same-frame Site1/Site2 valid pair",
            "successful_bundles": int(any_paired_frame.sum()),
            "eligible_bundles": len(evaluated),
            "success_fraction": float(any_paired_frame.mean()) if len(evaluated) else np.nan,
            "both_sites_observed_bundles": int(both_sites_observed.sum()),
            "both_sites_observed_fraction": (
                float(both_sites_observed.mean()) if len(evaluated) else np.nan
            ),
        },
        "coverage_association_scope": {
            "analysis_unit": "cell_mean_across_bundles",
            "multiple_testing": "Benjamini-Hochberg within T03 coverage screen",
            "interpretation": "descriptive technical association only",
        },
        "motion_dependent_missingness": _motion_missingness_summary(
            motion_associations, dropout_contrasts
        ),
        "exact_timing_heterogeneity": (
            {
                "status": "supported_descriptive",
                "analysis_unit": "acquisition",
                "acquisition_count": len(timing_hierarchy),
                "median_interval_min_s": float(timing_hierarchy["exact_interval_median_s"].min()),
                "median_interval_max_s": float(timing_hierarchy["exact_interval_median_s"].max()),
                "acquisitions_ratio_gt_1_5": int(timing_hierarchy["ratio_gt_1_5"].sum()),
                "acquisitions_ratio_gt_2": int(timing_hierarchy["ratio_gt_2"].sum()),
                "rule": (
                    "exact timestamps required for lag calculations; frames/crops not replicates"
                ),
            }
            if not timing_hierarchy.empty
            else {
                "status": "gated_unavailable",
                "analysis_unit": "acquisition",
                "acquisition_count": 0,
                "reason": (
                    "Frame-level exact interval columns are unavailable in this input; no timing "
                    "values were synthesized."
                ),
            }
        ),
        "qc_selection_sensitivity": {
            "grid_rows": len(qc_sensitivity),
            "included_bundles_min": int(qc_sensitivity["included_bundles"].min()),
            "included_bundles_max": int(qc_sensitivity["included_bundles"].max()),
            "inclusion_fraction_min": float(qc_sensitivity["inclusion_fraction"].min()),
            "inclusion_fraction_max": float(qc_sensitivity["inclusion_fraction"].max()),
            "evidence_scope": "technical selection sensitivity only",
        },
        "missingness_analysis_gate_counts": {
            str(key): int(value)
            for key, value in missingness_gates["status"].value_counts().items()
        },
    }
    return InitialQcResult(
        frame_table_qc=frames_qc,
        bundle_metadata_evaluated=evaluated.reset_index(drop=True),
        bundle_metadata_qc=included,
        bundle_exclusions=excluded.reset_index(drop=True),
        cohort_flowchart=cohort_flow,
        missingness_census=census,
        missingness_asymmetry=asymmetry,
        technical_hierarchy=hierarchy,
        timing_hierarchy=timing_hierarchy,
        qc_sensitivity=qc_sensitivity,
        coverage_associations=associations,
        motion_coverage_bundle_summary=motion_bundle,
        motion_coverage_associations=motion_associations,
        dropout_risk_contrasts=dropout_contrasts,
        missingness_gates=missingness_gates,
        summary=summary,
    )


def _summarize_group(group: pd.DataFrame, bundle_group: pd.DataFrame) -> dict[str, Any]:
    nominal = len(group)
    site1_count = int(group["site1_valid"].sum())
    site2_count = int(group["site2_valid"].sum())
    paired_count = int(group["paired_valid"].sum())
    bp1_count = int(group["bp1_valid"].sum())
    return {
        "bundle_count": int(bundle_group["bundle_id"].nunique()),
        "cell_count": int(bundle_group["cell_id"].nunique()),
        "nominal_frame_rows": nominal,
        "site1_valid_frames": site1_count,
        "site2_valid_frames": site2_count,
        "paired_valid_frames": paired_count,
        "bp1_valid_frames": bp1_count,
        "site1_coverage": site1_count / nominal if nominal else np.nan,
        "site2_coverage": site2_count / nominal if nominal else np.nan,
        "paired_coverage": paired_count / nominal if nominal else np.nan,
        "bp1_detection_fraction": bp1_count / nominal if nominal else np.nan,
        "median_bundle_paired_coverage": float(bundle_group["paired_coverage"].median()),
        "median_longest_site1_run": float(bundle_group["longest_site1_run"].median()),
        "median_longest_site2_run": float(bundle_group["longest_site2_run"].median()),
        "median_longest_paired_run": float(bundle_group["longest_paired_run"].median()),
    }


def build_missingness_census(
    frame_table: pd.DataFrame, bundle_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Return overall and hierarchy-stratified coverage summaries."""

    levels: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("overall", ()),
        ("hour", ("hour_post_delivery",)),
        ("acquisition", ("acquisition_id",)),
        ("fov", ("acquisition_id", "fov_id")),
        ("cell", ("acquisition_id", "fov_id", "cell_id")),
    )
    rows: list[dict[str, Any]] = []
    for level, columns in levels:
        if not columns:
            rows.append(
                {
                    "level": level,
                    **_summarize_group(frame_table, bundle_metadata),
                }
            )
            continue
        grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
        for keys, frames in frame_table.groupby(grouper, dropna=False, sort=True):
            keys_tuple = keys if isinstance(keys, tuple) else (keys,)
            selectors = pd.Series(True, index=bundle_metadata.index)
            row: dict[str, Any] = {"level": level}
            for column, value in zip(columns, keys_tuple, strict=True):
                row[column] = value
                if pd.isna(value):
                    selectors &= bundle_metadata[column].isna()
                else:
                    selectors &= bundle_metadata[column].eq(value)
            bundles = bundle_metadata.loc[selectors]
            rows.append({**row, **_summarize_group(frames, bundles)})
    return pd.DataFrame(rows)


def build_missingness_asymmetry(
    frame_table: pd.DataFrame, bundle_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Quantify neutral Site1/Site2 dropout asymmetry overall and by strata."""

    frame = frame_table.assign(
        site1_only=frame_table["site1_valid"] & ~frame_table["site2_valid"],
        site2_only=frame_table["site2_valid"] & ~frame_table["site1_valid"],
        both_missing=~frame_table["site1_valid"] & ~frame_table["site2_valid"],
    )
    levels: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("overall", ()),
        ("hour", ("hour_post_delivery",)),
        ("acquisition", ("acquisition_id",)),
    )
    rows: list[dict[str, Any]] = []
    for level, columns in levels:
        groups = (
            [((), frame)]
            if not columns
            else frame.groupby(
                columns[0] if len(columns) == 1 else list(columns), dropna=False, sort=True
            )
        )
        for keys, group in groups:
            keys_tuple = keys if isinstance(keys, tuple) else (keys,)
            row: dict[str, Any] = {"level": level}
            for column, value in zip(columns, keys_tuple, strict=True):
                row[column] = value
            nominal = len(group)
            site1_only = int(group["site1_only"].sum())
            site2_only = int(group["site2_only"].sum())
            discordant = site1_only + site2_only
            row.update(
                {
                    "nominal_frame_rows": nominal,
                    "site1_only_frames": site1_only,
                    "site2_only_frames": site2_only,
                    "both_missing_frames": int(group["both_missing"].sum()),
                    "discordant_frames": discordant,
                    "site1_minus_site2_valid_fraction": float(
                        group["site1_valid"].mean() - group["site2_valid"].mean()
                    ),
                    "discordant_site1_share": site1_only / discordant if discordant else np.nan,
                }
            )
            rows.append(row)

    bundle = bundle_metadata.assign(
        site1_minus_site2_coverage=(
            bundle_metadata["site1_coverage"] - bundle_metadata["site2_coverage"]
        )
    )
    for row in rows:
        if row["level"] == "overall":
            subset = bundle
        elif row["level"] == "hour":
            value = row["hour_post_delivery"]
            subset = bundle.loc[
                bundle["hour_post_delivery"].isna()
                if pd.isna(value)
                else bundle["hour_post_delivery"].eq(value)
            ]
        else:
            subset = bundle.loc[bundle["acquisition_id"].eq(row["acquisition_id"])]
        row["bundle_count"] = len(subset)
        row["median_site1_minus_site2_coverage"] = float(
            subset["site1_minus_site2_coverage"].median()
        )
    return pd.DataFrame(rows)


def _report_markdown(result: InitialQcResult) -> str:
    summary = result.summary
    hierarchy = summary["technical_hierarchy"]
    hours = summary["hour_label_provenance"]
    pairing = summary["bundle_pairing_success"]
    timing = summary["exact_timing_heterogeneity"]
    sensitivity = summary["qc_selection_sensitivity"]
    motion_missingness = summary["motion_dependent_missingness"]
    lines = [
        "# Initial QC and missingness report",
        "",
        (
            "This report is technical/descriptive only. Site1 and Site2 remain neutral labels; "
            "53BP1 absence is a tracking observation, not evidence of repair."
        ),
        "",
        "## Cohort flow",
        "",
        f"- Eligible formal bundles: **{summary['eligible_bundles']:,}**",
        f"- Included by primary QC: **{summary['included_bundles']:,}**",
        f"- Excluded by one or more rules: **{summary['excluded_bundles']:,}**",
        f"- Included nominal frame rows: **{summary['included_frame_rows']:,}**",
        (
            "- Operational bundle-pairing success (at least one same-frame Site1/Site2 valid "
            f"pair): **{pairing['successful_bundles']:,}/{pairing['eligible_bundles']:,} "
            f"({pairing['success_fraction']:.1%})**"
        ),
        (
            "- Bundles with each neutral channel observed at least once, irrespective of temporal "
            f"overlap: **{pairing['both_sites_observed_bundles']:,}/{pairing['eligible_bundles']:,} "
            f"({pairing['both_sites_observed_fraction']:.1%})**"
        ),
        "",
        "## Configured defaults",
        "",
    ]
    lines.extend(f"- `{key}`: {value:g}" for key, value in summary["thresholds"].items())
    lines.extend(["", "## Non-exclusive exclusion reasons", ""])
    lines.extend(
        f"- `{key}`: {value:,} bundles"
        for key, value in summary["exclusion_counts_nonexclusive"].items()
    )
    overall = result.missingness_asymmetry.loc[
        result.missingness_asymmetry["level"].eq("overall")
    ].iloc[0]
    lines.extend(
        [
            "",
            "## Channel missingness asymmetry",
            "",
            f"- Site1-only valid frames: {int(overall['site1_only_frames']):,}",
            f"- Site2-only valid frames: {int(overall['site2_only_frames']):,}",
            f"- Both missing frames: {int(overall['both_missing_frames']):,}",
            "",
            "## Technical hierarchy and supported coverage associations",
            "",
            (
                f"The snapshot contains **{hierarchy['acquisition_count']:,} acquisitions**, "
                f"**{hierarchy['fov_count']:,} FOV IDs**, and **{hierarchy['cell_count']:,} "
                "cells**. Biological-experiment IDs are unavailable, so no biological-replicate "
                "count or generalization claim is made."
            ),
            "",
        ]
    )
    if hierarchy["acquisition_fov_one_to_one"]:
        lines.extend(
            [
                (
                    "Acquisition and FOV IDs are one-to-one in this snapshot. Their contributions "
                    "cannot be separated; the tests below therefore describe a combined technical "
                    "grouping association."
                ),
                "",
            ]
        )
    for row in result.coverage_associations.itertuples(index=False):
        if np.isfinite(row.kruskal_h) and np.isfinite(row.p_value_bh):
            label = str(row.outcome).replace("_", " ")
            lines.append(
                f"- {label}: cell-level Kruskal-Wallis H = {row.kruskal_h:.2f}, "
                f"BH-adjusted p = {row.p_value_bh:.3g}, epsilon-squared = "
                f"{row.epsilon_squared:.3f}; n = {row.n_cells:,} cells across "
                f"{row.n_groups:,} technical groups."
            )
    lines.extend(
        [
            "",
            (
                "These are descriptive tracking-coverage associations. They do not identify a "
                "biological effect, a locus effect, or a causal mechanism."
            ),
            "",
            "## Motion-dependent tracking diagnostics",
            "",
            (
                "Motion uses consecutive valid coordinates divided by exact elapsed seconds, "
                "without interpolation. Bundle summaries are collapsed to cells; correlations "
                "are calculated separately within acquisition. High-speed and large-separation "
                "thresholds are acquisition-specific, and next-frame loss contrasts average "
                "within-cell risk differences before acquisitions are displayed."
            ),
            "",
        ]
    )
    if motion_missingness["status"] == "supported_descriptive":
        for item in motion_missingness["coverage_vs_motion"]:
            if item["supported_acquisitions"]:
                lines.append(
                    f"- Coverage vs {item['motion_metric']} speed: median within-acquisition "
                    f"Spearman rho = {item['median_within_acquisition_spearman_rho']:.3f} "
                    f"(range {item['minimum_rho']:.3f} to {item['maximum_rho']:.3f}); "
                    f"{item['supported_acquisitions']}/{item['assessed_acquisitions']} "
                    "acquisitions met support rules."
                )
        for item in motion_missingness["next_frame_track_loss"]:
            if item["supported_acquisitions"]:
                label = f"{item['channel']} {item['analysis']}".replace("_", " ")
                lines.append(
                    f"- {label}: median acquisition-level, cell-balanced high-minus-reference "
                    f"loss-risk difference = "
                    f"{item['median_acquisition_cell_mean_risk_difference']:.3f} "
                    f"(range {item['minimum_risk_difference']:.3f} to "
                    f"{item['maximum_risk_difference']:.3f}); "
                    f"{item['supported_acquisitions']}/{item['assessed_acquisitions']} "
                    "acquisitions met support rules."
                )
    else:
        lines.append(
            "- **gated**: exact-time coordinates or acquisition-level cell support were "
            "insufficient; no motion-dependent missingness estimate was synthesized."
        )
    lines.extend(
        [
            "",
            (
                "These are tracking-failure screens. Positive risk differences mean more "
                "next-frame loss in the within-acquisition upper predictor tail; they are not "
                "interpreted as locus biology or causation."
            ),
            "",
            "## Exact-time heterogeneity",
            "",
        ]
    )
    if timing["status"] == "supported_descriptive":
        lines.extend(
            [
                (
                    f"Across **{timing['acquisition_count']} acquisitions** (the analysis units), "
                    "the acquisition-level median exact frame interval ranges from "
                    f"**{timing['median_interval_min_s']:.3f} s to "
                    f"{timing['median_interval_max_s']:.3f} s**."
                ),
                (
                    f"- **{timing['acquisitions_ratio_gt_1_5']}/{timing['acquisition_count']}** "
                    "acquisitions have exact interval max/min > 1.5."
                ),
                (
                    f"- **{timing['acquisitions_ratio_gt_2']}/{timing['acquisition_count']}** "
                    "acquisitions have exact interval max/min > 2."
                ),
                (
                    "Frame rows and crops are not treated as independent replicates. Exact "
                    "`relative_time_s` values, rather than a single global or acquisition-wide "
                    "interval, are required for downstream lag and velocity calculations."
                ),
            ]
        )
    else:
        lines.append(f"**{timing['status']}**; {timing['reason']}")
    lines.extend(
        [
            "",
            "## Configured technical-selection sensitivity",
            "",
            (
                "The prespecified grid crosses paired coverage (0.40/0.60/0.80), minimum paired "
                "frames (10/20/30), and minimum contiguous paired run (5/10/20), while retaining "
                "the 20-frame minimum for each neutral Site1/Site2 channel."
            ),
            (
                f"Across **{sensitivity['grid_rows']} configurations**, inclusion ranges from "
                f"**{sensitivity['included_bundles_min']:,} "
                f"({sensitivity['inclusion_fraction_min']:.1%})** to "
                f"**{sensitivity['included_bundles_max']:,} "
                f"({sensitivity['inclusion_fraction_max']:.1%})** bundles."
            ),
            (
                "This quantifies cohort-selection sensitivity only; it is not a robustness claim "
                "for any biological endpoint."
            ),
            "",
            "## Macro-time provenance and analysis gates",
            "",
            (
                f"- **{hours['folder_label_acquisitions']} acquisitions** retain a frozen "
                "`folder label` hour as provenance."
            ),
            (
                f"- **{hours['inferred_inventory_label_left_unmapped_acquisitions']} acquisition** "
                "has an inferred inventory label that was deliberately left unmapped."
            ),
            (
                "- Macro-time biological inference remains **gated** until an authoritative "
                "acquisition-to-hour and biological-experiment mapping is supplied."
            ),
            "",
            "Prespecified missingness analyses:",
            "",
        ]
    )
    for row in result.missingness_gates.itertuples(index=False):
        lines.append(f"- `{row.analysis}`: **{row.status}**; {row.reason}")
    lines.extend(
        [
            "",
            (
                "In particular, site intensity, localization confidence, 53BP1 intensity/SBR, "
                "and 53BP1 component analyses are not estimated from point-trajectory presence. "
                "They remain explicitly gated until validated source measurements exist."
            ),
            "",
            "## Reproducible outputs",
            "",
            (
                "Full counts by hour and acquisition are in `cohort_flowchart.csv`, "
                "`missingness_census.csv`, and `missingness_asymmetry.csv`. Missing hours are "
                "retained as missing and never inferred by this stage."
            ),
            (
                "Technical-group summaries and tests are in `technical_hierarchy.csv`, "
                "`timing_hierarchy.csv`, `qc_sensitivity.csv`, `coverage_associations.csv`, and "
                "`missingness_analysis_gates.csv`. Motion-dependent tracking screens are in "
                "`motion_coverage_bundle_summary.csv`, `motion_coverage_associations.csv`, and "
                "`dropout_risk_contrasts.csv`. Each QC figure has PNG/PDF/SVG outputs and exact "
                "source CSVs in the configured figure directory."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_initial_qc(
    result: InitialQcResult,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    allowed_roots: tuple[str | Path, ...],
    forbidden_roots: tuple[str | Path, ...] = (),
    figure_dir: str | Path | None = None,
) -> dict[str, Path]:
    output = validate_analysis_output_dir(
        output_dir,
        project_root=project_root,
        allowed_roots=allowed_roots,
        forbidden_roots=forbidden_roots,
    )
    figure_output = validate_analysis_output_dir(
        figure_dir if figure_dir is not None else output / "figures",
        project_root=project_root,
        allowed_roots=allowed_roots,
        forbidden_roots=forbidden_roots,
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "frame_table_qc": output / "frame_table_qc.parquet",
        "bundle_metadata_qc": output / "bundle_metadata_qc.parquet",
        "bundle_exclusions": output / "bundle_exclusions.csv",
        "cohort_flowchart": output / "cohort_flowchart.csv",
        "missingness_census": output / "missingness_census.csv",
        "missingness_asymmetry": output / "missingness_asymmetry.csv",
        "technical_hierarchy": output / "technical_hierarchy.csv",
        "timing_hierarchy": output / "timing_hierarchy.csv",
        "qc_sensitivity": output / "qc_sensitivity.csv",
        "coverage_associations": output / "coverage_associations.csv",
        "motion_coverage_bundle_summary": output / "motion_coverage_bundle_summary.csv",
        "motion_coverage_associations": output / "motion_coverage_associations.csv",
        "dropout_risk_contrasts": output / "dropout_risk_contrasts.csv",
        "missingness_analysis_gates": output / "missingness_analysis_gates.csv",
        "summary": output / "initial_qc_summary.json",
        "report": output / "initial_qc_report.md",
    }
    result.frame_table_qc.to_parquet(paths["frame_table_qc"], index=False)
    result.bundle_metadata_qc.to_parquet(paths["bundle_metadata_qc"], index=False)
    result.bundle_exclusions.to_csv(paths["bundle_exclusions"], index=False)
    result.cohort_flowchart.to_csv(paths["cohort_flowchart"], index=False)
    result.missingness_census.to_csv(paths["missingness_census"], index=False)
    result.missingness_asymmetry.to_csv(paths["missingness_asymmetry"], index=False)
    result.technical_hierarchy.to_csv(paths["technical_hierarchy"], index=False)
    result.timing_hierarchy.to_csv(paths["timing_hierarchy"], index=False)
    result.qc_sensitivity.to_csv(paths["qc_sensitivity"], index=False)
    result.coverage_associations.to_csv(paths["coverage_associations"], index=False)
    result.motion_coverage_bundle_summary.to_csv(
        paths["motion_coverage_bundle_summary"], index=False
    )
    result.motion_coverage_associations.to_csv(paths["motion_coverage_associations"], index=False)
    result.dropout_risk_contrasts.to_csv(paths["dropout_risk_contrasts"], index=False)
    result.missingness_gates.to_csv(paths["missingness_analysis_gates"], index=False)
    paths["summary"].write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(_report_markdown(result), encoding="utf-8")
    overall_flow = result.cohort_flowchart.loc[
        result.cohort_flowchart["flow_type"].eq("overall"), ["stage", "bundle_count"]
    ].copy()
    flow_order = {
        "eligible_formal_bundles": (0, "Eligible formal bundles", "neutral"),
        "included_primary_qc": (1, "Included by primary QC", "bp1"),
        "excluded_primary_qc": (2, "Excluded by primary QC", "warning"),
    }
    overall_flow["display_order"] = overall_flow["stage"].map(
        {key: value[0] for key, value in flow_order.items()}
    )
    overall_flow["stage_label"] = overall_flow["stage"].map(
        {key: value[1] for key, value in flow_order.items()}
    )
    overall_flow["color_key"] = overall_flow["stage"].map(
        {key: value[2] for key, value in flow_order.items()}
    )
    coverage_columns = [
        "bundle_id",
        "acquisition_id",
        "fov_id",
        "cell_id",
        "hour_post_delivery",
        "hour_mapping_source",
        "hour_mapping_status",
        "site1_coverage",
        "site2_coverage",
        "paired_coverage",
        "site1_valid_count",
        "site2_valid_count",
        "paired_valid_count",
        "qc_included",
    ]
    coverage_source = result.bundle_metadata_evaluated.loc[
        :, [column for column in coverage_columns if column in result.bundle_metadata_evaluated]
    ].copy()
    figure_paths = write_initial_qc_figures(
        cohort_flow_source=overall_flow,
        coverage_source=coverage_source,
        missingness_asymmetry_source=result.missingness_asymmetry,
        timing_hierarchy_source=result.timing_hierarchy,
        qc_sensitivity_source=result.qc_sensitivity,
        motion_coverage_source=result.motion_coverage_associations,
        dropout_risk_source=result.dropout_risk_contrasts,
        output_dir=figure_output,
    )
    paths.update({f"figure_{key}": value for key, value in figure_paths.items()})
    return paths
