"""Transparent four-tier cohort views for DSB trajectory analysis.

The tiers deliberately separate observation availability from analysis support:

* T1 retains every same-frame Site1/Site2 observation.
* T2 retains Site1 and Site2 observations as independent cohorts.
* T3 is a simple relaxed paired-trajectory cohort.
* T4 reproduces the existing primary/strict bundle QC.

No tier interpolates coordinates.  Metric-specific valid-pair requirements remain
downstream metric gates and are not silently folded into cohort QC.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io import sha256_file
from .tables import canonical_output_roots, validate_analysis_output_dir

FRAME_REQUIRED = {
    "biological_experiment_id",
    "acquisition_id",
    "fov_id",
    "cell_id",
    "bundle_id",
    "allele_index",
    "hour_post_delivery",
    "frame",
    "micro_time_s",
    "frame_interval_s",
    "timing_axis_primary",
    "site1_x_nm",
    "site1_y_nm",
    "site1_valid",
    "site2_x_nm",
    "site2_y_nm",
    "site2_valid",
    "paired_valid",
}

BUNDLE_REQUIRED = {
    "biological_experiment_id",
    "acquisition_id",
    "fov_id",
    "cell_id",
    "bundle_id",
    "allele_index",
    "hour_post_delivery",
    "nominal_frame_count",
    "site1_valid_count",
    "site2_valid_count",
    "paired_valid_count",
    "longest_site1_run",
    "longest_site2_run",
    "longest_paired_run",
    "site1_coverage",
    "site2_coverage",
    "paired_coverage",
    "trajectory_usability_class",
}

STRICT_RULES = (
    ("min_site1_valid_frames", "min_left_valid_frames", "site1_valid_count"),
    ("min_site2_valid_frames", "min_right_valid_frames", "site2_valid_count"),
    ("min_paired_valid_frames", None, "paired_valid_count"),
    ("min_longest_paired_run", None, "longest_paired_run"),
    ("min_paired_coverage", None, "paired_coverage"),
)

RELAXED_RULES = (
    ("min_site1_valid_frames", "site1_valid_count"),
    ("min_site2_valid_frames", "site2_valid_count"),
    ("min_paired_valid_frames", "paired_valid_count"),
    ("min_longest_paired_run", "longest_paired_run"),
)


@dataclass(frozen=True)
class TieredQcResult:
    shared_frame_observations: pd.DataFrame
    site1_observations: pd.DataFrame
    site2_observations: pd.DataFrame
    site_trajectory_membership: pd.DataFrame
    relaxed_trajectory_membership: pd.DataFrame
    relaxed_frame_table: pd.DataFrame
    strict_trajectory_membership: pd.DataFrame
    strict_frame_table: pd.DataFrame
    tier_summary: pd.DataFrame
    tier_census_by_acquisition: pd.DataFrame
    exclusion_counts: pd.DataFrame
    summary: dict[str, Any]


def load_tiered_qc_contract(path: str | Path) -> dict[str, Any]:
    """Load a versioned tier contract without resolving or guessing data paths."""

    contract_path = Path(path).resolve()
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise TypeError(f"Tiered QC contract must be a mapping: {contract_path}")
    if contract.get("schema_version") != 1:
        raise ValueError("Tiered QC contract requires schema_version: 1")
    return contract


def _nonnegative_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(numeric)


def _relaxed_thresholds(contract: Mapping[str, Any]) -> dict[str, int]:
    try:
        relaxed = contract["tiers"]["trajectory_relaxed"]
    except KeyError as exc:
        raise ValueError(f"Tiered QC contract lacks trajectory_relaxed.{exc.args[0]}") from exc
    thresholds = {
        name: _nonnegative_integer(relaxed.get(name), name, minimum=1) for name, _ in RELAXED_RULES
    }
    if relaxed.get("min_paired_coverage") is not None:
        raise ValueError("Relaxed trajectory QC must not silently add a coverage threshold")
    if relaxed.get("interpolation") != "prohibited":
        raise ValueError("Relaxed trajectory QC must explicitly prohibit interpolation")
    return thresholds


def _strict_thresholds(analysis_config: Mapping[str, Any]) -> dict[str, float]:
    qc = analysis_config.get("qc", {})
    thresholds: dict[str, float] = {}
    for name, alias, _ in STRICT_RULES:
        value = qc.get(name, qc.get(alias) if alias else None)
        if value is None:
            raise ValueError(f"Analysis QC configuration lacks strict threshold: {name}")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"Strict threshold {name} must be finite")
        if name == "min_paired_coverage":
            if not 0 <= numeric <= 1:
                raise ValueError("min_paired_coverage must lie in [0, 1]")
        elif numeric < 0 or not numeric.is_integer():
            raise ValueError(f"Strict threshold {name} must be a nonnegative integer")
        thresholds[name] = numeric
    return thresholds


def _validate_nontrajectory_tier_semantics(contract: Mapping[str, Any]) -> None:
    """Keep the retention-only tiers from acquiring hidden quality filters."""

    tiers = contract.get("tiers", {})
    frame = tiers.get("frame_shared", {})
    site = tiers.get("site_independent", {})
    strict = tiers.get("primary_strict", {})
    if frame.get("minimum_bundle_length", "missing") is not None:
        raise ValueError("T1 shared-frame tier must not impose a bundle-length threshold")
    if frame.get("interpolation") != "prohibited":
        raise ValueError("T1 shared-frame tier must explicitly prohibit interpolation")
    if (
        _nonnegative_integer(site.get("minimum_valid_frames"), "minimum_valid_frames", minimum=1)
        != 1
    ):
        raise ValueError(
            "T2 independent-site tier must retain every site trajectory with >=1 frame"
        )
    if site.get("paired_site_required") is not False:
        raise ValueError("T2 independent-site tier must not require the other site")
    if site.get("interpolation") != "prohibited":
        raise ValueError("T2 independent-site tier must explicitly prohibit interpolation")
    if strict.get("threshold_source") != "analysis_config.qc":
        raise ValueError("T4 thresholds must come from analysis_config.qc")
    if strict.get("preserve_existing_primary_qc") is not True:
        raise ValueError("T4 must explicitly preserve the existing primary QC")
    if strict.get("interpolation") != "prohibited":
        raise ValueError("T4 must explicitly prohibit interpolation")
    if contract.get("interpretation", {}).get("biological_speed_endpoint") != "prohibited":
        raise ValueError("Tiered QC must not promote a biological speed endpoint")


def _require_source_contract(
    frame_table: pd.DataFrame,
    bundle_metadata: pd.DataFrame,
    analysis_config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    missing_frame = FRAME_REQUIRED.difference(frame_table.columns)
    missing_bundle = BUNDLE_REQUIRED.difference(bundle_metadata.columns)
    if missing_frame or missing_bundle:
        raise ValueError(
            "Tiered QC inputs lack required fields: "
            f"frame={sorted(missing_frame)}; bundle={sorted(missing_bundle)}"
        )
    if frame_table.empty or bundle_metadata.empty:
        raise ValueError("Tiered QC inputs must be non-empty")
    if frame_table.duplicated(["bundle_id", "frame"]).any():
        raise ValueError("frame_table is not unique by (bundle_id, frame)")
    if bundle_metadata["bundle_id"].duplicated().any():
        raise ValueError("bundle_metadata is not unique by bundle_id")

    data = analysis_config.get("data", {})
    source = contract.get("source_contract", {})
    if (
        source.get("require_formal_usable_only", True)
        and data.get("formal_usable_only") is not True
    ):
        raise ValueError("Tiered QC requires data.formal_usable_only: true")
    if source.get("require_review_excluded", True) and data.get("review_data_policy") != "exclude":
        raise ValueError("Tiered QC requires data.review_data_policy: exclude")
    if (
        source.get("require_exact_timestamps", True)
        and data.get("use_exact_timestamps") is not True
    ):
        raise ValueError("Tiered QC requires data.use_exact_timestamps: true")

    expected_axis = str(source.get("timing_axis_value", "exact_relative_time_s"))
    axes = set(frame_table["timing_axis_primary"].dropna().astype(str).unique())
    if axes != {expected_axis}:
        raise ValueError(f"Tiered QC requires timing_axis_primary={expected_axis}; observed={axes}")

    allowed_classes = set(source.get("allowed_trajectory_usability_classes", ()))
    observed_classes = set(
        bundle_metadata["trajectory_usability_class"].dropna().astype(str).unique()
    )
    if not allowed_classes or not observed_classes.issubset(allowed_classes):
        raise ValueError(
            "Tiered QC input includes undeclared/review usability classes: "
            f"{sorted(observed_classes.difference(allowed_classes))}"
        )

    paired_expected = frame_table["site1_valid"].astype(bool) & frame_table["site2_valid"].astype(
        bool
    )
    if not np.array_equal(
        frame_table["paired_valid"].astype(bool).to_numpy(), paired_expected.to_numpy()
    ):
        raise ValueError("paired_valid is not exactly site1_valid AND site2_valid")
    for site in ("site1", "site2"):
        valid = frame_table[f"{site}_valid"].astype(bool)
        coordinates_finite = np.isfinite(
            frame_table[[f"{site}_x_nm", f"{site}_y_nm"]].to_numpy(dtype=float)
        ).all(axis=1)
        if not np.all(coordinates_finite[valid.to_numpy()]):
            raise ValueError(f"{site}_valid rows contain non-finite coordinates")

    ordered = frame_table.sort_values(["bundle_id", "frame"], kind="stable")
    times = pd.to_numeric(ordered["micro_time_s"], errors="coerce")
    intervals = pd.to_numeric(ordered["frame_interval_s"], errors="coerce")
    if not np.isfinite(times).all() or not (np.isfinite(intervals) & (intervals > 0)).all():
        raise ValueError("Exact micro-times and frame intervals must be finite and positive")
    within_bundle_dt = times.groupby(ordered["bundle_id"], sort=False).diff()
    if not (within_bundle_dt.dropna() > 0).all():
        raise ValueError("micro_time_s must increase strictly within each bundle")

    frame_ids = set(frame_table["bundle_id"].astype(str))
    bundle_ids = set(bundle_metadata["bundle_id"].astype(str))
    if frame_ids != bundle_ids:
        raise ValueError("frame_table and bundle_metadata bundle IDs do not reconcile")
    grouped = frame_table.groupby("bundle_id", sort=False)
    checks = grouped.agg(
        nominal_frame_count=("frame", "size"),
        site1_valid_count=("site1_valid", "sum"),
        site2_valid_count=("site2_valid", "sum"),
        paired_valid_count=("paired_valid", "sum"),
    ).reset_index()
    declared = bundle_metadata[
        [
            "bundle_id",
            "nominal_frame_count",
            "site1_valid_count",
            "site2_valid_count",
            "paired_valid_count",
        ]
    ]
    reconciled = checks.merge(
        declared, on="bundle_id", suffixes=("_frame", "_bundle"), validate="1:1"
    )
    for field in (
        "nominal_frame_count",
        "site1_valid_count",
        "site2_valid_count",
        "paired_valid_count",
    ):
        if not np.array_equal(
            reconciled[f"{field}_frame"].to_numpy(dtype=int),
            reconciled[f"{field}_bundle"].to_numpy(dtype=int),
        ):
            raise ValueError(f"Frame/bundle reconciliation failed for {field}")


def _membership(
    bundle_metadata: pd.DataFrame,
    rules: tuple[tuple[str, str], ...] | tuple[tuple[str, str | None, str], ...],
    thresholds: Mapping[str, float],
    *,
    prefix: str,
) -> pd.DataFrame:
    result = bundle_metadata.copy()
    failure_names: list[str] = []
    for rule in rules:
        name = rule[0]
        column = rule[-1]
        failure_name = f"fails_{name}"
        result[failure_name] = pd.to_numeric(result[column], errors="coerce").isna() | (
            pd.to_numeric(result[column], errors="coerce") < thresholds[name]
        )
        failure_names.append(failure_name)
    result[f"{prefix}_included"] = ~result[failure_names].any(axis=1)
    result[f"{prefix}_exclusion_reasons"] = [
        "|".join(name.removeprefix("fails_") for name in failure_names if bool(row[name]))
        for _, row in result.iterrows()
    ]
    return result


def _site_observations(frame_table: pd.DataFrame, site: str) -> pd.DataFrame:
    columns = [
        "biological_experiment_id",
        "acquisition_id",
        "fov_id",
        "cell_id",
        "bundle_id",
        "allele_index",
        "hour_post_delivery",
        "frame",
        "micro_time_s",
        "frame_interval_s",
        "timing_axis_primary",
    ]
    optional = [
        "exact_interval_median_s",
        "hour_mapping_source",
        "hour_mapping_status",
        f"{site}_source_file",
    ]
    columns.extend(column for column in optional if column in frame_table.columns)
    selected = frame_table.loc[frame_table[f"{site}_valid"].astype(bool), columns].copy()
    selected["site_id"] = site
    selected["x_nm"] = frame_table.loc[selected.index, f"{site}_x_nm"].to_numpy(dtype=float)
    selected["y_nm"] = frame_table.loc[selected.index, f"{site}_y_nm"].to_numpy(dtype=float)
    selected["observation_policy"] = "recorded_site_localization_no_interpolation"
    return selected.reset_index(drop=True)


def _shared_observations(frame_table: pd.DataFrame) -> pd.DataFrame:
    selected = frame_table.loc[frame_table["paired_valid"].astype(bool)].copy()
    selected["observation_policy"] = "same_frame_site1_site2_no_interpolation"
    return selected.reset_index(drop=True)


def _site_membership(bundle_metadata: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    identity = [
        "biological_experiment_id",
        "acquisition_id",
        "fov_id",
        "cell_id",
        "bundle_id",
        "allele_index",
        "hour_post_delivery",
        "nominal_frame_count",
        "trajectory_usability_class",
    ]
    for site in ("site1", "site2"):
        part = bundle_metadata[identity].copy()
        part["site_id"] = site
        part["valid_frame_count"] = bundle_metadata[f"{site}_valid_count"].to_numpy(dtype=int)
        part["coverage"] = bundle_metadata[f"{site}_coverage"].to_numpy(dtype=float)
        part["longest_observed_run"] = bundle_metadata[f"longest_{site}_run"].to_numpy(dtype=int)
        part["site_cohort_included"] = part["valid_frame_count"] >= 1
        part["site_cohort_rule"] = "at_least_one_recorded_valid_localization"
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["site_id", "bundle_id"], kind="stable")
        .reset_index(drop=True)
    )


def _summary_row(
    tier_id: str,
    cohort: str,
    unit: str,
    frames: pd.DataFrame,
    bundles: pd.DataFrame | None = None,
) -> dict[str, Any]:
    source = frames if bundles is None else bundles
    return {
        "tier_id": tier_id,
        "cohort": cohort,
        "unit": unit,
        "observation_frame_count": len(frames),
        "nominal_frame_count": (
            int(pd.to_numeric(bundles["nominal_frame_count"], errors="raise").sum())
            if bundles is not None
            else pd.NA
        ),
        "bundle_or_site_trajectory_count": int(source["bundle_id"].astype(str).nunique()),
        "cell_count": int(source["cell_id"].astype(str).nunique()),
        "acquisition_count": int(source["acquisition_id"].astype(str).nunique()),
        "biological_replicate_count": pd.NA,
    }


def _tier_summary(
    shared: pd.DataFrame,
    site1: pd.DataFrame,
    site2: pd.DataFrame,
    relaxed: pd.DataFrame,
    strict: pd.DataFrame,
) -> pd.DataFrame:
    relaxed_pass = relaxed.loc[relaxed["relaxed_qc_included"]]
    strict_pass = strict.loc[strict["strict_qc_included"]]
    rows = [
        _summary_row("T1", "shared_same_frame", "observed_same_frame_pair", shared),
        _summary_row("T2", "site1_independent", "recorded_site_localization", site1),
        _summary_row("T2", "site2_independent", "recorded_site_localization", site2),
        _summary_row(
            "T3",
            "relaxed_paired_trajectory",
            "bundle",
            shared.loc[
                shared["bundle_id"].astype(str).isin(set(relaxed_pass["bundle_id"].astype(str)))
            ],
            relaxed_pass,
        ),
        _summary_row(
            "T4",
            "strict_primary",
            "bundle",
            shared.loc[
                shared["bundle_id"].astype(str).isin(set(strict_pass["bundle_id"].astype(str)))
            ],
            strict_pass,
        ),
    ]
    return pd.DataFrame(rows)


def _census_by_acquisition(
    bundle_metadata: pd.DataFrame,
    shared: pd.DataFrame,
    site1: pd.DataFrame,
    site2: pd.DataFrame,
    relaxed: pd.DataFrame,
    strict: pd.DataFrame,
) -> pd.DataFrame:
    groups = bundle_metadata[["acquisition_id", "hour_post_delivery"]].drop_duplicates()
    definitions = (
        ("T1", "shared_same_frame", shared, None),
        ("T2", "site1_independent", site1, None),
        ("T2", "site2_independent", site2, None),
        (
            "T3",
            "relaxed_paired_trajectory",
            shared,
            relaxed.loc[relaxed["relaxed_qc_included"]],
        ),
        ("T4", "strict_primary", shared, strict.loc[strict["strict_qc_included"]]),
    )
    rows: list[pd.DataFrame] = []
    for tier, cohort, observations, members in definitions:
        if members is not None:
            ids = set(members["bundle_id"].astype(str))
            observations = observations.loc[observations["bundle_id"].astype(str).isin(ids)]
            member_counts = (
                members.groupby(["acquisition_id", "hour_post_delivery"], dropna=False)
                .agg(
                    bundle_count=("bundle_id", "nunique"),
                    cell_count=("cell_id", "nunique"),
                    nominal_frame_count=("nominal_frame_count", "sum"),
                )
                .reset_index()
            )
        else:
            member_counts = (
                observations.groupby(["acquisition_id", "hour_post_delivery"], dropna=False)
                .agg(bundle_count=("bundle_id", "nunique"), cell_count=("cell_id", "nunique"))
                .reset_index()
            )
            member_counts["nominal_frame_count"] = pd.NA
        observed_counts = (
            observations.groupby(["acquisition_id", "hour_post_delivery"], dropna=False)
            .size()
            .rename("observation_frame_count")
            .reset_index()
        )
        block = groups.merge(
            member_counts, on=["acquisition_id", "hour_post_delivery"], how="left"
        ).merge(observed_counts, on=["acquisition_id", "hour_post_delivery"], how="left")
        for column in ("bundle_count", "cell_count", "observation_frame_count"):
            block[column] = block[column].fillna(0).astype(int)
        block.insert(0, "cohort", cohort)
        block.insert(0, "tier_id", tier)
        rows.append(block)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["tier_id", "cohort", "hour_post_delivery", "acquisition_id"], kind="stable"
    )


def _exclusion_counts(
    relaxed: pd.DataFrame,
    strict: pd.DataFrame,
    relaxed_thresholds: Mapping[str, float],
    strict_thresholds: Mapping[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tier, source, included_column, thresholds in (
        ("T3", relaxed, "relaxed_qc_included", relaxed_thresholds),
        ("T4", strict, "strict_qc_included", strict_thresholds),
    ):
        rows.extend(
            [
                {
                    "tier_id": tier,
                    "scope": "overall",
                    "reason": "eligible_formal_bundles",
                    "threshold": pd.NA,
                    "bundle_count": len(source),
                },
                {
                    "tier_id": tier,
                    "scope": "overall",
                    "reason": "included_all_rules_passed",
                    "threshold": pd.NA,
                    "bundle_count": int(source[included_column].sum()),
                },
                {
                    "tier_id": tier,
                    "scope": "overall",
                    "reason": "excluded_one_or_more_rules_failed",
                    "threshold": pd.NA,
                    "bundle_count": int((~source[included_column]).sum()),
                },
            ]
        )
        for name, threshold in thresholds.items():
            rows.append(
                {
                    "tier_id": tier,
                    "scope": "reason_total_nonexclusive",
                    "reason": name,
                    "threshold": threshold,
                    "bundle_count": int(source[f"fails_{name}"].sum()),
                }
            )
    return pd.DataFrame(rows)


def build_tiered_qc(
    frame_table: pd.DataFrame,
    bundle_metadata: pd.DataFrame,
    analysis_config: Mapping[str, Any],
    tier_contract: Mapping[str, Any],
    *,
    strict_reference: pd.DataFrame | None = None,
) -> TieredQcResult:
    """Build four transparent cohort views from formal, exact-time tables."""

    _validate_nontrajectory_tier_semantics(tier_contract)
    _require_source_contract(frame_table, bundle_metadata, analysis_config, tier_contract)
    relaxed_thresholds = _relaxed_thresholds(tier_contract)
    strict_thresholds = _strict_thresholds(analysis_config)

    relaxed = _membership(
        bundle_metadata,
        RELAXED_RULES,
        relaxed_thresholds,
        prefix="relaxed_qc",
    )
    strict = _membership(
        bundle_metadata,
        STRICT_RULES,
        strict_thresholds,
        prefix="strict_qc",
    )
    relaxed_ids = set(relaxed.loc[relaxed["relaxed_qc_included"], "bundle_id"].astype(str))
    strict_ids = set(strict.loc[strict["strict_qc_included"], "bundle_id"].astype(str))
    if not strict_ids.issubset(relaxed_ids):
        raise ValueError("Strict primary QC is not a subset of relaxed trajectory QC")

    strict_reference_match: bool | None = None
    if strict_reference is not None:
        if "bundle_id" not in strict_reference:
            raise ValueError("Strict reference table lacks bundle_id")
        reference_ids = set(strict_reference["bundle_id"].astype(str))
        strict_reference_match = reference_ids == strict_ids
        if not strict_reference_match:
            raise ValueError(
                "Recomputed strict QC does not match the supplied existing primary QC table: "
                f"missing={len(reference_ids - strict_ids)}, extra={len(strict_ids - reference_ids)}"
            )

    shared = _shared_observations(frame_table)
    site1 = _site_observations(frame_table, "site1")
    site2 = _site_observations(frame_table, "site2")
    site_membership = _site_membership(bundle_metadata)
    relaxed_frames = frame_table.loc[frame_table["bundle_id"].astype(str).isin(relaxed_ids)].copy()
    strict_frames = frame_table.loc[frame_table["bundle_id"].astype(str).isin(strict_ids)].copy()

    tier_summary = _tier_summary(shared, site1, site2, relaxed, strict)
    census = _census_by_acquisition(
        bundle_metadata, shared, site1, site2, relaxed, strict
    ).reset_index(drop=True)
    exclusions = _exclusion_counts(relaxed, strict, relaxed_thresholds, strict_thresholds)
    summary = {
        "status": "complete_descriptive_census",
        "source_bundle_count": len(bundle_metadata),
        "source_nominal_frame_count": len(frame_table),
        "tiers": {
            row.cohort: {
                "tier_id": row.tier_id,
                "observation_frame_count": int(row.observation_frame_count),
                "bundle_or_site_trajectory_count": int(row.bundle_or_site_trajectory_count),
                "cell_count": int(row.cell_count),
                "acquisition_count": int(row.acquisition_count),
            }
            for row in tier_summary.itertuples(index=False)
        },
        "relaxed_thresholds": relaxed_thresholds,
        "strict_thresholds": strict_thresholds,
        "strict_reference_supplied": strict_reference is not None,
        "strict_reference_match": strict_reference_match,
        "exact_timestamp_required": True,
        "interpolation_used": False,
        "review_data_included": False,
        "metric_specific_pair_support_deferred": True,
        "independent_biological_replicates_available": False,
        "inference_scope": "technical/descriptive cohort census; higher-level uncertainty groups by acquisition",
    }
    return TieredQcResult(
        shared,
        site1,
        site2,
        site_membership,
        relaxed.reset_index(drop=True),
        relaxed_frames.reset_index(drop=True),
        strict.reset_index(drop=True),
        strict_frames.reset_index(drop=True),
        tier_summary,
        census,
        exclusions,
        summary,
    )


def _report(result: TieredQcResult) -> str:
    tier = result.tier_summary.copy()
    display_columns = [
        "tier_id",
        "cohort",
        "unit",
        "observation_frame_count",
        "bundle_or_site_trajectory_count",
        "cell_count",
        "acquisition_count",
    ]
    header = "| " + " | ".join(display_columns) + " |"
    separator = "|" + "|".join("---" for _ in display_columns) + "|"
    table_rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in tier[display_columns].itertuples(index=False, name=None)
    ]
    lines = [
        "# Four-tier QC census",
        "",
        "This report defines analysis cohorts; it does not score biological quality or make a biological claim.",
        "",
        header,
        separator,
        *table_rows,
        "",
        "## Exact definitions",
        "",
        "- **T1 shared frame:** every recorded frame where Site1 and Site2 are both valid; no bundle-length threshold.",
        "- **T2 independent sites:** every recorded valid Site1 localization and every recorded valid Site2 localization, retained as separate cohorts; the other site is not required.",
        "- **T3 relaxed paired trajectory:** Site1 >= 10 frames, Site2 >= 10 frames, >= 10 same-frame pairs, and longest paired run >= 5 frames. There is no coverage threshold.",
        "- **T4 strict primary:** the existing five rules are unchanged: Site1/Site2/paired >= 20 frames, longest paired run >= 10, and paired coverage >= 0.50.",
        "",
        "## Contracts and caveats",
        "",
        "- Coordinates are copied only from recorded localizations. No interpolation or gap filling is used.",
        "- `micro_time_s` is the exact recorded clock. Nominal frame number is not substituted for time.",
        "- Review trajectories are excluded; only formal-usable source classes are admitted.",
        "- T1/T2 are observation views, whereas T3/T4 are bundle cohorts; they should not be interpreted as a single monotone quality score.",
        "- Valid-pair requirements for a particular MSD/MSCD/VCC lag remain metric-level gates, not hidden cohort exclusions.",
        "- Frame rows are not independent samples. Acquisition is the highest available technical grouping unit and is not a biological replicate.",
        "- No biological `speed` endpoint is defined here. Any future finite-difference displacement-per-time quantity is a technical diagnostic unless separately justified.",
        "",
    ]
    return "\n".join(lines)


def write_tiered_qc(
    result: TieredQcResult,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    tier_contract: Mapping[str, Any],
    input_paths: Mapping[str, str | Path],
    forbidden_roots: tuple[str | Path, ...] = (),
) -> dict[str, Path]:
    """Write an immutable, provenance-bearing tiered-QC result directory."""

    output = validate_analysis_output_dir(
        output_dir,
        project_root=project_root,
        allowed_roots=canonical_output_roots(project_root),
        forbidden_roots=forbidden_roots,
    )
    if output.exists():
        raise FileExistsError(f"Tiered QC output is immutable and already exists: {output}")
    for name, source in input_paths.items():
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Tiered QC input {name} is missing: {path}")
    output.mkdir(parents=True)

    paths = {
        "shared_frame_observations": output / "T1_shared_frame_observations.parquet",
        "site1_observations": output / "T2_site1_independent_observations.parquet",
        "site2_observations": output / "T2_site2_independent_observations.parquet",
        "site_trajectory_membership": output / "T2_site_trajectory_membership.parquet",
        "relaxed_trajectory_membership": output / "T3_relaxed_trajectory_membership.parquet",
        "relaxed_frame_table": output / "T3_relaxed_frame_table.parquet",
        "strict_trajectory_membership": output / "T4_strict_trajectory_membership.parquet",
        "strict_frame_table": output / "T4_strict_frame_table.parquet",
        "tier_summary": output / "tier_summary.csv",
        "tier_census_by_acquisition": output / "tier_census_by_acquisition.csv",
        "exclusion_counts": output / "tier_exclusion_counts.csv",
        "summary": output / "tiered_qc_summary.json",
        "contract": output / "tiered_qc_contract_resolved.yaml",
        "provenance": output / "input_provenance.json",
        "report": output / "TIERED_QC_REPORT.md",
    }
    result.shared_frame_observations.to_parquet(paths["shared_frame_observations"], index=False)
    result.site1_observations.to_parquet(paths["site1_observations"], index=False)
    result.site2_observations.to_parquet(paths["site2_observations"], index=False)
    result.site_trajectory_membership.to_parquet(paths["site_trajectory_membership"], index=False)
    result.relaxed_trajectory_membership.to_parquet(
        paths["relaxed_trajectory_membership"], index=False
    )
    result.relaxed_frame_table.to_parquet(paths["relaxed_frame_table"], index=False)
    result.strict_trajectory_membership.to_parquet(
        paths["strict_trajectory_membership"], index=False
    )
    result.strict_frame_table.to_parquet(paths["strict_frame_table"], index=False)
    result.tier_summary.to_csv(paths["tier_summary"], index=False)
    result.tier_census_by_acquisition.to_csv(paths["tier_census_by_acquisition"], index=False)
    result.exclusion_counts.to_csv(paths["exclusion_counts"], index=False)
    paths["summary"].write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["contract"].write_text(
        yaml.safe_dump(dict(tier_contract), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    provenance = {
        name: {
            "path": str(Path(source).resolve()),
            "sha256": sha256_file(Path(source).resolve()),
        }
        for name, source in sorted(input_paths.items())
    }
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(_report(result), encoding="utf-8", newline="\n")

    manifest_path = output / "artifact_manifest.json"
    manifest = {
        name: {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in sorted(paths.items())
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["artifact_manifest"] = manifest_path
    return paths
