from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io import load_json, read_tsv, validate_archive_layout

REQUIRED_CONTRACTS = {
    "archive_root": "data.archive_root_resolved",
    "acquisition_id": "hierarchy.acquisition_id_source",
    "fov_id": "hierarchy.fov_id_source",
    "cell_id": "hierarchy.cell_id_source",
    "bundle_id": "hierarchy.bundle_id_source",
    "biological_experiment_id": "hierarchy.biological_experiment_id_source",
    "macro_time_mapping": "time_axes.macro_time_mapping_source",
    "micro_time_axis": "time_axes.micro_time_source",
    "coordinate_units": "coordinates.units",
    "drift_correction": "coordinates.drift_correction_evidence",
    "bp1_dna_coordinate_system": "coordinates.bp1_dna_same_coordinate_system",
    "site1_site2_flank_mapping": "channels.flank_mapping_status",
    "site1_probe_to_cut_bp": "channels.site1_probe_to_cut_bp",
    "site2_probe_to_cut_bp": "channels.site2_probe_to_cut_bp",
}

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED_TOKENS = {
    "",
    "missing",
    "n/a",
    "na",
    "none",
    "not_available",
    "not_provided",
    "null",
    "pending",
    "tbd",
    "unknown",
    "unresolved",
}
TRUE_TOKENS = {"1", "true", "yes", "y"}
EXPECTED_HIERARCHY_SOURCES = {
    "acquisition_id": {"nd2_id", "cell_index.nd2_id"},
    "fov_id": {"fov", "cell_index.fov"},
    "cell_id": {"crop_id", "cell_index.crop_id"},
    "bundle_id": {
        "bundle_id",
        "bundle_level_metrics.bundle_id",
        "nd2_id|crop_id|allele_index",
    },
}
EXPECTED_MICRO_TIME_SOURCE = "exact_crop_metadata.relative_time_s"
EXPECTED_DRIFT_EVIDENCE = "run_manifest_corrected_channels_and_drift_alignment"
EXPECTED_BP1_COORDINATE_EVIDENCE = "production_operational_guide_fixed_corrected_coordinate_system"
BIOLOGICAL_EXPERIMENT_UNAVAILABLE_SOURCE = "unavailable_by_design"
EVIDENCE_PREFIXES = {
    "acquisition_metadata",
    "experimental_design_record",
    "genomic_annotation",
    "laboratory_record",
    "probe_design_record",
    "validated_manifest",
}
GENOMIC_COORDINATE_PATTERN = re.compile(
    r"^(?P<chrom>(?:chr)?[A-Za-z0-9_.-]+):(?P<start>[0-9]+)(?:-(?P<end>[0-9]+))?$"
)


def _get_nested(config: dict, dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _resolved(value: Any) -> bool:
    """Return whether a scalar is more than a placeholder token.

    This helper is intentionally *not* sufficient for M0.  The authoritative
    gate below applies a typed, evidence-backed validator to every contract.
    """

    if value is None or isinstance(value, (dict, list, tuple, set)):
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in UNRESOLVED_TOKENS
    return not (isinstance(value, (float, np.floating)) and math.isnan(float(value)))


def _present_text(value: Any) -> bool:
    return isinstance(value, str) and _resolved(value)


def _finite_positive(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0


def _valid_evidence_reference(value: Any) -> bool:
    """Require a typed provenance reference, not free-form assertion text."""

    if not _present_text(value):
        return False
    prefix, separator, reference = str(value).strip().partition(":")
    return bool(
        separator
        and prefix.casefold() in EVIDENCE_PREFIXES
        and len(reference.strip()) >= 2
        and reference.strip().casefold() not in UNRESOLVED_TOKENS
        and not reference.strip().casefold().startswith("inferred:")
    )


def _parse_genomic_coordinate(value: Any) -> tuple[str, int, int] | None:
    if not isinstance(value, str):
        return None
    match = GENOMIC_COORDINATE_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if start < 1 or end < start:
        return None
    return match.group("chrom").casefold(), start, end


def _resolve_project_path(config: dict, value: Any) -> Path | None:
    if not _present_text(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(config.get("_project_root", Path.cwd())) / path
    return path.resolve()


def _assessment(value: Any, resolved: bool, reason: str, **evidence: Any) -> dict[str, Any]:
    return {
        "value": value,
        "resolved": bool(resolved),
        "reason": reason,
        "evidence": evidence,
    }


def _series_complete(table: pd.DataFrame, column: str) -> bool:
    if column not in table or table.empty:
        return False
    values = table[column]
    text = values.astype("string").str.strip()
    return bool(
        values.notna().all()
        and text.ne("").all()
        and ~text.str.casefold().isin(UNRESOLVED_TOKENS).any()
    )


def _assess_hierarchy(
    config: dict, cell_index: pd.DataFrame, layout_root: Path
) -> dict[str, dict[str, Any]]:
    hierarchy = config.get("hierarchy", {})
    assessments: dict[str, dict[str, Any]] = {}
    columns = {
        "acquisition_id": "nd2_id",
        "fov_id": "fov",
        "cell_id": "crop_id",
    }
    for contract, column in columns.items():
        value = hierarchy.get(f"{contract}_source")
        source_valid = value in EXPECTED_HIERARCHY_SOURCES[contract]
        complete = _series_complete(cell_index, column)
        assessments[contract] = _assessment(
            value,
            source_valid and complete,
            (
                f"verified archive cell_index.{column} coverage"
                if source_valid and complete
                else f"requires a recognized source and complete archive cell_index.{column}"
            ),
            formal_usable_rows=len(cell_index),
            nonmissing_rows=int(cell_index[column].notna().sum()) if column in cell_index else 0,
        )

    bundle_source = hierarchy.get("bundle_id_source")
    bundle_source_valid = bundle_source in EXPECTED_HIERARCHY_SOURCES["bundle_id"]
    bundle_path = _resolve_project_path(config, config.get("data", {}).get("bundle_inventory"))
    if bundle_path is None or not bundle_path.is_file():
        canonical = (
            layout_root / "provenance_v5.1" / "fullrun_inventory" / "bundle_level_metrics.tsv"
        )
        bundle_path = canonical if canonical.is_file() else bundle_path
    bundle_valid = False
    bundle_count = 0
    bundle_reason = "authoritative bundle inventory is missing"
    if bundle_path is not None and bundle_path.is_file():
        try:
            bundles = read_tsv(bundle_path)
            required = {"bundle_id", "nd2_id", "crop_id", "trajectory_artifacts_usable"}
            if required.issubset(bundles.columns):
                usable = bundles.loc[
                    bundles["trajectory_artifacts_usable"]
                    .astype("string")
                    .str.strip()
                    .str.casefold()
                    .isin(TRUE_TOKENS)
                ].copy()
                bundle_count = len(usable)
                formal_keys = set(
                    cell_index.loc[:, ["nd2_id", "crop_id"]]
                    .astype("string")
                    .itertuples(index=False, name=None)
                )
                bundle_keys = set(
                    usable.loc[:, ["nd2_id", "crop_id"]]
                    .astype("string")
                    .itertuples(index=False, name=None)
                )
                bundle_valid = bool(
                    not usable.empty
                    and _series_complete(usable, "bundle_id")
                    and not usable["bundle_id"].duplicated().any()
                    and bundle_keys == formal_keys
                )
                bundle_reason = (
                    "verified unique authoritative bundle IDs with complete formal-crop coverage"
                    if bundle_valid
                    else "bundle IDs are missing, duplicated, or do not cover exactly the formal crops"
                )
            else:
                bundle_reason = (
                    f"bundle inventory lacks columns: {sorted(required - set(bundles.columns))}"
                )
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            bundle_reason = f"bundle inventory could not be validated: {type(exc).__name__}"
    assessments["bundle_id"] = _assessment(
        bundle_source,
        bundle_source_valid and bundle_valid,
        bundle_reason
        if bundle_source_valid
        else "bundle_id_source is not a recognized authoritative source",
        bundle_inventory=str(bundle_path) if bundle_path is not None else None,
        bundle_count=bundle_count,
    )
    return assessments


def _assess_exact_micro_time(config: dict, cell_index: pd.DataFrame, archive_root: Path) -> dict:
    source = _get_nested(config, "time_axes.micro_time_source")
    if (
        source != EXPECTED_MICRO_TIME_SOURCE
        or config.get("data", {}).get("use_exact_timestamps") is not True
    ):
        return _assessment(
            source,
            False,
            "micro-time requires the exact per-crop relative_time_s source and use_exact_timestamps=true",
        )
    invalid: list[str] = []
    checked = 0
    for row in cell_index.itertuples(index=False):
        crop_root = archive_root / str(row.archive_relative_path)
        metadata_path = crop_root / f"{row.crop_id}_metadata.json"
        try:
            metadata = load_json(metadata_path)
            raw = metadata.get("time", {}).get("relative_time_s")
            values = np.asarray(raw, dtype=float)
            expected = int(row.total_frames)
            valid = bool(
                raw is not None
                and values.ndim == 1
                and values.size == expected
                and np.isfinite(values).all()
                and (values.size == 1 or np.all(np.diff(values) > 0))
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            valid = False
        checked += 1
        if not valid and len(invalid) < 10:
            invalid.append(str(metadata_path))
    resolved = bool(checked and not invalid)
    return _assessment(
        source,
        resolved,
        (
            "all formal crops have finite, strictly increasing exact timestamps of nominal length"
            if resolved
            else "one or more formal crops lack a valid exact timestamp vector"
        ),
        formal_crops_checked=checked,
        invalid_examples=invalid,
    )


def _assess_drift(config: dict, cell_index: pd.DataFrame, archive_root: Path) -> dict:
    value = _get_nested(config, "coordinates.drift_correction_evidence")
    if value != EXPECTED_DRIFT_EVIDENCE:
        return _assessment(
            value,
            False,
            "drift evidence must name the locked run-manifest plus drift-alignment contract",
        )
    missing: list[str] = []
    for row in cell_index.itertuples(index=False):
        qc_root = archive_root / str(row.archive_relative_path) / "spt_results" / "qc"
        required = (
            qc_root / "run_manifest.json",
            qc_root / "mask_alignment" / "drift_alignment.csv",
        )
        for path in required:
            if not path.is_file() and len(missing) < 10:
                missing.append(str(path))
    return _assessment(
        value,
        not missing and not cell_index.empty,
        (
            "locked run manifests and drift-alignment tables cover all formal crops"
            if not missing and not cell_index.empty
            else "drift provenance files are incomplete"
        ),
        formal_crops_checked=len(cell_index),
        missing_examples=missing,
    )


def _assess_authoritative_metadata(config: dict, cell_index: pd.DataFrame) -> dict[str, dict]:
    hierarchy = config.get("hierarchy", {})
    time_axes = config.get("time_axes", {})
    path = _resolve_project_path(config, hierarchy.get("m0_metadata_path"))
    base_evidence: dict[str, Any] = {"mapping_path": str(path) if path else None}
    bio_source = hierarchy.get("biological_experiment_id_source")
    hour_source = time_axes.get("macro_time_mapping_source")
    bio_source_valid = bio_source == "m0_metadata.biological_experiment_id"
    bio_unavailable_by_design = bio_source == BIOLOGICAL_EXPERIMENT_UNAVAILABLE_SOURCE
    unavailable_design_valid = bool(
        bio_unavailable_by_design
        and hierarchy.get("analysis_group_id_source") == "acquisition_id"
        and hierarchy.get("independent_biological_replicates_available") is False
        and str(hierarchy.get("biological_population_inference", "")).strip().casefold()
        == "prohibited"
    )
    hour_source_valid = hour_source == "m0_metadata.hour_post_delivery"
    if path is None or not path.is_file():
        return {
            "biological_experiment_id": _assessment(
                bio_source,
                False,
                "authoritative M0 metadata CSV is not configured",
                **base_evidence,
            ),
            "macro_time_mapping": _assessment(
                hour_source,
                False,
                "authoritative M0 metadata CSV is not configured",
                **base_evidence,
            ),
        }
    required = {
        "nd2_id",
        "fov_id",
        "hour_post_delivery",
        "hour_mapping_evidence",
        "hour_mapping_confirmed",
        "biological_experiment_id",
        "biological_experiment_evidence",
        "biological_experiment_confirmed",
    }
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        reason = f"authoritative M0 metadata CSV is unreadable: {type(exc).__name__}"
        return {
            "biological_experiment_id": _assessment(bio_source, False, reason, **base_evidence),
            "macro_time_mapping": _assessment(hour_source, False, reason, **base_evidence),
        }
    missing = required - set(table.columns)
    if missing:
        reason = f"authoritative M0 metadata CSV lacks columns: {sorted(missing)}"
        return {
            "biological_experiment_id": _assessment(bio_source, False, reason, **base_evidence),
            "macro_time_mapping": _assessment(hour_source, False, reason, **base_evidence),
        }
    observed_keys = set(
        table.loc[:, ["nd2_id", "fov_id"]].astype("string").itertuples(index=False, name=None)
    )
    expected_keys = set(
        cell_index.loc[:, ["nd2_id", "fov"]].astype("string").itertuples(index=False, name=None)
    )
    complete_coverage = bool(
        expected_keys
        and observed_keys == expected_keys
        and not table.duplicated(["nd2_id", "fov_id"]).any()
    )
    bio_complete = _series_complete(table, "biological_experiment_id") and bool(
        table["biological_experiment_id"].map(_present_text).all()
    )
    bio_values = table["biological_experiment_id"]
    bio_absent_as_declared = bool(
        bio_values.map(
            lambda value: pd.isna(value) or str(value).strip().casefold() in UNRESOLVED_TOKENS
        ).all()
    )
    hour_values = pd.to_numeric(table["hour_post_delivery"], errors="coerce")
    allowed = {float(value) for value in time_axes.get("allowed_hours", [])}
    hour_numeric = bool(hour_values.notna().all() and np.isfinite(hour_values).all())
    hour_allowed = bool(hour_numeric and allowed and hour_values.map(float).isin(allowed).all())
    hour_evidence = table["hour_mapping_evidence"].astype("string").str.strip()
    hour_evidence_valid = bool(hour_evidence.map(_valid_evidence_reference).all())
    hour_confirmed = bool(
        table["hour_mapping_confirmed"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .isin(TRUE_TOKENS)
        .all()
    )
    experiment_evidence = table["biological_experiment_evidence"].astype("string").str.strip()
    experiment_evidence_valid = bool(experiment_evidence.map(_valid_evidence_reference).all())
    experiment_confirmed = bool(
        table["biological_experiment_confirmed"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .isin(TRUE_TOKENS)
        .all()
    )
    # Acquisition is the minimum scientific grouping unit. Multiple FOV rows
    # may describe it, but they may not disagree on hour, experiment, or the
    # provenance supporting either assignment.
    per_acquisition = table.assign(_hour_numeric=hour_values).groupby("nd2_id", dropna=False)
    acquisition_consistent = bool(
        (per_acquisition["_hour_numeric"].nunique(dropna=False) <= 1).all()
        and (per_acquisition["biological_experiment_id"].nunique(dropna=False) <= 1).all()
        and (per_acquisition["hour_mapping_evidence"].nunique(dropna=False) <= 1).all()
        and (per_acquisition["biological_experiment_evidence"].nunique(dropna=False) <= 1).all()
    )
    shared = {
        **base_evidence,
        "expected_acquisition_fov_pairs": len(expected_keys),
        "mapped_acquisition_fov_pairs": len(observed_keys),
        "complete_coverage": complete_coverage,
        "acquisition_level_assignments_consistent": acquisition_consistent,
    }
    bio_ok = bool(
        complete_coverage
        and experiment_evidence_valid
        and experiment_confirmed
        and acquisition_consistent
        and (
            bio_source_valid and bio_complete or unavailable_design_valid and bio_absent_as_declared
        )
    )
    hour_ok = (
        hour_source_valid
        and complete_coverage
        and hour_numeric
        and hour_allowed
        and hour_evidence_valid
        and hour_confirmed
        and acquisition_consistent
    )
    return {
        "biological_experiment_id": _assessment(
            bio_source,
            bio_ok,
            (
                "biological experiment/day/replicate IDs are explicitly unavailable by design; acquisition_id is the highest technical grouping unit and biological-population inference is prohibited"
                if bio_ok and bio_unavailable_by_design
                else "authoritative biological-experiment IDs cover every formal acquisition/FOV"
                if bio_ok
                else "requires complete acquisition-consistent experiment IDs with confirmed typed evidence"
            ),
            **shared,
            unavailable_by_design=bio_unavailable_by_design,
            analysis_group_id_source=hierarchy.get("analysis_group_id_source"),
            independent_biological_replicates_available=hierarchy.get(
                "independent_biological_replicates_available"
            ),
            biological_population_inference=hierarchy.get("biological_population_inference"),
        ),
        "macro_time_mapping": _assessment(
            hour_source,
            hour_ok,
            (
                "confirmed authoritative hours cover every formal acquisition/FOV"
                if hour_ok
                else "requires complete acquisition-consistent allowed hours with confirmed typed evidence"
            ),
            **shared,
        ),
    }


def load_authoritative_m0_metadata(
    config: dict, cell_index: pd.DataFrame
) -> tuple[pd.DataFrame, Path]:
    """Load the typed, complete acquisition/FOV map after the M0 checks pass.

    This is the only interface allowed to promote external hour and biological
    experiment metadata into the analysis snapshot.
    """

    assessments = _assess_authoritative_metadata(config, cell_index)
    failed = {
        name: assessment["reason"]
        for name, assessment in assessments.items()
        if not assessment["resolved"]
    }
    if failed:
        raise ValueError(f"Authoritative M0 metadata failed validation: {failed}")
    path = _resolve_project_path(config, config.get("hierarchy", {}).get("m0_metadata_path"))
    if path is None:  # pragma: no cover - guarded by the assessment above
        raise ValueError("Authoritative M0 metadata path is absent")
    table = pd.read_csv(path)
    columns = [
        "nd2_id",
        "fov_id",
        "hour_post_delivery",
        "hour_mapping_evidence",
        "hour_mapping_confirmed",
        "biological_experiment_id",
        "biological_experiment_evidence",
        "biological_experiment_confirmed",
    ]
    optional = [column for column in ("imaging_day", "biological_replicate") if column in table]
    return table.loc[:, columns + optional].copy(), path


def _assess_genomic_mapping(config: dict) -> dict[str, dict]:
    channels = config.get("channels", {})
    mapping_path = _resolve_project_path(config, channels.get("genomic_mapping_path"))
    flank_value = channels.get("flank_mapping_status")
    distance_values = {
        "site1_probe_to_cut_bp": channels.get("site1_probe_to_cut_bp"),
        "site2_probe_to_cut_bp": channels.get("site2_probe_to_cut_bp"),
    }
    if mapping_path is None or not mapping_path.is_file():
        return {
            "site1_site2_flank_mapping": _assessment(
                flank_value,
                False,
                "an authoritative genomic mapping YAML is not configured",
                mapping_path=str(mapping_path) if mapping_path else None,
            ),
            **{
                contract: _assessment(
                    value,
                    False,
                    "an authoritative genomic mapping YAML is not configured",
                    mapping_path=str(mapping_path) if mapping_path else None,
                )
                for contract, value in distance_values.items()
            },
        }
    try:
        payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        payload = None
        load_reason = f"genomic mapping YAML is unreadable: {type(exc).__name__}"
    else:
        load_reason = ""
    if not isinstance(payload, dict):
        reason = load_reason or "genomic mapping YAML must contain a mapping"
        return {
            "site1_site2_flank_mapping": _assessment(flank_value, False, reason),
            **{
                contract: _assessment(value, False, reason)
                for contract, value in distance_values.items()
            },
        }
    site1 = payload.get("site1") if isinstance(payload.get("site1"), dict) else {}
    site2 = payload.get("site2") if isinstance(payload.get("site2"), dict) else {}
    cut = payload.get("cut_site") if isinstance(payload.get("cut_site"), dict) else {}
    language = (
        payload.get("language_gate") if isinstance(payload.get("language_gate"), dict) else {}
    )
    side1 = str(site1.get("genomic_side_of_cut", "")).strip().casefold()
    side2 = str(site2.get("genomic_side_of_cut", "")).strip().casefold()
    opposing = {side1, side2} in ({"left", "right"}, {"upstream", "downstream"})
    left = str(channels.get("site_left_source", "")).strip().casefold()
    right = str(channels.get("site_right_source", "")).strip().casefold()
    distinct_sources = {left, right} == {"site1", "site2"}
    site_by_name = {"site1": site1, "site2": site2}
    declared_side_by_name = {"site1": side1, "site2": side2}
    site1_coordinate = _parse_genomic_coordinate(site1.get("genomic_coordinates"))
    site2_coordinate = _parse_genomic_coordinate(site2.get("genomic_coordinates"))
    cut_coordinate = _parse_genomic_coordinate(cut.get("genomic_coordinate"))
    coordinates_parse = bool(site1_coordinate and site2_coordinate and cut_coordinate)
    coordinate_order = False
    if coordinates_parse:
        assert site1_coordinate is not None
        assert site2_coordinate is not None
        assert cut_coordinate is not None
        same_chromosome = len({site1_coordinate[0], site2_coordinate[0], cut_coordinate[0]}) == 1
        left_coordinate = _parse_genomic_coordinate(
            site_by_name.get(left, {}).get("genomic_coordinates")
        )
        right_coordinate = _parse_genomic_coordinate(
            site_by_name.get(right, {}).get("genomic_coordinates")
        )
        coordinate_order = bool(
            same_chromosome
            and left_coordinate
            and right_coordinate
            and left_coordinate[2] < cut_coordinate[1]
            and cut_coordinate[2] < right_coordinate[1]
        )
    if {side1, side2} == {"left", "right"}:
        directional_sources = bool(
            distinct_sources
            and declared_side_by_name.get(left) == "left"
            and declared_side_by_name.get(right) == "right"
        )
    else:
        # Upstream/downstream depends on strand orientation. Left/right naming
        # is therefore verified from genomic coordinate order, not by assuming
        # upstream always has the smaller coordinate.
        directional_sources = distinct_sources and coordinate_order
    documented = bool(
        coordinates_parse
        and coordinate_order
        and _valid_evidence_reference(site1.get("evidence_source"))
        and _valid_evidence_reference(site2.get("evidence_source"))
        and _present_text(cut.get("genome_build"))
        and _valid_evidence_reference(cut.get("evidence_source"))
    )
    flank_ok = bool(
        str(flank_value).strip().casefold() == "verified"
        and opposing
        and directional_sources
        and documented
        and language.get("allow_left_right_names") is True
    )
    assessments = {
        "site1_site2_flank_mapping": _assessment(
            flank_value,
            flank_ok,
            (
                "verified opposing Site1/Site2 mapping with distinct left/right sources and evidence"
                if flank_ok
                else "requires verified directionally consistent sources, parseable ordered coordinates, and typed evidence"
            ),
            mapping_path=str(mapping_path),
            site1_side=side1 or None,
            site2_side=side2 or None,
            site_left_source=left or None,
            site_right_source=right or None,
            directional_sources=directional_sources,
            genomic_coordinate_order_valid=coordinate_order,
        )
    }
    for contract, site in (
        ("site1_probe_to_cut_bp", site1),
        ("site2_probe_to_cut_bp", site2),
    ):
        configured = distance_values[contract]
        mapped = site.get("probe_to_cut_distance_bp")
        definition = str(site.get("probe_to_cut_distance_definition", "")).strip().casefold()
        site_coordinate = _parse_genomic_coordinate(site.get("genomic_coordinates"))
        coordinate_distance: float | None = None
        if (
            site_coordinate is not None
            and cut_coordinate is not None
            and cut_coordinate[1] == cut_coordinate[2]
        ):
            if definition == "probe_interval_midpoint_to_cut":
                midpoint = (site_coordinate[1] + site_coordinate[2]) / 2.0
                coordinate_distance = abs(midpoint - cut_coordinate[1])
            elif definition == "nearest_probe_edge_to_cut":
                coordinate_distance = min(
                    abs(site_coordinate[1] - cut_coordinate[1]),
                    abs(site_coordinate[2] - cut_coordinate[1]),
                )
        matching = bool(
            _finite_positive(configured)
            and _finite_positive(mapped)
            and math.isclose(float(configured), float(mapped), rel_tol=0.0, abs_tol=1e-9)
            and coordinate_distance is not None
            and math.isclose(float(mapped), coordinate_distance, rel_tol=0.0, abs_tol=1e-9)
        )
        site_documented = bool(
            documented and _valid_evidence_reference(site.get("evidence_source"))
        )
        allow_nominal_distance = bool(
            language.get("allow_probe_to_nominal_cut_distance") is True
            or language.get("allow_broken_end_distance") is True
        )
        ok = matching and site_documented and flank_ok and allow_nominal_distance
        assessments[contract] = _assessment(
            configured,
            ok,
            (
                "finite positive probe-to-cut distance matches the documented genomic mapping"
                if ok
                else "requires verified flanks, explicit nominal-distance permission, and a coordinate-derived documented distance"
            ),
            mapping_path=str(mapping_path),
            mapped_distance_bp=mapped,
            distance_definition=definition or None,
            coordinate_derived_distance_bp=coordinate_distance,
            allow_broken_end_distance=language.get("allow_broken_end_distance"),
            allow_probe_to_nominal_cut_distance=language.get("allow_probe_to_nominal_cut_distance"),
        )
    return assessments


def _typed_contract_assessments(
    config: dict,
    *,
    archive_root: Path,
    archive_issues: list[dict],
    cell_index: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    assessments: dict[str, dict[str, Any]] = {}
    archive_errors = [issue for issue in archive_issues if issue.get("severity") == "error"]
    configured_archive = _resolve_project_path(
        config, _get_nested(config, "data.archive_root_resolved")
    )
    archive_ok = bool(
        configured_archive == archive_root.resolve()
        and archive_root.is_dir()
        and not archive_errors
    )
    assessments["archive_root"] = _assessment(
        str(configured_archive) if configured_archive else None,
        archive_ok,
        "archive layout/version contract verified"
        if archive_ok
        else "archive layout/version failed",
    )
    assessments.update(_assess_hierarchy(config, cell_index, archive_root))
    assessments.update(_assess_authoritative_metadata(config, cell_index))
    assessments["micro_time_axis"] = _assess_exact_micro_time(config, cell_index, archive_root)
    unit = _get_nested(config, "coordinates.units")
    assessments["coordinate_units"] = _assessment(
        unit,
        isinstance(unit, str) and unit.strip().casefold() == "nm",
        "coordinates are explicitly nanometres"
        if str(unit).strip().casefold() == "nm"
        else "coordinates.units must be exactly nm",
    )
    assessments["drift_correction"] = _assess_drift(config, cell_index, archive_root)
    same_coordinates = _get_nested(config, "coordinates.bp1_dna_same_coordinate_system")
    coordinate_evidence = _get_nested(config, "coordinates.bp1_dna_coordinate_evidence")
    bp1_ok = bool(
        same_coordinates is True and coordinate_evidence == EXPECTED_BP1_COORDINATE_EVIDENCE
    )
    assessments["bp1_dna_coordinate_system"] = _assessment(
        same_coordinates,
        bp1_ok,
        (
            "BP1 and DNA are documented in the locked corrected coordinate system"
            if bp1_ok
            else "requires true co-registration plus the recognized production evidence contract"
        ),
        coordinate_evidence=coordinate_evidence,
    )
    assessments.update(_assess_genomic_mapping(config))
    return assessments


def audit_m0(config: dict) -> dict:
    layout, archive_issues = validate_archive_layout(config["data"]["archive_root_resolved"])
    summary = load_json(layout.archive_summary) if layout.archive_summary.exists() else {}
    manifest_reconciliation: dict = {}
    if layout.archive_manifest.exists():
        manifest = read_tsv(layout.archive_manifest)
        category_counts = manifest["category"].value_counts().sort_index().to_dict()
        summary_counts = dict(sorted(summary.get("category_counts", {}).items()))
        declared_bytes = int(manifest["bytes"].sum())
        representative_checks = []
        for category, block in manifest.groupby("category", sort=True):
            row = block.iloc[0]
            local = layout.archive_root / str(row["archive_relative_path"])
            representative_checks.append(
                {
                    "category": category,
                    "path": str(local),
                    "exists": local.exists(),
                    "declared_bytes": int(row["bytes"]),
                    "local_bytes": local.stat().st_size if local.exists() else None,
                    "size_matches": local.exists() and local.stat().st_size == int(row["bytes"]),
                }
            )
        hashes = manifest["sha256"].astype(str)
        manifest_reconciliation = {
            "row_count": len(manifest),
            "row_count_matches_summary": len(manifest) == summary.get("file_count"),
            "declared_bytes": declared_bytes,
            "bytes_match_summary": declared_bytes == summary.get("bytes"),
            "category_counts": category_counts,
            "category_counts_match_summary": category_counts == summary_counts,
            "sha256_value_count": int(
                hashes.map(lambda value: bool(SHA256_PATTERN.fullmatch(value))).sum()
            ),
            "deferred_hash_count": int((hashes == "DEFERRED_TO_RCLONE_CHECK").sum()),
            "unexpected_hash_value_count": int(
                (
                    ~hashes.map(lambda value: bool(SHA256_PATTERN.fullmatch(value)))
                    & (hashes != "DEFERRED_TO_RCLONE_CHECK")
                ).sum()
            ),
            "materialization_values": sorted(manifest["materialization"].astype(str).unique()),
            "representative_category_file_checks": representative_checks,
            "representative_checks_all_pass": all(
                item["exists"] and item["size_matches"] for item in representative_checks
            ),
            "scope_note": (
                "Manifest reconciliation and one file-stat check per category; no exhaustive "
                "330-GiB content rehash was performed."
            ),
        }
        for check, passed in (
            ("ARCHIVE_MANIFEST_ROW_COUNT", manifest_reconciliation["row_count_matches_summary"]),
            ("ARCHIVE_MANIFEST_BYTES", manifest_reconciliation["bytes_match_summary"]),
            (
                "ARCHIVE_MANIFEST_CATEGORY_COUNTS",
                manifest_reconciliation["category_counts_match_summary"],
            ),
            (
                "ARCHIVE_REPRESENTATIVE_FILE_STATS",
                manifest_reconciliation["representative_checks_all_pass"],
            ),
        ):
            if not passed:
                archive_issues.append(
                    {"severity": "error", "code": check, "message": "Archive reconciliation failed"}
                )

    if layout.cell_index.exists():
        all_cells = read_tsv(layout.cell_index)
        cell_index = all_cells.loc[
            all_cells["trajectory_artifacts_usable"].astype(str) == "1"
        ].copy()
        cell_counts = {
            "rows": len(all_cells),
            "nd2": int(all_cells["nd2_id"].nunique()),
            "fov": int(all_cells["fov"].nunique()),
            "formal_usable": len(cell_index),
        }
    else:
        cell_index = pd.DataFrame(
            columns=["nd2_id", "fov", "crop_id", "total_frames", "archive_relative_path"]
        )
        cell_counts = {}

    blocking = [issue for issue in archive_issues if issue["severity"] == "error"]
    assessments = _typed_contract_assessments(
        config,
        archive_root=layout.archive_root,
        archive_issues=archive_issues,
        cell_index=cell_index,
    )
    contracts = []
    for name, source in REQUIRED_CONTRACTS.items():
        assessment = assessments[name]
        contracts.append(
            {
                "contract": name,
                "source": source,
                "value": assessment["value"],
                "status": "resolved" if assessment["resolved"] else "unresolved",
                "reason": assessment["reason"],
                "evidence": assessment["evidence"],
            }
        )
    unresolved = [item["contract"] for item in contracts if item["status"] == "unresolved"]
    # M0 is fail-closed: every required contract participates in the gate.
    # Archive problems also appear through the typed archive_root contract.
    biological_blockers = list(dict.fromkeys(unresolved))
    return {
        "milestone": "M0",
        "archive_status": "pass" if not blocking else "fail",
        "scientific_status": "stop" if blocking or biological_blockers else "go",
        "archive_root": str(layout.archive_root),
        "archive_summary": summary,
        "cell_index_counts": cell_counts,
        "archive_manifest_reconciliation": manifest_reconciliation,
        "contracts": contracts,
        "issues": archive_issues,
        "blocking_contracts": biological_blockers,
        "allowed_while_stopped": [
            "schema_and_validation",
            "synthetic_tests",
            "trajectory_snapshot",
            "neutral_site1_site2_qc",
            "gated_interface_scaffolds",
        ],
        "prohibited_while_stopped": [
            "left_right_flank_labels",
            "broken_end_language",
            "macro_time_biological_modeling_without_authoritative_hour_map",
            "grouped_prediction_claiming_biological_experiment_holdout",
            "real_data_state_interpretation",
        ],
    }


def write_m0_reports(report: dict, json_path: str | Path, markdown_path: str | Path) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# M0 data-readiness report",
        "",
        f"- Archive status: **{report['archive_status'].upper()}**",
        f"- Scientific status: **{report['scientific_status'].upper()}**",
        f"- Archive root: `{report['archive_root']}`",
    ]
    reconciliation = report.get("archive_manifest_reconciliation", {})
    if reconciliation:
        lines.extend(
            [
                f"- Archive manifest rows: {reconciliation['row_count']:,} (summary match: {reconciliation['row_count_matches_summary']})",
                f"- Declared bytes: {reconciliation['declared_bytes']:,} (summary match: {reconciliation['bytes_match_summary']})",
                f"- SHA-256 values / deferred checks: {reconciliation['sha256_value_count']:,} / {reconciliation['deferred_hash_count']:,}",
                f"- One local file/size check per category: {reconciliation['representative_checks_all_pass']}",
                f"- Verification scope: {reconciliation['scope_note']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Contract status",
            "",
            "| Contract | Status | Config source | Value | Evidence-based assessment |",
            "|---|---|---|---|---|",
        ]
    )
    for item in report["contracts"]:
        reason = str(item.get("reason", "")).replace("|", "\\|")
        lines.append(
            f"| {item['contract']} | {item['status']} | `{item['source']}` | "
            f"`{item['value']}` | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Blocking contracts",
            "",
        ]
    )
    if report["blocking_contracts"]:
        lines.extend(f"- `{value}`" for value in report["blocking_contracts"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            (
                "The STOP gate forbids biological left/right interpretation, broken-end language, "
                "macro-time biological modeling without an authoritative map, and claims of "
                "biological-experiment generalization. Neutral Site1/Site2 QC and synthetic method "
                "validation remain allowed."
            ),
            "",
        ]
    )
    markdown_output.write_text("\n".join(lines), encoding="utf-8")
