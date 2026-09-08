from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage


@dataclass(frozen=True)
class PointBP1Summary:
    """Point-track evidence with explicit technical-assessability semantics.

    ``bp1_present_bundle`` means that at least one valid point was observed; a
    false value is not a biological-negative call.  Missing input, detection
    failure, assessable non-detection, and unknown technical status remain
    separate.

    Fraction denominators are deliberately strict:

    - ``bp1_detection_fraction`` uses technically evaluable frames only.
    - ``bp1_colocalized_fraction`` uses paired-DNA, technically evaluable
      frames; explicit assessable non-detections contribute zeros.
    - ``bp1_colocalized_fraction_detected_paired`` is proximity conditional on
      a detected BP1 point and paired DNA.
    - ``bp1_detection_fraction_nominal_technical`` is retained only as an
      explicitly named tracking-density statistic over all nominal rows.

    Missing, failed, unassessable, or unknown frames never enter a biological
    fraction denominator. ``bp1_biological_negative`` is deliberately always
    ``None`` because point tracking alone cannot establish that state.
    """

    bp1_present_bundle: bool
    bp1_observed_frames: int
    paired_dna_frames: int
    jointly_observed_frames: int
    bp1_detection_fraction: float
    bp1_colocalized_fraction: float
    bp1_distance_to_pair_median_nm: float
    bp1_distance_to_pair_mean_nm: float
    longest_bp1_run: int
    longest_colocalized_run: int
    bp1_assessment_metadata_available: bool = False
    bp1_assessable_bundle: bool = False
    bp1_assessable_frames: int = 0
    bp1_unassessable_frames: int = 0
    bp1_missing_frames: int = 0
    bp1_detection_failure_frames: int = 0
    bp1_assessable_nondetection_frames: int = 0
    bp1_unknown_status_frames: int = 0
    bp1_detection_fraction_assessable: float = np.nan
    bp1_colocalization_assessable_frames: int = 0
    bp1_colocalized_fraction_assessable: float = np.nan
    bp1_bundle_observation_status: str = "unassessable_or_unknown"
    bp1_biological_negative: bool | None = None
    bp1_evaluable_frames: int = 0
    bp1_paired_evaluable_frames: int = 0
    bp1_detection_fraction_nominal_technical: float = np.nan
    bp1_detected_paired_frames: int = 0
    bp1_colocalized_fraction_detected_paired: float = np.nan


@dataclass(frozen=True)
class ComponentMeasurement:
    component_label: int
    area_px2: int
    equivalent_radius_nm: float
    integrated_intensity: float
    peak_intensity: float
    background_intensity: float
    background_corrected_intensity: float
    signal_to_background: float
    centroid_x_nm: float
    centroid_y_nm: float
    centroid_distance_to_pair_nm: float
    boundary_distance_to_pair_nm: float
    pair_proxy_inside_component: bool


_POINT_STATUS_COLUMNS = (
    "bp1_assessable",
    "bp1_missing",
    "bp1_detection_failed",
)

_COMPONENT_PLACEHOLDERS = frozenset(
    {
        "",
        "example",
        "missing",
        "n/a",
        "na",
        "none",
        "not available",
        "not_available",
        "pending",
        "placeholder",
        "tbd",
        "todo",
        "unknown",
        "unresolved",
    }
)
_COMPONENT_EVIDENCE_REFERENCE = re.compile(
    r"^(?:acquisition_record|analysis_record|laboratory_record|read_only_archive_manifest|"
    r"schema_record|validated_manifest|validation_report):[A-Za-z0-9][A-Za-z0-9_.:/-]+$",
    re.IGNORECASE,
)


def _boolean_column(frame_table: pd.DataFrame, name: str) -> np.ndarray:
    """Return a strict boolean column, rejecting missing or truthy strings."""

    values = frame_table[name].to_numpy(dtype=object)
    result = np.zeros(values.shape, dtype=bool)
    invalid: list[object] = []
    for position, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            result[position] = bool(value)
        elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            if int(value) not in (0, 1):
                invalid.append(frame_table.index[position])
            else:
                result[position] = bool(value)
        elif isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0.0, 1.0):
            result[position] = bool(value)
        else:
            invalid.append(frame_table.index[position])
    if invalid:
        raise ValueError(
            f"{name} must contain only non-missing booleans/0/1; invalid rows: {invalid[:10]}"
        )
    return result


def _coordinate_column(frame_table: pd.DataFrame, name: str) -> np.ndarray:
    try:
        return frame_table[name].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _validate_valid_coordinates(
    frame_table: pd.DataFrame,
    *,
    channel: str,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    invalid = valid & ~(np.isfinite(x) & np.isfinite(y))
    if np.any(invalid):
        rows = frame_table.index[np.flatnonzero(invalid)].tolist()
        raise ValueError(
            f"{channel}_valid=True requires finite {channel}_x_nm and "
            f"{channel}_y_nm coordinates; invalid rows: {rows[:10]}"
        )


def _longest_consecutive_run(frames: np.ndarray) -> int:
    if frames.size == 0:
        return 0
    values = np.unique(np.asarray(frames, dtype=int))
    boundaries = np.flatnonzero(np.diff(values) != 1) + 1
    return int(max(map(len, np.split(values, boundaries))))


def summarize_point_bp1(
    frame_table: pd.DataFrame,
    *,
    colocalization_radius_nm: float,
) -> PointBP1Summary:
    """Summarize point-trajectory 53BP1 relative to the neutral pair midpoint.

    This function quantifies observation, persistence, and geometry only.  A
    missing point trajectory is never interpreted as repair or even biological
    53BP1 negativity.

    Existing frame tables remain valid.  Three optional boolean columns add
    technical provenance without changing the required production schema:
    ``bp1_assessable``, ``bp1_missing``, and ``bp1_detection_failed``.  When
    they are absent, only a valid observed point is known to be assessable and
    every non-detection remains technically unknown.  A detection failure may
    document that an assessment was attempted, but is not technically
    evaluable and is never counted as a biological negative.
    """

    required = {
        "frame",
        "site1_x_nm",
        "site1_y_nm",
        "site1_valid",
        "site2_x_nm",
        "site2_y_nm",
        "site2_valid",
        "bp1_x_nm",
        "bp1_y_nm",
        "bp1_valid",
    }
    missing = required.difference(frame_table.columns)
    if missing:
        raise ValueError(f"Missing point-53BP1 columns: {sorted(missing)}")
    try:
        radius_nm = float(colocalization_radius_nm)
    except (TypeError, ValueError) as error:
        raise ValueError("colocalization_radius_nm must be finite and positive") from error
    if not np.isfinite(radius_nm) or radius_nm <= 0:
        raise ValueError("colocalization_radius_nm must be finite and positive")

    site1 = _boolean_column(frame_table, "site1_valid")
    site2 = _boolean_column(frame_table, "site2_valid")
    bp1 = _boolean_column(frame_table, "bp1_valid")
    site1_x = _coordinate_column(frame_table, "site1_x_nm")
    site1_y = _coordinate_column(frame_table, "site1_y_nm")
    site2_x = _coordinate_column(frame_table, "site2_x_nm")
    site2_y = _coordinate_column(frame_table, "site2_y_nm")
    bp1_x = _coordinate_column(frame_table, "bp1_x_nm")
    bp1_y = _coordinate_column(frame_table, "bp1_y_nm")
    _validate_valid_coordinates(
        frame_table,
        channel="site1",
        valid=site1,
        x=site1_x,
        y=site1_y,
    )
    _validate_valid_coordinates(
        frame_table,
        channel="site2",
        valid=site2,
        x=site2_x,
        y=site2_y,
    )
    _validate_valid_coordinates(
        frame_table,
        channel="bp1",
        valid=bp1,
        x=bp1_x,
        y=bp1_y,
    )

    status_metadata_available = any(column in frame_table for column in _POINT_STATUS_COLUMNS)
    detection_failed = (
        _boolean_column(frame_table, "bp1_detection_failed")
        if "bp1_detection_failed" in frame_table
        else np.zeros(len(frame_table), dtype=bool)
    )
    missing_input = (
        _boolean_column(frame_table, "bp1_missing")
        if "bp1_missing" in frame_table
        else np.zeros(len(frame_table), dtype=bool)
    )
    assessable = (
        _boolean_column(frame_table, "bp1_assessable")
        if "bp1_assessable" in frame_table
        else (bp1 | detection_failed)
    )
    contradictory = {
        "valid_and_unassessable": bp1 & ~assessable,
        "valid_and_missing": bp1 & missing_input,
        "valid_and_detection_failed": bp1 & detection_failed,
        "missing_and_assessable": missing_input & assessable,
        "missing_and_detection_failed": missing_input & detection_failed,
        "failure_and_unassessable": detection_failed & ~assessable,
    }
    conflicts = {
        name: frame_table.index[np.flatnonzero(mask)].tolist()[:10]
        for name, mask in contradictory.items()
        if np.any(mask)
    }
    if conflicts:
        raise ValueError(f"Contradictory 53BP1 technical-status flags: {conflicts}")

    paired = site1 & site2
    joint = paired & bp1
    distance = np.full(len(frame_table), np.nan, dtype=float)
    if joint.any():
        midpoint_x = (site1_x[joint] + site2_x[joint]) / 2.0
        midpoint_y = (site1_y[joint] + site2_y[joint]) / 2.0
        dx = bp1_x[joint] - midpoint_x
        dy = bp1_y[joint] - midpoint_y
        distance[np.flatnonzero(joint)] = np.hypot(dx, dy)
    colocalized = joint & (distance <= radius_nm)
    bp1_count = int(bp1.sum())
    paired_count = int(paired.sum())
    joint_count = int(joint.sum())
    assessable_count = int(assessable.sum())
    missing_count = int(missing_input.sum())
    failure_count = int(detection_failed.sum())
    evaluable = assessable & ~detection_failed & ~missing_input
    evaluable_count = int(evaluable.sum())
    paired_evaluable = paired & evaluable
    paired_evaluable_count = int(paired_evaluable.sum())
    assessable_nondetection = evaluable & ~bp1
    unknown_status = ~bp1 & ~assessable & ~missing_input & ~detection_failed
    if bp1_count:
        observation_status = "point_detected"
    elif failure_count:
        observation_status = "detection_failure_no_point"
    elif np.any(assessable_nondetection):
        observation_status = "assessable_no_point_detected"
    elif missing_count == len(frame_table) and len(frame_table):
        observation_status = "source_missing"
    else:
        observation_status = "unassessable_or_unknown"
    return PointBP1Summary(
        bp1_present_bundle=bp1_count > 0,
        bp1_observed_frames=bp1_count,
        paired_dna_frames=paired_count,
        jointly_observed_frames=joint_count,
        bp1_detection_fraction=(bp1_count / evaluable_count if evaluable_count else np.nan),
        bp1_colocalized_fraction=(
            float(np.sum(colocalized)) / paired_evaluable_count
            if paired_evaluable_count
            else np.nan
        ),
        bp1_distance_to_pair_median_nm=(float(np.nanmedian(distance)) if joint_count else np.nan),
        bp1_distance_to_pair_mean_nm=(float(np.nanmean(distance)) if joint_count else np.nan),
        longest_bp1_run=_longest_consecutive_run(frame_table.loc[bp1, "frame"].to_numpy(int)),
        longest_colocalized_run=_longest_consecutive_run(
            frame_table.loc[colocalized, "frame"].to_numpy(int)
        ),
        bp1_assessment_metadata_available=status_metadata_available,
        bp1_assessable_bundle=assessable_count > 0,
        bp1_assessable_frames=assessable_count,
        bp1_unassessable_frames=int(len(frame_table) - assessable_count),
        bp1_missing_frames=missing_count,
        bp1_detection_failure_frames=failure_count,
        bp1_assessable_nondetection_frames=int(assessable_nondetection.sum()),
        bp1_unknown_status_frames=int(unknown_status.sum()),
        bp1_detection_fraction_assessable=(
            bp1_count / evaluable_count if evaluable_count else np.nan
        ),
        bp1_colocalization_assessable_frames=paired_evaluable_count,
        bp1_colocalized_fraction_assessable=(
            float(np.sum(colocalized)) / paired_evaluable_count
            if paired_evaluable_count
            else np.nan
        ),
        bp1_bundle_observation_status=observation_status,
        bp1_biological_negative=None,
        bp1_evaluable_frames=evaluable_count,
        bp1_paired_evaluable_frames=paired_evaluable_count,
        bp1_detection_fraction_nominal_technical=(
            bp1_count / len(frame_table) if len(frame_table) else np.nan
        ),
        bp1_detected_paired_frames=joint_count,
        bp1_colocalized_fraction_detected_paired=(
            float(np.sum(colocalized)) / joint_count if joint_count else np.nan
        ),
    )


def component_quantification_contract_template() -> dict[str, dict[str, Any]]:
    """Return a fresh, deliberately incomplete image-component contract.

    The template is an interface only.  ``None`` and empty values are blockers,
    not defaults, and copying the template can never ungate real-data analysis
    without explicit validation evidence for every workflow stage.
    """

    required_output_fields = [field.name for field in fields(ComponentMeasurement)]
    return {
        "input": {
            "image_source": None,
            "channel_identity": None,
            "pixel_size_nm": None,
            "read_only_source_confirmed": None,
            "evidence_source": None,
        },
        "qc": {
            "validated": None,
            "frame_exclusion_rules": [],
            "evidence_source": None,
        },
        "registration": {
            "validated": None,
            "transform_definition": None,
            "max_registration_error_nm": None,
            "evidence_source": None,
        },
        "threshold": {
            "validated": None,
            "method": None,
            "parameters": {},
            "background_method": None,
            "evidence_source": None,
        },
        "linking": {
            "validated": None,
            "method": None,
            "parameters": {},
            "evidence_source": None,
        },
        "output_schema": {
            "validated": None,
            "schema_version": None,
            "required_fields": required_output_fields,
            "evidence_source": None,
        },
    }


def _normalized_placeholder_token(value: str) -> str:
    return " ".join(value.strip().casefold().replace("_", " ").replace("-", " ").split())


def _is_placeholder(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        token = _normalized_placeholder_token(value)
        return (
            token in {_normalized_placeholder_token(item) for item in _COMPONENT_PLACEHOLDERS}
            or token in {"fill me", "replace me", "path/to/evidence"}
            or (token.startswith("<") and token.endswith(">"))
        )
    if isinstance(value, Mapping):
        return not value or any(_is_placeholder(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return not value or any(_is_placeholder(item) for item in value)
    return False


def _finite_number(value: object, *, positive: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(numeric):
        return False
    return numeric > 0.0 if positive else numeric >= 0.0


def _valid_component_evidence_reference(value: object) -> bool:
    return isinstance(value, str) and bool(_COMPONENT_EVIDENCE_REFERENCE.fullmatch(value.strip()))


def evaluate_component_quantification_gate(
    contract: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every image-component workflow contract is evidenced.

    This validates metadata only; it does not run image analysis or certify the
    scientific quality of supplied evidence.  Placeholder strings, empty
    sections, and truthy non-booleans never satisfy validation requirements.
    """

    if contract is not None and not isinstance(contract, Mapping):
        raise TypeError("component contract must be a mapping or None")
    candidate: Mapping[str, object] = {} if contract is None else contract
    template = component_quantification_contract_template()
    blockers: list[str] = []
    placeholder_fields: list[str] = []

    sections: dict[str, Mapping[str, object]] = {}
    for section_name in template:
        section = candidate.get(section_name)
        if not isinstance(section, Mapping):
            blockers.append(f"{section_name}:missing_or_invalid_section")
            sections[section_name] = {}
        else:
            sections[section_name] = section

    def require_value(section_name: str, field_name: str) -> object:
        value = sections[section_name].get(field_name)
        if _is_placeholder(value):
            path = f"{section_name}.{field_name}"
            blockers.append(f"{path}:missing_or_placeholder")
            placeholder_fields.append(path)
        return value

    def require_evidence(section_name: str, field_name: str = "evidence_source") -> object:
        value = require_value(section_name, field_name)
        if not _is_placeholder(value) and not _valid_component_evidence_reference(value):
            blockers.append(f"{section_name}.{field_name}:invalid_typed_reference")
        return value

    input_section = sections["input"]
    image_source = require_evidence("input", "image_source")
    channel_identity = require_value("input", "channel_identity")
    pixel_size = require_value("input", "pixel_size_nm")
    read_only = input_section.get("read_only_source_confirmed")
    input_evidence = require_evidence("input")
    if read_only is not True:
        blockers.append("input.read_only_source_confirmed:must_be_true")
    if not _finite_number(pixel_size, positive=True):
        blockers.append("input.pixel_size_nm:must_be_finite_positive")
    if isinstance(channel_identity, str):
        normalized_channel = "".join(
            character for character in channel_identity.casefold() if character.isalnum()
        )
        if normalized_channel not in {"53bp1", "bp1"}:
            blockers.append("input.channel_identity:must_identify_53BP1")
    elif not _is_placeholder(channel_identity):
        blockers.append("input.channel_identity:must_be_text")
    del image_source, input_evidence

    for section_name in ("qc", "registration", "threshold", "linking", "output_schema"):
        if sections[section_name].get("validated") is not True:
            blockers.append(f"{section_name}.validated:must_be_true")
        require_evidence(section_name)

    require_value("qc", "frame_exclusion_rules")

    require_value("registration", "transform_definition")
    registration_error = require_value("registration", "max_registration_error_nm")
    if not _finite_number(registration_error, positive=False):
        blockers.append("registration.max_registration_error_nm:must_be_finite_nonnegative")

    require_value("threshold", "method")
    require_value("threshold", "parameters")
    require_value("threshold", "background_method")

    require_value("linking", "method")
    require_value("linking", "parameters")

    require_value("output_schema", "schema_version")
    output_fields = require_value("output_schema", "required_fields")
    expected_fields = {field.name for field in fields(ComponentMeasurement)}
    if isinstance(output_fields, Sequence) and not isinstance(
        output_fields,
        (str, bytes, bytearray),
    ):
        supplied_fields = {
            value
            for value in output_fields
            if isinstance(value, str) and not _is_placeholder(value)
        }
        missing_fields = sorted(expected_fields.difference(supplied_fields))
        if missing_fields:
            blockers.append("output_schema.required_fields:missing:" + ",".join(missing_fields))
    elif not _is_placeholder(output_fields):
        blockers.append("output_schema.required_fields:must_be_a_sequence")

    blockers = sorted(set(blockers))
    placeholder_fields = sorted(set(placeholder_fields))
    ready = not blockers
    return {
        "status": "ready" if ready else "gated",
        "ready": ready,
        "reason": (
            "All image input, QC, registration, threshold, linking, and output-schema "
            "contracts carry explicit validation evidence."
            if ready
            else (
                "Production final trajectories contain point coordinates only. Component masks, "
                "intensity, background, SBR, and area require an independently validated read-only "
                "image-derived workflow."
            )
        ),
        "blocking_requirements": blockers,
        "placeholder_fields": placeholder_fields,
        "required_sections": list(template),
        "contract_template": deepcopy(template),
        "prohibited_interpretation": "53BP1 trajectory absent does not mean repaired",
    }


def component_quantification_gate(
    contract: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Declare or evaluate the image-derived 53BP1 component gate."""

    return evaluate_component_quantification_gate(contract)


def quantify_local_component(
    image: np.ndarray,
    *,
    pair_proxy_x_px: float,
    pair_proxy_y_px: float,
    pixel_size_nm: float,
    local_radius_px: int = 12,
    threshold_k: float = 0.5,
    nucleus_mask: np.ndarray | None = None,
) -> ComponentMeasurement | None:
    """Quantify the component nearest a pair proxy in one corrected 53BP1 frame.

    This is an analysis-layer primitive, not a production extraction change.
    Real-data use remains gated until threshold/background/linking choices are
    validated against image QC. The function is deterministic and testable on
    synthetic images.
    """

    values = np.asarray(image, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("image must be a finite two-dimensional array")
    if not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be finite and positive")
    if local_radius_px < 2:
        raise ValueError("local_radius_px must be at least two pixels")
    support = np.ones(values.shape, dtype=bool)
    if nucleus_mask is not None:
        support = np.asarray(nucleus_mask, dtype=bool)
        if support.shape != values.shape:
            raise ValueError("nucleus_mask must match image shape")
    yy, xx = np.ogrid[: values.shape[0], : values.shape[1]]
    local = (
        (xx - float(pair_proxy_x_px)) ** 2 + (yy - float(pair_proxy_y_px)) ** 2
        <= local_radius_px**2
    ) & support
    background_values = values[local]
    if background_values.size < 5:
        return None
    background = float(np.median(background_values))
    robust_sigma = float(1.4826 * np.median(np.abs(background_values - background)))
    if robust_sigma == 0:
        robust_sigma = float(np.std(background_values))
    threshold = background + threshold_k * robust_sigma
    binary = (values > threshold) & local
    labels, count = ndimage.label(binary)
    if count == 0:
        return None
    coordinates = np.column_stack(np.nonzero(labels))
    component_labels = labels[coordinates[:, 0], coordinates[:, 1]]
    proxy = np.array([pair_proxy_y_px, pair_proxy_x_px], dtype=float)
    best_label = None
    best_distance = np.inf
    for label in range(1, count + 1):
        component_coordinates = coordinates[component_labels == label]
        distance = float(np.min(np.linalg.norm(component_coordinates - proxy, axis=1)))
        if distance < best_distance:
            best_label = label
            best_distance = distance
    if best_label is None:
        return None
    component = labels == best_label
    area = int(component.sum())
    weighted_center = ndimage.center_of_mass(values, labels, best_label)
    centroid_y, centroid_x = map(float, weighted_center)
    component_values = values[component]
    integrated = float(component_values.sum())
    corrected = float((component_values - background).sum())
    sbr = float(component_values.max() / background) if background > 0 else np.nan
    proxy_y = int(np.clip(round(pair_proxy_y_px), 0, values.shape[0] - 1))
    proxy_x = int(np.clip(round(pair_proxy_x_px), 0, values.shape[1] - 1))
    inside = bool(component[proxy_y, proxy_x])
    boundary_distance = 0.0 if inside else best_distance * pixel_size_nm
    return ComponentMeasurement(
        component_label=int(best_label),
        area_px2=area,
        equivalent_radius_nm=float(np.sqrt(area / np.pi) * pixel_size_nm),
        integrated_intensity=integrated,
        peak_intensity=float(component_values.max()),
        background_intensity=background,
        background_corrected_intensity=corrected,
        signal_to_background=sbr,
        centroid_x_nm=centroid_x * pixel_size_nm,
        centroid_y_nm=centroid_y * pixel_size_nm,
        centroid_distance_to_pair_nm=float(
            np.hypot(centroid_x - pair_proxy_x_px, centroid_y - pair_proxy_y_px) * pixel_size_nm
        ),
        boundary_distance_to_pair_nm=float(boundary_distance),
        pair_proxy_inside_component=inside,
    )
