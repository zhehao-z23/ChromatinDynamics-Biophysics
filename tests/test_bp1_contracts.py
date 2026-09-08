from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from dsb_states.bp1 import (
    component_quantification_contract_template,
    component_quantification_gate,
    evaluate_component_quantification_gate,
    summarize_point_bp1,
)


def _point_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [1, 2, 3, 4],
            "site1_x_nm": [0.0, 0.0, 0.0, 0.0],
            "site1_y_nm": [0.0, 0.0, 0.0, 0.0],
            "site1_valid": [True, True, True, True],
            "site2_x_nm": [2.0, 2.0, 2.0, 2.0],
            "site2_y_nm": [0.0, 0.0, 0.0, 0.0],
            "site2_valid": [True, True, True, True],
            "bp1_x_nm": [1.0, np.nan, np.nan, np.nan],
            "bp1_y_nm": [0.0, np.nan, np.nan, np.nan],
            "bp1_valid": [True, False, False, False],
        }
    )


@pytest.mark.parametrize("coordinate", ["bp1_x_nm", "bp1_y_nm"])
def test_valid_bp1_requires_finite_coordinates(coordinate: str) -> None:
    frame = _point_frame()
    frame.loc[0, coordinate] = np.inf

    with pytest.raises(ValueError, match="bp1_valid=True requires finite"):
        summarize_point_bp1(frame, colocalization_radius_nm=100.0)


@pytest.mark.parametrize(
    ("channel", "coordinate"),
    [
        ("site1", "site1_x_nm"),
        ("site1", "site1_y_nm"),
        ("site2", "site2_x_nm"),
        ("site2", "site2_y_nm"),
    ],
)
def test_valid_dna_sites_require_finite_coordinates(channel: str, coordinate: str) -> None:
    frame = _point_frame()
    frame.loc[2, coordinate] = np.nan

    with pytest.raises(ValueError, match=rf"{channel}_valid=True requires finite"):
        summarize_point_bp1(frame, colocalization_radius_nm=100.0)


def test_unassessable_point_absence_never_becomes_zero_colocalization_or_negative() -> None:
    frame = _point_frame()
    frame["bp1_valid"] = False
    frame[["bp1_x_nm", "bp1_y_nm"]] = np.nan

    summary = summarize_point_bp1(frame, colocalization_radius_nm=100.0)

    assert not summary.bp1_present_bundle
    assert not summary.bp1_assessable_bundle
    assert not summary.bp1_assessment_metadata_available
    assert summary.bp1_unknown_status_frames == 4
    assert summary.bp1_bundle_observation_status == "unassessable_or_unknown"
    assert summary.bp1_biological_negative is None
    assert np.isnan(summary.bp1_colocalized_fraction)
    assert np.isnan(summary.bp1_colocalized_fraction_assessable)
    assert np.isnan(summary.bp1_detection_fraction_assessable)


def test_optional_technical_flags_separate_missing_failure_and_nondetection() -> None:
    frame = _point_frame()
    frame["bp1_assessable"] = [True, True, True, False]
    frame["bp1_missing"] = [False, False, False, True]
    frame["bp1_detection_failed"] = [False, False, True, False]

    summary = summarize_point_bp1(frame, colocalization_radius_nm=100.0)

    assert summary.bp1_assessment_metadata_available
    assert summary.bp1_assessable_frames == 3
    assert summary.bp1_unassessable_frames == 1
    assert summary.bp1_missing_frames == 1
    assert summary.bp1_detection_failure_frames == 1
    assert summary.bp1_assessable_nondetection_frames == 1
    assert summary.bp1_unknown_status_frames == 0
    assert summary.bp1_evaluable_frames == 2
    assert summary.bp1_detection_fraction == 0.5
    assert summary.bp1_detection_fraction_assessable == 0.5
    assert summary.bp1_detection_fraction_nominal_technical == 0.25
    assert summary.bp1_colocalization_assessable_frames == 2
    assert summary.bp1_paired_evaluable_frames == 2
    assert summary.bp1_colocalized_fraction == 0.5
    assert summary.bp1_colocalized_fraction_assessable == 0.5
    assert summary.bp1_detected_paired_frames == 1
    assert summary.bp1_colocalized_fraction_detected_paired == 1.0
    assert summary.bp1_bundle_observation_status == "point_detected"
    assert summary.bp1_biological_negative is None


def test_bp1_burden_and_detected_conditional_colocalization_have_explicit_denominators() -> None:
    frame = pd.DataFrame(
        {
            "frame": [1, 2, 3, 4, 5, 6],
            "site1_x_nm": [0.0] * 6,
            "site1_y_nm": [0.0] * 6,
            "site1_valid": [True] * 6,
            "site2_x_nm": [2.0] * 6,
            "site2_y_nm": [0.0] * 6,
            "site2_valid": [True] * 6,
            "bp1_x_nm": [1.0, 20.0, np.nan, np.nan, np.nan, np.nan],
            "bp1_y_nm": [0.0, 0.0, np.nan, np.nan, np.nan, np.nan],
            "bp1_valid": [True, True, False, False, False, False],
            "bp1_assessable": [True, True, True, True, False, False],
            "bp1_detection_failed": [False, False, False, True, False, False],
            "bp1_missing": [False, False, False, False, True, False],
        }
    )

    summary = summarize_point_bp1(frame, colocalization_radius_nm=3.0)

    # The burden denominator is three paired/evaluable frames: one colocalized
    # point, one non-colocalized point, and one explicit assessable nondetection.
    assert summary.bp1_paired_evaluable_frames == 3
    assert summary.bp1_colocalized_fraction == pytest.approx(1 / 3)
    assert summary.bp1_colocalized_fraction_assessable == pytest.approx(1 / 3)
    # Conditional proximity uses only the two detected paired frames.
    assert summary.bp1_detected_paired_frames == 2
    assert summary.bp1_colocalized_fraction_detected_paired == 0.5
    # Failure, missing input, and unknown status never enter either denominator.
    assert summary.bp1_detection_failure_frames == 1
    assert summary.bp1_missing_frames == 1
    assert summary.bp1_unknown_status_frames == 1


def test_contradictory_or_ambiguous_status_flags_fail_closed() -> None:
    frame = _point_frame()
    frame["bp1_assessable"] = [False, False, False, False]

    with pytest.raises(ValueError, match="Contradictory"):
        summarize_point_bp1(frame, colocalization_radius_nm=100.0)

    frame = _point_frame()
    frame["bp1_assessable"] = ["True", False, False, False]
    with pytest.raises(ValueError, match="booleans/0/1"):
        summarize_point_bp1(frame, colocalization_radius_nm=100.0)


def _complete_component_contract() -> dict:
    contract = component_quantification_contract_template()
    contract["input"].update(
        {
            "image_source": "read_only_archive_manifest:image-index-v1",
            "channel_identity": "53BP1",
            "pixel_size_nm": 108.0,
            "read_only_source_confirmed": True,
            "evidence_source": "acquisition_record:channel-map-v1",
        }
    )
    contract["qc"].update(
        {
            "validated": True,
            "frame_exclusion_rules": ["saturation", "empty_nucleus_mask"],
            "evidence_source": "validation_report:image-qc-v1",
        }
    )
    contract["registration"].update(
        {
            "validated": True,
            "transform_definition": "locked_corrected_coordinate_transform:v1",
            "max_registration_error_nm": 75.0,
            "evidence_source": "validation_report:registration-v1",
        }
    )
    contract["threshold"].update(
        {
            "validated": True,
            "method": "local_median_plus_robust_sigma",
            "parameters": {"threshold_k": 0.5, "radius_px": 12},
            "background_method": "local_median",
            "evidence_source": "validation_report:threshold-v1",
        }
    )
    contract["linking"].update(
        {
            "validated": True,
            "method": "nearest_component_with_gap_guard",
            "parameters": {"max_gap_frames": 1, "max_distance_px": 4.0},
            "evidence_source": "validation_report:linking-v1",
        }
    )
    contract["output_schema"].update(
        {
            "validated": True,
            "schema_version": "component-measurement-v1",
            "evidence_source": "schema_record:component-measurement-v1",
        }
    )
    return contract


def test_empty_or_placeholder_component_contract_stays_gated() -> None:
    template = component_quantification_contract_template()
    duplicate = component_quantification_contract_template()
    template["input"]["image_source"] = "mutated"
    assert duplicate["input"]["image_source"] is None

    empty_gate = component_quantification_gate()
    assert empty_gate["status"] == "gated"
    assert not empty_gate["ready"]
    assert set(empty_gate["required_sections"]) == {
        "input",
        "qc",
        "registration",
        "threshold",
        "linking",
        "output_schema",
    }

    placeholder = _complete_component_contract()
    placeholder["registration"]["evidence_source"] = "TODO"
    placeholder_gate = evaluate_component_quantification_gate(placeholder)
    assert placeholder_gate["status"] == "gated"
    assert "registration.evidence_source" in placeholder_gate["placeholder_fields"]

    arbitrary_assertions = _complete_component_contract()
    arbitrary_assertions["qc"]["evidence_source"] = "banana"
    arbitrary_gate = evaluate_component_quantification_gate(arbitrary_assertions)
    assert arbitrary_gate["status"] == "gated"
    assert "qc.evidence_source:invalid_typed_reference" in arbitrary_gate["blocking_requirements"]


def test_component_gate_requires_every_evidence_stage_and_complete_output_schema() -> None:
    complete = _complete_component_contract()
    ready = evaluate_component_quantification_gate(complete)
    assert ready["status"] == "ready"
    assert ready["ready"]
    assert ready["blocking_requirements"] == []

    missing_evidence = deepcopy(complete)
    missing_evidence["linking"]["evidence_source"] = None
    assert evaluate_component_quantification_gate(missing_evidence)["status"] == "gated"

    incomplete_schema = deepcopy(complete)
    incomplete_schema["output_schema"]["required_fields"] = ["component_label"]
    report = evaluate_component_quantification_gate(incomplete_schema)
    assert report["status"] == "gated"
    assert any(
        blocker.startswith("output_schema.required_fields:missing:")
        for blocker in report["blocking_requirements"]
    )
