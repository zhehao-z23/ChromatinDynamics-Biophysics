from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dsb_states.config import load_config
from dsb_states.schema import (
    BUNDLE_NEUTRAL_REQUIRED_FIELDS,
    BUNDLE_REQUIRED_FIELDS,
    DEFAULT_ALLOWED_HOURS,
    DataContractError,
    ValidationIssue,
    ValidationReport,
    detect_bundle_schema,
    partition_validation_errors,
    resolve_allowed_hours,
    valid_pair_mask,
    validate_bundle_table,
    validate_data_contract,
    validate_frame_table,
    validate_m0_readiness,
    write_validation_reports,
)


def _neutral_frame_table() -> pd.DataFrame:
    rows = []
    for bundle_index, (bundle_id, hour) in enumerate(
        zip(("b1", "b2"), (1.5, 2.0), strict=True), start=1
    ):
        for frame in range(3):
            rows.append(
                {
                    "biological_experiment_id": "experiment_1",
                    "acquisition_id": f"acq_{bundle_index}",
                    "fov_id": f"fov_{bundle_index}",
                    "cell_id": f"cell_{bundle_index}",
                    "bundle_id": bundle_id,
                    "hour_post_delivery": hour,
                    "frame": frame,
                    "micro_time_s": float(frame),
                    "frame_interval_s": 1.0,
                    "site1_x_nm": 100.0 * bundle_index + frame,
                    "site1_y_nm": 200.0 + frame,
                    "site1_valid": True,
                    "site2_x_nm": 130.0 * bundle_index + frame,
                    "site2_y_nm": 240.0 + frame,
                    "site2_valid": frame != 1,
                    "bp1_valid": frame == 0,
                }
            )
    return pd.DataFrame(rows)


def _canonical_frame_table() -> pd.DataFrame:
    return _neutral_frame_table().rename(
        columns={
            "site1_x_nm": "site_left_x_nm",
            "site1_y_nm": "site_left_y_nm",
            "site1_valid": "site_left_valid",
            "site2_x_nm": "site_right_x_nm",
            "site2_y_nm": "site_right_y_nm",
            "site2_valid": "site_right_valid",
        }
    )


def _bundle_table() -> pd.DataFrame:
    rows = []
    for bundle_index, (bundle_id, hour) in enumerate(
        zip(("b1", "b2"), (1.5, 2.0), strict=True), start=1
    ):
        rows.append(
            {
                "bundle_id": bundle_id,
                "biological_experiment_id": "experiment_1",
                "acquisition_id": f"acq_{bundle_index}",
                "fov_id": f"fov_{bundle_index}",
                "cell_id": f"cell_{bundle_index}",
                "hour_post_delivery": hour,
                "nominal_frame_count": 3,
                "left_valid_count": 3,
                "right_valid_count": 2,
                "paired_valid_count": 2,
                "left_coverage": 1.0,
                "right_coverage": 2 / 3,
                "paired_coverage": 2 / 3,
                "longest_left_run": 3,
                "longest_right_run": 1,
                "longest_paired_run": 1,
                "median_site_intensity": 10.0,
                "bp1_detection_fraction": 1 / 3,
                "manual_or_pipeline_qc_flags": "",
            }
        )
    return pd.DataFrame(rows)


def _neutral_bundle_table() -> pd.DataFrame:
    return _bundle_table().rename(
        columns={
            "left_valid_count": "site1_valid_count",
            "right_valid_count": "site2_valid_count",
            "left_coverage": "site1_coverage",
            "right_coverage": "site2_coverage",
            "longest_left_run": "longest_site1_run",
            "longest_right_run": "longest_site2_run",
        }
    )


def _resolved_config() -> dict:
    return {
        "hierarchy": {"biological_experiment_id_source": "experiment_manifest"},
        "channels": {
            "flank_mapping_status": "verified",
            "site_left_source": "site1",
            "site_right_source": "site2",
            "bp1_dna_same_coordinate_system": True,
        },
        "units": {"coordinate_unit": "nm"},
    }


def test_neutral_site_schema_is_structurally_valid_but_does_not_infer_flanks() -> None:
    frames = _neutral_frame_table()
    structural = validate_frame_table(frames)
    assert structural.valid
    assert structural.metadata["coordinate_schema"] == "neutral_site1_site2"

    m0 = validate_m0_readiness(frames, _bundle_table(), config={})
    assert not m0.valid
    assert m0.has_code("m0_flank_mapping_unresolved")
    assert m0.has_code("m0_nm_unit_evidence_missing")


def test_resolved_metadata_allows_complete_contract_to_pass() -> None:
    report = validate_data_contract(
        _neutral_frame_table(), _bundle_table(), config=_resolved_config()
    )
    assert report.valid, report.to_dict()


def test_snapshot_default_macro_hours_are_explicit_and_include_half_hours() -> None:
    assert DEFAULT_ALLOWED_HOURS == (
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        9.5,
        10.0,
        10.5,
    )
    frames = _neutral_frame_table()
    frames.loc[frames["bundle_id"].eq("b2"), "hour_post_delivery"] = 10.5
    assert validate_frame_table(frames).valid

    frames.loc[frames["bundle_id"].eq("b2"), "hour_post_delivery"] = 1.0
    assert validate_frame_table(frames).has_code("invalid_hour_post_delivery")


def test_missing_and_invalid_macro_hours_have_distinct_issue_records() -> None:
    frames = _neutral_frame_table()
    frames["hour_post_delivery"] = frames["hour_post_delivery"].astype("object")
    frames.loc[0, "hour_post_delivery"] = pd.NA
    frames.loc[3, "hour_post_delivery"] = "not-a-timepoint"
    report = validate_frame_table(frames)

    missing = next(issue for issue in report.errors if issue.code == "missing_hour_post_delivery")
    invalid = next(issue for issue in report.errors if issue.code == "invalid_hour_post_delivery")
    assert missing.count == 1
    assert missing.row_indices == (0,)
    assert invalid.count == 1
    assert invalid.row_indices == (3,)

    bundles = _neutral_bundle_table()
    bundles["hour_post_delivery"] = bundles["hour_post_delivery"].astype("object")
    bundles.loc[0, "hour_post_delivery"] = pd.NA
    bundles.loc[1, "hour_post_delivery"] = 8.0
    bundle_report = validate_bundle_table(bundles)
    assert (
        next(
            issue for issue in bundle_report.errors if issue.code == "missing_hour_post_delivery"
        ).count
        == 1
    )
    assert (
        next(
            issue for issue in bundle_report.errors if issue.code == "invalid_hour_post_delivery"
        ).count
        == 1
    )


def test_active_config_coordinate_and_design_evidence_close_m0() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "provenance" / "source_analysis.yaml")
    assert config["coordinates"]["units"] == "nm"
    assert config["coordinates"]["bp1_dna_same_coordinate_system"] is True

    report = validate_m0_readiness(
        _neutral_frame_table(),
        _neutral_bundle_table(),
        config,
    )
    codes = {issue.code for issue in report.errors}
    assert "m0_nm_unit_evidence_missing" not in codes
    assert "m0_bp1_coordinate_system_unverified" not in codes
    assert "m0_flank_mapping_unresolved" not in codes
    assert "m0_hierarchy_source_unresolved" not in codes
    assert "m0_unavailable_biological_group_contract_invalid" not in codes
    assert report.valid


def test_neutral_bundle_schema_is_detected_and_validated() -> None:
    bundles = _neutral_bundle_table()
    report = validate_bundle_table(bundles)
    assert report.valid, report.to_dict()
    assert report.metadata["bundle_schema"] == "neutral_site1_site2"
    assert detect_bundle_schema(bundles) == "neutral_site1_site2"
    assert set(BUNDLE_NEUTRAL_REQUIRED_FIELDS).issubset(bundles.columns)

    bundles.loc[0, "site1_coverage"] = 0.25
    assert validate_bundle_table(bundles).has_code("bundle_coverage_count_mismatch")


def test_neutral_frame_and_bundle_tables_reconcile_without_flank_renaming() -> None:
    report = validate_data_contract(
        _neutral_frame_table(), _neutral_bundle_table(), config=_resolved_config()
    )
    assert report.valid, report.to_dict()
    assert report.metadata["coordinate_schema"] == "neutral_site1_site2"
    assert report.metadata["bundle_schema"] == "neutral_site1_site2"


def test_canonical_frame_and_bundle_schema_remains_supported() -> None:
    report = validate_data_contract(
        _canonical_frame_table(), _bundle_table(), config=_resolved_config()
    )
    assert report.valid, report.to_dict()
    assert report.metadata["coordinate_schema"] == "canonical_left_right"
    assert report.metadata["bundle_schema"] == "canonical_left_right"


def test_data_contract_resolves_allowed_hours_from_config_or_explicit_override() -> None:
    frames = _neutral_frame_table()
    bundles = _neutral_bundle_table()
    frames["hour_post_delivery"] = 7.0
    bundles["hour_post_delivery"] = 7.0
    config = _resolved_config() | {"time_axes": {"allowed_hours": [7.0]}}
    configured = validate_data_contract(frames, bundles, config=config)
    assert configured.valid, configured.to_dict()
    assert configured.metadata["allowed_hours"] == [7.0]
    assert resolve_allowed_hours(config) == (7.0,)

    config["time_axes"]["allowed_hours"] = [8.0]
    explicit = validate_data_contract(frames, bundles, config=config, allowed_hours=[7.0])
    assert explicit.valid, explicit.to_dict()


def test_required_frame_and_bundle_fields_are_machine_readable() -> None:
    frame_report = validate_frame_table(pd.DataFrame({"bundle_id": ["b1"]}))
    assert not frame_report.valid
    issue = next(issue for issue in frame_report.issues if issue.code == "missing_required_fields")
    assert "micro_time_s" in issue.details["missing_fields"]

    bundle = _bundle_table().drop(columns=[BUNDLE_REQUIRED_FIELDS[-1]])
    bundle_report = validate_bundle_table(bundle)
    assert bundle_report.has_code("missing_required_fields")
    serialized = bundle_report.to_dict()
    assert serialized["valid"] is False
    assert isinstance(serialized["issues"], list)


def test_frame_validator_detects_uniqueness_time_containment_hours_and_pair_logic() -> None:
    frames = _neutral_frame_table()
    frames["paired_valid"] = frames["site1_valid"] & frames["site2_valid"]
    duplicate = frames.iloc[[0]].copy()
    broken = pd.concat([frames, duplicate], ignore_index=True)
    broken.loc[2, "micro_time_s"] = 0.5
    broken.loc[1, "acquisition_id"] = "crossed_acquisition"
    broken.loc[3, "hour_post_delivery"] = 99
    broken.loc[0, "paired_valid"] = False

    report = validate_frame_table(broken)
    codes = {issue.code for issue in report.errors}
    assert "duplicate_bundle_frame" in codes
    assert "nonmonotonic_micro_time" in codes
    assert "bundle_crosses_acquisition_id" in codes
    assert "invalid_hour_post_delivery" in codes
    assert "invalid_paired_valid_logic" in codes


def test_valid_pair_mask_is_exact_boolean_intersection() -> None:
    frames = _neutral_frame_table()
    mask = valid_pair_mask(frames)
    assert mask.tolist() == [True, False, True, True, False, True]
    frames["site1_valid"] = frames["site1_valid"].astype("boolean")
    frames.loc[0, "site1_valid"] = pd.NA
    with pytest.raises(DataContractError):
        valid_pair_mask(frames)


def test_bundle_counts_and_coverage_are_validated() -> None:
    bundles = _bundle_table()
    bundles.loc[0, "paired_valid_count"] = 4
    bundles.loc[1, "left_coverage"] = 0.1
    report = validate_bundle_table(bundles)
    assert report.has_code("bundle_count_exceeds_nominal")
    assert report.has_code("paired_count_exceeds_site_count")
    assert report.has_code("bundle_coverage_count_mismatch")


def test_validation_reports_write_json_and_markdown(tmp_path) -> None:
    report = ValidationReport(
        "example",
        [
            ValidationIssue(
                code="example_issue",
                message="structured",
                count=1,
                keys=({"bundle_id": "b1"},),
            )
        ],
    )
    json_path = tmp_path / "data_validation_report.json"
    md_path = tmp_path / "data_validation_report.md"
    write_validation_reports(report, json_path, md_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["issues"][0]["code"] == "example_issue"
    assert payload["issues"][0]["keys"] == [{"bundle_id": "b1"}]
    assert "example_issue" in md_path.read_text(encoding="utf-8")


def test_validation_partition_allows_only_declared_metadata_gaps() -> None:
    report = ValidationReport(
        "partition",
        [
            ValidationIssue(
                code="missing_hierarchy_id_values",
                message="missing experiment",
                details={"field": "biological_experiment_id"},
            ),
            ValidationIssue(
                code="missing_hour_post_delivery",
                message="missing hour",
                details={"field": "hour_post_delivery"},
            ),
            ValidationIssue(code="m0_flank_mapping_unresolved", message="missing flank"),
            ValidationIssue(code="duplicate_bundle_frame", message="duplicate"),
            ValidationIssue(code="m0_nm_unit_evidence_missing", message="missing units"),
        ],
    )
    expected, structural = partition_validation_errors(
        report,
        {
            "biological_experiment_id",
            "macro_time_mapping",
            "site1_site2_flank_mapping",
            "coordinate_units",
        },
    )
    assert [issue.code for issue in expected] == [
        "missing_hierarchy_id_values",
        "missing_hour_post_delivery",
        "m0_flank_mapping_unresolved",
    ]
    assert [issue.code for issue in structural] == [
        "duplicate_bundle_frame",
        "m0_nm_unit_evidence_missing",
    ]
