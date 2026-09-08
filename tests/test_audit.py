from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from dsb_states import audit
from dsb_states.audit import (
    REQUIRED_CONTRACTS,
    _assess_authoritative_metadata,
    _assess_genomic_mapping,
    _finite_positive,
    _resolved,
    audit_m0,
    load_authoritative_m0_metadata,
)
from dsb_states.config import load_config
from dsb_states.io import SourceLayout


def test_unresolved_contract_tokens() -> None:
    assert not _resolved(None)
    assert not _resolved("unresolved")
    assert not _resolved(" pending ")
    assert _resolved("nd2_id")
    assert not _finite_positive("banana")
    assert not _finite_positive(float("nan"))
    assert _finite_positive(1000)


@pytest.mark.parametrize("placeholder", ["unknown", "pending", "n/a"])
def test_hierarchy_values_reject_placeholder_tokens(tmp_path: Path, placeholder: str) -> None:
    cells = pd.DataFrame(
        {
            "nd2_id": [placeholder],
            "fov": [placeholder],
            "crop_id": [placeholder],
        }
    )
    assessments = audit._assess_hierarchy(
        {
            "hierarchy": {
                "acquisition_id_source": "nd2_id",
                "fov_id_source": "cell_index.fov",
                "cell_id_source": "crop_id",
                "bundle_id_source": "bundle_id",
            }
        },
        cells,
        tmp_path,
    )
    for contract in ("acquisition_id", "fov_id", "cell_id"):
        assert not assessments[contract]["resolved"]


def test_m0_is_fail_closed_for_every_required_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = SourceLayout(tmp_path)
    monkeypatch.setattr(audit, "validate_archive_layout", lambda _root: (layout, []))
    for unresolved_contract in REQUIRED_CONTRACTS:
        assessments = {
            name: {
                "value": "verified",
                "resolved": name != unresolved_contract,
                "reason": "synthetic assessment",
                "evidence": {},
            }
            for name in REQUIRED_CONTRACTS
        }
        monkeypatch.setattr(
            audit,
            "_typed_contract_assessments",
            lambda *_args, _assessments=assessments, **_kwargs: _assessments,
        )
        report = audit_m0({"data": {"archive_root_resolved": str(tmp_path)}})
        assert report["scientific_status"] == "stop"
        assert unresolved_contract in report["blocking_contracts"]


def test_adversarial_flank_and_distance_placeholders_never_resolve(tmp_path: Path) -> None:
    mapping = {
        "site1": {
            "genomic_side_of_cut": "left",
            "genomic_coordinates": "chr1:100-120",
            "probe_to_cut_distance_bp": 990,
            "probe_to_cut_distance_definition": "probe_interval_midpoint_to_cut",
            "evidence_source": "probe_design_record:PD-001",
        },
        "site2": {
            "genomic_side_of_cut": "right",
            "genomic_coordinates": "chr1:2100-2120",
            "probe_to_cut_distance_bp": 1010,
            "probe_to_cut_distance_definition": "probe_interval_midpoint_to_cut",
            "evidence_source": "probe_design_record:PD-001",
        },
        "cut_site": {
            "genome_build": "test-build",
            "genomic_coordinate": "chr1:1100",
            "evidence_source": "genomic_annotation:GA-001",
        },
        "language_gate": {
            "allow_left_right_names": True,
            "allow_broken_end_distance": True,
        },
    }
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    config = {
        "_project_root": str(tmp_path),
        "channels": {
            "genomic_mapping_path": str(path),
            "flank_mapping_status": "pending",
            "site_left_source": "site1",
            "site_right_source": "site2",
            "site1_probe_to_cut_bp": "banana",
            "site2_probe_to_cut_bp": "banana",
        },
    }
    assessments = _assess_genomic_mapping(config)
    assert not assessments["site1_site2_flank_mapping"]["resolved"]
    assert not assessments["site1_probe_to_cut_bp"]["resolved"]
    assert not assessments["site2_probe_to_cut_bp"]["resolved"]

    config["channels"].update(
        {
            "flank_mapping_status": "verified",
            "site1_probe_to_cut_bp": 990,
            "site2_probe_to_cut_bp": 1010,
        }
    )
    assessments = _assess_genomic_mapping(config)
    assert all(item["resolved"] for item in assessments.values())

    config["channels"].update({"site_left_source": "site2", "site_right_source": "site1"})
    reversed_sources = _assess_genomic_mapping(config)
    assert not reversed_sources["site1_site2_flank_mapping"]["resolved"]

    config["channels"].update({"site_left_source": "site1", "site_right_source": "site2"})
    mapping["site1"]["genomic_coordinates"] = "not-a-coordinate"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    malformed_coordinate = _assess_genomic_mapping(config)
    assert not malformed_coordinate["site1_site2_flank_mapping"]["resolved"]


def test_authoritative_metadata_requires_complete_confirmed_coverage(tmp_path: Path) -> None:
    cells = pd.DataFrame(
        {
            "nd2_id": ["a.nd2", "b.nd2"],
            "fov": ["fov1", "fov2"],
        }
    )
    mapping = pd.DataFrame(
        {
            "nd2_id": ["a.nd2"],
            "fov_id": ["fov1"],
            "hour_post_delivery": [2.0],
            "hour_mapping_evidence": ["experimental_design_record:ED-001"],
            "hour_mapping_confirmed": [True],
            "biological_experiment_id": ["experiment-1"],
            "biological_experiment_evidence": ["experimental_design_record:ED-001"],
            "biological_experiment_confirmed": [True],
        }
    )
    path = tmp_path / "m0.csv"
    mapping.to_csv(path, index=False)
    config = {
        "_project_root": str(tmp_path),
        "hierarchy": {
            "m0_metadata_path": str(path),
            "biological_experiment_id_source": "m0_metadata.biological_experiment_id",
        },
        "time_axes": {
            "macro_time_mapping_source": "m0_metadata.hour_post_delivery",
            "allowed_hours": [2.0, 3.0],
        },
    }
    incomplete = _assess_authoritative_metadata(config, cells)
    assert not incomplete["biological_experiment_id"]["resolved"]
    assert not incomplete["macro_time_mapping"]["resolved"]

    mapping.loc[len(mapping)] = [
        "b.nd2",
        "fov2",
        3.0,
        "inferred: folder",
        True,
        "experiment-1",
        "experimental_design_record:ED-001",
        True,
    ]
    mapping.to_csv(path, index=False)
    inferred = _assess_authoritative_metadata(config, cells)
    assert inferred["biological_experiment_id"]["resolved"]
    assert not inferred["macro_time_mapping"]["resolved"]


def test_authoritative_metadata_rejects_arbitrary_evidence_and_acquisition_conflicts(
    tmp_path: Path,
) -> None:
    cells = pd.DataFrame({"nd2_id": ["a.nd2", "a.nd2"], "fov": ["fov1", "fov2"]})
    mapping = pd.DataFrame(
        {
            "nd2_id": ["a.nd2", "a.nd2"],
            "fov_id": ["fov1", "fov2"],
            "hour_post_delivery": [2.0, 3.0],
            "hour_mapping_evidence": ["banana", "banana"],
            "hour_mapping_confirmed": [True, True],
            "biological_experiment_id": ["experiment-1", "experiment-2"],
            "biological_experiment_evidence": ["banana", "banana"],
            "biological_experiment_confirmed": [True, True],
        }
    )
    path = tmp_path / "m0.csv"
    mapping.to_csv(path, index=False)
    config = {
        "_project_root": str(tmp_path),
        "hierarchy": {
            "m0_metadata_path": str(path),
            "biological_experiment_id_source": "m0_metadata.biological_experiment_id",
        },
        "time_axes": {
            "macro_time_mapping_source": "m0_metadata.hour_post_delivery",
            "allowed_hours": [2.0, 3.0],
        },
    }
    assessments = _assess_authoritative_metadata(config, cells)
    assert not assessments["biological_experiment_id"]["resolved"]
    assert not assessments["macro_time_mapping"]["resolved"]


def test_verified_authoritative_metadata_loader_returns_typed_interface(tmp_path: Path) -> None:
    cells = pd.DataFrame({"nd2_id": ["a.nd2"], "fov": ["fov1"]})
    mapping = pd.DataFrame(
        {
            "nd2_id": ["a.nd2"],
            "fov_id": ["fov1"],
            "hour_post_delivery": [2.0],
            "hour_mapping_evidence": ["experimental_design_record:ED-001"],
            "hour_mapping_confirmed": [True],
            "biological_experiment_id": ["experiment-1"],
            "biological_experiment_evidence": ["experimental_design_record:ED-001"],
            "biological_experiment_confirmed": [True],
            "imaging_day": ["day-1"],
            "biological_replicate": ["replicate-1"],
        }
    )
    path = tmp_path / "m0.csv"
    mapping.to_csv(path, index=False)
    config = {
        "_project_root": str(tmp_path),
        "hierarchy": {
            "m0_metadata_path": str(path),
            "biological_experiment_id_source": "m0_metadata.biological_experiment_id",
        },
        "time_axes": {
            "macro_time_mapping_source": "m0_metadata.hour_post_delivery",
            "allowed_hours": [2.0],
        },
    }
    loaded, loaded_path = load_authoritative_m0_metadata(config, cells)
    assert loaded_path == path.resolve()
    assert loaded.loc[0, "biological_experiment_id"] == "experiment-1"
    assert loaded.loc[0, "hour_post_delivery"] == 2.0


