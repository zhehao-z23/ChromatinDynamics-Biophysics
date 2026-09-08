from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dsb_states.io import sha256_file
from dsb_states.snapshot import SnapshotError, _enforce_output_scope, _load_hour_map
from dsb_states.tables import (
    AnalysisTablesResult,
    OutputScopeError,
    _interval_per_frame,
    build_analysis_tables,
    canonical_output_roots,
    longest_true_run,
    summarize_bundles,
    write_analysis_tables,
)


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "biological_experiment_id": pd.Series([pd.NA] * 5, dtype="string"),
            "acquisition_id": ["a1"] * 5,
            "fov_id": ["f1"] * 5,
            "cell_id": ["c1"] * 5,
            "bundle_id": ["b1"] * 5,
            "hour_post_delivery": [2.0] * 5,
            "frame": [1, 2, 3, 4, 5],
            "micro_time_s": [0.0, 1.0, 2.1, 3.2, 4.2],
            "frame_interval_s": [1.0, 1.0, 1.1, 1.1, 1.0],
            "exact_interval_median_s": [1.0] * 5,
            "production_uniform_frame_interval_s": [1.05] * 5,
            "site1_valid": [True, True, False, True, True],
            "site2_valid": [True, False, False, True, True],
            "paired_valid": [True, False, False, True, True],
            "bp1_valid": [False, True, False, False, True],
        }
    )


def _bundle_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bundle_id": ["b1"],
            "biological_experiment_id": pd.Series([pd.NA], dtype="string"),
            "nd2_id": ["a1"],
            "fov": ["f1"],
            "crop_id": ["c1"],
            "allele_index": [1],
            "anchor_locus": [1],
            "hour_post_delivery": [2.0],
            "hour_mapping_source": ["frozen inventory"],
            "hour_mapping_status": ["mapped_from_frozen_inventory"],
            "run_manifest_status": ["complete"],
            "trajectory_usability_class": ["PIPELINE_COMPLETE"],
        }
    )


def test_longest_true_run_does_not_bridge_gaps() -> None:
    assert longest_true_run([True, True, False, True, True, True, False]) == 3
    assert longest_true_run([False, False]) == 0


def test_bundle_summary_uses_neutral_site_counts_and_preserves_unknown_experiment() -> None:
    summary = summarize_bundles(_frames(), _bundle_index()).iloc[0]
    assert summary.site1_valid_count == 4
    assert summary.site2_valid_count == 3
    assert summary.paired_valid_count == 3
    assert summary.longest_site1_run == 2
    assert summary.longest_paired_run == 2
    assert np.isclose(summary.bp1_detection_fraction, 0.4)
    assert summary.flank_mapping_status == "unresolved"
    assert pd.isna(summary.biological_experiment_id)
    assert not any("left" in column or "right" in column for column in summary.index)


def test_snapshot_manifest_contract_never_marks_review_as_included(tmp_path: Path) -> None:
    # The fixture directly checks the machine-readable invariant expected by table discovery.
    manifest = pd.DataFrame(
        {
            "record_type": ["formal_final_trajectory"],
            "formal_usable": [True],
            "review_included": [False],
            "bundle_id": ["b1"],
            "site_id": ["site1"],
        }
    )
    path = tmp_path / "source_manifest.parquet"
    manifest.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    assert restored.formal_usable.all()
    assert not restored.review_included.any()


def test_inferred_inventory_hour_is_not_promoted_to_analysis_label(tmp_path: Path) -> None:
    inventory = tmp_path / "nd2.tsv"
    pd.DataFrame(
        {
            "file_name": ["authoritative.nd2", "inferred.nd2"],
            "timepoint_h": [2.5, 2.0],
            "timepoint_source": ["folder label", "inferred: adjacent folders agree"],
            "retained_after_dedup": [True, True],
        }
    ).to_csv(inventory, sep="\t", index=False)
    mapping, _ = _load_hour_map(
        {"_project_root": str(tmp_path), "data": {"nd2_inventory": str(inventory)}}
    )
    indexed = mapping.set_index("nd2_id")
    assert indexed.loc["authoritative.nd2", "hour_post_delivery"] == 2.5
    assert pd.isna(indexed.loc["inferred.nd2", "hour_post_delivery"])
    assert indexed.loc["inferred.nd2", "hour_mapping_source"].startswith("inferred:")


def test_exact_and_uniform_timing_provenance_are_distinct() -> None:
    frames = _frames()
    assert frames.micro_time_s.tolist() == [0.0, 1.0, 2.1, 3.2, 4.2]
    assert frames.production_uniform_frame_interval_s.nunique() == 1
    assert not np.allclose(
        np.diff(frames.micro_time_s), frames.production_uniform_frame_interval_s.iloc[0]
    )


def test_exact_intervals_align_with_interval_to_next_frame() -> None:
    times = np.array([0.0, 1.0, 2.1, 3.4])
    intervals = _interval_per_frame(times, representative_interval=1.1)
    assert np.allclose(intervals[:-1], np.diff(times))
    assert intervals[-1] == intervals[-2]


def test_build_analysis_tables_end_to_end_from_minimal_snapshot(tmp_path: Path) -> None:
    exact = tmp_path / "exact_metadata.json"
    exact.write_text(json.dumps({"time": {"relative_time_s": [0.0, 1.0, 2.2]}}), encoding="utf-8")
    manifest_rows = []
    trajectories = {
        "site1": pd.DataFrame({"frame": [1, 3], "x_nm": [0.0, 2.0], "y_nm": [0.0, 0.0]}),
        "site2": pd.DataFrame(
            {"frame": [1, 2, 3], "x_nm": [1.0, 2.0, 3.0], "y_nm": [0.0, 0.0, 0.0]}
        ),
        "53bp1": pd.DataFrame({"frame": [2], "x_nm": [1.5], "y_nm": [0.5]}),
    }
    for site, trajectory in trajectories.items():
        path = tmp_path / f"allele_001_{site}_longest_spt_cleaned.csv"
        trajectory.to_csv(path, index=False)
        manifest_rows.append(
            {
                "record_type": "formal_final_trajectory",
                "bundle_id": "b1",
                "site_id": site,
                "snapshot_relative_path": path.name,
                "sha256": sha256_file(path),
                "allele_index": 1,
                "localization_rows": len(trajectory),
            }
        )
    pd.DataFrame(manifest_rows).to_parquet(tmp_path / "source_manifest.parquet", index=False)
    pd.DataFrame(
        {
            "bundle_id": ["b1"],
            "biological_experiment_id": pd.Series([pd.NA], dtype="string"),
            "nd2_id": ["a1"],
            "crop_id": ["c1"],
            "total_frames": [3],
            "fov": ["f1"],
            "allele_index": [1],
            "anchor_locus": [1],
            "hour_post_delivery": [2.0],
            "hour_mapping_source": ["folder label"],
            "hour_mapping_status": ["mapped_from_frozen_inventory"],
            "run_manifest_status": ["complete"],
            "trajectory_usability_class": ["PIPELINE_COMPLETE"],
        }
    ).to_parquet(tmp_path / "bundle_index_formal_usable.parquet", index=False)
    pd.DataFrame(
        {
            "nd2_id": ["a1"],
            "cell_id": ["c1"],
            "exact_metadata_snapshot_relative_path": [exact.name],
            "exact_metadata_sha256": [sha256_file(exact)],
            "exact_interval_median_s": [1.1],
            "production_uniform_interval_s": [1.1],
            "production_uniform_policy": ["per-ND2 median"],
        }
    ).to_parquet(tmp_path / "crop_metadata.parquet", index=False)

    result = build_analysis_tables(tmp_path)
    assert len(result.frame_table) == 3
    assert result.frame_table["micro_time_s"].tolist() == [0.0, 1.0, 2.2]
    assert result.frame_table["paired_valid"].tolist() == [True, False, True]
    assert result.bundle_metadata.loc[0, "paired_valid_count"] == 2
    assert all(
        value is True for key, value in result.reconciliation.items() if key.endswith("reconciled")
    )


def test_reconciliation_values_are_json_serializable() -> None:
    payload = SimpleNamespace(
        snapshot_bundle_count=1,
        frame_table_rows=5,
        localization_count_reconciled=True,
    )
    assert "localization_count_reconciled" in json.dumps(vars(payload))


def _minimal_table_result() -> AnalysisTablesResult:
    return AnalysisTablesResult(
        frame_table=pd.DataFrame({"bundle_id": ["b1"], "frame": [1]}),
        bundle_metadata=pd.DataFrame({"bundle_id": ["b1"]}),
        reconciliation={"reconciled": True},
    )


def test_table_writer_requires_explicit_project_allowlist(tmp_path: Path) -> None:
    project = tmp_path / "analysis"
    allowed = canonical_output_roots(project)
    output = project / "data_snapshot" / "unit" / "tables"
    paths = write_analysis_tables(
        _minimal_table_result(),
        output,
        project_root=project,
        allowed_roots=allowed,
    )
    assert paths["frame_table"].is_file()
    assert paths["bundle_metadata"].is_file()


def test_table_writer_rejects_outside_or_source_overlapping_output_before_write(
    tmp_path: Path,
) -> None:
    project = tmp_path / "analysis"
    allowed = canonical_output_roots(project)
    outside = tmp_path / "arbitrary" / "tables"
    with pytest.raises(OutputScopeError, match="outside"):
        write_analysis_tables(
            _minimal_table_result(),
            outside,
            project_root=project,
            allowed_roots=allowed,
        )
    assert not outside.exists()

    source = project / "data_snapshot" / "source_archive"
    overlapping = source / "derived"
    with pytest.raises(OutputScopeError, match="overlaps"):
        write_analysis_tables(
            _minimal_table_result(),
            overlapping,
            project_root=project,
            allowed_roots=allowed,
            forbidden_roots=(source,),
        )
    assert not overlapping.exists()


def test_caller_cannot_declare_external_source_as_allowed_root(tmp_path: Path) -> None:
    project = tmp_path / "analysis"
    source = tmp_path / "production_repo"
    output = source / "generated"
    with pytest.raises(OutputScopeError, match="Allowed output roots"):
        write_analysis_tables(
            _minimal_table_result(),
            output,
            project_root=project,
            allowed_roots=(source,),
            forbidden_roots=(source,),
        )
    assert not output.exists()


def test_snapshot_scope_requires_project_data_snapshot_tree(tmp_path: Path) -> None:
    project = tmp_path / "analysis"
    archive = tmp_path / "archive"
    config = {
        "_project_root": str(project),
        "project": {"snapshot_root": "data_snapshot"},
    }
    _enforce_output_scope(config, project / "data_snapshot" / "v1", archive)
    with pytest.raises(SnapshotError, match="must remain beneath"):
        _enforce_output_scope(config, tmp_path / "outside", archive)

    external_config = {
        "_project_root": str(project),
        "project": {"snapshot_root": str(tmp_path / "production_repo")},
    }
    with pytest.raises(SnapshotError, match="Configured snapshot root"):
        _enforce_output_scope(external_config, tmp_path / "production_repo" / "v1", archive)
