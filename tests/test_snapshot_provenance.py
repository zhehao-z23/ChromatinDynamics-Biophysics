from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dsb_states import snapshot
from dsb_states.io import SourceLayout, sha256_file


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _immutable_snapshot_fixture(tmp_path: Path) -> tuple[dict, Path, SourceLayout]:
    project = tmp_path / "analysis"
    archive = tmp_path / "archive"
    layout = SourceLayout(archive.resolve())
    source = _write(
        archive / "dsb_v5_results" / "a" / "crop_1" / "trajectory.csv",
        b"frame,x_nm,y_nm\n1,1.0,2.0\n",
    )
    _write(layout.archive_manifest, b"relative_path\tbytes\ntrajectory.csv\t32\n")
    _write(layout.archive_summary, b'{"pipeline_version":"v5.1.0"}\n')
    _write(layout.cell_index, b"nd2_id\tcrop_id\nacq_1\tcrop_1\n")
    bundle_inventory = _write(tmp_path / "bundle_level_metrics.tsv", b"bundle_id\nbundle_1\n")
    nd2_inventory = _write(tmp_path / "nd2_inventory.tsv", b"file_name\nacq_1\n")
    config = {
        "_project_root": str(project),
        "project": {"snapshot_root": "data_snapshot"},
        "data": {
            "archive_root_resolved": str(archive),
            "bundle_inventory": str(bundle_inventory),
            "nd2_inventory": str(nd2_inventory),
            "formal_usable_only": True,
            "review_data_policy": "exclude",
            "use_exact_timestamps": True,
        },
        "hierarchy": {"m0_metadata_path": None},
    }
    root = snapshot.default_snapshot_root(config)
    copied = _write(root / "source_files" / "trajectory.csv", source.read_bytes())
    copied_record = {
        "record_type": "formal_final_trajectory",
        "source_path": str(source.resolve()),
        "snapshot_relative_path": copied.relative_to(root).as_posix(),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    manifest = pd.DataFrame([copied_record])
    manifest.to_parquet(root / "source_manifest.parquet", index=False)
    manifest.to_csv(root / "source_manifest.csv", index=False)
    pd.DataFrame([{"crop_id": "crop_1"}]).to_parquet(
        root / "cell_index_formal_usable.parquet", index=False
    )
    pd.DataFrame([{"bundle_id": "bundle_1"}]).to_parquet(
        root / "bundle_index_formal_usable.parquet", index=False
    )
    pd.DataFrame([{"crop_id": "crop_1"}]).to_parquet(root / "crop_metadata.parquet", index=False)
    reconciliation = {
        "formal_usable_crop_count": 1,
        "authoritative_bundle_count": 1,
        "trajectory_file_count_reconciled": True,
        "localization_count_reconciled": True,
        "review_exclusion_reconciled": True,
    }
    (root / "snapshot_reconciliation.json").write_text(
        json.dumps(reconciliation) + "\n", encoding="utf-8"
    )
    pd.DataFrame([{"check": key, "value": value} for key, value in reconciliation.items()]).to_csv(
        root / "snapshot_reconciliation.csv", index=False
    )
    source_indexes = snapshot._configured_snapshot_source_indexes(config, layout)
    source_hashes = {
        "schema_version": snapshot.SNAPSHOT_SCHEMA_VERSION,
        "snapshot_identity": snapshot.snapshot_identity(config),
        "pipeline_version": snapshot.PIPELINE_VERSION,
        "pipeline_commit": snapshot.PIPELINE_COMMIT,
        "selection": {
            "formal_usable_only": True,
            "review_data_policy": "exclude",
            "exact_relative_time_primary": True,
        },
        "source_indexes": source_indexes,
        "snapshot_artifacts": snapshot.snapshot_control_artifact_hashes(root),
        "copied_files": [copied_record],
    }
    (root / "source_hashes.json").write_text(
        json.dumps(source_hashes, indent=2) + "\n", encoding="utf-8"
    )
    return config, root, layout


def _minimal_buildable_config(tmp_path: Path) -> dict:
    project = tmp_path / "analysis"
    archive = tmp_path / "archive"
    layout = SourceLayout(archive.resolve())
    crop_relative = Path("dsb_v5_results") / "a" / "crop_1"
    crop = archive / crop_relative
    trajectory = _write(
        crop / "spt_results" / "final_trajectories" / "allele_001_site1_longest_spt_cleaned.csv",
        b"frame,x_nm,y_nm\n1,1.0,2.0\n2,2.0,3.0\n",
    )
    exact = crop / "crop_1_metadata.json"
    exact.parent.mkdir(parents=True, exist_ok=True)
    exact.write_text(
        json.dumps(
            {
                "acquisition_date": "test instrument timestamp",
                "time": {
                    "relative_time_s": [0.0, 1.0],
                    "finterval_source": "synthetic_exact",
                },
            }
        ),
        encoding="utf-8",
    )
    uniform = crop / "crop_1_metadata_uniform.json"
    uniform.write_text(
        json.dumps(
            {
                "time": {"finterval_s": 1.0},
                "uniform_timing_override": {
                    "policy": "synthetic",
                    "original_sidecar_sha256": sha256_file(exact),
                },
            }
        ),
        encoding="utf-8",
    )
    _write(layout.archive_manifest, b"relative_path\tbytes\ntrajectory.csv\t42\n")
    layout.archive_summary.write_text(
        json.dumps(
            {
                "pipeline_version": snapshot.PIPELINE_VERSION,
                "pipeline_commit": snapshot.PIPELINE_COMMIT,
                "missing_required_count": 0,
            }
        ),
        encoding="utf-8",
    )
    _write(layout.missing_required, b"relative_path\n")
    pd.DataFrame(
        [
            {
                "dsb_id": "DSB001",
                "nd2_id": "acq_1.nd2",
                "fov": "fov_1",
                "crop_id": "crop_1",
                "total_frames": 2,
                "trajectory_artifacts_usable": 1,
                "trajectory_usability_class": "formal_usable",
                "formal_final_trajectory_count": 1,
                "formal_final_localization_count": 2,
                "archive_relative_path": crop_relative.as_posix(),
            }
        ]
    ).to_csv(layout.cell_index, sep="\t", index=False)
    bundle_inventory = tmp_path / "bundle_level_metrics.tsv"
    pd.DataFrame(
        [
            {
                "bundle_id": "bundle_1",
                "nd2_id": "acq_1.nd2",
                "crop_id": "crop_1",
                "allele_index": 1,
                "anchor_locus": "site1",
                "total_frames": 2,
                "trajectory_artifacts_usable": 1,
            }
        ]
    ).to_csv(bundle_inventory, sep="\t", index=False)
    nd2_inventory = tmp_path / "nd2_inventory.tsv"
    pd.DataFrame(
        [
            {
                "file_name": "acq_1.nd2",
                "timepoint_h": 1.5,
                "timepoint_source": "folder label",
                "retained_after_dedup": 1,
            }
        ]
    ).to_csv(nd2_inventory, sep="\t", index=False)
    assert trajectory.stat().st_size == 36
    return {
        "_project_root": str(project),
        "project": {"snapshot_root": "data_snapshot"},
        "data": {
            "archive_root_resolved": str(archive),
            "bundle_inventory": str(bundle_inventory),
            "nd2_inventory": str(nd2_inventory),
            "formal_usable_only": True,
            "review_data_policy": "exclude",
            "use_exact_timestamps": True,
        },
        "hierarchy": {"m0_metadata_path": None},
    }


def test_existing_snapshot_is_verified_and_never_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root, layout = _immutable_snapshot_fixture(tmp_path)
    monkeypatch.setattr(snapshot, "validate_archive_layout", lambda _path: (layout, []))
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    result = snapshot.build_trajectory_snapshot(config, snapshot_root=root)

    after = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    assert result.summary["snapshot_access"] == "verified_reuse"
    assert result.summary["archive_manifest"]["role"] == "archive_manifest"
    assert result.summary["archive_manifest"]["sha256"] == sha256_file(layout.archive_manifest)
    assert after == before


def test_new_snapshot_is_published_at_content_addressed_final_path(tmp_path: Path) -> None:
    config = _minimal_buildable_config(tmp_path)
    final_root = snapshot.default_snapshot_root(config)

    result = snapshot.build_trajectory_snapshot(config)

    assert result.root == final_root
    assert result.summary["snapshot_access"] == "created"
    assert final_root.is_dir()
    assert not list(final_root.parent.glob(f".{final_root.name}.building-*"))
    source_payload = json.loads((final_root / "source_hashes.json").read_text(encoding="utf-8"))
    assert source_payload["snapshot_identity"] == snapshot.snapshot_identity(config)
    assert next(
        record
        for record in source_payload["source_indexes"]
        if record["role"] == "archive_manifest"
    )["sha256"] == sha256_file(
        SourceLayout(Path(config["data"]["archive_root_resolved"])).archive_manifest
    )


def test_tampered_existing_snapshot_fails_closed_without_repairing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root, layout = _immutable_snapshot_fixture(tmp_path)
    monkeypatch.setattr(snapshot, "validate_archive_layout", lambda _path: (layout, []))
    copied = root / "source_files" / "trajectory.csv"
    copied.write_bytes(b"tampered\n")

    with pytest.raises(snapshot.SnapshotError, match="snapshot file hash changed"):
        snapshot.build_trajectory_snapshot(config, snapshot_root=root)

    assert copied.read_bytes() == b"tampered\n"


def test_legacy_snapshot_is_left_untouched_and_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root, layout = _immutable_snapshot_fixture(tmp_path)
    monkeypatch.setattr(snapshot, "validate_archive_layout", lambda _path: (layout, []))
    source_hashes_path = root / "source_hashes.json"
    payload = json.loads(source_hashes_path.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    source_hashes_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    with pytest.raises(snapshot.SnapshotError, match="predates immutable provenance"):
        snapshot.build_trajectory_snapshot(config, snapshot_root=root)

    after = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    assert after == before


def test_snapshot_path_is_content_bound_to_archive_manifest(tmp_path: Path) -> None:
    config, first_root, layout = _immutable_snapshot_fixture(tmp_path)

    _write(layout.archive_manifest, b"relative_path\tbytes\ntrajectory.csv\t999\n")
    second_root = snapshot.default_snapshot_root(config)

    assert second_root != first_root
    assert first_root.exists()
    assert not second_root.exists()


