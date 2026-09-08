from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsb_states.tables import OutputScopeError
from dsb_states.tiered_qc import build_tiered_qc, write_tiered_qc


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    patterns = {
        "b1": (np.ones(10, dtype=bool), np.ones(10, dtype=bool)),
        "b2": (
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
            np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool),
        ),
        "b3": (
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
            np.zeros(10, dtype=bool),
        ),
    }
    frame_rows: list[dict] = []
    bundle_rows: list[dict] = []
    for index, (bundle_id, (site1, site2)) in enumerate(patterns.items()):
        paired = site1 & site2
        for frame_index in range(10):
            frame_rows.append(
                {
                    "biological_experiment_id": pd.NA,
                    "acquisition_id": f"a{index + 1}",
                    "fov_id": f"f{index + 1}",
                    "cell_id": f"c{index + 1}",
                    "bundle_id": bundle_id,
                    "allele_index": 1,
                    "hour_post_delivery": 2.0,
                    "frame": frame_index + 1,
                    "micro_time_s": float(frame_index),
                    "frame_interval_s": 1.0,
                    "timing_axis_primary": "exact_relative_time_s",
                    "site1_x_nm": float(frame_index) if site1[frame_index] else np.nan,
                    "site1_y_nm": 0.0 if site1[frame_index] else np.nan,
                    "site1_valid": site1[frame_index],
                    "site2_x_nm": float(frame_index + 1) if site2[frame_index] else np.nan,
                    "site2_y_nm": 0.0 if site2[frame_index] else np.nan,
                    "site2_valid": site2[frame_index],
                    "paired_valid": paired[frame_index],
                }
            )
        bundle_rows.append(
            {
                "biological_experiment_id": pd.NA,
                "acquisition_id": f"a{index + 1}",
                "fov_id": f"f{index + 1}",
                "cell_id": f"c{index + 1}",
                "bundle_id": bundle_id,
                "allele_index": 1,
                "hour_post_delivery": 2.0,
                "nominal_frame_count": 10,
                "site1_valid_count": int(site1.sum()),
                "site2_valid_count": int(site2.sum()),
                "paired_valid_count": int(paired.sum()),
                "longest_site1_run": int(site1.sum()),
                "longest_site2_run": int(site2.sum()),
                "longest_paired_run": int(paired.sum()),
                "site1_coverage": float(site1.mean()),
                "site2_coverage": float(site2.mean()),
                "paired_coverage": float(paired.mean()),
                "trajectory_usability_class": "PIPELINE_COMPLETE",
            }
        )
    return pd.DataFrame(frame_rows), pd.DataFrame(bundle_rows)


def _analysis_config() -> dict:
    return {
        "data": {
            "formal_usable_only": True,
            "review_data_policy": "exclude",
            "use_exact_timestamps": True,
        },
        "qc": {
            "min_site1_valid_frames": 8,
            "min_site2_valid_frames": 8,
            "min_paired_valid_frames": 8,
            "min_longest_paired_run": 8,
            "min_paired_coverage": 0.8,
        },
    }


def _contract() -> dict:
    return {
        "schema_version": 1,
        "tiers": {
            "frame_shared": {
                "minimum_bundle_length": None,
                "interpolation": "prohibited",
            },
            "site_independent": {
                "minimum_valid_frames": 1,
                "paired_site_required": False,
                "interpolation": "prohibited",
            },
            "trajectory_relaxed": {
                "min_site1_valid_frames": 5,
                "min_site2_valid_frames": 5,
                "min_paired_valid_frames": 5,
                "min_longest_paired_run": 3,
                "min_paired_coverage": None,
                "interpolation": "prohibited",
            },
            "primary_strict": {
                "threshold_source": "analysis_config.qc",
                "preserve_existing_primary_qc": True,
                "interpolation": "prohibited",
            },
        },
        "source_contract": {
            "timing_axis_value": "exact_relative_time_s",
            "allowed_trajectory_usability_classes": ["PIPELINE_COMPLETE"],
        },
        "interpretation": {"biological_speed_endpoint": "prohibited"},
    }


def test_four_tiers_preserve_all_observed_frames_and_existing_strict_membership() -> None:
    frames, bundles = _inputs()
    strict_reference = bundles.loc[bundles.bundle_id.eq("b1")]
    result = build_tiered_qc(
        frames,
        bundles,
        _analysis_config(),
        _contract(),
        strict_reference=strict_reference,
    )
    assert len(result.shared_frame_observations) == 16
    assert len(result.site1_observations) == 19
    assert len(result.site2_observations) == 18
    site = result.site_trajectory_membership.set_index(["site_id", "bundle_id"])
    assert site.loc[("site1", "b3"), "site_cohort_included"]
    assert not site.loc[("site2", "b3"), "site_cohort_included"]
    assert set(
        result.relaxed_trajectory_membership.loc[
            result.relaxed_trajectory_membership.relaxed_qc_included, "bundle_id"
        ]
    ) == {"b1", "b2"}
    assert set(
        result.strict_trajectory_membership.loc[
            result.strict_trajectory_membership.strict_qc_included, "bundle_id"
        ]
    ) == {"b1"}
    assert result.summary["strict_reference_match"] is True
    assert not result.summary["interpolation_used"]


def test_exact_time_formal_source_and_strict_reference_fail_closed() -> None:
    frames, bundles = _inputs()
    wrong_time = frames.copy()
    wrong_time["timing_axis_primary"] = "uniform_override"
    with pytest.raises(ValueError, match="timing_axis_primary"):
        build_tiered_qc(wrong_time, bundles, _analysis_config(), _contract())

    review_config = _analysis_config()
    review_config["data"]["review_data_policy"] = "include"
    with pytest.raises(ValueError, match="review_data_policy"):
        build_tiered_qc(frames, bundles, review_config, _contract())

    with pytest.raises(ValueError, match="does not match"):
        build_tiered_qc(
            frames,
            bundles,
            _analysis_config(),
            _contract(),
            strict_reference=bundles.loc[bundles.bundle_id.eq("b2")],
        )


def test_relaxed_qc_rejects_hidden_coverage_threshold() -> None:
    frames, bundles = _inputs()
    contract = _contract()
    contract["tiers"]["trajectory_relaxed"]["min_paired_coverage"] = 0.1
    with pytest.raises(ValueError, match="coverage threshold"):
        build_tiered_qc(frames, bundles, _analysis_config(), contract)


def test_writer_is_scoped_and_immutable(tmp_path: Path) -> None:
    frames, bundles = _inputs()
    result = build_tiered_qc(
        frames,
        bundles,
        _analysis_config(),
        _contract(),
        strict_reference=bundles.loc[bundles.bundle_id.eq("b1")],
    )
    inputs = {}
    for name in ("frame", "bundle", "strict"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        inputs[name] = path
    project = tmp_path / "project"
    with pytest.raises(OutputScopeError):
        write_tiered_qc(
            result,
            tmp_path / "outside",
            project_root=project,
            tier_contract=_contract(),
            input_paths=inputs,
        )
    output = project / "results" / "tiered"
    paths = write_tiered_qc(
        result,
        output,
        project_root=project,
        tier_contract=_contract(),
        input_paths=inputs,
    )
    assert paths["artifact_manifest"].is_file()
    assert "No interpolation" in paths["report"].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        write_tiered_qc(
            result,
            output,
            project_root=project,
            tier_contract=_contract(),
            input_paths=inputs,
        )
