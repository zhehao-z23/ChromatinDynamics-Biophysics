from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsb_states.qc import (
    _resolved_thresholds,
    apply_primary_qc,
    build_missingness_analysis_gates,
    build_missingness_asymmetry,
    build_motion_coverage_diagnostics,
    build_track_loss_diagnostics,
    write_initial_qc,
)
from dsb_states.tables import OutputScopeError, canonical_output_roots


def _bundle_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bundle_id": ["pass", "left_low", "multi_fail"],
            "biological_experiment_id": pd.Series([pd.NA] * 3, dtype="string"),
            "acquisition_id": ["a1", "a1", "a2"],
            "fov_id": ["f1", "f1", "f2"],
            "cell_id": ["c1", "c2", "c3"],
            "hour_post_delivery": [2.0, 2.0, np.nan],
            "hour_mapping_source": [
                "folder label",
                "folder label",
                "inferred: adjacent labels agree",
            ],
            "hour_mapping_status": [
                "mapped_from_frozen_inventory",
                "mapped_from_frozen_inventory",
                "inferred_inventory_label_not_used",
            ],
            "site1_valid_count": [25, 19, 30],
            "site2_valid_count": [25, 30, 10],
            "paired_valid_count": [24, 18, 8],
            "longest_paired_run": [12, 12, 4],
            "paired_coverage": [0.60, 0.60, 0.20],
            "site1_coverage": [0.7, 0.4, 0.6],
            "site2_coverage": [0.7, 0.7, 0.2],
            "longest_site1_run": [12, 10, 8],
            "longest_site2_run": [12, 14, 4],
        }
    )


def _frames() -> pd.DataFrame:
    rows = []
    patterns = {
        "pass": ([1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 0, 0]),
        "left_low": ([1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]),
        "multi_fail": ([1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]),
    }
    metadata = _bundle_metadata().set_index("bundle_id")
    for bundle_id, (site1, site2, bp1) in patterns.items():
        source = metadata.loc[bundle_id]
        for index in range(4):
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "acquisition_id": source.acquisition_id,
                    "fov_id": source.fov_id,
                    "cell_id": source.cell_id,
                    "hour_post_delivery": source.hour_post_delivery,
                    "frame": index + 1,
                    "site1_valid": bool(site1[index]),
                    "site2_valid": bool(site2[index]),
                    "paired_valid": bool(site1[index] and site2[index]),
                    "bp1_valid": bool(bp1[index]),
                }
            )
    return pd.DataFrame(rows)


def _config() -> dict:
    return {
        "qc": {
            "min_site1_valid_frames": 20,
            "min_site2_valid_frames": 20,
            "min_paired_valid_frames": 20,
            "min_longest_paired_run": 10,
            "min_paired_coverage": 0.5,
        }
    }


def _motion_fixture() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame_rows = []
    bundle_rows = []
    for acquisition_index, acquisition in enumerate(("a1", "a2")):
        for cell_index in range(6):
            cell_id = f"{acquisition}_c{cell_index}"
            bundle_id = f"{cell_id}_b"
            site1_x = 0.0
            site2_x = 100.0
            valid_flags = {4: False, 8: False}
            for frame in range(1, 11):
                step = (1000.0 if frame in {3, 7} else 2.0) * (cell_index + 1)
                if frame > 1:
                    site1_x += step
                separation = 1200.0 if frame in {3, 7} else 100.0 + cell_index
                site2_x = site1_x + separation
                valid = valid_flags.get(frame, True)
                frame_rows.append(
                    {
                        "bundle_id": bundle_id,
                        "acquisition_id": acquisition,
                        "fov_id": f"f{acquisition_index + 1}",
                        "cell_id": cell_id,
                        "frame": frame,
                        "micro_time_s": float(frame - 1),
                        "site1_x_nm": site1_x,
                        "site1_y_nm": 0.0,
                        "site1_valid": valid,
                        "site2_x_nm": site2_x,
                        "site2_y_nm": 0.0,
                        "site2_valid": valid,
                    }
                )
            coverage = 0.35 + 0.1 * cell_index
            bundle_rows.append(
                {
                    "bundle_id": bundle_id,
                    "acquisition_id": acquisition,
                    "fov_id": f"f{acquisition_index + 1}",
                    "cell_id": cell_id,
                    "site1_coverage": coverage,
                    "site2_coverage": coverage,
                    "paired_coverage": coverage,
                }
            )
    config = {
        "qc": {
            "motion_missingness_high_quantile": 0.75,
            "motion_missingness_min_valid_steps_per_bundle": 1,
            "motion_missingness_min_cells_per_acquisition": 2,
        }
    }
    return pd.DataFrame(frame_rows), pd.DataFrame(bundle_rows), config


def test_primary_qc_uses_config_and_records_all_nonexclusive_reasons() -> None:
    result = apply_primary_qc(_frames(), _bundle_metadata(), _config())
    assert result.bundle_metadata_qc.bundle_id.tolist() == ["pass"]
    exclusions = result.bundle_exclusions.set_index("bundle_id")
    assert "min_site1_valid_frames" in exclusions.loc["left_low", "qc_exclusion_reasons"]
    assert "min_site2_valid_frames" in exclusions.loc["multi_fail", "qc_exclusion_reasons"]
    assert "min_paired_valid_frames" in exclusions.loc["multi_fail", "qc_exclusion_reasons"]
    assert result.summary["unmapped_hours_not_inferred"] == 1


def test_legacy_flank_config_keys_are_accepted_only_as_aliases() -> None:
    config = _config()
    config["qc"]["min_left_valid_frames"] = config["qc"].pop("min_site1_valid_frames")
    config["qc"]["min_right_valid_frames"] = config["qc"].pop("min_site2_valid_frames")
    result = apply_primary_qc(_frames(), _bundle_metadata(), config)
    assert "min_site1_valid_frames" in result.summary["thresholds"]
    assert not any("left" in reason for reason in result.summary["exclusion_counts_nonexclusive"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_site1_valid_frames", np.nan),
        ("min_site2_valid_frames", np.inf),
        ("min_paired_valid_frames", -1),
        ("min_longest_paired_run", 1.5),
        ("min_paired_coverage", -0.1),
        ("min_paired_coverage", 1.1),
    ],
)
def test_primary_qc_thresholds_are_finite_and_domain_valid(field: str, value: float) -> None:
    config = _config()
    config["qc"][field] = value
    with pytest.raises(ValueError, match="Invalid QC thresholds"):
        _resolved_thresholds(config)


def test_exclusion_flow_contains_reason_hour_and_acquisition_rows() -> None:
    result = apply_primary_qc(_frames(), _bundle_metadata(), _config())
    detailed = result.cohort_flowchart.loc[
        result.cohort_flowchart.flow_type.eq("reason_by_hour_acquisition_nonexclusive")
    ]
    assert {"reason", "hour_post_delivery", "acquisition_id", "bundle_count"}.issubset(
        detailed.columns
    )
    unmapped = detailed.loc[detailed.acquisition_id.eq("a2"), "hour_post_delivery"]
    assert unmapped.isna().all()


def test_missingness_asymmetry_is_neutral_and_exact() -> None:
    asymmetry = build_missingness_asymmetry(_frames(), _bundle_metadata())
    overall = asymmetry.loc[asymmetry.level.eq("overall")].iloc[0]
    assert overall.site1_only_frames == 3
    assert overall.site2_only_frames == 4
    assert overall.discordant_frames == 7
    assert "left" not in " ".join(asymmetry.columns).lower()


def test_t03_summary_reports_pairing_provenance_associations_and_explicit_gates() -> None:
    result = apply_primary_qc(_frames(), _bundle_metadata(), _config())
    pairing = result.summary["bundle_pairing_success"]
    assert pairing["definition"] == "at least one same-frame Site1/Site2 valid pair"
    assert pairing["successful_bundles"] == 3
    hours = result.summary["hour_label_provenance"]
    assert hours["folder_label_acquisitions"] == 1
    assert hours["inferred_inventory_label_left_unmapped_acquisitions"] == 1
    assert result.summary["technical_hierarchy"]["acquisition_fov_one_to_one"] is True
    assert result.summary["exact_timing_heterogeneity"]["status"] == "gated_unavailable"
    sensitivity = result.qc_sensitivity
    assert len(sensitivity) == 27
    assert set(sensitivity["min_paired_coverage"]) == {0.4, 0.6, 0.8}
    assert set(sensitivity["min_paired_valid_frames"]) == {10, 20, 30}
    assert set(sensitivity["min_longest_paired_run"]) == {5, 10, 20}
    assert set(result.coverage_associations["analysis_unit"]) == {"cell_mean_across_bundles"}
    gates = result.missingness_gates.set_index("analysis")
    for analysis in ("site_intensity", "bp1_intensity", "bp1_sbr", "bp1_component_metrics"):
        assert gates.loc[analysis, "status"] == "gated_unavailable"
    assert gates.loc["acquisition_fov_coverage", "status"] == "supported_descriptive"


def test_exact_timing_is_collapsed_to_acquisition_units_when_available() -> None:
    frames = _frames().copy()
    frames["frame_interval_s"] = [1.0, 1.1, 0.9, 1.0] * 3
    frames["exact_interval_median_s"] = [1.0] * len(frames)
    result = apply_primary_qc(frames, _bundle_metadata(), _config())
    timing = result.summary["exact_timing_heterogeneity"]
    assert timing["status"] == "supported_descriptive"
    assert timing["acquisition_count"] == 2
    assert len(result.timing_hierarchy) == 2
    assert set(result.timing_hierarchy["analysis_unit"]) == {"acquisition"}


def test_motion_missingness_diagnostics_use_cells_within_acquisition() -> None:
    frames, bundles, config = _motion_fixture()
    bundle_summary, associations = build_motion_coverage_diagnostics(frames, bundles, config)
    contrasts = build_track_loss_diagnostics(frames, config)

    assert set(bundle_summary["motion_metric"]) == {"site1", "site2", "common_mode"}
    supported_associations = associations.loc[associations["supported"]]
    assert len(supported_associations) == 6
    assert set(supported_associations["analysis_unit"]) == {"cell_within_acquisition"}
    assert supported_associations["n_cells"].eq(6).all()
    assert supported_associations["spearman_rho"].gt(0.8).all()

    supported_contrasts = contrasts.loc[contrasts["supported"]]
    assert len(supported_contrasts) == 6
    assert supported_contrasts["n_cells_with_both_strata"].ge(2).all()
    assert supported_contrasts["cell_mean_risk_difference"].gt(0).all()
    assert set(supported_contrasts["effect_analysis_unit"]) == {
        "acquisition_mean_of_within_cell_risk_differences"
    }

    gates = build_missingness_analysis_gates(
        frames,
        bundles.assign(hour_post_delivery=2.0),
        motion_coverage_associations=associations,
        dropout_risk_contrasts=contrasts,
    ).set_index("analysis")
    for analysis in (
        "coverage_vs_motion_magnitude",
        "track_loss_after_high_speed",
        "track_loss_after_large_separation",
    ):
        assert gates.loc[analysis, "status"] == "supported_descriptive"


def test_motion_missingness_settings_fail_closed_when_invalid() -> None:
    frames, bundles, config = _motion_fixture()
    config["qc"]["motion_missingness_high_quantile"] = 1.0
    with pytest.raises(ValueError, match="strictly between"):
        build_motion_coverage_diagnostics(frames, bundles, config)
    config["qc"]["motion_missingness_high_quantile"] = 0.75
    config["qc"]["motion_missingness_min_cells_per_acquisition"] = 1
    with pytest.raises(ValueError, match=">= 2"):
        build_track_loss_diagnostics(frames, config)


def test_localization_confidence_is_an_explicit_unavailable_gate() -> None:
    gates = build_missingness_analysis_gates(_frames(), _bundle_metadata()).set_index("analysis")
    assert gates.loc["localization_confidence", "status"] == "gated_unavailable"
    assert "localization" in gates.loc["localization_confidence", "reason"].lower()


def test_qc_writer_applies_scope_guard_before_creating_outputs(tmp_path: Path) -> None:
    result = apply_primary_qc(_frames(), _bundle_metadata(), _config())
    project = tmp_path / "analysis"
    allowed = canonical_output_roots(project)
    source = project / "results" / "production_source"
    output = source / "qc"
    with pytest.raises(OutputScopeError, match="overlaps"):
        write_initial_qc(
            result,
            output,
            project_root=project,
            allowed_roots=allowed,
            forbidden_roots=(source,),
        )
    assert not output.exists()


def test_qc_writer_validates_figure_scope_before_writing_tables(tmp_path: Path) -> None:
    result = apply_primary_qc(_frames(), _bundle_metadata(), _config())
    project = tmp_path / "analysis"
    output = project / "results" / "run" / "tables"
    source = project / "results" / "production_source"
    with pytest.raises(OutputScopeError, match="overlaps"):
        write_initial_qc(
            result,
            output,
            project_root=project,
            allowed_roots=canonical_output_roots(project),
            forbidden_roots=(source,),
            figure_dir=source / "figures",
        )
    assert not output.exists()


def test_qc_writer_succeeds_inside_results_allowlist(tmp_path: Path) -> None:
    result = apply_primary_qc(_frames(), _bundle_metadata(), _config())
    project = tmp_path / "analysis"
    output = project / "results" / "run" / "tables"
    paths = write_initial_qc(
        result,
        output,
        project_root=project,
        allowed_roots=canonical_output_roots(project),
    )
    assert paths["frame_table_qc"].is_file()
    assert paths["report"].is_file()
    assert paths["technical_hierarchy"].is_file()
    assert paths["timing_hierarchy"].is_file()
    assert paths["qc_sensitivity"].is_file()
    assert paths["coverage_associations"].is_file()
    assert paths["motion_coverage_bundle_summary"].is_file()
    assert paths["motion_coverage_associations"].is_file()
    assert paths["dropout_risk_contrasts"].is_file()
    assert paths["missingness_analysis_gates"].is_file()
    for figure in ("cohort_flow", "coverage_hierarchy", "missingness_asymmetry", "qc_sensitivity"):
        for extension in ("png", "pdf", "svg"):
            assert paths[f"figure_{figure}_{extension}"].is_file()
        assert paths[f"figure_{figure}_source"].is_file()
    report = paths["report"].read_text(encoding="utf-8")
    assert "technical/descriptive only" in report
    assert "folder label" in report
    assert "inferred inventory label" in report
    assert "component analyses" in report
    assert "not identify a biological effect" in report
