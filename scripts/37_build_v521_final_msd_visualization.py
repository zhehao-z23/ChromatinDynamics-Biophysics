#!/usr/bin/env python
"""Build the compact final MSD figure and select complete-coverage alpha cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dsb_states.v521_msd_final_visualization import (
    build_complete_coverage_alpha_table,
    build_final_msd_figure,
    build_window_mean_summary,
    save_figure_set,
    select_alpha_cases,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-run", type=Path, required=True)
    parser.add_argument("--complete-pair-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    physics = args.physics_run.resolve()
    selection_path = args.complete_pair_csv.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tables = physics / "tables"
    sources = {
        "hour_curves": tables / "hour_curves.parquet",
        "fits": tables / "fits.csv",
        "predicted_curves": tables / "predicted_curves.parquet",
        "unit_curves": tables / "unit_curves.parquet",
        "primary_bins": tables / "primary_unit_lag_bins.parquet",
        "selection": selection_path,
        "config": config_path,
        "physics_method": physics / "FORMULAS_AND_METHODS.md",
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    fit_lower, fit_upper = (float(value) for value in config["fit_window_s"])
    support = config["case_selection"]

    staging.mkdir(parents=True)
    try:
        figure_dir = staging / "figures"
        table_dir = staging / "tables"
        config_dir = staging / "config"
        figure_dir.mkdir()
        table_dir.mkdir()
        config_dir.mkdir()

        hour_curves = pd.read_parquet(sources["hour_curves"])
        fits = pd.read_csv(sources["fits"])
        predicted = pd.read_parquet(sources["predicted_curves"])
        unit_curves = pd.read_parquet(sources["unit_curves"])
        primary_bins = pd.read_parquet(sources["primary_bins"])
        complete = pd.read_csv(selection_path)

        window = build_window_mean_summary(
            fits,
            primary_bins,
            lower_s=fit_lower,
            upper_s=fit_upper,
            bootstrap_iterations=int(config["bootstrap_iterations"]),
            bootstrap_seed=int(config["bootstrap_seed"]),
        )
        alpha = build_complete_coverage_alpha_table(
            unit_curves,
            complete,
            lower_s=fit_lower,
            upper_s=fit_upper,
            minimum_pairs=int(support["minimum_pairs_per_lag"]),
            minimum_points=int(support["minimum_fit_lags"]),
            minimum_span_fold=float(support["minimum_span_fold"]),
        )
        targets = tuple(
            (str(item["case_id"]), float(item["target_alpha"]))
            for item in support["targets"]
        )
        selected = select_alpha_cases(
            alpha,
            primary_hour=float(support["primary_hour"]),
            targets=targets,
            minimum_r2=float(support["minimum_r2"]),
        )

        window.to_csv(table_dir / "fitted_window_mean_msd_by_hour.csv", index=False)
        alpha.to_csv(table_dir / "complete_coverage_trajectory_alpha.csv", index=False)
        alpha.to_parquet(table_dir / "complete_coverage_trajectory_alpha.parquet", index=False)
        selected.to_csv(table_dir / "selected_alpha_case_fits.csv", index=False)
        selected_rows = complete.merge(
            selected[
                [
                    "bundle_id",
                    "case_id",
                    "display_label",
                    "site",
                    "target_alpha",
                    "alpha",
                    "r2_log",
                    "distance_to_target",
                ]
            ],
            on="bundle_id",
            how="inner",
            validate="one_to_one",
        )
        selected_rows.to_csv(table_dir / "selected_alpha_cases_for_review.csv", index=False)

        case_config = {
            "schema_version": 4,
            "cases": [
                {
                    "case_id": str(row.case_id),
                    "display_label": str(row.display_label),
                    "bundle_id": str(row.bundle_id),
                }
                for row in selected.itertuples(index=False)
            ],
            "display": config["case_display"],
        }
        (config_dir / "selected_alpha_case_review.json").write_text(
            json.dumps(case_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        figure = build_final_msd_figure(
            hour_curves,
            fits,
            predicted,
            window,
            representative_hours=config["representative_hours"],
            lower_s=fit_lower,
            upper_s=fit_upper,
        )
        figure_paths = save_figure_set(
            figure, figure_dir / "fig_final_msd_representative_fits_and_hour_trend"
        )
        plt.close(figure)

        contract = {
            "population_input": "frozen v5.2.1 unfiltered time-resolved physics v2 tables",
            "population_unit": "trajectory; time-average first, then equal-trajectory hour mean",
            "representative_hours": config["representative_hours"],
            "fit_model": "raw MSD(tau)=A*tau^alpha without localization or motion-blur correction",
            "fit_window_s": [fit_lower, fit_upper],
            "uncertainty": {
                "raw_curve": "between-trajectory sample SD",
                "whole_window_trend": "crop-cluster bootstrap 95% CI",
            },
            "whole_window_endpoint": (
                "mean of the fitted MSD curve over d(log lag) from 10 to 50 s; "
                "units remain um^2"
            ),
            "case_scope": (
                "previously frozen 100%-coverage Site1/Site2 paired cohort after exclusion "
                "of acquisitions with only 5 or 10 movie frames"
            ),
            "case_fit_support": support,
            "case_selection_role": "visual case study only; not a new population endpoint",
            "coordinate_interpolation": False,
            "excluded_case_filters": ["motion", "separation", "53BP1", "manual appearance"],
        }
        (staging / "METHOD_CONTRACT.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        selected_lines = []
        for row in selected.itertuples(index=False):
            selected_lines.append(
                f"- `{row.case_id}`: {row.bundle_id}, {row.site}, alpha={row.alpha:.3f}, "
                f"R2={row.r2_log:.3f}, target={row.target_alpha:g}."
            )
        (staging / "REPORT.md").write_text(
            "\n".join(
                [
                    "# Compact final v5.2.1 MSD visualization",
                    "",
                    "## PPT figure",
                    "",
                    (
                        "Three representative raw ensemble-MSD panels (1.5, 3 and 10 h) retain "
                        "between-trajectory SD and show the fitted alpha in-panel. The fourth "
                        "panel summarizes the complete fitted 10--50 s curve by its log-lag "
                        "window mean."
                    ),
                    "",
                    "## Complete-coverage 3 h alpha cases",
                    "",
                    *selected_lines,
                    "",
                    (
                        "The cases were selected numerically from the frozen complete-coverage "
                        "cohort, not by image appearance. Alpha is descriptive raw-MSD scaling "
                        "and is not a validated discrete motion-state label."
                    ),
                    "",
                    "## Limits",
                    "",
                    "- Folder-hour is time after Cas9 delivery, not observed time since cutting.",
                    "- The 10--50 s window spans only five-fold.",
                    "- The hollow 5 h trend marker flags sparse Site1 support.",
                    "- No localization-error or motion-blur correction is available.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        all_sources = [*sources.values(), Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "src" / "dsb_states" / "v521_msd_final_visualization.py"]
        artifacts = [path for path in staging.rglob("*") if path.is_file()]
        (staging / "OUTPUT_MANIFEST.json").write_text(
            json.dumps(
                {
                    "sources": [
                        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                        for path in all_sources
                    ],
                    "artifacts": [
                        {
                            "path": path.relative_to(staging).as_posix(),
                            "bytes": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                        for path in sorted(artifacts)
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise

    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "figures": [str(output / path.relative_to(staging)) for path in figure_paths],
                "selected_cases": selected[
                    ["case_id", "bundle_id", "site", "target_alpha", "alpha", "r2_log"]
                ].to_dict("records"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
