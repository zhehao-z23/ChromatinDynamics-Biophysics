from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dsb_states.oligo_vcc import (
    RouseVccReference,
    build_vcc_fit_figures,
    build_vcc_matrix_figure,
    compute_oligo_vcc,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_text() -> str:
    return """# Pair VCC implementation audit

## Reference MATLAB logic retained

- Finite-window velocity: `v_delta(t)=[r(t+delta)-r(t)]/delta`.
- Two-locus cross-correlation: `C12_delta(tau)=<v1(t+tau) v2(t)^T>`.
- Oligo-LiveFISH display coordinate `tau/delta`, evaluated to 2.5.
- Corrected `calc_vel_corr_new.m` numerical tables and the
  `WLS_vel_corr_new.m` 1--1000 s communication-time search.
- The reference WLS value `alpha0=0.9` is fixed for the primary temporal comparison.

## Errors or dataset-incompatible assumptions corrected

- The supplied empirical MATLAB arrays are initialized with zeros. Missing coordinates and
  unsupported velocities therefore enter some means as real zero motion. The revised code uses
  NaN/finite-endpoint support only.
- The supplied code checks only the later y coordinate (`Dfiny~=0`), does not require both velocity
  endpoints, and treats a valid y=0 position as missing. The revised code requires finite x/y at
  both endpoints of both velocities.
- The supplied cross-correlation is normalized only by the first channel's autocorrelation energy.
  The revised primary denominator is the fixed symmetric scale
  `sqrt(<|v1|^2><|v2|^2>)`, so Site1/Site2 naming cannot change the result.
- The supplied code computes only one lead/lag direction. The revised result stores both
  `C12(tau)` and `C21(tau)`, transposes the reverse matrix, and uses their equal mean for the
  equilibrium-style Rouse comparison.
- The supplied code hard-codes one frame interval and later overwrites it for plotting. The revised
  code uses each acquisition's exact time schedule and matches integer frame offsets separately to
  10, 20, and 40 s measurement windows.
- The supplied final cross-correlation matrix allocates one extra row that is never filled. The
  revised tables emit only supported lags.

## Final analysis contract

- Primary empirical object: `delta x tau x 2 x 2` normalized VCC matrix for every bundle.
- Rouse fit readout: the rotation-invariant trace of the bidirectionally symmetrized matrix.
- Pooling: time average within bundle, then equal bundle mean within folder-hour.
- Metric support only: at least eight directly observed velocity pairs at a lag.
- No coordinate interpolation, trajectory compression, separation threshold, motion threshold,
  T3/T4 global QC, 53BP1 requirement, learned-state filter, or acquisition exclusion.
- Folder-hour is time after Cas9 delivery, not synchronized time since an observed cut.
- `tau_Delta_n` is a descriptive model parameter unless the Rouse curve shape is adequate; it is
  not a direct measurement of cutting time or repair kinetics.
"""


def _report(result: object) -> str:
    fits = result.communication_time_fits.sort_values("hour_post_delivery")
    included = result.support_census.loc[result.support_census["unit_included"].astype(bool)]
    lines = [
        "# Oligo-LiveFISH-style pair VCC and Rouse communication-time fit",
        "",
        "## Output summary",
        "",
        f"- {len(result.directional_unit_curves):,} supported directional scalar curve points.",
        f"- {len(result.directional_unit_matrices):,} supported directional matrix-component rows.",
        f"- {included['unit_id'].nunique():,} Site1/Site2 bundles contribute at least one VCC point.",
        (
            f"- {included['crop_id'].nunique():,} crops and "
            f"{included['nd2_id'].nunique():,} acquisitions are represented."
        ),
        "- Full bidirectional 2 x 2 matrices are retained; only their symmetric trace is fitted.",
        "",
        "## Folder-hour fits (fixed Rouse alpha0=0.9)",
        "",
    ]
    for row in fits.itertuples(index=False):
        if np.isfinite(row.communication_time_ci_low_s):
            interval = (
                f" [{row.communication_time_ci_low_s:.0f},"
                f" {row.communication_time_ci_high_s:.0f}]"
            )
        else:
            interval = " [CI unavailable]"
        lines.append(
            f"- {row.hour_post_delivery:g} h: tau_Delta_n={row.communication_time_s:.0f} s"
            f"{interval}; RMSE={row.rmse:.3f}; R2={row.r_squared:.3f}; "
            f"n={row.n_units} bundles; {row.fit_status}."
        )
    if len(fits) >= 2:
        rho = fits[["hour_post_delivery", "communication_time_s"]].corr(
            method="spearman"
        ).iloc[0, 1]
        minimum = fits.loc[fits["communication_time_s"].idxmin()]
        maximum = fits.loc[fits["communication_time_s"].idxmax()]
        lines.extend(
            [
                "",
                "## Preliminary temporal reading",
                "",
                (
                    f"- Descriptive Spearman rho(hour, tau_Delta_n)={rho:+.3f}; no p-value or "
                    "biological-replicate claim is made."
                ),
                (
                    f"- The smallest fitted communication time is {minimum.communication_time_s:.0f} s "
                    f"at {minimum.hour_post_delivery:g} h; the largest is "
                    f"{maximum.communication_time_s:.0f} s at {maximum.hour_post_delivery:g} h."
                ),
                "- Temporal interpretation must be conditioned on fit RMSE, boundary status, and alpha sensitivity.",
            ]
        )
    predictions = result.fit_predictions.copy()
    predictions["residual"] = predictions["observed_mean"] - predictions["predicted_vcc"]
    late = predictions.loc[predictions["scaled_lag_bin"].ge(1.25)]
    late_by_hour = late.groupby("hour_post_delivery")["residual"].mean()
    weak_fits = fits.loc[fits["r_squared"].lt(0.5), "hour_post_delivery"].to_numpy(float)
    lines.extend(
        [
            "",
            "## Model adequacy is the main result",
            "",
            (
                f"- {len(weak_fits)}/{len(fits)} folder-hours have R2<0.5; the fitted communication "
                "times should therefore not be read as a clean kinetic trajectory."
            ),
            (
                f"- Mean empirical-minus-Rouse residual at tau/delta>=1.25 is positive at "
                f"{int(np.count_nonzero(late_by_hour.gt(0)))}/{len(late_by_hour)} hours. The data "
                "retain a positive long-lag component that the supplied equilibrium Rouse curves do not capture."
            ),
            "- This mismatch is compatible with unresolved slow common motion or model misspecification; it does not by itself identify a biological mechanism.",
        ]
    )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- The primary alpha0=0.9 is fixed from the supplied corrected Oligo-LiveFISH WLS workflow; alpha0=0.7, 0.8, and 1.0 are archived as sensitivity fits.",
            "- Crop-cluster intervals are descriptive because formal biological replicate IDs are unavailable.",
            "- Unequal acquisition cadence is handled explicitly; acquisitions are otherwise pooled as the user-confirmed same batch.",
            "- Tracking missingness can remain outcome-dependent and can still bias long-lag estimates.",
            "",
        ]
    )
    return "\n".join(lines)


def _figure_index(hours: list[float]) -> str:
    lines = [
        "# Figure index",
        "",
        "- `fig_vcc_rouse_fits_by_hour`: At every folder-hour, do the 10/20/40 s pair-VCC curves agree with one Rouse communication time?",
        "- `fig_vcc_communication_time_by_hour`: How does fitted communication time vary with folder-hour?",
        "- `fig_vcc_matrix_components_by_hour`: Do all four components of the primary 2 x 2 VCC tensor change with folder-hour?",
    ]
    lines.extend(
        f"- `vcc_{hour:g}h_rouse_fit`: Full-size empirical and Rouse-fit panel for {hour:g} h."
        for hour in hours
    )
    lines.extend(
        [
            "",
            "Solid lines are equal-bundle empirical means; same-color dashed lines are corrected numerical-table Rouse predictions. The MATLAB-style display omits uncertainty ribbons; uncertainty remains in the source tables.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--empirical-reference-dir", type=Path, required=True)
    parser.add_argument("--vvcf-reference-dir", type=Path, required=True)
    parser.add_argument("--reference-readme-pdf", type=Path, required=True)
    args = parser.parse_args()

    cache = args.cache.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    empirical_reference = args.empirical_reference_dir.resolve()
    vvcf_reference = args.vvcf_reference_dir.resolve()
    reference_pdf = args.reference_readme_pdf.resolve()
    frames_path = cache / "tables" / "bundle_frame_master.parquet"
    index_path = cache / "tables" / "bundle_index.parquet"
    table_dir = vvcf_reference / "tables"
    empirical_sources = [
        empirical_reference / "v_autocorrelation.m",
        empirical_reference / "v_cross_correlation.m",
        empirical_reference / "v_cross_correlation_GPR.m",
    ]
    model_sources = [
        vvcf_reference / "calc_vel_corr_new.m",
        vvcf_reference / "WLS_vel_corr_new.m",
        vvcf_reference / "plot_vel_corr.m",
        vvcf_reference / "exp_corr.m",
        reference_pdf,
        *sorted(table_dir.glob("table*.mat")),
    ]
    sources = [
        frames_path,
        index_path,
        cache / "CACHE_CONTRACT.json",
        config_path,
        *empirical_sources,
        *model_sources,
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite result: {output}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["oligo_vcc"]
    reference = RouseVccReference.from_directory(table_dir)
    result = compute_oligo_vcc(
        pd.read_parquet(frames_path),
        pd.read_parquet(index_path),
        reference,
        target_delta_s=tuple(float(value) for value in settings["target_delta_s"]),
        delta_relative_tolerance=float(settings["delta_relative_tolerance"]),
        maximum_scaled_lag=float(settings["maximum_scaled_lag"]),
        minimum_velocity_pairs=int(settings["minimum_velocity_pairs"]),
        scaled_lag_bin_centers=tuple(
            float(value) for value in settings["scaled_lag_bin_centers"]
        ),
        rouse_alpha_fixed=float(settings["rouse_alpha_fixed"]),
        rouse_alpha_sensitivity=tuple(
            float(value) for value in settings["rouse_alpha_sensitivity"]
        ),
        communication_time_bounds_s=tuple(
            int(value) for value in settings["communication_time_bounds_s"]
        ),
        minimum_hour_bin_units_for_fit=int(
            settings["minimum_hour_bin_units_for_fit"]
        ),
        bootstrap_iterations=int(settings["bootstrap_iterations"]),
        bootstrap_seed=int(settings["bootstrap_seed"]),
    )

    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        tables = staging / "tables"
        figures = staging / "figures"
        tables.mkdir()
        figures.mkdir()
        parquet_tables = {
            "directional_unit_curves.parquet": result.directional_unit_curves,
            "directional_unit_matrices.parquet": result.directional_unit_matrices,
            "symmetric_unit_curves.parquet": result.symmetric_unit_curves,
            "symmetric_unit_matrices.parquet": result.symmetric_unit_matrices,
            "support_census.parquet": result.support_census,
            "binned_unit_curves.parquet": result.binned_unit_curves,
            "hour_curves.parquet": result.hour_curves,
            "hour_matrix_curves.parquet": result.hour_matrix_curves,
        }
        for filename, table in parquet_tables.items():
            table.to_parquet(tables / filename, index=False, compression="zstd")
        csv_tables = {
            "hour_curves.csv": result.hour_curves,
            "hour_matrix_curves.csv": result.hour_matrix_curves,
            "communication_time_fits.csv": result.communication_time_fits,
            "fit_predictions.csv": result.fit_predictions,
            "rouse_alpha_sensitivity.csv": result.alpha_sensitivity,
        }
        for filename, table in csv_tables.items():
            table.to_csv(tables / filename, index=False, lineterminator="\n")
        support_summary = result.support_census.groupby(
            ["hour_post_delivery", "target_delta_s", "unit_included", "exclusion_reason"],
            as_index=False,
            dropna=False,
        ).agg(n_units=("unit_id", "nunique"), n_crops=("crop_id", "nunique"))
        support_summary.to_csv(
            tables / "support_summary.csv", index=False, lineterminator="\n"
        )

        figure_sets = build_vcc_fit_figures(result, figures)
        matrix_figure = build_vcc_matrix_figure(
            result,
            figures / "fig_vcc_matrix_components_by_hour",
            target_delta_s=float(settings["target_delta_s"][0]),
        )
        contract = {
            "schema_version": 1,
            "metric": "two-locus finite-window velocity cross-correlation",
            "primary_object": "delta x tau x 2 x 2 bidirectional VCC tensor",
            "equations": {
                "velocity": "v_delta(t)=[r(t+delta)-r(t)]/delta",
                "directional_matrix": "C12_delta(tau)=<v1(t+tau) v2(t)^T>",
                "normalization": "sqrt(<|v1|^2><|v2|^2>) fixed across lag",
                "symmetrization": "0.5*(C12(tau)+transpose(C21(tau)))",
                "fit_readout": "trace of the symmetrized normalized matrix",
                "rouse": "corrected calc_vel_corr_new numerical tables; alpha0 fixed; tau_Delta_n grid WLS",
            },
            "target_delta_s": list(settings["target_delta_s"]),
            "integer_frame_matching_relative_tolerance": float(
                settings["delta_relative_tolerance"]
            ),
            "maximum_scaled_lag": float(settings["maximum_scaled_lag"]),
            "minimum_velocity_pairs_per_direction_per_unit_lag": int(
                settings["minimum_velocity_pairs"]
            ),
            "minimum_hour_bin_units_for_fit": int(
                settings["minimum_hour_bin_units_for_fit"]
            ),
            "rouse_alpha_fixed": float(settings["rouse_alpha_fixed"]),
            "rouse_alpha_sensitivity": list(settings["rouse_alpha_sensitivity"]),
            "communication_time_bounds_s": list(settings["communication_time_bounds_s"]),
            "pooling": "time-average within bundle and direction; transpose reverse direction; equal direction mean; equal bundle mean within folder-hour",
            "uncertainty": "between-bundle sample SD; crop-cluster bootstrap 95% interval for communication time",
            "coordinate_interpolation": False,
            "excluded_filters": [
                "T3/T4 global pair QC",
                "motion magnitude",
                "Site1/Site2 separation",
                "53BP1 detection or recruitment",
                "hour or acquisition identity",
                "learned state",
            ],
            "macro_time_warning": "folder-hour after Cas9 delivery is not observed time since cutting",
        }
        (staging / "METHOD_CONTRACT.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (staging / "IMPLEMENTATION_AUDIT.md").write_text(
            _audit_text(), encoding="utf-8"
        )
        (staging / "REPORT.md").write_text(_report(result), encoding="utf-8")
        hours = sorted(
            float(value)
            for value in result.communication_time_fits["hour_post_delivery"].unique()
        )
        (staging / "FIGURE_INDEX.md").write_text(
            _figure_index(hours), encoding="utf-8"
        )

        code_sources = [
            Path(__file__).resolve(),
            PROJECT_ROOT / "src" / "dsb_states" / "oligo_vcc.py",
            PROJECT_ROOT / "src" / "dsb_states" / "physical_metrics.py",
            PROJECT_ROOT / "src" / "dsb_states" / "time_resolved_physics.py",
        ]
        artifacts = [path for path in staging.rglob("*") if path.is_file()]
        manifest = {
            "sources": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sources + code_sources
            ],
            "artifacts": [
                {
                    "path": path.relative_to(staging).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(artifacts)
            ],
            "figure_sets": {
                "atlas": [str(path.relative_to(staging)) for path in figure_sets["atlas"]],
                "trend": [str(path.relative_to(staging)) for path in figure_sets["trend"]],
                "matrix": [str(path.relative_to(staging)) for path in matrix_figure],
                "per_hour": {
                    f"{hour:g}": [str(path.relative_to(staging)) for path in paths]
                    for hour, paths in figure_sets["per_hour"].items()
                },
            },
        }
        (staging / "OUTPUT_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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
                "symmetric_curve_rows": len(result.symmetric_unit_curves),
                "fit_rows": len(result.communication_time_fits),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
