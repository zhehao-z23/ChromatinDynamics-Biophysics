#!/usr/bin/env python
"""Build the Oligo-LiveFISH-style multi-delta VAC analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dsb_states.oligo_vac import (
    build_oligo_vac_alpha_figure,
    build_oligo_vac_hour_figures,
    build_vac_msd_alpha_consistency_figure,
    compute_oligo_vac,
)
from dsb_states.time_resolved_physics_summary import fbm_normalized_vac


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_lines(fits: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for record in fits.sort_values(["hour_post_delivery", "site"]).itertuples():
        interval = ""
        if pd.notna(record.alpha_ci_low) and pd.notna(record.alpha_ci_high):
            interval = (
                f" (95% crop-bootstrap CI {record.alpha_ci_low:.3f}-{record.alpha_ci_high:.3f})"
            )
        lines.append(
            f"- {record.hour_post_delivery:g} h {record.site}: alpha={record.alpha:.3f}{interval}; "
            f"RMSE={record.rmse:.3f}; n={record.n_units} trajectories."
        )
    return lines


def _audit_text() -> str:
    return """# VAC implementation audit

## What was already correct

- The old project calculation used finite-difference velocities
  `v_delta(t)=[r(t+delta)-r(t)]/delta`, the dot product
  `<v_delta(t+tau) dot v_delta(t)>`, and zero-lag energy normalization.
- Missing frames stayed on the original one-based acquisition schedule; coordinates were not
  interpolated or compressed.
- Trajectories were time-averaged first and then equal-weighted within folder-hour.

## What was incomplete or potentially misleading

- The old headline plot retained only the acquisition-specific delta nearest 10 s. It therefore
  could show an anticorrelation trough but could not test the Oligo-LiveFISH multi-delta collapse.
- The old fit used only that single delta and treated VAC-derived alpha as a headline result.
  Oligo-LiveFISH instead overlays Equation 9 using alpha estimated independently from MSD. The
  revised main figures follow that self-consistency test. A joint 10/20/40 s VAC-only alpha fit is
  retained only as a diagnostic and uses the local/reference range 0.25-1.0.
- The old physical-time binning was optimized for cross-acquisition comparison, not direct
  tau/delta collapse. The revised collapse bins tau/delta itself.

## Important separation of VAC and VCC reference code

- `exp_corr.m` and the Oligo-LiveFISH Figure 5J layout motivate the raw-tau and rescaled-tau/delta
  panels used here.
- `calc_vel_corr_new.m` fixes reversed interpolation weights in an older numerical-table
  implementation. That table and `WLS_vel_corr_new.m` describe Rouse-model two-locus VCC
  communication-time fitting, not the single-locus VAC Eq. 9 fit.
- The local file `site_1and2_canonical_group_280kb_vcctheory.eps` and Oligo-LiveFISH Methods
  Eq. 10/Movie S2 independently identify that numerical-table workflow as VCC. It is therefore
  intentionally not applied to Site1 or Site2 VAC.

## Revised VAC contract

- Oligo-LiveFISH Eq. 7: `C_v^delta(tau)=<v_delta(t+tau) dot v_delta(t)>`.
- Oligo-LiveFISH Eq. 8: `v_delta(t)=[r(t+delta)-r(t)]/delta`.
- Normalize each trajectory/delta curve by its own `C_v^delta(0)` before pooling.
- Match each acquisition to the nearest integer frame offset for target delta=10, 20, or 40 s;
  require the schedule-median delta to lie within +/-25% of target.
- A reported trajectory/lag requires at least eight directly observed velocity pairs.
- No trajectory interpolation, pair-QC requirement, motion/separation threshold, 53BP1 filter,
  hour filter, acquisition filter, or learned-state filter is used.
- At each tau/delta bin, first average supported deltas within trajectory, then equal-weight
  trajectories within folder-hour.
- Overlay Oligo-LiveFISH Eq. 9 using the independently estimated folder-hour/site MSD exponent.
- Fit the collapsed hour curve to Eq. 9 over 0 < tau/delta <= 2.5 only as a diagnostic. Confidence
  intervals are crop-cluster bootstraps and remain descriptive because biological replicates are
  unavailable.
- Suppress only displayed hour/delta bins with fewer than eight contributing trajectories; full
  values remain in the tables and this display rule does not affect fitting.
"""


def _report(result: Any, consistency: pd.DataFrame) -> str:
    included = result.support_census.loc[result.support_census["unit_included"].astype(bool)]
    return "\n".join(
        [
            "# Oligo-LiveFISH-style multi-delta VAC",
            "",
            "## Output summary",
            "",
            f"- {len(result.unit_curves):,} supported trajectory/delta/lag points.",
            f"- {included['unit_id'].nunique():,} trajectories contribute at least one VAC point.",
            (
                f"- {included['crop_id'].nunique():,} crops and "
                f"{included['nd2_id'].nunique():,} acquisitions represented."
            ),
            "- Site1 and Site2 are calculated independently; paired-site availability is not required.",
            "- Target velocity windows are 10, 20, and 40 s with no coordinate interpolation.",
            "",
            "## Primary Oligo-LiveFISH consistency test",
            "",
            "- The blue Equation 9 overlays use alpha from the independently fitted MSD curves.",
            "- The VAC-only alpha estimates below are diagnostic, not replacement MSD exponents.",
            "",
            *[
                (
                    f"- {row.hour_post_delivery:g} h {row.site}: "
                    f"MSD alpha={row.alpha_msd:.3f}; VAC-only alpha={row.alpha_vac:.3f}; "
                    f"Eq. 9 RMSE at MSD alpha={row.rmse_at_msd_alpha:.3f}."
                )
                for row in consistency.sort_values(["hour_post_delivery", "site"]).itertuples()
            ],
            "",
            "## Interpretation",
            "",
            "- The original-style raw-lag panels show whether the anticorrelation trough moves with delta.",
            "- The rescaled panels show whether the curves collapse near tau/delta=1, the key fBM diagnostic.",
            "- Agreement between the VAC collapse and the MSD-alpha Eq. 9 curve supports fBM self-consistency; disagreement is informative and is not refitted away in the primary panel.",
            "- Folder-hour is time after Cas9 delivery, not synchronized time since an observed cut.",
            "- No biological-replicate inference or causal claim is made.",
            "",
        ]
    )


def _figure_index(hours: list[float]) -> str:
    lines = [
        "# Figure index",
        "",
        "- `vac_alpha_multi_delta_by_hour`: How does the VAC-only diagnostic Eq. 9 fit vary with folder-hour?",
        "- `vac_msd_alpha_consistency_by_hour`: Does the VAC-only diagnostic exponent agree with the independently estimated MSD exponent?",
    ]
    lines.extend(
        f"- `vac_{hour:g}h_oligo_style`: At {hour:g} h, do Site1/Site2 VAC troughs shift with delta "
        "and collapse after rescaling tau/delta?"
        for hour in hours
    )
    lines.extend(
        [
            "",
            (
                "Each hour figure uses Oligo-LiveFISH-style raw lag panels on the left and "
                "rescaled tau/delta panels with the MSD-alpha Eq. 9 prediction on the right."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _load_msd_alphas(path: Path) -> pd.DataFrame:
    fits = pd.read_csv(path)
    required = {
        "metric",
        "site",
        "hour_post_delivery",
        "estimate_name",
        "estimate",
        "ci_low",
        "ci_high",
    }
    missing = sorted(required.difference(fits.columns))
    if missing:
        raise ValueError(f"MSD fit table is missing required columns: {missing}")
    selected = fits.loc[
        fits["metric"].astype(str).eq("msd")
        & fits["estimate_name"].astype(str).eq("exponent"),
        ["hour_post_delivery", "site", "estimate", "ci_low", "ci_high", "n_units"],
    ].copy()
    selected = selected.rename(
        columns={
            "estimate": "alpha_msd",
            "ci_low": "alpha_msd_ci_low",
            "ci_high": "alpha_msd_ci_high",
            "n_units": "n_units_msd",
        }
    )
    if selected.duplicated(["hour_post_delivery", "site"]).any():
        raise ValueError("MSD fit table has duplicate hour/site exponent rows")
    return selected.sort_values(["hour_post_delivery", "site"], kind="stable").reset_index(
        drop=True
    )


def _build_consistency_table(
    result: Any,
    msd_alphas: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fit in result.fits.itertuples(index=False):
        reference = msd_alphas.loc[
            np.isclose(msd_alphas["hour_post_delivery"], float(fit.hour_post_delivery))
            & msd_alphas["site"].eq(str(fit.site))
        ]
        if len(reference) != 1:
            raise ValueError(
                f"Expected one MSD alpha for {fit.hour_post_delivery:g} h {fit.site}; "
                f"found {len(reference)}"
            )
        reference_row = reference.iloc[0]
        curve = result.collapsed_hour_curves.loc[
            np.isclose(
                result.collapsed_hour_curves["hour_post_delivery"],
                float(fit.hour_post_delivery),
            )
            & result.collapsed_hour_curves["site"].eq(str(fit.site))
            & result.collapsed_hour_curves["scaled_lag_bin"].gt(0.0)
        ]
        x = curve["scaled_lag_bin"].to_numpy(float)
        observed = curve["mean"].to_numpy(float)
        weights = curve["n_units"].to_numpy(float)
        alpha_msd = float(reference_row["alpha_msd"])
        predicted = fbm_normalized_vac(x, alpha_msd)
        rmse = float(np.sqrt(np.average((observed - predicted) ** 2, weights=weights)))
        rows.append(
            {
                "hour_post_delivery": float(fit.hour_post_delivery),
                "site": str(fit.site),
                "alpha_msd": alpha_msd,
                "alpha_msd_ci_low": float(reference_row["alpha_msd_ci_low"]),
                "alpha_msd_ci_high": float(reference_row["alpha_msd_ci_high"]),
                "alpha_vac": float(fit.alpha),
                "alpha_vac_ci_low": float(fit.alpha_ci_low),
                "alpha_vac_ci_high": float(fit.alpha_ci_high),
                "alpha_vac_minus_msd": float(fit.alpha) - alpha_msd,
                "rmse_at_msd_alpha": rmse,
                "rmse_at_vac_alpha": float(fit.rmse),
                "n_units_vac": int(fit.n_units),
                "n_units_msd": int(reference_row["n_units_msd"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["hour_post_delivery", "site"], kind="stable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msd-fits", type=Path, required=True)
    parser.add_argument("--reference-vvcf-dir", type=Path, required=True)
    parser.add_argument("--reference-readme-pdf", type=Path, required=True)
    args = parser.parse_args()

    cache = args.cache.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    msd_fits_path = args.msd_fits.resolve()
    reference_dir = args.reference_vvcf_dir.resolve()
    reference_pdf = args.reference_readme_pdf.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite result: {output}")
    frames_path = cache / "tables" / "bundle_frame_master.parquet"
    index_path = cache / "tables" / "bundle_index.parquet"
    cache_contract = cache / "CACHE_CONTRACT.json"
    reference_files = [
        reference_dir / "exp_corr.m",
        reference_dir / "calc_vel_corr_new.m",
        reference_dir / "WLS_vel_corr_new.m",
        reference_dir / "plot_vel_corr.m",
        reference_dir / "site_1and2_canonical_group_280kb_vcctheory.eps",
        reference_pdf,
    ]
    sources = [
        frames_path,
        index_path,
        cache_contract,
        config_path,
        msd_fits_path,
        *reference_files,
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["oligo_vac"]
    targets = tuple(float(value) for value in settings["target_delta_s"])
    msd_alphas = _load_msd_alphas(msd_fits_path)
    result = compute_oligo_vac(
        pd.read_parquet(frames_path),
        pd.read_parquet(index_path),
        target_delta_s=targets,
        delta_relative_tolerance=float(settings["delta_relative_tolerance"]),
        maximum_scaled_lag=float(settings["maximum_scaled_lag"]),
        minimum_velocity_pairs=int(settings["minimum_velocity_pairs"]),
        raw_lag_bin_centers_s=tuple(float(value) for value in settings["raw_lag_bin_centers_s"]),
        scaled_lag_bin_centers=tuple(float(value) for value in settings["scaled_lag_bin_centers"]),
        alpha_bounds=tuple(float(value) for value in settings["alpha_bounds"]),
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
        table_specs = {
            "unit_vac_curves.parquet": result.unit_curves,
            "support_census.parquet": result.support_census,
            "raw_hour_curves.parquet": result.raw_hour_curves,
            "scaled_delta_hour_curves.parquet": result.scaled_delta_hour_curves,
            "collapsed_unit_curves.parquet": result.collapsed_unit_curves,
            "collapsed_hour_curves.parquet": result.collapsed_hour_curves,
        }
        for filename, table in table_specs.items():
            table.to_parquet(tables / filename, index=False, compression="zstd")
        result.fits.to_csv(
            tables / "vac_only_alpha_fits.csv", index=False, lineterminator="\n"
        )
        consistency = _build_consistency_table(result, msd_alphas)
        consistency.to_csv(
            tables / "vac_msd_consistency.csv", index=False, lineterminator="\n"
        )
        result.raw_hour_curves.to_csv(
            tables / "raw_hour_curves.csv", index=False, lineterminator="\n"
        )
        result.scaled_delta_hour_curves.to_csv(
            tables / "scaled_delta_hour_curves.csv", index=False, lineterminator="\n"
        )
        result.collapsed_hour_curves.to_csv(
            tables / "collapsed_hour_curves.csv", index=False, lineterminator="\n"
        )
        support_summary = result.support_census.groupby(
            [
                "hour_post_delivery",
                "site",
                "target_delta_s",
                "unit_included",
                "exclusion_reason",
            ],
            as_index=False,
            dropna=False,
        ).agg(n_units=("unit_id", "nunique"), n_crops=("crop_id", "nunique"))
        support_summary.to_csv(tables / "support_summary.csv", index=False, lineterminator="\n")
        hour_figures = build_oligo_vac_hour_figures(
            result,
            msd_alphas,
            figures,
            target_delta_s=targets,
            maximum_scaled_lag=float(settings["maximum_scaled_lag"]),
            raw_lag_limit_s=float(settings["raw_lag_plot_limit_s"]),
            minimum_hour_bin_units=int(settings["minimum_hour_bin_units_for_display"]),
        )
        alpha_figure = build_oligo_vac_alpha_figure(
            result.fits, figures / "vac_alpha_multi_delta_by_hour"
        )
        consistency_figure = build_vac_msd_alpha_consistency_figure(
            result.fits,
            msd_alphas,
            figures / "vac_msd_alpha_consistency_by_hour",
        )
        contract = {
            "schema_version": 1,
            "metric": "single-locus normalized velocity autocorrelation",
            "equations": {
                "velocity": "v_delta(t)=[r(t+delta)-r(t)]/delta",
                "vac": "C_v_delta(tau)=<v_delta(t+tau) dot v_delta(t)>",
                "normalization": "each trajectory/delta divided by its own C_v_delta(0)",
                "primary_theory_overlay": "Oligo-LiveFISH Eq.9 with alpha fixed to the independently fitted MSD exponent",
                "diagnostic_fit": "Oligo-LiveFISH Eq.9 fit to the joint multi-delta tau/delta collapse",
            },
            "targets_delta_s": list(targets),
            "integer_frame_matching_relative_tolerance": float(
                settings["delta_relative_tolerance"]
            ),
            "maximum_scaled_lag": float(settings["maximum_scaled_lag"]),
            "minimum_velocity_pairs_per_unit_lag": int(settings["minimum_velocity_pairs"]),
            "minimum_hour_bin_units_for_display_only": int(
                settings["minimum_hour_bin_units_for_display"]
            ),
            "alpha_bounds": list(settings["alpha_bounds"]),
            "pooling": "time-average within trajectory; average supported deltas within trajectory; equal trajectory mean within folder-hour",
            "uncertainty": "between-trajectory sample SD in tables; crop-cluster bootstrap 95% CI for alpha",
            "coordinate_interpolation": False,
            "excluded_filters": [
                "paired bundle requirement",
                "T3/T4 QC",
                "motion magnitude",
                "Site1/Site2 separation",
                "53BP1 detection or recruitment",
                "hour or acquisition identity",
                "learned state",
            ],
            "reference_code_interpretation": {
                "adopted": "exp_corr.m multi-delta display and Oligo-LiveFISH VAC Eq.7-9",
                "not_applied_to_vac": "Rouse numerical-table WLS in calc_vel_corr_new/WLS_vel_corr_new; this is VCC communication-time modeling",
            },
            "macro_time_warning": "folder-hour after Cas9 delivery is not observed time since cutting",
        }
        (staging / "METHOD_CONTRACT.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (staging / "IMPLEMENTATION_AUDIT.md").write_text(_audit_text(), encoding="utf-8")
        (staging / "REPORT.md").write_text(
            _report(result, consistency), encoding="utf-8"
        )
        hours = sorted(float(value) for value in result.fits["hour_post_delivery"].unique())
        (staging / "FIGURE_INDEX.md").write_text(_figure_index(hours), encoding="utf-8")

        project_root = Path(__file__).resolve().parents[1]
        code_sources = [
            Path(__file__).resolve(),
            project_root / "src" / "dsb_states" / "oligo_vac.py",
            project_root / "src" / "dsb_states" / "physical_metrics.py",
            project_root / "src" / "dsb_states" / "time_resolved_physics_summary.py",
        ]
        source_manifest = sources + code_sources
        artifacts = [path for path in staging.rglob("*") if path.is_file()]
        manifest = {
            "sources": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in source_manifest
            ],
            "artifacts": [
                {
                    "path": path.relative_to(staging).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(artifacts)
            ],
            "hour_figures": {
                f"{hour:g}": [str(path.relative_to(staging)) for path in paths]
                for hour, paths in hour_figures.items()
            },
            "alpha_figure": [str(path.relative_to(staging)) for path in alpha_figure],
            "consistency_figure": [
                str(path.relative_to(staging)) for path in consistency_figure
            ],
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
                "unit_curve_rows": len(result.unit_curves),
                "fit_rows": len(result.fits),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
