#!/usr/bin/env python
"""Build the metric-support-only, hour-resolved v5.2.1 physics analysis."""

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
from scipy.stats import spearmanr

from dsb_states.time_resolved_physics import TimeResolvedPhysicsUnits, compute_time_resolved_physics
from dsb_states.time_resolved_physics_plots import write_time_resolved_physics_figures
from dsb_states.time_resolved_physics_summary import (
    build_time_resolved_physics_summary,
    build_vcc_matrix_hour_summary,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trend_table(fits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if fits.empty:
        return pd.DataFrame()
    for (metric, site, name), group in fits.groupby(
        ["metric", "site", "estimate_name"], sort=True
    ):
        hour = pd.to_numeric(group["hour_post_delivery"], errors="coerce").to_numpy(float)
        value = pd.to_numeric(group["estimate"], errors="coerce").to_numpy(float)
        valid = np.isfinite(hour) & np.isfinite(value)
        if np.count_nonzero(valid) < 3:
            continue
        rho = float(spearmanr(hour[valid], value[valid]).statistic)
        rows.append(
            {
                "metric": metric,
                "site": site,
                "estimate_name": name,
                "n_hours": int(np.count_nonzero(valid)),
                "first_hour": float(np.min(hour[valid])),
                "last_hour": float(np.max(hour[valid])),
                "first_estimate": float(value[valid][np.argmin(hour[valid])]),
                "last_estimate": float(value[valid][np.argmax(hour[valid])]),
                "spearman_rho_hour": rho,
                "inference_status": "descriptive_only_no_biological_replicates",
            }
        )
    return pd.DataFrame(rows)


def _formula_notes() -> str:
    return """# Formula and method contract

## Frozen pooling assumption

The user confirmed on 2026-08-23 that all acquisitions belong to one experimental batch.
Acquisition identity is retained only for provenance, cadence and support diagnostics; no batch
effect or acquisition-equal biological model is fitted. Folder-hour is time after Cas9 delivery,
not verified time since cutting.

## Metric-specific support rule

- Coordinates must be finite at the required observed endpoints on the exact one-based acquisition
  schedule; coordinates are never interpolated and missing frames are never compressed.
- Each trajectory/bundle and lag requires at least eight endpoint or velocity pairs.
- Maximum lag is `min(50 frames, floor(0.25 * movie_frames))`.
- No T3/T4, motion, separation, 53BP1, hour, acquisition or learned-state filter is used.
- Site1/Site2 MSD and VAC enter independently. MSCD/VCC require only their own paired support.
- A trajectory is first time-averaged. Trajectories (or paired bundles) are then equal-weighted
  within folder-hour. Display bands are between-unit sample SD, not SEM.

## MSD

`MSD(tau) = <|r(t+tau)-r(t)|^2>` in µm². Raw MSD is shown. The common 10–50 s curve is fitted
descriptively as `A*tau^alpha`. Oligo-LiveFISH Eq. 5 also models exposure-time motion blur and
localization error sigma, but exposure/localization contracts are unavailable here; therefore no
`MSD-4 sigma²`, effective diffusion coefficient or error-corrected claim is made.

## MSCD

`DeltaR(t) = r_site2(t)-r_site1(t)` and
`MSCD(tau) = <|DeltaR(t+tau)-DeltaR(t)|^2>` in µm². This is relative-vector change, not change in
scalar separation. The common 10–50 s curve is fitted descriptively as `A*tau^beta`.

## VAC

`v_delta(t) = [r(t+delta)-r(t)]/delta` and
`C_v_delta(tau) = <v_delta(t+tau) dot v_delta(t)>`. Raw VAC is µm²/s²; displayed VAC is normalized
by its zero-lag velocity energy and is dimensionless. Primary cross-hour comparison uses the
integer frame offset nearest 10 s within 7.5–12.5 s for each acquisition. The fitted shape is
Oligo-LiveFISH Eq. 9:

`C(tau)/C(0) = (|tau-delta|^alpha + |tau+delta|^alpha - 2|tau|^alpha)/(2*delta^alpha)`.

The additional delta=1/2/4/8-frame families remain in the unit tables for later collapse checks.

## VCC

The primary object is the normalized lag-indexed 2×2 matrix
`C12_delta(tau)=<v1_delta(t+tau) v2_delta(t)^T>`, with raw units µm²/s². The denominator is the
fixed symmetric zero-lag energy scale `sqrt(<|v1|²><|v2|²>)`. Both lead directions are stored.
The headline curve is the mean of their rotation-invariant traces; it does not replace the matrix.
Only a nonmechanistic smoothing spline is drawn. A viscoelastic-Rouse communication-time fit is
not attempted because the authoritative theoretical discretization and fit mask are not yet fully
implemented and validated.

## References inspected

- Zhu et al., *Cell* (2025), Oligo-LiveFISH:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12167157/
- Official Data S1:
  https://ars.els-cdn.com/content/image/1-s2.0-S0092867425003502-mmc3.pdf
- Local MATLAB reference (read-only):
  `G:/共享云端硬盘/LivevFISH data/All_codes/Single Particle Tracking/diffusion distribution/spt_dispmoments.m`

The local MATLAB code supports the order “trajectory time average, then equal trajectory average”,
but its fixed-cadence Brownian linear fit and SEM display are not copied into this analysis.
"""


def _fit_markdown(fits: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if fits.empty:
        return ["No supported model fits were produced."]
    chosen = fits.loc[
        fits["estimate_name"].isin(
            ["exponent", "value_at_10s", "zero_lag_trace", "positive_auc_0_5delta"]
        )
    ].copy()
    for (metric, site, name), group in chosen.groupby(
        ["metric", "site", "estimate_name"], sort=True
    ):
        values = ", ".join(
            f"{row.hour_post_delivery:g}h={row.estimate:.3g}"
            for row in group.sort_values("hour_post_delivery").itertuples()
        )
        lines.append(f"- `{metric}/{site}/{name}`: {values}")
    return lines or ["No selected headline fit parameters were available."]


def _report(
    *,
    units: Any,
    summary: Any,
    matrix_summary: pd.DataFrame,
    trends: pd.DataFrame,
) -> str:
    primary = units.unit_curves.loc[
        units.unit_curves["metric"].isin(["msd", "mscd"])
        | units.unit_curves["is_primary_matched_10s"].astype(bool)
    ]
    lines = [
        "# v5.2.1 time-resolved physical curves",
        "",
        "## What was calculated",
        "",
        f"- {len(units.unit_curves):,} supported per-unit scalar curve points.",
        f"- {len(units.vcc_matrices):,} supported directional VCC matrix-component rows.",
        f"- {len(units.support_census):,} metric/delta support-census rows, including failures.",
        f"- {primary['unit_id'].nunique():,} units contribute at least one primary curve point.",
        f"- {len(summary.hour_curves):,} equal-unit hour/lag summary rows.",
        f"- {len(matrix_summary):,} symmetric hour/lag VCC matrix-component summary rows.",
        "",
        "## Preliminary descriptive fit parameters",
        "",
        *_fit_markdown(summary.fits),
        "",
        "## Interpretation limits",
        "",
        "- All acquisitions are pooled as one user-confirmed batch, but exact cadence and movie length remain unequal.",
        "- The common 10–50 s fit window spans only five-fold; exponents are weakly identified.",
        "- 5 h has very sparse Site1/pair support and remains preliminary.",
        "- These are folder-hours after Cas9 delivery, not synchronized times since an observed cut.",
        "- Raw MSD is not localization- or motion-blur-corrected; no `MSD-4σ²` claim is made.",
        "- VCC Rouse communication time is gated; the empirical 2×2 matrix is complete and retained.",
        "- No biological-replicate population inference or causal claim is made.",
        "",
        "## Temporal trend table",
        "",
        f"`tables/fit_parameter_time_trends.csv` contains {len(trends)} descriptive Spearman summaries; p-values are intentionally omitted.",
        "",
    ]
    return "\n".join(lines)


def _figure_index() -> str:
    return """# Figure index

- `fig_msd_curves_by_hour`: Does mean Site1/Site2 displacement grow differently with lag and folder-hour? Mean ± between-trajectory SD; dashed 10–50 s descriptive fit.
- `fig_msd_alpha_by_hour`: How does the weakly identified raw-MSD power-law exponent vary across folder-hour?
- `fig_mscd_curves_by_hour`: Does Site1/Site2 relative-vector motion change with lag and folder-hour? Mean ± between-bundle SD.
- `fig_mscd_beta_by_hour`: How does the descriptive MSCD exponent vary across folder-hour?
- `vac_site1_matched_10s`, `vac_site2_matched_10s`: Does normalized velocity memory at a comparable delta≈10 s change across folder-hour, and is its shape fBM-like?
- `fig_vac_alpha_by_hour`: How does the Eq. 9 shape parameter vary across folder-hour?
- `fig_vcc_curves_by_hour`: How does the bidirectional, rotation-invariant VCC trace evolve with tau/delta? Dashed line is a nonmechanistic guide.
- `fig_vcc_descriptive_endpoints_by_hour`: Compact descriptive VCC summaries; not a Rouse relaxation fit.

All full directional 2×2 VCC matrices remain in source tables even when the figure uses their trace.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-unit-result",
        type=Path,
        help="Reuse immutable per-unit tables from a validated prior result and rebuild summaries/figures.",
    )
    args = parser.parse_args()
    cache = args.cache.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite result: {output}")
    frames_path = cache / "tables" / "bundle_frame_master.parquet"
    index_path = cache / "tables" / "bundle_index.parquet"
    contract_path = cache / "CACHE_CONTRACT.json"
    for source in (frames_path, index_path, contract_path, config_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["time_resolved_physics"]
    if not settings.get("same_experimental_batch"):
        raise ValueError("same_experimental_batch must be explicitly frozen true")

    reused_sources: list[Path] = []
    if args.reuse_unit_result is not None:
        reuse = args.reuse_unit_result.resolve()
        unit_path = reuse / "tables" / "unit_curves.parquet"
        matrix_path = reuse / "tables" / "vcc_unit_matrices.parquet"
        census_path = reuse / "tables" / "metric_support_census.parquet"
        reused_sources = [unit_path, matrix_path, census_path]
        for source in reused_sources:
            if not source.is_file():
                raise FileNotFoundError(source)
        units = TimeResolvedPhysicsUnits(
            unit_curves=pd.read_parquet(unit_path),
            vcc_matrices=pd.read_parquet(matrix_path),
            support_census=pd.read_parquet(census_path),
        )
    else:
        units = compute_time_resolved_physics(
            pd.read_parquet(frames_path),
            pd.read_parquet(index_path),
            max_lag_cap=int(settings["max_lag_frames_cap"]),
            max_lag_fraction=float(settings["max_lag_nominal_fraction"]),
            paper_delta_frames=tuple(
                int(value) for value in settings["oligo_livefish_delta_frames"]
            ),
            min_pairs=int(settings["min_pairs_per_unit_lag"]),
            primary_delta_target_s=float(settings["primary_velocity_delta_target_s"]),
            primary_delta_acceptable_s=tuple(
                float(value) for value in settings["primary_velocity_delta_allowed_s"]
            ),
        )
    centers = tuple(float(value) for value in settings["physical_lag_bin_centers_s"])
    summary = build_time_resolved_physics_summary(
        units.unit_curves,
        lag_bin_centers_s=centers,
        common_fit_window_s=tuple(
            float(value) for value in settings["common_descriptive_fit_window_s"]
        ),
        minimum_fit_points=int(settings["minimum_fit_lag_points"]),
        minimum_fit_span_fold=float(settings["minimum_fit_time_span_fold"]),
        bootstrap_iterations=int(settings["bootstrap_iterations"]),
        bootstrap_seed=int(settings["bootstrap_seed"]),
    )
    matrix_summary = build_vcc_matrix_hour_summary(
        units.vcc_matrices,
        lag_bin_centers_s=centers,
    )
    trends = _trend_table(summary.fits)

    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        tables = staging / "tables"
        figures = staging / "figures"
        tables.mkdir()
        figures.mkdir()
        table_specs = (
            ("unit_curves.parquet", units.unit_curves),
            ("vcc_unit_matrices.parquet", units.vcc_matrices),
            ("metric_support_census.parquet", units.support_census),
            ("primary_unit_lag_bins.parquet", summary.unit_bins),
            ("hour_curves.parquet", summary.hour_curves),
            ("vcc_matrix_hour_summary.parquet", matrix_summary),
            ("predicted_curves.parquet", summary.predicted_curves),
        )
        for filename, table in table_specs:
            table.to_parquet(tables / filename, index=False, compression="zstd")
        summary.hour_curves.to_csv(tables / "hour_curves.csv", index=False, lineterminator="\n")
        summary.fits.to_csv(tables / "fits.csv", index=False, lineterminator="\n")
        summary.support_by_hour.to_csv(
            tables / "support_by_hour.csv", index=False, lineterminator="\n"
        )
        trends.to_csv(tables / "fit_parameter_time_trends.csv", index=False, lineterminator="\n")
        units.support_census.groupby(
            ["metric", "site", "hour_post_delivery", "unit_included", "exclusion_reason"],
            dropna=False,
        ).size().rename("n_units").reset_index().to_csv(
            tables / "support_census_by_hour.csv", index=False, lineterminator="\n"
        )
        (staging / "METHOD_CONTRACT.json").write_text(
            json.dumps(summary.method_contract, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging / "FORMULAS_AND_METHODS.md").write_text(_formula_notes(), encoding="utf-8")
        (staging / "REPORT.md").write_text(
            _report(units=units, summary=summary, matrix_summary=matrix_summary, trends=trends),
            encoding="utf-8",
        )
        (staging / "FIGURE_INDEX.md").write_text(_figure_index(), encoding="utf-8")
        write_time_resolved_physics_figures(
            hour_curves=summary.hour_curves,
            fits=summary.fits,
            output_dir=figures,
            predicted_curves=summary.predicted_curves,
        )
        project_root = Path(__file__).resolve().parents[1]
        code_sources = [
            Path(__file__).resolve(),
            project_root / "src" / "dsb_states" / "physical_metrics.py",
            project_root / "src" / "dsb_states" / "time_resolved_physics.py",
            project_root / "src" / "dsb_states" / "time_resolved_physics_summary.py",
            project_root / "src" / "dsb_states" / "time_resolved_physics_plots.py",
        ]
        sources = [
            frames_path,
            index_path,
            contract_path,
            config_path,
            *code_sources,
            *reused_sources,
        ]
        artifacts = [path for path in staging.rglob("*") if path.is_file()]
        (staging / "OUTPUT_MANIFEST.json").write_text(
            json.dumps(
                {
                    "sources": [
                        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                        for path in sources
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
                "unit_curve_rows": len(units.unit_curves),
                "vcc_matrix_rows": len(units.vcc_matrices),
                "hour_curve_rows": len(summary.hour_curves),
                "fit_rows": len(summary.fits),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
