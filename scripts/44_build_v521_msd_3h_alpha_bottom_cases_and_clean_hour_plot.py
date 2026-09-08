#!/usr/bin/env python
"""Render the lowest supported 3 h Site1-alpha cases and a clean hour plot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dsb_states.plots import COLORS, configure_publication_style, save_figure
from dsb_states.v521_case_study_multichannel_review import (
    configure_case_style,
    load_multichannel_review_cases,
    write_multichannel_case_review,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_bottom(alpha_table: pd.DataFrame, contract: dict[str, object]) -> pd.DataFrame:
    hour = float(contract["hour_post_delivery"])
    site = str(contract["site"])
    minimum_r2 = float(contract["minimum_r2"])
    bottom_n = int(contract["bottom_n"])
    eligible = alpha_table.loc[
        alpha_table["site"].astype(str).eq(site)
        & np.isclose(alpha_table["hour_post_delivery"].astype(float), hour)
        & (pd.to_numeric(alpha_table["r2_log"], errors="coerce") >= minimum_r2)
        & np.isfinite(pd.to_numeric(alpha_table["alpha"], errors="coerce"))
    ].copy()
    if len(eligible) < bottom_n:
        raise ValueError(f"Only {len(eligible)} supported cases; cannot select bottom {bottom_n}")
    selected = eligible.sort_values(
        ["alpha", "r2_log", "fit_endpoint_pairs"],
        ascending=[True, False, False],
        kind="stable",
    ).head(bottom_n)
    selected = selected.reset_index(drop=True)
    selected.insert(0, "alpha_rank_from_low", np.arange(1, len(selected) + 1, dtype=int))
    selected.insert(
        1,
        "case_id",
        [f"alpha_bottom{rank:02d}" for rank in selected["alpha_rank_from_low"]],
    )
    return selected


def _build_clean_alpha_figure(fits: pd.DataFrame, output_stem: Path) -> list[Path]:
    selected = fits.loc[
        fits["metric"].astype(str).str.casefold().eq("msd")
        & fits["estimate_name"].astype(str).str.casefold().str.contains("exponent")
    ].copy()
    for column in ("hour_post_delivery", "estimate", "ci_low", "ci_high"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.loc[
        np.isfinite(selected["estimate"])
        & ~np.isclose(selected["hour_post_delivery"].astype(float), 5.0)
    ].copy()

    configure_publication_style()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 17,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 17,
        }
    )
    figure, axis = plt.subplots(figsize=(7.6, 4.9))
    colors = {"site1": COLORS["site1"], "site2": COLORS["site2"]}
    labels = {"site1": "Site1", "site2": "Site2"}
    for site in ("site1", "site2"):
        group = selected.loc[selected["site"].astype(str).str.casefold().eq(site)].sort_values(
            "hour_post_delivery", kind="stable"
        )
        estimate = group["estimate"].to_numpy(float)
        low = group["ci_low"].to_numpy(float)
        high = group["ci_high"].to_numpy(float)
        finite = np.isfinite(low) & np.isfinite(high) & (low <= estimate) & (high >= estimate)
        errors = np.vstack((estimate - low, high - estimate)) if np.all(finite) else None
        axis.errorbar(
            group["hour_post_delivery"],
            estimate,
            yerr=errors,
            color=colors[site],
            marker="o",
            markersize=7.5,
            linewidth=2.2,
            capsize=3.5,
            label=labels[site],
        )
    axis.set_xlabel("Folder-hour after delivery")
    axis.set_ylabel(r"Fitted $\alpha$")
    axis.set_xlim(1.1, 10.4)
    axis.set_ylim(0.28, 1.25)
    axis.set_xticks([2, 4, 6, 8, 10])
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    axis.tick_params(width=1.6, length=5.5)
    figure.tight_layout(pad=0.7)
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-table", type=Path, required=True)
    parser.add_argument("--fits-table", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fullrun-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-exe", type=Path, required=True)
    parser.add_argument("--reference-script-dir", type=Path, required=True)
    parser.add_argument("--original-matlab", type=Path, required=True)
    parser.add_argument(
        "--resume-after-cases",
        action="store_true",
        help="Reuse an already completed case_studies/ manifest after a late-stage failure.",
    )
    args = parser.parse_args()

    alpha_path = args.alpha_table.resolve()
    fits_path = args.fits_table.resolve()
    selection_path = args.selection_csv.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists() and not args.resume_after_cases:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = _select_bottom(pd.read_csv(alpha_path), config["selection"])
    configured = [str(item["bundle_id"]) for item in config["cases"]]
    if configured != selected["bundle_id"].astype(str).tolist():
        raise ValueError("Frozen case configuration does not match recomputed bottom ranking")
    alpha_table = pd.read_csv(alpha_path)
    paired = alpha_table.loc[
        alpha_table["site"].astype(str).eq("site2"),
        ["bundle_id", "alpha", "r2_log", "n_fit_lags", "fit_lag_min_s", "fit_lag_max_s"],
    ].rename(
        columns={
            "alpha": "paired_site2_alpha",
            "r2_log": "paired_site2_r2_log",
            "n_fit_lags": "paired_site2_n_fit_lags",
            "fit_lag_min_s": "paired_site2_fit_lag_min_s",
            "fit_lag_max_s": "paired_site2_fit_lag_max_s",
        }
    )
    ranked = selected.merge(paired, on="bundle_id", how="left", validate="one_to_one")
    ranked["selection_role"] = (
        "ascending Site1 raw-MSD alpha at 3 h in the frozen complete-coverage cohort; "
        "R2 support only"
    )

    cases, review_config = load_multichannel_review_cases(
        config_path=config_path,
        fullrun_root=args.fullrun_root.resolve(),
        selection_csv=selection_path,
    )
    configure_case_style()
    reference_dir = args.reference_script_dir.resolve()
    script_path = Path(__file__).resolve()
    case_module = (
        script_path.parents[1] / "src" / "dsb_states" / "v521_case_study_multichannel_review.py"
    )
    plots_module = script_path.parents[1] / "src" / "dsb_states" / "plots.py"
    if args.resume_after_cases:
        case_manifest_path = output / "case_studies" / "OUTPUT_MANIFEST.json"
        if not case_manifest_path.is_file():
            raise FileNotFoundError(f"Missing completed case manifest: {case_manifest_path}")
        case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
        products = case_manifest.get("products", [])
        if len(products) != len(cases):
            raise ValueError(
                f"Case manifest has {len(products)} products; expected {len(cases)}"
            )
        for product in products:
            for artifact in (product["png"], product["pdf"], product["video"]["path"]):
                artifact_path = Path(artifact)
                if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing or empty case artifact: {artifact_path}")
        case_result = {"products": products}
    else:
        case_result = write_multichannel_case_review(
            cases=cases,
            config=review_config,
            config_path=config_path,
            output_dir=output / "case_studies",
            ffmpeg_exe=args.ffmpeg_exe.resolve(),
            extra_source_paths=(
                alpha_path,
                fits_path,
                selection_path,
                script_path,
                case_module,
                plots_module,
                reference_dir / "README.md",
                reference_dir / "animation_core.py",
                reference_dir / "build_demo_videos.py",
                args.original_matlab.resolve(),
            ),
        )
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(exist_ok=args.resume_after_cases)
    figure_dir.mkdir(exist_ok=args.resume_after_cases)
    ranking_path = table_dir / "3h_site1_alpha_bottom3.csv"
    ranked.to_csv(ranking_path, index=False)
    alpha_figure_paths = _build_clean_alpha_figure(
        pd.read_csv(fits_path), figure_dir / "fig_msd_alpha_by_hour_no5_clean"
    )
    contract = {
        "case_selection": {
            "cohort": "frozen 100%-coverage cohort after excluding 5/10-frame acquisitions",
            "hour_post_delivery": 3.0,
            "ranking_site": "site1",
            "ranking": "ascending raw-MSD alpha",
            "bottom_n": 3,
            "minimum_r2": 0.9,
            "excluded_variables": ["appearance", "distance", "motion amplitude", "53BP1"],
        },
        "alpha_hour_figure": {
            "source": str(fits_path),
            "excluded_display_hour": 5.0,
            "removed_elements": ["title", "lower-right explanatory annotation"],
            "fit_values_or_intervals_recomputed": False,
        },
    }
    contract_path = output / "METHOD_CONTRACT.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# 3 h Site1 alpha bottom-three cases and clean hour plot",
        "",
    ]
    for row in ranked.itertuples(index=False):
        report_lines.append(
            f"- Bottom {row.alpha_rank_from_low}: `{row.bundle_id}`; Site1 alpha={row.alpha:.3f}, "
            f"R2={row.r2_log:.3f}; paired Site2 alpha={row.paired_site2_alpha:.3f}, "
            f"R2={row.paired_site2_r2_log:.3f}."
        )
    report_lines.extend(
        [
            "",
            "The clean alpha-by-hour figure omits 5 h, its title and the in-panel grey note only.",
            "No fit estimate or confidence interval was recomputed.",
        ]
    )
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    sources = [
        alpha_path,
        fits_path,
        selection_path,
        config_path,
        script_path,
        case_module,
        plots_module,
        reference_dir / "README.md",
        reference_dir / "animation_core.py",
        reference_dir / "build_demo_videos.py",
        args.original_matlab.resolve(),
    ]
    artifacts = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "OUTPUT_MANIFEST.json"
    )
    manifest_path = output / "OUTPUT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                    for path in sources
                ],
                "artifacts": [
                    {
                        "path": path.relative_to(output).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in artifacts
                ],
                "case_products": case_result["products"],
                "alpha_figure_paths": [str(path) for path in alpha_figure_paths],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "bottom_cases": len(ranked),
                "ranking_table": str(ranking_path),
                "alpha_figure_paths": [str(path) for path in alpha_figure_paths],
                "case_products": case_result["products"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
