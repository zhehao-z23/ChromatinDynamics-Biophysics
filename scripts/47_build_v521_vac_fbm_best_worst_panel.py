#!/usr/bin/env python
"""Compare the frozen best- and worst-fitting folder-hours to fBM Equation 9."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from dsb_states.time_resolved_physics_summary import fbm_normalized_vac

DELTA_COLORS = {10.0: "#0072BD", 20.0: "#D95319", 40.0: "#EDB120"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.linewidth": 2.0,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.fontsize": 15,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _matlab_axis(axis: mpl.axes.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.0)
    axis.tick_params(direction="out", top=True, right=True, width=1.8, length=6, pad=5)
    axis.axhline(0.0, color="#8C8C8C", lw=0.9, zorder=0)


def _select_hours(consistency: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    required = {
        "hour_post_delivery",
        "site",
        "alpha_msd",
        "rmse_at_msd_alpha",
    }
    missing = required.difference(consistency.columns)
    if missing:
        raise ValueError(f"Consistency table missing columns: {sorted(missing)}")
    eligible = consistency.loc[
        consistency["site"].isin(["site1", "site2"])
        & ~np.isclose(consistency["hour_post_delivery"], 5.0)
    ].copy()
    summary = (
        eligible.groupby("hour_post_delivery", as_index=False)
        .agg(
            mean_rmse_at_msd_alpha=("rmse_at_msd_alpha", "mean"),
            n_sites=("site", "nunique"),
        )
        .sort_values("mean_rmse_at_msd_alpha", kind="stable")
    )
    if not summary["n_sites"].eq(2).all():
        raise ValueError("Every ranked hour must contain both Site1 and Site2")
    best = float(summary.iloc[0]["hour_post_delivery"])
    worst = float(summary.iloc[-1]["hour_post_delivery"])
    return best, worst, summary


def _plot_panel(
    dense: pd.DataFrame,
    consistency: pd.DataFrame,
    *,
    best_hour: float,
    worst_hour: float,
    output_stem: Path,
) -> list[Path]:
    _configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.35),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    hours = (best_hour, worst_hour)
    grid = np.linspace(0.0, 2.5, 501)
    site = "site1"
    for column, hour in enumerate(hours):
            axis = axes[0, column]
            points = dense.loc[
                dense["site"].eq(site) & np.isclose(dense["hour_post_delivery"], hour)
            ]
            parameter = consistency.loc[
                consistency["site"].eq(site)
                & np.isclose(consistency["hour_post_delivery"], hour)
            ]
            if len(parameter) != 1:
                raise ValueError(f"Expected one consistency row for {site}, {hour:g} h")
            parameter = parameter.iloc[0]
            alpha = float(parameter["alpha_msd"])
            rmse = float(parameter["rmse_at_msd_alpha"])
            axis.plot(
                grid,
                fbm_normalized_vac(grid, alpha),
                color="black",
                lw=2.6,
                zorder=3,
            )
            for delta, color in DELTA_COLORS.items():
                curve = points.loc[np.isclose(points["target_delta_s"], delta)]
                axis.scatter(
                    curve["x_mean"],
                    curve["mean"],
                    s=31,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=5,
                )
            axis.text(
                0.96,
                0.94,
                f"RMSE={rmse:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=15,
            )
            letter = "A" if column == 0 else "B"
            axis.text(
                -0.12,
                1.08,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=20,
                fontweight="bold",
                clip_on=False,
            )
            axis.text(
                0.04,
                0.94,
                f"{hour:g} h",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=17,
                fontweight="bold",
            )
            axis.set_xlim(-0.03, 2.53)
            axis.set_ylim(-0.5, 1.05)
            axis.set_xticks(np.arange(0.0, 2.51, 0.5))
            axis.set_yticks([-0.5, 0.0, 0.5, 1.0])
            _matlab_axis(axis)
    for axis in axes[0, :]:
        axis.set_xlabel(r"Rescaled lag time $\tau/\delta$")
    axes[0, 0].set_ylabel("Normalized VAC")
    handles = [
        Line2D([0], [0], color="black", lw=2.6, label="fBM Eq. 9"),
        *[
            Line2D(
                [0],
                [0],
                marker="o",
                ms=7,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                color="none",
                label=rf"$\delta$={delta:g} s",
            )
            for delta, color in DELTA_COLORS.items()
        ],
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.80, wspace=0.16)
    outputs = [Path(f"{output_stem}{suffix}") for suffix in (".png", ".pdf", ".svg")]
    figure.savefig(outputs[0], dpi=300, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    figure.savefig(outputs[2], bbox_inches="tight")
    plt.close(figure)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-vac-table", type=Path, required=True)
    parser.add_argument("--consistency-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dense_path = args.dense_vac_table.resolve()
    consistency_path = args.consistency_table.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    dense = pd.read_csv(dense_path)
    consistency = pd.read_csv(consistency_path)
    best, worst, ranking = _select_hours(consistency)
    if not (np.isclose(best, 4.0) and np.isclose(worst, 2.0)):
        raise ValueError(f"Unexpected frozen best/worst hours: best={best}, worst={worst}")

    figure_dir = output / "figures"
    table_dir = output / "tables"
    figure_dir.mkdir(parents=True)
    table_dir.mkdir()
    figure_paths = _plot_panel(
        dense,
        consistency,
        best_hour=best,
        worst_hour=worst,
        output_stem=figure_dir / "fig_vac_fbm_best_4h_vs_worst_2h",
    )
    ranking_path = table_dir / "vac_fbm_hour_fit_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    selected_rows_path = table_dir / "vac_fbm_best_worst_site_parameters.csv"
    consistency.loc[
        consistency["hour_post_delivery"].isin([best, worst])
    ].sort_values(["hour_post_delivery", "site"], kind="stable").to_csv(
        selected_rows_path, index=False
    )

    contract = {
        "fit_quality_metric": "RMSE of normalized VAC against fBM Equation 9 fixed by independent MSD alpha",
        "hour_selection": "lowest/highest equal-weight mean RMSE across Site1 and Site2",
        "excluded_hour": 5.0,
        "best_hour": best,
        "worst_hour": worst,
        "alpha_refitted_to_vac": False,
        "displayed_site": "Site1 only",
        "panel_titles": "panel letters only; hour is an in-panel label",
        "alpha_annotation_displayed": False,
        "display_points": (
            "frozen equal-trajectory means in 0.05-wide rescaled-lag bins; at most one real "
            "observation per trajectory/delta/bin; minimum eight trajectories; no interpolation"
        ),
        "interpretation": "model-consistency comparison, not proof or disproof of a motion mechanism",
    }
    contract_path = output / "METHOD_CONTRACT.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report_path = output / "REPORT.md"
    report_path.write_text(
        "# VAC versus fixed-fBM best/worst folder-hours\n\n"
        "Using the equal-weight mean of the frozen Site1 and Site2 RMSE values, 4 h is the best "
        "fitting folder-hour (mean RMSE 0.0757) and 2 h is the worst (mean RMSE 0.1504). "
        "The display contains Site1 only. Equation 9 is fixed by the independently fitted MSD "
        "alpha; alpha is not refitted to VAC or annotated inside the panels.\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()
    theory_module = script_path.parents[1] / "src" / "dsb_states" / "time_resolved_physics_summary.py"
    sources = [dense_path, consistency_path, script_path, theory_module]
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
                "figure_paths": [str(path) for path in figure_paths],
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
                "best_hour": best,
                "worst_hour": worst,
                "figure_paths": [str(path) for path in figure_paths],
                "ranking_table": str(ranking_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
