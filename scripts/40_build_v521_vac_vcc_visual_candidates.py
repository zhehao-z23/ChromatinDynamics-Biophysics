"""Build MATLAB-style VAC/VCC displays from frozen, validated estimates.

This script changes presentation only. Experimental observations are rebuilt
on a fine display grid from archived per-unit curves so the plots use all
supported lag samples. The formal 9-bin fits, fitted alpha values, Rouse
communication times, cohorts, and correlation definitions are never refit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

from dsb_states.time_resolved_physics_summary import fbm_normalized_vac


DELTA_COLORS = {10.0: "#0072BD", 20.0: "#D95319", 40.0: "#EDB120"}
SELECTED_HOURS = (1.5, 3.0, 4.5, 10.0)
DISPLAY_BIN_WIDTH = 0.05
MINIMUM_DISPLAY_UNITS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
            "legend.fontsize": 16,
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
    axis.tick_params(
        direction="out", top=True, right=True, width=1.8, length=6, pad=5
    )
    axis.axhline(0.0, color="#8C8C8C", lw=0.9, zorder=0)


def _save(figure: mpl.figure.Figure, stem: Path) -> list[Path]:
    outputs = [Path(f"{stem}{suffix}") for suffix in (".png", ".pdf", ".svg")]
    figure.savefig(outputs[0], dpi=300, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    figure.savefig(outputs[2], bbox_inches="tight")
    plt.close(figure)
    return outputs


def _delta_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "measurement_delta",
        [
            (0.0, DELTA_COLORS[10.0]),
            ((20.0 - 10.0) / 30.0, DELTA_COLORS[20.0]),
            (1.0, DELTA_COLORS[40.0]),
        ],
    )


def _add_delta_colorbar(
    figure: mpl.figure.Figure,
    axes: np.ndarray,
    *,
    rectangle: tuple[float, float, float, float] = (0.915, 0.17, 0.018, 0.66),
) -> None:
    scalar = mpl.cm.ScalarMappable(norm=Normalize(10.0, 40.0), cmap=_delta_cmap())
    del axes  # Kept in the call signature to make the figure-level intent explicit.
    colorbar_axis = figure.add_axes(rectangle)
    colorbar = figure.colorbar(scalar, cax=colorbar_axis, ticks=[10.0, 20.0, 40.0])
    colorbar.set_label(r"$\delta$ (s)", fontsize=17)
    colorbar.ax.tick_params(labelsize=16, width=1.5, length=5)
    colorbar.outline.set_linewidth(1.6)


def _display_legend(*, include_cloud: bool, model_name: str) -> list[Line2D]:
    handles = [
        Line2D([0], [0], color="black", lw=2.5, label=model_name),
        Line2D(
            [0],
            [0],
            marker="o",
            ms=7,
            markerfacecolor="#666666",
            markeredgecolor="black",
            markeredgewidth=0.5,
            color="none",
            label="equal-unit mean",
        ),
    ]
    if include_cloud:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                ms=5,
                markerfacecolor="#999999",
                markeredgecolor="none",
                color="none",
                alpha=0.28,
                label="unit estimate",
            )
        )
    return handles


def _fine_equal_unit_curve(
    source: pd.DataFrame,
    *,
    value_column: str,
    support_column: str,
    include_site: bool,
    width: float = DISPLAY_BIN_WIDTH,
    minimum_units: int = MINIMUM_DISPLAY_UNITS,
) -> pd.DataFrame:
    """Create a dense display summary without interpolating observations.

    Each archived observation is assigned to its nearest fine display bin. At
    most one real lag observation per unit/delta/bin is retained, then units are
    averaged equally. Empty bins stay empty and are never drawn.
    """

    data = source.loc[
        np.isfinite(pd.to_numeric(source["scaled_lag"], errors="coerce"))
        & np.isfinite(pd.to_numeric(source[value_column], errors="coerce"))
        & source["scaled_lag"].between(0.0, 2.5)
    ].copy()
    data["display_lag_bin"] = np.round(data["scaled_lag"] / width) * width
    data["display_lag_bin"] = data["display_lag_bin"].round(6)
    data["display_bin_distance"] = np.abs(
        data["scaled_lag"] - data["display_lag_bin"]
    )
    keys = ["hour_post_delivery"]
    if include_site:
        keys.append("site")
    keys.extend(["unit_id", "target_delta_s", "display_lag_bin"])
    selected = (
        data.sort_values(
            keys + ["display_bin_distance", support_column],
            ascending=[True] * len(keys) + [True, False],
            kind="stable",
        )
        .drop_duplicates(keys, keep="first")
        .drop(columns="display_bin_distance")
    )
    group = ["hour_post_delivery"]
    if include_site:
        group.append("site")
    group.extend(["target_delta_s", "display_lag_bin"])
    summary = (
        selected.groupby(group, as_index=False, dropna=False)
        .agg(
            x_mean=("scaled_lag", "mean"),
            mean=(value_column, "mean"),
            sd=(value_column, "std"),
            median=(value_column, "median"),
            n_units=("unit_id", "nunique"),
            n_crops=("crop_id", "nunique"),
            total_pairs=(support_column, "sum"),
        )
        .sort_values(group, kind="stable")
        .reset_index(drop=True)
    )
    return summary.loc[summary["n_units"].ge(minimum_units)].reset_index(drop=True)


def _smooth_rouse(curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = curve.sort_values("scaled_lag_bin", kind="stable")
    x = ordered["scaled_lag_bin"].to_numpy(float)
    y = ordered["predicted_vcc"].to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return x, y
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 401)
    return grid, PchipInterpolator(x, y)(grid)


def _plot_unit_cloud(
    axis: mpl.axes.Axes,
    data: pd.DataFrame,
    *,
    value_column: str,
    seed: int,
) -> None:
    for index, (target, color) in enumerate(DELTA_COLORS.items()):
        cloud = data.loc[np.isclose(data["target_delta_s"], target)]
        if cloud.empty:
            continue
        rng = np.random.default_rng(seed + index)
        jitter = rng.uniform(-0.010, 0.010, len(cloud))
        axis.scatter(
            cloud["scaled_lag"].to_numpy(float) + jitter,
            cloud[value_column],
            s=8,
            color=color,
            edgecolor="none",
            alpha=0.11,
            rasterized=True,
            zorder=1,
        )


def _plot_dense_means(axis: mpl.axes.Axes, data: pd.DataFrame) -> None:
    for target, color in DELTA_COLORS.items():
        curve = data.loc[np.isclose(data["target_delta_s"], target)]
        axis.scatter(
            curve["x_mean"],
            curve["mean"],
            s=31,
            facecolor=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=5,
        )


def _plot_vcc_grid(
    dense: pd.DataFrame,
    raw: pd.DataFrame,
    predictions: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    hours: tuple[float, ...],
    shape: tuple[int, int],
    include_cloud: bool,
    stem: Path,
) -> list[Path]:
    figure, axes = plt.subplots(
        *shape,
        figsize=(5.2 * shape[1], 4.25 * shape[0]),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for panel_index, (axis, hour) in enumerate(zip(axes.ravel(), hours, strict=True)):
        raw_panel = raw.loc[
            np.isclose(raw["hour_post_delivery"], hour)
            & raw["scaled_lag"].between(0.0, 2.5)
        ]
        dense_panel = dense.loc[np.isclose(dense["hour_post_delivery"], hour)]
        prediction_panel = predictions.loc[
            np.isclose(predictions["hour_post_delivery"], hour)
        ]
        if include_cloud:
            _plot_unit_cloud(
                axis,
                raw_panel,
                value_column="normalized_vcc",
                seed=20260824 + panel_index * 10,
            )
        for target, color in DELTA_COLORS.items():
            model = prediction_panel.loc[
                np.isclose(prediction_panel["target_delta_s"], target)
            ]
            model_x, model_y = _smooth_rouse(model)
            axis.plot(model_x, model_y, color=color, lw=2.6, zorder=3)
        _plot_dense_means(axis, dense_panel)
        fit = fits.loc[np.isclose(fits["hour_post_delivery"], hour)].iloc[0]
        axis.text(
            0.96,
            0.94,
            f"$R^2$={fit['r_squared']:.2f}\n$n$={int(fit['n_units'])}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=16,
        )
        axis.set_title(f"{hour:g} h", loc="left", pad=6)
        axis.set_xlim(-0.03, 2.53)
        axis.set_ylim(-0.5, 1.0)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
        axis.set_yticks([-0.5, 0.0, 0.5, 1.0])
        _matlab_axis(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"Rescaled lag time $\tau/\delta$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Normalized VCC")
    figure.legend(
        handles=_display_legend(include_cloud=include_cloud, model_name="Rouse fit"),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.47, 1.005),
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.885,
        bottom=0.10,
        top=0.92,
        wspace=0.18,
        hspace=0.24,
    )
    _add_delta_colorbar(figure, axes)
    return _save(figure, stem)


def _plot_vac_selected(
    dense: pd.DataFrame,
    raw: pd.DataFrame,
    msd_alphas: pd.DataFrame,
    *,
    include_cloud: bool,
    stem: Path,
) -> list[Path]:
    figure, axes = plt.subplots(
        2,
        len(SELECTED_HOURS),
        figsize=(17.2, 8.1),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    grid = np.linspace(0.0, 2.5, 501)
    for row, site in enumerate(("site1", "site2")):
        for column, hour in enumerate(SELECTED_HOURS):
            axis = axes[row, column]
            raw_panel = raw.loc[
                raw["site"].eq(site)
                & np.isclose(raw["hour_post_delivery"], hour)
                & raw["scaled_lag"].between(0.0, 2.5)
            ]
            dense_panel = dense.loc[
                dense["site"].eq(site)
                & np.isclose(dense["hour_post_delivery"], hour)
            ]
            alpha_row = msd_alphas.loc[
                msd_alphas["site"].eq(site)
                & np.isclose(msd_alphas["hour_post_delivery"], hour)
            ].iloc[0]
            alpha = float(alpha_row["alpha_msd"])
            if include_cloud:
                _plot_unit_cloud(
                    axis,
                    raw_panel,
                    value_column="normalized_vac",
                    seed=20260924 + row * 100 + column * 10,
                )
            axis.plot(
                grid,
                fbm_normalized_vac(grid, alpha),
                color="black",
                lw=2.6,
                zorder=3,
            )
            _plot_dense_means(axis, dense_panel)
            axis.text(
                0.96,
                0.94,
                rf"$\alpha_{{MSD}}$={alpha:.2f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=16,
            )
            if row == 0:
                axis.set_title(f"{hour:g} h", pad=6)
            axis.set_xlim(-0.03, 2.53)
            axis.set_ylim(-0.5, 1.05)
            axis.set_xticks(np.arange(0.0, 2.51, 0.5))
            axis.set_yticks([-0.5, 0.0, 0.5, 1.0])
            _matlab_axis(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"Rescaled lag time $\tau/\delta$")
    axes[0, 0].set_ylabel("Site 1\nNormalized VAC")
    axes[1, 0].set_ylabel("Site 2\nNormalized VAC")
    figure.legend(
        handles=_display_legend(
            include_cloud=include_cloud, model_name="fixed fBM reference"
        ),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.47, 1.005),
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.115,
        top=0.91,
        wspace=0.16,
        hspace=0.22,
    )
    _add_delta_colorbar(
        figure,
        axes,
        rectangle=(0.905, 0.16, 0.014, 0.69),
    )
    return _save(figure, stem)


def _load_msd_alphas(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data.loc[
        data["metric"].eq("msd")
        & data["estimate_name"].eq("exponent")
        & data["site"].isin(["site1", "site2"])
    ].rename(
        columns={
            "estimate": "alpha_msd",
            "ci_low": "alpha_low",
            "ci_high": "alpha_high",
        }
    )


def _figure_index() -> str:
    return """# MATLAB-style VAC/VCC visualization choices

- **A — VCC selected hours, clean:** recommended PPT version. Fine-grid equal-bundle means are points; colored continuous curves are the archived Rouse fit.
- **B — VCC selected hours, cloud:** the same figure with all supported bundle estimates shown faintly, exposing heterogeneity.
- **C — VCC all hours, clean:** complete 3 x 3 temporal audit using the denser display grid.
- **D — VAC selected hours, clean:** Site 1 and Site 2 in matched rows; points use all supported trajectory lags and the black curve is Equation 9 with independent MSD alpha.
- **E — VAC selected hours, cloud:** the same VAC comparison with all supported trajectory estimates behind the equal-trajectory means.

The fine display grid is not interpolation. Each point is an equal-unit mean of archived real lag observations assigned to a 0.05-wide rescaled-lag bin, with at most one observation per unit/delta/bin and at least eight supporting units. Empty bins remain empty. Formal fits and biological conclusions remain those of the frozen VAC/VCC result directories.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vac-result", type=Path, required=True)
    parser.add_argument("--vcc-result", type=Path, required=True)
    parser.add_argument("--msd-fits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    vac_result = args.vac_result.resolve()
    vcc_result = args.vcc_result.resolve()
    msd_fits_path = args.msd_fits.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite result: {output}")

    sources = {
        "vac_unit_curves": vac_result / "tables" / "unit_vac_curves.parquet",
        "vcc_unit_curves": vcc_result / "tables" / "symmetric_unit_curves.parquet",
        "vcc_predictions": vcc_result / "tables" / "fit_predictions.csv",
        "vcc_fits": vcc_result / "tables" / "communication_time_fits.csv",
        "msd_fits": msd_fits_path,
        "script": Path(__file__).resolve(),
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    staging = output.with_name(f"{output.name}.staging")
    if staging.exists():
        raise FileExistsError(f"Staging path exists: {staging}")
    figure_dir = staging / "figures"
    table_dir = staging / "tables"
    figure_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)
    _configure_style()

    try:
        vac_raw = pd.read_parquet(sources["vac_unit_curves"])
        vcc_raw = pd.read_parquet(sources["vcc_unit_curves"])
        predictions = pd.read_csv(sources["vcc_predictions"])
        fits = pd.read_csv(sources["vcc_fits"])
        msd_alphas = _load_msd_alphas(sources["msd_fits"])

        vac_dense = _fine_equal_unit_curve(
            vac_raw,
            value_column="normalized_vac",
            support_column="pair_count",
            include_site=True,
        )
        vcc_dense = _fine_equal_unit_curve(
            vcc_raw,
            value_column="normalized_vcc",
            support_column="pair_count_min",
            include_site=False,
        )
        vac_dense.to_csv(table_dir / "vac_fine_display_means.csv", index=False)
        vcc_dense.to_csv(table_dir / "vcc_fine_display_means.csv", index=False)
        msd_alphas.to_csv(table_dir / "msd_alpha_reference.csv", index=False)

        all_hours = tuple(
            sorted(float(value) for value in fits["hour_post_delivery"].unique())
        )
        figure_sets = {
            "A_vcc_selected_clean": _plot_vcc_grid(
                vcc_dense,
                vcc_raw,
                predictions,
                fits,
                hours=SELECTED_HOURS,
                shape=(2, 2),
                include_cloud=False,
                stem=figure_dir / "option_a_vcc_selected_hours_matlab_clean",
            ),
            "B_vcc_selected_cloud": _plot_vcc_grid(
                vcc_dense,
                vcc_raw,
                predictions,
                fits,
                hours=SELECTED_HOURS,
                shape=(2, 2),
                include_cloud=True,
                stem=figure_dir / "option_b_vcc_selected_hours_matlab_cloud",
            ),
            "C_vcc_all_hours_clean": _plot_vcc_grid(
                vcc_dense,
                vcc_raw,
                predictions,
                fits,
                hours=all_hours,
                shape=(3, 3),
                include_cloud=False,
                stem=figure_dir / "option_c_vcc_all_hours_matlab_clean",
            ),
            "D_vac_selected_clean": _plot_vac_selected(
                vac_dense,
                vac_raw,
                msd_alphas,
                include_cloud=False,
                stem=figure_dir / "option_d_vac_selected_hours_matlab_clean",
            ),
            "E_vac_selected_cloud": _plot_vac_selected(
                vac_dense,
                vac_raw,
                msd_alphas,
                include_cloud=True,
                stem=figure_dir / "option_e_vac_selected_hours_matlab_cloud",
            ),
        }

        (staging / "FIGURE_INDEX.md").write_text(_figure_index(), encoding="utf-8")
        method = {
            "analysis_change": False,
            "visualization_only": True,
            "experimental_interpolation": False,
            "display_bin_width_scaled_lag": DISPLAY_BIN_WIDTH,
            "minimum_supporting_units_per_display_point": MINIMUM_DISPLAY_UNITS,
            "maximum_one_observation_per_unit_delta_display_bin": True,
            "formal_fit_grid_changed": False,
            "formal_fit_parameters_changed": False,
            "cloud_x_jitter": "deterministic uniform +/-0.010, display only",
            "selected_hours": list(SELECTED_HOURS),
            "vcc_model": "archived hour-specific Rouse fit",
            "vac_model": "fBM Equation 9 with archived independent MSD alpha",
        }
        (staging / "METHOD_NOTES.json").write_text(
            json.dumps(method, indent=2) + "\n", encoding="utf-8"
        )

        artifacts = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = {
            "sources": [
                {
                    "role": role,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for role, path in sources.items()
            ],
            "artifacts": [
                {
                    "path": path.relative_to(staging).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in artifacts
            ],
            "figure_sets": {
                key: [str(path.relative_to(staging)) for path in paths]
                for key, paths in figure_sets.items()
            },
        }
        (staging / "OUTPUT_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise

    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
