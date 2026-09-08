#!/usr/bin/env python
"""Render the frozen equal-bundle 10-s MSCD estimate across folder-hour without 5 h."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dsb_states.plots import COLORS, configure_publication_style, save_figure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_frozen_mscd10(fits: pd.DataFrame) -> pd.DataFrame:
    selected = fits.loc[
        fits["metric"].astype(str).eq("mscd")
        & fits["site"].astype(str).eq("pair")
        & fits["estimate_name"].astype(str).eq("value_at_10s")
    ].copy()
    for column in ("hour_post_delivery", "estimate", "ci_low", "ci_high"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.loc[
        np.isfinite(selected[["hour_post_delivery", "estimate", "ci_low", "ci_high"]]).all(
            axis=1
        )
        & ~np.isclose(selected["hour_post_delivery"], 5.0)
    ].sort_values("hour_post_delivery", kind="stable")
    if len(selected) != 8:
        raise ValueError(f"Expected eight non-5-h MSCD rows, found {len(selected)}")
    if not ((selected["ci_low"] <= selected["estimate"]) & (selected["estimate"] <= selected["ci_high"])).all():
        raise ValueError("Invalid supplied confidence interval")
    return selected.reset_index(drop=True)


def _build_figure(selected: pd.DataFrame, output_stem: Path) -> list[Path]:
    configure_publication_style()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 17,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )
    figure, axis = plt.subplots(figsize=(7.6, 4.9))
    estimate = selected["estimate"].to_numpy(float)
    low = selected["ci_low"].to_numpy(float)
    high = selected["ci_high"].to_numpy(float)
    errors = np.vstack((estimate - low, high - estimate))
    axis.errorbar(
        selected["hour_post_delivery"],
        estimate,
        yerr=errors,
        color=COLORS["neutral"],
        marker="o",
        markersize=8.0,
        markerfacecolor=COLORS["neutral"],
        markeredgecolor=COLORS["neutral"],
        linewidth=2.4,
        elinewidth=1.8,
        capsize=4.0,
        capthick=1.8,
    )
    axis.set_xlabel("Time after Cas9 delivery (h)")
    axis.set_ylabel(r"MSCD at 10 s ($\mathrm{\mu m^2}$)")
    axis.set_xlim(1.2, 10.4)
    axis.set_ylim(0.0, float(selected["ci_high"].max()) * 1.12)
    axis.set_xticks(selected["hour_post_delivery"])
    axis.set_xticklabels([f"{value:g}" for value in selected["hour_post_delivery"]])
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=1.6, length=5.5)
    figure.tight_layout(pad=0.7)
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    fits_path = args.fits_table.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    selected = _select_frozen_mscd10(pd.read_csv(fits_path))
    figure_dir = output / "figures"
    table_dir = output / "tables"
    figure_dir.mkdir(parents=True)
    table_dir.mkdir()
    figure_paths = _build_figure(selected, figure_dir / "fig_mscd10_by_delivery_hour_no5_clean")
    source_table_path = table_dir / "fig_mscd10_by_delivery_hour_no5_clean_source.csv"
    selected.to_csv(source_table_path, index=False)

    contract = {
        "metric": "paired relative-vector MSCD",
        "display_estimate": "frozen equal-bundle fitted value at 10 s",
        "source_fit_model": "MSCD(tau)=A*tau^beta over the frozen common 10--50 s window",
        "uncertainty": "supplied crop-cluster bootstrap 95% confidence interval",
        "excluded_display_hour": 5.0,
        "fit_values_or_intervals_recomputed": False,
        "biological_boundary": (
            "folder-hour is time after Cas9 delivery; acquisitions are technical units and no "
            "biological-replicate population inference is made"
        ),
    }
    contract_path = output / "METHOD_CONTRACT.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report_path = output / "REPORT.md"
    report_path.write_text(
        "# Clean MSCD-at-10-s macro-time display\n\n"
        "The figure shows the frozen equal-bundle fitted MSCD value at 10 s with supplied "
        "crop-cluster bootstrap 95% confidence intervals. The 5 h point is omitted for display "
        "because it has only nine bundles from one acquisition. No estimate or interval was "
        "recomputed.\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()
    plots_module = script_path.parents[1] / "src" / "dsb_states" / "plots.py"
    sources = [fits_path, script_path, plots_module]
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
                "displayed_hours": selected["hour_post_delivery"].tolist(),
                "figure_paths": [str(path) for path in figure_paths],
                "source_table": str(source_table_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
