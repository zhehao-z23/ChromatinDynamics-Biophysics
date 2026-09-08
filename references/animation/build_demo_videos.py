"""Build three reusable dynamic QC demos from a completed Oligo-LiveFISH run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import ndimage
from skimage import measure

from animation_core import (
    CHANNEL_META,
    CropBounds,
    add_scale_bar,
    configure_matplotlib,
    crop_around_points,
    encode_mp4,
    figure_to_rgb,
    fixed_intensity_limits,
    load_corrected_tcyx,
    load_json,
    load_track,
    plot_time_colored_track,
    plot_track_with_gaps,
    radial_displacement_nm,
    reconstruct_reference_components,
    relative_times,
    rolling_track,
    sha256_file,
    show_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "fov7_cell00_demo.json",
        help="JSON configuration containing source paths and display settings.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override config output_dir.")
    parser.add_argument("--fps", type=float, help="Override MP4 playback fps.")
    parser.add_argument(
        "--only",
        choices=("all", "multichannel", "time-colored", "connected-region"),
        default="all",
        help="Build all demos or only one named demo.",
    )
    return parser.parse_args()


def resolve_config(path: Path, output_override: Path | None, fps_override: float | None) -> dict:
    config = load_json(path)
    config["_config_path"] = str(path.resolve())
    if output_override is not None:
        config["output_dir"] = str(output_override)
    if fps_override is not None:
        config["playback_fps"] = float(fps_override)
    return config


def select_tracks(config: dict, pixel_size_nm: float) -> tuple[pd.DataFrame, dict[str, dict]]:
    manifest_path = Path(config["baseline_manifest"])
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"No selected baselines in {manifest_path}")
    selected: dict[str, dict] = {}
    for channel in ("G", "R", "P"):
        subset = manifest[manifest["channel"] == channel].copy()
        if subset.empty:
            raise ValueError(f"No {channel} baseline in {manifest_path}")
        subset.sort_values(
            ["points", "frame_span", "first_frame", "allele_index"],
            ascending=[False, False, True, True],
            inplace=True,
            kind="stable",
        )
        row = subset.iloc[0].to_dict()
        row["track"] = load_track(row["baseline_csv"], pixel_size_nm)
        selected[channel] = row
    return manifest, selected


def save_poster(render_frame, frame_index: int, output_path: Path) -> None:
    image = render_frame(frame_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image)


def make_multichannel_renderer(
    movie: np.ndarray,
    selected: dict[str, dict],
    times_s: np.ndarray,
    pixel_size_nm: float,
    tail_frames: int,
):
    channels = ("G", "R", "P")
    bounds: dict[str, CropBounds] = {}
    limits: dict[str, tuple[float, float]] = {}
    for channel in channels:
        track = selected[channel]["track"]
        channel_stack = movie[:, CHANNEL_META[channel]["raw_index"]]
        bounds[channel] = crop_around_points(
            track["x_px"], track["y_px"], channel_stack.shape[1:], minimum_size_px=52, padding_px=12
        )
        limits[channel] = fixed_intensity_limits(channel_stack, bounds[channel])

    maximum_displacement = 0.0
    for channel in channels:
        maximum_displacement = max(
            maximum_displacement,
            float(np.nanmax(radial_displacement_nm(selected[channel]["track"], pixel_size_nm))),
        )
    y_limit = max(250.0, 1.12 * maximum_displacement)

    def render(frame_index: int) -> np.ndarray:
        fig = plt.figure(figsize=(16, 9), dpi=100)
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=(1.0, 1.0, 1.34),
            wspace=0.20,
            hspace=0.22,
            left=0.045,
            right=0.975,
            bottom=0.075,
            top=0.92,
        )
        image_axes = {
            "G": fig.add_subplot(grid[0, 0]),
            "R": fig.add_subplot(grid[0, 1]),
            "P": fig.add_subplot(grid[1, 0]),
        }
        note_axis = fig.add_subplot(grid[1, 1])
        time_axis = fig.add_subplot(grid[:, 2])

        for channel in channels:
            meta = CHANNEL_META[channel]
            track = selected[channel]["track"]
            axis = image_axes[channel]
            channel_stack = movie[:, meta["raw_index"]]
            show_frame(axis, channel_stack, frame_index, bounds[channel], limits[channel])
            history = rolling_track(track, frame_index + 1, tail_frames)
            plot_track_with_gaps(axis, history, meta["color"], linewidth=2.3)
            current = track[track["frame"] == frame_index + 1]
            if not current.empty:
                axis.scatter(
                    current["x_px"],
                    current["y_px"],
                    s=72,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=1.8,
                    zorder=22,
                )
            add_scale_bar(axis, bounds[channel], pixel_size_nm / 1000.0, length_um=1.0)
            axis.set_title(
                f"{channel} · {meta['name']}  |  {int(selected[channel]['points'])} localizations",
                color=meta["color"],
                fontweight="bold",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#363636")
                spine.set_linewidth(1.3)

        note_axis.axis("off")
        note_axis.text(
            0.02,
            0.92,
            "Best recovered trajectory\nfrom each channel",
            transform=note_axis.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            color="#202020",
        )
        lines = []
        for row_index, channel in enumerate(channels):
            meta = CHANNEL_META[channel]
            row = selected[channel]
            lines.append(
                f"{channel}  allele {int(row['allele_index'])}   "
                f"frames {int(row['first_frame'])}–{int(row['last_frame'])}"
            )
            note_axis.text(
                0.03,
                0.62 - 0.13 * row_index,
                lines[-1],
                transform=note_axis.transAxes,
                color=meta["color"],
                fontsize=13,
                fontweight="bold",
                ha="left",
                va="center",
            )
        note_axis.text(
            0.03,
            0.14,
            f"Rolling tail: {tail_frames} frames\nDashed segment: gap-linked step",
            transform=note_axis.transAxes,
            color="#555555",
            fontsize=11,
            ha="left",
            va="bottom",
        )

        for channel in channels:
            track = selected[channel]["track"]
            frames = track["frame"].to_numpy(int) - 1
            visible = frames <= frame_index
            displacement = radial_displacement_nm(track, pixel_size_nm)
            if visible.any():
                time_axis.plot(
                    times_s[frames[visible]],
                    displacement[visible],
                    color=CHANNEL_META[channel]["color"],
                    linewidth=2.2,
                    marker="o",
                    markersize=3.8,
                    label=f"{channel} · {CHANNEL_META[channel]['name']}",
                )
                if np.any(frames == frame_index):
                    point_index = int(np.flatnonzero(frames == frame_index)[-1])
                    time_axis.scatter(
                        [times_s[frame_index]],
                        [displacement[point_index]],
                        s=70,
                        facecolor="white",
                        edgecolor=CHANNEL_META[channel]["color"],
                        linewidth=2,
                        zorder=8,
                    )
        time_axis.axvline(times_s[frame_index], color="#333333", linewidth=1.3, alpha=0.75)
        time_axis.set_xlim(float(times_s[0]), float(times_s[-1]))
        time_axis.set_ylim(0, y_limit)
        time_axis.set_xlabel("Acquisition time (s)")
        time_axis.set_ylabel("Displacement from first localization (nm)")
        time_axis.set_title(
            f"t = {times_s[frame_index]:.1f} s  |  frame {frame_index + 1}/{len(times_s)}",
            fontweight="bold",
        )
        time_axis.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
        time_axis.legend(frameon=False, loc="upper left")
        fig.suptitle("Dynamic tracking of longest recovered candidates", fontsize=20, fontweight="bold", y=0.975)
        image = figure_to_rgb(fig)
        plt.close(fig)
        return image

    return render, bounds


def make_time_colored_renderer(
    movie: np.ndarray,
    track: pd.DataFrame,
    channel: str,
    times_s: np.ndarray,
    pixel_size_nm: float,
    tail_frames: int,
):
    stack = movie[:, CHANNEL_META[channel]["raw_index"]]
    bounds = crop_around_points(track["x_px"], track["y_px"], stack.shape[1:], minimum_size_px=58, padding_px=14)
    limits = fixed_intensity_limits(stack, bounds)
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=float(times_s[0]), vmax=float(times_s[-1]))
    pixel_size_um = pixel_size_nm / 1000.0
    x_um = (track["x_px"].to_numpy(float) - float(track["x_px"].iloc[0])) * pixel_size_um
    y_um = -(track["y_px"].to_numpy(float) - float(track["y_px"].iloc[0])) * pixel_size_um
    max_range = max(float(np.ptp(x_um)), float(np.ptp(y_um)), 0.6)
    margin = 0.18 * max_range
    x_mid = 0.5 * (float(x_um.min()) + float(x_um.max()))
    y_mid = 0.5 * (float(y_um.min()) + float(y_um.max()))
    half = 0.5 * max_range + margin

    def render(frame_index: int) -> np.ndarray:
        fig = plt.figure(figsize=(14, 7.5), dpi=100)
        grid = fig.add_gridspec(1, 2, width_ratios=(1.08, 1.0), wspace=0.19, left=0.055, right=0.92, bottom=0.09, top=0.88)
        raw_axis = fig.add_subplot(grid[0, 0])
        trajectory_axis = fig.add_subplot(grid[0, 1])
        show_frame(raw_axis, stack, frame_index, bounds, limits)
        history = rolling_track(track, frame_index + 1, tail_frames)
        plot_time_colored_track(raw_axis, history, times_s, norm, cmap, linewidth=3.2)
        current = track[track["frame"] == frame_index + 1]
        if not current.empty:
            raw_axis.scatter(
                current["x_px"], current["y_px"], s=94, facecolors="none", edgecolors="white", linewidths=2.0, zorder=30
            )
        add_scale_bar(raw_axis, bounds, pixel_size_um, length_um=1.0)
        raw_axis.set_title(f"Corrected {CHANNEL_META[channel]['name']} image", fontweight="bold")
        raw_axis.set_xticks([])
        raw_axis.set_yticks([])

        visible = track[track["frame"] <= frame_index + 1]
        if len(visible) >= 2:
            indices = visible.index.to_numpy(int)
            points = np.column_stack([x_um[indices], y_um[indices]])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            frame_numbers = visible["frame"].to_numpy(int)[1:] - 1
            line = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm, linewidths=3.0)
            line.set_array(times_s[frame_numbers])
            trajectory_axis.add_collection(line)
        if not visible.empty:
            last_index = int(visible.index[-1])
            last_frame_index = int(visible["frame"].iloc[-1]) - 1
            trajectory_axis.scatter(
                [x_um[last_index]], [y_um[last_index]], s=86, facecolor="white", edgecolor=cmap(norm(times_s[last_frame_index])), linewidth=2.2, zorder=20
            )
        trajectory_axis.set_xlim(x_mid - half, x_mid + half)
        trajectory_axis.set_ylim(y_mid - half, y_mid + half)
        trajectory_axis.set_aspect("equal")
        trajectory_axis.set_xticks([])
        trajectory_axis.set_yticks([])
        trajectory_axis.set_title("Time-coded trajectory", fontweight="bold")
        for spine in trajectory_axis.spines.values():
            spine.set_visible(False)
        bar_length_um = 0.5
        x0 = x_mid - half + 0.10 * (2 * half)
        y0 = y_mid - half + 0.10 * (2 * half)
        trajectory_axis.plot([x0, x0 + bar_length_um], [y0, y0], color="#202020", linewidth=4, solid_capstyle="butt")
        trajectory_axis.text(x0 + 0.5 * bar_length_um, y0 + 0.055 * (2 * half), "0.5 µm", ha="center", va="bottom", fontsize=11)

        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar_axis = fig.add_axes([0.935, 0.18, 0.018, 0.60])
        colorbar = fig.colorbar(scalar, cax=colorbar_axis)
        colorbar.set_label("Acquisition time (s)")
        fig.suptitle(
            f"Site 2 trajectory · t = {times_s[frame_index]:.1f} s · frame {frame_index + 1}/{len(times_s)}",
            fontsize=19,
            fontweight="bold",
            y=0.965,
        )
        image = figure_to_rgb(fig)
        plt.close(fig)
        return image

    return render, bounds


def component_contours(mask: np.ndarray, bounds: CropBounds) -> list[np.ndarray]:
    crop = mask[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
    contours = measure.find_contours(crop.astype(float), 0.5)
    converted = []
    for contour in contours:
        converted.append(np.column_stack([contour[:, 1] + bounds.x0, contour[:, 0] + bounds.y0]))
    return converted


def make_connected_region_renderer(
    movie: np.ndarray,
    reference_track: pd.DataFrame,
    seed_xy_px: tuple[float, float],
    times_s: np.ndarray,
    pixel_size_nm: float,
    tail_frames: int,
    tracking_k: float,
    adjacency_px: float,
):
    stack = movie[:, CHANNEL_META["P"]["raw_index"]]
    records = reconstruct_reference_components(stack, seed_xy_px, tracking_k=tracking_k, adjacency_px=adjacency_px)
    all_x = [record["centroid_x_px"] for record in records if np.isfinite(record["centroid_x_px"])] + [seed_xy_px[0]]
    all_y = [record["centroid_y_px"] for record in records if np.isfinite(record["centroid_y_px"])] + [seed_xy_px[1]]
    bounds = crop_around_points(all_x, all_y, stack.shape[1:], minimum_size_px=72, padding_px=18)
    limits = fixed_intensity_limits(stack, bounds)
    pixel_size_um = pixel_size_nm / 1000.0

    reconstructed = pd.DataFrame(
        {
            "frame": [record["frame"] for record in records],
            "x_px": [record["centroid_x_px"] for record in records],
            "y_px": [record["centroid_y_px"] for record in records],
        }
    )

    def render(frame_index: int) -> np.ndarray:
        fig = plt.figure(figsize=(14, 7.5), dpi=100)
        grid = fig.add_gridspec(1, 2, wspace=0.17, left=0.055, right=0.96, bottom=0.08, top=0.88)
        raw_axis = fig.add_subplot(grid[0, 0])
        binary_axis = fig.add_subplot(grid[0, 1])
        record = records[frame_index]

        show_frame(raw_axis, stack, frame_index, bounds, limits)
        for contour in component_contours(record["selected_mask"], bounds):
            raw_axis.plot(contour[:, 0], contour[:, 1], color="#00D7D7", linewidth=2.8, zorder=12)
        history = rolling_track(reconstructed, frame_index + 1, tail_frames)
        plot_track_with_gaps(raw_axis, history, "#FF4FA3", linewidth=2.0, zorder=13)
        if record["selected_label"]:
            raw_axis.scatter(
                [record["centroid_x_px"]],
                [record["centroid_y_px"]],
                marker="+",
                s=180,
                c="#FFD23F",
                linewidths=2.5,
                zorder=18,
            )
        raw_axis.scatter([seed_xy_px[0]], [seed_xy_px[1]], marker="x", s=80, c="white", linewidths=1.6, zorder=17)
        add_scale_bar(raw_axis, bounds, pixel_size_um, length_um=1.0)
        raw_axis.set_title("Corrected Site 2 image", fontweight="bold")
        raw_axis.set_xticks([])
        raw_axis.set_yticks([])

        binary_crop = record["binary"][bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
        selected_crop = record["selected_mask"][bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
        display = np.zeros((*binary_crop.shape, 3), dtype=float)
        display[binary_crop] = (0.38, 0.38, 0.38)
        display[selected_crop] = (0.0, 0.84, 0.84)
        binary_axis.imshow(display, origin="upper", extent=bounds.extent, interpolation="nearest")
        binary_axis.set_xlim(bounds.x0 - 0.5, bounds.x1 - 0.5)
        binary_axis.set_ylim(bounds.y1 - 0.5, bounds.y0 - 0.5)
        binary_axis.set_aspect("equal")
        search_circle = plt.Circle(
            seed_xy_px,
            adjacency_px,
            fill=False,
            edgecolor="#B0B0B0",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        binary_axis.add_patch(search_circle)
        binary_axis.scatter([seed_xy_px[0]], [seed_xy_px[1]], marker="x", s=90, c="white", linewidths=1.8, zorder=18)
        if record["selected_label"]:
            binary_axis.scatter(
                [record["centroid_x_px"]],
                [record["centroid_y_px"]],
                marker="+",
                s=200,
                c="#FFD23F",
                linewidths=2.8,
                zorder=19,
            )
            binary_axis.plot(
                [seed_xy_px[0], record["centroid_x_px"]],
                [seed_xy_px[1], record["centroid_y_px"]],
                color="white",
                linewidth=1.0,
                alpha=0.75,
                zorder=17,
            )
        binary_axis.set_title("Thresholded connected regions", fontweight="bold")
        binary_axis.set_xticks([])
        binary_axis.set_yticks([])
        legend_handles = [
            Line2D([0], [0], color="#00D7D7", linewidth=5, label="selected connected region"),
            Line2D([0], [0], marker="+", linestyle="none", color="#FFD23F", markersize=12, markeredgewidth=2, label="intensity-weighted centroid"),
            Line2D([0], [0], marker="x", linestyle="none", color="white", markersize=9, label="time-average seed centroid"),
        ]
        binary_axis.legend(handles=legend_handles, loc="lower right", frameon=False, labelcolor="white")

        status = (
            f"t = {times_s[frame_index]:.1f} s · frame {frame_index + 1}/{len(times_s)}   |   "
            f"threshold = μ + {tracking_k:g}σ = {record['threshold']:.1f} ADU   |   "
            f"selected area = {record['area_px']} px"
        )
        fig.suptitle(status, fontsize=17, fontweight="bold", y=0.965)
        image = figure_to_rgb(fig)
        plt.close(fig)
        return image

    return render, bounds, records


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    config = resolve_config(args.config.resolve(), args.output_dir, args.fps)
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    poster_dir = output_dir / "posters"
    poster_dir.mkdir(parents=True, exist_ok=True)

    corrected_tiff = Path(config["corrected_tiff"])
    metadata_path = Path(config["metadata_json"])
    movie = load_corrected_tcyx(corrected_tiff)
    metadata = load_json(metadata_path)
    n_frames = int(movie.shape[0])
    times_s = relative_times(metadata, n_frames)
    pixel_size_nm = float(config.get("pixel_size_nm", metadata["pixel_size"]["x_um"] * 1000.0))
    playback_fps = float(config.get("playback_fps", 6.0))
    tail_frames = int(config.get("track_tail_frames", 30))
    manifest, selected = select_tracks(config, pixel_size_nm)

    products: list[dict] = []
    selections = {
        channel: {
            "allele_index": int(row["allele_index"]),
            "points": int(row["points"]),
            "first_frame": int(row["first_frame"]),
            "last_frame": int(row["last_frame"]),
            "baseline_csv": str(Path(row["baseline_csv"]).resolve()),
        }
        for channel, row in selected.items()
    }

    if args.only in ("all", "multichannel"):
        renderer, _ = make_multichannel_renderer(movie, selected, times_s, pixel_size_nm, tail_frames)
        product = encode_mp4(
            output_dir / "01_multichannel_longest_tracking.mp4",
            renderer,
            range(n_frames),
            fps=playback_fps,
        )
        poster = poster_dir / "01_multichannel_longest_tracking_frame25.png"
        save_poster(renderer, min(24, n_frames - 1), poster)
        product["poster"] = str(poster.resolve())
        product["selection"] = selections
        products.append(product)

    if args.only in ("all", "time-colored"):
        p_track = selected["P"]["track"]
        renderer, _ = make_time_colored_renderer(movie, p_track, "P", times_s, pixel_size_nm, tail_frames)
        product = encode_mp4(
            output_dir / "02_site2_time_colored_tracking.mp4",
            renderer,
            range(n_frames),
            fps=playback_fps,
        )
        poster = poster_dir / "02_site2_time_colored_tracking_frame25.png"
        save_poster(renderer, min(24, n_frames - 1), poster)
        product["poster"] = str(poster.resolve())
        product["selection"] = selections["P"]
        products.append(product)

    component_metrics = None
    if args.only in ("all", "connected-region"):
        connected_locus = config.get("connected_reference_locus", "match_longest_site2")
        if connected_locus == "match_longest_site2":
            locus = int(selected["P"]["anchor_locus"])
        else:
            locus = int(connected_locus)
        audit = load_json(config["reference_audit_json"])
        returned = {int(row["rank"]): row for row in audit["returned_components"]}
        if locus not in returned:
            raise ValueError(f"Reference locus/rank {locus} is absent from reference audit")
        seed = returned[locus]
        seed_xy = (float(seed["centroid_x_px"]), float(seed["centroid_y_px"]))
        reference_csv = Path(config["reference_trajectory_pattern"].format(locus=locus))
        reference_track = load_track(reference_csv, pixel_size_nm)
        tracking_k = float(audit.get("reference_tracking_k", 0.5))
        adjacency_px = float(config.get("reference_adjacency_px", 30.0))
        renderer, _, records = make_connected_region_renderer(
            movie,
            reference_track,
            seed_xy,
            times_s,
            pixel_size_nm,
            tail_frames,
            tracking_k,
            adjacency_px,
        )
        product = encode_mp4(
            output_dir / "03_connected_region_centroid_tracking.mp4",
            renderer,
            range(n_frames),
            fps=playback_fps,
        )
        poster = poster_dir / "03_connected_region_centroid_tracking_frame25.png"
        save_poster(renderer, min(24, n_frames - 1), poster)
        computed = pd.DataFrame(
            {
                "frame": [record["frame"] for record in records],
                "x_px_reconstructed": [record["centroid_x_px"] for record in records],
                "y_px_reconstructed": [record["centroid_y_px"] for record in records],
                "area_px": [record["area_px"] for record in records],
                "threshold_adu": [record["threshold"] for record in records],
                "distance_to_seed_px": [record["distance_to_seed_px"] for record in records],
            }
        )
        joined = computed.merge(
            reference_track.loc[:, ["frame", "x_px", "y_px"]], on="frame", how="left", validate="one_to_one"
        )
        joined["centroid_difference_px"] = np.hypot(
            joined["x_px_reconstructed"] - joined["x_px"],
            joined["y_px_reconstructed"] - joined["y_px"],
        )
        table_path = output_dir / "03_connected_region_reconstruction.tsv"
        joined.to_csv(table_path, sep="\t", index=False)
        component_metrics = {
            "reference_locus": locus,
            "seed_component_id": int(seed["component_id"]),
            "seed_centroid_x_px": seed_xy[0],
            "seed_centroid_y_px": seed_xy[1],
            "tracking_k": tracking_k,
            "adjacency_px": adjacency_px,
            "matched_frames": int(joined["centroid_difference_px"].notna().sum()),
            "centroid_rmse_px": float(np.sqrt(np.nanmean(joined["centroid_difference_px"] ** 2))),
            "centroid_max_abs_difference_px": float(np.nanmax(joined["centroid_difference_px"])),
            "reconstruction_table": str(table_path.resolve()),
        }
        product["poster"] = str(poster.resolve())
        product["selection"] = {
            "reference_csv": str(reference_csv.resolve()),
            **component_metrics,
        }
        products.append(product)

    provenance_paths = [
        corrected_tiff,
        metadata_path,
        Path(config["baseline_manifest"]),
        Path(config["reference_audit_json"]),
    ] + [Path(item["baseline_csv"]) for item in selections.values()]
    manifest_out = {
        "config": config,
        "coordinate_contract": {
            "image": "corrected TCYX, zero-based pixel centres, x=column, y=row, origin upper-left",
            "automatic_csv": "whole-corrected-image one-based pixel centres in nm",
            "overlay_formula": "image_px = automatic_nm / pixel_size_nm - 1",
            "frame_formula": "array_index = csv_frame - 1",
        },
        "movie_shape_tcyx": [int(value) for value in movie.shape],
        "pixel_size_nm": pixel_size_nm,
        "exact_time_axis_used": True,
        "acquisition_duration_s": float(times_s[-1] - times_s[0]),
        "best_per_channel_selection": selections,
        "connected_region_reconstruction": component_metrics,
        "products": products,
        "source_files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in provenance_paths
            if path.exists()
        ],
    }
    manifest_path = output_dir / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "products": products, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
