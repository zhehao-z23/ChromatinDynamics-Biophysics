"""Shared, coordinate-safe helpers for Oligo-LiveFISH trajectory animations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.collections import LineCollection
from scipy import ndimage


CHANNEL_META = {
    "G": {"name": "53BP1", "raw_index": 0, "color": "#00A67A"},
    "R": {"name": "Site 1", "raw_index": 1, "color": "#E69F00"},
    "P": {"name": "Site 2", "raw_index": 2, "color": "#B04CC2"},
}


@dataclass(frozen=True)
class CropBounds:
    """Pixel-centre-aligned crop bounds with exclusive upper edges."""

    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x0 - 0.5, self.x1 - 0.5, self.y1 - 0.5, self.y0 - 0.5)


def configure_matplotlib() -> None:
    """Set a restrained, presentation-ready style without changing global files."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 1.1,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": None,
        }
    )


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_corrected_tcyx(path: str | Path) -> np.ndarray:
    """Read a corrected movie and normalize it to TCYX."""

    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes.upper()
    if axes == "TCYX":
        return data
    if axes == "CTYX":
        return np.moveaxis(data, 0, 1)
    raise ValueError(f"Expected corrected TCYX/CTYX TIFF, got axes={axes!r} at {path}")


def load_track(path: str | Path, pixel_size_nm: float) -> pd.DataFrame:
    """Load an automatic trajectory and convert whole-image nm to image pixels."""

    frame = pd.read_csv(path)
    required = {"frame", "x_nm", "y_nm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    frame = frame.loc[:, ["frame", "x_nm", "y_nm"]].copy()
    frame["frame"] = frame["frame"].astype(int)
    frame["x_px"] = frame["x_nm"].astype(float) / float(pixel_size_nm) - 1.0
    frame["y_px"] = frame["y_nm"].astype(float) / float(pixel_size_nm) - 1.0
    frame.sort_values("frame", inplace=True, kind="stable")
    frame.reset_index(drop=True, inplace=True)
    return frame


def relative_times(metadata: dict, n_frames: int) -> np.ndarray:
    """Prefer exact acquisition timestamps and fall back to the declared interval."""

    time_meta = metadata.get("time", {})
    exact = np.asarray(time_meta.get("relative_time_s", []), dtype=float)
    if exact.size == n_frames and np.all(np.isfinite(exact)):
        return exact - exact[0]
    interval = float(time_meta.get("finterval_s", 1.0))
    return np.arange(n_frames, dtype=float) * interval


def crop_around_points(
    x: Iterable[float],
    y: Iterable[float],
    image_shape: tuple[int, int],
    minimum_size_px: int = 48,
    padding_px: int = 10,
) -> CropBounds:
    """Return a square crop that encloses all finite points and remains in-frame."""

    h, w = image_shape
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not finite.any():
        raise ValueError("Cannot define a crop from zero finite coordinates")
    min_x, max_x = float(x_arr[finite].min()), float(x_arr[finite].max())
    min_y, max_y = float(y_arr[finite].min()), float(y_arr[finite].max())
    required = int(math.ceil(max(max_x - min_x, max_y - min_y))) + 2 * int(padding_px) + 1
    size = min(max(int(minimum_size_px), required), min(h, w))
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = min(max(x0, 0), w - size)
    y0 = min(max(y0, 0), h - size)
    return CropBounds(x0=x0, x1=x0 + size, y0=y0, y1=y0 + size)


def fixed_intensity_limits(stack_tyx: np.ndarray, bounds: CropBounds) -> tuple[float, float]:
    """Compute one robust display range for all movie frames to avoid flicker."""

    values = stack_tyx[:, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1].astype(np.float32)
    vmin, vmax = np.percentile(values, [1.0, 99.8])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    return float(vmin), float(vmax)


def show_frame(
    ax: mpl.axes.Axes,
    stack_tyx: np.ndarray,
    frame_index: int,
    bounds: CropBounds,
    limits: tuple[float, float],
) -> None:
    """Render a corrected-frame crop in whole-image pixel coordinates."""

    image = stack_tyx[frame_index, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
    ax.imshow(
        image,
        cmap="gray",
        origin="upper",
        extent=bounds.extent,
        interpolation="nearest",
        norm=mpl.colors.PowerNorm(gamma=0.78, vmin=limits[0], vmax=limits[1], clip=True),
    )
    ax.set_xlim(bounds.x0 - 0.5, bounds.x1 - 0.5)
    ax.set_ylim(bounds.y1 - 0.5, bounds.y0 - 0.5)
    ax.set_aspect("equal")


def add_scale_bar(
    ax: mpl.axes.Axes,
    bounds: CropBounds,
    pixel_size_um: float,
    length_um: float = 1.0,
    color: str = "white",
) -> None:
    """Draw a physically calibrated bar inside an image-coordinate axes."""

    bar_px = float(length_um) / float(pixel_size_um)
    x_start = bounds.x0 + 0.09 * bounds.width
    y = bounds.y1 - 0.12 * bounds.height
    ax.plot(
        [x_start, x_start + bar_px],
        [y, y],
        color=color,
        linewidth=4.0,
        solid_capstyle="butt",
        zorder=20,
    )
    ax.text(
        x_start + 0.5 * bar_px,
        y - 0.055 * bounds.height,
        f"{length_um:g} µm",
        color=color,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        zorder=20,
    )


def rolling_track(track: pd.DataFrame, current_frame: int, tail_frames: int) -> pd.DataFrame:
    lower = max(1, int(current_frame) - int(tail_frames) + 1)
    return track[(track["frame"] >= lower) & (track["frame"] <= int(current_frame))]


def plot_track_with_gaps(
    ax: mpl.axes.Axes,
    track: pd.DataFrame,
    color: str,
    linewidth: float = 2.0,
    zorder: int = 10,
) -> None:
    """Connect localizations while marking gap-linked segments as dashed."""

    if track.empty:
        return
    if len(track) == 1:
        ax.scatter(track["x_px"], track["y_px"], s=10, c=color, zorder=zorder)
        return
    rows = track.reset_index(drop=True)
    for index in range(1, len(rows)):
        first = rows.iloc[index - 1]
        second = rows.iloc[index]
        gap = int(second["frame"] - first["frame"])
        ax.plot(
            [first["x_px"], second["x_px"]],
            [first["y_px"], second["y_px"]],
            color=color,
            linewidth=linewidth,
            linestyle="-" if gap == 1 else "--",
            alpha=0.98 if gap == 1 else 0.78,
            zorder=zorder,
        )


def plot_time_colored_track(
    ax: mpl.axes.Axes,
    track: pd.DataFrame,
    times_s: np.ndarray,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    linewidth: float = 3.0,
) -> None:
    """Plot trajectory segments colored by destination-frame acquisition time."""

    if len(track) < 2:
        return
    points = track.loc[:, ["x_px", "y_px"]].to_numpy(float)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    destination_frames = track["frame"].to_numpy(int)[1:] - 1
    line = LineCollection(segments, cmap=cmap, norm=norm, linewidths=linewidth, zorder=10)
    line.set_array(times_s[destination_frames])
    ax.add_collection(line)


def figure_to_rgb(fig: mpl.figure.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def encode_mp4(
    output_path: str | Path,
    render_frame: Callable[[int], np.ndarray],
    frame_indices: Iterable[int],
    fps: float = 6.0,
    crf: int = 18,
) -> dict:
    """Stream RGB frames to H.264 without accumulating the movie in memory."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = list(frame_indices)
    if not indices:
        raise ValueError("At least one frame is required")
    first = render_frame(indices[0])
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise ValueError(f"H.264 yuv420p requires even dimensions, got {width}x{height}")
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        size=(width, height),
        fps=float(fps),
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-crf", str(int(crf)), "-preset", "medium", "-movflags", "+faststart"],
        ffmpeg_log_level="warning",
    )
    writer.send(None)
    try:
        writer.send(first)
        for frame_index in indices[1:]:
            writer.send(render_frame(frame_index))
    finally:
        writer.close()
    return {
        "path": str(output_path.resolve()),
        "frames": len(indices),
        "playback_fps": float(fps),
        "duration_s": len(indices) / float(fps),
        "width": int(width),
        "height": int(height),
        "codec": "H.264/libx264",
        "pixel_format": "yuv420p",
    }


def compute_valid_mask(image: np.ndarray, fill_ratio: float = 0.85) -> np.ndarray:
    """Production-equivalent exclusion of border-connected drift-fill pixels."""

    background_reference = float(np.percentile(image, 75))
    candidate = image < background_reference * float(fill_ratio)
    labeled, _ = ndimage.label(candidate)
    border_labels: set[int] = set()
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border_labels.update(int(value) for value in edge if value > 0)
    if not border_labels:
        return np.ones(image.shape, dtype=bool)
    fill = np.zeros(image.shape, dtype=bool)
    for label_id in border_labels:
        fill |= labeled == label_id
    return ~fill


def reconstruct_reference_components(
    stack_tyx: np.ndarray,
    seed_xy_px: tuple[float, float],
    tracking_k: float = 0.5,
    adjacency_px: float = 30.0,
) -> list[dict]:
    """Rebuild the exact per-frame connected region chosen by reference tracking."""

    seed_x, seed_y = map(float, seed_xy_px)
    records: list[dict] = []
    for frame_index, frame in enumerate(stack_tyx):
        valid = compute_valid_mask(frame)
        mean = float(frame[valid].mean())
        standard_deviation = float(frame[valid].std())
        threshold = mean + float(tracking_k) * standard_deviation
        binary = (frame > threshold) & valid
        labeled, component_count = ndimage.label(binary)
        labels = np.arange(1, component_count + 1, dtype=int)
        if component_count:
            centers_yx = np.asarray(ndimage.center_of_mass(frame, labeled, labels), dtype=float)
        else:
            centers_yx = np.empty((0, 2), dtype=float)
        selected_label = 0
        selected_distance = float(adjacency_px)
        selected_y = float("nan")
        selected_x = float("nan")
        for label_id, (center_y, center_x) in zip(labels, centers_yx):
            distance = float(np.hypot(center_y - seed_y, center_x - seed_x))
            if distance < selected_distance:
                selected_label = int(label_id)
                selected_distance = distance
                selected_y = float(center_y)
                selected_x = float(center_x)
        selected_mask = labeled == selected_label if selected_label else np.zeros_like(binary)
        records.append(
            {
                "frame": frame_index + 1,
                "mean": mean,
                "std": standard_deviation,
                "threshold": threshold,
                "binary": binary,
                "labeled": labeled,
                "component_count": int(component_count),
                "selected_label": selected_label,
                "selected_mask": selected_mask,
                "centroid_x_px": selected_x,
                "centroid_y_px": selected_y,
                "distance_to_seed_px": selected_distance if selected_label else float("nan"),
                "area_px": int(selected_mask.sum()),
            }
        )
    return records


def radial_displacement_nm(track: pd.DataFrame, pixel_size_nm: float) -> np.ndarray:
    if track.empty:
        return np.empty(0, dtype=float)
    x = track["x_px"].to_numpy(float)
    y = track["y_px"].to_numpy(float)
    return np.hypot(x - x[0], y - y[0]) * float(pixel_size_nm)


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
