"""Final multichannel review figures for selected v5.2.1 paired trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class CropBounds:
    """Square image bounds with exclusive upper edges."""

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


@dataclass(frozen=True)
class CaseData:
    """Validated production arrays and coordinates for one paired trajectory."""

    case_id: str
    display_label: str
    bundle_id: str
    nd2_id: str
    crop_id: str
    allele_index: int
    hour_post_delivery: float
    mean_separation_nm: float
    sd_separation_nm: float
    pixel_size_nm: float
    times_s: np.ndarray
    site1_stack: np.ndarray
    site2_stack: np.ndarray
    nucleus_mask_stack: np.ndarray
    nucleus_mask_path: Path
    site1_track: pd.DataFrame
    site2_track: pd.DataFrame
    raw_bounds: CropBounds
    site1_limits: tuple[float, float]
    site2_limits: tuple[float, float]
    spatial_limits_um: tuple[float, float, float, float]
    trace_limit_nm: float
    source_paths: tuple[Path, ...]

    @property
    def n_frames(self) -> int:
        return len(self.times_s)


def configure_case_style() -> None:
    """Use the compact, heavy-axis style of the published Oligo-LiveFISH traces."""

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "axes.linewidth": 2.2,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _load_tyx(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        array = np.asarray(series.asarray())
        axes = series.axes.upper()
    if axes != "TYX":
        raise ValueError(f"Expected corrected TYX tracking input, got {axes} at {path}")
    return array


def _validated_nucleus_mask(mask: np.ndarray, reference_shape: tuple[int, ...]) -> np.ndarray:
    """Return a non-empty, non-scaffold TYX production mask as boolean."""

    if mask.shape != reference_shape:
        raise ValueError(
            f"Nucleus-mask shape {mask.shape} does not match image shape {reference_shape}"
        )
    binary = np.asarray(mask) > 0
    frame_area = binary.reshape(binary.shape[0], -1).sum(axis=1)
    total_area = binary.shape[1] * binary.shape[2]
    if np.any(frame_area == 0) or np.any(frame_area == total_area):
        raise ValueError("Nucleus mask contains an empty or all-foreground scaffold frame")
    return binary


def _load_track(path: Path, pixel_size_nm: float, n_frames: int) -> pd.DataFrame:
    track = pd.read_csv(path)
    required = {"frame", "x_nm", "y_nm"}
    missing = sorted(required.difference(track.columns))
    if missing:
        raise ValueError(f"Track {path} is missing columns: {missing}")
    track = track[["frame", "x_nm", "y_nm"]].copy()
    track["frame"] = track["frame"].astype(int)
    track = track.sort_values("frame", kind="stable").reset_index(drop=True)
    expected_frames = np.arange(1, n_frames + 1, dtype=int)
    if not np.array_equal(track["frame"].to_numpy(int), expected_frames):
        raise ValueError(f"Case track does not cover every acquisition frame: {path}")
    if not np.isfinite(track[["x_nm", "y_nm"]].to_numpy(float)).all():
        raise ValueError(f"Case track contains non-finite coordinates: {path}")
    track["x_px"] = track["x_nm"].astype(float) / pixel_size_nm - 1.0
    track["y_px"] = track["y_nm"].astype(float) / pixel_size_nm - 1.0
    return track


def _relative_times(metadata: dict[str, Any], n_frames: int) -> np.ndarray:
    exact = np.asarray(metadata.get("time", {}).get("relative_time_s", []), dtype=float)
    if exact.size != n_frames or not np.isfinite(exact).all():
        raise ValueError("Exact frame timestamps are required for case-study visualization")
    if np.any(np.diff(exact) <= 0):
        raise ValueError("Exact frame timestamps must increase strictly")
    return exact - exact[0]


def _crop_around_tracks(
    tracks: Iterable[pd.DataFrame],
    image_shape: tuple[int, int],
    *,
    minimum_size_px: int,
    padding_px: int,
) -> CropBounds:
    x = np.concatenate([track["x_px"].to_numpy(float) for track in tracks])
    y = np.concatenate([track["y_px"].to_numpy(float) for track in tracks])
    height, width = image_shape
    required = math.ceil(max(np.ptp(x), np.ptp(y))) + 2 * padding_px + 1
    size = min(max(minimum_size_px, required), min(height, width))
    center_x = 0.5 * (float(x.min()) + float(x.max()))
    center_y = 0.5 * (float(y.min()) + float(y.max()))
    x0 = min(max(round(center_x - size / 2), 0), width - size)
    y0 = min(max(round(center_y - size / 2), 0), height - size)
    bounds = CropBounds(x0=x0, x1=x0 + size, y0=y0, y1=y0 + size)
    if (
        x.min() < bounds.x0 - 0.5
        or x.max() > bounds.x1 - 0.5
        or y.min() < bounds.y0 - 0.5
        or y.max() > bounds.y1 - 0.5
    ):
        raise ValueError("A tracked localization lies outside the declared raw-image crop")
    return bounds


def _fixed_limits(stack: np.ndarray, bounds: CropBounds) -> tuple[float, float]:
    values = stack[:, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1].astype(np.float32)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    source = positive if positive.size >= 100 else finite
    low, high = np.percentile(source, [50.0, 99.75])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(source)), float(np.nanmax(source))
    return float(low), float(high)


def _case_spatial_limits(
    site1: pd.DataFrame,
    site2: pd.DataFrame,
) -> tuple[float, float, float, float]:
    midpoint_x = 0.5 * (site1["x_nm"].to_numpy(float) + site2["x_nm"].to_numpy(float))
    midpoint_y = -0.5 * (site1["y_nm"].to_numpy(float) + site2["y_nm"].to_numpy(float))
    center_x = float(np.median(midpoint_x))
    center_y = float(np.median(midpoint_y))
    all_x = np.concatenate(
        [site1["x_nm"].to_numpy(float) - center_x, site2["x_nm"].to_numpy(float) - center_x]
    ) / 1000.0
    all_y = np.concatenate(
        [-site1["y_nm"].to_numpy(float) - center_y, -site2["y_nm"].to_numpy(float) - center_y]
    ) / 1000.0
    span = max(float(np.ptp(all_x)), float(np.ptp(all_y)), 0.75)
    half = 0.5 * span + 0.18 * span
    center_plot_x = 0.5 * (float(all_x.min()) + float(all_x.max()))
    center_plot_y = 0.5 * (float(all_y.min()) + float(all_y.max()))
    return (
        center_plot_x - half,
        center_plot_x + half,
        center_plot_y - half,
        center_plot_y + half,
    )


def _source_trace_limit(site1: pd.DataFrame, site2: pd.DataFrame) -> float:
    values: list[np.ndarray] = []
    for track in (site1, site2):
        x = track["x_nm"].to_numpy(float)
        y = -track["y_nm"].to_numpy(float)
        values.extend([x - np.median(x), y - np.median(y)])
    maximum = max(float(np.max(np.abs(value))) for value in values)
    target = max(100.0, 1.10 * maximum)
    step = 50.0 if target <= 500 else 100.0
    return float(step * math.ceil(target / step))


def _figure_to_rgb(figure: mpl.figure.Figure) -> np.ndarray:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def encode_mp4(
    output_path: Path,
    render_frame: Callable[[int], np.ndarray],
    frame_indices: Iterable[int],
    *,
    fps: float,
    ffmpeg_exe: Path,
    crf: int = 20,
) -> dict[str, Any]:
    """Stream rendered RGB frames to an H.264 MP4 using a declared FFmpeg binary."""

    indices = list(frame_indices)
    if not indices:
        raise ValueError("At least one movie frame is required")
    ffmpeg_exe = ffmpeg_exe.resolve()
    if not ffmpeg_exe.is_file():
        raise FileNotFoundError(ffmpeg_exe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = render_frame(indices[0])
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise ValueError(f"H.264 yuv420p requires even dimensions, got {width}x{height}")
    encoder_probe = subprocess.run(
        [str(ffmpeg_exe), "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    available_encoders = encoder_probe.stdout
    if "libx264" in available_encoders:
        encoder = "libx264"
        quality_arguments = ["-crf", str(int(crf)), "-preset", "medium"]
        codec_label = "H.264/libx264"
    elif "h264_mf" in available_encoders:
        # Some Windows desktop applications ship a small FFmpeg build with
        # Media Foundation H.264 but without libx264 or CRF support.  This
        # changes only the file encoder, never the rendered scientific frame.
        encoder = "h264_mf"
        quality = int(np.clip(round(100.0 - 2.0 * int(crf)), 1, 100))
        quality_arguments = [
            "-rate_control",
            "quality",
            "-quality",
            str(quality),
            "-hw_encoding",
            "0",
        ]
        codec_label = "H.264/Media Foundation"
    else:
        raise RuntimeError(f"No supported H.264 encoder in {ffmpeg_exe}")
    command = [
        str(ffmpeg_exe),
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{float(fps):g}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        encoder,
        "-pix_fmt",
        "yuv420p",
        *quality_arguments,
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("FFmpeg raw-video stdin pipe was not created")
    try:
        process.stdin.write(first.tobytes())
        for index in indices[1:]:
            process.stdin.write(render_frame(index).tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg exited with code {return_code}: {output_path}")
    return {
        "path": str(output_path.resolve()),
        "frames": len(indices),
        "playback_fps": float(fps),
        "playback_duration_s": len(indices) / float(fps),
        "width": width,
        "height": height,
        "codec": codec_label,
        "pixel_format": "yuv420p",
        "ffmpeg_executable": str(ffmpeg_exe),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hour_slug(hour: float) -> str:
    return f"{hour:g}".replace(".", "p") + "h"


def _channel_paths(case: CaseData) -> dict[str, Path]:
    crop_dir = case.source_paths[0].parent
    stem = case.crop_id
    paths = {
        "crop_dir": crop_dir,
        "composite_tiff": crop_dir / f"{stem}_composite.tif",
        "corrected_crop_tiff": crop_dir / f"{stem}_correct.tif",
        "green_53bp1_tiff": crop_dir / f"{stem}_green.tif",
        "purple_site2_tiff": crop_dir / f"{stem}_purple.tif",
        "red_site1_tiff": crop_dir / f"{stem}_red.tif",
        "nucleus_scaffold_tiff": crop_dir / f"{stem}_Nucleus.tif",
        "aligned_nucleus_mask_tiff": case.nucleus_mask_path,
    }
    for label, path in paths.items():
        if label != "crop_dir" and not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_multichannel_review_cases(
    *,
    config_path: Path,
    fullrun_root: Path,
    selection_csv: Path,
) -> tuple[list[CaseData], dict[str, Any]]:
    """Load the frozen poster-demo cases directly from CSV and production files.

    This intentionally avoids a Parquet runtime dependency. The frozen selection CSV supplies
    identities and trajectory summaries; every coordinate, timestamp and image is reloaded from
    the production sources and reconciled before plotting.
    """

    config = _load_json(config_path)
    display = config["display"]
    selection = pd.read_csv(selection_csv)
    cases: list[CaseData] = []
    for item in config["cases"]:
        bundle_id = str(item["bundle_id"])
        selected = selection.loc[selection["bundle_id"] == bundle_id]
        if len(selected) != 1:
            raise ValueError(f"Expected one selected row for {bundle_id}, found {len(selected)}")
        row = selected.iloc[0]
        nd2_id = str(row["nd2_id"])
        crop_id = str(row["crop_id"])
        crop_candidates = [
            fullrun_root / "03_new_all_candidates" / nd2_id / crop_id,
            fullrun_root / "04_crops" / "old" / nd2_id / crop_id,
        ]
        crop_dirs = [path for path in crop_candidates if path.is_dir()]
        if len(crop_dirs) != 1:
            raise ValueError(f"Expected one production crop directory for {crop_id}: {crop_dirs}")
        crop_dir = crop_dirs[0]
        metadata_path = crop_dir.parent / f"{crop_id}_metadata.json"
        site1_tiff = crop_dir / f"{crop_id}_red.tif"
        site2_tiff = crop_dir / f"{crop_id}_purple.tif"
        allele = int(row["allele_index"])
        result_dirs: list[Path] = []
        for candidate in crop_dir.iterdir():
            if not candidate.is_dir() or not (candidate / "run_manifest.json").is_file():
                continue
            baseline = candidate / "baseline_longest"
            site1_candidate = baseline / f"allele_{allele:03d}_site1_longest_spt_cleaned.csv"
            site2_candidate = baseline / f"allele_{allele:03d}_site2_longest_spt_cleaned.csv"
            if site1_candidate.is_file() and site2_candidate.is_file():
                result_dirs.append(candidate)
        if len(result_dirs) != 1:
            raise ValueError(f"Expected one production result directory for {bundle_id}: {result_dirs}")
        result_dir = result_dirs[0]
        run_manifest_path = result_dir / "run_manifest.json"
        site1_track_path = (
            result_dir
            / "baseline_longest"
            / f"allele_{allele:03d}_site1_longest_spt_cleaned.csv"
        )
        site2_track_path = (
            result_dir
            / "baseline_longest"
            / f"allele_{allele:03d}_site2_longest_spt_cleaned.csv"
        )
        nucleus_mask_path = result_dir / "mask_alignment" / "microsam_mask_aligned_raw.tif"
        source_paths = (
            site1_tiff,
            site2_tiff,
            metadata_path,
            run_manifest_path,
            site1_track_path,
            site2_track_path,
            nucleus_mask_path,
        )
        for source in source_paths:
            if not source.is_file():
                raise FileNotFoundError(source)
        site1_stack = _load_tyx(site1_tiff)
        site2_stack = _load_tyx(site2_tiff)
        if site1_stack.shape != site2_stack.shape:
            raise ValueError(f"Site channel stack mismatch for {bundle_id}")
        nucleus_mask_stack = _validated_nucleus_mask(
            _load_tyx(nucleus_mask_path), site1_stack.shape
        )
        n_frames = int(site1_stack.shape[0])
        if n_frames != int(row["movie_frames"]):
            raise ValueError(f"Movie-frame mismatch for {bundle_id}")
        metadata = _load_json(metadata_path)
        times_s = _relative_times(metadata, n_frames)
        pixel_size_nm = float(metadata["pixel_size"]["x_um"]) * 1000.0
        site1_track = _load_track(site1_track_path, pixel_size_nm, n_frames)
        site2_track = _load_track(site2_track_path, pixel_size_nm, n_frames)
        separation = np.hypot(
            site2_track["x_nm"].to_numpy(float) - site1_track["x_nm"].to_numpy(float),
            site2_track["y_nm"].to_numpy(float) - site1_track["y_nm"].to_numpy(float),
        )
        if not np.isclose(float(np.mean(separation)), float(row["mean_separation_nm"]), atol=1e-9):
            raise ValueError(f"Mean separation does not reconcile for {bundle_id}")
        if not np.isclose(float(np.std(separation, ddof=1)), float(row["sd_separation_nm"]), atol=1e-9):
            raise ValueError(f"Separation SD does not reconcile for {bundle_id}")
        bounds = _crop_around_tracks(
            (site1_track, site2_track),
            site1_stack.shape[1:],
            minimum_size_px=int(display["raw_minimum_crop_px"]),
            padding_px=int(display["raw_padding_px"]),
        )
        cases.append(
            CaseData(
                case_id=str(item["case_id"]),
                display_label=str(item["display_label"]),
                bundle_id=bundle_id,
                nd2_id=nd2_id,
                crop_id=crop_id,
                allele_index=allele,
                hour_post_delivery=float(row["hour_post_delivery"]),
                mean_separation_nm=float(row["mean_separation_nm"]),
                sd_separation_nm=float(row["sd_separation_nm"]),
                pixel_size_nm=pixel_size_nm,
                times_s=times_s,
                site1_stack=site1_stack,
                site2_stack=site2_stack,
                nucleus_mask_stack=nucleus_mask_stack,
                nucleus_mask_path=nucleus_mask_path,
                site1_track=site1_track,
                site2_track=site2_track,
                raw_bounds=bounds,
                site1_limits=_fixed_limits(site1_stack, bounds),
                site2_limits=_fixed_limits(site2_stack, bounds),
                spatial_limits_um=_case_spatial_limits(site1_track, site2_track),
                trace_limit_nm=_source_trace_limit(site1_track, site2_track),
                source_paths=source_paths,
            )
        )
    return cases, config


def _limits(stack: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    finite = stack[np.isfinite(stack)].astype(np.float32, copy=False)
    if not finite.size:
        raise ValueError("Cannot display an image stack with no finite pixels")
    low, high = np.percentile(finite, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    return float(low), float(high)


def _normalize(image: np.ndarray, limits: tuple[float, float], gamma: float) -> np.ndarray:
    denominator = limits[1] - limits[0]
    if denominator <= 0:
        raise ValueError("Display limits must increase")
    scaled = np.clip((image.astype(float) - limits[0]) / denominator, 0, 1)
    return scaled**gamma


def _single_channel_rgb(
    image: np.ndarray,
    limits: tuple[float, float],
    gamma: float,
    color: str,
) -> np.ndarray:
    return np.clip(
        _normalize(image, limits, gamma)[..., None] * np.asarray(to_rgb(color)),
        0,
        1,
    )


def _middle_frame(case: CaseData) -> int:
    return case.n_frames // 2


def _case_display_limits(
    case: CaseData,
    green_stack: np.ndarray,
    display: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    site_percentiles = tuple(float(value) for value in display["site_percentiles"])
    bp1_percentiles = tuple(float(value) for value in display["bp1_percentiles"])
    bounds = case.raw_bounds
    return {
        "full_site1": _limits(case.site1_stack, site_percentiles),
        "full_site2": _limits(case.site2_stack, site_percentiles),
        "full_bp1": _limits(green_stack, bp1_percentiles),
        "crop_site1": _limits(
            case.site1_stack[:, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1],
            site_percentiles,
        ),
        "crop_site2": _limits(
            case.site2_stack[:, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1],
            site_percentiles,
        ),
    }


def _full_cell_composite(
    case: CaseData,
    green_stack: np.ndarray,
    display: dict[str, Any],
    limits: dict[str, tuple[float, float]],
    frame_index: int,
) -> np.ndarray:
    if green_stack.shape != case.site1_stack.shape or green_stack.shape != case.site2_stack.shape:
        raise ValueError(f"Full-cell channel shape mismatch for {case.bundle_id}")
    site1 = _single_channel_rgb(
        case.site1_stack[frame_index],
        limits["full_site1"],
        float(display["site_gamma"]),
        str(display["site1_color"]),
    )
    site2 = _single_channel_rgb(
        case.site2_stack[frame_index],
        limits["full_site2"],
        float(display["site_gamma"]),
        str(display["site2_color"]),
    )
    bp1 = _single_channel_rgb(
        green_stack[frame_index],
        limits["full_bp1"],
        float(display["bp1_gamma"]),
        str(display["bp1_color"]),
    )
    return np.clip(
        float(display["site1_weight"]) * site1
        + float(display["site2_weight"]) * site2
        + float(display["bp1_weight"]) * bp1,
        0,
        1,
    )


def _draw_pixel_track(
    axis: mpl.axes.Axes,
    track: pd.DataFrame,
    color: str,
    display: dict[str, Any],
    visible_count: int,
) -> None:
    visible = track.iloc[:visible_count]
    x = visible["x_px"].to_numpy(float)
    y = visible["y_px"].to_numpy(float)
    axis.plot(
        x,
        y,
        color="white",
        linewidth=float(display["track_halo_linewidth"]),
        alpha=0.82,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=8,
    )
    axis.plot(
        x,
        y,
        color=color,
        linewidth=float(display["track_linewidth"]),
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=9,
    )
    axis.scatter(
        [x[0]],
        [y[0]],
        s=35,
        facecolor="white",
        edgecolor=color,
        linewidth=1.7,
        zorder=10,
    )
    axis.scatter(
        [x[-1]],
        [y[-1]],
        s=42,
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=11,
    )


def _add_pixel_scale_bar(
    axis: mpl.axes.Axes,
    bounds: CropBounds,
    pixel_size_nm: float,
    length_um: float,
    display: dict[str, Any],
) -> None:
    """Draw a labeled, high-contrast physical scale bar on a pixel panel."""

    length_px = length_um * 1000.0 / pixel_size_nm
    x0 = bounds.x0 + 0.08 * bounds.width
    y = bounds.y1 - 0.10 * bounds.height
    halo_width = float(display.get("pixel_scale_bar_halo_linewidth", 8.0))
    line_width = float(display.get("pixel_scale_bar_linewidth", 4.5))
    axis.plot(
        [x0, x0 + length_px],
        [y, y],
        color="black",
        linewidth=halo_width,
        solid_capstyle="butt",
        alpha=0.82,
        zorder=19,
    )
    axis.plot(
        [x0, x0 + length_px],
        [y, y],
        color="white",
        linewidth=line_width,
        solid_capstyle="butt",
        zorder=20,
    )
    label = f"{length_um:g} µm"
    text = axis.text(
        x0 + 0.5 * length_px,
        y - 0.025 * bounds.height,
        label,
        color="white",
        fontsize=float(display.get("pixel_scale_bar_fontsize", 17.0)),
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=21,
    )
    text.set_path_effects(
        [
            patheffects.Stroke(
                linewidth=float(display.get("pixel_label_halo_linewidth", 3.2)),
                foreground="black",
            ),
            patheffects.Normal(),
        ]
    )


def _add_full_cell_time_label(
    axis: mpl.axes.Axes,
    case: CaseData,
    frame_index: int,
    display: dict[str, Any],
) -> None:
    """Annotate the exact acquisition time and current frame in the full-cell panel."""

    label = (
        f"t = {case.times_s[frame_index]:.1f} s\n"
        f"frame {frame_index + 1}/{case.n_frames}"
    )
    text = axis.text(
        0.965,
        0.965,
        label,
        transform=axis.transAxes,
        color="white",
        fontsize=float(display.get("full_cell_time_fontsize", 17.0)),
        fontweight="bold",
        ha="right",
        va="top",
        linespacing=1.15,
        zorder=30,
    )
    text.set_path_effects(
        [
            patheffects.Stroke(
                linewidth=float(display.get("pixel_label_halo_linewidth", 3.2)),
                foreground="black",
            ),
            patheffects.Normal(),
        ]
    )


def _show_full_cell(
    axis: mpl.axes.Axes,
    case: CaseData,
    green_stack: np.ndarray,
    display: dict[str, Any],
    limits: dict[str, tuple[float, float]],
    frame_index: int,
    visible_count: int,
) -> None:
    image = _full_cell_composite(case, green_stack, display, limits, frame_index)
    height, width = image.shape[:2]
    axis.imshow(image, origin="upper", interpolation="nearest")
    mask = case.nucleus_mask_stack[frame_index].astype(float, copy=False)
    axis.contour(
        np.arange(width),
        np.arange(height),
        mask,
        levels=[0.5],
        colors=[str(display.get("nucleus_outline_color", "#66D9EF"))],
        linewidths=float(display.get("nucleus_outline_linewidth", 1.8)),
        alpha=float(display.get("nucleus_outline_alpha", 0.95)),
        zorder=8,
    )
    _draw_pixel_track(
        axis,
        case.site1_track,
        str(display["site1_color"]),
        display,
        visible_count,
    )
    _draw_pixel_track(
        axis,
        case.site2_track,
        str(display["site2_color"]),
        display,
        visible_count,
    )
    full_bounds = CropBounds(x0=0, x1=width, y0=0, y1=height)
    _add_pixel_scale_bar(axis, full_bounds, case.pixel_size_nm, 1.0, display)
    _add_full_cell_time_label(axis, case, frame_index, display)
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title("Full cell", fontsize=17, fontweight="bold", pad=7)


def _show_site_crop(
    axis: mpl.axes.Axes,
    case: CaseData,
    display: dict[str, Any],
    *,
    site: str,
    limits: dict[str, tuple[float, float]],
    frame_index: int,
    visible_count: int,
) -> None:
    if site == "site1":
        stack = case.site1_stack
        track = case.site1_track
        color = str(display["site1_color"])
        title = "Site 1"
    elif site == "site2":
        stack = case.site2_stack
        track = case.site2_track
        color = str(display["site2_color"])
        title = "Site 2"
    else:
        raise ValueError(site)
    bounds = case.raw_bounds
    frame = stack[frame_index, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
    rgb = _single_channel_rgb(
        frame,
        limits[f"crop_{site}"],
        float(display["site_gamma"]),
        color,
    )
    axis.imshow(
        rgb,
        origin="upper",
        interpolation="nearest",
        extent=bounds.extent,
    )
    _draw_pixel_track(axis, track, color, display, visible_count)
    _add_pixel_scale_bar(axis, bounds, case.pixel_size_nm, 0.5, display)
    axis.set_xlim(bounds.x0 - 0.5, bounds.x1 - 0.5)
    axis.set_ylim(bounds.y1 - 0.5, bounds.y0 - 0.5)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(title, color=color, fontsize=17, fontweight="bold", pad=7)


def _reference_shifted_nm(case: CaseData, track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Apply the source MATLAB Xpos/Ypos = position - joint minimum + 50 rule."""

    minimum_x = min(
        float(case.site1_track["x_nm"].min()),
        float(case.site2_track["x_nm"].min()),
    )
    minimum_y = min(
        float(case.site1_track["y_nm"].min()),
        float(case.site2_track["y_nm"].min()),
    )
    return (
        track["x_nm"].to_numpy(float) - minimum_x + 50.0,
        track["y_nm"].to_numpy(float) - minimum_y + 50.0,
    )


def _reference_axis_limit_nm(case: CaseData) -> float:
    x1, y1 = _reference_shifted_nm(case, case.site1_track)
    x2, y2 = _reference_shifted_nm(case, case.site2_track)
    required = max(float(np.max(x1)), float(np.max(x2)), float(np.max(y1)), float(np.max(y2)))
    return max(600.0, 200.0 * math.ceil((required + 50.0) / 200.0))


def _add_time_path(
    axis: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    times_s: np.ndarray,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    site_color: str,
    marker: str,
) -> None:
    points = np.column_stack([x, y])
    if len(points) > 1:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        halo = LineCollection(
            segments,
            colors=site_color,
            linewidths=5.2,
            alpha=0.72,
            capstyle="round",
            joinstyle="round",
            zorder=4,
        )
        axis.add_collection(halo)
        time_path = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=2.5,
            capstyle="round",
            joinstyle="round",
            zorder=5,
        )
        time_path.set_array(0.5 * (times_s[:-1] + times_s[1:]))
        axis.add_collection(time_path)
    point_colors = cmap(norm(times_s))
    axis.scatter(
        x,
        y,
        s=13,
        c=point_colors,
        edgecolors="none",
        zorder=6,
    )
    axis.scatter(
        [x[0]],
        [y[0]],
        marker=marker,
        s=38,
        facecolor="white",
        edgecolor=site_color,
        linewidth=1.7,
        zorder=7,
    )
    axis.scatter(
        [x[-1]],
        [y[-1]],
        marker=marker,
        s=44,
        facecolor=site_color,
        edgecolor="white",
        linewidth=1.0,
        zorder=8,
    )


def _show_joint_trajectory(
    axis: mpl.axes.Axes,
    colorbar_axis: mpl.axes.Axes,
    case: CaseData,
    display: dict[str, Any],
    visible_count: int,
    physical_axis_limit_nm: float,
) -> None:
    cmap = mpl.colormaps[str(display["time_colormap"])]
    norm = mpl.colors.Normalize(vmin=float(case.times_s[0]), vmax=float(case.times_s[-1]))
    x1, y1 = _reference_shifted_nm(case, case.site1_track)
    x2, y2 = _reference_shifted_nm(case, case.site2_track)
    _add_time_path(
        axis,
        x1[:visible_count],
        y1[:visible_count],
        case.times_s[:visible_count],
        norm,
        cmap,
        str(display["site1_color"]),
        "o",
    )
    _add_time_path(
        axis,
        x2[:visible_count],
        y2[:visible_count],
        case.times_s[:visible_count],
        norm,
        cmap,
        str(display["site2_color"]),
        "s",
    )
    required_axis_limit = _reference_axis_limit_nm(case)
    if physical_axis_limit_nm < required_axis_limit:
        raise ValueError(
            f"Shared physical axis {physical_axis_limit_nm:g} nm clips {case.bundle_id}; "
            f"at least {required_axis_limit:g} nm is required"
        )
    axis.set_xlim(0.0, physical_axis_limit_nm)
    axis.set_ylim(0.0, physical_axis_limit_nm)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (nm)", labelpad=2, fontsize=14)
    axis.set_ylabel("y (nm)", labelpad=2, fontsize=14)
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=6, width=2.0, labelsize=12)
    axis.text(
        0.02,
        1.04,
        f"Allele {case.allele_index}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )
    axis.text(
        0.48,
        1.04,
        "Site 1 ○",
        transform=axis.transAxes,
        color=str(display["site1_color"]),
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
    )
    axis.text(
        0.83,
        1.04,
        "Site 2 □",
        transform=axis.transAxes,
        color=str(display["site2_color"]),
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
    )
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = axis.figure.colorbar(scalar, cax=colorbar_axis)
    indices = np.rint(np.linspace(0, case.n_frames - 1, 5)).astype(int)
    ticks = case.times_s[indices]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(
        [f"{value:.1f}".rstrip("0").rstrip(".") for value in ticks]
    )
    colorbar.ax.set_title("Time (s)", fontsize=12, pad=8)
    colorbar.ax.tick_params(labelsize=11, width=1.5, length=4)
    colorbar.outline.set_linewidth(1.2)


def _median_centered_nm(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = track["x_nm"].to_numpy(float)
    y = -track["y_nm"].to_numpy(float)
    return x - np.median(x), y - np.median(y)


def _trace_limit(case: CaseData) -> float:
    arrays: list[np.ndarray] = []
    for track in (case.site1_track, case.site2_track):
        x, y = _median_centered_nm(track)
        arrays.extend([x, y])
    maximum = max(float(np.max(np.abs(values))) for values in arrays)
    target = max(100.0, 1.02 * maximum)
    step = 50.0 if target <= 500 else 100.0
    return float(step * math.ceil(target / step))


def _shared_trace_limit_nm(cases: Iterable[CaseData]) -> float:
    """Return one symmetric position limit large enough for both X/Y traces in every case."""

    limits = [_trace_limit(case) for case in cases]
    if not limits:
        raise ValueError("At least one case is required to define a shared position scale")
    return float(max(limits))


def _show_trace(
    axis: mpl.axes.Axes,
    case: CaseData,
    display: dict[str, Any],
    *,
    coordinate: str,
    visible_count: int,
    trace_axis_limit_nm: float,
) -> None:
    x1, y1 = _median_centered_nm(case.site1_track)
    x2, y2 = _median_centered_nm(case.site2_track)
    first = x1 if coordinate == "x" else y1
    second = x2 if coordinate == "x" else y2
    axis.plot(
        case.times_s[:visible_count],
        first[:visible_count],
        color=str(display["site1_color"]),
        linewidth=2.3,
        label="Site 1",
    )
    axis.plot(
        case.times_s[:visible_count],
        second[:visible_count],
        color=str(display["site2_color"]),
        linewidth=2.3,
        label="Site 2",
    )
    required_limit = _trace_limit(case)
    if trace_axis_limit_nm < required_limit:
        raise ValueError(
            f"Shared position limit {trace_axis_limit_nm:g} nm clips {case.bundle_id}; "
            f"at least {required_limit:g} nm is required"
        )
    axis.set_xlim(float(case.times_s[0]), float(case.times_s[-1]))
    axis.set_ylim(-trace_axis_limit_nm, trace_axis_limit_nm)
    axis.set_xlabel("Time (s)", labelpad=2, fontsize=14)
    axis.set_ylabel(f"{coordinate} (nm)", labelpad=2, fontsize=14)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=6, width=2.0, labelsize=12)
    axis.text(
        0.03,
        0.96,
        coordinate.upper(),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    if coordinate == "x":
        legend = axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
            handlelength=1.2,
            columnspacing=1.0,
            borderaxespad=0,
            fontsize=11,
        )
        for text, color in zip(
            legend.get_texts(),
            (str(display["site1_color"]), str(display["site2_color"])),
            strict=True,
        ):
            text.set_color(color)


def make_multichannel_case_figure(
    case: CaseData,
    green_stack: np.ndarray,
    display: dict[str, Any],
    *,
    frame_index: int | None = None,
    visible_count: int | None = None,
    limits: dict[str, tuple[float, float]] | None = None,
    physical_axis_limit_nm: float | None = None,
    trace_axis_limit_nm: float | None = None,
) -> mpl.figure.Figure:
    """Create the final five-panel manual-review figure for one selected case."""

    if frame_index is None:
        frame_index = _middle_frame(case)
    if visible_count is None:
        visible_count = case.n_frames
    if not 0 <= frame_index < case.n_frames:
        raise ValueError(f"Frame index out of range: {frame_index}")
    if not 1 <= visible_count <= case.n_frames:
        raise ValueError(f"Visible count out of range: {visible_count}")
    if limits is None:
        limits = _case_display_limits(case, green_stack, display)
    if physical_axis_limit_nm is None:
        physical_axis_limit_nm = _reference_axis_limit_nm(case)
    if trace_axis_limit_nm is None:
        trace_axis_limit_nm = _trace_limit(case)
    figure = plt.figure(
        figsize=tuple(float(value) for value in display["figure_size_inches"]),
        facecolor="white",
    )
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.82, 1.0, 1.0, 1.42),
        height_ratios=(1.02, 0.98),
        left=0.025,
        right=0.958,
        bottom=0.125,
        top=0.93,
        wspace=0.27,
        hspace=0.36,
    )
    full_axis = figure.add_subplot(grid[:, 0])
    site1_axis = figure.add_subplot(grid[0, 1])
    site2_axis = figure.add_subplot(grid[0, 2])
    x_axis = figure.add_subplot(grid[1, 1])
    y_axis = figure.add_subplot(grid[1, 2])
    trajectory_grid = grid[:, 3].subgridspec(1, 2, width_ratios=(1.0, 0.065), wspace=0.11)
    trajectory_axis = figure.add_subplot(trajectory_grid[0, 0])
    colorbar_axis = figure.add_subplot(trajectory_grid[0, 1])
    _show_full_cell(
        full_axis,
        case,
        green_stack,
        display,
        limits,
        frame_index,
        visible_count,
    )
    _show_site_crop(
        site1_axis,
        case,
        display,
        site="site1",
        limits=limits,
        frame_index=frame_index,
        visible_count=visible_count,
    )
    _show_site_crop(
        site2_axis,
        case,
        display,
        site="site2",
        limits=limits,
        frame_index=frame_index,
        visible_count=visible_count,
    )
    _show_trace(
        x_axis,
        case,
        display,
        coordinate="x",
        visible_count=visible_count,
        trace_axis_limit_nm=trace_axis_limit_nm,
    )
    _show_trace(
        y_axis,
        case,
        display,
        coordinate="y",
        visible_count=visible_count,
        trace_axis_limit_nm=trace_axis_limit_nm,
    )
    _show_joint_trajectory(
        trajectory_axis,
        colorbar_axis,
        case,
        display,
        visible_count,
        physical_axis_limit_nm,
    )
    return figure


def _shared_reference_axis_limit_nm(cases: Iterable[CaseData]) -> float:
    """Return one joint-origin physical scale that contains every selected case."""

    limits = [_reference_axis_limit_nm(case) for case in cases]
    if not limits:
        raise ValueError("At least one case is required to define a shared physical scale")
    return float(max(limits))


def _show_joint_origin_trace(
    axis: mpl.axes.Axes,
    case: CaseData,
    display: dict[str, Any],
    *,
    coordinate: str,
    visible_count: int,
    shared_axis_limit_nm: float,
) -> None:
    """Show uncentered Site1/Site2 positions after one shared translation per case."""

    x1, y1 = _reference_shifted_nm(case, case.site1_track)
    x2, y2 = _reference_shifted_nm(case, case.site2_track)
    first = x1 if coordinate == "x" else y1
    second = x2 if coordinate == "x" else y2
    required = max(float(np.max(first)), float(np.max(second)))
    if shared_axis_limit_nm < required:
        raise ValueError(
            f"Shared joint-origin position limit {shared_axis_limit_nm:g} nm clips "
            f"{case.bundle_id}; at least {required:g} nm is required"
        )
    axis.plot(
        case.times_s[:visible_count],
        first[:visible_count],
        color=str(display["site1_color"]),
        linewidth=2.3,
        label="Site 1",
    )
    axis.plot(
        case.times_s[:visible_count],
        second[:visible_count],
        color=str(display["site2_color"]),
        linewidth=2.3,
        label="Site 2",
    )
    axis.set_xlim(float(case.times_s[0]), float(case.times_s[-1]))
    axis.set_ylim(0.0, shared_axis_limit_nm)
    axis.set_xlabel("Time (s)", labelpad=2, fontsize=14)
    axis.set_ylabel(f"{coordinate} (nm)", labelpad=2, fontsize=14)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=6, width=2.0, labelsize=12)
    axis.text(
        0.03,
        0.96,
        coordinate.upper(),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    if coordinate == "x":
        legend = axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
            handlelength=1.2,
            columnspacing=1.0,
            borderaxespad=0,
            fontsize=11,
        )
        for text, color in zip(
            legend.get_texts(),
            (str(display["site1_color"]), str(display["site2_color"])),
            strict=True,
        ):
            text.set_color(color)


def make_distance_demo_variant_figure(
    case: CaseData,
    green_stack: np.ndarray,
    display: dict[str, Any],
    *,
    variant: str,
    frame_index: int | None = None,
    visible_count: int | None = None,
    limits: dict[str, tuple[float, float]] | None = None,
    shared_axis_limit_nm: float | None = None,
) -> mpl.figure.Figure:
    """Create one of the two final distance-demo layouts."""

    supported = {"compact_no_xy", "shared_joint_origin_xy"}
    if variant not in supported:
        raise ValueError(f"Unsupported distance-demo variant: {variant}")
    if frame_index is None:
        frame_index = _middle_frame(case)
    if visible_count is None:
        visible_count = case.n_frames
    if not 0 <= frame_index < case.n_frames:
        raise ValueError(f"Frame index out of range: {frame_index}")
    if not 1 <= visible_count <= case.n_frames:
        raise ValueError(f"Visible count out of range: {visible_count}")
    if limits is None:
        limits = _case_display_limits(case, green_stack, display)
    if shared_axis_limit_nm is None:
        shared_axis_limit_nm = _reference_axis_limit_nm(case)

    if variant == "compact_no_xy":
        figure = plt.figure(figsize=(13.8, 6.4), facecolor="white")
        grid = figure.add_gridspec(
            2,
            4,
            width_ratios=(1.72, 0.96, 1.48, 0.055),
            height_ratios=(1.0, 1.0),
            left=0.025,
            right=0.965,
            bottom=0.09,
            top=0.88,
            wspace=0.27,
            hspace=0.24,
        )
        full_axis = figure.add_subplot(grid[:, 0])
        site1_axis = figure.add_subplot(grid[0, 1])
        site2_axis = figure.add_subplot(grid[1, 1])
        trajectory_axis = figure.add_subplot(grid[:, 2])
        colorbar_axis = figure.add_subplot(grid[:, 3])
        physical_axis_limit_nm = _reference_axis_limit_nm(case)
    else:
        figure = plt.figure(
            figsize=tuple(float(value) for value in display["figure_size_inches"]),
            facecolor="white",
        )
        grid = figure.add_gridspec(
            2,
            4,
            width_ratios=(1.82, 1.0, 1.0, 1.42),
            height_ratios=(1.02, 0.98),
            left=0.025,
            right=0.958,
            bottom=0.125,
            top=0.89,
            wspace=0.27,
            hspace=0.36,
        )
        full_axis = figure.add_subplot(grid[:, 0])
        site1_axis = figure.add_subplot(grid[0, 1])
        site2_axis = figure.add_subplot(grid[0, 2])
        x_axis = figure.add_subplot(grid[1, 1])
        y_axis = figure.add_subplot(grid[1, 2])
        trajectory_grid = grid[:, 3].subgridspec(
            1, 2, width_ratios=(1.0, 0.065), wspace=0.11
        )
        trajectory_axis = figure.add_subplot(trajectory_grid[0, 0])
        colorbar_axis = figure.add_subplot(trajectory_grid[0, 1])
        physical_axis_limit_nm = shared_axis_limit_nm

    _show_full_cell(
        full_axis,
        case,
        green_stack,
        display,
        limits,
        frame_index,
        visible_count,
    )
    _show_site_crop(
        site1_axis,
        case,
        display,
        site="site1",
        limits=limits,
        frame_index=frame_index,
        visible_count=visible_count,
    )
    _show_site_crop(
        site2_axis,
        case,
        display,
        site="site2",
        limits=limits,
        frame_index=frame_index,
        visible_count=visible_count,
    )
    if variant == "shared_joint_origin_xy":
        _show_joint_origin_trace(
            x_axis,
            case,
            display,
            coordinate="x",
            visible_count=visible_count,
            shared_axis_limit_nm=shared_axis_limit_nm,
        )
        _show_joint_origin_trace(
            y_axis,
            case,
            display,
            coordinate="y",
            visible_count=visible_count,
            shared_axis_limit_nm=shared_axis_limit_nm,
        )
    _show_joint_trajectory(
        trajectory_axis,
        colorbar_axis,
        case,
        display,
        visible_count,
        physical_axis_limit_nm,
    )
    return figure


def _make_distance_demo_variant_video_renderer(
    case: CaseData,
    green_stack: np.ndarray,
    display: dict[str, Any],
    limits: dict[str, tuple[float, float]],
    *,
    variant: str,
    shared_axis_limit_nm: float,
) -> Callable[[int], np.ndarray]:
    def render_frame(frame_index: int) -> np.ndarray:
        dynamic = make_distance_demo_variant_figure(
            case,
            green_stack,
            display,
            variant=variant,
            frame_index=frame_index,
            visible_count=frame_index + 1,
            limits=limits,
            shared_axis_limit_nm=shared_axis_limit_nm,
        )
        rgb = _figure_to_rgb(dynamic)
        plt.close(dynamic)
        even_height = rgb.shape[0] - rgb.shape[0] % 2
        even_width = rgb.shape[1] - rgb.shape[1] % 2
        return np.ascontiguousarray(rgb[:even_height, :even_width])

    return render_frame


def _make_video_renderer(
    case: CaseData,
    green_stack: np.ndarray,
    display: dict[str, Any],
    limits: dict[str, tuple[float, float]],
    physical_axis_limit_nm: float,
    trace_axis_limit_nm: float,
) -> Callable[[int], np.ndarray]:
    def render_frame(frame_index: int) -> np.ndarray:
        dynamic = make_multichannel_case_figure(
            case,
            green_stack,
            display,
            frame_index=frame_index,
            visible_count=frame_index + 1,
            limits=limits,
            physical_axis_limit_nm=physical_axis_limit_nm,
            trace_axis_limit_nm=trace_axis_limit_nm,
        )
        rgb = _figure_to_rgb(dynamic)
        plt.close(dynamic)
        even_height = rgb.shape[0] - rgb.shape[0] % 2
        even_width = rgb.shape[1] - rgb.shape[1] % 2
        return np.ascontiguousarray(rgb[:even_height, :even_width])

    return render_frame


def write_multichannel_case_review(
    *,
    cases: list[CaseData],
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    ffmpeg_exe: Path,
    extra_source_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Write final PNG/PDF figures, source table, method contract and hash manifest."""

    output_dir = output_dir.resolve()
    figure_dir = output_dir / "figures"
    video_dir = output_dir / "videos"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=False)
    video_dir.mkdir(parents=True, exist_ok=False)
    table_dir.mkdir(parents=True, exist_ok=False)
    display = config["display"]
    physical_axis_limits_nm = {
        case.case_id: _reference_axis_limit_nm(case) for case in cases
    }
    required_trace_limit_nm = _shared_trace_limit_nm(cases)
    trace_axis_limit_nm = float(
        display.get("shared_trace_axis_limit_nm", required_trace_limit_nm)
    )
    if trace_axis_limit_nm < required_trace_limit_nm:
        raise ValueError(
            f"Configured shared position limit {trace_axis_limit_nm:g} nm is smaller than "
            f"the required {required_trace_limit_nm:g} nm"
        )
    products: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = [config_path.resolve(), *extra_source_paths]
    for case in cases:
        physical_axis_limit_nm = physical_axis_limits_nm[case.case_id]
        paths = _channel_paths(case)
        green_stack = _load_tyx(paths["green_53bp1_tiff"])
        display_limits = _case_display_limits(case, green_stack, display)
        figure = make_multichannel_case_figure(
            case,
            green_stack,
            display,
            limits=display_limits,
            physical_axis_limit_nm=physical_axis_limit_nm,
            trace_axis_limit_nm=trace_axis_limit_nm,
        )
        hour = _hour_slug(case.hour_post_delivery)
        nd2 = case.nd2_id.replace("LiveFISH ", "")
        candidate = case.crop_id.rsplit("_candidate_", maxsplit=1)[-1]
        stem = (
            f"{hour}_{case.case_id}_{nd2}_candidate{candidate}_"
            f"allele{case.allele_index}_multichannel_case_review"
        )
        png_path = figure_dir / f"{stem}.png"
        pdf_path = figure_dir / f"{stem}.pdf"
        video_path = video_dir / f"{stem}.mp4"
        figure.savefig(png_path, dpi=int(display["png_dpi"]), facecolor="white")
        figure.savefig(pdf_path, facecolor="white")
        plt.close(figure)

        video = encode_mp4(
            video_path,
            _make_video_renderer(
                case,
                green_stack,
                display,
                display_limits,
                physical_axis_limit_nm,
                trace_axis_limit_nm,
            ),
            range(case.n_frames),
            fps=float(display["playback_fps"]),
            ffmpeg_exe=ffmpeg_exe,
            crf=int(display["video_crf"]),
        )
        metadata_path = case.source_paths[2]
        metadata = _load_json(metadata_path)
        separation_nm = np.hypot(
            case.site2_track["x_nm"].to_numpy(float)
            - case.site1_track["x_nm"].to_numpy(float),
            case.site2_track["y_nm"].to_numpy(float)
            - case.site1_track["y_nm"].to_numpy(float),
        )
        distance_rows.append(
            {
                "case_id": case.case_id,
                "display_label": case.display_label,
                "hour_post_delivery": case.hour_post_delivery,
                "bundle_id": case.bundle_id,
                "shared_frames": int(separation_nm.size),
                "mean_separation_nm": float(np.mean(separation_nm)),
                "sd_separation_nm": float(np.std(separation_nm, ddof=1)),
                "median_separation_nm": float(np.median(separation_nm)),
                "minimum_separation_nm": float(np.min(separation_nm)),
                "maximum_separation_nm": float(np.max(separation_nm)),
                "physical_axis_limit_nm": physical_axis_limit_nm,
                "shared_trace_axis_limit_nm": trace_axis_limit_nm,
            }
        )
        path_rows.append(
            {
                "case_id": case.case_id,
                "display_label": case.display_label,
                "hour_post_delivery": case.hour_post_delivery,
                "bundle_id": case.bundle_id,
                "mean_separation_nm": float(np.mean(separation_nm)),
                "sd_separation_nm": float(np.std(separation_nm, ddof=1)),
                "physical_axis_limit_nm": physical_axis_limit_nm,
                "shared_trace_axis_limit_nm": trace_axis_limit_nm,
                "crop_directory": str(paths["crop_dir"]),
                "composite_tiff": str(paths["composite_tiff"]),
                "corrected_crop_tiff": str(paths["corrected_crop_tiff"]),
                "green_53bp1_tiff": str(paths["green_53bp1_tiff"]),
                "purple_site2_tiff": str(paths["purple_site2_tiff"]),
                "red_site1_tiff": str(paths["red_site1_tiff"]),
                "nucleus_scaffold_tiff": str(paths["nucleus_scaffold_tiff"]),
                "aligned_nucleus_mask_tiff": str(paths["aligned_nucleus_mask_tiff"]),
                "metadata_json": str(metadata_path),
                "frozen_source_nd2": str(metadata.get("source_nd2", "")),
                "output_png": str(png_path),
                "output_pdf": str(pdf_path),
                "output_video": str(video_path),
            }
        )
        products.append(
            {
                "case_id": case.case_id,
                "png": str(png_path),
                "pdf": str(pdf_path),
                "video": video,
            }
        )
        source_paths.extend(path for label, path in paths.items() if label != "crop_dir")
        source_paths.extend(case.source_paths[2:])
    path_table = table_dir / "selected_case_source_and_output_paths.csv"
    pd.DataFrame(path_rows).to_csv(path_table, index=False)
    distance_table = table_dir / "distance_demo_summary.csv"
    pd.DataFrame(distance_rows).to_csv(distance_table, index=False)
    contract = {
        "schema_version": 5,
        "analysis_unit": "one user-selected complete Site1/Site2 bundle trajectory",
        "layout": {
            "left": (
                "large production crop with Site1/Site2/53BP1 pseudocolor, both tracks, "
                "and the frame-aligned production micro-SAM boundary"
            ),
            "upper_middle": "separate Site1 and Site2 pixel panels with identical crop bounds",
            "lower_middle": "median-centered X and Y traces in Site1 yellow and Site2 purple",
            "right": "joint physical trajectory; site-color halo plus shared jet acquisition-time color",
        },
        "channel_mapping": {
            "red": {"identity": "Site1", "display_color": display["site1_color"]},
            "purple": {"identity": "Site2", "display_color": display["site2_color"]},
            "green": {"identity": "53BP1 context only", "display_color": display["bp1_color"]},
        },
        "green_channel_role": "visual nuclear context only; never selection, filter or outcome",
        "nucleus_boundary": {
            "source": "mask_alignment/microsam_mask_aligned_raw.tif",
            "role": "production micro-SAM instance boundary overlaid for spatial context",
            "not_used_as_new_filter": True,
            "root_nucleus_tiff_role": "all-foreground compatibility scaffold; never displayed",
        },
        "background_frame": "middle acquisition frame",
        "site_pixel_coordinate_contract": "same production pixel frame, same CropBounds and same scale",
        "track_coordinate_formula": "image_px = automatic_nm/pixel_size_nm - 1",
        "physical_trajectory": {
            "units": "nm",
            "origin": "joint Site1/Site2 minimum x/y shifted to 50 nm, matching source MATLAB",
            "y_direction": "production y_nm direction, matching source MATLAB",
            "aspect": "equal",
            "axis_limits_nm_by_case": {
                case_id: [0.0, limit] for case_id, limit in physical_axis_limits_nm.items()
            },
            "axis_scale_scope": "equal x/y aspect within each case; case-specific range",
            "time_encoding": display["time_colormap"],
            "site_identity": "yellow/purple halo and circle/square endpoint symbols",
            "colorbar": "exact relative acquisition time in seconds at five MATLAB-style ticks",
        },
        "xy_traces": {
            "centering": "each site median-centered independently",
            "time": "exact acquisition time",
            "y_direction": "positive upward",
            "shared_y_limits_nm": [-trace_axis_limit_nm, trace_axis_limit_nm],
            "shared_scale_scope": "identical for X and Y panels across both poster demo cases",
        },
        "interpolation_or_refit": "none",
        "video": {
            "dynamic_panels": "all five panels update synchronously",
            "track_history": "revealed through the current frame",
            "future_coordinates": "not shown",
            "playback_fps": display["playback_fps"],
            "playback_time_is_not_acquisition_time": True,
        },
        "filename_hour": "folder-hour after Cas9 delivery",
        "display": display,
    }
    contract_path = output_dir / "METHOD_CONTRACT.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "REPORT.md"
    report_path.write_text(
        "# Final paired-distance poster demos\n\n"
        "The two frozen cases are rendered with one large production-crop three-channel pseudocolor "
        "panel and the aligned production micro-SAM boundary, "
        "separate Site1 and Site2 raw-pixel panels on identical bounds, one joint MATLAB-style "
        "time-colored physical trajectory with a Time (s) color bar, and yellow/purple X/Y traces. "
        "The physical XY panels use case-specific equal x/y ranges (the previous display mode). "
        f"All lower X/Y position panels use the same -{trace_axis_limit_nm:g} to "
        f"{trace_axis_limit_nm:g} nm vertical limits. "
        "The 53BP1 channel provides visual context only and is not a filter or outcome. The cyan "
        "outline is the saved production micro-SAM mask; the root-level all-one Nucleus TIFF is a "
        "compatibility scaffold and is not displayed. "
        "All five panels update synchronously in the companion MP4. No localization is "
        "interpolated or re-fitted, and playback time is not acquisition time.\n",
        encoding="utf-8",
    )
    unique_sources = list(dict.fromkeys(path.resolve() for path in source_paths))
    artifact_paths = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "OUTPUT_MANIFEST.json"
    ]
    manifest = {
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in unique_sources
        ],
        "artifacts": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(artifact_paths)
        ],
        "products": products,
        "physical_axis_limits_nm": physical_axis_limits_nm,
        "shared_trace_axis_limit_nm": trace_axis_limit_nm,
    }
    manifest_path = output_dir / "OUTPUT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "products": products,
        "path_table": path_table,
        "distance_table": distance_table,
        "contract": contract_path,
        "manifest": manifest_path,
    }


def write_distance_demo_variants(
    *,
    cases: list[CaseData],
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    ffmpeg_exe: Path,
    extra_source_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Write compact and common-scale position variants for both frozen cases."""

    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {output}")
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    variants = ("compact_no_xy", "shared_joint_origin_xy")
    figure_dirs = {variant: staging / "figures" / variant for variant in variants}
    video_dirs = {variant: staging / "videos" / variant for variant in variants}
    for directory in (*figure_dirs.values(), *video_dirs.values()):
        directory.mkdir(parents=True)
    table_dir = staging / "tables"
    table_dir.mkdir()

    display = config["display"]
    shared_axis_limit_nm = _shared_reference_axis_limit_nm(cases)
    products: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = [config_path.resolve(), *extra_source_paths]
    for case in cases:
        paths = _channel_paths(case)
        green_stack = _load_tyx(paths["green_53bp1_tiff"])
        display_limits = _case_display_limits(case, green_stack, display)
        hour = _hour_slug(case.hour_post_delivery)
        nd2 = case.nd2_id.replace("LiveFISH ", "")
        candidate = case.crop_id.rsplit("_candidate_", maxsplit=1)[-1]
        base_stem = (
            f"{hour}_{case.case_id}_{nd2}_candidate{candidate}_"
            f"allele{case.allele_index}"
        )
        separation_nm = np.hypot(
            case.site2_track["x_nm"].to_numpy(float)
            - case.site1_track["x_nm"].to_numpy(float),
            case.site2_track["y_nm"].to_numpy(float)
            - case.site1_track["y_nm"].to_numpy(float),
        )
        distance_rows.append(
            {
                "case_id": case.case_id,
                "display_label": case.display_label,
                "hour_post_delivery": case.hour_post_delivery,
                "bundle_id": case.bundle_id,
                "shared_frames": int(separation_nm.size),
                "mean_separation_nm": float(np.mean(separation_nm)),
                "sd_separation_nm": float(np.std(separation_nm, ddof=1)),
                "median_separation_nm": float(np.median(separation_nm)),
                "minimum_separation_nm": float(np.min(separation_nm)),
                "maximum_separation_nm": float(np.max(separation_nm)),
                "compact_trajectory_axis_limit_nm": _reference_axis_limit_nm(case),
                "shared_joint_origin_axis_limit_nm": shared_axis_limit_nm,
            }
        )
        metadata_path = case.source_paths[2]
        metadata = _load_json(metadata_path)
        for variant in variants:
            figure = make_distance_demo_variant_figure(
                case,
                green_stack,
                display,
                variant=variant,
                limits=display_limits,
                shared_axis_limit_nm=shared_axis_limit_nm,
            )
            stem = f"{base_stem}_{variant}"
            png_path = figure_dirs[variant] / f"{stem}.png"
            pdf_path = figure_dirs[variant] / f"{stem}.pdf"
            video_path = video_dirs[variant] / f"{stem}.mp4"
            figure.savefig(png_path, dpi=int(display["png_dpi"]), facecolor="white")
            figure.savefig(pdf_path, facecolor="white")
            plt.close(figure)
            video = encode_mp4(
                video_path,
                _make_distance_demo_variant_video_renderer(
                    case,
                    green_stack,
                    display,
                    display_limits,
                    variant=variant,
                    shared_axis_limit_nm=shared_axis_limit_nm,
                ),
                range(case.n_frames),
                fps=float(display["playback_fps"]),
                ffmpeg_exe=ffmpeg_exe,
                crf=int(display["video_crf"]),
            )
            products.append(
                {
                    "case_id": case.case_id,
                    "variant": variant,
                    "png": str(output / png_path.relative_to(staging)),
                    "pdf": str(output / pdf_path.relative_to(staging)),
                    "video": {
                        **video,
                        "path": str(output / video_path.relative_to(staging)),
                    },
                }
            )
            path_rows.append(
                {
                    "case_id": case.case_id,
                    "variant": variant,
                    "hour_post_delivery": case.hour_post_delivery,
                    "bundle_id": case.bundle_id,
                    "mean_separation_nm": float(np.mean(separation_nm)),
                    "sd_separation_nm": float(np.std(separation_nm, ddof=1)),
                    "trajectory_axis_limit_nm": (
                        _reference_axis_limit_nm(case)
                        if variant == "compact_no_xy"
                        else shared_axis_limit_nm
                    ),
                    "xy_position_transform": (
                        "not_displayed"
                        if variant == "compact_no_xy"
                        else "joint_minimum_translation_plus_50nm; no median centering"
                    ),
                    "crop_directory": str(paths["crop_dir"]),
                    "green_53bp1_tiff": str(paths["green_53bp1_tiff"]),
                    "purple_site2_tiff": str(paths["purple_site2_tiff"]),
                    "red_site1_tiff": str(paths["red_site1_tiff"]),
                    "metadata_json": str(metadata_path),
                    "frozen_source_nd2": str(metadata.get("source_nd2", "")),
                    "output_png": str(output / png_path.relative_to(staging)),
                    "output_pdf": str(output / pdf_path.relative_to(staging)),
                    "output_video": str(output / video_path.relative_to(staging)),
                }
            )
        source_paths.extend(path for label, path in paths.items() if label != "crop_dir")
        source_paths.extend(case.source_paths[2:])

    path_table = table_dir / "selected_case_variant_source_and_output_paths.csv"
    pd.DataFrame(path_rows).to_csv(path_table, index=False)
    distance_table = table_dir / "distance_demo_summary.csv"
    pd.DataFrame(distance_rows).to_csv(distance_table, index=False)
    contract = {
        "schema_version": 7,
        "analysis_unit": "one user-selected complete Site1/Site2 bundle trajectory",
        "variants": {
            "compact_no_xy": {
                "panels": "Full cell, Site1 pixels, Site2 pixels, joint trajectory with time bar",
                "xy_time_traces": "omitted",
                "joint_trajectory_axis": "case-specific equal x/y scale for legibility",
            },
            "shared_joint_origin_xy": {
                "panels": (
                    "Full cell, Site1 pixels, Site2 pixels, X position, Y position, "
                    "joint trajectory with time bar"
                ),
                "position_transform": (
                    "one translation shared by Site1 and Site2: coordinate minus joint minimum "
                    "plus 50 nm; no site-wise median centering"
                ),
                "shared_xy_position_limits_nm": [0.0, shared_axis_limit_nm],
                "shared_joint_trajectory_limits_nm": [0.0, shared_axis_limit_nm],
                "scale_scope": "identical X, Y and joint-trajectory axes across both cases",
            },
        },
        "channel_mapping": {
            "red": {"identity": "Site1", "display_color": display["site1_color"]},
            "purple": {"identity": "Site2", "display_color": display["site2_color"]},
            "green": {"identity": "53BP1 context only", "display_color": display["bp1_color"]},
        },
        "green_channel_role": "visual nuclear context only; never selection, filter or outcome",
        "track_coordinate_formula": "image_px = automatic_nm/pixel_size_nm - 1",
        "interpolation_or_refit": "none",
        "background_frame": "middle acquisition frame",
        "pixel_scale_bars": {
            "full_cell": "1 µm",
            "site1": "0.5 µm",
            "site2": "0.5 µm",
            "labels_and_high_contrast_halo": True,
        },
        "video": {
            "all_visible_panels_update_synchronously": True,
            "track_history": "revealed through current frame",
            "future_coordinates": "not shown",
            "full_cell_dynamic_label": "exact relative acquisition time and frame i/N",
            "playback_fps": display["playback_fps"],
            "playback_time_is_not_acquisition_time": True,
        },
        "display": display,
    }
    contract_path = staging / "METHOD_CONTRACT.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report_path = staging / "REPORT.md"
    report_path.write_text(
        "# Final paired-distance visualization variants\n\n"
        "Both frozen trajectories are shown in two static and dynamic layouts. The compact layout "
        "omits X/Y time traces and enlarges the full-cell, channel and joint-trajectory panels. "
        "The common-scale layout retains X/Y time traces without independent median centering: "
        "one joint translation is applied to both sites, preserving their separation. Its X, Y "
        f"and joint-trajectory axes are all fixed to 0--{shared_axis_limit_nm:g} nm across both "
        "cases. Full-cell panels carry an exact dynamic acquisition-time/frame label; all pixel "
        "panels carry labeled, high-contrast physical scale bars. No localization is interpolated "
        "or re-fitted.\n",
        encoding="utf-8",
    )
    unique_sources = list(dict.fromkeys(path.resolve() for path in source_paths))
    for source in unique_sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    artifact_paths = sorted(
        path for path in staging.rglob("*") if path.is_file() and path.name != "OUTPUT_MANIFEST.json"
    )
    manifest_path = staging / "OUTPUT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                    for path in unique_sources
                ],
                "artifacts": [
                    {
                        "path": str(path.relative_to(staging)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in artifact_paths
                ],
                "products": products,
                "shared_axis_limit_nm": shared_axis_limit_nm,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    staging.rename(output)
    return {
        "products": products,
        "path_table": output / path_table.relative_to(staging),
        "distance_table": output / distance_table.relative_to(staging),
        "contract": output / contract_path.relative_to(staging),
        "manifest": output / manifest_path.relative_to(staging),
    }
