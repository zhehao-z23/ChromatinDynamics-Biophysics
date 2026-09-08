# Oligo-LiveFISH dynamic trajectory visualizations

This folder generates three reusable H.264 MP4 demos from a completed pipeline run:

1. `01_multichannel_longest_tracking.mp4` — the longest selected trajectory in each channel, over the corrected raw movie, plus displacement over acquisition time.
2. `02_site2_time_colored_tracking.mp4` — the longest Site 2 trajectory on the corrected movie and as a MATLAB-like time-colored path.
3. `03_connected_region_centroid_tracking.mp4` — the production reference detector's per-frame connected region and intensity-weighted centroid.

The supplied FOV7/cell00 configuration uses the finalized v5.1 corrected movie and coordinate contract. It deliberately does **not** overlay trajectories on the earlier 288×276 uncorrected crop.

## Rebuild the demos

```powershell
python -m pip install -r .\trajectory_visualization_scripts\requirements.txt
python .\trajectory_visualization_scripts\build_demo_videos.py
```

Build only one product or change playback speed:

```powershell
python .\trajectory_visualization_scripts\build_demo_videos.py --only connected-region --fps 8
```

Use another completed run by copying `configs/fov7_cell00_demo.json` and changing the five source paths. The baseline manifest is used to select the longest baseline independently in G, R, and P; the connected-region demo follows the reference locus belonging to the longest P baseline by default.

## Coordinate and time contracts

- Image: corrected TCYX array; zero-based pixel centres; x is column, y is row; upper-left origin.
- Automatic CSV: whole-corrected-image one-based pixel centres in nm.
- Overlay: `x_px = x_nm / pixel_size_nm - 1`, and likewise for y.
- Frame: CSV is one-based; array index is `frame - 1`.
- Time: exact `relative_time_s` from the crop sidecar is used when available; MP4 fps controls playback only.
- Scale bars: calculated from `pixel_size_nm`, never hard-coded in display pixels.

The connected-region reconstruction mirrors the production reference tracker: each corrected Site 2 frame is thresholded at `mean(valid) + 0.5 × SD(valid)`, connected components are labelled globally, intensity-weighted centroids are calculated, and the component nearest the fixed time-average seed within 30 px is selected. The generated reconstruction table and manifest quantify agreement with the archived reference trajectory.

## Outputs and provenance

Outputs are written to `demo_outputs/fov7_cell00/`. `demo_manifest.json` records selections, coordinate rules, encoder settings, input hashes, and connected-centroid reconstruction error. Poster PNGs are supplied for rapid review and use as video cover images.
