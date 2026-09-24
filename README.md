# ChromatinDynamics-Biophysics

Compute physical observables, statistical support, QC figures, and multichannel dynamic reviews of chromatin motion from **trajectories and exact timing produced by CrisprTrack2**. Defaults follow the formal v5.2.1 inputs (extraction commit `9f4e28a7bdaa6847e876fa9e3de059de197d0441`) and the final physics analysis working tree completed by August 26, 2026.

Image segmentation and trajectory extraction belong to CrisprTrack2. Fingerprinting, clustering, and supervised learning belong to `ChromatinDynamics-ML`, which can consume the tables generated here or install this package to import physical kernels such as `dsb_states.physical_metrics`.

| Repository | Start → finish | Shared interface |
|---|---|---|
| [CrisprTrack2](https://github.com/zhehao-z23/CrisprTrack2) | ND2/TIFF → nucleus segmentation → corrected trajectories | Formal run root, `frame,x_nm,y_nm`, exact timing, and provenance metadata |
| ChromatinDynamics-Biophysics | Formal trajectories → intake/cache → MSD/MSCD/VAC/VCC, QC, image/video review | Parquet/CSV physics tables and bundle/crop/acquisition identifiers |
| [ChromatinDynamics-ML](https://github.com/zhehao-z23/ChromatinDynamics-ML) | Shared frozen intake/physics tables → fingerprints → unsupervised and supervised learning | Join by identifiers; extraction and physics kernels are not duplicated |

## Version and final analysis scope

At packaging time, the original analysis repository's Git HEAD was still the older `d3bd4ab`; final v5.2.1 files were present as uncommitted working-tree changes. This repository preserves **that working tree with per-file SHA256 records**. See [source provenance](provenance/SOURCE_MANIFEST.json), [packaging changes](docs/PACKAGING.md), and the [frozen-run index](docs/FROZEN_RESULTS.md).

Included workflows cover per-hour MSD/MSCD v2, VAC v2 at multiple velocity windows, the full bidirectional VCC tensor and Rouse communication-time v4, final MSD/separation figures, complete-coverage case selection, multichannel dynamic review, 3 h high/low-alpha and highest-MSCD cases, and final presentation figures. `provenance/frozen_runs` contains historical evidence; its absolute paths must be mapped to your local data explicitly.

## Installation

Python 3.11+ is required. Use this repository's minimal dependencies for a new environment; the original analysis environment also included unrelated ML methods.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
```

On Linux/macOS, use `.venv/bin/python`. Numerical Rouse reference tables are included in `references/rouse/tables`; videos additionally require FFmpeg. See [NOTICE](NOTICE.md) for reference-code attribution.

## Required inputs

The three supported starting points have distinct requirements:

1. **Complete formal run root:** extraction outputs, exact per-frame timing, crop metadata, trajectory manifests, and 53BP1 sidecars, from which the full intake can be built. Dynamic visualization also requires corrected TIFFs and nucleus masks.
2. **Frozen intake:** `table_registry.json` and its registered Parquet datasets. The frozen identifier is `v5_2_1_formal_70909c2e6326`; this input can be used directly to build a cache.
3. **Frozen flat cache:** `CACHE_CONTRACT.json`, `tables/bundle_frame_master.parquet`, `tables/bundle_index.parquet`, and associated tables. This input can be used directly for physics analysis.

A CSV archive containing only `frame,x_nm,y_nm` allows trajectory reuse but cannot independently recover exact timing, scientific asset status, 53BP1 measurements, or image/video content. Assigning an arbitrary frame interval does not reproduce the formal input contract. See the [data dictionary](docs/INTAKE_DATA_DICTIONARY.md) for column definitions.

Write every analysis to a **new output directory**. Treat the source run root and frozen intake/cache as read-only. Choose a new output name for a new run. Example paths:

```powershell
$cache = 'F:\DSB_v51\path_to_frozen_cache'
$run = 'C:\analysis_results\physics_20260908'
```

Replace these example paths with your actual archive locations. Original source locations are documented in [data retention and path migration](docs/DATA_AND_STORAGE.md).
The driver requires `output-root`, `cache`, and optional `snapshot` paths to be mutually non-overlapping, preventing outputs from being written inside frozen data.

## Build intake and cache from a formal run root

Copy `config/v521_intake.yaml` to `config/v521_intake.local.yaml` and edit `source.root`, `output.parent`, and required metadata paths. Keep the extraction version, commit, exact-timing requirements, and field schema unchanged unless deliberately defining a new data version.

```powershell
& $py scripts/20_build_v521_intake.py --config config/v521_intake.local.yaml
# The preceding command prints the generated content-addressed snapshot path.
& $py scripts/21_build_v521_unfiltered_cache.py `
  --snapshot 'C:\analysis_data\v5_2_1_formal_70909c2e6326' `
  --output-dir 'C:\analysis_data\v5_2_1_unfiltered_cache_70909c2e6326'
```

The cache applies neither T1–T4 nor 53BP1-positive filtering and does not fill missing positions. Global 53BP1 objects are stored in a separate one-to-many table; expanding that table directly would duplicate bundle-frame rows.

## Full and staged execution

Preview the command list without reading large tables or writing results:

```powershell
& $py scripts/run_pipeline.py --cache $cache --output-root $run --dry-run
```

Run the full static physics and QC workflow:

```powershell
& $py scripts/run_pipeline.py --cache $cache --output-root $run
```

Stage order: `qc → physics → selection → msd → vac → vcc → display → separation → mscd-clean → vac-best-worst`. Later stages read earlier outputs. Select only stages not yet run, or invoke individual scripts below with an existing run directory. The driver does not rebuild existing stage outputs.

```powershell
& $py scripts/run_pipeline.py --cache $cache --output-root $run --stages qc physics
& $py scripts/run_pipeline.py --cache $cache --output-root $run --stages selection msd vac vcc display separation mscd-clean vac-best-worst
```

The full workflow covers physics statistics and static QC. Videos require image assets and selected cases and are run separately below. CSV/Parquet physics analysis can still run without image assets.
The default ten-stage workflow targets the frozen cohort; the `mscd-clean` reporting script expects eight estimates outside the 5 h group.
For other hour-group designs, omit this reporting stage with `--stages` or explicitly define a new reporting contract. Its fixed-count check is not a general-purpose data-cleaning rule.

| Script | Input | Main outputs / purpose |
|---|---|---|
| `20_build_v521_intake.py` | Formal run root + YAML | Content-addressed intake, asset/status audit, data dictionary |
| `21_build_v521_unfiltered_cache.py` | Intake | Bundle-frame cache before scientific QC |
| `22_build_v521_pair_galleries.py` | Cache | T3/T4 membership, exclusion reasons, fixed-scale trajectory galleries |
| `23_build_v521_time_resolved_physics.py` | Cache + `physics.yaml` | Unit curves, support counts, hourly curves, fits, bootstrap intervals |
| `26_select_v521_complete_pair_cases_without_t5_t10.py` | Cache | Cases with 100% coverage, excluding 5/10-frame movies |
| `37_build_v521_final_msd_visualization.py` | Physics + selection | Final MSD figures, per-trajectory alpha, candidate cases |
| `38_build_v521_oligo_vac.py` | Cache + MSD fits + references | VAC at delta=10/20/40 s; consistency diagnostics against MSD alpha |
| `39_build_v521_oligo_vcc.py` | Cache + references | Bidirectional 2×2 tensor, symmetric trace, Rouse time and adequacy |
| `40_build_v521_vac_vcc_visual_candidates.py` | VAC/VCC + MSD fits | MATLAB-style clean/cloud panels; fine-grid displays of actual observations |
| `42_build_v521_separation_final_ppt.py` | Cache | Descriptive 500 nm separation threshold, crop-level comparisons, final static figures |
| `43/44/45` | Frozen alpha/MSCD tables + selection + full run | High/low-alpha and high-MSCD case PNG/PDF/MP4 files |
| `46/47` | Frozen fit/display tables | MSCD macro-time and best/worst VAC consistency panels |
| `41_archive_v521_trajectory_csvs.py` | Formal run root | Compact trajectory CSV archive and hash manifest; does not replace the full run |

Scripts `24/25/28/29` provide additional entry points for separation, complete coverage, distance animations, and T4 ranking. See the [CLI parameter table](docs/CLI_PARAMETERS.md) or each script's `--help` for all options and defaults.

## Dynamic visualization and image QC

Example: review the three cases with the lowest Site1 alpha at 3 h:

```powershell
& $py scripts/44_build_v521_msd_3h_alpha_bottom_cases_and_clean_hour_plot.py `
  --alpha-table "$run\msd\tables\complete_coverage_trajectory_alpha.csv" `
  --fits-table "$run\physics\tables\fits.csv" `
  --selection-csv "$run\selection\tables\eligible_complete_pair_trajectory_summary.csv" `
  --config config/v521_msd_3h_site1_alpha_bottom3.json `
  --fullrun-root 'F:\DSB_v51\dsb_v521_fullrun_20260819T084304Z' `
  --output-dir "$run\alpha-bottom-review" `
  --ffmpeg-exe 'C:\tools\ffmpeg\bin\ffmpeg.exe' `
  --reference-script-dir references/animation `
  --original-matlab ../CrisprTrack2/trajectory_extraction/pipeline/plot_longest_trajectories.m
```

Images must match the exact identifiers, trajectories, and time axis of the current run root. Displays include full-cell context, nuclear boundaries, two-locus crops, scale bars, trajectories, and X/Y traces. Configuration controls brightness percentiles, gamma, channel weights, fonts, line widths, and video encoding without changing trajectory coordinates. `playback_fps=8` controls playback speed; annotations use exact experimental times. Display lines may cross missing frames, but **calculations neither interpolate nor compress missing frames**.

## Equations and statistical conventions

Let the observed two-dimensional coordinates be $\mathbf r_1(t),\mathbf r_2(t)$, converted from nm to µm. Calculations use only frames where all required endpoints are observed.

| Observable | Definition | Output / interpretation |
|---|---|---|
| Separation | $d(t)=\lVert\mathbf r_2(t)-\mathbf r_1(t)\rVert$ | nm; geometric distance between the two probes |
| MSD | $\langle\lVert\mathbf r(t+\tau)-\mathbf r(t)\rVert^2\rangle_t$ | µm²; Site1 and Site2 analyzed separately |
| MSCD | $\langle\lVert\Delta\mathbf R(t+\tau)-\Delta\mathbf R(t)\rVert^2\rangle_t$, $\Delta\mathbf R=\mathbf r_2-\mathbf r_1$ | µm²; change in the relative **vector**, not in scalar separation |
| Velocity | $\mathbf v_\delta(t)=[\mathbf r(t+\delta)-\mathbf r(t)]/\delta_t$ | Each velocity uses its own exact elapsed time |
| VAC | $\langle\mathbf v_\delta(t+\tau)\cdot\mathbf v_\delta(t)\rangle/\langle\lVert\mathbf v_\delta(t)\rVert^2\rangle$ | Dimensionless; normalized at zero lag |
| VCC tensor | $\langle\mathbf v_{1,\delta}(t+\tau)\mathbf v_{2,\delta}(t)^T\rangle / \sqrt{\langle\lVert\mathbf v_1\rVert^2\rangle\langle\lVert\mathbf v_2\rVert^2\rangle}$ | Full bidirectional delta×tau×2×2; plots use the symmetrized trace |

Raw MSD/MSCD curves over 10–50 s are fitted descriptively as $A\tau^\alpha$ / $A\tau^\beta$. No localization-error or exposure-time contract was supplied; localization-error subtraction and motion-blur correction are therefore not assumed.

The fBM reference for VAC is

$$
C(\tau)/C(0)=\frac{|\tau-\delta|^\alpha+|\tau+\delta|^\alpha-2|\tau|^\alpha}{2\delta^\alpha}.
$$

The primary reference alpha comes from an **independent MSD fit for the same hour and site**; VAC-only alpha is diagnostic. VCC uses interpolated numerical Rouse tables, fixes `alpha0=0.9`, and searches communication times from 1–1000 s. Interpret parameters only after assessing fit shape and residuals. See the [VCC implementation audit](provenance/frozen_runs/20260824T_v521_oligo_vcc_v4/IMPLEMENTATION_AUDIT.md) for endpoint and normalization corrections relative to the original MATLAB implementation.

Primary physics analysis uses **observable-specific support**: at least 8 observed endpoint/velocity pairs per unit×lag, with maximum lag `min(50 frames, floor(movie_frames/4))`. T3/T4 membership, 53BP1 positivity, motion amplitude, and separation are not global inclusion gates. Each trajectory/bundle is time-averaged first, then units receive equal weight within each hour. Curve bands show sample SD across units; fit intervals use crop-cluster bootstrap.

T3/T4 are review views for different purposes: T3 requires `10/10/10` site/paired frames and at least 5 contiguous paired frames; T4 requires `20/20/20`, at least 10 contiguous paired frames, and paired coverage ≥0.5. These frozen gallery constants are explicit in code and are not dynamically overwritten by `config/tiered_qc.yaml`. New thresholds require a new analysis version.

See the [configuration parameter guide](docs/PARAMETERS.md) for run and display settings. Support thresholds, fit windows, delta matching, and bootstrap settings change estimates or intervals; colors, fonts, and playback speed affect display only. The driver explicitly applies the final 500 nm threshold and Bonferroni convention. Historical script `24` still defaults to 550 nm, so filenames alone do not identify the active settings.

## Interpretation limits and reproducibility checks

These analyses describe continuous physical differences and QC/observation effects. UMAP/Leiden groups and physical curves are not established discrete repair states. Hour denotes time after Cas9 delivery, not a confirmed cleavage onset; acquisition is a technical grouping. Short-window negative VAC and estimated Rouse communication times alone do not establish an fBM mechanism, repair kinetics, or causality.

```powershell
& $py -m pytest tests
```

See [VALIDATION](VALIDATION.md) for completed packaging checks and outstanding validation. Repository preparation did not rerun the full study or change scientific parameters.
