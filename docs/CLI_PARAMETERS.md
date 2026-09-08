# Command-line options

由 Python AST 提取；`required` 意味着必须显式指定。`--help` 给出 argparse 的实际界面。
所有输出路径应为新目录；fullrun/cache/snapshot 是外部科学资产。

## 20_build_v521_intake.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| inherited | — | — | 入口委托至同名 package main；运行 `--help`。 |

## 21_build_v521_unfiltered_cache.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--snapshot` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |

## 22_build_v521_pair_galleries.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--cache` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |

## 23_build_v521_time_resolved_physics.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--cache` | `Path` | `required` |   |
| `--config` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--reuse-unit-result` | `Path` | `not specified` |  'Reuse immutable per-unit tables from a validated prior result and rebuild summaries/figures.' |

## 24_build_v521_shared_frame_separation_500nm.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--cache` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--threshold-nm` | `float` | `550.0` |   |
| `--bin-width-nm` | `float` | `50.0` |   |

## 25_select_v521_complete_pair_cases.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--cache` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--n-extreme` | `int` | `5` |   |
| `--display-min-frames` | `int` | `20` |   |

## 26_select_v521_complete_pair_cases_without_t5_t10.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--cache` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--n-extreme` | `int` | `5` |   |

## 28_build_v521_case_study_multichannel_review.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--config` | `Path` | `required` |   |
| `--fullrun-root` | `Path` | `required` |   |
| `--selection-csv` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--ffmpeg-exe` | `Path` | `required` |   |
| `--reference-script-dir` | `Path` | `required` |   |
| `--original-matlab` | `Path` | `required` |   |

## 29_rank_v521_t4_msd_extremes.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--qc-membership` | `Path` | `required` |   |
| `--unit-lag-bins` | `Path` | `required` |   |
| `--output` | `Path` | `required` |   |

## 37_build_v521_final_msd_visualization.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--physics-run` | `Path` | `required` |   |
| `--complete-pair-csv` | `Path` | `required` |   |
| `--config` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |

## 38_build_v521_oligo_vac.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--cache` | `Path` | `required` |   |
| `--config` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--msd-fits` | `Path` | `required` |   |
| `--reference-vvcf-dir` | `Path` | `required` |   |
| `--reference-readme-pdf` | `Path` | `required` |   |

## 39_build_v521_oligo_vcc.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--cache` | `Path` | `required` |   |
| `--config` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--empirical-reference-dir` | `Path` | `required` |   |
| `--vvcf-reference-dir` | `Path` | `required` |   |
| `--reference-readme-pdf` | `Path` | `required` |   |

## 40_build_v521_vac_vcc_visual_candidates.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--vac-result` | `Path` | `required` |   |
| `--vcc-result` | `Path` | `required` |   |
| `--msd-fits` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |

## 41_archive_v521_trajectory_csvs.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--source-root` | `Path` | `required` |   |
| `--output-root` | `Path` | `required` |   |
| `--progress-every` | `int` | `250` |   |
| `--workers` | `int` | `16` |   |

## 42_build_v521_separation_final_ppt.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--cache` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--threshold-nm` | `float` | `500.0` |   |
| `--multiple-testing` | `str` | `'holm'` | ('none', 'holm', 'bonferroni')  |

## 43_build_v521_msd_3h_alpha_top_cases.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--alpha-table` | `Path` | `required` |   |
| `--selection-csv` | `Path` | `required` |   |
| `--config` | `Path` | `required` |   |
| `--fullrun-root` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--ffmpeg-exe` | `Path` | `required` |   |
| `--reference-script-dir` | `Path` | `required` |   |
| `--original-matlab` | `Path` | `required` |   |

## 44_build_v521_msd_3h_alpha_bottom_cases_and_clean_hour_plot.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--alpha-table` | `Path` | `required` |   |
| `--fits-table` | `Path` | `required` |   |
| `--selection-csv` | `Path` | `required` |   |
| `--config` | `Path` | `required` |   |
| `--fullrun-root` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--ffmpeg-exe` | `Path` | `required` |   |
| `--reference-script-dir` | `Path` | `required` |   |
| `--original-matlab` | `Path` | `required` |   |
| `--resume-after-cases` | `str` | `not specified` |  'Reuse an already completed case_studies/ manifest after a late-stage failure.' |

## 45_build_v521_mscd10_top4_cases.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--unit-curves` | `Path` | `required` |   |
| `--selection-csv` | `Path` | `required` |   |
| `--config` | `Path` | `required` |   |
| `--fullrun-root` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |
| `--ffmpeg-exe` | `Path` | `required` |   |
| `--reference-script-dir` | `Path` | `required` |   |
| `--original-matlab` | `Path` | `required` |   |

## 46_build_v521_mscd10_macro_time_clean.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--fits-table` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |

## 47_build_v521_vac_fbm_best_worst_panel.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--dense-vac-table` | `Path` | `required` |   |
| `--consistency-table` | `Path` | `required` |   |
| `--output-dir` | `Path` | `required` |   |

## run_pipeline.py

| 选项 | 类型 | 默认 / 必需 | 可选值 / 帮助 |
|---|---|---|---|
| `--cache` | `Path` | `required` |  'Existing frozen flat cache, or a NEW path when --snapshot is set.' |
| `--snapshot` | `Path` | `not specified` |  'Optional frozen intake to build a new cache first; existing cache is never overwritten.' |
| `--output-root` | `Path` | `required` |  'New analysis root. For partial resumption use only uncompleted --stages.' |
| `--config` | `Path` | `ROOT / 'config/physics.yaml'` |   |
| `--stages` | `str` | `list(STAGES)` | STAGES  |
| `--dry-run` | `str` | `not specified` |  'Print command plan only; create no files.' |

