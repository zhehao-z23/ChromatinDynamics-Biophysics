# ChromatinDynamics-Biophysics

从 **CrisprTrack2 已提取的轨迹及精确时间信息**开始，完成染色质运动的物理量、统计支持度、质控图和多通道动态复核。当前默认是 v5.2.1 正式输入（提取 commit `9f4e28a7bdaa6847e876fa9e3de059de197d0441`）及 2026-08-26 前完成的物理分析工作树。

本仓库不做图像分割和轨迹提取，也不训练 fingerprint/聚类/监督模型。后两者由 `ChromatinDynamics-ML` 负责；ML 可安装本仓库后导入 `dsb_states.physical_metrics` 等物理内核，或直接读取这里产生的表。

| 仓库 | 起点 → 终点 | 共享接口 |
|---|---|---|
| [CrisprTrack2](https://github.com/zhehao-z23/CrisprTrack2) | ND2/TIFF → nucleus segmentation → corrected trajectories | 正式 runroot、`frame,x_nm,y_nm`、精确时间与来源 metadata |
| ChromatinDynamics-Biophysics | 正式轨迹 → intake/cache → MSD/MSCD/VAC/VCC、QC、图像/视频复核 | Parquet/CSV 物理表、bundle/crop/acquisition 标识 |
| [ChromatinDynamics-ML](https://github.com/zhehao-z23/ChromatinDynamics-ML) | 相同冻结 intake / 物理表 → fingerprint → 无监督与监督 | 通过标识 join；不复制图像提取或物理内核 |

## 版本和最终分析定位

原分析仓库 Git HEAD 仍为旧 `d3bd4ab`，最终 v5.2.1 文件是尚未提交的工作树内容。本次保留的是**工作树及文件 SHA256**，不是仅复制旧 Git HEAD。见 [来源记录](provenance/SOURCE_MANIFEST.json)、[打包变更](docs/PACKAGING.md) 和 [历史运行索引](docs/FROZEN_RESULTS.md)。

主要保留：逐小时 MSD/MSCD v2；多速度窗口 VAC v2；完整双向 VCC 张量与 Rouse communication-time v4；最终 MSD/分离距离图；完整覆盖 case selection；多通道动态复核；3 h 高/低 alpha、最高 MSCD 个案和最终清爽展示图。旧 `provenance/frozen_runs` 是可追溯的运行证据，里面的绝对路径不会自动转为本机数据路径。

## 安装

需要 Python 3.11+。新环境请使用本仓库最小依赖，而不是旧分析项目包含全部 ML 方法的环境清单。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
```

Linux/macOS 对应使用 `.venv/bin/python`。数值 Rouse 参考表已包含在 `references/rouse/tables`；动态视频还需要可用的 FFmpeg。参考代码权利归属见 [NOTICE](NOTICE.md)。

## 输入必须是什么

三种起点不能混用：

1. **完整正式 runroot**：提取输出、精确逐帧时间、crop metadata、轨迹清单及 53BP1 sidecars，可从中构建完整 intake；动态图额外需要修正 TIFF 和 nucleus mask。
2. **冻结 intake**：包含 `table_registry.json` 和注册的 Parquet 数据集。当前冻结标识为 `v5_2_1_formal_70909c2e6326`，可直接构建 cache。
3. **冻结 flat cache**：包含 `CACHE_CONTRACT.json`、`tables/bundle_frame_master.parquet`、`tables/bundle_index.parquet` 等文件，可直接运行物理分析。

仅有 `frame,x_nm,y_nm` 的纯 CSV 归档可以复用轨迹，但不能独立恢复精确时间、科学 asset 状态、53BP1 指标或图像视频。不要给 CSV 人为补一个时间间隔来冒充当前正式输入。完整列合同见 [数据字典](docs/INTAKE_DATA_DICTIONARY.md)。

所有分析写入**新的输出目录**。原始 runroot 和冻结 intake/cache 只读，重复运行更换输出名。典型配置：

```powershell
$cache = 'F:\DSB_v51\path_to_frozen_cache'
$run = 'C:\analysis_results\physics_20260908'
```

上述是示例路径，需要替换为实际归档位置。原项目当前来源路径列在 [数据保留与路径迁移](docs/DATA_AND_STORAGE.md)。
驱动器要求 `output-root`、`cache`、可选 `snapshot` 互不重叠，避免将新输出写进冻结数据。

## 从正式 runroot 构建 intake 和 cache

复制 `config/v521_intake.yaml` 为 `config/v521_intake.local.yaml`，修改 `source.root`、`output.parent` 与必要 metadata 路径；提取版本、commit、精确时间要求及字段 schema 保持原值，除非明确创建新的数据版本。

```powershell
& $py scripts/20_build_v521_intake.py --config config/v521_intake.local.yaml
# 上一命令会打印实际生成的 content-addressed snapshot 路径。
& $py scripts/21_build_v521_unfiltered_cache.py `
  --snapshot 'C:\analysis_data\v5_2_1_formal_70909c2e6326' `
  --output-dir 'C:\analysis_data\v5_2_1_unfiltered_cache_70909c2e6326'
```

Cache 不应用 T1–T4 或 53BP1 阳性筛选，也不填补缺失位置。Global 53BP1 对象是独立一对多表，不能直接展开并重复 bundle-frame 行。

## 完整运行与分阶段运行

先查看命令清单（不读大表、不写结果）：

```powershell
& $py scripts/run_pipeline.py --cache $cache --output-root $run --dry-run
```

执行完整静态物理与 QC 工作流：

```powershell
& $py scripts/run_pipeline.py --cache $cache --output-root $run
```

顺序为 `qc → physics → selection → msd → vac → vcc → display → separation → mscd-clean → vac-best-worst`。后续阶段读取前序结果。分阶段只选择尚未运行的阶段，或用下表脚本传入已有运行目录；驱动器不会重建已经存在的阶段结果。

```powershell
& $py scripts/run_pipeline.py --cache $cache --output-root $run --stages qc physics
& $py scripts/run_pipeline.py --cache $cache --output-root $run --stages selection msd vac vcc display separation mscd-clean vac-best-worst
```

这里的“完整”是物理统计与静态 QC；视频需要原始图像及所选 case，因此单独执行下一节。没有 image asset 时，CSV/Parquet 物理流程仍可运行。
默认十阶段流程对应当前冻结cohort；`mscd-clean` 的汇报脚本要求8个非5 h估计值。
分析其他hour设计时，使用 `--stages` 省略该特定汇报阶段，或明确建立新的汇报合同，不能把其固定数量检查当作通用数据清洗。

| 脚本 | 输入 | 主要输出 / 目的 |
|---|---|---|
| `20_build_v521_intake.py` | 正式 runroot + YAML | 内容寻址 intake、asset/status audit、数据字典 |
| `21_build_v521_unfiltered_cache.py` | intake | 无科学 QC 的 bundle-frame cache |
| `22_build_v521_pair_galleries.py` | cache | T3/T4 membership、剔除原因、固定比例轨迹画廊 |
| `23_build_v521_time_resolved_physics.py` | cache + `physics.yaml` | unit curves、支持度、hour curves、fits、bootstrap 区间 |
| `26_select_v521_complete_pair_cases_without_t5_t10.py` | cache | 100% 覆盖且去掉 5/10-frame movie 的候选个案 |
| `37_build_v521_final_msd_visualization.py` | physics + selection | 最终 MSD 图、逐轨迹 alpha、候选 case |
| `38_build_v521_oligo_vac.py` | cache + MSD fits + references | delta=10/20/40 s VAC、MSD-alpha 一致性诊断 |
| `39_build_v521_oligo_vcc.py` | cache + references | 双向 2×2 张量、对称 trace、Rouse 时间与 adequacy |
| `40_build_v521_vac_vcc_visual_candidates.py` | VAC/VCC + MSD fits | MATLAB 风格 clean/cloud 面板；真实观测细网格展示 |
| `42_build_v521_separation_final_ppt.py` | cache | 500 nm 分离阈值描述、crop-level 比较、最终静态图 |
| `43/44/45` | 冻结 alpha/MSCD 表 + selection + fullrun | 高/低 alpha 与高 MSCD 个案 PNG/PDF/MP4 |
| `46/47` | 冻结 fit/display 表 | MSCD macro-time 与 VAC 最佳/最差一致性面板 |
| `41_archive_v521_trajectory_csvs.py` | 正式 runroot | 精简轨迹 CSV 归档与 hash manifest；不替代完整 runroot |

其余 `24/25/28/29` 保留专门的分离距离、完整覆盖、距离动画和 T4 ranking 入口。全部命令行参数与默认值见 [CLI 参数表](docs/CLI_PARAMETERS.md)，也可运行任一脚本 `--help`。

## 动态可视化和图像质控

以最终 3 h 最低 Site1 alpha 三个 case 为例：

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

图像必须和当前 runroot 的精确标识、轨迹、时间轴一致。显示含 full-cell context、核边缘、两位点 crop、比例尺、轨迹和 X/Y traces。配置控制亮度百分位、gamma、channel weight、字体、线宽和视频编码，不改变轨迹坐标。`playback_fps=8` 是播放速度；视频注释使用精确实验时间。绘图连线可以跨缺失帧，**计算不插值、也不压缩缺失帧**。

## 公式和统计口径

令二维记录坐标为 \(\mathbf r_1(t),\mathbf r_2(t)\)，由 nm 转为 µm；计算只使用所需端点真实存在的帧。

| 量 | 定义 | 输出 / 解释 |
|---|---|---|
| Separation | \(d(t)=\|\mathbf r_2(t)-\mathbf r_1(t)\|\) | nm；双探针几何距离 |
| MSD | \(\langle\|\mathbf r(t+\tau)-\mathbf r(t)\|^2\rangle_t\) | µm²；Site1/Site2 独立进入 |
| MSCD | \(\langle\|\Delta\mathbf R(t+\tau)-\Delta\mathbf R(t)\|^2\rangle_t\), \(\Delta\mathbf R=\mathbf r_2-\mathbf r_1\) | µm²；相对**向量**变化，不是标量距离变化 |
| Velocity | \(\mathbf v_\delta(t)=[\mathbf r(t+\delta)-\mathbf r(t)]/\delta_t\) | 每个速度使用自己的精确 elapsed time |
| VAC | \(\langle\mathbf v_\delta(t+\tau)\cdot\mathbf v_\delta(t)\rangle/\langle\|\mathbf v_\delta(t)\|^2\rangle\) | 无量纲；零时滞归一化 |
| VCC tensor | \(\langle\mathbf v_{1,\delta}(t+\tau)\mathbf v_{2,\delta}(t)^T\rangle / \sqrt{\langle\|\mathbf v_1\|^2\rangle\langle\|\mathbf v_2\|^2\rangle}\) | 完整双向 delta×tau×2×2；绘图用双向对称 trace |

MSD/MSCD 的 10–50 s 原始曲线作描述性拟合 \(A\tau^\alpha\) / \(A\tau^\beta\)。未提供 localization-error / exposure-time 合同，因此没有擅自减去定位误差或声称 motion-blur correction。

VAC 的 fBM 参考为
\[
C(\tau)/C(0)=\frac{|\tau-\delta|^\alpha+|\tau+\delta|^\alpha-2|\tau|^\alpha}{2\delta^\alpha}.
\]
主参考的 alpha 来自**同小时同位点独立 MSD 拟合**；VAC-only alpha 是诊断。VCC 用数值 Rouse 表插值，固定 `alpha0=0.9`，搜索 communication time 1–1000 s；评价拟合形状与残差后才解释参数。实现细节、对原 MATLAB 缺失端点与归一化处理的修正见 [VCC audit](provenance/frozen_runs/20260824T_v521_oligo_vcc_v4/IMPLEMENTATION_AUDIT.md)。

主要物理分析使用**指标自身的支持度**：unit×lag 至少 8 个真实端点/速度对；最大 lag 为 `min(50 frames, floor(movie_frames/4))`。不把 T3/T4、53BP1 阳性、运动幅度或 separation 当作全局入选条件。每条轨迹/bundle 先时间平均，再在小时内等权；曲线带是 unit 间 sample SD；拟合区间使用 crop-cluster bootstrap。

T3/T4 是不同用途的审查视图：T3=`10/10/10` 位点/配对帧、连续配对≥5；T4=`20/20/20`、连续配对≥10、配对 coverage≥0.5。这些 frozen gallery 常数在代码中明示，不由 `config/tiered_qc.yaml` 动态改写；新阈值必须建立新的分析版本。

全部运行/显示配置解释见 [配置参数表](docs/PARAMETERS.md)。改变支持度、拟合窗口、delta 匹配或 bootstrap 设置会改变估计/区间；颜色、字体、播放速率仅改变展示。驱动器明确传入最终 500 nm 和 Bonferroni 口径；历史脚本 `24` 的默认仍是 550 nm，不能仅凭文件名推断。

## 当前结论边界与复现检查

这里描述持续的物理差异和 QC/观察效应，不把 UMAP/Leiden 或物理曲线命名为离散修复状态。Hour 是 Cas9 delivery 后时间，非确认的切割起点；acquisition 是技术分组。短窗口负 VAC 及 Rouse communication-time 估计本身不证明 fBM 机制、repair kinetics 或因果关系。

```powershell
& $py -m pytest tests
```

当前打包验证及尚未执行的工作见 [VALIDATION](VALIDATION.md)。本次整理不重跑完整研究、不改变科学参数、不移动或删除原始 D/F 数据。
