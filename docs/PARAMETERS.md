# Active parameter catalog

此表从已打包配置生成，数值与来源一致。路径可按机器修改；科学阈值修改后必须使用新输出目录并记录配置。
物理主配置为 `physics.yaml`。T3/T4 gallery 的冻结阈值在 README 解释；遗留 `tiered_qc.yaml` 是历史合同，不被 script 22 读取。

## physics.yaml

| 参数 | 默认值 | 含义 |
|---|---|---|
| `time_resolved_physics.same_experimental_batch` | `true` | 当前研究所有 acquisitions 属于同一个实验 batch 的冻结汇总假设；不是声称 cadence 相同。 |
| `time_resolved_physics.batch_effect_model` | `"none"` | 当前物理曲线不拟合 acquisition/batch effect；保持 none。 |
| `time_resolved_physics.acquisition_role` | `"provenance_and_support_diagnostic_only"` | acquisition 仅用于来源、cadence、支持度诊断；不是 biological replicate。 |
| `time_resolved_physics.macro_time_group` | `"hour_post_delivery"` | 宏观时间分组列，单位为 Cas9 delivery 后小时。 |
| `time_resolved_physics.within_hour_weighting` | `"equal_trajectory_or_bundle"` | 每轨迹或 bundle 先时间平均，然后小时内等权。 |
| `time_resolved_physics.coordinate_interpolation` | `"prohibited"` | 禁止位置插值或压缩缺失帧。 |
| `time_resolved_physics.min_pairs_per_unit_lag` | `8` | 每单位每 lag 至少多少个真实端点/速度对；低于此数不估计该点。 |
| `time_resolved_physics.max_lag_frames_cap` | `50` | 最大 frame lag 的绝对上限。 |
| `time_resolved_physics.max_lag_nominal_fraction` | `0.25` | 最大 frame lag 同时不得超过 movie 总帧数的该比例。 |
| `time_resolved_physics.physical_lag_bin_centers_s` | `[1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]` | 跨 cadence 汇总的物理时滞中心（秒）；不会生成插值坐标。 |
| `time_resolved_physics.common_descriptive_fit_window_s` | `[10, 50]` | MSD/MSCD 描述性幂律拟合物理窗口（秒）。 |
| `time_resolved_physics.minimum_fit_lag_points` | `5` | 支持拟合的最少 lag 点数。 |
| `time_resolved_physics.minimum_fit_time_span_fold` | `4.0` | 拟合最大/最小物理时滞至少达到此倍数。 |
| `time_resolved_physics.primary_velocity_delta_target_s` | `10.0` | 历史单窗口 VAC/VCC 的目标速度间隔（秒）；最终多窗口见 oligo_*。 |
| `time_resolved_physics.primary_velocity_delta_allowed_s` | `[7.5, 12.5]` | 最近整数帧速度间隔允许的真实秒数范围。 |
| `time_resolved_physics.oligo_livefish_delta_frames` | `[1, 2, 4, 8]` | 保留在 unit tables 中的附加速度窗口（frame offsets）。 |
| `time_resolved_physics.bootstrap_iterations` | `500` | crop-cluster bootstrap 重抽样次数；增加可提高区间数值稳定性并增加耗时。 |
| `time_resolved_physics.bootstrap_seed` | `20260823` | bootstrap 随机种子；变更会改变 Monte Carlo 区间。 |
| `oligo_vac.target_delta_s` | `[10, 20, 40]` | 目标速度窗口（秒）；每 acquisition 独立选择最近整数 frame offset。 |
| `oligo_vac.delta_relative_tolerance` | `0.25` | 目标速度窗口的相对容差；0.25 表示 ±25%。 |
| `oligo_vac.maximum_scaled_lag` | `2.5` | 最大无量纲 lag τ/δ。 |
| `oligo_vac.minimum_velocity_pairs` | `8` | 每单位每 lag 最少真实速度向量对数。 |
| `oligo_vac.minimum_hour_bin_units_for_display` | `8` | 图点最少贡献轨迹数；低支持点仍保留在数据表。 |
| `oligo_vac.raw_lag_bin_centers_s` | `[0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]` | 原始 VAC lag 汇总中心（秒）。 |
| `oligo_vac.scaled_lag_bin_centers` | `[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5]` | 无量纲 τ/δ 汇总中心。 |
| `oligo_vac.alpha_bounds` | `[0.25, 1.0]` | VAC-only 诊断 alpha 的搜索范围；不替换主图的独立 MSD alpha。 |
| `oligo_vac.bootstrap_iterations` | `300` | crop-cluster bootstrap 重抽样次数；增加可提高区间数值稳定性并增加耗时。 |
| `oligo_vac.bootstrap_seed` | `20260824` | bootstrap 随机种子；变更会改变 Monte Carlo 区间。 |
| `oligo_vac.raw_lag_plot_limit_s` | `100` | 原始 lag 图 x 轴最大秒数，仅显示限制。 |
| `oligo_vcc.target_delta_s` | `[10, 20, 40]` | 目标速度窗口（秒）；每 acquisition 独立选择最近整数 frame offset。 |
| `oligo_vcc.delta_relative_tolerance` | `0.25` | 目标速度窗口的相对容差；0.25 表示 ±25%。 |
| `oligo_vcc.maximum_scaled_lag` | `2.5` | 最大无量纲 lag τ/δ。 |
| `oligo_vcc.minimum_velocity_pairs` | `8` | 每单位每 lag 最少真实速度向量对数。 |
| `oligo_vcc.minimum_hour_bin_units_for_fit` | `8` | 拟合点至少贡献多少 bundle。 |
| `oligo_vcc.scaled_lag_bin_centers` | `[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]` | 无量纲 τ/δ 汇总中心。 |
| `oligo_vcc.rouse_alpha_fixed` | `0.9` | Rouse 模型固定 viscoelastic exponent；主分析 0.9。 |
| `oligo_vcc.rouse_alpha_sensitivity` | `[0.7, 0.8, 0.9, 1.0]` | Rouse alpha 敏感性值；与主分析结果区分。 |
| `oligo_vcc.communication_time_bounds_s` | `[1, 1000]` | communication-time 整秒搜索下限/上限（秒）。 |
| `oligo_vcc.bootstrap_iterations` | `200` | crop-cluster bootstrap 重抽样次数；增加可提高区间数值稳定性并增加耗时。 |
| `oligo_vcc.bootstrap_seed` | `20260824` | bootstrap 随机种子；变更会改变 Monte Carlo 区间。 |

## v521_intake.yaml

| 参数 | 默认值 | 含义 |
|---|---|---|
| `schema_version` | `"v5.2.1-analysis-intake-v1"` | 配置结构版本标识；不是可自由调整的科学参数。 |
| `source.root` | `"E:/DSB_v51/dsb_v521_fullrun_20260819T084304Z"` | 输入的正式提取 runroot；更换磁盘后显式更新。 |
| `source.expected_version` | `"5.2.1"` | 接受的提取 pipeline 版本。 |
| `source.expected_commit` | `"9f4e28a7bdaa6847e876fa9e3de059de197d0441"` | 正式提取源码 commit；校验来源用。 |
| `output.parent` | `"data_snapshot"` | 新 intake 输出父目录；禁止放入只读源 runroot。 |
| `output.prefix` | `"v5_2_1_formal"` | 内容寻址 intake 文件夹前缀。 |
| `output.compression` | `"zstd"` | Parquet 压缩编码。 |
| `metadata.existing_m0_path` | `"config/m0_acquisition_metadata_v521.csv"` | 冻结 acquisition-hour metadata CSV 路径。 |
| `metadata.new_10h_hour_post_delivery` | `10.0` | 新增 10h cohort 的已确认 hour 值。 |
| `metadata.new_10h_hour_evidence` | `"laboratory_record:user_confirmation_2026-08-22_new_10h_cohort"` | 新增 cohort 宏观时间证据标识。 |
| `metadata.macro_time_semantics` | `"hours_after_cas9_delivery_not_cut_onset"` | 宏观时间语义，保持 delivery 后时间。 |
| `metadata.acquisition_role` | `"technical_acquisition_not_biological_replicate"` | acquisition 仅用于来源、cadence、支持度诊断；不是 biological replicate。 |
| `validation.minimum_trajectory_points` | `4` | intake 保留轨迹的最少位置数；与物理指标每 lag 支持度不同。 |
| `validation.require_exact_frame_times` | `true` | 要求精确逐帧时间，不能用推测 cadence 替代。 |
| `validation.require_bp1_sidecar_schema` | `"v5.2.1-bp1-locus-sidecar-v1"` | 要求的正式 53BP1 sidecar schema。 |
| `materialize.trajectory_points` | `true` | 是否物化独立位点轨迹点表。 |
| `materialize.bp1_allele_frames` | `true` | 是否物化 Site2-linked dense 53BP1 帧表。 |
| `materialize.bp1_allele_summary` | `true` | 是否物化 allele 级 53BP1 汇总。 |
| `materialize.bp1_frame_segmentation` | `true` | 是否物化逐帧 53BP1 分割摘要。 |
| `materialize.bp1_focus_objects` | `true` | 是否物化 global 53BP1 对象表。 |

## v521_final_msd_visualization.json

| 参数 | 默认值 | 含义 |
|---|---|---|
| `representative_hours` | `[1.5, 3.0, 10.0]` | MSD 展示的代表小时；不改变全时段拟合。 |
| `fit_window_s` | `[10.0, 50.0]` | 逐轨迹 MSD 幂律拟合窗口（秒）。 |
| `bootstrap_iterations` | `500` | crop-cluster bootstrap 重抽样次数；增加可提高区间数值稳定性并增加耗时。 |
| `bootstrap_seed` | `20260823` | bootstrap 随机种子；变更会改变 Monte Carlo 区间。 |
| `case_selection.primary_hour` | `3.0` | 个案候选主要小时；不改变群体汇总。 |
| `case_selection.minimum_pairs_per_lag` | `8` | 逐轨迹 case alpha 拟合每 lag 最少端点对。 |
| `case_selection.minimum_fit_lags` | `5` | 逐轨迹 case alpha 拟合的最少 lag 点。 |
| `case_selection.minimum_span_fold` | `4.0` | 逐轨迹拟合时滞覆盖倍数。 |
| `case_selection.minimum_r2` | `0.9` | 个案支持的 log-log 幂律 R² 最低要求；只影响个案筛选。 |
| `case_selection.targets` | `[{"case_id": "alpha1", "target_alpha": 1.0}, {"case_id": "alpha05", "target_alpha": 0.5}]` | 个案的命名和目标 alpha 列表；选择最接近的受支持候选。 |
| `case_display.raw_minimum_crop_px` | `72` | 图像局部视窗最小边长（像素）。 |
| `case_display.raw_padding_px` | `12` | 轨迹包围框外的图像边缘余量（像素）。 |
| `case_display.site1_color` | `"#F4B400"` | Site1 显示颜色。 |
| `case_display.site2_color` | `"#7A3DB8"` | Site2 显示颜色。 |
| `case_display.bp1_color` | `"#00B050"` | 53BP1 显示颜色。 |
| `case_display.nucleus_outline_color` | `"#66D9EF"` | 核 mask 轮廓颜色。 |
| `case_display.nucleus_outline_linewidth` | `1.8` | 核轮廓线宽。 |
| `case_display.nucleus_outline_alpha` | `0.95` | 核轮廓透明度（0–1）。 |
| `case_display.site1_weight` | `0.85` | 融合图中 Site1 通道亮度权重。 |
| `case_display.site2_weight` | `0.85` | 融合图中 Site2 通道亮度权重。 |
| `case_display.bp1_weight` | `0.6` | 融合图中 53BP1 通道亮度权重。 |
| `case_display.site_gamma` | `0.85` | 位点图像亮度 gamma，仅影响显示。 |
| `case_display.bp1_gamma` | `0.85` | 53BP1 图像亮度 gamma，仅影响显示。 |
| `case_display.site_percentiles` | `[50.0, 99.75]` | 位点图像显示强度归一化的下/上百分位。 |
| `case_display.bp1_percentiles` | `[5.0, 99.5]` | 53BP1 图像显示强度归一化的下/上百分位。 |
| `case_display.track_linewidth` | `2.4` | 轨迹主线宽。 |
| `case_display.track_halo_linewidth` | `4.8` | 轨迹外层衬色线宽，提高背景对比。 |
| `case_display.time_colormap` | `"jet"` | 轨迹时间颜色映射。 |
| `case_display.playback_fps` | `8.0` | 视频播放帧率；不改变实验 timestamps。 |
| `case_display.video_crf` | `20` | H.264 CRF 压缩质量；越低质量/体积越高。 |
| `case_display.figure_size_inches` | `[15.8, 7.6]` | 图大小（宽×高，英寸）。 |
| `case_display.png_dpi` | `220` | PNG 导出分辨率。 |
| `case_display.shared_trace_axis_limit_nm` | `1000.0` | X/Y trace 共享轴范围（±nm）。 |

## v521_case_study_multichannel_review.json

| 参数 | 默认值 | 含义 |
|---|---|---|
| `schema_version` | `5` | 配置结构版本标识；不是可自由调整的科学参数。 |
| `cases` | `[{"case_id": "shortest_06", "display_label": "Shortest 6", "bundle_id": "LiveFISH DSB006\|LiveFISH DSB006_candidate_5\|a002"}, {"case_id": "longest_03", "display_label": "Longest 3", "bundle_id": "LiveFISH DSB091\|LiveFISH DSB091_candidate_43\|a002"}]` | 固定个案列表；bundle_id 必须与当前 frozen input 匹配。 |
| `display.raw_minimum_crop_px` | `72` | 图像局部视窗最小边长（像素）。 |
| `display.raw_padding_px` | `12` | 轨迹包围框外的图像边缘余量（像素）。 |
| `display.site1_color` | `"#F4B400"` | Site1 显示颜色。 |
| `display.site2_color` | `"#7A3DB8"` | Site2 显示颜色。 |
| `display.bp1_color` | `"#00B050"` | 53BP1 显示颜色。 |
| `display.site1_weight` | `0.85` | 融合图中 Site1 通道亮度权重。 |
| `display.site2_weight` | `0.85` | 融合图中 Site2 通道亮度权重。 |
| `display.bp1_weight` | `0.6` | 融合图中 53BP1 通道亮度权重。 |
| `display.site_gamma` | `0.85` | 位点图像亮度 gamma，仅影响显示。 |
| `display.bp1_gamma` | `0.85` | 53BP1 图像亮度 gamma，仅影响显示。 |
| `display.site_percentiles` | `[50.0, 99.75]` | 位点图像显示强度归一化的下/上百分位。 |
| `display.bp1_percentiles` | `[5.0, 99.5]` | 53BP1 图像显示强度归一化的下/上百分位。 |
| `display.track_linewidth` | `2.4` | 轨迹主线宽。 |
| `display.track_halo_linewidth` | `4.8` | 轨迹外层衬色线宽，提高背景对比。 |
| `display.pixel_scale_bar_linewidth` | `5.0` | 图像比例尺主线宽。 |
| `display.pixel_scale_bar_halo_linewidth` | `8.5` | 图像比例尺衬色线宽。 |
| `display.pixel_scale_bar_fontsize` | `18.0` | 图像比例尺字体大小。 |
| `display.pixel_label_halo_linewidth` | `3.2` | 图像标注文字衬色宽。 |
| `display.full_cell_time_fontsize` | `18.0` | 全细胞画面的时间标注字体大小。 |
| `display.time_colormap` | `"jet"` | 轨迹时间颜色映射。 |
| `display.playback_fps` | `8.0` | 视频播放帧率；不改变实验 timestamps。 |
| `display.video_crf` | `20` | H.264 CRF 压缩质量；越低质量/体积越高。 |
| `display.figure_size_inches` | `[15.8, 7.6]` | 图大小（宽×高，英寸）。 |
| `display.png_dpi` | `220` | PNG 导出分辨率。 |
| `display.shared_trace_axis_limit_nm` | `1000.0` | X/Y trace 共享轴范围（±nm）。 |

## v521_msd_3h_site1_alpha_top5.json

| 参数 | 默认值 | 含义 |
|---|---|---|
| `schema_version` | `1` | 配置结构版本标识；不是可自由调整的科学参数。 |
| `selection.hour_post_delivery` | `3.0` | 按 Cas9 delivery 后小时筛选展示个案。 |
| `selection.site` | `"site1"` | 个案排名所用位点。 |
| `selection.top_n` | `5` | 选择排名最高的 N 个个案。 |
| `selection.minimum_r2` | `0.9` | 个案支持的 log-log 幂律 R² 最低要求；只影响个案筛选。 |
| `selection.fit_window_s` | `[10.0, 50.0]` | 逐轨迹 MSD 幂律拟合窗口（秒）。 |
| `cases` | `[{"case_id": "alpha_top01", "display_label": "Site1 alpha top 1", "bundle_id": "LiveFISH DSB014\|LiveFISH DSB014_candidate_45\|a001"}, {"case_id": "alpha_top02", "display_label": "Site1 alpha top 2", "bundle_id": "LiveFISH DSB013\|LiveFISH DSB013_candidate_12\|a002"}, {"case_id": "alpha_top03", "display_label": "Site1 alpha top 3", "bundle_id": "LiveFISH 3h_DSB016\|LiveFISH 3h_DSB016_candidate_16\|a001"}, {"case_id": "alpha_top04", "display_label": "Site1 alpha top 4", "bundle_id": "LiveFISH DSB013\|LiveFISH DSB013_candidate_13\|a002"}, {"case_id": "alpha_top05", "display_label": "Site1 alpha top 5", "bundle_id": "LiveFISH DSB013\|LiveFISH DSB013_candidate_7\|a001"}]` | 固定个案列表；bundle_id 必须与当前 frozen input 匹配。 |
| `display.raw_minimum_crop_px` | `72` | 图像局部视窗最小边长（像素）。 |
| `display.raw_padding_px` | `12` | 轨迹包围框外的图像边缘余量（像素）。 |
| `display.site1_color` | `"#F4B400"` | Site1 显示颜色。 |
| `display.site2_color` | `"#7A3DB8"` | Site2 显示颜色。 |
| `display.bp1_color` | `"#00B050"` | 53BP1 显示颜色。 |
| `display.nucleus_outline_color` | `"#66D9EF"` | 核 mask 轮廓颜色。 |
| `display.nucleus_outline_linewidth` | `1.8` | 核轮廓线宽。 |
| `display.nucleus_outline_alpha` | `0.95` | 核轮廓透明度（0–1）。 |
| `display.site1_weight` | `0.85` | 融合图中 Site1 通道亮度权重。 |
| `display.site2_weight` | `0.85` | 融合图中 Site2 通道亮度权重。 |
| `display.bp1_weight` | `0.6` | 融合图中 53BP1 通道亮度权重。 |
| `display.site_gamma` | `0.85` | 位点图像亮度 gamma，仅影响显示。 |
| `display.bp1_gamma` | `0.85` | 53BP1 图像亮度 gamma，仅影响显示。 |
| `display.site_percentiles` | `[50.0, 99.75]` | 位点图像显示强度归一化的下/上百分位。 |
| `display.bp1_percentiles` | `[5.0, 99.5]` | 53BP1 图像显示强度归一化的下/上百分位。 |
| `display.track_linewidth` | `2.4` | 轨迹主线宽。 |
| `display.track_halo_linewidth` | `4.8` | 轨迹外层衬色线宽，提高背景对比。 |
| `display.pixel_scale_bar_linewidth` | `5.0` | 图像比例尺主线宽。 |
| `display.pixel_scale_bar_halo_linewidth` | `8.5` | 图像比例尺衬色线宽。 |
| `display.pixel_scale_bar_fontsize` | `18.0` | 图像比例尺字体大小。 |
| `display.pixel_label_halo_linewidth` | `3.2` | 图像标注文字衬色宽。 |
| `display.full_cell_time_fontsize` | `18.0` | 全细胞画面的时间标注字体大小。 |
| `display.time_colormap` | `"jet"` | 轨迹时间颜色映射。 |
| `display.playback_fps` | `8.0` | 视频播放帧率；不改变实验 timestamps。 |
| `display.video_crf` | `20` | H.264 CRF 压缩质量；越低质量/体积越高。 |
| `display.figure_size_inches` | `[15.8, 7.6]` | 图大小（宽×高，英寸）。 |
| `display.png_dpi` | `220` | PNG 导出分辨率。 |

## v521_msd_3h_site1_alpha_bottom3.json

| 参数 | 默认值 | 含义 |
|---|---|---|
| `schema_version` | `1` | 配置结构版本标识；不是可自由调整的科学参数。 |
| `selection.hour_post_delivery` | `3.0` | 按 Cas9 delivery 后小时筛选展示个案。 |
| `selection.site` | `"site1"` | 个案排名所用位点。 |
| `selection.bottom_n` | `3` | 选择排名最低的 N 个个案。 |
| `selection.minimum_r2` | `0.9` | 个案支持的 log-log 幂律 R² 最低要求；只影响个案筛选。 |
| `selection.fit_window_s` | `[10.0, 50.0]` | 逐轨迹 MSD 幂律拟合窗口（秒）。 |
| `cases` | `[{"case_id": "alpha_bottom01", "display_label": "Site1 alpha bottom 1", "bundle_id": "LiveFISH DSB015\|LiveFISH DSB015_candidate_27\|a001"}, {"case_id": "alpha_bottom02", "display_label": "Site1 alpha bottom 2", "bundle_id": "LiveFISH DSB013\|LiveFISH DSB013_candidate_3\|a003"}, {"case_id": "alpha_bottom03", "display_label": "Site1 alpha bottom 3", "bundle_id": "LiveFISH DSB014\|LiveFISH DSB014_candidate_11\|a002"}]` | 固定个案列表；bundle_id 必须与当前 frozen input 匹配。 |
| `display.raw_minimum_crop_px` | `72` | 图像局部视窗最小边长（像素）。 |
| `display.raw_padding_px` | `12` | 轨迹包围框外的图像边缘余量（像素）。 |
| `display.site1_color` | `"#F4B400"` | Site1 显示颜色。 |
| `display.site2_color` | `"#7A3DB8"` | Site2 显示颜色。 |
| `display.bp1_color` | `"#00B050"` | 53BP1 显示颜色。 |
| `display.nucleus_outline_color` | `"#66D9EF"` | 核 mask 轮廓颜色。 |
| `display.nucleus_outline_linewidth` | `1.8` | 核轮廓线宽。 |
| `display.nucleus_outline_alpha` | `0.95` | 核轮廓透明度（0–1）。 |
| `display.site1_weight` | `0.85` | 融合图中 Site1 通道亮度权重。 |
| `display.site2_weight` | `0.85` | 融合图中 Site2 通道亮度权重。 |
| `display.bp1_weight` | `0.6` | 融合图中 53BP1 通道亮度权重。 |
| `display.site_gamma` | `0.85` | 位点图像亮度 gamma，仅影响显示。 |
| `display.bp1_gamma` | `0.85` | 53BP1 图像亮度 gamma，仅影响显示。 |
| `display.site_percentiles` | `[50.0, 99.75]` | 位点图像显示强度归一化的下/上百分位。 |
| `display.bp1_percentiles` | `[5.0, 99.5]` | 53BP1 图像显示强度归一化的下/上百分位。 |
| `display.track_linewidth` | `2.4` | 轨迹主线宽。 |
| `display.track_halo_linewidth` | `4.8` | 轨迹外层衬色线宽，提高背景对比。 |
| `display.pixel_scale_bar_linewidth` | `5.0` | 图像比例尺主线宽。 |
| `display.pixel_scale_bar_halo_linewidth` | `8.5` | 图像比例尺衬色线宽。 |
| `display.pixel_scale_bar_fontsize` | `18.0` | 图像比例尺字体大小。 |
| `display.pixel_label_halo_linewidth` | `3.2` | 图像标注文字衬色宽。 |
| `display.full_cell_time_fontsize` | `18.0` | 全细胞画面的时间标注字体大小。 |
| `display.time_colormap` | `"jet"` | 轨迹时间颜色映射。 |
| `display.playback_fps` | `8.0` | 视频播放帧率；不改变实验 timestamps。 |
| `display.video_crf` | `20` | H.264 CRF 压缩质量；越低质量/体积越高。 |
| `display.figure_size_inches` | `[15.8, 7.6]` | 图大小（宽×高，英寸）。 |
| `display.png_dpi` | `220` | PNG 导出分辨率。 |

## v521_mscd10_top4_complete_coverage.json

| 参数 | 默认值 | 含义 |
|---|---|---|
| `schema_version` | `1` | 配置结构版本标识；不是可自由调整的科学参数。 |
| `selection.top_n` | `4` | 选择排名最高的 N 个个案。 |
| `selection.target_lag_s` | `10.0` | 个案 MSCD 排名的目标 lag（秒）。 |
| `selection.accepted_lag_window_s` | `[7.5, 12.5]` | 个案 MSCD 排名允许的真实 lag 范围（秒），不插值。 |
| `selection.minimum_endpoint_pairs` | `8` | 个案 MSCD 排名需要的最少真实端点对数。 |
| `selection.ranking` | `"descending directly observed paired relative-vector MSCD at the supported lag nearest 10 s"` | 固定的候选排序规则；只影响个案选择。 |
| `cases` | `[{"case_id": "mscd10_top01", "display_label": "MSCD top 1", "bundle_id": "LiveFISH DSB009\|LiveFISH DSB009_candidate_46\|a001"}, {"case_id": "mscd10_top02", "display_label": "MSCD top 2", "bundle_id": "LiveFISH DSB023\|LiveFISH DSB023_candidate_44\|a003"}, {"case_id": "mscd10_top03", "display_label": "MSCD top 3", "bundle_id": "LiveFISH DSB028\|LiveFISH DSB028_candidate_55\|a002"}, {"case_id": "mscd10_top04", "display_label": "MSCD top 4", "bundle_id": "LiveFISH DSB008\|LiveFISH DSB008_candidate_3\|a001"}]` | 固定个案列表；bundle_id 必须与当前 frozen input 匹配。 |
| `display.raw_minimum_crop_px` | `72` | 图像局部视窗最小边长（像素）。 |
| `display.raw_padding_px` | `12` | 轨迹包围框外的图像边缘余量（像素）。 |
| `display.site1_color` | `"#F4B400"` | Site1 显示颜色。 |
| `display.site2_color` | `"#7A3DB8"` | Site2 显示颜色。 |
| `display.bp1_color` | `"#00B050"` | 53BP1 显示颜色。 |
| `display.nucleus_outline_color` | `"#66D9EF"` | 核 mask 轮廓颜色。 |
| `display.nucleus_outline_linewidth` | `1.8` | 核轮廓线宽。 |
| `display.nucleus_outline_alpha` | `0.95` | 核轮廓透明度（0–1）。 |
| `display.site1_weight` | `0.85` | 融合图中 Site1 通道亮度权重。 |
| `display.site2_weight` | `0.85` | 融合图中 Site2 通道亮度权重。 |
| `display.bp1_weight` | `0.6` | 融合图中 53BP1 通道亮度权重。 |
| `display.site_gamma` | `0.85` | 位点图像亮度 gamma，仅影响显示。 |
| `display.bp1_gamma` | `0.85` | 53BP1 图像亮度 gamma，仅影响显示。 |
| `display.site_percentiles` | `[50.0, 99.75]` | 位点图像显示强度归一化的下/上百分位。 |
| `display.bp1_percentiles` | `[5.0, 99.5]` | 53BP1 图像显示强度归一化的下/上百分位。 |
| `display.track_linewidth` | `2.4` | 轨迹主线宽。 |
| `display.track_halo_linewidth` | `4.8` | 轨迹外层衬色线宽，提高背景对比。 |
| `display.pixel_scale_bar_linewidth` | `5.0` | 图像比例尺主线宽。 |
| `display.pixel_scale_bar_halo_linewidth` | `8.5` | 图像比例尺衬色线宽。 |
| `display.pixel_scale_bar_fontsize` | `18.0` | 图像比例尺字体大小。 |
| `display.pixel_label_halo_linewidth` | `3.2` | 图像标注文字衬色宽。 |
| `display.full_cell_time_fontsize` | `18.0` | 全细胞画面的时间标注字体大小。 |
| `display.time_colormap` | `"jet"` | 轨迹时间颜色映射。 |
| `display.playback_fps` | `8.0` | 视频播放帧率；不改变实验 timestamps。 |
| `display.video_crf` | `20` | H.264 CRF 压缩质量；越低质量/体积越高。 |
| `display.figure_size_inches` | `[15.8, 7.6]` | 图大小（宽×高，英寸）。 |
| `display.png_dpi` | `220` | PNG 导出分辨率。 |

