# Table dictionary

## `acquisition_index`

- Layout: `single_parquet_plus_csv`
- Rows: 40
- Path: `tables/acquisition_index.parquet`
- Columns: `cohort`, `nd2_id`, `fov_id`, `hour_post_delivery`, `hour_mapping_evidence`, `macro_time_semantics`, `acquisition_role`, `source_nd2`, `movie_frames`, `z_planes`, `median_interval_s`, `trajectory_eligible`, `acquisition_exclusion_reason`, `crop_policy`, `crops_site1_usable`, `crops_site2_usable`, `crops_paired_frames_usable`, `crops_bp1_global_usable`, `crops_bp1_locus_usable`, `candidate_crops`

## `crop_index`

- Layout: `single_parquet_plus_csv`
- Rows: 2070
- Path: `tables/crop_index.parquet`
- Columns: `cohort`, `nd2_id`, `crop_id`, `crop_index`, `fov_id`, `hour_post_delivery`, `movie_frames`, `exact_timing_points`, `frame_interval_s_production`, `task_status`, `runner_exit_code`, `site1_usable`, `site2_usable`, `bp1_spt_usable`, `paired_tracks_usable`, `paired_frames_usable`, `bp1_global_usable`, `bp1_locus_usable`, `bp1_continuous_intensity_usable`, `bp1_component_association_observed`, `availability_class`, `unavailable_reason`, `crop_tif_source_relative_path`, `metadata_source_relative_path`, `result_source_relative_path`, `bp1_labels_source_relative_path`

## `trajectory_index`

- Layout: `single_parquet_plus_csv`
- Rows: 10322
- Path: `tables/trajectory_index.parquet`
- Columns: `trajectory_id`, `bundle_id`, `cohort`, `nd2_id`, `crop_id`, `fov_id`, `hour_post_delivery`, `allele_index`, `site_id`, `source_channel`, `usable`, `validation_reason`, `points`, `first_frame`, `last_frame`, `frame_span`, `temporal_coverage_fraction`, `maximum_missing_frames_between_points`, `median_step_px`, `p95_step_px`, `frame_interval_s_production`, `movie_frame_count`, `pixel_size_nm_per_px`, `source_relative_path`

## `allele_index`

- Layout: `single_parquet_plus_csv`
- Rows: 4645
- Path: `tables/allele_index.parquet`
- Columns: `bundle_id`, `cohort`, `nd2_id`, `crop_id`, `fov_id`, `hour_post_delivery`, `allele_index`, `site1_usable`, `site2_usable`, `bp1_spt_usable`, `site1_points`, `site2_points`, `bp1_spt_points`, `paired_tracks_usable`, `shared_site_frames`, `paired_frames_usable`, `bp1_locus_usable`, `bp1_continuous_intensity_valid_frames`, `bp1_continuous_intensity_usable`, `bp1_component_associated_frames`, `bp1_component_association_observed`

## `unavailable_crops`

- Layout: `single_parquet_plus_csv`
- Rows: 491
- Path: `tables/unavailable_crops.parquet`
- Columns: `cohort`, `nd2_id`, `crop_id`, `crop_index`, `fov_id`, `hour_post_delivery`, `movie_frames`, `exact_timing_points`, `frame_interval_s_production`, `task_status`, `runner_exit_code`, `site1_usable`, `site2_usable`, `bp1_spt_usable`, `paired_tracks_usable`, `paired_frames_usable`, `bp1_global_usable`, `bp1_locus_usable`, `bp1_continuous_intensity_usable`, `bp1_component_association_observed`, `availability_class`, `unavailable_reason`, `crop_tif_source_relative_path`, `metadata_source_relative_path`, `result_source_relative_path`, `bp1_labels_source_relative_path`

## `trajectory_points`

- Layout: `parquet_dataset_partitioned_by_acquisition_file`
- Rows: 473211
- Path: `tables/trajectory_points`
- Columns: `trajectory_id`, `bundle_id`, `cohort`, `nd2_id`, `crop_id`, `fov_id`, `hour_post_delivery`, `allele_index`, `site_id`, `frame`, `x_nm`, `y_nm`, `time_s`, `uniform_time_s`, `x_um`, `y_um`

## `bp1_allele_frames`

- Layout: `parquet_dataset_partitioned_by_acquisition_file`
- Rows: 512596
- Path: `tables/bp1_allele_frames`
- Columns: `cohort`, `nd2_id`, `crop_id`, `fov_id`, `hour_post_delivery`, `bundle_id`, `allele_index`, `frame`, `time_s`, `pixel_size_nm`, `site2_track_csv`, `site1_track_csv`, `site2_valid`, `site1_valid`, `site2_x_nm`, `site2_y_nm`, `site2_x_px`, `site2_y_px`, `site1_x_nm`, `site1_y_nm`, `site1_x_px`, `site1_y_px`, `site2_bp1_aperture_mean_au`, `site2_bp1_aperture_sum_au`, `site2_bp1_local_background_median_au`, `site2_bp1_background_subtracted_mean_au`, `site2_bp1_background_subtracted_sum_au`, `site2_bp1_nonnegative_excess_mean_au`, `site2_bp1_nonnegative_excess_sum_au`, `site2_bp1_local_excess_ratio`, `site2_bp1_continuous_intensity_score`, `intensity_aperture_nominal_pixel_count`, `intensity_aperture_valid_pixel_count`, `intensity_background_nominal_pixel_count`, `intensity_background_valid_pixel_count`, `intensity_aperture_valid_fraction`, `intensity_background_valid_fraction`, `intensity_metric_valid`, `intensity_missing_reason`, `assignment_status`, `assigned_object_id`, `nearest_candidate_object_id`, `nearest_candidate_signed_boundary_distance_nm`, `nearest_candidate_unsigned_boundary_distance_nm`, `second_candidate_margin_nm`, `candidate_count_in_harvest_domain`, `candidate_count_within_association_gate`, `site2_to_bp1_signed_boundary_distance_nm`, `site2_inside_bp1`, `site2_bp1_distance_score`, `assigned_focus_track_id`, `assigned_focus_lineage_event`, `assigned_focus_area_px`, `assigned_focus_equivalent_radius_nm`, `assigned_focus_centroid_x_nm`, `assigned_focus_centroid_y_nm`, `site1_one_frame_step_nm`, `site2_one_frame_step_nm`, `site1_site2_separation_nm`

## `bp1_allele_summary`

- Layout: `parquet_dataset_partitioned_by_acquisition_file`
- Rows: 4602
- Path: `tables/bp1_allele_summary`
- Columns: `cohort`, `nd2_id`, `crop_id`, `fov_id`, `hour_post_delivery`, `bundle_id`, `allele_index`, `movie_frames`, `site2_localizations`, `site1_localizations`, `shared_site_frames`, `intensity_valid_frames`, `associated_component_frames`, `associated_fraction_of_site2_frames`, `median_continuous_intensity_score`, `median_distance_score`

## `bp1_frame_segmentation`

- Layout: `parquet_dataset_partitioned_by_acquisition_file`
- Rows: 188633
- Path: `tables/bp1_frame_segmentation`
- Columns: `cohort`, `nd2_id`, `crop_id`, `fov_id`, `hour_post_delivery`, `frame`, `time_s`, `segmentation_status`, `segmentation_gaussian_sigma_px`, `segmentation_threshold_au`, `nuclear_background_median_au`, `robust_background_scale_au`, `threshold_method`, `threshold_robust_z`, `nucleus_area_px`, `raw_threshold_component_count`, `eligible_component_count`, `raw_foreground_area_px`, `eligible_foreground_area_px`

## `bp1_focus_objects`

- Layout: `parquet_dataset_partitioned_by_acquisition_file`
- Rows: 4071647
- Path: `tables/bp1_focus_objects`
- Columns: `cohort`, `nd2_id`, `crop_id`, `fov_id`, `hour_post_delivery`, `frame`, `time_s`, `object_id`, `source_threshold_component_id`, `area_px`, `area_um2`, `equivalent_radius_nm`, `centroid_x_px`, `centroid_y_px`, `centroid_x_nm`, `centroid_y_nm`, `bbox_x_min_px`, `bbox_x_max_px`, `bbox_y_min_px`, `bbox_y_max_px`, `mean_intensity_au`, `median_intensity_au`, `maximum_intensity_au`, `integrated_excess_intensity_au`, `dtype_saturated_pixel_fraction`, `touches_nucleus_boundary`, `touches_image_boundary`, `focus_track_id`, `lineage_event`, `previous_frame_object_ids`, `parent_focus_track_ids`, `overlap_px_previous`, `dilated_overlap_px_previous`, `next_frame_object_ids`, `child_focus_track_ids`, `focus_track_first_frame`, `focus_track_last_frame`, `focus_track_length_frames`

