# Data retention and relocation contract

Read-only source root at packaging time:
`D:\UGVR\Stanley\Genomic DNA Dynamics Predict Chromatin Density\code_phase2_DSB\analysis_dsb_physical_states`.

Keep the following as scientific assets, even where the directory name contains `cache`:

| Asset | Why it is retained |
|---|---|
| `data_snapshot/v5_2_1_formal_70909c2e6326` | Validated trajectory/intake tables, registry, exact time and per-asset eligibility |
| `data_snapshot/v5_2_1_unfiltered_cache_70909c2e6326` | Direct input to the final analyses; compact ~50 MB core frame table preserves all source positions |
| Latest `dsb_v521_fullrun_20260819T084304Z` | Corrected image pixels, masks, exact timing/metadata and diagnostic sidecars needed by dynamic reviews |
| `dsb_v521_trajectory_csv_archive_20260825` | Compact trajectory-only access and per-file manifest; does not replace the full runroot |
| Final run directories in `FROZEN_RESULTS.md` | Model fits, diagnostics, selected cases, images/videos and evidence manifests |
| `code/trajectory_visualization_scripts/.../VVCF` | Supplied Rouse numerical reference; small required files are now copied into `references/` with hashes |

The former result manifests refer to `E:`. Moving that payload to `F:` does not make those source strings correct automatically. The executable scripts accept explicit current roots; do not bulk-edit historical manifests or source hashes. Resolve path relocation in a separate mapping ledger.

Archive obsolete runs as whole directories before deleting convenience copies. Do not judge a full runroot by CSV file counts alone: 53BP1 object/allele-frame data and missing-asset state cannot be reconstructed from a pure trajectory CSV set.

This repository extraction is not evidence that all D/F payloads have been independently SHA256 verified. Destructive storage cleanup remains contingent on the project-level audit and a checked authoritative copy.
