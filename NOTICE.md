# Source and reference notice

This repository consolidates the user's analysis working tree frozen from
`code_phase2_DSB/analysis_dsb_physical_states` on 2026-09-08. Scientific implementations
and numeric defaults are preserved. The former repository HEAD (`d3bd4ab`) predates the
final v5.2.1 working-tree additions, so it is not a sufficient source snapshot.

`provenance/SOURCE_MANIFEST.json` records original paths, sizes and SHA256 values.
`provenance/DELIVERY_MANIFEST.json` records the assembled files after packaging changes.
The original analysis configuration and result manifests are provenance, not runnable
machine-independent defaults. Use `config/physics.yaml` for the current workflow.

`references/rouse`, `references/empirical`, and `references/Read_Me.pdf` preserve supplied
Oligo-LiveFISH/MATLAB reference assets with source attribution and hashes. Their original
copyright and licensing apply. No new permissive license is asserted for those assets.
`references/animation` is the earlier local animation source used in final-output provenance;
current multichannel rendering lives in `dsb_states.v521_case_study_multichannel_review`.

Raw ND2/TIFF, extracted trajectories, trained ML weights, and scientific result payloads
remain external. The repository contains no replacement for the full runroot.
