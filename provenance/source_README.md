# DSB physical-state analysis

This isolated module implements `DSB_trajectory_analysis_master_plan.md` without modifying the v5.1 production extraction pipeline or archive.

## Scientific guardrails

- Macro-time (hours after delivery) and within-movie micro-time are separate axes.
- Site1-Site2 separation is local/flanking-locus geometry, not a broken-end distance.
- A missing 53BP1 trajectory is not evidence of repair.
- Predictive splits are grouped by acquisition at minimum and occur before windowing or preprocessing.
- qPCR, NGS, weak supervision, and perturbation analyses remain gated until their schemas, replicate structure, uncertainty, and mappings are present.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.lock
.\.venv\Scripts\python -m pip install -e . --no-deps
$env:DSB_V51_ROOT = 'E:\DSB_v51\dsb_v51_current_formal_20260811T053950Z_with_review'
```

## Run

```powershell
.\.venv\Scripts\python -m dsb_states.run --config config/analysis.yaml
```

The canonical CLI reserves an immutable `results/<run_id>/`; reruns must use a new ID. Numbered
scripts expose guarded stage-level interfaces for diagnostics, gated templates, and explicitly
provisional analyses. The canonical T04-T12/T19 path remains fail-closed under M0 STOP. Under the
user-authorized exploratory scope, standalone scripts 03/04 can compute neutral Site1/Site2
geometry and conditional point-track sensitivity, while scripts 06/07 can infer frozen DeepSPT
readouts on the current cohort and learn acquisition-grouped candidate kinematic regimes. These
standalone results do not change M0 status and cannot be promoted to repair-state or causal claims.

Trajectory snapshots are content-addressed from the archive/control-index hashes and selection
contract. A completed snapshot is verified and reused read-only; it is never merged, repaired, or
overwritten in place. Physical-metric primary APIs use original frame offsets while retaining the
exact elapsed-time distribution for every lag, so timestamp jitter does not erase support and no
coordinates are interpolated.

The frozen v5.2.1 full run has a separate asset-aware intake builder:

```powershell
.\.venv\Scripts\python scripts\20_build_v521_intake.py --config config\v521_intake.yaml
```

It materializes compact Parquet datasets for Site1/Site2 trajectories, paired allele frames,
Site2-linked 53BP1 metrics, and global 53BP1 focus objects. Task status and each scientific asset
are audited separately; production ND2/TIFF/results remain read-only and are referenced by path.

Build the losslessly reconciled, no-analysis-QC frame cache and the frozen-rule review galleries:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python scripts\21_build_v521_unfiltered_cache.py `
  --snapshot data_snapshot\v5_2_1_formal_70909c2e6326 `
  --output-dir data_snapshot\v5_2_1_unfiltered_cache_70909c2e6326
.\.venv\Scripts\python scripts\22_build_v521_pair_galleries.py `
  --cache data_snapshot\v5_2_1_unfiltered_cache_70909c2e6326 `
  --output-dir results\20260822T_v521_pair_qc_review_galleries_v2
```

The cache applies no T1-T4 or 53BP1-positive filter. The gallery builder then applies the historical
T3 (`10/10/10`, longest shared run `>=5`, no coverage threshold) and T4 (`20/20/20`, longest shared
run `>=10`, shared coverage `>=0.50`) rules exactly and records every exclusion reason.

## M0 metadata interfaces

Formal biological interpretation, accepted state naming, prediction, and causal stages remain
fail-closed until every M0 contract is verified. The neutral exploratory executors above are the
only documented exception and retain Site1/Site2 labels plus explicit provenance. To unblock the
formal path without editing source data:

1. Copy `config/m0_metadata_template.csv` to `config/m0_metadata.csv`; fill one authoritative,
   confirmed row for every formal acquisition/FOV, including acquisition-consistent
   `biological_experiment_id` and `hour_post_delivery`. Evidence fields use typed references
   such as `experimental_design_record:ED-001`; free-form assertions do not pass. Set
   `hierarchy.m0_metadata_path` to that file, set
   `hierarchy.biological_experiment_id_source` to
   `m0_metadata.biological_experiment_id`, and set `time_axes.macro_time_mapping_source` to
   `m0_metadata.hour_post_delivery`. Inferred hour labels never pass this gate.
2. Copy `config/genomic_mapping_template.yaml` to `config/genomic_mapping.yaml`; fill documented
   Site1/Site2 sides, genomic coordinates, positive probe-to-cut distances, cut coordinate,
   genome build, and typed evidence sources such as `probe_design_record:PD-001`. Coordinates
   must be parseable as `chrom:start-end` (or `chrom:position`) and ordered around the cut. Each
   distance must use one declared coordinate-derived definition from the template and match it. Set
   `channels.genomic_mapping_path`,
   `channels.flank_mapping_status: verified`, distinct `site_left_source` / `site_right_source`,
   and matching numeric `site1_probe_to_cut_bp` / `site2_probe_to_cut_bp` values.

Placeholder strings such as `pending`, `unknown`, or arbitrary nonempty values do not satisfy M0.

## Future-data interfaces (gated)

`external_data` in `config/analysis.yaml` contains explicit null paths for qPCR, NGS,
weak-supervision, perturbation, and image-derived 53BP1 inputs. Null paths are intentional: the
project has no such completed data, and enabling a gate without every required table and evidence
file must fail closed. Templates written by the canonical run contain column contracts only and no
values.

When data arrive, copy them into a versioned analysis input location (never the production archive),
set the corresponding paths, and run the schema loaders/gates before any model. qPCR requires assay
semantics, supplied uncertainty or raw Ct evidence, biological replicates, and sample-to-acquisition
mapping. NGS requires sample-level count reconciliation, QC evidence, class definitions, and explicit
sample mapping; trajectory-level labels remain prohibited unless a validated bundle linkage exists.
Weak supervision additionally requires a one-row-per-bag qPCR/NGS label-source ledger, exact bag
membership, a predeclared bag-size threshold and major hours, a strictly hour-only baseline, and
held-out-biological-experiment protocol evidence. Perturbational language requires observed
treatment/control arms and the predeclared control measurements specified by the master plan.
