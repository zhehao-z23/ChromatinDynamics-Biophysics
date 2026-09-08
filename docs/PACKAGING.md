# Packaging changes

- Kept the final working-tree implementations under their original `dsb_states` import namespace.
- Removed ML/state-learning entry points from this distribution. Their successor is the sibling ML repository.
- Added minimal install metadata, `.gitignore`, a parameter catalog, source/delivery hash manifests, and a sequential workflow driver with a no-write dry run.
- `config/physics.yaml` contains the unchanged final `time_resolved_physics`, `oligo_vac`, and `oligo_vcc` sections extracted from the source configuration. No biological or fit defaults changed.
- The workflow driver resolves included MATLAB references relative to the repository, passes the final separation threshold 500 nm and Bonferroni setting explicitly, and refuses to overwrite completed stage directories.
- Original full configuration, original dependency lock, and result method/source manifests remain under `provenance/` for historical audit. They include old machine-specific paths and superseded methods; they are not active workflow defaults.
- Original multichannel image-review scripts keep explicit `--fullrun-root`, `--ffmpeg-exe`, and reference-path arguments. The extraction MATLAB file is supplied by CrisprTrack2, so its implementation is not duplicated here.
- Data payloads, movies and large result tables stay outside Git. Small numerical Rouse reference tables are retained because they are required to reproduce the fitted model.

An unmodified source hash and a final delivery hash are intentionally distinct. A copied file can be traced to its source even after documentation or path-portability changes.

## Test scope adaptations

- Kept five standalone snapshot immutability/provenance tests. The former
  `test_run_provenance_binds_archive_manifest_and_snapshot_controls` targets the removed
  monolithic `dsb_states.run` orchestrator; its original source is archived in
  `provenance/excluded_tests/test_snapshot_provenance.py.txt`.
- The former `test_active_config_m0_contracts_are_consistent` requires the author's actual
  historical E-drive production archive to report `archive_status=pass`. It is an
  environment integration assertion, not a portable unit test. Its original source is
  archived in `provenance/excluded_tests/test_audit.py.txt`; synthetic audit tests remain active.
- The standalone schema/config test now reads the preserved source configuration from
  `provenance/source_analysis.yaml`; all of its scientific assertions remain unchanged.
- No failing scientific test was disabled, and no numerical expectations were relaxed.
