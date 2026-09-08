# Packaging validation — 2026-09-08

**159 active inherited tests passed** in 75.28 seconds. These cover physical metric support,
missing frames, exact timestamps, MSD/MSCD/VAC/VCC behavior, Rouse fits and symmetrization,
QC, intake/cache reconciliation, case selection, image-review construction and separation
statistics. No research dataset analysis was rerun.

Additional checks:

- All 21 delivered command-line entry points returned success for `--help`.
- All delivered Python files parsed successfully.
- The full workflow driver produced a complete ten-stage command plan in no-write dry-run mode.
- An additional six workflow path-isolation checks reject output/cache/snapshot overlap; no science kernel changed.
- All 31 included Rouse `.mat` tables loaded through the production reader as a finite
  `(31, 25, 501)` reference tensor.
- Every packaged `dsb_states` relative import resolves within this repository; ML modules
  live in the sibling repository's separate namespace.
- The active physics/VAC/VCC settings equal the source sections; scientific core source
  files are byte-for-byte copies. Source and final hashes are recorded under `provenance/`.

Validation used Python 3.13.14 from the existing local scientific environment, plus pytest,
statsmodels and patsy in workspace-only test dependencies. Automatic third-party pytest plugin
loading was disabled to avoid unrelated napari plugins. The Conda `Library/bin` directory was
added to the test process DLL search path; an initial run without that activation hit a native
SciPy DLL failure. The full successful run did not require scientific code changes.

The two historical tests tied to the removed monolithic orchestrator and the original physical
archive are retained as source evidence outside the active suite. See
[the exact scope explanation](docs/PACKAGING.md).

This is code/contract verification, not a full new-environment installation test or a rerun of
the frozen 2,070-crop cohort. Original D/F files and scientific result payloads were read only.
Dynamic rendering was tested on synthetic images; the full stored MP4 set was not regenerated.
