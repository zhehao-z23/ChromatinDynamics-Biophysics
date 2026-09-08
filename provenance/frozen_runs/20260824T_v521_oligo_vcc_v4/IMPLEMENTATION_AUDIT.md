# Pair VCC implementation audit

## Reference MATLAB logic retained

- Finite-window velocity: `v_delta(t)=[r(t+delta)-r(t)]/delta`.
- Two-locus cross-correlation: `C12_delta(tau)=<v1(t+tau) v2(t)^T>`.
- Oligo-LiveFISH display coordinate `tau/delta`, evaluated to 2.5.
- Corrected `calc_vel_corr_new.m` numerical tables and the
  `WLS_vel_corr_new.m` 1--1000 s communication-time search.
- The reference WLS value `alpha0=0.9` is fixed for the primary temporal comparison.

## Errors or dataset-incompatible assumptions corrected

- The supplied empirical MATLAB arrays are initialized with zeros. Missing coordinates and
  unsupported velocities therefore enter some means as real zero motion. The revised code uses
  NaN/finite-endpoint support only.
- The supplied code checks only the later y coordinate (`Dfiny~=0`), does not require both velocity
  endpoints, and treats a valid y=0 position as missing. The revised code requires finite x/y at
  both endpoints of both velocities.
- The supplied cross-correlation is normalized only by the first channel's autocorrelation energy.
  The revised primary denominator is the fixed symmetric scale
  `sqrt(<|v1|^2><|v2|^2>)`, so Site1/Site2 naming cannot change the result.
- The supplied code computes only one lead/lag direction. The revised result stores both
  `C12(tau)` and `C21(tau)`, transposes the reverse matrix, and uses their equal mean for the
  equilibrium-style Rouse comparison.
- The supplied code hard-codes one frame interval and later overwrites it for plotting. The revised
  code uses each acquisition's exact time schedule and matches integer frame offsets separately to
  10, 20, and 40 s measurement windows.
- The supplied final cross-correlation matrix allocates one extra row that is never filled. The
  revised tables emit only supported lags.

## Final analysis contract

- Primary empirical object: `delta x tau x 2 x 2` normalized VCC matrix for every bundle.
- Rouse fit readout: the rotation-invariant trace of the bidirectionally symmetrized matrix.
- Pooling: time average within bundle, then equal bundle mean within folder-hour.
- Metric support only: at least eight directly observed velocity pairs at a lag.
- No coordinate interpolation, trajectory compression, separation threshold, motion threshold,
  T3/T4 global QC, 53BP1 requirement, learned-state filter, or acquisition exclusion.
- Folder-hour is time after Cas9 delivery, not synchronized time since an observed cut.
- `tau_Delta_n` is a descriptive model parameter unless the Rouse curve shape is adequate; it is
  not a direct measurement of cutting time or repair kinetics.
