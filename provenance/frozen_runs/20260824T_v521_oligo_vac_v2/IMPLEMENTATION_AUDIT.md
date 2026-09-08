# VAC implementation audit

## What was already correct

- The old project calculation used finite-difference velocities
  `v_delta(t)=[r(t+delta)-r(t)]/delta`, the dot product
  `<v_delta(t+tau) dot v_delta(t)>`, and zero-lag energy normalization.
- Missing frames stayed on the original one-based acquisition schedule; coordinates were not
  interpolated or compressed.
- Trajectories were time-averaged first and then equal-weighted within folder-hour.

## What was incomplete or potentially misleading

- The old headline plot retained only the acquisition-specific delta nearest 10 s. It therefore
  could show an anticorrelation trough but could not test the Oligo-LiveFISH multi-delta collapse.
- The old fit used only that single delta and treated VAC-derived alpha as a headline result.
  Oligo-LiveFISH instead overlays Equation 9 using alpha estimated independently from MSD. The
  revised main figures follow that self-consistency test. A joint 10/20/40 s VAC-only alpha fit is
  retained only as a diagnostic and uses the local/reference range 0.25-1.0.
- The old physical-time binning was optimized for cross-acquisition comparison, not direct
  tau/delta collapse. The revised collapse bins tau/delta itself.

## Important separation of VAC and VCC reference code

- `exp_corr.m` and the Oligo-LiveFISH Figure 5J layout motivate the raw-tau and rescaled-tau/delta
  panels used here.
- `calc_vel_corr_new.m` fixes reversed interpolation weights in an older numerical-table
  implementation. That table and `WLS_vel_corr_new.m` describe Rouse-model two-locus VCC
  communication-time fitting, not the single-locus VAC Eq. 9 fit.
- The local file `site_1and2_canonical_group_280kb_vcctheory.eps` and Oligo-LiveFISH Methods
  Eq. 10/Movie S2 independently identify that numerical-table workflow as VCC. It is therefore
  intentionally not applied to Site1 or Site2 VAC.

## Revised VAC contract

- Oligo-LiveFISH Eq. 7: `C_v^delta(tau)=<v_delta(t+tau) dot v_delta(t)>`.
- Oligo-LiveFISH Eq. 8: `v_delta(t)=[r(t+delta)-r(t)]/delta`.
- Normalize each trajectory/delta curve by its own `C_v^delta(0)` before pooling.
- Match each acquisition to the nearest integer frame offset for target delta=10, 20, or 40 s;
  require the schedule-median delta to lie within +/-25% of target.
- A reported trajectory/lag requires at least eight directly observed velocity pairs.
- No trajectory interpolation, pair-QC requirement, motion/separation threshold, 53BP1 filter,
  hour filter, acquisition filter, or learned-state filter is used.
- At each tau/delta bin, first average supported deltas within trajectory, then equal-weight
  trajectories within folder-hour.
- Overlay Oligo-LiveFISH Eq. 9 using the independently estimated folder-hour/site MSD exponent.
- Fit the collapsed hour curve to Eq. 9 over 0 < tau/delta <= 2.5 only as a diagnostic. Confidence
  intervals are crop-cluster bootstraps and remain descriptive because biological replicates are
  unavailable.
- Suppress only displayed hour/delta bins with fewer than eight contributing trajectories; full
  values remain in the tables and this display rule does not affect fitting.
