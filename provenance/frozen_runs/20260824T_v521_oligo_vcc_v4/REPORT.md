# Oligo-LiveFISH-style pair VCC and Rouse communication-time fit

## Output summary

- 124,628 supported directional scalar curve points.
- 498,512 supported directional matrix-component rows.
- 1,458 Site1/Site2 bundles contribute at least one VCC point.
- 842 crops and 35 acquisitions are represented.
- Full bidirectional 2 x 2 matrices are retained; only their symmetric trace is fitted.

## Folder-hour fits (fixed Rouse alpha0=0.9)

- 1.5 h: tau_Delta_n=51 s [38, 66]; RMSE=0.151; R2=-0.226; n=138 bundles; descriptive_interior_solution.
- 2 h: tau_Delta_n=58 s [43, 77]; RMSE=0.131; R2=0.082; n=245 bundles; descriptive_interior_solution.
- 2.5 h: tau_Delta_n=74 s [63, 86]; RMSE=0.114; R2=0.104; n=309 bundles; descriptive_interior_solution.
- 3 h: tau_Delta_n=52 s [37, 74]; RMSE=0.102; R2=0.523; n=147 bundles; descriptive_interior_solution.
- 3.5 h: tau_Delta_n=70 s [48, 110]; RMSE=0.119; R2=-0.047; n=68 bundles; descriptive_interior_solution.
- 4 h: tau_Delta_n=70 s [51, 102]; RMSE=0.079; R2=0.657; n=121 bundles; descriptive_interior_solution.
- 4.5 h: tau_Delta_n=90 s [70, 117]; RMSE=0.129; R2=-0.092; n=153 bundles; descriptive_interior_solution.
- 5 h: tau_Delta_n=57 s [20, 104]; RMSE=0.046; R2=0.927; n=9 bundles; descriptive_interior_solution.
- 10 h: tau_Delta_n=74 s [61, 93]; RMSE=0.110; R2=0.271; n=268 bundles; descriptive_interior_solution.

## Preliminary temporal reading

- Descriptive Spearman rho(hour, tau_Delta_n)=+0.496; no p-value or biological-replicate claim is made.
- The smallest fitted communication time is 51 s at 1.5 h; the largest is 90 s at 4.5 h.
- Temporal interpretation must be conditioned on fit RMSE, boundary status, and alpha sensitivity.

## Model adequacy is the main result

- 6/9 folder-hours have R2<0.5; the fitted communication times should therefore not be read as a clean kinetic trajectory.
- Mean empirical-minus-Rouse residual at tau/delta>=1.25 is positive at 9/9 hours. The data retain a positive long-lag component that the supplied equilibrium Rouse curves do not capture.
- This mismatch is compatible with unresolved slow common motion or model misspecification; it does not by itself identify a biological mechanism.

## Limits

- The primary alpha0=0.9 is fixed from the supplied corrected Oligo-LiveFISH WLS workflow; alpha0=0.7, 0.8, and 1.0 are archived as sensitivity fits.
- Crop-cluster intervals are descriptive because formal biological replicate IDs are unavailable.
- Unequal acquisition cadence is handled explicitly; acquisitions are otherwise pooled as the user-confirmed same batch.
- Tracking missingness can remain outcome-dependent and can still bias long-lag estimates.
