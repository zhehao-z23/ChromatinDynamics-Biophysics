# v5.2.1 time-resolved physical curves

## What was calculated

- 900,712 supported per-unit scalar curve points.
- 925,240 supported directional VCC matrix-component rows.
- 96,163 metric/delta support-census rows, including failures.
- 4,196 units contribute at least one primary curve point.
- 624 equal-unit hour/lag summary rows.
- 432 symmetric hour/lag VCC matrix-component summary rows.

## Preliminary descriptive fit parameters

- `mscd/pair/exponent`: 1.5h=0.223, 2h=0.401, 2.5h=0.378, 3h=0.378, 3.5h=0.346, 4h=0.382, 4.5h=0.394, 5h=0.339, 10h=0.414
- `mscd/pair/value_at_10s`: 1.5h=0.0389, 2h=0.0458, 2.5h=0.0481, 3h=0.0619, 3.5h=0.0544, 4h=0.055, 4.5h=0.0573, 5h=0.0485, 10h=0.0618
- `msd/site1/exponent`: 1.5h=0.713, 2h=0.456, 2.5h=0.703, 3h=1, 3.5h=0.717, 4h=0.674, 4.5h=0.748, 5h=0.567, 10h=0.703
- `msd/site1/value_at_10s`: 1.5h=0.0376, 2h=0.0567, 2.5h=0.046, 3h=0.0741, 3.5h=0.0432, 4h=0.0602, 4.5h=0.0537, 5h=0.0347, 10h=0.0587
- `msd/site2/exponent`: 1.5h=0.722, 2h=0.491, 2.5h=0.741, 3h=0.562, 3.5h=0.692, 4h=0.697, 4.5h=0.683, 5h=0.511, 10h=0.731
- `msd/site2/value_at_10s`: 1.5h=0.0297, 2h=0.042, 2.5h=0.033, 3h=0.057, 3.5h=0.0363, 4h=0.0407, 4.5h=0.0391, 5h=0.0551, 10h=0.0444
- `vac/site1/exponent`: 1.5h=0.866, 2h=0.568, 2.5h=0.731, 3h=0.532, 3.5h=0.567, 4h=0.531, 4.5h=0.577, 5h=0.458, 10h=0.561
- `vac/site2/exponent`: 1.5h=0.819, 2h=0.504, 2.5h=0.736, 3h=0.406, 3.5h=0.418, 4h=0.481, 4.5h=0.523, 5h=0.383, 10h=0.519
- `vcc/pair/positive_auc_0_5delta`: 1.5h=0.515, 2h=0.398, 2.5h=0.421, 3h=0.528, 3.5h=0.364, 4h=0.379, 4.5h=0.402, 5h=0.36, 10h=0.49
- `vcc/pair/zero_lag_trace`: 1.5h=0.346, 2h=0.36, 2.5h=0.289, 3h=0.407, 3.5h=0.252, 4h=0.307, 4.5h=0.29, 5h=0.367, 10h=0.331

## Interpretation limits

- All acquisitions are pooled as one user-confirmed batch, but exact cadence and movie length remain unequal.
- The common 10–50 s fit window spans only five-fold; exponents are weakly identified.
- 5 h has very sparse Site1/pair support and remains preliminary.
- These are folder-hours after Cas9 delivery, not synchronized times since an observed cut.
- Raw MSD is not localization- or motion-blur-corrected; no `MSD-4σ²` claim is made.
- VCC Rouse communication time is gated; the empirical 2×2 matrix is complete and retained.
- No biological-replicate population inference or causal claim is made.

## Temporal trend table

`tables/fit_parameter_time_trends.csv` contains 20 descriptive Spearman summaries; p-values are intentionally omitted.
