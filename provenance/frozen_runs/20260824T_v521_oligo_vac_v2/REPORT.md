# Oligo-LiveFISH-style multi-delta VAC

## Output summary

- 291,121 supported trajectory/delta/lag points.
- 4,098 trajectories contribute at least one VAC point.
- 1,449 crops and 35 acquisitions represented.
- Site1 and Site2 are calculated independently; paired-site availability is not required.
- Target velocity windows are 10, 20, and 40 s with no coordinate interpolation.

## Primary Oligo-LiveFISH consistency test

- The blue Equation 9 overlays use alpha from the independently fitted MSD curves.
- The VAC-only alpha estimates below are diagnostic, not replacement MSD exponents.

- 1.5 h site1: MSD alpha=0.713; VAC-only alpha=0.957; Eq. 9 RMSE at MSD alpha=0.124.
- 1.5 h site2: MSD alpha=0.722; VAC-only alpha=1.000; Eq. 9 RMSE at MSD alpha=0.152.
- 2 h site1: MSD alpha=0.456; VAC-only alpha=0.836; Eq. 9 RMSE at MSD alpha=0.149.
- 2 h site2: MSD alpha=0.491; VAC-only alpha=0.847; Eq. 9 RMSE at MSD alpha=0.152.
- 2.5 h site1: MSD alpha=0.703; VAC-only alpha=0.841; Eq. 9 RMSE at MSD alpha=0.095.
- 2.5 h site2: MSD alpha=0.741; VAC-only alpha=0.968; Eq. 9 RMSE at MSD alpha=0.116.
- 3 h site1: MSD alpha=1.004; VAC-only alpha=0.723; Eq. 9 RMSE at MSD alpha=0.129.
- 3 h site2: MSD alpha=0.562; VAC-only alpha=0.592; Eq. 9 RMSE at MSD alpha=0.094.
- 3.5 h site1: MSD alpha=0.717; VAC-only alpha=0.866; Eq. 9 RMSE at MSD alpha=0.101.
- 3.5 h site2: MSD alpha=0.692; VAC-only alpha=0.879; Eq. 9 RMSE at MSD alpha=0.112.
- 4 h site1: MSD alpha=0.674; VAC-only alpha=0.669; Eq. 9 RMSE at MSD alpha=0.067.
- 4 h site2: MSD alpha=0.697; VAC-only alpha=0.739; Eq. 9 RMSE at MSD alpha=0.084.
- 4.5 h site1: MSD alpha=0.748; VAC-only alpha=0.755; Eq. 9 RMSE at MSD alpha=0.087.
- 4.5 h site2: MSD alpha=0.683; VAC-only alpha=0.878; Eq. 9 RMSE at MSD alpha=0.114.
- 5 h site1: MSD alpha=0.567; VAC-only alpha=0.792; Eq. 9 RMSE at MSD alpha=0.129.
- 5 h site2: MSD alpha=0.511; VAC-only alpha=0.565; Eq. 9 RMSE at MSD alpha=0.085.
- 10 h site1: MSD alpha=0.703; VAC-only alpha=0.771; Eq. 9 RMSE at MSD alpha=0.082.
- 10 h site2: MSD alpha=0.731; VAC-only alpha=0.875; Eq. 9 RMSE at MSD alpha=0.106.

## Interpretation

- The original-style raw-lag panels show whether the anticorrelation trough moves with delta.
- The rescaled panels show whether the curves collapse near tau/delta=1, the key fBM diagnostic.
- Agreement between the VAC collapse and the MSD-alpha Eq. 9 curve supports fBM self-consistency; disagreement is informative and is not refitted away in the primary panel.
- Folder-hour is time after Cas9 delivery, not synchronized time since an observed cut.
- No biological-replicate inference or causal claim is made.
