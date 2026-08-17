# Data audit and interpretation boundary

## Headline current scheme

Source: `优化方案对比（平面、纵断面联合协同优化）/results/joint_results_w500_dens.json`.

| Quantity | M-A existing | M-C current | Reported change |
|---|---:|---:|---:|
| Life-cycle cost, C (10^8 RMB) | 26.4061 | 24.2078 | -8.325% |
| Monetized traffic energy, E (10^8 RMB) | 13.9460 | 13.5278 | -2.999% |
| Length (km) | 22.4618 | 22.4741 | +0.0548% |
| Minimum horizontal radius (m) | 422.43 | 420.91 | — |
| Mean slope-hazard index | 3.3094 | 3.2979 | -0.347% |
| Hard-constraint penalty | — | 0 | feasible |
| Tier-1 / Tier-2 exposure (km) | — | 1.591 / 0 | — |

The cost reduction is driven primarily by bridge/tunnel cost decreasing from 18.1536 to 15.7859 ×10^8 RMB. Earthwork rises from 0.6717 to 0.8234 ×10^8 RMB and route length rises slightly; the paper reports this trade-off rather than attributing all improvement to shorter geometry.

The current M-C solution contains approximately 4.08 km of OSM-triggered crossing bridges and 0.750 km of ecological tunnel. Crossing-bridge length is recomputed from the current geometry with the repository's crossing detector; the tunnel value is stored in the result JSON.

## Controlled comparison and statistical experiments

- The joint-versus-two-stage comparison is a controlled **pre-density** experiment with equal evaluation budgets. The two-stage solution has penalty 0.0736 and is infeasible; the joint solution has zero penalty. It is not mixed into the current density-enabled scheme table.
- Ablation statistics use 30 independent runs per variant. Mean normalized objective decreases from 0.9133 (JS) to 0.6808 (full IJS), a 25.45% improvement; paired Wilcoxon p = 3.02 × 10^-11.
- The six-problem benchmark uses 10 runs per algorithm/problem. IJS has Friedman rank 1; the manuscript separately discloses that an IJS generation uses roughly three population-equivalent objective-evaluation stages.
- Sensitivity plots contain 237 re-optimized scenarios, all feasible. They are not frozen-line perturbation evaluations.

## Scope limitations retained in the manuscripts

- The quasi-natural DEM is reconstructed from the available corridor data; its reported validation error is about 8.8 m and it is not a final survey surface.
- OSM completeness is not equivalent to a field asset inventory. Missing obstacle elevations prevent a full vertical-clearance check.
- Building-density Tier 2 is a hard exclusion; Tier 1 is a soft planning proxy, not parcel-level acquisition compensation.
- Energy currently represents one anchored representative direction. Fuel and electric traffic are both monetized, but a two-direction operating profile is future work.
- The mathematical model includes constructive grade enforcement and endpoint elevation anchoring, but final design still requires explicit clothoids, vertical curves, sight distance, drainage and geotechnical verification.
- The latest density-enabled current scheme is one reported optimization run. Statistical robustness is supplied by separate ablation/benchmark experiments, not claimed for that single design realization.

## Integrity checks

- All headline percentages were recalculated from raw JSON values.
- Figure-generation code writes only under this version directory.
- Existing experiment results and unrelated repository assets were not modified.
- Author identity, funding and repository DOI placeholders must be resolved before submission.
