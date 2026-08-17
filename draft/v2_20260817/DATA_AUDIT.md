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
| Tier-1 / Tier-2 exposure (km) | 2.010 / 0 | 1.591 / 0 | -20.85% / feasible |

The cost reduction is driven primarily by bridge/tunnel cost decreasing from 18.1536 to 15.7859 ×10^8 RMB. Earthwork rises from 0.6717 to 0.8234 ×10^8 RMB and route length rises slightly; the paper reports this trade-off rather than attributing all improvement to shorter geometry.

The current M-C solution contains approximately 4.08 km of OSM-triggered crossing bridges and 0.750 km of ecological tunnel. Crossing-bridge length is recomputed from the current geometry with the repository's crossing detector; the tunnel value is stored in the result JSON. The authoritative JSON SHA-256 is `e7c652922252f97f906c55cef550b9181a360d5bd6fbdf073db42ed97c22291d`; the crossing-detector SHA-256 is `5cd840fc828e3104c300f67c38a0cc136ab4c9ba3f1a03b565a73b38843f1a98` at repository commit `ce69f0ed37958cb7ff1a83a4cca0b7b1a757bbdc`.

## Controlled comparison and statistical experiments

- The joint-versus-two-stage comparison is a controlled **pre-density, pre-endpoint-anchoring** experiment. It is exploratory and not mixed into the current scheme table.
- Ablation statistics use 30 runs per variant under an older free-elevation profile objective. Mean normalized objective decreases from 0.9133 (JS) to 0.6808 (full IJS), but JS and IJS use approximately 100,200 and 300,400 NFE; 25.45% is therefore a same-generation observation, not an equal-budget effect.
- Historical p-values are independent-sample Mann–Whitney/rank-sum values, not paired Wilcoxon signed-rank results; they are unadjusted for multiple comparisons.
- The six-problem benchmark uses 10 runs per algorithm/problem under the older free-endpoint objective and unequal NFE. Version 2 reports mean (SD) and explicitly treats it as exploratory.
- Sensitivity plots contain 237 re-optimized scenarios under the same older free-endpoint, pre-density model. They are not evidence of current-model robustness.

## Scope limitations retained in the manuscripts

- The quasi-natural DEM is reconstructed from the available corridor data; its reported validation error is about 8.8 m and it is not a final survey surface.
- OSM completeness is not equivalent to a field asset inventory. Missing obstacle elevations prevent a full vertical-clearance check.
- Building-density Tier 2 is a hard exclusion; Tier 1 is a soft planning proxy, not parcel-level acquisition compensation.
- Energy currently represents one anchored representative direction. Fuel and electric traffic are both monetized, but a two-direction operating profile is future work.
- The mathematical model includes constructive grade enforcement and endpoint elevation anchoring, but final design still requires explicit clothoids, vertical curves, sight distance, drainage and geotechnical verification.
- The latest density-enabled current scheme is one reported optimization run. Historical ablation/benchmark experiments do not supply statistical robustness for it.
- The stored current vector has 275 slots, but its reserved start-elevation gene is ignored when endpoints are tied; there are 274 active variables.
- `W=500 m` is the first sine-mode amplitude, not a pointwise hard corridor bound. The selected line remains within the displayed ±500 m context band.

## Integrity checks

- All headline percentages were recalculated from raw JSON values.
- `tables/paper_numbers.csv` contains 210 machine-readable values with evidence-scope labels and source SHA-256 hashes; `tables/source_manifest.csv` records the source set.
- Figure-generation code writes only under this version directory.
- Existing experiment results and unrelated repository assets were not modified.
- Author identity, funding and repository DOI placeholders must be resolved before submission.
