# v3 revision notes after independent v2 review

Review source: `../v2_20260817/review/agent_review.md`.

## Resolved in v3

- Distinguished 275 stored slots, 274 referenced genes and at most 273 locally independent degrees of freedom; the centering zero direction is now explicit.
- Replaced the simplified earthwork minimum with the implemented stationwise hard 30-m/cost trigger and trapezoidal integration.
- Added the implemented non-negative truncation to the EV drive-energy expression.
- Renamed the numerical artifact as `source_value_snapshot.csv`; the script now states that only five headline relationships are asserted and does not claim to parse all LaTeX numbers.
- Expanded `source_manifest.csv` to current objective/algorithm/geometry code and primary DEM/OSM/alignment inputs.
- Replaced residual “corridor width” prose with spectral-width terminology and explained the legacy internal figure label.
- Downgraded residual contribution/comparison wording to historical associations, screening and hypothesis generation.
- Added a reproducibility supplement listing model parameters, code variables, seeds, initial-population construction and hashes.
- Added a verified 2026 hybrid reinforcement-learning/metaheuristic highway-alignment reference.
- Added an immutable-result metadata sidecar for the saved JSON's stale bridge/corridor note and the `mu_tent` documentation mismatch.
- Synchronized the Chinese abstract with the 275/274/273 definition, expanded five truncated author lists, and removed two uncited statistical references.

## Unresolved experimental P0 items

These require new, high-cost optimization runs and cannot be repaired editorially:

1. Reparameterize the profile to remove both the inactive endpoint slot and grade-centering zero direction, then freeze one authoritative endpoint-anchored+density objective.
2. Replicate the current M-C with 20–30 independent seeds.
3. Rerun ablation and algorithm comparisons under identical objective-call budgets, paired seeds, multiplicity control, effect sizes and confidence intervals.
4. Rerun key sensitivity scenarios with the current endpoint/density formulation.
5. Add transport-model component tests: 1D versus 2D DEM, fixed versus OSM-triggered structures, density off/on and one-way versus two-way energy.

Until these experiments are complete, v3 is an honest review-ready draft and experiment specification, not a final Q1 submission package. No wording change can substitute for these P0 runs.
