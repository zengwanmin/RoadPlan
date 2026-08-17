# v2 revision notes after independent v1 review

Review source: `../v1_20260816/review/agent_review.md`.

## Resolved in v2

- Corrected M-A Tier-1 from a hard-coded 2.001 km to JSON-derived 2.010162 km; the reduction is now 20.85%.
- Corrected the nominal 275-dimensional claim: the stored vector has 275 slots but one endpoint gene is inactive, leaving 274 active variables.
- Clarified that `W=500 m` is a spectral first-mode amplitude, not a pointwise hard corridor boundary.
- Replaced “feasible knee solution” in the hand-drawn flowchart with “entropy-weighted compromise”.
- Rewrote the abstract below 250 English words and removed unfair algorithm-superiority claims.
- Added curvature, component LCC, earthwork, fuel, EV, penalty, entropy-weight and IJS update equations plus a principal parameter table.
- Defined the slope-hazard index and added fuel/electric energy components to the current result table.
- Replaced best-of-10 algorithm values with mean (SD), named the actual independent-sample rank tests and exposed exact NFE differences.
- Relabelled ablation, algorithm and 237-point sensitivity evidence as historical free-endpoint/pre-density exploratory evidence.
- Added `paper_numbers.csv`, a source hash manifest and automatic assertions for headline values.

## Unresolved experimental P0 items

These require new, high-cost optimization runs and cannot be repaired editorially:

1. Remove the inactive endpoint gene and freeze one authoritative endpoint-anchored+density objective.
2. Replicate the current M-C with 20–30 independent seeds.
3. Rerun ablation and algorithm comparisons under identical objective-call budgets, paired seeds, multiplicity control, effect sizes and confidence intervals.
4. Rerun key sensitivity scenarios with the current endpoint/density formulation.
5. Add transport-model component tests: 1D versus 2D DEM, fixed versus OSM-triggered structures, density off/on and one-way versus two-way energy.

Until these experiments are complete, v2 is an honest review-ready draft and experiment specification, not a final Q1 submission package.
