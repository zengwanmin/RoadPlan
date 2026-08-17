# Confirmatory current-model experiments

This isolated directory closes the experimental P0 items identified by the v3 independent review. It does not edit the upstream optimizer or overwrite historical evidence.

## Frozen scope

- Upstream repository commit: `ce69f0ed`
- Current fixed-endpoint, DEM/OSM, endogenous-crossing and density-enabled model
- Non-redundant 273-dimensional quotient parameterization
- Frozen entropy weights/scales and fixed hard feasibility treatment
- Exact, common 600,000-NFE budget
- 20 paired initial populations, each persisted with SHA-256
- Full 2^3 Tent/Lévy/DE ablation plus JS, GA, PSO, GWO and NSGA-II

The scientific rationale and pre-specified tests are in `PROTOCOL.md`.

## Files

- `model_adapter.py`: exact quotient mapping and frozen objective adapter
- `algorithms_nfe.py`: exact-NFE implementations
- `run_confirmatory.py`: resumable multiprocessing runner
- `validate_results.py`: fail-closed raw-result and provenance audit
- `analyze_confirmatory.py`: paired statistics, multiplicity correction and figures
- `run_collaboration.py`: exact-budget plan/profile structural ablation
- `run_robustness.py`: eight pre-specified current-model robustness reruns
- `evaluate_operational_scenarios.py`: post-optimal fleet/demand/price scenarios
- `validate_*.py`, `analyze_*.py`: fail-closed audits and paired analyses
- `tests/test_protocol.py`: exact-NFE and model-equivalence regression tests
- `requirements-lock.txt`: direct runtime dependency versions
- `CODE_FREEZE.json`: SHA-256 of the exact main-run code and upstream baseline
- `results/initial_populations/`: persisted paired initial populations

## Reproduction

```bash
python tests/test_protocol.py
python run_confirmatory.py --smoke --workers 2 --fresh
python validate_results.py results/confirmatory_smoke.json
python run_confirmatory.py --workers 12
python validate_results.py
python analyze_confirmatory.py
python run_collaboration.py --workers 12
python validate_collaboration.py
python analyze_collaboration.py
python run_robustness.py --workers 12
python validate_robustness.py
python analyze_robustness.py
python evaluate_operational_scenarios.py
python make_manifest.py
```

The long runner checkpoints every completed task in
`results/confirmatory_raw.partial.json`. Re-running the same command resumes only when algorithms, run count, population size and budget match. Post-run diagnostic evaluations are explicitly outside the optimization NFE and are not exposed to the optimizers.

## Interpretation guardrail

The fixed hard-penalty confirmatory target is intentionally identical for all algorithms. It differs from the historical two-stage soft/hard search schedule, so these runs support current-model equal-budget comparisons; they are not claimed as bitwise replications of the original single M-C trajectory.
