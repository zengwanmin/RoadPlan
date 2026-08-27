# v5 — c1327df fixed-evidence manuscript

This manuscript version is written exclusively against Git commit
`c1327df1ea2dc64cdde826bcb1df7141d80a0533`.

## Evidence boundary

- No optimization or supplementary experiment is started for this version.
- The authoritative engineering result is `优化方案对比（平面、纵断面联合协同优化）/results/joint_results.json` at the pinned commit.
- The sequential comparison is `优化方案对比（平面、纵断面联合协同优化）/results/twostage_results.json` at the pinned commit.
- Ablation, algorithm-comparison and sensitivity statements use only files already committed in `c1327df`.
- Later endpoint-anchored or building-density variants are outside this manuscript's evidence boundary.

## Main result

The feasible entropy-selected joint scheme reduces life-cycle cost from
26.4061 to 24.2239 hundred-million RMB (8.26%) and monetized 30-year traffic
energy from 13.9460 to 13.4766 hundred-million RMB (3.37%). Its minimum
horizontal radius is 401.03 m and its stored hard-constraint penalty is zero.

The two-stage comparison reaches lower raw cost and energy but is infeasible:
its minimum radius is 397.06 m against the 400 m requirement and its stored
penalty is 0.0735928. It is therefore evidence for the need for coupled search,
not a feasible preferred design.

## Reproduction of manuscript-only figures

Run `python scripts/make_c132_figures.py`. The script reads the two JSON files
directly from the pinned Git object and writes only manuscript figures/tables in
this directory. It does not import or execute any optimizer.

