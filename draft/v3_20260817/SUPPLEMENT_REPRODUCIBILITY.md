# Reproducibility supplement for v3

This supplement maps the manuscript notation to the saved current implementation. It does not upgrade the single current run into a multi-seed experiment.

## Authoritative entry points

- Main run: `优化方案对比（平面、纵断面联合协同优化）/run_joint.py`
- Joint geometry/objective: `objective_joint.py`; common cost/energy model: `objective.py`
- Parameters: `params.py`; optimizer: `algorithms.py`
- Crossings/density/terrain: `crossings.py`, `building_mask.py`, `dem.py`, `data_loader.py`
- Current result: `results/joint_results_w500_dens.json`

`tables/source_manifest.csv` records SHA-256 hashes for these files and the principal input layers. The repository commit captured by the v2 audit is `ce69f0ed37958cb7ff1a83a4cca0b7b1a757bbdc`.

## Geometry and optimization state

| Item | Value | Code variable / source |
|---|---:|---|
| Plan modes | 50 | `objective_joint.N_MODE` |
| Spectral coefficient bound | `|a_k| <= 500/k^2` m | `PLANE_MODE_A1`, current result `meta.corridor_half_w` (legacy key name) |
| Plan reconstruction/evaluation | 150 m / 10 m | `STEP_PLANE_CTRL_M`, `STEP_EVAL_M` |
| Profile station spacing | about 100 m | `STEP_PROFILE_CTRL_M` |
| Stored/profile state | 275 slots / 225-slot profile block | current result `meta.dim`, `meta.M_prof` |
| Referenced/independent state | 274 referenced genes / at most 273 locally independent degrees of freedom | one ignored endpoint slot plus one grade-centering zero direction |
| Population / iterations | 200 / 1,000 | `params.ALGO.pop_size`, `run_joint.MAX_ITER` |
| Pareto weight points | 21 | current result `meta.n_pareto` |
| Initial-population seed | 2025 | `objective_joint.joint_baseline` default |
| Search seed | 1000, hard phase 1001 | `run_joint.py`, `run_ijs_two_phase` |
| Initial population | `default_rng(2025)` uniform `[0,1]`, with M-A seed inserted as row 0 | `joint_baseline`; shared by M-B/M-C/front tasks |
| Initial scalar weights | `w_C=0.50838155`, `w_E=0.49161845` | current result `meta` |
| Front decision weights | `w_C=0.42925073`, `w_E=0.57074927` | current result `entropy_point` |

The current result file does not store a separate hash of the initial-population array. Future confirmatory runs should persist that array or its SHA-256 per seed.

## Engineering, traffic and energy parameters

| Parameter | Value | Code variable |
|---|---:|---|
| Design speed / formation width | 100 km/h / 25.5 m | `CASE.design_speed_kmh`, `CASE.road_width_m` |
| Minimum radius / maximum grade | 400 m / 0.04 | `FLAT_STD_100.R_extreme_m`, `LONG_STD_100.grade_max` |
| Analysis period / discount rate | 30 yr / 0.05 | `LCC.analysis_years`, `LCC.bank_rate` |
| Land / subgrade / pavement | 58,641 RMB/mu / 200 / 30,000 RMB/m | `COST_UNIT` |
| Cut / fill / haul / side slope | 30 / 25 RMB/m3 / 100,000 RMB / 1.5 | `EARTHWORK` |
| Bridge / tunnel | 156 / 270 million RMB/km | `BRIDGE_TUNNEL` |
| Structural hard trigger | 30 m fill or cut | `fill_height_bridge_m`, `cut_depth_tunnel_m` |
| Crossing extension / merge / minimum skew | 75 m each side / 100 m / 15 deg | `BRIDGE_TUNNEL.crossing_trigger` |
| AADT / fuel share / EV share | 30,000 / 0.70 / 0.30 | `TRAFFIC` |
| Fuel/electricity price | 8 RMB/L / 0.8 RMB/kWh | `ENERGY_PRICE` |
| Fuel car mass / `Cd` / area / `Cr` | 1,500 kg / 0.28 / 2.0 m2 / 0.015 | `FUEL_CAR` |
| Fuel `NV` / `CS` / efficiency / internal power | 4 / 43 / 0.85 / 27 hp | `FUEL_CAR` |
| EV mass / `Cd` / area / `Cr` | 1,800 kg / 0.28 / 2.2 m2 / 0.012 | `EV` |
| EV drive efficiencies / auxiliaries | 0.90 x 0.95 / 15 kWh per 100 km | `EV.ea`, `EV.eb`, `EV.EH_kwh_per_100` |
| Air density / gravity | 1.25 kg/m3 / 9.8 m/s2 | `PHYS` |
| Maximum superelevation `S_p` | 0.08 | `FLAT_STD_100.superelev_max` |
| Maintenance `gamma/tau/c_soil/alpha/beta` | 10 / 1e-6 / 2 / 0.3 / 0.3 | `MAINTENANCE` |

Density V1 uses building-footprint coverage, Gaussian sigma 200 m, margin 1.15, 100-m morphological closing and 1-ha minimum cluster. The exported `density_tiers_V1.npz` stores the realized thresholds and grid metadata. Tier-2 penalty depth uses a 100-m reference, clipped at 5; Tier-1 scalar weight is 0.22.

## IJS settings

- Tent map uses `mu=1.99` in the implemented optimizer; `params.py` retains the earlier documentation value 2.0 and should be synchronized before new runs.
- Levy stable exponent `beta=1.5`; search-domain scale 0.01.
- DE crossover probability `CR=0.5`; greedy/elitist acceptance after trials.
- Two-phase penalty scale: 0.3 for the first 500 iterations and 3.0 for the final 500.
- Current IJS performs about three trial evaluations per individual per generation. Fair comparisons must stop on a shared objective-call counter, not a shared generation count.

## Local reproduction

From `draft/v3_20260817`:

```bash
python3 scripts/make_v1_figures.py
python3 scripts/make_case_data_layers.py
python3 scripts/make_historical_spectral_plot.py
python3 scripts/audit_paper_numbers.py
(cd en && tectonic main_en.tex)
(cd zh && tectonic main_zh.tex)
```

The second command extracts source values and checks only the five stated headline relationships. It does not parse all LaTeX numerals. Exact reproduction of the current optimization itself requires the repository's original Python environment and restricted project data; new P0 experiments should use a locked environment file and persist raw per-evaluation logs.
