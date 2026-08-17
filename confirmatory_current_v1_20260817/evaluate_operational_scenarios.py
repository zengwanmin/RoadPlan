# -*- coding: utf-8 -*-
"""Post-optimal operational scenarios on all confirmed IJS designs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model_adapter import OJ, existing_reduced, expand_reduced, make_context


HERE = Path(__file__).resolve().parent
RESULTS, TABLES, FIGURES = HERE / "results", HERE / "tables", HERE / "figures"
SCENARIOS = {
    "base": {}, "EV10": {"ev": 0.10}, "EV50": {"ev": 0.50},
    "traffic_growth2": {"traffic_growth": 0.02},
    "traffic_growth4": {"traffic_growth": 0.04},
    "fuel_price_growth2": {"fuel_price_growth": 0.02},
    "elec_price_growth2": {"elec_price_growth": 0.02},
    "both_price_growth2": {"fuel_price_growth": 0.02, "elec_price_growth": 0.02},
    "fuel_save10": {"fuel_save": 0.10},
    "elec_save10": {"elec_save": 0.10},
}


raw = json.loads((RESULTS / "confirmatory_raw.json").read_text(encoding="utf-8"))
designs = sorted((r for r in raw["records"] if r["algorithm"] == "V5_IJS"),
                 key=lambda r: r["repetition"])
if len(designs) != raw["config"]["n_runs"]:
    raise SystemExit("Missing confirmed IJS designs")
pc = make_context()
x_a = expand_reduced(existing_reduced(pc))
rng = np.random.default_rng(20260819)
rows, samples = [], {}
for name, scenario in SCENARIOS.items():
    C_a, E_a, _, _ = OJ.objectives_joint(x_a, pc, scenario=scenario)
    values = []
    for record in designs:
        x = expand_reduced(np.asarray(record["best_x_reduced"], float))
        C, E, pen, info = OJ.objectives_joint(x, pc, scenario=scenario)
        values.append(dict(repetition=record["repetition"], C=float(C), E=float(E),
                           penalty=float(pen), C_improve_pct=(C_a - C) / C_a * 100,
                           E_improve_pct=(E_a - E) / E_a * 100))
    samples[name] = values
    c = np.array([x["C_improve_pct"] for x in values])
    e = np.array([x["E_improve_pct"] for x in values])
    idx = rng.integers(0, len(values), size=(20_000, len(values)))
    bc, be = np.median(c[idx], axis=1), np.median(e[idx], axis=1)
    rows.append(dict(
        scenario=name, parameters=json.dumps(scenario, sort_keys=True),
        M_A_C=float(C_a), M_A_E=float(E_a), n=len(values),
        C_improve_median_pct=float(np.median(c)),
        C_improve_ci_low=float(np.quantile(bc, .025)),
        C_improve_ci_high=float(np.quantile(bc, .975)),
        E_improve_median_pct=float(np.median(e)),
        E_improve_ci_low=float(np.quantile(be, .025)),
        E_improve_ci_high=float(np.quantile(be, .975)),
        feasible_rate=float(np.mean([x["penalty"] <= 1e-6 for x in values])),
    ))

with (TABLES / "operational_scenario_summary.csv").open(
        "w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(RESULTS / "operational_scenarios.json").write_text(
    json.dumps(dict(schema="operational-scenarios-v1", scenarios=SCENARIOS,
                    summary=rows, raw_samples=samples,
                    interpretation="post-optimal evaluation; not re-optimization"),
               ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7.8, 4.3))
y = np.arange(len(rows))
v = np.array([x["E_improve_median_pct"] for x in rows])
lo = v - np.array([x["E_improve_ci_low"] for x in rows])
hi = np.array([x["E_improve_ci_high"] for x in rows]) - v
ax.errorbar(v, y, xerr=np.vstack((lo, hi)), fmt="o", color="#E76F51",
            ecolor="#F4A261", capsize=3)
ax.axvline(0, color="0.4", ls="--", lw=1)
ax.set_yticks(y, [x["scenario"] for x in rows]); ax.invert_yaxis()
ax.set_xlabel("Energy-cost improvement versus M-A (%), median [95% CI]")
ax.grid(axis="x", alpha=.2); fig.tight_layout()
fig.savefig(FIGURES / "operational_scenario_energy.pdf")
fig.savefig(FIGURES / "operational_scenario_energy.png", dpi=300)
plt.close(fig)
print("Wrote current-model post-optimal operational scenarios")

