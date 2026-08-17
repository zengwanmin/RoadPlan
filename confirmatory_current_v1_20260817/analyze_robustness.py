# -*- coding: utf-8 -*-
"""Paired robustness summaries relative to base V5_IJS repetitions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon


HERE = Path(__file__).resolve().parent
RESULTS, TABLES, FIGURES = HERE / "results", HERE / "tables", HERE / "figures"


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    running = 0.0
    m = len(p)
    for j, i in enumerate(order):
        running = max(running, (m - j) * p[i])
        out[i] = min(1.0, running)
    return out


def rbc(d):
    d = np.asarray(d)
    d = d[np.abs(d) > 1e-14]
    if not len(d):
        return 0.0
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def paired(d, rng):
    d = np.asarray(d, float)
    p = 1.0 if np.all(np.abs(d) <= 1e-14) else float(wilcoxon(d).pvalue)
    sample = np.median(d[rng.integers(0, len(d), size=(20_000, len(d)))], axis=1)
    lo, hi = np.quantile(sample, [.025, .975])
    return dict(n=len(d), median_diff=float(np.median(d)), ci_low=float(lo),
                ci_high=float(hi), p_raw=p, rank_biserial=rbc(d))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


main = json.loads((RESULTS / "confirmatory_raw.json").read_text(encoding="utf-8"))
rob = json.loads((RESULTS / "robustness_raw.json").read_text(encoding="utf-8"))
n = rob["config"]["n_runs"]
base = {r["repetition"]: r for r in main["records"]
        if r["algorithm"] == "V5_IJS" and r["repetition"] < n}
index = {(r["scenario"], r["repetition"]): r for r in rob["records"]}
scenarios = rob["config"]["scenarios"]
rng = np.random.default_rng(20260818)
rows = []
for scenario in scenarios:
    rr = [index[(scenario, i)] for i in range(n)]
    dF = np.array([rr[i]["F_hard"] - base[i]["F_hard"] for i in range(n)])
    stat = paired(dF, rng)
    rows.append(dict(
        scenario=scenario, direction="scenario_minus_base; zero_is_robust", **stat,
        feasible_rate=np.mean([x["feasible"] for x in rr]),
        median_F=np.median([x["F_hard"] for x in rr]),
        median_C_change_pct=np.median(
            [(rr[i]["C"] / base[i]["C"] - 1) * 100 for i in range(n)]),
        median_E_change_pct=np.median(
            [(rr[i]["E"] / base[i]["E"] - 1) * 100 for i in range(n)]),
        median_Tier1_change_km=np.median(
            [rr[i]["L_dense1_km"] - base[i]["L_dense1_km"] for i in range(n)]),
    ))
for row, p_adjusted in zip(rows, holm([x["p_raw"] for x in rows])):
    row["p_holm"] = float(p_adjusted)
write_csv(TABLES / "robustness_paired_summary.csv", rows)
(RESULTS / "robustness_analysis.json").write_text(
    json.dumps(dict(schema="robustness-analysis-v1", rows=rows,
                    multiplicity="Holm across 8 pre-specified scenario comparisons",
                    bootstrap="20,000 paired resamples, seed 20260818"),
               ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7.6, 4.2))
y = np.arange(len(rows))
values = np.array([x["median_diff"] for x in rows])
low = values - np.array([x["ci_low"] for x in rows])
high = np.array([x["ci_high"] for x in rows]) - values
ax.errorbar(values, y, xerr=np.vstack((low, high)), fmt="o", color="#264653",
            ecolor="#2A9D8F", capsize=3)
ax.axvline(0, color="0.4", lw=1, ls="--")
ax.set_yticks(y, [x["scenario"] for x in rows])
ax.invert_yaxis()
ax.set_xlabel("Paired change in frozen F (scenario - base), median [95% CI]")
ax.grid(axis="x", alpha=.2)
fig.tight_layout()
fig.savefig(FIGURES / "current_model_robustness_forest.pdf")
fig.savefig(FIGURES / "current_model_robustness_forest.png", dpi=300)
plt.close(fig)
print("Wrote paired robustness analysis")

