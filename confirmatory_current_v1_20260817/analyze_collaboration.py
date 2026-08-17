# -*- coding: utf-8 -*-
"""Paired current-model collaboration ablation analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon


HERE = Path(__file__).resolve().parent
RESULTS, TABLES, FIGURES = HERE / "results", HERE / "tables", HERE / "figures"
main = json.loads((RESULTS / "confirmatory_raw.json").read_text(encoding="utf-8"))
raw = json.loads((RESULTS / "collaboration_raw.json").read_text(encoding="utf-8"))
n = raw["config"]["n_runs"]
joint = {r["repetition"]: r for r in main["records"]
         if r["algorithm"] == "V5_IJS" and r["repetition"] < n}
index = {(r["mode"], r["repetition"]): r for r in raw["records"]}
rng = np.random.default_rng(20260820)
rows = []
for mode in raw["config"]["modes"]:
    rr = [index[(mode, i)] for i in range(n)]
    d = np.array([rr[i]["F_hard"] - joint[i]["F_hard"] for i in range(n)])
    p = 1.0 if np.all(np.abs(d) <= 1e-14) else float(wilcoxon(d).pvalue)
    nz = d[np.abs(d) > 1e-14]
    ranks = rankdata(np.abs(nz)) if len(nz) else np.array([])
    effect = (float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
              if len(nz) else 0.0)
    boot = np.median(d[rng.integers(0, n, size=(20_000, n))], axis=1)
    rows.append(dict(
        mode=mode, direction="restricted_minus_joint; positive_favors_joint",
        n=n, median_diff=float(np.median(d)), ci_low=float(np.quantile(boot, .025)),
        ci_high=float(np.quantile(boot, .975)), p_raw=p,
        rank_biserial=effect, feasible_rate=np.mean([x["feasible"] for x in rr]),
        F_median=np.median([x["F_hard"] for x in rr]),
    ))
order = np.argsort([r["p_raw"] for r in rows]); running = 0.0
for j, idx in enumerate(order):
    running = max(running, (len(rows) - j) * rows[idx]["p_raw"])
    rows[idx]["p_holm"] = min(1.0, running)
with (TABLES / "collaboration_ablation.csv").open(
        "w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
(RESULTS / "collaboration_analysis.json").write_text(
    json.dumps(dict(schema="collaboration-analysis-v1", rows=rows,
                    multiplicity="Holm across three pre-specified comparisons",
                    bootstrap="20,000 paired resamples, seed 20260820"),
               ensure_ascii=False, indent=2), encoding="utf-8")
fig, ax = plt.subplots(figsize=(6.8, 3.2))
y = np.arange(len(rows)); v = np.array([r["median_diff"] for r in rows])
lo = v - np.array([r["ci_low"] for r in rows]); hi = np.array([r["ci_high"] for r in rows]) - v
ax.errorbar(v, y, xerr=np.vstack((lo, hi)), fmt="o", color="#264653",
            ecolor="#457B9D", capsize=3)
ax.axvline(0, color=".4", ls="--", lw=1)
ax.set_yticks(y, [r["mode"] for r in rows]); ax.invert_yaxis()
ax.set_xlabel("Paired F difference (restricted - joint), median [95% CI]")
ax.grid(axis="x", alpha=.2); fig.tight_layout()
fig.savefig(FIGURES / "current_model_collaboration_ablation.pdf")
fig.savefig(FIGURES / "current_model_collaboration_ablation.png", dpi=300)
plt.close(fig)
print("Wrote current-model collaboration ablation analysis")

