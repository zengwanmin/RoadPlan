# -*- coding: utf-8 -*-
"""Pre-specified paired statistics and publication figures for confirmed runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, rankdata, wilcoxon


HERE = Path(__file__).resolve().parent
RESULTS, TABLES, FIGURES = HERE / "results", HERE / "tables", HERE / "figures"
PRIMARY = "V5_IJS"
MAIN_COMPARATORS = ["V1_JS", "GA", "PSO", "GWO", "NSGA-II"]
ABLATION = ["V1_JS", "V2_JS+Tent", "V3_JS+Levy", "V4_JS+DE",
            "V6_JS+Tent+Levy", "V7_JS+Tent+DE", "V8_JS+Levy+DE"]
VARIANT_BITS = {
    "V1_JS": (0, 0, 0), "V2_JS+Tent": (1, 0, 0),
    "V3_JS+Levy": (0, 1, 0), "V4_JS+DE": (0, 0, 1),
    "V6_JS+Tent+Levy": (1, 1, 0), "V7_JS+Tent+DE": (1, 0, 1),
    "V8_JS+Levy+DE": (0, 1, 1), "V5_IJS": (1, 1, 1),
}


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _holm(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _paired_rank_biserial(diff):
    d = np.asarray(diff, float)
    d = d[np.abs(d) > 1e-14]
    if not len(d):
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    pos, neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def _paired_test(diff, rng, n_boot=20_000):
    d = np.asarray(diff, float)
    if len(d) < 2:
        return dict(n_pairs=len(d), median_diff=float(np.median(d)) if len(d) else np.nan,
                    ci_low=np.nan, ci_high=np.nan, p_raw=np.nan,
                    rank_biserial=np.nan)
    if np.all(np.abs(d) <= 1e-14):
        p = 1.0
    else:
        p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                           method="auto").pvalue)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = np.median(d[idx], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(n_pairs=len(d), median_diff=float(np.median(d)),
                ci_low=float(lo), ci_high=float(hi), p_raw=p,
                rank_biserial=_paired_rank_biserial(d))


def _group(records):
    return {(r["algorithm"], r["repetition"]): r for r in records}


def _summary(records, algorithms):
    rows = []
    for name in algorithms:
        rr = [r for r in records if r["algorithm"] == name]
        f = np.array([r["F_hard"] for r in rr])
        C = np.array([r["C"] for r in rr]) / 1e8
        E = np.array([r["E"] for r in rr]) / 1e8
        rows.append(dict(
            algorithm=name, n=len(rr), feasible_n=sum(r["feasible"] for r in rr),
            feasible_rate=np.mean([r["feasible"] for r in rr]),
            F_median=np.median(f), F_q1=np.quantile(f, .25), F_q3=np.quantile(f, .75),
            C_median_1e8=np.median(C), C_q1_1e8=np.quantile(C, .25), C_q3_1e8=np.quantile(C, .75),
            E_median_1e8=np.median(E), E_q1_1e8=np.quantile(E, .25), E_q3_1e8=np.quantile(E, .75),
            Tier1_median_km=np.median([r["L_dense1_km"] for r in rr]),
            runtime_median_s=np.median([r["runtime_s"] for r in rr]),
        ))
    return rows


def _paired_family(index, comparators, metric, rng):
    rows = []
    for competitor in comparators:
        reps = sorted(set(r for a, r in index if a == PRIMARY) &
                      set(r for a, r in index if a == competitor))
        # Positive difference means the primary IJS has a smaller outcome.
        diff = np.array([index[(competitor, r)][metric] - index[(PRIMARY, r)][metric]
                         for r in reps], float)
        stat = _paired_test(diff, rng)
        primary_only = sum(index[(PRIMARY, r)]["feasible"] and
                           not index[(competitor, r)]["feasible"] for r in reps)
        comparator_only = sum(index[(competitor, r)]["feasible"] and
                              not index[(PRIMARY, r)]["feasible"] for r in reps)
        discordant = primary_only + comparator_only
        feasibility_p = (float(binomtest(primary_only, discordant, 0.5).pvalue)
                         if discordant else 1.0)
        rows.append(dict(primary=PRIMARY, comparator=competitor, metric=metric,
                         direction="comparator_minus_IJS_positive_favors_IJS",
                         IJS_feasible_comparator_not=primary_only,
                         comparator_feasible_IJS_not=comparator_only,
                         feasibility_exact_p_raw=feasibility_p, **stat))
    adjusted = _holm([r["p_raw"] for r in rows])
    adjusted_feas = _holm([r["feasibility_exact_p_raw"] for r in rows])
    for r, p, pf in zip(rows, adjusted, adjusted_feas):
        r["p_holm"] = float(p)
        r["feasibility_exact_p_holm"] = float(pf)
    return rows


def _factorial_main_effects(index, rng):
    reps = sorted(set(r for a, r in index if a == PRIMARY))
    rows = []
    for bit, component in enumerate(("Tent", "Levy", "DE")):
        effects = []
        for rep in reps:
            on = [index[(a, rep)]["F_hard"] for a, bits in VARIANT_BITS.items()
                  if bits[bit] == 1 and (a, rep) in index]
            off = [index[(a, rep)]["F_hard"] for a, bits in VARIANT_BITS.items()
                   if bits[bit] == 0 and (a, rep) in index]
            if len(on) == 4 and len(off) == 4:
                effects.append(float(np.mean(off) - np.mean(on)))
        stat = _paired_test(np.array(effects), rng)
        rows.append(dict(component=component,
                         direction="mean_off_minus_mean_on_positive_is_improvement", **stat))
    adjusted = _holm([r["p_raw"] for r in rows])
    for r, p in zip(rows, adjusted):
        r["p_holm"] = float(p)
    return rows


def _step_curve(curve, grid):
    nfe = np.array([x[0] for x in curve], int)
    val = np.array([x[1] for x in curve], float)
    idx = np.searchsorted(nfe, grid, side="right") - 1
    idx = np.clip(idx, 0, len(val) - 1)
    return val[idx]


def _figures(records, cfg):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    main = [PRIMARY] + MAIN_COMPARATORS
    data = [[r["F_hard"] for r in records if r["algorithm"] == a] for a in main]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bp = ax.boxplot(data, tick_labels=main, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#E76F51", "#457B9D", "#2A9D8F",
                                          "#E9C46A", "#8D99AE", "#6D597A"]):
        patch.set_facecolor(color); patch.set_alpha(.72)
    ax.set_ylabel("Frozen scalar objective F (lower is better)")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "equal_nfe_algorithm_boxplot.pdf")
    fig.savefig(FIGURES / "equal_nfe_algorithm_boxplot.png", dpi=300)
    plt.close(fig)

    grid = np.linspace(cfg["pop_size"], cfg["budget"], 201).astype(int)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    colors = plt.cm.tab10(np.linspace(0, .8, len(main)))
    for name, color in zip(main, colors):
        rr = [r for r in records if r["algorithm"] == name]
        curves = np.array([_step_curve(r["curve"], grid) for r in rr])
        med, q1, q3 = np.median(curves, 0), np.quantile(curves, .25, 0), np.quantile(curves, .75, 0)
        ax.plot(grid, med, label=name, color=color, lw=1.7)
        ax.fill_between(grid, q1, q3, color=color, alpha=.12)
    ax.set_xlabel("Objective evaluations (NFE)")
    ax.set_ylabel("Best-so-far F, median [IQR]")
    ax.grid(alpha=.2); ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "equal_nfe_convergence.pdf")
    fig.savefig(FIGURES / "equal_nfe_convergence.png", dpi=300)
    plt.close(fig)

    variants = list(VARIANT_BITS)
    ab = [[r["F_hard"] for r in records if r["algorithm"] == a] for a in variants]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.boxplot(ab, tick_labels=[x.replace("JS+", "+") for x in variants],
               patch_artist=True, showmeans=True)
    ax.set_ylabel("F (lower is better)")
    ax.tick_params(axis="x", rotation=24)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "current_model_factorial_ablation.pdf")
    fig.savefig(FIGURES / "current_model_factorial_ablation.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/confirmatory_raw.json")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    path = HERE / args.path
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg, records = raw["config"], raw["records"]
    expected = len(cfg["algorithms"]) * cfg["n_runs"]
    if len(records) != expected and not args.allow_partial:
        raise SystemExit(f"Refusing incomplete data: {len(records)}/{expected}")
    TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260817)
    index = _group(records)
    summary = _summary(records, cfg["algorithms"])
    main = _paired_family(index, MAIN_COMPARATORS, "F_hard", rng)
    secondary_C = _paired_family(index, MAIN_COMPARATORS, "C", rng)
    secondary_E = _paired_family(index, MAIN_COMPARATORS, "E", rng)
    ablation = _paired_family(index, ABLATION, "F_hard", rng)
    effects = _factorial_main_effects(index, rng)
    _write_csv(TABLES / "algorithm_summary.csv", summary)
    _write_csv(TABLES / "paired_main_comparisons.csv", main)
    _write_csv(TABLES / "paired_secondary_cost.csv", secondary_C)
    _write_csv(TABLES / "paired_secondary_energy.csv", secondary_E)
    _write_csv(TABLES / "paired_ablation_comparisons.csv", ablation)
    _write_csv(TABLES / "factorial_main_effects.csv", effects)
    _figures(records, cfg)
    report = dict(schema="confirmatory-analysis-v1", complete=len(records) == expected,
                  n_records=len(records), expected=expected, summary=summary,
                  paired_main=main, paired_ablation=ablation,
                  paired_secondary_cost=secondary_C,
                  paired_secondary_energy=secondary_E,
                  factorial_main_effects=effects,
                  multiplicity="Holm within each pre-specified family",
                  bootstrap="20,000 paired resamples, seed 20260817")
    (RESULTS / "statistical_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(summary)} summaries and confirmatory paired statistics")


if __name__ == "__main__":
    main()
