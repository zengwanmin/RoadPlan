#!/usr/bin/env python3
"""Build manuscript-only plots from immutable c1327df Git objects.

This script performs no optimization. It reads already-committed JSON and writes
only to draft/v5_c132_20260818/{figures,tables}.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


COMMIT = "c1327df1ea2dc64cdde826bcb1df7141d80a0533"
HERE = Path(__file__).resolve().parent
VERSION = HERE.parent
REPO = VERSION.parents[1]
FIG = VERSION / "figures"
TAB = VERSION / "tables"
JOINT = "优化方案对比（平面、纵断面联合协同优化）/results/joint_results.json"
TWOSTAGE = "优化方案对比（平面、纵断面联合协同优化）/results/twostage_results.json"


def git_json(path: str) -> dict:
    raw = subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=REPO)
    return json.loads(raw)


DATA = git_json(JOINT)
TS = git_json(TWOSTAGE)
MA, MB, MC = DATA["M_A"], DATA["M_B"], DATA["M_C"]

plt.rcParams.update({
    "figure.dpi": 180,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
})


def pct(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def scheme_summary() -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 7.0), constrained_layout=True)

    a = ax[0, 0]
    a.plot(np.asarray(MA["plane_x"]) / 1000, np.asarray(MA["plane_y"]) / 1000,
           "--", color="#555555", lw=1.3, label="M-A existing")
    a.plot(np.asarray(MC["plane_x"]) / 1000, np.asarray(MC["plane_y"]) / 1000,
           color="#1565c0", lw=2.0, label="M-C joint")
    a.set(xlabel="Easting (km)", ylabel="Northing (km)", title="(a) Horizontal alignment")
    a.axis("equal")
    a.grid(alpha=.18)
    a.legend(frameon=False)

    a = ax[0, 1]
    sa = np.asarray(MA["sta"]) / 1000
    sc = np.asarray(MC["sta"]) / 1000
    a.plot(sc, MC["gz_new"], color="#9a8f79", lw=1.0, label="Terrain along M-C")
    a.plot(sa, MA["design_z"], "--", color="#555555", lw=1.1, label="M-A profile")
    a.plot(sc, MC["design_z"], color="#1565c0", lw=1.6, label="M-C profile")
    a.set(xlabel="Station (km)", ylabel="Elevation (m)", title="(b) Vertical profile")
    a.grid(alpha=.18)
    a.legend(frameon=False, fontsize=8)

    parts = ["CR", "CB", "CS", "CQ", "C_TU"]
    labels = ["Land", "Bridge/\ntunnel", "Basic\nworks", "Maintenance", "Earthwork"]
    x = np.arange(len(parts))
    width = .36
    a = ax[1, 0]
    a.bar(x - width / 2, [MA[k] / 1e8 for k in parts], width, color="#9e9e9e", label="M-A")
    a.bar(x + width / 2, [MC[k] / 1e8 for k in parts], width, color="#4b83c3", label="M-C")
    a.set_xticks(x, labels)
    a.set_ylabel(r"Cost ($10^8$ RMB)")
    a.set_title("(c) Life-cycle cost decomposition")
    a.grid(axis="y", alpha=.18)
    a.legend(frameon=False)

    labels = ["Life-cycle\ncost", "Traffic\nenergy", "Length", "Slope\nhazard"]
    changes = [pct(MC["C"], MA["C"]), pct(MC["E"], MA["E"]),
               pct(MC["L_km"], MA["L_km"]), pct(MC["Q_mean"], MA["Q_mean"])]
    a = ax[1, 1]
    bars = a.bar(np.arange(4), changes,
                 color=["#2a9d8f" if v <= 0 else "#d1495b" for v in changes])
    a.axhline(0, color="#222222", lw=.7)
    a.set_xticks(np.arange(4), labels)
    a.set_ylabel("Change from M-A (%)")
    a.set_title("(d) Engineering performance")
    for bar, value in zip(bars, changes):
        a.text(bar.get_x() + bar.get_width() / 2, value + .12, f"{value:+.2f}%",
               ha="center", va="bottom", fontsize=8)
    a.text(.50, .04, f"$R_{{min}}$={MC['Rmin']:.1f} m; penalty={MC['penalty']:.0e}",
           transform=a.transAxes, ha="center", fontsize=8)
    save(fig, "c132_scheme_summary")


def pareto_front() -> None:
    pts = [p for p in DATA["pareto_sweep"] if p["pen"] <= 1e-6]
    c = np.asarray([p["C"] for p in pts]) / 1e8
    e = np.asarray([p["E"] for p in pts]) / 1e8
    nd = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        nd[i] = not np.any((c <= c[i]) & (e <= e[i]) & ((c < c[i]) | (e < e[i])))
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(e[~nd], c[~nd], s=28, color="#bdbdbd", label="Feasible dominated")
    order = np.argsort(e[nd])
    ax.plot(e[nd][order], c[nd][order], "o-", ms=5, lw=1.4,
            color="#2a9d8f", label="Feasible non-dominated")
    ax.scatter(MC["E"] / 1e8, MC["C"] / 1e8, marker="*", s=190,
               color="#d1495b", edgecolor="white", lw=.8, zorder=5,
               label="Entropy-selected M-C")
    ax.scatter(MA["E"] / 1e8, MA["C"] / 1e8, marker="s", s=55,
               color="#555555", label="M-A existing")
    ax.set(xlabel=r"Monetized traffic energy ($10^8$ RMB)",
           ylabel=r"Life-cycle cost ($10^8$ RMB)",
           title="Feasible cost--energy solutions at the pinned c1327df model")
    ax.grid(alpha=.20)
    ax.legend(frameon=False)
    iax = inset_axes(ax, width="43%", height="43%", loc="center left",
                     bbox_to_anchor=(.05, -.02, 1, 1), bbox_transform=ax.transAxes)
    iax.scatter(e[~nd], c[~nd], s=12, color="#bdbdbd")
    iax.plot(e[nd][order], c[nd][order], "o-", ms=3, lw=1, color="#2a9d8f")
    iax.scatter(MC["E"] / 1e8, MC["C"] / 1e8, marker="*", s=70, color="#d1495b")
    iax.scatter(MA["E"] / 1e8, MA["C"] / 1e8, marker="s", s=25, color="#555555")
    iax.set_xlim(12.8, 14.1)
    iax.set_ylim(22.8, 28.3)
    iax.set_title("Decision-scale detail", fontsize=7)
    iax.tick_params(labelsize=6)
    iax.grid(alpha=.18)
    mark_inset(ax, iax, loc1=1, loc2=3, fc="none", ec="#777777", lw=.6)
    save(fig, "c132_pareto")


def write_summary() -> None:
    TAB.mkdir(exist_ok=True)
    rows = []
    for name, obj in (("M-A", MA), ("M-B", MB), ("M-C", MC), ("Two-stage", TS["M_C"])):
        rows.append([name, obj["C"] / 1e8, obj["E"] / 1e8, obj["L_km"],
                     obj["Rmin"], obj["Q_mean"], obj["penalty"]])
    with (TAB / "c132_scheme_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scheme", "C_1e8_RMB", "E_1e8_RMB", "L_km", "Rmin_m", "Q_mean", "penalty"])
        writer.writerows(rows)


if __name__ == "__main__":
    scheme_summary()
    pareto_front()
    write_summary()
    print(f"Built pinned-c132 manuscript figures in {FIG}")
