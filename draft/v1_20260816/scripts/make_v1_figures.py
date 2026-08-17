#!/usr/bin/env python3
"""Regenerate manuscript-only figures from the current RoadPlan result JSON.

This script never edits experiment outputs. It reads the endpoint-anchored,
building-density-enabled W=500 m result and writes only into draft/v1_20260816.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


HERE = Path(__file__).resolve().parent
VERSION = HERE.parent
EXP = VERSION.parents[1]
SCHEME = EXP / "优化方案对比（平面、纵断面联合协同优化）"
RESULT = SCHEME / "results" / "joint_results_w500_dens.json"
FIG = VERSION / "figures"
TAB = VERSION / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 180,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
})
COL = {"existing": "#555555", "optimized": "#1565c0", "ground": "#9a8f79",
       "bridge": "#e58e26", "tunnel": "#6a51a3", "accent": "#d1495b"}

with RESULT.open(encoding="utf-8") as f:
    DATA = json.load(f)
MA, MC = DATA["M_A"], DATA["M_C"]


def save(fig, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def percent(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def scheme_summary() -> None:
    """Plan/profile, LCC decomposition and headline performance."""
    xa, ya = np.asarray(MA["plane_x"]), np.asarray(MA["plane_y"])
    xc, yc = np.asarray(MC["plane_x"]), np.asarray(MC["plane_y"])
    sa, sc = np.asarray(MA["sta"]) / 1000.0, np.asarray(MC["sta"]) / 1000.0
    za, zc = np.asarray(MA["design_z"]), np.asarray(MC["design_z"])
    ga, gc = np.asarray(MA["gz_new"]), np.asarray(MC["gz_new"])

    fig, ax = plt.subplots(2, 2, figsize=(11.2, 7.0), constrained_layout=True)
    a = ax[0, 0]
    a.plot(xa / 1000, ya / 1000, color=COL["existing"], lw=1.3, ls="--",
           label="M-A existing")
    a.plot(xc / 1000, yc / 1000, color=COL["optimized"], lw=2.0,
           label="M-C current")
    a.set(xlabel="Easting (km)", ylabel="Northing (km)", title="(a) Horizontal alignment")
    a.axis("equal"); a.grid(alpha=.18); a.legend(frameon=False)

    a = ax[0, 1]
    a.plot(sc, gc, color=COL["ground"], lw=1.0, alpha=.85,
           label="Quasi-natural terrain (M-C path)")
    a.plot(sa, za, color=COL["existing"], lw=1.1, ls="--", label="M-A profile")
    a.plot(sc, zc, color=COL["optimized"], lw=1.6, label="M-C profile")
    a.scatter([sc[0], sc[-1]], [zc[0], zc[-1]], s=22, color=COL["accent"], zorder=4,
              label="Anchored tie-in elevations")
    a.set(xlabel="Station (km)", ylabel="Elevation (m)", title="(b) Vertical profile")
    a.grid(alpha=.18); a.legend(frameon=False, fontsize=7.7, ncol=2)

    parts = ["CR", "CB", "CS", "CQ", "C_TU"]
    labels = ["Land", "Bridge/\ntunnel", "Base\nworks", "Maintenance", "Earthwork"]
    v0 = np.array([MA[k] for k in parts]) / 1e8
    v1 = np.array([MC[k] for k in parts]) / 1e8
    x = np.arange(len(parts)); w = .36
    a = ax[1, 0]
    a.bar(x - w/2, v0, width=w, color="#9e9e9e", label="M-A")
    a.bar(x + w/2, v1, width=w, color="#4b83c3", label="M-C")
    a.set_xticks(x, labels); a.set_ylabel(r"Cost ($10^8$ RMB)")
    a.set_title("(c) Life-cycle cost decomposition"); a.grid(axis="y", alpha=.18)
    a.legend(frameon=False)

    metrics = ["Life-cycle\ncost", "Traffic\nenergy", "Length", "Slope\nhazard"]
    changes = [percent(MC["C"], MA["C"]), percent(MC["E"], MA["E"]),
               percent(MC["L_km"], MA["L_km"]), percent(MC["Q_mean"], MA["Q_mean"])]
    a = ax[1, 1]
    colors = ["#2a9d8f" if v <= 0 else "#d1495b" for v in changes]
    bars = a.bar(np.arange(4), changes, color=colors)
    a.axhline(0, color="#222", lw=.7)
    a.set_xticks(np.arange(4), metrics); a.set_ylabel("Change from M-A (%)")
    a.set_title("(d) Engineering performance")
    a.set_ylim(min(changes)-1.0, max(changes)+.75)
    for b, v in zip(bars, changes):
        a.text(b.get_x()+b.get_width()/2, v + (.22 if v < 0 else .12), f"{v:+.2f}%",
               ha="center", va="bottom", fontsize=8)
    a.text(.47, .08, f"$R_{{min}}$={MC['Rmin']:.0f} m; penalty={MC['penalty']:.0e}\n"
           f"Tier 1={MC['L_dense1_km']:.2f} km; Tier 2={MC['L_dense2_km']:.2f} km",
           transform=a.transAxes, fontsize=8, va="bottom")
    save(fig, "current_scheme_summary")


def pareto_current() -> None:
    pts = DATA.get("pareto_sweep", DATA.get("pareto", []))
    p = [q for q in pts if float(q.get("pen", q.get("penalty", 0))) <= 1e-6]
    c = np.asarray([q["C"] for q in p]) / 1e8
    e = np.asarray([q["E"] for q in p]) / 1e8
    nd = np.ones(len(p), dtype=bool)
    for i in range(len(p)):
        nd[i] = not np.any((c <= c[i]) & (e <= e[i]) & ((c < c[i]) | (e < e[i])))
    fig, ax = plt.subplots(figsize=(6.7, 4.7))
    ax.scatter(e[~nd], c[~nd], c="#bdbdbd", s=28, label="Feasible dominated")
    order = np.argsort(e[nd])
    ax.plot(e[nd][order], c[nd][order], "o-", color="#2a9d8f", ms=5, lw=1.4,
            label="Feasible non-dominated")
    ax.scatter(MC["E"]/1e8, MC["C"]/1e8, marker="*", s=190, color="#d1495b",
               edgecolor="white", linewidth=.8, zorder=5, label="Entropy-selected M-C")
    ax.scatter(MA["E"]/1e8, MA["C"]/1e8, marker="s", s=60, color="#555",
               label="M-A existing")
    ax.set(xlabel=r"Life-cycle traffic energy ($10^8$ RMB)",
           ylabel=r"Life-cycle cost ($10^8$ RMB)",
           title="Current Pareto screening: endpoint anchored, density constrained")
    ax.grid(alpha=.2); ax.legend(frameon=False)
    # The full front contains several structure-heavy, low-energy solutions.
    # Keep them visible while adding a decision-scale inset around M-A/M-C.
    iax = inset_axes(ax, width="43%", height="43%", loc="center left",
                     bbox_to_anchor=(.05, -.02, 1, 1), bbox_transform=ax.transAxes)
    iax.scatter(e[~nd], c[~nd], c="#bdbdbd", s=12)
    iax.plot(e[nd][order], c[nd][order], "o-", color="#2a9d8f", ms=3, lw=1)
    iax.scatter(MC["E"]/1e8, MC["C"]/1e8, marker="*", s=75,
                color="#d1495b", edgecolor="white", linewidth=.5, zorder=5)
    iax.scatter(MA["E"]/1e8, MA["C"]/1e8, marker="s", s=25, color="#555")
    iax.set_xlim(12.8, 14.1); iax.set_ylim(22.8, 28.3)
    iax.set_title("Decision-scale detail", fontsize=7)
    iax.tick_params(labelsize=6); iax.grid(alpha=.18)
    mark_inset(ax, iax, loc1=1, loc2=3, fc="none", ec="#777", lw=.6)
    save(fig, "current_pareto_density")


def three_dimensional_current() -> None:
    sys.path.insert(0, str(SCHEME))
    from data_loader import load_alignment  # type: ignore
    import dem  # type: ignore
    import crossings as crmod  # type: ignore

    xx, yy = np.asarray(MC["plane_x"]), np.asarray(MC["plane_y"])
    zz, sta = np.asarray(MC["design_z"]), np.asarray(MC["sta"])
    align = load_alignment()
    xa, ya, za = align["X"], align["Y"], align["ground_z"]
    lat0, lon0 = float(align["lat"][0]), float(align["lon"][0])
    pad = 600.0
    x0, x1 = min(xx.min(), xa.min())-pad, max(xx.max(), xa.max())+pad
    y0, y1 = min(yy.min(), ya.min())-pad, max(yy.max(), ya.max())+pad
    gx, gy = np.linspace(x0, x1, 380), np.linspace(y0, y1, 150)
    GX, GY = np.meshgrid(gx, gy)
    GZ = dem.ground_elev_xy(GX.ravel(), GY.ravel(), lat0, lon0,
                            natural=True).reshape(GX.shape)
    eco = dem.eco_mask_xy(xx, yy, lat0, lon0)
    cr = crmod.detect_crossings(xx, yy, lat0, lon0)
    keep = np.interp(cr["s"], sta, eco.astype(float)) <= .5 if len(cr["s"]) else np.zeros(0, bool)
    intervals, bridge_len = crmod.bridge_intervals(cr, keep=keep)
    bridge = crmod.mask_from_intervals(sta, intervals)
    normal = ~(eco | bridge)

    cmap = LinearSegmentedColormap.from_list("paper_terrain",
        ["#f4f1ea", "#e4ddc7", "#cfc09a", "#a58b60", "#e8e4dd"])
    rgb = LightSource(azdeg=315, altdeg=42).shade(GZ, cmap=cmap, blend_mode="soft",
                                                   vert_exag=10, vmin=GZ.min(), vmax=GZ.max())
    fig = plt.figure(figsize=(12.2, 5.1))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_position([-.04, -.18, .91, 1.34])
    kx, ky = (GX-x0)/1000, (GY-y0)/1000
    ax.plot_surface(kx, ky, GZ, facecolors=rgb, shade=False, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, zorder=1)
    lx, ly = (xx-x0)/1000, (yy-y0)/1000
    halo = [pe.Stroke(linewidth=3.5, foreground="white", alpha=.9), pe.Normal()]

    def plot_mask(mask, color, lw, ls="-"):
        idx = np.where(mask)[0]
        for seg in np.split(idx, np.where(np.diff(idx) > 1)[0]+1):
            if len(seg) < 2:
                continue
            line, = ax.plot(lx[seg], ly[seg], zz[seg]+8, color=color, lw=lw, ls=ls,
                            zorder=12, solid_capstyle="round")
            line.set_path_effects(halo)

    ax.plot((xa-x0)/1000, (ya-y0)/1000, za+4, color="#666", lw=.8, ls=(0,(2,2)), zorder=10)
    plot_mask(normal, "#b6321f", 1.8)
    plot_mask(bridge, COL["bridge"], 2.6)
    plot_mask(eco, COL["tunnel"], 2.2, (0,(4,2)))
    Lx, Ly = (x1-x0)/1000, (y1-y0)/1000
    zmax = float(np.ceil(GZ.max()/100)*100)
    ax.set_box_aspect((Lx, Ly, 3.6)); ax.set_zlim(0, zmax); ax.view_init(elev=42, azim=-112)
    ax.set_xlabel("Easting (km)", labelpad=14)
    ax.set_ylabel("Northing (km)", labelpad=-4)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Elev. (m)", labelpad=10, rotation=90)
    ax.set_yticks(np.arange(0, np.ceil(Ly)+.1, 2.0))
    ax.set_zticks(np.arange(0, zmax+1, 200))
    ax.tick_params(pad=1, labelsize=8.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1,1,1,0)); axis._axinfo["grid"].update(color=(0,0,0,.10), linewidth=.4)
    ax.set_title("Current M-C alignment on quasi-natural DEM (endpoint anchored; density constrained)", y=.88)
    sm = ScalarMappable(cmap=cmap); sm.set_array(GZ)
    cb = fig.colorbar(sm, ax=ax, shrink=.42, aspect=22, pad=.005)
    cb.set_label("Ground elevation (m)")
    handles = [
        Line2D([0],[0],color="#b6321f",lw=2,label="M-C at-grade"),
        Line2D([0],[0],color=COL["bridge"],lw=2.8,label=f"OSM-triggered bridges ({bridge_len/1000:.2f} km)"),
        Line2D([0],[0],color=COL["tunnel"],lw=2.4,ls=(0,(4,2)),label=f"Baiyun eco-tunnel ({MC['L_eco_km']:.2f} km)"),
        Line2D([0],[0],color="#666",lw=1,ls=(0,(2,2)),label="M-A existing centerline"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(.04,.80))
    save(fig, "current_alignment_3d")


def write_tables() -> None:
    rows = [
        ["Life-cycle cost C", MA["C"]/1e8, MC["C"]/1e8, percent(MC["C"], MA["C"]), "1e8 RMB"],
        ["Traffic energy E", MA["E"]/1e8, MC["E"]/1e8, percent(MC["E"], MA["E"]), "1e8 RMB"],
        ["Length", MA["L_km"], MC["L_km"], percent(MC["L_km"], MA["L_km"]), "km"],
        ["Minimum radius", MA["Rmin"], MC["Rmin"], percent(MC["Rmin"], MA["Rmin"]), "m"],
        ["Slope hazard Q", MA["Q_mean"], MC["Q_mean"], percent(MC["Q_mean"], MA["Q_mean"]), "-"],
        ["Tier-1 exposure", 2.001, MC["L_dense1_km"], percent(MC["L_dense1_km"], 2.001), "km"],
        ["Tier-2 exposure", 0.0, MC["L_dense2_km"], 0.0, "km"],
    ]
    with (TAB / "current_scheme_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Metric", "M-A", "M-C", "Change_percent", "Unit"]); w.writerows(rows)


if __name__ == "__main__":
    scheme_summary()
    pareto_current()
    three_dimensional_current()
    write_tables()
    print(f"Generated manuscript figures and table under {VERSION}")
