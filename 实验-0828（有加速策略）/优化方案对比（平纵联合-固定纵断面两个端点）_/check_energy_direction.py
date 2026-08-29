# -*- coding: utf-8 -*-
"""
能耗方向性口径核验: 现行 E 用【单程(桩号递增方向)】纵坡算单车能耗, 再乘全量 AADT。
本脚本重算【反向行驶】与【双向平均】能耗, 量化单程口径的方向偏置。

反向行驶: sta' = L - sta[::-1], z' = z[::-1] (纵坡自动反号), R_seg' = R_seg[::-1]。
Fa(横向阻力) 只依赖 R 与 v, 与坡向无关, 故按段反序即精确对应。
"""
import json
import os
import sys

import numpy as np

from objective import fuel_energy, ev_energy
from objective_joint import _plane_metrics
from params import TRAFFIC, ENERGY_PRICE, LCC, CASE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def r_seg_from_plane(px, py, sta):
    """复现 decode_joint 的 R_seg: 密集点曲率 -> 插值到桩号 -> 相邻均值取倒数。"""
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    _, R = _plane_metrics(px, py)
    sarc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    kappa = 1.0 / np.maximum(R, 1e-9)
    kappa_sta = np.interp(sta, sarc, kappa)
    kappa_seg = 0.5 * (kappa_sta[:-1] + kappa_sta[1:])
    return 1.0 / np.maximum(kappa_seg, 1e-12)


def per_vehicle(sta, z, R_seg, v):
    """返回 (ml 单车单程油耗, kwh 单车单程电耗)。"""
    return (fuel_energy(sta, z, v, R=R_seg),
            ev_energy(sta, z, v, R=R_seg))


def to_money(ml, kwh, ev_share):
    """单车单程物理量 -> 全生命周期货币(元), 复现 objective_joint 基准情景聚合。"""
    AADT = TRAFFIC["AADT"]
    n1, n2 = 1.0 - ev_share, ev_share
    t = LCC["analysis_years"]
    ru = LCC["bank_rate"]
    Ef = Ee = 0.0
    for j in range(1, t + 1):
        disc = 1.0 / (1 + ru) ** j
        Ef += AADT * n1 * (ml / 1000.0) * ENERGY_PRICE["Kf_fuel"] * disc * 365.0
        Ee += AADT * n2 * kwh * ENERGY_PRICE["ZQ_elec"] * disc * 365.0
    return Ef, Ee


def analyse(tag, sta, z, px, py, v, ev_share):
    sta = np.asarray(sta, float)
    z = np.asarray(z, float)
    R_seg = r_seg_from_plane(px, py, sta)

    ml_f, kwh_f = per_vehicle(sta, z, R_seg, v)

    sta_r = sta[-1] - sta[::-1]
    z_r = z[::-1]
    ml_r, kwh_r = per_vehicle(sta_r, z_r, R_seg[::-1], v)

    ml_b, kwh_b = 0.5 * (ml_f + ml_r), 0.5 * (kwh_f + kwh_r)

    out = {}
    for nm, (ml, kwh) in (("fwd", (ml_f, kwh_f)),
                          ("rev", (ml_r, kwh_r)),
                          ("bi", (ml_b, kwh_b))):
        Ef, Ee = to_money(ml, kwh, ev_share)
        out[nm] = dict(ml=ml, kwh=kwh, E_fuel=Ef, E_ele=Ee, E=Ef + Ee)
    out["dz_net"] = float(z[-1] - z[0])
    out["tag"] = tag
    return out


def main():
    v = CASE["design_speed_kmh"]
    ev_share = TRAFFIC["n2_ev"]
    YI = 1e8
    rows = []
    for w in (500, 1000):
        fn = os.path.join(RESULTS, f"joint_results_w{w}_dens.json")
        if not os.path.exists(fn):
            continue
        d = json.load(open(fn, encoding="utf-8"))
        ev_share = d.get("meta", {}).get("ev_share", ev_share)
        for k, nm in (("M_A", "M-A"), ("M_B", "M-B"), ("M_C", "M-C")):
            m = d[k]
            r = analyse(f"w{w} {nm}", m["sta"], m["design_z"],
                        m["plane_x"], m["plane_y"], v, ev_share)
            r["E_json"] = m["E"]
            rows.append(r)

    print(f"设计速度 {v} km/h, EV 占比 {ev_share}, AADT {TRAFFIC['AADT']}")
    print()
    print(f"{'方案':<11}{'净高差m':>8}{'正向ml':>9}{'反向ml':>9}"
          f"{'正向kWh':>9}{'反向kWh':>9}")
    for r in rows:
        f, rv = r["fwd"], r["rev"]
        print(f"{r['tag']:<11}{r['dz_net']:>8.1f}{f['ml']:>9.1f}{rv['ml']:>9.1f}"
              f"{f['kwh']:>9.3f}{rv['kwh']:>9.3f}")
    print()
    print(f"{'方案':<11}{'E正向亿':>10}{'E反向亿':>10}{'E双向亿':>10}"
          f"{'反-正%':>9}{'双-正%':>9}{'E(JSON)亿':>11}")
    for r in rows:
        Ef, Er, Eb = r["fwd"]["E"], r["rev"]["E"], r["bi"]["E"]
        print(f"{r['tag']:<11}{Ef/YI:>10.4f}{Er/YI:>10.4f}{Eb/YI:>10.4f}"
              f"{(Er/Ef-1)*100:>9.2f}{(Eb/Ef-1)*100:>9.2f}"
              f"{r['E_json']/YI:>11.4f}")

    print()
    print("排序稳健性检查(同组内 M-C 相对 M-A 的降幅):")
    for w in (500, 1000):
        sel = [r for r in rows if r["tag"].startswith(f"w{w} ")]
        if len(sel) < 3:
            continue
        A = next(r for r in sel if r["tag"].endswith("M-A"))
        C = next(r for r in sel if r["tag"].endswith("M-C"))
        for nm, lab in (("fwd", "单程正向"), ("rev", "单程反向"), ("bi", "双向平均")):
            print(f"  w{w} {lab}: ΔE = {(C[nm]['E']/A[nm]['E']-1)*100:+.2f}%")


if __name__ == "__main__":
    main()
