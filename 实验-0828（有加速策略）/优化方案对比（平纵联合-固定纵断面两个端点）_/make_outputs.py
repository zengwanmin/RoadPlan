# -*- coding: utf-8 -*-
"""
make_outputs.py — 由【平纵联合协同优化】结果生成实验三全部图表

数据源: results/joint_results_w500_nodens.json (run_joint.py 输出, 固定端点 W500 主结果)
        results/twostage_results_w500_nodens.json (run_twostage.py 输出, 两阶段对照, 供表C3)
表: 表C1(三模式四维指标 + M-B→M-C 变化率)  表C2(现状 M-A vs 本文 M-C 关键指标 + 变化%)
    表C3(现状/两阶段/平纵联合协同 三方案对比表)
图: 图C1(Pareto解集+熵权决策点)
    图C2(平面线形: 现状 vs 优化, 同一张图)
    图C3(纵断面线形: 现状 vs 优化, 同一张图)
    图C4(全生命周期成本分项堆积柱, 三方案)
    图C5(优化后边坡稳定性评估云图)
    图C6(平纵联合优化 IJS 收敛曲线)
    图C7(权重 wC 从 0 到 1 变化时优化方案帕累托前沿的变化趋势)
    图C8(能耗-成本完整Pareto解集)
    图C9(全图层叠加: DEM + OSM 路网/铁路/水系 + OSM 建筑 + 最终线位 M-C + 桥隧段标注)
图的横轴/纵轴/图例/图名均为英文。能耗单位 元/日, 桥隧费用 0。
"""
import os, json, csv, argparse, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from params import DESIGN_STD   # 表6.4 竖曲线半径(供图C3竖曲线平滑)
from alllayers import fig_C9_alllayers   # 图C9 全图层叠加(DEM+OSM+建筑+线位+桥隧)
from fair_pareto import SCHEMA as FAIR_PARETO_SCHEMA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "joint_results_w500_nodens.json")
RES_TWO = os.path.join(HERE, "results", "twostage_results_w500_nodens.json")
FIG = os.path.join(HERE, "figures"); TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

for fn in ["Arial Unicode MS", "Heiti TC", "Songti SC", "STHeiti"]:
    try:
        font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fn]; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

C_YI = 1e8   # 元 -> 亿元


def load():
    with open(RES, encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(data):
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_sources(d):
    """
    Fail closed before writing any table or figure from stale mixed sources.

    联合结果只保存原始 Pareto 前沿；两阶段结果在两条前沿都完成后
    保存唯一的公共范围熵权决策。本函数校验绑定后，把公共决策选出的
    联合 M-C 注入内存中的 d，供所有图表统一使用；不回写原始结果文件。
    """
    provenance = d.get("provenance", {})
    config = provenance.get("config")
    fingerprint = provenance.get("config_fingerprint")
    if not isinstance(config, dict) or fingerprint != _fingerprint(config):
        raise RuntimeError(
            "Joint result has no valid current provenance fingerprint; "
            "rerun run_joint.py instead of generating outputs from a legacy JSON.")
    if (float(config.get("corridor_half_w", -1)) != 500.0 or
            config.get("density_on") is not False or
            config.get("profile_endpoints_fixed") is not True or
            config.get("smoke") is not False or
            config.get("schema") != "joint-pareto-resume-v3"):
        raise RuntimeError(
            "Main-paper outputs require full W500 fixed-endpoint, density-disabled results")
    if not os.path.isfile(RES_TWO):
        raise FileNotFoundError(
            "Current two-stage result is required for Table C3; rerun run_twostage.py")
    with open(RES_TWO, encoding="utf-8") as fp:
        two_stage = json.load(fp)
    two_provenance = two_stage.get("provenance", {})
    two_config = two_provenance.get("config")
    if (not isinstance(two_config, dict) or
            two_provenance.get("config_fingerprint") != _fingerprint(two_config) or
            float(two_config.get("corridor_half_w", -1)) != 500.0 or
            two_config.get("density_on") is not False or
            two_config.get("profile_endpoints_fixed") is not True or
            two_config.get("smoke") is not False or
            two_config.get("schema") != "two-stage-pareto-resume-v3"):
        raise RuntimeError(
            "Two-stage result is not a full W500 fixed-endpoint, density-disabled result")
    if two_provenance.get("joint_result_sha256") != _sha256_file(RES):
        raise RuntimeError(
            "Two-stage result is not bound to this exact W500 joint result; "
            "refusing to generate a mixed-version Table C3.")

    joint_tie = np.asarray(config.get("profile_endpoint_elevations_m", []),
                           dtype=float)
    two_tie = np.asarray(two_config.get("profile_endpoint_elevations_m", []),
                         dtype=float)
    if (joint_tie.shape != (2,) or two_tie.shape != (2,) or
            not np.allclose(joint_tie, two_tie, rtol=0.0, atol=1e-9)):
        raise RuntimeError(
            "Joint and two-stage results do not use the same two fixed profile elevations")

    if "M_C_scalar" in d or "M_C_scalar" in two_stage:
        raise RuntimeError(
            "Fixed-weight M-C results are forbidden in fair Pareto comparison outputs")
    joint_grid = np.asarray(config.get("weight_grid", []), dtype=float)
    two_grid = np.asarray(two_config.get("weight_grid", []), dtype=float)
    if (joint_grid.shape != two_grid.shape or
            not np.allclose(joint_grid, two_grid, rtol=0.0, atol=1e-12)):
        raise RuntimeError("Joint and two-stage results use different Pareto weight grids")

    fair = two_stage.get("fair_decision")
    if (not isinstance(fair, dict) or fair.get("schema") != FAIR_PARETO_SCHEMA or
            fair.get("selection_scope") != "joint_and_two_stage_front_union"):
        raise RuntimeError(
            "Missing common-range fair Pareto decision; rerun run_twostage.py")
    if (fair.get("joint_result_sha256") != _sha256_file(RES) or
            fair.get("joint_config_fingerprint") != fingerprint or
            fair.get("two_stage_config_fingerprint") !=
            two_provenance.get("config_fingerprint")):
        raise RuntimeError("Fair Pareto decision is not bound to this joint front")
    fair_grid = np.asarray(fair.get("weight_grid", []), dtype=float)
    if (fair_grid.shape != joint_grid.shape or
            not np.allclose(fair_grid, joint_grid, rtol=0.0, atol=1e-12)):
        raise RuntimeError("Fair Pareto decision used a different weight grid")

    joint_selected = fair.get("joint", {})
    two_selected = fair.get("two_stage", {})

    def verify_selected(label, selected, sweep):
        matches = [p for p in sweep if p.get("tag") == selected.get("tag")]
        if len(matches) != 1:
            raise RuntimeError(f"{label} selected Pareto tag is missing or duplicated")
        raw = matches[0]
        if not np.allclose(
                [float(raw["w1"]), float(raw["C"]), float(raw["E"]),
                 float(raw.get("pen", float("inf")))],
                [float(selected["w1"]), float(selected["C"]),
                 float(selected["E"]), float(selected["pen"])],
                rtol=1e-12, atol=1e-9):
            raise RuntimeError(f"{label} common decision does not match its raw front")

    verify_selected("joint", joint_selected, d.get("pareto_sweep", []))
    verify_selected("two-stage", two_selected,
                    two_stage.get("pareto_sweep", []))
    joint_mc = joint_selected.get("M_C", {})
    two_mc = two_selected.get("M_C", {})
    if two_stage.get("M_C") != two_mc:
        raise RuntimeError("Two-stage M-C is not the common-range Pareto selection")
    for label, payload in (("joint", joint_mc), ("two-stage", two_mc)):
        if (float(payload.get("penalty", float("inf"))) > 1e-6 or
                float(payload.get("Rmin", -float("inf"))) < 400.0 - 1e-6):
            raise RuntimeError(
                f"{label} M-C is infeasible; refusing to label Table C3 as R>=400")
        design_z = np.asarray(payload.get("design_z", []), dtype=float)
        if (design_z.size < 2 or
                not np.allclose(design_z[[0, -1]], joint_tie,
                                rtol=0.0, atol=1e-8)):
            raise RuntimeError(
                f"{label} M-C does not satisfy the two fixed profile elevations")

    # 两种方法的最终解必须使用同一熵权和同一归一化范围。
    entropy = fair["entropy"]
    d["M_C"] = joint_mc
    d["entropy_point"] = dict(
        C=joint_mc["C"], E=joint_mc["E"],
        wC=float(entropy["wC"]), wE=float(entropy["wE"]),
        w1_selected=float(joint_selected["w1"]),
        budget_tol=float(fair["filter"]["budget_tol"]),
        selection_scope=fair["selection_scope"])
    d["convergence"] = list(joint_selected["convergence"])
    d["length_reduction_pct"] = (
        (d["M_A"]["L_km"] - joint_mc["L_km"]) / d["M_A"]["L_km"] * 100.0)
    d["fair_decision"] = fair
    d["decision_status"] = "common_range_fair_pareto_selected"
    return two_stage


def _write_source_provenance(d, two_stage):
    payload = {
        "schema": "current-output-sources-v1",
        "joint_result": os.path.relpath(RES, HERE),
        "joint_result_sha256": _sha256_file(RES),
        "joint_config_fingerprint": d["provenance"]["config_fingerprint"],
        "two_stage_result": os.path.relpath(RES_TWO, HERE),
        "two_stage_result_sha256": _sha256_file(RES_TWO),
        "two_stage_joint_binding": two_stage["provenance"]["joint_result_sha256"],
        "decision_schema": two_stage["fair_decision"]["schema"],
        "decision_scope": two_stage["fair_decision"]["selection_scope"],
        "common_entropy": two_stage["fair_decision"]["entropy"],
        "profile_endpoints_fixed": True,
        "profile_endpoint_elevations_m": d["provenance"]["config"][
            "profile_endpoint_elevations_m"],
    }
    path = os.path.join(TAB, "SOURCE_PROVENANCE.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _write_table(name, hdr, rows):
    with open(os.path.join(TAB, name + ".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    with open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(hdr) + " |\n")
        f.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(map(str, r)) + " |\n")
    print(f"[表] {name}")


def _pct(new, old):
    if old == 0:
        return "-"
    return f"{(new - old) / old * 100:+.2f}%"


def _save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close()
    print(f"[图] {name}")


def _vertical_curve_profile(sta, z, R_v=None, n_sub=6):
    """在【分段线性设计纵断面】的每个变坡点处插入对称抛物线竖曲线, 返回加密后的
    (s, z), 使纵断面在变坡点处平滑相切过渡(而非折线尖角)。仅用于绘图, 不改动优化。

    竖曲线(式4.28-4.29 的形态): 对称二次抛物线。变坡点 i 入坡 i_in、出坡 i_out,
    坡差 A=i_out-i_in, 竖曲线长度 L=R_v·|A|(R_v 取表6.4 最小竖曲线半径),
    半长 L/2 再裁剪到不超过相邻两段各自的一半 —— 使相邻竖曲线在切点处正好衔接,
    整条设计线 C1 连续。变坡点桩号 s_i、高程 z_i, 则:
      BVC(竖曲线起点) 桩号 s_i-L/2、高程 z_i-i_in·L/2;
      沿弧长 t∈[0,L]:  z(t) = z_BVC + i_in·t + (i_out-i_in)/(2L)·t²。
    地面线不加竖曲线(属地形, 非设计线)。
    """
    sta = np.asarray(sta, float); z = np.asarray(z, float)
    n = sta.size
    if n < 3:
        return sta, z
    if R_v is None:
        R_v = float(DESIGN_STD["Lv_min_m"])       # 最小竖曲线半径(表6.4)
    seg = np.diff(sta)                             # 段长, n-1
    g = np.diff(z) / np.where(seg > 0, seg, 1e-9)  # 各段纵坡, n-1
    # 内部变坡点 i=1..n-2: 入坡 g[i-1]、出坡 g[i]; 竖曲线半长 h_i(裁剪防相邻重叠)
    dA = np.abs(g[1:] - g[:-1])                    # |坡差|, 长度 n-2, 对应 sta[1:-1]
    h = 0.5 * R_v * dA
    h = np.minimum(h, 0.5 * seg[:-1])              # 不越过左段中点
    h = np.minimum(h, 0.5 * seg[1:])               # 不越过右段中点
    out_s = [sta[0]]; out_z = [z[0]]
    for i in range(1, n - 1):
        hi = h[i - 1]
        if hi <= 1e-9:                             # 坡差≈0 或段太短: 直连该变坡点
            out_s.append(sta[i]); out_z.append(z[i]); continue
        g_in, g_out = g[i - 1], g[i]
        s_bvc = sta[i] - hi; z_bvc = z[i] - g_in * hi
        L = 2.0 * hi
        t = np.linspace(0.0, L, n_sub + 1)
        sc = s_bvc + t
        zc = z_bvc + g_in * t + (g_out - g_in) / (2.0 * L) * t * t
        out_s.extend(sc.tolist()); out_z.extend(zc.tolist())   # BVC 切点 -> 抛物线 -> EVC 切点
    out_s.append(sta[-1]); out_z.append(z[-1])
    return np.asarray(out_s), np.asarray(out_z)


# ---- 表C1: 三模式四维指标 + M-B→M-C 变化率 ----
def table_C1(d):
    A, B, C = d["M_A"], d["M_B"], d["M_C"]
    hdr = ["Dimension", "Metric", "M-A (existing)", "M-B (cost-only)",
           "M-C (joint bi-objective)", "M-B→M-C change"]

    def row(dim, metric, key, scale=1.0, unit=""):
        va, vb, vc = A[key] / scale, B[key] / scale, C[key] / scale
        return [dim, metric, f"{va:.4f}{unit}", f"{vb:.4f}{unit}",
                f"{vc:.4f}{unit}", _pct(C[key], B[key])]
    rows = [
        row("Economy", "Life-cycle cost C (10^8 RMB)", "C", C_YI),
        row("Economy", "  Land acquisition CR (10^8)", "CR", C_YI),
        row("Economy", "  Bridge/tunnel CB (10^8)", "CB", C_YI),
        row("Economy", "  Basic construction CS (10^8)", "CS", C_YI),
        row("Economy", "  Maintenance CQ (10^8)", "CQ", C_YI),
        row("Economy", "  Earthwork C_TU (10^8)", "C_TU", C_YI),
        row("Energy", "Life-cycle traffic energy E (10^8 RMB)", "E", C_YI),
        row("Energy", "  Fuel-vehicle E_fuel (10^8)", "E_fuel", C_YI),
        row("Energy", "  Electric-vehicle E_ele (10^8)", "E_ele", C_YI),
        row("Efficiency", "Length L (km)", "L_km"),
        row("Safety", "Slope hazard degree Q", "Q_mean"),
    ]
    _write_table("表C1_三模式四维指标对比表", hdr, rows)


# ---- 表C2: 现状 M-A vs 本文 M-C 关键指标 + 变化% ----
def table_C2(d):
    A, C = d["M_A"], d["M_C"]
    hdr = ["Metric", "M-A (existing)", "M-C (optimized)", "Change (%)"]
    rows = [
        ["Length L (km)", f"{A['L_km']:.3f}", f"{C['L_km']:.3f}", _pct(C['L_km'], A['L_km'])],
        ["Earthwork C_TU (10^8 RMB)", f"{A['C_TU']/C_YI:.4f}", f"{C['C_TU']/C_YI:.4f}", _pct(C['C_TU'], A['C_TU'])],
        ["Land acquisition CR (10^8 RMB)", f"{A['CR']/C_YI:.4f}", f"{C['CR']/C_YI:.4f}", _pct(C['CR'], A['CR'])],
        ["Life-cycle cost C (10^8 RMB)", f"{A['C']/C_YI:.4f}", f"{C['C']/C_YI:.4f}", _pct(C['C'], A['C'])],
        ["Life-cycle traffic energy E (10^8 RMB)", f"{A['E']/C_YI:.4f}", f"{C['E']/C_YI:.4f}", _pct(C['E'], A['E'])],
        ["Slope hazard Q", f"{A['Q_mean']:.3f}", f"{C['Q_mean']:.3f}", _pct(C['Q_mean'], A['Q_mean'])],
        ["Min plane radius (m)", "-", f"{C['Rmin']:.0f}", "(>=400 OK)"],
    ]
    _write_table("表C2_优化前后关键指标对比表", hdr, rows)


# ---- 表C3: 现状 / 两阶段(先平面后纵断面) / 平纵联合协同 三方案对比 ----
def table_C3(d, two_stage):
    """三方案对比: 现状(M-A) vs 两阶段优化 vs 平纵联合协同优化。
    两种方法都在同一权重网格上生成 Pareto 前沿，再使用两条前沿合并
    后的公共数据范围、归一化和熵权规则分别选出；两者固定同一对
    既有道路纵断面接线高程。"""
    dt = two_stage
    A = d["M_A"]              # 现状(联合与两阶段口径一致, 取联合的现状)
    TS = dt["M_C"]            # 两阶段优化方案
    JT = d["M_C"]             # 平纵联合协同优化方案(本文)

    hdr = ["Metric", "M-A (existing)", "Two-stage (plane→profile)",
           "Joint (plane+profile)", "Two-stage vs A", "Joint vs A"]

    def rowk(metric, key, scale=1.0, fmt="{:.4f}"):
        a, ts, jt = A[key] / scale, TS[key] / scale, JT[key] / scale
        return [metric, fmt.format(a), fmt.format(ts), fmt.format(jt),
                _pct(TS[key], A[key]), _pct(JT[key], A[key])]

    rows = [
        rowk("Life-cycle cost C (10^8 RMB)", "C", C_YI),
        rowk("  Land acquisition CR (10^8)", "CR", C_YI),
        rowk("  Bridge/tunnel CB (10^8)", "CB", C_YI),
        rowk("  Basic construction CS (10^8)", "CS", C_YI),
        rowk("  Maintenance CQ (10^8)", "CQ", C_YI),
        rowk("  Earthwork C_TU (10^8)", "C_TU", C_YI),
        rowk("Life-cycle traffic energy E (10^8 RMB)", "E", C_YI),
        rowk("Length L (km)", "L_km", 1.0, "{:.3f}"),
        rowk("Slope hazard Q", "Q_mean", 1.0, "{:.3f}"),
        ["Min plane radius (m)", "-", f"{TS['Rmin']:.0f}", f"{JT['Rmin']:.0f}",
         "(>=400)", "(>=400)"],
        ["Common entropy decision score S (higher is better)", "-",
         f"{d['fair_decision']['two_stage']['score']:.6f}",
         f"{d['fair_decision']['joint']['score']:.6f}", "-", "-"],
        ["Selected Pareto scan weight wC", "-",
         f"{d['fair_decision']['two_stage']['w1']:.2f}",
         f"{d['fair_decision']['joint']['w1']:.2f}", "-", "-"],
    ]
    _write_table("表C3_现状_两阶段_联合协同_三方案对比表", hdr, rows)
    _append_c3_budget_note()


def _append_c3_budget_note():
    """
    表C3 附注(求解与决策口径): 每个权重点的联合方案按 iter=1000 求解，
    两阶段由 Stage1 iter=500 + Stage2 iter=500 构成:
      联合  : 单次 IJS,            200 + 3·200·1000 ≈ 600,200
      两阶段: Stage1+Stage2 各一次 IJS, 2·(200 + 3·200·500) ≈ 600,400
    两者使用同一 Pareto 权重网格，最终解共用两条前沿合并后的公共
    极差范围、归一化和唯一熵权规则，不保留固定权重 M-C。
    """
    fn = os.path.join(TAB, "表C3_现状_两阶段_联合协同_三方案对比表.md")
    note = (
        "\n> **附注(公平Pareto决策)**: 联合和两阶段使用完全相同的"
        "Pareto权重网格；每个最终解的逻辑预算为联合 iter=1000，两阶段 "
        "Stage1+Stage2 各 iter=500。最终解共用两条前沿合并后的数据范围、"
        "归一化和唯一熵权，不使用固定权重解；两种方法均将纵断面"
        "首末端点锚定到同一对既有道路接线高程。\n"
    )
    with open(fn, "a", encoding="utf-8") as f:
        f.write(note)
    print("[表C3] 已附加求解预算说明")


# ---- 图C1: Pareto 解集 + 熵权决策点 ----
def fig_C1(d, two_stage):
    P = d["pareto"]
    PT = two_stage["pareto"]
    C = np.array([p["C"] for p in P]) / C_YI
    E = np.array([p["E"] for p in P]) / C_YI
    CT = np.array([p["C"] for p in PT]) / C_YI
    ET = np.array([p["E"] for p in PT]) / C_YI
    ep = d["entropy_point"]
    fair = d["fair_decision"]
    ts = fair["two_stage"]["M_C"]
    plt.figure(figsize=(7.2, 5.2))
    order = np.argsort(E)
    plt.plot(E[order], C[order], "o-", color="#4c72b0", ms=6, lw=1.3,
             label="Joint Pareto front", alpha=0.85)
    order_t = np.argsort(ET)
    plt.plot(ET[order_t], CT[order_t], "s--", color="#55a868", ms=5, lw=1.2,
             label="Two-stage Pareto front", alpha=0.85)
    plt.scatter([ep["E"] / C_YI], [ep["C"] / C_YI], s=190, marker="*",
                color="#c44e52", zorder=5, edgecolor="k",
                label=f"Joint decision (common entropy weights)\n"
                      f"(wC={ep['wC']:.3f}, wE={ep['wE']:.3f})")
    plt.scatter([ts["E"] / C_YI], [ts["C"] / C_YI], s=145, marker="D",
                color="#dd8452", zorder=5, edgecolor="k",
                label="Two-stage decision (same weights/range)")
    plt.xlabel("Life-cycle traffic energy E (10^8 RMB)")
    plt.ylabel("Life-cycle cost C (10^8 RMB)")
    plt.title("Fig. C1  Fair Pareto fronts and common-range entropy decisions")
    plt.legend(frameon=False); plt.grid(alpha=0.3)
    _save("图C1_Pareto解集与熵权决策点")


# ---- 图C2: 平面线形 现状 vs 优化 (同一张图) ----
def fig_C2(d):
    A, C = d["M_A"], d["M_C"]
    mx = np.array(d["measured"]["x"]) / 1000.0
    my = np.array(d["measured"]["y"]) / 1000.0
    ax_ = np.array(A["plane_x"]) / 1000.0; ay_ = np.array(A["plane_y"]) / 1000.0
    cx_ = np.array(C["plane_x"]) / 1000.0; cy_ = np.array(C["plane_y"]) / 1000.0
    plt.figure(figsize=(8.4, 5.4))
    plt.plot(mx, my, color="#bbbbbb", lw=0.8, ls=":", label="Measured trajectory")
    plt.plot(ax_, ay_, color="#333333", lw=1.7,
             label=f"M-A existing plane ({A['L_km']:.2f} km)")
    plt.plot(cx_, cy_, color="#c44e52", lw=2.1,
             label=f"M-C joint-optimized ({C['L_km']:.2f} km, {_pct(C['L_km'], A['L_km'])})")
    plt.scatter([cx_[0]], [cy_[0]], c="#2ca02c", s=55, zorder=6, label="Start")
    plt.scatter([cx_[-1]], [cy_[-1]], c="#8c564b", s=55, zorder=6, label="End")
    plt.xlabel("Easting X (km)"); plt.ylabel("Northing Y (km)")
    plt.title("Fig. C2  Horizontal alignment: existing (M-A) vs joint-optimized (M-C)")
    plt.legend(frameon=False); plt.grid(alpha=0.3); plt.axis("equal")
    _save("图C2_平面线形对比")


# ---- 图C3: 纵断面线形 现状 vs 优化 (同一张图, 设计线含竖曲线平滑) ----
def fig_C3(d):
    A, C = d["M_A"], d["M_C"]
    sa = np.array(A["sta"]) / 1000.0; gA = np.array(A["gz_new"])
    # 设计纵断面在变坡点处加竖曲线(对称抛物线)平滑; 地面线保持实测原样。
    # 竖曲线在原始里程(m)上构造后再转 km 绘图, 使变坡点处相切、无折线尖角。
    saV, zAv = _vertical_curve_profile(np.array(A["sta"]), np.array(A["design_z"]))
    scV, zCv = _vertical_curve_profile(np.array(C["sta"]), np.array(C["design_z"]))
    plt.figure(figsize=(9.2, 4.8))
    plt.plot(sa, gA, color="#bbbbbb", lw=1.0, ls=":", label="Ground line (measured)")
    plt.plot(saV / 1000.0, zAv, color="#333333", lw=1.5,
             label="M-A existing profile (with vertical curves)")
    plt.plot(scV / 1000.0, zCv, color="#c44e52", lw=1.9,
             label="M-C joint-optimized profile (with vertical curves)")
    plt.xlabel("Chainage (km)"); plt.ylabel("Elevation (m)")
    plt.title("Fig. C3  Longitudinal profile: existing (M-A) vs joint-optimized (M-C)")
    plt.legend(frameon=False); plt.grid(alpha=0.3)
    _save("图C3_纵断面线形对比")


# ---- 图C4: 全生命周期成本分项堆积柱 (M-A/M-B/M-C) ----
def fig_C4(d):
    modes = ["M_A", "M_B", "M_C"]
    labels = ["M-A (existing)", "M-B (cost-only)", "M-C (joint bi-obj.)"]
    keys = ["CR", "CB", "CS", "CQ", "C_TU"]
    names = ["Land acquisition CR", "Bridge/tunnel CB", "Basic construction CS",
             "Maintenance CQ", "Earthwork C_TU"]
    colors = ["#4c72b0", "#55a868", "#ccb974", "#8172b3", "#c44e52"]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(modes)); bottom = np.zeros(len(modes))
    for k, nm, col in zip(keys, names, colors):
        vals = np.array([d[m][k] / C_YI for m in modes])
        ax.bar(x, vals, bottom=bottom, label=nm, color=col, alpha=0.9,
               edgecolor="white", linewidth=0.6)
        bottom += vals
    for i, tot in enumerate(bottom):
        ax.text(i, tot, f"{tot:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Life-cycle cost (10^8 RMB)")
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.set_title("Fig. C4  Composition of life-cycle cost across the three schemes")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=0.3)
    _save("图C4_全生命周期成本分项构成")


# ---- 图C5: 优化后边坡稳定性评估云图 (里程-危险度) ----
def fig_C5(d):
    C = d["M_C"]
    sta = np.array(C["sta"]) / 1000.0
    QC = np.array(C["Q_series"])
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    im = ax.imshow(QC[None, :], aspect="auto", cmap="RdYlGn_r",
                   extent=[sta[0], sta[-1], 0, 1], vmin=1, vmax=5)
    ax.set_yticks([])
    ax.set_xlabel("Chainage (km)")
    ax.set_title("Fig. C5  Slope-stability hazard assessment of the optimized alignment (M-C)")
    cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cb.set_label("Hazard degree Q  (low → high)")
    _save("图C5_边坡稳定性评估云图")


# ---- 图C7: 权重 wC 从 0 到 1 变化时优化方案帕累托前沿的变化趋势 ----
def fig_C7(d):
    P = d.get("pareto_sweep") or d.get("pareto")
    P = sorted(P, key=lambda p: p["w1"])
    w = np.array([p["w1"] for p in P])
    C = np.array([p["C"] for p in P]) / C_YI
    E = np.array([p["E"] for p in P]) / C_YI
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    # 前沿曲线(按能耗排序连线) + 按权重 wC 着色的散点
    order = np.argsort(E)
    ax.plot(E[order], C[order], "-", color="#999999", lw=1.0, zorder=1,
            label="Pareto front (weight sweep)")
    sc = ax.scatter(E, C, c=w, cmap="viridis", s=70, zorder=3,
                    edgecolor="k", linewidth=0.4, vmin=0, vmax=1)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Cost weight wC  (0 = energy-priority, 1 = cost-priority)")
    # 两端标注
    i0 = int(np.argmin(w)); i1 = int(np.argmax(w))
    ax.annotate(f"wC=0\n(energy-priority)", (E[i0], C[i0]), fontsize=8,
                textcoords="offset points", xytext=(6, 6))
    ax.annotate(f"wC=1\n(cost-priority)", (E[i1], C[i1]), fontsize=8,
                textcoords="offset points", xytext=(6, -14))
    ax.set_xlabel("Life-cycle traffic energy E (10^8 RMB)")
    ax.set_ylabel("Life-cycle cost C (10^8 RMB)")
    ax.set_title("Fig. C7  Pareto front evolution as cost weight wC varies from 0 to 1")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    _save("图C7_权重0到1帕累托前沿变化趋势")


# ---- 图C6: 平纵联合优化 IJS 收敛曲线 ----
def fig_C6(d):
    cC = np.array(d["convergence"]); cB = np.array(d.get("convergence_B", []))
    plt.figure(figsize=(7.6, 4.8))
    if cB.size:
        plt.plot(np.arange(len(cB)), cB, color="#4c72b0", lw=1.4, ls="--",
                 label="M-B cost-only (wC=1)")
    plt.plot(np.arange(len(cC)), cC, color="#c44e52", lw=1.9,
             label="M-C joint bi-objective (selected Pareto scan weight)")
    plt.xlabel("Iteration"); plt.ylabel("Scalarized objective F")
    plt.yscale("log")
    plt.title("Fig. C6  Convergence of joint plane+profile optimization (IJS)")
    plt.legend(frameon=False); plt.grid(alpha=0.3, which="both")
    _save("图C6_平纵联合优化收敛曲线")


# ---- 图C8: 完整 Pareto 解集(横轴能耗E, 纵轴全生命周期成本C) ----
def fig_C8(d, two_stage):
    sweep = d.get("pareto_sweep", d["pareto"])
    sweep_t = two_stage.get("pareto_sweep", two_stage["pareto"])
    A, MC = d["M_A"], d["M_C"]
    MCT = two_stage["M_C"]
    ep = d["entropy_point"]
    budget = (1.0 + ep.get("budget_tol", 0.10)) * A["C"]
    C = np.array([p["C"] for p in sweep]) / C_YI
    E = np.array([p["E"] for p in sweep]) / C_YI
    CT = np.array([p["C"] for p in sweep_t]) / C_YI
    ET = np.array([p["E"] for p in sweep_t]) / C_YI
    feas = np.array([p["C"] <= budget and p.get("pen", 0.0) <= 1e-6
                     for p in sweep])
    feas_t = np.array([p["C"] <= budget and p.get("pen", 0.0) <= 1e-6
                       for p in sweep_t])
    # 两条方法前沿合并后的全局非支配前沿。
    front = sorted((p["E"] / C_YI, p["C"] / C_YI)
                   for p in d["fair_decision"]["combined_global_front"])
    plt.figure(figsize=(7.8, 5.6))
    plt.scatter(E[~feas], C[~feas], s=42, marker="x", color="#999999",
                label="Joint solutions excluded by feasibility/budget filter")
    plt.scatter(ET[~feas_t], CT[~feas_t], s=38, marker="+", color="#b0b0b0",
                label="Two-stage solutions excluded by feasibility/budget filter")
    plt.scatter(E[feas], C[feas], s=46, color="#4c72b0", alpha=0.85,
                label="Joint feasible weight-scan solutions")
    plt.scatter(ET[feas_t], CT[feas_t], s=42, marker="s", color="#55a868",
                alpha=0.78, label="Two-stage feasible weight-scan solutions")
    if front:
        fe, fc = zip(*front)
        plt.plot(fe, fc, "-", color="#333333", lw=1.4, alpha=0.9,
                 label="Combined non-dominated front")
    plt.scatter([A["E"] / C_YI], [A["C"] / C_YI], s=130, marker="s",
                color="#333333", zorder=5, label="M-A existing")
    plt.scatter([MC["E"] / C_YI], [MC["C"] / C_YI], s=210, marker="*",
                color="#c44e52", edgecolor="k", zorder=6,
                label=f"Joint decision (w1={ep.get('w1_selected', 0):.2f})")
    plt.scatter([MCT["E"] / C_YI], [MCT["C"] / C_YI], s=145, marker="D",
                color="#dd8452", edgecolor="k", zorder=6,
                label=f"Two-stage decision (w1={d['fair_decision']['two_stage']['w1']:.2f})")
    plt.axhline(budget / C_YI, ls="--", color="#e8a33d", lw=1.2,
                label=f"Budget constraint C = {budget/C_YI:.1f}")
    plt.xlabel("Life-cycle traffic energy E (10^8 RMB)")
    plt.ylabel("Life-cycle cost C (10^8 RMB)")
    plt.title("Fig. C8  Joint and two-stage Pareto solution sets")
    plt.legend(frameon=False, fontsize=9)
    plt.grid(alpha=0.3)
    _save("图C8_能耗-成本完整Pareto解集")


def main():
    global RES, RES_TWO
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=RES,
                    help="当前联合主结果，默认W500固定端点、无密度约束结果")
    ap.add_argument("--two-stage", default=RES_TWO,
                    help="与当前联合结果绑定的两阶段对照结果")
    args = ap.parse_args()
    RES = os.path.abspath(args.result)
    RES_TWO = os.path.abspath(args.two_stage)
    d = load()
    two_stage = _validate_sources(d)
    table_C1(d); table_C2(d); table_C3(d, two_stage)
    fig_C1(d, two_stage); fig_C2(d); fig_C3(d); fig_C4(d); fig_C5(d)
    fig_C6(d); fig_C7(d); fig_C8(d, two_stage)
    fig_C9_alllayers(d, _save)
    _write_source_provenance(d, two_stage)
    print("[完成] 全部图表已输出到 figures/ 与 tables/")


if __name__ == "__main__":
    main()
