# -*- coding: utf-8 -*-
"""
objective_joint.py — 平纵断面【联合】优化目标 (方案B: 真正的平纵一体化协同优化)

区别于主实验 objective.py(固定平面, 只优化纵断面变坡点高程):
  本模块把【平面控制点法向偏移】与【纵断面变坡点高程】放入【同一决策向量】,
  在同一次 IJS 寻优中一起搜索, 实现平面走向与纵断面坡度的一体化协同。

求解方法: 三维解空间(x, y, z)协同——平面走向由每个桩号控制点的法向偏移决定其
  (x, y)坐标, 纵断面由每个桩号的高程 z 决定; 二者放入【同一决策向量】, 在同一次
  IJS 寻优中对 (x, y, z) 三维立体线形进行全面彻底的搜索。平面与纵断面的求解桩号
  步长均为 10 m(STEP_M), 即沿全线每 10 m 布设一个平面控制点与一个变坡点, 使平面
  与纵断面在全线范围内均可被充分调整。

决策向量 x ∈ [0,1]^(N_CTRL + M_PROF):
  x[:N_CTRL]  -> 平面 N_CTRL 个控制点法向偏移 δ_k ∈ [-W, W](走廊带半宽 W) -> 决定 (x,y)
  x[N_CTRL:]  -> 沿【新平面线位】等分的 M_PROF 个变坡点高程 z 调整
  其中 N_CTRL = M_PROF = 全线里程 / 10 m + 1(≈2247), 平面/纵断面步长均为 10 m。

公式全部沿用林坤锐学位论文(见 objective.py 式号), 模型公式不变:
  平面里程/坐标 式3.1-3.4; 平曲线最小半径 表3.2(≥400m);
  土方 式4.3; LCC 式3.41-3.55; 能耗 式4.4-4.18; 纵坡/竖曲线约束 式4.27-4.29。

【数据口径的诚实声明】
  数据.xlsx 仅提供实测中线一条轨迹的地面高程, 无面状 DEM(论文图6.6 有 DEM,
  本数据集未提供)。平面横向偏移后, 新点位的地面高程无法从数据直接读出, 本模块
  采用"取最近实测中线点的地面高程"作为近似(cKDTree 最近邻)。此近似在走廊带内、
  地形沿纵向变化为主时成立; 属数据限制下不杜撰的必要处理, 已在报告中说明。
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from params import (CASE, EARTHWORK, TRAFFIC, ENERGY_PRICE, LONG_STD_100, LCC,
                    FLAT_STD_100)
from objective import (earthwork_cost, lcc_ping, fuel_energy, ev_energy,
                       _grades)
from data_loader import load_alignment

CORRIDOR_HALF_W = 800.0     # 走廊带半宽(m), 与 GapB 一致
STEP_M = 10.0               # 平面/纵断面求解桩号步长(m): 全面彻底搜索

# 平面控制点数 N_CTRL 与纵断面变坡点数 M_PROF 均按 10 m 步长沿全线布置:
# n = 全线里程 / STEP_M + 1(≈2247), 使平面(x,y)与纵断面(z)在全线范围内均可充分调整。
def _n_stations(step_m=STEP_M):
    a = load_alignment()
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(a["X"]), np.diff(a["Y"])))])
    return int(np.floor(s[-1] / step_m)) + 1

N_CTRL = _n_stations()      # 平面控制点个数(每 10 m 一个) -> 决定 (x, y)
M_PROF = N_CTRL             # 纵断面变坡点个数(每 10 m 一个) -> 决定 z


def make_plane_context(align):
    """由实测线位预计算: 控制点、法向、KDTree(供地面高程最近邻查询)、原始里程。"""
    X, Y = align["X"], align["Y"]
    z = align["ground_z"]
    # 等弧长重采样 N_CTRL 个平面控制点 + 单位左法向
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    si = np.linspace(0, s[-1], N_CTRL)
    cx = np.interp(si, s, X); cy = np.interp(si, s, Y)
    tx = np.gradient(cx); ty = np.gradient(cy)
    tn = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tn, tx / tn
    tree = cKDTree(np.column_stack([X, Y]))     # 实测点 KDTree
    L0 = float(s[-1])
    amp = max(z.max() - z.min(), 10.0) * 0.6    # 纵断面高程可调幅度(同 objective.decode)
    return dict(cx=cx, cy=cy, nx=nx, ny=ny, tree=tree, gz_meas=z,
                X=X, Y=Y, L0=L0, amp=amp)


def build_plane(pc, delta):
    """由控制点法向偏移 delta 生成平滑平面线形(三次样条), 端点固定。返回密集点。"""
    px = pc["cx"] + delta * pc["nx"]
    py = pc["cy"] + delta * pc["ny"]
    px[0], py[0] = pc["cx"][0], pc["cy"][0]
    px[-1], py[-1] = pc["cx"][-1], pc["cy"][-1]
    try:
        tck, _ = splprep([px, py], s=0.0, k=3)
        # 样条密集采样点数与 10 m 步长匹配(至少覆盖每个控制点), 保证平面几何精度
        n_dense = max(N_CTRL * 2, 1200)
        uu = np.linspace(0, 1, n_dense)
        xx, yy = splev(uu, tck)
        return np.asarray(xx), np.asarray(yy)
    except Exception:
        return px, py


def _plane_metrics(xx, yy):
    """平面里程 L(m) 与曲率半径序列 R(m)。"""
    dx = np.gradient(xx); dy = np.gradient(yy)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    L = float(np.sum(np.hypot(np.diff(xx), np.diff(yy))))
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx * dx + dy * dy) ** 1.5 + 1e-9
    kappa = num / den
    R = np.where(kappa > 1e-9, 1.0 / kappa, 1e9)
    return L, R


def decode_joint(x, pc):
    """
    解码联合决策向量为 (新平面, 新桩号 sta, 新地面高程 gz_new, 设计高程 design_z)。
    """
    W = CORRIDOR_HALF_W
    delta = (x[:N_CTRL] - 0.5) * 2.0 * W
    xx, yy = build_plane(pc, delta)
    L_new, R = _plane_metrics(xx, yy)

    # 沿新线位弧长等分 M_PROF 个桩号
    sarc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))])
    sta = np.linspace(0, sarc[-1], M_PROF)
    xj = np.interp(sta, sarc, xx); yj = np.interp(sta, sarc, yy)

    # 新桩号点的地面高程 = 最近实测中线点高程(数据限制下的近似, 见模块说明)
    _, idx = pc["tree"].query(np.column_stack([xj, yj]))
    gz_new = pc["gz_meas"][idx]

    # 纵断面变坡点高程(同 objective.decode 口径)
    design_z = gz_new + (x[N_CTRL:] - 0.5) * 2.0 * pc["amp"]
    return dict(xx=xx, yy=yy, L_new=L_new, R=R, sta=sta,
                gz_new=gz_new, design_z=design_z)


def objectives_joint(x, pc):
    """联合目标: 返回 (C, E, penalty, info)。C/E 用【新里程、新桩号】计算。"""
    d = decode_joint(x, pc)
    sta, gz_new, design_z, L_new = d["sta"], d["gz_new"], d["design_z"], d["L_new"]
    v = CASE["design_speed_kmh"]

    # 目标一 C = 土方(式4.3) + 平面全周期(式3.41-3.55), 均用新里程/桩号
    C_TU, Vs, Vh = earthwork_cost(sta, gz_new, design_z, CASE["road_width_m"])
    C_PING, cinfo = lcc_ping(L_new, sta, gz_new, design_z)
    C = C_PING + C_TU

    # 目标二 E = 油电混合车流【全生命周期】能耗(式4.4-4.18, 用新桩号/新里程)
    # 与 LCC 同口径: 日能耗 × 365 天 × 30 年等额年金现值系数(5%折现) -> 全周期元,
    # 使 C 与 E 均为全生命周期货币量(亿元), 与论文表6.8 能耗单位"亿元"一致。
    ml = fuel_energy(sta, design_z, v)
    kwh = ev_energy(sta, design_z, v)
    AADT = TRAFFIC["AADT"]; n1, n2 = TRAFFIC["n1_fuel"], TRAFFIC["n2_ev"]
    # 单日能耗货币量 × 365 × 30年等额年金现值(5%折现) = 全周期货币量(亿元), 不另乘标定系数
    _t = LCC["analysis_years"]; _ru = LCC["bank_rate"]
    lc_factor = 365.0 * sum(1.0 / (1 + _ru) ** k for k in range(1, _t + 1))
    E_fuel = AADT * n1 * (ml / 1000.0) * ENERGY_PRICE["Kf_fuel"] * lc_factor  # 全周期 元
    E_ele = AADT * n2 * kwh * ENERGY_PRICE["ZQ_elec"] * lc_factor            # 全周期 元
    E = E_fuel + E_ele                                                       # 全周期 元

    # ---- 联合约束惩罚 ----
    pen = 0.0
    # (a) 平面: 最小平曲线半径 ≥ 极限值 400m (表3.2)
    Rmin = FLAT_STD_100["R_extreme_m"]
    R = d["R"]
    pen += np.sum(np.where(R < Rmin, (Rmin - R) / Rmin, 0.0)) * 5e7
    # (b) 纵断面: 纵坡 |i| ≤ 4% (式4.27)
    grades = _grades(sta, design_z)
    over = np.abs(grades) - LONG_STD_100["grade_max"]
    pen += np.sum(np.where(over > 0, over, 0.0)) * 1e9
    # (c) 竖曲线: 相邻纵坡代数差限制(式4.28-4.29 近似)
    dgrade = np.abs(np.diff(grades))
    pen += np.sum(np.where(dgrade > 0.03, dgrade - 0.03, 0.0)) * 5e8

    info = dict(C_PING=C_PING, C_TU=C_TU, E_fuel=E_fuel, E_ele=E_ele,
                Vs=Vs, Vh=Vh, ml=ml, kwh=kwh, L_km=L_new / 1000.0,
                Rmin=float(R.min()), **cinfo)
    return C, E, pen, info


def make_scalar_joint(pc, wC, wE, C_ref, E_ref):
    """标量化联合目标 F = wC·Cnorm + wE·Enorm + 惩罚/C_ref。"""
    def f(x):
        C, E, pen, _ = objectives_joint(x, pc)
        return wC * (C / C_ref) + wE * (E / E_ref) + pen / C_ref
    return f


# =============================================================
#  两阶段(先平面后纵断面)对照所需的平面阶段 helper
#  仅供 run_twostage.py 复现"先平面优化、再纵断面优化"的分阶段方法作为对比;
#  平面/纵断面步长同为 10m(N_CTRL=M_PROF, 见上), 桥隧数据同联合方案(params.py)。
# =============================================================
def build_plane_from_delta(pc, delta_norm):
    """由平面控制点归一化偏移 delta_norm∈[0,1]^N_CTRL 生成新平面。返回密集点+里程+R。"""
    delta = (np.asarray(delta_norm) - 0.5) * 2.0 * CORRIDOR_HALF_W
    xx, yy = build_plane(pc, delta)
    L_new, R = _plane_metrics(xx, yy)
    return xx, yy, L_new, R


def plane_lcc(L_m):
    """
    第一阶段平面目标: 平面相关全生命周期成本(随里程变化的项), 论文 §3.4.2:
      工程占地 CR(式3.42-3.44) + 基本建设 CS(式3.52-3.54) + 养护 CQ 的基础/交通量项(式3.55)。
    这些项均正比于里程 L, 故最小化平面 LCC 等价于在满足平曲线约束下缩短里程/占地。
    桥隧 CB 与土方 C_TU 分别属结构/纵断面阶段, 此处平面阶段不计(最终 C 仍由 objectives_joint 全量计)。
    """
    from params import COST_UNIT
    width = CASE["road_width_m"]
    area_mu = (L_m * width) / 666.67
    CR = area_mu * COST_UNIT["farmland_per_mu"] + (L_m / 1000.0) * 5000.0   # 式3.42-3.44
    CS = COST_UNIT["subgrade_per_m"] * L_m + COST_UNIT["pavement_per_m"] * (L_m / 1000.0)  # 式3.52-3.54
    t = LCC["analysis_years"]; ru = LCC["bank_rate"]
    pv = sum(1.0 / (1 + ru) ** k for k in range(1, t + 1))
    T_year = TRAFFIC["AADT"] * 365
    CQ = 0.02 * CS * pv + 1e-4 * T_year * pv                                # 式3.55 基础/交通量项
    return CR + CS + CQ


def make_scalar_plane(pc, C_ref_plane):
    """第一阶段平面标量目标: min 平面LCC + 平曲线半径惩罚(R>=400m, 表3.2)。"""
    Rmin_req = FLAT_STD_100["R_extreme_m"]
    def f(delta_norm):
        _, _, L_new, R = build_plane_from_delta(pc, delta_norm)
        C_plane = plane_lcc(L_new)
        pen = np.sum(np.where(R < Rmin_req, (Rmin_req - R) / Rmin_req, 0.0)) * 5e7
        return C_plane / C_ref_plane + pen / C_ref_plane
    return f
