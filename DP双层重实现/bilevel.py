# -*- coding: utf-8 -*-
"""
bilevel.py — 双层求解核心: 外层搜平面(N_MODE 维), 内层 DP 全局解纵断面。

【为何这样分解】
  纵断面子问题在"坡度上下界 + 相邻坡差"约束下是 (桩号 × 高程 × 进入坡度) 状态
  空间上的最短路问题, dp_profile.solve_profile 可全局最优求解; 于是 275 维联合
  搜索退化为 50 维外层(平面) + 精确内层(纵断面)。

【规范约束如何被满足】
  纵坡上界 |i| ≤ 4%(式4.27 上界)   : DP 状态空间构造保证(kmax)
  纵坡下界 |i| ≥ 0.3%(排水)        : DP 状态空间构造保证(IMIN_ENFORCE)
  相邻坡差 ≤ 3e-4×步长(竖曲线)     : DP 状态转移构造保证(kjump)
  首末接线高程                      : DP 起止状态锚定(z_tie)
  平曲线半径 R ≥ 400 m             : 外层惩罚 PENALTY["k_R"]=30(平面自由度所致)
  建筑 Tier2 禁区                   : 外层惩罚 DENSITY["k_forbid"](硬约束)
  建筑 Tier1 可穿越区               : 外层软代价 DENSITY["w_dense1"](被抑制但可行)
  即: 所有纵断面类约束由构造满足(惩罚恒为 0), 只有平面类约束需要惩罚。

【数据来源】
  高程   : dem.py 准天然地面(AWS Terrain Tiles z14 + road-removal 重建)
  河流等 : crossings.py OSM 障碍物(road≥primary / rail / water(river,canal))交叉触发桥
  建筑   : building_mask.py 密度三级分区(V1, 阈值由现状线位标定)
"""
import numpy as np

import objective_joint as OJ
from objective_joint import (DIM, N_MODE, decode_joint, objectives_joint,
                             make_plane_context, STEP_PROFILE_CTRL_M)
from params import (DENSITY, LONG_STD_100, ALGO)
from dp_profile import solve_profile

# 竖曲线坡差限值(与主线同口径: 3e-4/m × 变坡点步长)
DG_LIM = 3e-4 * STEP_PROFILE_CTRL_M


def modes_to_x(modes):
    """外层平面向量 -> 完整决策向量(纵断面位占位, 实际由 DP 覆盖)。"""
    x = np.full(DIM, 0.5)
    x[:N_MODE] = np.clip(np.asarray(modes, float), 0.0, 1.0)
    return x


def plane_terrain(modes, pc):
    """只取平面与沿线地形(纵断面为占位), 供 DP 使用。"""
    return decode_joint(modes_to_x(modes), pc)


def z_tie_of(pc):
    """既有路网接线高程(实测路面高程首末点)。"""
    return (float(pc["gz_meas"][0]), float(pc["gz_meas"][-1]))


# DP 内层权重比的饱和上限: wC->0(纯能耗端点)时比值发散, 而 DP 的 argmin 只依赖
# 该比值, 故用一个足够大的有界值代替无穷即可让能耗项主导, 避免除零。
EW_CAP = 1e6


def dp_energy_weight(wC, wE, C_ref, E_ref):
    """DP 内层的能耗/土方权重比 = (wE/E_ref)/(wC/C_ref)(熵权口径)。"""
    if wE <= 0.0:
        return 0.0                      # 纯成本端点: 内层只最小化土方费
    a = wC / C_ref
    if a <= 1e-30:
        return EW_CAP                   # 纯能耗端点: 能耗项主导
    return min((wE / E_ref) / a, EW_CAP)


def solve_inner(d, pc, energy_weight):
    """给定平面(已解码 d), 用 DP 求该平面下的最优纵断面(变坡点高程)。"""
    return solve_profile(d["sta_ctrl"], d["gz_ctrl"], dg_lim=DG_LIM,
                         include_energy=True, energy_weight=energy_weight,
                         z_tie=z_tie_of(pc))[0]


def evaluate(modes, pc, energy_weight, pen_scale=1.0, scenario=None):
    """外层平面向量 -> (C, E, pen, info, design_z_ctrl)。纵断面为 DP 全局最优。

    scenario: 敏感性情景(见 objectives_joint 文档), None 为基准口径。
    """
    x = modes_to_x(modes)
    d = decode_joint(x, pc)
    z_star = solve_inner(d, pc, energy_weight)
    C, E, pen, info = objectives_joint(x, pc, pen_scale=pen_scale,
                                       scenario=scenario,
                                       design_z_ctrl=z_star)
    return C, E, pen, info, z_star


def scalar_of(C, E, pen, info, wC, wE, C_ref, E_ref):
    """统一标量目标 F(四个实验共用同一定义)。

    Tier1 是软项、不进 pen: 否则可行性门控(pen<=0)会把它变成硬约束, 而现状线位
    M-A 本身就有约 2.0 km 落在 Tier1, "允许穿越"的设计目标会被静默摧毁。
    """
    soft = DENSITY["w_dense1"] * info["soft_dense1"] if OJ.DENSITY_ON else 0.0
    return wC * (C / C_ref) + wE * (E / E_ref) + pen + soft


def make_outer_f(pc, wC, wE, C_ref, E_ref, pen_scale=1.0, scenario=None):
    """构造外层标量目标 f(modes)->F, 供 IJS/JS/GA/PSO/GWO 等任意算法调用。"""
    ew = dp_energy_weight(wC, wE, C_ref, E_ref)

    def f(modes):
        C, E, pen, info, _ = evaluate(modes, pc, ew, pen_scale=pen_scale,
                                      scenario=scenario)
        return scalar_of(C, E, pen, info, wC, wE, C_ref, E_ref)
    return f


def make_outer_biobj(pc, wC, wE, C_ref, E_ref, pen_scale=1.0):
    """双目标版 f(modes)->[Cn, En], 供 NSGA-II 在 (C,E) 空间做非支配排序。"""
    ew = dp_energy_weight(wC, wE, C_ref, E_ref)

    def f(modes):
        C, E, pen, info, _ = evaluate(modes, pc, ew, pen_scale=pen_scale)
        soft = DENSITY["w_dense1"] * info["soft_dense1"] if OJ.DENSITY_ON else 0.0
        return np.array([C / C_ref + pen + soft, E / E_ref + pen + soft])
    return f


def existing_profile_ctrl(d, pc):
    """现状纵断面(实测路面高程)在变坡点上的取值, 供 M-A 基准使用。"""
    return np.interp(d["sta_ctrl"], pc["s_meas"], pc["gz_meas"])


def evaluate_existing(pc, scenario=None):
    """M-A 现状方案: 平面 δ=0(实测中线) + 实测路面高程(既有设计线)。"""
    x = modes_to_x(np.full(N_MODE, 0.5))
    d = decode_joint(x, pc)
    z_meas = existing_profile_ctrl(d, pc)
    C, E, pen, info = objectives_joint(x, pc, scenario=scenario,
                                       design_z_ctrl=z_meas)
    return C, E, pen, info, z_meas


def entropy_weights(C_arr, E_arr):
    """熵权法(式5.3-5.4, 极差标准化)。C/E 均为成本型指标(越小越优)。"""
    M = np.vstack([np.asarray(C_arr, float), np.asarray(E_arr, float)]).T
    mn, mx = M.min(axis=0, keepdims=True), M.max(axis=0, keepdims=True)
    rng = np.where(mx - mn < 1e-30, 1.0, mx - mn)
    Z = (mx - M) / rng                       # 成本型: 越小越优 -> 归一后越大越好
    S = Z.sum(axis=0, keepdims=True)
    S = np.where(S < 1e-30, 1.0, S)
    P = Z / S
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=0)
    ent = ent / np.log(max(M.shape[0], 2))
    dvar = 1.0 - ent
    if dvar.sum() < 1e-30:
        return 0.5, 0.5
    w = dvar / dvar.sum()
    return float(w[0]), float(w[1])


def baseline(pc, pop_size, seed=2025):
    """基准种群 -> 熵权与归一化基准(C_ref, E_ref), 四个实验共用同一口径。

    基准种群在【外层平面空间】采样, 每个个体的纵断面由 DP 给出, 故 C_ref/E_ref
    已经是"平面随机 + 纵断面最优"的尺度, 与后续搜索同口径。
    """
    rng = np.random.default_rng(seed)
    pop0 = rng.random((pop_size, N_MODE))
    pop0[0] = 0.5                                    # 现状平面必在初始种群内
    # 先用等权 DP 权重取得 C/E 样本(权重尚未知, 用 1.0 起步)
    Cs, Es = [], []
    for i in range(pop_size):
        C, E, _, _, _ = evaluate(pop0[i], pc, energy_weight=1.0)
        Cs.append(C); Es.append(E)
    Cs, Es = np.array(Cs), np.array(Es)
    wC, wE = entropy_weights(Cs, Es)
    return pop0, wC, wE, float(Cs.mean()), float(Es.mean())
