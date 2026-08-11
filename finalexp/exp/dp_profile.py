# -*- coding: utf-8 -*-
"""
dp_profile.py — 给定平面线位下的纵断面全局优化(动态规划)。

依据: 双向平均口径下, |i|≤4% 的缓坡对能耗近似免费(实测同里程全平坡的 E 反而比
现状高 0.5%), 纵断面子问题退化为「坡度 ±4%、坡差 ≤dg_lim 约束下的最小土方费」。
该问题在 (桩号 × 高程 × 上一段坡度) 状态空间上可用 DP 全局求解, 避开进化搜索的
局部最优。DP 目标只含 C_TU(含结构替代封顶), 求解后用真实目标复核。

状态: (变坡点 i, 高程格 z_idx, 进入坡度格 dz_idx)。
  z 步长 DZ=0.5 m; 每段 Δz ∈ [−gmax·ds, +gmax·ds]; 相邻段 |Δz−Δz'| ≤ dg_lim·ds。
"""
import os as _os
import numpy as np

from params import EARTHWORK, CASE, BRIDGE_TUNNEL, LONG_STD_100

DP_DZ = float(_os.environ.get('DP_DZ', 0.5))      # 高程格(m), 敏感性可调
IMIN_ENFORCE = _os.environ.get('IMIN_ENFORCE', '1') == '1'  # 式4.27 下界 i_min=0.3%(排水)


def _cost_per_m(h_signed):
    """每米路线的土方费(含桥/隧封顶), h_signed = 设计-地面(m)。"""
    W = CASE["road_width_m"]; m = EARTHWORK["side_slope"]
    h = np.abs(h_signed)
    area = W * h + m * h * h
    fill = EARTHWORK["Kh_fill_per_m3"] * area
    cut = EARTHWORK["Ks_cut_per_m3"] * area
    bcap = BRIDGE_TUNNEL["bridge_cost_per_km"] / 1000.0
    tcap = BRIDGE_TUNNEL["tunnel_cost_per_km"] / 1000.0
    c = np.where(h_signed > 0, np.minimum(fill, bcap), np.minimum(cut, tcap))
    return np.where(h == 0, 0.0, c)


def solve_profile(sta_ctrl, gz_ctrl, z0_range=20.0, DZ=None, dg_lim=0.028,
                  include_energy=True, energy_weight=1.0):
    # energy_weight: 能耗相对土方的权重比; 熵权口径取 (wE/E_ref)/(wC/C_ref)。
    # DP 的 argmin 只与两项比值有关, 整体缩放不影响。
    """
    返回变坡点最优设计高程 z*(len = len(sta_ctrl)), 最小化 Σ 每米土方费×段长。
    """
    if DZ is None:
        DZ = DP_DZ
    n = len(sta_ctrl)
    ds = float(np.median(np.diff(sta_ctrl)))
    gmax = LONG_STD_100["grade_max"]
    kmax = int(np.floor(gmax * ds / DZ + 1e-9))          # 每段 Δz 格数上限
    kjump = max(int(np.floor(dg_lim * ds / DZ + 1e-9)), 1)
    dzs = np.arange(-kmax, kmax + 1)                      # Δz 选择(格)
    if IMIN_ENFORCE:
        # 论文式(4.27) 下界: |i| ≥ i_min=0.3%(排水)。剔除坡度过小的 Δz 选择,
        # 约束由构造满足(此前原仓库/主实验均未实施该下界)。
        gmin_dr = 0.003
        dzs = dzs[np.abs(dzs * DZ / ds) >= gmin_dr]
    nd = len(dzs)

    zlo = float(gz_ctrl.min() - 40.0)
    zhi = float(gz_ctrl.max() + 40.0)
    zg = np.arange(zlo, zhi + DZ, DZ)
    nz = len(zg)

    # 每个变坡点 × 高程格 的节点费用(该点每米费 × 代表段长)
    seg_len = np.empty(n)
    seg_len[0] = (sta_ctrl[1] - sta_ctrl[0]) / 2
    seg_len[-1] = (sta_ctrl[-1] - sta_ctrl[-2]) / 2
    seg_len[1:-1] = (sta_ctrl[2:] - sta_ctrl[:-2]) / 2
    node_cost = _cost_per_m(zg[None, :] - gz_ctrl[:, None]) * seg_len[:, None]

    BIG = 1e18
    # DP 表: V[z_idx, dz_idx] = 到达变坡点 i、高程 zg[z]、进入坡度 dzs[dz] 的最小费
    V = np.full((nz, nd), BIG)
    z0lo = np.searchsorted(zg, gz_ctrl[0] - z0_range)
    z0hi = np.searchsorted(zg, gz_ctrl[0] + z0_range)
    # 起点: 无进入坡度限制 -> 对所有 dz 同置起点费
    V[z0lo:z0hi + 1, :] = node_cost[0, z0lo:z0hi + 1][:, None]
    parent = np.zeros((n, nz, nd), dtype=np.int16)        # 记录前驱 dz 索引

    e_per_m = None
    if include_energy:
        g_choices = dzs * DZ / ds
        e_per_m = energy_money_per_m(g_choices) * float(energy_weight)
    for i in range(1, n):
        Vn = np.full((nz, nd), BIG)
        par = np.zeros((nz, nd), dtype=np.int16)
        for jd, dz in enumerate(dzs):
            # 从 (z-dz, dz_prev) 转移到 (z, dz), 要求 |dz-dz_prev|<=kjump
            lo, hi = max(0, jd - kjump), min(nd, jd + kjump + 1)
            Wv = V[:, lo:hi].min(axis=1)
            Wa = V[:, lo:hi].argmin(axis=1) + lo
            if dz > 0:
                cand = np.full(nz, BIG); pa = np.zeros(nz, dtype=np.int64)
                cand[dz:] = Wv[:-dz]; pa[dz:] = Wa[:-dz]
            elif dz < 0:
                cand = np.full(nz, BIG); pa = np.zeros(nz, dtype=np.int64)
                cand[:dz] = Wv[-dz:]; pa[:dz] = Wa[-dz:]
            else:
                cand = Wv.copy(); pa = Wa
            tot = cand + node_cost[i]
            if e_per_m is not None:
                tot = tot + e_per_m[jd] * ds             # 该段能耗货币(转移费)
            better = tot < Vn[:, jd]
            Vn[better, jd] = tot[better]
            par[better, jd] = pa[better]
        V = Vn
        parent[i] = par

    # 回溯
    flat = np.argmin(V)
    zi, di = np.unravel_index(flat, V.shape)
    z_idx = np.empty(n, dtype=np.int64)
    dz_idx = np.empty(n, dtype=np.int64)
    z_idx[-1], dz_idx[-1] = zi, di
    for i in range(n - 1, 0, -1):
        pd = parent[i, z_idx[i], dz_idx[i]]
        z_idx[i - 1] = z_idx[i] - dzs[dz_idx[i]]
        dz_idx[i - 1] = pd
    return zg[z_idx], float(V[zi, di])


def profile_to_x(z_star, gz_ctrl, sta_ctrl, x_base):
    """把 DP 高程序列编码回决策向量(起点高程 + 各段纵坡)。"""
    from objective_joint import N_MODE, START_AMP_M
    gmax = LONG_STD_100["grade_max"]
    x = np.array(x_base, float)
    x[N_MODE] = np.clip(0.5 + (z_star[0] - gz_ctrl[0]) / (2.0 * START_AMP_M), 0, 1)
    g = np.diff(z_star) / np.diff(sta_ctrl)
    x[N_MODE + 1:] = np.clip(0.5 + g / (2.0 * gmax), 0.0, 1.0)
    return x


def energy_money_per_m(grades):
    """
    每米路线的全生命周期能耗货币(元/m), 双向平均, 只依赖坡度(忽略 Fa 的微小曲率项)。
    与 objective.fuel_energy / ev_energy 完全同参数(式4.5-4.18 + 年金现值口径)。
    """
    import numpy as np
    from params import (FUEL_CAR, EV, PHYS, TRAFFIC, ENERGY_PRICE, LCC, CASE)
    v = CASE["design_speed_kmh"] / 3.6
    g = PHYS["g"]; rho = PHYS["rho_air"]
    th = np.arctan(np.asarray(grades, float))
    out = 0.0
    lc = 365.0 * sum(1.0 / (1 + LCC["bank_rate"]) ** k
                     for k in range(1, LCC["analysis_years"] + 1))
    AADT = TRAFFIC["AADT"]

    def one_dir(theta):
        fc = FUEL_CAR
        Fair = 0.5 * rho * fc["C_aero"] * fc["A_front"] * v * v
        Fr = fc["Cr"] * fc["mass_kg"] * g * np.cos(theta)
        Fg = fc["mass_kg"] * g * np.sin(theta)
        HPex = np.maximum((Fair + Fr + Fg) * v / (PHYS["W_denom"] * fc["eta"]), 0.0)
        UFC = fc["phi"] * (fc["HPin"] + HPex)          # ml/s
        ml_per_m = UFC / v                              # ml/m
        fuel = ml_per_m / 1000.0 * ENERGY_PRICE["Kf_fuel"]
        ev = EV
        Fair2 = 0.5 * rho * ev["C_aero"] * ev["A_front"] * v * v
        Fr2 = ev["Cr"] * ev["mass_kg"] * g * np.cos(theta)
        Fg2 = ev["mass_kg"] * g * np.sin(theta)
        Fe = np.where(theta >= 0, Fair2 + Fr2 + Fg2, Fair2 + Fr2 - np.abs(Fg2))
        eff = ev["ea"] * ev["eb"]
        E_m = np.where(Fe >= 0, Fe / eff, Fe * eff)     # J/m
        kwh_per_m = E_m / 3.6e6 + ev["EH_kwh_per_100"] / 100.0 / 1000.0
        elec = kwh_per_m * ENERGY_PRICE["ZQ_elec"]
        return (TRAFFIC["n1_fuel"] * fuel + TRAFFIC["n2_ev"] * elec)

    if _os.environ.get('E_DIRECTION', 'avg') == 'single':
        money = one_dir(th)                              # 论文式(4.4) 单向口径
    else:
        money = 0.5 * (one_dir(th) + one_dir(-th))       # 双向平均
    return AADT * money * lc                             # 元/m(全周期)
