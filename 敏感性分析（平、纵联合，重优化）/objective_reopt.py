# -*- coding: utf-8 -*-
"""
objective_reopt.py — 参数化的平纵联合优化目标（敏感性分析"每点重优化"版）

与 objective_joint.objectives_joint 完全同源、同公式（林坤锐学位论文式号见
objective.py / objective_joint.py），仅把四类敏感性外部参数由固定值改为可传入，
使得在每一个采样点上都能对【同一套公式、随该点参数变化的目标函数】重新寻优线形：

  P = dict(
    ev                : 电动车渗透率 p  -> n1=1-p(燃油), n2=p(电动)   (式4.4/4.26)
    traffic_growth    : 交通量增长率 rj -> 养护费(式3.55)与能耗当量交通量
    fuel_price_growth : 油价增长率      -> Kf·(1+·)                   (式4.5)
    elec_price_growth : 电价增长率      -> ZQ·(1+·)                   (式4.14)
    fuel_save         : 燃油节油率      -> ml·(1-·)                   (§6.5.3)
    elec_save         : 电动节能率      -> kwh·(1-·)                  (§6.5.3)
  )

口径与 objective_joint 一致：
  · C = 土方(式4.3) + 平面全周期(式3.41-3.55)，用【新里程/新桩号】；
    交通量增长通过养护费增量 ΔCQ 进入 C（与 run_sensitivity.py 同口径）。
  · E = 油电混合车流【全生命周期】货币化能耗（式4.4-4.18 + 365×30年5%年金现值），
    单位亿元，与 C 同口径、与论文表6.8 一致。
  · 约束惩罚（平曲线半径≥400m、纵坡≤4%、相邻坡差）与 objective_joint 完全一致。

【诚实声明】新点位地面高程取最近实测中线点（cKDTree），同 objective_joint 的数据
限制处理，非杜撰。桥隧费用系数论文未给取值，统一置 0（见 params.BRIDGE_TUNNEL）。
"""
import numpy as np

from params import (CASE, TRAFFIC, ENERGY_PRICE, LONG_STD_100, LCC, FLAT_STD_100,
                    MAINTENANCE)
from objective import (earthwork_cost, lcc_ping, fuel_energy, ev_energy, _grades)
from objective_joint import decode_joint


DEFAULT_P = dict(ev=TRAFFIC["n2_ev"], traffic_growth=0.0,
                 fuel_price_growth=0.0, elec_price_growth=0.0,
                 fuel_save=0.0, elec_save=0.0)


def _lc_factor():
    """全生命周期折现系数 365×30年等额年金现值(5%)，与 objective 完全一致。"""
    t = LCC["analysis_years"]; ru = LCC["bank_rate"]
    return 365.0 * sum(1.0 / (1 + ru) ** k for k in range(1, t + 1))


def _pv_factor():
    """等额年金现值系数（不含365），供养护费交通量增量用。"""
    t = LCC["analysis_years"]; ru = LCC["bank_rate"]
    return sum(1.0 / (1 + ru) ** k for k in range(1, t + 1))


def objectives_reopt(x, pc, P=None):
    """
    参数化联合目标：返回 (C, E, penalty, info)。C/E 用【新里程、新桩号】计算。
    P 缺省即基准情景（等价 objective_joint.objectives_joint）。
    """
    if P is None:
        P = DEFAULT_P
    ev = P.get("ev", DEFAULT_P["ev"])
    rj = P.get("traffic_growth", 0.0)
    fpg = P.get("fuel_price_growth", 0.0)
    epg = P.get("elec_price_growth", 0.0)
    fsave = P.get("fuel_save", 0.0)
    esave = P.get("elec_save", 0.0)

    d = decode_joint(x, pc)
    sta, gz_new, design_z, L_new = d["sta"], d["gz_new"], d["design_z"], d["L_new"]
    v = CASE["design_speed_kmh"]

    # ---- 目标一 C：土方(式4.3) + 平面全周期(式3.41-3.55) ----
    C_TU, Vs, Vh = earthwork_cost(sta, gz_new, design_z, CASE["road_width_m"])
    C_PING, cinfo = lcc_ping(L_new, sta, gz_new, design_z)
    # 养护费随交通量增长的增量 ΔCQ（式3.55 车流量项的敏感性偏差）:
    #   lcc_ping 已含基准增长率 r_growth 下的 CQ; 此处叠加"本采样点增长率 rj
    #   与基准增长率之差"引起的车流量项增量, 系数用 MAINTENANCE.tau(与式3.55一致)。
    AADT0 = TRAFFIC["AADT"]; t = LCC["analysis_years"]; ru = LCC["bank_rate"]
    r_base_pct = TRAFFIC["r_growth"] * 100          # 基准增长率(与 lcc_ping 一致)
    tau = MAINTENANCE["tau"]
    dCQ = 0.0
    for j in range(1, t + 1):
        T_j_sens = AADT0 * (1 + rj) ** j                       # 采样点增长率下第j年日交通量
        T_j_base = AADT0 * (1 + 0.014 * r_base_pct) ** j       # 基准增长下第j年日交通量
        dCQ += (1.0 / (1 + ru) ** j) * L_new * 365.0 * (T_j_sens - T_j_base) * tau
    C = C_PING + C_TU + dCQ

    # ---- 目标二 E：油电混合车流全生命周期能耗(式4.4-4.18) ----
    ml = fuel_energy(sta, design_z, v) * (1.0 - fsave)      # 节油率
    kwh = ev_energy(sta, design_z, v) * (1.0 - esave)       # 节能率
    n1 = 1.0 - ev; n2 = ev                                  # 车流构成随渗透率
    Kf = ENERGY_PRICE["Kf_fuel"] * (1.0 + fpg)              # 油价增长
    ZQ = ENERGY_PRICE["ZQ_elec"] * (1.0 + epg)              # 电价增长
    AADT = AADT0 * (1 + rj) ** (t / 2.0)                    # 当量日交通量(全周期中期)
    lc = _lc_factor()
    E_fuel = AADT * n1 * (ml / 1000.0) * Kf * lc   # 全周期 元
    E_ele = AADT * n2 * kwh * ZQ * lc              # 全周期 元
    E = E_fuel + E_ele

    # ---- 联合约束惩罚（与 objective_joint 完全一致） ----
    pen = 0.0
    Rmin = FLAT_STD_100["R_extreme_m"]; R = d["R"]
    pen += np.sum(np.where(R < Rmin, (Rmin - R) / Rmin, 0.0)) * 5e7   # 平曲线半径≥400m
    grades = _grades(sta, design_z)
    over = np.abs(grades) - LONG_STD_100["grade_max"]
    pen += np.sum(np.where(over > 0, over, 0.0)) * 1e9                # 纵坡≤4%(式4.27)
    dgrade = np.abs(np.diff(grades))
    pen += np.sum(np.where(dgrade > 0.03, dgrade - 0.03, 0.0)) * 5e8  # 相邻坡差(式4.28-4.29)

    info = dict(C_PING=C_PING, C_TU=C_TU, dCQ=dCQ, E_fuel=E_fuel, E_ele=E_ele,
                Vs=Vs, Vh=Vh, ml=ml, kwh=kwh, L_km=L_new / 1000.0,
                Rmin=float(R.min()), **cinfo)
    return C, E, pen, info


def make_scalar_reopt(pc, wC, wE, C_ref, E_ref, P=None):
    """标量化参数化联合目标 F = wC·Cnorm + wE·Enorm + 惩罚/C_ref。"""
    def f(x):
        C, E, pen, _ = objectives_reopt(x, pc, P)
        return wC * (C / C_ref) + wE * (E / E_ref) + pen / C_ref
    return f
