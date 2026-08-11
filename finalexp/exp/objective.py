# -*- coding: utf-8 -*-
"""
objective.py — 立体线形双目标(全生命周期成本 C / 油电混合能耗 E)与熵权标量化 F

公式全部取自林坤锐学位论文, 关键式号已在代码中标注:
  C = C_PING + C_TU
    C_PING = CR + CB + CS + CQ          (式3.41)
    C_TU   = EL + ΣKs·Vs + ΣKh·Vh       (式4.3)
  E = ω1·n1·E_fuel + ω2·n2·E_ele        (式4.4 / 4.26)
    E_fuel: 式4.5-4.12   E_ele: 式4.13-4.18
  F = wC·Cnorm + wE·Enorm               (熵权法 式5.3-5.4)

决策变量 X (实数编码, 归一到[0,1], 由 alignment_decode 映射到工程量纲):
  纵断面变坡点高程 H_i (§4.5.1 第4组变量: 里程固定, 优化高程)
    —— 里程按等间距桩号固定(数据重采样), 与论文式(4.24)一致
"""
import numpy as np
from params import (FUEL_CAR, EV, PHYS, TRAFFIC, ENERGY_PRICE, COST_UNIT,
                    LCC, EARTHWORK, BRIDGE_TUNNEL, DESIGN_STD, LONG_STD_100,
                    CASE, FLAT_STD_100, MAINTENANCE)


# =============================================================
#  决策变量 <-> 纵断面线形
# =============================================================
def decode(x, sta, gz):
    """
    将[0,1]决策向量解码为各变坡点设计高程。
    变量维度 = 变坡点个数(等于重采样桩号数)。
    设计高程 = 地面高程 ± 允许调整幅度, 幅度取 ±[地面起伏范围] 内。
    """
    z_min, z_max = gz.min(), gz.max()
    span = (z_max - z_min)
    amp = max(span * 0.6, 10.0)          # 允许高程调整幅度(m)
    design_z = gz + (x - 0.5) * 2.0 * amp  # x=0.5 -> 贴合地面
    return design_z


def _grades(sta, z):
    """相邻变坡点纵坡 g_i = ΔH/ΔP (式4.19)。"""
    return np.diff(z) / np.diff(sta)


# =============================================================
#  土石方工程成本 C_TU  (式4.3 平均断面法)
# =============================================================
def earthwork_cost(sta, gz, design_z, road_width,
                   bridge_cap_per_m=None, tunnel_cap_per_m=None):
    """
    平均断面法(§4.3.2): 填/挖面积 ≈ 断面宽 × |设计高程-地面高程|,
    体积 = 相邻断面平均面积 × 桩号间距, 再乘单位费用并加调运费(式4.3)。

    【结构替代(论文§3.4.2 判据的成本形式)】
      bridge_cap_per_m / tunnel_cap_per_m: 桥梁/隧道单位造价(元/m)。给出时,
      每米路线的填方费用按 min(土方费, 桥单价)、挖方费用按 min(土方费, 隧单价)
      计——土方随高差二次增长, 超过结构单价时工程上会改建桥/隧, 费用以结构单价封顶。
      性质: ①单调, 抬高/压低设计线不会更便宜, 无"悬浮免费"漏洞;
            ②连续, 不存在"凑到 30 m 阈值突然变便宜"的畸形激励;
            ③论文 30 m 判据由单价内生地近似给出(挖方在 h≈69 m 处交叉;
              填方因表6.6 桥单价 2000 元/m 偏低, 在 h≈2.8 m 即交叉, 需在论文中声明)。
      None 时为纯土方(原行为)。

    返回 (C_TU, Vs, Vh, extra):
      Vs/Vh 只统计【未被结构替代】段的挖/填方量;
      extra = dict(L_bridge_new_km, L_tunnel_new_km) 为结构替代段诊断。
    """
    dz = design_z - gz                    # >0 填方, <0 挖方
    # 断面填/挖面积(含边坡放坡, 近似梯形): A = W*h + m*h^2
    m = EARTHWORK["side_slope"]
    h = np.abs(dz)
    area = road_width * h + m * h * h     # m^2 (即每米路线的土方体积 m^3/m)
    fill_cost_pm = np.where(dz > 0, EARTHWORK["Kh_fill_per_m3"] * area, 0.0)
    cut_cost_pm = np.where(dz < 0, EARTHWORK["Ks_cut_per_m3"] * area, 0.0)

    use_bridge = np.zeros(len(sta), dtype=bool)
    use_tunnel = np.zeros(len(sta), dtype=bool)
    if bridge_cap_per_m is not None:
        use_bridge = (dz > 0) & (fill_cost_pm > bridge_cap_per_m)
        fill_cost_pm = np.where(use_bridge, bridge_cap_per_m, fill_cost_pm)
    if tunnel_cap_per_m is not None:
        use_tunnel = (dz < 0) & (cut_cost_pm > tunnel_cap_per_m)
        cut_cost_pm = np.where(use_tunnel, tunnel_cap_per_m, cut_cost_pm)

    dL = np.diff(sta)
    seg_pm = 0.5 * (fill_cost_pm[:-1] + fill_cost_pm[1:]) \
        + 0.5 * (cut_cost_pm[:-1] + cut_cost_pm[1:])
    C_TU = EARTHWORK["EL_haul"] + float(np.sum(seg_pm * dL))

    # 土方量诊断: 只统计未被结构替代的断面
    fill_area = np.where((dz > 0) & ~use_bridge, area, 0.0)
    cut_area = np.where((dz < 0) & ~use_tunnel, area, 0.0)
    Vh = float(np.sum(0.5 * (fill_area[:-1] + fill_area[1:]) * dL))
    Vs = float(np.sum(0.5 * (cut_area[:-1] + cut_area[1:]) * dL))
    # 结构替代段诊断: 等效桥/隧长度(按段中点判定)与结构费
    br_seg = 0.5 * (use_bridge[:-1] + use_bridge[1:])
    tu_seg = 0.5 * (use_tunnel[:-1] + use_tunnel[1:])
    extra = dict(
        L_bridge_new_km=float(np.sum(br_seg * dL)) / 1000.0,
        L_tunnel_new_km=float(np.sum(tu_seg * dL)) / 1000.0,
    )
    return C_TU, Vs, Vh, extra


# =============================================================
#  平面全生命周期成本 C_PING = CR + CB + CS + CQ  (式3.41)
# =============================================================
def lcc_ping(total_len_m, sta, gz, design_z):
    """
    平面/结构全生命周期成本(式3.41-3.55)。
    本消融实验中里程固定, 主要随纵断面填挖高度变化 CB(桥隧)与 CQ(养护)。
    """
    L = total_len_m
    dz = design_z - gz

    # (1) 工程占地总费用 CR (式3.42-3.44): 永久占地(按耕地估) + 临时占地
    width = CASE["road_width_m"]
    area_mu = (L * width) / 666.67        # m^2 -> 亩
    # 永久占地: 按耕地净产值折现30年(式3.42简化: 直接按单价计)
    CM = area_mu * COST_UNIT["farmland_per_mu"]
    CP = (L / 1000.0) * 5000.0            # 临时占地(式3.43, 每公里工程标定)
    CR = CM + CP

    # (2) 桥梁隧道建设费用 CB: 实测桥隧长度 × 分类单位造价, 优化方案按里程缩短比例同比缩短
    #   现状桥隧长度取自 数据/北环高速现状桥梁隧道统计.xlsx(桥7.52km, 隧1.35km);
    #   单位造价 桥5.5074e6元/km、隧1.10705e8元/km(550.74万元/km、11070.5万元/km)。
    #   优化后平面里程 L 相对现状基准 L_ref 缩短, 桥隧长度按同一比例缩短(平面走向缩短
    #   -> 需跨越的桥隧同比减少)。现状方案 L=L_ref 时比例=1.0, 用实测桥隧全长。
    ratio = (L / 1000.0) / BRIDGE_TUNNEL["L_ref_km"]     # 优化里程/现状里程
    L_bridge = BRIDGE_TUNNEL["bridge_exist_km"] * ratio  # 桥梁长度(km)
    L_tunnel = BRIDGE_TUNNEL["tunnel_exist_km"] * ratio  # 隧道长度(km)
    CB = L_bridge * BRIDGE_TUNNEL["bridge_cost_per_km"] \
        + L_tunnel * BRIDGE_TUNNEL["tunnel_cost_per_km"]

    # (3) 基本建设费用 CS (式3.52-3.54): 路基相关 + 路面
    CV = COST_UNIT["subgrade_per_m"] * L                     # 式3.53 (δ·LU)
    CX = COST_UNIT["pavement_per_m"] * L                     # 式3.54 表6.6 路面 元/m
    CS = CV + CX

    # (4) 养护费用 CQ (式3.55):
    # CQ = Σ_j [1/(1+ru)^j] × Σ_i l_i × {γ + 365·T_j·τ + c_ij·[α·Ws·ms² + β·Wh·mh²
    #       + (1-α-β)·(Ws·ms² + Wh·mh²)]}
    # 其中: T_j = 第j年预测日交通量 = AADT × (1+0.014·rj)^j; 365·T_j = 年交通量
    #       Ws,Wh = 挖/填处横断面宽度; ms,mh = 挖/填边坡坡率
    t = LCC["analysis_years"]
    ru = LCC["bank_rate"]
    AADT = TRAFFIC["AADT"]
    r_j_pct = TRAFFIC["r_growth"] * 100   # 增长系数转百分数(式3.55中 rj 为百分数)
    m_s = EARTHWORK["side_slope"]          # 挖方边坡坡率 ms (1:1.5)
    m_h = EARTHWORK["side_slope"]          # 填方边坡坡率 mh (同值)
    gamma = MAINTENANCE["gamma"]
    tau = MAINTENANCE["tau"]
    c_ij = MAINTENANCE["c_soil"]
    alpha = MAINTENANCE["alpha"]
    beta = MAINTENANCE["beta"]

    dL = np.diff(sta)
    # 各段填/挖横断面宽度: 有填挖时取路基宽度, 无则为0
    W_s_arr = np.where(dz[:-1] < 0, width, 0.0)   # 挖方段 Ws = 路基宽度
    W_h_arr = np.where(dz[:-1] > 0, width, 0.0)   # 填方段 Wh = 路基宽度

    # 坡面养护项 (式3.55 大括号内第三项)
    slope_term = (alpha * W_s_arr * m_s**2
                  + beta * W_h_arr * m_h**2
                  + (1 - alpha - beta) * (W_s_arr * m_s**2 + W_h_arr * m_h**2))

    # 逐年折现累加
    CQ = 0.0
    for j in range(1, t + 1):
        # T_j = 第j年预测日交通量; 365·T_j = 年交通量 (式3.55 定义)
        T_j = AADT * (1 + 0.014 * r_j_pct) ** j
        discount_j = 1.0 / (1 + ru) ** j
        yearly = np.sum(dL * (gamma + 365.0 * T_j * tau + c_ij * slope_term))
        CQ += discount_j * yearly

    C_PING = CR + CB + CS + CQ            # 式3.41
    return C_PING, dict(CR=CR, CB=CB, CS=CS, CQ=CQ, L_bridge=L_bridge, L_tunnel=L_tunnel)


# =============================================================
#  燃油车运行油耗 E_fuel  (式4.5-4.12)
# =============================================================
def _lateral_force(mass, v, R, SP_max, NV, CS, g):
    """
    横向阻力 Fa (式4.12): Fa = (m·v²/R − m·g·SP)/(NV·CS)。

    超高 SP 按规范取值: 理论平衡超高为 v²/(gR), 但规范限定最大超高 SP_max,
    故 SP = min(SP_max, v²/(gR))。半径足够大时超高可完全平衡离心力 -> Fa=0(直线段
    口径 UFC_Z); 半径偏小时残余离心力形成横向阻力 -> Fa>0(曲线段口径 UFC_Q)。
    R 为 None 时按直线处理(Fa=0)。

    【对论文的偏离: 去掉式(4.12) 末尾的 10⁻³ 因子】
      按论文原式保留 10⁻³ 时, 全线最急弯(R=430 m)处残余离心力 1515 N 被 NV·CS=172
      再乘 10⁻³ 缩小 17.2 万倍, 仅剩 0.0088 N, 而同工况空气阻力为 270 N, 相差约
      3 万倍 -> 实测曲率对全线能耗的影响为 +0.0000%, 式(4.5) 区分 UFC_Z / UFC_Q
      失去意义, 平面线形无法通过能耗通道影响优化。
      判定 10⁻³ 为原文笔误并去掉后, 同一工况 Fa≈8.8 N, 占空气阻力约 3%, 量级合理,
      曲率-能耗耦合成立。此偏离须在论文中显式声明。
    """
    if R is None:
        return 0.0
    R = np.maximum(np.asarray(R, float), 1e-9)
    SP = np.minimum(SP_max, v * v / (g * R))
    Fa = (mass * v * v / R - mass * g * SP) / (NV * CS)
    return np.maximum(Fa, 0.0)


def fuel_energy(sta, design_z, v_kmh, R=None):
    """
    小汽车运行油耗(式4.5-4.12)。逐坡段按纵坡 θ 与曲率半径 R 计算外部功率->油耗率->油耗。
    R 为各坡段的平曲线半径(m), 缺省 None 表示按直线段处理(式4.5 的 UFC_Z 口径)。
    返回单车单程油耗(ml)。
    """
    v = v_kmh / 3.6                       # m/s
    fc = FUEL_CAR
    m = fc["mass_kg"]; g = PHYS["g"]; rho = PHYS["rho_air"]
    grades = _grades(sta, design_z)       # 各坡段纵坡(tanθ)
    theta = np.arctan(grades)             # 坡度角
    dL = np.diff(sta)                     # 坡段长(m)

    # 外部各阻力 (式4.9-4.12)
    Fair = 0.5 * rho * fc["C_aero"] * fc["A_front"] * v * v      # 空气阻力(式4.9)
    Fr = fc["Cr"] * m * g * np.cos(theta)                        # 滚动阻力(式4.10)
    Fg = m * g * np.sin(theta)                                   # 坡度阻力(式4.11)
    # 横向阻力 Fa (式4.12): 由曲率半径决定, 直线段为 0
    Fa = _lateral_force(m, v, R, FLAT_STD_100["superelev_max"],
                        fc["NV"], fc["CS"], g)
    F_ext = Fair + Fr + Fg + Fa                                  # 合力(N)
    # 外部功率 HPex (式4.7): (F·v)/(736·ω), ω 为发动机效率 85%
    HPex = (F_ext * v) / (PHYS["W_denom"] * fc["eta"])            # hp
    HPex = np.maximum(HPex, 0.0)                                  # 下坡不回收(燃油)
    # 油耗率 UFC = φ·(HPin+HPex) (式4.6); Fa=0 为 UFC_Z, Fa>0 为 UFC_Q (式4.5)
    UFC = fc["phi"] * (fc["HPin"] + HPex)                        # ml/s
    # 该坡段行驶时间 = dL/v, 油耗 = UFC × t
    seg_time = dL / v                                            # s
    ml = np.sum(UFC * seg_time)                                  # ml/单车单程
    return ml


# =============================================================
#  电动车运行能耗 E_ele  (式4.13-4.18)
# =============================================================
def ev_energy(sta, design_z, v_kmh, R=None):
    """
    电动车运行能耗(式4.13-4.18)。上坡耗电, 下坡再生制动回收(式4.17)。
    R 为各坡段的平曲线半径(m), 缺省 None 表示按直线段处理。
    返回单车单程耗电量(kWh)。
    """
    v = v_kmh / 3.6
    ev = EV
    m = ev["mass_kg"]; g = PHYS["g"]; rho = PHYS["rho_air"]
    grades = _grades(sta, design_z)
    theta = np.arctan(grades)
    dL = np.diff(sta)

    Fair = 0.5 * rho * ev["C_aero"] * ev["A_front"] * v * v
    Fr = ev["Cr"] * m * g * np.cos(theta)
    Fg = m * g * np.sin(theta)
    # 横向阻力 Fa (式4.12): 式4.17 规定上坡与下坡的阻力合力均含横向阻力
    Fa = _lateral_force(m, v, R, FLAT_STD_100["superelev_max"],
                        ev["NV"], ev["CS"], g)
    # 式4.17: 上坡阻力合力=空气+滚动+坡度+横向; 下坡=空气+滚动+横向(坡度分量转为回收)
    Fe = np.where(theta >= 0, Fair + Fr + Fg + Fa,
                  Fair + Fr + Fa - np.abs(Fg))
    # 牵引功率 Pe=Fe·ve (式4.16)
    Pe = Fe * v                                                  # W
    seg_time = dL / v                                            # s
    # 机械能(J) -> 考虑机械/放电效率(式4.15) -> kWh
    E_mech = Pe * seg_time                                       # J (可正可负)
    # 效率: 驱动时除以效率(更耗电), 回收时乘效率(打折回收)
    eff = ev["ea"] * ev["eb"]
    E_elec = np.where(E_mech >= 0, E_mech / eff, E_mech * eff)
    E_drive_kwh = np.sum(E_elec) / 3.6e6                         # J->kWh
    # 空调能耗 (式4.18): Eair=EH/100 每公里, 乘里程
    L_km = (sta[-1] - sta[0]) / 1000.0
    E_air_kwh = (ev["EH_kwh_per_100"] / 100.0) * L_km
    return max(E_drive_kwh, 0.0) + E_air_kwh


# =============================================================
#  两目标 C, E  (式4.25-4.26)
# =============================================================
def objectives(x, ctx):
    """
    输入决策向量 x∈[0,1]^dim, 返回 (C, E, penalty, info)。
    C: 全生命周期成本(元)  E: 油电混合车流日能耗(等价元, 折算日车流)
    """
    sta = ctx["sta"]; gz = ctx["gz"]; L = ctx["total_len_m"]
    v = CASE["design_speed_kmh"]
    design_z = decode(x, sta, gz)

    # ---- 目标一 C (式4.25) ----
    C_TU, Vs, Vh, _ = earthwork_cost(sta, gz, design_z, CASE["road_width_m"])
    C_PING, cinfo = lcc_ping(L, sta, gz, design_z)
    C = C_PING + C_TU                                           # 式4.25

    # ---- 目标二 E (式4.26): 全生命周期货币化能耗 (元) ----
    # 与 LCC 成本同口径: 日能耗 × 365 天 × 30 年等额年金现值系数(5%折现),
    # 使 C 与 E 均为全生命周期货币量(亿元)(与论文表6.8 能耗单位"亿元"一致)。
    ml = fuel_energy(sta, design_z, v)                          # ml/单车单程
    kwh = ev_energy(sta, design_z, v)                           # kWh/单车单程
    AADT = TRAFFIC["AADT"]
    n1, n2 = TRAFFIC["n1_fuel"], TRAFFIC["n2_ev"]
    # 全生命周期折现系数 (与养护费 CQ 式3.55 完全相同的 365×等额年金现值口径)
    # 单日能耗货币量 × 365 × 30年等额年金现值(5%折现) = 全周期货币量(亿元), 不另乘标定系数
    _t = LCC["analysis_years"]; _ru = LCC["bank_rate"]
    lc_factor = 365.0 * sum(1.0 / (1 + _ru) ** k for k in range(1, _t + 1))
    E_fuel = AADT * n1 * (ml / 1000.0) * ENERGY_PRICE["Kf_fuel"] * lc_factor  # 全周期 元
    E_ele = AADT * n2 * kwh * ENERGY_PRICE["ZQ_elec"] * lc_factor            # 全周期 元
    E = E_fuel + E_ele                                          # 式4.26(ω,n并入), 全周期元

    # ---- 约束惩罚 (§4.5.3 坡度/竖曲线) ----
    grades = _grades(sta, design_z)
    pen = 0.0
    gmax = LONG_STD_100["grade_max"]; gmin = LONG_STD_100["grade_min"]
    # 坡度约束 imin<=|i|<=imax (式4.27): 超限惩罚
    over = np.abs(grades) - gmax
    pen += np.sum(np.where(over > 0, over, 0.0)) * 1e9
    # 相邻坡度代数差过大(竖曲线约束 式4.28 近似): 限制变坡率
    dgrade = np.abs(np.diff(grades))
    pen += np.sum(np.where(dgrade > 0.03, dgrade - 0.03, 0.0)) * 5e8

    info = dict(C_PING=C_PING, C_TU=C_TU, E_fuel=E_fuel, E_ele=E_ele,
                Vs=Vs, Vh=Vh, ml=ml, kwh=kwh, **cinfo)
    return C, E, pen, info


# =============================================================
#  熵权法权重  (式5.3-5.4)
# =============================================================
def entropy_weights(C_arr, E_arr):
    """
    对 (C, E) 两指标用熵权法客观赋权(式5.3-5.4)。返回 (wC, wE)。

    流程按标准熵权法: ①极差标准化 -> ②比重 p_ij -> ③信息熵 e_j -> ④权重 w_j。

    【为何必须先做极差标准化】
      原实现直接用比重法 p_ij = X_ij/ΣX_ij(式5.3 的字面写法), 未消除两指标离散度的
      量级差异。实测在真实地形 DEM 下, 成本 C 的变异系数为 0.370(土方随纵坡剧变),
      而能耗 E 仅 0.022, 于是得到 wC=0.996、wE=0.004 —— 能耗被完全忽略, 双目标退化
      为单目标。补上极差标准化后权重反映分布形态而非量纲尺度。
      C 与 E 均为成本型指标(越小越优), 故按 (max − x)/(max − min) 标准化。
    """
    M = np.vstack([C_arr, E_arr]).T          # n_samples × 2
    n = M.shape[0]
    mn = M.min(axis=0, keepdims=True)
    mx = M.max(axis=0, keepdims=True)
    Z = (mx - M) / (mx - mn + 1e-12)         # ①极差标准化(成本型)
    Z = Z + 1e-6                             # 避免 p=0 时 log 发散
    P = Z / Z.sum(axis=0, keepdims=True)     # ②比重
    E_j = -(1.0 / np.log(n)) * np.sum(P * np.log(P), axis=0)   # ③信息熵(式5.3)
    d = 1.0 - E_j                                              # 差异系数
    w = d / (d.sum() + 1e-12)                                  # ④权重(式5.4)
    return float(w[0]), float(w[1])


def make_scalar_fn(ctx, wC, wE, C_ref, E_ref):
    """
    生成标量化目标 F = wC·Cnorm + wE·Enorm (实验设计方案2 §1.1)。
    用初始种群的参考尺度(C_ref,E_ref)做归一, 保证跨算法同口径。
    返回 f(x)->F (越小越优)。
    """
    def f(x):
        C, E, pen, _ = objectives(x, ctx)
        F = wC * (C / C_ref) + wE * (E / E_ref) + pen / C_ref
        return F
    return f
