# -*- coding: utf-8 -*-
"""
objective_joint.py — 平纵断面【联合】优化目标 (方案B: 真正的平纵一体化协同优化)

区别于主实验 objective.py(固定平面, 只优化纵断面变坡点高程):
  本模块把【平面控制点法向偏移】与【纵断面变坡点高程】放入【同一决策向量】,
  在同一次 IJS 寻优中一起搜索, 实现平面走向与纵断面坡度的一体化协同。

求解方法: 三维解空间(x, y, z)协同——平面走向由平面控制点的法向偏移决定其 (x, y)
  坐标, 纵断面由每个桩号的高程 z 决定; 二者放入【同一决策向量】, 在同一次 IJS 寻优
  中对 (x, y, z) 三维立体线形一起搜索。

【决策变量的离散尺度 vs 目标函数的评价尺度(二者分开)】
  · 决策变量都取 400 m 一个(STEP_CTRL_M):
      平面 N_CTRL=57 个控制点(法向偏移 δ) + 纵断面 M_PROF=57 个【变坡点】高程。
  · 目标函数一律在 10 m 桩号上积分(STEP_EVAL_M): M_EVAL=2247 个评价桩号,
      土方(式4.3)、能耗(式4.4-4.18)、纵坡/坡差约束、边坡危险度全部按 10 m 计,
      评价精度不受决策维度下降的影响。变坡点高程线性插值到评价桩号, 故变坡点之间
      为等坡段、坡度只在变坡点处改变 —— 与真实纵断面设计一致。

  为何平面必须取 400 m(不能取 10 m): 平面受平曲线最小半径 R ≥ 400 m(表3.2 极限值)
  约束, 相邻控制点的横向错动 A 在波长 λ=2Δs 上产生的曲率半径为 R = λ²/(4π²A), 即
  Δs 越小、可行的 A 越小: Δs=10 m 时 A ≤ 2.5 cm, Δs=400 m 时 A ≤ 40 m。若平面取
  10 m, 则 ±800 m 走廊带内几乎全域不可行(实测中线自身在 10 m 控制点下 Rmin 仅
  53.5 m, 系拟合 GPS 噪声), 初始种群线形长度可达 1084 km, 平面搜索退化为整体平移。

  为何纵断面变坡点也必须取 400 m(不能取 10 m): 两方面。
   (a) 工程上, 每 10 m 一个变坡点等价于全线 2246 处独立变坡, 本就不是有效的纵断面
       设计(论文 §4.5.1 的决策变量是"变坡点"高程, 个数应远小于评价桩号数), 这也是
       相邻坡差(式4.28-4.29)频繁违规的来源;
   (b) 算法上, 57 个平面维若与 2247 个纵断面维放在同一决策向量里, 会被彻底淹没:
       IJS 每次移动同时扰动全部维度并按整体贪婪接受, "改善平面但恶化纵断面"的候选
       会被拒绝, 而纵断面在贴地附近已接近最优、任何随机扰动都使其变差, 于是平面
       无法取得任何进展。实测: dim=2304 时联合优化 L=22.442 km, 相对现状仅缩短
       0.01%(即 δ≈0, 平面完全没动), 且末 100 代 F 仅改善 1.7e-4(结构性停滞,
       加迭代无用)。把两块维度调平(57+57)后, 平面立刻起作用: L=21.852 km(−2.6%),
       约束惩罚从 1.66e8 降到 2.16e4(近乎完全可行)。
  详见 运行记录与问题定位.md §3.3-3.4。

决策向量 x ∈ [0,1]^(N_CTRL + M_PROF) = [0,1]^114:
  x[:N_CTRL]  -> 平面 N_CTRL=57 个控制点法向偏移 δ_k ∈ [-W, W](走廊带半宽 W) -> (x,y)
  x[N_CTRL:]  -> 沿【新平面线位】等分的 M_PROF=57 个变坡点高程 z 调整

公式全部沿用林坤锐学位论文(见 objective.py 式号), 模型公式不变:
  平面里程/坐标 式3.1-3.4; 平曲线最小半径 表3.2(≥400m);
  土方 式4.3; LCC 式3.41-3.55; 能耗 式4.4-4.18; 纵坡/竖曲线约束 式4.27-4.29。

【数据口径的诚实声明】
  数据.xlsx 仅提供实测中线一条轨迹的地面高程, 无面状 DEM(论文图6.6 有 DEM,
  本数据集未提供)。平面横向偏移后, 新点位的地面高程无法从数据直接读出, 本模块
  采用"取【同里程】实测中线点的地面高程"作为近似(按里程比例 np.interp)。
  此近似在走廊带内、地形沿纵向变化为主时成立; 属数据限制下不杜撰的必要处理。

  【为何不用"最近实测点"(cKDTree 最近邻)】
  最近邻写法在平面大幅侧移后会失效: 实测中线自身会折返/绕行, 侧移几百米后
  相邻两个 10 m 桩号可能取到实测路线上相距数百米的两个点的高程, 产生虚假陡坡。
  实测(两阶段最优平面, 侧移中位 569.7 m, 最大 736.2 m):
      最近邻  -> 地面纵坡 |i|max = 104.66% (桩号 12354/12363 m 相距 9.5 m,
                 最近邻索引跳了 154 个实测点≈246 m, 高程 32.72->22.79 m)
      里程对应 -> 地面纵坡 |i|max =   6.66%
  两者在 δ≈0 区(最近邻唯一成立的区域)差异为: C 恰好 0、E ≤ 0.014%,
  故改用里程对应不影响原有结论, 只是把近似写成在全走廊带都成立的形式。
  副产物: 单次目标函数求值 7.65 ms -> 0.80 ms(最近邻查询原占 92% 耗时)。
  详见 运行记录与问题定位.md §3.2。
"""
import numpy as np
from scipy.interpolate import splprep, splev

from params import (CASE, EARTHWORK, TRAFFIC, ENERGY_PRICE, LONG_STD_100, LCC,
                    FLAT_STD_100)
from objective import (earthwork_cost, lcc_ping, fuel_energy, ev_energy,
                       _grades)
from data_loader import load_alignment

CORRIDOR_HALF_W = 800.0     # 走廊带半宽(m), 与 GapB 一致
STEP_CTRL_M = 400.0         # 【决策变量】间距(m): 平面控制点 与 纵断面变坡点 同为 400 m
STEP_EVAL_M = 10.0          # 【目标函数评价】桩号间距(m): 土方/能耗/约束均按 10 m 积分
# 兼容旧名(部分脚本/文档以此指代)
STEP_PLANE_M = STEP_CTRL_M
STEP_PROFILE_M = STEP_CTRL_M


def _n_stations(step_m):
    """沿实测平面线位按 step_m 等分的桩号个数 = 里程/step_m + 1。"""
    a = load_alignment()
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(a["X"]), np.diff(a["Y"])))])
    return int(np.floor(s[-1] / step_m)) + 1

N_CTRL = _n_stations(STEP_CTRL_M)   # 平面控制点个数(每 400 m 一个) -> 决定 (x, y)
M_PROF = _n_stations(STEP_CTRL_M)   # 纵断面变坡点个数(每 400 m 一个) -> 决定 z
M_EVAL = _n_stations(STEP_EVAL_M)   # 目标函数评价桩号个数(每 10 m 一个), 非决策变量


def make_plane_context(align):
    """由实测线位预计算: 平面控制点、单位左法向、实测里程剖面、高程可调幅度。"""
    X, Y = align["X"], align["Y"]
    z = align["ground_z"]
    # 等弧长重采样 N_CTRL 个平面控制点 + 单位左法向
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    si = np.linspace(0, s[-1], N_CTRL)
    cx = np.interp(si, s, X); cy = np.interp(si, s, Y)
    tx = np.gradient(cx); ty = np.gradient(cy)
    tn = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tn, tx / tn
    L0 = float(s[-1])
    amp = max(z.max() - z.min(), 10.0) * 0.6    # 纵断面高程可调幅度(同 objective.decode)
    return dict(cx=cx, cy=cy, nx=nx, ny=ny, gz_meas=z,
                s_meas=s,          # 实测中线累计里程(m), 供按里程对应取地面高程
                X=X, Y=Y, L0=L0, amp=amp)


def build_plane(pc, delta):
    """由控制点法向偏移 delta 生成平滑平面线形(三次样条), 端点固定。返回密集点。"""
    px = pc["cx"] + delta * pc["nx"]
    py = pc["cy"] + delta * pc["ny"]
    px[0], py[0] = pc["cx"][0], pc["cy"][0]
    px[-1], py[-1] = pc["cx"][-1], pc["cy"][-1]
    try:
        tck, _ = splprep([px, py], s=0.0, k=3)
        # 样条密集采样点数按【10 m 评价步长】取(M_EVAL), 而非按控制点数(57):
        # 平面里程 L 与曲率 R 都在这组密集点上算, 采样需比控制点间距(400 m)细得多,
        # 才能准确解析样条在控制点之间的曲率极值。
        n_dense = max(M_EVAL, 1200)
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
    解码联合决策向量为 (新平面, 评价桩号 sta, 新地面高程 gz_new, 设计高程 design_z)。

    决策变量(各 400 m 一个): 平面 N_CTRL 个法向偏移 + 纵断面 M_PROF 个变坡点高程。
    返回的 sta / gz_new / design_z 都是 M_EVAL 个(10 m 一个)评价桩号上的量,
    供土方、能耗、纵坡约束、边坡危险度按 10 m 分辨率积分。
    """
    W = CORRIDOR_HALF_W
    delta = (x[:N_CTRL] - 0.5) * 2.0 * W
    xx, yy = build_plane(pc, delta)
    L_new, R = _plane_metrics(xx, yy)

    # 沿新线位弧长等分 M_EVAL 个【评价桩号】(每 10 m 一个)
    sarc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))])
    sta = np.linspace(0, sarc[-1], M_EVAL)

    # 评价桩号的地面高程 = 【同里程】实测中线点高程(按里程比例线性插值)。
    # 平面侧移后新线位里程 sarc[-1] 与实测里程 s_meas[-1] 不同, 故按比例对应。
    # (不用最近邻: 侧移较大时最近邻会取到实测路线上相距数百米的点, 产生虚假陡坡,
    #  见模块 docstring 的实测数据)
    s_meas = pc["s_meas"]
    gz_new = np.interp(sta / sarc[-1] * s_meas[-1], s_meas, pc["gz_meas"])

    # 纵断面: M_PROF 个变坡点(400 m 一个)的设计高程, 再线性插值到评价桩号。
    # 变坡点高程 = 该处地面高程 ± 可调幅度(同 objective.decode 口径);
    # 插值后变坡点之间为等坡段, 坡度仅在变坡点处改变(真实纵断面设计形态)。
    sta_ctrl = np.linspace(0.0, sarc[-1], M_PROF)
    gz_ctrl = np.interp(sta_ctrl / sarc[-1] * s_meas[-1], s_meas, pc["gz_meas"])
    design_z_ctrl = gz_ctrl + (x[N_CTRL:] - 0.5) * 2.0 * pc["amp"]
    design_z = np.interp(sta, sta_ctrl, design_z_ctrl)

    return dict(xx=xx, yy=yy, L_new=L_new, R=R, sta=sta,
                gz_new=gz_new, design_z=design_z,
                sta_ctrl=sta_ctrl, gz_ctrl=gz_ctrl,
                design_z_ctrl=design_z_ctrl)


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
#  平面步长 400m / 纵断面步长 10m(同联合方案, 见 STEP_PLANE_M / STEP_PROFILE_M),
#  桥隧数据同联合方案(params.py), 保证两种方法在同一离散精度下可比。
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
    from params import COST_UNIT, MAINTENANCE
    width = CASE["road_width_m"]
    area_mu = (L_m * width) / 666.67
    CR = area_mu * COST_UNIT["farmland_per_mu"] + (L_m / 1000.0) * 5000.0   # 式3.42-3.44
    CS = COST_UNIT["subgrade_per_m"] * L_m + COST_UNIT["pavement_per_m"] * (L_m / 1000.0)  # 式3.52-3.54
    # 养护费 CQ (式3.55): 平面阶段无逐桩填挖信息, 仅计与里程相关的
    #   基础项 γ 与车流量项 365·T_j·τ(坡面项属纵断面阶段, 由 objectives_joint 全量计)。
    t = LCC["analysis_years"]; ru = LCC["bank_rate"]
    AADT = TRAFFIC["AADT"]; r_j_pct = TRAFFIC["r_growth"] * 100
    gamma = MAINTENANCE["gamma"]; tau = MAINTENANCE["tau"]
    CQ = 0.0
    for j in range(1, t + 1):
        T_j = AADT * (1 + 0.014 * r_j_pct) ** j          # 第j年预测日交通量
        CQ += (1.0 / (1 + ru) ** j) * L_m * (gamma + 365.0 * T_j * tau)
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
