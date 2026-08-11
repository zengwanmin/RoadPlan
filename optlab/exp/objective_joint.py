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
  · 平面决策变量 150 m 一个(STEP_PLANE_CTRL_M, 约束 R≥400m 下最小可行步长):
      N_CTRL=150 个控制点(法向偏移 δ)
  · 纵断面决策变量 100 m 一个(STEP_PROFILE_CTRL_M, 与消融实验同口径):
      M_PROF=225 个变坡点高程
  · 目标函数一律在 10 m 桩号上积分(STEP_EVAL_M): M_EVAL=2247 个评价桩号,
      土方(式4.3)、能耗(式4.4-4.18)、纵坡/坡差约束、边坡危险度全部按 10 m 计,
      评价精度不受决策维度下降的影响。变坡点高程线性插值到评价桩号, 故变坡点之间
      为等坡段、坡度只在变坡点处改变 —— 与真实纵断面设计一致。

  为何平面取 150 m(不能取 10 m): 平面受平曲线最小半径 R ≥ 400 m(表3.2 极限值)
  约束, 相邻控制点的横向错动 A 在波长 λ=2Δs 上产生的曲率半径为 R = λ²/(4π²A), 即
  Δs 越小、可行的 A 越小: Δs=10 m 时 A ≤ 2.5 cm(实测中线自身在 10 m 控制点下 Rmin
  仅 53.5 m, 系拟合 GPS 噪声); Δs≥150 m 时贴地方案 Rmin≈422 m, 约束才可满足。

  为何纵断面取 100 m(与消融实验同口径): 与消融实验/多算法对比的纵断面搜索空间一致,
  保证跨实验的数值口径可比; 平面维(N_CTRL=150)与纵断面维(M_PROF=225)同数量级,
  不会被淹没。

决策向量 x ∈ [0,1]^(N_CTRL + M_PROF) = [0,1]^375:
  x[:N_CTRL]  -> 平面 N_CTRL=150 个控制点法向偏移 δ_k ∈ [-W, W](走廊带半宽 W) -> (x,y)
  x[N_CTRL:]  -> 沿【新平面线位】等分的 M_PROF=225 个变坡点高程 z 调整

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
                    FLAT_STD_100, BRIDGE_TUNNEL)
from objective import (earthwork_cost, lcc_ping, fuel_energy, ev_energy,
                       entropy_weights, _grades)
from data_loader import load_alignment
import dem

import os as _os
# 走廊带半宽(m): 场景参数, 可用环境变量 CORRIDOR_HALF_W 覆盖(敏感性分析用)。
# 历史: v0=800, v2=1500, v3+=2500。模态幅值和≈1.62·A1, 须小于 DEM 缓冲(4.5 km)。
CORRIDOR_HALF_W = float(_os.environ.get("CORRIDOR_HALF_W", 2500.0))
STEP_PLANE_CTRL_M = 150.0   # 平面决策变量间距(m): 受R≥400m约束, ≥150m贴地才可行
STEP_PROFILE_CTRL_M = 100.0 # 纵断面决策变量间距(m): 变坡点步长, 与消融实验同口径
STEP_EVAL_M = 10.0          # 【目标函数评价】桩号间距(m): 土方/能耗/约束均按 10 m 积分
# 兼容旧名(部分脚本/文档以此指代)
STEP_PLANE_M = STEP_PLANE_CTRL_M
STEP_PROFILE_M = STEP_PROFILE_CTRL_M
STEP_CTRL_M = STEP_PLANE_CTRL_M  # 向后兼容


def _n_stations(step_m):
    """沿实测平面线位按 step_m 等分的桩号个数 = 里程/step_m + 1。"""
    a = load_alignment()
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(a["X"]), np.diff(a["Y"])))])
    return int(np.floor(s[-1] / step_m)) + 1

N_CTRL = _n_stations(STEP_PLANE_CTRL_M)   # 平面控制点个数 -> 决定 (x,y)
M_PROF = _n_stations(STEP_PROFILE_CTRL_M) # 纵断面变坡点个数 -> 决定 z
M_EVAL = _n_stations(STEP_EVAL_M)         # 目标函数评价桩号个数(每 10 m 一个), 非决策变量


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
                lat0=float(align["lat"][0]), lon0=float(align["lon"][0]),
                X=X, Y=Y, L0=L0, amp=amp)


# 平面模态数, 可用环境变量 N_MODE 覆盖(方法矩阵用)。历史: v3=25, v8=40。
N_MODE = int(_os.environ.get("N_MODE", 40))
PLANE_MODE_A1 = CORRIDOR_HALF_W   # 一阶模态幅值上限(m); 第k阶为 A1/k²
START_AMP_M = 20.0   # 纵断面起点高程相对地面的可调幅度(m)
DIM_PLANE = N_MODE
DIM_PROF = M_PROF    # 1 个起点高程 + (M_PROF-1) 个坡段纵坡
DIM = DIM_PLANE + DIM_PROF


def _mode_amps():
    """各阶模态幅值上限: A_k = A1/k²。

    依据: 第k阶 δ_k(s)=a_k·sin(kπs/L) 的曲率量级为 a_k·(kπ/L)², 取 a_k∝1/k² 可使
    各阶对曲率的贡献等量, 从而用固定的模态数与幅值上限把平曲线半径控制在规范之上
    (实测: 25 阶全取上限时叠加曲率对应 R≈2.5 km, 远大于 R_extreme=400 m),
    使 R≥400 m 近似【由构造满足】, 惩罚项仅作安全网。
    """
    k = np.arange(1, N_MODE + 1, dtype=float)
    # v8: 衰减 1/k^2 -> 1/k^1.5。实测最优解 12/40 阶模态顶死幅值上限而 Rmin=454 m
    # 仍有 54 m 余量 —— 绑定的是基函数幅值(自设), 不是半径规范。放宽后曲率安全网
    # 由惩罚项兜底(两段式先软后硬, 历轮全部收敛到可行)。
    return PLANE_MODE_A1 / k ** 1.5


MODE_AMPS = _mode_amps()


def delta_from_modes(coef_norm):
    """由归一化模态系数 [0,1]^N_MODE 生成平面控制点法向偏移 δ(长度 N_CTRL)。"""
    a = (np.asarray(coef_norm, float) - 0.5) * 2.0 * MODE_AMPS
    u = np.linspace(0.0, 1.0, N_CTRL)
    k = np.arange(1, N_MODE + 1, dtype=float)
    # δ(u) = Σ a_k sin(kπu); 正弦基在 u=0,1 处为零 -> 端点自动固定
    return (np.sin(np.outer(u, k) * np.pi) * a).sum(axis=1)


def profile_from_grades(prof_norm, gz_ctrl, ds_ctrl, grade_max):
    """
    由归一化纵断面变量生成变坡点设计高程。

    prof_norm[0]   : 起点高程相对该处地面的偏移(归一化到 ±START_AMP_M)
    prof_norm[1:]  : 各坡段纵坡(归一化到 ±grade_max) —— 纵坡约束由构造满足(式4.27)
    设计高程由起点高程沿各段纵坡累加积分得到, 等价于论文式(4.22) 的变坡点高程 H_i。
    """
    z0 = gz_ctrl[0] + (float(prof_norm[0]) - 0.5) * 2.0 * START_AMP_M
    g = (np.asarray(prof_norm[1:], float) - 0.5) * 2.0 * grade_max
    return np.concatenate([[z0], z0 + np.cumsum(g * ds_ctrl)])


def build_plane(pc, delta):
    """由控制点法向偏移 delta 生成平滑平面线形(三次样条), 端点固定。返回密集点。"""
    px = pc["cx"] + delta * pc["nx"]
    py = pc["cy"] + delta * pc["ny"]
    px[0], py[0] = pc["cx"][0], pc["cy"][0]
    px[-1], py[-1] = pc["cx"][-1], pc["cy"][-1]
    try:
        tck, _ = splprep([px, py], s=0.0, k=3)
        # 样条密集采样点数按【10 m 评价步长】取(M_EVAL): 平面里程 L 与曲率 R 都在这组
        # 密集点上算。当决策步长取 400 m 时该采样比控制点间距细得多; 取 10 m 时二者相当。
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

    决策向量 x ∈ [0,1]^DIM:
      x[:N_MODE]  平面偏移的低频正弦模态系数 -> δ(s) -> 新平面 (x,y)
      x[N_MODE]   纵断面起点高程偏移
      x[N_MODE+1:] 各坡段纵坡(取值域即规范限值, 坡度约束由构造满足)

    返回的 sta / gz_new / design_z 均为 M_EVAL 个(10 m 一个)评价桩号上的量,
    供土方、能耗、竖曲线约束、边坡危险度按 10 m 分辨率积分。
    """
    delta = delta_from_modes(x[:N_MODE])
    xx, yy = build_plane(pc, delta)
    L_new, R = _plane_metrics(xx, yy)

    # 沿新线位弧长等分 M_EVAL 个【评价桩号】(每 10 m 一个)
    sarc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))])
    sta = np.linspace(0, sarc[-1], M_EVAL)

    # 评价桩号处的平面坐标(按弧长插值), 用于 DEM 采样
    x_sta = np.interp(sta, sarc, xx)
    y_sta = np.interp(sta, sarc, yy)

    # 地面高程: 按【实际平面坐标】从走廊带 DEM 双线性采样(见 dem.py)。
    # 取代原先"按同里程比例取实测中线高程"的近似——该近似下平面横移完全不改变
    # 所穿越地形(实测横移 400 m 时地面高程变化为 0.0000 m), 平面只能通过里程影响
    # 目标, 无法为减少土方服务, 从结构上抹掉了平纵联合优化的主要收益来源。
    gz_new = dem.ground_elev_xy(x_sta, y_sta, pc["lat0"], pc["lon0"])

    # 纵断面: 起点高程 + 各段纵坡积分 -> 变坡点设计高程, 再线性插值到评价桩号。
    # 纵坡取值域即规范限值, 故式(4.27) 的坡度约束由构造满足, 无需惩罚。
    # 插值后变坡点之间为等坡段, 坡度仅在变坡点处改变(真实纵断面设计形态)。
    sta_ctrl = np.linspace(0.0, sarc[-1], M_PROF)
    x_ctrl = np.interp(sta_ctrl, sarc, xx)
    y_ctrl = np.interp(sta_ctrl, sarc, yy)
    gz_ctrl = dem.ground_elev_xy(x_ctrl, y_ctrl, pc["lat0"], pc["lon0"])
    ds_ctrl = np.diff(sta_ctrl)
    design_z_ctrl = profile_from_grades(x[N_MODE:], gz_ctrl, ds_ctrl,
                                        LONG_STD_100["grade_max"])
    design_z = np.interp(sta, sta_ctrl, design_z_ctrl)

    # 各评价坡段的平曲线半径: 曲率沿弧长插值到评价桩号, 再取相邻桩号曲率均值,
    # 反推坡段代表半径, 供能耗模型按式(4.12) 计算横向阻力。
    kappa = 1.0 / np.maximum(R, 1e-9)
    kappa_sta = np.interp(sta, sarc, kappa)
    kappa_seg = 0.5 * (kappa_sta[:-1] + kappa_sta[1:])
    R_seg = 1.0 / np.maximum(kappa_seg, 1e-12)

    return dict(xx=xx, yy=yy, L_new=L_new, R=R, R_seg=R_seg, sta=sta,
                gz_new=gz_new, design_z=design_z,
                sta_ctrl=sta_ctrl, gz_ctrl=gz_ctrl,
                design_z_ctrl=design_z_ctrl)


def objectives_joint(x, pc, pen_scale=1.0):
    """联合目标: 返回 (C, E, penalty, info)。C/E 用【新里程、新桩号】计算。

    pen_scale: 惩罚权重倍率, 供分阶段收紧使用(先软后硬)。
    """
    d = decode_joint(x, pc)
    sta, gz_new, design_z, L_new = d["sta"], d["gz_new"], d["design_z"], d["L_new"]
    v = CASE["design_speed_kmh"]

    # 目标一 C = 土方(式4.3, 含结构替代封顶) + 平面全周期(式3.41-3.55), 均用新里程/桩号
    # 结构替代(§3.4.2 判据的成本形式): 逐段取 min(土方费, 结构单价)——
    #   挖方以隧道单价(2.7e8元/km)封顶, 交叉深度 h≈69 m;
    #   填方以桥梁单价(8.8669e6元/km, 数据表手工标注)封顶, 交叉高度 h≈9 m。
    # 全线悬浮(22.4 km 全架桥 ≈1.99 亿)贵于贴地土方(≈1 亿), 无退化解;
    # 交叉高度 9 m 低于论文 30 m 判据, 系桥单价偏低的直接后果, 须在论文中声明。
    C_TU, Vs, Vh, ew = earthwork_cost(
        sta, gz_new, design_z, CASE["road_width_m"],
        bridge_cap_per_m=BRIDGE_TUNNEL["bridge_cost_per_km"] / 1000.0,
        tunnel_cap_per_m=BRIDGE_TUNNEL["tunnel_cost_per_km"] / 1000.0)
    C_PING, cinfo = lcc_ping(L_new, sta, gz_new, design_z)
    C = C_PING + C_TU

    # 目标二 E = 油电混合车流【全生命周期】能耗(式4.4-4.18, 用新桩号/新里程)
    # 与 LCC 同口径: 日能耗 × 365 天 × 30 年等额年金现值系数(5%折现) -> 全周期元,
    # 使 C 与 E 均为全生命周期货币量(亿元), 与论文表6.8 能耗单位"亿元"一致。
    # 能耗方向口径(环境变量 E_DIRECTION):
    #   avg(默认) = 双向平均(v2 修正, 消除"净下坡"套利);
    #   single    = 论文式(4.4) 原口径, 只计单一行驶方向(供与论文同口径对照)。
    Rs = d["R_seg"]
    if _os.environ.get("E_DIRECTION", "avg") == "single":
        ml = fuel_energy(sta, design_z, v, R=Rs)
        kwh = ev_energy(sta, design_z, v, R=Rs)
    else:
        ml = 0.5 * (fuel_energy(sta, design_z, v, R=Rs)
                    + fuel_energy(sta, design_z[::-1], v, R=Rs[::-1]))
        kwh = 0.5 * (ev_energy(sta, design_z, v, R=Rs)
                     + ev_energy(sta, design_z[::-1], v, R=Rs[::-1]))
    AADT = TRAFFIC["AADT"]; n1, n2 = TRAFFIC["n1_fuel"], TRAFFIC["n2_ev"]
    # 单日能耗货币量 × 365 × 30年等额年金现值(5%折现) = 全周期货币量(亿元), 不另乘标定系数
    _t = LCC["analysis_years"]; _ru = LCC["bank_rate"]
    lc_factor = 365.0 * sum(1.0 / (1 + _ru) ** k for k in range(1, _t + 1))
    E_fuel = AADT * n1 * (ml / 1000.0) * ENERGY_PRICE["Kf_fuel"] * lc_factor  # 全周期 元
    E_ele = AADT * n2 * kwh * ENERGY_PRICE["ZQ_elec"] * lc_factor            # 全周期 元
    E = E_fuel + E_ele                                                       # 全周期 元

    # ---- 联合约束惩罚 ----
    # 尺度: 用【相对违反量的 均值+最大值】×权重, 使惩罚与标量目标 F(约 0.19)可比。
    # 只取均值会严重低估局部违反(2247 个评价点里少数越限时均值被摊薄, 实测 Rmin=1 m
    # 的个体惩罚仅 0.2), 故叠加最大值项, 使任何单点的严重越限都产生量级为 1 的惩罚。
    # 原实现用绝对系数(5e7/1e9/5e8)再除以 C_ref, 实测 pen/C_ref≈3.2 而 F≈0.19,
    # 搜索几乎只看得见惩罚、看不见成本与能耗。
    # 纵坡约束(式4.27)已由纵坡编码构造满足, 不再设惩罚项。
    Rmin_req = FLAT_STD_100["R_extreme_m"]
    R = d["R"]
    rel_R = np.maximum(0.0, (Rmin_req - R) / Rmin_req)             # 平曲线半径(表3.2)
    grades = _grades(sta, design_z)
    dg_lim = 0.03
    rel_V = np.maximum(0.0, (np.abs(np.diff(grades)) - dg_lim) / dg_lim)
    pen = pen_scale * 10.0 * (rel_R.mean() + rel_R.max()
                              + rel_V.mean() + rel_V.max())        # 竖曲线(式4.28-4.29)

    info = dict(C_PING=C_PING, C_TU=C_TU, E_fuel=E_fuel, E_ele=E_ele,
                Vs=Vs, Vh=Vh, ml=ml, kwh=kwh, L_km=L_new / 1000.0,
                Rmin=float(R.min()),
                L_bridge_new=ew["L_bridge_new_km"],
                L_tunnel_new=ew["L_tunnel_new_km"], **cinfo)
    return C, E, pen, info


def make_scalar_joint(pc, wC, wE, C_ref, E_ref, pen_scale=1.0):
    """标量化联合目标 F = wC·Cnorm + wE·Enorm + 惩罚(已与 F 同尺度)。"""
    def f(x):
        C, E, pen, _ = objectives_joint(x, pc, pen_scale=pen_scale)
        return wC * (C / C_ref) + wE * (E / E_ref) + pen
    return f


def run_ijs_two_phase(make_f, lb, ub, pop0, max_iter, seed,
                      soft=0.3, hard=3.0):
    """
    分阶段收紧惩罚的 IJS 寻优(先软后硬)。

    前半迭代用较小惩罚倍率, 允许解穿越不可行区搜索; 后半迭代放大倍率, 把解拉回
    可行域。第二阶段从第一阶段返回的末代种群继续(本目录 algorithms.run 会返回 pop)。

    make_f: pen_scale -> 目标函数 的构造器。
    """
    from algorithms import run, VARIANTS
    it1 = max_iter // 2
    it2 = max_iter - it1
    r1 = run(make_f(soft), lb, ub, pop0, it1, seed, **VARIANTS["V5_IJS"])
    r2 = run(make_f(hard), lb, ub, r1["pop"], it2, seed + 1, **VARIANTS["V5_IJS"])
    return dict(best_x=r2["best_x"], best_f=r2["best_f"],
                curve=np.concatenate([r1["curve"], r2["curve"]]),
                nfe=r1["nfe"] + r2["nfe"])


def joint_baseline(pc, pop_size, seed=2025, x_seed=None):
    """
    联合基准种群 + 【联合方案与两阶段对照共用】的熵权与参考尺度。

    返回 (base, wC, wE, C_ref, E_ref)。

    为何必须共用: 熵权由种群客观计算, 若两种方法各自用自己的初始种群算权重, 得到的
    是不同的标量目标(实测联合 wC≈0.35 而两阶段 wC≈0.26), 此时 C 与 E 的单项数值不可
    直接比较——同一组解在不同权重下可给出相反的优劣结论。共用同一组 (wC,wE,C_ref,E_ref)
    后, 两种方法最小化同一个 F, 表中的 C/E 才具备可比性。

    种群在 [0,1]^DIM 内均匀取值即可: 模态系数与纵坡的取值域本身就是规范限值,
    故随机个体天然满足坡度约束、且平曲线半径远大于限值(见 _mode_amps)。

    x_seed: 若给出(通常为现状方案 M-A), 则置入种群首位, 保证寻优结果不劣于现状方案。
    两种方法均由本函数取种群, 故该注入对联合与两阶段是同等的。
    """
    rng = np.random.default_rng(seed)
    base = rng.random((pop_size, DIM))
    if x_seed is not None:
        base[0] = np.clip(np.asarray(x_seed, float), 0.0, 1.0)
    C0 = np.array([objectives_joint(base[i], pc)[0] for i in range(pop_size)])
    E0 = np.array([objectives_joint(base[i], pc)[1] for i in range(pop_size)])
    wC, wE = entropy_weights(C0, E0)
    return base, float(wC), float(wE), float(C0.mean()), float(E0.mean())


# =============================================================
#  两阶段(先平面后纵断面)对照所需的平面阶段 helper
#  仅供 run_twostage.py 复现"先平面优化、再纵断面优化"的分阶段方法作为对比;
#  平面步长 10m / 纵断面步长 10m(同联合方案, 见 STEP_PLANE_M / STEP_PROFILE_M),
#  桥隧数据同联合方案(params.py), 保证两种方法在同一离散精度下可比。
# =============================================================
def build_plane_from_delta(pc, coef_norm):
    """由平面低频模态系数 coef_norm∈[0,1]^N_MODE 生成新平面。返回密集点+里程+R。"""
    xx, yy = build_plane(pc, delta_from_modes(coef_norm))
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
    CS = COST_UNIT["subgrade_per_m"] * L_m + COST_UNIT["pavement_per_m"] * L_m  # 式3.53-3.54
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


def make_scalar_plane(pc, C_ref_plane, pen_scale=1.0):
    """第一阶段平面标量目标: min 平面LCC + 平曲线半径惩罚(R>=400m, 表3.2)。"""
    Rmin_req = FLAT_STD_100["R_extreme_m"]

    def f(coef_norm):
        _, _, L_new, R = build_plane_from_delta(pc, coef_norm)
        vio_R = np.maximum(0.0, (Rmin_req - R) / Rmin_req).mean()
        return plane_lcc(L_new) / C_ref_plane + pen_scale * 10.0 * vio_R
    return f
