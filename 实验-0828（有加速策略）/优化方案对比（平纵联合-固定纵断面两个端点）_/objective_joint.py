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

【地面高程口径】
  地面高程一律按新线位实际平面坐标从走廊带 DEM 的【准天然地面】双线性采样
  (dem.py: AWS Terrain Tiles z14 + 道路影响带掩膜插值重建, 移除既有路基)。
  所有方案(含现状 M-A)统一以准天然地面计土方, 口径公平; 平面横移会真实改变
  所穿越地形, 平纵联合优化的收益通道(绕避高地形省土方)成立。
  白云山生态强制隧道区(dem.eco_mask): 穿越长度×隧道单价计入 CB, 区内土方豁免;
  7 座立交按弦投影带锚定: 土方豁免, 结构费以常数计入 CB(params.BRIDGE_TUNNEL)。
"""
import os

import numpy as np
from scipy.interpolate import splprep, splev

from params import (CASE, EARTHWORK, TRAFFIC, ENERGY_PRICE, LONG_STD_100, LCC,
                    FLAT_STD_100, BRIDGE_TUNNEL, PENALTY)
from objective import (earthwork_cost, lcc_ping, fuel_energy, ev_energy,
                       entropy_weights, _grades)
from data_loader import load_alignment
import dem
from acceleration import ScalarObjective, evaluate_many_ordered

CORRIDOR_HALF_W = 500.0     # 走廊带半宽(m): 主实验口径 ±500m(±250m 作敏感性情景)
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


OSM_NPZ = None  # 延迟定位: 数据/OSM走廊带障碍物/obstacles.npz


def _ic_bands_from_osm(X, Y, s, t_meas):
    """
    7座立交带的弦投影区间, 由 OSM 被交道路锚定(带磁盘缓存)。

    对 params.BRIDGE_TUNNEL["interchanges"] 中每座立交:
      在【镜像估计里程 ±2.5 km】窗口内(数据.xlsx 里程零点在东端, 与统计表桩号
      方向相反), 找名称含关键字的 OSM 道路折线与实测中线的最近交叉点里程
      s_cross(要求最近距离<150 m), 豁免带 = s_cross ± 统计长度/2 对应的弦投影区间。
    缓存: 数据/OSM走廊带障碍物/ic_anchor_cache.json (主进程先算, worker 直接读)。
    """
    import json
    from params import BRIDGE_TUNNEL
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "数据", "OSM走廊带障碍物")
    cache_fn = os.path.join(base, "ic_anchor_cache.json")
    if os.path.exists(cache_fn):
        with open(cache_fn, encoding="utf-8") as f:
            anchors = json.load(f)
    else:
        from scipy.spatial import cKDTree
        d = np.load(os.path.join(base, "obstacles.npz"), allow_pickle=True)
        lon_f, lat_f, offs = d["lines_lon"], d["lines_lat"], d["offsets"]
        names = [str(x) for x in d["name"]]
        R_E = 6378137.0
        # 与 data_loader 相同的局部平面(以中线首点为原点)
        lat0 = None
        # 中线首点经纬度由 X/Y=0 处反推不可行, 直接重读对齐数据
        from data_loader import load_alignment
        a = load_alignment()
        lat0 = np.radians(a["lat"][0]); lon0 = np.radians(a["lon"][0])
        tree = cKDTree(np.column_stack([X, Y]))
        L_total = float(s[-1])
        anchors = {}
        for name, kw, L_km, K_mid in BRIDGE_TUNNEL["interchanges"]:
            s_est = L_total - float(K_mid)      # 镜像估计
            best = (1e18, None)
            for i in range(len(offs) - 1):
                if kw not in names[i]:
                    continue
                lx = R_E * np.cos(lat0) * (np.radians(lon_f[offs[i]:offs[i+1]]) - lon0)
                ly = R_E * (np.radians(lat_f[offs[i]:offs[i+1]]) - lat0)
                for j in range(len(lx) - 1):
                    seg = float(np.hypot(lx[j+1]-lx[j], ly[j+1]-ly[j]))
                    n = max(2, int(seg / 20) + 1)
                    px = np.linspace(lx[j], lx[j+1], n)
                    py = np.linspace(ly[j], ly[j+1], n)
                    dd, ii = tree.query(np.column_stack([px, py]))
                    for k in range(len(dd)):
                        if abs(s[ii[k]] - s_est) <= 2500.0 and dd[k] < best[0]:
                            best = (float(dd[k]), float(s[ii[k]]))
            dist, s_cross = best
            if s_cross is None or dist > 150.0:
                raise RuntimeError(
                    f"立交[{name}]的被交道路[{kw}]OSM 锚定失败: 最近距离 {dist:.0f} m")
            anchors[name] = dict(kw=kw, L_km=float(L_km),
                                 s_cross=s_cross, dist_m=dist)
        with open(cache_fn, "w", encoding="utf-8") as f:
            json.dump(anchors, f, ensure_ascii=False, indent=2)
    bands = []
    for name, rec in anchors.items():
        half = rec["L_km"] * 1000.0 / 2.0
        ta = float(np.interp(rec["s_cross"] - half, s, t_meas))
        tb = float(np.interp(rec["s_cross"] + half, s, t_meas))
        bands.append((min(ta, tb), max(ta, tb)))
    return bands


def make_plane_context(align):
    """由实测线位预计算: 平面控制点、单位左法向、实测里程剖面、高程可调幅度、
    起终点弦与立交带弦投影区间(空间锚定)。"""
    # DEM/OSM 在算法计时开始前一次加载，后续所有算法共享相同只读数据。
    dem.preload_readonly()
    X, Y = align["X"], align["Y"]
    z = align["ground_z"].copy()
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    # 【隧道段 GPS 高程修复】白云山隧道内无卫星信号, 数据.xlsx 的 Altitude 为
    # 设备外插漂移值(实测该段"路面"高出准天然地面 7-17m, 物理不可能)。
    # 修复: 生态强制隧道区内的连续段, 用洞口两端实测高程线性内插替换。
    eco_line = dem.eco_mask_xy(X, Y, float(align["lat"][0]),
                               float(align["lon"][0]))
    i = 0
    n = len(z)
    while i < n:
        if eco_line[i]:
            j = i
            while j + 1 < n and eco_line[j + 1]:
                j += 1
            i0, j1 = max(i - 1, 0), min(j + 1, n - 1)
            z[i:j + 1] = np.interp(s[i:j + 1], [s[i0], s[j1]], [z[i0], z[j1]])
            i = j + 1
        else:
            i += 1
    si = np.linspace(0, s[-1], N_CTRL)
    cx = np.interp(si, s, X); cy = np.interp(si, s, Y)
    tx = np.gradient(cx); ty = np.gradient(cy)
    tn = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tn, tx / tn
    L0 = float(s[-1])
    amp = max(z.max() - z.min(), 10.0) * 0.6    # 纵断面高程可调幅度(同 objective.decode)
    # 起终点弦(空间锚定坐标轴): 立交带 -> 弦投影区间。
    # 带中心由 OSM 被交道路与实测中线的交叉点锚定(_ic_bands_from_osm),
    # 修复了"统计表桩号直接当数据里程"的方向镜像错误。
    ux, uy = X[-1] - X[0], Y[-1] - Y[0]
    un = float(np.hypot(ux, uy)); ux, uy = ux / un, uy / un
    t_meas = (X - X[0]) * ux + (Y - Y[0]) * uy       # 实测点弦投影
    bands = _ic_bands_from_osm(X, Y, s, t_meas)
    # ---- 交叉桥内生口径(问题21)的现状线校准: 匝道功能费常数 + 基线诊断 ----
    import crossings as _cr
    lat0 = float(align["lat"][0]); lon0 = float(align["lon"][0])
    _cr.preload_readonly(lat0, lon0)
    cr0 = _cr.detect_crossings(X, Y, lat0, lon0)
    eco_at = np.interp(cr0["s"], s, eco_line.astype(float)) > 0.5
    iv0, L0m = _cr.bridge_intervals(cr0, keep=~eco_at)
    ramp_cost, L_ic_main_km = _cr.ramp_cost_from_baseline(iv0)
    return dict(cx=cx, cy=cy, nx=nx, ny=ny, gz_meas=z,
                s_meas=s,          # 实测中线累计里程(m), 供按里程对应取地面高程
                lat0=lat0, lon0=lon0,
                X=X, Y=Y, L0=L0, amp=amp,
                chord=(float(X[0]), float(Y[0]), ux, uy),
                ic_bands=bands,
                ramp_cost=float(ramp_cost),
                baseline_cross=dict(intervals=iv0,
                                    L_cross_km=L0m / 1000.0,
                                    L_ic_main_km=float(L_ic_main_km),
                                    n_cross=int(len(cr0["s"]))))


N_MODE = 50          # 平面偏移的低频正弦模态个数(决策变量数)
PLANE_MODE_A1 = CORRIDOR_HALF_W   # 一阶模态幅值上限(m); 第k阶为 A1/k²
START_AMP_M = 20.0   # 仅供现状解的兼容编码使用；固定端点解码时不调整起点高程
DIM_PLANE = N_MODE
DIM_PROF = M_PROF    # 1 个兼容保留位 + (M_PROF-1) 个坡段纵坡形状变量
DIM = DIM_PLANE + DIM_PROF


def _mode_amps():
    """各阶模态幅值上限: A_k = A1/k²。

    依据: 第k阶 δ_k(s)=a_k·sin(kπs/L) 的曲率量级为 a_k·(kπ/L)², 取 a_k∝1/k² 可使
    各阶对曲率的贡献等量(每阶 A1π²/L²), 最坏总曲率 ∝ 模态数 N。
    N=50、A1=500m(走廊带半宽) 时全模态取上限的最坏曲率对应 R≈2000 m, 远大于
    规范 R_extreme=400 m(表3.2), 故 R≥400 m 近似【由构造满足】, 惩罚项仅作安全网。
    """
    k = np.arange(1, N_MODE + 1, dtype=float)
    return PLANE_MODE_A1 / k ** 2


MODE_AMPS = _mode_amps()
MODE_AMPS.setflags(write=False)
_MODE_U = np.linspace(0.0, 1.0, N_CTRL)
_MODE_K = np.arange(1, N_MODE + 1, dtype=np.float64)
_MODE_BASIS = np.sin(np.outer(_MODE_U, _MODE_K) * np.pi)
_MODE_BASIS.setflags(write=False)


def set_corridor(half_w_m):
    """
    运行时切换走廊带半宽(供敏感性分析逐任务调用), 同步更新模态幅值上限。

    构造最坏平曲线半径 R ≈ 2000m × (500/half_w): ±200m→R≈5000m, ±1000m→R≈1000m,
    ±2500m→R≈400m(恰在 R_extreme 限值, 样条过冲可能轻微越限, 由惩罚项兜底)。
    多进程下每个 worker 在任务开始时调用一次即可(进程内全局)。
    """
    global CORRIDOR_HALF_W, PLANE_MODE_A1, MODE_AMPS
    CORRIDOR_HALF_W = float(half_w_m)
    PLANE_MODE_A1 = CORRIDOR_HALF_W
    MODE_AMPS = _mode_amps()
    MODE_AMPS.setflags(write=False)


PROFILE_ENDPOINTS_FIXED = True   # 联合/两阶段统一口径: 纵断面首末高程锚定到既有道路接线高程


def set_profile_step(step_m):
    """
    运行时切换纵断面变坡点步长(供多算法对比 PJ1-PJ6 联合规模阶梯调用,
    待办清单2 问题19)。同步更新 M_PROF / DIM; 平面模态数 N_MODE 固定不变
    (规模变量只取纵断面步长, 避免双变量混淆)。
    多进程下每个 worker 在任务开始时调用一次即可(进程内全局)。
    """
    global STEP_PROFILE_CTRL_M, STEP_PROFILE_M, M_PROF, DIM_PROF, DIM
    STEP_PROFILE_CTRL_M = float(step_m)
    STEP_PROFILE_M = STEP_PROFILE_CTRL_M
    M_PROF = _n_stations(STEP_PROFILE_CTRL_M)
    DIM_PROF = M_PROF
    DIM = DIM_PLANE + DIM_PROF


def delta_from_modes(coef_norm):
    """由归一化模态系数 [0,1]^N_MODE 生成平面控制点法向偏移 δ(长度 N_CTRL)。"""
    a = (np.asarray(coef_norm, float) - 0.5) * 2.0 * MODE_AMPS
    # δ(u) = Σ a_k sin(kπu); 正弦基在 u=0,1 处为零 -> 端点自动固定
    return (_MODE_BASIS * a).sum(axis=1)


def profile_from_grades(prof_norm, gz_ctrl, ds_ctrl, grade_max, z_tie):
    """
    由归一化纵断面变量生成变坡点设计高程。

    prof_norm[0]   : 为了与原始种群维度及其他实验编码兼容而保留；
                     固定端点口径下不用于改变起点高程。
    prof_norm[1:]  : 各坡段的原始纵坡形状变量。
    z_tie=(z0, zL): 既有道路首、末接线高程。

    先扣除原始纵坡的里程加权均值，再叠加为满足两端高程所需的
    平均纵坡 g_req，从而使 sum(g*ds) = zL-z0 恒成立。同时缩放纵坡波动，
    使每一段仍由构造满足 |g| <= grade_max。
    """
    ds = np.asarray(ds_ctrl, float)
    z0, zL = float(z_tie[0]), float(z_tie[1])
    total_length = float(ds.sum())
    if total_length <= 0.0:
        raise ValueError("纵断面控制点总里程必须大于 0")

    g_req = (zL - z0) / total_length
    if abs(g_req) > grade_max + 1e-12:
        raise ValueError(
            f"固定端点所需平均纵坡 {g_req:.6f} 超过规范限值 "
            f"{grade_max:.6f}")

    raw = (np.asarray(prof_norm[1:], float) - 0.5) * 2.0
    if raw.shape != ds.shape:
        raise ValueError(
            f"纵断面坡段变量数 {raw.size} 与坡段数 {ds.size} 不一致")
    raw_mean = float(np.dot(raw, ds) / total_length)
    fluctuation = raw - raw_mean
    max_fluctuation = float(np.max(np.abs(fluctuation))) if fluctuation.size else 0.0
    grade_room = max(0.0, grade_max - abs(g_req))
    scale = (1.0 if max_fluctuation * grade_max <= grade_room else
             grade_room / (max_fluctuation * grade_max))
    g = g_req + scale * fluctuation * grade_max

    z = np.concatenate([[z0], z0 + np.cumsum(g * ds)])
    z[-1] = zL  # 消除累加浮点误差，保证末端点精确固定
    return z


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
      x[N_MODE]   纵断面兼容保留位(不改变已固定的起点高程)
      x[N_MODE+1:] 各坡段纵坡形状变量(经端点锚定变换后仍由构造满足坡度约束)

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

    # 地面高程: 按【实际平面坐标】从走廊带 DEM 采样【准天然地面】(见 dem.py:
    # 移除既有路基后的 road-removal 重建地面)。所有方案(含现状 M-A)统一以准天然
    # 地面计土方, 恢复"现状方案历史土方成本"的口径公平(修复"沉没成本计零"问题)。
    gz_new = dem.ground_elev_xy(x_sta, y_sta, pc["lat0"], pc["lon0"], natural=True)

    # 生态强制隧道区掩膜(白云山, dem.eco_mask): 穿越段按隧道计费、土方豁免
    eco = dem.eco_mask_xy(x_sta, y_sta, pc["lat0"], pc["lon0"])
    # 立交带掩膜(空间锚定): 仅作诊断输出(豁免与计费已由交叉触发桥区间接管)
    x0c, y0c, ux, uy = pc["chord"]
    t_sta = (x_sta - x0c) * ux + (y_sta - y0c) * uy
    ic = np.zeros(len(sta), dtype=bool)
    for ta, tb in pc["ic_bands"]:
        ic |= (t_sta >= ta) & (t_sta <= tb)
    # ---- 交叉触发跨越桥(问题21): OSM 障碍物交叉 -> 桥区间(几何内生) ----
    import crossings as _cr
    cross = _cr.detect_crossings(xx, yy, pc["lat0"], pc["lon0"])
    # 生态隧道区内的交叉不设桥(隧道下穿, 优先级更高)
    keep = np.interp(cross["s"], sta, eco.astype(float)) <= 0.5 \
        if len(cross["s"]) else np.zeros(0, dtype=bool)
    bridge_iv, L_cross_m = _cr.bridge_intervals(cross, keep=keep)
    bridge = _cr.mask_from_intervals(sta, bridge_iv)
    exempt = eco | bridge
    # 生态隧道穿越长度(km): 按段中点判定
    eco_seg = 0.5 * (eco[:-1].astype(float) + eco[1:].astype(float))
    dL_sta = np.diff(sta)
    L_eco_km = float(np.sum(eco_seg * dL_sta)) / 1000.0

    # 纵断面: 固定起点高程 + 经终点约束校正的各段纵坡积分
    # -> 变坡点设计高程, 再线性插值到评价桩号。
    # 纵坡取值域即规范限值, 故式(4.27) 的坡度约束由构造满足, 无需惩罚。
    # 插值后变坡点之间为等坡段, 坡度仅在变坡点处改变(真实纵断面设计形态)。
    sta_ctrl = np.linspace(0.0, sarc[-1], M_PROF)
    x_ctrl = np.interp(sta_ctrl, sarc, xx)
    y_ctrl = np.interp(sta_ctrl, sarc, yy)
    gz_ctrl = dem.ground_elev_xy(x_ctrl, y_ctrl, pc["lat0"], pc["lon0"],
                                 natural=True)
    ds_ctrl = np.diff(sta_ctrl)
    # 联合与两阶段方案共用本解码器：两者的纵断面首、末设计高程
    # 都精确锚定到既有道路接线高程。平面端点也已由 build_plane 固定，
    # 因此所有候选方案的两个接线点是同一对物理位置。
    z_tie = (float(pc["gz_meas"][0]), float(pc["gz_meas"][-1]))
    design_z_ctrl = profile_from_grades(x[N_MODE:], gz_ctrl, ds_ctrl,
                                        LONG_STD_100["grade_max"], z_tie=z_tie)
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
                design_z_ctrl=design_z_ctrl,
                eco=eco, ic=ic, exempt=exempt, L_eco_km=L_eco_km,
                cross=cross, cross_keep=keep, bridge_iv=bridge_iv,
                bridge=bridge, L_cross_km=L_cross_m / 1000.0)


def _joint_penalty(d, pen_scale):
    """只依赖已解码几何的非负罚项，是完整标量目标的安全下界。"""
    Rmin_req = FLAT_STD_100["R_extreme_m"]
    rel_R = np.maximum(0.0, (Rmin_req - d["R"]) / Rmin_req)
    grades = _grades(d["sta"], d["design_z"])
    dg_lim = 3e-4 * STEP_PROFILE_CTRL_M
    rel_V = np.maximum(0.0, (np.abs(np.diff(grades)) - dg_lim) / dg_lim)
    pen = pen_scale * (
        PENALTY["k_R"] * (rel_R.mean() + rel_R.max())
        + PENALTY["k_V"] * (rel_V.mean() + rel_V.max()))
    cr = d["cross"]; kp = d["cross_keep"]
    if len(cr["s"]) and np.any(kp):
        th_min = np.radians(
            BRIDGE_TUNNEL["crossing_trigger"]["skew_min_deg"])
        rel_S = np.maximum(0.0, (th_min - cr["theta"][kp]) / th_min)
        pen += pen_scale * PENALTY["k_skew"] * (rel_S.mean() + rel_S.max())
    return float(pen)


def objectives_joint(x, pc, pen_scale=1.0, scenario=None,
                     reject_penalty=None):
    """联合目标: 返回 (C, E, penalty, info)。C/E 用【新里程、新桩号】计算。

    pen_scale: 惩罚权重倍率, 供分阶段收紧使用(先软后硬)。
    scenario : 敏感性分析情景参数(None=基准, 等价原行为), dict 可含:
      ev                 电动车渗透率 n2 (基准 TRAFFIC.n2_ev)
      traffic_growth     交通量年增长率 rj (基准 0, 即 E 用恒定 AADT;
                         增长同时叠加养护费车流量项增量 dCQ, 式3.55)
      fuel_price_growth  油价年增长率 (逐年 (1+g)^j 进入折现求和)
      elec_price_growth  电价年增长率
      fuel_save          燃油车节油率 (乘 (1-s) 于单车油耗)
      elec_save          电动车节能率
    """
    P = scenario or {}
    ev_share = float(P.get("ev", TRAFFIC["n2_ev"]))
    rj = float(P.get("traffic_growth", 0.0))
    fpg = float(P.get("fuel_price_growth", 0.0))
    epg = float(P.get("elec_price_growth", 0.0))
    fsave = float(P.get("fuel_save", 0.0))
    esave = float(P.get("elec_save", 0.0))

    d = decode_joint(x, pc)
    pen = _joint_penalty(d, pen_scale)
    if reject_penalty is not None and pen >= float(reject_penalty):
        return np.nan, np.nan, pen, {"early_reject": True}
    sta, gz_new, design_z, L_new = d["sta"], d["gz_new"], d["design_z"], d["L_new"]
    v = CASE["design_speed_kmh"]

    # 目标一 C = 土方(式4.3, 含结构替代封顶+豁免) + 平面全周期(式3.41-3.55)
    # 口径(见 params.BRIDGE_TUNNEL): 立交带与生态隧道区土方豁免(结构费在 CB),
    #   其余区段结构替代封顶 min(土方费, 结构单价)防退化解; 地面为准天然地面。
    C_TU, Vs, Vh, ew = earthwork_cost(
        sta, gz_new, design_z, CASE["road_width_m"],
        bridge_cap_per_m=BRIDGE_TUNNEL["bridge_cost_per_km"] / 1000.0,
        tunnel_cap_per_m=BRIDGE_TUNNEL["tunnel_cost_per_km"] / 1000.0,
        exempt=d["exempt"])
    C_PING, cinfo = lcc_ping(L_new, sta, gz_new, design_z,
                             L_eco_tunnel_km=d["L_eco_km"], exempt=d["exempt"],
                             L_bridge_km=d["L_cross_km"],
                             ramp_cost=pc["ramp_cost"])
    C = C_PING + C_TU
    _t = LCC["analysis_years"]; _ru = LCC["bank_rate"]
    if rj > 0.0:
        # 养护费车流量项增量(式3.55): 采样点增长率 rj 相对基准增长率的偏差
        from params import MAINTENANCE as _MT
        r_base_pct = TRAFFIC["r_growth"] * 100
        dCQ = 0.0
        for j in range(1, _t + 1):
            T_sens = TRAFFIC["AADT"] * (1 + rj) ** j
            T_base = TRAFFIC["AADT"] * (1 + 0.014 * r_base_pct) ** j
            dCQ += (1.0 / (1 + _ru) ** j) * L_new * 365.0 \
                * (T_sens - T_base) * _MT["tau"]
        C += dCQ

    # 目标二 E = 油电混合车流【全生命周期】能耗(式4.4-4.18, 用新桩号/新里程)
    # 与 LCC 同口径: 逐年 [交通量增长 × 能源价格增长] × 5%折现求和 × 365 天,
    # 基准情景(增长率均为0)退化为原"日能耗×365×等额年金现值", 全周期货币量(亿元)。
    ml = fuel_energy(sta, design_z, v, R=d["R_seg"]) * (1.0 - fsave)
    kwh = ev_energy(sta, design_z, v, R=d["R_seg"]) * (1.0 - esave)
    AADT = TRAFFIC["AADT"]
    n1, n2 = 1.0 - ev_share, ev_share
    E_fuel = E_ele = 0.0
    for j in range(1, _t + 1):
        disc = (1 + rj) ** j / (1 + _ru) ** j
        E_fuel += AADT * n1 * (ml / 1000.0) * ENERGY_PRICE["Kf_fuel"] \
            * (1 + fpg) ** j * disc * 365.0
        E_ele += AADT * n2 * kwh * ENERGY_PRICE["ZQ_elec"] \
            * (1 + epg) ** j * disc * 365.0
    E = E_fuel + E_ele                                                       # 全周期 元

    R = d["R"]
    cr = d["cross"]
    ic_seg = 0.5 * (d["ic"][:-1].astype(float) + d["ic"][1:].astype(float))
    L_ic_km = float(np.sum(ic_seg * np.diff(sta))) / 1000.0
    L_km_new = L_new / 1000.0
    info = dict(C_PING=C_PING, C_TU=C_TU, E_fuel=E_fuel, E_ele=E_ele,
                Vs=Vs, Vh=Vh, ml=ml, kwh=kwh, L_km=L_km_new,
                Rmin=float(R.min()),
                L_eco_km=d["L_eco_km"], L_ic_km=L_ic_km,
                L_cross_km=d["L_cross_km"],
                n_cross=int(np.sum(d["cross_keep"])) if len(cr["s"]) else 0,
                n_bridge_iv=len(d["bridge_iv"]),
                ramp_cost=float(pc["ramp_cost"]),
                L_bridge_new=ew["L_bridge_new_km"],
                L_tunnel_new=ew["L_tunnel_new_km"],
                **cinfo)
    return C, E, pen, info


def make_scalar_joint(pc, wC, wE, C_ref, E_ref, pen_scale=1.0, scenario=None):
    """标量化联合目标 F = wC·Cnorm + wE·Enorm + 惩罚(已与 F 同尺度)。"""
    def f(x):
        C, E, pen, _ = objectives_joint(x, pc, pen_scale=pen_scale,
                                        scenario=scenario)
        return wC * (C / C_ref) + wE * (E / E_ref) + pen

    def bounded(x, limit):
        C, E, pen, info = objectives_joint(
            x, pc, pen_scale=pen_scale, scenario=scenario,
            reject_penalty=limit)
        if info.get("early_reject", False):
            return limit
        return wC * (C / C_ref) + wE * (E / E_ref) + pen

    return ScalarObjective(f, bounded)


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
    def ce(x):
        return np.asarray(objectives_joint(x, pc)[:2], dtype=np.float64)

    CE0 = evaluate_many_ordered(ce, base)
    C0, E0 = CE0[:, 0], CE0[:, 1]
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


def plane_lcc(L_m, L_eco_km=0.0, L_cross_km=None, ramp_cost=0.0):
    """
    第一阶段平面目标: 平面相关全生命周期成本(随平面走向变化的项), 论文 §3.4.2/式3.41:
      工程占地 CR(式3.42-3.44) + 桥隧 CB(式3.45-3.51: 交叉触发桥内生 + 匝道常数
      + 生态隧道×穿越长度) + 基本建设 CS(式3.52-3.54) + 养护 CQ 的基础/交通量项(式3.55)。
    CR/CS/CQ 正比于里程 L; CB 的桥项由平面交叉几何内生(问题21)、隧道项由平面是否
    穿越白云山生态区内生决定——平面阶段即可体现"跨越/绕行"与"隧道/绕行"的权衡。
    土方 C_TU 属纵断面阶段, 此处不计(最终 C 仍由 objectives_joint 全量计)。
    """
    from params import COST_UNIT, MAINTENANCE
    width = CASE["road_width_m"]
    area_mu = (L_m * width) / 666.67
    CR = area_mu * COST_UNIT["farmland_per_mu"] + (L_m / 1000.0) * 5000.0   # 式3.42-3.44
    CS = COST_UNIT["subgrade_per_m"] * L_m + COST_UNIT["pavement_per_m"] * L_m  # 式3.53-3.54
    if L_cross_km is None:                # 旧口径兜底(常数计费)
        CB = BRIDGE_TUNNEL["interchange_total_km"] * BRIDGE_TUNNEL["bridge_cost_per_km"] \
            + float(L_eco_km) * BRIDGE_TUNNEL["tunnel_cost_per_km"]
    else:
        CB = float(L_cross_km) * BRIDGE_TUNNEL["bridge_cost_per_km"] \
            + float(ramp_cost) \
            + float(L_eco_km) * BRIDGE_TUNNEL["tunnel_cost_per_km"]         # 式3.45-3.51
    # 养护费 CQ (式3.55): 平面阶段无逐桩填挖信息, 仅计与里程相关的
    #   基础项 γ 与车流量项 365·T_j·τ(坡面项属纵断面阶段, 由 objectives_joint 全量计)。
    t = LCC["analysis_years"]; ru = LCC["bank_rate"]
    AADT = TRAFFIC["AADT"]; r_j_pct = TRAFFIC["r_growth"] * 100
    gamma = MAINTENANCE["gamma"]; tau = MAINTENANCE["tau"]
    CQ = 0.0
    for j in range(1, t + 1):
        T_j = AADT * (1 + 0.014 * r_j_pct) ** j          # 第j年预测日交通量
        CQ += (1.0 / (1 + ru) ** j) * L_m * (gamma + 365.0 * T_j * tau)
    return CR + CB + CS + CQ


def make_scalar_plane(pc, C_ref_plane, pen_scale=1.0):
    """第一阶段平面标量目标: min 平面LCC(交叉桥+生态隧道内生)
    + 平曲线半径惩罚(R>=400m)。建筑密度不进入目标或约束。
    """
    import crossings as _cr
    Rmin_req = FLAT_STD_100["R_extreme_m"]

    def value(coef_norm, limit=None):
        xx, yy, L_new, R = build_plane_from_delta(pc, coef_norm)
        vio_R = np.maximum(0.0, (Rmin_req - R) / Rmin_req).mean()
        pen = pen_scale * PENALTY["k_R"] * vio_R
        if limit is not None and pen >= limit:
            return limit
        # 生态区穿越长度(沿密集平面点)
        eco = dem.eco_mask_xy(xx, yy, pc["lat0"], pc["lon0"])
        seg = np.hypot(np.diff(xx), np.diff(yy))
        eco_seg = 0.5 * (eco[:-1].astype(float) + eco[1:].astype(float))
        L_eco_km = float(np.sum(eco_seg * seg)) / 1000.0
        # 交叉触发桥长(问题21, 与联合口径一致; 生态区内交叉不设桥)
        sarc = np.concatenate([[0], np.cumsum(seg)])
        cross = _cr.detect_crossings(xx, yy, pc["lat0"], pc["lon0"])
        keep = np.interp(cross["s"], sarc, eco.astype(float)) <= 0.5 \
            if len(cross["s"]) else np.zeros(0, dtype=bool)
        _, L_cross_m = _cr.bridge_intervals(cross, keep=keep)
        return plane_lcc(L_new, L_eco_km, L_cross_km=L_cross_m / 1000.0,
                         ramp_cost=pc["ramp_cost"]) / C_ref_plane \
            + pen

    return ScalarObjective(
        lambda x: value(x),
        lambda x, limit: value(x, limit))
