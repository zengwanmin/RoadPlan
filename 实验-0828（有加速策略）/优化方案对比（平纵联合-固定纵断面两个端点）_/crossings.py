# -*- coding: utf-8 -*-
"""
crossings.py — 平面线形与 OSM 障碍物(道路/铁路/水系)的交叉检测与跨越桥内生触发。

口径(待办清单2 问题21, 2026-08-11 定案):
  触发类别: road ∈ {motorway, trunk, primary} + rail + water(river/canal);
            secondary 不触发(现实为下穿通道/涵洞, 属路基附属造价, 声明忽略)。
  桥区间长度 = 障碍物缺省宽度 / sinθ (θ=交叉角, sinθ 下限 clamp 15°防病态)
              + 两侧各 ext_m 延伸;
  区间合并: 间隙 < merge_gap_m 合并为连续高架(消除双向车道/平行 way 重复计数)。

数据: 数据/OSM走廊带障碍物/obstacles.npz (© OpenStreetMap contributors, ODbL),
      已剔除北环高速自身/伴行线(见该目录 README)。

实现: 障碍物折线一次性转局部平面坐标(与 data_loader 相同投影), 建 250m 均匀
      网格索引; 每次评估对线形折线(~50m 重采样)做候选段矢量化求交, 增量 ~ms 级。
"""
import os

import numpy as np

from params import BRIDGE_TUNNEL
from acceleration import NUMBA_AVAILABLE, segment_intersections_kernel

_TRIG = BRIDGE_TUNNEL["crossing_trigger"]
_GRID_CELL = 250.0
_KEY_MUL = 1_000_000     # cell 键编码 cx*_KEY_MUL+cy (|cell 索引|<<50万, 保序)
_PAIR_MUL = 10_000_000   # (段,障碍)对编码 ia*_PAIR_MUL+ib (障碍数~3万<<1千万)

_OBS = None          # 惰性单例: dict(px1,py1,px2,py2,width,hclear,kind_code,grid)
_KIND_NAMES = ("road", "rail", "water")


def _width_of(kind, hclass, name):
    w = _TRIG["width_m"]
    if kind == "road":
        return float(w.get(hclass, w["primary"]))
    if kind == "rail":
        return float(w["rail"])
    for key, wv in _TRIG["water_width_by_name"]:
        if key in name:
            return float(wv)
    return float(w["water"])


def _load_obstacles(lat0, lon0):
    """加载并预处理障碍物线段(局部平面坐标, 与 data_loader 相同投影)。"""
    global _OBS
    if _OBS is not None:
        return _OBS
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "数据", "OSM走廊带障碍物")
    d = np.load(os.path.join(base, "obstacles.npz"), allow_pickle=True)
    lon_f, lat_f, offs = d["lines_lon"], d["lines_lat"], d["offsets"]
    kinds = [str(x) for x in d["kind"]]
    hcls = [str(x) for x in d["highway_class"]]
    names = [str(x) for x in d["name"]]
    R_E = 6378137.0
    la0 = np.radians(lat0); lo0 = np.radians(lon0)
    kx = R_E * np.cos(la0)

    x1l, y1l, x2l, y2l, wl, hl, kl = [], [], [], [], [], [], []
    for i in range(len(offs) - 1):
        kind = kinds[i]
        if kind == "road" and hcls[i] not in _TRIG["road_classes"]:
            continue
        lx = kx * (np.radians(lon_f[offs[i]:offs[i+1]]) - lo0)
        ly = R_E * (np.radians(lat_f[offs[i]:offs[i+1]]) - la0)
        if len(lx) < 2:
            continue
        w = _width_of(kind, hcls[i], names[i])
        hc = float(_TRIG["clearance_m"][kind])
        kc = _KIND_NAMES.index(kind)
        x1l.append(lx[:-1]); y1l.append(ly[:-1])
        x2l.append(lx[1:]);  y2l.append(ly[1:])
        n = len(lx) - 1
        wl.append(np.full(n, w)); hl.append(np.full(n, hc))
        kl.append(np.full(n, kc, dtype=np.int8))
    px1 = np.concatenate(x1l); py1 = np.concatenate(y1l)
    px2 = np.concatenate(x2l); py2 = np.concatenate(y2l)
    width = np.concatenate(wl); hclear = np.concatenate(hl)
    kcode = np.concatenate(kl)

    # 均匀网格索引: cell -> 障碍段索引数组(段 bbox 覆盖的所有 cell)
    cx_min = np.floor(np.minimum(px1, px2) / _GRID_CELL).astype(int)
    cx_max = np.floor(np.maximum(px1, px2) / _GRID_CELL).astype(int)
    cy_min = np.floor(np.minimum(py1, py2) / _GRID_CELL).astype(int)
    cy_max = np.floor(np.maximum(py1, py2) / _GRID_CELL).astype(int)
    grid = {}
    for i in range(len(px1)):
        for cx in range(cx_min[i], cx_max[i] + 1):
            for cy in range(cy_min[i], cy_max[i] + 1):
                grid.setdefault((cx, cy), []).append(i)
    grid = {c: np.asarray(v, dtype=int) for c, v in grid.items()}
    # CSR 形式(供 detect_crossings 向量化候选收集, 避免逐 cell 字典查询+concatenate+unique):
    #   cells 按 (cx,cy) 排序; cell_start 为前缀和偏移; cell_obs 为按同序拼接的障碍段索引。
    cells = sorted(grid.keys())
    kx = np.array([c[0] for c in cells], dtype=np.int64)
    ky = np.array([c[1] for c in cells], dtype=np.int64)
    ckey = kx * _KEY_MUL + ky
    lens = np.array([len(grid[c]) for c in cells], dtype=np.int64)
    cell_start = np.concatenate([[0], np.cumsum(lens)])
    cell_obs = (np.concatenate([grid[c] for c in cells]).astype(np.int64)
                if cells else np.empty(0, dtype=np.int64))
    _OBS = dict(px1=px1, py1=py1, px2=px2, py2=py2,
                width=width, hclear=hclear, kcode=kcode, grid=grid,
                csr=dict(kx=kx, ky=ky, ckey=ckey,
                         cell_start=cell_start, cell_obs=cell_obs))
    # 所有方案/算法共享同一份只读 OSM 几何与空间索引，避免重复拷贝和意外改写。
    for key in ("px1", "py1", "px2", "py2", "width", "hclear", "kcode"):
        _OBS[key].setflags(write=False)
    for value in _OBS["grid"].values():
        value.setflags(write=False)
    for value in _OBS["csr"].values():
        value.setflags(write=False)
    return _OBS


def preload_readonly(lat0, lon0):
    """在算法计时前加载一次 OSM 障碍物和网格索引。"""
    return _load_obstacles(lat0, lon0)


def detect_crossings(xx, yy, lat0, lon0, step_m=50.0):
    """
    线形折线与障碍物求交。

    xx, yy: 平面线形密集点(局部平面坐标)。内部按 step_m 重采样后逐段求交。
    返回 dict of arrays(按交点):
      s     交点里程(m, 沿重采样折线弧长)
      theta 交叉角(rad, ∈(0, π/2])
      width/hclear/kcode  障碍物属性
      x, y  交点坐标
    """
    obs = _load_obstacles(lat0, lon0)
    sarc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))])
    n = max(int(sarc[-1] / step_m) + 1, 2)
    ss = np.linspace(0.0, sarc[-1], n)
    ax = np.interp(ss, sarc, xx); ay = np.interp(ss, sarc, yy)

    grid = obs["grid"]
    px1, py1 = obs["px1"], obs["py1"]
    px2, py2 = obs["px2"], obs["py2"]

    # --- 向量化候选收集(CSR 网格) ---
    # 段 bbox 覆盖的 cell 用二分定位(cell 键编码为单个 int64), 候选障碍段按 CSR 切片取出。
    # 最后对 (段,障碍) 对去重: 障碍横跨多 cell 时会重复出现, 而下游求交不去重,
    # 重复会把同一交点计两次 -> 桥长多算。去重后与旧实现的逐段 unique 结果完全一致。
    csr = obs["csr"]
    ckey = csr["ckey"]                            # 预计算；按(kx,ky)升序
    cell_start, cell_obs = csr["cell_start"], csr["cell_obs"]
    cx0 = np.floor(np.minimum(ax[:-1], ax[1:]) / _GRID_CELL).astype(np.int64)
    cx1 = np.floor(np.maximum(ax[:-1], ax[1:]) / _GRID_CELL).astype(np.int64)
    cy0 = np.floor(np.minimum(ay[:-1], ay[1:]) / _GRID_CELL).astype(np.int64)
    cy1 = np.floor(np.maximum(ay[:-1], ay[1:]) / _GRID_CELL).astype(np.int64)
    ia_l, ib_l = [], []
    for i in range(n - 1):
        for cx in range(cx0[i], cx1[i] + 1):
            base = cx * _KEY_MUL
            # 该 cx 列内 cy0..cy1 的 cell 键区间, 二分批量定位
            lo = np.searchsorted(ckey, base + cy0[i])
            hi = np.searchsorted(ckey, base + cy1[i], side="right")
            for j in range(lo, hi):
                s0, s1 = cell_start[j], cell_start[j + 1]
                if s1 > s0:
                    ib_l.append(cell_obs[s0:s1])
                    ia_l.append(np.full(s1 - s0, i, dtype=np.int64))
    if not ia_l:
        return dict(s=np.empty(0), theta=np.empty(0), width=np.empty(0),
                    hclear=np.empty(0), kcode=np.empty(0, dtype=np.int8),
                    x=np.empty(0), y=np.empty(0))
    ia = np.concatenate(ia_l); ib = np.concatenate(ib_l)
    # (段,障碍) 对去重
    pair = ia * _PAIR_MUL + ib
    uniq = np.unique(pair)
    ia = (uniq // _PAIR_MUL).astype(int)
    ib = (uniq % _PAIR_MUL).astype(int)

    # 矢量化线段求交: A=(p, r), B=(q, w); t=cross(q-p,w)/cross(r,w), u=cross(q-p,r)/...
    if NUMBA_AVAILABLE:
        ok, t, u = segment_intersections_kernel(
            np.asarray(ax, dtype=np.float64), np.asarray(ay, dtype=np.float64),
            px1, py1, px2, py2,
            np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64))
        rpx = ax[ia + 1] - ax[ia]; rpy = ay[ia + 1] - ay[ia]
        qwx = px2[ib] - px1[ib];   qwy = py2[ib] - py1[ib]
    else:
        rpx = ax[ia + 1] - ax[ia]; rpy = ay[ia + 1] - ay[ia]
        qwx = px2[ib] - px1[ib];   qwy = py2[ib] - py1[ib]
        dqx = px1[ib] - ax[ia];    dqy = py1[ib] - ay[ia]
        den = rpx * qwy - rpy * qwx
        ok = np.abs(den) > 1e-12
        t = np.where(ok, (dqx * qwy - dqy * qwx) / np.where(ok, den, 1.0), -1.0)
        u = np.where(ok, (dqx * rpy - dqy * rpx) / np.where(ok, den, 1.0), -1.0)
    hit = ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    if not np.any(hit):
        return dict(s=np.empty(0), theta=np.empty(0), width=np.empty(0),
                    hclear=np.empty(0), kcode=np.empty(0, dtype=np.int8),
                    x=np.empty(0), y=np.empty(0))
    ia_h, ib_h, t_h = ia[hit], ib[hit], t[hit]
    s_hit = ss[ia_h] + t_h * (ss[ia_h + 1] - ss[ia_h])
    x_hit = ax[ia_h] + t_h * (ax[ia_h + 1] - ax[ia_h])
    y_hit = ay[ia_h] + t_h * (ay[ia_h + 1] - ay[ia_h])
    va = np.arctan2(rpy[hit], rpx[hit])
    vb = np.arctan2(qwy[hit], qwx[hit])
    dang = np.abs(va - vb) % np.pi
    theta = np.minimum(dang, np.pi - dang)          # 交叉角 ∈ [0, π/2]
    return dict(s=s_hit, theta=theta,
                width=obs["width"][ib_h], hclear=obs["hclear"][ib_h],
                kcode=obs["kcode"][ib_h], x=x_hit, y=y_hit)


def bridge_intervals(cr, keep=None):
    """
    由交叉点生成跨越桥里程区间并合并。

    cr: detect_crossings 的返回; keep: 可选布尔掩膜(如剔除生态隧道区内交叉)。
    区间 = s ± (width/sinθ_clamped/2 + ext_m); 间隙 < merge_gap_m 合并。
    返回 (intervals[(a,b)...], L_total_m)。
    """
    s, th, w = cr["s"], cr["theta"], cr["width"]
    if keep is not None:
        s, th, w = s[keep], th[keep], w[keep]
    if len(s) == 0:
        return [], 0.0
    sin_min = np.sin(np.radians(_TRIG["sin_clamp_deg"]))
    half = 0.5 * w / np.maximum(np.sin(th), sin_min) + _TRIG["ext_m"]
    a = s - half; b = s + half
    order = np.argsort(a)
    a, b = a[order], b[order]
    gap = _TRIG["merge_gap_m"]
    iv = [[a[0], b[0]]]
    for i in range(1, len(a)):
        if a[i] - iv[-1][1] < gap:
            iv[-1][1] = max(iv[-1][1], b[i])
        else:
            iv.append([a[i], b[i]])
    L = float(sum(v[1] - v[0] for v in iv))
    return [(float(v[0]), float(v[1])) for v in iv], L


def mask_from_intervals(sta, intervals):
    """里程区间 -> 评价桩号布尔掩膜。"""
    m = np.zeros(len(sta), dtype=bool)
    for va, vb in intervals:
        m |= (sta >= va) & (sta <= vb)
    return m


def ramp_cost_from_baseline(intervals_baseline):
    """
    7座立交匝道功能费常数(元)的现状线校准。

    统计口径 7.8km(interchange_total_km) = 立交范围内主线桥 + 匝道/加减速段;
    交叉触发已内生了主线桥, 剩余为匝道功能费:
      ramp = max(0, 7.8km − Σ 现状线触发桥区间与立交统计带 [s_cross±L/2] 的重叠长)
             × 桥单价。
    匝道费与线位无关(任何线位都须与这7条被交道路互通), 对所有候选为常数。
    依赖 ic_anchor_cache.json(OSM 锚定缓存, 由 objective_joint 首次构建)。
    """
    import json
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "数据", "OSM走廊带障碍物")
    with open(os.path.join(base, "ic_anchor_cache.json"), encoding="utf-8") as f:
        anchors = json.load(f)
    L_ic_main = 0.0
    for rec in anchors.values():
        a0 = rec["s_cross"] - rec["L_km"] * 500.0
        b0 = rec["s_cross"] + rec["L_km"] * 500.0
        for va, vb in intervals_baseline:
            L_ic_main += max(0.0, min(b0, vb) - max(a0, va))
    L_ramp_km = max(0.0, BRIDGE_TUNNEL["interchange_total_km"] - L_ic_main / 1000.0)
    return L_ramp_km * BRIDGE_TUNNEL["bridge_cost_per_km"], L_ic_main / 1000.0
