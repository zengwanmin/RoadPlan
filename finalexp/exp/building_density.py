# -*- coding: utf-8 -*-
"""
building_density.py — 建筑密度场构建、聚类分区与阈值标定

【目的】把完备 OSM 建筑轮廓集(22590 个)转成线位优化的空间约束:
  Tier2 严格禁行(硬约束) / Tier1 可穿越但计惩罚 / Tier0 自由。

【方法与既有生态区界定同构】
  dem.py::build_natural 用"阈值化 + 形态学闭运算 + 连通域 + 双指标标定"界定白云山
  生态强制隧道区; 本模块对建筑密度用同一套范式, 保持全项目方法论一致:
    多边形栅格化 -> 高斯平滑得邻域密度 -> 阈值化 -> 闭运算 -> 连通域 -> 最小簇过滤
  不引入 sklearn(仓库无此依赖); scipy.ndimage 已是既有依赖。

【阈值不是手填, 而是由现状线位标定】
  现状高速已建成并穿过部分居民区 —— 这就是"该密度可以穿过"的经验铁证。故取
    D_A_max  = 现状线位沿程密度最大值
    θ_forbid = D_A_max × margin      (margin >= 1)
    θ_pass   = θ_forbid × pass_ratio
  这样 M-A 基准方案在 Tier2 上天然可行, 不需要任何特例豁免。

【栅格化必须按真实多边形】
  先导测算曾把每栋建筑的全部占地面积记到质心格, 使大于网格的建筑覆盖率 >1(90 分位 1.248)。
  本模块改为在 5 m 子网格上做点在多边形判定(外环填充、内环即天井扣除), 再聚合到 25 m,
  并以"栅格化总面积 ≈ b_area_m2 之和"作为正确性硬指标。

© OpenStreetMap contributors (建筑数据), ODbL v1.0。
"""
import math
import os

import numpy as np
from matplotlib.path import Path
from scipy import ndimage

R_E = 6378137.0

# 栅格范围(模型局部笛卡尔, m): 覆盖建筑数据全域并留边
X0, X1 = -24000.0, 4000.0
Y0, Y1 = -7200.0, 5800.0
CELL = 25.0        # 最终密度栅格
SUB = 5.0          # 栅格化子网格(小建筑中位 519 m² 约 23×23 m, 25 m 格心判定会整栋漏掉)

NX = int(round((X1 - X0) / CELL))       # 1120
NY = int(round((Y1 - Y0) / CELL))       # 520
K = int(round(CELL / SUB))              # 5


def ll2xy(lon, lat, lat0_deg, lon0_deg):
    """与 data_loader.load_alignment 完全相同的局部平面投影。"""
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    return (R_E * math.cos(lat0) * (np.radians(lon) - lon0),
            R_E * (np.radians(lat) - lat0))


# =====================================================================
# 一、多边形栅格化
# =====================================================================
def rasterize(npz_path, lat0_deg, lon0_deg, cache=None, verbose=True):
    """
    返回 dict(area=G_area, count=G_count, vol=G_vol, meta=...)
      G_area  : 每 25 m 格内建筑占地面积(m²)
      G_count : 每格建筑栋数(按质心计)
      G_vol   : 每格建筑体量(占地面积 × 层数, 层数缺失按 1)
    """
    if cache and os.path.exists(cache):
        d = np.load(cache)
        if verbose:
            print(f"[栅格化] 命中缓存 {cache}")
        return dict(area=d["area"], count=d["count"], vol=d["vol"])

    b = np.load(npz_path, allow_pickle=False)
    PLON, PLAT = b["poly_lon"], b["poly_lat"]
    OFF, RING, BID = b["poly_off"], b["poly_ring"], b["poly_bid"]
    PX, PY = ll2xy(PLON, PLAT, lat0_deg, lon0_deg)

    lev = np.nan_to_num(b["b_levels"], nan=1.0)
    lev = np.clip(lev, 1.0, 100.0)

    nxs, nys = NX * K, NY * K
    sub_area = SUB * SUB
    A_sub = np.zeros((nys, nxs), dtype=np.float32)
    V_sub = np.zeros((nys, nxs), dtype=np.float32)

    # 逐环在其包围盒的子网格上做点在多边形判定; 外环 +1, 内环(天井) -1
    n_ring = len(RING)
    for k in range(n_ring):
        i0, i1 = OFF[k], OFF[k + 1]
        xs, ys = PX[i0:i1], PY[i0:i1]
        jx0 = int(math.floor((xs.min() - X0) / SUB))
        jx1 = int(math.ceil((xs.max() - X0) / SUB))
        jy0 = int(math.floor((ys.min() - Y0) / SUB))
        jy1 = int(math.ceil((ys.max() - Y0) / SUB))
        jx0, jx1 = max(jx0, 0), min(jx1, nxs - 1)
        jy0, jy1 = max(jy0, 0), min(jy1, nys - 1)
        if jx1 < jx0 or jy1 < jy0:
            continue
        gx = X0 + (np.arange(jx0, jx1 + 1) + 0.5) * SUB
        gy = Y0 + (np.arange(jy0, jy1 + 1) + 0.5) * SUB
        MX, MY = np.meshgrid(gx, gy)
        inside = Path(np.c_[xs, ys]).contains_points(
            np.c_[MX.ravel(), MY.ravel()]).reshape(MX.shape)
        if not inside.any():
            # 建筑小于子网格且未含格心: 记到其包围盒中心格, 保证面积不丢
            cxk = int((0.5 * (xs.min() + xs.max()) - X0) / SUB)
            cyk = int((0.5 * (ys.min() + ys.max()) - Y0) / SUB)
            if 0 <= cxk < nxs and 0 <= cyk < nys:
                inside = None
                sgn = 1.0 if RING[k] == "outer" else -1.0
                w = lev[BID[k]]
                A_sub[cyk, cxk] += sgn * sub_area
                V_sub[cyk, cxk] += sgn * sub_area * w
            continue
        sgn = 1.0 if RING[k] == "outer" else -1.0
        w = lev[BID[k]]
        blkA = A_sub[jy0:jy1 + 1, jx0:jx1 + 1]
        blkV = V_sub[jy0:jy1 + 1, jx0:jx1 + 1]
        blkA[inside] += sgn * sub_area
        blkV[inside] += sgn * sub_area * w

        if verbose and (k + 1) % 5000 == 0:
            print(f"  栅格化 {k+1}/{n_ring} 环", flush=True)

    A_sub = np.clip(A_sub, 0.0, None)
    V_sub = np.clip(V_sub, 0.0, None)

    # 重叠去重: 同一子格被多栋建筑覆盖时会累加, 使覆盖率 >1(实测最高 4.0), 而地面
    # 覆盖率物理上不可能超过 1, 且阈值是按覆盖率标定的, 必须先去重。
    # 做法: 子格占地面积截断到一个满格; 体量按同一比例缩放以保留平均层数。
    over = A_sub > sub_area
    n_over = int(over.sum())
    if n_over:
        scale = np.ones_like(A_sub)
        scale[over] = sub_area / A_sub[over]
        A_sub *= scale
        V_sub *= scale
    if verbose:
        print(f"[栅格化] 重叠去重: {n_over} 个 5 m 子格覆盖率曾 >1, 已截断至 1")

    # 5 m 子网格聚合到 25 m
    G_area = A_sub.reshape(NY, K, NX, K).sum(axis=(1, 3))
    G_vol = V_sub.reshape(NY, K, NX, K).sum(axis=(1, 3))

    # 栋数密度: 按质心记入所在格
    BX, BY = ll2xy(b["b_lon"], b["b_lat"], lat0_deg, lon0_deg)
    jx = ((BX - X0) / CELL).astype(int)
    jy = ((BY - Y0) / CELL).astype(int)
    ok = (jx >= 0) & (jx < NX) & (jy >= 0) & (jy < NY)
    G_count = np.zeros((NY, NX), dtype=np.float32)
    np.add.at(G_count, (jy[ok], jx[ok]), 1.0)

    if verbose:
        true_km2 = float(b["b_area_m2"].sum()) / 1e6
        got_km2 = float(G_area.sum()) / 1e6
        err = (got_km2 - true_km2) / true_km2 * 100
        print(f"[栅格化自检] 栅格总面积 {got_km2:.3f} km²  "
              f"轮廓真实总面积 {true_km2:.3f} km²  偏差 {err:+.2f}%")
        print(f"[栅格化] 建筑落域 {int(ok.sum())}/{len(BX)}, "
              f"非零格 {int((G_area>0).sum())}/{G_area.size}")

    out = dict(area=G_area, count=G_count, vol=G_vol)
    if cache:
        np.savez_compressed(cache, **out)
        if verbose:
            print(f"[栅格化] 已缓存 {cache}")
    return out


# =====================================================================
# 二、密度场与聚类分区
# =====================================================================
def density_field(G, metric, sigma_m):
    """
    归一化密度场:
      area  -> 建筑占地覆盖率 (m²/m², 0~1)
      count -> 栋数密度 (栋/公顷)
      vol   -> 体量覆盖率 (m³/m², 即等效层数)
    再做高斯平滑得邻域密度。
    """
    if metric == "area":
        D = G / (CELL * CELL)
    elif metric == "vol":
        D = G / (CELL * CELL)
    elif metric == "count":
        D = G / (CELL * CELL / 1e4)          # 栋/公顷
    else:
        raise ValueError(metric)
    return ndimage.gaussian_filter(D.astype(np.float64), sigma=sigma_m / CELL)


def calibrate(S, rx, ry, margin, pass_ratio=0.5):
    """由现状线位沿程密度导出阈值。返回 (D_A_max, θ_forbid, θ_pass, D_A 剖面)。"""
    jx = np.clip(((rx - X0) / CELL).astype(int), 0, NX - 1)
    jy = np.clip(((ry - Y0) / CELL).astype(int), 0, NY - 1)
    D_A = S[jy, jx]
    D_A_max = float(D_A.max())
    th_f = D_A_max * margin
    return D_A_max, th_f, th_f * pass_ratio, D_A


def cluster_tiers(S, th_pass, th_forbid, close_m=100.0, min_area_ha=1.0):
    """
    阈值化 + 闭运算 + 连通域 + 最小簇过滤 -> (tier2, tier1, n_clusters)
    close_m=0 或 min_area_ha=0 时跳过对应后处理(用于最严版本)。
    """
    raw = S > th_forbid
    if close_m > 0:
        it = max(1, int(round(close_m / CELL / 2)))
        raw = ndimage.binary_closing(raw, structure=np.ones((3, 3), bool),
                                     iterations=it)
        raw = ndimage.binary_fill_holes(raw)
    lab, n = ndimage.label(raw)
    if min_area_ha > 0 and n > 0:
        min_cells = int(min_area_ha * 1e4 / (CELL * CELL))
        sizes = ndimage.sum(raw, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:][sizes >= min_cells] = True
        raw = keep[lab]
        lab, n = ndimage.label(raw)
    tier2 = raw
    tier1 = (S > th_pass) & ~tier2
    return tier2, tier1, int(n)


# =====================================================================
# 三、诊断
# =====================================================================
def _mask_at(mask, qx, qy):
    jx = np.clip(((qx - X0) / CELL).astype(int), 0, NX - 1)
    jy = np.clip(((qy - Y0) / CELL).astype(int), 0, NY - 1)
    return mask[jy, jx]


def route_exposure(mask, rx, ry, rs):
    """现状线位穿越掩膜的里程(km), 用段中点平均(与 decode_joint 同惯用法)。"""
    m = _mask_at(mask, rx, ry).astype(float)
    return float(np.sum(0.5 * (m[:-1] + m[1:]) * np.diff(rs))) / 1000.0


def corridor_passability(tier2, X, Y, s, half_w, step=50.0, dq=25.0):
    """
    沿线位法向扫断面, 检查走廊带是否被 Tier2 封堵。
    返回 (完全封堵断面数, 断面总数, 最窄可通宽度 m, 可通宽度中位数 m)
    """
    tx, ty = np.gradient(X), np.gradient(Y)
    tn = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tn, tx / tn
    sm = np.arange(0, s[-1], step)
    px, py = np.interp(sm, s, X), np.interp(sm, s, Y)
    pnx, pny = np.interp(sm, s, nx), np.interp(sm, s, ny)
    off = np.arange(-half_w, half_w + dq, dq)
    blocked = 0
    widths = []
    for i in range(len(sm)):
        qx = px[i] + off * pnx[i]
        qy = py[i] + off * pny[i]
        free = ~_mask_at(tier2, qx, qy)
        if not free.any():
            blocked += 1
            widths.append(0.0)
            continue
        e = np.diff(np.concatenate(([0], free.view(np.int8), [0])))
        runs = (np.flatnonzero(e == -1) - np.flatnonzero(e == 1)) * dq
        widths.append(float(runs.max()))
    w = np.array(widths)
    return blocked, len(sm), float(w.min()), float(np.median(w))
