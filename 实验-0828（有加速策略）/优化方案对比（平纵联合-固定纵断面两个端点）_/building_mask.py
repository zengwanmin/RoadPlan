# -*- coding: utf-8 -*-
"""
building_mask.py — 建筑密度分区的逐桩号采样（仅供图C10可视化诊断）

【数据来源】数据/OSM走廊带障碍物/density_tiers_V1.npz
  由 data 分支 data/osm/buildings/building_density.py::export_tiers 生成(选定版本 V1 = 方案A)。
  完整方法、标定原理与局限见 data 分支 data/docs/数据来源与处理说明.md §2.8。

【三级分区与阈值标定】
  Tier2 高密度核心: D > θ_forbid
  Tier1 中密度区域: θ_pass < D <= θ_forbid
  Tier0 自由
  最终实验不把上述分区加入目标、惩罚或可行性约束，只用其绘制空间背景并统计
  线位叠加长度。
  阈值由现状线位标定: θ_forbid = 现状线位沿程最大密度 D_A_max × margin(1.15)。
  既有高速已建成并穿过居民区, 即"该密度可穿"的经验铁证; margin>=1 使 M-A 基准方案
  在 Tier2 上天然可行, 无需任何特例豁免。

【为什么 Tier2 用深度场而不是布尔掩膜】
  布尔掩膜在禁区【内部】是平台状的、梯度恒为 0, IJS 在里面感受不到任何方向, 无法被
  "推"出禁区; distance_transform_edt 得到的深度场处处指向最近的禁区边界, 提供明确的
  逃离方向。故 depth2 用【双线性插值】采样以保连续可微; tier1/tier2 布尔量只用于长度
  诊断, 用最近邻即可。这与 dem.py 中"高程用 _bilinear 保连续、掩膜用最近邻"同一取舍。

【坐标系】
  本栅格直接建在模型局部笛卡尔坐标上(原点=实测线位首点, 25 m 网格), 故 XY->索引是直接
  线性映射, 不需要像 dem.py 那样经 lon/lat -> 瓦片像素两次换算。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "数据")
TIERS_NPZ = os.path.join(DATA_DIR, "OSM走廊带障碍物", "density_tiers_V1.npz")

_T = None


def _load():
    global _T
    if _T is None:
        d = np.load(TIERS_NPZ)
        _T = dict(
            depth2=d["depth2"].astype(np.float64),
            tier1=d["tier1"].astype(bool),
            tier2=d["tier2"].astype(bool),
            S=d["S"].astype(np.float64),
            theta_pass=float(d["theta_pass"]),
            theta_forbid=float(d["theta_forbid"]),
            D_A_max=float(d["D_A_max"]),
            X0=float(d["X0"]), Y0=float(d["Y0"]), CELL=float(d["CELL"]),
            NX=int(d["NX"]), NY=int(d["NY"]),
            n_clusters=int(d["n_clusters"]),
        )
    return _T


def thresholds():
    """返回 (θ_pass, θ_forbid, D_A_max)。"""
    t = _load()
    return t["theta_pass"], t["theta_forbid"], t["D_A_max"]


def _to_grid(X, Y):
    """模型局部 XY(m) -> 栅格浮点索引 (gx 列, gy 行)。"""
    t = _load()
    gx = (np.asarray(X, float) - t["X0"]) / t["CELL"] - 0.5
    gy = (np.asarray(Y, float) - t["Y0"]) / t["CELL"] - 0.5
    return gx, gy


def _bilinear(F, gx, gy):
    """双线性插值采样(越界按边界值), 保证对平面偏移连续可微。"""
    t = _load()
    NX, NY = t["NX"], t["NY"]
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    fx = gx - x0
    fy = gy - y0
    x0 = np.clip(x0, 0, NX - 1); x1 = np.clip(x0 + 1, 0, NX - 1)
    y0 = np.clip(y0, 0, NY - 1); y1 = np.clip(y0 + 1, 0, NY - 1)
    return ((1 - fx) * (1 - fy) * F[y0, x0] + fx * (1 - fy) * F[y0, x1]
            + (1 - fx) * fy * F[y1, x0] + fx * fy * F[y1, x1])


def _nearest(M, gx, gy):
    t = _load()
    j = np.clip(np.round(gx).astype(int), 0, t["NX"] - 1)
    i = np.clip(np.round(gy).astype(int), 0, t["NY"] - 1)
    return M[i, j]


def depth2_at_xy(X, Y):
    """Tier2 高密度核心内部深度(m)，供可视化诊断。"""
    gx, gy = _to_grid(X, Y)
    return np.maximum(_bilinear(_load()["depth2"], gx, gy), 0.0)


def tier2_at_xy(X, Y):
    """是否落在 Tier2 高密度核心(布尔, 最近邻; 仅用于长度诊断)。"""
    gx, gy = _to_grid(X, Y)
    return _nearest(_load()["tier2"], gx, gy)


def tier1_at_xy(X, Y):
    """是否落在 Tier1 中密度区域(布尔, 最近邻; 仅用于长度诊断)。"""
    gx, gy = _to_grid(X, Y)
    return _nearest(_load()["tier1"], gx, gy)


def density_at_xy(X, Y):
    """建筑占地覆盖率密度场(双线性; 供诊断与出图)。"""
    gx, gy = _to_grid(X, Y)
    return _bilinear(_load()["S"], gx, gy)
