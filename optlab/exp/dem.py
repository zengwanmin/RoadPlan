# -*- coding: utf-8 -*-
"""
dem.py — 走廊带数字高程模型(DEM)的加载与采样

【数据来源】AWS Terrain Tiles(terrarium 编码, 公开免密钥), 缩放级 z=14。
  覆盖: lon 113.1812~113.4009, lat 23.1404~23.2010(实测线位外扩约 1 km),
  30 个 256×256 瓦片拼接为 768×2560 栅格, 赤道分辨率 9.55 m,
  本纬度(23.16°N)约 8.8 m/像素。
  解码: h = R·256 + G + B/256 − 32768 (terrarium 规范)。
  文件: 数据/走廊带DEM_z14.npz (已清理 2 个 <−100 m 的异常像素, 邻域中值填补)。

【与实测数据的一致性(核验结果)】
  沿实测中线采样 DEM 与 GPS 实测高程: 相关系数 0.915, 中位偏差 +1.57 m。
  98.7% 的桩号处 |DEM − 路面| ≤ 20 m; 仅 0.256 km(两处, 最大 53.1 m)DEM 显著
  高于路面, 对应白云山隧道段——DEM 给的是隧道上方山体地面, 物理上正确。

【口径声明】
  1. DEM 为【现状地表】, 已含既有路基/桥隧/城市建筑, 非天然原地面。但本模型原先
     使用的"地面高程"是 GPS 实测的【路面】高程, 同样非天然地面, 故此偏差不是新引入
     的问题, 仅需在论文中写明口径。
  2. 垂直基准差异(DEM 正高 vs GPS 高程)以常数 BIAS_M 消除, 该常数取"至平地段"
     (|DEM − 路面| ≤ 20 m, 占 98.7%)的中位偏差, 使 DEM 在平地段与实测路面对齐。
  3. 采样用双线性插值, 保证目标函数对平面偏移连续可微, 利于寻优。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# v2: 宽版 DEM(走廊带±3 km 缓冲, 48 瓦片)。与窄版在实测中线上逐点一致(差异 0.000 m),
# 仅扩大覆盖, 基线不变。
DEM_NPZ = os.path.join(HERE, "dem_xwide_z14.npz")   # v3: ±4.5 km 缓冲(84瓦片), 中线一致性 0.000 m

BIAS_M = 1.57      # 至平地段 DEM 相对实测路面的中位偏差(m), 见模块 docstring
R_EARTH = 6378137.0

_D = None


def _load():
    global _D
    if _D is None:
        d = np.load(DEM_NPZ)
        _D = dict(elev=d["elev"].astype(np.float64), z=int(d["z"]),
                  x0=int(d["x0"]), y0=int(d["y0"]))
        _D["H"], _D["W"] = _D["elev"].shape
    return _D


def lonlat_to_px(lon, lat):
    """经纬度 -> DEM 栅格像素坐标(Web Mercator 瓦片像素, 浮点)。"""
    d = _load()
    n = 2 ** d["z"]
    la = np.radians(lat)
    px = ((np.asarray(lon, float) + 180.0) / 360.0 * n - d["x0"]) * 256.0
    py = ((1.0 - np.log(np.tan(la) + 1.0 / np.cos(la)) / np.pi) / 2.0 * n
          - d["y0"]) * 256.0
    return px, py


def xy_to_lonlat(X, Y, lat0_deg, lon0_deg):
    """
    模型局部平面坐标 -> 经纬度。与 data_loader 的正向变换严格互逆:
      X = R·cos(lat0)·(lon − lon0),  Y = R·(lat − lat0)   [lon/lat 为弧度]
    """
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    lon = lon0 + np.asarray(X, float) / (R_EARTH * np.cos(lat0))
    lat = lat0 + np.asarray(Y, float) / R_EARTH
    return np.degrees(lon), np.degrees(lat)


def ground_elev_lonlat(lon, lat):
    """按经纬度双线性采样地面高程(m), 已扣除垂直基准偏差 BIAS_M。"""
    d = _load()
    E, Hh, Ww = d["elev"], d["H"], d["W"]
    px, py = lonlat_to_px(lon, lat)
    px = np.clip(px, 0.0, Ww - 1.001)
    py = np.clip(py, 0.0, Hh - 1.001)
    j0 = np.floor(px).astype(int); i0 = np.floor(py).astype(int)
    fx = px - j0; fy = py - i0
    j1 = np.minimum(j0 + 1, Ww - 1); i1 = np.minimum(i0 + 1, Hh - 1)
    v = ((1 - fx) * (1 - fy) * E[i0, j0] + fx * (1 - fy) * E[i0, j1]
         + (1 - fx) * fy * E[i1, j0] + fx * fy * E[i1, j1])
    return v - BIAS_M


def ground_elev_xy(X, Y, lat0_deg, lon0_deg):
    """按模型局部平面坐标采样地面高程(m)。"""
    lon, lat = xy_to_lonlat(X, Y, lat0_deg, lon0_deg)
    return ground_elev_lonlat(lon, lat)
