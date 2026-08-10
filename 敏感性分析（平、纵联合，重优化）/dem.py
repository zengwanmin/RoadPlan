# -*- coding: utf-8 -*-
"""
dem.py — 走廊带数字高程模型(DEM)的加载、准天然地面重建与生态区掩膜

【数据来源】AWS Terrain Tiles(terrarium 编码, 公开免密钥), 缩放级 z=14。
  扩展覆盖(fetch_dem_ext.py): lon 113.159~113.423, lat 23.095~23.201
  (实测线位外扩约 2.8 km, 支撑 ±2.5 km 走廊带), 60 瓦片拼接 1280×3072 栅格,
  本纬度(23.16°N)约 8.8 m/像素。解码 h = R·256+G+B/256−32768 (terrarium 规范)。
  文件: 数据/走廊带DEM_z14_ext.npz (<−100 m 坏点已按邻域中值填补)。

【准天然地面重建(natural ground)】
  原始 DEM 为【现状地表】, 既有北环路基已"长"在里面——沿现状线位 |设计−地面|≈0,
  现状方案的历史土方成本被计为零, 与优化新线位(踩未整形地面)不同口径。
  修复: 将实测中线两侧 ±MASK_HALF_W(60 m, 路基+边坡影响带)的像素掩膜移除,
  用影响带外邻域像素线性插值回填, 得到"移除道路后的准天然地面"(标准 DEM
  restoration / road removal 做法)。所有方案(含现状)统一对准天然地面计算土方,
  恢复口径公平。因建路前实测地形不可得, 论文中声明该近似即可。
  结果缓存: 数据/走廊带DEM_z14_ext_natural.npz (含 natural 高程与 eco 掩膜)。

【白云山生态强制隧道区(eco mask)】
  依论文 §2.1.1 选线原则(4)"优先避让自然保护区…确需跨越时实施最小干扰",
  穿越白云山风景区的线位强制以隧道通过(不允许大开挖)。风景区边界以
  【准天然地面高程 ≥ ECO_ELEV_M(70 m) 的连片山体 + 形态学闭运算(约500 m)】
  客观近似界定(闭运算将山体内部的沟谷/垭口并入连片区, 与"隧道连续通过山体"
  的工程事实一致)。参数经双指标校准: 界定面积 19.2 km² ≈ 白云山风景区实测
  约 21 km², 现状线穿越 1.28 km ≈ 实际隧道 1.35 km。可复现、不依赖人工划界。

【与实测数据的一致性(核验结果, 原始地表口径)】
  沿实测中线采样 DEM 与 GPS 实测高程: 相关系数 0.915, 中位偏差 +1.57 m。
  垂直基准差(DEM 正高 vs GPS 高程)以常数 BIAS_M 消除。
  采样用双线性插值, 保证目标函数对平面偏移连续可微。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "数据")
DEM_NPZ = os.path.join(DATA_DIR, "走廊带DEM_z14_ext.npz")
NATURAL_NPZ = os.path.join(DATA_DIR, "走廊带DEM_z14_ext_natural.npz")

BIAS_M = 1.57          # DEM 相对实测路面的垂直基准中位偏差(m)
R_EARTH = 6378137.0
PX_M = 8.78            # z14 在 23.16°N 的像素地面尺寸(m/px)
MASK_HALF_W = 60.0     # 道路影响带半宽(m): 路基 25.5/2 + 边坡 + 施工带
ECO_ELEV_M = 70.0      # 生态强制隧道区高程阈值(m, 准天然地面)
ECO_CLOSE_M = 500.0    # 形态学闭运算尺度(m): 山体内部沟谷并入连片区
# 校准依据(双指标): 70m/500m 下连片山体面积 19.2 km² ≈ 白云山风景名胜区实测
# 约 21 km²; 现状线穿越长度 1.28 km ≈ 实际白云山隧道 1.35 km(偏差 -5%)。

_D = None              # 现状地表
_N = None              # 准天然地面 + 生态掩膜


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


# =============================================================
#  准天然地面重建 + 生态区掩膜(一次构建, 磁盘缓存)
# =============================================================
def build_natural(force=False):
    """构建准天然地面与生态掩膜并缓存。多进程场景请在主进程先调用一次。"""
    global _N
    if not force and os.path.exists(NATURAL_NPZ):
        return
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata
    from scipy import ndimage
    from data_loader import load_alignment

    d = _load()
    elev = d["elev"].copy()
    H, W = elev.shape

    # 1) 实测中线 -> 像素坐标, 掩膜道路影响带(±MASK_HALF_W)
    a = load_alignment()
    cx, cy = lonlat_to_px(a["lon"], a["lat"])
    tree = cKDTree(np.column_stack([cx, cy]))
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.column_stack([jj.ravel(), ii.ravel()]).astype(float)
    dist_px, _ = tree.query(pix, k=1, distance_upper_bound=(MASK_HALF_W / PX_M) + 1)
    road_mask = (dist_px * PX_M <= MASK_HALF_W).reshape(H, W)

    # 2) 影响带内像素 -> 用带外邻域像素线性插值回填(road removal)
    src_ring = ndimage.binary_dilation(road_mask, iterations=6) & ~road_mask
    ri, rj = np.nonzero(src_ring)
    mi, mj = np.nonzero(road_mask)
    fill = griddata(np.column_stack([ri, rj]), elev[ri, rj],
                    np.column_stack([mi, mj]), method="linear")
    nan = np.isnan(fill)
    if nan.any():
        fill[nan] = griddata(np.column_stack([ri, rj]), elev[ri, rj],
                             np.column_stack([mi[nan], mj[nan]]),
                             method="nearest")
    natural = elev.copy()
    natural[mi, mj] = fill

    # 3) 生态强制隧道区: 准天然地面 ≥ ECO_ELEV_M 的连片山体 + 闭运算并入沟谷
    it = max(1, int(round(ECO_CLOSE_M / PX_M / 2)))
    eco = ndimage.binary_closing(natural - BIAS_M >= ECO_ELEV_M,
                                 structure=np.ones((3, 3), bool), iterations=it)
    eco = ndimage.binary_fill_holes(eco)

    np.savez_compressed(NATURAL_NPZ, natural=natural.astype(np.float32),
                        eco=eco.astype(np.uint8),
                        mask_half_w=MASK_HALF_W, eco_elev=ECO_ELEV_M,
                        eco_close=ECO_CLOSE_M)
    _N = None   # 强制重载
    print(f"[dem] 准天然地面重建完成: 掩膜像素 {road_mask.sum()}, "
          f"生态区像素 {eco.sum()} ({eco.sum()*PX_M*PX_M/1e6:.1f} km²)")


def _load_natural():
    global _N
    if _N is None:
        build_natural()
        d = np.load(NATURAL_NPZ)
        _N = dict(natural=d["natural"].astype(np.float64),
                  eco=d["eco"].astype(bool))
    return _N


# =============================================================
#  采样接口
# =============================================================
def _bilinear(E, px, py):
    d = _load()
    Hh, Ww = d["H"], d["W"]
    px = np.clip(px, 0.0, Ww - 1.001)
    py = np.clip(py, 0.0, Hh - 1.001)
    j0 = np.floor(px).astype(int); i0 = np.floor(py).astype(int)
    fx = px - j0; fy = py - i0
    j1 = np.minimum(j0 + 1, Ww - 1); i1 = np.minimum(i0 + 1, Hh - 1)
    return ((1 - fx) * (1 - fy) * E[i0, j0] + fx * (1 - fy) * E[i0, j1]
            + (1 - fx) * fy * E[i1, j0] + fx * fy * E[i1, j1])


def ground_elev_lonlat(lon, lat, natural=False):
    """双线性采样地面高程(m), 已扣除垂直基准偏差 BIAS_M。
    natural=True 采样准天然地面(移除既有路基), False 采样现状地表。"""
    E = _load_natural()["natural"] if natural else _load()["elev"]
    px, py = lonlat_to_px(lon, lat)
    return _bilinear(E, px, py) - BIAS_M


def ground_elev_xy(X, Y, lat0_deg, lon0_deg, natural=False):
    """按模型局部平面坐标采样地面高程(m)。"""
    lon, lat = xy_to_lonlat(X, Y, lat0_deg, lon0_deg)
    return ground_elev_lonlat(lon, lat, natural=natural)


def eco_mask_xy(X, Y, lat0_deg, lon0_deg):
    """按模型局部平面坐标查询是否落入白云山生态强制隧道区(bool 数组)。"""
    n = _load_natural()
    lon, lat = xy_to_lonlat(X, Y, lat0_deg, lon0_deg)
    px, py = lonlat_to_px(lon, lat)
    d = _load()
    j = np.clip(np.round(px).astype(int), 0, d["W"] - 1)
    i = np.clip(np.round(py).astype(int), 0, d["H"] - 1)
    return n["eco"][i, j]
