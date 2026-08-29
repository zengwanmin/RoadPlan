# -*- coding: utf-8 -*-
"""
fetch_dem_ext.py — 下载扩展走廊带 DEM(±2.5 km)并保存为 npz

【数据来源】AWS Terrain Tiles(terrarium 编码, 公开免密钥, 官方源
  https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png), z=14。
【覆盖】实测线位外扩约 2.8 km: x=13342..13353(12列), y=7106..7110(5行),
  即 lon 113.159~113.423, lat 23.095~23.201, 栅格 1280×3072。
【清理】terrarium 源数据存在个别 <-100 m 坏点(如 -936 m), 按 3×3 邻域中值填补,
  与原 走廊带DEM_z14.npz 的清理规则一致。
运行: python3 fetch_dem_ext.py   (瓦片缓存在 数据/tiles_z14/, 已存在则跳过下载)
"""
import io
import os
import urllib.request

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "数据")
TILE_DIR = os.path.join(DATA_DIR, "tiles_z14")
OUT_NPZ = os.path.join(DATA_DIR, "走廊带DEM_z14_ext.npz")

Z = 14
X0, X1 = 13342, 13353          # 含端点, 12 列
Y0, Y1 = 7106, 7110            # 含端点, 5 行
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def fetch_tile(x, y):
    os.makedirs(TILE_DIR, exist_ok=True)
    fn = os.path.join(TILE_DIR, f"{x}_{y}.png")
    if not os.path.exists(fn):
        with urllib.request.urlopen(URL.format(z=Z, x=x, y=y), timeout=60) as r:
            data = r.read()
        with open(fn, "wb") as f:
            f.write(data)
    a = np.asarray(Image.open(fn), dtype=np.float64)
    return a[:, :, 0] * 256 + a[:, :, 1] + a[:, :, 2] / 256 - 32768


def main():
    H = (Y1 - Y0 + 1) * 256
    W = (X1 - X0 + 1) * 256
    elev = np.zeros((H, W), dtype=np.float32)
    for iy, ty in enumerate(range(Y0, Y1 + 1)):
        for ix, tx in enumerate(range(X0, X1 + 1)):
            elev[iy * 256:(iy + 1) * 256, ix * 256:(ix + 1) * 256] = \
                fetch_tile(tx, ty)
        print(f"行 y={ty} 完成")
    # 清理 <-100 m 坏点(邻域中值)
    bad = np.argwhere(elev < -100)
    for r, c in bad:
        win = elev[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        good = win[win >= -100]
        elev[r, c] = np.median(good) if len(good) else 0.0
    print(f"清理坏点 {len(bad)} 个")
    np.savez_compressed(OUT_NPZ, elev=elev, z=Z, x0=X0, y0=Y0, H=H, W=W)
    print(f"保存 {OUT_NPZ}: {H}x{W}, 范围[{elev.min():.1f},{elev.max():.1f}]m")


if __name__ == "__main__":
    main()
