# -*- coding: utf-8 -*-
"""
verify_data.py — 本地数据一键完整性复核（不联网）

校验 data/ 下各数据文件的自洽性与关键指标, 逐项与
docs/数据来源与处理说明.md §5 速查表中的期望值比对。
任一项不符即以非零码退出, 便于接入 CI 或提交前自查。

用法:  python3 verify/verify_data.py
"""
import json
import os
import sys
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEM = os.path.join(ROOT, "dem")
OBS = os.path.join(ROOT, "osm", "obstacles")
BLD = os.path.join(ROOT, "osm", "buildings")
MEA = os.path.join(ROOT, "measured")

fails = []
notes = []


def check(name, got, want, ok=None):
    good = (got == want) if ok is None else ok
    print(f"  {'OK  ' if good else 'FAIL'}  {name}: 实得 {got}  期望 {want}")
    if not good:
        fails.append(name)


def section(t):
    print(f"\n=== {t} ===")


# ---------------- DEM ----------------
section("DEM 高程")
p = os.path.join(DEM, "走廊带DEM_z14_ext.npz")
d = np.load(p)
E = d["elev"].astype(float)
check("DEM 栅格尺寸", f"{E.shape[0]}x{E.shape[1]}", "1280x3072")
check("DEM 缩放级 z", int(d["z"]), 14)
bad = int(np.sum(E < -100.0))
check("DEM 坏点(<-100m)残留", bad, 0)
print(f"        高程范围 {np.nanmin(E):.1f} ~ {np.nanmax(E):.1f} m")

p = os.path.join(DEM, "走廊带DEM_z14_ext_natural.npz")
if os.path.exists(p):
    n = np.load(p)
    NAT = n["natural"].astype(float)
    check("准天然地面尺寸与 DEM 一致", NAT.shape == E.shape, True, NAT.shape == E.shape)
    eco = n["eco"].astype(bool)
    # 生态区面积: 像元数 x 像元面积(8.78 m)
    area_km2 = eco.sum() * (8.78 ** 2) / 1e6
    check("生态区面积 19.2 km²(容差 ±1.5)", f"{area_km2:.1f}", "19.2",
          abs(area_km2 - 19.2) <= 1.5)
    check("准天然地面掩膜半宽", float(n["mask_half_w"]), 60.0)
    check("生态区高程阈值", float(n["eco_elev"]), 70.0)
    # 道路影响带应被抹平: 天然地面与现状地表存在差异
    diff = np.abs(NAT - E)
    print(f"        天然地面 vs 现状地表: 非零差异像元 {int((diff>0.01).sum())} 个, "
          f"最大 {diff.max():.1f} m")
else:
    notes.append("走廊带DEM_z14_ext_natural.npz 不存在(可由 dem.build_natural() 生成)")

# ---------------- OSM 障碍物 ----------------
section("OSM 障碍物(道路/铁路/水系)")
o = np.load(os.path.join(OBS, "obstacles.npz"), allow_pickle=False)
kind = o["kind"]
check("折线总数", len(kind), 6002)
for k, want in (("road", 4807), ("rail", 857), ("water", 338)):
    check(f"  {k} 折线数", int((kind == k).sum()), want)
check("offsets 长度 = 折线数+1", len(o["offsets"]), len(kind) + 1)
check("顶点数与 offsets 末位一致", len(o["lines_lon"]), int(o["offsets"][-1]))

q = os.path.join(OBS, "query.overpass")
check("查询语句原文存在", os.path.exists(q), True)
if os.path.exists(q):
    txt = open(q, encoding="utf-8").read()
    check("  查询 bbox 与 DEM 一致", "23.095,113.159,23.201,113.423" in txt, True)

# ---------------- OSM 建筑 ----------------
section("OSM 建筑完整集")
b = np.load(os.path.join(BLD, "buildings_full.npz"), allow_pickle=False)
btype = b["b_type"]
n_way = int((btype == "way").sum())
n_rel = int((btype == "rel").sum())
check("way 建筑数", n_way, 22414)
check("rel 建筑数(22591-1 退化)", n_rel, 176)
check("建筑总数", n_way + n_rel, 22590)
check("building:part 数", len(b["part_b_id"]), 397)
check("环数", len(b["poly_ring"]), 22787)
check("顶点数", len(b["poly_lon"]), 171039)
check("poly_off 长度 = 环数+1", len(b["poly_off"]), len(b["poly_ring"]) + 1)
check("顶点数与 poly_off 末位一致", len(b["poly_lon"]), int(b["poly_off"][-1]))

km2 = float(b["b_area_m2"].sum()) / 1e6
check("总占地 24.41 km²(容差 ±0.05)", f"{km2:.2f}", "24.41", abs(km2 - 24.41) <= 0.05)

bases = [str(x) for x in b["osm_base"]]
days = sorted({s[:10] for s in bases})
check("快照一致性(全部同日)", days, ["2026-08-11"], days == ["2026-08-11"])
check("bbox 与 DEM 一致",
      list(np.round(b["bbox"], 3)), [23.095, 113.159, 23.201, 113.423],
      list(np.round(b["bbox"], 3)) == [23.095, 113.159, 23.201, 113.423])

# 几何自洽: 每个环至少 4 个顶点且首尾闭合
off, PL, PA = b["poly_off"], b["poly_lon"], b["poly_lat"]
short = unclosed = 0
for k in range(len(b["poly_ring"])):
    i0, i1 = off[k], off[k + 1]
    if i1 - i0 < 4:
        short += 1
    elif PL[i0] != PL[i1 - 1] or PA[i0] != PA[i1 - 1]:
        unclosed += 1
check("环顶点数 >= 4", short, 0)
check("环首尾闭合", unclosed, 0)

tgz = os.path.join(BLD, "buildings_raw_tiles.tar.gz")
if os.path.exists(tgz):
    with tarfile.open(tgz) as t:
        n_json = sum(1 for m in t.getmembers() if m.name.endswith(".json"))
    check("原始分块响应数", n_json, 384)
else:
    notes.append("buildings_raw_tiles.tar.gz 不存在(不影响使用, 仅影响断点续抓)")

# ---------------- 实测数据 ----------------
section("实测数据")
try:
    import pandas as pd
    df = pd.read_excel(os.path.join(MEA, "数据.xlsx"))
    check("实测中线点数", len(df), 14018)
    for c in ("Latitude", "Longitude", "Altitude", "Distance", "R"):
        check(f"  字段 {c}", c in df.columns, True)
    lat0, lon0 = float(df["Latitude"][0]), float(df["Longitude"][0])
    check("首点纬度 23.15442(容差 1e-4)", f"{lat0:.5f}", "23.15442",
          abs(lat0 - 23.15442) < 1e-4)
    check("首点经度 113.38835(容差 1e-4)", f"{lon0:.5f}", "113.38835",
          abs(lon0 - 113.38835) < 1e-4)
except ImportError:
    notes.append("未安装 pandas, 跳过实测数据校验")

# ---------------- 跨图层投影一致性 ----------------
section("跨图层投影一致性(各图层是否落在同一坐标系)")
R_E = 6378137.0
lat0r = np.radians(23.15442)
lon0r = np.radians(113.38835)


def ll2xy(lon, lat):
    return (R_E * np.cos(lat0r) * (np.radians(lon) - lon0r),
            R_E * (np.radians(lat) - lat0r))


bx, by = ll2xy(b["b_lon"], b["b_lat"])
ox, oy = ll2xy(o["lines_lon"], o["lines_lat"])
# DEM 像元中心 -> XY
z, x0, y0 = int(d["z"]), int(d["x0"]), int(d["y0"])
H, W = E.shape
nn = 2 ** z
lon_px = (x0 + (np.arange(W) + 0.5) / 256.0) / nn * 360.0 - 180.0
ty = (y0 + (np.arange(H) + 0.5) / 256.0) / nn
lat_px = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * ty))))
dxs, _ = ll2xy(lon_px, np.full(W, 23.15442))
_, dys = ll2xy(np.full(H, 113.38835), lat_px)
print(f"        DEM   X {dxs.min():8.0f} ~ {dxs.max():8.0f}   "
      f"Y {dys.min():8.0f} ~ {dys.max():8.0f}")
print(f"        建筑  X {bx.min():8.0f} ~ {bx.max():8.0f}   "
      f"Y {by.min():8.0f} ~ {by.max():8.0f}")
print(f"        障碍  X {ox.min():8.0f} ~ {ox.max():8.0f}   "
      f"Y {oy.min():8.0f} ~ {oy.max():8.0f}")

# 真正要保证的是: 线位 + 最宽搜索走廊带完全落在 DEM 内(否则采样会越界)。
# 建筑/障碍物延伸到 DEM 之外属正常: Overpass 按 bbox 返回相交要素, 且 DEM 栅格
# 为整数张瓦片, 南边比标称 bbox 短约 555 m(见下方提示), 与模型无关。
try:
    import pandas as pd
    df = pd.read_excel(os.path.join(MEA, "数据.xlsx"))
    AY = R_E * (np.radians(df["Latitude"].to_numpy()) - lat0r)
    AX = R_E * np.cos(lat0r) * (np.radians(df["Longitude"].to_numpy()) - lon0r)
    print(f"        线位  X {AX.min():8.0f} ~ {AX.max():8.0f}   "
          f"Y {AY.min():8.0f} ~ {AY.max():8.0f}")
    for c in (500, 2500):
        ok = (AX.min() - c >= dxs.min() and AX.max() + c <= dxs.max()
              and AY.min() - c >= dys.min() and AY.max() + c <= dys.max())
        check(f"线位 + 走廊带 ±{c} m 落在 DEM 覆盖内", ok, True)
except ImportError:
    notes.append("未安装 pandas, 跳过线位/走廊带落域校验")

# DEM 南边覆盖比标称 bbox 短的量, 作为已知事实显式报告(非失败项)
lat_min_dem = 23.15442 + np.degrees(dys.min() / R_E)
gap_m = (lat_min_dem - 23.095) * np.pi / 180.0 * R_E
print(f"        [已知] DEM 实际南界 lat {lat_min_dem:.5f}, 比标称 bbox 23.095 "
      f"短 {gap_m:.0f} m (栅格 1280 行 = 5 张 z14 瓦片, 覆盖 bbox 需 6 张)")
print(f"        [已知] 该缺口位于线位以南约 6 km, 不影响采样与优化, "
      f"仅使最南端建筑落在 DEM 底图之外")

# ---------------- 汇总 ----------------
print("\n" + "=" * 58)
for n in notes:
    print(f"[提示] {n}")
if fails:
    print(f"[结论] {len(fails)} 项不符: {', '.join(fails)}")
    sys.exit(1)
print("[结论] 全部检查通过, 数据与文档 §5 速查表一致")
sys.exit(0)
