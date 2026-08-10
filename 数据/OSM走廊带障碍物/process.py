# -*- coding: utf-8 -*-
"""处理 Overpass 原始数据 -> obstacles.npz，并做立交带验证。
仅依赖 pandas / numpy / scipy / json / 标准库。
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = "/root/roadplan/RoadPlan_remote/数据/OSM走廊带障碍物"
R_EARTH = 6378137.0

# ---------- 1. 读取原始 Overpass 数据 ----------
with open(f"{BASE}/raw_overpass.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

ways = []
for el in raw["elements"]:
    if el.get("type") != "way" or "geometry" not in el:
        continue
    tags = el.get("tags", {})
    if "highway" in tags:
        kind, fclass = "road", tags["highway"]
    elif tags.get("railway") == "rail":
        kind, fclass = "rail", "rail"
    elif "waterway" in tags:
        kind, fclass = "water", tags["waterway"]
    else:
        continue
    lon = np.array([g["lon"] for g in el["geometry"]], dtype=np.float64)
    lat = np.array([g["lat"] for g in el["geometry"]], dtype=np.float64)
    name = tags.get("name", "")
    ways.append(dict(osm_id=el["id"], kind=kind, fclass=fclass, name=name,
                     lon=lon, lat=lat))

print(f"原始 way 数(带几何): {len(ways)}")

# ---------- 2. 读取实测中线 ----------
df = pd.read_excel("/root/roadplan/RoadPlan_remote/数据/数据.xlsx")
cl_lat = df["Latitude"].to_numpy(dtype=np.float64)
cl_lon = df["Longitude"].to_numpy(dtype=np.float64)
print(f"实测中线点数: {len(cl_lat)}")

lat0 = np.deg2rad(cl_lat[0])
KX = R_EARTH * np.cos(lat0)   # X = KX * rad(lon)
KY = R_EARTH                  # Y = KY * rad(lat)

def to_xy(lon, lat):
    return np.column_stack([KX * np.deg2rad(lon), KY * np.deg2rad(lat)])

cl_xy = to_xy(cl_lon, cl_lat)
cl_tree = cKDTree(cl_xy)

# 中线累计里程
seg = np.hypot(np.diff(cl_xy[:, 0]), np.diff(cl_xy[:, 1]))
chain = np.concatenate([[0.0], np.cumsum(seg)])
print(f"中线全长: {chain[-1]:.1f} m")

# ---------- 3. 剔除北环高速自身(motorway/trunk, >60%顶点距中线<150m) ----------
kept, removed = [], []
for w in ways:
    if w["kind"] == "road" and w["fclass"] in ("motorway", "trunk"):
        d, _ = cl_tree.query(to_xy(w["lon"], w["lat"]))
        frac = float(np.mean(d < 150.0))
        if frac > 0.60:
            removed.append((w, frac))
            continue
    kept.append(w)

print(f"剔除本线/伴行线 way 数: {len(removed)}")
rm_names = {}
for w, frac in removed:
    rm_names.setdefault(w["name"] or "(无名)", 0)
    rm_names[w["name"] or "(无名)"] += 1
for n, c in sorted(rm_names.items(), key=lambda x: -x[1]):
    print(f"  剔除: {n} x{c}")

# ---------- 4. 保存 npz ----------
lines_lon = np.concatenate([w["lon"] for w in kept])
lines_lat = np.concatenate([w["lat"] for w in kept])
counts = np.array([len(w["lon"]) for w in kept], dtype=np.int64)
offsets = np.concatenate([[0], np.cumsum(counts)])  # 第 i 条: [offsets[i], offsets[i+1])
kind = np.array([w["kind"] for w in kept])
fclass = np.array([w["fclass"] for w in kept])
name = np.array([w["name"] for w in kept])
osm_id = np.array([w["osm_id"] for w in kept], dtype=np.int64)

np.savez_compressed(
    f"{BASE}/obstacles.npz",
    lines_lon=lines_lon, lines_lat=lines_lat, offsets=offsets,
    kind=kind, highway_class=fclass, name=name, osm_id=osm_id,
)
print(f"保留折线: {len(kept)} 条, 顶点总数: {len(lines_lon)}")

# 统计
print("\n按 kind 统计:")
for k in ("road", "rail", "water"):
    print(f"  {k}: {int(np.sum(kind == k))}")
print("按 highway_class/要素类型 统计:")
for fc in sorted(set(fclass.tolist())):
    print(f"  {fc}: {int(np.sum(fclass == fc))}")

# ---------- 5. 立交带验证 ----------
BANDS = [
    ("沙贝", 200, 1000), ("广清", 2400, 4600), ("广花和三元里", 6030, 7630),
    ("广园路", 8770, 9670), ("沙河", 14210, 15010), ("岑村", 17800, 18700),
    ("科韵路", 20300, 20900),
]

# 障碍物折线加密到 <=30m 间距后建 KDTree
dense_pts, dense_widx = [], []
for i, w in enumerate(kept):
    xy = to_xy(w["lon"], w["lat"])
    for j in range(len(xy) - 1):
        p, q = xy[j], xy[j + 1]
        L = np.hypot(*(q - p))
        n = max(int(np.ceil(L / 30.0)), 1)
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        dense_pts.append(p + t * (q - p))
        dense_widx.append(np.full(n, i, dtype=np.int64))
    dense_pts.append(xy[-1:])
    dense_widx.append(np.array([i], dtype=np.int64))
dense_pts = np.vstack(dense_pts)
dense_widx = np.concatenate(dense_widx)
obs_tree = cKDTree(dense_pts)

print("\n立交带验证 (300 m 半径):")
hits_total = 0
for nm, s0, s1 in BANDS:
    m = (chain >= s0) & (chain <= s1)
    pts = cl_xy[m]
    idx_lists = obs_tree.query_ball_point(pts, 300.0)
    widx = sorted({int(dense_widx[i]) for lst in idx_lists for i in lst})
    if widx:
        hits_total += 1
        feats = {}
        for i in widx:
            w = kept[i]
            key = (w["kind"], w["fclass"], w["name"] or "(无名)")
            feats[key] = feats.get(key, 0) + 1
        desc = "; ".join(f"{k[2]}[{k[0]}/{k[1]}]x{v}"
                         for k, v in sorted(feats.items(), key=lambda x: -x[1]))
        print(f"  [命中] {nm} ({s0}-{s1}m): {len(widx)} 条 -> {desc}")
    else:
        print(f"  [未命中] {nm} ({s0}-{s1}m)")
print(f"\n命中率: {hits_total}/7")
