# -*- coding: utf-8 -*-
"""
process_buildings.py — 把 fetch_buildings_full.py 抓下的分块原始响应处理成建筑轮廓数据集

【输入】osm/buildings_raw/{way,rel,part}_II_JJ.json  (Overpass `out geom` 原始响应)
【输出】osm/buildings_full.npz

【处理步骤】
  1. 按 (type, id) 去重 —— 跨块边界的建筑会在相邻块中重复返回;
  2. way 直接取 geometry 作外环; relation(多重多边形)按成员 role 分 outer/inner,
     并把首尾相接的成员线段缝合成闭合环(OSM 多重多边形的外环常被拆成多段 way);
  3. 过滤退化几何(顶点 < 4 或首尾不闭合且无法闭合的环);
  4. 计算每个建筑的质心与面积(等积投影下的鞋带公式), 供密度图与统计用;
  5. 与 Overpass `out count` 的目标量逐项核对, 不一致则在报告中显式列出差额。

【输出字段】
  poly_lon / poly_lat : 所有环的顶点坐标(展平)
  poly_off            : 环的分界下标, 长度 = 环数 + 1
  poly_ring           : 每个环的类型, 'outer' / 'inner'
  poly_bid            : 每个环所属建筑在 b_* 数组中的下标
  b_id / b_type       : 建筑的 OSM id 与类型('way'/'rel')
  b_lon / b_lat       : 建筑质心
  b_area_m2           : 建筑占地面积(m², 已扣除内环)
  b_building          : building 标签值(yes/house/apartments/...)
  b_levels / b_height : 层数与高度(缺失为 nan)
  b_name              : 名称(缺失为空串)
  part_*              : building:part 要素同结构单独存放
© OpenStreetMap contributors, ODbL v1.0。
"""
import json, os, glob, math
import numpy as np

CACHE = "osm/buildings_raw"
OUT = "osm/buildings_full.npz"
# Overpass `out count` 实测目标量 (osm_base 2026-08-11T16:19:45Z)
TARGET = {"way": 22414, "rel": 177, "part": 397}

R_E = 6378137.0
LAT0 = math.radians(23.148)          # 走廊带中心纬度, 用于等积近似


def _load_raw():
    """读取所有分块响应, 按 (kind, id) 去重。"""
    got = {"way": {}, "rel": {}, "part": {}}
    bases, files = set(), 0
    for path in sorted(glob.glob(os.path.join(CACHE, "*.json"))):
        kind = os.path.basename(path).split("_")[0]
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("osm_base"):
            bases.add(d["osm_base"])
        files += 1
        for el in d.get("elements", []):
            got[kind][el["id"]] = el
    return got, bases, files


def _rings_from_way(el):
    g = el.get("geometry") or []
    pts = [(p["lon"], p["lat"]) for p in g if p]
    return [("outer", pts)] if len(pts) >= 4 else []


def _stitch(segs):
    """把若干首尾相接的线段缝合成闭合环(OSM 多重多边形外环常被拆成多段)。"""
    segs = [list(s) for s in segs if len(s) >= 2]
    rings = []
    while segs:
        cur = segs.pop(0)
        changed = True
        while changed and (cur[0] != cur[-1]):
            changed = False
            for k, s in enumerate(segs):
                if s[0] == cur[-1]:
                    cur += s[1:]; segs.pop(k); changed = True; break
                if s[-1] == cur[-1]:
                    cur += s[::-1][1:]; segs.pop(k); changed = True; break
                if s[-1] == cur[0]:
                    cur = s[:-1] + cur; segs.pop(k); changed = True; break
                if s[0] == cur[0]:
                    cur = s[::-1][:-1] + cur; segs.pop(k); changed = True; break
        if len(cur) >= 4 and cur[0] == cur[-1]:
            rings.append(cur)
    return rings


def _rings_from_rel(el):
    """多重多边形: 按 role 收集成员线段并缝合成环。"""
    out = []
    for role in ("outer", "inner"):
        segs = []
        for m in el.get("members", []):
            if m.get("type") != "way":
                continue
            r = m.get("role") or "outer"      # role 缺省视为 outer
            if r != role:
                continue
            g = m.get("geometry") or []
            pts = [(p["lon"], p["lat"]) for p in g if p]
            if len(pts) >= 2:
                segs.append(pts)
        for ring in _stitch(segs):
            out.append((role, ring))
    return out


def _area_m2(pts):
    """等积近似下的鞋带公式面积(m²)。"""
    x = np.array([p[0] for p in pts]) * (math.pi / 180.0) * R_E * math.cos(LAT0)
    y = np.array([p[1] for p in pts]) * (math.pi / 180.0) * R_E
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def _centroid(pts):
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return float(x[:-1].mean()), float(y[:-1].mean())


def _num(tags, *keys):
    for k in keys:
        v = tags.get(k)
        if v is None:
            continue
        try:
            return float(str(v).split()[0].replace(",", "."))
        except ValueError:
            continue
    return np.nan


def build(kinds, label):
    P_lon, P_lat, P_off, P_ring, P_bid = [], [], [0], [], []
    b_id, b_type, b_lon, b_lat, b_area = [], [], [], [], []
    b_bld, b_lev, b_hgt, b_name = [], [], [], []
    degenerate = 0

    for kind, elems in kinds:
        for oid, el in sorted(elems.items()):
            rings = _rings_from_way(el) if el["type"] == "way" else _rings_from_rel(el)
            rings = [(r, p) for r, p in rings if len(p) >= 4]
            if not rings:
                degenerate += 1
                continue
            outer = [p for r, p in rings if r == "outer"]
            if not outer:
                degenerate += 1
                continue
            bi = len(b_id)
            area = sum(_area_m2(p) for p in outer) \
                - sum(_area_m2(p) for r, p in rings if r == "inner")
            cx, cy = _centroid(max(outer, key=len))
            tags = el.get("tags") or {}
            b_id.append(oid); b_type.append(kind)
            b_lon.append(cx); b_lat.append(cy); b_area.append(max(area, 0.0))
            b_bld.append(str(tags.get("building") or tags.get("building:part") or "yes"))
            b_lev.append(_num(tags, "building:levels", "levels"))
            b_hgt.append(_num(tags, "height", "building:height"))
            b_name.append(str(tags.get("name") or ""))
            for r, p in rings:
                P_lon.extend(q[0] for q in p); P_lat.extend(q[1] for q in p)
                P_off.append(len(P_lon)); P_ring.append(r); P_bid.append(bi)

    print(f"  [{label}] 建筑 {len(b_id)} 个, 环 {len(P_ring)} 个, "
          f"顶点 {len(P_lon)} 个, 退化剔除 {degenerate} 个")
    return dict(
        poly_lon=np.array(P_lon), poly_lat=np.array(P_lat),
        poly_off=np.array(P_off, np.int64), poly_ring=np.array(P_ring),
        poly_bid=np.array(P_bid, np.int64),
        b_id=np.array(b_id, np.int64), b_type=np.array(b_type),
        b_lon=np.array(b_lon), b_lat=np.array(b_lat),
        b_area_m2=np.array(b_area), b_building=np.array(b_bld),
        b_levels=np.array(b_lev), b_height=np.array(b_hgt),
        b_name=np.array(b_name)), degenerate


def main():
    got, bases, files = _load_raw()
    print(f"[读取] {files} 个分块响应, osm_base={sorted(bases)}")
    print(f"[去重] way={len(got['way'])} rel={len(got['rel'])} part={len(got['part'])}")

    print("[核对] 去重后数量 vs Overpass out count 目标量:")
    ok = True
    for k in ("way", "rel", "part"):
        n, t = len(got[k]), TARGET[k]
        flag = "一致" if n == t else f"差 {n - t:+d}"
        if n != t:
            ok = False
        print(f"  {k:5s} 实得 {n:6d}  目标 {t:6d}  {flag}")

    main_pack, dg1 = build([("way", got["way"]), ("rel", got["rel"])], "buildings")
    part_pack, dg2 = build([("part", got["part"])], "building:part")
    pack = dict(main_pack)
    pack.update({"part_" + k: v for k, v in part_pack.items()})
    pack["osm_base"] = np.array(sorted(bases))
    pack["bbox"] = np.array([23.095, 113.159, 23.201, 113.423])
    np.savez_compressed(OUT, **pack)

    a = main_pack["b_area_m2"]
    print(f"[面积] 总占地 {a.sum()/1e6:.3f} km²  中位 {np.median(a):.0f} m²  "
          f"最大 {a.max():.0f} m²")
    types, cnt = np.unique(main_pack["b_building"], return_counts=True)
    top = sorted(zip(cnt, types), reverse=True)[:8]
    print("[类型] " + ", ".join(f"{t}={c}" for c, t in top))
    print(f"[保存] {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")
    print("[结论] " + ("与目标量完全一致, 数据完整" if ok else "存在差额, 见上表"))


if __name__ == "__main__":
    main()
