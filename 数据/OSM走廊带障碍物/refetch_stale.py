# -*- coding: utf-8 -*-
"""
refetch_stale.py — 重抓快照过期的分块, 保证全集来自同一时刻的 OSM 快照

【问题】首轮抓取里镜像 overpass.kumi.systems 返回的是它自己的旧快照, 27 个分块的
`timestamp_osm_base` 最早到 2026-05-06(比目标快照旧近 3 个月)。混用不同时刻的快照会
让去重后的总数与 `out count` 目标量对不上(实测 way 少 10 个), 且数据自身不自洽。

【做法】删除这些分块的缓存, 只用主端点 overpass-api.de 重抓, 使全部分块的 osm_base
落在同一天; 重抓后再跑 process_buildings.py 复核数量。
© OpenStreetMap contributors, ODbL v1.0。
"""
import json, os, time, gzip, random
import urllib.request, urllib.parse

import fetch_buildings_full as F

EP = "https://overpass-api.de/api/interpreter"      # 只用主端点, 不用镜像
FRESH_DAY = "2026-08-11"


def _request(query, timeout=200):
    req = urllib.request.Request(
        EP, data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"Accept-Encoding": "gzip",
                 "User-Agent": "roadplan-research/1.0 (academic; ODbL)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            buf = gzip.decompress(buf)
    return json.loads(buf)


def refetch(name):
    kind, i, j = name[:-5].split("_")
    i, j = int(i), int(j)
    la0, lo0, la1, lo1 = F._bbox(i, j)
    bb = f"{la0:.6f},{lo0:.6f},{la1:.6f},{lo1:.6f}"
    q = f'[out:json][timeout:180];{F.SELECTORS[kind]}({bb});out geom;'
    path = os.path.join(F.CACHE, name)

    for attempt in range(10):
        try:
            d = _request(q)
            base = d.get("osm3s", {}).get("timestamp_osm_base", "")
            if not base.startswith(FRESH_DAY):
                raise RuntimeError(f"仍是旧快照 {base}")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"elements": d.get("elements", []),
                           "osm_base": base, "bbox": bb}, f)
            return True, base, len(d.get("elements", []))
        except Exception as ex:
            wait = min(90, 5 * (2 ** attempt)) + random.uniform(0, 4)
            print(f"  [{name}] try{attempt+1}/10 {type(ex).__name__} "
                  f"{str(ex)[:44]} -> wait {wait:.0f}s", flush=True)
            time.sleep(wait)
    return False, "", -1


def main():
    names = [n.strip() for n in open("stale_tiles.txt") if n.strip()]
    print(f"[重抓] {len(names)} 个快照过期分块, 仅用主端点", flush=True)
    bad = 0
    for k, name in enumerate(names, 1):
        ok, base, n = refetch(name)
        if ok:
            print(f"[{k}/{len(names)}] {name}  osm_base={base}  els={n}", flush=True)
        else:
            bad += 1
            print(f"[{k}/{len(names)}] {name}  失败", flush=True)
        time.sleep(2)
    print(f"[完成] 失败 {bad}", flush=True)


if __name__ == "__main__":
    main()
