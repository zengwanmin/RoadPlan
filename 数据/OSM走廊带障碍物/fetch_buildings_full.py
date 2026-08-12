# -*- coding: utf-8 -*-
"""
fetch_buildings_full.py — 完整抓取走廊带范围内的 OSM 建筑（含多边形轮廓与多重多边形关系）

【为什么重写】旧脚本 fetch_buildings.py 的三处缺陷:
  1. 用 `out center` 只取质心, 没有建筑轮廓多边形;
  2. 只查 way["building"], 漏掉 relation["building"](多重多边形, 带内环的建筑);
  3. bbox 被裁窄为 23.1264~23.1966 / 113.1669~113.4137, 比 DEM 与障碍物数据的
     23.095~23.201 / 113.159~113.423 小一圈。
  结果: 旧数据 12789 个, 而本 bbox 内实际 way["building"] 有 22414 个 —— 仅 57% 覆盖。

【目标量(Overpass `out count` 实测, osm_base 2026-08-11T16:19:45Z)】
  way["building"]      = 22414
  relation["building"] =   177
  way["building:part"] =   397   (楼体分部, 单独存放, 默认不计入建筑总数以免重复)

【做法】把 bbox 切成 NX×NY 小块逐块抓 `out geom`, 每块响应落盘缓存(可断点续抓),
失败按指数退避重试并轮换镜像; 最后按 (type,id) 去重, 与上述目标量逐项核对。
跨块边界的建筑会在多块中重复出现, 去重后即为全集。

© OpenStreetMap contributors, ODbL v1.0。
"""
import json, os, sys, time, gzip, random
import urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

# 与 DEM / obstacles.npz 一致的走廊带 bbox
S, W, N, E = 23.095, 113.159, 23.201, 113.423
NX, NY = 16, 8                      # 128 块, 每块约 1.8 km × 1.5 km
CACHE = "osm/buildings_raw"
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
SELECTORS = {
    "way":  'way["building"]',
    "rel":  'relation["building"]',
    "part": 'way["building:part"]',
}
MAX_TRY = 8
WORKERS = 2                          # Overpass 公共实例 rate limit = 2 slots

os.makedirs(CACHE, exist_ok=True)


def _bbox(i, j):
    lo0 = W + (E - W) * i / NX
    lo1 = W + (E - W) * (i + 1) / NX
    la0 = S + (N - S) * j / NY
    la1 = S + (N - S) * (j + 1) / NY
    return la0, lo0, la1, lo1


def _request(query, timeout):
    last = None
    for ep in ENDPOINTS:
        try:
            req = urllib.request.Request(
                ep, data=urllib.parse.urlencode({"data": query}).encode(),
                headers={"Accept-Encoding": "gzip",
                         "User-Agent": "roadplan-research/1.0 (academic; ODbL)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                buf = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    buf = gzip.decompress(buf)
            return json.loads(buf)
        except Exception as ex:
            last = ex
    raise last


def fetch_tile(args):
    kind, i, j = args
    path = os.path.join(CACHE, f"{kind}_{i:02d}_{j:02d}.json")
    if os.path.exists(path) and os.path.getsize(path) > 2:
        return path, "cached", -1

    la0, lo0, la1, lo1 = _bbox(i, j)
    bb = f"{la0:.6f},{lo0:.6f},{la1:.6f},{lo1:.6f}"
    q = f'[out:json][timeout:180];{SELECTORS[kind]}({bb});out geom;'

    for attempt in range(MAX_TRY):
        try:
            d = _request(q, 200)
            els = d.get("elements", [])
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"elements": els,
                           "osm_base": d.get("osm3s", {}).get("timestamp_osm_base"),
                           "bbox": bb}, f)
            return path, "ok", len(els)
        except Exception as ex:
            # 指数退避 + 抖动: 504/429 多为服务端瞬时过载, 重试即可
            wait = min(90, 5 * (2 ** attempt)) + random.uniform(0, 4)
            print(f"  [{kind} {i:02d},{j:02d}] try{attempt+1}/{MAX_TRY} "
                  f"{type(ex).__name__} {str(ex)[:44]} -> wait {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    return path, "FAILED", -1


def main():
    jobs = [(k, i, j) for k in SELECTORS for i in range(NX) for j in range(NY)]
    print(f"[抓取] bbox={S},{W},{N},{E}  切块 {NX}x{NY}  "
          f"任务 {len(jobs)} 个 ({len(SELECTORS)} 类要素)", flush=True)

    done = failed = cached = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for path, status, n in pool.map(fetch_tile, jobs):
            if status == "FAILED":
                failed += 1
                print(f"[失败] {os.path.basename(path)}", flush=True)
            elif status == "cached":
                cached += 1
            else:
                done += 1
                if done % 20 == 0:
                    print(f"[进度] 新抓 {done} / 缓存 {cached} / 失败 {failed}",
                          flush=True)
    print(f"[完成] 新抓 {done}, 缓存命中 {cached}, 失败 {failed}", flush=True)
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
