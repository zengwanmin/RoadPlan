# -*- coding: utf-8 -*-
"""fetch_buildings.py — 分块从 Overpass 抓走廊带建筑质心(大范围 504, 按经度切块)。
© OpenStreetMap contributors (ODbL).
"""
import json, time, urllib.request, urllib.parse, gzip
import numpy as np

S, W, N, E = 23.1264, 113.1669, 23.1966, 113.4137
NCHUNK = 10
EDGES = np.linspace(W, E, NCHUNK+1)
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]

def fetch(bbox):
    q = f'[out:json][timeout:120];(way["building"]({bbox});out center;'
    q = f'[out:json][timeout:120];way["building"]({bbox});out center;'
    for ep in ENDPOINTS:
        try:
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({"data": q}).encode(),
                    headers={"Accept-Encoding":"gzip","User-Agent":"roadplan-research/1.0"})
            with urllib.request.urlopen(req, timeout=150) as r:
                buf = r.read()
                if r.headers.get("Content-Encoding")=="gzip": buf=gzip.decompress(buf)
            return json.loads(buf)["elements"]
        except Exception as ex:
            print("   ", ep.split("//")[1][:20], type(ex).__name__, str(ex)[:60], flush=True)
    return None

lon, lat = [], []
for i in range(NCHUNK):
    bbox = f"{S},{EDGES[i]:.5f},{N},{EDGES[i+1]:.5f}"
    for attempt in range(3):
        els = fetch(bbox)
        if els is not None: break
        time.sleep(8)
    if els is None:
        print(f"[chunk {i+1}/{NCHUNK}] FAILED", flush=True); continue
    c0 = len(lon)
    for el in els:
        cc = el.get("center")
        if cc: lon.append(cc["lon"]); lat.append(cc["lat"])
    print(f"[chunk {i+1}/{NCHUNK}] lon<{EDGES[i+1]:.3f}  +{len(lon)-c0} buildings (total {len(lon)})", flush=True)
    time.sleep(3)

lon=np.array(lon); lat=np.array(lat)
np.savez_compressed("osm/buildings.npz", lon=lon, lat=lat)
print("saved buildings=%d lon %.4f~%.4f lat %.4f~%.4f"%(len(lon),lon.min(),lon.max(),lat.min(),lat.max()))
