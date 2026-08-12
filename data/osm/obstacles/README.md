# OSM 走廊带障碍物（道路 / 铁路 / 水系）

**用途**：优化线位横穿既有城市道路、铁路、河渠时必须跨越（上跨或下穿），
这决定了立交/桥梁的空间位置与计费。**进入目标函数。**

## 抓取记录（可追溯）

| 项 | 内容 |
|---|---|
| 端点 | `https://overpass-api.de/api/interpreter` |
| 查询时间 | 2026-08-09 15:04:28 UTC |
| 数据快照 `timestamp_osm_base` | 2026-08-09T15:03:11Z |
| Overpass 版本 | 0.7.62.11 |
| bbox | `23.095, 113.159, 23.201, 113.423`（与 DEM 一致） |
| 原始返回 | `raw_overpass.json`（5.43 MB，6132 个 way，全部带 geometry 节点坐标） |

> 抓取过程如实记录：主端点首次遇服务器繁忙、二次因传输超时中断，
> 镜像 `overpass.kumi.systems` 亦繁忙，最终主端点加 gzip 压缩成功。

查询语句原文见 `query.overpass`：

```
[out:json][timeout:60][bbox:23.095,113.159,23.201,113.423];
(
  way[highway~"^(motorway|trunk|primary|secondary)$"];
  way[railway=rail];
  way[waterway~"^(river|canal)$"];
);
out geom;
```

## 产出

`obstacles.npz` —— **6002 条折线**（road 4807 / rail 857 / water 338），字段
`lines_lon / lines_lat / offsets / kind / highway_class / name / osm_id`。

## 验证：立交带命中检查

论文给出既有 7 座功能性立交的桩号带。用本数据**独立检查**"这些桩号带内是否真的存在
横贯走廊带的被交道路" —— **7 个带全部命中**（最少 6 个 way、最多 54 个），
命中要素如广园西路、机场路、机场高速、许广高速、石井河等。
结论：无需补充 tertiary 级道路，数据足以支撑立交锚定。

> **坐标方向注意**：实测中线首点位于走廊带**东端**（lon 113.38835, lat 23.15442），
> 即**累计里程 0 在东侧（黄村端）**。由命中要素可见，桩号带 20300–20900 命中
> "金沙洲路"（西端浔峰洲侧），200–1000 命中"广州东环城际线"（东端），
> 说明论文表中给定的立交**名称**与实际地理位置呈**东→西反序对应**。
> 不影响计算，但读图时必须知道。

## 模型接入点

`objective_joint._ic_bands_from_osm(X, Y, s, t_meas)` 以 OSM 被交道路与实测中线的
**交叉点**空间锚定 7 座立交的弦区间 → `ic` 掩膜 → 落入该带的桩号**土方豁免**
（结构费已计入 CB 常数项），并输出 `L_ic_km`。锚定结果缓存于 `ic_anchor_cache.json`。

**为什么用空间锚定而非固定桩号**：优化改变线位后里程轴会变，
写死桩号会让立交"漂移"到错误的地理位置。锚定到 OSM 实际交叉点才是正确做法。

## 复现

```bash
python3 process.py     # 由 raw_overpass.json 重新生成 obstacles.npz
```

---

© OpenStreetMap contributors, ODbL v1.0
