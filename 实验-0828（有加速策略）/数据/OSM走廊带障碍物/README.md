# OSM 走廊带障碍物数据（广州北环高速 浔峰洲—黄村）

用于高速公路线形优化中"横向障碍物"交叉判定（跨线桥自动触发）的矢量数据。

## 1. 数据来源

- 数据源：OpenStreetMap，经 Overpass API 下载
- 端点：`https://overpass-api.de/api/interpreter`（主端点；首次请求遇服务器繁忙、二次因传输超时中断，镜像 `overpass.kumi.systems` 亦繁忙，最终主端点加 gzip 压缩成功）
- 查询发起时间：2026-08-09 15:04:28 UTC；返回数据 `timestamp_osm_base`：2026-08-09T15:03:11Z
- 服务器版本：Overpass API 0.7.62.11
- bbox（south,west,north,east）：`23.095, 113.159, 23.201, 113.423`
- 原始返回：`raw_overpass.json`（5.69 MB，6132 个 way，全部带 `geometry` 节点坐标）

### 完整 Overpass 查询语句（见 `query.overpass`）

```
[out:json][timeout:60][bbox:23.095,113.159,23.201,113.423];
(
  way[highway~"^(motorway|trunk|primary|secondary)$"];
  way[railway=rail];
  way[waterway~"^(river|canal)$"];
);
out geom;
```

### 许可声明（ODbL）

本目录数据 © OpenStreetMap 贡献者，依据 Open Database License (ODbL) v1.0
（https://opendatabase.org/licenses/odbl/ 与 https://www.openstreetmap.org/copyright）提供。
任何公开使用须署名 "© OpenStreetMap contributors"；基于本数据的衍生数据库须以 ODbL 同等条款共享。

## 2. 处理与过滤规则（脚本 `process.py`，仅依赖 pandas/numpy/scipy/json）

1. 逐 way 提取 `geometry` 为经纬度折线；按标签分类：
   - `kind=road`：highway ∈ {motorway, trunk, primary, secondary}
   - `kind=rail`：railway=rail
   - `kind=water`：waterway ∈ {river, canal}
2. **剔除北环高速自身/伴行线**：读取实测中线 `/root/roadplan/RoadPlan_remote/数据/数据.xlsx`
   （14018 点，沿线全长 22463.9 m）。局部平面近似：
   `X = R·cos(lat0)·Δlon(rad), Y = R·Δlat(rad)`，R=6378137 m，lat0 取中线首点纬度 23.15442°。
   对每条 motorway/trunk 折线，用 cKDTree 求各顶点到中线的最近距离；
   若 **>60% 顶点距离 < 150 m**，判为本线/伴行线并剔除。
3. 其余全部保留，存入 `obstacles.npz`。

### 剔除结果：共 130 条 way

| 名称 | 条数 | | 名称 | 条数 |
|---|---|---|---|---|
| 广州环城高速 | 74 | | 广园中路辅路 | 6 |
| (无名) | 14 | | 广佛高速 | 4 |
| 广园中路 | 12 | | 白云大道南 | 2 |
| 燕岭路 | 8 | | 华南快速 / 科韵北路 / 机场高速 / 广州大道北 | 各 1 |
| 机场路 | 6 | | | |

（北环高速在 OSM 中挂名"广州环城高速"；广园中路等与其长距离并行段按规则一并剔除，
其未并行的路段仍保留在数据中，不影响交叉判定。）

### 保留结果：6002 条折线，49497 个顶点

按 kind：road 4807、rail 857、water 338。

按类型：motorway 183、trunk 1069、primary 1727、secondary 1828、rail 857、river 238、canal 100。

## 3. `obstacles.npz` 结构

| 数组 | dtype | 长度 | 说明 |
|---|---|---|---|
| `lines_lon` / `lines_lat` | float64 | 49497 | 全部折线顶点经/纬度（WGS84），扁平拼接 |
| `offsets` | int64 | 6003 | 第 i 条折线的顶点为 `[offsets[i], offsets[i+1])` |
| `kind` | str | 6002 | road / rail / water |
| `highway_class` | str | 6002 | motorway/trunk/primary/secondary/rail/river/canal |
| `name` | str | 6002 | OSM name 标签，可为空串 |
| `osm_id` | int64 | 6002 | OSM way id |

读取示例：

```python
d = np.load("obstacles.npz", allow_pickle=False)
i = 0  # 第 i 条折线
lon = d["lines_lon"][d["offsets"][i]:d["offsets"][i+1]]
lat = d["lines_lat"][d["offsets"][i]:d["offsets"][i+1]]
```

## 4. 验证：7 座现状立交桩号带 300 m 命中检查

方法：沿实测中线累计里程取各桩号带内的中线点；将保留障碍物折线按 ≤30 m 间距加密采样建
cKDTree，检查带内任一中线点 300 m 半径内是否存在障碍物。**结果 7/7 全部命中。**

| 桩号带 (m) | 名称 | 结果 | 命中要素（名称[kind/类型]×way 数） |
|---|---|---|---|
| 200–1000 | 沙贝 | 命中(15) | 健明六路[road/secondary]×6、广州环城高速[road/motorway]×2、广佛高速[road/motorway]×2、风庄涌[water/river]×2、广州东环城际线[rail/rail]×2、车陂涌[water/canal]×1 |
| 2400–4600 | 广清 | 命中(6) | 华南快速[road/motorway]×5、车陂涌[water/canal]×1 |
| 6030–7630 | 广花和三元里 | 命中(15) | 元岗横路[road/secondary]×8、沙太南路[road/primary]×5、燕岭路[road/trunk]×2 |
| 8770–9670 | 广园路 | 命中(7) | 广州大道北[road/trunk]×5、广州大道北[road/primary]×1、南蛇坑涌[water/river]×1 |
| 14210–15010 | 沙河 | 命中(54) | 广园西路[road/primary]×15、广园中路[road/primary]×12、机场路[road/trunk]×9、三元里大道[road/secondary]×7、景泰河[water/canal]×6、机场高速[road/motorway]×3 等 |
| 17800–18700 | 岑村 | 命中(12) | 许广高速[road/motorway]×5、增槎路[road/primary]×5、石井河[water/river]×1、(无名)[water/canal]×1 |
| 20300–20900 | 科韵路 | 命中(7) | 金沙洲路[road/primary]×7 |

**重要说明（里程方向）**：`数据.xlsx` 中线首点位于走廊带东端
（lon 113.38835, lat 23.15442），即累计里程 0 在东侧（黄村端）。由命中要素可见，
桩号带 20300–20900 处命中"金沙洲路"（西端浔峰洲侧），200–1000 处命中"广州东环城际线"
（东端），说明上表给定的立交名称与实际地理位置呈**东→西反序对应**
（各带中点坐标：200–1000 → 113.38261,23.15395；20300–20900 → 113.19867,23.16442）。
无论名称如何对应，7 个桩号带内均存在保留的横向障碍物要素，验证通过，
无需补充 tertiary 级道路。

## 5. 建筑分布数据（完整集 `buildings_full.npz`）

- 用途：可视化图层（图C9 全图层叠加）与走廊带宽度的数据依据。
  **不进入目标函数、不参与任何成本/能耗计算。**
- 抓取脚本：`fetch_buildings_full.py`（分块 + 断点续抓 + 指数退避）
- 处理脚本：`process_buildings.py`（去重、多重多边形缝环、面积/质心、数量核对）
- 快照修复：`refetch_stale.py`（见下文"快照一致性"）

### 数据规模与完整性核验

抓取范围与 DEM、`obstacles.npz` 一致：`23.095, 113.159, 23.201, 113.423`。
把 bbox 切成 16×8 = 128 块逐块请求 `out geom`，三类要素共 384 次查询，**零失败**。
去重后数量与 Overpass `out count` 的权威计数**逐项完全一致**：

| 要素 | 实得 | `out count` 目标 | 结论 |
|---|---|---|---|
| `way["building"]` | 22414 | 22414 | 一致 |
| `relation["building"]` | 177 | 177 | 一致 |
| `way["building:part"]` | 397 | 397 | 一致 |

合计 22414 + 177 = 22591 个建筑要素，其中 1 个几何退化（顶点不足无法成环）被剔除，
最终 **22590 个建筑、22787 个环、171039 个顶点、总占地 24.41 km²**
（中位 519 m²，最大 145133 m²）。类型分布：`yes` 11407、`residential` 3526、
`apartments` 3163、`commercial` 865、`university` 660、`house` 547 等。

### 快照一致性（重要修正）

首轮抓取中镜像 `overpass.kumi.systems` 返回的是它自己的**旧快照**，27 个分块的
`timestamp_osm_base` 最早到 2026-05-06（比目标快照旧近 3 个月）。混用不同时刻的快照
会使总数对不上（当时 way 少 10 个）且数据自身不自洽。已用 `refetch_stale.py`
仅经主端点重抓这 27 块，使**全部 384 块的 `osm_base` 落在同一天**，重新核对后三项计数
全部一致。数据集内保存了 `osm_base` 字段可供复核。

### 与旧文件 `buildings.npz` 的关系（已弃用）

旧文件由 `fetch_buildings.py` 生成，存在三处缺陷，**不应再使用**：

1. 用 `out center` 只存质心，**没有建筑轮廓**；
2. 只查 `way["building"]`，**漏掉 177 个多重多边形关系**；
3. bbox 被裁窄为 `23.1264~23.1966 / 113.1669~113.4137`，比 DEM/障碍物范围小一圈。

结果旧文件只有 12789 个点，**仅为实际 22414 个 way 的 57%**。保留该文件仅为历史可追溯，
新图与新分析一律使用 `buildings_full.npz`。

### 为什么轮廓比质心重要

横向净距应量到建筑**边缘**而非质心：最大建筑占地 14.5 万 m²，其边缘比质心近上百米，
用质心会系统性**高估**可用宽度。按轮廓顶点重算的走廊带可行半宽（两侧均无建筑冲突的
里程占比）：

| 半宽 | ±100 m | ±200 m | ±300 m | ±500 m | ±800 m |
|---|---|---|---|---|---|
| 无冲突里程占比 | 71.6% | 58.0% | 49.3% | 34.6% | 18.2% |

左侧净距中位 550 m、右侧 719 m，最紧断面仅 **10 m**。这组数据支持"走廊带不应全线
取同一半宽"的结论。

### 字段说明

| 字段 | 含义 |
|---|---|
| `poly_lon` / `poly_lat` | 所有环的顶点坐标（展平） |
| `poly_off` | 环的分界下标，长度 = 环数 + 1 |
| `poly_ring` | 每个环的类型，`outer` / `inner` |
| `poly_bid` | 每个环所属建筑在 `b_*` 数组中的下标 |
| `b_id` / `b_type` | 建筑的 OSM id 与类型（`way` / `rel`） |
| `b_lon` / `b_lat` | 建筑质心 |
| `b_area_m2` | 占地面积（m²，已扣除内环） |
| `b_building` | `building` 标签值 |
| `b_levels` / `b_height` | 层数与高度（缺失为 `nan`） |
| `b_name` | 名称（缺失为空串） |
| `part_*` | `building:part` 要素，同结构单独存放 |
| `osm_base` / `bbox` | 快照时刻与抓取范围 |

### 仍需注意的局限

数量已与 Overpass 计数完全一致，即**本 bbox 内 OSM 数据库中的建筑已全部取到**。
但 OSM 本身在国内城区的建筑测绘完备度并非 100%（郊区与村镇尤其），
因此"图上空白"只能说明 **OSM 中无记录**，不等于实地无建筑。此为数据源固有局限，
非本次抓取的缺漏。

## 6. 文件清单

- `raw_overpass.json` — Overpass 原始返回（备查）
- `obstacles.npz` — 处理后的障碍物折线集合（numpy compressed）
- `buildings_full.npz` — **建筑完整轮廓集**（22590 个，已核验，见第 5 节）
- `buildings.npz` — 旧的建筑质心点（仅 57% 覆盖，已弃用，见第 5 节）
- `query.overpass` — Overpass 查询语句原文
- `process.py` — 处理与验证脚本（可复现）
- `fetch_buildings_full.py` — 完整建筑抓取脚本（分块/续抓/退避）
- `process_buildings.py` — 轮廓处理与数量核对脚本
- `refetch_stale.py` — 快照过期分块重抓脚本
- `fetch_buildings.py` — 旧质心抓取脚本（已弃用）
- `README.md` — 本文件
