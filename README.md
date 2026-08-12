# RoadPlan — `data` 分支：数据与数据处理

本分支是**数据专用分支**，与代码分支（`main` / `opt` / `optlab-finalexp`）相互独立，
自成一体：把广州北环高速平纵联合线形优化用到的**全部外部数据、抓取与处理脚本、
来源说明与完整性复核**集中在 `data/` 一个文件夹内。

仓库：https://github.com/zengwanmin/RoadPlan　·　分支：`data`

---

## 先读这一份

**→ [`data/docs/数据来源与处理说明.md`](data/docs/数据来源与处理说明.md)**

这份文档回答三个问题：**数据从哪来**、**凭什么可信**、**在模型里被用在哪一步**，
并给出从零复现每个派生文件的命令。内容包括来源网址、许可与署名要求、
使用广泛程度、可信度评估（含必须承认的局限）、逐步处理链、模型接入点。

## 一键复核

```bash
cd data && python3 verify/verify_data.py
```

不联网，只校验本地文件的自洽性与关键指标（栅格尺寸、要素计数、环闭合性、
快照一致性、投影落域等），逐项对比文档 §5 速查表。通过则退出码 0。

---

## 数据一览

| 类别 | 主用文件 | 规模 | 是否进入目标函数 |
|---|---|---|---|
| 实测中线 | `data/measured/数据.xlsx` | 14018 点 | **是**（基准线位、地面高程、曲率） |
| DEM 高程 | `data/dem/走廊带DEM_z14_ext.npz` | 1280×3072，约 8.78 m/像元 | **是**（新线位地面线） |
| DEM 准天然地面 | `data/dem/走廊带DEM_z14_ext_natural.npz` | 同上 + 生态掩膜 | **是**（统一土方计费口径、生态强制隧道区） |
| OSM 障碍物 | `data/osm/obstacles/obstacles.npz` | 6002 条折线 | **是**（立交带空间锚定） |
| OSM 建筑 | `data/osm/buildings/buildings_full.npz` | 22590 个轮廓，171039 顶点 | **否**（仅可视化与走廊带论证） |

## 目录结构

```
data/
├── docs/数据来源与处理说明.md        主文档
├── dem/          DEM 数据 + 下载/重建/采样脚本
├── osm/
│   ├── obstacles/   道路/铁路/水系 + 查询语句 + 原始返回 + 处理脚本
│   └── buildings/   建筑完整轮廓集 + 抓取/续抓/处理脚本 + 384 块原始响应
├── measured/     实测中线与现状桥隧统计 + 加载投影脚本
└── verify/       一键完整性复核
```

## 数据可信度速览

| 指标 | 结果 |
|---|---|
| DEM vs GPS 实测高程 | 相关系数 **0.915**，中位偏差 **+1.57 m**（已以常数消除） |
| 生态区双指标校准 | 面积 19.2 km²（实际约 21）、现状穿越 1.28 km（实际 1.35） |
| OSM 立交带验证 | 7 座立交桩号带 **7/7 全部命中**横向障碍物 |
| OSM 建筑计数核对 | way 22414/22414、relation 177/177、part 397/397 —— **逐项一致** |
| 建筑快照一致性 | 384 块 `osm_base` 全部同日（已修正镜像旧快照问题） |

局限已在文档 §7 集中列出（DEM 过采样、准天然地面为插值近似、OSM 建筑受数据源
完备度限制、DEM 南界比标称 bbox 短 555 m 等），便于论文如实声明。

## 许可与署名

- **OSM 派生数据**（`data/osm/`）：© OpenStreetMap contributors，依 **ODbL v1.0** 提供。
  公开使用须署名，衍生数据库须以同等条款共享。https://www.openstreetmap.org/copyright
- **DEM 派生数据**（`data/dem/`）：源自 AWS Terrain Tiles（terrarium），
  底数为 SRTM、GMTED2010 等公开数据集。https://registry.opendata.aws/terrain-tiles/
- **实测数据**（`data/measured/`）：项目内部数据。
