# OSM 矢量数据

详细说明见 [`../docs/数据来源与处理说明.md`](../docs/数据来源与处理说明.md) 第 2 节。

## 来源与许可

- 数据库：OpenStreetMap　https://www.openstreetmap.org
- 查询接口：Overpass API　https://overpass-api.de/api/interpreter
- 接口状态页：https://overpass-api.de/api/status
- 接口文档：https://wiki.openstreetmap.org/wiki/Overpass_API
- 查询语言：https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL
- 许可：**ODbL v1.0**　https://www.openstreetmap.org/copyright

> **署名要求（ODbL 强制）**：公开使用须署名 **"© OpenStreetMap contributors"**；
> 基于本数据的衍生数据库须以 ODbL 同等条款共享。
> 本目录下所有脚本头部均已写入该署名。

## 两个子集

| 子目录 | 内容 | 是否进入目标函数 |
|---|---|---|
| `obstacles/` | 主干道路 / 铁路 / 河渠，6002 条折线 | **是** —— 立交带空间锚定 → 土方豁免与桥隧计费 |
| `buildings/` | 建筑完整轮廓，22590 个 | **否** —— 仅可视化与走廊带宽度论证 |

## 为什么区别对待

OSM 是众源数据，完备度**按要素类型和地区差异极大**，不能一概而论：

- **道路网**在多数国家已接近完备，主干路（motorway/trunk/primary）尤其可靠
  —— 显著、易制图、且被多个商业用户反复校核。**本项目只用四级主干路**，
  正落在 OSM 最可靠的部分。铁路与河渠同属显著地物。
- **建筑轮廓**是 OSM 最不均衡的图层之一，中国境内完备度整体低于欧美，
  郊区与村镇缺失明显。

因此建筑数据被严格限定为"仅可视化"：其**数量已与 OSM 数据库完全一致**
（逐项核对见 `buildings/README.md`），但"图上空白"只能说明 **OSM 无记录**，
不等于实地无建筑。这是数据源固有局限，不是抓取缺漏。

我们不依赖对 OSM 的笼统信任 —— 每一处具体用法都做了独立验证
（障碍物见立交带 7/7 命中，建筑见三项计数逐项一致）。
