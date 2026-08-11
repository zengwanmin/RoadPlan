# 原理图（问题定义与抽象）

两个版本，内容一致，用途不同：

| 文件 | 生成方式 | 用途 |
|---|---|---|
| `problem_schematic_aigen.png` | AI 生图（卡通手绘风，1536×1024） | 汇报/答辩讲解页，观感最好 |
| `problem_schematic_cartoon.png` | `draw_schematic.py`（matplotlib xkcd 模式） | 论文正式插图：几何精确、数据可驱动、可随结果重绘 |

图中要点：

- 上格 PLAN VIEW (x, y)：黄色带 = 走廊带可行域 ±W；放大镜 = 最小半径 R≥400 m（表3.2 极限值）；起终点固定
- 下格 PROFILE VIEW (M, H)：M 为沿平面线位的累计里程；坡度锥 |i|≤4%（表4.4）、竖曲线 L≥k_P·A_P（式4.28）；红/绿 = 挖方/填方
- 橙色箭头 = 平纵耦合的根源：平面决定里程轴 M 与地面线 g(M)——两阶段法冻结平面等于焊死了纵断面可行域的坐标轴与地板
- 橙色格纹条 = no-PVI zone：平面圆曲线段投影到 M 轴形成的变坡点禁区（平纵组合约束）
- 右列 SOLVER LOOP：IJS 搜平面（40 正弦模态）→ DP 精确解纵断面 → 熵权法权衡 C 与 E，回环最小化 F = wC·C/C_ref + wE·E/E_ref

注意：AI 版为示意图，其中 "bridge deck" 重复标注、山顶积雪等细节属生图冗余，无工程含义；
论文引用请以 `draw_schematic.py` 生成的版本为准（该脚本可改为直接读取 results/ 的真实线位重绘）。

---

# 建设区域真实数据图（Fig1–8）

由 `exp/draw_region.py`、`exp/draw_osm.py`、`exp/draw_buildings.py` 生成，坐标为模型局部
笛卡尔系（X 东 / Y 北，米，原点=实测起点）。

| 图 | 文件 | 内容 | 数据来源 |
|---|---|---|---|
| Fig1 | `region_fig1_plan.png` | 实测平面线形 + 走廊带可行域（±500 / ±2500，shapely 缓冲） | 数据.xlsx GPS 中线 |
| Fig2 | `region_fig2_dem.png` | DEM 宽泛平面地形 + 线位 | AWS Terrain Tiles z14 |
| Fig3 | `region_fig3_overview.png` | 地形 + 线位 + 走廊 + 7 桥 1 隧 | +北环桥隧统计.xlsx |
| Fig4 | `region_fig4_osm.png` | OSM 交通网络（路/铁/水）+ 线位 | main 分支 OSM 障碍物 |
| Fig5 | `region_fig5_integrated.png` | DEM+交通网+线位+走廊+高架触发点（平面跨越判据） | 综合 |
| Fig6 | `region_fig6_buildings.png` | 建筑分布 hexbin 密度 vs 走廊宽 | OSM building 质心 |
| Fig7 | `region_fig7_corridor_width.png` | 沿线左右侧到最近建筑净距（走廊宽度数据依据） | OSM building |
| Fig8 | `region_fig8_alllayers.png` | 六层全叠加：DEM+建筑+交通网+线位+走廊 | 综合 |

## 数据来源与依赖
- OSM 障碍物 `osm/obstacles.npz` 由 `osm/extract.py` 从本仓库 main 分支 `数据/OSM走廊带障碍物/` 取出。
- OSM 建筑 `osm/buildings.npz` 由 `fetch_buildings.py` 从 Overpass API 分块抓取（走廊带外扩 2.6 km）。
- DEM `*.npz` 见 `DEM_重建说明.md`。以上 `*.npz` 均按 .gitignore 排除，需脚本重建。

## 重要局限（务必在论文/汇报中声明）
**OSM 建筑数据不完整**：抓取时广州城区若干经度块 Overpass 504 超时，最终仅得 12789 栋，
对整片建成区明显偏少。因此：
- Fig6 hexbin 计数偏低、Fig8 建筑点偏稀，均非真实密度；
- **Fig7 的“到最近建筑净距”是乐观高估**——真实建筑更多更近，可行走廊只会更窄。

但方向性结论稳健：即便用这份偏少数据，±500 m 走廊两侧均无建筑冲突的里程仅 **37.8%**，
±200 m 也仅 **60.9%**。这证明走廊带**不应全线取 ±500 m（更不能取 ±2500 m）**——宽走廊只
适用于少数近郊空地段，城区段实际可行摆动被建筑严格限制。用于论文时建议补抓完整建筑数据
（或改用政府用地红线/影像判读）后重算 Fig7，作为走廊带宽度的正式依据。
