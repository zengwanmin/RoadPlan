# history_figures — 历史优化结果全图层叠加可视化

每张图 = 全图层底图（DEM 地形 + OSM 交通网 + OSM 建筑 + 现状中线 + 走廊带）
叠加**该版本优化后的平面线位**（红实线；现状为黑点线）。

## 命名规则
`{来源}_{方法}_corr{走廊带半宽:04d}m_Cdown{成本降幅%}_Edown{能耗降幅%}_{口径}.png`
- 来源：finalexp（定稿 熵权+IJS+DP）/ optlab-v9（宽走廊里程碑）/ main（准天然DEM+OSM桥隧+官方造价口径）
- 方法：joint（联合）/ twostage（两阶段）/ joint-refined / joint-entropyDP
- 口径：avg（双向能耗，推荐）/ single（单向，论文式4.4 原口径）/ ewm（main 的前沿熵权决策）
- 降幅 = 相对各自版本 M_A（现状）的下降百分比

## 三个来源的口径差异（重要，降幅不可跨来源直接比绝对值）

| 来源 | 地面/成本口径 | 平面编码 | M_A 基准 C | 坐标 |
|---|---|---|---|---|
| finalexp | AWS DEM z14 + 桥886.69万/km + 隧27000万/km | 40 正弦模态(1/k^1.5) | 13.22 亿 | 由 best_x 按走廊带解码 |
| optlab-v9 | 同 finalexp | 同 finalexp | — | 由 best_x 解码 |
| main | 准天然DEM + OSM锚定桥隧 + 官方造价 | dim=275, n_mode=50 | 23.71 亿 | 结果直接存储 plane_x/y |

→ finalexp/optlab 与 main 是**两套独立口径**，M_A 成本基准不同（13.22 vs 23.71 亿）。
同来源内可比降幅；跨来源只比"平面走向/取直程度"，不比降幅绝对值。

## 覆盖清单（21 张）
- finalexp 联合：走廊 500–1000 m × {avg, single} 共 12 张
- finalexp 两阶段：走廊 500 m × {avg, single} 共 2 张
- finalexp 精修终解：走廊 500 m avg（见 refine_w500_final）— 已并入联合命名
- optlab-v9：走廊 2500 m avg（C−11.2% / E−7.9%）1 张
- main：联合/两阶段 × 走廊 250/500 m（ewm）共 6 张

## 结果规律（同来源内）
- finalexp **avg 口径**：成本主降（C −7%）、能耗随走廊放宽小幅增益（E 3.2→4.2%）
- finalexp **single 口径**：能耗大降（E 7.4→8.4%）但成本几乎不降（C 0.2→1.9%）
  ——单向口径诱导"净下坡"套利，见报告§能耗方向；双向重评后 E 回落至 3% 量级
- main **ewm**：成本降幅最大（C 10–13.7%），因其口径下 M_A 基准更高、可压缩空间更大

## 复现
`python3 exp/draw_history.py`（主控自动 fork 子进程，按走廊带隔离环境变量解码）。
main 结果由 `git show origin/main:.../results/*.json` 导出至 `exp/results/main_branch/`。
底图 DEM/OSM 数据见 `../figures/README.md` 与 `DEM_重建说明.md`。
