# DP 双层重实现（外层 IJS 搜平面 + 内层 DP 解纵断面）

本目录是四个实验的重新实现，方法全部取自本仓库各分支已用过的实现，不引入新方法。

## 为什么改成双层

主线是 **275 维联合搜索**（平面 50 正弦模态 + 纵断面 225 变坡点），纵断面的规范
约束靠惩罚项软约束，导致两个后果：一是搜索维度高、易陷局部最优；二是惩罚强度一
旦偏弱，解就会带着违规交付（旧消融实验 8 个变体的 `best_pen` 全在 5.9e8~1.5e9，
即全部违反纵坡/坡差，上报的 C/E 不是可施工方案）。

纵断面子问题在"纵坡上下界 + 相邻坡差"约束下，是 (桩号 × 高程 × 进入坡度) 状态空
间上的最短路问题，可用 **动态规划全局最优求解**（`dp_profile.solve_profile`，
单次约 0.19 s）。于是问题分解为：

```
外层 IJS/JS/GA/PSO/GWO/NSGA-II  搜 50 维平面模态
   └─ 每次求值 → 内层 DP 给出该平面下的纵断面全局最优解 → 完整目标 (C,E,pen) 评价
```

275 维软约束搜索 → **50 维外层 + 精确内层**，且纵断面类约束由构造满足。

## 规范合规的分工

| 约束 | 限值 | 满足方式 |
|---|---|---|
| 纵坡上界 | \|i\| ≤ 4% | DP 状态空间构造（`kmax`） |
| 纵坡下界（排水） | \|i\| ≥ 0.3% | DP 状态空间构造（`IMIN_ENFORCE`） |
| 相邻坡差（竖曲线） | ≤ 3e-4×步长 = 0.03 | DP 状态转移构造（`kjump`，按 Δz **数值**取窗口） |
| 首末接线高程 | 8.590 / −4.700 m | DP 起止状态锚定（`z_tie`）+ 残差线性斜坡分摊 |
| 平曲线半径 | R ≥ 400 m | 外层惩罚 `PENALTY["k_R"]=30` |
| 建筑 Tier2 禁区 | 穿越 = 0 | 外层惩罚 `DENSITY["k_forbid"]`（硬约束） |
| 建筑 Tier1 可穿越区 | 抑制但可行 | 外层软代价 `DENSITY["w_dense1"]`，**不进 pen** |

Tier1 必须留在 pen 之外：否则可行性门控（`pen ≤ 0`）会把它变成硬约束，而现状线位
M-A 本身就有约 2.0 km 落在 Tier1，"允许穿越"的设计目标会被静默摧毁。

结果：纵断面类惩罚恒为 0，`pen > 0` 只可能来自平面类（R / Tier2），职责清晰。

## 数据来源

| 数据 | 模块 | 内容 |
|---|---|---|
| 高程 | `dem.py` | AWS Terrain Tiles z14 terrarium + road-removal 重建的**准天然地面** |
| 河流 | `crossings.py` | OSM `water(river, canal)` + `road ≥ primary` + `rail`，交叉触发跨越桥，桥长由几何内生 |
| 建筑 | `building_mask.py` | 密度三级分区 V1；阈值由现状线位标定 `θ_forbid = D_A_max×1.15 = 0.2111` |
| 实测线位 | `data_loader.py` | 广州北环高速 22.462 km 中线与路面高程 |

## 目录内容

```
bilevel.py            双层求解核心（统一标量目标、熵权、基准种群）
dp_profile.py         内层 DP（含本次两处修复，见下）
run_main.py           实验一 主实验（M-A/M-B/M-C + Pareto 前沿 + 熵权决策）
run_ablation.py       实验二 消融（IJS 8 变体对外层平面搜索的贡献）
run_comparison.py     实验三 多算法对比（IJS/JS/NSGA-II/GA/PSO/GWO）
run_sensitivity.py    实验四 敏感性（10 项逐点重优化）
其余模块              params / objective / objective_joint / crossings /
                      building_mask / dem / data_loader / algorithms /
                      benchmarks / safety  —— 均自主线与多算法分支原样取用
```

## 本次对原 DP 的两处修复

1. **坡差约束失效（原版 bug）**：`i_min` 过滤会把 `dzs` 数组挖去中间一段（Δz=0 及
   过小坡度被剔除），而状态转移窗口 `[jd-kjump, jd+kjump+1]` 是按**数组下标**取
   的，跨过缺口后实际 Δz 跨度超过 `kjump` —— 实测坡差达 **0.0349 > 限值 0.0300**。
   改为按 Δz **数值** `searchsorted` 取窗口（`dzs` 升序，仍是连续切片，效率不变）。
   该缺陷在 `IMIN_ENFORCE=1`（默认）时始终触发，`finalexp/` 与 `optlab/` 两份原版
   都有。
2. **端点自由**：原版起点仅受 `z0_range` 限制、终点完全自由（回溯取全局 argmin），
   优化器会把全线整体下倾以套取燃油模型"下坡不回收"的免费油耗，产出接不上路网的
   纵断面。加 `z_tie` 锚定首末状态；网格相位对齐起点，终点格点残差（≤DZ/2）按里程
   线性斜坡分摊消除，附加坡度约 1e-5，不破坏构造合规性。

## 验证（冒烟）

四个入口均已 `--smoke` 跑通：

- 主实验：M-A C=26.4100 亿 / E=13.9455 亿 pen=0；M-C pen=0、Rmin=421 m、
  纵坡 0.497%~3.986%、坡差 0.02990、Tier2 0.000 km、端点 8.590/−4.700 精确
- 消融：8 变体 **全部 pen=0**（旧版 8 个全违规），V5_IJS 均值最优
- 多算法：6 种算法全部跑通、全部 pen=0，IJS 最优
- 敏感性：16 个采样点 **违规 0/16**

另在 5 个不同平面（现状 + 4 随机）上逐一复核：坡差 ≤0.0299、纵坡 3.99%/0.49%、
端点残差 0.0e+00，全部由构造满足。

## 算力（单次外层求值约 199 ms，DP 占大头）

| 实验 | 默认规模 | 估算 |
|---|---|---|
| 主实验 | pop40/iter200×2惩罚阶段, 11 任务 | ~29 核·时 |
| 消融 | pop40/iter150, 8 变体×10 次 | ~80 核·时 |
| 多算法 | pop40/iter150, 6 算法×1 规模×5 种子 | ~30 核·时 |
| 敏感性 | pop40/iter150, 51 采样点 | ~51 核·时 |

合计约 **190 核·时**；本机 cgroup 配额 16 核，约 12 小时。规模均可由命令行调整。
`--scales all`（多算法六档）会显著加倍，慎用。

## 用法

```bash
python3 run_main.py        --smoke              # 冒烟
python3 run_main.py        --corridor 1000 --pareto 9
python3 run_ablation.py    --corridor 1000 --n_runs 10
python3 run_ablation.py    --serial             # 运行时间口径与旧版严格一致
python3 run_comparison.py  --scales PJ5 --n_runs 5
python3 run_sensitivity.py --corridor 1000
```

## 口径提醒

- 内层 DP 的能耗按 **双向平均**（`E_DIRECTION=avg`，`energy_money_per_m`），而主线
  `objectives_joint` 的 E 是**单程**。故 DP 用于选形的能耗与最终上报的 E 口径不同；
  敏感性 item10 专门量化这一差异。若要全链条双向，需改 `objective.py` 的能耗聚合。
- 消融默认按变体并行；运行时间是上报列，绝对值随并发变化，需在表注声明，或用
  `--serial`。解的质量与并行无关（`algorithms.run` 只用 `default_rng(seed)`）。
- 各实验的 `iter/pop` 远小于主线（275 维联合用 pop200/iter1000），因为内层 DP 使
  单次求值成本从毫秒级升到 199 ms；外层维度已从 275 降到 50，所需迭代也相应减少。
