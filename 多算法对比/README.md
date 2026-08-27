# 多算法对比（PJ1–PJ6 平纵联合规模阶梯）

本目录仅保留 PJ1–PJ6 版本。模型采用平纵联合决策、OSM 交叉桥内生触发和
生态隧道内生判定；六个规模只改变纵断面变坡点步长，平面模态数固定为 50。

## 对比设置

- 规模：PJ1–PJ6，对应纵断面步长 500、400、300、200、100、50 m。
- 算法：IJS、JS、NSGA-II、GA、PSO、GWO。
- 正式运行：种群 200，迭代 500，每算法每规模 30 次独立运行；种子为
  1000–1029，各算法在同一运行序号共享初始种群。
- 标量评价：最优值、均值、标准差、收敛代数、运行时间、Wilcoxon 秩和检验和
  Friedman 检验。
- Pareto 评价：每个 PJ 规模均计算前沿。IJS、JS、GA、PSO、GWO 采用 11 点
  权重扫描（成本权重 0.1–0.9），NSGA-II 使用原生双目标第一前沿；以各算法
  前沿并集的非支配集作为统一参考前沿，计算 HV、IGD、Spacing。

## 运行方式

```bash
python3 run_comparison.py --workers 30
python3 make_outputs.py
```

也可以执行 `run_comparison_pj.sh` 顺序完成求解和出图。`--smoke` 仅用于代码通路
检查，会改用 PJ1/PJ6、2 次独立运行、5 次迭代和 3 点权重扫描。

## 输出

- `results/comparison_results.json`：30 次独立运行结果、Pareto 前沿、统一参考前沿、
  HV 参考点及 HV/IGD/Spacing。
- `tables/表B1*`：运行时间与最优综合效益。
- `tables/表B2_PJ1*` 至 `表B2_PJ6*`：描述统计与显著性检验。
- `tables/表B3_PJ1*` 至 `表B3_PJ6*`：Pareto 前沿质量指标。
- `tables/表B4*`：可行性及最优解工程指标。
- `figures/图B1*`：六规模收敛曲线。
- `figures/图B2_PJ1*` 至 `图B2_PJ6*`：Pareto 前沿。
- `figures/图B3*`：运行时间随规模增长趋势。
- `figures/图B4_PJ1*` 至 `图B4_PJ6*`：30 次运行箱线图。

当前已有结果文件和 PJ 图表是上一次运行产物；修改代码本身不会刷新这些结果。
重新正式运行后，结果中的 `meta.n_runs` 才会更新为 30，并生成新的 PJ Pareto 图表和表格。
