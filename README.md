# RoadPlan：固定端点 W500 平纵联合优化实验

本仓库用于广州北环高速平面—纵断面联合协同优化研究。当前论文主口径为：

- 模型逻辑基线：`ce69f0e`；
- 固定平面起终点与纵断面接线高程；
- 平面谱宽 `W=500 m`；
- 建筑密度 Tier-2 硬约束与 Tier-1 软代价开启；
- DEM、OSM交叉桥、生态隧道和工程几何约束开启；
- 275个上游槽位，读取274个基因，确认性实验使用273个独立自由度；
- 成本与全生命周期交通能耗采用熵权决策；
- 算法比较采用共享初始种群、严格相同 NFE 和配对统计。

旧自由端点结果不得作为当前论文主证据。历史论文、表格、图件和108条探索性确认实验已保存在 Git 标签：

```text
archive/pre-ce69-w500-fixed-endpoint-20260818
```

## 当前状态

断点续跑、配置指纹和跨结果绑定代码已完成：

- 代码提交：`fcf820579e8611b1f32d856926ca9909a7bdd24c`
- 冻结记录提交：`d1f1110`
- 正式固定端点 W500 结果：尚未启动生成
- 旧 `joint_results_w500_dens.json`：自由端点历史数据，确认性脚本和图表脚本会拒绝使用

核心目录：

```text
实验/
├── 数据/
├── 优化方案对比（平面、纵断面联合协同优化）/
│   ├── run_joint.py
│   ├── run_twostage.py
│   ├── make_outputs.py
│   ├── results/
│   ├── tables/
│   └── figures/
├── confirmatory_current_v1_20260817/
│   ├── run_confirmatory.py
│   ├── tests/
│   ├── results/
│   └── PROTOCOL.md
└── draft/
```

## 环境

建议使用 Python 3.13。主要依赖为 NumPy、SciPy、Pandas、Matplotlib、OpenPyXL、Rasterio、Shapely 和 PyProj。

```bash
cd 实验
python -m pip install -r requirements.txt
```

正式计算前建议关闭系统自动休眠，并保证模型与数据文件在运行期间不被修改。运行时配置指纹包含代码、数据、权重、尺度、初始种群、软件环境和上游结果哈希；任何变化都会使续跑失败关闭。

## 预检

```bash
cd confirmatory_current_v1_20260817
python tests/test_protocol.py
python tests/test_resume_guards.py
```

预期输出包括：

```text
PASS exact-NFE checks
PASS quotient profile equivalence
PASS current-model M-A equivalence and feasibility
PASS repository baseline parity at complete-generation budgets
PASS joint checkpoint fingerprint guard
PASS confirmatory checkpoint fingerprint guard
```

## 正式运行顺序

### 1. 固定端点 W500 联合主实验

```bash
cd "优化方案对比（平面、纵断面联合协同优化）"
python run_joint.py --corridor 500 --pareto 21 --workers 12 --fresh
```

首次运行使用 `--fresh`。中断后必须使用完全相同的命令，但去掉 `--fresh`：

```bash
python run_joint.py --corridor 500 --pareto 21 --workers 12
```

每个 M-B、M-C 和 Pareto 权重点完成后都会原子写入：

```text
results/joint_results_w500_dens.partial.json
```

全部完成且最终方案可行后，才会原子覆盖：

```text
results/joint_results_w500_dens.json
```

随后自动删除冗余 partial 文件。

### 2. 同口径两阶段对照

两阶段对照必须绑定上一步联合结果的精确 SHA-256：

```bash
python run_twostage.py \
  --corridor 500 \
  --joint-result results/joint_results_w500_dens.json \
  --fresh
```

中断后去掉 `--fresh` 恢复。Stage 1和Stage 2分别保存检查点。若联合结果、代码、数据、权重或尺度变化，程序拒绝续接。

### 3. 重新生成 C2、C3和图件

```bash
python make_outputs.py
```

生成器会在写任何文件前检查：

- 联合结果具有有效配置指纹；
- 联合结果为正式 W500、密度开启口径；
- 两阶段结果绑定同一联合结果哈希；
- 两种 M-C 方案均满足惩罚和最小半径要求。

检查失败时不会用旧 JSON 静默生成新表。输出来源记录在：

```text
tables/SOURCE_PROVENANCE.json
```

### 4. 确认性冒烟测试

```bash
cd ../confirmatory_current_v1_20260817
python run_confirmatory.py --smoke --workers 2 --fresh
```

确认性脚本只有在新的正式 W500 联合结果存在、来源指纹有效且模型代码/数据与其一致时才允许启动。

### 5. 12算法 × 20配对重复确认性实验

```bash
python run_confirmatory.py --workers 12 --fresh
```

中断后去掉 `--fresh`：

```bash
python run_confirmatory.py --workers 12
```

每完成一个“算法×重复”任务即写入原子检查点。全部240条完成后生成 `confirmatory_raw.json` 并移除 partial 文件。

### 6. 验证、统计与制图

```bash
python validate_results.py
python analyze_confirmatory.py
python run_collaboration.py --workers 12
python validate_collaboration.py
python analyze_collaboration.py
python run_robustness.py --workers 12
python validate_robustness.py
python analyze_robustness.py
python evaluate_operational_scenarios.py
python make_manifest.py
```

统计采用配对 Wilcoxon signed-rank、Holm多重校正、配对效应量和95%置信区间。

## 断点续跑规则

以下任一内容变化，程序都会拒绝读取旧检查点：

- Git提交和核心代码哈希；
- DEM、OSM、密度栅格和原始线路数据哈希；
- W、密度开关、维数、种群规模、迭代数或 NFE；
- `wC`、`wE`、`C_ref`、`E_ref`；
- 上游联合结果 SHA-256；
- 初始种群内容与随机种子；
- Python、NumPy、SciPy和运行平台。

配置不一致时不得删除报错后强行拼接。确认旧证据已提交到 Git 后，才能使用 `--fresh` 建立全新实验。

## 结果与 Git 管理

当前版本只保留最新权威结果；历史结果通过提交和标签恢复，不在当前目录复制多套同名数据。

查看历史快照：

```bash
git show archive/pre-ce69-w500-fixed-endpoint-20260818
```

读取历史文件：

```bash
git show "archive/pre-ce69-w500-fixed-endpoint-20260818:优化方案对比（平面、纵断面联合协同优化）/tables/表C2_优化前后关键指标对比表.csv"
```

正式实验全部完成后，应执行一次结果清理提交：

1. 新 JSON、C2/C3、图件和统计结果覆盖旧文件；
2. 从当前版本删除旧 draft 和探索性输出；
3. 生成并核对结果 manifest；
4. 保证 `git status` 干净；
5. 提交当前统一口径结果。

## 重要解释边界

- 不根据降低率大小选择历史版本；
- 不把自由端点结果解释为固定端点结果；
- 不把不同 W、不同密度口径或不同 NFE 的结果直接比较；
- 冒烟结果只验证管线，不用于论文结论；
- 单次最优解不能代替多随机种子统计；
- 当前论文结论必须由同一模型、数据、预算和来源指纹下的主实验、对照、消融与稳健性结果共同支持。
