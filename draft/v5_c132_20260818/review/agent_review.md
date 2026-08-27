# v5_c132_20260818 独立审阅报告（固定证据：c1327df）

## 1. 审阅边界与结论摘要

- 审阅对象：`en/main_en.tex`、`zh/main_zh.tex`。
- 唯一证据版本：Git commit `c1327df1ea2dc64cdde826bcb1df7141d80a0533`。所有代码、参数、JSON/CSV 均从该不可变 Git 对象读取；未用任何后续固定端点或密度模型评判本文。
- 审阅范围仅限：（1）句子逻辑；（2）中英文流畅度与一致性；（3）公式与该提交代码/参数的一致性；（4）实验结论能否由该提交已有 JSON/CSV 支持。
- 未启动、未要求、也不建议以补充实验解决本报告问题；下述问题均可用现有 c132 证据完成文稿修正。

总体判断：核心模型口径与绝大多数结果已正确绑定 c132。275 个活跃变量的组成（50 个平面模态、1 个起点高程变量、224 个坡度变量）、`W=500 m` 的谱幅定义、自由纵断面端点、单向能耗、土方梯形积分及 30 m 强制结构替代、交叉桥触发、正部截断、`mu=1.99`、同代数但不等 NFE 等均与提交代码一致。主结果的总成本、总能耗、路线长度、半径、罚值、结构费/土方费变化、两阶段结果、消融及多算法表的主要数值也有 c132 记录支持。

但当前稿件仍有 **1 组 P0 数据错误**：燃油与电耗分项的百分比均算错，而且中英文正文和主表重复出现。在修正前，不能判为四项审查通过。

## 2. P0：必须立即修正

### P0-1 燃油与电耗分项降幅不受 c132 JSON 支持

**位置**

- 英文：`en/main_en.tex` 第 246 行；表 1 第 259--260 行。
- 中文：`zh/main_zh.tex` 第 246 行；表 1 第 259--260 行。

**证据与复算**

`results/joint_results.json` 在 c132 中给出：

- 燃油：M-A = `1110723154.4208326`，M-C = `1075118396.0989451`，变化率为
  `(M-C/M-A-1)×100% = -3.2055475%`，按两位小数应为 **-3.21%**，不是 -3.31%。
- 电耗：M-A = `283881428.3388862`，M-C = `272542116.21649563`，变化率为
  `(M-C/M-A-1)×100% = -3.9943832%`，按两位小数应为 **-3.99%**，不是 -4.23%。

表下注明“变化率由 JSON 未四舍五入数值计算”，因此这不是展示精度差异，而是实质性算术错误。

**可直接替换的措辞**

- 英文正文改为：`Fuel and electricity components fall by 3.21\% and 3.99\%, respectively.`
- 英文表格对应两格改为：`$-3.21\%$`、`$-3.99\%$`。
- 中文正文改为：`燃油和电耗分项分别下降3.21\%和3.99\%。`
- 中文表格对应两格改为：`$-3.21\%$`、`$-3.99\%$`。

建议同时让现有只读审计脚本断言这两个由未取整数值计算的百分比，以免改稿后再次漂移；这只是现有证据的自动核算，不涉及任何新实验。

## 3. P1：重要但可直接由现有证据修正

### P1-1 横向阻力公式缺少 `S_p` 的代码定义，且未披露 c132 对来源公式的已提交勘误

**位置**

- 英文：第 173--193 行，尤其式中 `S_p`。
- 中文：第 174--193 行，尤其式中 `S_p`。

**问题**

c132 的 `objective.py::_lateral_force` 实际采用
`S_p=min(S_{p,max},v^2/(gR))`，其中 `S_{p,max}=0.08`，稿件却未定义 `S_p`。此外，代码明确记录：来源论文式（4.12）末尾的 `10^{-3}` 被判为量纲/数量级不一致而在实现中删除，并要求论文显式声明。当前稿件的 `F_a` 公式数值形式与代码一致（没有该因子），但读者无法知道这是有意的已提交实现修正，而非漏项。

**可直接增加的英文句子**

`The implemented superelevation is $S_p=\min(S_{p,\max},v^2/(gR))$, with $S_{p,\max}=0.08$. The $10^{-3}$ multiplier printed at the end of the source-thesis lateral-force equation is omitted in the pinned implementation because it is dimensionally and numerically inconsistent; Eq. (...) reports the committed implementation.`

**可直接增加的中文句子**

`实现中超高取 $S_p=\min(S_{p,\max},v^2/(gR))$，其中 $S_{p,\max}=0.08$。来源学位论文横向阻力式末尾印有的 $10^{-3}$ 因量纲和数量级不一致而未被固定实现采用；本文公式报告的是 c132 已提交实现。`

### P1-2 两阶段对照的目标叙述把第一阶段也写成了使用共同双目标参考与权重

**位置**

- 英文第 236 行：`it uses the same objective references and weights but 500 generations per stage.`
- 中文第 236 行：`两阶段使用共同目标参考与权重，每阶段500代。`

**问题**

c132 的 `run_twostage.py` 显示：第一阶段只最小化平面相关 LCC，并以 `C_ref_plane` 归一化；冻结平面后，第二阶段才使用联合实验的共同 `C_ref`、`E_ref` 和熵权。现句容易被读成两个阶段都使用共同 C/E 参考和权重。

**可直接替换**

- 英文：`A two-stage control first minimizes plan-related LCC and freezes the resulting plan; the second stage then optimizes the profile using the joint experiment's common $C/E$ references and entropy weights. Each stage runs for 500 generations.`
- 中文：`两阶段对照先以平面相关全寿命成本为目标优化平面并将其冻结，第二阶段再使用与联合实验相同的 $C/E$ 参考值和熵权优化纵断面；每阶段运行500代。`

### P1-3 统计检验名称应按两个数据源分别写清

**位置**

- 英文第 238 行。
- 中文第 238 行。

**问题**

“Mann--Whitney/rank-sum”并列会让读者误以为所有表均由同一函数生成。c132 中消融表使用 Mann--Whitney U；多算法表使用独立样本 Wilcoxon rank-sum（`scipy.stats.ranksums`）。二者都不是配对符号秩检验，稿件关于“非配对、未校正、NFE 不等”的边界判断是正确的，只需精确分流名称。

**可直接替换**

- 英文：`Ablation $p$ values use the independent-sample Mann--Whitney U test, whereas benchmark $p$ values use the independent Wilcoxon rank-sum test (SciPy \texttt{ranksums}); neither analysis is paired.`
- 中文：`消融表的 $p$ 值采用独立样本 Mann--Whitney U 检验，多算法表采用独立样本 Wilcoxon 秩和检验（SciPy \texttt{ranksums}）；两者均非配对检验。`

### P1-4 DE“贡献”措辞仍略带因果色彩，与稿件自己声明的不等 NFE 边界不完全协调

**位置**

- 英文第 315、384 行，尤其 `strong DE contribution`。
- 中文第 315、384 行，尤其“DE贡献较强”。

**问题**

c132 CSV 支持“JS+DE 在单机制变体中具有最大的同代数观测差距（24.17%）”，但额外算子阶段带来更多评价，不能把差距严格归因于 DE 的内在贡献。第 315 行已有 NFE 限定，第 384 行最好保持同一非因果口径。

**可直接替换**

- 英文：`Among the single-operator variants, JS+DE has the largest observed same-generation gap from JS; this observation does not isolate an evaluation-budget-normalized DE effect.`
- 中文：`在单算子变体中，JS+DE 相对 JS 的同代数观测差距最大；该结果不能分离出按评价预算归一化后的 DE 效应。`

### P1-5 公式符号定义不完整，影响读者按 c132 复核

**位置**

- 英文第 115--123、165--193 行。
- 中文第 116--124、166--193 行。

**问题**

文中给出核心结构和“主要参数”表，但 `\Phi_i`、`\gamma`、`\tau`、`c_{soil}`、`\rho`、`C_d`、`A_f`、`C_r`、`m_v`、`N_v`、`C_s`、`\phi`、`HP_{in}`、`\eta_f` 等未在正文定义。公式与 c132 代码的结构吻合，但符号闭合度不足，尤其 `\Phi_i` 无法从上下文判断是什么量。

**可执行修正**

在参数表后增加一段双语符号说明或一张紧凑参数表，按 c132 的 `params.py`/`objective.py` 给出含义与实际值；至少必须定义 `\Phi_i`，并说明燃油式输出为单车单程 L、电耗式输出为单车单程 kWh。此项只需转录已提交参数，不涉及重算结果。

### P1-6 多算法六分辨率表的源清单覆盖不完整

**位置**

- 英文/中文第 336--355 行的六分辨率表；`tables/source_manifest.csv`。

**问题**

表中六个 PJ 分辨率的均值均能在 c132 已有 CSV 中找到，数值本身受支持；但当前 source manifest 仅列出 `表B2_PJ6...csv`，未逐项列出承载 PJ1--PJ5 表格数值的 B2 CSV。因而“数据存在”成立，“清单足以独立追踪整张表”尚不成立。

**可执行修正**

将 c132 中 PJ1--PJ5 对应 B2 CSV 的路径和 SHA-256 追加到 source manifest。仅补源文件指纹，不生成或改变实验数据。

## 4. P2：语言与表达优化

### P2-1 英文摘要和方法中的若干搭配不自然

**位置与建议措辞**

- 英文第 22 行：`a road-removed quasi-natural digital elevation model` 建议改为 `a quasi-natural digital elevation model reconstructed after masking the existing roadway`。
- 英文第 22 行：`Spatial intersections trigger crossing bridges` 建议改为 `Intersections with qualifying OSM features trigger bridge intervals`，更贴合代码对区间的生成与合并。
- 英文第 78 行：`Its plan endpoints define the endpoints of every candidate.` 建议改为 `All candidates share the measured plan endpoints.`
- 英文第 48 行 `a transparent evaluation of IJS` 与中文“边界清晰的解释”力度略有不同，建议统一为 `a boundary-aware evaluation of IJS` / “对 IJS 进行边界清晰的评价”。

### P2-2 英文“hundred-million RMB”反复出现，生硬且削弱句子可读性

**位置**

- 英文第 280、376、404 行。

**可直接改写第 280 行核心句**

`Bridge/tunnel cost decreases by 261.4 million RMB, whereas earthwork cost increases by 45.5 million RMB; maintenance rises by 0.88 million RMB, yielding a net cost reduction of 218.2 million RMB.`

第 376、404 行同样统一为 `261.4 million RMB` 和 `45.5 million RMB`。这与中文 2.6141 亿元、0.4550 亿元完全等值。

### P2-3 中文参数单位搭配应规范化

**位置**

- 中文第 157--160 行。

**建议替换**

- `1.56/2.70亿元每km` → `1.56/2.70亿元/km`
- `30/25元每m$^3$` → `30/25元/m$^3$`
- `30,000辆每日` → `30,000辆/日`
- `8元每L/0.8元每kWh` → `8元/L/0.8元/kWh`

### P2-4 “约两倍耗时”可用现有 B1 CSV 写得更准确

**位置**

- 英文/中文第 336 行。

c132 的 B1 表显示 IJS/JS 耗时比在 PJ1--PJ5 约为 1.9 倍，而 PJ6 约为 2.8 倍。现有“about twice/约两倍”不构成数据错误，但可改为：

- 英文：`IJS takes about 1.9 times the wall time of JS at PJ1--PJ5 and about 2.8 times at PJ6, while also performing more objective calls.`
- 中文：`IJS 在 PJ1--PJ5 的耗时约为 JS 的1.9倍，在 PJ6 约为2.8倍，同时还执行更多目标调用。`

## 5. 已通过的重点核验

1. **中英文一致性**：章节结构、关键数字、表格、限制条件基本逐项一致；未发现英文使用后续模型而中文保留 c132，或反向混用的情况。
2. **275 维定义**：50 个正弦系数 + 1 个起始高程变量 + 224 个逐段坡度变量，与 c132 `objective_joint.py` 完全一致；纵断面首尾均未锚定，稿件多处明确披露。
3. **谱宽 `W`**：稿件正确将 `W` 解释为第一模态幅值，并给出约 `1.625W` 的保守偏移包络，没有将其误写为点态硬走廊半宽。
4. **成本与结构模型**：土方截面积、平均端面积/梯形积分、`|h|>30 m` 强制结构替代、交叉桥 `1/sin(theta)` 延长、两侧各 75 m、100 m 间隙合并，均与 c132 实现吻合。
5. **能耗边界**：EV 正负功分支、总驱动能量正部截断及空调项与代码一致；稿件明确写明只算一个代表方向，并在局限中说明自由终点会影响单向能耗。
6. **主要实验结论**：总成本 -8.26%、总能耗 -3.37%、里程 -0.44%、坡险指标 +3.36%、`R_min=401.03 m`、零罚值，以及结构费减少 2.6141 亿元、土方增加 0.4550 亿元，均可由 c132 JSON 复算。
7. **两阶段结论**：成本/能耗较低但 `R_min=397.06 m`、罚值 0.07359，因此不可行，受 c132 JSON 支持；稿件没有把它误称为更优可行设计。
8. **算法结论边界**：IJS 在已存同代数结果中具有最低观测均值、PJ6 可行性记录更好，均有 CSV 支持；稿件同时明确不等 NFE，未声称等预算优越性。
9. **历史/后续口径隔离**：稿件明确绑定完整 c132 家族，声明无建筑密度约束，且没有用后续固定端点结果替代当前数字。

## 6. 仅针对四项指定范围的最终结论

**当前结论：Fail（存在可直接修正的 P0），修正后可达到 Pass；不需要任何补充实验。**

判定理由：句子逻辑和双语一致性整体合格，核心公式与 c132 代码/参数总体吻合，绝大多数实验结论由 c132 已有 JSON/CSV 支持；但燃油、电耗两个分项降幅在中英文正文和主表中均与固定 JSON 不符，属于明确的数据正确性 P0。将其分别改为 **3.21%** 与 **3.99%**，并完成 P1-1 至 P1-3 的公式/实验描述闭合后，四项范围内可判为通过。P1-4 至 P2 为措辞、可审计性和语言质量提升，不构成要求新增实验的条件。

---

## 7. 修正后复核（P0/P1 闭环）

复核日期：2026-08-18。复核仍以 commit `c1327df1ea2dc64cdde826bcb1df7141d80a0533` 为唯一证据，仅检查上文 P0/P1，不扩展范围，也未运行优化实验。现有只读审计脚本执行结果为 `PASS`；新增 manifest 各行的 SHA-256 亦已逐项与该提交中的 Git 对象复核一致。

### 已解决

1. **P0-1 已解决。** 中英文正文和主表均已将燃油/电耗变化率改为 **-3.21% / -3.99%**；旧值 `-3.31% / -4.23%` 已不存在。审计脚本已增加这两个百分比的 JSON 复算和旧值拦截。
2. **P1-1 已解决。** 双语稿已给出 `S_p=\min(S_{p,max},v^2/(gR))`、`S_{p,max}=0.08`，并明确披露 c132 删除来源式末尾 `10^{-3}` 的实现勘误；表述与 `objective.py::_lateral_force` 一致。
3. **P1-2 已解决。** 双语稿现已明确第一阶段只优化平面相关全生命周期成本，第二阶段才使用联合实验的共同 `C/E` 参考值和熵权，与 `run_twostage.py` 一致。
4. **P1-3 已解决。** 双语稿已分别写明消融使用独立样本 Mann--Whitney U、多算法表使用 SciPy `ranksums`，并明确二者均非配对检验。
5. **P1-6 已解决。** source manifest 已加入 PJ1--PJ5 的 B2 CSV；PJ1--PJ6、B1、B4 及其余已列来源的哈希均与 c132 Git 对象一致。

### 尚未完全解决

#### P1-4 DE 因果措辞仍有一处残留

- 英文第 316 行仍写：`DE produces the largest isolated gap`。
- 中文第 316 行仍写：“DE产生最大的单机制差距”。

讨论部分第 385 行附近已经正确改成“最大同代数观测差距、不能分离预算归一化效应”，但结果部分仍使用 `produces/产生` 与 `isolated/单机制`，语气比现有不等 NFE 证据更强。

可直接替换为：

- 英文：`Among the single-operator variants, JS+DE has the largest observed same-generation gap from JS; Tent alone has a small and statistically inconclusive gap in the unadjusted rank-sum test.`
- 中文：`在单算子变体中，JS+DE相对JS的同代数观测差距最大；Tent单独变体的差距较小，且在未调整秩和检验中不显著。`

#### P1-5 符号/参数补充基本完成，但仍有两个闭合点

新增参数数值与 c132 一致，`\Phi_i` 的代数形式也与代码一致；但：

1. `W_{s,i}`、`W_{h,i}`、`m_s`、`m_h` 尚未明确定义。c132 中前两者分别为挖/填段的横断面宽度（对应段取路基宽 `B`，否则取 0，结构豁免段也取 0），且 `m_s=m_h=1.5`。
2. 同一稿件中 `\alpha` 既在养护项表示权重 `\alpha=0.3`，又在能耗阻力式 `\cos\alpha,\sin\alpha` 表示坡度角，构成符号冲突；代码对坡度角使用 `theta=arctan(grades)`。

可直接补充/替换为：

- 英文养护定义后增加：`Here $W_{s,i}$ and $W_{h,i}$ are the cut- and fill-segment cross-sectional widths (equal to $B$ on the corresponding earthwork segment and zero otherwise, including structure-exempt segments), and $m_s=m_h=1.5$.`
- 中文养护定义后增加：`其中$W_{s,i}$和$W_{h,i}$分别为挖方段、填方段横断面宽度（对应土方段取$B$，其余及结构豁免段取0），且$m_s=m_h=1.5$。`
- 将双语能耗阻力式中的坡度角统一由 `\alpha` 改为 `\vartheta`，并补一句 `\vartheta_i=\arctan g_i` / “坡度角 `\vartheta_i=\arctan g_i`”，保留 `\alpha,\beta` 专用于养护权重。

### 修正后结论

**P0 已全部关闭；P1-1、P1-2、P1-3、P1-6 已关闭；P1-4、P1-5 尚有上述两处可立即完成的文字/符号闭环。**

因此，仅针对本次限定四项，当前状态由原先的 `Fail（P0）` 提升为 **Conditional Pass（无数据硬错误，尚余两项 P1 文稿修正）**。完成上述两处替换后即可判为 **Pass**；不需要、也不建议为此启动任何补充实验。

### 最终快速复核（最后两处）

复核仅针对上一小节剩余的 P1-4 与 P1-5：

1. **P1-4 已关闭。** 英文结果段现表述为 `Among the single-operator variants, JS+DE has the largest observed same-generation gap from JS`，中文同步为“JS+DE 相对 JS 的同代数观测差距最大”；`produces/产生最大 isolated/单机制差距` 的因果化残留已删除。
2. **P1-5 已关闭。** 双语稿已定义 `W_{s,i}`、`W_{h,i}` 为挖/填坡面宽度，定义 `m_s`、`m_h` 为对应边坡坡率，并说明结构豁免段宽度取 0；能耗式中的坡度角已统一改为 `\vartheta=\arctan g_i`，不再与养护权重 `\alpha=0.3` 冲突。相关参数值仍由同节文字及主参数表闭合。

只读自动审计再次返回 `PASS`。未检查任何新范围，未启动实验。

**最终结论：Pass。原报告 P0/P1 已全部关闭。**
