# -*- coding: utf-8 -*-
"""
metrics.py — Pareto 前沿质量指标(HV/IGD/Spacing)与统计检验

多目标最小化, 目标已归一化(C_norm, E_norm)。
  HV      : 超体积(相对参考点, 越大越好)
  IGD     : 反世代距离(相对参考前沿, 越小越好)
  Spacing : 解分布均匀性(越小越均匀)
统计检验(实验设计方案2 §1.4): 配对Wilcoxon符号秩 + Friedman。
"""
import numpy as np
from scipy import stats


def nondominated(F):
    """提取非支配解(最小化)。F: (n,2)。"""
    F = np.asarray(F, float)
    if F.size == 0:
        return np.empty((0, 2), float)
    F = np.atleast_2d(F)
    F = F[np.all(np.isfinite(F), axis=1)]
    F = np.unique(F, axis=0)
    if F.shape[1] == 2:
        # 二目标专用 O(n log n) 扫描；10次Pareto前沿池化后仍可高效处理。
        order = np.lexsort((F[:, 1], F[:, 0]))
        sorted_F = F[order]
        keep = np.zeros(len(sorted_F), dtype=bool)
        best_y = np.inf
        for i, (_, y) in enumerate(sorted_F):
            if y < best_y:
                keep[i] = True
                best_y = y
        return sorted_F[keep]

    # 非二目标的通用回退实现。
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)):
            keep[i] = False
    return F[keep]


def hypervolume_2d(F, ref):
    """
    2D 超体积(最小化): 参考点 ref 在右上方。
    对非支配前沿按第一目标升序, 逐点累加矩形面积。
    """
    P = nondominated(F)
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    # 从大到小(右->左)累加, 保证不重叠
    P = P[np.argsort(-P[:, 0])]
    prev_x = ref[0]
    for x, y in P:
        if x >= ref[0] or y >= ref[1]:
            continue
        width = prev_x - x
        height = ref[1] - y
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x
    return hv


def igd(F, ref_front):
    """反世代距离: 参考前沿每点到 F 的最小距离均值。"""
    P = nondominated(F)
    if len(P) == 0:
        return np.inf
    d = []
    for r in ref_front:
        d.append(np.min(np.sqrt(np.sum((P - r) ** 2, axis=1))))
    return float(np.mean(d))


def spacing(F):
    """Spacing 指标: 相邻解间距的标准差(越小越均匀)。"""
    P = nondominated(F)
    if len(P) < 2:
        return 0.0
    n = len(P)
    di = np.zeros(n)
    for i in range(n):
        dd = np.sum(np.abs(P[i] - P), axis=1)
        dd[i] = np.inf
        di[i] = dd.min()
    dbar = di.mean()
    return float(np.sqrt(np.sum((dbar - di) ** 2) / (n - 1)))


def build_reference_front(all_fronts):
    """由所有算法前沿的并集提取参考前沿(§Pareto质量评估基准)。"""
    U = np.vstack([f for f in all_fronts if len(f) > 0])
    return nondominated(U)


# ---------------- 统计检验 ----------------
def wilcoxon_signedrank(target_samples, other_samples):
    """相同种子配对结果的双侧Wilcoxon符号秩检验，返回p值。"""
    target = np.asarray(target_samples, dtype=float)
    other = np.asarray(other_samples, dtype=float)
    if target.shape != other.shape:
        raise ValueError("配对Wilcoxon要求两组样本数量与顺序一致")
    if np.allclose(target, other, rtol=0.0, atol=1e-15):
        return 1.0
    try:
        _, p = stats.wilcoxon(target, other, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


def holm_adjust(p_values):
    """Holm逐步校正。输入/输出均为{name: p}字典。"""
    finite = [(k, float(p)) for k, p in p_values.items() if np.isfinite(p)]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted = {k: np.nan for k in p_values}
    running = 0.0
    m = len(ordered)
    for i, (key, p) in enumerate(ordered):
        running = max(running, (m - i) * p)
        adjusted[key] = float(min(running, 1.0))
    return adjusted


def friedman_ranks(sample_dict):
    """
    Friedman 检验: sample_dict {algo: [10次F值]}。
    返回 (chi2, p, 平均秩 dict)。
    对每次run(列)按 F 升序给秩, 再对算法求平均秩。
    """
    names = list(sample_dict.keys())
    mat = np.array([sample_dict[k] for k in names])   # algos × runs
    try:
        chi2, p = stats.friedmanchisquare(*[mat[i] for i in range(len(names))])
    except Exception:
        chi2, p = np.nan, np.nan
    # 平均秩(每列排序)
    ranks = np.zeros_like(mat, float)
    for j in range(mat.shape[1]):
        ranks[:, j] = stats.rankdata(mat[:, j])
    avg_rank = {names[i]: float(ranks[i].mean()) for i in range(len(names))}
    return float(chi2), float(p), avg_rank
