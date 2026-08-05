# -*- coding: utf-8 -*-
"""
metrics.py — Pareto 前沿质量指标(HV/IGD/Spacing)与统计检验

多目标最小化, 目标已归一化(C_norm, E_norm)。
  HV      : 超体积(相对参考点, 越大越好)
  IGD     : 反世代距离(相对参考前沿, 越小越好)
  Spacing : 解分布均匀性(越小越均匀)
统计检验(实验设计方案2 §1.4): Wilcoxon 秩和 + Friedman。
"""
import numpy as np
from scipy import stats


def nondominated(F):
    """提取非支配解(最小化)。F: (n,2)。"""
    F = np.asarray(F, float)
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False
                break
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
def wilcoxon_ranksum(target_samples, other_samples):
    """Wilcoxon 秩和检验(成对, IJS vs 其它), 返回 p 值。"""
    try:
        _, p = stats.ranksums(target_samples, other_samples)
        return float(p)
    except Exception:
        return np.nan


def friedman_ranks(sample_dict):
    """
    Friedman 检验: sample_dict {algo: [30次F值]}。
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
