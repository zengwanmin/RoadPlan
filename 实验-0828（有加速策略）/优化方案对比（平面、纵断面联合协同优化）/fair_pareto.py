# -*- coding: utf-8 -*-
"""联合求解器/两阶段求解器的公共 Pareto 决策规则。

两种方法先在完全相同的权重网格上各自产生前沿；本模块在两条
前沿合并后统一执行：可行/预算过滤、非支配筛选、公共极差归一化、
一次性熵权计算和同一评分函数选点。不允许任一方法用自己的数据
范围单独计算熵权。
"""
import numpy as np


SCHEMA = "fair-pareto-common-range-v1"


def _records(sweep, method, penalty_tol, budget):
    """转为统一记录并严格过滤可行性与共同预算上限。"""
    out = []
    for k, raw in enumerate(sweep):
        C = float(raw["C"]); E = float(raw["E"])
        pen = float(raw.get("pen", raw.get("penalty", float("inf"))))
        if not np.all(np.isfinite([C, E, pen])):
            continue
        if pen > penalty_tol or C > budget:
            continue
        rec = dict(raw)
        rec.update(method=method, tag=raw.get("tag", f"pareto_{k}"),
                   w1=float(raw["w1"]), C=C, E=E, pen=pen)
        out.append(rec)
    if not out:
        raise RuntimeError(
            f"{method} has no Pareto candidate satisfying "
            f"pen<={penalty_tol:g} and C<={budget:.6g}")
    return out


def _nondominated(records):
    """返回成本 C 和能耗 E 都是越小越好的非支配集。"""
    return [
        r for r in records
        if not any(
            (o["C"] <= r["C"] and o["E"] <= r["E"])
            and (o["C"] < r["C"] or o["E"] < r["E"])
            for o in records
        )
    ]


def _common_entropy(M):
    """
    在合并后的两条方法前沿上计算唯一的极差范围和熵权。

    C/E 均为成本型指标，效用归一化 Z=(max-X)/(max-min)。某列无变异
    时其信息效用为 0；两列都无变异时退化为等权。
    """
    M = np.asarray(M, dtype=float)
    mn = M.min(axis=0); mx = M.max(axis=0); span = mx - mn
    scale = np.maximum(np.maximum(np.abs(mn), np.abs(mx)), 1.0)
    varying = span > 1e-12 * scale
    Z = np.zeros_like(M)
    Z[:, varying] = (mx[varying] - M[:, varying]) / span[varying]
    eps = 1e-6
    P = (Z + eps) / (Z + eps).sum(axis=0, keepdims=True)
    denom = np.log(max(len(M), 2))
    entropy = -(P * np.log(P)).sum(axis=0) / denom
    utility = np.where(varying, np.maximum(0.0, 1.0 - entropy), 0.0)
    if float(utility.sum()) <= 1e-15:
        weights = np.array([0.5, 0.5])
    else:
        weights = utility / utility.sum()
    return mn, mx, weights, entropy


def _score(records, mn, mx, weights):
    M = np.array([[r["C"], r["E"]] for r in records], dtype=float)
    span = mx - mn
    scale = np.maximum(np.maximum(np.abs(mn), np.abs(mx)), 1.0)
    varying = span > 1e-12 * scale
    Z = np.zeros_like(M)
    Z[:, varying] = (mx[varying] - M[:, varying]) / span[varying]
    scores = Z @ weights
    idx = int(np.argmax(scores))
    return records[idx], float(scores[idx])


def select_common_pareto(joint_sweep, two_stage_sweep, existing_cost,
                         budget_tol=0.10, penalty_tol=1e-6):
    """
    用同一数据范围、归一化和熵权规则分别选出联合/两阶段解。

    返回 (decision, selected_joint_record, selected_two_stage_record)。
    decision 不携带大型 best_x/curve，便于结果文件存档与审核。
    """
    jw = np.array([float(p["w1"]) for p in joint_sweep], dtype=float)
    tw = np.array([float(p["w1"]) for p in two_stage_sweep], dtype=float)
    if jw.shape != tw.shape or not np.allclose(jw, tw, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Joint and two-stage Pareto fronts must use the exact same weight grid")

    budget = (1.0 + float(budget_tol)) * float(existing_cost)
    joint_front = _nondominated(
        _records(joint_sweep, "joint", penalty_tol, budget))
    two_front = _nondominated(
        _records(two_stage_sweep, "two_stage", penalty_tol, budget))

    # 公共范围和熵权使用两条“方法内部非支配前沿”的并集。
    # 这保证两方选点都落在相同 min/max 范围内，且两方样本对熵权
    # 的贡献对称。
    pool = joint_front + two_front
    M = np.array([[r["C"], r["E"]] for r in pool], dtype=float)
    mn, mx, weights, entropy = _common_entropy(M)
    selected_joint, score_joint = _score(joint_front, mn, mx, weights)
    selected_two, score_two = _score(two_front, mn, mx, weights)
    global_front = _nondominated(pool)

    def compact(r):
        return dict(method=r["method"], tag=r["tag"], w1=float(r["w1"]),
                    C=float(r["C"]), E=float(r["E"]), pen=float(r["pen"]))

    decision = dict(
        schema=SCHEMA,
        selection_scope="joint_and_two_stage_front_union",
        filter=dict(penalty_tol=float(penalty_tol), budget_tol=float(budget_tol),
                    budget_C=float(budget)),
        weight_grid=jw.tolist(),
        normalization=dict(
            method="cost_type_min_max_benefit",
            formula="Z=(max-X)/(max-min)",
            min_C=float(mn[0]), max_C=float(mx[0]),
            min_E=float(mn[1]), max_E=float(mx[1])),
        entropy=dict(
            formula="w_j=(1-e_j)/sum(1-e_j)",
            eC=float(entropy[0]), eE=float(entropy[1]),
            wC=float(weights[0]), wE=float(weights[1])),
        candidate_counts=dict(joint_front=len(joint_front),
                              two_stage_front=len(two_front),
                              combined_global_front=len(global_front)),
        combined_global_front=[compact(r) for r in global_front],
        joint=dict(**compact(selected_joint), score=score_joint),
        two_stage=dict(**compact(selected_two), score=score_two),
    )
    return decision, selected_joint, selected_two
