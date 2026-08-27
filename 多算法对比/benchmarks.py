# -*- coding: utf-8 -*-
"""
benchmarks.py — 对比算法(GA/PSO/GWO)与 NSGA-II 的 Python 实现

- GA / PSO / GWO 移植自参考代码 GA.m / PSO.m / GWO.m, 在同一熵权标量化目标 F 下寻优,
  与 IJS/JS (algorithms.run) 同口径比较 (实验设计方案2 §1.4 公平性)。
- NSGA-II 依论文 §5.2-5.3 实现: 快速非支配排序 + 拥挤距离 + 锦标赛 + SBX交叉 + 多项式变异,
  在双目标 (C, E) 空间求 Pareto 前沿。参考 Deb et al. 2002 (论文式5.1-5.2 支配准则)。
返回统一 dict(best_x, best_f, curve, nfe) (标量算法) 或 front 相关 (NSGA-II)。
"""
import numpy as np
from acceleration import evaluate_many_ordered, evaluate_one


# =============================================================
#  GA  (移植 GA.m: 锦标赛选择 + 均匀交叉 + 均匀变异)
# =============================================================
def run_GA(fobj, lb, ub, pop0, max_iter, seed, p1=0.8, p2=0.2, record=True):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    x = np.array(pop0, float).copy()
    n, dim = x.shape
    y = evaluate_many_ordered(fobj, x); nfe = n
    yp = y.copy(); xp = x.copy()
    gi = np.argmin(y); yg = y[gi]; xg = x[gi].copy()
    curve = [yg]
    for _ in range(max_iter):
        # 锦标赛选择(GA.m): 选 2n 个父代
        parent = np.empty(2 * n, int)
        for i in range(2 * n):
            k = int(np.ceil(rng.integers(1, n + 1) * p1))
            idx = rng.choice(n, size=max(k, 1), replace=False)
            parent[i] = idx[np.argmin(y[idx])]
        newX = x.copy()
        for i in range(n):
            newX[i] = x[parent[i]]
            if rng.random() < p1:                        # 均匀交叉
                m = rng.random(dim) > 0.5
                newX[i, m] = x[parent[i + n], m]
            if rng.random() < p2:                        # 均匀变异
                m = rng.random(dim) > 0.5
                tmp = rng.random(dim) * (ub - lb) + lb
                newX[i, m] = tmp[m]
        newX = np.clip(newX, lb, ub)
        newY = evaluate_many_ordered(fobj, newX, reject_above=yp)
        nfe += n
        for i in range(n):
            ny = newY[i]
            if ny < yp[i]:
                yp[i] = ny; xp[i] = newX[i]
                if ny < yg:
                    yg = ny; xg = newX[i].copy()
        x = xp.copy(); y = yp.copy()
        if record:
            curve.append(yg)
    return dict(best_x=xg, best_f=yg, curve=np.array(curve), nfe=nfe)


# =============================================================
#  PSO  (移植 PSO.m)
# =============================================================
def run_PSO(fobj, lb, ub, pop0, max_iter, seed, w=0.8, c1=2.0, c2=2.0, record=True):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    x = np.array(pop0, float).copy()
    n, dim = x.shape
    v = rng.standard_normal((n, dim))
    y = evaluate_many_ordered(fobj, x); nfe = n
    yp = y.copy(); xp = x.copy()
    gi = np.argmin(y); yg = y[gi]; xg = x[gi].copy()
    curve = [yg]
    for _ in range(max_iter):
        r1 = rng.random((n, dim)); r2 = rng.random((n, dim))
        for i in range(n):
            v[i] = w * v[i] + c1 * r1[i] * (xg - x[i]) + c2 * r2[i] * (xp[i] - x[i])
            x[i] = np.clip(x[i] + v[i], lb, ub)
            yi = evaluate_one(fobj, x[i], reject_above=yp[i]); nfe += 1
            if yi < yp[i]:
                yp[i] = yi; xp[i] = x[i].copy()
                if yi < yg:
                    yg = yi; xg = x[i].copy()
        if record:
            curve.append(yg)
    return dict(best_x=xg, best_f=yg, curve=np.array(curve), nfe=nfe)


# =============================================================
#  GWO  (移植 GWO.m: 灰狼优化)
# =============================================================
def run_GWO(fobj, lb, ub, pop0, max_iter, seed, record=True):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    x = np.array(pop0, float).copy()
    n, dim = x.shape
    y = evaluate_many_ordered(fobj, x); nfe = n
    order = np.argsort(y)
    Apos, Bpos, Dpos = x[order[0]].copy(), x[order[1]].copy(), x[order[2]].copy()
    Asc, Bsc, Dsc = y[order[0]], y[order[1]], y[order[2]]
    yg = Asc; xg = Apos.copy()
    curve = [yg]
    for it in range(max_iter):
        a = 2 - it * (2.0 / max_iter)
        for i in range(n):
            # 向量化维度更新(GWO.m 三头狼引导)
            A1 = 2 * a * rng.random(dim) - a; C1 = 2 * rng.random(dim)
            X1 = Apos - A1 * np.abs(C1 * Apos - x[i])
            A2 = 2 * a * rng.random(dim) - a; C2 = 2 * rng.random(dim)
            X2 = Bpos - A2 * np.abs(C2 * Bpos - x[i])
            A3 = 2 * a * rng.random(dim) - a; C3 = 2 * rng.random(dim)
            X3 = Dpos - A3 * np.abs(C3 * Dpos - x[i])
            x[i] = np.clip((X1 + X2 + X3) / 3.0, lb, ub)
            nf = evaluate_one(fobj, x[i], reject_above=Dsc); nfe += 1
            if nf < Asc:
                Dsc, Dpos = Bsc, Bpos.copy(); Bsc, Bpos = Asc, Apos.copy()
                Asc, Apos = nf, x[i].copy()
                if Asc < yg:
                    yg = Asc; xg = Apos.copy()
            elif nf < Bsc:
                Dsc, Dpos = Bsc, Bpos.copy(); Bsc, Bpos = nf, x[i].copy()
            elif nf < Dsc:
                Dsc, Dpos = nf, x[i].copy()
        if record:
            curve.append(yg)
    return dict(best_x=xg, best_f=yg, curve=np.array(curve), nfe=nfe)


# =============================================================
#  NSGA-II  (论文 §5.2-5.3, Deb et al. 2002)
# =============================================================
def _domination_matrix(F):
    """向量化支配矩阵 D[p,q]=True 表示 p 支配 q (最小化, 式5.1-5.2)。"""
    # le[p,q,k] = F[p,k] <= F[q,k]
    le = F[:, None, :] <= F[None, :, :]
    lt = F[:, None, :] < F[None, :, :]
    return le.all(axis=2) & lt.any(axis=2)


def _fast_nondominated_sort(F):
    """快速非支配排序(§5.3.1 步骤3), 返回各前沿的个体索引列表。向量化支配矩阵。"""
    n = len(F)
    D = _domination_matrix(F)
    ndom = D.sum(axis=0)          # 被多少个体支配
    fronts = []
    remaining = np.ones(n, bool)
    cur = np.where((ndom == 0) & remaining)[0]
    while len(cur):
        fronts.append(list(cur))
        remaining[cur] = False
        # 被当前前沿支配的个体, 其支配计数减少
        ndom = ndom - D[cur].sum(axis=0)
        cur = np.where((ndom <= 0) & remaining)[0]
    return fronts


def _crowding_distance(F, idx):
    """拥挤距离(§5.3.1 步骤4): 边界解设为无穷。"""
    m = len(idx)
    if m == 0:
        return np.array([])
    dist = np.zeros(m)
    Fi = F[idx]
    for k in range(Fi.shape[1]):
        order = np.argsort(Fi[:, k])
        dist[order[0]] = dist[order[-1]] = np.inf
        rng_k = Fi[order[-1], k] - Fi[order[0], k]
        if rng_k < 1e-12:
            continue
        for j in range(1, m - 1):
            dist[order[j]] += (Fi[order[j + 1], k] - Fi[order[j - 1], k]) / rng_k
    return dist


def run_NSGA2(fobj_bi, lb, ub, pop0, max_iter, seed, pc=0.9, pm=0.1,
              eta=20.0, scalar_weights=None, record=True):
    """
    NSGA-II 双目标求解。fobj_bi(x)->[C_norm, E_norm]。
    scalar_weights: 给定(wC, wE)时，逐代记录当前种群的最优加权标量值，
                    使NSGA-II可与其余算法进行30次收敛轨迹统计。
    返回 dict(front_X, front_F, curve, nfe, all_F)。
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    P = np.array(pop0, float).copy()
    n, dim = P.shape
    F = evaluate_many_ordered(fobj_bi, P)
    nfe = n
    scalar_weights = (None if scalar_weights is None
                      else np.asarray(scalar_weights, dtype=float))
    if scalar_weights is not None and scalar_weights.shape != (2,):
        raise ValueError("scalar_weights必须为(wC, wE)两个权重")
    curve = ([float(np.min(F @ scalar_weights))]
             if record and scalar_weights is not None else [])

    def make_offspring(P, F):
        # 锦标赛选择(基于前沿层级+拥挤距离)(§5.3.1 步骤5)
        fronts = _fast_nondominated_sort(F)
        rank = np.zeros(n, int)
        cd = np.zeros(n)
        for fi, fr in enumerate(fronts):
            for idx in fr:
                rank[idx] = fi
            d = _crowding_distance(F, fr)
            for j, idx in enumerate(fr):
                cd[idx] = d[j]

        def tour():
            a, b = rng.integers(n), rng.integers(n)
            if rank[a] < rank[b]:
                return a
            if rank[b] < rank[a]:
                return b
            return a if cd[a] > cd[b] else b

        Q = np.empty_like(P)
        for i in range(0, n, 2):
            p1, p2 = P[tour()], P[tour()]
            c1, c2 = p1.copy(), p2.copy()
            # SBX 模拟二进制交叉(§5.3.1 精英保留)
            if rng.random() < pc:
                u = rng.random(dim)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            # 多项式变异
            for c in (c1, c2):
                mm = rng.random(dim) < pm
                c[mm] += (rng.random(mm.sum()) - 0.5) * 0.2 * (ub[mm] - lb[mm])
            Q[i] = np.clip(c1, lb, ub)
            if i + 1 < n:
                Q[i + 1] = np.clip(c2, lb, ub)
        return Q

    for _ in range(max_iter):
        Q = make_offspring(P, F)
        FQ = evaluate_many_ordered(fobj_bi, Q); nfe += n
        # 合并父子代 (§5.3.2 步骤3)
        R = np.vstack([P, Q]); FR = np.vstack([F, FQ])
        fronts = _fast_nondominated_sort(FR)
        newP_idx = []
        for fr in fronts:
            if len(newP_idx) + len(fr) <= n:
                newP_idx.extend(fr)
            else:
                d = _crowding_distance(FR, fr)
                order = np.argsort(-d)
                need = n - len(newP_idx)
                newP_idx.extend([fr[o] for o in order[:need]])
                break
        newP_idx = np.array(newP_idx)
        P = R[newP_idx]; F = FR[newP_idx]
        if record and scalar_weights is not None:
            curve.append(float(np.min(F @ scalar_weights)))

    # 输出第一前沿
    fronts = _fast_nondominated_sort(F)
    f0 = np.array(fronts[0])
    return dict(front_X=P[f0], front_F=F[f0], curve=np.asarray(curve),
                nfe=nfe, all_F=F)


BENCH_SCALAR = {"GA": run_GA, "PSO": run_PSO, "GWO": run_GWO}
