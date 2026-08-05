# -*- coding: utf-8 -*-
"""
algorithms.py — 水母搜索(JS)及其改进(IJS)算法, Python实现

严格移植自参考代码 js.m / IJS.m / levy.m, 并将三大改进组件做成开关,
以支持消融实验的 5 个变体:
  V1 JS(基线)      : tent=F, levy=F, de=F
  V2 JS+Tent       : tent=T, levy=F, de=F
  V3 JS+Levy       : tent=F, levy=T, de=F
  V4 JS+DE         : tent=F, levy=F, de=T
  V5 IJS(完整)     : tent=T, levy=T, de=T

对应论文 §4.2:
  洋流阶段 式(48)   水母群主动/被动 式(49)-(52)   时间控制 式(53)
  Tent初始化 式(47) Levy飞行 式(54)-(55)  DE 式(56)-(58)  边界回弹 式(59)
"""
import math
import numpy as np


def _simplebounds(s, lb, ub):
    """边界回弹(式59 / IJS.m simplebounds): 越界反射回相反边界。"""
    s = np.array(s, float)
    for _ in range(100):
        low = s < lb
        if low.any():
            s[low] = ub[low] + (s[low] - lb[low])
        high = s > ub
        if high.any():
            s[high] = lb[high] + (s[high] - ub[high])
        if not (s < lb).any() and not (s > ub).any():
            break
    return np.clip(s, lb, ub)


def _levy(dim, beta, rng):
    """Levy步长(levy.m / 式54)。"""
    num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = rng.normal(0, sigma_u, dim)
    v = rng.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))


def _tent_init(x0, lb, ub, mu, rng):
    """Tent混沌映射初始化(式47 / IJS.m)。在已随机种群基础上做混沌扰动择优。"""
    return x0  # 混沌扰动在主循环外统一处理(见 run)


def run(fobj, lb, ub, pop0, max_iter, seed,
        use_tent=False, use_levy=False, use_de=False,
        CR=0.5, levy_beta=1.5, mu_tent=2.0, beta_d=3.0, C0=0.5,
        record=True):
    """
    JS/IJS主循环。返回 dict(best_x, best_f, curve, nfe)。
      fobj    : 标量目标 f(x)->float (越小越优)
      pop0    : 初始种群 (nPop × dim), 各算法共享同一初始种群保证公平
      seed    : 随机种子(独立运行区分)
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    pop = np.array(pop0, float).copy()
    nPop, dim = pop.shape
    cost = np.array([fobj(pop[i]) for i in range(nPop)])
    nfe = nPop

    # ---- Tent 混沌映射初始化 (式47): 生成混沌序列择优替换 ----
    if use_tent:
        r0 = rng.random(dim)
        for i in range(nPop):
            p1 = r0 >= 0.5
            r0 = np.where(p1, mu_tent * (1 - r0), mu_tent * r0)
            r0 = np.clip(r0, 0, 1)
            cand = r0 * (ub - lb) + lb
            cand = _simplebounds(cand, lb, ub)
            fc = fobj(cand); nfe += 1
            if fc < cost[i]:
                cost[i] = fc; pop[i] = cand

    idx = np.argmin(cost)
    best_x = pop[idx].copy(); best_f = cost[idx]
    curve = [best_f]

    for it in range(1, max_iter + 1):
        meanvl = pop.mean(axis=0)
        bidx = np.argmin(cost)
        best_sol = pop[bidx].copy(); best_cost = cost[bidx]

        for i in range(nPop):
            # 时间控制函数 c(t) (式53): Ar=(1-t/Max)*(2*rand-1)
            Ar = (1 - it / max_iter) * (2 * rng.random() - 1)
            if abs(Ar) >= C0:
                # 洋流阶段 (式48): Xi + rand*(Xbest - βd*rand*mean)
                newsol = pop[i] + rng.random(dim) * (best_sol - beta_d * rng.random() * meanvl)
            else:
                if rng.random() <= (1 - Ar):
                    # 主动运动 (式50-52): 朝更优同伴方向
                    j = i
                    while j == i:
                        j = rng.integers(nPop)
                    step = pop[i] - pop[j]
                    if cost[j] < cost[i]:
                        step = -step
                    newsol = pop[i] + rng.random(dim) * step
                else:
                    # 被动运动 (式49): Xi + γ'*rand*(ub-lb)
                    newsol = pop[i] + 0.1 * (ub - lb) * rng.random()
            newsol = _simplebounds(newsol, lb, ub)
            fnew = fobj(newsol); nfe += 1
            if fnew < cost[i]:
                pop[i] = newsol; cost[i] = fnew
                if fnew < best_cost:
                    best_cost = fnew; best_sol = newsol.copy()

        # ---- Levy 飞行 (式54-55) + 贪婪选择 ----
        if use_levy:
            for i in range(nPop):
                cand = pop[i] + rng.random(dim) * _levy(dim, levy_beta, rng)
                cand = _simplebounds(cand, lb, ub)
                fc = fobj(cand); nfe += 1
                if fc < cost[i]:
                    pop[i] = cand; cost[i] = fc
                    if fc < best_cost:
                        best_cost = fc; best_sol = cand.copy()

        # ---- 差分进化 DE (式56-58) + 贪婪选择 ----
        if use_de:
            for i in range(nPop):
                r1, r2 = rng.integers(nPop), rng.integers(nPop)
                mutant = pop[i] + rng.random(dim) * (pop[r1] - pop[r2])   # 变异(式56)
                mask = rng.random(dim) < CR                                # 交叉(式57)
                cand = np.where(mask, pop[i], mutant)
                cand = _simplebounds(cand, lb, ub)
                fc = fobj(cand); nfe += 1
                if fc < cost[i]:                                           # 选择(式58)
                    pop[i] = cand; cost[i] = fc
                    if fc < best_cost:
                        best_cost = fc; best_sol = cand.copy()

        if best_cost < best_f:
            best_f = best_cost; best_x = best_sol.copy()
        if record:
            curve.append(best_f)

    return dict(best_x=best_x, best_f=best_f, curve=np.array(curve), nfe=nfe)


# 5个消融变体配置 (对应实验设计方案 表A1)
VARIANTS = {
    "V1_JS":       dict(use_tent=False, use_levy=False, use_de=False),
    "V2_JS+Tent":  dict(use_tent=True,  use_levy=False, use_de=False),
    "V3_JS+Levy":  dict(use_tent=False, use_levy=True,  use_de=False),
    "V4_JS+DE":    dict(use_tent=False, use_levy=False, use_de=True),
    "V5_IJS":      dict(use_tent=True,  use_levy=True,  use_de=True),
}
