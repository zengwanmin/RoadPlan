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
from acceleration import evaluate_many_ordered, evaluate_one


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


def _levy(dim, beta, rng, sigma_u=None):
    """Levy步长(levy.m / 式54)。"""
    if sigma_u is None:
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
        CR=0.5, levy_beta=1.5, mu_tent=1.99, beta_d=3.0, C0=0.5,
        tent_chains=10, record=True, track=False):
    """
    JS/IJS主循环。返回 dict(best_x, best_f, curve, nfe)。
      fobj    : 标量目标 f(x)->float (越小越优)
      pop0    : 初始种群 (nPop × dim), 各算法共享同一初始种群保证公平
      seed    : 随机种子(独立运行区分)
      track   : True 时额外返回机制对齐插桩(待办清单2 问题15):
        trace = dict(
          phase_dF   : {main/levy/de: 每代该阶段带来的 best_f 改进量Σ}
          phase_acc  : {main/levy/de: 每代该阶段候选接受次数}
          diversity  : 每代种群多样性(个体到质心的平均欧氏距离/√dim)
          tent_dF    : Tent 初始化替换带来的初始 cost 改进量Σ(标量)
          tent_n_rep : Tent 替换个体数
        )

    【实现修正声明】
      mu_tent=1.99(原2.0): μ=2 的 tent 映射在 float64 下等价二进制尾数左移,
        52 位尾数耗尽后(约54次迭代)轨道精确塌缩为 0 不动点; μ=1.99 保持混沌
        遍历性且无二进制退化。
      Tent 独立链(问题16): 原实现全种群共用一条混沌链(单链条), 插桩实测 61%
        个体落在同一轨道邻域(同质化); 现每个体持有独立初始值 r0_i, 各自迭代
        tent_chains 次后作为候选, 消除链条传染。
      Levy 步长缩放见主循环内注释。
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    pop = np.array(pop0, float).copy()
    nPop, dim = pop.shape
    span = ub - lb
    # IJS专用算子常数只计算一次；不消费随机数，也不改变候选生成顺序。
    levy_sigma_u = None
    if use_levy:
        num = math.gamma(1 + levy_beta) * np.sin(np.pi * levy_beta / 2)
        den = (math.gamma((1 + levy_beta) / 2) * levy_beta
               * 2 ** ((levy_beta - 1) / 2))
        levy_sigma_u = (num / den) ** (1 / levy_beta)
    # 初始种群候选相互独立：有序批量评价不改变个体位置或随机数序列。
    cost = evaluate_many_ordered(fobj, pop)
    nfe = nPop

    trace = dict(phase_dF={"main": [], "levy": [], "de": []},
                 phase_acc={"main": [], "levy": [], "de": []},
                 diversity=[], tent_dF=0.0, tent_n_rep=0) if track else None

    # ---- Tent 混沌映射初始化 (式47, 每个体独立链) ----
    if use_tent:
        r0 = rng.random((nPop, dim))          # 每个体独立初始值(问题16)
        for _ in range(tent_chains):
            r0 = np.where(r0 >= 0.5, mu_tent * (1 - r0), mu_tent * r0)
            r0 = np.clip(r0, 0, 1)
        tent_pop = _simplebounds(r0 * span + lb, lb, ub)
        tent_cost = evaluate_many_ordered(fobj, tent_pop, reject_above=cost)
        for i in range(nPop):
            cand = tent_pop[i]
            fc = tent_cost[i]; nfe += 1
            if fc < cost[i]:
                if track:
                    trace["tent_dF"] += float(cost[i] - fc)
                    trace["tent_n_rep"] += 1
                cost[i] = fc; pop[i] = cand

    idx = np.argmin(cost)
    best_x = pop[idx].copy(); best_f = cost[idx]
    curve = [best_f]

    for it in range(1, max_iter + 1):
        meanvl = pop.mean(axis=0)
        bidx = np.argmin(cost)
        best_sol = pop[bidx].copy(); best_cost = cost[bidx]

        acc_main = 0; f_before = best_cost
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
                    newsol = pop[i] + 0.1 * span * rng.random()
            newsol = _simplebounds(newsol, lb, ub)
            fnew = evaluate_one(fobj, newsol, reject_above=cost[i]); nfe += 1
            if fnew < cost[i]:
                pop[i] = newsol; cost[i] = fnew
                acc_main += 1
                if fnew < best_cost:
                    best_cost = fnew; best_sol = newsol.copy()
        if track:
            trace["phase_dF"]["main"].append(float(f_before - best_cost))
            trace["phase_acc"]["main"].append(acc_main)

        # ---- Levy 飞行 (式54-55) + 贪婪选择 ----
        # 【实现修正】步长按搜索域缩放: step = α·(ub−lb)·levy, α=0.01
        # (Yang & Deb 2009 布谷鸟搜索标准取值)。原实现无缩放, 每维步长中位
        # ≈0.26·域宽、P90 超过整个定义域, 插桩实测接受率仅 0.48%——Levy 阶段
        # 退化为无效随机重启; 缩放后成为围绕当前解的重尾局部探索。
        if use_levy:
            acc_levy = 0; f_before = best_cost
            # Levy候选只依赖各自阶段起点，可先按原循环顺序消费随机数，再有序批量评价。
            levy_pop = np.empty_like(pop)
            for i in range(nPop):
                step = 0.01 * span * _levy(
                    dim, levy_beta, rng, sigma_u=levy_sigma_u)
                cand = pop[i] + rng.random(dim) * step
                levy_pop[i] = _simplebounds(cand, lb, ub)
            levy_cost = evaluate_many_ordered(fobj, levy_pop,
                                               reject_above=cost.copy())
            for i in range(nPop):
                cand = levy_pop[i]
                fc = levy_cost[i]; nfe += 1
                if fc < cost[i]:
                    pop[i] = cand; cost[i] = fc
                    acc_levy += 1
                    if fc < best_cost:
                        best_cost = fc; best_sol = cand.copy()
            if track:
                trace["phase_dF"]["levy"].append(float(f_before - best_cost))
                trace["phase_acc"]["levy"].append(acc_levy)

        # ---- 差分进化 DE (式56-58) + 贪婪选择 ----
        if use_de:
            acc_de = 0; f_before = best_cost
            for i in range(nPop):
                r1, r2 = rng.integers(nPop), rng.integers(nPop)
                # 复用差分临时量，数学次序与原式完全一致，减少一次大数组分配。
                diff = pop[r1] - pop[r2]
                diff *= rng.random(dim)
                mutant = pop[i] + diff                                  # 变异(式56)
                mask = rng.random(dim) < CR                                # 交叉(式57)
                mutant[mask] = pop[i, mask]
                cand = mutant
                cand = _simplebounds(cand, lb, ub)
                fc = evaluate_one(fobj, cand, reject_above=cost[i]); nfe += 1
                if fc < cost[i]:                                           # 选择(式58)
                    pop[i] = cand; cost[i] = fc
                    acc_de += 1
                    if fc < best_cost:
                        best_cost = fc; best_sol = cand.copy()
            if track:
                trace["phase_dF"]["de"].append(float(f_before - best_cost))
                trace["phase_acc"]["de"].append(acc_de)

        if best_cost < best_f:
            best_f = best_cost; best_x = best_sol.copy()
        if record:
            curve.append(best_f)
        if track:
            centroid = pop.mean(axis=0)
            trace["diversity"].append(
                float(np.mean(np.linalg.norm(pop - centroid, axis=1))
                      / np.sqrt(dim)))

    out = dict(best_x=best_x, best_f=best_f, curve=np.array(curve), nfe=nfe,
               pop=pop.copy(), cost=cost.copy())
    if track:
        out["trace"] = trace
    return out


# 消融变体配置: 2³ 全因子设计(V1-V5 为原方案, V6-V8 为组合变体, 支持交互效应分解)
VARIANTS = {
    "V1_JS":         dict(use_tent=False, use_levy=False, use_de=False),
    "V2_JS+Tent":    dict(use_tent=True,  use_levy=False, use_de=False),
    "V3_JS+Levy":    dict(use_tent=False, use_levy=True,  use_de=False),
    "V4_JS+DE":      dict(use_tent=False, use_levy=False, use_de=True),
    "V6_JS+Tent+Levy": dict(use_tent=True,  use_levy=True,  use_de=False),
    "V7_JS+Tent+DE":   dict(use_tent=True,  use_levy=False, use_de=True),
    "V8_JS+Levy+DE":   dict(use_tent=False, use_levy=True,  use_de=True),
    "V5_IJS":        dict(use_tent=True,  use_levy=True,  use_de=True),
}
