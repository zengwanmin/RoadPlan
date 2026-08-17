# -*- coding: utf-8 -*-
"""Exact-NFE optimizers for confirmatory comparison and 2^3 IJS ablation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


VARIANTS = {
    "V1_JS": dict(use_tent=False, use_levy=False, use_de=False),
    "V2_JS+Tent": dict(use_tent=True, use_levy=False, use_de=False),
    "V3_JS+Levy": dict(use_tent=False, use_levy=True, use_de=False),
    "V4_JS+DE": dict(use_tent=False, use_levy=False, use_de=True),
    "V6_JS+Tent+Levy": dict(use_tent=True, use_levy=True, use_de=False),
    "V7_JS+Tent+DE": dict(use_tent=True, use_levy=False, use_de=True),
    "V8_JS+Levy+DE": dict(use_tent=False, use_levy=True, use_de=True),
    "V5_IJS": dict(use_tent=True, use_levy=True, use_de=True),
}


def _reflect(x, lb, ub):
    x = np.asarray(x, float).copy()
    width = ub - lb
    z = np.mod(x - lb, 2.0 * width)
    return lb + np.where(z <= width, z, 2.0 * width - z)


def _levy(dim, beta, rng):
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u = rng.normal(0.0, sigma, dim)
    v = rng.normal(0.0, 1.0, dim)
    return u / np.maximum(np.abs(v), np.finfo(float).tiny) ** (1 / beta)


@dataclass
class Recorder:
    budget: int
    pop_size: int
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.interval = max(self.pop_size, self.budget // 200)
        self.next_mark = self.pop_size

    def add(self, nfe, best_f, force=False):
        if force or nfe >= self.next_mark:
            self.history.append([int(nfe), float(best_f)])
            while self.next_mark <= nfe:
                self.next_mark += self.interval


def _initialise(fobj, pop0, budget):
    pop = np.asarray(pop0, float).copy()
    if budget < len(pop):
        raise ValueError("NFE budget must be at least the population size")
    cost = np.empty(len(pop))
    for i in range(len(pop)):
        cost[i] = fobj(pop[i])
    return pop, cost, len(pop)


def _finish(pop, cost, nfe, recorder, extra=None):
    i = int(np.argmin(cost))
    recorder.add(nfe, cost[i], force=True)
    out = dict(best_x=pop[i].copy(), best_f=float(cost[i]), nfe=int(nfe),
               curve=recorder.history, pop=pop, cost=cost)
    if extra:
        out.update(extra)
    return out


def run_js(fobj, lb, ub, pop0, budget, seed, *, use_tent=False,
           use_levy=False, use_de=False, CR=0.5, levy_beta=1.5,
           mu_tent=1.99, beta_d=3.0, C0=0.5, tent_chains=10):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    pop, cost, nfe = _initialise(fobj, pop0, budget)
    n, dim = pop.shape
    rec = Recorder(budget, n)
    rec.add(nfe, float(cost.min()))
    accepted = dict(tent=0, main=0, levy=0, de=0)

    if use_tent and nfe < budget:
        r = rng.random((n, dim))
        for _ in range(tent_chains):
            r = np.where(r >= 0.5, mu_tent * (1.0 - r), mu_tent * r)
            r = np.clip(r, 0.0, 1.0)
        for i in range(n):
            if nfe >= budget:
                break
            cand = _reflect(lb + r[i] * (ub - lb), lb, ub)
            fc = fobj(cand); nfe += 1
            if fc < cost[i]:
                pop[i], cost[i] = cand, fc
                accepted["tent"] += 1
        rec.add(nfe, float(cost.min()))

    generation = 0
    while nfe < budget:
        generation += 1
        progress = nfe / budget
        mean = pop.mean(axis=0)
        bi = int(np.argmin(cost))
        best_x, best_f = pop[bi].copy(), float(cost[bi])

        for i in range(n):
            if nfe >= budget:
                break
            Ar = (1.0 - progress) * (2.0 * rng.random() - 1.0)
            if abs(Ar) >= C0:
                cand = pop[i] + rng.random(dim) * (
                    best_x - beta_d * rng.random() * mean)
            elif rng.random() <= 1.0 - Ar:
                j = int(rng.integers(n - 1))
                if j >= i:
                    j += 1
                step = pop[i] - pop[j]
                if cost[j] < cost[i]:
                    step = -step
                cand = pop[i] + rng.random(dim) * step
            else:
                cand = pop[i] + 0.1 * (ub - lb) * rng.random()
            cand = _reflect(cand, lb, ub)
            fc = fobj(cand); nfe += 1
            if fc < cost[i]:
                pop[i], cost[i] = cand, fc
                accepted["main"] += 1
                if fc < best_f:
                    best_x, best_f = cand.copy(), float(fc)
        rec.add(nfe, float(cost.min()))

        if use_levy:
            for i in range(n):
                if nfe >= budget:
                    break
                step = 0.01 * (ub - lb) * _levy(dim, levy_beta, rng)
                cand = _reflect(pop[i] + rng.random(dim) * step, lb, ub)
                fc = fobj(cand); nfe += 1
                if fc < cost[i]:
                    pop[i], cost[i] = cand, fc
                    accepted["levy"] += 1
            rec.add(nfe, float(cost.min()))

        if use_de:
            for i in range(n):
                if nfe >= budget:
                    break
                r1, r2 = rng.integers(n, size=2)
                mutant = pop[i] + rng.random(dim) * (pop[r1] - pop[r2])
                mask = rng.random(dim) < CR
                cand = _reflect(np.where(mask, pop[i], mutant), lb, ub)
                fc = fobj(cand); nfe += 1
                if fc < cost[i]:
                    pop[i], cost[i] = cand, fc
                    accepted["de"] += 1
            rec.add(nfe, float(cost.min()))

    return _finish(pop, cost, nfe, rec,
                   dict(generations=generation, accepted=accepted))


def run_ga(fobj, lb, ub, pop0, budget, seed, p1=0.8, p2=0.2):
    """Repository GA.m translation with only an exact-NFE stop added."""
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    pop, cost, nfe = _initialise(fobj, pop0, budget)
    n, dim = pop.shape
    xp, yp = pop.copy(), cost.copy()
    gi = int(np.argmin(cost)); global_x, global_f = pop[gi].copy(), float(cost[gi])
    rec = Recorder(budget, n); rec.add(nfe, float(cost.min()))
    generations = 0
    while nfe < budget:
        generations += 1
        parent = np.empty(2 * n, int)
        for i in range(2 * n):
            k = int(np.ceil(rng.integers(1, n + 1) * p1))
            chosen = rng.choice(n, size=max(k, 1), replace=False)
            parent[i] = chosen[np.argmin(cost[chosen])]
        offspring = pop.copy()
        for i in range(n):
            offspring[i] = pop[parent[i]]
            if rng.random() < p1:
                mask = rng.random(dim) > 0.5
                offspring[i, mask] = pop[parent[i + n], mask]
            if rng.random() < p2:
                mask = rng.random(dim) > 0.5
                random_gene = rng.random(dim) * (ub - lb) + lb
                offspring[i, mask] = random_gene[mask]
        offspring = np.clip(offspring, lb, ub)
        for i in range(n):
            if nfe >= budget:
                break
            fc = fobj(offspring[i]); nfe += 1
            if fc < yp[i]:
                xp[i], yp[i] = offspring[i], fc
                if fc < global_f:
                    global_x, global_f = offspring[i].copy(), float(fc)
        pop, cost = xp.copy(), yp.copy()
        rec.add(nfe, global_f)
    wi = int(np.argmax(cost)); pop[wi], cost[wi] = global_x, global_f
    return _finish(pop, cost, nfe, rec, dict(generations=generations))


def run_pso(fobj, lb, ub, pop0, budget, seed, w=0.8, c1=2.0, c2=2.0):
    """Repository PSO.m translation with only an exact-NFE stop added."""
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    pop, cost, nfe = _initialise(fobj, pop0, budget)
    n, dim = pop.shape
    velocity = rng.standard_normal(pop.shape)
    pbest, pbest_cost = pop.copy(), cost.copy()
    gi = int(np.argmin(cost)); global_x, global_f = pop[gi].copy(), float(cost[gi])
    rec = Recorder(budget, n); rec.add(nfe, float(cost.min()))
    generations = 0
    while nfe < budget:
        generations += 1
        r1, r2 = rng.random(pop.shape), rng.random(pop.shape)
        for i in range(n):
            if nfe >= budget:
                break
            velocity[i] = (w * velocity[i] + c1 * r1[i] * (global_x - pop[i])
                           + c2 * r2[i] * (pbest[i] - pop[i]))
            pop[i] = np.clip(pop[i] + velocity[i], lb, ub)
            fc = fobj(pop[i]); nfe += 1
            cost[i] = fc
            if fc < pbest_cost[i]:
                pbest[i], pbest_cost[i] = pop[i].copy(), fc
                if fc < global_f:
                    global_x, global_f = pop[i].copy(), float(fc)
        rec.add(nfe, global_f)
    wi = int(np.argmax(pbest_cost)); pbest[wi], pbest_cost[wi] = global_x, global_f
    return _finish(pbest, pbest_cost, nfe, rec, dict(generations=generations))


def run_gwo(fobj, lb, ub, pop0, budget, seed):
    """Repository GWO.m translation with NFE-normalized time and exact stop."""
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    pop, cost, nfe = _initialise(fobj, pop0, budget)
    n, dim = pop.shape
    order = np.argsort(cost)
    alpha, beta, delta = pop[order[0]].copy(), pop[order[1]].copy(), pop[order[2]].copy()
    alpha_f, beta_f, delta_f = float(cost[order[0]]), float(cost[order[1]]), float(cost[order[2]])
    global_x, global_f = alpha.copy(), alpha_f
    rec = Recorder(budget, n); rec.add(nfe, global_f)
    generations = 0
    while nfe < budget:
        generations += 1
        a = 2.0 * (1.0 - (nfe - n) / max(budget - n, 1))
        for i in range(n):
            if nfe >= budget:
                break
            A1 = 2 * a * rng.random(dim) - a; C1 = 2 * rng.random(dim)
            A2 = 2 * a * rng.random(dim) - a; C2 = 2 * rng.random(dim)
            A3 = 2 * a * rng.random(dim) - a; C3 = 2 * rng.random(dim)
            X1 = alpha - A1 * np.abs(C1 * alpha - pop[i])
            X2 = beta - A2 * np.abs(C2 * beta - pop[i])
            X3 = delta - A3 * np.abs(C3 * delta - pop[i])
            pop[i] = np.clip((X1 + X2 + X3) / 3.0, lb, ub)
            fc = fobj(pop[i]); nfe += 1; cost[i] = fc
            if fc < alpha_f:
                delta_f, delta = beta_f, beta.copy()
                beta_f, beta = alpha_f, alpha.copy()
                alpha_f, alpha = float(fc), pop[i].copy()
                global_x, global_f = alpha.copy(), alpha_f
            elif fc < beta_f:
                delta_f, delta = beta_f, beta.copy()
                beta_f, beta = float(fc), pop[i].copy()
            elif fc < delta_f:
                delta_f, delta = float(fc), pop[i].copy()
        rec.add(nfe, global_f)
    wi = int(np.argmax(cost))
    pop[wi], cost[wi] = global_x, global_f
    return _finish(pop, cost, nfe, rec, dict(generations=generations))


def _fronts_and_rank(F):
    """Vectorized nondominated sorting for a small two-objective population."""
    F = np.asarray(F, float)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominates = le & lt
    dominated_count = dominates.sum(axis=0).astype(int)
    remaining = np.ones(len(F), dtype=bool)
    fronts, rank = [], np.full(len(F), -1, dtype=int)
    level = 0
    current = np.flatnonzero((dominated_count == 0) & remaining)
    while len(current):
        fronts.append(current)
        rank[current] = level
        remaining[current] = False
        dominated_count -= dominates[current].sum(axis=0).astype(int)
        level += 1
        current = np.flatnonzero((dominated_count == 0) & remaining)
    if remaining.any():
        raise RuntimeError("Nondominated sorting did not consume the population")
    return fronts, rank


def _crowding(F, front):
    front = np.asarray(front, dtype=int)
    d = np.zeros(len(front), float)
    if len(front) <= 2:
        d[:] = np.inf
        return d
    values = F[front]
    for k in range(values.shape[1]):
        order = np.argsort(values[:, k])
        d[order[0]] = d[order[-1]] = np.inf
        span = values[order[-1], k] - values[order[0], k]
        if span > 1e-15:
            d[order[1:-1]] += ((values[order[2:], k] - values[order[:-2], k])
                               / span)
    return d


def _environmental_select(P, F, n):
    fronts, _ = _fronts_and_rank(F)
    chosen = []
    for front in fronts:
        if len(chosen) + len(front) <= n:
            chosen.extend(front.tolist())
        else:
            distance = _crowding(F, front)
            order = np.argsort(-distance)
            chosen.extend(front[order[:n - len(chosen)]].tolist())
            break
    idx = np.asarray(chosen, dtype=int)
    P2, F2 = P[idx], F[idx]
    fronts2, rank2 = _fronts_and_rank(F2)
    crowd2 = np.zeros(n, float)
    for front in fronts2:
        crowd2[front] = _crowding(F2, front)
    return P2, F2, rank2, crowd2


def run_nsgaii(fobj_bi, lb, ub, pop0, budget, seed, weights,
               crossover_rate=0.9, mutation_rate=0.1, eta=20.0):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    pop = np.asarray(pop0, float).copy()
    n, dim = pop.shape
    if budget < n:
        raise ValueError("NFE budget must be at least the population size")
    F = np.array([fobj_bi(x) for x in pop], dtype=float)
    nfe = n
    fronts, rank = _fronts_and_rank(F)
    crowd = np.zeros(n, float)
    for front in fronts:
        crowd[front] = _crowding(F, front)
    weights = np.asarray(weights, float)
    scalar = F @ weights
    bi = int(np.argmin(scalar))
    best_x, best_vec, best_f = pop[bi].copy(), F[bi].copy(), float(scalar[bi])
    rec = Recorder(budget, n); rec.add(nfe, best_f)
    generations = 0

    def tournament():
        a, b = rng.integers(n, size=2)
        if rank[a] != rank[b]:
            return int(a if rank[a] < rank[b] else b)
        return int(a if crowd[a] > crowd[b] else b)

    while nfe < budget:
        generations += 1
        Q = np.empty_like(pop)
        for i in range(0, n, 2):
            p1, p2 = pop[tournament()], pop[tournament()]
            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < crossover_rate:
                u = rng.random(dim)
                beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)),
                                (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
                c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
            for child in (c1, c2):
                mask = rng.random(dim) < mutation_rate
                if mask.any():
                    child[mask] += ((rng.random(mask.sum()) - 0.5) * 0.2
                                    * (ub[mask] - lb[mask]))
            Q[i] = np.clip(c1, lb, ub)
            if i + 1 < n:
                Q[i + 1] = np.clip(c2, lb, ub)
        m = min(n, budget - nfe)
        Q, FQ = Q[:m], np.array([fobj_bi(x) for x in Q[:m]], dtype=float)
        nfe += m
        sq = FQ @ weights
        qi = int(np.argmin(sq))
        if sq[qi] < best_f:
            best_x, best_vec, best_f = Q[qi].copy(), FQ[qi].copy(), float(sq[qi])
        pop, F, rank, crowd = _environmental_select(
            np.vstack((pop, Q)), np.vstack((F, FQ)), n)
        rec.add(nfe, best_f)

    scalar = F @ weights
    wi = int(np.argmax(scalar))
    pop[wi], F[wi] = best_x, best_vec
    # _finish only needs a scalar cost vector; preserve the archive-selected best.
    scalar[wi] = best_f
    return _finish(pop, scalar, nfe, rec,
                   dict(generations=generations, front_F=F))


def run_algorithm(name, fobj, lb, ub, pop0, budget, seed, weights=None):
    if name in VARIANTS:
        return run_js(fobj, lb, ub, pop0, budget, seed, **VARIANTS[name])
    if name == "GA":
        return run_ga(fobj, lb, ub, pop0, budget, seed)
    if name == "PSO":
        return run_pso(fobj, lb, ub, pop0, budget, seed)
    if name == "GWO":
        return run_gwo(fobj, lb, ub, pop0, budget, seed)
    if name == "NSGA-II":
        if weights is None:
            raise ValueError("NSGA-II requires final scalarization weights")
        return run_nsgaii(fobj, lb, ub, pop0, budget, seed, weights=weights)
    raise KeyError(f"Unknown algorithm: {name}")
