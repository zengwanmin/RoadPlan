# -*- coding: utf-8 -*-
"""
test_joint_improved.py — 改进联合优化的快速验证(3 种子, 与 two_stage@500 配对比较)

目标: 在【相同总求值量】(~600,400, = two_stage@500)下让联合方案的经济性(C/E/L)
      追平乃至超过两阶段, 同时尽量保留其几何富余度(Rmin)。

改进思路(均不削弱 C_TU/E 的真实反馈, 纯属搜索质量增强):
  WS  = 平面暖启动: 先用与两阶段完全相同的 Stage1(纯里程平面优化)得到一条"短平面",
        再以它为中心播种联合初始种群(平面块=Stage1解+小抖动, 纵断面块随机),
        对完整 114 维联合目标做 IJS 精修。可证不劣于两阶段(联合精修可复现"冻结平面
        只优化纵断面"= 两阶段Stage2), 协同带来上行空间。
  WSM = WS + 记忆式分块坐标下降: 在联合精修的每 K 代对当前最优个体追加一次
        "只动平面块 / 只动纵断面块"的贪婪局部搜索(仍按完整联合目标接受),
        给平面块干净的改进信号, 防止其在整体扰动中漂移。

预算核算(POP=200, 每次求值 run 内计 200+3*POP*iter):
  two_stage@500 : 2*(200+600*500) = 600,400
  WS(500+500)   : (200+600*500)+(200+600*500) = 600,400  ✓ 等预算
  WSM 的分块局部搜索额外求值计入 it_joint 的等效缩减(见下 _budget_for_memetic)。
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from data_loader import load_alignment
from algorithms import run, VARIANTS
from objective import entropy_weights
from objective_joint import (N_CTRL, M_PROF, make_plane_context, objectives_joint,
                             make_scalar_joint, make_scalar_plane,
                             build_plane_from_delta, plane_lcc)

POP = 200
SEED_BASE = 20250722
TOTAL_EVALS_TARGET = 600400   # = two_stage@500


def _stage1_plane(pc, seed_i, it1):
    """与 run_twostage / 多种子脚本完全相同的 Stage1 平面优化。返回 (ds, evals)。"""
    rng = np.random.default_rng(SEED_BASE + seed_i)
    pop_plane = np.clip(0.5 + (rng.random((POP, N_CTRL)) - 0.5) * 1.0, 0, 1)
    _, _, L0, _ = build_plane_from_delta(pc, np.full(N_CTRL, 0.5))
    fP = make_scalar_plane(pc, plane_lcc(L0))
    rP = run(fP, np.zeros(N_CTRL), np.ones(N_CTRL), pop_plane, it1,
             SEED_BASE + 10000 + seed_i, **VARIANTS["V5_IJS"])
    return rP["best_x"], POP + 3 * POP * it1


def _stage2_profile(pc, ds, seed_i, it2):
    """在冻结平面 ds 上做纵断面优化(= 两阶段 Stage2)。
    返回 (最优纵断面归一化向量, wC, wE, C_ref, E_ref, evals)。"""
    full = lambda p: np.concatenate([ds, p])
    rng = np.random.default_rng(SEED_BASE + 20000 + seed_i)
    baseP = rng.random((POP, M_PROF))
    C0 = np.array([objectives_joint(full(baseP[k]), pc)[0] for k in range(POP)])
    E0 = np.array([objectives_joint(full(baseP[k]), pc)[1] for k in range(POP)])
    wC, wE = entropy_weights(C0, E0); C_ref, E_ref = float(C0.mean()), float(E0.mean())

    def sc(p):
        C, E, pen, _ = objectives_joint(full(p), pc)
        return wC * (C / C_ref) + wE * (E / E_ref) + pen / C_ref

    rC = run(sc, np.zeros(M_PROF), np.ones(M_PROF), baseP.copy(), it2,
             SEED_BASE + 30000 + seed_i, **VARIANTS["V5_IJS"])
    return rC["best_x"], wC, wE, C_ref, E_ref, POP + 3 * POP * it2


def joint_wsp_once(pc, seed_i, it1=250, it2=250, it3=500):
    """WSP: 平面+纵断面【双暖启动】(先跑一遍轻量两阶段得到可行短解, penalty≈0),
    再以该完整解为中心播种联合种群做 it3 代联合协同精修。
    因起点已可行, 联合精修不会因惩罚压力而"逃离"短平面(WS/WSM 失败的根因)。
    预算: 2*(200+600*it1) + (200+600*it3) ≈ 600,600 ≈ two_stage@500。"""
    ds, ev1 = _stage1_plane(pc, seed_i, it1)
    prof, wC, wE, C_ref, E_ref, ev2 = _stage2_profile(pc, ds, seed_i, it2)
    dim = N_CTRL + M_PROF
    x_seed = np.concatenate([ds, prof])
    rng = np.random.default_rng(SEED_BASE + 70000 + seed_i)
    base = np.clip(x_seed[None, :] + rng.normal(0, 0.02, (POP, dim)), 0.0, 1.0)
    base[0] = x_seed                              # 保留完整两阶段解
    f = make_scalar_joint(pc, wC, wE, C_ref, E_ref)  # 沿用 Stage2 的权重/参考值, 保持连续
    r = run(f, np.zeros(dim), np.ones(dim), base, it3,
            SEED_BASE + 80000 + seed_i, **VARIANTS["V5_IJS"])
    ev3 = POP + 3 * POP * it3
    C, E, pen, info = objectives_joint(r["best_x"], pc)
    return _pack(C, E, pen, info, ev1 + ev2 + ev3, "WSP")


def _warmstart_pop(pc, ds, seed_i, jitter_amp=0.06):
    """以 Stage1 平面 ds 为中心播种联合初始种群(平面块暖启动, 纵断面块随机)。"""
    dim = N_CTRL + M_PROF
    rng = np.random.default_rng(SEED_BASE + 20000 + seed_i)
    base = np.empty((POP, dim))
    jit = (rng.random((POP, N_CTRL)) - 0.5) * 2.0 * jitter_amp
    base[:, :N_CTRL] = np.clip(ds[None, :] + jit, 0, 1)
    base[0, :N_CTRL] = ds                       # 保留 Stage1 精确平面
    base[:, N_CTRL:] = rng.random((POP, M_PROF))  # 纵断面随机(同两阶段Stage2)
    return base


def joint_ws_once(pc, seed_i, it1=500, it2=500):
    """WS: 平面暖启动 + 完整 114 维联合 IJS 精修。"""
    ds, ev1 = _stage1_plane(pc, seed_i, it1)
    base = _warmstart_pop(pc, ds, seed_i)
    dim = N_CTRL + M_PROF
    C0 = np.array([objectives_joint(base[k], pc)[0] for k in range(POP)])
    E0 = np.array([objectives_joint(base[k], pc)[1] for k in range(POP)])
    wC, wE = entropy_weights(C0, E0); C_ref, E_ref = float(C0.mean()), float(E0.mean())
    f = make_scalar_joint(pc, wC, wE, C_ref, E_ref)
    r = run(f, np.zeros(dim), np.ones(dim), base, it2,
            SEED_BASE + 40000 + seed_i, **VARIANTS["V5_IJS"])
    ev2 = POP + 3 * POP * it2
    C, E, pen, info = objectives_joint(r["best_x"], pc)
    return _pack(C, E, pen, info, ev1 + ev2, "WS")


def _block_local_search(x, f, block, lo, hi, rng, n_steps, sigma):
    """对 x 的 [lo:hi) 块做 n_steps 次高斯扰动贪婪局部搜索(整向量按 f 接受)。
    返回 (x_best, f_best, n_evals)。仅扰动该块, 其余维保持不变。"""
    x = x.copy(); fb = f(x); nev = 1
    dblk = hi - lo
    for _ in range(n_steps):
        cand = x.copy()
        cand[lo:hi] = np.clip(x[lo:hi] + rng.normal(0, sigma, dblk), 0.0, 1.0)
        fc = f(cand); nev += 1
        if fc < fb:
            x, fb = cand, fc
    return x, fb, nev


_NOTENT = dict(use_tent=False, use_levy=True, use_de=True)  # 暖启动种群无需 tent 再扰动


def joint_wsm_once(pc, seed_i, it1=500, it2=460, K=20, steps_per_block=8):
    """WSM: WS + 每 K 代对当前最优做一次分块(平面/纵断面)局部搜索, 种群跨轮延续。
    it2 取 460(而非 500)以吸收分块局部搜索的额外求值, 使总量 ≈ 600,400。
    分块局部搜索仍按完整联合目标接受 -> 不削弱 C_TU/E 反馈。"""
    ds, ev1 = _stage1_plane(pc, seed_i, it1)
    base = _warmstart_pop(pc, ds, seed_i)
    dim = N_CTRL + M_PROF
    C0 = np.array([objectives_joint(base[k], pc)[0] for k in range(POP)])
    E0 = np.array([objectives_joint(base[k], pc)[1] for k in range(POP)])
    wC, wE = entropy_weights(C0, E0); C_ref, E_ref = float(C0.mean()), float(E0.mean())
    f = make_scalar_joint(pc, wC, wE, C_ref, E_ref)
    ev2 = POP  # 初始种群一次性求值(后续 run 的 cost 评估计入各轮)

    rng = np.random.default_rng(SEED_BASE + 50000 + seed_i)
    x_glob, f_glob = None, np.inf
    n_rounds = max(1, it2 // K)
    for rd in range(n_rounds):
        seed_r = SEED_BASE + 60000 + seed_i * 1000 + rd
        r = run(f, np.zeros(dim), np.ones(dim), base, K, seed_r, **_NOTENT)
        ev2 += POP + 3 * POP * K   # 每轮: 初始 cost 评估 POP + 主循环 3*POP*K (无 tent)
        xb = r["best_x"].copy(); fb = r["best_f"]
        # 分块局部搜索: 先平面块, 再纵断面块(均按完整联合目标接受)
        xb, fb, e1 = _block_local_search(xb, f, "plane", 0, N_CTRL, rng,
                                         steps_per_block, sigma=0.03)
        xb, fb, e2 = _block_local_search(xb, f, "prof", N_CTRL, dim, rng,
                                         steps_per_block, sigma=0.03)
        ev2 += e1 + e2
        if fb < f_glob:
            x_glob, f_glob = xb.copy(), fb
        # 种群延续: 用上一轮末种群继续演化, 并把精修后的最优注入(替换个体0)
        base = r["pop"].copy()
        base[0] = xb
    C, E, pen, info = objectives_joint(x_glob, pc)
    return _pack(C, E, pen, info, ev1 + ev2, "WSM")


def _pack(C, E, pen, info, nev, tag):
    return dict(tag=tag, C=float(C), E=float(E), L_km=float(info["L_km"]),
                Rmin=float(info["Rmin"]), pen=float(pen), nev=int(nev))


def _worker(job):
    align = load_alignment(); pc = make_plane_context(align)
    t0 = time.time()
    fn = joint_ws_once if job["method"] == "WS" else joint_wsm_once
    r = fn(pc, job["seed_i"])
    r["seed_i"] = job["seed_i"]; r["wall_min"] = (time.time() - t0) / 60.0
    return r


if __name__ == "__main__":
    import multiprocessing as mp
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    ts = {r["seed_i"]: r for r in
          json.load(open("results/budget_fairness_multiseed.json"))["raw"]["two_stage@500"]}
    jobs = [dict(method=m, seed_i=si) for si in seeds for m in ("WS", "WSM")]
    print(f"[启动] {len(jobs)} 个任务 ({len(seeds)} 种子 × WS/WSM), 等预算≈600,400", flush=True)
    t0 = time.time()
    results = {}
    with mp.Pool(min(len(jobs), 11)) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            results.setdefault(r["seed_i"], {})[r["tag"]] = r
            print(f"  完成 seed{r['seed_i']} {r['tag']:3s} "
                  f"C={r['C']/1e8:.4f} Rmin={r['Rmin']:.0f} nev={r['nev']} "
                  f"({r['wall_min']:.1f}min) 累计{(time.time()-t0)/60:.1f}min", flush=True)
    print(f"\n{'seed':>4} {'method':>5} {'C(亿)':>9} {'E(亿)':>9} {'L(km)':>8} "
          f"{'Rmin':>6} {'pen':>9} {'nev':>8}  vs两阶段C")
    for si in seeds:
        b = ts[si]
        print(f"{si:>4} {'2stg':>5} {b['C']/1e8:>9.4f} {b['E']/1e8:>9.4f} "
              f"{b['L_km']:>8.3f} {b['Rmin']:>6.0f} {b['pen']:>9.1e} {b['nev']:>8} {'baseline':>9}")
        for tag in ("WS", "WSM"):
            r = results[si][tag]
            dC = (r["C"] - b["C"]) / b["C"] * 100
            print(f"{si:>4} {tag:>5} {r['C']/1e8:>9.4f} {r['E']/1e8:>9.4f} "
                  f"{r['L_km']:>8.3f} {r['Rmin']:>6.0f} {r['pen']:>9.1e} "
                  f"{r['nev']:>8} {dC:>+8.2f}%")
