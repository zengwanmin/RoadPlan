# -*- coding: utf-8 -*-
"""
run_ce.py — 目标: 直接最小化 C+E(全生命周期总账), 多种子并行 IJS。

与主实验(优化方案对比)的区别:
  · 标量目标就是 (C+E)/(C_A+E_A) + 惩罚 —— 不经熵权(目标即总账);
  · N_SEEDS 个独立种子并行(默认 8 核 8 种子), 取可行解中 C+E 最小者;
  · 每个种子的初始种群注入现状方案 x_A, 保证结果不劣于现状。

基线 M-A: 实测中线(δ=0) + 实测路面高程(既有设计线), 与主实验同口径。
"""
import os, json, time, argparse, multiprocessing as mp
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from params import LONG_STD_100
from data_loader import load_alignment
from objective_joint import (make_plane_context, objectives_joint, decode_joint,
                             run_ijs_two_phase, START_AMP_M,
                             DIM, N_MODE, M_PROF)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
POP = 200


def make_existing_x(pc):
    """现状方案: 平面 δ=0 + 实测路面高程按纵坡编码反解(与主实验同口径)。"""
    x = np.full(DIM, 0.5)
    d0 = decode_joint(x, pc)
    z_road = np.interp(d0["sta_ctrl"], pc["s_meas"], pc["gz_meas"])
    x[N_MODE] = np.clip(0.5 + (z_road[0] - d0["gz_ctrl"][0]) / (2.0 * START_AMP_M),
                        0.0, 1.0)
    g = np.diff(z_road) / np.diff(d0["sta_ctrl"])
    x[N_MODE + 1:] = np.clip(0.5 + g / (2.0 * LONG_STD_100["grade_max"]), 0.0, 1.0)
    return x


def _limit_dg(g, dg_lim=0.028):
    """前向-后向传递, 把相邻坡差 |Δg| 限制在 dg_lim 内(竖曲线约束代理, 留 7% 裕量)。"""
    g = np.array(g, float)
    for i in range(1, len(g)):
        g[i] = np.clip(g[i], g[i - 1] - dg_lim, g[i - 1] + dg_lim)
    for i in range(len(g) - 2, -1, -1):
        g[i] = np.clip(g[i], g[i + 1] - dg_lim, g[i + 1] + dg_lim)
    return g


def make_chord_x(pc, denom):
    """
    弦线播种(v2): 平面向"两端连线(弦)"方向偏移。全幅拟合经模态幅值截断后会产生
    扭结(实测 Rmin=1、里程反而变长), 故对正弦投影系数做 t∈[0,1] 缩放线搜索,
    每个 t 沿该线位重取地形限幅坡度(|i|≤4% 贴地)作纵断面, 取目标值最好的 t。
    """
    from objective_joint import MODE_AMPS, N_CTRL
    cx, cy, nx, ny = pc["cx"], pc["cy"], pc["nx"], pc["ny"]
    p0 = np.array([cx[0], cy[0]]); p1 = np.array([cx[-1], cy[-1]])
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    w = np.stack([cx - p0[0], cy - p0[1]], axis=1)
    foot = p0 + np.outer(w @ u, u)
    dvec = foot - np.stack([cx, cy], axis=1)
    delta_t = dvec[:, 0] * nx + dvec[:, 1] * ny
    uu = np.linspace(0.0, 1.0, N_CTRL)
    k = np.arange(1, N_MODE + 1, dtype=float)
    S = np.sin(np.outer(uu, k) * np.pi)
    a_full, *_ = np.linalg.lstsq(S, delta_t, rcond=None)

    gmax = LONG_STD_100["grade_max"]
    best = None
    for t in np.linspace(0.0, 1.0, 21):
        a = np.clip(t * a_full, -MODE_AMPS, MODE_AMPS)
        x = np.full(DIM, 0.5)
        x[:N_MODE] = np.clip(0.5 + a / (2.0 * MODE_AMPS), 0.0, 1.0)
        d = decode_joint(x, pc)
        g = _limit_dg(np.clip(np.diff(d["gz_ctrl"]) / np.diff(d["sta_ctrl"]),
                               -gmax, gmax))
        x[N_MODE + 1:] = 0.5 + g / (2.0 * gmax)
        C, E, pen, _ = objectives_joint(x, pc)
        f = (C + E) / denom + pen
        if best is None or f < best[0]:
            best = (f, t, x)
    return best[2], best[1]


_PC = None
_CTX = None


def _init(align, ctx):
    global _PC, _CTX
    _PC = make_plane_context(align)
    _CTX = ctx


def _solve(seed):
    pc, c = _PC, _CTX
    denom = c["denom"]

    def make_f(ps):
        def f(x):
            C, E, pen, _ = objectives_joint(x, pc, pen_scale=ps)
            return (C + E) / denom + pen
        return f

    rng = np.random.default_rng(seed)
    pop0 = rng.random((POP, DIM))
    pop0[0] = c["x_A"]
    if c.get("x_chord") is not None:
        pop0[1] = c["x_chord"]
    t0 = time.time()
    r = run_ijs_two_phase(make_f, np.zeros(DIM), np.ones(DIM), pop0,
                          c["max_iter"], seed)
    C, E, pen, info = objectives_joint(r["best_x"], pc)
    return dict(seed=seed, C=float(C), E=float(E), CE=float(C + E),
                pen=float(pen), L_km=info["L_km"], Rmin=info["Rmin"],
                C_TU=info["C_TU"], CB=info["CB"], CS=info["CS"], CR=info["CR"],
                CQ=info["CQ"], E_fuel=info["E_fuel"], E_ele=info["E_ele"],
                L_bridge_new=info["L_bridge_new"],
                L_tunnel_new=info["L_tunnel_new"],
                wall_min=(time.time() - t0) / 60.0,
                best_x=r["best_x"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--iter", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.iter, args.seeds = 5, 2

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    x_A = make_existing_x(pc)
    C_A, E_A, pen_A, infoA = objectives_joint(x_A, pc)
    denom = C_A + E_A
    print(f"[基线 M-A] C={C_A/1e8:.4f}亿 E={E_A/1e8:.4f}亿 "
          f"C+E={denom/1e8:.4f}亿 pen={pen_A:.2e}", flush=True)

    x_chord, t_chord = make_chord_x(pc, denom)
    C_c, E_c, pen_c, info_c = objectives_joint(x_chord, pc)
    print(f"[弦线种子 t={t_chord:.2f}] C+E={(C_c+E_c)/1e8:.4f}亿 "
          f"L={info_c['L_km']:.3f}km Rmin={info_c['Rmin']:.0f} pen={pen_c:.2e}",
          flush=True)

    ctx = dict(denom=denom, x_A=x_A, x_chord=x_chord, max_iter=args.iter)
    seeds = [3000 + 7 * k for k in range(args.seeds)]
    with mp.Pool(args.workers, initializer=_init, initargs=(align, ctx)) as pool:
        recs = []
        for r in pool.imap_unordered(_solve, seeds):
            recs.append(r)
            print(f"  seed={r['seed']} C+E={r['CE']/1e8:.4f}亿 "
                  f"({(r['CE']/denom-1)*100:+.2f}%) C={r['C']/1e8:.4f} "
                  f"E={r['E']/1e8:.4f} L={r['L_km']:.3f}km Rmin={r['Rmin']:.0f} "
                  f"pen={r['pen']:.1e} [{r['wall_min']:.1f}min]", flush=True)
    feas = [r for r in recs if r["pen"] < 1e-9]
    best = min(feas or recs, key=lambda r: r["CE"])
    imp = (1 - best["CE"] / denom) * 100
    print(f"[最优] seed={best['seed']} C+E={best['CE']/1e8:.4f}亿 "
          f"降幅 {imp:.2f}%  (可行解 {len(feas)}/{len(recs)})")
    out = dict(tag=args.tag, pop=POP, max_iter=args.iter, seeds=seeds,
               baseline=dict(C=C_A, E=E_A, CE=denom,
                             L_km=infoA["L_km"], CB=infoA["CB"],
                             C_TU=infoA["C_TU"]),
               runs=[{k: v for k, v in r.items() if k != "best_x"} for r in recs],
               best=best, improvement_pct=imp,
               wall_min=(time.time() - t0) / 60.0)
    fn = f"ce_{args.tag}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[完成] {fn}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
