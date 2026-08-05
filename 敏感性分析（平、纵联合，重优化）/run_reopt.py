# -*- coding: utf-8 -*-
"""
run_reopt.py — 敏感性分析【每点重优化】主程序（实验四·重优化版）

与固定方案口径不同：在【每一个参数采样点】上都重新跑一次完整的平纵联合 IJS 寻优
(pop=200, iter=500)，记录该点真正的最优线形与 C、E（亿元）、里程 L、最小平曲线半径、
约束惩罚。四项：
  项目①: 交通量增长率 {0-10%} × EV 渗透率 {0-100%}  -> 每点重优化 C、E   (§6.5.1)
  项目②: 油价增长率 {0-5%} × 电价增长率 {0-5%}        -> 每点重优化 E     (§6.5.2)
  项目③: 燃油节油率 {0-5%} × 电动节能率 {0-5%}        -> 每点重优化 E     (§6.5.3)
  项目④: 目标权重 w1 {0.1-0.9} 的重优化 Pareto 前沿(权重分区+膝点)        (§5.3.4)

公式: 林坤锐学位论文(式号见 objective*.py)。数据: 数据.xlsx (北环高速实测)。
能耗单位: 全生命周期亿元(与 C 同口径)。pop=200, iter=500。
并行: multiprocessing.Pool(N_WORKERS)。用法:
  python3 run_reopt.py            # 正式全量 (pop200/iter500, ~2-2.5h)
  python3 run_reopt.py --smoke    # 冒烟测试 (iter=5, 少量点, 验证管线)
"""
import os, json, time, argparse, multiprocessing as mp
import numpy as np

from params import ALGO, TRAFFIC, LCC
from data_loader import load_alignment
from algorithms import run, VARIANTS
from objective import entropy_weights
from objective_joint import make_plane_context, N_CTRL, M_PROF
from objective_reopt import objectives_reopt, make_scalar_reopt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
YI = 1e8
N_WORKERS = 7          # 8 核留 1 核余量
SEED_BASE = 20250722   # 复现用固定种子基数

# --- worker 进程内的全局(由 initializer 设定, 兼容 macOS spawn) ---
_PC = None
_DIM = N_CTRL + M_PROF
_LB = np.zeros(_DIM); _UB = np.ones(_DIM)
POP_SIZE = ALGO["pop_size"]      # 主进程默认; worker 由 initargs 覆盖
MAX_ITER = ALGO["max_iter"]


def _init_worker(align, pop_size, max_iter):
    """worker 进程初始化: 建 pc(含 cKDTree) + 设定 pop/iter, 每进程只做一次。
    (macOS 用 spawn, 子进程会重新 import 本模块, 故 pop/iter 必须显式下发。)"""
    global _PC, POP_SIZE, MAX_ITER
    _PC = make_plane_context(align)
    POP_SIZE = pop_size
    MAX_ITER = max_iter


def _base_pop(seed):
    """联合优化初始种群(与优化方案对比一致: 平面全走廊探索, 纵断面全域)。"""
    rng = np.random.default_rng(seed)
    base = np.empty((POP_SIZE, _DIM))
    base[:, :N_CTRL] = 0.5 + (rng.random((POP_SIZE, N_CTRL)) - 0.5) * 1.0
    base[:, N_CTRL:] = rng.random((POP_SIZE, M_PROF))
    return np.clip(base, 0, 1)


def _optimize_one(task):
    """
    单个采样点的完整重优化(在 worker 进程执行)。
    task = dict(kind, idx, seed, P, w1(可选))
      kind: 'grid'  -> 熵权法自适应权重, 用 P 参数化目标
            'front' -> 指定权重 w1 的 Pareto 前沿点(P 为基准)
    返回记录 dict(含 C,E,L,Rmin,pen,ml,kwh,wC,wE 及 task 索引)。
    """
    pc = _PC
    P = task["P"]; seed = task["seed"]
    base = _base_pop(seed)
    # 用该点参数化目标在初始种群上算 (C,E) -> 熵权法客观权重(式5.3-5.4)
    C0 = np.empty(POP_SIZE); E0 = np.empty(POP_SIZE)
    for i in range(POP_SIZE):
        c, e, _, _ = objectives_reopt(base[i], pc, P)
        C0[i] = c; E0[i] = e
    if task["kind"] == "front":
        wC = task["w1"]; wE = 1.0 - task["w1"]
    else:
        wC, wE = entropy_weights(C0, E0)
    C_ref, E_ref = float(C0.mean()), float(E0.mean())
    f = make_scalar_reopt(pc, wC, wE, C_ref, E_ref, P)
    r = run(f, _LB, _UB, base.copy(), MAX_ITER, seed + 1, **VARIANTS["V5_IJS"])
    C, E, pen, info = objectives_reopt(r["best_x"], pc, P)
    return dict(kind=task["kind"], item=task["item"], idx=task["idx"],
                C=float(C), E=float(E), pen=float(pen),
                L_km=float(info["L_km"]), Rmin=float(info["Rmin"]),
                ml=float(info["ml"]), kwh=float(info["kwh"]),
                wC=float(wC), wE=float(wE),
                P={k: float(v) for k, v in P.items()})


def build_tasks(grids):
    """构造全部采样点任务列表(带全局序号->唯一 seed)。"""
    tasks = []; gid = 0
    tr = grids["traffic"]; ev = grids["ev"]
    fp = grids["fuel_price"]; ep = grids["elec_price"]
    fs = grids["fuel_save"]; es = grids["elec_save"]
    w1s = grids["w1"]

    # 项目①: 交通量 × EV
    for i, rj in enumerate(tr):
        for j, p in enumerate(ev):
            tasks.append(dict(kind="grid", item=1, idx=(i, j), seed=SEED_BASE + gid,
                              P=dict(ev=float(p), traffic_growth=float(rj),
                                     fuel_price_growth=0.0, elec_price_growth=0.0,
                                     fuel_save=0.0, elec_save=0.0))); gid += 1
    # 项目②: 油价 × 电价 (EV 固定基准 n2_ev)
    for i, a in enumerate(fp):
        for j, b in enumerate(ep):
            tasks.append(dict(kind="grid", item=2, idx=(i, j), seed=SEED_BASE + gid,
                              P=dict(ev=TRAFFIC["n2_ev"], traffic_growth=0.0,
                                     fuel_price_growth=float(a), elec_price_growth=float(b),
                                     fuel_save=0.0, elec_save=0.0))); gid += 1
    # 项目③: 节油率 × 节能率
    for i, a in enumerate(fs):
        for j, b in enumerate(es):
            tasks.append(dict(kind="grid", item=3, idx=(i, j), seed=SEED_BASE + gid,
                              P=dict(ev=TRAFFIC["n2_ev"], traffic_growth=0.0,
                                     fuel_price_growth=0.0, elec_price_growth=0.0,
                                     fuel_save=float(a), elec_save=float(b)))); gid += 1
    # 项目④: 权重前沿 (基准参数, 指定 w1)
    for k, w1 in enumerate(w1s):
        tasks.append(dict(kind="front", item=4, idx=(k,), seed=SEED_BASE + gid,
                          w1=float(w1),
                          P=dict(ev=TRAFFIC["n2_ev"], traffic_growth=0.0,
                                 fuel_price_growth=0.0, elec_price_growth=0.0,
                                 fuel_save=0.0, elec_save=0.0))); gid += 1
    # 图D6/D7/D8: 三段权重重优化(成本为主/能耗为主/折中), 基准参数
    for item_id, key in [(5, "w1_cost"), (6, "w1_energy"), (7, "w1_balanced")]:
        for k, w1 in enumerate(grids.get(key, [])):
            tasks.append(dict(kind="front", item=item_id, idx=(k,), seed=SEED_BASE + gid,
                              w1=float(w1),
                              P=dict(ev=TRAFFIC["n2_ev"], traffic_growth=0.0,
                                     fuel_price_growth=0.0, elec_price_growth=0.0,
                                     fuel_save=0.0, elec_save=0.0))); gid += 1
    return tasks


def main():
    global POP_SIZE, MAX_ITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="冒烟测试(iter=5,少量点)")
    args = ap.parse_args()

    t_all = time.time()
    align = load_alignment()
    print(f"[数据] 北环高速 {align['total_km']:.3f} km  dim={_DIM} "
          f"(平面{N_CTRL}+纵断面{M_PROF})  pop={POP_SIZE} iter={MAX_ITER}")

    if args.smoke:
        MAX_ITER = 5
        grids = dict(traffic=[0.0, 0.1], ev=[0.0, 0.5, 1.0],
                     fuel_price=[0.0, 0.05], elec_price=[0.0, 0.05],
                     fuel_save=[0.0, 0.05], elec_save=[0.0, 0.05],
                     w1=[0.1, 0.5, 0.9],
                     w1_cost=[0.7, 0.85, 1.0],       # 图D6 重成本
                     w1_energy=[0.0, 0.15, 0.3],     # 图D7 重能耗(能耗比例0.7-1.0)
                     w1_balanced=[0.4, 0.5, 0.6])    # 图D8 折中
        print(f"[冒烟] iter={MAX_ITER}")
    else:
        # 用户选定: EV 轴加密 (① 6×21, ② 6×6, ③ 6×6, ④ 9) = 207 点
        grids = dict(
            traffic=list(np.array([0, 2, 4, 6, 8, 10]) / 100.0),   # 6
            ev=list(np.linspace(0, 1.0, 21)),                       # 21
            fuel_price=list(np.linspace(0, 0.05, 6)),               # 6
            elec_price=list(np.linspace(0, 0.05, 6)),               # 6
            fuel_save=list(np.linspace(0, 0.05, 6)),                # 6
            elec_save=list(np.linspace(0, 0.05, 6)),                # 6
            w1=list(np.linspace(0.1, 0.9, 9)),                      # 9
            # 三段权重重优化(图D6/D7/D8): 成本权重 wC(w1)
            w1_cost=list(np.linspace(0.7, 1.0, 7)),      # 图D6 重成本: 成本比例 0.7-1.0
            w1_energy=list(np.linspace(0.0, 0.3, 7)),    # 图D7 重能耗: 能耗比例 0.7-1.0(即 wC 0-0.3)
            w1_balanced=list(np.linspace(0.4, 0.6, 5)),  # 图D8 折中: 成本/能耗比例各 0.4-0.6
        )

    tasks = build_tasks(grids)
    n = len(tasks)
    print(f"[任务] 共 {n} 个采样点, 各重跑一次 IJS(pop{POP_SIZE}/iter{MAX_ITER}), "
          f"并行 {N_WORKERS} 进程")

    recs = []
    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(align, POP_SIZE, MAX_ITER)) as pool:
        for k, rec in enumerate(pool.imap_unordered(_optimize_one, tasks), 1):
            recs.append(rec)
            el = time.time() - t_all
            eta = el / k * (n - k)
            print(f"  [{k:3d}/{n}] item{rec['item']} "
                  f"idx={rec['idx']} C={rec['C']/YI:.4f}亿 E={rec['E']/YI:.4f}亿 "
                  f"L={rec['L_km']:.3f}km Rmin={rec['Rmin']:.0f} pen={rec['pen']:.1e} "
                  f"| 用时{el/60:.1f}min ETA{eta/60:.1f}min", flush=True)

    # 组织为四项结构 + 基准
    out = _assemble(recs, tasks, grids, align)
    fn = os.path.join(RESULTS, "reopt_results.json")
    with open(fn, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[完成] {fn}  总耗时 {(time.time()-t_all)/60:.1f} min")


def _assemble(recs, tasks, grids, align):
    """把无序回收的记录重排成网格, 并挑出基准 M-C(项目①的 EV=n2_ev 最近点不便, 单跑)。"""
    by = {}
    for r in recs:
        by.setdefault(r["item"], []).append(r)

    tr = np.array(grids["traffic"]); ev = np.array(grids["ev"])
    fp = np.array(grids["fuel_price"]); ep = np.array(grids["elec_price"])
    fs = np.array(grids["fuel_save"]); es = np.array(grids["elec_save"])
    w1s = np.array(grids["w1"])

    def grid(item, na, nb, key):
        G = np.full((na, nb), np.nan)
        for r in by.get(item, []):
            i, j = r["idx"]; G[i, j] = r[key]
        return G

    # 项目①
    C1 = grid(1, len(tr), len(ev), "C"); E1 = grid(1, len(tr), len(ev), "E")
    L1 = grid(1, len(tr), len(ev), "L_km"); R1 = grid(1, len(tr), len(ev), "Rmin")
    W1 = grid(1, len(tr), len(ev), "wC"); P1 = grid(1, len(tr), len(ev), "pen")
    iC = np.unravel_index(np.nanargmax(C1), C1.shape)
    iE = np.unravel_index(np.nanargmax(E1), E1.shape)
    item1 = dict(traffic_rates=(tr * 100).tolist(), ev_pens=(ev * 100).tolist(),
                 C_grid=C1.tolist(), E_grid=E1.tolist(), L_grid=L1.tolist(),
                 Rmin_grid=R1.tolist(), wC_grid=W1.tolist(), pen_grid=P1.tolist(),
                 C_max=dict(value=float(C1[iC]), traffic=float(tr[iC[0]]*100), ev=float(ev[iC[1]]*100)),
                 E_max=dict(value=float(E1[iE]), traffic=float(tr[iE[0]]*100), ev=float(ev[iE[1]]*100)))

    # 项目②
    E2 = grid(2, len(fp), len(ep), "E"); C2 = grid(2, len(fp), len(ep), "C")
    L2 = grid(2, len(fp), len(ep), "L_km")
    item2 = dict(fuel_price=(fp*100).tolist(), elec_price=(ep*100).tolist(),
                 E_grid=E2.tolist(), C_grid=C2.tolist(), L_grid=L2.tolist(),
                 E_base=float(E2[0, 0]))

    # 项目③
    E3 = grid(3, len(fs), len(es), "E"); C3 = grid(3, len(fs), len(es), "C")
    L3 = grid(3, len(fs), len(es), "L_km")
    item3 = dict(fuel_save=(fs*100).tolist(), elec_save=(es*100).tolist(),
                 E_grid=E3.tolist(), C_grid=C3.tolist(), L_grid=L3.tolist(),
                 E_base=float(E3[0, 0]))

    # 项目④ 权重前沿
    front = []
    fr_recs = sorted(by.get(4, []), key=lambda r: r["idx"][0])
    for r in fr_recs:
        front.append(dict(w1=float(w1s[r["idx"][0]]), w2=float(1 - w1s[r["idx"][0]]),
                          C=r["C"], E=r["E"], pen=r["pen"], L_km=r["L_km"], Rmin=r["Rmin"]))
    # 熵权膝点: 用基准参数(item2 的[0,0]即 EV=n2_ev、无扰动)对应的自适应权重解
    # 取项目②基准点(0,0)作为熵权解参照(其 P 即基准), 其权重由该点重优化时的 wC 决定
    base_rec = None
    for r in by.get(2, []):
        if r["idx"] == (0, 0):
            base_rec = r; break
    item4 = dict(front=front,
                 entropy_wC=(base_rec["wC"] if base_rec else float("nan")),
                 entropy_wE=(base_rec["wE"] if base_rec else float("nan")),
                 C_opt=(base_rec["C"] if base_rec else float("nan")),
                 E_opt=(base_rec["E"] if base_rec else float("nan")))

    # 图D6/D7/D8: 三段权重重优化(每点 wC 已知, 按 wC 升序整理 C/E/L)
    def _sweep(item_id, key):
        ws = np.array(grids.get(key, []))
        pts = []
        for r in sorted(by.get(item_id, []), key=lambda r: r["idx"][0]):
            w1 = float(ws[r["idx"][0]])
            pts.append(dict(w1=w1, wE=float(1 - w1), C=r["C"], E=r["E"],
                            L_km=r["L_km"], Rmin=r["Rmin"], pen=r["pen"]))
        return pts
    reweight_cost = _sweep(5, "w1_cost")        # 成本比例 0.7-1.0
    reweight_energy = _sweep(6, "w1_energy")    # 能耗比例 0.7-1.0 (wC 0-0.3)
    reweight_balanced = _sweep(7, "w1_balanced")  # 折中 wC 0.4-0.6

    meta = dict(dim=_DIM, pop_size=POP_SIZE, max_iter=MAX_ITER,
                total_km=align["total_km"], n_points=len(recs),
                energy_unit="全生命周期元(亿元, 与C同口径)",
                scheme="每采样点重新优化线形(平纵联合 IJS)",
                C_base=(base_rec["C"] if base_rec else float("nan")),
                E_base=(base_rec["E"] if base_rec else float("nan")),
                L_base=(base_rec["L_km"] if base_rec else float("nan")),
                Rmin_base=(base_rec["Rmin"] if base_rec else float("nan")),
                ml_base=(base_rec["ml"] if base_rec else float("nan")),
                kwh_base=(base_rec["kwh"] if base_rec else float("nan")))
    return dict(meta=meta, item1=item1, item2=item2, item3=item3, item4=item4,
                reweight_cost=reweight_cost, reweight_energy=reweight_energy,
                reweight_balanced=reweight_balanced)


if __name__ == "__main__":
    main()
