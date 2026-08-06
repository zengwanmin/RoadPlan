# -*- coding: utf-8 -*-
"""
run_ablation.py — 消融实验主程序 (实验设计方案2 · 实验一)

流程:
  1. 加载北环高速轨迹, 生成纵断面优化上下文(数据.xlsx, 不可杜撰)
  2. 生成初始种群, 用熵权法(式5.3-5.4)客观确定权重, 构建同口径标量目标F
  3. 5个变体(V1 JS基线 → V5 IJS完整) × 30次独立运行(pop=200, iter=500)
  4. 记录 F最优/均值/标准差、收敛代数、运行时间, 保存结果供出图出表

用法:
  python3 run_ablation.py            # 正式全量 (pop200/iter500/30次, ~1.1 h)
  python3 run_ablation.py --smoke    # 冒烟测试 (iter=5, 3次, 验证管线)

【为何本实验串行执行】表A2 把"平均运行时间(s)"作为结果列上报, 若跨进程并行,
各变体的耗时会受内存带宽/缓存争用干扰而不再可比。本实验总耗时仅约 1.1 h,
故坚持单进程串行, 保证 5 个变体的运行时间在同一条件下测得、可直接横向比较。
"""
import os, json, time, argparse
import numpy as np

from params import ALGO
from data_loader import load_alignment, resample_profile
from objective import objectives, entropy_weights, make_scalar_fn
from algorithms import run, VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def convergence_gen(curve, frac=0.99):
    """达到最优解 99% 所需迭代次数(实验设计方案 §2.3)。
    对最小化问题: 从初值 f0 收敛到终值 f*, 达到 f0-0.99*(f0-f*) 的代数。"""
    f0, fstar = curve[0], curve[-1]
    if abs(f0 - fstar) < 1e-12:
        return 0
    target = f0 - frac * (f0 - fstar)
    below = np.where(curve <= target)[0]
    return int(below[0]) if len(below) else len(curve) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟测试(iter=5, n_runs=3), 仅验证管线可跑通")
    args = ap.parse_args()

    t_all = time.time()
    pop_size = ALGO["pop_size"]      # 200
    max_iter = ALGO["max_iter"]      # 500
    n_runs = ALGO["n_runs"]          # 30
    if args.smoke:
        max_iter, n_runs = 5, 3
        print(f"[冒烟] max_iter={max_iter}, n_runs={n_runs}")

    # ---- 1. 数据与上下文 ----
    align = load_alignment()
    sta, gz = resample_profile(align, step_m=100.0)
    ctx = dict(sta=sta, gz=gz, total_len_m=align["s"][-1])
    dim = len(sta)
    lb, ub = np.zeros(dim), np.ones(dim)
    print(f"[数据] 北环高速 {align['total_km']:.3f} km, 变坡点/桩号 dim={dim}")

    # ---- 2. 熵权法权重(用固定基准种群客观确定, 全实验统一口径) ----
    base_rng = np.random.default_rng(2025)
    base_pop = base_rng.random((pop_size, dim))
    C0 = np.array([objectives(base_pop[i], ctx)[0] for i in range(pop_size)])
    E0 = np.array([objectives(base_pop[i], ctx)[1] for i in range(pop_size)])
    wC, wE = entropy_weights(C0, E0)
    C_ref, E_ref = float(C0.mean()), float(E0.mean())
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f}; C_ref={C_ref/1e8:.3f}亿, E_ref={E_ref:.0f}")

    # ---- 3. 5变体 × 30次独立运行 ----
    all_res = {}
    curves = {}     # 每变体保存一条代表性收敛曲线(取中位run)
    for vname, cfg in VARIANTS.items():
        tv = time.time()
        best_fs, conv_gens, runtimes = [], [], []
        run_curves = []
        best_x_overall, best_f_overall = None, np.inf
        for r in range(n_runs):
            # 每次独立运行用不同种子生成独立初始种群(公平: 同一run内5变体共享)
            rng = np.random.default_rng(1000 + r)
            pop0 = rng.random((pop_size, dim))
            f = make_scalar_fn(ctx, wC, wE, C_ref, E_ref)
            t0 = time.time()
            res = run(f, lb, ub, pop0, max_iter=max_iter, seed=1000 + r, **cfg)
            rt = time.time() - t0
            best_fs.append(res["best_f"])
            conv_gens.append(convergence_gen(res["curve"]))
            runtimes.append(rt)
            run_curves.append(res["curve"])
            if res["best_f"] < best_f_overall:
                best_f_overall = res["best_f"]; best_x_overall = res["best_x"]
        best_fs = np.array(best_fs)
        # 代表性曲线: 取 best_f 中位数对应的那次run
        med_idx = int(np.argsort(best_fs)[len(best_fs) // 2])
        curves[vname] = np.array(run_curves[med_idx]).tolist()
        # 最优方案的工程指标(供方案对比参考)
        Cb, Eb, penb, infob = objectives(best_x_overall, ctx)
        all_res[vname] = dict(
            best=float(best_fs.min()), mean=float(best_fs.mean()),
            std=float(best_fs.std()), median=float(np.median(best_fs)),
            conv_gen_mean=float(np.mean(conv_gens)),
            runtime_mean=float(np.mean(runtimes)),
            best_fs=best_fs.tolist(),
            conv_gens=list(map(int, conv_gens)),
            runtimes=list(map(float, runtimes)),
            best_C=float(Cb), best_E=float(Eb), best_pen=float(penb),
        )
        print(f"[{vname}] best={best_fs.min():.4f} mean={best_fs.mean():.4f} "
              f"std={best_fs.std():.4f} conv={np.mean(conv_gens):.1f} "
              f"t={np.mean(runtimes):.2f}s  (变体耗时 {time.time()-tv:.1f}s)")

    # ---- 4. 保存 ----
    out = dict(
        meta=dict(pop_size=pop_size, max_iter=max_iter, n_runs=n_runs,
                  dim=dim, total_km=align["total_km"],
                  wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                  CR=ALGO["CR"], levy_beta=ALGO["levy_beta"],
                  smoke=bool(args.smoke), execution="串行单进程(保证运行时间可比)"),
        variants=all_res,
        curves=curves,
    )
    fn = "ablation_results_smoke.json" if args.smoke else "ablation_results.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[完成] 结果已保存 results/{fn}  总耗时 {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
