# -*- coding: utf-8 -*-
"""
run_ablation.py — 消融实验主程序 (实验设计方案2 · 实验一)

流程:
  1. 加载北环高速轨迹, 构建平面模态+纵断面坡段的联合优化上下文
  2. 生成联合初始种群, 用熵权法客观确定权重, 构建同口径标量目标F
  3. 8个全因子变体 × 30次独立运行(pop=200, iter=500)
  4. 记录 F最优/均值/标准差、收敛代数、运行时间及每次运行的阶段直接贡献,
     保存结果供出图出表

用法:
  python3 run_ablation.py            # 正式全量 (pop200/iter500/30次)
  python3 run_ablation.py --smoke    # 冒烟测试 (iter=5, 3次, 验证管线)

【为何本实验串行执行】表A2 把"平均运行时间(s)"作为结果列上报, 若跨进程并行,
各变体的耗时会受内存带宽/缓存争用干扰而不再可比。平纵联合评价开销显著高于
固定平面版本，仍坚持单进程串行以保证8个变体的运行时间可直接横向比较。
"""
import os, json, time, argparse
import numpy as np

from params import ALGO
from data_loader import load_alignment
from algorithms import run, VARIANTS
from objective_joint import (make_plane_context, decode_joint, objectives_joint,
                             make_scalar_joint, joint_baseline, DIM, N_MODE,
                             M_PROF, START_AMP_M, CORRIDOR_HALF_W,
                             STEP_PLANE_CTRL_M, STEP_PROFILE_CTRL_M,
                             STEP_EVAL_M, PROFILE_ENDPOINTS_FIXED)
from acceleration import evaluation_workers

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

PENALTY_SOFT = 0.3
PENALTY_HARD = 3.0


def make_existing_x(pc, dim):
    """按联合主实验口径构造现状线：平面零偏移＋实测道路纵断面。"""
    from params import LONG_STD_100
    x = np.full(dim, 0.5)
    d0 = decode_joint(x, pc)
    sta_ctrl, gz_ctrl = d0["sta_ctrl"], d0["gz_ctrl"]
    z_road = np.interp(sta_ctrl, pc["s_meas"], pc["gz_meas"])
    x[N_MODE] = np.clip(
        0.5 + (z_road[0] - gz_ctrl[0]) / (2.0 * START_AMP_M), 0.0, 1.0)
    grades = np.diff(z_road) / np.diff(sta_ctrl)
    x[N_MODE + 1:] = np.clip(
        0.5 + grades / (2.0 * LONG_STD_100["grade_max"]), 0.0, 1.0)
    return x


def _merge_traces(first, second):
    """合并软、硬惩罚两阶段插桩，保持原图表所需字段结构。"""
    phases = ("main", "levy", "de")
    return dict(
        phase_dF={p: list(first["phase_dF"][p])
                  + list(second["phase_dF"][p]) for p in phases},
        phase_acc={p: list(first["phase_acc"][p])
                   + list(second["phase_acc"][p]) for p in phases},
        diversity=list(first["diversity"]) + list(second["diversity"]),
        tent_dF=float(first["tent_dF"]) + float(second["tent_dF"]),
        tent_n_rep=int(first["tent_n_rep"]) + int(second["tent_n_rep"]),
    )


def run_variant_joint(pc, cfg, lb, ub, pop0, max_iter, seed,
                      wC, wE, C_ref, E_ref):
    """所有消融变体共用联合主实验的先软后硬惩罚调度。"""
    it1 = max_iter // 2
    it2 = max_iter - it1
    f1 = make_scalar_joint(pc, wC, wE, C_ref, E_ref,
                           pen_scale=PENALTY_SOFT)
    r1 = run(f1, lb, ub, pop0, it1, seed, track=True, **cfg)
    f2 = make_scalar_joint(pc, wC, wE, C_ref, E_ref,
                           pen_scale=PENALTY_HARD)
    r2 = run(f2, lb, ub, r1["pop"], it2, seed + 1, track=True, **cfg)
    return dict(best_x=r2["best_x"], best_f=r2["best_f"],
                curve=np.concatenate([r1["curve"], r2["curve"]]),
                nfe=int(r1["nfe"] + r2["nfe"]),
                trace=_merge_traces(r1["trace"], r2["trace"]))


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

    # ---- 1. 平纵联合上下文（与优化方案对比联合主实验同口径） ----
    align = load_alignment()
    pc = make_plane_context(align)
    dim = DIM
    lb, ub = np.zeros(dim), np.ones(dim)
    if dim != N_MODE + M_PROF:
        raise RuntimeError(
            f"联合决策维数不一致: DIM={dim}, N_MODE+M_PROF={N_MODE + M_PROF}")
    print(f"[数据] 北环高速 {align['total_km']:.3f} km, 联合 dim={dim} "
          f"(平面模态{N_MODE}+纵断面{M_PROF}), "
          f"走廊带±{CORRIDOR_HALF_W:.0f}m, 纵断面端点自由, 建筑密度约束=OFF")

    # ---- 2. 联合基准种群、熵权和参考尺度（与联合主实验同一构造） ----
    x_existing = make_existing_x(pc, dim)
    _, wC, wE, C_ref, E_ref = joint_baseline(
        pc, pop_size, seed=2025, x_seed=x_existing)
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f}; C_ref={C_ref/1e8:.3f}亿, E_ref={E_ref:.0f}")

    # ---- 3. 8变体 × 30次独立运行(全部运行带机制插桩, 问题15) ----
    # 全部运行统一 track=True, 使阶段直接贡献可以按30个配对种子报告分布,
    # 同时避免运行时间统计混合“有插桩/无插桩”两种口径。
    all_res = {}
    curves = {}     # 每变体保存一条代表性收敛曲线(取中位run)
    traces = {}     # {vname: [trace_dict × n_runs]} 机制对齐插桩(问题15)
    for vname, cfg in VARIANTS.items():
        tv = time.time()
        best_fs, conv_gens, runtimes = [], [], []
        run_curves = []
        v_traces = []
        best_x_overall, best_f_overall = None, np.inf
        for r in range(n_runs):
            # 每次独立运行用不同种子生成独立初始种群(公平: 同一run内8变体共享)
            rng = np.random.default_rng(1000 + r)
            pop0 = rng.random((pop_size, dim))
            pop0[0] = x_existing
            t0 = time.time()
            res = run_variant_joint(
                pc, cfg, lb, ub, pop0, max_iter, 1000 + r,
                wC, wE, C_ref, E_ref)
            rt = time.time() - t0
            best_fs.append(res["best_f"])
            conv_gens.append(convergence_gen(res["curve"]))
            runtimes.append(rt)
            run_curves.append(res["curve"])
            v_traces.append(res["trace"])
            if res["best_f"] < best_f_overall:
                best_f_overall = res["best_f"]; best_x_overall = res["best_x"]
        best_fs = np.array(best_fs)
        traces[vname] = v_traces
        # 代表性曲线: 取 best_f 中位数对应的那次run
        med_idx = int(np.argsort(best_fs)[len(best_fs) // 2])
        curves[vname] = np.array(run_curves[med_idx]).tolist()
        # 最优方案的工程指标(供方案对比参考)
        Cb, Eb, penb, infob = objectives_joint(best_x_overall, pc)
        all_res[vname] = dict(
            best=float(best_fs.min()), mean=float(best_fs.mean()),
            std=float(best_fs.std()), median=float(np.median(best_fs)),
            conv_gen_mean=float(np.mean(conv_gens)),
            runtime_mean=float(np.mean(runtimes)),
            best_fs=best_fs.tolist(),
            conv_gens=list(map(int, conv_gens)),
            runtimes=list(map(float, runtimes)),
            best_C=float(Cb), best_E=float(Eb), best_pen=float(penb),
            best_L_km=float(infob["L_km"]),
            best_Rmin=float(infob["Rmin"]),
            best_L_eco_km=float(infob["L_eco_km"]),
            best_L_cross_km=float(infob["L_cross_km"]),
        )
        print(f"[{vname}] best={best_fs.min():.4f} mean={best_fs.mean():.4f} "
              f"std={best_fs.std():.4f} conv={np.mean(conv_gens):.1f} "
              f"t={np.mean(runtimes):.2f}s  (变体耗时 {time.time()-tv:.1f}s)")

    # ---- 4. 保存 ----
    out = dict(
        meta=dict(pop_size=pop_size, max_iter=max_iter, n_runs=n_runs,
                  run_seeds=[1000 + r for r in range(n_runs)],
                  dim=dim, total_km=align["total_km"],
                  model="平面线形与纵断面联合协同优化",
                  n_mode=N_MODE, M_prof=M_PROF,
                  step_plane_ctrl_m=STEP_PLANE_CTRL_M,
                  step_profile_ctrl_m=STEP_PROFILE_CTRL_M,
                  step_eval_m=STEP_EVAL_M,
                  corridor_half_w=CORRIDOR_HALF_W,
                  profile_endpoints_fixed=bool(PROFILE_ENDPOINTS_FIXED),
                  density_on=False, osm_crossing_on=True,
                  wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                  CR=ALGO["CR"], levy_beta=ALGO["levy_beta"],
                  baseline_seed=2025,
                  existing_solution_injected=True,
                  penalty_schedule=dict(soft=PENALTY_SOFT,
                                        hard=PENALTY_HARD,
                                        split_iteration=max_iter // 2),
                  evaluation_workers=evaluation_workers(),
                  smoke=bool(args.smoke), execution="串行单进程(保证运行时间可比)"),
        variants=all_res,
        curves=curves,
    )
    fn = "ablation_results_smoke.json" if args.smoke else "ablation_results.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    # 机制对齐插桩(问题15)单独保存(体积大, 供 make_outputs 计算表A3/图A4-A6)
    tfn = "ablation_traces_smoke.json" if args.smoke else "ablation_traces.json"
    with open(os.path.join(RESULTS, tfn), "w", encoding="utf-8") as fp:
        json.dump(dict(
            n_track=n_runs,
            n_runs=n_runs,
            run_seeds=[1000 + r for r in range(n_runs)],
            attribution_order=["main", "levy", "de"],
            attribution_definition=(
                "联合模型软/硬惩罚两阶段中，各算子当场刷新全局最优产生的best-F下降量"),
            model="平面线形与纵断面联合协同优化",
            penalty_schedule=dict(soft=PENALTY_SOFT, hard=PENALTY_HARD,
                                  split_iteration=max_iter // 2),
            traces=traces,
        ), fp)
    print(f"[完成] 结果已保存 results/{fn} + results/{tfn}  总耗时 {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
