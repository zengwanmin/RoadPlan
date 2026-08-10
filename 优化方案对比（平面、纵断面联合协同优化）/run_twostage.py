# -*- coding: utf-8 -*-
"""
run_twostage.py — 两阶段(先平面, 后纵断面)优化【对照】程序

【定位】本脚本不作为独立实验, 仅生成"两阶段优化方案"的结果, 供表C3 与本文
  "平纵联合协同优化"方案对比。两阶段方法严格复现学位论文 §3.4 的分阶段框架:
    第一阶段(平面): 只搜平面控制点法向偏移, 目标=平面相关全周期成本(占地CR+基建CS
        +养护CQ, 随里程变化), 约束 平曲线半径 R>=400m(表3.2), 得最优平面后【冻结】。
    第二阶段(纵断面): 在冻结的最优平面上, 只搜变坡点高程, 双目标 min C + min E,
        熵权法客观标量化 IJS 寻优。二者串联、非同一次寻优(区别于联合协同的同时寻优)。

  求解桩号步长: 平面控制点 10 m、纵断面变坡点 10 m(用户指定; 见 objective_joint 的
  STEP_PLANE_M / STEP_PROFILE_M), 与联合协同优化完全一致, 保证两种方法在同一
  离散精度下可比。
  桥隧长度/单位造价、能耗口径、成本口径均与联合协同方案一致(共用 params.py/objective*.py),
  数据来自 数据/ (北环高速实测轨迹 + 现状桥梁隧道统计), 不杜撰。

三方案(与联合方案同口径):
  M-A  现状方案(人工选线)   : 实测平面(δ=0) + 人工粗放纵断面(0.5km 平滑地面线), 未优化。
  M-S1 仅第一阶段(平面优化) : 最优平面 + 贴地纵断面, 体现平面阶段(里程/占地)贡献。
  M-C  两阶段优化方案       : Stage1 平面 -> Stage2 纵断面, 最终两阶段优化方案。

用法:
  python3 run_twostage.py            # 正式全量 (两阶段各一次 IJS, ~45 min)
  python3 run_twostage.py --smoke    # 冒烟测试 (iter=5, 验证管线)

【为何本脚本不并行】两阶段是严格串联的: 第二阶段必须在第一阶段冻结的最优平面上
进行, 无法并行; 且各阶段内部只有一次 IJS 寻优。故本脚本单进程执行。
"""
import os, json, time, argparse
import numpy as np

from params import ALGO
from data_loader import load_alignment
from algorithms import run, VARIANTS
from objective_joint import (make_plane_context, objectives_joint, decode_joint,
                             make_scalar_plane, build_plane_from_delta, plane_lcc,
                             joint_baseline, run_ijs_two_phase, START_AMP_M,
                             DIM, N_MODE, M_PROF, CORRIDOR_HALF_W,
                             STEP_PLANE_M, STEP_PROFILE_M)
from safety import hazard_profile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

POP_SIZE = ALGO["pop_size"]     # 200
MAX_ITER = ALGO["max_iter"]     # 500


def evaluate(x, pc):
    """对完整决策向量(平面+纵断面)计算四维指标 + 平纵线形序列。"""
    C, E, pen, info = objectives_joint(x, pc)
    d = decode_joint(x, pc)
    Q_series, Q_mean = hazard_profile(d["sta"], d["gz_new"], d["design_z"])
    return dict(C=C, E=E, penalty=pen, L_km=info["L_km"], Rmin=info["Rmin"],
                C_PING=info["C_PING"], C_TU=info["C_TU"], CR=info["CR"],
                CB=info["CB"], CS=info["CS"], CQ=info["CQ"],
                E_fuel=info["E_fuel"], E_ele=info["E_ele"],
                Vs=info["Vs"], Vh=info["Vh"], Q_mean=Q_mean,
                L_eco_km=info["L_eco_km"], L_ic_km=info["L_ic_km"],
                L_bridge_new=info["L_bridge_new"],
                L_tunnel_new=info["L_tunnel_new"],
                plane_x=d["xx"].tolist(), plane_y=d["yy"].tolist(),
                design_z=d["design_z"].tolist(), sta=d["sta"].tolist(),
                gz_new=d["gz_new"].tolist(), Q_series=Q_series.tolist())
def make_existing_x(pc, dim):
    """现状方案 M-A: 平面 δ=0(实测中线) + 实测路面高程作为既有设计线。
    与 run_joint.make_existing_x 完全同口径(纵坡编码反解)。"""
    from params import LONG_STD_100
    x = np.full(dim, 0.5)
    d0 = decode_joint(x, pc)
    sta_ctrl = d0["sta_ctrl"]; gz_ctrl = d0["gz_ctrl"]
    z_road = np.interp(sta_ctrl, pc["s_meas"], pc["gz_meas"])
    x[N_MODE] = np.clip(0.5 + (z_road[0] - gz_ctrl[0]) / (2.0 * START_AMP_M), 0.0, 1.0)
    g = np.diff(z_road) / np.diff(sta_ctrl)
    x[N_MODE + 1:] = np.clip(0.5 + g / (2.0 * LONG_STD_100["grade_max"]), 0.0, 1.0)
    return x


def main():
    global MAX_ITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="冒烟测试(iter=5)")
    args = ap.parse_args()
    if args.smoke:
        MAX_ITER = 5
        print(f"[冒烟] iter={MAX_ITER}")

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    dim = DIM
    print(f"[数据] 北环高速 {align['total_km']:.3f} km")
    print(f"[两阶段对照] dim={dim} (平面模态{N_MODE} + "
          f"纵断面{M_PROF}@{STEP_PROFILE_M:.0f}m), "
          f"走廊带±{CORRIDOR_HALF_W:.0f}m, pop={POP_SIZE}, iter={MAX_ITER}")

    # ---------- M-A 现状方案 ----------
    x_A = make_existing_x(pc, dim)
    res_A = evaluate(x_A, pc)
    print(f"[M-A] C={res_A['C']/1e8:.4f}亿 E={res_A['E']/1e8:.4f}亿 "
          f"L={res_A['L_km']:.3f}km Q={res_A['Q_mean']:.3f}")

    # ========== 第一阶段: 平面优化(只搜平面变量, min 平面LCC, R>=400m) ==========
    print("[Stage 1] 平面线形优化 ...")
    rng = np.random.default_rng(2025)
    pop_plane = rng.random((POP_SIZE, N_MODE))
    pop_plane[0] = 0.5                       # 注入现状平面(δ=0), 与联合同等
    lbP, ubP = np.zeros(N_MODE), np.ones(N_MODE)
    _, _, L0_plane, _ = build_plane_from_delta(pc, np.full(N_MODE, 0.5))
    C_ref_plane = plane_lcc(L0_plane)
    rP = run_ijs_two_phase(lambda ps: make_scalar_plane(pc, C_ref_plane, pen_scale=ps),
                           lbP, ubP, pop_plane, MAX_ITER, 1000)
    delta_star = rP["best_x"]                       # 最优平面(冻结)
    _, _, L_star, R_star = build_plane_from_delta(pc, delta_star)
    print(f"[Stage 1] 最优平面 L={L_star/1000:.3f}km Rmin={R_star.min():.0f}m "
          f"(现状 {L0_plane/1000:.3f}km)")

    # M-S1: 最优平面 + 贴地纵断面
    # M-S1: 最优平面 + 【现状纵断面】(不是"纵坡全0"的水平线 —— 新编码下 0.5 表示
    #       纵坡为零, 会得到一条水平线, 不能代表"仅完成第一阶段"的方案)
    x_S1 = np.concatenate([delta_star, x_A[N_MODE:]])
    res_S1 = evaluate(x_S1, pc)
    print(f"[M-S1] C={res_S1['C']/1e8:.4f}亿 E={res_S1['E']/1e8:.4f}亿 L={res_S1['L_km']:.3f}km")

    # ========== 第二阶段: 纵断面优化(冻结平面, 只搜变坡点高程, 双目标熵权) ==========
    print("[Stage 2] 纵断面线形优化(固定最优平面, 双目标 C+E 熵权 IJS) ...")

    def full_x(prof_norm):
        return np.concatenate([delta_star, prof_norm])

    # 熵权与参考尺度: 与联合方案【共用】同一组(由联合基准种群一次算出), 使两种方法
    # 最小化同一个标量目标 F。若各自从自己的种群另算权重, C/E 单项将不可比。
    baseP = rng.random((POP_SIZE, M_PROF))
    baseP[0] = x_A[N_MODE:]                  # 注入现状纵断面, 与联合同等
    _, wC, wE, C_ref, E_ref = joint_baseline(pc, POP_SIZE, x_seed=x_A)
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f} (与联合方案共用)")

    lb2, ub2 = np.zeros(M_PROF), np.ones(M_PROF)
    pop2 = baseP.copy()

    def make_prof_f(ps):
        def f(prof_norm):
            C, E, pen, _ = objectives_joint(full_x(prof_norm), pc, pen_scale=ps)
            return wC * (C / C_ref) + wE * (E / E_ref) + pen
        return f

    rC = run_ijs_two_phase(make_prof_f, lb2, ub2, pop2, MAX_ITER, 2000)
    res_C = evaluate(full_x(rC["best_x"]), pc)
    print(f"[M-C 两阶段] C={res_C['C']/1e8:.4f}亿 E={res_C['E']/1e8:.4f}亿 "
          f"L={res_C['L_km']:.3f}km Rmin={res_C['Rmin']:.0f}m pen={res_C['penalty']:.2e} "
          f"Q={res_C['Q_mean']:.3f}")

    reduce_pct = (res_A["L_km"] - res_C["L_km"]) / res_A["L_km"] * 100
    print(f"[里程] 现状 {res_A['L_km']:.3f}km -> 两阶段 {res_C['L_km']:.3f}km 缩短 {reduce_pct:.2f}%")

    out = dict(
        meta=dict(dim=dim, n_mode=N_MODE, M_prof=M_PROF,
                  step_plane_m=STEP_PLANE_M, step_profile_m=STEP_PROFILE_M,
                  corridor_half_w=CORRIDOR_HALF_W, pop_size=POP_SIZE,
                  max_iter=MAX_ITER, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                  total_km=align["total_km"], Rmin_req=400,
                  smoke=bool(args.smoke),
                  energy_unit="全生命周期元(亿元, 与C同口径)",
                  method="两阶段对照: 先平面(min平面LCC)后纵断面(双目标C+E熵权), 平面固定后串联; "
                         f"平面步长{STEP_PLANE_M:.0f}m/纵断面步长{STEP_PROFILE_M:.0f}m, "
                         "桥隧/成本/能耗口径同联合协同方案",
                  L_plane_existing_km=L0_plane / 1000.0,
                  L_plane_optimized_km=L_star / 1000.0,
                  Rmin_plane=float(R_star.min())),
        M_A=res_A, M_S1=res_S1, M_C=res_C,
        length_reduction_pct=reduce_pct,
        convergence_stage1=list(rP["curve"]),
        convergence_stage2=list(rC["curve"]),
    )
    fn = "twostage_results_smoke.json" if args.smoke else "twostage_results.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[完成] {fn}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
