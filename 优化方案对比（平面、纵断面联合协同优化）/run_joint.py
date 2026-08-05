# -*- coding: utf-8 -*-
"""
run_joint.py — 优化方案对比主程序（平纵断面联合协同优化, pop=200, iter=500）

【本实验的唯一方法：三维解空间(x,y,z)的平纵一体化协同优化】
  平面走向(每 10 m 控制点法向偏移 -> x,y)与纵断面坡度(每 10 m 变坡点高程 -> z)
  放入【同一决策向量】、在【同一次 IJS 寻优】中对 (x,y,z) 三维立体线形一起做
  全面彻底的搜索, 实现平面与纵断面的真正协同(区别于论文"先平面后纵断面"的分阶段
  串联)。平面与纵断面的求解桩号步长均为 10 m(见 objective_joint.STEP_M)。

三模式对比 (同一联合模型、同一约束下):
  M-A 现状方案(人工选线) : 实测平面中线(δ=0) + 人工粗放纵断面(0.5km 平滑地面线),
                            未做全局优化, 作为基线。
  M-B 单目标成本最优     : 平纵联合优化, 仅 min C (wC=1, wE=0)。
  M-C 平纵联合双目标(本文): 平纵联合优化, min C 与 min E 协同 + 熵权法客观决策。

  M-B → M-C 的差值 = "引入车流能耗协同优化"的净贡献。

数据: 数据.xlsx (北环高速实测)   公式: 林坤锐学位论文(式号见 objective*.py)
能耗单位: 全生命周期货币量(亿元, 与C同口径)   桥隧费用: 0(系数论文未给, 见 params/分析总结)
"""
import os, json, time
import numpy as np

from params import ALGO, CASE
from data_loader import load_alignment
from algorithms import run, VARIANTS
from objective import entropy_weights
from objective_joint import (make_plane_context, objectives_joint,
                             make_scalar_joint, decode_joint,
                             N_CTRL, M_PROF, CORRIDOR_HALF_W)
from safety import hazard_profile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

POP_SIZE = ALGO["pop_size"]     # 200 (用户指定)
MAX_ITER = ALGO["max_iter"]     # 500 (用户指定)


def evaluate_joint(x, pc):
    """对联合决策向量计算四维指标 + 平面/纵断面线形序列。"""
    C, E, pen, info = objectives_joint(x, pc)
    d = decode_joint(x, pc)
    Q_series, Q_mean = hazard_profile(d["sta"], d["gz_new"], d["design_z"])
    return dict(C=C, E=E, penalty=pen, L_km=info["L_km"], Rmin=info["Rmin"],
                C_PING=info["C_PING"], C_TU=info["C_TU"], CR=info["CR"],
                CB=info["CB"], CS=info["CS"], CQ=info["CQ"],
                E_fuel=info["E_fuel"], E_ele=info["E_ele"],
                Vs=info["Vs"], Vh=info["Vh"], Q_mean=Q_mean,
                plane_x=d["xx"].tolist(), plane_y=d["yy"].tolist(),
                design_z=d["design_z"].tolist(), sta=d["sta"].tolist(),
                gz_new=d["gz_new"].tolist(), Q_series=Q_series.tolist())


def make_existing_x(pc, dim):
    """
    构造现状方案 M-A 的联合决策向量:
      平面: δ=0 (x[:N_CTRL]=0.5) -> 实测中线, 里程/走向不变;
      纵断面: 人工粗放设计线 = 沿实测中线地面高程的 0.5km 尺度平滑
              (未做全局精细优化), 反解为对应的 x[N_CTRL:]。
    依据: 现状为人工选线, 依实测地面线按较粗控制尺度布设纵断面(局部平滑、
          长直坡衔接), 故以 0.5km 平滑近似其未精细优化的纵断面。
    """
    x = np.full(dim, 0.5)                       # 平面 δ=0, 纵断面暂置贴地
    d0 = decode_joint(x, pc)
    gz = d0["gz_new"]; sta = d0["sta"]
    amp = pc["amp"]
    step = np.median(np.diff(sta))              # 桩号间距(≈100m)
    win = max(int(round(500.0 / step)), 3)      # 0.5km 平滑窗口
    if win % 2 == 0:
        win += 1
    kern = np.ones(win) / win
    design_A = np.convolve(gz, kern, mode="same")
    x[N_CTRL:] = np.clip(0.5 + (design_A - gz) / (2.0 * amp), 0.0, 1.0)
    return x


def main():
    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    dim = N_CTRL + M_PROF
    lb, ub = np.zeros(dim), np.ones(dim)
    print(f"[数据] 北环高速 {align['total_km']:.3f} km")
    print(f"[联合] 决策维度 dim={dim} (平面{N_CTRL} + 纵断面{M_PROF}), "
          f"走廊带±{CORRIDOR_HALF_W:.0f}m, pop={POP_SIZE}, iter={MAX_ITER}")

    # ---------- 熵权法权重(基准种群客观确定, 式5.3-5.4) ----------
    # 平面分量给足初始探索幅度(全走廊带), 避免平面子空间(仅 N_CTRL 维)在高维
    # 联合搜索中被纵断面(M_PROF 维)淹没。
    rng = np.random.default_rng(2025)
    base = np.empty((POP_SIZE, dim))
    base[:, :N_CTRL] = 0.5 + (rng.random((POP_SIZE, N_CTRL)) - 0.5) * 1.0
    base[:, N_CTRL:] = rng.random((POP_SIZE, M_PROF))
    base = np.clip(base, 0, 1)
    C0 = np.array([objectives_joint(base[i], pc)[0] for i in range(POP_SIZE)])
    E0 = np.array([objectives_joint(base[i], pc)[1] for i in range(POP_SIZE)])
    wC, wE = entropy_weights(C0, E0)
    C_ref, E_ref = float(C0.mean()), float(E0.mean())
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f}")

    pop0 = base.copy()          # M-B/M-C/Pareto 共享同一初始种群保证公平

    # ---------- M-A 现状方案(人工选线, 未优化) ----------
    x_A = make_existing_x(pc, dim)
    res_A = evaluate_joint(x_A, pc)
    print(f"[M-A] C={res_A['C']/1e8:.4f}亿 E={res_A['E']/1e8:.4f}亿(全周期) "
          f"L={res_A['L_km']:.3f}km Q={res_A['Q_mean']:.3f}")

    # ---------- M-B 单目标成本最优 (wC=1, wE=0) ----------
    fB = make_scalar_joint(pc, 1.0, 0.0, C_ref, E_ref)
    rB = run(fB, lb, ub, pop0, MAX_ITER, 1000, **VARIANTS["V5_IJS"])
    res_B = evaluate_joint(rB["best_x"], pc)
    print(f"[M-B] C={res_B['C']/1e8:.4f}亿 E={res_B['E']/1e8:.4f}亿(全周期) "
          f"L={res_B['L_km']:.3f}km Rmin={res_B['Rmin']:.0f}m pen={res_B['penalty']:.2e}")

    # ---------- M-C 平纵联合双目标协同 (熵权法, 本文方案) ----------
    fC = make_scalar_joint(pc, wC, wE, C_ref, E_ref)
    rC = run(fC, lb, ub, pop0, MAX_ITER, 1000, **VARIANTS["V5_IJS"])
    res_C = evaluate_joint(rC["best_x"], pc)
    print(f"[M-C] C={res_C['C']/1e8:.4f}亿 E={res_C['E']/1e8:.4f}亿(全周期) "
          f"L={res_C['L_km']:.3f}km Rmin={res_C['Rmin']:.0f}m pen={res_C['penalty']:.2e} "
          f"Q={res_C['Q_mean']:.3f}")

    # ---------- Pareto 权重扫描: wC 从 0 到 1 全区间 (图C1 参考前沿 + 图C7 前沿变化趋势) ----------
    # wC 从 0(纯能耗最优)到 1(纯成本最优)细分扫描, 每个权重解出一个联合最优立体线形,
    # 得到成本-能耗 Pareto 前沿随权重的完整变化趋势。
    print("[Pareto] 权重 wC 从 0 到 1 全区间扫描 ...")
    pareto_sweep = []
    for w1 in np.linspace(0.0, 1.0, 21):
        f = make_scalar_joint(pc, w1, 1 - w1, C_ref, E_ref)
        r = run(f, lb, ub, pop0, MAX_ITER, 1000, **VARIANTS["V5_IJS"])
        C, E, pen, _ = objectives_joint(r["best_x"], pc)
        pareto_sweep.append(dict(w1=float(w1), C=float(C), E=float(E), pen=float(pen)))
        print(f"    w_C={w1:.2f}: C={C/1e8:.4f}亿 E={E/1e8:.4f}亿(全周期) pen={pen:.1e}")
    # 图C1 沿用中间区间(0.1-0.9)作参考前沿, 保持与原图口径一致
    pareto = [p for p in pareto_sweep if 0.1 - 1e-9 <= p["w1"] <= 0.9 + 1e-9]
    entropy_point = dict(C=res_C["C"], E=res_C["E"], wC=wC, wE=wE)

    # 里程缩短(现状 M-A -> 本文 M-C)
    reduce_pct = (res_A["L_km"] - res_C["L_km"]) / res_A["L_km"] * 100
    print(f"[里程] 现状 {res_A['L_km']:.3f}km -> 联合优化 {res_C['L_km']:.3f}km "
          f"缩短 {reduce_pct:.2f}%")

    out = dict(
        meta=dict(dim=dim, N_ctrl=N_CTRL, M_prof=M_PROF,
                  corridor_half_w=CORRIDOR_HALF_W, pop_size=POP_SIZE,
                  max_iter=MAX_ITER, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                  total_km=align["total_km"], Rmin_req=400, step_m=10.0,
                  energy_unit="全生命周期元(亿元)",
                  note="平纵联合协同优化(三维解空间x/y/z, 步长10m全面搜索): "
                       "三方案 M-A/M-B/M-C 均在同一联合模型下评估/寻优"),
        M_A=res_A, M_B=res_B, M_C=res_C,
        pareto=pareto, pareto_sweep=pareto_sweep, entropy_point=entropy_point,
        length_reduction_pct=reduce_pct,
        measured=dict(x=align["X"].tolist(), y=align["Y"].tolist()),
        convergence=rC["curve"].tolist(), convergence_B=rB["curve"].tolist(),
    )
    with open(os.path.join(RESULTS, "joint_results.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[完成] joint_results.json  总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
