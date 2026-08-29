# -*- coding: utf-8 -*-
"""复现前沿熵权决策选中的权重点(wC=0.40, wE=0.60), 输出完整评价指标。
与 run_joint.py 同种群(joint_baseline seed=2025 + M-A注入)、同寻优 seed=1000,
结果与扫描中的 pareto_8 逐位一致。"""
import os
import json

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from params import ALGO
from data_loader import load_alignment
from objective_joint import (make_plane_context, joint_baseline,
                             make_scalar_joint, run_ijs_two_phase, DIM, N_MODE)
from run_joint import evaluate_joint, make_existing_x

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

W_C, W_E = 0.40, 0.60          # 前沿熵权决策选中的权重点(pareto_8)
MAX_ITER = 1000

align = load_alignment()
pc = make_plane_context(align)
x_A = make_existing_x(pc, DIM)
base, wC0, wE0, C_ref, E_ref = joint_baseline(pc, ALGO["pop_size"], x_seed=x_A)
lb, ub = np.zeros(DIM), np.ones(DIM)


def make_f(ps):
    return make_scalar_joint(pc, W_C, W_E, C_ref, E_ref, pen_scale=ps)


r = run_ijs_two_phase(make_f, lb, ub, base.copy(), MAX_ITER, seed=1000)
res = evaluate_joint(np.array(r["best_x"]), pc)
res_A = evaluate_joint(x_A, pc)
print(f"[选中点 wC={W_C} wE={W_E}] C={res['C']/1e8:.4f}亿 E={res['E']/1e8:.4f}亿 "
      f"L={res['L_km']:.3f}km Rmin={res['Rmin']:.0f}m pen={res['penalty']:.2e} "
      f"Q={res['Q_mean']:.3f}")
print(f"  分项: CR={res['CR']/1e8:.3f} CB={res['CB']/1e8:.3f} CS={res['CS']/1e8:.3f} "
      f"CQ={res['CQ']/1e8:.3f} CTU={res['C_TU']/1e8:.3f} | "
      f"Ef={res['E_fuel']/1e8:.3f} Ee={res['E_ele']/1e8:.3f} | "
      f"Vs={res['Vs']/1e4:.0f}万m3 Vh={res['Vh']/1e4:.0f}万m3 "
      f"桥替代={res['L_bridge_new']:.2f}km 隧替代={res['L_tunnel_new']:.2f}km")
print(f"  vs现状: C {res['C']/res_A['C']*100-100:+.2f}%  "
      f"E {res['E']/res_A['E']*100-100:+.2f}%  "
      f"L {res['L_km']/res_A['L_km']*100-100:+.2f}%  "
      f"Q {res['Q_mean']/res_A['Q_mean']*100-100:+.2f}%")
out = dict(wC=W_C, wE=W_E, note="前沿熵权决策选中点(pareto_8 复现)",
           best_x=list(map(float, r["best_x"])), **{
               k: res[k] for k in ("C", "E", "penalty", "L_km", "Rmin", "Q_mean",
                                   "CR", "CB", "CS", "CQ", "C_TU", "E_fuel",
                                   "E_ele", "Vs", "Vh",
                                   "L_bridge_new", "L_tunnel_new")})
with open(os.path.join(RESULTS, "front_ewm_selected.json"), "w",
          encoding="utf-8") as fp:
    json.dump(out, fp, ensure_ascii=False, indent=2)
print("[完成] results/front_ewm_selected.json")
