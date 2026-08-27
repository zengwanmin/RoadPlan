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
import objective_joint as OJ
from objective_joint import (make_plane_context, objectives_joint, decode_joint,
                             make_scalar_plane, build_plane_from_delta, plane_lcc,
                             joint_baseline, run_ijs_two_phase, START_AMP_M,
                             DIM, N_MODE, M_PROF, CORRIDOR_HALF_W,
                             STEP_PLANE_M, STEP_PROFILE_M)
from safety import hazard_profile
from run_joint import (_atomic_json, _file_manifest, _fingerprint, _git_head,
                       _load_checkpoint, _sha256_array, _sha256_file)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

POP_SIZE = ALGO["pop_size"]     # 200
MAX_ITER = ALGO["max_iter"]     # 500


def _load_joint_reference(path, smoke):
    """Load and validate the exact joint result that defines this control run."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing joint result {path}; run the free-endpoint W500 joint experiment first")
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    provenance = data.get("provenance", {})
    config = provenance.get("config")
    if (not isinstance(config, dict) or
            provenance.get("config_fingerprint") != _fingerprint(config)):
        raise RuntimeError("Joint reference has no valid provenance fingerprint")
    if config.get("smoke") is not bool(smoke):
        raise RuntimeError("Joint reference smoke/full mode does not match two-stage run")
    return data, _sha256_file(path)


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
    ap.add_argument("--corridor", type=float, default=None,
                    help="走廊带半宽 m(默认沿用模块设置 500)")
    ap.add_argument("--joint-result", default=None,
                    help="绑定的联合主结果；默认按smoke/W500当前口径选择")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略并覆盖同名检查点；默认仅续跑完全相同配置")
    args = ap.parse_args()
    if args.smoke:
        MAX_ITER = 5
        print(f"[冒烟] iter={MAX_ITER}")

    # 走廊带必须在 make_plane_context 之前设置(影响模态幅值)
    if args.corridor is not None:
        OJ.set_corridor(args.corridor)
    if args.smoke:
        result_name = "twostage_results_smoke.json"
        default_joint_name = "joint_results_smoke.json"
    else:
        result_name = f"twostage_results_w{int(OJ.CORRIDOR_HALF_W)}_nodens.json"
        default_joint_name = f"joint_results_w{int(OJ.CORRIDOR_HALF_W)}_nodens.json"
    result_path = os.path.join(RESULTS, result_name)
    partial_path = result_path.replace(".json", ".partial.json")
    joint_path = os.path.abspath(args.joint_result or os.path.join(RESULTS, default_joint_name))
    joint_result, joint_result_sha256 = _load_joint_reference(joint_path, args.smoke)
    joint_config = joint_result["provenance"]["config"]
    if (float(joint_config.get("corridor_half_w", -1)) != float(OJ.CORRIDOR_HALF_W) or
            joint_config.get("density_on") is not False or
            joint_config.get("profile_endpoints_fixed") is not False):
        raise RuntimeError(
            "Joint reference corridor/density/profile-endpoint configuration does not match")

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    dim = DIM
    print(f"[数据] 北环高速 {align['total_km']:.3f} km")
    print(f"[两阶段对照] dim={dim} (平面模态{N_MODE} + "
          f"纵断面{M_PROF}@{STEP_PROFILE_M:.0f}m), "
          f"走廊带±{OJ.CORRIDOR_HALF_W:.0f}m, pop={POP_SIZE}, iter={MAX_ITER}, "
          "纵断面首末端自由, 建筑密度不进入约束")

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
    baseP = rng.random((POP_SIZE, M_PROF))
    baseP[0] = x_A[N_MODE:]                  # 注入现状纵断面, 与联合同等

    # 重新计算一次只用于一致性核验；正式权重/尺度必须取自绑定的联合主结果。
    _, wC_check, wE_check, C_ref_check, E_ref_check = joint_baseline(
        pc, POP_SIZE, x_seed=x_A)
    wC = float(joint_result["meta"]["wC"])
    wE = float(joint_result["meta"]["wE"])
    C_ref = float(joint_result["meta"]["C_ref"])
    E_ref = float(joint_result["meta"]["E_ref"])
    np.testing.assert_allclose(
        [wC_check, wE_check, C_ref_check, E_ref_check],
        [wC, wE, C_ref, E_ref], rtol=2e-12, atol=1e-6,
        err_msg="Two-stage baseline weights/scales do not match the bound joint result")

    files = _file_manifest()
    files[os.path.relpath(__file__, os.path.dirname(HERE))] = _sha256_file(__file__)
    config = dict(
        schema="two-stage-resume-v2", repository_head=_git_head(),
        joint_result=os.path.relpath(joint_path, os.path.dirname(HERE)),
        joint_result_sha256=joint_result_sha256,
        joint_config_fingerprint=joint_result["provenance"]["config_fingerprint"],
        corridor_half_w=float(OJ.CORRIDOR_HALF_W),
        density_on=False,
        profile_endpoints_fixed=bool(OJ.PROFILE_ENDPOINTS_FIXED),
        smoke=bool(args.smoke),
        dim=int(dim), n_mode=int(N_MODE), M_prof=int(M_PROF),
        pop_size=int(POP_SIZE), max_iter_each_stage=int(MAX_ITER),
        stage1_seed=1000, stage2_seed=2000,
        weights=dict(wC=wC, wE=wE),
        reference_scales=dict(C_ref=C_ref, E_ref=E_ref),
        existing_solution_sha256=_sha256_array(x_A),
        stage1_population_sha256=_sha256_array(pop_plane),
        stage2_population_sha256=_sha256_array(baseP), files=files,
    )
    expected_tags = ["stage1", "stage2"]
    checkpoint = _load_checkpoint(partial_path, config, expected_tags,
                                  fresh=bool(args.fresh))
    solved = {record["tag"]: record for record in checkpoint["records"]}
    _atomic_json(partial_path, checkpoint)

    lbP, ubP = np.zeros(N_MODE), np.ones(N_MODE)
    _, _, L0_plane, _ = build_plane_from_delta(pc, np.full(N_MODE, 0.5))
    C_ref_plane = plane_lcc(L0_plane)
    if "stage1" not in solved:
        rP = run_ijs_two_phase(
            lambda ps: make_scalar_plane(pc, C_ref_plane, pen_scale=ps),
            lbP, ubP, pop_plane, MAX_ITER, 1000)
        solved["stage1"] = dict(tag="stage1", best_x=rP["best_x"].tolist(),
                                curve=np.asarray(rP["curve"]).tolist(),
                                nfe=int(rP["nfe"]))
        checkpoint["records"] = [solved[tag] for tag in expected_tags if tag in solved]
        _atomic_json(partial_path, checkpoint)
    rP = solved["stage1"]
    delta_star = np.asarray(rP["best_x"], float)    # 最优平面(冻结)
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

    # 熵权与参考尺度严格读取绑定联合结果，并已在上方用相同基准种群复算核验。
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f} (与联合方案共用)")

    lb2, ub2 = np.zeros(M_PROF), np.ones(M_PROF)
    pop2 = baseP.copy()

    def make_prof_f(ps):
        def f(prof_norm):
            C, E, pen, _ = objectives_joint(full_x(prof_norm), pc, pen_scale=ps)
            return wC * (C / C_ref) + wE * (E / E_ref) + pen
        return f

    if "stage2" not in solved:
        rC = run_ijs_two_phase(make_prof_f, lb2, ub2, pop2, MAX_ITER, 2000)
        solved["stage2"] = dict(tag="stage2", best_x=rC["best_x"].tolist(),
                                curve=np.asarray(rC["curve"]).tolist(),
                                nfe=int(rC["nfe"]))
        checkpoint["records"] = [solved[tag] for tag in expected_tags if tag in solved]
        _atomic_json(partial_path, checkpoint)
    rC = solved["stage2"]
    best_profile = np.asarray(rC["best_x"], float)
    res_C = evaluate(full_x(best_profile), pc)
    print(f"[M-C 两阶段] C={res_C['C']/1e8:.4f}亿 E={res_C['E']/1e8:.4f}亿 "
          f"L={res_C['L_km']:.3f}km Rmin={res_C['Rmin']:.0f}m pen={res_C['penalty']:.2e} "
          f"Q={res_C['Q_mean']:.3f}")
    if (not args.smoke and
            (res_C["penalty"] > 1e-6 or res_C["Rmin"] < 400.0 - 1e-6)):
        raise RuntimeError(
            "Two-stage control is not publication-feasible: "
            f"penalty={res_C['penalty']:.6g}, Rmin={res_C['Rmin']:.6g}. "
            "Checkpoint retained for diagnosis; no authoritative result was overwritten.")

    reduce_pct = (res_A["L_km"] - res_C["L_km"]) / res_A["L_km"] * 100
    print(f"[里程] 现状 {res_A['L_km']:.3f}km -> 两阶段 {res_C['L_km']:.3f}km 缩短 {reduce_pct:.2f}%")

    out = dict(
        meta=dict(dim=dim, n_mode=N_MODE, M_prof=M_PROF,
                  step_plane_m=STEP_PLANE_M, step_profile_m=STEP_PROFILE_M,
                  corridor_half_w=OJ.CORRIDOR_HALF_W, pop_size=POP_SIZE,
                  density_on=False,
                  profile_endpoints_fixed=bool(OJ.PROFILE_ENDPOINTS_FIXED),
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
        provenance=dict(
            config_fingerprint=_fingerprint(config), config=config,
            joint_result_sha256=joint_result_sha256,
            joint_config_fingerprint=joint_result["provenance"]["config_fingerprint"],
        ),
        M_A=res_A, M_S1=res_S1, M_C=res_C,
        length_reduction_pct=reduce_pct,
        convergence_stage1=list(rP["curve"]),
        convergence_stage2=list(rC["curve"]),
    )
    _atomic_json(result_path, out)
    if os.path.exists(partial_path):
        os.unlink(partial_path)
    print(f"[完成] {result_name}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
