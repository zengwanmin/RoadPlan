# -*- coding: utf-8 -*-
"""
run_joint.py — 优化方案对比主程序（平纵断面联合协同优化, pop=200, iter=1000）

【本实验的唯一方法：三维解空间(x,y,z)的平纵一体化协同优化】
  平面走向(每 10 m 控制点法向偏移 -> x,y)与纵断面坡度(每 10 m 变坡点高程 -> z)
  放入【同一决策向量】、在【同一次 IJS 寻优】中对 (x,y,z) 三维立体线形一起做
  全面彻底的搜索, 实现平面与纵断面的真正协同(区别于论文"先平面后纵断面"的分阶段
  串联)。求解桩号步长: 平面控制点 10 m、纵断面变坡点 10 m(用户指定, 全线加密;
  平面在 10 m 步长下受 R≥400m 约束的代价见 objective_joint 模块说明)。

三模式对比 (同一联合模型、同一约束下):
  M-A 现状方案(人工选线) : 实测平面中线(δ=0) + 人工粗放纵断面(0.5km 平滑地面线),
                            未做全局优化, 作为基线。
  M-B 单目标成本最优     : 平纵联合优化, 仅 min C (wC=1, wE=0)。
  M-C 平纵联合双目标(本文): 平纵联合优化, min C 与 min E 协同 + 熵权法客观决策。

  M-B → M-C 的差值 = "引入车流能耗协同优化"的净贡献。

数据: 数据.xlsx (北环高速实测)   公式: 林坤锐学位论文(式号见 objective*.py)
能耗单位: 全生命周期货币量(亿元, 与C同口径)   桥隧费用: 0(系数论文未给, 见 params/分析总结)
"""
import os, json, time, argparse, hashlib, subprocess, multiprocessing as mp
# 多进程下禁用 BLAS 内部多线程(每进程 1 核), 必须在 import numpy 之前设置。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from params import ALGO, CASE
from data_loader import load_alignment
from algorithms import run, VARIANTS
from objective import entropy_weights
import objective_joint as OJ
from objective_joint import (make_plane_context, objectives_joint,
                             make_scalar_joint, decode_joint, joint_baseline,
                             run_ijs_two_phase, START_AMP_M,
                             DIM, N_MODE, M_PROF, CORRIDOR_HALF_W,
                             STEP_PLANE_M, STEP_PROFILE_M)
from safety import hazard_profile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

SOURCE_FINGERPRINT_FILES = (
    "run_joint.py", "objective_joint.py", "objective.py", "algorithms.py",
    "params.py", "data_loader.py", "dem.py",
    "crossings.py", "safety.py",
)
DATA_FINGERPRINT_FILES = (
    os.path.join("数据", "数据.xlsx"),
    os.path.join("数据", "走廊带DEM_z14_ext.npz"),
    os.path.join("数据", "走廊带DEM_z14_ext_natural.npz"),
    os.path.join("数据", "OSM走廊带障碍物", "obstacles.npz"),
    os.path.join("数据", "OSM走廊带障碍物", "ic_anchor_cache.json"),
)

POP_SIZE = ALGO["pop_size"]     # 200 (用户指定)
MAX_ITER = 1000                 # 联合优化迭代次数(用户指定 1000)
# 说明: 联合为单次 IJS, 总求值量 ≈ POP + 3·POP·iter = 200 + 3·200·1000 = 600,200,
#       恰与两阶段(Stage1+Stage2 各 iter=500)的 2·(200+3·200·500) = 600,400 大致相等,
#       即本设置下联合(iter=1000)与两阶段(iter=500)在【等总求值预算】下比较。

# ---- worker 进程内的全局(由 initializer 设定, 兼容 macOS spawn) ----
_PC = None
_CTX = None      # dict(C_ref, E_ref, pop0, lb, ub, max_iter)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(a):
    a = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii"))
    h.update(b"|<f8|")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _fingerprint(data):
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_manifest():
    repo_root = os.path.dirname(HERE)
    files = {}
    for name in SOURCE_FINGERPRINT_FILES:
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing source file required by fingerprint: {path}")
        files[os.path.relpath(path, repo_root)] = _sha256_file(path)
    for name in DATA_FINGERPRINT_FILES:
        path = os.path.join(repo_root, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing input file required by fingerprint: {path}")
        files[name] = _sha256_file(path)
    return files


def _load_checkpoint(path, config, expected_tags, fresh=False):
    """Return prior records only for a byte-for-byte compatible experiment."""
    fingerprint = _fingerprint(config)
    if fresh or not os.path.exists(path):
        return dict(schema="joint-checkpoint-v1", config=config,
                    config_fingerprint=fingerprint, records=[])
    with open(path, encoding="utf-8") as fp:
        state = json.load(fp)
    old_fingerprint = state.get("config_fingerprint")
    if old_fingerprint != fingerprint:
        raise RuntimeError(
            "Checkpoint configuration mismatch; refusing to mix old and new results. "
            f"old={old_fingerprint!r}, current={fingerprint!r}. "
            "Archive the old checkpoint and rerun with --fresh only if intentional.")
    records = state.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Checkpoint records must be a list")
    tags = [r.get("tag") for r in records]
    if len(tags) != len(set(tags)) or not set(tags) <= set(expected_tags):
        raise RuntimeError("Checkpoint contains duplicate or unexpected task tags")
    # Replace non-fingerprinted timestamps while preserving immutable config.
    state["config"] = config
    return state


def _init_worker(align, ctx, corridor):
    """worker 初始化: 重建 plane context(含 cKDTree)与共享寻优上下文, 每进程一次。
    先同步走廊带(fork 会继承, 但 spawn 不会 -> 显式设置更稳健)。"""
    global _PC, _CTX
    OJ.set_corridor(corridor)
    _PC = make_plane_context(align)
    _CTX = ctx


def _solve_one(task):
    """
    单个权重点的完整 IJS 联合寻优(在 worker 进程执行)。
    task = dict(tag, wC, wE, seed)
    结果与串行执行完全一致: 初始种群 pop0 与 seed 均由主进程固定下发, 与执行顺序无关。
    """
    pc, c = _PC, _CTX
    def make_f(ps):
        return make_scalar_joint(pc, task["wC"], task["wE"], c["C_ref"],
                                 c["E_ref"], pen_scale=ps)
    r = run_ijs_two_phase(make_f, c["lb"], c["ub"], c["pop0"],
                          c["max_iter"], task["seed"])
    C, E, pen, _ = objectives_joint(r["best_x"], pc)
    return dict(tag=task["tag"], wC=task["wC"], wE=task["wE"],
                C=float(C), E=float(E), pen=float(pen),
                best_x=r["best_x"].tolist(), curve=r["curve"].tolist())


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
                L_eco_km=info["L_eco_km"], L_ic_km=info["L_ic_km"],
                L_bridge_new=info["L_bridge_new"],
                L_tunnel_new=info["L_tunnel_new"],
                plane_x=d["xx"].tolist(), plane_y=d["yy"].tolist(),
                design_z=d["design_z"].tolist(), sta=d["sta"].tolist(),
                gz_new=d["gz_new"].tolist(), Q_series=Q_series.tolist())


def make_existing_x(pc, dim):
    """
    构造现状方案 M-A 的联合决策向量:
      平面: 模态系数全取 0.5 -> δ=0 -> 实测中线, 里程/走向不变;
      纵断面: 直接采用【实测路面高程】作为既有设计线(GPS 实测的就是现状道路的
              设计线), 按纵坡编码反解为起点高程偏移与各段纵坡。

    注: 地面高程现已改为真实地形 DEM, 对地形做平滑并不等于既有道路的设计线,
        故此处不再用"地面线平滑"近似现状纵断面, 而是直接用实测路面高程。
    """
    from params import LONG_STD_100
    x = np.full(dim, 0.5)                        # 平面 δ=0
    d0 = decode_joint(x, pc)
    sta_ctrl = d0["sta_ctrl"]; gz_ctrl = d0["gz_ctrl"]
    # δ=0 时评价桩号与实测里程一致, 直接按里程取实测路面高程
    z_road = np.interp(sta_ctrl, pc["s_meas"], pc["gz_meas"])
    x[N_MODE] = np.clip(0.5 + (z_road[0] - gz_ctrl[0]) / (2.0 * START_AMP_M), 0.0, 1.0)
    g = np.diff(z_road) / np.diff(sta_ctrl)      # 现状道路各段纵坡
    x[N_MODE + 1:] = np.clip(0.5 + g / (2.0 * LONG_STD_100["grade_max"]), 0.0, 1.0)
    return x


def main():
    global MAX_ITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟测试(iter=5, Pareto 仅 3 个权重点)")
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数(默认: 按任务数与 CPU 核数自适应)")
    ap.add_argument("--corridor", type=float, default=None,
                    help="走廊带半宽 m(默认沿用模块设置 500)")
    ap.add_argument("--pareto", type=int, default=21,
                    help="Pareto 权重扫描点数(默认21; 减少可显著缩短总耗时)")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略并覆盖同名检查点；默认仅续跑完全相同配置")
    args = ap.parse_args()

    # 走廊带必须在 make_plane_context 之前设置(影响模态幅值 -> 平面上下文)
    if args.corridor is not None:
        OJ.set_corridor(args.corridor)

    if args.smoke:
        result_name = "joint_results_smoke.json"
    else:
        result_name = f"joint_results_w{int(OJ.CORRIDOR_HALF_W)}_nodens.json"
    result_path = os.path.join(RESULTS, result_name)
    partial_path = result_path.replace(".json", ".partial.json")

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    dim = DIM
    lb, ub = np.zeros(dim), np.ones(dim)
    n_pareto = args.pareto
    if args.smoke:
        MAX_ITER = 5
        n_pareto = 3
        print(f"[冒烟] iter={MAX_ITER}, Pareto 权重点={n_pareto}")
    print(f"[数据] 北环高速 {align['total_km']:.3f} km")
    print(f"[联合] 决策维度 dim={dim} (平面模态{N_MODE} + 纵断面{M_PROF}), "
          f"走廊带±{OJ.CORRIDOR_HALF_W:.0f}m, pop={POP_SIZE}, iter={MAX_ITER}, "
          "纵断面端点自由, 建筑密度约束=OFF")

    # ---------- 熵权法权重(基准种群客观确定, 式5.3-5.4) ----------
    # 由 joint_baseline 统一产出, 两阶段对照(run_twostage.py)共用同一组
    # (wC, wE, C_ref, E_ref), 使两种方法最小化同一个标量目标 F, 结果可直接比较。
    x_A = make_existing_x(pc, dim)
    base, wC, wE, C_ref, E_ref = joint_baseline(pc, POP_SIZE, x_seed=x_A)
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f} (与两阶段对照共用)")

    pop0 = base.copy()          # M-B/M-C/Pareto 共享同一初始种群保证公平

    config = dict(
        schema="joint-main-resume-v2",
        repository_head=_git_head(),
        corridor_half_w=float(OJ.CORRIDOR_HALF_W),
        density_on=False,
        profile_endpoints_fixed=bool(OJ.PROFILE_ENDPOINTS_FIXED),
        smoke=bool(args.smoke),
        dim=int(dim), n_mode=int(N_MODE), M_prof=int(M_PROF),
        pop_size=int(POP_SIZE), max_iter=int(MAX_ITER),
        n_pareto=int(n_pareto), optimizer_seed=1000,
        baseline_seed=2025, step_plane_m=float(STEP_PLANE_M),
        step_profile_m=float(STEP_PROFILE_M),
        weights=dict(wC=float(wC), wE=float(wE)),
        reference_scales=dict(C_ref=float(C_ref), E_ref=float(E_ref)),
        initial_population_sha256=_sha256_array(pop0),
        existing_solution_sha256=_sha256_array(x_A),
        files=_file_manifest(),
    )

    # ---------- M-A 现状方案(人工选线, 未优化) ----------
    res_A = evaluate_joint(x_A, pc)
    print(f"[M-A] C={res_A['C']/1e8:.4f}亿 E={res_A['E']/1e8:.4f}亿(全周期) "
          f"L={res_A['L_km']:.3f}km Q={res_A['Q_mean']:.3f}")

    # ---------- 并行求解 M-B / M-C / Pareto 权重扫描 ----------
    # 三者都是"同一初始种群 pop0、同一 seed=1000、只有权重不同"的独立 IJS 寻优,
    # 彼此无数据依赖, 故放到进程池里一起跑。每个任务的 pop0 与 seed 由主进程固定
    # 下发, 结果与串行执行逐位一致(仅执行顺序不同)。
    #   M-B: wC=1, wE=0            单目标成本最优
    #   M-C: wC,wE 为熵权法客观权重  本文方案
    #   Pareto: wC 从 0 到 1 扫描 n_pareto 个点(图C1 参考前沿 + 前沿变化趋势)
    w_grid = np.linspace(0.0, 1.0, n_pareto)
    tasks = ([dict(tag="M_B", wC=1.0, wE=0.0, seed=1000),
              dict(tag="M_C", wC=wC, wE=wE, seed=1000)]
             + [dict(tag=f"pareto_{k}", wC=float(w), wE=float(1 - w), seed=1000)
                for k, w in enumerate(w_grid)])
    expected_tags = [task["tag"] for task in tasks]
    checkpoint = _load_checkpoint(partial_path, config, expected_tags,
                                  fresh=bool(args.fresh))
    solved = {record["tag"]: record for record in checkpoint["records"]}
    tasks = [task for task in tasks if task["tag"] not in solved]
    _atomic_json(partial_path, checkpoint)
    ctx = dict(C_ref=C_ref, E_ref=E_ref, pop0=pop0, lb=lb, ub=ub,
               max_iter=MAX_ITER)
    n_workers = args.workers or min(max(len(tasks), 1),
                                    max(1, (os.cpu_count() or 2) - 2))
    print(f"[并行] 剩余 {len(tasks)}/{len(expected_tags)} 个寻优任务 "
          f"(M-B, M-C, Pareto×{n_pareto}), "
          f"{n_workers} 进程", flush=True)

    if tasks:
        with mp.Pool(n_workers, initializer=_init_worker,
                     initargs=(align, ctx, OJ.CORRIDOR_HALF_W)) as pool:
            for k, rec in enumerate(pool.imap_unordered(_solve_one, tasks), 1):
                solved[rec["tag"]] = rec
                checkpoint["records"] = [solved[tag] for tag in expected_tags if tag in solved]
                _atomic_json(partial_path, checkpoint)
                el = time.time() - t0
                print(f"  [{len(solved):2d}/{len(expected_tags)}] {rec['tag']:10s} "
                      f"wC={rec['wC']:.2f} C={rec['C']/1e8:.4f}亿 "
                      f"E={rec['E']/1e8:.4f}亿 pen={rec['pen']:.1e} "
                      f"| 本次用时{el/60:.1f}min "
                      f"ETA{el/k*(len(tasks)-k)/60:.1f}min", flush=True)

    missing = sorted(set(expected_tags) - set(solved))
    if missing:
        raise RuntimeError(f"Incomplete joint experiment; missing tasks: {missing}")

    # ---------- M-B 单目标成本最优 ----------
    rB = solved["M_B"]
    res_B = evaluate_joint(np.array(rB["best_x"]), pc)
    print(f"[M-B] C={res_B['C']/1e8:.4f}亿 E={res_B['E']/1e8:.4f}亿(全周期) "
          f"L={res_B['L_km']:.3f}km Rmin={res_B['Rmin']:.0f}m pen={res_B['penalty']:.2e}")

    # ---------- M-C 标量寻优点(初始种群熵权, 保留作对照) ----------
    rC = solved["M_C"]
    res_C_scalar = evaluate_joint(np.array(rC["best_x"]), pc)
    print(f"[M-C标量] C={res_C_scalar['C']/1e8:.4f}亿 E={res_C_scalar['E']/1e8:.4f}亿 "
          f"L={res_C_scalar['L_km']:.3f}km pen={res_C_scalar['penalty']:.2e}")

    # ---------- Pareto 权重扫描结果整理(含 best_x) ----------
    pareto_sweep = []
    for k, w1 in enumerate(w_grid):
        rec = solved[f"pareto_{k}"]
        pareto_sweep.append(dict(w1=float(w1), C=rec["C"], E=rec["E"],
                                 pen=rec["pen"], best_x=rec["best_x"]))

    # ---------- M-C 最终方案: 前沿熵权决策(论文第5章"先前沿、后决策"流程) ----------
    # ①候选 = 扫描解 + 标量寻优点, 要求可行(pen≈0);
    # ②预算约束: C ≤ (1+BUDGET_TOL)×现状成本(改扩建工程预算约束, 剔除
    #   "全线高架"等端部退化解, 避免熵权法对离群极差敏感);
    # ③非支配筛选; ④熵权法(式5.3-5.4, 极差标准化)加权得分取最大者。
    BUDGET_TOL = 0.10
    cands = [dict(tag=f"pareto_{k}", **p) for k, p in enumerate(pareto_sweep)]
    cands.append(dict(tag="M_C_scalar", w1=wC, C=res_C_scalar["C"],
                      E=res_C_scalar["E"], pen=res_C_scalar["penalty"],
                      best_x=rC["best_x"]))
    feas = [c for c in cands if c["pen"] <= 1e-6
            and c["C"] <= (1 + BUDGET_TOL) * res_A["C"]]
    if not feas:
        # 空集会让下面的 front[...] / M.min(0) 直接抛异常, 且看不出原因。
        pen_ok = [c for c in cands if c["pen"] <= 1e-6]
        pen_min = min((c["pen"] for c in cands), default=float("nan"))
        if pen_ok:
            print(f"[警告] 无候选同时满足 pen<=1e-6 与预算 C<=(1+{BUDGET_TOL})×现状; "
                  f"{len(pen_ok)} 个候选满足惩罚但超预算 -> 放宽预算约束降级选择",
                  flush=True)
            feas = pen_ok
        else:
            print(f"[警告] 无候选满足 pen<=1e-6(最小惩罚 {pen_min:.6g}) -> "
                  "退化为按最小惩罚选择。请检查平曲线半径、纵坡与坡差等线形约束",
                  flush=True)
            feas = [c for c in cands if c["pen"] <= pen_min + 1e-12]
    front = [c for c in feas
             if not any((o["C"] <= c["C"] and o["E"] <= c["E"])
                        and (o["C"] < c["C"] or o["E"] < c["E"]) for o in feas)]
    M = np.array([[c["C"], c["E"]] for c in front])
    mn, mx = M.min(0), M.max(0)
    if np.all(mx - mn < 1e-6):                      # 退化: 前沿点全相同(冒烟等)
        w_front = np.array([0.5, 0.5])
        score = np.zeros(len(front))
    else:
        Zn = (mx - M) / (mx - mn + 1e-12)           # 成本型极差标准化
        P = (Zn + 1e-6) / (Zn + 1e-6).sum(0)
        Ej = -(P * np.log(P)).sum(0) / np.log(max(len(front), 2))   # 式5.3
        w_front = (1 - Ej) / ((1 - Ej).sum() + 1e-12)               # 式5.4
        score = Zn @ w_front
    sel = front[int(score.argmax())]
    print(f"[决策] 可行{len(feas)}/{len(cands)}, 前沿{len(front)}个, "
          f"前沿熵权 wC={w_front[0]:.4f}/wE={w_front[1]:.4f} "
          f"-> 选中 {sel['tag']}(w1={sel['w1']:.2f})")
    res_C = evaluate_joint(np.array(sel["best_x"]), pc)
    print(f"[M-C] C={res_C['C']/1e8:.4f}亿 E={res_C['E']/1e8:.4f}亿(全周期) "
          f"L={res_C['L_km']:.3f}km Rmin={res_C['Rmin']:.0f}m pen={res_C['penalty']:.2e} "
          f"Q={res_C['Q_mean']:.3f}")
    if (not args.smoke and
            (res_C["penalty"] > 1e-6 or res_C["Rmin"] < 400.0 - 1e-6)):
        raise RuntimeError(
            "Selected M-C is not publication-feasible: "
            f"penalty={res_C['penalty']:.6g}, Rmin={res_C['Rmin']:.6g}. "
            "Checkpoint retained for diagnosis; no authoritative result was overwritten.")

    # 图C1 沿用中间区间(0.1-0.9)作参考前沿, 保持与原图口径一致
    pareto = [dict(w1=p["w1"], C=p["C"], E=p["E"], pen=p["pen"])
              for p in pareto_sweep if 0.1 - 1e-9 <= p["w1"] <= 0.9 + 1e-9]
    entropy_point = dict(C=res_C["C"], E=res_C["E"], wC=float(w_front[0]),
                         wE=float(w_front[1]), w1_selected=sel["w1"],
                         budget_tol=BUDGET_TOL)

    # 里程缩短(现状 M-A -> 本文 M-C)
    reduce_pct = (res_A["L_km"] - res_C["L_km"]) / res_A["L_km"] * 100
    print(f"[里程] 现状 {res_A['L_km']:.3f}km -> 联合优化 {res_C['L_km']:.3f}km "
          f"缩短 {reduce_pct:.2f}%")

    out = dict(
        meta=dict(dim=dim, n_mode=N_MODE, M_prof=M_PROF,
                  corridor_half_w=OJ.CORRIDOR_HALF_W, pop_size=POP_SIZE,
                  density_on=False,
                  profile_endpoints_fixed=bool(OJ.PROFILE_ENDPOINTS_FIXED),
                  max_iter=MAX_ITER, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                  total_km=align["total_km"], Rmin_req=400,
                  step_plane_m=STEP_PLANE_M, step_profile_m=STEP_PROFILE_M,
                  n_pareto=n_pareto, smoke=bool(args.smoke),
                  n_workers=n_workers,
                  energy_unit="全生命周期元(亿元)",
                  note="平纵联合协同优化(准天然地面DEM口径): OSM交叉桥内生触发, "
                       "白云山隧道由生态区穿越长度内生, 纵断面首末端自由, "
                       f"走廊带±{OJ.CORRIDOR_HALF_W:.0f}m, 建筑密度不进入约束; "
                       "M-C=前沿熵权决策"
                       "(可行+预算约束C≤1.1×现状+非支配+熵权, 论文第5章流程)"),
        provenance=dict(config_fingerprint=_fingerprint(config), config=config),
        M_A=res_A, M_B=res_B, M_C=res_C, M_C_scalar=res_C_scalar,
        pareto=pareto, pareto_sweep=pareto_sweep, entropy_point=entropy_point,
        length_reduction_pct=reduce_pct,
        measured=dict(x=align["X"].tolist(), y=align["Y"].tolist()),
        convergence=rC["curve"], convergence_B=rB["curve"],
    )
    _atomic_json(result_path, out)
    if os.path.exists(partial_path):
        os.unlink(partial_path)
    print(f"[完成] {result_name}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
