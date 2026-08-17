# -*- coding: utf-8 -*-
"""Run resumable, paired, exact-NFE confirmatory experiments."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from algorithms_nfe import VARIANTS, run_algorithm
from model_adapter import (DIM_FULL, DIM_REDUCED, FROZEN, PEN_SCALE, RESULT_JSON,
                           W_C, W_E, C_REF, E_REF, array_sha256, diagnostics,
                           biobjective_value, existing_reduced, lower_bounds, make_context,
                           scalar_value, upper_bounds)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
POP_DIR = RESULTS / "initial_populations"
PARTIAL = RESULTS / "confirmatory_raw.partial.json"
FINAL = RESULTS / "confirmatory_raw.json"
ENVIRONMENT = RESULTS / "environment.json"

DEFAULT_ALGORITHMS = list(VARIANTS) + ["GA", "PSO", "GWO", "NSGA-II"]
POPULATION_SEED_BASE = 2026080700
ALGORITHM_SEED_BASE = 2026081700

_PC = None


class CountingObjective:
    def __init__(self, pc):
        self.pc = pc
        self.count = 0

    def __call__(self, y):
        self.count += 1
        return scalar_value(y, self.pc, pen_scale=PEN_SCALE)


class CountingBiObjective:
    def __init__(self, pc):
        self.pc = pc
        self.count = 0

    def __call__(self, y):
        self.count += 1
        return biobjective_value(y, self.pc, pen_scale=PEN_SCALE)


def _init_worker():
    global _PC
    _PC = make_context()


def _make_population(rep, pop_size, seed_y):
    """Sample the legacy box, then map it to the non-redundant quotient."""
    rng = np.random.default_rng(POPULATION_SEED_BASE + rep)
    plan = rng.random((pop_size, 50))
    raw_grade = rng.random((pop_size, 224))
    contrast = raw_grade[:, :-1] - raw_grade[:, -1, None] + 0.5
    pop = np.hstack((plan, contrast))
    pop[0] = seed_y
    return np.asarray(pop, dtype="<f8")


def _prepare_populations(n_runs, pop_size):
    POP_DIR.mkdir(parents=True, exist_ok=True)
    pc = make_context()
    seed_y = existing_reduced(pc)
    manifest = []
    for rep in range(n_runs):
        pop = _make_population(rep, pop_size, seed_y)
        path = POP_DIR / f"rep_{rep:02d}.npz"
        np.savez_compressed(path, pop=pop)
        manifest.append(dict(rep=rep, path=str(path.relative_to(HERE)),
                             shape=list(pop.shape), sha256=array_sha256(pop),
                             population_seed=POPULATION_SEED_BASE + rep,
                             algorithm_seed=ALGORITHM_SEED_BASE + rep))
    return manifest


def _job(task):
    algorithm, rep, budget, expected_hash = task
    pop_path = POP_DIR / f"rep_{rep:02d}.npz"
    with np.load(pop_path) as d:
        pop0 = np.asarray(d["pop"], dtype=float)
    actual_hash = array_sha256(pop0)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Initial population hash mismatch for rep {rep}")
    objective = CountingBiObjective(_PC) if algorithm == "NSGA-II" else CountingObjective(_PC)
    lb, ub = lower_bounds(), upper_bounds()
    t0 = time.perf_counter()
    out = run_algorithm(algorithm, objective, lb, ub, pop0, budget,
                        ALGORITHM_SEED_BASE + rep, weights=(W_C, W_E))
    elapsed = time.perf_counter() - t0
    if out["nfe"] != budget or objective.count != budget:
        raise RuntimeError(
            f"NFE mismatch {algorithm}/{rep}: algorithm={out['nfe']}, "
            f"objective={objective.count}, budget={budget}")
    y = np.asarray(out["best_x"], float)
    record = dict(
        key=f"{algorithm}::{rep}", algorithm=algorithm, repetition=rep,
        population_seed=POPULATION_SEED_BASE + rep,
        algorithm_seed=ALGORITHM_SEED_BASE + rep,
        initial_population_sha256=actual_hash,
        pop_size=int(len(pop0)), dimension=DIM_REDUCED,
        nfe=int(out["nfe"]), nfe_exact=True, runtime_s=float(elapsed),
        best_f_optimizer=float(out["best_f"]), best_x_reduced=y.tolist(),
        curve=out["curve"], generations=int(out.get("generations", 0)),
    )
    if "accepted" in out:
        record["accepted"] = out["accepted"]
    record.update(diagnostics(y, _PC))
    return record


def _atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _environment_record():
    return dict(
        generated_at=datetime.now().astimezone().isoformat(),
        python=sys.version, platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__,
        executable=sys.executable,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--pop-size", type=int, default=None)
    ap.add_argument("--algorithms", default=",")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore a matching partial result; does not delete populations")
    args = ap.parse_args()

    smoke = bool(args.smoke)
    n_runs = args.n_runs if args.n_runs is not None else (1 if smoke else 20)
    budget = args.budget if args.budget is not None else (40 if smoke else 600_000)
    pop_size = args.pop_size if args.pop_size is not None else (8 if smoke else 200)
    algorithms = (DEFAULT_ALGORITHMS if args.algorithms == "," else
                  [x.strip() for x in args.algorithms.split(",") if x.strip()])
    unknown = sorted(set(algorithms) - set(DEFAULT_ALGORITHMS))
    if unknown:
        ap.error(f"unknown algorithms: {unknown}")
    if n_runs < 1 or pop_size < 4 or budget < pop_size:
        ap.error("require n_runs>=1, pop_size>=4 and budget>=pop_size")
    workers = args.workers or min(len(algorithms) * n_runs, os.cpu_count() or 1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    _atomic_json(ENVIRONMENT, _environment_record())
    populations = _prepare_populations(n_runs, pop_size)
    pop_hash = {x["rep"]: x["sha256"] for x in populations}

    config = dict(
        schema="confirmatory-current-v1", created_at=datetime.now().astimezone().isoformat(),
        smoke=smoke, algorithms=algorithms, n_runs=n_runs, pop_size=pop_size,
        budget=budget, dimension_reduced=DIM_REDUCED, dimension_full=DIM_FULL,
        penalty_scale=PEN_SCALE, weights=dict(wC=W_C, wE=W_E),
        reference_scales=dict(C_ref=C_REF, E_ref=E_REF),
        upstream_result=str(RESULT_JSON.relative_to(HERE.parent)),
        upstream_meta=FROZEN, populations=populations,
    )
    state = dict(config=config, records=[])
    if PARTIAL.exists() and not args.fresh:
        old = json.loads(PARTIAL.read_text(encoding="utf-8"))
        comparable = (old.get("config", {}).get("algorithms") == algorithms and
                      old.get("config", {}).get("n_runs") == n_runs and
                      old.get("config", {}).get("pop_size") == pop_size and
                      old.get("config", {}).get("budget") == budget)
        if comparable:
            state["records"] = old.get("records", [])
    completed = {r["key"] for r in state["records"]}
    tasks = [(a, r, budget, pop_hash[r]) for r in range(n_runs) for a in algorithms
             if f"{a}::{r}" not in completed]
    # Persist the active configuration before the first long task completes, so
    # interruption cannot leave a stale smoke manifest beside new populations.
    _atomic_json(PARTIAL, state)
    print(f"[confirmatory] algorithms={len(algorithms)} runs={n_runs} "
          f"budget={budget:,} pop={pop_size} tasks_remaining={len(tasks)} "
          f"workers={workers}", flush=True)

    t_all = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futures = {pool.submit(_job, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                record = future.result()
                state["records"].append(record)
                state["records"].sort(key=lambda x: (x["repetition"], x["algorithm"]))
                _atomic_json(PARTIAL, state)
                print(f"[{len(completed)+i}/{len(completed)+len(tasks)}] "
                      f"{record['algorithm']} rep={record['repetition']} "
                      f"F={record['F_hard']:.6f} feasible={record['feasible']} "
                      f"time={record['runtime_s']:.1f}s", flush=True)

    expected = len(algorithms) * n_runs
    if len(state["records"]) == expected:
        state["completed_at"] = datetime.now().astimezone().isoformat()
        state["total_wall_s_this_call"] = float(time.perf_counter() - t_all)
        _atomic_json(FINAL if not smoke else RESULTS / "confirmatory_smoke.json", state)
        print(f"[complete] {expected} records; all exact NFE", flush=True)
    else:
        _atomic_json(PARTIAL, state)
        print(f"[partial] {len(state['records'])}/{expected}", flush=True)


if __name__ == "__main__":
    main()
