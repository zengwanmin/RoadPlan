# -*- coding: utf-8 -*-
"""Current-model plan/profile collaboration ablation at exact total NFE."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms_nfe import run_algorithm
from model_adapter import (W_C, W_E, array_sha256, diagnostics, existing_reduced,
                           lower_bounds, make_context, scalar_value, upper_bounds)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
POP_DIR = RESULTS / "initial_populations"
PARTIAL = RESULTS / "collaboration_raw.partial.json"
FINAL = RESULTS / "collaboration_raw.json"
MODES = ("plan_only", "profile_only", "sequential")
SEED_BASE = 2026081700
_PC = None
_XA = None


def _init_worker():
    global _PC, _XA
    _PC = make_context()
    _XA = existing_reduced(_PC)


class SubObjective:
    def __init__(self, mode, fixed):
        self.mode, self.fixed, self.count = mode, np.asarray(fixed, float), 0

    def __call__(self, free):
        self.count += 1
        if self.mode == "plan":
            y = np.concatenate((free, self.fixed))
        else:
            y = np.concatenate((self.fixed, free))
        return scalar_value(y, _PC)


def _offset_curve(curve, offset):
    return [[int(x[0] + offset), float(x[1])] for x in curve]


def _job(task):
    mode, rep, budget, expected_hash = task
    with np.load(POP_DIR / f"rep_{rep:02d}.npz") as d:
        pop = np.asarray(d["pop"], float)
    if array_sha256(pop) != expected_hash:
        raise RuntimeError(f"population hash mismatch rep={rep}")
    lb, ub = lower_bounds(), upper_bounds()
    t0 = time.perf_counter()
    if mode == "plan_only":
        obj = SubObjective("plan", _XA[50:])
        out = run_algorithm("V5_IJS", obj, lb[:50], ub[:50], pop[:, :50],
                            budget, SEED_BASE + rep)
        y = np.concatenate((out["best_x"], _XA[50:]))
        counts = [obj.count]
        curve = out["curve"]
    elif mode == "profile_only":
        obj = SubObjective("profile", _XA[:50])
        out = run_algorithm("V5_IJS", obj, lb[50:], ub[50:], pop[:, 50:],
                            budget, SEED_BASE + rep)
        y = np.concatenate((_XA[:50], out["best_x"]))
        counts = [obj.count]
        curve = out["curve"]
    else:
        b1 = budget // 2
        b2 = budget - b1
        obj1 = SubObjective("plan", _XA[50:])
        out1 = run_algorithm("V5_IJS", obj1, lb[:50], ub[:50], pop[:, :50],
                             b1, SEED_BASE + rep)
        obj2 = SubObjective("profile", out1["best_x"])
        out2 = run_algorithm("V5_IJS", obj2, lb[50:], ub[50:], pop[:, 50:],
                             b2, SEED_BASE + rep + 1)
        y = np.concatenate((out1["best_x"], out2["best_x"]))
        counts = [obj1.count, obj2.count]
        curve = out1["curve"] + _offset_curve(out2["curve"], b1)
        out = out2
    if sum(counts) != budget:
        raise RuntimeError(f"NFE mismatch {mode}/{rep}: {counts}")
    elapsed = time.perf_counter() - t0
    result = dict(
        key=f"{mode}::{rep}", mode=mode, repetition=rep, algorithm="V5_IJS",
        algorithm_seed=SEED_BASE + rep, initial_population_sha256=expected_hash,
        nfe=budget, phase_nfe=counts, nfe_exact=True, runtime_s=float(elapsed),
        best_x_reduced=np.asarray(y).tolist(), curve=curve,
        best_f_optimizer=float(scalar_value(y, _PC)),
    )
    result.update(diagnostics(y, _PC))
    return result


def _write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--budget", type=int, default=600_000)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    main_path = RESULTS / "confirmatory_raw.json"
    if not main_path.exists():
        raise SystemExit("Complete the main comparison first")
    main = json.loads(main_path.read_text(encoding="utf-8"))
    if args.n_runs > main["config"]["n_runs"]:
        ap.error("n-runs exceeds main paired populations")
    manifests = {x["rep"]: x for x in main["config"]["populations"]}
    config = dict(schema="collaboration-ablation-v1", modes=list(MODES),
                  n_runs=args.n_runs, budget=args.budget,
                  paired_baseline="V5_IJS in confirmatory_raw.json",
                  weights=dict(wC=W_C, wE=W_E),
                  created_at=datetime.now().astimezone().isoformat())
    state = dict(config=config, records=[])
    if PARTIAL.exists() and not args.fresh:
        old = json.loads(PARTIAL.read_text(encoding="utf-8"))
        if (old.get("config", {}).get("n_runs") == args.n_runs and
                old.get("config", {}).get("budget") == args.budget):
            state["records"] = old.get("records", [])
    done = {r["key"] for r in state["records"]}
    tasks = [(mode, rep, args.budget, manifests[rep]["sha256"])
             for rep in range(args.n_runs) for mode in MODES
             if f"{mode}::{rep}" not in done]
    _write(PARTIAL, state)
    workers = args.workers or min(len(tasks), os.cpu_count() or 1)
    print(f"[collaboration] tasks={len(tasks)} workers={workers}", flush=True)
    if tasks:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            future_map = {pool.submit(_job, task): task for task in tasks}
            for i, future in enumerate(as_completed(future_map), 1):
                r = future.result()
                state["records"].append(r)
                state["records"].sort(key=lambda x: (x["repetition"], x["mode"]))
                _write(PARTIAL, state)
                print(f"[{i}/{len(tasks)}] {r['mode']} rep={r['repetition']} "
                      f"F={r['F_hard']:.6f} feasible={r['feasible']} "
                      f"time={r['runtime_s']:.1f}s", flush=True)
    if len(state["records"]) == len(MODES) * args.n_runs:
        state["completed_at"] = datetime.now().astimezone().isoformat()
        _write(FINAL, state)
        print("[complete] collaboration ablation", flush=True)


if __name__ == "__main__":
    main()

