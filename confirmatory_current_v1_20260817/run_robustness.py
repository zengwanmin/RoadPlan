# -*- coding: utf-8 -*-
"""Resumable current-model robustness reruns after the main comparison."""
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
from model_adapter import (DATA_LOADER, DIM_REDUCED, OJ, PARAMS, PEN_SCALE,
                           W_C, W_E, array_sha256, diagnostics, lower_bounds,
                           scalar_value, upper_bounds)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
POP_DIR = RESULTS / "initial_populations"
PARTIAL = RESULTS / "robustness_raw.partial.json"
FINAL = RESULTS / "robustness_raw.json"
ALGORITHM_SEED_BASE = 2026081700

SCENARIOS = {
    "W250": dict(W=250.0),
    "W1000": dict(W=1000.0),
    "density_off": dict(density=False),
    "raw_dem": dict(raw_dem=True),
    "bridge_ext50": dict(bridge_ext_m=50.0),
    "bridge_ext100": dict(bridge_ext_m=100.0),
    "structure20": dict(structure_threshold_m=20.0),
    "structure40": dict(structure_threshold_m=40.0),
}

_ORIGINAL_GROUND = OJ.dem.ground_elev_xy


class CountingObjective:
    def __init__(self, pc):
        self.pc = pc
        self.count = 0

    def __call__(self, y):
        self.count += 1
        return scalar_value(y, self.pc, pen_scale=PEN_SCALE)


def _configure(name):
    s = SCENARIOS[name]
    OJ.set_profile_step(100.0)
    OJ.set_corridor(float(s.get("W", 500.0)))
    OJ.set_density(bool(s.get("density", True)))
    PARAMS.BRIDGE_TUNNEL["crossing_trigger"]["ext_m"] = float(
        s.get("bridge_ext_m", 75.0))
    threshold = float(s.get("structure_threshold_m", 30.0))
    PARAMS.BRIDGE_TUNNEL["fill_height_bridge_m"] = threshold
    PARAMS.BRIDGE_TUNNEL["cut_depth_tunnel_m"] = threshold
    OJ.dem.ground_elev_xy = _ORIGINAL_GROUND
    if s.get("raw_dem"):
        def raw_ground(x, y, lat0, lon0, natural=True):
            return _ORIGINAL_GROUND(x, y, lat0, lon0, natural=False)
        OJ.dem.ground_elev_xy = raw_ground
    return OJ.make_plane_context(DATA_LOADER.load_alignment())


def _job(task):
    name, rep, budget, expected_hash = task
    with np.load(POP_DIR / f"rep_{rep:02d}.npz") as d:
        pop0 = np.asarray(d["pop"], float)
    if array_sha256(pop0) != expected_hash:
        raise RuntimeError(f"population hash mismatch rep={rep}")
    pc = _configure(name)
    objective = CountingObjective(pc)
    t0 = time.perf_counter()
    out = run_algorithm("V5_IJS", objective, lower_bounds(), upper_bounds(),
                        pop0, budget, ALGORITHM_SEED_BASE + rep)
    elapsed = time.perf_counter() - t0
    if out["nfe"] != budget or objective.count != budget:
        raise RuntimeError(f"NFE mismatch {name}/{rep}")
    y = np.asarray(out["best_x"], float)
    r = dict(key=f"{name}::{rep}", scenario=name, repetition=rep,
             algorithm="V5_IJS", algorithm_seed=ALGORITHM_SEED_BASE + rep,
             initial_population_sha256=expected_hash, dimension=DIM_REDUCED,
             nfe=budget, nfe_exact=True, runtime_s=float(elapsed),
             best_f_optimizer=float(out["best_f"]), best_x_reduced=y.tolist(),
             curve=out["curve"], generations=int(out.get("generations", 0)),
             accepted=out.get("accepted", {}))
    r.update(diagnostics(y, pc))
    return r


def _write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--budget", type=int, default=600_000)
    ap.add_argument("--scenarios", default=",")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    names = list(SCENARIOS) if args.scenarios == "," else [
        x.strip() for x in args.scenarios.split(",") if x.strip()]
    unknown = sorted(set(names) - set(SCENARIOS))
    if unknown:
        ap.error(f"unknown scenarios: {unknown}")
    main_path = RESULTS / "confirmatory_raw.json"
    if not main_path.exists():
        raise SystemExit("Run and validate confirmatory_raw.json before robustness")
    main = json.loads(main_path.read_text(encoding="utf-8"))
    if len(main["records"]) != len(main["config"]["algorithms"]) * main["config"]["n_runs"]:
        raise SystemExit("Main comparison is incomplete")
    if args.n_runs > main["config"]["n_runs"]:
        ap.error("robustness n-runs exceeds persisted main populations")
    manifests = {x["rep"]: x for x in main["config"]["populations"]}
    config = dict(schema="current-robustness-v1", scenarios=names,
                  definitions={k: SCENARIOS[k] for k in names}, n_runs=args.n_runs,
                  budget=args.budget, pop_size=main["config"]["pop_size"],
                  paired_baseline="V5_IJS records in confirmatory_raw.json",
                  created_at=datetime.now().astimezone().isoformat())
    state = dict(config=config, records=[])
    if PARTIAL.exists() and not args.fresh:
        old = json.loads(PARTIAL.read_text(encoding="utf-8"))
        if (old.get("config", {}).get("scenarios") == names and
                old.get("config", {}).get("n_runs") == args.n_runs and
                old.get("config", {}).get("budget") == args.budget):
            state["records"] = old.get("records", [])
    done = {r["key"] for r in state["records"]}
    tasks = [(name, rep, args.budget, manifests[rep]["sha256"])
             for rep in range(args.n_runs) for name in names
             if f"{name}::{rep}" not in done]
    workers = args.workers or min(len(tasks), os.cpu_count() or 1)
    print(f"[robustness] tasks={len(tasks)} workers={workers}", flush=True)
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_job, t): t for t in tasks}
            for i, future in enumerate(as_completed(future_map), 1):
                r = future.result()
                state["records"].append(r)
                state["records"].sort(key=lambda x: (x["repetition"], x["scenario"]))
                _write(PARTIAL, state)
                print(f"[{i}/{len(tasks)}] {r['scenario']} rep={r['repetition']} "
                      f"F={r['F_hard']:.6f} feasible={r['feasible']} "
                      f"time={r['runtime_s']:.1f}s", flush=True)
    if len(state["records"]) == len(names) * args.n_runs:
        state["completed_at"] = datetime.now().astimezone().isoformat()
        _write(FINAL, state)
        print("[complete] robustness queue", flush=True)


if __name__ == "__main__":
    main()

