# -*- coding: utf-8 -*-
"""Run resumable, paired, exact-NFE confirmatory experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
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

SOURCE_FINGERPRINT_FILES = (
    HERE / "run_confirmatory.py",
    HERE / "model_adapter.py",
    HERE / "algorithms_nfe.py",
    HERE.parent / "优化方案对比（平面、纵断面联合协同优化）" / "objective_joint.py",
    HERE.parent / "优化方案对比（平面、纵断面联合协同优化）" / "objective.py",
    HERE.parent / "优化方案对比（平面、纵断面联合协同优化）" / "params.py",
    HERE.parent / "优化方案对比（平面、纵断面联合协同优化）" / "data_loader.py",
    HERE.parent / "优化方案对比（平面、纵断面联合协同优化）" / "dem.py",
    HERE.parent / "优化方案对比（平面、纵断面联合协同优化）" / "building_mask.py",
    HERE.parent / "优化方案对比（平面、纵断面联合协同优化）" / "crossings.py",
)
DATA_FINGERPRINT_FILES = (
    HERE.parent / "数据" / "数据.xlsx",
    HERE.parent / "数据" / "走廊带DEM_z14_ext.npz",
    HERE.parent / "数据" / "走廊带DEM_z14_ext_natural.npz",
    HERE.parent / "数据" / "OSM走廊带障碍物" / "density_tiers_V1.npz",
    HERE.parent / "数据" / "OSM走廊带障碍物" / "obstacles.npz",
    HERE.parent / "数据" / "OSM走廊带障碍物" / "ic_anchor_cache.json",
)

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


def _prepare_populations(n_runs, pop_size, fresh=False):
    POP_DIR.mkdir(parents=True, exist_ok=True)
    pc = make_context()
    seed_y = existing_reduced(pc)
    manifest = []
    for rep in range(n_runs):
        pop = _make_population(rep, pop_size, seed_y)
        path = POP_DIR / f"rep_{rep:02d}.npz"
        expected_hash = array_sha256(pop)
        if path.exists() and not fresh:
            with np.load(path) as stored:
                actual = np.asarray(stored["pop"], dtype="<f8")
            actual_hash = array_sha256(actual)
            if actual.shape != pop.shape or actual_hash != expected_hash:
                raise RuntimeError(
                    f"Initial population {path.name} differs from the current model. "
                    "Refusing to overwrite it during resume; archive first and use --fresh.")
        else:
            np.savez_compressed(path, pop=pop)
        manifest.append(dict(rep=rep, path=str(path.relative_to(HERE)),
                             shape=list(pop.shape), sha256=expected_hash,
                             population_seed=POPULATION_SEED_BASE + rep,
                             algorithm_seed=ALGORITHM_SEED_BASE + rep))
    return manifest


def _job(task):
    algorithm, rep, budget, expected_hash, config_fingerprint = task
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
        config_fingerprint=config_fingerprint,
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
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(data):
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_manifest():
    manifest = {}
    for path in SOURCE_FINGERPRINT_FILES + DATA_FINGERPRINT_FILES + (RESULT_JSON,):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing file required by fingerprint: {path}")
        manifest[str(path.relative_to(HERE.parent))] = _sha256_file(path)
    return manifest


def _validated_upstream_result(current_files):
    """Require a current, provenance-bearing W500 result before any long run."""
    payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    provenance = payload.get("provenance", {})
    config = provenance.get("config")
    fingerprint = provenance.get("config_fingerprint")
    if not isinstance(config, dict) or fingerprint != _fingerprint(config):
        raise RuntimeError(
            "The W500 upstream result is legacy or has invalid provenance; "
            "run the fixed-endpoint joint experiment first.")
    if (float(config.get("corridor_half_w", -1)) != 500.0 or
            config.get("density_on") is not True or config.get("smoke") is not False):
        raise RuntimeError("Confirmatory experiments require full W500 density-enabled results")
    upstream_files = config.get("files", {})
    for path, digest in upstream_files.items():
        if path in current_files and current_files[path] != digest:
            raise RuntimeError(
                f"Current code/data differs from the W500 source manifest: {path}")
    return payload, fingerprint


def _load_checkpoint(path, config, expected_keys, pop_hash, fresh=False):
    fingerprint = _fingerprint(config)
    path = Path(path)
    if fresh or not path.exists():
        return dict(schema="confirmatory-checkpoint-v2", config=config,
                    config_fingerprint=fingerprint, records=[])
    old = json.loads(path.read_text(encoding="utf-8"))
    old_fingerprint = old.get("config_fingerprint")
    if old_fingerprint != fingerprint:
        raise RuntimeError(
            "Checkpoint configuration mismatch; refusing to mix old and new results. "
            f"old={old_fingerprint!r}, current={fingerprint!r}. "
            "The upstream result, weights, code, data, environment and populations "
            "must all match. Use --fresh only after the old checkpoint is archived.")
    records = old.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Checkpoint records must be a list")
    keys = [record.get("key") for record in records]
    if len(keys) != len(set(keys)) or not set(keys) <= set(expected_keys):
        raise RuntimeError("Checkpoint contains duplicate or unexpected task keys")
    for record in records:
        rep = int(record.get("repetition", -1))
        if (record.get("config_fingerprint") != fingerprint or
                record.get("initial_population_sha256") != pop_hash.get(rep) or
                record.get("nfe") != config["budget"] or
                record.get("nfe_exact") is not True):
            raise RuntimeError(f"Checkpoint record failed provenance audit: {record.get('key')}")
    old["config"] = config
    return old


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
                    help="覆盖旧检查点/种群并从头运行；仅可在旧结果已归档后使用")
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

    current_files = _file_manifest()
    _, upstream_fingerprint = _validated_upstream_result(current_files)
    RESULTS.mkdir(parents=True, exist_ok=True)
    populations = _prepare_populations(n_runs, pop_size, fresh=bool(args.fresh))
    pop_hash = {x["rep"]: x["sha256"] for x in populations}
    _atomic_json(ENVIRONMENT, _environment_record())
    config = dict(
        schema="confirmatory-current-v2",
        repository_head=_git_head(),
        smoke=smoke, algorithms=algorithms, n_runs=n_runs, pop_size=pop_size,
        budget=budget, dimension_reduced=DIM_REDUCED, dimension_full=DIM_FULL,
        penalty_scale=PEN_SCALE, weights=dict(wC=W_C, wE=W_E),
        reference_scales=dict(C_ref=C_REF, E_ref=E_REF),
        upstream_result=str(RESULT_JSON.relative_to(HERE.parent)),
        upstream_result_sha256=_sha256_file(RESULT_JSON),
        upstream_config_fingerprint=upstream_fingerprint,
        upstream_meta=FROZEN, populations=populations,
        runtime=dict(python=sys.version, platform=platform.platform(),
                     numpy=np.__version__, scipy=scipy.__version__,
                     executable=sys.executable),
        files=current_files,
    )
    fingerprint = _fingerprint(config)
    expected_keys = [f"{algorithm}::{rep}" for rep in range(n_runs)
                     for algorithm in algorithms]
    state = _load_checkpoint(PARTIAL, config, expected_keys, pop_hash,
                             fresh=bool(args.fresh))
    state.setdefault("created_at", datetime.now().astimezone().isoformat())
    completed = {r["key"] for r in state["records"]}
    tasks = [(a, r, budget, pop_hash[r], fingerprint)
             for r in range(n_runs) for a in algorithms
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
        final_path = FINAL if not smoke else RESULTS / "confirmatory_smoke.json"
        _atomic_json(final_path, state)
        if PARTIAL.exists():
            PARTIAL.unlink()
        print(f"[complete] {expected} records; all exact NFE", flush=True)
    else:
        _atomic_json(PARTIAL, state)
        print(f"[partial] {len(state['records'])}/{expected}", flush=True)


if __name__ == "__main__":
    main()
