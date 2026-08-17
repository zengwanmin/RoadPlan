# -*- coding: utf-8 -*-
"""Fail-closed audit of confirmatory raw records and population identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from model_adapter import DIM_FULL, DIM_REDUCED, array_sha256


HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/confirmatory_raw.json")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    path = (HERE / args.path).resolve() if not Path(args.path).is_absolute() else Path(args.path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg, records = data["config"], data["records"]
    expected = len(cfg["algorithms"]) * cfg["n_runs"]
    errors = []
    freeze = json.loads((HERE / "CODE_FREEZE.json").read_text(encoding="utf-8"))
    for relative, expected_hash in freeze["files"].items():
        frozen_path = (HERE / relative).resolve()
        actual_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"runtime code freeze mismatch: {relative}")
    result_hash = hashlib.sha256(
        (HERE.parent / "优化方案对比（平面、纵断面联合协同优化）" /
         "results" / "joint_results_w500_dens.json").read_bytes()).hexdigest()
    if result_hash != freeze["authoritative_result_sha256"]:
        errors.append("authoritative upstream result hash mismatch")
    if not args.allow_partial and len(records) != expected:
        errors.append(f"record count {len(records)} != {expected}")
    keys = [r["key"] for r in records]
    if len(keys) != len(set(keys)):
        errors.append("duplicate algorithm/repetition keys")
    pop_manifest = {x["rep"]: x for x in cfg["populations"]}
    for rep, item in pop_manifest.items():
        p = HERE / item["path"]
        if not p.exists():
            errors.append(f"missing population {p}")
            continue
        with np.load(p) as d:
            pop = d["pop"]
        if array_sha256(pop) != item["sha256"]:
            errors.append(f"population hash mismatch rep={rep}")
    for r in records:
        prefix = r["key"]
        if r["nfe"] != cfg["budget"] or not r.get("nfe_exact"):
            errors.append(f"{prefix}: non-exact NFE")
        if r["dimension"] != DIM_REDUCED:
            errors.append(f"{prefix}: reduced dimension mismatch")
        if len(r["best_x_reduced"]) != DIM_REDUCED:
            errors.append(f"{prefix}: reduced vector length mismatch")
        if len(r["best_x_full"]) != DIM_FULL:
            errors.append(f"{prefix}: full vector length mismatch")
        if r["initial_population_sha256"] != pop_manifest[r["repetition"]]["sha256"]:
            errors.append(f"{prefix}: record population hash mismatch")
        if not np.isfinite([r["F_hard"], r["C"], r["E"], r["penalty"]]).all():
            errors.append(f"{prefix}: nonfinite result")
        if abs(r["best_f_optimizer"] - r["F_hard"]) > 5e-9:
            errors.append(f"{prefix}: optimizer/diagnostic F mismatch "
                          f"{r['best_f_optimizer']} vs {r['F_hard']}")
        curve = r.get("curve", [])
        if not curve or curve[-1][0] != cfg["budget"]:
            errors.append(f"{prefix}: convergence curve does not end at budget")
        if any(curve[i][0] > curve[i + 1][0] for i in range(len(curve) - 1)):
            errors.append(f"{prefix}: nonmonotone NFE axis")
        if any(curve[i][1] + 1e-12 < curve[i + 1][1] for i in range(len(curve) - 1)):
            errors.append(f"{prefix}: best-so-far curve worsens")
    report = dict(path=str(path), expected_records=expected, audited_records=len(records),
                  passed=not errors, errors=errors)
    out = HERE / "results" / "validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
