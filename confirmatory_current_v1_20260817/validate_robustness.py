# -*- coding: utf-8 -*-
"""Fail-closed audit for the pre-specified robustness queue."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from model_adapter import DIM_FULL, DIM_REDUCED


HERE = Path(__file__).resolve().parent
path = HERE / "results" / "robustness_raw.json"
data = json.loads(path.read_text(encoding="utf-8"))
cfg, records = data["config"], data["records"]
expected = len(cfg["scenarios"]) * cfg["n_runs"]
errors = []
if len(records) != expected:
    errors.append(f"record count {len(records)} != {expected}")
keys = [r["key"] for r in records]
if len(keys) != len(set(keys)):
    errors.append("duplicate scenario/repetition keys")
for r in records:
    p = r["key"]
    if r["nfe"] != cfg["budget"] or not r.get("nfe_exact"):
        errors.append(f"{p}: NFE mismatch")
    if r["dimension"] != DIM_REDUCED or len(r["best_x_reduced"]) != DIM_REDUCED:
        errors.append(f"{p}: reduced dimension mismatch")
    if len(r["best_x_full"]) != DIM_FULL:
        errors.append(f"{p}: full dimension mismatch")
    if not np.isfinite([r["F_hard"], r["C"], r["E"], r["penalty"]]).all():
        errors.append(f"{p}: nonfinite metric")
    if abs(r["best_f_optimizer"] - r["F_hard"]) > 5e-9:
        errors.append(f"{p}: optimizer/diagnostic mismatch")
    curve = r.get("curve", [])
    if not curve or curve[-1][0] != cfg["budget"]:
        errors.append(f"{p}: curve endpoint mismatch")
    if any(curve[i][1] + 1e-12 < curve[i + 1][1] for i in range(len(curve) - 1)):
        errors.append(f"{p}: best-so-far curve worsens")
report = dict(expected_records=expected, audited_records=len(records),
              passed=not errors, errors=errors)
(HERE / "results" / "robustness_validation_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
if errors:
    sys.exit(1)

