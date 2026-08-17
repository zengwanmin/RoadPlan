# -*- coding: utf-8 -*-
"""Fail-closed audit of the current-model collaboration ablation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from model_adapter import DIM_FULL, DIM_REDUCED


HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "results" / "collaboration_raw.json").read_text(encoding="utf-8"))
cfg, records = data["config"], data["records"]
expected = len(cfg["modes"]) * cfg["n_runs"]
errors = []
if len(records) != expected:
    errors.append(f"record count {len(records)} != {expected}")
if len({r["key"] for r in records}) != len(records):
    errors.append("duplicate mode/repetition keys")
for r in records:
    p = r["key"]
    if r["nfe"] != cfg["budget"] or sum(r["phase_nfe"]) != cfg["budget"]:
        errors.append(f"{p}: total NFE mismatch")
    if len(r["best_x_reduced"]) != DIM_REDUCED or len(r["best_x_full"]) != DIM_FULL:
        errors.append(f"{p}: vector dimension mismatch")
    if not np.isfinite([r["F_hard"], r["C"], r["E"], r["penalty"]]).all():
        errors.append(f"{p}: nonfinite metrics")
    if abs(r["best_f_optimizer"] - r["F_hard"]) > 5e-9:
        errors.append(f"{p}: optimizer/diagnostic mismatch")
    if not r["curve"] or r["curve"][-1][0] != cfg["budget"]:
        errors.append(f"{p}: curve endpoint mismatch")
report = dict(expected_records=expected, audited_records=len(records),
              passed=not errors, errors=errors)
(HERE / "results" / "collaboration_validation_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
if errors:
    sys.exit(1)

