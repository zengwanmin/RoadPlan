#!/usr/bin/env python3
"""Fail if the v5 manuscript drifts from the pinned c1327df evidence."""
from __future__ import annotations

import hashlib
import csv
import json
import re
import subprocess
from pathlib import Path


COMMIT = "c1327df1ea2dc64cdde826bcb1df7141d80a0533"
HERE = Path(__file__).resolve().parent
VERSION = HERE.parent
REPO = VERSION.parents[1]
JOINT = "优化方案对比（平面、纵断面联合协同优化）/results/joint_results.json"
TWOSTAGE = "优化方案对比（平面、纵断面联合协同优化）/results/twostage_results.json"


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=REPO)


def close(actual: float, expected: float, tol: float = 5e-7) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{actual} != {expected}")


jraw = git_bytes(JOINT)
traw = git_bytes(TWOSTAGE)
j = json.loads(jraw)
t = json.loads(traw)
ma, mc, ts = j["M_A"], j["M_C"], t["M_C"]

assert hashlib.sha256(jraw).hexdigest() == "641557fb3a436f1c07917a614c90f1652a04a5e5f092a027036e4bf30325b88d"
assert hashlib.sha256(traw).hexdigest() == "fa8b4d25bf7a7f00a20d9c1f07641bfab93cb35664c317fd94aa4ede211b4b45"
with (VERSION / "tables/source_manifest.csv").open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        assert row["source_commit"] == COMMIT
        actual = hashlib.sha256(git_bytes(row["source_file"])).hexdigest()
        if actual != row["sha256"]:
            raise AssertionError(f"manifest hash mismatch: {row['source_file']}")
close(100 * (1 - mc["C"] / ma["C"]), 8.264036528125207)
close(100 * (1 - mc["E"] / ma["E"]), 3.366120477776058)
close(100 * (1 - mc["L_km"] / ma["L_km"]), 0.4375240843179447)
close(100 * (1 - mc["E_fuel"] / ma["E_fuel"]), 3.205547539300013)
close(100 * (1 - mc["E_ele"] / ma["E_ele"]), 3.994383169722554)
close(mc["Rmin"], 401.02567056600003)
close(mc["penalty"], 0.0)
close(ts["Rmin"], 397.05792459454506)
close(ts["penalty"], 0.07359275057726418)

required = (
    "26.4061", "24.2239", "13.9460", "13.4766", "8.26", "3.37",
    "401.03", "397.06", "0.07359", "3.21", "3.99", "0.6808", "0.9133",
)
manuscripts = (VERSION / "en/main_en.tex", VERSION / "zh/main_zh.tex")
combined = ""
for manuscript in manuscripts:
    text = manuscript.read_text(encoding="utf-8")
    combined += text
    missing = [value for value in required if value not in text]
    if missing:
        raise AssertionError(f"{manuscript}: missing {missing}")
    for pattern in (r"(?<!\d)8\.33\s*\\%", r"(?<!\d)3\.00\s*\\%",
                    r"3\.31\s*\\%", r"4\.23\s*\\%",
                    r"24\.2078", r"13\.5278", r"Tier-[12]"):
        if re.search(pattern, text):
            raise AssertionError(f"{manuscript}: forbidden later-version value {pattern}")
    assert "vertical endpoints are free" in text or "纵断面端点自由" in text
    assert "unequal" in text or "不等" in text

for value in ("18.1536", "15.5395", "0.6717", "1.1267", "2.6141", "0.4550"):
    if value not in combined:
        raise AssertionError(f"bilingual manuscripts missing decomposition value {value}")

print("PASS: bilingual manuscript matches pinned c1327df headline evidence")
