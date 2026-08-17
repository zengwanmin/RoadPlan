#!/usr/bin/env python3
"""Create a machine-readable manuscript-number audit without changing experiments."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = HERE.parent
EXP = VERSION.parents[1]
OUT = VERSION / "tables"
OUT.mkdir(exist_ok=True)

CURRENT = EXP / "优化方案对比（平面、纵断面联合协同优化）/results/joint_results_w500_dens.json"
ABLATION = EXP / "消融实验/tables_5变体/表A2_各变体性能对比表.csv"
BENCH_DIR = EXP / "多算法对比/tables"
SENS_DIR = EXP / "敏感性分析（平、纵联合，重优化）/tables"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


rows: list[dict[str, str]] = []


def add(scope: str, experiment: str, item: str, metric: str, value, unit: str, source: Path) -> None:
    rows.append({
        "evidence_scope": scope,
        "experiment": experiment,
        "item": item,
        "metric": metric,
        "value": str(value),
        "unit": unit,
        "source": str(source.relative_to(EXP)),
        "source_sha256": sha256(source),
    })


with CURRENT.open(encoding="utf-8") as f:
    data = json.load(f)
ma, mc = data["M_A"], data["M_C"]
for key, unit in [
    ("C", "RMB"), ("E", "RMB"), ("E_fuel", "RMB"), ("E_ele", "RMB"),
    ("L_km", "km"), ("Rmin", "m"), ("Q_mean", "index"),
    ("CR", "RMB"), ("CB", "RMB"), ("CS", "RMB"), ("CQ", "RMB"),
    ("C_TU", "RMB"), ("L_dense1_km", "km"), ("L_dense2_km", "km"),
    ("penalty", "dimensionless"),
]:
    add("current-confirmatory-single-run", "current W500 density anchored", "M-A", key, ma[key], unit, CURRENT)
    add("current-confirmatory-single-run", "current W500 density anchored", "M-C", key, mc[key], unit, CURRENT)

# Exact assertions for manuscript headline values.
assert abs(100 * (mc["C"] / ma["C"] - 1) + 8.3252944825) < 1e-8
assert abs(100 * (mc["E"] / ma["E"] - 1) + 2.9993191869) < 1e-8
assert abs(100 * (mc["L_dense1_km"] / ma["L_dense1_km"] - 1) + 20.8522012282) < 1e-8
assert mc["penalty"] <= 1e-12 and mc["L_dense2_km"] <= 1e-12

with ABLATION.open(encoding="utf-8-sig", newline="") as f:
    for rec in csv.DictReader(f):
        name = rec["变体"]
        for metric, unit in [("均值F", "F"), ("标准差", "F"), ("每代NFE(×pop)", "population-equivalent")]:
            add("historical-exploratory-free-endpoint-unequal-NFE", "ablation", name,
                metric, rec[metric], unit, ABLATION)

for j in range(1, 7):
    src = BENCH_DIR / f"表B2_PJ{j}_算法最优均值标准差与统计检验汇总表.csv"
    with src.open(encoding="utf-8-sig", newline="") as f:
        for rec in csv.DictReader(f):
            for metric in ("Mean F", "Std F", "Wilcoxon p (vs IJS)"):
                add("historical-exploratory-free-endpoint-unequal-NFE", f"benchmark PJ{j}",
                    rec["Algorithm"], metric, rec[metric], "F" if metric != "Wilcoxon p (vs IJS)" else "p", src)

for filename, experiment in [
    ("表D9_走廊带半宽敏感性.csv", "historical spectral-width sensitivity"),
    ("表D10_交叉桥延伸Eext敏感性.csv", "historical bridge-extension sensitivity"),
]:
    src = SENS_DIR / filename
    with src.open(encoding="utf-8-sig", newline="") as f:
        for i, rec in enumerate(csv.DictReader(f), 1):
            for metric, value in rec.items():
                add("historical-exploratory-free-endpoint", experiment, f"row-{i}", metric, value, "as-source", src)

fieldnames = list(rows[0])
with (OUT / "paper_numbers.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

manifest_sources = [CURRENT, ABLATION]
manifest_sources.extend(BENCH_DIR / f"表B2_PJ{j}_算法最优均值标准差与统计检验汇总表.csv" for j in range(1, 7))
manifest_sources.extend(SENS_DIR / p for p in ("表D9_走廊带半宽敏感性.csv", "表D10_交叉桥延伸Eext敏感性.csv"))
with (OUT / "source_manifest.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "sha256"])
    for src in manifest_sources:
        writer.writerow([str(src.relative_to(EXP)), sha256(src)])

print(f"Audited {len(rows)} manuscript values -> {OUT / 'paper_numbers.csv'}")
