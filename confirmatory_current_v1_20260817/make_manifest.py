# -*- coding: utf-8 -*-
"""Create a SHA-256 manifest for code, frozen inputs and completed evidence."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEME = ROOT / "优化方案对比（平面、纵断面联合协同优化）"
FILES = [
    HERE / "PROTOCOL.md", HERE / "README.md", HERE / "requirements-lock.txt",
    HERE / "CODE_FREEZE.json", HERE / "V4_INTEGRATION_MAP.md",
    HERE / "OPERATIONAL_SCENARIOS_PROTOCOL.md",
    HERE / "COLLABORATION_PROTOCOL.md",
    HERE / "model_adapter.py", HERE / "algorithms_nfe.py",
    HERE / "run_confirmatory.py", HERE / "validate_results.py",
    HERE / "analyze_confirmatory.py", HERE / "run_robustness.py",
    HERE / "validate_robustness.py", HERE / "analyze_robustness.py",
    HERE / "evaluate_operational_scenarios.py",
    HERE / "run_collaboration.py", HERE / "validate_collaboration.py",
    HERE / "analyze_collaboration.py",
    HERE / "make_manifest.py",
    HERE / "tests" / "test_protocol.py",
    SCHEME / "objective_joint.py", SCHEME / "objective.py", SCHEME / "params.py",
    SCHEME / "algorithms.py", SCHEME / "run_joint.py", SCHEME / "crossings.py",
    SCHEME / "dem.py", SCHEME / "building_mask.py", SCHEME / "data_loader.py",
    SCHEME / "results" / "joint_results_w500_dens.json",
    ROOT / "数据" / "数据.xlsx",
    ROOT / "数据" / "OSM走廊带障碍物" / "obstacles.npz",
    ROOT / "数据" / "OSM走廊带障碍物" / "density_tiers_V1.npz",
]
FILES += sorted((HERE / "results" / "initial_populations").glob("rep_*.npz"))
for name in ("confirmatory_raw.json", "validation_report.json",
             "statistical_analysis.json", "robustness_raw.json",
             "robustness_validation_report.json", "robustness_analysis.json",
             "collaboration_raw.json", "collaboration_validation_report.json",
             "collaboration_analysis.json", "operational_scenarios.json",
             "environment.json"):
    p = HERE / "results" / name
    if p.exists():
        FILES.append(p)
FILES += [p for p in sorted((HERE / "tables").glob("*.csv"))
          if p.name != "confirmatory_manifest.csv"]
FILES += sorted((HERE / "figures").glob("*.pdf"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


rows = []
for path in dict.fromkeys(FILES):
    if not path.exists():
        raise SystemExit(f"Missing required manifest input: {path}")
    rows.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                     sha256=sha256(path)))
out = HERE / "tables" / "confirmatory_manifest.csv"
with out.open("w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.DictWriter(fp, fieldnames=("path", "bytes", "sha256"))
    writer.writeheader(); writer.writerows(rows)
print(f"Wrote {len(rows)} manifest entries to {out}")
