# -*- coding: utf-8 -*-
"""Fail-closed regression checks for experiment checkpoint resumption."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SCHEME = HERE.parent / "优化方案对比（平面、纵断面联合协同优化）"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCHEME))

import run_confirmatory as confirm  # noqa: E402
import run_joint as joint  # noqa: E402


def _write(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_joint_checkpoint_guard():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "joint.partial.json")
        config = {"schema": "test", "corridor": 500, "weights": [0.5, 0.5]}
        state = joint._load_checkpoint(path, config, ["M_B", "M_C"])
        state["records"] = [{"tag": "M_B"}]
        joint._atomic_json(path, state)
        resumed = joint._load_checkpoint(path, config, ["M_B", "M_C"])
        assert [r["tag"] for r in resumed["records"]] == ["M_B"]

        changed = dict(config, weights=[0.4, 0.6])
        try:
            joint._load_checkpoint(path, changed, ["M_B", "M_C"])
        except RuntimeError as exc:
            assert "refusing to mix" in str(exc)
        else:
            raise AssertionError("joint checkpoint accepted changed weights")

        fresh = joint._load_checkpoint(path, changed, ["M_B", "M_C"], fresh=True)
        assert fresh["records"] == []


def test_confirmatory_checkpoint_guard():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "confirmatory.partial.json"
        config = {"schema": "test", "budget": 600000, "weights": [0.5, 0.5]}
        fingerprint = confirm._fingerprint(config)
        pop_hash = {0: "population-hash"}
        record = {
            "key": "GA::0", "repetition": 0, "nfe": 600000,
            "nfe_exact": True, "config_fingerprint": fingerprint,
            "initial_population_sha256": pop_hash[0],
        }
        state = {
            "schema": "confirmatory-checkpoint-v2", "config": config,
            "config_fingerprint": fingerprint, "records": [record],
        }
        _write(path, state)
        resumed = confirm._load_checkpoint(path, config, ["GA::0"], pop_hash)
        assert resumed["records"][0]["key"] == "GA::0"

        changed = dict(config, upstream_result_sha256="new-result")
        try:
            confirm._load_checkpoint(path, changed, ["GA::0"], pop_hash)
        except RuntimeError as exc:
            assert "refusing to mix" in str(exc)
        else:
            raise AssertionError("confirmatory checkpoint accepted changed source result")

        fresh = confirm._load_checkpoint(
            path, changed, ["GA::0"], pop_hash, fresh=True)
        assert fresh["records"] == []


if __name__ == "__main__":
    test_joint_checkpoint_guard()
    print("PASS joint checkpoint fingerprint guard")
    test_confirmatory_checkpoint_guard()
    print("PASS confirmatory checkpoint fingerprint guard")
