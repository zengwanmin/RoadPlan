# -*- coding: utf-8 -*-
"""Regression checks for quotient parameterization and exact-NFE stopping."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from algorithms_nfe import VARIANTS, run_algorithm  # noqa: E402
from model_adapter import (DIM_FULL, DIM_REDUCED, OJ, RUN_JOINT,  # noqa: E402
                           diagnostics, expand_reduced, lower_bounds,
                           make_context, reduce_full, upper_bounds)


def test_exact_nfe():
    rng = np.random.default_rng(7)
    pop = rng.random((5, 9))
    lb, ub = np.zeros(9), np.ones(9)
    names = list(VARIANTS) + ["GA", "PSO", "GWO"]
    for budget in (5, 6, 17, 37):
        for name in names:
            calls = [0]
            def sphere(x):
                calls[0] += 1
                return float(np.dot(x, x))
            out = run_algorithm(name, sphere, lb, ub, pop, budget, seed=31)
            assert out["nfe"] == budget, (name, budget, out["nfe"])
            assert calls[0] == budget, (name, budget, calls[0])
        calls = [0]
        def sphere_bi(x):
            calls[0] += 1
            s = float(np.dot(x, x))
            return np.array([s, float(np.dot(x - 0.25, x - 0.25))])
        out = run_algorithm("NSGA-II", sphere_bi, lb, ub, pop, budget,
                            seed=31, weights=(0.5, 0.5))
        assert out["nfe"] == budget
        assert calls[0] == budget


def test_quotient_profile_equivalence():
    rng = np.random.default_rng(11)
    x = rng.random(DIM_FULL)
    y = reduce_full(x)
    xq = expand_reduced(y)
    assert y.shape == (DIM_REDUCED,)
    assert xq[50] == 0.5 and xq[-1] == 0.5
    ds = rng.uniform(70.0, 130.0, 224)
    gz = rng.normal(100.0, 5.0, 225)
    z_tie = (91.5, 99.0)
    z1 = OJ.profile_from_grades(x[50:], gz, ds, 0.05, z_tie=z_tie)
    z2 = OJ.profile_from_grades(xq[50:], gz, ds, 0.05, z_tie=z_tie)
    np.testing.assert_allclose(z1, z2, rtol=2e-13, atol=2e-11)
    assert np.all(lower_bounds()[:50] == 0.0)
    assert np.all(upper_bounds()[:50] == 1.0)
    assert np.all(lower_bounds()[50:] == -0.5)
    assert np.all(upper_bounds()[50:] == 1.5)


def test_current_model_equivalence():
    pc = make_context()
    full = RUN_JOINT.make_existing_x(pc, DIM_FULL)
    quotient = expand_reduced(reduce_full(full))
    a = OJ.objectives_joint(full, pc, pen_scale=1.0)
    b = OJ.objectives_joint(quotient, pc, pen_scale=1.0)
    np.testing.assert_allclose(a[:3], b[:3], rtol=2e-11, atol=1e-5)
    d = diagnostics(reduce_full(full), pc)
    assert d["feasible"], d["penalty"]


def test_repository_baseline_parity():
    benchmark_dir = HERE.parent / "多算法对比"
    sys.path.insert(0, str(benchmark_dir))
    from benchmarks import run_GA, run_GWO, run_NSGA2, run_PSO
    from algorithms_nfe import run_ga, run_gwo, run_nsgaii, run_pso
    rng = np.random.default_rng(9)
    pop = rng.random((7, 11))
    lb, ub = np.zeros(11), np.ones(11)
    scalar = lambda x: float(np.sum((x - 0.3) ** 2))
    for old, new in ((run_GA, run_ga), (run_PSO, run_pso), (run_GWO, run_gwo)):
        expected = old(scalar, lb, ub, pop, 4, 123)
        actual = new(scalar, lb, ub, pop, 35, 123)
        assert expected["nfe"] == actual["nfe"] == 35
        np.testing.assert_allclose(expected["best_x"], actual["best_x"],
                                   rtol=0, atol=1e-14)
        assert abs(expected["best_f"] - actual["best_f"]) < 1e-14

    bi = lambda x: np.array([np.sum((x - 0.2) ** 2), np.sum((x - 0.7) ** 2)])
    expected = run_NSGA2(bi, lb, ub, pop, 3, 321)
    expected_best = float(np.min(expected["front_F"] @ np.array([0.4, 0.6])))
    actual = run_nsgaii(bi, lb, ub, pop, 28, 321, weights=(0.4, 0.6))
    assert actual["nfe"] == 28
    assert actual["best_f"] <= expected_best + 1e-14


if __name__ == "__main__":
    test_exact_nfe()
    print("PASS exact-NFE checks")
    test_quotient_profile_equivalence()
    print("PASS quotient profile equivalence")
    test_current_model_equivalence()
    print("PASS current-model M-A equivalence and feasibility")
    test_repository_baseline_parity()
    print("PASS repository baseline parity at complete-generation budgets")
