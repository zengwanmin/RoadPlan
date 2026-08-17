# -*- coding: utf-8 -*-
"""Adapter from the independent 273-D confirmatory space to the current model."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
SCHEME_DIR = EXPERIMENT_ROOT / "优化方案对比（平面、纵断面联合协同优化）"
RESULT_JSON = SCHEME_DIR / "results" / "joint_results_w500_dens.json"

# The current implementation uses unqualified imports (params, dem, ...).
# Put its directory first and import it exactly once in each worker process.
if str(SCHEME_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEME_DIR))

OJ = importlib.import_module("objective_joint")
RUN_JOINT = importlib.import_module("run_joint")
PARAMS = importlib.import_module("params")
DATA_LOADER = importlib.import_module("data_loader")

N_PLAN = 50
N_GRADE_RAW = 224
N_GRADE_FREE = 223
DIM_REDUCED = N_PLAN + N_GRADE_FREE
DIM_FULL = 275
IGNORED_START_SLOT = N_PLAN
FIXED_GRADE_SLOT = DIM_FULL - 1


def _load_frozen_meta() -> dict:
    with RESULT_JSON.open(encoding="utf-8") as fp:
        d = json.load(fp)
    return d["meta"]


FROZEN = _load_frozen_meta()
W_C = float(FROZEN["wC"])
W_E = float(FROZEN["wE"])
C_REF = float(FROZEN["C_ref"])
E_REF = float(FROZEN["E_ref"])
PEN_SCALE = 3.0
FEAS_TOL = 1e-6


def make_context():
    """Construct the frozen current-model context in a fresh process."""
    OJ.set_profile_step(100.0)
    OJ.set_corridor(500.0)
    OJ.set_density(True)
    if OJ.N_MODE != N_PLAN or OJ.M_PROF != 225 or OJ.DIM != DIM_FULL:
        raise RuntimeError(
            f"Unexpected upstream dimensions: N_MODE={OJ.N_MODE}, "
            f"M_PROF={OJ.M_PROF}, DIM={OJ.DIM}")
    return OJ.make_plane_context(DATA_LOADER.load_alignment())


def expand_reduced(y: np.ndarray) -> np.ndarray:
    """Map [plan50, grade223] to the upstream 275-slot representation."""
    y = np.asarray(y, dtype=float)
    if y.shape != (DIM_REDUCED,):
        raise ValueError(f"Expected ({DIM_REDUCED},), got {y.shape}")
    x = np.full(DIM_FULL, 0.5, dtype=float)
    x[:N_PLAN] = y[:N_PLAN]
    # x[50] is ignored under endpoint anchoring.
    x[N_PLAN + 1:FIXED_GRADE_SLOT] = y[N_PLAN:]
    # x[274] is fixed, which removes the centering translation direction.
    return x


def reduce_full(x: np.ndarray) -> np.ndarray:
    """Project a full vector to a quotient chart anchored at last grade=0.5."""
    x = np.asarray(x, dtype=float)
    if x.shape != (DIM_FULL,):
        raise ValueError(f"Expected ({DIM_FULL},), got {x.shape}")
    raw = x[N_PLAN + 1:]
    grade_contrasts = raw[:-1] - raw[-1] + 0.5
    return np.concatenate((x[:N_PLAN], grade_contrasts))


def canonicalize_full(x: np.ndarray) -> np.ndarray:
    """Return the unique 275-slot representative used by this experiment."""
    return expand_reduced(reduce_full(x))


def existing_reduced(pc) -> np.ndarray:
    """M-A-like seed represented in the independent coordinates."""
    return reduce_full(RUN_JOINT.make_existing_x(pc, DIM_FULL))


def lower_bounds() -> np.ndarray:
    return np.concatenate((np.zeros(N_PLAN), np.full(N_GRADE_FREE, -0.5)))


def upper_bounds() -> np.ndarray:
    return np.concatenate((np.ones(N_PLAN), np.full(N_GRADE_FREE, 1.5)))


def objectives(y: np.ndarray, pc, pen_scale: float = 1.0):
    return OJ.objectives_joint(expand_reduced(y), pc, pen_scale=pen_scale)


def scalar_value(y: np.ndarray, pc, pen_scale: float = PEN_SCALE) -> float:
    C, E, pen, info = objectives(y, pc, pen_scale=pen_scale)
    soft = (PARAMS.DENSITY["w_dense1"] * info["soft_dense1"]
            if OJ.DENSITY_ON else 0.0)
    return float(W_C * C / C_REF + W_E * E / E_REF + pen + soft)


def biobjective_value(y: np.ndarray, pc, pen_scale: float = PEN_SCALE) -> np.ndarray:
    """Two penalized normalized objectives whose weighted sum equals scalar_value."""
    C, E, pen, info = objectives(y, pc, pen_scale=pen_scale)
    soft = (PARAMS.DENSITY["w_dense1"] * info["soft_dense1"]
            if OJ.DENSITY_ON else 0.0)
    common = pen + soft
    return np.array([C / C_REF + common, E / E_REF + common], dtype=float)


def diagnostics(y: np.ndarray, pc) -> dict:
    """Evaluate final metrics with the unscaled diagnostic penalty."""
    x = expand_reduced(y)
    C, E, pen, info = OJ.objectives_joint(x, pc, pen_scale=1.0)
    hard_f = scalar_value(y, pc, pen_scale=PEN_SCALE)
    return dict(
        C=float(C), E=float(E), F_hard=float(hard_f), penalty=float(pen),
        feasible=bool(pen <= FEAS_TOL), L_km=float(info["L_km"]),
        Rmin=float(info["Rmin"]), L_dense1_km=float(info["L_dense1_km"]),
        L_dense2_km=float(info["L_dense2_km"]),
        soft_dense1=float(info["soft_dense1"]),
        dense_depth_max=float(info["dense_depth_max"]),
        L_eco_km=float(info["L_eco_km"]),
        L_cross_km=float(info["L_cross_km"]),
        L_bridge_new=float(info["L_bridge_new"]),
        L_tunnel_new=float(info["L_tunnel_new"]),
        E_fuel=float(info["E_fuel"]), E_ele=float(info["E_ele"]),
        C_PING=float(info["C_PING"]), C_TU=float(info["C_TU"]),
        best_x_full=x.tolist(),
    )


def array_sha256(a: np.ndarray) -> str:
    """Hash shape, dtype and C-order bytes so population identity is auditable."""
    a = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii"))
    h.update(b"|<f8|")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()
