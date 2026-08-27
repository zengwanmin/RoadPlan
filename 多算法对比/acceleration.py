# -*- coding: utf-8 -*-
"""结果保真的公共加速工具。

约束：float64、随机数消费顺序、候选解顺序和 ``<`` 贪婪接受判据保持不变。
批量评价只应放在候选之间不存在代内状态依赖的阶段；返回顺序与输入严格一致。
Numba 仅以 ``fastmath=False`` 编译确定性数值热点，未安装时自动退回 NumPy。
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # 可复现回退：模型与算法均不改变
    njit = None
    NUMBA_AVAILABLE = False


_EVAL_WORKERS = max(1, int(os.environ.get("ALIGN_OPT_EVAL_WORKERS", "1")))
_EXECUTOR = None
_EXECUTOR_PID = None


def set_evaluation_workers(n):
    """设置同一进程内所有算法共用的有序批量评价线程数。"""
    global _EVAL_WORKERS, _EXECUTOR, _EXECUTOR_PID
    n = max(1, int(n))
    if n != _EVAL_WORKERS and _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=True)
        _EXECUTOR = None
        _EXECUTOR_PID = None
    _EVAL_WORKERS = n
    os.environ["ALIGN_OPT_EVAL_WORKERS"] = str(n)


def evaluation_workers():
    return _EVAL_WORKERS


def _executor():
    global _EXECUTOR, _EXECUTOR_PID
    pid = os.getpid()
    if _EXECUTOR is None or _EXECUTOR_PID != pid:
        _EXECUTOR = ThreadPoolExecutor(max_workers=_EVAL_WORKERS,
                                       thread_name_prefix="ordered-eval")
        _EXECUTOR_PID = pid
    return _EXECUTOR


def evaluate_one(fobj, x, reject_above=None):
    """评价一个候选；若目标支持严格下界剪枝，则传入当前接受阈值。"""
    method = getattr(fobj, "evaluate", None)
    if method is not None:
        value = method(np.asarray(x, dtype=np.float64), reject_above)
    else:
        value = fobj(np.asarray(x, dtype=np.float64))
    value = np.asarray(value, dtype=np.float64)
    return float(value) if value.ndim == 0 else value


def evaluate_many_ordered(fobj, points, reject_above=None):
    """有序批量评价；线程数对所有算法统一，结果位置不随完成次序变化。"""
    X = np.asarray(points, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("批量候选必须是二维数组(n_candidate, dim)")
    n = len(X)
    if reject_above is None:
        limits = [None] * n
    else:
        a = np.asarray(reject_above, dtype=np.float64)
        if a.ndim == 0:
            limits = [float(a)] * n
        elif a.shape == (n,):
            limits = a.tolist()
        else:
            raise ValueError("reject_above必须为标量或与候选数相同的一维数组")

    batch = getattr(fobj, "evaluate_batch", None)
    if batch is not None:
        values = batch(X, None if reject_above is None else np.asarray(limits))
        return np.asarray(values, dtype=np.float64)

    def task(pair):
        i, limit = pair
        return evaluate_one(fobj, X[i], limit)

    indexed = list(enumerate(limits))
    if _EVAL_WORKERS == 1 or n < 2:
        values = [task(pair) for pair in indexed]
    else:
        values = list(_executor().map(task, indexed))
    return np.asarray(values, dtype=np.float64)


class ScalarObjective:
    """为标量目标附加安全拒绝接口，不改变普通 ``f(x)`` 调用语义。"""
    __slots__ = ("_full", "_bounded")

    def __init__(self, full, bounded=None):
        self._full = full
        self._bounded = bounded

    def __call__(self, x):
        return float(self._full(np.asarray(x, dtype=np.float64)))

    def evaluate(self, x, reject_above=None):
        if reject_above is None or self._bounded is None:
            return self(x)
        return float(self._bounded(np.asarray(x, dtype=np.float64),
                                   float(reject_above)))


class MappedObjective:
    """保持候选顺序的决策向量映射（如两阶段纵断面 -> 完整联合向量）。"""
    __slots__ = ("inner", "mapper")

    def __init__(self, inner, mapper):
        self.inner = inner
        self.mapper = mapper

    def __call__(self, x):
        return self.inner(self.mapper(np.asarray(x, dtype=np.float64)))

    def evaluate(self, x, reject_above=None):
        y = self.mapper(np.asarray(x, dtype=np.float64))
        method = getattr(self.inner, "evaluate", None)
        if method is None:
            return self.inner(y)
        return method(y, reject_above)


if NUMBA_AVAILABLE:
    @njit(cache=False, fastmath=False, nogil=True)
    def bilinear_kernel(E, px, py, height, width):
        out = np.empty(px.size, dtype=np.float64)
        for q in range(px.size):
            x = min(max(px[q], 0.0), width - 1.001)
            y = min(max(py[q], 0.0), height - 1.001)
            j0 = int(np.floor(x)); i0 = int(np.floor(y))
            fx = x - j0; fy = y - i0
            j1 = min(j0 + 1, width - 1); i1 = min(i0 + 1, height - 1)
            out[q] = ((1.0 - fx) * (1.0 - fy) * E[i0, j0]
                      + fx * (1.0 - fy) * E[i0, j1]
                      + (1.0 - fx) * fy * E[i1, j0]
                      + fx * fy * E[i1, j1])
        return out


    @njit(cache=False, fastmath=False, nogil=True)
    def earthwork_arrays_kernel(dz, road_width, side_slope, kh, ks,
                                bridge_cap, tunnel_cap, h_bridge, h_tunnel,
                                exempt):
        n = dz.size
        area = np.empty(n, dtype=np.float64)
        fill = np.zeros(n, dtype=np.float64)
        cut = np.zeros(n, dtype=np.float64)
        use_bridge = np.zeros(n, dtype=np.bool_)
        use_tunnel = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            h = abs(dz[i])
            a = road_width * h + side_slope * h * h
            area[i] = a
            if dz[i] > 0.0:
                value = kh * a
                if bridge_cap >= 0.0 and (value > bridge_cap or h > h_bridge):
                    value = bridge_cap
                    use_bridge[i] = True
                fill[i] = value
            elif dz[i] < 0.0:
                value = ks * a
                if tunnel_cap >= 0.0 and (value > tunnel_cap or h > h_tunnel):
                    value = tunnel_cap
                    use_tunnel[i] = True
                cut[i] = value
            if exempt[i]:
                fill[i] = 0.0
                cut[i] = 0.0
                use_bridge[i] = False
                use_tunnel[i] = False
        return area, fill, cut, use_bridge, use_tunnel


    @njit(cache=False, fastmath=False, nogil=True)
    def fuel_segments_kernel(sta, design_z, v, radius, use_radius,
                             mass, gravity, rho_air, c_aero, area_front,
                             c_roll, superelev_max, nv, cs, w_denom, eta,
                             phi, hp_in):
        """逐坡段燃油量(float64)；求和仍交给 NumPy，保持既有归约口径。"""
        n = sta.size - 1
        out = np.empty(n, dtype=np.float64)
        fair = 0.5 * rho_air * c_aero * area_front * v * v
        for i in range(n):
            grade = (design_z[i + 1] - design_z[i]) / (sta[i + 1] - sta[i])
            theta = np.arctan(grade)
            fr = c_roll * mass * gravity * np.cos(theta)
            fg = mass * gravity * np.sin(theta)
            fa = 0.0
            if use_radius:
                r = max(radius[i], 1e-9)
                sp = min(superelev_max, v * v / (gravity * r))
                fa = max((mass * v * v / r - mass * gravity * sp) / (nv * cs),
                         0.0)
            hp_ex = max(((fair + fr + fg + fa) * v) / (w_denom * eta), 0.0)
            ufc = phi * (hp_in + hp_ex)
            out[i] = ufc * ((sta[i + 1] - sta[i]) / v)
        return out


    @njit(cache=False, fastmath=False, nogil=True)
    def ev_segments_kernel(sta, design_z, v, radius, use_radius,
                           mass, gravity, rho_air, c_aero, area_front,
                           c_roll, superelev_max, nv, cs, eff):
        """逐坡段电能(J, 可正可负)；求和仍交给 NumPy。"""
        n = sta.size - 1
        out = np.empty(n, dtype=np.float64)
        fair = 0.5 * rho_air * c_aero * area_front * v * v
        for i in range(n):
            grade = (design_z[i + 1] - design_z[i]) / (sta[i + 1] - sta[i])
            theta = np.arctan(grade)
            fr = c_roll * mass * gravity * np.cos(theta)
            fg = mass * gravity * np.sin(theta)
            fa = 0.0
            if use_radius:
                r = max(radius[i], 1e-9)
                sp = min(superelev_max, v * v / (gravity * r))
                fa = max((mass * v * v / r - mass * gravity * sp) / (nv * cs),
                         0.0)
            if theta >= 0.0:
                force = fair + fr + fg + fa
            else:
                force = fair + fr + fa - abs(fg)
            e_mech = force * v * ((sta[i + 1] - sta[i]) / v)
            out[i] = e_mech / eff if e_mech >= 0.0 else e_mech * eff
        return out


    @njit(cache=False, fastmath=False, nogil=True)
    def segment_intersections_kernel(ax, ay, px1, py1, px2, py2, ia, ib):
        n = ia.size
        ok = np.zeros(n, dtype=np.bool_)
        t = np.full(n, -1.0, dtype=np.float64)
        u = np.full(n, -1.0, dtype=np.float64)
        for q in range(n):
            i = ia[q]; j = ib[q]
            rpx = ax[i + 1] - ax[i]; rpy = ay[i + 1] - ay[i]
            qwx = px2[j] - px1[j]; qwy = py2[j] - py1[j]
            dqx = px1[j] - ax[i]; dqy = py1[j] - ay[i]
            den = rpx * qwy - rpy * qwx
            if abs(den) > 1e-12:
                ok[q] = True
                t[q] = (dqx * qwy - dqy * qwx) / den
                u[q] = (dqx * rpy - dqy * rpx) / den
        return ok, t, u
else:
    bilinear_kernel = None
    earthwork_arrays_kernel = None
    fuel_segments_kernel = None
    ev_segments_kernel = None
    segment_intersections_kernel = None
