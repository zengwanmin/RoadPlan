# -*- coding: utf-8 -*-
"""
objective_reopt.py — 敏感性分析的参数化联合目标(薄封装层)

自本版本起, 情景参数注入已内置于主实验联合模型 objective_joint.objectives_joint
的 scenario 参数(准天然地面 DEM、OSM 锚定立交、生态区内生隧道、官方桥价,
与优化方案对比实验完全同口径)。本模块仅保留原接口名, 供 run_reopt.py 调用:

  objectives_reopt(x, pc, P)      == objectives_joint(x, pc, scenario=P)
  make_scalar_reopt(..., P)       == make_scalar_joint(..., scenario=P)

情景参数 P(缺省即基准情景):
  ev                 电动车渗透率 n2            (§6.5.1)
  traffic_growth     交通量年增长率 rj          (§6.5.1, 同时叠加养护费增量 式3.55)
  fuel_price_growth  油价年增长率               (§6.5.2, 逐年 (1+g)^j 进入折现求和)
  elec_price_growth  电价年增长率               (§6.5.2)
  fuel_save          燃油车节油率               (§6.5.3)
  elec_save          电动车节能率               (§6.5.3)
"""
from params import TRAFFIC
from objective_joint import objectives_joint, make_scalar_joint

DEFAULT_P = dict(ev=TRAFFIC["n2_ev"], traffic_growth=0.0,
                 fuel_price_growth=0.0, elec_price_growth=0.0,
                 fuel_save=0.0, elec_save=0.0)


def objectives_reopt(x, pc, P=None):
    """参数化联合目标: 返回 (C, E, penalty, info)。P 缺省即基准情景。"""
    return objectives_joint(x, pc, scenario=(P or DEFAULT_P))


def make_scalar_reopt(pc, wC, wE, C_ref, E_ref, P=None):
    """参数化标量目标 F = wC·Cnorm + wE·Enorm + 惩罚。"""
    return make_scalar_joint(pc, wC, wE, C_ref, E_ref,
                             scenario=(P or DEFAULT_P))
