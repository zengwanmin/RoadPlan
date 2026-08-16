# -*- coding: utf-8 -*-
"""版本B: 冷灰单色晕渲 — Nature 风黑白地形, 线形高饱和撞色, 对比最强。"""
from fig3d_core import render

render(dict(
    tag="vB", name="冷灰单色",
    terrain=["#f7f7f7", "#e2e2e2", "#c6c6c6", "#a0a0a0", "#787878", "#ececec"],
    line_colors=("#d62728", "#ff9500", "#7b3fbf"),
    elev=42, azim=-112, light_alt=48,
))
