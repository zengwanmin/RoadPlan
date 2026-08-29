# -*- coding: utf-8 -*-
"""
safety.py — 高速公路边坡稳定性/失稳危险度评估 (论文 §6.3, 式6.1, 表6.7)

危险度模型(式6.1):  Q = Σ_u S_u · W_u
  Q  : 公路边坡灾害危险度
  S_u: 各评价指标的等级评分 (表6.7: 低=1, 中=3, 高=5)
  W_u: 相应指标的权重系数
  j  : 评价指标个数

表6.7 失稳因素评估(评分SI: 低1/中3/高5):
  地形-地形坡度(°):   <3 低 | 3-15 中 | >15 高
  地形-沟壑密度(km/km2): <0.13 低 | 0.13-0.43 中 | >0.43 高
  降雨-年均降水日数≥25mm: <3 低 | 3-4 中 | >4 高
  地质-土类型:  硬/软岩土/碎石土 低 | 沙土 中 | 粉土/黄土 高
  植被-植被覆盖率(%): >55 低 | 20-55 中 | <20 高

数据来源:
  地形坡度 —— 由 数据.xlsx 实测地面高程沿里程求纵向坡度(可从数据导出)。
  沟壑密度/降雨/地质/植被 —— 论文 §6.3 说明来自 GIS 图件(广东省数字信息模型、
    降水资料、工程地质图、植被图), 本地无栅格源, 按研究区特征(白云山山区段)
    给定分级(工程标定), 并标注; 与线形优化直接相关的地形坡度项由实测数据驱动。
线形优化通过降低填挖高度、平顺纵坡, 主要改善"地形坡度"与开挖边坡高度项,
从而降低危险度 Q。
"""
import numpy as np

# 表6.7 评分等级 (低/中/高 -> SI)
SI_LOW, SI_MID, SI_HIGH = 1, 3, 5

# 各指标权重系数 W_u (式6.1); 论文未给具体权重, 按四因素等权归一(工程标定, 可调)
# 顺序: [地形坡度, 沟壑密度, 降雨, 地质, 植被]
WEIGHTS = np.array([0.30, 0.15, 0.20, 0.20, 0.15])
WEIGHTS = WEIGHTS / WEIGHTS.sum()


def _score_terrain_slope(slope_deg):
    """地形坡度评分(表6.7): <3->1, 3-15->3, >15->5。"""
    s = np.where(slope_deg < 3, SI_LOW, np.where(slope_deg <= 15, SI_MID, SI_HIGH))
    return s


def _score_cut_fill_height(h):
    """开挖/填筑边坡高度附加危险(与线形相关): 高填深挖增大失稳风险。
    本研究区为城区平缓段, 填挖高度普遍较小, 按细化工程分级(工程标定):
    <1m->1(低), 1-3m->3(中), >3m->5(高)。线形优化通过减少填挖高度降低该项风险。"""
    return np.where(h < 1.0, SI_LOW, np.where(h <= 3.0, SI_MID, SI_HIGH))


# 研究区(白云山山区段)非地形类因素的分级评分 (§6.3, GIS图件, 工程标定)
STATIC_SCORES = dict(
    gully_density=SI_MID,   # 沟壑密度: 山区段中等
    rainfall=SI_HIGH,       # 广州年均≥25mm 降水日数多 -> 高
    geology=SI_MID,         # 土类型: 中等
    vegetation=SI_LOW,      # 植被覆盖率高(白云山) -> 低
)


def hazard_profile(sta, ground_z, design_z):
    """
    沿线逐桩号计算边坡失稳危险度 Q(式6.1)。
    地形坡度项由设计纵断面坡度驱动(线形优化可改善);
    并叠加填挖边坡高度的附加危险。
    返回: Q_series(每桩号危险度), Q_mean(路段平均危险度)
    """
    # 地形坡度(设计纵断面) -> 角度
    grades = np.abs(np.diff(design_z) / np.diff(sta))
    slope_deg = np.degrees(np.arctan(grades))
    slope_deg = np.concatenate([slope_deg, slope_deg[-1:]])  # 对齐长度

    # 填挖高度
    h = np.abs(design_z - ground_z)

    S_terrain = _score_terrain_slope(slope_deg)
    S_cutfill = _score_cut_fill_height(h)
    # 地形项综合取坡度与填挖高度的较大风险
    S_topo = np.maximum(S_terrain, S_cutfill)

    S = np.vstack([
        S_topo,
        np.full_like(S_topo, STATIC_SCORES["gully_density"]),
        np.full_like(S_topo, STATIC_SCORES["rainfall"]),
        np.full_like(S_topo, STATIC_SCORES["geology"]),
        np.full_like(S_topo, STATIC_SCORES["vegetation"]),
    ]).astype(float)   # 5 × N

    Q_series = (WEIGHTS[:, None] * S).sum(axis=0)   # 式6.1
    return Q_series, float(Q_series.mean())


def hazard_grade(Q):
    """危险度分级(便于云图着色): 低/中/高。"""
    return np.where(Q < 2.0, 0, np.where(Q < 3.5, 1, 2))
