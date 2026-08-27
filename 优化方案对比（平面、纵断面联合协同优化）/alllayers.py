# -*- coding: utf-8 -*-
"""
alllayers.py — 图C9: 全图层叠加图(DEM + OSM 路网/铁路/水系 + OSM 建筑 + 最终线位 + 桥隧标注)

把优化结果放回真实地理背景中核验: 优化线位穿过什么地形、是否压占既有路网/铁路/水系、
与OSM建筑的空间关系, 以及哪些桩号是靠桥梁/隧道通过的。建筑图层仅用于增强
可视化效果，不进入优化目标或约束。

【图层与数据源】
  DEM 底图   : 数据/走廊带DEM_z14_ext.npz (AWS Terrain Tiles z14, 约 8.8 m/px, 现状地表)
  OSM 障碍物 : 数据/OSM走廊带障碍物/obstacles.npz (road/rail/water 折线)
  OSM 建筑   : 数据/OSM走廊带障碍物/buildings_full.npz (完整轮廓多边形, 22590 个建筑,
               已与 Overpass `out count` 逐项核对一致, 见该目录 README 第 5 节)
  线位        : results/joint_results_w500_nodens.json 的 M_A(现状) 与
                M_C(平纵联合协同优化, 最终方案)

【桥隧判据 — 与成本模型完全一致, 不是另立标准】
  与 objective.earthwork_cost 相同: 填方段 dz>0 且(土方费>桥单价 或 填高>30m) -> 桥;
  挖方段 dz<0 且(土方费>隧单价 或 挖深>30m) -> 隧; 再扣除豁免掩膜 exempt。
  exempt = 生态强制隧道区(白云山) ∪ 立交带, 这两类结构费已计入 CB 常数项, 土方豁免,
  故【必须扣除】: 本线位若不扣 exempt 会把 108 个深挖桩号误标为隧道, 实际仅 19 个,
  其余 89 个落在立交带内(按立交结构计费)。生态区与立交带另作独立图层标出。

  掩膜由已落盘的 plane_x/plane_y 反算(dem.eco_mask_xy + pc["ic_bands"] 弦投影),
  无需决策变量; 反算结果与 JSON 中已存的 L_eco_km / L_ic_km 量级一致(见控制台自检输出)。

坐标系: 模型局部笛卡尔(X 东, Y 北, 单位 m, 原点为实测线位首点), 与 data_loader 同一投影。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import dem
from data_loader import load_alignment
from objective_joint import make_plane_context
from params import BRIDGE_TUNNEL, EARTHWORK, CASE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "数据")
DEM_NPZ = os.path.join(DATA_DIR, "走廊带DEM_z14_ext.npz")
OSM_NPZ = os.path.join(DATA_DIR, "OSM走廊带障碍物", "obstacles.npz")
BLD_NPZ = os.path.join(DATA_DIR, "OSM走廊带障碍物", "buildings_full.npz")

R_EARTH = 6378137.0
LINE_COLOR = {"road": "#777777", "rail": "#7b3fa0", "water": "#2b8cbe"}
C_BRIDGE = "#e6550d"
C_TUNNEL = "#54278f"
C_ECO = "#00701a"
C_IC = "#f2a900"


def _lonlat_to_xy(lon, lat, lat0_deg, lon0_deg):
    """与 data_loader.load_alignment 完全相同的局部平面投影(式3.3-3.4)。"""
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    x = R_EARTH * np.cos(lat0) * (np.radians(lon) - lon0)
    y = R_EARTH * (np.radians(lat) - lat0)
    return x, y


def _dem_grid_xy(lat0_deg, lon0_deg):
    """DEM 栅格每像素中心 -> 局部 XY 网格; 坏点(<-100m)置 nan。"""
    d = np.load(DEM_NPZ)
    E = d["elev"].astype(float)
    z, x0, y0 = int(d["z"]), int(d["x0"]), int(d["y0"])
    H, W = E.shape
    n = 2 ** z
    lon_px = (x0 + (np.arange(W) + 0.5) / 256.0) / n * 360.0 - 180.0
    ty = (y0 + (np.arange(H) + 0.5) / 256.0) / n
    lat_px = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * ty))))
    GLON, GLAT = np.meshgrid(lon_px, lat_px)
    GX, GY = _lonlat_to_xy(GLON, GLAT, lat0_deg, lon0_deg)
    return GX, GY, np.where(E < -100.0, np.nan, E)


def _runs(mask):
    """布尔掩膜 -> 连续 True 区段的 (起, 止) 下标对(含止)。"""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    edge = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1))


def _seg_km(mask, sta):
    """按段中点判定的掩膜覆盖长度(km), 与 objective/decode_joint 口径一致。"""
    m = np.asarray(mask, float)
    return float(np.sum(0.5 * (m[:-1] + m[1:]) * np.diff(sta))) / 1000.0


def structure_masks(scheme, pc):
    """
    复现 objective.earthwork_cost 的结构替代判据, 返回
    (use_bridge, use_tunnel, eco, ic) 四个逐桩号布尔掩膜。
    """
    sta = np.asarray(scheme["sta"], float)
    gz = np.asarray(scheme["gz_new"], float)
    design_z = np.asarray(scheme["design_z"], float)
    px = np.asarray(scheme["plane_x"], float)
    py = np.asarray(scheme["plane_y"], float)

    eco = dem.eco_mask_xy(px, py, pc["lat0"], pc["lon0"])
    x0c, y0c, ux, uy = pc["chord"]
    t = (px - x0c) * ux + (py - y0c) * uy
    ic = np.zeros(len(sta), dtype=bool)
    for ta, tb in pc["ic_bands"]:
        ic |= (t >= ta) & (t <= tb)
    exempt = eco | ic

    dz = design_z - gz
    h = np.abs(dz)
    area = CASE["road_width_m"] * h + EARTHWORK["side_slope"] * h * h
    fill_pm = np.where(dz > 0, EARTHWORK["Kh_fill_per_m3"] * area, 0.0)
    cut_pm = np.where(dz < 0, EARTHWORK["Ks_cut_per_m3"] * area, 0.0)
    b_cap = BRIDGE_TUNNEL["bridge_cost_per_km"] / 1000.0
    t_cap = BRIDGE_TUNNEL["tunnel_cost_per_km"] / 1000.0

    use_bridge = (dz > 0) & ((fill_pm > b_cap) |
                             (h > BRIDGE_TUNNEL["fill_height_bridge_m"])) & ~exempt
    use_tunnel = (dz < 0) & ((cut_pm > t_cap) |
                             (h > BRIDGE_TUNNEL["cut_depth_tunnel_m"])) & ~exempt
    return use_bridge, use_tunnel, eco, ic


def fig_C9_alllayers(d, save):
    """d: 当前自由端点、无密度约束主结果; save: make_outputs._save 回调。"""
    C = d["M_C"]
    align = load_alignment()
    lat0_deg = float(align["lat"][0])
    lon0_deg = float(align["lon"][0])
    pc = make_plane_context(align)

    sta = np.asarray(C["sta"], float)
    cx = np.asarray(C["plane_x"], float)
    cy = np.asarray(C["plane_y"], float)

    GX, GY, E = _dem_grid_xy(lat0_deg, lon0_deg)
    # 坐标对齐自检: 线位必须落在 DEM 覆盖范围内, 否则是投影原点不一致, 直接报错
    if not (GX.min() <= cx.min() and cx.max() <= GX.max()
            and GY.min() <= cy.min() and cy.max() <= GY.max()):
        raise RuntimeError(
            "线位包围盒超出 DEM 覆盖范围, 疑似投影原点不一致: "
            f"线位 X[{cx.min():.0f},{cx.max():.0f}] Y[{cy.min():.0f},{cy.max():.0f}] "
            f"vs DEM X[{GX.min():.0f},{GX.max():.0f}] Y[{GY.min():.0f},{GY.max():.0f}]")

    use_bridge, use_tunnel, eco, ic = structure_masks(C, pc)
    kmB, kmT = _seg_km(use_bridge, sta), _seg_km(use_tunnel, sta)
    kmE, kmI = _seg_km(eco, sta), _seg_km(ic, sta)
    print(f"[自检] 图C9 桥隧标注(与成本模型同判据, 已扣 exempt): "
          f"桥 {kmB:.3f} km / 隧 {kmT:.3f} km; "
          f"生态强制隧道区 {kmE:.3f} km / 立交带 {kmI:.3f} km")
    print(f"[自检] JSON 已存计费口径: L_bridge_new={C.get('L_bridge_new')} "
          f"L_tunnel_new={C.get('L_tunnel_new')} "
          f"L_eco_km={C.get('L_eco_km')} L_ic_km={C.get('L_ic_km')}")

    fig, ax = plt.subplots(figsize=(15.2, 5.4))
    # DEM 栅格(1280×3072)、建筑散点与 6002 条 OSM 折线若以矢量写入 PDF 会达 70 MB 以上,
    # 故 zorder<3 的密集图层在矢量输出中光栅化; 线位/标记/图例仍为矢量。
    ax.set_rasterization_zorder(3)
    mesh = ax.pcolormesh(GX, GY, E, cmap="terrain", shading="auto",
                         alpha=0.60, zorder=0)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.01)
    cb.set_label("Ground elevation (m, current surface)")

    b = np.load(BLD_NPZ)
    bpx, bpy = _lonlat_to_xy(b["poly_lon"], b["poly_lat"], lat0_deg, lon0_deg)
    boff, bring = b["poly_off"], b["poly_ring"]
    # 只填充外环(内环即天井, 对占压判读无意义)
    foot = [np.c_[bpx[boff[k]:boff[k + 1]], bpy[boff[k]:boff[k + 1]]]
            for k in range(len(bring)) if bring[k] == "outer"]
    ax.add_collection(PolyCollection(foot, facecolors="#8b0000",
                                     edgecolors="none", alpha=0.72, zorder=1))
    n_bld = len(b["b_id"])
    km2 = float(b["b_area_m2"].sum()) / 1e6

    o = np.load(OSM_NPZ, allow_pickle=False)
    ox, oy = _lonlat_to_xy(o["lines_lon"], o["lines_lat"], lat0_deg, lon0_deg)
    off, kind = o["offsets"], o["kind"]
    for i in range(len(kind)):
        sl = slice(off[i], off[i + 1])
        ax.plot(ox[sl], oy[sl], color=LINE_COLOR.get(str(kind[i]), "#999999"),
                lw=0.5, alpha=0.55, zorder=2)

    A = d["M_A"]
    ax.plot(np.asarray(A["plane_x"]), np.asarray(A["plane_y"]),
            color="#222222", lw=1.5, zorder=4)
    ax.plot(cx, cy, color="#d62728", lw=2.6, zorder=6)

    # 立交带作为浅色衬底(7.25 km, 若与桥隧同粗会把红色线位整段盖住);
    # 桥/隧/生态段用粗线; 极短段(<400 m, 在 25 km 幅宽上不足 1 px)另加中点环标以便定位。
    for m, col, lw, zo in ((ic, C_IC, 7.5, 5), (eco, C_ECO, 6.0, 7),
                           (use_bridge, C_BRIDGE, 6.0, 8),
                           (use_tunnel, C_TUNNEL, 6.0, 8)):
        for i0, i1 in _runs(m):
            ax.plot(cx[i0:i1 + 1], cy[i0:i1 + 1], color=col, lw=lw,
                    solid_capstyle="butt",
                    alpha=0.55 if col == C_IC else 0.9, zorder=zo)
            if col != C_IC and sta[i1] - sta[i0] < 400.0:
                mid = (i0 + i1) // 2
                ax.plot([cx[mid]], [cy[mid]], marker="o", ms=9, mfc="none",
                        mec=col, mew=2.0, zorder=zo + 1)

    ax.scatter([cx[0]], [cy[0]], c="#2ca02c", s=70, zorder=9, edgecolors="k")
    ax.scatter([cx[-1]], [cy[-1]], c="#8c564b", s=70, zorder=9, edgecolors="k")

    handles = [
        Line2D([0], [0], color="#d62728", lw=2.6,
               label=f"M-C joint-optimized ({C['L_km']:.2f} km)"),
        Line2D([0], [0], color="#222222", lw=1.5,
               label=f"M-A existing ({A['L_km']:.2f} km)"),
        Line2D([0], [0], color=C_BRIDGE, lw=6,
               label=(f"Bridge {kmB:.2f} km" if use_bridge.any()
                      else "Bridge: none in this scheme")),
        Line2D([0], [0], color=C_TUNNEL, lw=6,
               label=(f"Tunnel {kmT:.2f} km" if use_tunnel.any()
                      else "Tunnel: none in this scheme")),
        Line2D([0], [0], color=C_ECO, lw=6, label=f"Eco forced tunnel {kmE:.2f} km"),
        Line2D([0], [0], color=C_IC, lw=6, label=f"Interchange band {kmI:.2f} km"),
        Line2D([0], [0], color="#777777", lw=2, label="OSM road"),
        Line2D([0], [0], color="#7b3fa0", lw=2, label="OSM rail"),
        Line2D([0], [0], color="#2b8cbe", lw=2, label="OSM water"),
        Patch(facecolor="#8b0000", edgecolor="none",
               label=f"OSM building footprint (visual only; {n_bld}, {km2:.1f} km$^2$)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, fontsize=8.5, framealpha=0.9)

    # 裁剪到线位包围盒 + 2 km 余量: OSM 折线延伸到 ±15 km, 不裁剪则线位只占画面一薄条
    mg = 2000.0
    ax.set_xlim(cx.min() - mg, cx.max() + mg)
    ax.set_ylim(cy.min() - mg, cy.max() + mg)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting X (m)")
    ax.set_ylabel("Northing Y (m)")
    ax.set_title("Fig. C9  All-layer overlay: DEM + OSM network + buildings + "
                 "final alignment M-C with structures")
    fig.text(0.5, -0.02,
             "Bridge/tunnel criterion identical to cost model (objective.earthwork_cost): "
             "earthwork cost > structure unit price, or fill>30 m / cut>30 m; "
             "exempt bands (eco tunnel, interchanges) excluded and drawn separately.",
             ha="center", fontsize=7.5)
    save("图C9_全图层叠加")
