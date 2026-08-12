# -*- coding: utf-8 -*-
"""
run_density_versions.py — 产出 6 个建筑密度聚类版本 + 诊断表 + 分区图, 供审核选型

分级约定(用户确认):
  Tier2 严格禁行(硬约束) / Tier1 可穿越但计惩罚 / Tier0 自由
阈值标定(用户确认): θ_forbid = 现状线位沿程最大密度 × margin
  —— 现状高速已建成并穿过部分居民区, 是"该密度可穿"的经验铁证;
     阈值定在其之上则 M-A 基准方案在 Tier2 上天然可行, 无需任何特例。

任何"Tier2 封堵走廊带"或"使 M-A 超阈"的版本一律标记为【不可用】并说明原因。
"""
import os
import sys
import math
import numpy as np
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 路径解析: 同一份脚本要能在两种布局下跑
#   (a) finalexp/exp/                 数据在 osm/, data_loader 同级, 出图到 ../figures
#   (b) data 分支 data/osm/buildings/ 数据同级, data_loader 在 ../../measured/
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find(name):
    for c in (os.path.join(_HERE, name),
              os.path.join(_HERE, 'osm', name),
              os.path.join(os.getcwd(), 'osm', name),
              os.path.join(os.getcwd(), name)):
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(name)


for _c in (_HERE, os.path.join(_HERE, '..', '..', 'measured')):
    if os.path.exists(os.path.join(_c, 'data_loader.py')):
        sys.path.insert(0, os.path.abspath(_c))
        break
sys.path.insert(0, _HERE)

import building_density as bd
from data_loader import load_alignment
from shapely.geometry import LineString

NPZ = _find('buildings_full.npz')
GRIDS = os.path.join(os.path.dirname(NPZ), 'density_grids.npz')
FIG = os.path.abspath(os.path.join(_HERE, '..', '..', '..', 'figures'))
if not os.path.isdir(FIG):
    FIG = os.path.join(_HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

a = load_alignment()
X, Y, s = a['X'], a['Y'], a['s']
lat0d, lon0d = float(a['lat'][0]), float(a['lon'][0])

G = bd.rasterize(NPZ, lat0d, lon0d, cache=GRIDS, verbose=True)

b = np.load(NPZ, allow_pickle=False)
lev = b['b_levels']
print(f"\n[层数标签] 有层数标签的建筑 {int(np.isfinite(lev).sum())}/{len(lev)} "
      f"({100*np.isfinite(lev).mean():.1f}%), "
      f"最大 {np.nanmax(lev):.0f} 层 —— V5(体量)受标签覆盖率与个别异常标签影响, 仅作参照")

# 现状线位密集采样(25 m), 用于标定与暴露量统计
rs = np.arange(0, s[-1], 25.0)
rx = np.interp(rs, s, X)
ry = np.interp(rs, s, Y)

_CL = LineString(np.c_[X, Y])


def _corridor(half_w):
    g = _CL.buffer(half_w, cap_style=2, join_style=1)
    gs = g.geoms if g.geom_type == 'MultiPolygon' else [g]
    return [np.array(gm.exterior.coords) for gm in gs]

VERSIONS = [
    dict(name='V1', metric='area',  sigma=200, margin=1.15, close=100.0, minha=1.0,
         desc='占地覆盖率 σ200 margin1.15',
         edesc='footprint coverage, sigma=200m, margin=1.15'),
    dict(name='V2', metric='area',  sigma=100, margin=1.15, close=100.0, minha=1.0,
         desc='占地覆盖率 σ100 margin1.15',
         edesc='footprint coverage, sigma=100m, margin=1.15'),
    dict(name='V3', metric='area',  sigma=300, margin=1.15, close=100.0, minha=1.0,
         desc='占地覆盖率 σ300 margin1.15',
         edesc='footprint coverage, sigma=300m, margin=1.15'),
    dict(name='V4', metric='count', sigma=200, margin=1.15, close=100.0, minha=1.0,
         desc='栋数密度 σ200 margin1.15',
         edesc='building count density, sigma=200m, margin=1.15'),
    dict(name='V5', metric='vol',   sigma=200, margin=1.15, close=100.0, minha=1.0,
         desc='体量(占地×层数) σ200 margin1.15',
         edesc='volume (footprint x levels), sigma=200m, margin=1.15'),
    dict(name='V6', metric='area',  sigma=200, margin=1.00, close=0.0, minha=0.0,
         desc='占地覆盖率 σ200 margin1.00 无形态学后处理(最严)',
         edesc='footprint coverage, sigma=200m, margin=1.00, no morphology (strictest)'),
]
UNIT = {'area': '覆盖率', 'count': '栋/公顷', 'vol': '等效层数'}
EUNIT = {'area': 'coverage ratio', 'count': 'bldg/ha', 'vol': 'equiv. storeys'}

rows = []
for v in VERSIONS:
    S = bd.density_field(G[v['metric']], v['metric'], v['sigma'])
    D_A_max, th_f, th_p, D_A = bd.calibrate(S, rx, ry, v['margin'])
    tier2, tier1, nclu = bd.cluster_tiers(S, th_p, th_f, v['close'], v['minha'])

    cell_km2 = bd.CELL ** 2 / 1e6
    a2 = float(tier2.sum()) * cell_km2
    a1 = float(tier1.sum()) * cell_km2
    ex2 = bd.route_exposure(tier2, rx, ry, rs)
    ex1 = bd.route_exposure(tier1, rx, ry, rs)
    blk5, n5, minw5, medw5 = bd.corridor_passability(tier2, X, Y, s, 500.0)
    blk25, n25, minw25, medw25 = bd.corridor_passability(tier2, X, Y, s, 2500.0)

    usable = (ex2 <= 1e-9) and (blk5 == 0)
    why = []
    if ex2 > 1e-9:
        why.append(f'M-A 穿越 Tier2 {ex2:.2f} km')
    if blk5 > 0:
        why.append(f'±500m 走廊带 {blk5} 个断面被完全封堵')

    rows.append(dict(name=v['name'], desc=v['desc'], metric=v['metric'],
                     D_A_max=D_A_max, th_p=th_p, th_f=th_f,
                     a2=a2, a1=a1, nclu=nclu, ex2=ex2, ex1=ex1,
                     blk5=blk5, minw5=minw5, medw5=medw5,
                     blk25=blk25, minw25=minw25,
                     usable=usable, why='; '.join(why) or '—'))

    print(f"\n=== {v['name']}  {v['desc']} ===")
    print(f"  D_A_max={D_A_max:.4f}  θ_pass={th_p:.4f}  θ_forbid={th_f:.4f} "
          f"({UNIT[v['metric']]})")
    print(f"  Tier2 面积 {a2:6.2f} km² ({nclu} 簇)   Tier1 面积 {a1:6.2f} km²")
    print(f"  现状线位: Tier2 {ex2:.3f} km   Tier1 {ex1:.3f} km "
          f"({100*ex1/(s[-1]/1000):.1f}% 里程)")
    print(f"  ±500m 走廊带: 封堵 {blk5}/{n5} 断面, 最窄可通 {minw5:.0f} m, "
          f"中位 {medw5:.0f} m")
    print(f"  ±2500m 走廊带: 封堵 {blk25}/{n25} 断面, 最窄可通 {minw25:.0f} m")
    print(f"  {'【可用】' if usable else '【不可用】 ' + rows[-1]['why']}")

    # -------- 分区图 --------
    fig, ax = plt.subplots(figsize=(15.2, 5.6), dpi=140)
    ax.set_rasterization_zorder(3)
    ext = [bd.X0, bd.X1, bd.Y0, bd.Y1]
    ax.imshow(S, origin='lower', extent=ext, cmap='Greys', zorder=0,
              vmin=0, vmax=np.percentile(S[S > 0], 99) if (S > 0).any() else 1)
    ov = np.zeros(S.shape, dtype=int)
    ov[tier1] = 1
    ov[tier2] = 2
    ax.imshow(np.ma.masked_where(ov == 0, ov), origin='lower', extent=ext,
              cmap=ListedColormap(['#f4a582', '#b2182b']), vmin=1, vmax=2,
              alpha=0.72, zorder=1, interpolation='nearest')
    ax.plot(X, Y, color='#0050ff', lw=2.0, zorder=4, label='existing centerline (M-A)')
    # 走廊带边界用 shapely buffer: 简单法向偏移在曲线段会自交成一团乱线
    # (偏移量接近曲率半径时尤甚), 与 draw_buildings_full.py / draw_osm.py 做法一致
    for hw, ls in ((500, '--'), (2500, ':')):
        for ring in _corridor(hw):
            ax.plot(ring[:, 0], ring[:, 1], color='#00a000', lw=1.0, ls=ls, zorder=4)
    ax.set_aspect('equal')
    ax.set_xlim(X.min() - 3000, X.max() + 3000)
    ax.set_ylim(Y.min() - 3500, Y.max() + 3500)
    ax.set_xlabel('X East (m)'); ax.set_ylabel('Y North (m)')
    ax.set_title(f"{v['name']}  {v['edesc']}  |  "
                 f"theta_pass={th_p:.3f}  theta_forbid={th_f:.3f} "
                 f"({EUNIT[v['metric']]})  |  "
                 f"Tier2 {a2:.1f} km$^2$ / {nclu} clusters  |  "
                 f"{'USABLE' if usable else 'NOT USABLE'}")
    ax.legend(handles=[
        Patch(facecolor='#b2182b', label=f'Tier2 forbidden ({a2:.1f} km²)'),
        Patch(facecolor='#f4a582', label=f'Tier1 passable w/ penalty ({a1:.1f} km²)'),
        Line2D([0], [0], color='#0050ff', lw=2,
               label=f'M-A existing (Tier2 {ex2:.2f} km, Tier1 {ex1:.2f} km)'),
        Line2D([0], [0], color='#00a000', lw=1, ls='--', label='corridor ±500 m'),
        Line2D([0], [0], color='#00a000', lw=1, ls=':', label='corridor ±2500 m'),
    ], loc='lower left', ncol=3, fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/density_{v["name"]}.png', facecolor='white')
    plt.close(fig)
    print(f"  [图] density_{v['name']}.png")

# ---------------- 汇总表 ----------------
hdr = ['版本', '说明', 'D_A_max', 'θ_pass', 'θ_forbid', 'Tier2 km²', 'Tier2 簇数',
       'Tier1 km²', 'M-A∩Tier2 km', 'M-A∩Tier1 km', '±500封堵', '±500最窄m',
       '±2500最窄m', '可用', '原因']
lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '|'.join(['---'] * len(hdr)) + '|']
csv = [','.join(hdr)]
for r in rows:
    cells = [r['name'], r['desc'], f"{r['D_A_max']:.4f}", f"{r['th_p']:.4f}",
             f"{r['th_f']:.4f}", f"{r['a2']:.2f}", str(r['nclu']), f"{r['a1']:.2f}",
             f"{r['ex2']:.3f}", f"{r['ex1']:.3f}", str(r['blk5']),
             f"{r['minw5']:.0f}", f"{r['minw25']:.0f}",
             '可用' if r['usable'] else '不可用', r['why']]
    lines.append('| ' + ' | '.join(cells) + ' |')
    csv.append(','.join(c.replace(',', ';') for c in cells))

with open(f'{FIG}/density_versions.md', 'w', encoding='utf-8') as f:
    f.write('# 建筑密度聚类版本对比（供选型审核）\n\n')
    f.write('分级：Tier2 严格禁行（硬约束）/ Tier1 可穿越但计惩罚 / Tier0 自由\n\n')
    f.write('阈值标定：`θ_forbid = 现状线位沿程最大密度 × margin`，'
            '故 margin ≥ 1 时 M-A 在 Tier2 上天然可行。\n\n')
    f.write('单位：V1/V2/V3/V6 为建筑占地覆盖率（0~1）；V4 为栋/公顷；V5 为等效层数。\n')
    f.write('**不同指标的阈值数值不可横向直接比较**，只能比各自的分区形态与诊断量。\n\n')
    f.write('\n'.join(lines) + '\n')

    n_lev = int(np.isfinite(lev).sum())
    f.write('\n## 审核要点\n\n')
    f.write(f'1. **V5（体量）证据基础最弱**：全集只有 {n_lev}/{len(lev)} '
            f'（{100*n_lev/len(lev):.1f}%）的建筑带 `building:levels` 标签，'
            f'其余按 1 层计；标签最大值 {np.nanmax(lev):.0f} 层，'
            '本区域内明显是个别异常标注。故 V5 的"体量"实际由不到 1/4 的样本驱动，'
            '建议仅作参照，不宜作为约束依据。\n\n')
    f.write('2. **`M-A∩Tier2 = 0` 对所有版本都成立，不是区分标准**：'
            '这是标定规则（θ_forbid 取现状线位最大密度之上）的必然结果，属设计保证，'
            '不能用来说明某个版本"更好"。真正要看的是下面第 5 点。\n\n')
    f.write('3. **σ 的取舍**：σ 小（V2=100m）→ 密度场保留细节，禁区碎裂成很多小簇'
            f'（{rows[1]["nclu"]} 簇），更像"逐个街区"判定；'
            'σ 大（V3=300m）→ 邻域平均，禁区合并为少数连片区域'
            f'（{rows[2]["nclu"]} 簇），更像"片区级"管控。'
            'V1（200m）居中。选哪个取决于你希望约束表达"街区尺度"还是"片区尺度"。\n\n')
    f.write('4. **OSM 完备度局限**：数量已与 OSM 数据库完全一致，但 OSM 本身在国内城区'
            '建筑测绘并非 100% 完备。图上空白只说明 **OSM 无记录**，不等于实地无建筑；'
            '因此密度偏低的区域可能被低估，郊区与村镇尤甚。\n\n')
    f.write('5. **建议横向比较的量**（按重要性）：\n'
            '   - `M-A∩Tier1 km`：现状线位穿越"居民区"的里程。这是"既有高速确实穿过居民区"'
            '这一事实的量化，太小说明 Tier1 没抓住该现象（V5 仅 0.43 km、V2 仅 0.85 km）。\n'
            '   - `±500最窄m`：走廊带被挤压后的最窄可通宽度，决定优化还有多少横向余地。\n'
            '   - `Tier2 簇数` 与 `Tier2 km²`：禁区是碎斑还是连片，面积是否过大到不合理。\n'
            '   - 目视：Tier2 应落在南侧主城区连片建成带，而非零散村镇。\n')
with open(f'{FIG}/density_versions.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(csv) + '\n')
print('\n[表] density_versions.md / .csv')
print('\n' + '\n'.join(lines))
