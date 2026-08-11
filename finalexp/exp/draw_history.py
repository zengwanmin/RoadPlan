# -*- coding: utf-8 -*-
"""
draw_history.py — 历史优化结果的全图层叠加可视化。
无参数: 主控, 遍历结果文件, 逐个 fork 子进程(隔离走廊带环境变量以正确解码 best_x)。
带参数 <task.json>: worker, 画一张全 layer 图。

底图: DEM + OSM 交通网 + OSM 建筑 + 现状中线。 叠加: 该版本优化后平面线位 + 走廊带。
命名: {source}_{method}_corr{W:04d}m_Cdown{降幅}_Edown{降幅}_{口径}.png
"""
import os, sys, json, math, glob, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), 'history_figures')
R_E = 6378137.0


def load_result(fp):
    d = json.load(open(fp))
    best = d.get('best', d)
    bx = best.get('best_x', d.get('best_x'))
    C = best.get('C', d.get('C')); E = best.get('E', d.get('E'))
    bl = d['baseline']; CA = bl['C']; EA = bl['E']
    return np.array(bx, float), C, E, CA, EA


def worker(task):
    t = json.load(open(task))
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from shapely.geometry import LineString
    from data_loader import load_alignment

    a = load_alignment(); X, Y = a['X'], a['Y']
    lat0 = math.radians(a['lat'][0]); lon0 = math.radians(a['lon'][0])
    def ll2xy(lon, lat):
        return (R_E*math.cos(lat0)*(np.radians(lon)-lon0), R_E*(np.radians(lat)-lat0))

    if t.get('coord') == 'stored':
        # main 分支结果: 直接读存储的平面坐标, 不解码(编码不兼容)
        d = json.load(open(t['file'])); A = d['M_A']; M = d[t['key']]
        ox = np.array(M['plane_x'], float); oy = np.array(M['plane_y'], float)
        C, E, CA, EA = M['C'], M['E'], A['C'], A['E']
    else:
        os.environ['CORRIDOR_HALF_W'] = str(t['W'])   # 必须在 import 前设定
        os.environ['E_DIRECTION'] = t['mode']
        import objective_joint as oj
        bx, C, E, CA, EA = load_result(t['file'])
        pc = oj.make_plane_context(a)
        d = oj.decode_joint(bx, pc)
        ox, oy = d['xx'], d['yy']                      # 优化后平面线位
    dcC = (1 - C/CA)*100; dcE = (1 - E/EA)*100

    # 底图数据
    dem = np.load(os.path.join(HERE, 'dem_xwide_z14.npz'))
    Em = dem['elev']; Z = int(dem['z']); x0 = int(dem['x0']); y0 = int(dem['y0'])
    Hh, Ww = Em.shape; n = 2**Z
    lon_px = (x0+(np.arange(Ww)+0.5)/256.0)/n*360.0-180.0
    lat_rd = np.arctan(np.sinh(np.pi*(1-2*(y0+(np.arange(Hh)+0.5)/256.0)/n)))
    GLON, GLAT = np.meshgrid(np.radians(lon_px), lat_rd)
    DX = R_E*math.cos(lat0)*(GLON-lon0); DY = R_E*(GLAT-lat0)
    Emask = np.where(Em < -100, np.nan, Em)
    ob = np.load(os.path.join(HERE, 'osm/obstacles.npz'), allow_pickle=False)
    OX, OY = ll2xy(ob['lines_lon'], ob['lines_lat']); OFF = ob['offsets']; KND = ob['kind']
    COL = {'road': '#777777', 'rail': '#7b3fa0', 'water': '#2b8cbe'}
    bd = np.load(os.path.join(HERE, 'osm/buildings.npz'), allow_pickle=False)
    BX, BY = ll2xy(bd['lon'], bd['lat'])

    cl = LineString(np.c_[X, Y])
    g = cl.buffer(t['W'], cap_style=2, join_style=1)
    rings = [np.array(gm.exterior.coords) for gm in
             (g.geoms if g.geom_type == 'MultiPolygon' else [g])]

    fig, ax = plt.subplots(figsize=(15, 7.6), dpi=140)
    ax.pcolormesh(DX, DY, Emask, cmap='terrain', shading='auto', alpha=.55, zorder=0)
    ax.scatter(BX, BY, s=2, color='#8b0000', alpha=.28, zorder=1)
    for i in range(len(KND)):
        sl = slice(OFF[i], OFF[i+1])
        ax.plot(OX[sl], OY[sl], color=COL[KND[i]], lw=0.5, alpha=.5, zorder=2)
    for k, ring in enumerate(rings):
        ax.plot(ring[:, 0], ring[:, 1], color='#1f6fd6', lw=1.4, ls='--', zorder=3,
                label=f'corridor +/-{t["W"]} m' if k == 0 else None)
    ax.plot(X, Y, color='0.25', lw=1.6, ls=':', zorder=4, label='existing centerline')
    ax.plot(ox, oy, 'r-', lw=2.4, zorder=5, label='optimized alignment')
    ax.plot(X[0], Y[0], 'o', ms=11, color='green', zorder=6)
    ax.plot(X[-1], Y[-1], 's', ms=10, color='red', zorder=6)
    ax.set_aspect('equal'); ax.set_xlim(-20800, 800); ax.set_ylim(-1400, 3000)
    ax.set_xlabel('X East (m)'); ax.set_ylabel('Y North (m)')
    ax.set_title('%s | %s | corridor +/-%dm | %s\nC down %.1f%%  E down %.1f%%  (C=%.3f E=%.3f (1e8 CNY))'
                 % (t['source'], t['method'], t['W'], t['mode'],
                    dcC, dcE, C/1e8, E/1e8))
    h = [Line2D([0],[0],color='#777777',lw=2,label='road'),
         Line2D([0],[0],color='#7b3fa0',lw=2,label='rail'),
         Line2D([0],[0],color='#2b8cbe',lw=2,label='water'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor='#8b0000',ms=6,label='building'),
         Line2D([0],[0],color='0.25',lw=1.6,ls=':',label='existing'),
         Line2D([0],[0],color='r',lw=2,label='optimized'),
         Line2D([0],[0],color='#1f6fd6',lw=1.4,ls='--',label='corridor')]
    ax.legend(handles=h, loc='upper center', ncol=4, fontsize=9)
    out = os.path.join(FIGDIR, t['out'])
    fig.tight_layout(); fig.savefig(out, facecolor='white'); plt.close(fig)
    print('  saved', t['out'], 'Cdown=%.1f Edown=%.1f' % (dcC, dcE), flush=True)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    R = os.path.join(HERE, 'results')
    tasks = []
    # finalexp 联合 6 走廊 × 2 口径
    for mode in ('avg', 'single'):
        for W in (500, 600, 700, 800, 900, 1000):
            f = os.path.join(R, f'entropy_dp_{mode}_w{W}.json')
            if os.path.exists(f):
                tasks.append(dict(file=f, W=W, mode=mode, method='joint',
                                  source='finalexp', out=None))
    # finalexp 两阶段 ±500 双口径
    for mode in ('avg', 'single'):
        f = os.path.join(R, f'twostage_{mode}_w500.json')
        if os.path.exists(f):
            tasks.append(dict(file=f, W=500, mode=mode, method='twostage',
                              source='finalexp', out=None))
    # finalexp ±500 精修终解
    f = os.path.join(R, 'refine_w500_final.json')
    if os.path.exists(f):
        tasks.append(dict(file=f, W=500, mode='avg', method='joint-refined',
                          source='finalexp', out=None))
    # optlab v9 宽走廊里程碑 (定稿编码 1/k^1.5, W2500, 双向)
    f = os.path.join(R, 'opt_refine_v9.json')
    if os.path.exists(f):
        tasks.append(dict(file=f, W=2500, mode='avg', method='joint-entropyDP',
                          source='optlab-v9', out=None, coord='decode'))
    # main 分支结果 (存储坐标, 不解码; 准天然DEM+OSM桥隧+官方造价口径, ewm 熵权决策)
    MB = os.path.join(R, 'main_branch')
    for fn, method, W in [
        ('joint_results.json', 'joint', 500),
        ('twostage_results.json', 'twostage', 500),
        ('joint_results_corridor250.json', 'joint', 250),
        ('joint_results_corridor500.json', 'joint', 500),
        ('twostage_results_corridor250.json', 'twostage', 250),
        ('twostage_results_corridor500.json', 'twostage', 500)]:
        fp = os.path.join(MB, fn)
        if os.path.exists(fp):
            tasks.append(dict(file=fp, W=W, mode='ewm', method=method,
                              source='main', key='M_C', coord='stored', out=None))

    for t in tasks:
        if t.get('coord') == 'stored':
            d = json.load(open(t['file'])); C = d[t['key']]['C']; E = d[t['key']]['E']
            CA = d['M_A']['C']; EA = d['M_A']['E']
        else:
            _, C, E, CA, EA = load_result(t['file'])
        t['out'] = ('%s_%s_corr%04dm_Cdown%.1f_Edown%.1f_%s.png'
                    % (t['source'], t['method'], t['W'],
                       (1-C/CA)*100, (1-E/EA)*100, t['mode']))
        tj = os.path.join(FIGDIR, '_task.json'); json.dump(t, open(tj, 'w'))
        print('[%d/%d] %s' % (tasks.index(t)+1, len(tasks), t['out']), flush=True)
        subprocess.run([sys.executable, __file__, tj], cwd=HERE, check=True)
    os.remove(os.path.join(FIGDIR, '_task.json'))
    print('done: %d history overlay figures -> history_figures/' % len(tasks))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        worker(sys.argv[1])
    else:
        main()
