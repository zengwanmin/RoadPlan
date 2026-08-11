# -*- coding: utf-8 -*-
"""prep.py — 生成两种口径的熵权参数 + 各宽度的 warm 启动文件。"""
import os, json, sys
import numpy as np

os.environ.setdefault("CORRIDOR_HALF_W", "500")


def gen_weights(mode):
    os.environ["E_DIRECTION"] = mode
    for m in list(sys.modules):
        if m in ("objective_joint", "objective", "dp_profile", "run_ce"):
            del sys.modules[m]
    from data_loader import load_alignment
    import objective_joint as oj
    import run_ce
    pc = oj.make_plane_context(load_alignment())
    xA = run_ce.make_existing_x(pc)
    base, wC, wE, C_ref, E_ref = oj.joint_baseline(pc, 200, x_seed=xA)
    fw = "%.10f,%.10f,%.6f,%.6f" % (wC, wE, C_ref, E_ref)
    open("fixw_%s.txt" % mode, "w").write(fw)
    print("[%s] wC=%.4f wE=%.4f C_ref=%.2f亿 E_ref=%.2f亿" % (
        mode, wC, wE, C_ref / 1e8, E_ref / 1e8))


def gen_warms():
    # warm500 的模态是 W=500 归一化坐标; 物理形状不变 -> 宽度 W 下按 500/W 缩放
    d = json.load(open("results/warm500.json"))
    m0 = np.array(d["best"]["modes"], float)
    for W in (500, 600, 700, 800, 900, 1000):
        m = 0.5 + (m0 - 0.5) * (500.0 / W)
        out = dict(best=dict(modes=np.clip(m, 0, 1).tolist()))
        json.dump(out, open("results/warm_w%d.json" % W, "w"))
        print("warm_w%d.json 物理形状等价缩放 ok" % W)


if __name__ == "__main__":
    gen_weights("avg")
    gen_weights("single")
    gen_warms()
