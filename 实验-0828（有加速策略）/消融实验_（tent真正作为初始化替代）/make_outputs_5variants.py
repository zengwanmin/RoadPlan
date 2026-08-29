# -*- coding: utf-8 -*-
"""
make_outputs_5variants.py — 消融图表的 5 变体精简版(V1 JS / V2 +Tent / V3 +Levy /
V4 +DE / V5 IJS), 复用 make_outputs.py 的全部绘图/制表函数, 仅过滤变体集合。

数据源不变(results/ablation_results.json + ablation_traces.json, 10 种子);
输出到独立目录 figures_5变体/ 与 tables_5变体/。
用法: python3 make_outputs_5variants.py
"""
import os

import make_outputs as M

HERE = os.path.dirname(os.path.abspath(__file__))
M.FIG = os.path.join(HERE, "figures_5变体")
M.TAB = os.path.join(HERE, "tables_5变体")
os.makedirs(M.FIG, exist_ok=True)
os.makedirs(M.TAB, exist_ok=True)

KEEP = ["V1_JS", "V2_JS+Tent", "V3_JS+Levy", "V4_JS+DE", "V5_IJS"]
M.ORDER = KEEP


def table_A1_5():
    rows = [
        ("V1", "JS(基线)", "✗", "✗", "✗"),
        ("V2", "JS + Tent", "✓", "✗", "✗"),
        ("V3", "JS + Levy", "✗", "✓", "✗"),
        ("V4", "JS + DE", "✗", "✗", "✓"),
        ("V5", "IJS(完整)", "✓", "✓", "✓"),
    ]
    hdr = ["编号", "变体名称", "Tent初始化", "Levy飞行", "差分进化DE"]
    M._write_table("表A1_消融变体组件配置表", hdr, rows)


def main():
    d = M.load()
    # 过滤到 5 变体(fig_A2 的 comps 与 traces 按存在性自适应)
    d["variants"] = {k: v for k, v in d["variants"].items() if k in KEEP}
    d["curves"] = {k: v for k, v in d["curves"].items() if k in KEEP}
    print(f"[5变体] {list(d['variants'])}")
    table_A1_5()
    M.table_A2(d)
    M.fig_A1(d)
    M.fig_A2(d)
    M.fig_A3(d)
    td = M.load_traces()
    if td is not None:
        td["traces"] = {k: v for k, v in td["traces"].items() if k in KEEP}
        M.table_A3(d, td)
        M.fig_A4(d, td)
        M.fig_A5(d, td)
        M.fig_A6(d, td)
    print("[完成] 输出到 figures_5变体/ 与 tables_5变体/")


if __name__ == "__main__":
    main()
