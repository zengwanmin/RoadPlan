# -*- coding: utf-8 -*-
"""
data_loader.py — 北环高速公路车辆轨迹数据加载与线形特征提取

数据来源: 实验/数据/数据.xlsx
  列: Num, Latitude, Longitude, Altitude, Distance, R  (14018组有效样本)
处理流程严格遵循论文 §3.3:
  1) GCS_WGS_1984 经纬度 -> 平面笛卡尔坐标 (式3.1-3.4, 大圆距离)
  2) 生成沿线里程、地面高程剖面, 供纵断面土方/能耗计算
"""
import os
import numpy as np
import pandas as pd

# 数据文件绝对路径(所有数据来自 实验/数据/)
DATA_XLSX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "数据", "数据.xlsx",
)

R_EARTH = 6378137.0  # WGS-84 长半轴 (m)  (bh_line.m)


def _haversine_seg(lat, lon):
    """相邻点间大圆距离(m). 对应论文式(3.1)(3.2)的两点地表距离公式。"""
    lat = np.radians(lat)
    lon = np.radians(lon)
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    seg = 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return seg


def load_alignment():
    """
    读取北环高速轨迹, 返回沿线里程/地面高程/平面坐标等线形特征。
    返回 dict:
      s        : 沿线累计里程 (m), 长度 N
      ground_z : 地面高程剖面 (m), 长度 N  —— 用作纵断面优化的原地面线
      X, Y     : 平面笛卡尔坐标 (m)  (式3.3-3.4, 以起点为原点)
      lat, lon : 原始经纬度
      total_km : 路段总长(km)
      R_curve  : 平面曲率半径序列(R列, m)
    """
    df = pd.read_excel(DATA_XLSX)
    lat = df["Latitude"].to_numpy(float)
    lon = df["Longitude"].to_numpy(float)
    z = df["Altitude"].to_numpy(float)

    # --- 里程 (式3.1-3.2 大圆距离累计) ---
    seg = _haversine_seg(lat, lon)
    s = np.concatenate([[0.0], np.cumsum(seg)])  # 累计里程 (m)

    # --- 平面笛卡尔坐标 (式3.3-3.4): 以首点为零点, 用局部平面近似 ---
    lat0, lon0 = np.radians(lat[0]), np.radians(lon[0])
    X = R_EARTH * np.cos(lat0) * (np.radians(lon) - lon0)   # 东向
    Y = R_EARTH * (np.radians(lat) - lat0)                  # 北向

    return dict(
        s=s,
        ground_z=z,
        X=X, Y=Y,
        lat=lat, lon=lon,
        total_km=s[-1] / 1000.0,
        R_curve=df["R"].to_numpy(float),
        N=len(s),
    )


def resample_profile(align, step_m=100.0):
    """
    将地面高程按等间距里程重采样, 得到纵断面优化的原地面线 (每 step_m 一个桩号)。
    对应论文纵断面按桩号离散、连续坡段近似为直线坡 (§4.2 假设3)。
    返回: sta (桩号里程 m), gz (对应地面高程 m)
    """
    s = align["s"]
    z = align["ground_z"]
    sta = np.arange(0.0, s[-1], step_m)
    gz = np.interp(sta, s, z)
    return sta, gz


if __name__ == "__main__":
    a = load_alignment()
    print(f"样本点数 N = {a['N']}")
    print(f"路段总长 = {a['total_km']:.3f} km")
    print(f"地面高程范围 = [{a['ground_z'].min():.2f}, {a['ground_z'].max():.2f}] m")
    sta, gz = resample_profile(a, 100.0)
    print(f"重采样桩号数 = {len(sta)} (步长100m)")
    print(f"平面坐标范围 X=[{a['X'].min():.1f},{a['X'].max():.1f}] Y=[{a['Y'].min():.1f},{a['Y'].max():.1f}] m")
