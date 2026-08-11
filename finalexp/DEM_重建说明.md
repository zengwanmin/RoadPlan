# optlab / finalexp — DEM 数据重建说明

为控制仓库体积，走廊带 DEM 文件（`*.npz`，合计约 30 MB）未纳入版本控制，需运行时重新下载。

## 需要的文件
- `optlab/exp/dem_wide_z14.npz`（±3 km 缓冲，48 瓦片）
- `optlab/exp/dem_xwide_z14.npz`、`finalexp/exp/dem_xwide_z14.npz`（±4.5 km 缓冲，84 瓦片）

## 重建方式
数据源：AWS Terrain Tiles（terrarium 编码，公开免密钥），缩放级 z=14。
解码：`h = R*256 + G + B/256 - 32768`。范围取实测中线经纬度外扩相应缓冲。
生成脚本逻辑见 `finalexp/exp/prep.py` 与提交历史中的下载片段；
`finalexp` 的 DEM 与 `optlab` 版在实测中线上逐点一致（差异 0.000 m），仅覆盖范围不同。

## 一致性校验
沿实测中线采样 DEM 与 GPS 实测高程相关系数 0.916，垂直基准中位偏差 1.57 m（已在 `dem.py` 以常数校正）。
