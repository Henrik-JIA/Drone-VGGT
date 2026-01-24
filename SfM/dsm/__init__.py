"""
DSM (Digital Surface Model) 构建模块

从密集点云生成数字表面模型。

特性:
- 支持 PLY 和 LAS 格式点云输入
- 基于点云边界轮廓（凸包/凹包）裁剪，只保留有效覆盖区域
- 多种空洞填充方法
- 输出 GeoTIFF 格式

使用示例:
    from SfM.dsm import export_dsm_from_point_cloud
    
    # 通用用法（推荐）
    export_dsm_from_point_cloud(
        point_cloud_path="merged.ply",
        output_path="dsm.tif",
        resolution=0.1,
        epsg_code=32648,  # UTM Zone 48N
    )
    
    # 底层用法
    from SfM.dsm import DSMBuilder
    builder = DSMBuilder(resolution=0.05, interpolation_method="linear")
    xyz, colors = builder.load_point_cloud("merged.ply")
    dsm, metadata = builder.build_dsm(
        xyz,
        boundary_mask=True,      # 启用边界掩码
        boundary_alpha=0.0,      # 0=凸包, >0=凹包
        boundary_buffer=10,      # 边缘缓冲像素
    )
    builder.save_geotiff(dsm, metadata, "dsm.tif")
"""

from .build_dsm import (
    DSMBuilder,
    build_dsm_from_reconstruction,
)

from .export_dsm import (
    export_dsm_from_point_cloud,
    find_point_cloud_in_directory,
)

__all__ = [
    "DSMBuilder",
    "build_dsm_from_reconstruction",
    "export_dsm_from_point_cloud",
    "find_point_cloud_in_directory",
]
