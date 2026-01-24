"""
通用 DSM 导出工具

从点云文件生成 DSM (数字表面模型)。
"""

from pathlib import Path
from typing import Optional, Union

from .build_dsm import build_dsm_from_reconstruction


def export_dsm_from_point_cloud(
    point_cloud_path: Union[str, Path],
    output_path: Union[str, Path],
    resolution: float = 0.1,
    epsg_code: Optional[int] = None,
    interpolation_method: str = "nearest",
    aggregation: str = "max",
    boundary_mask: bool = True,
    boundary_alpha: float = 0.005,
    boundary_buffer: int = 2,
    verbose: bool = True,
) -> bool:
    """
    从点云文件导出 DSM (数字表面模型)。
    
    这是一个通用的 DSM 导出函数，可以处理任意点云文件。
    
    Args:
        point_cloud_path: 输入点云文件路径 (.ply 或 .las 格式)
        output_path: 输出 DSM 文件路径 (.tif)
        resolution: DSM 分辨率（米），默认 0.1m (10cm)
        epsg_code: EPSG 坐标系代码，用于 GeoTIFF 输出
        interpolation_method: 空洞填充方法，可选:
            - "nearest": 最近邻插值（快速，推荐）
            - "linear": 线性插值
            - "cubic": 三次插值（更平滑）
            - "idw": 反距离加权插值
            - "none": 不进行空洞填充
        aggregation: 单元格聚合方式:
            - "max": 取最大值（标准 DSM）
            - "min": 取最小值
            - "mean": 取平均值
        boundary_mask: 是否使用点云边界轮廓裁剪（非矩形），默认 True
        boundary_alpha: 边界轮廓参数:
            - 0: 使用凸包（convex hull）
            - >0: 使用凹包（concave hull），值越大边界越紧密
        boundary_buffer: 边界向外扩展的像素数，默认 2
        verbose: 是否输出详细日志
            
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> from SfM.dsm import export_dsm_from_point_cloud
        >>> 
        >>> # 从 PLY 点云生成 DSM
        >>> export_dsm_from_point_cloud(
        ...     point_cloud_path="merged_points.ply",
        ...     output_path="output/dsm.tif",
        ...     resolution=0.1,
        ...     epsg_code=32648,  # UTM Zone 48N
        ... )
    """
    point_cloud_path = Path(point_cloud_path)
    output_path = Path(output_path)
    
    # 验证输入文件存在
    if not point_cloud_path.exists():
        print(f"Error: 点云文件不存在: {point_cloud_path}")
        return False
    
    # 验证输入文件格式
    valid_extensions = {'.ply', '.las', '.laz'}
    if point_cloud_path.suffix.lower() not in valid_extensions:
        print(f"Error: 不支持的点云格式: {point_cloud_path.suffix}")
        print(f"  支持的格式: {', '.join(valid_extensions)}")
        return False
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print("导出数字表面模型 (DSM)")
        print(f"{'='*60}")
        print(f"  点云路径: {point_cloud_path}")
        print(f"  输出路径: {output_path}")
        print(f"  分辨率: {resolution} 米")
        if epsg_code:
            print(f"  EPSG: {epsg_code}")
        print(f"  聚合方式: {aggregation}")
        print(f"  插值方法: {interpolation_method}")
    
    # 构建 DSM
    success = build_dsm_from_reconstruction(
        point_cloud_path=point_cloud_path,
        output_path=output_path,
        resolution=resolution,
        epsg_code=epsg_code,
        interpolation_method=interpolation_method,
        aggregation=aggregation,
        boundary_mask=boundary_mask,
        boundary_alpha=boundary_alpha,
        boundary_buffer=boundary_buffer,
        verbose=verbose,
    )
    
    if success and verbose:
        print(f"  ✓ DSM 导出成功: {output_path}")
    
    return success

def find_point_cloud_in_directory(
    directory: Union[str, Path],
    preferred_names: Optional[list] = None,
) -> Optional[Path]:
    """
    在目录中查找点云文件。
    
    Args:
        directory: 搜索目录
        preferred_names: 优先查找的文件名列表（不含扩展名）
        
    Returns:
        找到的点云文件路径，未找到返回 None
    """
    directory = Path(directory)
    
    if not directory.exists():
        return None
    
    # 默认优先查找的文件名
    if preferred_names is None:
        preferred_names = [
            "sparse_points",
            "points3D", 
            "merged",
            "point_cloud",
            "dense",
        ]
    
    # 支持的扩展名
    extensions = ['.ply', '.las', '.laz']
    
    # 优先按名称查找
    for name in preferred_names:
        for ext in extensions:
            path = directory / f"{name}{ext}"
            if path.exists():
                return path
    
    # 如果没找到，返回目录中第一个点云文件
    for ext in extensions:
        files = list(directory.glob(f"*{ext}"))
        if files:
            return files[0]
    
    return None


