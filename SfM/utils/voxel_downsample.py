"""
体素降采样工具模块

提供点云体素降采样相关功能：
- voxel_downsample: 字典版本的体素降采样（返回质心和平均颜色）
- voxel_downsample_array: numpy数组版本的体素降采样
- voxel_dedup: 体素去重（保留每个体素中第一个点，内存友好）
"""

import gc
import numpy as np
from typing import Dict, Tuple, List, Optional


def voxel_downsample(
    points_xyz: Dict[int, np.ndarray],
    points_color: Dict[int, np.ndarray],
    voxel_size: float = 0.1,
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, List[int]]]:
    """
    对点云进行体素下采样，每个体素只保留一个代表点（质心）
    使用numpy向量化操作实现高效处理
    
    原理：
    - 将3D空间划分为均匀的体素网格
    - 每个体素内的所有点被合并为一个代表点
    - 代表点的位置为体素内所有点的质心
    - 代表点的颜色为体素内所有点颜色的平均值
    
    适用场景：
    - 点云密度不均匀时的均匀化处理
    - 减少点云数量以加速后续处理
    - 合并来自不同重建的重叠点
    
    Args:
        points_xyz: 点坐标字典 {point_id: xyz}，xyz 为 (3,) 的 numpy 数组
        points_color: 点颜色字典 {point_id: color}，color 为 (3,) 的 numpy 数组 (RGB, uint8)
        voxel_size: 体素大小（米），决定下采样的粒度
            - 值越小，保留的细节越多，但点数减少越少
            - 值越大，下采样越激进，点云更稀疏
            - 建议值：根据点云尺度，通常 0.01-0.5 米
        verbose: 是否打印详细信息
        
    Returns:
        downsampled_xyz: 下采样后的点坐标 {new_id: xyz}
        downsampled_color: 下采样后的点颜色 {new_id: color}
        voxel_to_original_ids: 每个新点对应的原始点ID列表 {new_id: [old_ids]}
            用于追溯合并关系，建立ID映射
            
    Example:
        >>> points_xyz = {1: np.array([0.1, 0.2, 0.3]), 2: np.array([0.15, 0.22, 0.31])}
        >>> points_color = {1: np.array([255, 0, 0]), 2: np.array([0, 255, 0])}
        >>> xyz, color, mapping = voxel_downsample(points_xyz, points_color, voxel_size=0.1)
        >>> # 两个相近的点会被合并为一个
    """
    n_points = len(points_xyz)
    if n_points == 0:
        return {}, {}, {}
    
    # Step 1: 批量提取数据到numpy数组（避免逐点操作）
    point_ids = np.array(list(points_xyz.keys()), dtype=np.int64)
    xyz_array = np.empty((n_points, 3), dtype=np.float64)
    color_array = np.empty((n_points, 3), dtype=np.float32)
    
    default_color = np.array([128, 128, 128], dtype=np.uint8)
    for i, pt_id in enumerate(point_ids):
        xyz_array[i] = points_xyz[pt_id]
        color_array[i] = points_color.get(pt_id, default_color)
    
    # Step 2: 向量化计算所有点的体素索引
    inv_voxel_size = 1.0 / voxel_size
    voxel_indices = np.floor(xyz_array * inv_voxel_size).astype(np.int64)
    
    # Step 3: 将3D体素索引编码为唯一的1D索引（用于分组）
    # 先计算偏移量使所有索引非负
    voxel_min = voxel_indices.min(axis=0)
    voxel_indices_shifted = voxel_indices - voxel_min
    
    # 计算每个维度的范围
    voxel_range = voxel_indices_shifted.max(axis=0) + 1
    
    # 编码为1D索引：idx = x + y*range_x + z*range_x*range_y
    multipliers = np.array([1, voxel_range[0], voxel_range[0] * voxel_range[1]], dtype=np.int64)
    voxel_1d = (voxel_indices_shifted * multipliers).sum(axis=1)
    
    # Step 4: 获取唯一体素和分组信息
    # argsort按体素分组排序，unique获取唯一体素和边界
    sort_idx = np.argsort(voxel_1d)
    voxel_1d_sorted = voxel_1d[sort_idx]
    
    # 找到唯一体素和它们的起始位置
    unique_voxels, unique_starts, unique_counts = np.unique(
        voxel_1d_sorted, return_index=True, return_counts=True
    )
    n_voxels = len(unique_voxels)
    
    # Step 5: 向量化计算每个体素的质心和平均颜色
    # 使用np.add.at进行分组求和
    xyz_sorted = xyz_array[sort_idx]
    color_sorted = color_array[sort_idx]
    point_ids_sorted = point_ids[sort_idx]
    
    # 为每个点分配其所属的体素索引（0到n_voxels-1）
    voxel_labels = np.zeros(n_points, dtype=np.int64)
    for i in range(n_voxels):
        start = unique_starts[i]
        end = start + unique_counts[i]
        voxel_labels[start:end] = i
    
    # 分组求和坐标
    xyz_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(xyz_sum, voxel_labels, xyz_sorted)
    
    # 分组求和颜色
    color_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(color_sum, voxel_labels, color_sorted)
    
    # 计算平均值
    counts_expanded = unique_counts[:, np.newaxis].astype(np.float64)
    centroids = xyz_sum / counts_expanded
    avg_colors = (color_sum / counts_expanded).astype(np.uint8)
    
    # Step 6: 构建输出字典
    downsampled_xyz = {}
    downsampled_color = {}
    voxel_to_original_ids = {}
    
    for i in range(n_voxels):
        new_id = i + 1
        downsampled_xyz[new_id] = centroids[i]
        downsampled_color[new_id] = avg_colors[i]
        
        # 提取该体素的原始点ID
        start = unique_starts[i]
        end = start + unique_counts[i]
        voxel_to_original_ids[new_id] = point_ids_sorted[start:end].tolist()
    
    if verbose:
        print(f"  Voxel downsampling: {n_points} -> {n_voxels} points (voxel_size={voxel_size}m)")
    
    return downsampled_xyz, downsampled_color, voxel_to_original_ids


def voxel_downsample_array(
    xyz_array: np.ndarray,
    color_array: Optional[np.ndarray] = None,
    voxel_size: float = 0.1,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], List[List[int]]]:
    """
    对点云数组进行体素下采样（numpy数组版本）
    
    与 voxel_downsample 的区别：
    - 输入/输出为 numpy 数组而非字典
    - 更适合直接处理大规模点云数据
    - 返回的映射是索引列表而非ID列表
    
    Args:
        xyz_array: 点坐标数组 (N, 3)
        color_array: 点颜色数组 (N, 3)，可选，RGB uint8
        voxel_size: 体素大小（米）
        verbose: 是否打印详细信息
        
    Returns:
        downsampled_xyz: 下采样后的点坐标 (M, 3)
        downsampled_color: 下采样后的点颜色 (M, 3)，如果输入无颜色则为 None
        voxel_to_original_indices: 每个新点对应的原始点索引列表 [[indices], ...]
    """
    n_points = len(xyz_array)
    if n_points == 0:
        empty_color = np.empty((0, 3), dtype=np.uint8) if color_array is not None else None
        return np.empty((0, 3), dtype=np.float64), empty_color, []
    
    # 向量化计算体素索引
    inv_voxel_size = 1.0 / voxel_size
    voxel_indices = np.floor(xyz_array * inv_voxel_size).astype(np.int64)
    
    # 编码为1D索引
    voxel_min = voxel_indices.min(axis=0)
    voxel_indices_shifted = voxel_indices - voxel_min
    voxel_range = voxel_indices_shifted.max(axis=0) + 1
    multipliers = np.array([1, voxel_range[0], voxel_range[0] * voxel_range[1]], dtype=np.int64)
    voxel_1d = (voxel_indices_shifted * multipliers).sum(axis=1)
    
    # 分组
    sort_idx = np.argsort(voxel_1d)
    voxel_1d_sorted = voxel_1d[sort_idx]
    unique_voxels, unique_starts, unique_counts = np.unique(
        voxel_1d_sorted, return_index=True, return_counts=True
    )
    n_voxels = len(unique_voxels)
    
    # 排序后的数据
    xyz_sorted = xyz_array[sort_idx]
    
    # 计算质心
    voxel_labels = np.zeros(n_points, dtype=np.int64)
    for i in range(n_voxels):
        start = unique_starts[i]
        end = start + unique_counts[i]
        voxel_labels[start:end] = i
    
    xyz_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(xyz_sum, voxel_labels, xyz_sorted)
    counts_expanded = unique_counts[:, np.newaxis].astype(np.float64)
    centroids = xyz_sum / counts_expanded
    
    # 处理颜色
    if color_array is not None:
        color_sorted = color_array[sort_idx].astype(np.float64)
        color_sum = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(color_sum, voxel_labels, color_sorted)
        avg_colors = (color_sum / counts_expanded).astype(np.uint8)
    else:
        avg_colors = None
    
    # 构建索引映射
    voxel_to_original_indices = []
    for i in range(n_voxels):
        start = unique_starts[i]
        end = start + unique_counts[i]
        original_indices = sort_idx[start:end].tolist()
        voxel_to_original_indices.append(original_indices)
    
    if verbose:
        print(f"  Voxel downsampling: {n_points} -> {n_voxels} points (voxel_size={voxel_size}m)")
    
    return centroids, avg_colors, voxel_to_original_indices


def voxel_dedup(
    all_xyz: np.ndarray,
    all_colors: np.ndarray,
    voxel_size: float,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用体素下采样进行近似去重（适合大数据集，内存友好）
    
    与 voxel_downsample_array 的区别：
    - 不计算质心，直接保留每个体素中第一个点
    - 内存更高效，适合处理大规模点云
    - 使用质数哈希减少碰撞
    
    内存优化：
    - 使用 numpy 向量化操作
    - 及时释放中间变量
    
    Args:
        all_xyz: 点坐标数组 (N, 3)
        all_colors: 点颜色数组 (N, 3)
        voxel_size: 体素大小（米）
        verbose: 是否打印详细信息
        
    Returns:
        merged_xyz: 去重后的点坐标 (M, 3)
        merged_colors: 去重后的点颜色 (M, 3)
    """
    n = len(all_xyz)
    
    # 计算体素索引（使用 int64 避免溢出）
    voxel_indices = np.floor(all_xyz / voxel_size).astype(np.int64)
    
    # 将 3D 体素索引转换为唯一的 1D 键（使用质数避免碰撞）
    # 使用较大的质数以减少碰撞
    p1, p2 = 73856093, 19349663
    
    # 偏移到正数范围
    offset = voxel_indices.min(axis=0)
    voxel_indices -= offset
    
    # 计算唯一键
    voxel_keys = (voxel_indices[:, 0] * p1 + 
                  voxel_indices[:, 1] * p2 + 
                  voxel_indices[:, 2])
    
    del voxel_indices
    gc.collect()
    
    # 使用 numpy.unique 找到第一次出现的索引
    _, first_indices = np.unique(voxel_keys, return_index=True)
    
    del voxel_keys
    gc.collect()
    
    # 排序索引以保持原始顺序
    first_indices = np.sort(first_indices)
    
    # 复制结果（允许原数组被释放）
    merged_xyz = all_xyz[first_indices].copy()
    merged_colors = all_colors[first_indices].copy()
    
    del first_indices
    gc.collect()
    
    if verbose:
        print(f"    Voxel dedup: {n} -> {len(merged_xyz)} points")
    
    return merged_xyz, merged_colors


# 为了向后兼容，保留 _voxel_dedup 作为别名
_voxel_dedup = voxel_dedup
