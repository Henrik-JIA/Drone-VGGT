"""
边界处理工具模块

提供边界相关的点云处理功能：
- 边界密度过滤 (Boundary-aware Filtering)
- 边界点平滑 (Boundary Point Smoothing)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Set
from scipy.spatial import cKDTree


def boundary_aware_filter(
    points_xyz: Dict[int, np.ndarray],
    points_color: Dict[int, np.ndarray],
    nb_neighbors: int = 10,
    density_threshold_ratio: float = 0.3,
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    边界感知滤波：移除边缘稀疏区域的点
    
    原理：
    - 通过检测每个点的局部密度，移除密度过低的边缘点
    - 密度定义为：1 / (到k个最近邻的平均距离)
    - 密度低于 (全局中位数密度 * ratio) 的点被移除
    
    与 statistical_outlier_removal 的区别：
    - SOR 主要针对离群点（距离异常大的点）
    - 本方法主要针对边缘稀疏区域（密度异常低的点）
    - 两者可以配合使用
    
    适用场景：
    - 移除点云边缘的飘散点
    - 移除重叠区域边界的低密度点
    - 清理点云的不规则边缘
    
    Args:
        points_xyz: 点坐标 {point_id: xyz}
        points_color: 点颜色 {point_id: color}
        nb_neighbors: 用于计算密度的邻居数量
        density_threshold_ratio: 密度阈值比例
            - 小于 (全局中位数密度 * ratio) 的点将被移除
            - 值越小，滤波越激进
            - 建议值：0.2-0.5
        verbose: 是否打印详细信息
        
    Returns:
        filtered_xyz: 滤波后的点坐标
        filtered_color: 滤波后的点颜色
    """
    if len(points_xyz) < nb_neighbors + 1:
        return points_xyz, points_color
    
    # 构建KD-Tree
    ids = list(points_xyz.keys())
    xyz_array = np.array([points_xyz[pid] for pid in ids])
    tree = cKDTree(xyz_array)
    
    # 计算每个点的局部密度（用平均距离的倒数表示）
    distances, _ = tree.query(xyz_array, k=nb_neighbors + 1)
    mean_distances = np.mean(distances[:, 1:], axis=1)  # 排除自己
    
    # 避免除零
    mean_distances = np.maximum(mean_distances, 1e-10)
    densities = 1.0 / mean_distances
    
    # 使用中位数密度作为参考（比平均值更鲁棒）
    median_density = np.median(densities)
    threshold = median_density * density_threshold_ratio
    
    # 筛选密度足够高的点
    filtered_xyz = {}
    filtered_color = {}
    for i, pt_id in enumerate(ids):
        if densities[i] >= threshold:
            filtered_xyz[pt_id] = points_xyz[pt_id]
            filtered_color[pt_id] = points_color[pt_id]
    
    if verbose:
        removed = len(points_xyz) - len(filtered_xyz)
        print(f"  Boundary-aware filter (ratio={density_threshold_ratio}): {len(points_xyz)} -> {len(filtered_xyz)} points ({removed} removed)")
    
    return filtered_xyz, filtered_color


def boundary_aware_filter_array(
    xyz_array: np.ndarray,
    color_array: Optional[np.ndarray] = None,
    nb_neighbors: int = 10,
    density_threshold_ratio: float = 0.3,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    边界感知滤波（numpy数组版本）
    
    Args:
        xyz_array: 点坐标数组 (N, 3)
        color_array: 点颜色数组 (N, 3)，可选
        nb_neighbors: 邻居数量
        density_threshold_ratio: 密度阈值比例
        verbose: 是否打印详细信息
        
    Returns:
        filtered_xyz: 滤波后的点坐标 (M, 3)
        filtered_color: 滤波后的点颜色 (M, 3) 或 None
        inlier_mask: 内点掩码 (N,)，True 表示保留
    """
    n_points = len(xyz_array)
    if n_points < nb_neighbors + 1:
        inlier_mask = np.ones(n_points, dtype=bool)
        return xyz_array, color_array, inlier_mask
    
    # 构建KD-Tree并查询
    tree = cKDTree(xyz_array)
    distances, _ = tree.query(xyz_array, k=nb_neighbors + 1)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # 计算密度
    mean_distances = np.maximum(mean_distances, 1e-10)
    densities = 1.0 / mean_distances
    
    # 计算阈值
    median_density = np.median(densities)
    threshold = median_density * density_threshold_ratio
    
    # 筛选
    inlier_mask = densities >= threshold
    filtered_xyz = xyz_array[inlier_mask]
    filtered_color = color_array[inlier_mask] if color_array is not None else None
    
    if verbose:
        removed = n_points - len(filtered_xyz)
        print(f"  Boundary-aware filter (ratio={density_threshold_ratio}): {n_points} -> {len(filtered_xyz)} points ({removed} removed)")
    
    return filtered_xyz, filtered_color, inlier_mask


def smooth_boundary_points(
    points: Dict[int, np.ndarray],
    boundary_point_ids: Set[int],
    smoothing_radius: float = 0.5,
    smoothing_strength: float = 0.3,
    verbose: bool = True
) -> Dict[int, np.ndarray]:
    """
    对边界区域的点进行平滑处理
    
    通过将边界点向其邻域的平均位置移动，实现平滑过渡效果。
    
    Args:
        points: 点云 {point_id: xyz}
        boundary_point_ids: 边界区域点的ID集合
        smoothing_radius: 平滑搜索半径（米）
        smoothing_strength: 平滑强度 (0-1)
            - 0 = 不平滑
            - 1 = 完全移动到邻域平均位置
            - 建议值：0.2-0.5
        verbose: 是否打印详细信息
        
    Returns:
        smoothed_points: 平滑后的点云
    """
    if len(boundary_point_ids) == 0 or smoothing_strength <= 0:
        return points
    
    # 构建KD-Tree
    all_ids = list(points.keys())
    all_xyz = np.array([points[pid] for pid in all_ids])
    id_to_idx = {pid: idx for idx, pid in enumerate(all_ids)}
    tree = cKDTree(all_xyz)
    
    # 对边界点进行平滑
    smoothed_points = dict(points)
    smoothed_count = 0
    
    for pt_id in boundary_point_ids:
        if pt_id not in points:
            continue
        
        idx = id_to_idx[pt_id]
        xyz = all_xyz[idx]
        
        # 查找邻域内的点
        neighbor_indices = tree.query_ball_point(xyz, smoothing_radius)
        
        if len(neighbor_indices) > 1:  # 至少要有邻居
            # 计算邻域平均位置
            neighbor_xyz = all_xyz[neighbor_indices]
            avg_xyz = np.mean(neighbor_xyz, axis=0)
            
            # 加权平滑
            smoothed_xyz = (1 - smoothing_strength) * xyz + smoothing_strength * avg_xyz
            smoothed_points[pt_id] = smoothed_xyz
            smoothed_count += 1
    
    if verbose:
        print(f"  Smoothed {smoothed_count} boundary points")
    
    return smoothed_points

