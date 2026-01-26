"""
统计滤波工具模块

提供统计滤波相关的点云处理功能：
- 统计滤波去除离群点 (Statistical Outlier Removal)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.spatial import cKDTree


def statistical_outlier_removal(
    points_xyz: Dict[int, np.ndarray],
    points_color: Dict[int, np.ndarray],
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    统计滤波去除离群点
    
    原理：
    - 对于每个点，计算其到k个最近邻的平均距离
    - 计算所有点的平均距离的全局均值和标准差
    - 如果某点的平均距离大于 (全局均值 + std_ratio * 标准差)，则认为是离群点
    
    适用场景：
    - 去除重建中的噪声点
    - 去除错误匹配产生的离群点
    - 清理点云边缘的飘散点
    
    Args:
        points_xyz: 点坐标 {point_id: xyz}
        points_color: 点颜色 {point_id: color}
        nb_neighbors: 用于计算平均距离的邻居数量
            - 值越大，统计越稳定，但计算量越大
            - 建议值：10-50
        std_ratio: 标准差倍数阈值
            - 值越小，滤波越激进（移除更多点）
            - 值越大，滤波越宽松（保留更多点）
            - 建议值：1.5-3.0
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
    
    # 计算每个点到k个最近邻的平均距离
    distances, _ = tree.query(xyz_array, k=nb_neighbors + 1)  # +1 因为包含自己
    mean_distances = np.mean(distances[:, 1:], axis=1)  # 排除自己（距离为0）
    
    # 计算全局统计量
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # 确定阈值
    threshold = global_mean + std_ratio * global_std
    
    # 筛选内点
    inlier_mask = mean_distances < threshold
    
    filtered_xyz = {}
    filtered_color = {}
    for i, pt_id in enumerate(ids):
        if inlier_mask[i]:
            filtered_xyz[pt_id] = points_xyz[pt_id]
            filtered_color[pt_id] = points_color[pt_id]
    
    if verbose:
        removed = len(points_xyz) - len(filtered_xyz)
        print(f"  Statistical outlier removal: {len(points_xyz)} -> {len(filtered_xyz)} points ({removed} removed)")
    
    return filtered_xyz, filtered_color


def statistical_outlier_removal_array(
    xyz_array: np.ndarray,
    color_array: Optional[np.ndarray] = None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    统计滤波去除离群点（numpy数组版本）
    
    Args:
        xyz_array: 点坐标数组 (N, 3)
        color_array: 点颜色数组 (N, 3)，可选
        nb_neighbors: 邻居数量
        std_ratio: 标准差倍数阈值
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
    
    # 计算阈值
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    threshold = global_mean + std_ratio * global_std
    
    # 筛选
    inlier_mask = mean_distances < threshold
    filtered_xyz = xyz_array[inlier_mask]
    filtered_color = color_array[inlier_mask] if color_array is not None else None
    
    if verbose:
        removed = n_points - len(filtered_xyz)
        print(f"  Statistical outlier removal: {n_points} -> {len(filtered_xyz)} points ({removed} removed)")
    
    return filtered_xyz, filtered_color, inlier_mask

