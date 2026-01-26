"""
点云处理工具模块

提供通用的点云处理功能：
- 体素降采样 (Voxel Downsampling) - 从 utils.voxel_downsample 导入
- 统计滤波去除离群点 (Statistical Outlier Removal)
- 边界密度过滤 (Boundary-aware Filtering) - 从 utils.boundary_filter 导入
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Set
from scipy.spatial import cKDTree

# 从独立的体素降采样模块导入（为了向后兼容，在此重新导出）
from utils.voxel_downsample import voxel_downsample

# 从独立的边界处理模块导入（为了向后兼容，在此重新导出）
from utils.boundary_filter import (
    boundary_aware_filter,
    boundary_aware_filter_array,
    smooth_boundary_points
)

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




def filter_by_track_length(
    reconstruction,  # pycolmap.Reconstruction
    min_track_length: int = 3,
    verbose: bool = True
) -> Set[int]:
    """
    根据track长度过滤点云，返回需要保留的点ID集合
    
    边缘的飘散点通常只被很少的影像观测到，track长度短。
    通过设置最小track长度阈值可以有效移除这些点。
    
    Args:
        reconstruction: pycolmap.Reconstruction 对象
        min_track_length: 最小track长度，小于此值的点将被过滤
            - 0-1: 不过滤
            - 2: COLMAP默认最小值
            - 3+: 更严格的过滤
        verbose: 是否打印详细信息
        
    Returns:
        valid_point_ids: 满足条件的点ID集合
    """
    valid_point_ids = set()
    removed_count = 0
    
    for pt_id, point3D in reconstruction.points3D.items():
        track_length = len(point3D.track.elements)
        if track_length >= min_track_length:
            valid_point_ids.add(pt_id)
        else:
            removed_count += 1
    
    if verbose:
        print(f"  Track length filter (min={min_track_length}): {len(reconstruction.points3D)} -> {len(valid_point_ids)} points ({removed_count} removed)")
    
    return valid_point_ids


def find_duplicate_points(
    points1: Dict[int, np.ndarray],
    points2: Dict[int, np.ndarray],
    distance_threshold: float = 0.1
) -> Dict[int, int]:
    """
    使用KD-Tree找到两组点云中的重复点
    
    Args:
        points1: 第一组点 {point_id: xyz}
        points2: 第二组点 {point_id: xyz}
        distance_threshold: 距离阈值（米），小于此距离认为是重复点
        
    Returns:
        duplicates: 映射字典 {points2_id: points1_id}
            表示points2中的点与points1中哪个点重复
    """
    if len(points1) == 0 or len(points2) == 0:
        return {}
    
    # 构建points1的KD-Tree
    ids1 = list(points1.keys())
    xyz1 = np.array([points1[pid] for pid in ids1])
    tree = cKDTree(xyz1)
    
    # 查询points2中每个点的最近邻
    ids2 = list(points2.keys())
    xyz2 = np.array([points2[pid] for pid in ids2])
    
    distances, indices = tree.query(xyz2, k=1)
    
    # 找到距离小于阈值的点对
    duplicates = {}
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist < distance_threshold:
            duplicates[ids2[i]] = ids1[idx]
    
    return duplicates


# ========== 便捷函数 ==========

def process_point_cloud(
    points_xyz: Dict[int, np.ndarray],
    points_color: Dict[int, np.ndarray],
    voxel_size: float = 0.0,
    statistical_filter: bool = True,
    stat_nb_neighbors: int = 20,
    stat_std_ratio: float = 2.0,
    boundary_filter: bool = False,
    boundary_density_ratio: float = 0.3,
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int]]:
    """
    点云处理流水线：组合多种滤波操作
    
    处理顺序：
    1. 体素下采样（可选）
    2. 统计滤波去除离群点（可选）
    3. 边界密度过滤（可选）
    
    Args:
        points_xyz: 输入点坐标
        points_color: 输入点颜色
        voxel_size: 体素大小，0表示不下采样
        statistical_filter: 是否进行统计滤波
        stat_nb_neighbors: 统计滤波邻居数
        stat_std_ratio: 统计滤波标准差倍数
        boundary_filter: 是否进行边界滤波
        boundary_density_ratio: 边界滤波密度比例
        verbose: 是否打印详细信息
        
    Returns:
        processed_xyz: 处理后的点坐标
        processed_color: 处理后的点颜色
        id_mapping: 原始ID到新ID的映射
    """
    current_xyz = points_xyz
    current_color = points_color
    id_mapping = {pid: pid for pid in points_xyz.keys()}  # 初始为恒等映射
    
    # Step 1: 体素下采样
    if voxel_size > 0:
        if verbose:
            print(f"\n[Step 1] Voxel downsampling (voxel_size={voxel_size}m):")
        
        ds_xyz, ds_color, voxel_to_original = voxel_downsample(
            current_xyz, current_color, voxel_size, verbose
        )
        
        # 更新ID映射（多对一）
        new_id_mapping = {}
        for new_id, orig_ids in voxel_to_original.items():
            for orig_id in orig_ids:
                if orig_id in id_mapping:
                    # 追踪原始ID
                    original_original_id = None
                    for k, v in id_mapping.items():
                        if v == orig_id:
                            original_original_id = k
                            break
                    if original_original_id is not None:
                        new_id_mapping[original_original_id] = new_id
        
        if not new_id_mapping:
            # 如果追踪失败，使用简单映射
            for new_id, orig_ids in voxel_to_original.items():
                for orig_id in orig_ids:
                    new_id_mapping[orig_id] = new_id
        
        id_mapping = new_id_mapping
        current_xyz = ds_xyz
        current_color = ds_color
    
    # Step 2: 统计滤波
    if statistical_filter and len(current_xyz) > 0:
        if verbose:
            print(f"\n[Step 2] Statistical outlier removal (nb={stat_nb_neighbors}, std={stat_std_ratio}):")
        
        filtered_xyz, filtered_color = statistical_outlier_removal(
            current_xyz, current_color, stat_nb_neighbors, stat_std_ratio, verbose
        )
        
        # 更新ID映射
        valid_ids = set(filtered_xyz.keys())
        id_mapping = {k: v for k, v in id_mapping.items() if v in valid_ids}
        
        current_xyz = filtered_xyz
        current_color = filtered_color
    
    # Step 3: 边界滤波
    if boundary_filter and len(current_xyz) > 0:
        if verbose:
            print(f"\n[Step 3] Boundary-aware filter (ratio={boundary_density_ratio}):")
        
        filtered_xyz, filtered_color = boundary_aware_filter(
            current_xyz, current_color, 10, boundary_density_ratio, verbose
        )
        
        # 更新ID映射
        valid_ids = set(filtered_xyz.keys())
        id_mapping = {k: v for k, v in id_mapping.items() if v in valid_ids}
        
        current_xyz = filtered_xyz
        current_color = filtered_color
    
    if verbose:
        print(f"\n[Final] Processed point cloud: {len(current_xyz)} points")
    
    return current_xyz, current_color, id_mapping