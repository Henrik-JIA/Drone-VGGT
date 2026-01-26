"""
点云过滤工具模块

提供点云过滤相关功能：
- filter_by_track_length: 根据track长度过滤点云
- find_duplicate_points: 使用KD-Tree找到重复点
- process_point_cloud: 点云处理流水线（组合多种滤波操作）
"""

import numpy as np
from typing import Dict, Set, Tuple
from scipy.spatial import cKDTree

# 导入其他滤波模块（用于 process_point_cloud）
from utils.voxel_downsample import voxel_downsample
from utils.statistical_filter import statistical_outlier_removal
from utils.boundary_filter import boundary_aware_filter


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

