"""
点云处理工具模块

提供通用的点云处理功能：
- 体素降采样 (Voxel Downsampling)
- 统计滤波去除离群点 (Statistical Outlier Removal)
- 边界密度过滤 (Boundary-aware Filtering)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Set
from scipy.spatial import cKDTree


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

