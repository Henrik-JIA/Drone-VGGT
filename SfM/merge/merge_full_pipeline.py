"""
合并两个 pycolmap Reconstruction 的工具模块

通过共同影像进行Sim3对齐后合并两个reconstruction。
支持点云融合：去除重复点、平滑过渡。
"""

import copy
import numpy as np
import pycolmap
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Set
from scipy.spatial import cKDTree

# 从体素降采样模块导入
from utils.voxel_downsample import voxel_downsample

# 从统计滤波模块导入
from utils.statistical_filter import statistical_outlier_removal

# 从边界处理模块导入
from utils.boundary_filter import (
    boundary_aware_filter,
    boundary_aware_filter_array,
    smooth_boundary_points,
)

# 从点云过滤模块导入
from utils.point_filter import (
    filter_by_track_length,
    find_duplicate_points,
    process_point_cloud,
)

def estimate_translation_only(src_points: np.ndarray, tgt_points: np.ndarray) -> np.ndarray:
    """
    只估计平移量（假设尺度=1，无旋转）
    
    变换公式: tgt = src + t
    
    Args:
        src_points: 源点坐标 (N, 3)
        tgt_points: 目标点坐标 (N, 3)
        
    Returns:
        t: 平移向量 (3,)
    """
    assert src_points.shape == tgt_points.shape
    
    # 计算质心差作为平移量
    src_mean = np.mean(src_points, axis=0)
    tgt_mean = np.mean(tgt_points, axis=0)
    
    t = tgt_mean - src_mean
    return t


def estimate_scale_and_translation(src_points: np.ndarray, tgt_points: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    估计缩放和平移（无旋转）
    
    变换公式: tgt = scale * src + t
    
    使用最小二乘法：
    - 先对齐质心
    - 根据到质心的距离比值估计缩放因子
    - 然后计算平移
    
    Args:
        src_points: 源点坐标 (N, 3)
        tgt_points: 目标点坐标 (N, 3)
        
    Returns:
        scale: 缩放因子
        t: 平移向量 (3,)
    """
    assert src_points.shape == tgt_points.shape
    n = src_points.shape[0]
    
    # 计算质心
    src_mean = np.mean(src_points, axis=0)
    tgt_mean = np.mean(tgt_points, axis=0)
    
    # 去中心化
    src_centered = src_points - src_mean
    tgt_centered = tgt_points - tgt_mean
    
    # 计算缩放因子：使用最小二乘法
    # min sum ||tgt_centered - scale * src_centered||^2
    # 解: scale = sum(tgt_centered · src_centered) / sum(src_centered · src_centered)
    numerator = np.sum(tgt_centered * src_centered)
    denominator = np.sum(src_centered * src_centered)
    
    if denominator < 1e-10:
        scale = 1.0
    else:
        scale = numerator / denominator
    
    # 确保缩放因子为正且合理
    if scale <= 0:
        scale = 1.0
    
    # 计算平移向量
    t = tgt_mean - scale * src_mean
    
    return scale, t


def estimate_sim3_umeyama(src_points: np.ndarray, tgt_points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    使用 Umeyama 算法估计从源点到目标点的 Sim3 变换 (scale, rotation, translation)
    
    变换公式: tgt = scale * R @ src + t
    
    Args:
        src_points: 源点坐标 (N, 3)
        tgt_points: 目标点坐标 (N, 3)
        
    Returns:
        scale: 缩放因子
        R: 旋转矩阵 (3, 3)
        t: 平移向量 (3,)
    """
    assert src_points.shape == tgt_points.shape
    n = src_points.shape[0]
    
    # 计算质心
    src_mean = np.mean(src_points, axis=0)
    tgt_mean = np.mean(tgt_points, axis=0)
    
    # 去中心化
    src_centered = src_points - src_mean
    tgt_centered = tgt_points - tgt_mean
    
    # 计算方差
    src_var = np.sum(src_centered ** 2) / n
    
    # 计算协方差矩阵
    cov_matrix = (tgt_centered.T @ src_centered) / n
    
    # SVD 分解
    U, S, Vt = np.linalg.svd(cov_matrix)
    
    # 处理反射情况
    d = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2, 2] = -1
    
    # 计算旋转矩阵
    R = U @ d @ Vt
    
    # 计算缩放因子
    scale = np.trace(np.diag(S) @ d) / src_var
    
    # 计算平移向量
    t = tgt_mean - scale * R @ src_mean
    
    return scale, R, t


def estimate_sim3_with_ransac(
    src_points: np.ndarray, 
    tgt_points: np.ndarray,
    max_iterations: int = 1000,
    inlier_threshold: float = 0.5,
    min_inliers: int = 2
) -> Tuple[Optional[pycolmap.Sim3d], np.ndarray]:
    """
    使用 RANSAC 估计 Sim3 变换
    
    Args:
        src_points: 源点坐标 (N, 3)
        tgt_points: 目标点坐标 (N, 3)
        max_iterations: 最大迭代次数
        inlier_threshold: 内点阈值（米）
        min_inliers: 最小内点数量
        
    Returns:
        sim3: pycolmap.Sim3d 变换对象，失败返回 None
        inlier_mask: 内点掩码
    """
    n = src_points.shape[0]
    
    # 如果点数少于3，直接使用所有点估计（不用RANSAC）
    if n < 3:
        try:
            scale, R, t = estimate_sim3_umeyama(src_points, tgt_points)
            rotation = pycolmap.Rotation3d(R)
            sim3 = pycolmap.Sim3d(scale, rotation, t)
            return sim3, np.ones(n, dtype=bool)
        except Exception as e:
            print(f"  Warning: Failed to estimate Sim3 with {n} points: {e}")
            return None, np.zeros(n, dtype=bool)
    
    best_sim3 = None
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_inlier_count = 0
    
    for _ in range(max_iterations):
        # 随机选择3个点
        indices = np.random.choice(n, size=3, replace=False)
        src_sample = src_points[indices]
        tgt_sample = tgt_points[indices]
        
        try:
            # 估计变换
            scale, R, t = estimate_sim3_umeyama(src_sample, tgt_sample)
            
            # 应用变换到所有源点
            transformed = scale * (src_points @ R.T) + t
            
            # 计算误差
            errors = np.linalg.norm(transformed - tgt_points, axis=1)
            
            # 统计内点
            inlier_mask = errors < inlier_threshold
            inlier_count = np.sum(inlier_mask)
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask = inlier_mask
                
                # 使用所有内点重新估计
                if inlier_count >= 3:
                    scale, R, t = estimate_sim3_umeyama(
                        src_points[inlier_mask], 
                        tgt_points[inlier_mask]
                    )
                
                rotation = pycolmap.Rotation3d(R)
                best_sim3 = pycolmap.Sim3d(scale, rotation, t)
                
        except Exception as e:
            continue
    
    if best_inlier_count < min_inliers:
        print(f"  Warning: Only {best_inlier_count} inliers found, less than minimum {min_inliers}")
        return None, best_inlier_mask
    
    return best_sim3, best_inlier_mask


def find_common_images(
    recon1: pycolmap.Reconstruction, 
    recon2: pycolmap.Reconstruction
) -> Dict[int, int]:
    """
    通过影像名称找到两个 reconstruction 中的共同影像
    
    Args:
        recon1: 第一个 reconstruction
        recon2: 第二个 reconstruction
        
    Returns:
        映射字典 {recon1_image_id: recon2_image_id}
    """
    # 建立 recon2 的影像名称到 ID 的映射
    name_to_id2 = {img.name: img_id for img_id, img in recon2.images.items()}
    
    # 查找共同影像
    common_images = {}
    for img_id1, img1 in recon1.images.items():
        if img1.name in name_to_id2:
            common_images[img_id1] = name_to_id2[img1.name]
    
    return common_images


def get_camera_centers(
    reconstruction: pycolmap.Reconstruction,
    image_ids: List[int]
) -> np.ndarray:
    """
    获取指定影像的相机中心坐标
    
    Args:
        reconstruction: pycolmap.Reconstruction 对象
        image_ids: 影像 ID 列表
        
    Returns:
        相机中心坐标 (N, 3)
    """
    centers = []
    for img_id in image_ids:
        if img_id in reconstruction.images:
            image = reconstruction.images[img_id]
            # 相机中心 = -R^T @ t (从 cam_from_world 提取)
            R = np.array(image.cam_from_world.rotation.matrix())
            t = np.array(image.cam_from_world.translation)
            center = -R.T @ t
            centers.append(center)
    return np.array(centers)


def compute_overlap_region_bbox(
    recon1: pycolmap.Reconstruction,
    recon2_aligned: pycolmap.Reconstruction,
    common_image_ids_recon1: List[int],
    common_image_ids_recon2: List[int],
    expand_ratio: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算重叠区域的边界框
    
    Args:
        recon1: 第一个重建
        recon2_aligned: 对齐后的第二个重建
        common_image_ids_recon1: recon1中共同影像的ID列表
        common_image_ids_recon2: recon2中共同影像的ID列表
        expand_ratio: 边界框扩展比例
        
    Returns:
        bbox_min: 边界框最小点
        bbox_max: 边界框最大点
    """
    # 获取共同影像的相机中心
    centers1 = get_camera_centers(recon1, common_image_ids_recon1)
    centers2 = get_camera_centers(recon2_aligned, common_image_ids_recon2)
    
    all_centers = np.vstack([centers1, centers2])
    
    # 计算边界框
    center_mean = np.mean(all_centers, axis=0)
    center_range = np.max(all_centers, axis=0) - np.min(all_centers, axis=0)
    
    # 扩展边界框
    half_size = center_range * expand_ratio / 2
    bbox_min = center_mean - half_size
    bbox_max = center_mean + half_size
    
    return bbox_min, bbox_max


def is_point_in_bbox(xyz: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
    """检查点是否在边界框内"""
    return np.all(xyz >= bbox_min) and np.all(xyz <= bbox_max)


def merge_all_points_simple(
    recon1: pycolmap.Reconstruction,
    recon2_aligned: pycolmap.Reconstruction,
    overlap_image_ids_recon2: Set[int],
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int], Dict[int, int]]:
    """
    简单合并两个重建的所有点（不做任何下采样）
    
    Args:
        recon1: 基准重建
        recon2_aligned: 已对齐的第二个重建
        overlap_image_ids_recon2: recon2中重叠影像的ID集合
        verbose: 是否打印详细信息
        
    Returns:
        all_points: 所有点 {new_id: xyz}
        all_colors: 所有颜色 {new_id: color}
        point3D_id_map1: recon1点ID到新ID的映射
        point3D_id_map2: recon2点ID到新ID的映射
    """
    all_points = {}
    all_colors = {}
    point3D_id_map1 = {}
    point3D_id_map2 = {}
    
    next_id = 1
    
    # 添加recon1的所有点
    for pt_id, point3D in recon1.points3D.items():
        all_points[next_id] = point3D.xyz
        all_colors[next_id] = point3D.color
        point3D_id_map1[pt_id] = next_id
        next_id += 1
    
    # 添加recon2的非重叠影像的点
    for pt_id, point3D in recon2_aligned.points3D.items():
        # 只添加非重叠影像观测到的点
        has_non_overlap_obs = False
        for track_elem in point3D.track.elements:
            if track_elem.image_id not in overlap_image_ids_recon2:
                has_non_overlap_obs = True
                break
        
        if has_non_overlap_obs:
            all_points[next_id] = point3D.xyz
            all_colors[next_id] = point3D.color
            point3D_id_map2[pt_id] = next_id
            next_id += 1
    
    if verbose:
        print(f"\nSimple merge:")
        print(f"  Points from Recon1: {len(point3D_id_map1)}")
        print(f"  Points from Recon2 (non-overlap): {len(point3D_id_map2)}")
        print(f"  Total before processing: {len(all_points)}")
    
    return all_points, all_colors, point3D_id_map1, point3D_id_map2


def merge_points_with_spatial_dedup(
    recon1: pycolmap.Reconstruction,
    recon2_aligned: pycolmap.Reconstruction,
    duplicate_threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int], Dict[int, int]]:
    """
    基于空间距离的重复点检测和合并
    
    策略：
    1. 以recon1的所有点为基准
    2. 对于recon2的每个点，检查是否与recon1的点距离过近
    3. 如果距离小于阈值，认为是重复点，不添加
    4. 如果距离大于阈值，添加为新点
    
    Args:
        recon1: 基准重建
        recon2_aligned: 已对齐的第二个重建
        duplicate_threshold: 重复点判定距离阈值（米）
        verbose: 是否打印详细信息
        
    Returns:
        all_points: 所有点 {new_id: xyz}
        all_colors: 所有颜色 {new_id: color}
        point3D_id_map1: recon1点ID到新ID的映射
        point3D_id_map2: recon2点ID到新ID的映射
    """
    all_points = {}
    all_colors = {}
    point3D_id_map1 = {}
    point3D_id_map2 = {}
    
    next_id = 1
    
    # Step 1: 添加recon1的所有点
    recon1_ids = list(recon1.points3D.keys())
    recon1_xyz = np.array([recon1.points3D[pid].xyz for pid in recon1_ids])
    
    for pt_id, point3D in recon1.points3D.items():
        all_points[next_id] = point3D.xyz
        all_colors[next_id] = point3D.color
        point3D_id_map1[pt_id] = next_id
        next_id += 1
    
    if verbose:
        print(f"\nSpatial deduplication merge:")
        print(f"  Points from Recon1: {len(point3D_id_map1)}")
    
    # Step 2: 构建recon1点云的KD-Tree
    if len(recon1_xyz) > 0:
        tree = cKDTree(recon1_xyz)
    else:
        tree = None
    
    # Step 3: 检查recon2的每个点
    duplicate_count = 0
    added_count = 0
    
    for pt_id, point3D in recon2_aligned.points3D.items():
        xyz = point3D.xyz
        
        # 检查是否与recon1的点重复
        is_duplicate = False
        if tree is not None:
            dist, _ = tree.query(xyz, k=1)
            if dist < duplicate_threshold:
                is_duplicate = True
                duplicate_count += 1
        
        if not is_duplicate:
            all_points[next_id] = xyz
            all_colors[next_id] = point3D.color
            point3D_id_map2[pt_id] = next_id
            next_id += 1
            added_count += 1
    
    if verbose:
        print(f"  Points from Recon2: {len(recon2_aligned.points3D)}")
        print(f"    - Duplicates removed: {duplicate_count}")
        print(f"    - Unique points added: {added_count}")
        print(f"  Total merged points: {len(all_points)}")
    
    return all_points, all_colors, point3D_id_map1, point3D_id_map2


def refine_alignment_from_matched_points(
    recon2: pycolmap.Reconstruction,
    matched_pairs: Dict[Tuple[int, int], Dict],
    transform_mode: str = "translation",
    min_cell_size: Optional[int] = None,
    max_cell_size: Optional[int] = None,
    max_distance_3d: Optional[float] = None,
    use_ransac: bool = True,
    ransac_threshold: float = 0.3,
    verbose: bool = True
) -> Tuple[pycolmap.Reconstruction, Optional[pycolmap.Sim3d]]:
    """
    基于2D匹配得到的3D点对，重新精化对齐变换
    
    核心思想：
    - 初始对齐使用相机中心（点数少，可能不够精确）
    - 2D匹配后得到大量对应的3D点对
    - 利用这些点对重新估计更精确的变换
    
    Args:
        recon2: 原始的第二个重建（未变换的）
        matched_pairs: 匹配的点对信息，包含recon1_xyz、recon2_xyz和cell_size
        transform_mode: 变换模式
            - "translation": 只估计平移（scale=1, R=I）
            - "scale_translation": 估计缩放和平移，不旋转（R=I）
            - "sim3": 完整Sim3变换（scale, R, t）
        min_cell_size: 只使用cell_size >= min_cell_size的匹配点（None=不限制下限）
        max_cell_size: 只使用cell_size <= max_cell_size的匹配点（None=不限制上限）
        max_distance_3d: 只使用3D距离 <= max_distance_3d的匹配点（None=不限制）
        use_ransac: 是否使用RANSAC（仅transform_mode="sim3"时有效）
        ransac_threshold: RANSAC内点阈值（米）
        verbose: 是否打印详细信息
        
    Returns:
        recon2_refined: 精化对齐后的recon2
        refined_transform: 精化后的变换
    """
    # 过滤匹配点对（根据cell_size范围和3D距离筛选）
    filtered_pairs = {}
    for key, data in matched_pairs.items():
        cell_size = data.get('cell_size', 999)
        distance_3d = data.get('distance_3d', float('inf'))
        
        # 检查是否在指定的cell_size范围内
        if min_cell_size is not None and cell_size < min_cell_size:
            continue
        if max_cell_size is not None and cell_size > max_cell_size:
            continue
        # 检查3D距离是否在阈值内
        if max_distance_3d is not None and distance_3d > max_distance_3d:
            continue
        filtered_pairs[key] = data
    
    if verbose and (min_cell_size is not None or max_cell_size is not None or max_distance_3d is not None):
        filter_parts = []
        if min_cell_size == max_cell_size and min_cell_size is not None:
            filter_parts.append(f"cell_size == {min_cell_size}")
        else:
            if min_cell_size is not None and max_cell_size is not None:
                filter_parts.append(f"{min_cell_size} <= cell_size <= {max_cell_size}")
            elif min_cell_size is not None:
                filter_parts.append(f"cell_size >= {min_cell_size}")
            elif max_cell_size is not None:
                filter_parts.append(f"cell_size <= {max_cell_size}")
        if max_distance_3d is not None:
            filter_parts.append(f"distance_3d <= {max_distance_3d}m")
        filter_desc = " AND ".join(filter_parts) if filter_parts else "none"
        print(f"\n=== Refinement using matched 3D points ===")
        print(f"  Filter: {filter_desc}")
        print(f"  Filtered pairs: {len(filtered_pairs)} / {len(matched_pairs)}")
    
    if len(filtered_pairs) < 3:
        if verbose:
            print(f"  Warning: Not enough matched pairs ({len(filtered_pairs)}) for refinement")
        return recon2, None
    
    # 提取匹配点对的坐标
    # 注意：matched_pairs中的recon2_xyz是已经变换过的坐标
    # 我们需要用原始的recon2坐标来重新估计变换
    src_points = []  # recon2原始坐标
    tgt_points = []  # recon1坐标
    
    for (r1_id, r2_id), data in filtered_pairs.items():
        # recon1的xyz是目标
        tgt_points.append(data['recon1_xyz'])
        # recon2的原始xyz需要从原始recon2中获取
        if r2_id in recon2.points3D:
            src_points.append(recon2.points3D[r2_id].xyz)
        else:
            # 如果找不到，跳过这个点对
            continue
    
    if len(src_points) < 3:
        if verbose:
            print(f"  Warning: Not enough valid source points ({len(src_points)}) for refinement")
        return recon2, None
    
    src_points = np.array(src_points)
    tgt_points = np.array(tgt_points)
    
    if verbose:
        print(f"  Valid point pairs for refinement: {len(src_points)}")
        print(f"  Transform mode: {transform_mode}")
    
    # 估计变换
    if transform_mode == "translation":
        # 只估计平移
        translation = estimate_translation_only(src_points, tgt_points)
        identity_rotation = pycolmap.Rotation3d(np.eye(3))
        refined_transform = pycolmap.Sim3d(1.0, identity_rotation, translation)
        
        if verbose:
            print(f"  Translation-only refinement:")
            print(f"    Translation: {translation}")
            
            # 计算残差
            transformed_src = src_points + translation
            errors = np.linalg.norm(transformed_src - tgt_points, axis=1)
            print(f"    Residual errors:")
            print(f"      Mean: {np.mean(errors):.6f} m")
            print(f"      Median: {np.median(errors):.6f} m")
            print(f"      Max: {np.max(errors):.6f} m")
            print(f"      Std: {np.std(errors):.6f} m")
            
    elif transform_mode == "scale_translation":
        # 估计缩放和平移，不旋转
        scale, translation = estimate_scale_and_translation(src_points, tgt_points)
        identity_rotation = pycolmap.Rotation3d(np.eye(3))
        refined_transform = pycolmap.Sim3d(scale, identity_rotation, translation)
        
        if verbose:
            print(f"  Scale + Translation refinement (no rotation):")
            print(f"    Scale: {scale:.6f}")
            print(f"    Translation: {translation}")
            
            # 计算残差
            transformed_src = scale * src_points + translation
            errors = np.linalg.norm(transformed_src - tgt_points, axis=1)
            print(f"    Residual errors:")
            print(f"      Mean: {np.mean(errors):.6f} m")
            print(f"      Median: {np.median(errors):.6f} m")
            print(f"      Max: {np.max(errors):.6f} m")
            print(f"      Std: {np.std(errors):.6f} m")
            
    else:  # sim3
        # 完整的Sim3变换
        if use_ransac:
            refined_transform, inlier_mask = estimate_sim3_with_ransac(
                src_points, tgt_points,
                inlier_threshold=ransac_threshold
            )
            if refined_transform is None:
                if verbose:
                    print(f"  Warning: RANSAC failed to estimate transform")
                return recon2, None
            
            if verbose:
                print(f"  Sim3 refinement with RANSAC:")
                print(f"    Inliers: {np.sum(inlier_mask)}/{len(inlier_mask)}")
        else:
            scale, R, t = estimate_sim3_umeyama(src_points, tgt_points)
            rotation = pycolmap.Rotation3d(R)
            refined_transform = pycolmap.Sim3d(scale, rotation, t)
        
        if verbose:
            print(f"    Scale: {refined_transform.scale:.6f}")
            print(f"    Translation: {refined_transform.translation}")
    
    # 应用精化后的变换
    # recon2_refined = pycolmap.Reconstruction(recon2)
    recon2_refined = copy.deepcopy(recon2)
    recon2_refined.transform(refined_transform)
    
    if verbose:
        # 验证变换后的点对距离
        refined_errors = []
        for (r1_id, r2_id), data in filtered_pairs.items():
            if r2_id in recon2_refined.points3D:
                refined_xyz = recon2_refined.points3D[r2_id].xyz
                target_xyz = data['recon1_xyz']
                error = np.linalg.norm(refined_xyz - target_xyz)
                refined_errors.append(error)
        
        if refined_errors:
            print(f"  After refinement, matched point distances:")
            print(f"    Mean: {np.mean(refined_errors):.6f} m")
            print(f"    Median: {np.median(refined_errors):.6f} m")
            print(f"    Max: {np.max(refined_errors):.6f} m")
    
    return recon2_refined, refined_transform


def update_matched_pairs_coordinates(
    matched_pairs: Dict[Tuple[int, int], Dict],
    recon1: pycolmap.Reconstruction,
    recon2_aligned: pycolmap.Reconstruction,
    verbose: bool = True
) -> Dict[Tuple[int, int], Dict]:
    """
    更新已有匹配对中的recon2坐标信息（不重新做2D匹配）
    
    因为2D像素坐标是固定的，匹配对（点ID对应关系）不会因3D变换而改变。
    这个函数只更新 recon2_xyz、average_xyz、distance_3d 等坐标相关信息。
    
    Args:
        matched_pairs: 已有的匹配对
        recon1: 基准重建（用于获取recon1的颜色信息）
        recon2_aligned: 精化对齐后的recon2
        verbose: 是否打印详细信息
        
    Returns:
        updated_pairs: 更新坐标后的匹配对
    """
    updated_pairs = {}
    distances_3d = []
    
    for (r1_id, r2_id), data in matched_pairs.items():
        if r2_id not in recon2_aligned.points3D:
            continue
        if r1_id not in recon1.points3D:
            continue
        
        # 获取recon1坐标和颜色
        recon1_xyz = data['recon1_xyz']
        recon1_color = np.array(recon1.points3D[r1_id].color)
        
        # 获取更新后的recon2坐标和颜色
        new_recon2_xyz = np.array(recon2_aligned.points3D[r2_id].xyz)
        new_recon2_color = np.array(recon2_aligned.points3D[r2_id].color)
        
        # 计算新的平均坐标和距离
        new_average_xyz = (recon1_xyz + new_recon2_xyz) / 2.0
        new_average_color = ((recon1_color.astype(float) + new_recon2_color.astype(float)) / 2.0).astype(np.uint8)
        new_distance_3d = np.linalg.norm(recon1_xyz - new_recon2_xyz)
        
        updated_pairs[(r1_id, r2_id)] = {
            'recon1_xyz': recon1_xyz,
            'recon2_xyz': new_recon2_xyz,
            'average_xyz': new_average_xyz,
            'average_color': new_average_color,
            'cell_size': data['cell_size'],
            'distance_2d': data['distance_2d'],
            'distance_3d': new_distance_3d
        }
        distances_3d.append(new_distance_3d)
    
    if verbose and distances_3d:
        print(f"\n  Updated matched pairs coordinates:")
        print(f"    Total pairs: {len(updated_pairs)}")
        print(f"    3D distance after refinement:")
        print(f"      Mean: {np.mean(distances_3d):.4f}m")
        print(f"      Median: {np.median(distances_3d):.4f}m")
        print(f"      Max: {np.max(distances_3d):.4f}m")
    
    return updated_pairs


def match_3d_points_by_2d_observations(
    recon1: pycolmap.Reconstruction,
    recon2: pycolmap.Reconstruction,
    common_images: Dict[int, int],
    cell_sizes: List[int] = None,
    verbose: bool = True
) -> Dict[Tuple[int, int], Dict]:
    """
    通过2D像素坐标匹配两个重建中的对应3D点
    
    核心思路：同一个物理点在不同重建中的2D观测位置应该一致
    
    Args:
        recon1: 第一个重建（基准）
        recon2: 第二个重建（已对齐）
        common_images: 共同影像映射 {recon1_image_id: recon2_image_id}
        cell_sizes: 多尺度匹配的cell大小列表
        verbose: 是否打印详细信息
        
    Returns:
        matched_pairs: {(recon1_point3D_id, recon2_point3D_id): {
            'recon1_xyz': xyz,
            'recon2_xyz': xyz,
            'average_xyz': xyz,
            'average_color': color,
            'cell_size': int,
            'distance_2d': float,
            'distance_3d': float
        }}
    """
    from collections import defaultdict
    
    if cell_sizes is None:
        cell_sizes = [1, 3, 5, 10, 15, 20, 30, 50]
    
    # 获取影像尺寸
    sample_img_id = list(common_images.keys())[0]
    sample_image = recon1.images[sample_img_id]
    camera = recon1.cameras[sample_image.camera_id]
    img_width = int(camera.width)
    img_height = int(camera.height)
    
    # 建立recon1重叠区3D点到2D观测的映射
    recon1_overlap_image_ids = set(common_images.keys())
    recon1_3d_to_2d = {}  # {point3D_id: {'xyz': xyz, 'color': color, 'observations': [(image_id, pixel_xy), ...]}}
    
    for pt3d_id, point3D in recon1.points3D.items():
        observations = []
        for track_elem in point3D.track.elements:
            if track_elem.image_id in recon1_overlap_image_ids:
                image = recon1.images[track_elem.image_id]
                pixel_xy = image.points2D[track_elem.point2D_idx].xy
                observations.append((track_elem.image_id, np.array(pixel_xy)))
        
        if observations:
            recon1_3d_to_2d[pt3d_id] = {
                'xyz': np.array(point3D.xyz),
                'color': np.array(point3D.color),
                'observations': observations
            }
    
    # 建立recon2重叠区3D点到2D观测的映射
    recon2_overlap_image_ids = set(common_images.values())
    recon2_3d_to_2d = {}
    
    for pt3d_id, point3D in recon2.points3D.items():
        observations = []
        for track_elem in point3D.track.elements:
            if track_elem.image_id in recon2_overlap_image_ids:
                image = recon2.images[track_elem.image_id]
                pixel_xy = image.points2D[track_elem.point2D_idx].xy
                observations.append((track_elem.image_id, np.array(pixel_xy)))
        
        if observations:
            recon2_3d_to_2d[pt3d_id] = {
                'xyz': np.array(point3D.xyz),
                'color': np.array(point3D.color),
                'observations': observations
            }
    
    if verbose:
        print(f"\n2D-based 3D point matching:")
        print(f"  Recon1 overlap region 3D points: {len(recon1_3d_to_2d)}")
        print(f"  Recon2 overlap region 3D points: {len(recon2_3d_to_2d)}")
    
    # 建立影像ID映射（recon1 -> recon2）
    image_id_mapping = common_images
    
    # 多尺度匹配
    matched_pairs = {}
    matched_recon1_ids = set()
    matched_recon2_ids = set()
    
    for cell_size in cell_sizes:
        cell_width = cell_size
        cell_height = cell_size
        grid_cols = int(np.ceil(img_width / cell_width))
        grid_rows = int(np.ceil(img_height / cell_height))
        actual_cell_width = img_width / grid_cols
        actual_cell_height = img_height / grid_rows
        
        # 构建recon1的格网索引
        recon1_grid = defaultdict(lambda: defaultdict(list))  # {image_id: {(cell_y, cell_x): [point_data, ...]}}
        
        for pt3d_id, data in recon1_3d_to_2d.items():
            if pt3d_id in matched_recon1_ids:
                continue
            for img_id, pixel_xy in data['observations']:
                cell_x = min(int(pixel_xy[0] / actual_cell_width), grid_cols - 1)
                cell_y = min(int(pixel_xy[1] / actual_cell_height), grid_rows - 1)
                recon1_grid[img_id][(cell_y, cell_x)].append({
                    'point3D_id': pt3d_id,
                    'pixel_xy': pixel_xy,
                    'xyz': data['xyz'],
                    'color': data['color']
                })
        
        # 构建recon2的格网索引
        recon2_grid = defaultdict(lambda: defaultdict(list))
        
        for pt3d_id, data in recon2_3d_to_2d.items():
            if pt3d_id in matched_recon2_ids:
                continue
            for img_id, pixel_xy in data['observations']:
                cell_x = min(int(pixel_xy[0] / actual_cell_width), grid_cols - 1)
                cell_y = min(int(pixel_xy[1] / actual_cell_height), grid_rows - 1)
                recon2_grid[img_id][(cell_y, cell_x)].append({
                    'point3D_id': pt3d_id,
                    'pixel_xy': pixel_xy,
                    'xyz': data['xyz'],
                    'color': data['color']
                })
        
        # 在每个影像对的每个cell中进行匹配
        for recon1_img_id, recon2_img_id in image_id_mapping.items():
            for cell_key in recon1_grid[recon1_img_id]:
                recon1_points = recon1_grid[recon1_img_id][cell_key]
                recon2_points = recon2_grid[recon2_img_id].get(cell_key, [])
                
                if not recon1_points or not recon2_points:
                    continue
                
                # 收集候选匹配
                candidates = []
                for r1_pt in recon1_points:
                    if r1_pt['point3D_id'] in matched_recon1_ids:
                        continue
                    for r2_pt in recon2_points:
                        if r2_pt['point3D_id'] in matched_recon2_ids:
                            continue
                        
                        # 计算2D距离
                        dist_2d = np.linalg.norm(r1_pt['pixel_xy'] - r2_pt['pixel_xy'])
                        candidates.append({
                            'recon1_pt': r1_pt,
                            'recon2_pt': r2_pt,
                            'distance_2d': dist_2d
                        })
                
                # 按2D距离排序，贪心选择（保证一对一）
                candidates.sort(key=lambda x: x['distance_2d'])
                
                used_r1 = set()
                used_r2 = set()
                
                for cand in candidates:
                    r1_id = cand['recon1_pt']['point3D_id']
                    r2_id = cand['recon2_pt']['point3D_id']
                    
                    if r1_id in used_r1 or r2_id in used_r2:
                        continue
                    if r1_id in matched_recon1_ids or r2_id in matched_recon2_ids:
                        continue
                    
                    # 添加匹配
                    r1_xyz = cand['recon1_pt']['xyz']
                    r2_xyz = cand['recon2_pt']['xyz']
                    r1_color = cand['recon1_pt']['color']
                    r2_color = cand['recon2_pt']['color']
                    
                    matched_pairs[(r1_id, r2_id)] = {
                        'recon1_xyz': r1_xyz,
                        'recon2_xyz': r2_xyz,
                        'average_xyz': (r1_xyz + r2_xyz) / 2.0,
                        'average_color': ((r1_color.astype(float) + r2_color.astype(float)) / 2.0).astype(np.uint8),
                        # 'average_color': np.array([255, 0, 0], dtype=np.uint8),
                        'cell_size': cell_size,
                        'distance_2d': cand['distance_2d'],
                        'distance_3d': np.linalg.norm(r1_xyz - r2_xyz)
                    }
                    
                    matched_recon1_ids.add(r1_id)
                    matched_recon2_ids.add(r2_id)
                    used_r1.add(r1_id)
                    used_r2.add(r2_id)
        
        if verbose:
            print(f"    cell_size={cell_size}: 累计匹配 {len(matched_pairs)} 对")
    
    if verbose and matched_pairs:
        distances_3d = [v['distance_3d'] for v in matched_pairs.values()]
        print(f"  匹配点对3D距离统计:")
        print(f"    平均: {np.mean(distances_3d):.4f}m")
        print(f"    中位数: {np.median(distances_3d):.4f}m")
        print(f"    最大: {np.max(distances_3d):.4f}m")
    
    return matched_pairs


def merge_points_with_2d_matching(
    recon1: pycolmap.Reconstruction,
    recon2_aligned: pycolmap.Reconstruction,
    common_images: Dict[int, int],
    cell_sizes: List[int] = None,
    keep_unmatched_overlap: bool = False,
    spatial_dedup_threshold: float = 0.1,
    refine_alignment: bool = False,
    recon2_original: Optional[pycolmap.Reconstruction] = None,
    refine_transform_mode: str = "scale_translation",
    refine_cell_range: Tuple[int, int] = (1, 3),
    refine_stages: List = None,
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int], Dict[int, int], Optional[pycolmap.Reconstruction]]:
    """
    基于2D像素匹配的点云融合
    
    策略：
    1. 通过2D观测匹配重叠区的对应3D点
    2. （可选）利用匹配的点对精化对齐，重新匹配
    3. 匹配的点取平均坐标
    4. 重叠区域：未匹配的点根据keep_unmatched_overlap决定是否保留
    5. 非重叠区域：recon1和recon2的点都保留
    
    Args:
        recon1: 基准重建
        recon2_aligned: 已对齐的第二个重建
        common_images: 共同影像映射 {recon1_image_id: recon2_image_id}
        cell_sizes: 多尺度匹配的cell大小列表
        keep_unmatched_overlap: 是否保留重叠区域未匹配的点（False=丢弃，避免双层）
        spatial_dedup_threshold: 空间去重距离阈值（米），小于此距离的点被认为是重复点
        refine_alignment: 是否利用匹配的点对精化对齐
        recon2_original: 原始未变换的recon2（精化对齐时需要）
        refine_transform_mode: 精化时的全局变换模式（可被阶段配置覆盖）
            - "translation": 只平移
            - "scale_translation": 缩放+平移，不旋转（推荐）
            - "sim3": 完整Sim3
        refine_cell_range: 精化时使用的cell_size范围 (min_cell, max_cell)
            只有cell_size在此范围内的高质量匹配点才会用于精化
            默认 (1, 3) 表示使用最高质量的匹配点
            这个参数是全局的，只需指定一次，不需要在每个阶段重复
        refine_stages: 多阶段精化的配置列表，支持灵活格式：
            - 数字 (如 10.0): 只指定max_dist_3d，使用全局transform_mode
            - 字符串 (如 "translation"): 只指定变换模式，不筛选距离
            - 元组 (max_dist_3d, transform_mode): 同时指定距离和变换模式
            - None: 不筛选距离，使用全局transform_mode
            
            工作流程：
            1. 第一阶段：用 refine_cell_range 范围内的所有点进行粗对齐
            2. 后续阶段：在满足 cell_range 的点中，根据3D距离进一步筛选
            
            例如 refine_cell_range=(1,3), refine_stages=["translation", 10.0, 5.0, 2.0]：
            - 第1阶段：用cell_size=[1,3]的所有点，只平移粗对齐
            - 第2阶段：用dist_3d<=10m的点继续精化
            - 第3阶段：用dist_3d<=5m的点继续精化
            - 第4阶段：用dist_3d<=2m的点最终精化
        verbose: 是否打印详细信息
        
    Returns:
        all_points: 所有点 {new_id: xyz}
        all_colors: 所有颜色 {new_id: color}
        point3D_id_map1: recon1点ID到新ID的映射
        point3D_id_map2: recon2点ID到新ID的映射
        recon2_refined: 精化后的recon2（如果refine_alignment=True），否则为None
    """
    # 设置默认的精化阶段
    if refine_stages is None:
        refine_stages = [None]  # 默认单阶段精化，使用全局transform_mode，不筛选距离
    
    # 解析全局 cell_size 范围
    min_cell, max_cell = refine_cell_range
    
    # Step 1: 通过2D匹配找到对应的3D点对
    matched_pairs = match_3d_points_by_2d_observations(
        recon1, recon2_aligned, common_images, cell_sizes, verbose
    )
    
    # Step 1.5: 可选的多阶段精化对齐
    recon2_refined = None
    if refine_alignment and recon2_original is not None and len(matched_pairs) >= 3:
        if verbose:
            print(f"\n--- Multi-stage refinement using matched 3D points ---")
            print(f"    Cell range: {min_cell} <= cell_size <= {max_cell}")
            print(f"    Stages: {refine_stages}")
            print(f"    Global transform mode: {refine_transform_mode}")
        
        # 当前使用的recon2（会在每个阶段更新）
        current_recon2 = recon2_original
        
        # 注意：2D匹配关系(matched_pairs中的点对ID)不会因3D变换而改变
        # 因为2D像素坐标是固定的，所以不需要每个阶段重新匹配
        # refine_alignment_from_matched_points 会直接从 current_recon2 中读取最新的3D坐标
        
        # 依次执行每个精化阶段
        for stage_idx, stage_config in enumerate(refine_stages):
            # 解析阶段配置：支持灵活格式
            # - 数字 (如 10.0): 只指定max_dist_3d，使用全局transform_mode
            # - 字符串 (如 "translation"): 只指定变换模式，不筛选距离
            # - 元组 (max_dist_3d, transform_mode): 同时指定距离和变换模式
            # - None: 不筛选距离，使用全局transform_mode
            
            if stage_config is None:
                # None: 不筛选距离，使用全局transform_mode
                max_dist_3d = None
                stage_transform_mode = refine_transform_mode
            elif isinstance(stage_config, str):
                # 字符串: 只指定变换模式，不筛选距离
                max_dist_3d = None
                stage_transform_mode = stage_config
            elif isinstance(stage_config, (int, float)):
                # 数字: 只指定max_dist_3d，使用全局transform_mode
                max_dist_3d = float(stage_config)
                stage_transform_mode = refine_transform_mode
            elif isinstance(stage_config, tuple):
                # 元组: (max_dist_3d, transform_mode)
                if len(stage_config) == 2:
                    max_dist_3d, stage_transform_mode = stage_config
                    if max_dist_3d is not None:
                        max_dist_3d = float(max_dist_3d)
                else:
                    raise ValueError(f"Invalid stage config tuple: {stage_config}, expected (max_dist_3d, transform_mode)")
            else:
                raise ValueError(f"Invalid stage config type: {type(stage_config)}, value: {stage_config}")
            
            # 构建阶段描述
            stage_parts = []
            if max_dist_3d is not None:
                stage_parts.append(f"dist_3d <= {max_dist_3d}m")
            else:
                stage_parts.append("all points")
            stage_parts.append(f"mode={stage_transform_mode}")
            stage_desc = ", ".join(stage_parts)
            
            if verbose:
                print(f"\n  [Stage {stage_idx + 1}/{len(refine_stages)}] {stage_desc}")
            
            # 利用匹配的点对精化变换（使用指定范围的cell_size和3D距离）
            # 注意：这里传入的 matched_pairs 保持不变，但 recon2 会在每阶段更新
            # refine_alignment_from_matched_points 内部会从 current_recon2 读取最新的3D坐标
            stage_refined, stage_transform = refine_alignment_from_matched_points(
                recon2=current_recon2,
                matched_pairs=matched_pairs,  # 2D匹配对不变，无需重新计算
                transform_mode=stage_transform_mode,  # 使用该阶段指定的变换模式
                min_cell_size=min_cell,
                max_cell_size=max_cell,
                max_distance_3d=max_dist_3d,
                use_ransac=True,
                ransac_threshold=0.5,
                verbose=verbose
            )
            
            if stage_refined is not None and stage_transform is not None:
                # 更新当前recon2为精化后的版本
                current_recon2 = stage_refined
                recon2_refined = stage_refined
                
                # 每个阶段后更新matched_pairs中的distance_3d，以便下一阶段使用
                # 这是迭代优化的关键：精化后距离会变化，可以用更严格的阈值筛选
                matched_pairs = update_matched_pairs_coordinates(
                    matched_pairs, recon1, current_recon2, verbose=False
                )
            else:
                if verbose:
                    print(f"\n  [Stage {stage_idx + 1}] Refinement failed, skipping remaining stages")
                break
        
        if recon2_refined is not None:
            # 更新recon2_aligned为最终精化后的版本
            recon2_aligned = recon2_refined
            
            # 最终更新一次matched_pairs（带详细输出）
            matched_pairs = update_matched_pairs_coordinates(
                matched_pairs, recon1, recon2_aligned, verbose
            )
    
    # 建立映射（优化：一次遍历完成所有映射构建）
    matched_recon1_ids = set()
    matched_recon2_ids = set()
    recon2_to_recon1 = {}
    recon1_to_average = {}
    
    for (r1_id, r2_id), data in matched_pairs.items():
        matched_recon1_ids.add(r1_id)
        matched_recon2_ids.add(r2_id)
        recon2_to_recon1[r2_id] = r1_id
        recon1_to_average[r1_id] = (data['average_xyz'], data['average_color'])
    
    if verbose:
        print(f"\n2D-based merge:")
        print(f"  Matched 3D point pairs: {len(matched_pairs)}")
        print(f"  Recon1 total points: {len(recon1.points3D)}")
        print(f"  Recon2 total points: {len(recon2_aligned.points3D)}")
        print(f"  Spatial dedup threshold: {spatial_dedup_threshold}m")
    
    all_points = {}
    all_colors = {}
    point3D_id_map1 = {}
    point3D_id_map2 = {}
    
    next_id = 1
    
    # Step 2: 添加匹配点的平均坐标
    matched_added = 0
    for r1_id, (xyz, color) in recon1_to_average.items():
        all_points[next_id] = xyz
        all_colors[next_id] = color
        point3D_id_map1[r1_id] = next_id
        next_id += 1
        matched_added += 1
    
    # Step 3: 为recon2已匹配的点建立到recon1的映射
    for r2_id, r1_id in recon2_to_recon1.items():
        point3D_id_map2[r2_id] = point3D_id_map1[r1_id]
    
    # Step 4: 预先收集所有未匹配点数据到数组（批量处理优化）
    # Recon1 未匹配点
    recon1_unmatched_ids = []
    recon1_unmatched_xyz = []
    recon1_unmatched_colors = []
    for pt_id, point3D in recon1.points3D.items():
        if pt_id not in matched_recon1_ids:
            recon1_unmatched_ids.append(pt_id)
            recon1_unmatched_xyz.append(point3D.xyz)
            recon1_unmatched_colors.append(point3D.color)
    
    # Recon2 未匹配点
    recon2_unmatched_ids = []
    recon2_unmatched_xyz = []
    recon2_unmatched_colors = []
    for pt_id, point3D in recon2_aligned.points3D.items():
        if pt_id not in matched_recon2_ids:
            recon2_unmatched_ids.append(pt_id)
            recon2_unmatched_xyz.append(point3D.xyz)
            recon2_unmatched_colors.append(point3D.color)
    
    # 转换为 NumPy 数组
    recon1_unmatched_xyz_arr = np.array(recon1_unmatched_xyz) if recon1_unmatched_xyz else np.empty((0, 3))
    recon2_unmatched_xyz_arr = np.array(recon2_unmatched_xyz) if recon2_unmatched_xyz else np.empty((0, 3))
    
    # Step 5: 添加 recon1 未匹配点（批量 KD-Tree 查询）
    recon1_unmatched_added = 0
    recon1_unmatched_as_duplicate = 0
    
    if len(recon1_unmatched_ids) > 0:
        # 构建 recon2 未匹配点的 KD-Tree
        if len(recon2_unmatched_xyz_arr) > 0:
            recon2_tree = cKDTree(recon2_unmatched_xyz_arr)
            # 批量查询：一次查询所有 recon1 未匹配点到 recon2 的最近距离
            distances, _ = recon2_tree.query(recon1_unmatched_xyz_arr, k=1)
            near_recon2_mask = distances < spatial_dedup_threshold
            recon1_unmatched_as_duplicate = int(np.sum(near_recon2_mask))
        
        # recon1 的点总是保留
        for i, pt_id in enumerate(recon1_unmatched_ids):
            all_points[next_id] = recon1_unmatched_xyz_arr[i]
            all_colors[next_id] = np.array(recon1_unmatched_colors[i])
            point3D_id_map1[pt_id] = next_id
            next_id += 1
            recon1_unmatched_added += 1
    
    # Step 6: 添加 recon2 未匹配点（批量 KD-Tree 查询）
    recon2_unmatched_added = 0
    recon2_unmatched_discarded = 0
    
    if len(recon2_unmatched_ids) > 0:
        # 构建 recon1 所有已添加点的 KD-Tree
        # 收集所有 recon1 点（匹配 + 未匹配）
        all_recon1_pts = np.array([all_points[pid] for pid in point3D_id_map1.values()])
        
        if len(all_recon1_pts) > 0:
            all_recon1_tree = cKDTree(all_recon1_pts)
            # 批量查询：一次查询所有 recon2 未匹配点到 recon1 的最近距离
            distances, _ = all_recon1_tree.query(recon2_unmatched_xyz_arr, k=1)
            is_duplicate_mask = distances < spatial_dedup_threshold
        else:
            is_duplicate_mask = np.zeros(len(recon2_unmatched_ids), dtype=bool)
        
        # 根据结果添加或丢弃
        for i, pt_id in enumerate(recon2_unmatched_ids):
            if is_duplicate_mask[i] and not keep_unmatched_overlap:
                recon2_unmatched_discarded += 1
            else:
                all_points[next_id] = recon2_unmatched_xyz_arr[i]
                all_colors[next_id] = np.array(recon2_unmatched_colors[i])
                point3D_id_map2[pt_id] = next_id
                next_id += 1
                recon2_unmatched_added += 1
    
    if verbose:
        print(f"\nFinal merge statistics:")
        print(f"  Matched points (averaged): {matched_added}")
        print(f"  Recon1 unmatched points added: {recon1_unmatched_added}")
        print(f"    - Near recon2 points (as duplicates, kept): {recon1_unmatched_as_duplicate}")
        print(f"  Recon2 unmatched points added: {recon2_unmatched_added}")
        print(f"  Recon2 unmatched points discarded (duplicates): {recon2_unmatched_discarded}")
        print(f"  Total merged points: {len(all_points)}")
    
    return all_points, all_colors, point3D_id_map1, point3D_id_map2, recon2_refined


def merge_point_clouds_with_fusion(
    recon1: pycolmap.Reconstruction,
    recon2_aligned: pycolmap.Reconstruction,
    overlap_image_ids_recon1: Set[int],
    overlap_image_ids_recon2: Set[int],
    voxel_size: float = 0.05,
    overlap_bbox_min: Optional[np.ndarray] = None,
    overlap_bbox_max: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int], Dict[int, int]]:
    """
    融合两个重建的点云，使用体素下采样处理重叠区域
    
    改进策略：
    1. 识别重叠区域（通过空间边界框或影像观测）
    2. 对重叠区域内的所有点（来自两个重建）进行体素下采样，确保密度一致
    3. 非重叠区域的点保持不变
    
    Args:
        recon1: 基准重建
        recon2_aligned: 已对齐的第二个重建
        overlap_image_ids_recon1: recon1中重叠影像的ID集合
        overlap_image_ids_recon2: recon2中重叠影像的ID集合
        voxel_size: 体素大小（米），用于重叠区域下采样
        overlap_bbox_min: 重叠区域边界框最小点（可选）
        overlap_bbox_max: 重叠区域边界框最大点（可选）
        verbose: 是否打印详细信息
        
    Returns:
        final_points: 最终点云 {new_id: xyz}
        final_colors: 最终颜色 {new_id: color}
        point3D_id_map1: recon1点ID到新ID的映射
        point3D_id_map2: recon2点ID到新ID的映射
    """
    # 1. 收集并分类recon1的点
    points1_overlap_xyz = {}  # 在重叠影像中被观测的点
    points1_overlap_color = {}
    points1_non_overlap_xyz = {}  # 只在非重叠影像中被观测的点
    points1_non_overlap_color = {}
    
    for pt_id, point3D in recon1.points3D.items():
        in_overlap = False
        
        for track_elem in point3D.track.elements:
            if track_elem.image_id in overlap_image_ids_recon1:
                in_overlap = True
                break
        
        # 如果有边界框，也检查空间位置
        if overlap_bbox_min is not None and overlap_bbox_max is not None:
            if is_point_in_bbox(point3D.xyz, overlap_bbox_min, overlap_bbox_max):
                in_overlap = True
        
        if in_overlap:
            points1_overlap_xyz[pt_id] = point3D.xyz
            points1_overlap_color[pt_id] = point3D.color
        else:
            points1_non_overlap_xyz[pt_id] = point3D.xyz
            points1_non_overlap_color[pt_id] = point3D.color
    
    # 2. 收集并分类recon2的点
    points2_overlap_xyz = {}  # 在重叠影像中被观测的点
    points2_overlap_color = {}
    points2_non_overlap_xyz = {}  # 只在非重叠影像中被观测的点
    points2_non_overlap_color = {}
    
    for pt_id, point3D in recon2_aligned.points3D.items():
        in_overlap = False
        
        for track_elem in point3D.track.elements:
            if track_elem.image_id in overlap_image_ids_recon2:
                in_overlap = True
                break
        
        # 如果有边界框，也检查空间位置
        if overlap_bbox_min is not None and overlap_bbox_max is not None:
            if is_point_in_bbox(point3D.xyz, overlap_bbox_min, overlap_bbox_max):
                in_overlap = True
        
        if in_overlap:
            points2_overlap_xyz[pt_id] = point3D.xyz
            points2_overlap_color[pt_id] = point3D.color
        else:
            points2_non_overlap_xyz[pt_id] = point3D.xyz
            points2_non_overlap_color[pt_id] = point3D.color
    
    if verbose:
        print(f"\nPoint cloud fusion analysis:")
        print(f"  Recon1 overlap region points: {len(points1_overlap_xyz)}")
        print(f"  Recon1 non-overlap region points: {len(points1_non_overlap_xyz)}")
        print(f"  Recon2 overlap region points: {len(points2_overlap_xyz)}")
        print(f"  Recon2 non-overlap region points: {len(points2_non_overlap_xyz)}")
    
    # 3. 合并重叠区域的点并进行体素下采样
    # 为了区分来源，给recon2的点ID加一个大偏移
    id_offset = max(recon1.points3D.keys()) + 1 if recon1.points3D else 0
    
    overlap_combined_xyz = {}
    overlap_combined_color = {}
    overlap_source = {}  # 记录每个点来自哪个重建: 1 或 2
    
    for pt_id, xyz in points1_overlap_xyz.items():
        overlap_combined_xyz[pt_id] = xyz
        overlap_combined_color[pt_id] = points1_overlap_color[pt_id]
        overlap_source[pt_id] = 1
    
    for pt_id, xyz in points2_overlap_xyz.items():
        new_id = pt_id + id_offset
        overlap_combined_xyz[new_id] = xyz
        overlap_combined_color[new_id] = points2_overlap_color[pt_id]
        overlap_source[new_id] = 2
    
    if verbose:
        print(f"\n  Combined overlap region points: {len(overlap_combined_xyz)}")
    
    # 对重叠区域进行体素下采样
    if len(overlap_combined_xyz) > 0:
        overlap_downsampled_xyz, overlap_downsampled_color, voxel_to_original = voxel_downsample(
            overlap_combined_xyz,
            overlap_combined_color,
            voxel_size=voxel_size,
            verbose=verbose
        )
    else:
        overlap_downsampled_xyz = {}
        overlap_downsampled_color = {}
        voxel_to_original = {}
    
    # 4. 构建最终点云和映射
    final_points = {}
    final_colors = {}
    point3D_id_map1 = {}  # {old_recon1_id: new_id}
    point3D_id_map2 = {}  # {old_recon2_id: new_id}
    
    next_pt_id = 1
    
    # 4.1 添加recon1非重叠区域的点
    for pt_id in sorted(points1_non_overlap_xyz.keys()):
        final_points[next_pt_id] = points1_non_overlap_xyz[pt_id]
        final_colors[next_pt_id] = points1_non_overlap_color[pt_id]
        point3D_id_map1[pt_id] = next_pt_id
        next_pt_id += 1
    
    # 4.2 添加下采样后的重叠区域点，并建立映射
    for new_voxel_id in sorted(overlap_downsampled_xyz.keys()):
        final_points[next_pt_id] = overlap_downsampled_xyz[new_voxel_id]
        final_colors[next_pt_id] = overlap_downsampled_color[new_voxel_id]
        
        # 将原始点ID映射到这个新点
        original_ids = voxel_to_original[new_voxel_id]
        for orig_id in original_ids:
            if overlap_source.get(orig_id) == 1:
                # 来自recon1
                point3D_id_map1[orig_id] = next_pt_id
            else:
                # 来自recon2（需要减去偏移）
                real_id = orig_id - id_offset
                point3D_id_map2[real_id] = next_pt_id
        
        next_pt_id += 1
    
    # 4.3 添加recon2非重叠区域的点
    for pt_id in sorted(points2_non_overlap_xyz.keys()):
        final_points[next_pt_id] = points2_non_overlap_xyz[pt_id]
        final_colors[next_pt_id] = points2_non_overlap_color[pt_id]
        point3D_id_map2[pt_id] = next_pt_id
        next_pt_id += 1
    
    if verbose:
        print(f"\nFinal merged point cloud:")
        print(f"  Total points: {len(final_points)}")
        print(f"  From Recon1 (non-overlap): {len(points1_non_overlap_xyz)}")
        print(f"  From overlap region (downsampled): {len(overlap_downsampled_xyz)}")
        print(f"  From Recon2 (non-overlap): {len(points2_non_overlap_xyz)}")
        original_total = len(points1_overlap_xyz) + len(points2_overlap_xyz)
        if original_total > 0:
            reduction = (1 - len(overlap_downsampled_xyz) / original_total) * 100
            print(f"  Overlap region reduction: {original_total} -> {len(overlap_downsampled_xyz)} ({reduction:.1f}% reduced)")
    
    return final_points, final_colors, point3D_id_map1, point3D_id_map2


def _add_images_batch(
    source_recon,
    merged_recon,
    merged_points3D: dict,
    camera_id_map: Dict[int, int],
    point3D_id_map: Dict[int, int],
    start_image_id: int,
    filter_edge_margin: int = 0,
    skip_image_ids: Optional[Set[int]] = None
) -> Tuple[Dict[int, int], int, int]:
    """
    批量添加影像到合并后的 reconstruction（内部辅助函数）
    
    Args:
        source_recon: 源 reconstruction
        merged_recon: 目标合并 reconstruction
        merged_points3D: 合并后的 points3D 字典引用
        camera_id_map: 相机 ID 映射 {old_id: new_id}
        point3D_id_map: 3D点 ID 映射 {old_id: new_id}
        start_image_id: 起始影像 ID
        filter_edge_margin: 边缘过滤边距（像素）
        skip_image_ids: 需要跳过的影像 ID 集合
        
    Returns:
        image_id_map: 影像 ID 映射 {old_id: new_id}
        next_image_id: 下一个可用的影像 ID
        edge_filtered_count: 被边缘过滤的 2D 点数量
    """
    image_id_map = {}
    next_image_id = start_image_id
    edge_filtered_count = 0
    
    # 预转换 point3D_id_map 的键为集合，加速 in 查询
    valid_point3d_ids = set(point3D_id_map.keys())
    skip_set = skip_image_ids if skip_image_ids else set()
    
    sorted_img_ids = sorted(source_recon.images.keys())
    for img_id in sorted_img_ids:
        # 跳过指定的影像
        if img_id in skip_set:
            continue
        
        image = source_recon.images[img_id]
        points2D = image.points2D
        n_points = len(points2D)
        
        # 获取影像尺寸用于边缘过滤（缓存 camera_id 映射结果）
        new_camera_id = camera_id_map[image.camera_id]
        camera = merged_recon.cameras[new_camera_id]
        img_width = camera.width
        img_height = camera.height
        
        # 使用 NumPy 批量处理边缘检测
        if filter_edge_margin > 0 and n_points > 0:
            coords = np.array([pt.xy for pt in points2D], dtype=np.float32)
            edge_mask = (
                (coords[:, 0] < filter_edge_margin) | 
                (coords[:, 0] >= img_width - filter_edge_margin) |
                (coords[:, 1] < filter_edge_margin) | 
                (coords[:, 1] >= img_height - filter_edge_margin)
            )
        else:
            edge_mask = None
        
        # 重建 points2D 列表（预分配列表）
        new_points2D = [None] * n_points
        for pt2d_idx in range(n_points):
            point2D = points2D[pt2d_idx]
            pt3d_id = point2D.point3D_id
            
            if pt3d_id != -1 and pt3d_id in valid_point3d_ids:
                is_edge_point = edge_mask[pt2d_idx] if edge_mask is not None else False
                
                if is_edge_point:
                    new_points2D[pt2d_idx] = pycolmap.Point2D(point2D.xy)
                    edge_filtered_count += 1
                else:
                    new_pt3d_id = point3D_id_map[pt3d_id]
                    new_points2D[pt2d_idx] = pycolmap.Point2D(point2D.xy, new_pt3d_id)
                    # 更新 3D 点的 track
                    merged_points3D[new_pt3d_id].track.add_element(next_image_id, pt2d_idx)
            else:
                new_points2D[pt2d_idx] = pycolmap.Point2D(point2D.xy)
        
        # 创建新影像
        new_image = pycolmap.Image(
            image_id=next_image_id,
            name=image.name,
            camera_id=new_camera_id,
            cam_from_world=image.cam_from_world,
            points2D=new_points2D
        )
        merged_recon.add_image(new_image)
        image_id_map[img_id] = next_image_id
        next_image_id += 1
    
    return image_id_map, next_image_id, edge_filtered_count


def merge_reconstructions(
    model_dir1: str,
    model_dir2: str,
    output_dir: str,
    overlap_count: int = 2,
    translation_only: bool = False,
    use_ransac: bool = True,
    ransac_threshold: float = 0.5,
    # 点云融合参数
    point_fusion: bool = True,
    fusion_method: str = "2d_matching",
    duplicate_threshold: float = 0.3,
    cell_sizes: List[int] = None,
    keep_unmatched_overlap: bool = False,
    spatial_dedup_threshold: float = 0.1,
    # 精化对齐参数
    refine_alignment: bool = False,
    refine_transform_mode: str = "scale_translation",
    refine_cell_range: Tuple[int, int] = (1, 3),
    refine_stages: List = None,
    voxel_size: float = 0.0,
    statistical_filter: bool = True,
    stat_nb_neighbors: int = 20,
    stat_std_ratio: float = 2.0,
    # 边缘过滤参数
    min_track_length: int = 0,
    boundary_filter: bool = False,
    boundary_density_ratio: float = 0.3,
    # 影像边缘2D点过滤参数
    filter_edge_margin: int = 0,
    verbose: bool = True
) -> Optional[pycolmap.Reconstruction]:
    """
    合并两个 pycolmap Reconstruction
    
    Args:
        model_dir1: 第一个模型目录路径（作为基准）
        model_dir2: 第二个模型目录路径（将被变换并合并）
        output_dir: 输出目录路径
        overlap_count: 重叠影像数量
        translation_only: 是否只做平移对齐（不做旋转和缩放）
        use_ransac: 是否使用 RANSAC 估计变换（仅当 translation_only=False 时有效）
        ransac_threshold: RANSAC 内点阈值（米）
        point_fusion: 是否启用点云融合
        fusion_method: 融合方法 - "2d_matching"（基于2D像素匹配，推荐）或 "spatial_dedup"（基于3D距离去重）
        duplicate_threshold: 3D距离去重阈值（米），仅fusion_method="spatial_dedup"时有效
        cell_sizes: 2D匹配的多尺度cell大小列表，仅fusion_method="2d_matching"时有效
        keep_unmatched_overlap: 是否保留重叠区域未匹配的点（False=丢弃，消除双层）
        spatial_dedup_threshold: 空间去重距离阈值（米），小于此距离认为是重复点（默认0.1，太大会丢失太多点）
        refine_alignment: 是否利用2D匹配的3D点对精化对齐（推荐开启）
        refine_transform_mode: 精化时的全局变换模式（可被阶段配置覆盖）
            - "translation": 只平移
            - "scale_translation": 缩放+平移，不旋转（推荐）
            - "sim3": 完整Sim3
        refine_cell_range: 精化时使用的cell_size范围 (min_cell, max_cell)
            只有cell_size在此范围内的高质量匹配点才会用于精化
            默认 (1, 3) 表示使用最高质量的匹配点
            这个参数是全局的，只需指定一次
        refine_stages: 多阶段精化的配置列表，支持灵活格式：
            - 数字 (如 10.0): 只指定max_dist_3d，使用全局transform_mode
            - 字符串 (如 "translation"): 只指定变换模式，不筛选距离
            - 元组 (max_dist_3d, transform_mode): 同时指定距离和变换模式
            - None: 不筛选距离，使用全局transform_mode
            
            工作流程：
            1. 第一阶段：用 refine_cell_range 范围内的所有点进行粗对齐
            2. 后续阶段：在满足 cell_range 的点中，根据3D距离进一步筛选
            
            例如 refine_cell_range=(1,3), refine_stages=["translation", 10.0, 5.0, 2.0]：
            - 第1阶段：用cell_size=[1,3]的所有点，只平移粗对齐
            - 第2阶段：用dist_3d<=10m的点继续精化
            - 第3阶段：用dist_3d<=5m的点继续精化
            - 第4阶段：用dist_3d<=2m的点最终精化
        voxel_size: 体素大小（米），用于全局下采样，0表示不下采样
        statistical_filter: 是否进行统计滤波去除离群点
        stat_nb_neighbors: 统计滤波的邻居数量
        stat_std_ratio: 统计滤波的标准差倍数阈值
        min_track_length: 最小track长度，小于此值的点将被过滤（0=不过滤，推荐2-3）
        boundary_filter: 是否启用边界密度过滤（移除边缘稀疏点）
        boundary_density_ratio: 边界过滤的密度阈值比例（越小过滤越激进）
        filter_edge_margin: 影像边缘过滤范围（像素），2D点在此范围内的不关联3D点（0=不过滤，推荐50-100）
        verbose: 是否打印详细信息
        
    Returns:
        合并后的 Reconstruction 对象，失败返回 None
    """
    # 1. 读取两个 reconstruction
    if verbose:
        print(f"Loading reconstructions...")
        print(f"  Model 1: {model_dir1}")
        print(f"  Model 2: {model_dir2}")
    
    recon1 = pycolmap.Reconstruction(model_dir1)
    recon2 = pycolmap.Reconstruction(model_dir2)
    
    if verbose:
        print(f"\nReconstruction 1:")
        print(f"  Images: {len(recon1.images)}")
        print(f"  Points3D: {len(recon1.points3D)}")
        print(f"  Cameras: {len(recon1.cameras)}")
        print(f"\nReconstruction 2:")
        print(f"  Images: {len(recon2.images)}")
        print(f"  Points3D: {len(recon2.points3D)}")
        print(f"  Cameras: {len(recon2.cameras)}")
    
    # 2. 找到共同影像
    common_images = find_common_images(recon1, recon2)
    
    if verbose:
        print(f"\nCommon images found: {len(common_images)}")
        for id1, id2 in common_images.items():
            print(f"  Recon1 Image {id1} ({recon1.images[id1].name}) <-> Recon2 Image {id2}")
    
    if len(common_images) < overlap_count:
        print(f"Error: Need at least {overlap_count} common images for alignment, but only found {len(common_images)}")
        return None
    
    # 如果共同影像数量超过overlap_count，只使用前overlap_count个（保证一致性）
    if len(common_images) > overlap_count:
        if verbose:
            print(f"\nUsing first {overlap_count} common images for alignment (out of {len(common_images)})")
        common_images_keys = list(common_images.keys())[:overlap_count]
        common_images = {k: common_images[k] for k in common_images_keys}
    
    # 3. 使用共同影像的相机位置估计变换
    common_ids1 = list(common_images.keys())
    common_ids2 = [common_images[id1] for id1 in common_ids1]
    
    # 获取相机中心
    centers1 = get_camera_centers(recon1, common_ids1)
    centers2 = get_camera_centers(recon2, common_ids2)
    
    if verbose:
        print(f"\nCamera centers for alignment:")
        for i, (id1, id2) in enumerate(zip(common_ids1, common_ids2)):
            print(f"  Image {id1}/{id2}: Recon1={centers1[i]}, Recon2={centers2[i]}")
    
    # 4. 估计变换 (从 recon2 坐标系到 recon1 坐标系)
    if translation_only:
        # 只计算平移量，不做旋转和缩放
        translation = estimate_translation_only(centers2, centers1)
        # 创建只有平移的 Sim3 变换（scale=1, rotation=identity）
        identity_rotation = pycolmap.Rotation3d(np.eye(3))
        sim3_transform = pycolmap.Sim3d(1.0, identity_rotation, translation)
        
        if verbose:
            print(f"\nTranslation-only alignment:")
            print(f"  Translation: {translation}")
    else:
        # 使用完整的 Sim3 变换
        if use_ransac:
            sim3_transform, inlier_mask = estimate_sim3_with_ransac(
                centers2, centers1, 
                inlier_threshold=ransac_threshold
            )
            if sim3_transform is None:
                print("Error: Failed to estimate Sim3 transform with RANSAC")
                return None
            if verbose:
                print(f"\nSim3 transform estimated with RANSAC:")
                print(f"  Inliers: {np.sum(inlier_mask)}/{len(inlier_mask)}")
        else:
            scale, R, t = estimate_sim3_umeyama(centers2, centers1)
            rotation = pycolmap.Rotation3d(R)
            sim3_transform = pycolmap.Sim3d(scale, rotation, t)
        
        if verbose:
            print(f"  Scale: {sim3_transform.scale:.6f}")
            print(f"  Translation: {sim3_transform.translation}")
    
    # 5. 应用变换到 recon2
    # recon2_aligned = pycolmap.Reconstruction(recon2)
    recon2_aligned = copy.deepcopy(recon2)
    recon2_aligned.transform(sim3_transform)
    
    if verbose:
        print(f"\nRecon2 transformed to Recon1 coordinate system")
        # 验证变换后的相机中心
        centers2_transformed = get_camera_centers(recon2_aligned, common_ids2)
        for i, (id1, id2) in enumerate(zip(common_ids1, common_ids2)):
            error = np.linalg.norm(centers1[i] - centers2_transformed[i])
            print(f"  Image {id1}/{id2}: Error = {error:.6f} m")
    
    # 6. 创建合并后的 reconstruction
    merged_recon = pycolmap.Reconstruction()
    
    # 6.1 添加 recon1 的所有相机（保持原始 ID）
    camera_id_map1 = {}  # {old_id: new_id}
    next_cam_id = 1
    for cam_id in sorted(recon1.cameras.keys()):
        camera = recon1.cameras[cam_id]
        new_camera = pycolmap.Camera(
            camera_id=next_cam_id,
            model=camera.model,
            width=camera.width,
            height=camera.height,
            params=camera.params
        )
        merged_recon.add_camera(new_camera)
        camera_id_map1[cam_id] = next_cam_id
        next_cam_id += 1
    
    # 6.2 添加 recon2 的非重复相机
    camera_id_map2 = {}  # {old_id: new_id}
    
    for cam_id in sorted(recon2_aligned.cameras.keys()):
        camera = recon2_aligned.cameras[cam_id]
        # 为 recon2 的每个相机创建新 ID（即使参数相似也分开处理）
        new_camera = pycolmap.Camera(
            camera_id=next_cam_id,
            model=camera.model,
            width=camera.width,
            height=camera.height,
            params=camera.params
        )
        merged_recon.add_camera(new_camera)
        camera_id_map2[cam_id] = next_cam_id
        next_cam_id += 1
    
    if verbose:
        print(f"\nCameras added: {len(merged_recon.cameras)}")
    
    # 6.3 处理 3D 点 - 使用点云融合或简单合并
    recon1_overlap_image_ids = set(common_images.keys())
    recon2_overlap_image_ids = set(common_images.values())
    
    if point_fusion:
        if verbose:
            print(f"\n=== Point Cloud Fusion (method: {fusion_method}) ===")
        
        # Step 1: 根据选择的方法进行融合
        if fusion_method == "2d_matching":
            # 基于2D像素坐标匹配的融合（推荐）
            all_points, all_colors, point3D_id_map1, point3D_id_map2, recon2_refined = merge_points_with_2d_matching(
                recon1=recon1,
                recon2_aligned=recon2_aligned,
                common_images=common_images,
                cell_sizes=cell_sizes,
                keep_unmatched_overlap=keep_unmatched_overlap,
                spatial_dedup_threshold=spatial_dedup_threshold,
                refine_alignment=refine_alignment,
                recon2_original=recon2,  # 传入原始未变换的recon2
                refine_transform_mode=refine_transform_mode,
                refine_cell_range=refine_cell_range,
                refine_stages=refine_stages,
                verbose=verbose
            )
            
            # 如果进行了精化对齐，更新recon2_aligned
            if recon2_refined is not None:
                recon2_aligned = recon2_refined
                if verbose:
                    print(f"\n  Using refined recon2 alignment for image poses")
        else:
            # 基于3D空间距离去重
            all_points, all_colors, point3D_id_map1, point3D_id_map2 = merge_points_with_spatial_dedup(
                recon1=recon1,
                recon2_aligned=recon2_aligned,
                duplicate_threshold=duplicate_threshold,
                verbose=verbose
            )
        
        # Step 2: 可选的全局体素下采样
        if voxel_size > 0:
            if verbose:
                print(f"\nGlobal voxel downsampling (voxel_size={voxel_size}m):")
            
            downsampled_xyz, downsampled_color, voxel_to_original = voxel_downsample(
                all_points, all_colors,
                voxel_size=voxel_size,
                verbose=verbose
            )
            
            # 重建ID映射（多对一映射，取第一个原始ID）
            new_point3D_id_map1 = {}
            new_point3D_id_map2 = {}
            
            # 反向映射：原始ID -> 新ID
            original_to_new = {}
            for new_id, orig_ids in voxel_to_original.items():
                for orig_id in orig_ids:
                    original_to_new[orig_id] = new_id
            
            # 更新recon1的映射
            for old_r1_id, merged_id in point3D_id_map1.items():
                if merged_id in original_to_new:
                    new_point3D_id_map1[old_r1_id] = original_to_new[merged_id]
            
            # 更新recon2的映射
            for old_r2_id, merged_id in point3D_id_map2.items():
                if merged_id in original_to_new:
                    new_point3D_id_map2[old_r2_id] = original_to_new[merged_id]
            
            point3D_id_map1 = new_point3D_id_map1
            point3D_id_map2 = new_point3D_id_map2
            final_points = downsampled_xyz
            final_colors = downsampled_color
        else:
            final_points = all_points
            final_colors = all_colors
        
        # Step 3: 统计滤波去除离群点（可选）
        if statistical_filter:
            if verbose:
                print(f"\nStatistical outlier removal (nb_neighbors={stat_nb_neighbors}, std_ratio={stat_std_ratio}):")
            
            filtered_xyz, filtered_color = statistical_outlier_removal(
                final_points, final_colors,
                nb_neighbors=stat_nb_neighbors,
                std_ratio=stat_std_ratio,
                verbose=verbose
            )
            
            # 更新映射（移除被滤除点的映射）
            valid_new_ids = set(filtered_xyz.keys())
            point3D_id_map1 = {k: v for k, v in point3D_id_map1.items() if v in valid_new_ids}
            point3D_id_map2 = {k: v for k, v in point3D_id_map2.items() if v in valid_new_ids}
            
            final_points = filtered_xyz
            final_colors = filtered_color
        
        # Step 4: 边界密度过滤（移除边缘稀疏点）
        if boundary_filter:
            if verbose:
                print(f"\nBoundary-aware filtering (density_ratio={boundary_density_ratio}):")
            
            filtered_xyz, filtered_color = boundary_aware_filter(
                final_points, final_colors,
                nb_neighbors=10,
                density_threshold_ratio=boundary_density_ratio,
                verbose=verbose
            )
            
            # 更新映射
            valid_new_ids = set(filtered_xyz.keys())
            point3D_id_map1 = {k: v for k, v in point3D_id_map1.items() if v in valid_new_ids}
            point3D_id_map2 = {k: v for k, v in point3D_id_map2.items() if v in valid_new_ids}
            
            final_points = filtered_xyz
            final_colors = filtered_color
        
        if verbose:
            print(f"\nFinal point cloud: {len(final_points)} points")
        
        # 将融合后的点添加到merged_recon，并重新编号
        old_to_final_id = {}
        for new_pt_id in sorted(final_points.keys()):
            final_id = merged_recon.add_point3D(
                xyz=final_points[new_pt_id],
                track=pycolmap.Track(),
                color=final_colors[new_pt_id]
            )
            old_to_final_id[new_pt_id] = final_id
        
        # 更新映射到最终ID
        point3D_id_map1 = {k: old_to_final_id[v] for k, v in point3D_id_map1.items() if v in old_to_final_id}
        point3D_id_map2 = {k: old_to_final_id[v] for k, v in point3D_id_map2.items() if v in old_to_final_id}
        
    else:
        # 原始的简单合并方式
        # 6.3.1 添加 recon1 的所有 3D 点
        point3D_id_map1 = {}  # {old_id: new_id}
        for pt_id, point3D in recon1.points3D.items():
            new_pt_id = merged_recon.add_point3D(
                xyz=point3D.xyz,
                track=pycolmap.Track(),
                color=point3D.color
            )
            point3D_id_map1[pt_id] = new_pt_id
        
        if verbose:
            print(f"Points3D from Recon1: {len(point3D_id_map1)}")
        
        # 6.3.2 添加 recon2 的非重叠区域 3D 点
        point3D_id_map2 = {}  # {old_id: new_id}
        for pt_id, point3D in recon2_aligned.points3D.items():
            # 检查这个点是否只被非重叠影像观测
            has_non_overlap_obs = False
            for track_elem in point3D.track.elements:
                if track_elem.image_id not in recon2_overlap_image_ids:
                    has_non_overlap_obs = True
                    break
            
            if has_non_overlap_obs:
                new_pt_id = merged_recon.add_point3D(
                    xyz=point3D.xyz,
                    track=pycolmap.Track(),
                    color=point3D.color
                )
                point3D_id_map2[pt_id] = new_pt_id
        
        if verbose:
            print(f"Points3D from Recon2 (non-overlap): {len(point3D_id_map2)}")
    
    # 6.4 添加 recon1 的所有影像
    # 预取 merged_recon.points3D 引用，减少后续重复属性访问
    merged_points3D = merged_recon.points3D
    
    image_id_map1, next_image_id, edge_filtered_count_r1 = _add_images_batch(
        source_recon=recon1,
        merged_recon=merged_recon,
        merged_points3D=merged_points3D,
        camera_id_map=camera_id_map1,
        point3D_id_map=point3D_id_map1,
        start_image_id=1,
        filter_edge_margin=filter_edge_margin,
        skip_image_ids=None
    )
    
    if verbose:
        print(f"\nImages from Recon1: {len(image_id_map1)}")
        if filter_edge_margin > 0:
            print(f"  Edge-filtered 2D points (margin={filter_edge_margin}px): {edge_filtered_count_r1}")
    
    # 6.5 添加 recon2 的非重叠影像
    image_id_map2, next_image_id, edge_filtered_count_r2 = _add_images_batch(
        source_recon=recon2_aligned,
        merged_recon=merged_recon,
        merged_points3D=merged_points3D,
        camera_id_map=camera_id_map2,
        point3D_id_map=point3D_id_map2,
        start_image_id=next_image_id,
        filter_edge_margin=filter_edge_margin,
        skip_image_ids=recon2_overlap_image_ids  # 跳过重叠影像
    )
    
    if verbose:
        print(f"Images from Recon2 (non-overlap): {len(image_id_map2)}")
        if filter_edge_margin > 0:
            print(f"  Edge-filtered 2D points (margin={filter_edge_margin}px): {edge_filtered_count_r2}")
    
    # 6.6 清理观测不足的3D点（高效版：利用 track 直接定位受影响的影像）
    # min_track_length: 用户指定的最小track长度
    #   0 = 只删除track为空的点（最宽松，保留最多点）
    #   1 = 至少1个观测（与0效果相同）
    #   2 = 至少2个观测（COLMAP默认要求）
    #   3+ = 更严格的过滤
    effective_min_track = max(1, min_track_length)
    
    points3d_to_remove = []
    empty_track_count = 0
    insufficient_track_count = 0
    
    # 同时收集需删除的点和受影响的 (image_id, point2D_idx) 对
    # 使用 defaultdict 按 image_id 分组，避免后续遍历所有影像
    from collections import defaultdict
    affected_points_by_image = defaultdict(list)  # {image_id: [point2D_idx, ...]}
    
    for point3d_id, point3d in merged_points3D.items():
        track = point3d.track
        track_elements = track.elements
        track_len = len(track_elements)
        
        if track_len == 0:
            empty_track_count += 1
            points3d_to_remove.append(point3d_id)
        elif track_len < effective_min_track:
            insufficient_track_count += 1
            points3d_to_remove.append(point3d_id)
            # 从 track 中收集受影响的 (image_id, point2D_idx)
            for elem in track_elements:
                affected_points_by_image[elem.image_id].append(elem.point2D_idx)
    
    # 如果有需要删除的3D点，更新受影响的影像
    if points3d_to_remove:
        merged_images = merged_recon.images
        Point2D = pycolmap.Point2D
        Image = pycolmap.Image
        add_image = merged_recon.add_image
        
        # 只处理受影响的影像（而不是遍历所有影像）
        for img_id, affected_indices in affected_points_by_image.items():
            image = merged_images[img_id]
            points2D = image.points2D
            
            # 使用浅拷贝，只替换受影响的点
            new_points2D = list(points2D)
            for idx in affected_indices:
                new_points2D[idx] = Point2D(points2D[idx].xy)
            
            # 更新影像
            new_image = Image(
                image_id=image.image_id,
                name=image.name,
                camera_id=image.camera_id,
                cam_from_world=image.cam_from_world,
                points2D=new_points2D
            )
            del merged_images[img_id]
            add_image(new_image)
        
        # 批量删除3D点
        for point3d_id in points3d_to_remove:
            del merged_points3D[point3d_id]
        
        if verbose and affected_points_by_image:
            print(f"\nUpdated {len(affected_points_by_image)} images to fix 2D-3D references")
    
    if verbose:
        if empty_track_count > 0:
            print(f"Removed {empty_track_count} 3D points with empty track (no 2D observations)")
        if insufficient_track_count > 0:
            print(f"Removed {insufficient_track_count} 3D points with insufficient observations (< {effective_min_track})")
        if points3d_to_remove:
            print(f"Total removed: {len(points3d_to_remove)} points")
    
    # 7. 保存合并结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    merged_recon.write_text(str(output_path))
    merged_recon.export_PLY(str(output_path / "points3D.ply"))
    
    if verbose:
        print(f"\n=== Merged Reconstruction ===")
        print(f"  Images: {len(merged_recon.images)}")
        print(f"  Points3D: {len(merged_recon.points3D)}")
        print(f"  Cameras: {len(merged_recon.cameras)}")
        print(f"\nSaved to: {output_path}")
    
    return merged_recon


def main():
    """
    示例：合并 0_6 和 4_10 两个 reconstruction
    """
    # 定义路径
    base_dir = Path(__file__).parent.parent / "output" / "Ganluo_images" / "sparse_incremental_reconstruction" / "temp_aligned_to_prev_recon_overlay_image"
    
    model_dir1 = str(base_dir / "0_6")
    model_dir2 = str(base_dir / "4_10")
    output_dir = str(base_dir / "merged_0_6_4_10")

    # model_dir1 = str(base_dir / "merged_0_6_4_10")
    # model_dir2 = str(base_dir / "5_11")
    # output_dir = str(base_dir / "merged_0_6_4_10_5_11")

    # model_dir1 = str(base_dir / "merged_0_6_4_10_5_11")
    # model_dir2 = str(base_dir / "8_14")
    # output_dir = str(base_dir / "merged_0_6_4_10_5_11_8_14")

    # model_dir1 = str(base_dir / "merged_0_6_4_10_5_11_8_14")
    # model_dir2 = str(base_dir / "10_16")
    # output_dir = str(base_dir / "merged_0_6_4_10_5_11_8_14_10_16")

    # model_dir1 = str(base_dir / "merged_0_6_4_10_5_11_8_14_10_16")
    # model_dir2 = str(base_dir / "12_18")
    # output_dir = str(base_dir / "merged_0_6_4_10_5_11_8_14_10_16_12_18")

    # model_dir1 = str(base_dir / "merged_0_6_4_10_5_11_8_14_10_16_12_18")
    # model_dir2 = str(base_dir / "15_21")
    # output_dir = str(base_dir / "merged_0_6_4_10_5_11_8_14_10_16_12_18_15_21")

    # model_dir1 = str(base_dir / "merged_0_6_4_10_5_11_8_14_10_16_12_18_15_21")
    # model_dir2 = str(base_dir / "16_22")
    # output_dir = str(base_dir / "merged_0_6_4_10_5_11_8_14_10_16_12_18_15_21_16_22")

    print("=" * 60)
    print("Merging two COLMAP reconstructions")
    print("=" * 60)
    
    # 执行合并
    # translation_only=True: 初始对齐只做平移，适用于已预处理过的数据（尺度和旋转已一致）
    # fusion_method="2d_matching": 基于2D像素坐标匹配的融合（推荐）
    # refine_alignment=True: 利用2D匹配的3D点对精化对齐（推荐开启）
    # refine_transform_mode="scale_translation": 精化时做缩放+平移，不旋转
    # refine_max_cell_size=5: 精化时只用cell_size<=5的高质量匹配点
    # keep_unmatched_overlap=False: 丢弃重叠区未匹配的点，消除双层
    merged_recon = merge_reconstructions(
        model_dir1=model_dir1,
        model_dir2=model_dir2,
        output_dir=output_dir,
        overlap_count=2,
        translation_only=True,  # 初始对齐只做平移
        use_ransac=False,
        # 点云融合参数
        point_fusion=True,  # 启用点云融合
        fusion_method="2d_matching",  # 融合方法: "2d_matching"（推荐）或 "spatial_dedup"
        cell_sizes=[1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 500, 1000],  # 2D匹配的多尺度cell大小
        keep_unmatched_overlap=True,  # True=保留重叠区未匹配点（避免丢失太多点）
        spatial_dedup_threshold=0.1,  # 空间去重阈值（米），太大会丢失太多点
        # 精化对齐参数（利用2D匹配的高质量3D点对重新估计变换）
        refine_alignment=True,  # 开启精化对齐
        refine_transform_mode="scale_translation",  # 全局默认变换模式（可被阶段配置覆盖）
        refine_cell_range=(1, 3),  # 全局cell_size范围，只用[1,3]范围的高质量匹配点
        # 迭代优化：简化配置格式
        # - 数字: 只指定max_dist_3d，使用全局transform_mode
        # - 字符串: 只指定变换模式，不筛选距离
        # - 元组 (dist, mode): 同时指定距离和变换模式
        # - None: 不筛选距离，使用全局transform_mode
        refine_stages=[
            (None, "translation"),       # 第1阶段：不筛选距离，只平移
            (10.0, "scale_translation"), # 第2阶段：dist<=10m
            (5.0, "scale_translation"),  # 第3阶段：dist<=5m
            (2.0, "scale_translation"),  # 第4阶段：dist<=2m
            (1.0, "scale_translation"),  # 第5阶段：dist<=1m
            (0.5, "scale_translation"),  # 第6阶段：dist<=0.5m
        ],
        voxel_size=0,  # 体素下采样大小，0表示不下采样
        statistical_filter=False,  # 统计滤波去除离群点
        stat_nb_neighbors=10,  # 统计滤波邻居数
        stat_std_ratio=3.0,  # 统计滤波标准差倍数（越大越宽松）
        # 边缘过滤参数
        min_track_length=0,  # 最小track长度（0=不过滤，推荐2-3可过滤边缘飘点）
        boundary_filter=False,  # 启用边界密度过滤（移除边缘稀疏点）
        boundary_density_ratio=0.5,  # 密度阈值比例（越小过滤越激进，推荐0.2-0.5）
        # 影像边缘2D点过滤参数
        filter_edge_margin=0,  # 影像边缘过滤范围（像素），0=不过滤，推荐50-100
        verbose=True
    )
    
    if merged_recon is not None:
        print("\n" + "=" * 60)
        print("Merge completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Merge failed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
