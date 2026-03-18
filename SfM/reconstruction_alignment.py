#!/usr/bin/env python3
"""
Reconstruction alignment utilities for SfM.

This module provides functions for aligning pycolmap reconstructions
using various methods including point cloud matching and GPS-based alignment.
"""

from typing import Optional, List, Dict, Tuple

import numpy as np
import pycolmap

# 尝试导入 scipy 的 KDTree（更快的最近邻搜索）
try:
    from scipy.spatial import cKDTree
    HAS_KDTREE = True
except ImportError:
    HAS_KDTREE = False

# 模块级常量
MAX_VALID_POINT3D_ID = 2**31  # pycolmap 无效 ID 是大整数，合理 ID 应小于此值


def _extract_valid_points2d(image, max_valid_id: int = MAX_VALID_POINT3D_ID):
    """
    快速提取图像中有效的 2D 点及其 3D 点 ID。
    
    Returns:
        (coords_list, pt3d_ids_list): 坐标列表和 ID 列表
    """
    coords = []
    pt3d_ids = []
    for point2D in image.points2D:
        pt3d_id = point2D.point3D_id
        if 0 <= pt3d_id < max_valid_id:
            coords.append((point2D.xy[0], point2D.xy[1]))
            pt3d_ids.append(int(pt3d_id))
    return coords, pt3d_ids


def find_single_images_pair_matches_kdtree(
    prev_image,
    curr_image,
    curr_coords,
    curr_pt3d_ids,
    pixel_threshold: float,
) -> list:
    """
    使用 KD-Tree 进行快速最近邻匹配（优化版本）。
    
    Args:
        prev_image: 前一个重建的影像对象
        curr_coords: curr_image 的 2D 坐标数组 (M, 2) 或列表
        curr_pt3d_ids: curr_image 的 3D 点 ID 列表
        pixel_threshold: 像素距离阈值
        
    Returns:
        对应关系列表 [(prev_point3D_id, curr_point3D_id, dist)]
    """
    # 确保 curr_coords 是 numpy 数组
    if not isinstance(curr_coords, np.ndarray):
        curr_coords = np.asarray(curr_coords, dtype=np.float64)
    
    if len(curr_coords) == 0:
        return []
    
    # 构建 KD-Tree
    tree = cKDTree(curr_coords)
    
    # 批量提取 prev_image 的有效 2D 点
    prev_coords_list, prev_pt3d_ids = _extract_valid_points2d(prev_image)
    
    if not prev_coords_list:
        return []
    
    prev_coords = np.asarray(prev_coords_list, dtype=np.float64)
    
    # 批量查询最近邻
    distances, indices = tree.query(prev_coords, k=1, distance_upper_bound=pixel_threshold)
    
    # 过滤有效匹配：距离有限且未超出边界
    valid_mask = distances < pixel_threshold  # 比 isfinite 更快
    if not np.any(valid_mask):
        return []
    
    valid_indices = np.flatnonzero(valid_mask)
    valid_distances = distances[valid_indices]
    valid_curr_indices = indices[valid_indices]
    
    # 按距离排序索引
    sorted_order = np.argsort(valid_distances)
    
    # 去重并构建结果
    correspondences = []
    used_curr_ids = set()
    
    for order_idx in sorted_order:
        prev_idx = valid_indices[order_idx]
        curr_idx = valid_curr_indices[order_idx]
        
        curr_pt3d_id = curr_pt3d_ids[curr_idx]
        if curr_pt3d_id in used_curr_ids:
            continue
        
        correspondences.append((prev_pt3d_ids[prev_idx], curr_pt3d_id, valid_distances[order_idx]))
        used_curr_ids.add(curr_pt3d_id)
    
    return correspondences


def find_single_images_pair_matches(
    prev_image,
    curr_image,
    curr_spatial_index: dict,
    pixel_threshold: float,
) -> list:
    """
    在给定的 prev_image 与 curr_image 之间建立 3D 点对应关系（单对影像）。
    
    优化版本：使用平方距离和预生成邻域偏移。
    
    Args:
        prev_image: 前一个重建的影像对象
        curr_image: 当前重建的影像对象  
        curr_spatial_index: 以整数像素为键，值为 [(point3D_id, xy_float), ...] 的字典
        pixel_threshold: 像素距离阈值
        
    Returns:
        对应关系列表 [(prev_point3D_id, curr_point3D_id, dist)]
    """
    search_radius_sq = pixel_threshold * pixel_threshold
    window = int(pixel_threshold) + 1
    used_curr_ids = set()

    # 预生成邻域偏移列表
    neighbor_offsets = [(dx, dy) for dx in range(-window, window + 1) 
                                 for dy in range(-window, window + 1)]

    # 批量提取 prev_image 的有效 2D 点
    prev_coords_list, prev_pt3d_ids = _extract_valid_points2d(prev_image)
    
    if not prev_coords_list:
        return []

    # 收集所有匹配候选，然后排序去重
    all_matches = []
    
    for i, (px, py) in enumerate(prev_coords_list):
        prev_pt3d_id = prev_pt3d_ids[i]
        cx, cy = int(round(px)), int(round(py))
        
        best_curr_id = None
        best_dist_sq = search_radius_sq

        # 搜索邻域
        for dx, dy in neighbor_offsets:
            candidates = curr_spatial_index.get((cx + dx, cy + dy))
            if candidates is None:
                continue

            for curr_pt3d_id, curr_xy in candidates:
                if curr_pt3d_id in used_curr_ids:
                    continue

                # 使用平方距离避免 sqrt
                diff_x = px - curr_xy[0]
                diff_y = py - curr_xy[1]
                dist_sq = diff_x * diff_x + diff_y * diff_y
                
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_curr_id = curr_pt3d_id

        if best_curr_id is not None:
            all_matches.append((best_dist_sq, prev_pt3d_id, best_curr_id))
            used_curr_ids.add(best_curr_id)

    # 转换为最终格式（只在最后计算 sqrt）
    return [(prev_id, curr_id, np.sqrt(dist_sq)) for dist_sq, prev_id, curr_id in all_matches]


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    使用 Umeyama 算法计算从 src 到 dst 的相似变换。
    
    Args:
        src: 源点云 (N, 3)
        dst: 目标点云 (N, 3)
        with_scale: 是否估计尺度
        
    Returns:
        (scale, rotation_matrix, translation_vector)
    """
    assert src.shape == dst.shape
    N, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    Sigma = dst_c.T @ src_c / N  # (3,3)

    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c**2).sum() / N
        s = (D * S.diagonal()).sum() / var_src
    else:
        s = 1.0

    t = mu_dst - s * R @ mu_src

    return s, R, t


def estimate_sim3_transform(src_points: np.ndarray, tgt_points: np.ndarray) -> Optional[pycolmap.Sim3d]:
    """
    使用 Umeyama 算法估计 Sim3 变换。
    
    Args:
        src_points: 源点云 (N, 3)
        tgt_points: 目标点云 (N, 3)
    
    Returns:
        pycolmap.Sim3d 对象，如果失败则返回 None
    """
    try:
        # 使用 Umeyama 算法计算相似变换
        scale, R, t = umeyama_alignment(src_points, tgt_points, with_scale=True)
        
        # 创建 pycolmap.Sim3d 对象
        rotation = pycolmap.Rotation3d(R)
        sim3d = pycolmap.Sim3d(scale, rotation, t)
        
        return sim3d
        
    except Exception as e:
        print(f"  Error estimating Sim3 transform: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_sim3_with_ransac(
    src_points: np.ndarray, 
    tgt_points: np.ndarray,
    max_iterations: int = 1000,
    inlier_threshold: float = 0.5,  # 米
    min_inliers: int = 10,
    early_stop_ratio: float = 0.8,  # 早停阈值：内点比例达到此值时停止
) -> Tuple[Optional[pycolmap.Sim3d], np.ndarray]:
    """
    使用 RANSAC 鲁棒估计 Sim3 变换（优化版本）。
    
    优化：
    - 向量化残差计算
    - 早停策略
    - 一次性预生成所有随机采样索引
    
    Args:
        src_points: 源点云 (N, 3)
        tgt_points: 目标点云 (N, 3)
        max_iterations: RANSAC 最大迭代次数
        inlier_threshold: 内点阈值（米）
        min_inliers: 最小内点数
        early_stop_ratio: 早停内点比例
        
    Returns:
        (sim3_transform, inlier_mask)
    """
    n_points = len(src_points)
    if n_points < 3:
        return None, np.zeros(n_points, dtype=bool)
    
    best_inlier_count = 0
    best_inliers = np.zeros(n_points, dtype=bool)
    best_sim3 = None
    
    # 早停阈值
    early_stop_count = int(n_points * early_stop_ratio)
    
    # 预计算平方阈值
    inlier_threshold_sq = inlier_threshold * inlier_threshold
    
    # 一次性向量化生成所有随机索引，避免 Python 循环
    rng = np.random.default_rng()
    all_indices = rng.integers(0, n_points, size=(max_iterations, 3))
    
    for iteration in range(max_iterations):
        sample_idx = all_indices[iteration]
        # 跳过含重复索引的样本（n_points >> 3 时概率极低，约 3/n_points）
        if sample_idx[0] == sample_idx[1] or sample_idx[0] == sample_idx[2] or sample_idx[1] == sample_idx[2]:
            continue
        
        src_sample = src_points[sample_idx]
        tgt_sample = tgt_points[sample_idx]
        
        # 检查是否共线（使用向量化叉积）
        v1 = src_sample[1] - src_sample[0]
        v2 = src_sample[2] - src_sample[0]
        cross = np.cross(v1, v2)
        if cross @ cross < 1e-12:
            continue
        
        # 估计 Sim3
        try:
            scale, R, t = umeyama_alignment(src_sample, tgt_sample, with_scale=True)
        except:
            continue
        
        # 向量化计算残差（使用平方距离）
        transformed = scale * (src_points @ R.T) + t
        diff = transformed - tgt_points
        residuals_sq = np.einsum('ij,ij->i', diff, diff)
        inliers = residuals_sq < inlier_threshold_sq
        inlier_count = np.count_nonzero(inliers)  # 比 .sum() 快
        
        # 更新最佳结果
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            
            # 使用所有内点重新估计
            if inlier_count >= min_inliers:
                scale, R, t = umeyama_alignment(
                    src_points[inliers], 
                    tgt_points[inliers], 
                    with_scale=True
                )
                rotation = pycolmap.Rotation3d(R)
                best_sim3 = pycolmap.Sim3d(scale, rotation, t)
                
                # 早停检查
                if inlier_count >= early_stop_count:
                    break
    
    return best_sim3, best_inliers


def rescale_reconstruction_to_original_size(
    reconstruction: pycolmap.Reconstruction,
    ori_extrinsics: List[Dict],
    sfm_reconstructions: List[Dict],
    start_idx: int,
    end_idx: int,
    alignment_mode: str = 'auto',  # 'auto' | 'pcl_alignment' | 'image_alignment'
    image_alignment_max_error: float = 5.0,
    image_alignment_min_inlier_ratio: float = 0.3,
    verbose: bool = False,
) -> pycolmap.Reconstruction:
    """
    将 reconstruction 对齐到已知的影像 pose 位置。

    对齐方法：
        - 方法1（点云对齐）：使用与最新 SfM 重建的 3D 点对应关系估计 Sim3 变换
        - 方法2（影像位置对齐）：使用 RANSAC 将重建对齐到已知相机中心位置

    Args:
        reconstruction: pycolmap 重建结果
        ori_extrinsics: 原始外参列表，每个元素包含 'R_camera' 和 'tvec'
        sfm_reconstructions: SfM 重建结果列表，用于点云对齐
        start_idx: 起始影像索引
        end_idx: 结束影像索引
        alignment_mode: 对齐方式，'auto' | 'pcl_alignment' | 'image_alignment'
        image_alignment_max_error: 影像对齐 RANSAC 最大误差（米）
        image_alignment_min_inlier_ratio: 影像对齐 RANSAC 最小内点比例
        verbose: 是否输出详细信息

    Returns:
        对齐后的 reconstruction
    """
    valid_modes = {'auto', 'pcl_alignment', 'image_alignment'}
    if alignment_mode not in valid_modes:
        raise ValueError(f"alignment_mode 必须是 {valid_modes} 之一，当前为: {alignment_mode}")

    alignment_success = False
    use_method1 = alignment_mode in ('auto', 'pcl_alignment')
    use_method2 = alignment_mode in ('auto', 'image_alignment')

    # 方法1：通过3D点云配准到最新的SfM重建
    if use_method1 and sfm_reconstructions:
        if verbose:
            print("  Attempting alignment to latest SfM via 3D point cloud (pcl_alignment)...")

        sfm_result = sfm_reconstructions[-1]
        tgt_reconstruction = sfm_result['reconstruction']
        src_reconstruction = reconstruction
        num_tgt_images = len(tgt_reconstruction.images)
        num_src_images = len(src_reconstruction.images)

        if num_tgt_images > 0 and num_src_images > 0 and num_tgt_images == num_src_images:
            # 选取首、中、末三张影像进行匹配
            if num_tgt_images <= 3:
                sel_image_idx = list(range(1, num_tgt_images + 1))
            else:
                sel_image_idx = [1, (num_tgt_images + 1) // 2, num_tgt_images]

            pixel_threshold = 3.0
            tgt_images = tgt_reconstruction.images
            src_images = src_reconstruction.images
            tgt_points3D = tgt_reconstruction.points3D
            src_points3D = src_reconstruction.points3D

            # 合并对应关系收集与3D坐标提取（一趟遍历，省去中间 point_correspondences 列表）
            all_tgt_xyz = []
            all_src_xyz = []

            for index in sel_image_idx:
                tgt_image_obj = tgt_images[index]
                src_image_obj = src_images[index]

                src_coords, src_pt3d_ids = _extract_valid_points2d(src_image_obj)
                if not src_coords:
                    continue

                if HAS_KDTREE:
                    src_coords_arr = np.asarray(src_coords, dtype=np.float64)
                    correspondences = find_single_images_pair_matches_kdtree(
                        tgt_image_obj, src_image_obj,
                        src_coords_arr, src_pt3d_ids,
                        pixel_threshold,
                    )
                else:
                    src_spatial_index = {}
                    for i, (x, y) in enumerate(src_coords):
                        grid_key = (int(round(x)), int(round(y)))
                        if grid_key in src_spatial_index:
                            src_spatial_index[grid_key].append((src_pt3d_ids[i], (x, y)))
                        else:
                            src_spatial_index[grid_key] = [(src_pt3d_ids[i], (x, y))]
                    correspondences = find_single_images_pair_matches(
                        tgt_image_obj, src_image_obj,
                        src_spatial_index, pixel_threshold,
                    )

                # 直接提取3D坐标，避免存储中间对应关系列表
                for tgt_pt3d_id, src_pt3d_id, _ in correspondences:
                    if tgt_pt3d_id in tgt_points3D and src_pt3d_id in src_points3D:
                        all_tgt_xyz.append(tgt_points3D[tgt_pt3d_id].xyz)
                        all_src_xyz.append(src_points3D[src_pt3d_id].xyz)

            n_valid = len(all_tgt_xyz)
            if n_valid == 0:
                if verbose:
                    print("  Warning: No point correspondences found between overlapping regions")
            elif n_valid < 3:
                if verbose:
                    print(f"  Warning: Not enough 3D points for Sim3 estimation ({n_valid} pairs)")
            else:
                tgt_pts3d = np.array(all_tgt_xyz, dtype=np.float64)
                src_pts3d = np.array(all_src_xyz, dtype=np.float64)

                try:
                    sim3_transform = estimate_sim3_transform(src_pts3d, tgt_pts3d)
                    if sim3_transform is not None:
                        reconstruction.transform(sim3_transform)
                        alignment_success = True
                        if verbose:
                            print(f"  ✓ Reconstruction aligned to latest SfM via 3D point cloud")
                            print(f"    Scale: {sim3_transform.scale:.6f}, Used 3D points: {n_valid}")
                    elif verbose:
                        print("  Warning: Failed to compute Sim3 transform")
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Sim3 estimation failed: {e}")
                        import traceback
                        traceback.print_exc()

    if not alignment_success and verbose:
        print("  No successful alignment with SfM reconstructions, falling back to GPS-based alignment...")

    # 方法2：GPS位置对齐
    if use_method2 and not alignment_success:
        if verbose:
            print(f"  Using GPS-based alignment...")

        if end_idx - start_idx == 0:
            print("  Warning: No matching images found for alignment")
            return reconstruction

        # 建立 image_name -> (R, t) 映射，预转换为 numpy 数组（避免逐张影像重复转换）
        extrinsic_by_name = {}
        for ext in ori_extrinsics:
            extrinsic_by_name[ext['image_name']] = (
                np.asarray(ext['R_camera'], dtype=np.float64),
                np.asarray(ext['tvec'], dtype=np.float64).ravel(),
            )

        # 收集匹配的图像名称和 R, t
        valid_names = []
        R_list = []
        t_list = []

        for image in reconstruction.images.values():
            Rt = extrinsic_by_name.get(image.name)
            if Rt is not None:
                valid_names.append(image.name)
                R_list.append(Rt[0])
                t_list.append(Rt[1])

        n_valid = len(valid_names)
        if n_valid < 3:
            print(f"  Warning: Not enough matching images for alignment ({n_valid} found)")
            return reconstruction

        # 批量计算相机中心: camera_center_i = -R_i^T @ t_i
        R_batch = np.stack(R_list)  # (N, 3, 3)
        t_batch = np.stack(t_list)  # (N, 3)
        camera_centers = -np.einsum('nji,nj->ni', R_batch, t_batch)

        if verbose:
            print(f"    Found {n_valid} matching images in reconstruction")

        ransac_options = pycolmap.RANSACOptions()
        ransac_options.max_error = image_alignment_max_error
        ransac_options.min_inlier_ratio = image_alignment_min_inlier_ratio

        try:
            sim3d = pycolmap.align_reconstruction_to_locations(
                src=reconstruction,
                tgt_image_names=valid_names,
                tgt_locations=camera_centers,
                min_common_points=3,
                ransac_options=ransac_options
            )
            if sim3d is not None:
                reconstruction.transform(sim3d)
                if verbose:
                    print(f"  ✓ Reconstruction aligned to known poses")
                    print(f"    Scale: {sim3d.scale}, Aligned images: {n_valid}")
            else:
                print("  Warning: Failed to align reconstruction")

        except Exception as e:
            print(f"  Error aligning reconstruction: {e}")
            import traceback
            traceback.print_exc()

    return reconstruction

