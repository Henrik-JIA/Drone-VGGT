#!/usr/bin/env python3
"""
Reconstruction alignment utilities for SfM.

This module provides functions for aligning pycolmap reconstructions
using various methods including point cloud matching and GPS-based alignment.
"""

from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pycolmap


def find_single_images_pair_matches(
    prev_image,
    curr_image,
    curr_spatial_index: dict,
    pixel_threshold: float,
) -> list:
    """
    在给定的 prev_image 与 curr_image 之间建立 3D 点对应关系（单对影像）。
    
    Args:
        prev_image: 前一个重建的影像对象
        curr_image: 当前重建的影像对象  
        curr_spatial_index: 以整数像素为键，值为 [(point3D_id, xy_float), ...] 的字典
        pixel_threshold: 像素距离阈值
        
    Returns:
        对应关系列表 [(prev_point3D_id, curr_point3D_id, dist)]
    """
    correspondences = []

    search_radius = float(pixel_threshold)
    window = int(np.ceil(search_radius))
    used_curr_ids = set()  # 避免 curr 侧重复匹配

    for point2D in prev_image.points2D:
        prev_pt3d_id = int(point2D.point3D_id)
        if prev_pt3d_id == -1:
            continue

        prev_xy = np.asarray(point2D.xy, dtype=np.float64)
        cx, cy = int(round(prev_xy[0])), int(round(prev_xy[1]))

        best_curr_id = None
        best_dist = search_radius

        # 在 curr 图像索引中按窗口搜索邻域
        for dx in range(-window, window + 1):
            for dy in range(-window, window + 1):
                search_key = (cx + dx, cy + dy)
                if search_key not in curr_spatial_index:
                    continue

                for curr_pt3d_id, curr_xy in curr_spatial_index[search_key]:
                    curr_pt3d_id = int(curr_pt3d_id)
                    if curr_pt3d_id in used_curr_ids:
                        continue

                    dist = float(np.linalg.norm(prev_xy - np.asarray(curr_xy, dtype=np.float64)))
                    if dist < best_dist and dist < search_radius:
                        best_dist = dist
                        best_curr_id = curr_pt3d_id

        if best_curr_id is not None:
            correspondences.append((prev_pt3d_id, best_curr_id, best_dist))
            used_curr_ids.add(best_curr_id)

    return correspondences


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
    min_inliers: int = 10
) -> Tuple[Optional[pycolmap.Sim3d], np.ndarray]:
    """
    使用 RANSAC 鲁棒估计 Sim3 变换。
    
    Args:
        src_points: 源点云 (N, 3)
        tgt_points: 目标点云 (N, 3)
        max_iterations: RANSAC 最大迭代次数
        inlier_threshold: 内点阈值（米）
        min_inliers: 最小内点数
        
    Returns:
        (sim3_transform, inlier_mask)
    """
    n_points = len(src_points)
    if n_points < 3:
        return None, np.zeros(n_points, dtype=bool)
    
    best_inliers = np.zeros(n_points, dtype=bool)
    best_sim3 = None
    
    for _ in range(max_iterations):
        # 随机采样3个点
        sample_idx = np.random.choice(n_points, 3, replace=False)
        src_sample = src_points[sample_idx]
        tgt_sample = tgt_points[sample_idx]
        
        # 检查是否共线
        v1 = src_sample[1] - src_sample[0]
        v2 = src_sample[2] - src_sample[0]
        if np.linalg.norm(np.cross(v1, v2)) < 1e-6:
            continue
        
        # 估计 Sim3
        try:
            scale, R, t = umeyama_alignment(src_sample, tgt_sample, with_scale=True)
        except:
            continue
        
        # 计算所有点的残差
        transformed = scale * (src_points @ R.T) + t
        residuals = np.linalg.norm(transformed - tgt_points, axis=1)
        inliers = residuals < inlier_threshold
        
        # 更新最佳结果
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            
            # 使用所有内点重新估计
            if inliers.sum() >= min_inliers:
                scale, R, t = umeyama_alignment(
                    src_points[inliers], 
                    tgt_points[inliers], 
                    with_scale=True
                )
                rotation = pycolmap.Rotation3d(R)
                best_sim3 = pycolmap.Sim3d(scale, rotation, t)
    
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
    # 参数校验
    valid_modes = {'auto', 'pcl_alignment', 'image_alignment'}
    if alignment_mode not in valid_modes:
        raise ValueError(f"alignment_mode 必须是 {valid_modes} 之一，当前为: {alignment_mode}")

    alignment_success = False
    use_method1 = alignment_mode in ('auto', 'pcl_alignment')
    use_method2 = alignment_mode in ('auto', 'image_alignment')

    # 方法1：直接通过3D点云配准到最新的SfM重建
    if use_method1 and len(sfm_reconstructions) > 0:
        if verbose:
            print("  Attempting alignment to latest SfM via 3D point cloud (pcl_alignment)...")

        sfm_result = sfm_reconstructions[-1]
        tgt_reconstruction = sfm_result['reconstruction']
        src_reconstruction = reconstruction
        num_tgt_images = len(tgt_reconstruction.images)
        num_src_images = len(src_reconstruction.images)
        
        if num_tgt_images != 0 and num_src_images != 0 and num_tgt_images == num_src_images:
            # 优化：直接生成选择的图像索引列表
            if num_tgt_images <= 2:
                sel_image_idx = list(range(1, num_tgt_images + 1))
            elif num_tgt_images == 3:
                sel_image_idx = [1, 2, 3]
            else:
                sel_image_idx = [1, (num_tgt_images + 1) // 2, num_tgt_images]

            point_correspondences = []
            pixel_threshold = 3.0
            
            for index in sel_image_idx:
                tgt_image_obj = tgt_reconstruction.images[index]
                src_image_obj = src_reconstruction.images[index]

                # 使用 defaultdict 简化空间索引构建
                src_spatial_index = defaultdict(list)
                
                for point2D in src_image_obj.points2D:
                    if point2D.point3D_id != -1:
                        grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
                        src_spatial_index[grid_key].append(
                            (int(point2D.point3D_id), np.asarray(point2D.xy, dtype=np.float64))
                        )

                correspondences = find_single_images_pair_matches(
                    tgt_image_obj, 
                    src_image_obj, 
                    src_spatial_index, 
                    pixel_threshold, 
                )
                point_correspondences.extend(correspondences)

            if len(point_correspondences) == 0:
                print("  Warning: No point correspondences found between overlapping regions")
            else:
                # 批量提取3D点坐标
                tgt_points3D = tgt_reconstruction.points3D
                src_points3D = src_reconstruction.points3D
                
                # 使用列表推导式一次性过滤和提取
                valid_pairs = [
                    (tgt_pt3d_id, src_pt3d_id)
                    for tgt_pt3d_id, src_pt3d_id, _ in point_correspondences
                    if tgt_pt3d_id in tgt_points3D and src_pt3d_id in src_points3D
                ]
                
                if len(valid_pairs) >= 3:
                    # 批量提取坐标
                    tgt_pts3d = np.array([tgt_points3D[pid].xyz for pid, _ in valid_pairs], dtype=np.float64)
                    src_pts3d = np.array([src_points3D[pid].xyz for _, pid in valid_pairs], dtype=np.float64)

                    # 估计 Sim3（src → tgt）
                    try:
                        sim3_transform = estimate_sim3_transform(src_pts3d, tgt_pts3d)
                        if sim3_transform is not None:
                            reconstruction.transform(sim3_transform)
                            alignment_success = True
                            if verbose:
                                print(f"  ✓ Reconstruction aligned to latest SfM via 3D point cloud")
                                print(f"    Scale: {sim3_transform.scale:.6f}")
                                print(f"    Used 3D points: {len(src_pts3d)}")
                        else:
                            if verbose:
                                print("  Warning: Failed to compute Sim3 transform")
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Sim3 estimation failed: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    if verbose:
                        print(f"  Warning: Not enough 3D points for Sim3 estimation ({len(valid_pairs)} pairs)")

    if not alignment_success and verbose:
        print("  No successful alignment with SfM reconstructions, falling back to GPS-based alignment...")

    # 方法2：GPS位置对齐
    if use_method2 and not alignment_success:
        if verbose:
            print(f"  Using GPS-based alignment...")
    
        # 使用列表推导式提取影像名称
        tgt_image_names = [image.name for image in reconstruction.images.values()]
        
        # 向量化计算相机位置
        num_images = end_idx - start_idx
        indices = range(start_idx, end_idx)
        
        # 预提取所有外参数据
        R_cameras = []
        tvecs = []
        for idx in indices:
            extrinsic_info = ori_extrinsics[idx]
            R_cameras.append(extrinsic_info['R_camera'])
            tvecs.append(extrinsic_info['tvec'])
        
        # 转为 numpy 数组进行向量化计算
        R_cameras = np.array(R_cameras, dtype=np.float64)  # (N, 3, 3)
        tvecs = np.array(tvecs, dtype=np.float64)  # (N, 3) or (N, 3, 1)
        
        # 确保 tvecs 形状正确
        if tvecs.ndim == 2:
            tvecs = tvecs[..., np.newaxis]  # (N, 3, 1)
        
        # 向量化计算：camera_center = -R^T @ t
        R_cameras_T = np.transpose(R_cameras, (0, 2, 1))  # (N, 3, 3)
        camera_centers = -np.matmul(R_cameras_T, tvecs).squeeze(-1)  # (N, 3)
        
        tgt_locations = camera_centers
        
        # 生成有效名称列表
        valid_names = [f"image_{fidx + 1}" for fidx in range(num_images)]
        
        if len(valid_names) == 0:
            print("  Warning: No matching images found for alignment")
            return reconstruction

        # RANSAC 对齐
        ransac_options = pycolmap.RANSACOptions()
        ransac_options.max_error = image_alignment_max_error
        ransac_options.min_inlier_ratio = image_alignment_min_inlier_ratio

        try:
            sim3d = pycolmap.align_reconstruction_to_locations(
                src=reconstruction,
                tgt_image_names=valid_names,
                tgt_locations=tgt_locations,
                min_common_points=3,
                ransac_options=ransac_options
            )
            if sim3d is not None:
                reconstruction.transform(sim3d)
                
                if verbose:
                    print(f"  ✓ Reconstruction aligned to known poses")
                    print(f"    Scale: {sim3d.scale}")
                    print(f"    Number of aligned images: {len(valid_names)}")
            else:
                print("  Warning: Failed to align reconstruction")
                
        except Exception as e:
            print(f"  Error aligning reconstruction: {e}")
            import traceback
            traceback.print_exc()
        
    return reconstruction

