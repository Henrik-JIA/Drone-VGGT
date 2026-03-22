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


def _save_points_to_ply(points: np.ndarray, filepath: str,
                        color: Tuple[int, int, int] = (255, 255, 255)):
    """将 (N,3) 点云保存为带颜色的 ASCII PLY 文件。"""
    n = len(points)
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        r, g, b = color
        for x, y, z in points:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


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


def align_reconstruction_by_overlap(
    curr_recon: pycolmap.Reconstruction,
    prev_recon: pycolmap.Reconstruction,
    pixel_threshold: float = 5.0,
    max_correspondences: int = 0,
    verbose: bool = False,
) -> Tuple[pycolmap.Reconstruction, List[Tuple[int, int]]]:
    """
    通过重叠影像的共享特征像素，将 curr_recon 对齐到 prev_recon 的坐标系。

    适用于增量 SfM 中相邻 batch 的链式对齐：batch N 的前几张影像
    与 batch N-1 的后几张影像相同，利用这些共同影像上的 2D 特征
    像素匹配建立 3D-3D 对应关系，然后估计 Sim3 变换。

    步骤：
      1. 按影像名称找到 prev_recon 与 curr_recon 的共同影像
      2. Stage 1: 共同影像的相机中心 → Umeyama 粗对齐
      3. Stage 2: 逐影像批量 KDTree 查询建立 3D-3D 对应
      4. Umeyama + 离群点剔除精修

    Args:
        curr_recon: 当前 batch 的重建（将被变换）
        prev_recon: 前一个 batch 的重建（目标坐标系，不变）
        pixel_threshold: 像素匹配距离阈值
        max_correspondences: 控制返回的匹配点对数量，同时也控制对齐使用的点数：
            - 0: 找到并返回所有匹配对（默认）
            - N>0: 最多找 N 个匹配对并返回；若实际匹配不足 N 则返回全部
        verbose: 是否输出详细信息

    Returns:
        (aligned_curr_recon, matched_pairs):
            - aligned_curr_recon: 对齐后的 curr_recon
            - matched_pairs: 匹配的 3D 点 ID 对列表 [(prev_pt3d_id, curr_pt3d_id), ...]
              长度 = min(max_correspondences, 实际匹配数)（max_correspondences=0 时为全部）
              可直接用于后续点云合并，避免重复计算 2D 像素匹配
    """
    from pathlib import Path as _Path

    # ---- 按影像名称匹配 ----
    curr_name_map = {}
    for img_id in curr_recon.images:
        name = _Path(curr_recon.images[img_id].name).name
        curr_name_map[name] = img_id
    prev_name_map = {}
    for img_id in prev_recon.images:
        name = _Path(prev_recon.images[img_id].name).name
        prev_name_map[name] = img_id

    common_names = sorted(set(curr_name_map) & set(prev_name_map))
    n_common = len(common_names)

    if verbose:
        print(f"  [align_by_overlap] Common images: {n_common} "
              f"(curr={len(curr_recon.images)}, prev={len(prev_recon.images)})")

    if n_common < 2:
        if verbose:
            print("  ⚠ Not enough common images for alignment")
        return curr_recon, []

    # ---- Stage 1: 相机中心 Umeyama 粗对齐 ----
    curr_centers = []
    prev_centers = []
    for name in common_names:
        c_img = curr_recon.images[curr_name_map[name]]
        R_c = np.array(c_img.cam_from_world.rotation.matrix())
        t_c = np.array(c_img.cam_from_world.translation)
        curr_centers.append(-R_c.T @ t_c)

        p_img = prev_recon.images[prev_name_map[name]]
        R_p = np.array(p_img.cam_from_world.rotation.matrix())
        t_p = np.array(p_img.cam_from_world.translation)
        prev_centers.append(-R_p.T @ t_p)

    curr_cam_arr = np.array(curr_centers, dtype=np.float64)
    prev_cam_arr = np.array(prev_centers, dtype=np.float64)

    sim3_coarse = estimate_sim3_transform(curr_cam_arr, prev_cam_arr)
    if sim3_coarse is None:
        if verbose:
            print("  ⚠ Umeyama on camera centers failed")
        return curr_recon, []

    curr_recon.transform(sim3_coarse)

    if verbose:
        s_c = sim3_coarse.scale
        transformed = s_c * (curr_cam_arr @ np.array(sim3_coarse.rotation.matrix()).T) + np.array(sim3_coarse.translation)
        residuals = np.linalg.norm(transformed - prev_cam_arr, axis=1)
        print(f"    Stage 1 (camera centers): scale={s_c:.6f}, "
              f"residual mean={residuals.mean():.4f}m, max={residuals.max():.4f}m")

    # ---- Stage 2: 逐影像批量像素匹配建立 3D-3D 对应（优化版） ----
    prev_points3D = prev_recon.points3D
    curr_points3D = curr_recon.points3D
    target = max_correspondences if max_correspondences > 0 else float('inf')

    curr_pt3d_set = set(curr_points3D)
    prev_pt3d_set = set(prev_points3D)

    # 为共同影像在 curr 侧建立 KDTree + numpy ID 数组
    curr_img_search = {}
    for name in common_names:
        c_id = curr_name_map[name]
        c_img = curr_recon.images[c_id]
        obs_idxs = c_img.get_observation_point2D_idxs()
        if not obs_idxs:
            continue
        pts2d = c_img.points2D
        coords = []
        pt3d_ids = []
        for idx in obs_idxs:
            pt2d = pts2d[idx]
            pid = pt2d.point3D_id
            if pid in curr_pt3d_set:
                coords.append((float(pt2d.xy[0]), float(pt2d.xy[1])))
                pt3d_ids.append(pid)
        if not coords:
            continue
        curr_img_search[c_id] = (
            cKDTree(np.asarray(coords, dtype=np.float64)),
            np.array(pt3d_ids, dtype=np.int64),
        )

    # 收集所有影像的候选匹配（不做逐影像去重，统一全局去重）
    cand_dists_list = []
    cand_prev_list = []
    cand_curr_list = []

    for name in common_names:
        curr_img_id = curr_name_map[name]
        if curr_img_id not in curr_img_search:
            continue
        tree, curr_ids_arr = curr_img_search[curr_img_id]

        prev_img = prev_recon.images[prev_name_map[name]]
        obs_idxs = prev_img.get_observation_point2D_idxs()
        if not obs_idxs:
            continue

        pts2d = prev_img.points2D
        q_pixels = []
        q_pids = []
        for idx in obs_idxs:
            pt2d = pts2d[idx]
            pid = pt2d.point3D_id
            if pid in prev_pt3d_set:
                q_pixels.append((float(pt2d.xy[0]), float(pt2d.xy[1])))
                q_pids.append(pid)
        if not q_pixels:
            continue

        q_arr = np.asarray(q_pixels, dtype=np.float64)
        dists, idxs = tree.query(
            q_arr, k=1, distance_upper_bound=pixel_threshold,
            workers=-1 if len(q_arr) > 500 else 1,
        )

        valid = np.isfinite(dists)
        v_idx = np.where(valid)[0]
        if len(v_idx) == 0:
            continue

        q_pids_arr = np.array(q_pids, dtype=np.int64)
        cand_dists_list.append(dists[v_idx])
        cand_prev_list.append(q_pids_arr[v_idx])
        cand_curr_list.append(curr_ids_arr[idxs[v_idx]])

    # 全局排序 + 贪心去重（按像素距离由近到远，保证匹配质量）
    matched_id_pairs = []
    all_prev_xyz = []
    all_curr_xyz = []

    if cand_dists_list:
        all_cand_dists = np.concatenate(cand_dists_list)
        all_cand_prev = np.concatenate(cand_prev_list)
        all_cand_curr = np.concatenate(cand_curr_list)

        order = np.argsort(all_cand_dists, kind='quicksort')

        used_prev = set()
        used_curr = set()
        for oi in order:
            if len(matched_id_pairs) >= target:
                break
            ppid = int(all_cand_prev[oi])
            cpid = int(all_cand_curr[oi])
            if ppid in used_prev or cpid in used_curr:
                continue
            used_prev.add(ppid)
            used_curr.add(cpid)
            matched_id_pairs.append((ppid, cpid))
            all_prev_xyz.append(prev_points3D[ppid].xyz)
            all_curr_xyz.append(curr_points3D[cpid].xyz)

    n_corr = len(matched_id_pairs)
    if verbose:
        limit_str = f" (limit={max_correspondences})" if max_correspondences > 0 else ""
        print(f"    Stage 2: {n_corr} 3D-3D correspondences via batch pixel matching{limit_str}")

    if n_corr >= 3:
        prev_pts = np.array(all_prev_xyz, dtype=np.float64)
        curr_pts = np.array(all_curr_xyz, dtype=np.float64)

        try:
            s1, R1, t1 = umeyama_alignment(curr_pts, prev_pts, with_scale=True)
            transformed_1 = s1 * (curr_pts @ R1.T) + t1
            residuals_1 = np.linalg.norm(transformed_1 - prev_pts, axis=1)

            if verbose:
                print(f"    Pass 1 (all {n_corr} pts): scale={s1:.6f}, "
                      f"residual mean={residuals_1.mean():.4f}m, "
                      f"median={np.median(residuals_1):.4f}m")

            outlier_threshold = max(
                np.median(residuals_1) * 3.0,
                np.percentile(residuals_1, 75) * 2.0,
                0.5,
            )
            inlier_mask = residuals_1 < outlier_threshold
            n_inliers = int(np.count_nonzero(inlier_mask))

            if verbose:
                print(f"    Outlier rejection: threshold={outlier_threshold:.4f}m, "
                      f"inliers={n_inliers}/{n_corr}")

            if n_inliers >= 3:
                s2, R2, t2 = umeyama_alignment(
                    curr_pts[inlier_mask], prev_pts[inlier_mask], with_scale=True
                )
                sim3_fine = pycolmap.Sim3d(s2, pycolmap.Rotation3d(R2), t2)
                curr_recon.transform(sim3_fine)

                if verbose:
                    transformed_2 = s2 * (curr_pts[inlier_mask] @ R2.T) + t2
                    residuals_2 = np.linalg.norm(transformed_2 - prev_pts[inlier_mask], axis=1)
                    print(f"  ✓ Fine alignment: scale={s2:.6f}, "
                          f"inliers={n_inliers}/{n_corr}, "
                          f"residual mean={residuals_2.mean():.4f}m")
            else:
                sim3_p1 = pycolmap.Sim3d(s1, pycolmap.Rotation3d(R1), t1)
                curr_recon.transform(sim3_p1)
                if verbose:
                    print(f"  ⚠ Too few inliers ({n_inliers}), using Pass 1")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Fine alignment error ({e}), coarse alignment applied")
    else:
        if verbose:
            print(f"  ⚠ Too few 3D correspondences ({n_corr}), coarse alignment only")

    return curr_recon, matched_id_pairs


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

    # 方法1：两阶段粗到精对齐
    #   Stage 1: 相机中心配对 → Umeyama 完整 Sim3（含旋转）
    #   Stage 2: 3D 点像素匹配 → RANSAC Sim3 精修
    if use_method1 and sfm_reconstructions:
        if verbose:
            print("  Attempting coarse-to-fine alignment to latest SfM (pcl_alignment)...")

        sfm_result = sfm_reconstructions[-1]
        tgt_recon = sfm_result['reconstruction']

        # ---------- 按影像名称匹配（不依赖 ID 顺序） ----------
        from pathlib import Path as _Path
        src_name_map = {}
        for img_id in reconstruction.images:
            name = _Path(reconstruction.images[img_id].name).name
            src_name_map[name] = img_id
        tgt_name_map = {}
        for img_id in tgt_recon.images:
            name = _Path(tgt_recon.images[img_id].name).name
            tgt_name_map[name] = img_id

        common_names = sorted(set(src_name_map) & set(tgt_name_map))

        src_centers = []
        tgt_centers = []
        matched_pairs = []
        for name in common_names:
            src_id = src_name_map[name]
            tgt_id = tgt_name_map[name]
            src_img = reconstruction.images[src_id]
            R_s = np.array(src_img.cam_from_world.rotation.matrix())
            t_s = np.array(src_img.cam_from_world.translation)
            src_centers.append(-R_s.T @ t_s)
            tgt_img = tgt_recon.images[tgt_id]
            R_t = np.array(tgt_img.cam_from_world.rotation.matrix())
            t_t = np.array(tgt_img.cam_from_world.translation)
            tgt_centers.append(-R_t.T @ t_t)
            matched_pairs.append((src_id, tgt_id))

        n_common = len(matched_pairs)
        if verbose:
            print(f"    Matched {n_common} cameras by name "
                  f"(src={len(reconstruction.images)}, tgt={len(tgt_recon.images)})")

        if n_common >= 3:
            src_cam_arr = np.array(src_centers, dtype=np.float64)
            tgt_cam_arr = np.array(tgt_centers, dtype=np.float64)

            # ========== Stage 1: 粗对齐 — Umeyama 完整 Sim3（scale+rotation+translation） ==========
            sim3_coarse = estimate_sim3_transform(src_cam_arr, tgt_cam_arr)

            if sim3_coarse is not None:
                s_c = sim3_coarse.scale
                R_c = np.array(sim3_coarse.rotation.matrix())
                t_c = np.array(sim3_coarse.translation)
                transformed_centers = s_c * (src_cam_arr @ R_c.T) + t_c
                center_residuals = np.linalg.norm(transformed_centers - tgt_cam_arr, axis=1)

                reconstruction.transform(sim3_coarse)

                if verbose:
                    print(f"    Stage 1 (Umeyama on cameras): scale={s_c:.6f}")
                    print(f"      Camera residual: mean={center_residuals.mean():.4f}m, "
                          f"max={center_residuals.max():.4f}m")

                # ========== Stage 2: tgt 3D点 → 像素观测 → src 特征匹配 → RANSAC Sim3 精修 ==========
                pixel_threshold = 5.0
                tgt_points3D = tgt_recon.points3D
                src_points3D = reconstruction.points3D
                all_tgt_xyz = []
                all_src_xyz = []

                # 为共同影像在 src 侧建立 2D 特征空间索引
                src_img_search = {}
                for name in common_names:
                    s_id = src_name_map[name]
                    s_img = reconstruction.images[s_id]
                    coords, pt3d_ids = _extract_valid_points2d(s_img)
                    if not coords:
                        continue
                    if HAS_KDTREE:
                        coords_arr = np.asarray(coords, dtype=np.float64)
                        src_img_search[s_id] = (cKDTree(coords_arr), pt3d_ids)
                    else:
                        spatial_idx = {}
                        for i, (x, y) in enumerate(coords):
                            gk = (int(round(x)), int(round(y)))
                            spatial_idx.setdefault(gk, []).append((pt3d_ids[i], (x, y)))
                        src_img_search[s_id] = (spatial_idx, pt3d_ids)

                tgt_to_src_img = {}
                for name in common_names:
                    tgt_to_src_img[tgt_name_map[name]] = src_name_map[name]

                # 遍历 tgt 3D 点，通过其在共同影像上的像素观测寻找 src 对应 3D 点
                used_src_pt3d = set()
                window = int(pixel_threshold) + 1

                for tgt_pt3d_id, tgt_pt3d in tgt_points3D.items():
                    best_src_pt3d_id = None
                    best_dist = pixel_threshold

                    for track_elem in tgt_pt3d.track.elements:
                        tgt_img_id = track_elem.image_id
                        if tgt_img_id not in tgt_to_src_img:
                            continue
                        src_img_id = tgt_to_src_img[tgt_img_id]
                        if src_img_id not in src_img_search:
                            continue

                        tgt_pixel = tgt_recon.images[tgt_img_id].points2D[
                            track_elem.point2D_idx
                        ].xy
                        px, py = float(tgt_pixel[0]), float(tgt_pixel[1])

                        search_data = src_img_search[src_img_id]
                        if HAS_KDTREE:
                            tree, pt3d_ids = search_data
                            dist, idx = tree.query(
                                [px, py], k=1, distance_upper_bound=pixel_threshold
                            )
                            if dist < best_dist:
                                cand_id = pt3d_ids[idx]
                                if cand_id in src_points3D and cand_id not in used_src_pt3d:
                                    best_dist = dist
                                    best_src_pt3d_id = cand_id
                        else:
                            spatial_idx, _ = search_data
                            cx, cy = int(round(px)), int(round(py))
                            for dx in range(-window, window + 1):
                                for dy in range(-window, window + 1):
                                    cands = spatial_idx.get((cx + dx, cy + dy))
                                    if cands is None:
                                        continue
                                    for cand_id, (cand_x, cand_y) in cands:
                                        if cand_id in used_src_pt3d or cand_id not in src_points3D:
                                            continue
                                        d_sq = (px - cand_x) ** 2 + (py - cand_y) ** 2
                                        if d_sq < best_dist * best_dist:
                                            best_dist = d_sq ** 0.5
                                            best_src_pt3d_id = cand_id

                    if best_src_pt3d_id is not None:
                        used_src_pt3d.add(best_src_pt3d_id)
                        all_tgt_xyz.append(tgt_pt3d.xyz)
                        all_src_xyz.append(src_points3D[best_src_pt3d_id].xyz)

                n_valid = len(all_tgt_xyz)
                if verbose:
                    print(f"    Stage 2: {n_valid} 3D-3D correspondences via pixel observations")

                if n_valid >= 3:
                    tgt_pts3d = np.array(all_tgt_xyz, dtype=np.float64)
                    src_pts3d = np.array(all_src_xyz, dtype=np.float64)

                    # DEBUG: 保存对应点云到 PLY 供可视化检查
                    _save_points_to_ply(src_pts3d, "debug_src_pts3d.ply",
                                        color=(255, 0, 0))
                    _save_points_to_ply(tgt_pts3d, "debug_tgt_pts3d.ply",
                                        color=(0, 0, 255))
                    if verbose:
                        print(f"    [DEBUG] Saved {n_valid} pts → "
                              f"debug_src_pts3d.ply (red), debug_tgt_pts3d.ply (blue)")

                    raw_dists = np.linalg.norm(src_pts3d - tgt_pts3d, axis=1)
                    dist_median = float(np.median(raw_dists))
                    dist_mean = float(np.mean(raw_dists))
                    dist_p75 = float(np.percentile(raw_dists, 75))
                    if verbose:
                        print(f"    Raw 3D distances: mean={dist_mean:.4f}m, "
                              f"median={dist_median:.4f}m, p75={dist_p75:.4f}m, "
                              f"max={raw_dists.max():.4f}m")

                    adaptive_threshold = max(0.5, dist_median * 2.0, dist_p75)
                    adaptive_min_inliers = max(3, n_valid // 20)

                    if verbose:
                        print(f"    RANSAC params: threshold={adaptive_threshold:.4f}m, "
                              f"min_inliers={adaptive_min_inliers}")

                    try:
                        sim3_fine, inlier_mask = estimate_sim3_with_ransac(
                            src_pts3d, tgt_pts3d,
                            max_iterations=2000,
                            inlier_threshold=adaptive_threshold,
                            min_inliers=adaptive_min_inliers,
                            early_stop_ratio=0.85,
                        )

                        if sim3_fine is not None:
                            n_inliers = int(np.count_nonzero(inlier_mask))
                            s_f = sim3_fine.scale
                            R_f = np.array(sim3_fine.rotation.matrix())
                            t_f = np.array(sim3_fine.translation)
                            transformed = s_f * (src_pts3d[inlier_mask] @ R_f.T) + t_f
                            residuals = np.linalg.norm(
                                transformed - tgt_pts3d[inlier_mask], axis=1
                            )
                            reconstruction.transform(sim3_fine)
                            alignment_success = True
                            if verbose:
                                print(f"  ✓ Fine alignment: fine_scale={s_f:.6f}, "
                                      f"inliers={n_inliers}/{n_valid} "
                                      f"({100.*n_inliers/n_valid:.1f}%)")
                                print(f"    Residual: mean={residuals.mean():.4f}m, "
                                      f"median={np.median(residuals):.4f}m")
                        else:
                            alignment_success = True
                            if verbose:
                                print("  ⚠ Fine RANSAC failed (no model found), "
                                      "coarse alignment applied")
                    except Exception as e:
                        alignment_success = True
                        if verbose:
                            print(f"  ⚠ Fine alignment error ({e}), coarse alignment applied")
                else:
                    alignment_success = True
                    if verbose:
                        print(f"  ⚠ Too few 3D correspondences ({n_valid}), coarse alignment applied")
            else:
                if verbose:
                    print("  Warning: Umeyama on camera centers failed")
        else:
            if verbose:
                print(f"  Warning: Not enough matched cameras ({n_common}) for alignment")

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

