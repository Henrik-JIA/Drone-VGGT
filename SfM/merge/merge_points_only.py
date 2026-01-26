#!/usr/bin/env python3
"""
轻量级点云合并模块 - 只输出合并后的三维点云

相比 merge_confidence.py，此模块：
  • 不构建 pycolmap.Reconstruction 结构体
  • 不处理 track 信息
  • 不建立影像/相机映射
  • 只输出合并后的点云 (xyz, color)

适用场景：
  • 只需要最终点云用于可视化或后续处理
  • 不需要保留完整的 SfM 结构
  • 追求更快的合并速度

输出格式：
  • numpy 数组: (N, 3) xyz, (N, 3) rgb
  • 可选保存为 PLY 文件
"""

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union

import numpy as np
import pycolmap
from scipy.spatial import cKDTree

# 从体素降采样模块导入
from utils.voxel_downsample import _voxel_dedup

# 从 merge_confidence_blend 导入基础工具函数
from merge.merge_confidence_blend import (
    find_common_images,
    build_correspondences_parallel,
    build_2d_3d_correspondences,
    build_pixel_to_3d_mapping,
    find_corresponding_3d_points,
    estimate_sim3_ransac,
)


def apply_sim3_to_points(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    scale: float
) -> np.ndarray:
    """
    对点云应用 Sim3 变换: p' = scale * R @ p + t
    
    Args:
        points: (N, 3) 点云坐标
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量
        scale: 缩放因子
        
    Returns:
        变换后的点云 (N, 3)
    """
    return scale * (points @ R.T) + t


def _get_confidence_score(
    info: Dict, 
    img_id: int, 
    pixel_key: Tuple[int, int], 
    conf_cache: Dict
) -> float:
    """
    获取置信度分数（越大越好）
    """
    cached = conf_cache.get(img_id)
    if cached is not None:
        conf_map, H, W = cached
        px, py = pixel_key
        px_c = min(max(int(px), 0), W - 1)
        py_c = min(max(int(py), 0), H - 1)
        return float(conf_map[py_c, px_c])
    return -info.get('error', 1.0)


def _process_image_pair_for_merge(
    img_id1: int, 
    img_id2: int,
    pixel_map_recon1: Dict,
    pixel_map_recon2: Dict,
    match_radii: List[float],
    k_neighbors: int = 10,
) -> Optional[Tuple]:
    """
    处理单对影像的匹配（用于并行化）
    """
    pmap1 = pixel_map_recon1.get(img_id1)
    pmap2 = pixel_map_recon2.get(img_id2)
    
    if pmap1 is None or pmap2 is None or len(pmap1) == 0 or len(pmap2) == 0:
        return None
    
    if isinstance(match_radii, (int, float)):
        match_radii = [match_radii]
    match_radii = sorted(match_radii)
    
    pixels2_list = list(pmap2.keys())
    pixels1_list = list(pmap1.keys())
    n1 = len(pixels1_list)
    n2 = len(pixels2_list)
    
    pt3d_ids1 = [pmap1[pk]['point3D_id'] for pk in pixels1_list]
    pt3d_ids2 = [pmap2[pk]['point3D_id'] for pk in pixels2_list]
    
    tree2 = cKDTree(np.asarray(pixels2_list, dtype=np.float32))
    pixels1_arr = np.asarray(pixels1_list, dtype=np.float32)
    
    max_radius = match_radii[-1]
    k = min(k_neighbors, n2)
    all_distances, all_indices = tree2.query(
        pixels1_arr, k=k, distance_upper_bound=max_radius
    )
    
    if k == 1:
        all_distances = all_distances.reshape(-1, 1)
        all_indices = all_indices.reshape(-1, 1)
    
    i_indices, j_indices = np.meshgrid(np.arange(n1), np.arange(k), indexing='ij')
    i_flat = i_indices.ravel()
    dist_flat = all_distances.ravel()
    idx_flat = all_indices.ravel()
    
    valid_mask = (dist_flat <= max_radius) & (idx_flat < n2) & np.isfinite(dist_flat)
    valid_i = i_flat[valid_mask]
    valid_dist = dist_flat[valid_mask]
    valid_idx = idx_flat[valid_mask]
    
    if len(valid_dist) == 0:
        return None
    
    sort_order = np.argsort(valid_dist)
    sorted_i = valid_i[sort_order]
    sorted_dist = valid_dist[sort_order]
    sorted_idx = valid_idx[sort_order].astype(int)
    
    matches = []
    seen_r1 = set()
    seen_r2 = set()
    seen_pixel_idx1 = set()
    seen_pixel_idx2 = set()
    
    for order_idx in range(len(sorted_i)):
        i = sorted_i[order_idx]
        dist = sorted_dist[order_idx]
        idx2 = sorted_idx[order_idx]
        
        if i in seen_pixel_idx1 or idx2 in seen_pixel_idx2:
            continue
        
        pt3d_id1 = pt3d_ids1[i]
        pt3d_id2 = pt3d_ids2[idx2]
        
        if pt3d_id1 in seen_r1 or pt3d_id2 in seen_r2:
            continue
        
        seen_r1.add(pt3d_id1)
        seen_r2.add(pt3d_id2)
        seen_pixel_idx1.add(i)
        seen_pixel_idx2.add(idx2)
        
        pixel_key1 = pixels1_list[i]
        pixel_key2 = pixels2_list[idx2]
        info1 = pmap1[pixel_key1]
        info2 = pmap2[pixel_key2]
        
        matches.append((
            pixel_key1, pixel_key2, info1, info2,
            pt3d_id1, pt3d_id2, dist
        ))
    
    if len(matches) == 0:
        return None
    
    return (img_id1, img_id2, matches, pmap1, pmap2)


def merge_points_by_confidence(
    recon1: pycolmap.Reconstruction,
    pts2_xyz: np.ndarray,
    pts2_colors: np.ndarray,
    pts2_ids: np.ndarray,
    pixel_map_recon1: Dict[int, Dict],
    pixel_map_recon2: Dict[int, Dict],
    common_images: Dict[int, int],
    prev_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    curr_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    image_name_to_idx: Optional[Dict[str, int]] = None,
    match_radii: Optional[List[float]] = None,
    k_neighbors: int = 10,
    color_by_source: bool = False,
    blend_mode: str = 'select',
    blend_weight: float = 0.7,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    基于置信度的点云合并 - 轻量级版本
    
    Args:
        recon1: 第一个 reconstruction（用于读取点和置信度）
        pts2_xyz: recon2 变换后的点云坐标 (N2, 3)
        pts2_colors: recon2 的点云颜色 (N2, 3) uint8
        pts2_ids: recon2 的 point3D_id 数组 (N2,)
        pixel_map_recon1: recon1 的像素映射
        pixel_map_recon2: recon2 的像素映射（对应变换后的点）
        common_images: 共同影像映射 {img_id1: img_id2}
        prev_recon_conf: recon1 的置信度图
        curr_recon_conf: recon2 的置信度图
        image_name_to_idx: 图像名称到全局索引的映射
        match_radii: 匹配半径列表
        k_neighbors: 近邻数量
        color_by_source: 是否按来源着色
        blend_mode: 'select' 或 'weighted'
        blend_weight: 混合权重
        verbose: 是否打印信息
        
    Returns:
        merged_xyz: (M, 3) 合并后的点云坐标
        merged_colors: (M, 3) 合并后的颜色
        stats: 统计信息
    """
    if prev_recon_conf is None:
        prev_recon_conf = {}
    if curr_recon_conf is None:
        curr_recon_conf = {}
    if image_name_to_idx is None:
        image_name_to_idx = {}
    
    if match_radii is None:
        match_radii = [3, 5, 10, 20, 50]
    elif isinstance(match_radii, (int, float)):
        match_radii = [match_radii]
    match_radii = sorted(match_radii)
    
    # 构建 recon2 的 point3D_id -> index 映射
    pts2_id_to_idx = {int(pid): i for i, pid in enumerate(pts2_ids)}
    
    # 预计算置信度缓存
    def _build_conf_cache(recon, recon_conf, name_to_idx):
        cache = {}
        for img_id in recon.images:
            img_name = recon.images[img_id].name
            global_idx = name_to_idx.get(img_name)
            if global_idx is not None and global_idx in recon_conf:
                conf_map = recon_conf[global_idx]
                cache[img_id] = (conf_map, conf_map.shape[0], conf_map.shape[1])
            else:
                cache[img_id] = None
        return cache
    
    _conf_cache_r1 = _build_conf_cache(recon1, prev_recon_conf, image_name_to_idx)
    # 对于 recon2，需要特殊处理（因为没有完整 reconstruction）
    _conf_cache_r2 = {}
    
    # 统计信息
    stats = {
        'r1_total': len(recon1.points3D),
        'r2_total': len(pts2_xyz),
        'matched_pairs': 0,
        'r1_wins': 0,
        'r2_wins': 0,
        'avg_match_distance': 0.0,
    }
    
    # 记录保留和丢弃的点
    keep_r1 = set()  # point3D_id from recon1
    keep_r2 = set()  # point3D_id from recon2
    discard_r1 = set()
    discard_r2 = set()
    
    # 记录混合点
    blended_points = {}  # {(pt3d_id1, pt3d_id2): (w1, w2)}
    
    processed_r1 = set()
    processed_r2 = set()
    match_distances = []
    
    # 并行处理所有影像对
    image_pairs = list(common_images.items())
    
    if len(image_pairs) > 0:
        max_workers = min(16, len(image_pairs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda p: _process_image_pair_for_merge(
                    p[0], p[1], pixel_map_recon1, pixel_map_recon2, match_radii, k_neighbors
                ), 
                image_pairs
            ))
        
        # 收集所有匹配
        all_matches = []
        all_pmaps_r1 = {}
        all_pmaps_r2 = {}
        
        for result in results:
            if result is None:
                continue
            
            img_id1, img_id2, matches, pmap1, pmap2 = result
            
            for pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist in matches:
                all_matches.append((img_id1, img_id2, pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist))
            
            if img_id1 not in all_pmaps_r1:
                all_pmaps_r1[img_id1] = pmap1
            if img_id2 not in all_pmaps_r2:
                all_pmaps_r2[img_id2] = pmap2
        
        # 计算置信度并决策
        if len(all_matches) > 0:
            scores1 = []
            scores2 = []
            for (img_id1, img_id2, pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist) in all_matches:
                scores1.append(_get_confidence_score(info1, img_id1, pixel_key1, _conf_cache_r1))
                scores2.append(_get_confidence_score(info2, img_id2, pixel_key2, _conf_cache_r2))
            
            scores1 = np.array(scores1)
            scores2 = np.array(scores2)
            r1_wins_mask = scores1 >= scores2
            
            stats['r1_wins'] = int(np.sum(r1_wins_mask))
            stats['r2_wins'] = len(all_matches) - stats['r1_wins']
            stats['matched_pairs'] = len(all_matches)
            
            for idx, (img_id1, img_id2, pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist) in enumerate(all_matches):
                if blend_mode == 'weighted':
                    if r1_wins_mask[idx]:
                        w1, w2 = blend_weight, 1.0 - blend_weight
                    else:
                        w1, w2 = 1.0 - blend_weight, blend_weight
                    blended_points[(pt3d_id1, pt3d_id2)] = (w1, w2)
                else:
                    if r1_wins_mask[idx]:
                        keep_r1.add(pt3d_id1)
                        discard_r2.add(pt3d_id2)
                    else:
                        keep_r2.add(pt3d_id2)
                        discard_r1.add(pt3d_id1)
                
                processed_r1.add(pt3d_id1)
                processed_r2.add(pt3d_id2)
                match_distances.append(dist)
        
        # 收集未匹配的点
        def _collect_pt3d_ids(all_pmaps):
            return set(info['point3D_id'] for pmap in all_pmaps.values() for info in pmap.values())
        
        all_pt3d_ids_r1 = _collect_pt3d_ids(all_pmaps_r1)
        all_pt3d_ids_r2 = _collect_pt3d_ids(all_pmaps_r2)
        
        unmatched_r1 = all_pt3d_ids_r1 - processed_r1
        unmatched_r2 = all_pt3d_ids_r2 - processed_r2
        
        keep_r1.update(unmatched_r1)
        keep_r2.update(unmatched_r2)
    
    # 添加非共同影像的点
    common_img_ids_r1 = set(common_images.keys())
    common_img_ids_r2 = set(common_images.values())
    
    for img_id, img in recon1.images.items():
        if img_id in common_img_ids_r1:
            continue
        for pt2d in img.points2D:
            pt3d_id = pt2d.point3D_id
            if pt3d_id != -1 and pt3d_id not in processed_r1:
                keep_r1.add(pt3d_id)
    
    # 对于 recon2，需要从 pixel_map 中获取
    for img_id in pixel_map_recon2:
        if img_id in common_img_ids_r2:
            continue
        pmap = pixel_map_recon2[img_id]
        for info in pmap.values():
            pt3d_id = info['point3D_id']
            if pt3d_id not in processed_r2:
                keep_r2.add(pt3d_id)
    
    if len(match_distances) > 0:
        stats['avg_match_distance'] = float(np.mean(match_distances))
    
    if verbose:
        print(f"\n  2D matching (radii: {match_radii}):")
        print(f"    Total matched pairs: {stats['matched_pairs']}")
        print(f"    Avg match distance: {stats['avg_match_distance']:.2f}px")
        print(f"    R1 wins: {stats['r1_wins']}, R2 wins: {stats['r2_wins']}")
    
    # ========== 构建输出点云 ==========
    COLOR_BLUE = np.array([0, 0, 255], dtype=np.uint8)
    COLOR_GREEN = np.array([0, 255, 0], dtype=np.uint8)
    COLOR_RED = np.array([255, 0, 255], dtype=np.uint8)
    COLOR_MAGENTA = np.array([255, 0, 255], dtype=np.uint8)
    
    output_xyz = []
    output_colors = []
    
    # 添加 recon1 保留的点
    for pt3d_id in keep_r1:
        if pt3d_id not in recon1.points3D:
            continue
        pt3d = recon1.points3D[pt3d_id]
        output_xyz.append(pt3d.xyz)
        if color_by_source:
            is_conflict = pt3d_id in processed_r1 and pt3d_id not in (keep_r1 - processed_r1)
            output_colors.append(COLOR_RED if is_conflict else COLOR_BLUE)
        else:
            output_colors.append(pt3d.color)
    
    # 添加 recon2 保留的点
    for pt3d_id in keep_r2:
        if pt3d_id not in pts2_id_to_idx:
            continue
        idx = pts2_id_to_idx[pt3d_id]
        output_xyz.append(pts2_xyz[idx])
        if color_by_source:
            is_conflict = pt3d_id in processed_r2 and pt3d_id not in (keep_r2 - processed_r2)
            output_colors.append(COLOR_RED if is_conflict else COLOR_GREEN)
        else:
            output_colors.append(pts2_colors[idx])
    
    # 添加混合点
    for (pt3d_id1, pt3d_id2), (w1, w2) in blended_points.items():
        if pt3d_id1 not in recon1.points3D or pt3d_id2 not in pts2_id_to_idx:
            continue
        
        pt3d_r1 = recon1.points3D[pt3d_id1]
        idx2 = pts2_id_to_idx[pt3d_id2]
        
        blended_xyz = w1 * pt3d_r1.xyz + w2 * pts2_xyz[idx2]
        output_xyz.append(blended_xyz)
        
        if color_by_source:
            output_colors.append(COLOR_MAGENTA)
        else:
            blended_color = np.clip(
                w1 * pt3d_r1.color.astype(np.float32) + w2 * pts2_colors[idx2].astype(np.float32),
                0, 255
            ).astype(np.uint8)
            output_colors.append(blended_color)
    
    merged_xyz = np.array(output_xyz, dtype=np.float64) if output_xyz else np.zeros((0, 3), dtype=np.float64)
    merged_colors = np.array(output_colors, dtype=np.uint8) if output_colors else np.zeros((0, 3), dtype=np.uint8)
    
    stats['r1_kept'] = len(keep_r1)
    stats['r2_kept'] = len(keep_r2)
    stats['blended'] = len(blended_points)
    stats['total_points'] = len(merged_xyz)
    
    if verbose:
        print(f"\n  Points kept:")
        print(f"    R1: {stats['r1_kept']}, R2: {stats['r2_kept']}, Blended: {stats['blended']}")
        print(f"    Total: {stats['total_points']}")
    
    return merged_xyz, merged_colors, stats


def save_ply(
    filepath: Union[str, Path],
    xyz: np.ndarray,
    colors: np.ndarray
) -> None:
    """
    保存点云为 PLY 文件
    
    Args:
        filepath: 输出文件路径
        xyz: (N, 3) 点云坐标
        colors: (N, 3) RGB 颜色 (uint8)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    n_points = len(xyz)
    
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(n_points):
            x, y, z = xyz[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def save_ply_binary(
    filepath: Union[str, Path],
    xyz: np.ndarray,
    colors: np.ndarray,
    chunk_size: int = 100000
) -> None:
    """
    保存点云为二进制 PLY 文件（内存优化版）
    
    Args:
        filepath: 输出文件路径
        xyz: (N, 3) 点云坐标
        colors: (N, 3) RGB 颜色 (uint8)
        chunk_size: 分块写入的大小（减少内存峰值）
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    n_points = len(xyz)
    
    # 写入头部
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        
        # 分块写入以减少内存峰值
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            
            # 只转换当前块的数据
            chunk_xyz = xyz[start:end].astype(np.float32, copy=False)
            chunk_colors = colors[start:end].astype(np.uint8, copy=False)
            
            # 使用结构化数组一次性写入（更快）
            chunk_data = np.empty(end - start, dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('r', np.uint8), ('g', np.uint8), ('b', np.uint8)
            ])
            chunk_data['x'] = chunk_xyz[:, 0]
            chunk_data['y'] = chunk_xyz[:, 1]
            chunk_data['z'] = chunk_xyz[:, 2]
            chunk_data['r'] = chunk_colors[:, 0]
            chunk_data['g'] = chunk_colors[:, 1]
            chunk_data['b'] = chunk_colors[:, 2]
            
            f.write(chunk_data.tobytes())
            
            # 释放当前块
            del chunk_data, chunk_xyz, chunk_colors


def align_reconstruction_and_extract_points(
    prev_recon: pycolmap.Reconstruction,
    curr_recon: pycolmap.Reconstruction,
    inlier_threshold: float = 10,
    min_inliers: int = 5,
    min_sample_size: int = 3,
    ransac_iterations: int = 1000,
    rotation_mode: str = 'full',
    verbose: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    对齐 curr_recon 到 prev_recon 并提取变换后的点云（不合并）
    
    用于增量点云合并场景：计算对齐变换，返回变换后的 curr_recon 点云。
    调用者需要自行处理与累积点云的合并和去重。
    
    Args:
        prev_recon: 基准 reconstruction（不会被修改）
        curr_recon: 要对齐的 reconstruction
        inlier_threshold: RANSAC 内点阈值（米）
        min_inliers: RANSAC 最小内点数
        min_sample_size: RANSAC 采样点数
        ransac_iterations: RANSAC 迭代次数
        rotation_mode: 旋转模式 ('full', 'yaw_roll', 'yaw_pitch', 'yaw', 'none')
        verbose: 是否打印信息
        
    Returns:
        aligned_xyz: (N, 3) 变换后的 curr_recon 点云坐标（失败返回 None）
        aligned_colors: (N, 3) curr_recon 点云颜色（失败返回 None）
        info: 对齐信息字典，包含：
            - success: 是否成功
            - scale: 缩放因子
            - R: 旋转矩阵 (3,3)
            - t: 平移向量 (3,)
            - num_inliers: RANSAC 内点数
            - num_common_images: 共同影像数
    """
    info = {'success': False}
    
    if verbose:
        print(f"\n  Aligning reconstruction: {len(curr_recon.images)} images, {len(curr_recon.points3D)} points")
    
    # 1. 找到共同影像
    common_images = find_common_images(prev_recon, curr_recon)
    info['num_common_images'] = len(common_images)
    
    if len(common_images) == 0:
        if verbose:
            print("    No common images found!")
        return None, None, info
    
    # 2. 建立对应关系
    corr_r1, corr_r2, pmap_r1, pmap_r2 = build_correspondences_parallel(
        prev_recon, curr_recon, common_images,
        include_track_pixels=False,
        verbose=False
    )
    
    # 3. 找到对应点对
    pts1, pts2, match_info = find_corresponding_3d_points(
        pmap_r1, pmap_r2, common_images, corr_r1, corr_r2,
        verbose=False
    )
    
    info['num_point_pairs'] = len(pts1)
    
    if len(pts1) < 3:
        if verbose:
            print(f"    Not enough corresponding points ({len(pts1)})!")
        return None, None, info
    
    # 4. RANSAC 估计 Sim3
    R, t, scale, inlier_mask = estimate_sim3_ransac(
        pts2, pts1,  # src -> tgt
        max_iterations=ransac_iterations,
        inlier_threshold=inlier_threshold,
        min_inliers=min_inliers,
        min_sample_size=min_sample_size,
        verbose=False
    )
    
    info['num_inliers'] = int(np.sum(inlier_mask))
    info['scale'] = float(scale)
    
    if np.sum(inlier_mask) < 3:
        if verbose:
            print("    Too few inliers!")
        return None, None, info
    
    # 5. 计算最终旋转矩阵（根据 rotation_mode）
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    if rotation_mode == 'full':
        R_final = R.copy()
    elif rotation_mode == 'yaw_roll':
        R_final = np.array([
            [cy,  -sy * cr,  sy * sr],
            [sy,   cy * cr, -cy * sr],
            [0,    sr,       cr     ]
        ], dtype=np.float64)
    elif rotation_mode == 'yaw_pitch':
        R_final = np.array([
            [cy * cp, -sy, cy * sp],
            [sy * cp,  cy, sy * sp],
            [-sp,      0,  cp     ]
        ], dtype=np.float64)
    elif rotation_mode == 'yaw':
        R_final = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [0,   0,  1]
        ], dtype=np.float64)
    elif rotation_mode == 'none':
        R_final = np.eye(3, dtype=np.float64)
    else:
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")
    
    # 计算平移（使用质心）
    src_centroid = pts2.mean(axis=0)
    tgt_centroid = pts1.mean(axis=0)
    t_final = tgt_centroid - scale * (R_final @ src_centroid)
    
    info['R'] = R_final
    info['t'] = t_final
    info['success'] = True
    
    # 6. 提取并变换 curr_recon 的点云
    pts_ids = np.array(list(curr_recon.points3D.keys()), dtype=np.int64)
    pts_xyz = np.array([curr_recon.points3D[pid].xyz for pid in pts_ids], dtype=np.float64)
    pts_colors = np.array([curr_recon.points3D[pid].color for pid in pts_ids], dtype=np.uint8)
    
    # 应用 Sim3 变换
    aligned_xyz = apply_sim3_to_points(pts_xyz, R_final, t_final, scale)
    
    if verbose:
        print(f"    Aligned: scale={scale:.4f}, inliers={info['num_inliers']}, points={len(aligned_xyz)}")
    
    return aligned_xyz, pts_colors, info


def merge_two_reconstructions_points_only(
    recon1: pycolmap.Reconstruction,
    recon2: pycolmap.Reconstruction,
    inlier_threshold: float = 10,
    min_inliers: int = 5,
    min_sample_size: int = 3,
    ransac_iterations: int = 1000,
    prev_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    curr_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    image_name_to_idx: Optional[Dict[str, int]] = None,
    output_path: Optional[Union[str, Path]] = None,
    match_radii: Optional[List[float]] = None,
    k_neighbors: int = 10,
    color_by_source: bool = False,
    blend_mode: str = 'select',
    blend_weight: float = 0.7,
    rotation_mode: str = 'yaw_roll',
    binary_ply: bool = True,
    verbose: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    轻量级点云合并 - 只输出三维点云
    
    相比 merge_confidence.merge_two_reconstructions，此函数：
    • 不构建完整的 Reconstruction 结构体
    • 不处理 track/影像/相机映射
    • 只输出合并后的点云 (xyz, colors)
    • 执行速度更快
    
    Args:
        recon1: 第一个 reconstruction（基准）
        recon2: 第二个 reconstruction（会被变换）
        inlier_threshold: RANSAC 内点阈值（米）
        min_inliers: RANSAC 最小内点数
        min_sample_size: RANSAC 采样点数
        ransac_iterations: RANSAC 迭代次数
        prev_recon_conf: recon1 的置信度图 {global_img_idx: (H, W) array}
        curr_recon_conf: recon2 的置信度图
        image_name_to_idx: 图像名称到索引的映射
        output_path: 可选的 PLY 输出路径
        match_radii: 2D 匹配半径列表（默认 [3, 5, 10, 20, 50]）
        k_neighbors: 近邻数量（默认 10）
        color_by_source: 是否按来源着色
        blend_mode: 'select' 或 'weighted'
        blend_weight: 混合权重（默认 0.7）
        rotation_mode: 旋转模式 ('full', 'yaw_roll', 'yaw_pitch', 'yaw', 'none')
        binary_ply: 是否保存为二进制 PLY（默认 True，更快）
        verbose: 是否打印信息
        
    Returns:
        merged_xyz: (N, 3) 合并后的点云坐标（失败返回 None）
        merged_colors: (N, 3) 合并后的颜色（失败返回 None）
        info: 合并信息字典
    """
    info = {'success': False}
    
    if match_radii is None:
        match_radii = [3, 5, 10, 20, 50]
    
    if verbose:
        print("\n" + "=" * 60)
        print("Lightweight Point Cloud Merge (Points Only)")
        print("=" * 60)
        print(f"  R1: {len(recon1.images)} images, {len(recon1.points3D)} 3D points")
        print(f"  R2: {len(recon2.images)} images, {len(recon2.points3D)} 3D points")
    
    # 1. 找到共同影像
    common_images = find_common_images(recon1, recon2)
    info['num_common_images'] = len(common_images)
    
    if len(common_images) == 0:
        if verbose:
            print("  No common images found!")
        return None, None, info
    
    if verbose:
        print(f"\n  Step 1: Found {len(common_images)} common images")
    
    # 2. 建立对应关系
    if verbose:
        print(f"  Step 2: Building correspondences...")
    corr_r1, corr_r2, pmap_r1, pmap_r2 = build_correspondences_parallel(
        recon1, recon2, common_images,
        include_track_pixels=False,
        verbose=False
    )
    
    # 3. 找到对应点对
    if verbose:
        print(f"  Step 3: Finding corresponding 3D point pairs...")
    pts1, pts2, match_info = find_corresponding_3d_points(
        pmap_r1, pmap_r2, common_images, corr_r1, corr_r2,
        verbose=False
    )
    
    info['num_point_pairs'] = len(pts1)
    
    if len(pts1) < 3:
        if verbose:
            print(f"  Not enough corresponding points ({len(pts1)})!")
        return None, None, info
    
    if verbose:
        print(f"    Found {len(pts1)} corresponding pairs")
    
    # 4. RANSAC 估计 Sim3
    if verbose:
        print(f"  Step 4: RANSAC Sim3 estimation...")
    R, t, scale, inlier_mask = estimate_sim3_ransac(
        pts2, pts1,
        max_iterations=ransac_iterations,
        inlier_threshold=inlier_threshold,
        min_inliers=min_inliers,
        min_sample_size=min_sample_size,
        verbose=False
    )
    
    info['num_inliers'] = int(np.sum(inlier_mask))
    info['scale'] = float(scale)
    
    if np.sum(inlier_mask) < 3:
        if verbose:
            print("  Too few inliers!")
        return None, None, info
    
    if verbose:
        print(f"    Inliers: {info['num_inliers']}, Scale: {scale:.6f}")
    
    # 5. 提取 recon2 点云并应用变换
    if verbose:
        print(f"  Step 5: Extracting and transforming recon2 points...")
    
    # 提取 recon2 的点云数据
    pts2_ids = np.array(list(recon2.points3D.keys()), dtype=np.int64)
    pts2_xyz = np.array([recon2.points3D[pid].xyz for pid in pts2_ids], dtype=np.float64)
    pts2_colors = np.array([recon2.points3D[pid].color for pid in pts2_ids], dtype=np.uint8)
    
    # 计算旋转矩阵
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    if rotation_mode == 'full':
        R_final = R.copy()
    elif rotation_mode == 'yaw_roll':
        R_final = np.array([
            [cy,  -sy * cr,  sy * sr],
            [sy,   cy * cr, -cy * sr],
            [0,    sr,       cr     ]
        ], dtype=np.float64)
    elif rotation_mode == 'yaw_pitch':
        R_final = np.array([
            [cy * cp, -sy, cy * sp],
            [sy * cp,  cy, sy * sp],
            [-sp,      0,  cp     ]
        ], dtype=np.float64)
    elif rotation_mode == 'yaw':
        R_final = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [0,   0,  1]
        ], dtype=np.float64)
    elif rotation_mode == 'none':
        R_final = np.eye(3, dtype=np.float64)
    else:
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")
    
    # 计算平移
    src_centroid = pts2.mean(axis=0)
    tgt_centroid = pts1.mean(axis=0)
    t_final = tgt_centroid - scale * (R_final @ src_centroid)
    
    # 变换点云
    pts2_xyz_aligned = apply_sim3_to_points(pts2_xyz, R_final, t_final, scale)
    
    if verbose:
        print(f"    Rotation mode: {rotation_mode}, Scale: {scale:.6f}")
    
    # 6. 重建 pmap_r2（使用变换后的坐标）
    if verbose:
        print(f"  Step 6: Rebuilding pixel mappings...")
    
    # 需要复制 recon2 并应用变换以获取正确的像素映射
    recon2_temp = copy.deepcopy(recon2)
    for pt3d_id in recon2_temp.points3D:
        idx = np.where(pts2_ids == pt3d_id)[0]
        if len(idx) > 0:
            recon2_temp.points3D[pt3d_id].xyz = pts2_xyz_aligned[idx[0]]
    
    common_ids_r2 = list(common_images.values())
    corr_r2_aligned = build_2d_3d_correspondences(recon2_temp, common_ids_r2, verbose=False)
    pmap_r2_aligned = build_pixel_to_3d_mapping(corr_r2_aligned)
    
    # 7. 基于置信度合并点云
    if verbose:
        mode_str = "weighted blending" if blend_mode == 'weighted' else "confidence selection"
        print(f"  Step 7: Merging with {mode_str}...")
    
    merged_xyz, merged_colors, merge_stats = merge_points_by_confidence(
        recon1,
        pts2_xyz_aligned,
        pts2_colors,
        pts2_ids,
        pmap_r1,
        pmap_r2_aligned,
        common_images,
        prev_recon_conf=prev_recon_conf,
        curr_recon_conf=curr_recon_conf,
        image_name_to_idx=image_name_to_idx,
        match_radii=match_radii,
        k_neighbors=k_neighbors,
        color_by_source=color_by_source,
        blend_mode=blend_mode,
        blend_weight=blend_weight,
        verbose=verbose,
    )
    
    info.update(merge_stats)
    info['success'] = True
    
    # 保存 PLY（可选）
    if output_path is not None:
        if verbose:
            print(f"\n  Saving PLY to: {output_path}")
        if binary_ply:
            save_ply_binary(output_path, merged_xyz, merged_colors)
        else:
            save_ply(output_path, merged_xyz, merged_colors)
    
    if verbose:
        print(f"\n" + "=" * 60)
        print(f"  Merge completed!")
        print(f"    Input:  R1={len(recon1.points3D)} + R2={len(recon2.points3D)} = {len(recon1.points3D) + len(recon2.points3D)} points")
        print(f"    Output: {len(merged_xyz)} points")
        print("=" * 60)
    
    return merged_xyz, merged_colors, info


def merge_all_reconstructions_points_only(
    reconstructions: List[pycolmap.Reconstruction],
    conf_maps_list: Optional[List[Dict[int, np.ndarray]]] = None,
    image_name_to_idx: Optional[Dict[str, int]] = None,
    output_path: Optional[Union[str, Path]] = None,
    output_intermediate_dir: Optional[Union[str, Path]] = None,
    inlier_threshold: float = 10,
    min_inliers: int = 5,
    min_sample_size: int = 3,
    ransac_iterations: int = 1000,
    match_radii: Optional[List[float]] = None,
    k_neighbors: int = 10,
    dedup_threshold: float = 0.5,
    rotation_mode: str = 'full',
    binary_ply: bool = True,
    verbose: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    批量对齐并合并多个 reconstruction 的点云（points_only 模式）- 内存优化版
    
    核心流程：
    1. 第一阶段：顺序计算相邻对的对应关系并立即计算变换（不存储中间数据）
    2. 第二阶段：收集所有点云并应用变换（不复制整个 reconstruction）
       - 每合并一个 reconstruction 后输出中间结果
    3. 第三阶段：分批去重以减少内存峰值
    
    内存优化策略：
    - 不使用 copy.deepcopy() 复制整个 reconstruction
    - 只存储变换参数 (R, t, scale)，在提取点时即时应用
    - 处理完每对后立即清理中间数据
    - 分批进行去重以减少内存峰值
    
    Args:
        reconstructions: 所有 reconstruction 对象列表
        conf_maps_list: 每个 reconstruction 的置信度图列表
        image_name_to_idx: 图像名称到全局索引的映射
        output_path: 可选的最终 PLY 输出路径
        output_intermediate_dir: 可选的中间结果输出目录，若提供则每合并一组输出一个 PLY
        inlier_threshold: RANSAC 内点阈值（米）
        min_inliers: RANSAC 最小内点数
        min_sample_size: RANSAC 采样点数
        ransac_iterations: RANSAC 迭代次数
        match_radii: 2D 匹配半径列表
        k_neighbors: 近邻数量
        dedup_threshold: 去重距离阈值（米）
        rotation_mode: 旋转模式
        binary_ply: 是否保存为二进制 PLY
        verbose: 是否打印信息
        
    Returns:
        merged_xyz: (N, 3) 合并后的点云坐标
        merged_colors: (N, 3) 合并后的颜色
        info: 合并信息字典
    """
    import gc
    
    info = {
        'success': False,
        'num_reconstructions': len(reconstructions),
        'pairwise_alignments': [],
    }
    
    if len(reconstructions) == 0:
        if verbose:
            print("  No reconstructions to merge!")
        return None, None, info
    
    if match_radii is None:
        match_radii = [3, 5, 10, 20, 50]
    
    if verbose:
        print("\n" + "=" * 70)
        print("Batch Point Cloud Merge (Single-Pass Pipeline)")
        print("=" * 70)
        print(f"  Total reconstructions: {len(reconstructions)}")
        for i, recon in enumerate(reconstructions):
            print(f"    R{i}: {len(recon.images)} images, {len(recon.points3D)} 3D points")
    
    # 创建中间结果输出目录
    intermediate_dir = None
    if output_intermediate_dir is not None:
        intermediate_dir = Path(output_intermediate_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  Intermediate results will be saved to: {intermediate_dir}")
    
    # ========== 单遍流水线处理：对齐一个，提取一个，输出一个 ==========
    # 内存优化：每处理完一个 reconstruction，立即释放其相关数据
    if verbose:
        print(f"\n{'='*50}")
        print("Single-Pass: Align -> Extract -> Output")
        print(f"{'='*50}")
    
    # 使用列表累积点云（避免预分配时需要遍历所有 reconstruction）
    all_xyz_list = []
    all_colors_list = []
    total_points = 0
    
    # 存储前一个 reconstruction 的对应点信息（用于下一次对齐）
    # 只保留最小必要数据，而不是整个 reconstruction
    prev_aligned_corr_pts = None  # 对齐后的对应点坐标
    
    for i, recon in enumerate(reconstructions):
        if verbose:
            print(f"\n  Processing R{i}:")
        
        # ========== Step 1: 计算当前 reconstruction 的变换 ==========
        if i == 0:
            # 第一个 reconstruction：恒等变换
            R_curr, t_curr, s_curr = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0
            if verbose:
                print(f"    First reconstruction (identity transform)")
        else:
            # 后续 reconstruction：计算与前一个的变换
            recon_prev = reconstructions[i - 1]
            
            # 找到共同影像
            common_images = find_common_images(recon_prev, recon)
            
            if len(common_images) == 0:
                if verbose:
                    print(f"    No common images with R{i-1}!")
                # 使用恒等变换
                R_curr, t_curr, s_curr = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0
                info['pairwise_alignments'].append({
                    'pair': (i - 1, i),
                    'success': False,
                    'inliers': 0,
                })
            else:
                if verbose:
                    print(f"    Common images with R{i-1}: {len(common_images)}")
                
                # 建立对应关系
                corr_prev, corr_curr, pmap_prev, pmap_curr = build_correspondences_parallel(
                    recon_prev, recon, common_images,
                    include_track_pixels=False,
                    verbose=False
                )
                
                # 找到对应点对
                pts_prev, pts_curr, match_info = find_corresponding_3d_points(
                    pmap_prev, pmap_curr, common_images, corr_prev, corr_curr,
                    verbose=False
                )
                
                # 清理中间数据
                del corr_prev, corr_curr, pmap_prev, pmap_curr, match_info
                
                if verbose:
                    print(f"    Corresponding 3D point pairs: {len(pts_prev)}")
                
                if len(pts_prev) < 3:
                    if verbose:
                        print(f"    Not enough point pairs!")
                    R_curr, t_curr, s_curr = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0
                    info['pairwise_alignments'].append({
                        'pair': (i - 1, i),
                        'success': False,
                        'inliers': 0,
                    })
                    del pts_prev, pts_curr
                else:
                    # 如果前一个 reconstruction 已经被变换，需要将 pts_prev 也变换
                    # 使用已存储的对齐后对应点（如果有）
                    if prev_aligned_corr_pts is not None and len(prev_aligned_corr_pts) > 0:
                        # 使用已对齐的前一个对应点
                        pts_prev_aligned = prev_aligned_corr_pts
                    else:
                        pts_prev_aligned = pts_prev
                    
                    # RANSAC 估计 Sim3
                    R, t, scale, inlier_mask = estimate_sim3_ransac(
                        pts_curr, pts_prev_aligned,
                        max_iterations=ransac_iterations,
                        inlier_threshold=inlier_threshold,
                        min_inliers=min_inliers,
                        min_sample_size=min_sample_size,
                        verbose=False
                    )
                    
                    num_inliers = int(np.sum(inlier_mask))
                    del inlier_mask
                    
                    if num_inliers < 3:
                        if verbose:
                            print(f"    Too few inliers ({num_inliers})!")
                        R_curr, t_curr, s_curr = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0
                        info['pairwise_alignments'].append({
                            'pair': (i - 1, i),
                            'success': False,
                            'inliers': num_inliers,
                        })
                    else:
                        # 根据 rotation_mode 计算最终旋转矩阵
                        yaw = np.arctan2(R[1, 0], R[0, 0])
                        pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
                        roll = np.arctan2(R[2, 1], R[2, 2])
                        
                        cy, sy = np.cos(yaw), np.sin(yaw)
                        cp, sp = np.cos(pitch), np.sin(pitch)
                        cr, sr = np.cos(roll), np.sin(roll)
                        
                        if rotation_mode == 'full':
                            R_local = R.copy()
                        elif rotation_mode == 'yaw_roll':
                            R_local = np.array([
                                [cy,  -sy * cr,  sy * sr],
                                [sy,   cy * cr, -cy * sr],
                                [0,    sr,       cr     ]
                            ], dtype=np.float64)
                        elif rotation_mode == 'yaw_pitch':
                            R_local = np.array([
                                [cy * cp, -sy, cy * sp],
                                [sy * cp,  cy, sy * sp],
                                [-sp,      0,  cp     ]
                            ], dtype=np.float64)
                        elif rotation_mode == 'yaw':
                            R_local = np.array([
                                [cy, -sy, 0],
                                [sy,  cy, 0],
                                [0,   0,  1]
                            ], dtype=np.float64)
                        elif rotation_mode == 'none':
                            R_local = np.eye(3, dtype=np.float64)
                        else:
                            raise ValueError(f"Unknown rotation_mode: {rotation_mode}")
                        
                        # 计算平移
                        src_centroid = pts_curr.mean(axis=0)
                        tgt_centroid = pts_prev_aligned.mean(axis=0)
                        t_local = tgt_centroid - scale * (R_local @ src_centroid)
                        
                        R_curr, t_curr, s_curr = R_local, t_local, scale
                        
                        if verbose:
                            print(f"    Scale: {scale:.6f}, Inliers: {num_inliers}")
                        
                        info['pairwise_alignments'].append({
                            'pair': (i - 1, i),
                            'success': True,
                            'inliers': num_inliers,
                            'scale': float(scale),
                        })
                    
                    del pts_prev, pts_curr, R, t
            
            # 释放前一个 reconstruction 的引用（帮助 GC）
            del recon_prev
        
        # ========== Step 2: 提取并变换当前 reconstruction 的点云 ==========
        num_pts = len(recon.points3D)
        
        if num_pts == 0:
            if verbose:
                print(f"    0 points (skipped)")
            # 保存当前累积结果
            if intermediate_dir is not None and total_points > 0:
                intermediate_path = intermediate_dir / f"merged_{i + 1}.ply"
                cumulative_xyz = np.vstack(all_xyz_list) if all_xyz_list else np.zeros((0, 3))
                cumulative_colors = np.vstack(all_colors_list) if all_colors_list else np.zeros((0, 3), dtype=np.uint8)
                if binary_ply:
                    save_ply_binary(intermediate_path, cumulative_xyz, cumulative_colors)
                else:
                    save_ply(intermediate_path, cumulative_xyz, cumulative_colors)
                del cumulative_xyz, cumulative_colors
            continue
        
        # 提取点云
        pts_xyz = np.empty((num_pts, 3), dtype=np.float64)
        pts_colors = np.empty((num_pts, 3), dtype=np.uint8)
        
        for j, (pt3d_id, pt3d) in enumerate(recon.points3D.items()):
            pts_xyz[j] = pt3d.xyz
            pts_colors[j] = pt3d.color
        
        # 应用变换（第一个是恒等变换）
        if i > 0 and (np.abs(s_curr - 1.0) > 1e-6 or not np.allclose(R_curr, np.eye(3))):
            pts_xyz = apply_sim3_to_points(pts_xyz, R_curr, t_curr, s_curr)
        
        # 存储对齐后的对应点（用于下一次对齐）
        # 这里我们需要重新计算对应点的对齐后坐标，用于下一个 reconstruction 的对齐
        if i < len(reconstructions) - 1:
            # 预先计算下一对的共同影像和对应点
            next_recon = reconstructions[i + 1]
            next_common_images = find_common_images(recon, next_recon)
            if len(next_common_images) > 0:
                corr_curr, corr_next, pmap_curr, pmap_next = build_correspondences_parallel(
                    recon, next_recon, next_common_images,
                    include_track_pixels=False,
                    verbose=False
                )
                pts_this, pts_next_temp, _ = find_corresponding_3d_points(
                    pmap_curr, pmap_next, next_common_images, corr_curr, corr_next,
                    verbose=False
                )
                # 将当前的对应点变换到全局坐标系
                if i > 0 and (np.abs(s_curr - 1.0) > 1e-6 or not np.allclose(R_curr, np.eye(3))):
                    prev_aligned_corr_pts = apply_sim3_to_points(pts_this, R_curr, t_curr, s_curr)
                else:
                    prev_aligned_corr_pts = pts_this.copy()
                del corr_curr, corr_next, pmap_curr, pmap_next, pts_this, pts_next_temp
            else:
                prev_aligned_corr_pts = None
        
        # 添加到列表
        all_xyz_list.append(pts_xyz)
        all_colors_list.append(pts_colors)
        total_points += num_pts
        
        if verbose:
            print(f"    {num_pts} points extracted and transformed")
        
        # ========== Step 3: 保存中间结果 ==========
        if intermediate_dir is not None:
            intermediate_path = intermediate_dir / f"merged_{i + 1}.ply"
            # 合并当前所有点云
            cumulative_xyz = np.vstack(all_xyz_list)
            cumulative_colors = np.vstack(all_colors_list)
            
            if binary_ply:
                save_ply_binary(intermediate_path, cumulative_xyz, cumulative_colors)
            else:
                save_ply(intermediate_path, cumulative_xyz, cumulative_colors)
            
            if verbose:
                print(f"    ✓ Intermediate saved: {intermediate_path} ({total_points} points)")
            
            del cumulative_xyz, cumulative_colors
        
        # 强制垃圾回收
        gc.collect()
    
    # ========== 合并最终点云 ==========
    if total_points == 0:
        if verbose:
            print("\n  No points to merge!")
        return None, None, info
    
    all_xyz = np.vstack(all_xyz_list)
    all_colors = np.vstack(all_colors_list)
    
    # 释放列表
    del all_xyz_list, all_colors_list
    gc.collect()
    
    total_before_dedup = total_points
    
    if verbose:
        print(f"\n  Total points before deduplication: {total_before_dedup}")
    
    # ========== 第三阶段：内存优化的去重 ==========
    
    if dedup_threshold > 0 and total_points > 1:
        if verbose:
            print(f"  Deduplicating with threshold: {dedup_threshold}m...")
        
        # 对于大数据集，使用分块去重策略
        if total_points > 500000:
            # 大数据集：使用体素下采样近似去重（更快、更省内存）
            if verbose:
                print(f"    Using voxel-based deduplication for large dataset...")
            
            merged_xyz, merged_colors = _voxel_dedup(
                all_xyz, all_colors, dedup_threshold, verbose
            )
        else:
            # 中小数据集：使用并查集精确去重
            merged_xyz, merged_colors = _union_find_dedup(
                all_xyz, all_colors, dedup_threshold, verbose
            )
        
        # 立即释放原始数组
        del all_xyz, all_colors
        gc.collect()
        
        if verbose:
            print(f"  Points after deduplication: {len(merged_xyz)} (removed {total_before_dedup - len(merged_xyz)})")
    else:
        # 不需要去重时，直接使用原数组（已经是正确大小）
        merged_xyz = all_xyz
        merged_colors = all_colors
    
    info['success'] = True
    info['total_points_before_dedup'] = total_before_dedup
    info['total_points_after_dedup'] = len(merged_xyz)
    
    # 保存 PLY（可选）
    if output_path is not None:
        if verbose:
            print(f"\n  Saving PLY to: {output_path}")
        if binary_ply:
            save_ply_binary(output_path, merged_xyz, merged_colors)
        else:
            save_ply(output_path, merged_xyz, merged_colors)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Batch merge completed!")
        print(f"    Input reconstructions: {len(reconstructions)}")
        print(f"    Total input points: {sum(len(r.points3D) for r in reconstructions)}")
        print(f"    Output points: {len(merged_xyz)}")
        print(f"{'='*70}")
    
    return merged_xyz, merged_colors, info


def _union_find_dedup(
    all_xyz: np.ndarray,
    all_colors: np.ndarray,
    threshold: float,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用并查集进行精确去重（适合中小数据集）
    
    内存优化：
    - 分批查询近邻避免一次性生成所有 pairs
    - 使用 numpy 数组代替 set
    """
    import gc
    
    n = len(all_xyz)
    parent = np.arange(n, dtype=np.int32)
    
    # 路径压缩的并查集
    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        # 路径压缩
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root
    
    # 分批处理以减少内存峰值
    batch_size = 50000  # 每批处理的点数
    tree = cKDTree(all_xyz)
    
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_xyz = all_xyz[batch_start:batch_end]
        
        # 查询当前批次的近邻
        distances, indices = tree.query(batch_xyz, k=min(10, n), distance_upper_bound=threshold)
        
        # 处理匹配
        for local_i in range(len(batch_xyz)):
            global_i = batch_start + local_i
            for k_idx in range(len(indices[local_i])):
                j = indices[local_i][k_idx]
                if j >= n or j == global_i:  # 无效索引或自身
                    continue
                if distances[local_i][k_idx] > threshold:
                    break  # 后面的更远
                
                pi, pj = find(global_i), find(j)
                if pi != pj:
                    parent[pi] = pj
        
        # 每批后清理
        del distances, indices, batch_xyz
        gc.collect()
    
    del tree
    gc.collect()
    
    # 找到所有唯一的根（使用 numpy 向量化）
    # 先对所有点执行 find 以确保路径完全压缩
    for i in range(n):
        find(i)
    
    # 找到保留的点（每个连通分量选择一个代表）
    unique_roots, first_indices = np.unique(parent, return_index=True)
    keep_indices = np.sort(first_indices)
    
    merged_xyz = all_xyz[keep_indices].copy()  # copy 以允许原数组释放
    merged_colors = all_colors[keep_indices].copy()
    
    del parent, unique_roots, first_indices, keep_indices
    gc.collect()
    
    return merged_xyz, merged_colors


if __name__ == "__main__":
    print("=" * 70)
    print("Lightweight Point Cloud Merge Module")
    print("=" * 70)
    print("\n📍 Only outputs merged 3D point cloud (no Reconstruction structure)")
    print("\n✨ Advantages over merge_confidence.py:")
    print("  • No track building")
    print("  • No image/camera mapping")
    print("  • No point2D updates")
    print("  • Faster execution")
    print("\n📖 Usage:")
    print("  from merge.merge_points_only import merge_two_reconstructions_points_only")
    print("  ")
    print("  # Basic usage (two reconstructions)")
    print("  xyz, colors, info = merge_two_reconstructions_points_only(")
    print("      recon1, recon2,")
    print("      output_path='merged.ply',")
    print("  )")
    print("  ")
    print("  # Batch merge (all reconstructions at once)")
    print("  from merge.merge_points_only import merge_all_reconstructions_points_only")
    print("  xyz, colors, info = merge_all_reconstructions_points_only(")
    print("      [recon1, recon2, recon3, ...],")
    print("      output_path='merged.ply',")
    print("  )")
    print("\n📦 Output:")
    print("  • xyz: numpy array (N, 3) - point coordinates")
    print("  • colors: numpy array (N, 3) - RGB colors (uint8)")
    print("  • info: dict - merge statistics")
    print("=" * 70)

