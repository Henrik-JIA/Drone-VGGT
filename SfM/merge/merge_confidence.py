#!/usr/bin/env python3
"""
简单的基于置信度的重建合并模块

策略：基于2D像素匹配 + 置信度选择

三类点的处理方式：
  • recon1 非重叠区的点 → 直接保留
  • 重叠区匹配的点 → 基于置信度选择（选择置信度更高的）
  • recon2 非重叠区的点 → 直接保留

合并流程：
  1. 找到共同影像（重叠影像）
  2. 建立 2D-3D 对应关系
  3. 找到对应的 3D 点对
  4. RANSAC 估计 Sim3 变换
  5. 应用变换对齐 recon2
  6. 基于2D像素匹配找到对应点对
  7. 重叠区基于置信度选择，非重叠区全部保留
  8. 合并影像、相机和 3D 点

高级功能（边缘平滑、密度均衡等）请使用 merge_confidence_blend.py
"""

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pycolmap
from scipy.spatial import cKDTree

# 从 merge_confidence_blend 导入基础工具函数
from merge.merge_confidence_blend import (
    find_common_images,
    build_correspondences_parallel,
    build_2d_3d_correspondences,
    build_pixel_to_3d_mapping,
    find_corresponding_3d_points,
    estimate_sim3_ransac,
    apply_sim3_to_reconstruction,
)


# 缓存 pycolmap 类引用（模块级别，避免重复查找）
_Point2D = pycolmap.Point2D
_Camera = pycolmap.Camera
_Image = pycolmap.Image


def _add_cameras_and_images_batch(
    merged: pycolmap.Reconstruction,
    recon: pycolmap.Reconstruction,
    img_ids_to_add: List[int],
    camera_id_map: Dict[int, int],
    image_name_to_new_id: Dict[str, int],
    image_id_map: Dict[int, int],
    merged_img_pt2d_len: Dict[int, int],
    new_cam_id: int,
    new_img_id: int
) -> Tuple[int, int]:
    """
    批量添加多个影像及其相机到合并的 Reconstruction 中（优化版）。
    
    Returns:
        更新后的 (new_cam_id, new_img_id)
    """
    # 缓存本地引用
    recon_cameras = recon.cameras
    recon_images = recon.images
    add_camera = merged.add_camera
    add_image = merged.add_image
    
    for img_id in img_ids_to_add:
        img = recon_images[img_id]
        old_cam_id = img.camera_id
        
        # 添加相机（如果尚未添加）
        if old_cam_id not in camera_id_map:
            cam = recon_cameras[old_cam_id]
            add_camera(_Camera(
                camera_id=new_cam_id,
                model=cam.model,
                width=cam.width,
                height=cam.height,
                params=cam.params
            ))
            camera_id_map[old_cam_id] = new_cam_id
            new_cam_id += 1
        
        # 添加影像（直接使用列表推导式，避免函数调用开销）
        points2D = img.points2D
        new_points2D = [_Point2D(pt.xy) for pt in points2D]
        
        add_image(_Image(
            image_id=new_img_id,
            name=img.name,
            camera_id=camera_id_map[old_cam_id],
            cam_from_world=img.cam_from_world,
            points2D=new_points2D
        ))
        
        image_name_to_new_id[img.name] = new_img_id
        image_id_map[img_id] = new_img_id
        merged_img_pt2d_len[new_img_id] = len(points2D)
        new_img_id += 1
    
    return new_cam_id, new_img_id


def _add_camera_and_image(
    merged: pycolmap.Reconstruction,
    recon: pycolmap.Reconstruction,
    img_id: int,
    img: 'pycolmap.Image',
    camera_id_map: Dict[int, int],
    image_name_to_new_id: Dict[str, int],
    image_id_map: Dict[int, int],
    new_cam_id: int,
    new_img_id: int
) -> Tuple[int, int]:
    """
    添加单个影像及其相机到合并的 Reconstruction 中。
    保留此函数以兼容其他调用。
    """
    old_cam_id = img.camera_id
    
    # 添加相机（如果尚未添加）
    if old_cam_id not in camera_id_map:
        cam = recon.cameras[old_cam_id]
        merged.add_camera(_Camera(
            camera_id=new_cam_id,
            model=cam.model,
            width=cam.width,
            height=cam.height,
            params=cam.params
        ))
        camera_id_map[old_cam_id] = new_cam_id
        new_cam_id += 1
    
    # 添加影像
    new_points2D = [_Point2D(pt.xy) for pt in img.points2D]
    merged.add_image(_Image(
        image_id=new_img_id,
        name=img.name,
        camera_id=camera_id_map[old_cam_id],
        cam_from_world=img.cam_from_world,
        points2D=new_points2D
    ))
    
    image_name_to_new_id[img.name] = new_img_id
    image_id_map[img_id] = new_img_id
    new_img_id += 1
    
    return new_cam_id, new_img_id


def _get_confidence_score(
    info: Dict, 
    img_id: int, 
    pixel_key: Tuple[int, int], 
    conf_cache: Dict
) -> float:
    """
    获取置信度分数（越大越好）
    
    Args:
        info: 点信息字典，包含 'error' 字段
        img_id: 图像 ID
        pixel_key: 像素坐标 (x, y)
        conf_cache: 置信度缓存 {img_id: (conf_map, H, W) or None}
        
    Returns:
        置信度分数
    """
    cached = conf_cache.get(img_id)
    if cached is not None:
        conf_map, H, W = cached
        px, py = pixel_key
        px_c = min(max(int(px), 0), W - 1)
        py_c = min(max(int(py), 0), H - 1)
        return float(conf_map[py_c, px_c])
    # 如果没有置信度图，使用负的重投影误差（误差越小，分数越高）
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
    处理单对影像的匹配（用于并行化），支持多级半径匹配（优化版）
    
    使用多级半径策略：从小到大依次使用不同的匹配半径，
    已匹配的点不再参与后续匹配，从而实现更全面、更精确的覆盖。
    
    优化策略：
    - 一次性展平所有 (pixel1_idx, neighbor_idx, distance) 组合
    - 全局按距离排序，只排序一次
    - 使用 NumPy 向量化操作加速
    
    Args:
        img_id1: recon1 的图像 ID
        img_id2: recon2 的图像 ID
        pixel_map_recon1: recon1 的像素映射
        pixel_map_recon2: recon2 的像素映射
        match_radii: 匹配半径列表，从小到大排序
        k_neighbors: 每个点查询的近邻数量（默认10）
        
    Returns:
        匹配结果元组或 None
    """
    pmap1 = pixel_map_recon1.get(img_id1)
    pmap2 = pixel_map_recon2.get(img_id2)
    
    if pmap1 is None or pmap2 is None or len(pmap1) == 0 or len(pmap2) == 0:
        return None
    
    # 确保 match_radii 是排序的列表
    if isinstance(match_radii, (int, float)):
        match_radii = [match_radii]
    match_radii = sorted(match_radii)
    
    # 一次性转换为数组和列表
    pixels2_list = list(pmap2.keys())
    pixels1_list = list(pmap1.keys())
    n1 = len(pixels1_list)
    n2 = len(pixels2_list)
    
    # 预提取 point3D_id（避免重复字典查询）
    pt3d_ids1 = [pmap1[pk]['point3D_id'] for pk in pixels1_list]
    pt3d_ids2 = [pmap2[pk]['point3D_id'] for pk in pixels2_list]
    
    # 构建 KD-Tree
    tree2 = cKDTree(np.asarray(pixels2_list, dtype=np.float32))
    pixels1_arr = np.asarray(pixels1_list, dtype=np.float32)
    
    # 使用最大半径进行一次查询
    max_radius = match_radii[-1]
    k = min(k_neighbors, n2)
    all_distances, all_indices = tree2.query(
        pixels1_arr, k=k, distance_upper_bound=max_radius
    )
    
    # 确保 2D 数组
    if k == 1:
        all_distances = all_distances.reshape(-1, 1)
        all_indices = all_indices.reshape(-1, 1)
    
    # ========== 优化：一次性展平并排序所有候选 ==========
    # 创建索引网格
    i_indices, j_indices = np.meshgrid(np.arange(n1), np.arange(k), indexing='ij')
    i_flat = i_indices.ravel()
    j_flat = j_indices.ravel()
    dist_flat = all_distances.ravel()
    idx_flat = all_indices.ravel()
    
    # 筛选有效的候选（距离在最大半径内且索引有效）
    valid_mask = (dist_flat <= max_radius) & (idx_flat < n2) & np.isfinite(dist_flat)
    valid_i = i_flat[valid_mask]
    valid_j = j_flat[valid_mask]
    valid_dist = dist_flat[valid_mask]
    valid_idx = idx_flat[valid_mask]
    
    if len(valid_dist) == 0:
        return None
    
    # 全局按距离排序（只排序一次）
    sort_order = np.argsort(valid_dist)
    sorted_i = valid_i[sort_order]
    sorted_dist = valid_dist[sort_order]
    sorted_idx = valid_idx[sort_order].astype(int)
    
    # ========== 贪心匹配 ==========
    matches = []
    seen_r1 = set()
    seen_r2 = set()
    seen_pixel_idx1 = set()
    seen_pixel_idx2 = set()
    radius_stats = {r: 0 for r in match_radii}
    
    # 确定每个距离对应的半径区间
    radius_arr = np.array(match_radii)
    
    for order_idx in range(len(sorted_i)):
        i = sorted_i[order_idx]
        dist = sorted_dist[order_idx]
        idx2 = sorted_idx[order_idx]
        
        # 跳过已匹配的
        if i in seen_pixel_idx1 or idx2 in seen_pixel_idx2:
            continue
        
        pt3d_id1 = pt3d_ids1[i]
        pt3d_id2 = pt3d_ids2[idx2]
        
        if pt3d_id1 in seen_r1 or pt3d_id2 in seen_r2:
            continue
        
        # 确定这个匹配属于哪个半径区间
        radius_bin = radius_arr[np.searchsorted(radius_arr, dist, side='left')]
        if dist > radius_bin:
            # 找下一个更大的半径
            larger = radius_arr[radius_arr >= dist]
            if len(larger) == 0:
                continue
            radius_bin = larger[0]
        
        # 记录匹配
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
            pt3d_id1, pt3d_id2, dist, radius_bin
        ))
        radius_stats[radius_bin] = radius_stats.get(radius_bin, 0) + 1
    
    if len(matches) == 0:
        return None
    
    return (img_id1, img_id2, matches, pmap1, pmap2, radius_stats)


def merge_by_simple_confidence(
    recon1: pycolmap.Reconstruction,
    recon2_aligned: pycolmap.Reconstruction,
    pixel_map_recon1: Dict[int, Dict],
    pixel_map_recon2: Dict[int, Dict],
    common_images: Dict[int, int],
    prev_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    curr_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    image_name_to_idx: Optional[Dict[str, int]] = None,
    match_radii: Optional[List[float]] = None,
    k_neighbors: int = 10,
    color_by_source: bool = False,
    color_by_match_status: Optional[bool] = None,  # 兼容旧参数名
    blend_mode: str = 'select',  # 'select' 或 'weighted'
    blend_weight: float = 0.7,  # 高置信度点的权重（低置信度点权重 = 1 - blend_weight）
    verbose: bool = True,
) -> Tuple[pycolmap.Reconstruction, Dict]:
    """
    基于简单置信度选择的点云合并
    
    使用 KD-Tree 进行高效的 2D 像素匹配，基于置信度选择保留哪个点。
    
    处理逻辑：
    1. 遍历共同影像，使用多级半径的 2D 像素匹配找到对应的 3D 点对
    2. 对于匹配到的点对，基于置信度选择保留哪个点
    3. 未匹配的点全部保留
    
    Args:
        recon1: 第一个 reconstruction（基准）
        recon2_aligned: 已对齐的第二个 reconstruction
        pixel_map_recon1: recon1 的像素到 3D 点映射 {img_id: {(px,py): {'point3D_id': id, ...}}}
        pixel_map_recon2: recon2_aligned 的像素到 3D 点映射
        common_images: 共同影像映射 {img_id1: img_id2}
        prev_recon_conf: recon1 的置信度图 {global_img_idx: (H, W) array}
        curr_recon_conf: recon2 的置信度图
        image_name_to_idx: 图像名称到全局索引的映射
        match_radii: 多级 2D 像素匹配半径列表（默认 [3, 5, 10, 20, 50]）
        k_neighbors: 每个点查询的近邻数量（默认10），更大的值可以找到更多匹配
        color_by_source: 是否按来源着色（用于调试可视化）
        color_by_match_status: 同 color_by_source（兼容旧参数名）
            - True: 蓝色=R1独有, 绿色=R2独有, 红色=匹配点
            - False: 保留原始颜色
        blend_mode: 混合模式
            - 'select': 选择置信度高的点（默认）
            - 'weighted': 基于固定权重混合两个点的位置和颜色
        blend_weight: 高置信度点的权重（仅 weighted 模式，默认 0.7）
            - 高置信度点权重 = blend_weight
            - 低置信度点权重 = 1 - blend_weight
            - 例如 0.7 表示高置信度点占 70%，低置信度点占 30%
        verbose: 是否打印详细信息
        
    Returns:
        merged_recon: 合并后的 reconstruction
        info: 合并信息字典
    """
    # 兼容旧参数名 color_by_match_status
    if color_by_match_status is not None:
        color_by_source = color_by_match_status
    
    if prev_recon_conf is None:
        prev_recon_conf = {}
    if curr_recon_conf is None:
        curr_recon_conf = {}
    if image_name_to_idx is None:
        image_name_to_idx = {}
    
    # 设置默认的多级匹配半径
    if match_radii is None:
        match_radii = [3, 5, 10, 20, 50]
    elif isinstance(match_radii, (int, float)):
        match_radii = [match_radii]
    match_radii = sorted(match_radii)
    
    # ========== 预计算置信度缓存（并行） ==========
    def _build_conf_cache(recon, recon_conf, name_to_idx):
        """构建单个 reconstruction 的置信度缓存"""
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
    
    # 并行构建两个置信度缓存
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_r1 = executor.submit(_build_conf_cache, recon1, prev_recon_conf, image_name_to_idx)
        future_r2 = executor.submit(_build_conf_cache, recon2_aligned, curr_recon_conf, image_name_to_idx)
        _conf_cache_r1 = future_r1.result()
        _conf_cache_r2 = future_r2.result()
    
    # ========== 统计信息 ==========
    stats = {
        'r1_total': len(recon1.points3D),
        'r2_total': len(recon2_aligned.points3D),
        'matched_pairs': 0,
        'r1_wins': 0,
        'r2_wins': 0,
        'avg_match_distance': 0.0,
        'match_radii': match_radii,
        'matches_per_radius': {},
    }
    
    # ========== 第一步：2D 像素匹配 ==========
    # 使用 (source, pt3d_id) 作为保留点的 key
    keep_points = set()  # {('r1', pt3d_id) or ('r2', pt3d_id)}
    point_source_type = {}  # (source, pt3d_id) -> 'only' | 'conflict'
    
    # 记录被淘汰的 3D 点
    discard_r1 = set()
    discard_r2 = set()
    
    # 记录匹配距离
    match_distances = []
    
    # 记录每张影像处理过的点
    processed_r1 = set()
    processed_r2 = set()
    
    # 累计每个半径的匹配统计
    total_radius_stats = {r: 0 for r in match_radii}
    
    # 并行处理所有影像对
    image_pairs = list(common_images.items())
    
    if len(image_pairs) > 0:
        # 增加并行度
        max_workers = min(16, len(image_pairs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda p: _process_image_pair_for_merge(
                    p[0], p[1], pixel_map_recon1, pixel_map_recon2, match_radii, k_neighbors
                ), 
                image_pairs
            ))
        
        # ========== 优化：批量收集所有匹配结果 ==========
        all_matches = []  # (img_id1, img_id2, pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist, radius)
        all_pmaps_r1 = {}
        all_pmaps_r2 = {}
        
        for result in results:
            if result is None:
                continue
            
            img_id1, img_id2, matches, pmap1, pmap2, radius_stats = result
            
            # 累计半径匹配统计
            for r, count in radius_stats.items():
                if r in total_radius_stats:
                    total_radius_stats[r] += count
            
            # 收集所有匹配
            for pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist, radius in matches:
                all_matches.append((img_id1, img_id2, pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist))
            
            # 记录 pmap
            if img_id1 not in all_pmaps_r1:
                all_pmaps_r1[img_id1] = pmap1
            if img_id2 not in all_pmaps_r2:
                all_pmaps_r2[img_id2] = pmap2
        
        # ========== 批量计算置信度分数 ==========
        # 记录需要混合的点对 {blended_key: (pt3d_id1, pt3d_id2, weight1, weight2)}
        blended_points = {}
        
        if len(all_matches) > 0:
            # 预计算所有 score1, score2
            scores1 = []
            scores2 = []
            for (img_id1, img_id2, pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist) in all_matches:
                scores1.append(_get_confidence_score(info1, img_id1, pixel_key1, _conf_cache_r1))
                scores2.append(_get_confidence_score(info2, img_id2, pixel_key2, _conf_cache_r2))
            
            scores1 = np.array(scores1)
            scores2 = np.array(scores2)
            
            # 向量化比较
            r1_wins_mask = scores1 >= scores2
            
            # 统计
            stats['r1_wins'] = int(np.sum(r1_wins_mask))
            stats['r2_wins'] = len(all_matches) - stats['r1_wins']
            stats['matched_pairs'] = len(all_matches)
            stats['blended_count'] = 0
            
            # 处理每个匹配
            for idx, (img_id1, img_id2, pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist) in enumerate(all_matches):
                if blend_mode == 'weighted':
                    # 混合模式：使用固定权重混合两个点
                    # 高置信度点权重 = blend_weight，低置信度点权重 = 1 - blend_weight
                    if r1_wins_mask[idx]:
                        w1, w2 = blend_weight, 1.0 - blend_weight
                    else:
                        w1, w2 = 1.0 - blend_weight, blend_weight
                    
                    blended_key = ('blended', pt3d_id1, pt3d_id2)
                    blended_points[blended_key] = (pt3d_id1, pt3d_id2, w1, w2)
                    point_source_type[blended_key] = 'blended'
                    stats['blended_count'] += 1
                else:
                    # 选择模式：保留置信度高的点
                    if r1_wins_mask[idx]:
                        key = ('r1', pt3d_id1)
                        keep_points.add(key)
                        point_source_type[key] = 'conflict'
                        discard_r2.add(pt3d_id2)
                    else:
                        key = ('r2', pt3d_id2)
                        keep_points.add(key)
                        point_source_type[key] = 'conflict'
                        discard_r1.add(pt3d_id1)
                
                processed_r1.add(pt3d_id1)
                processed_r2.add(pt3d_id2)
                match_distances.append(dist)
        
        # 收集 pmap 中未匹配的点 - 优化：并行收集 + 集合运算
        def _collect_pt3d_ids(all_pmaps):
            """批量收集所有 point3D_id"""
            return set(info['point3D_id'] for pmap in all_pmaps.values() for info in pmap.values())
        
        # 并行收集两个 reconstruction 的所有 pt3d_id
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_r1 = executor.submit(_collect_pt3d_ids, all_pmaps_r1)
            future_r2 = executor.submit(_collect_pt3d_ids, all_pmaps_r2)
            all_pt3d_ids_r1 = future_r1.result()
            all_pt3d_ids_r2 = future_r2.result()
        
        # 使用集合差集找出未处理的点（O(n) 操作，比嵌套循环 in 检查更高效）
        unmatched_r1 = all_pt3d_ids_r1 - processed_r1
        unmatched_r2 = all_pt3d_ids_r2 - processed_r2
        
        # 批量添加 r1 未匹配的点
        for pt3d_id in unmatched_r1:
            key = ('r1', pt3d_id)
            keep_points.add(key)
            if key not in point_source_type:
                point_source_type[key] = 'only'
        processed_r1.update(unmatched_r1)
        
        # 批量添加 r2 未匹配的点
        for pt3d_id in unmatched_r2:
            key = ('r2', pt3d_id)
            keep_points.add(key)
            if key not in point_source_type:
                point_source_type[key] = 'only'
        processed_r2.update(unmatched_r2)
    
    # ========== 第二步：添加非共同影像的点（并行） ==========
    def _collect_non_common_pt3d_ids(recon, common_img_ids, already_processed):
        """收集非共同影像中未处理的 pt3d_id"""
        result = set()
        for img_id, img in recon.images.items():
            if img_id in common_img_ids:
                continue
            for pt2d in img.points2D:
                pt3d_id = pt2d.point3D_id
                if pt3d_id != -1 and pt3d_id not in already_processed:
                    result.add(pt3d_id)
        return result
    
    # 预计算共同影像 ID 集合
    common_img_ids_r1 = set(common_images.keys())
    common_img_ids_r2 = set(common_images.values())
    
    # 并行收集两个 reconstruction 的非共同影像点
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_r1 = executor.submit(_collect_non_common_pt3d_ids, recon1, common_img_ids_r1, processed_r1)
        future_r2 = executor.submit(_collect_non_common_pt3d_ids, recon2_aligned, common_img_ids_r2, processed_r2)
        non_common_r1 = future_r1.result()
        non_common_r2 = future_r2.result()
    
    # 批量添加 r1 非共同影像的点
    for pt3d_id in non_common_r1:
        key = ('r1', pt3d_id)
        keep_points.add(key)
        if key not in point_source_type:
            point_source_type[key] = 'only'
    processed_r1.update(non_common_r1)
    
    # 批量添加 r2 非共同影像的点
    for pt3d_id in non_common_r2:
        key = ('r2', pt3d_id)
        keep_points.add(key)
        if key not in point_source_type:
            point_source_type[key] = 'only'
    processed_r2.update(non_common_r2)
    
    # 计算平均匹配距离
    if len(match_distances) > 0:
        stats['avg_match_distance'] = float(np.mean(match_distances))
    stats['matches_per_radius'] = total_radius_stats
    
    if verbose:
        print(f"\n  2D matching (multi-radius: {match_radii}):")
        print(f"    Total matched pairs: {stats['matched_pairs']}")
        print(f"    Avg match distance: {stats['avg_match_distance']:.2f}px")
        for r in sorted(total_radius_stats.keys()):
            count = total_radius_stats[r]
            if count > 0:
                print(f"      radius={r}px: {count} pairs")
        print(f"    R1 wins: {stats['r1_wins']}, R2 wins: {stats['r2_wins']}")
        if blend_mode == 'weighted':
            print(f"    Blended points: {stats.get('blended_count', 0)} (high={blend_weight:.0%}, low={1-blend_weight:.0%})")
    
    # ========== 第三步：构建合并后的 Reconstruction ==========
    merged = pycolmap.Reconstruction()
    
    # 合并预计算：overlap_img_names、overlap_name_to_ids、r1/r2 统计（单次遍历 common_images + keep_points）
    overlap_img_names = set()
    overlap_name_to_ids = {}  # img_name -> (r1_id, r2_id, r1_cam_id, r2_cam_id)
    for r1_id, r2_id in common_images.items():
        r1_img = recon1.images[r1_id]
        r2_img = recon2_aligned.images[r2_id]
        overlap_img_names.add(r1_img.name)
        overlap_name_to_ids[r1_img.name] = (r1_id, r2_id, r1_img.camera_id, r2_img.camera_id)
    
    # 统计 r1/r2 保留点数（单次遍历）
    r1_kept = r2_kept = 0
    for s, _ in keep_points:
        if s == 'r1':
            r1_kept += 1
        else:
            r2_kept += 1
    use_r1_for_overlap = r1_kept >= r2_kept
    
    if verbose:
        print(f"\n  Overlap camera choice: {'R1' if use_r1_for_overlap else 'R2'} "
              f"(R1 kept: {r1_kept}, R2 kept: {r2_kept})")
    
    # 3.1 & 3.2 添加相机和影像（优化版：预筛选 + 批量处理）
    camera_id_map_r1, camera_id_map_r2 = {}, {}
    image_id_map_r1, image_id_map_r2 = {}, {}
    image_name_to_new_id = {}
    merged_img_pt2d_len = {}
    
    new_cam_id, new_img_id = 1, 1
    
    # 预筛选 recon1 需要添加的影像 ID（避免循环中重复检查）
    if use_r1_for_overlap:
        r1_img_ids_to_add = list(recon1.images.keys())
    else:
        r1_img_ids_to_add = [
            img_id for img_id, img in recon1.images.items()
            if img.name not in overlap_img_names
        ]
    
    # 批量添加 recon1 的影像
    new_cam_id, new_img_id = _add_cameras_and_images_batch(
        merged, recon1, r1_img_ids_to_add,
        camera_id_map_r1, image_name_to_new_id, image_id_map_r1,
        merged_img_pt2d_len, new_cam_id, new_img_id
    )
    
    # 处理 recon2 的影像：分离重叠和非重叠
    r2_overlap_ids = []  # 重叠影像（需要映射）
    r2_new_ids = []      # 非重叠影像（需要添加）
    
    for img_id, img in recon2_aligned.images.items():
        if img.name in image_name_to_new_id:
            r2_overlap_ids.append((img_id, img))
        else:
            r2_new_ids.append(img_id)
    
    # 处理重叠影像映射
    for img_id, img in r2_overlap_ids:
        image_id_map_r2[img_id] = image_name_to_new_id[img.name]
        _, _, r1_cam_id, _ = overlap_name_to_ids[img.name]
        camera_id_map_r2[img.camera_id] = camera_id_map_r1[r1_cam_id]
    
    # 批量添加 recon2 的非重叠影像
    new_cam_id, new_img_id = _add_cameras_and_images_batch(
        merged, recon2_aligned, r2_new_ids,
        camera_id_map_r2, image_name_to_new_id, image_id_map_r2,
        merged_img_pt2d_len, new_cam_id, new_img_id
    )
    
    # 补充 recon1 被跳过的重叠影像映射
    if not use_r1_for_overlap:
        for img_name, (r1_img_id, _, r1_cam_id, r2_cam_id) in overlap_name_to_ids.items():
            image_id_map_r1[r1_img_id] = image_name_to_new_id[img_name]
            camera_id_map_r1[r1_cam_id] = camera_id_map_r2[r2_cam_id]
    
    merged_img_ids = set(merged.images.keys())
    
    # 3.3 添加 3D 点（优化：缓存对象引用、分离 r1/r2 处理）
    point3d_id_map = {}
    COLOR_BLUE = np.array([0, 0, 255], dtype=np.uint8)
    COLOR_GREEN = np.array([0, 255, 0], dtype=np.uint8)
    COLOR_RED = np.array([255, 0, 255], dtype=np.uint8)
    COLOR_MAGENTA = np.array([255, 0, 255], dtype=np.uint8)
    
    # 缓存对象引用（避免循环内重复属性访问）
    merged_points3D = merged.points3D
    merged_images = merged.images
    recon1_points3D = recon1.points3D
    recon2_points3D = recon2_aligned.points3D
    merged_add_point3D = merged.add_point3D
    
    # 预分离 keep_points 为 r1/r2 两组（避免循环内字典查找）
    keep_r1 = [(pt3d_id, ('r1', pt3d_id)) for s, pt3d_id in keep_points if s == 'r1']
    keep_r2 = [(pt3d_id, ('r2', pt3d_id)) for s, pt3d_id in keep_points if s == 'r2']
    
    # 预计算冲突点集合（避免循环内 dict.get）
    conflict_keys = {k for k, v in point_source_type.items() if v == 'conflict'} if color_by_source else set()
    
    # 内联处理 r1 保留点
    for pt3d_id, key in keep_r1:
        if pt3d_id not in recon1_points3D:
            continue
        old_pt3d = recon1_points3D[pt3d_id]
        
        # 内联 collect_valid_tracks（无去重）
        valid_elements = []
        for te in old_pt3d.track.elements:
            new_img_id_val = image_id_map_r1.get(te.image_id)
            if new_img_id_val is not None and new_img_id_val in merged_img_ids:
                pt2d_idx = te.point2D_idx
                if pt2d_idx < merged_img_pt2d_len[new_img_id_val]:
                    valid_elements.append((new_img_id_val, pt2d_idx))
        
        if not valid_elements:
            continue
        
        point_color = (COLOR_RED if key in conflict_keys else COLOR_BLUE) if color_by_source else old_pt3d.color
        
        # 内联 add_point3d_with_track
        new_pt3d_id = merged_add_point3D(xyz=old_pt3d.xyz, track=pycolmap.Track(), color=point_color)
        pt3d_obj = merged_points3D[new_pt3d_id]
        pt3d_obj.error = old_pt3d.error
        point3d_id_map[key] = new_pt3d_id
        track_add = pt3d_obj.track.add_element
        for new_img_id_val, pt2d_idx in valid_elements:
            track_add(new_img_id_val, pt2d_idx)
            merged_images[new_img_id_val].points2D[pt2d_idx].point3D_id = new_pt3d_id
    
    # 内联处理 r2 保留点
    for pt3d_id, key in keep_r2:
        if pt3d_id not in recon2_points3D:
            continue
        old_pt3d = recon2_points3D[pt3d_id]
        
        # 内联 collect_valid_tracks（无去重）
        valid_elements = []
        for te in old_pt3d.track.elements:
            new_img_id_val = image_id_map_r2.get(te.image_id)
            if new_img_id_val is not None and new_img_id_val in merged_img_ids:
                pt2d_idx = te.point2D_idx
                if pt2d_idx < merged_img_pt2d_len[new_img_id_val]:
                    valid_elements.append((new_img_id_val, pt2d_idx))
        
        if not valid_elements:
            continue
        
        point_color = (COLOR_RED if key in conflict_keys else COLOR_GREEN) if color_by_source else old_pt3d.color
        
        # 内联 add_point3d_with_track
        new_pt3d_id = merged_add_point3D(xyz=old_pt3d.xyz, track=pycolmap.Track(), color=point_color)
        pt3d_obj = merged_points3D[new_pt3d_id]
        pt3d_obj.error = old_pt3d.error
        point3d_id_map[key] = new_pt3d_id
        track_add = pt3d_obj.track.add_element
        for new_img_id_val, pt2d_idx in valid_elements:
            track_add(new_img_id_val, pt2d_idx)
            merged_images[new_img_id_val].points2D[pt2d_idx].point3D_id = new_pt3d_id
    
    # 处理混合点（weighted 模式）- 内联优化
    for blended_key, (pt3d_id1, pt3d_id2, w1, w2) in blended_points.items():
        if pt3d_id1 not in recon1_points3D or pt3d_id2 not in recon2_points3D:
            continue
        
        pt3d_r1, pt3d_r2 = recon1_points3D[pt3d_id1], recon2_points3D[pt3d_id2]
        
        # 内联 collect_valid_tracks（带去重，优先权重高的来源）
        seen = set()
        valid_elements = []
        if w1 >= w2:
            for te in pt3d_r1.track.elements:
                new_img_id_val = image_id_map_r1.get(te.image_id)
                if new_img_id_val is not None and new_img_id_val in merged_img_ids:
                    pt2d_idx = te.point2D_idx
                    if pt2d_idx < merged_img_pt2d_len[new_img_id_val]:
                        k = (new_img_id_val, pt2d_idx)
                        if k not in seen:
                            valid_elements.append(k)
                            seen.add(k)
            for te in pt3d_r2.track.elements:
                new_img_id_val = image_id_map_r2.get(te.image_id)
                if new_img_id_val is not None and new_img_id_val in merged_img_ids:
                    pt2d_idx = te.point2D_idx
                    if pt2d_idx < merged_img_pt2d_len[new_img_id_val]:
                        k = (new_img_id_val, pt2d_idx)
                        if k not in seen:
                            valid_elements.append(k)
                            seen.add(k)
        else:
            for te in pt3d_r2.track.elements:
                new_img_id_val = image_id_map_r2.get(te.image_id)
                if new_img_id_val is not None and new_img_id_val in merged_img_ids:
                    pt2d_idx = te.point2D_idx
                    if pt2d_idx < merged_img_pt2d_len[new_img_id_val]:
                        k = (new_img_id_val, pt2d_idx)
                        if k not in seen:
                            valid_elements.append(k)
                            seen.add(k)
            for te in pt3d_r1.track.elements:
                new_img_id_val = image_id_map_r1.get(te.image_id)
                if new_img_id_val is not None and new_img_id_val in merged_img_ids:
                    pt2d_idx = te.point2D_idx
                    if pt2d_idx < merged_img_pt2d_len[new_img_id_val]:
                        k = (new_img_id_val, pt2d_idx)
                        if k not in seen:
                            valid_elements.append(k)
                            seen.add(k)
        
        if not valid_elements:
            continue
        
        # 计算混合属性
        blended_xyz = w1 * pt3d_r1.xyz + w2 * pt3d_r2.xyz
        blended_error = w1 * pt3d_r1.error + w2 * pt3d_r2.error
        if color_by_source:
            blended_color = COLOR_MAGENTA
        else:
            blended_color = np.clip(
                w1 * pt3d_r1.color.astype(np.float32) + w2 * pt3d_r2.color.astype(np.float32),
                0, 255
            ).astype(np.uint8)
        
        # 内联 add_point3d_with_track
        new_pt3d_id = merged_add_point3D(xyz=blended_xyz, track=pycolmap.Track(), color=blended_color)
        pt3d_obj = merged_points3D[new_pt3d_id]
        pt3d_obj.error = blended_error
        point3d_id_map[blended_key] = new_pt3d_id
        track_add = pt3d_obj.track.add_element
        for new_img_id_val, pt2d_idx in valid_elements:
            track_add(new_img_id_val, pt2d_idx)
            merged_images[new_img_id_val].points2D[pt2d_idx].point3D_id = new_pt3d_id
    
    # ========== 统计信息（优化：单次遍历）==========
    r1_only_count = 0
    r2_only_count = 0
    for key, t in point_source_type.items():
        if t == 'only':
            source = key[0]  # 第一个元素是来源标签
            if source == 'r1':
                r1_only_count += 1
            elif source == 'r2':
                r2_only_count += 1
    stats['r1_only'] = r1_only_count
    stats['r2_only'] = r2_only_count
    stats['total_points'] = len(merged.points3D)
    stats['total_images'] = len(merged.images)
    stats['total_cameras'] = len(merged.cameras)
    
    if verbose:
        print(f"\n  Points kept:")
        print(f"    R1 only (unmatched): {stats['r1_only']}")
        print(f"    R2 only (unmatched): {stats['r2_only']}")
        print(f"\n  Final merged reconstruction:")
        print(f"    Cameras: {stats['total_cameras']}")
        print(f"    Images: {stats['total_images']}")
        print(f"    3D Points: {stats['total_points']}")
    
    return merged, stats


def merge_two_reconstructions(
    recon1: pycolmap.Reconstruction,
    recon2: pycolmap.Reconstruction,
    inlier_threshold: float = 10,
    min_inliers: int = 5,
    min_sample_size: int = 3,
    ransac_iterations: int = 1000,
    prev_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    curr_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    image_name_to_idx: Optional[Dict[str, int]] = None,
    output_dir: Optional[Path] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    match_radii: Optional[List[float]] = None,
    k_neighbors: int = 10,
    color_by_source: bool = False,
    color_by_match_status: Optional[bool] = None,  # 兼容旧参数名
    blend_mode: str = 'select',
    blend_weight: float = 0.7,
    rotation_mode: str = 'yaw_roll',  # 旋转模式
    verbose: bool = True,
    **kwargs,  # 忽略其他高级参数（为了接口兼容）
) -> Tuple[Optional[pycolmap.Reconstruction], Dict]:
    """
    简单的基于置信度的两个 reconstruction 合并流程
    
    合并步骤：
    1. 找到共同影像（重叠影像）
    2. 建立 2D-3D 对应关系
    3. 找到对应的 3D 点对
    4. RANSAC 估计 Sim3 变换
    5. 应用变换对齐 recon2
    6. 基于 2D 像素匹配找到对应点对
    7. 重叠区基于置信度选择或加权混合，未匹配点全部保留
    8. 合并影像、相机和 3D 点
    
    如需高级功能（多级匹配、边缘平滑、密度均衡等），
    请使用 merge_confidence_blend.merge_two_reconstructions
    
    Args:
        recon1: 第一个 reconstruction（基准，不修改）
        recon2: 第二个 reconstruction（会被变换）
        inlier_threshold: RANSAC 内点阈值（单位：米）
        min_inliers: RANSAC 最小内点数
        min_sample_size: RANSAC 每次迭代采样的点数
        ransac_iterations: RANSAC 迭代次数
        prev_recon_conf: recon1 的像素级置信度图 {global_img_idx: (H, W) array}
        curr_recon_conf: recon2 的像素级置信度图 {global_img_idx: (H, W) array}
        image_name_to_idx: 图像名称到全局索引的映射
        output_dir: 可选的输出目录（用于保存中间结果）
        start_idx: 起始图像索引（用于输出目录命名）
        end_idx: 结束图像索引（用于输出目录命名）
        match_radii: 多级 2D 像素匹配半径列表（默认 [3, 5, 10, 20, 50]）
        k_neighbors: 每个点查询的近邻数量（默认10），更大的值可以找到更多匹配
        color_by_source: 是否按来源着色（用于调试可视化，默认 False）
        color_by_match_status: 同 color_by_source（兼容旧参数名）
            启用时的着色规则：
            - 🔴 红色 (255,0,0): 匹配后选择的点
            - 🔵 蓝色 (0,0,255): R1 独有的点
            - 🟢 绿色 (0,255,0): R2 独有的点
            - 🟣 洋红 (255,0,255): 加权混合的点
        blend_mode: 混合模式（默认 'select'）
            - 'select': 选择置信度高的点
            - 'weighted': 基于固定权重混合两个点的位置和颜色
        blend_weight: 高置信度点的权重（仅 weighted 模式，默认 0.7）
            - 高置信度点权重 = blend_weight（如 70%）
            - 低置信度点权重 = 1 - blend_weight（如 30%）
        rotation_mode: 旋转模式（默认 'yaw_roll'），控制对齐时使用哪些旋转分量
            - 'full': 使用完整旋转（yaw + pitch + roll）
            - 'yaw_roll': 只使用 yaw + roll，不做俯仰（适合无人机俯拍）
            - 'yaw_pitch': 只使用 yaw + pitch，不做横滚
            - 'yaw': 只使用 yaw（水平旋转）
            - 'none': 不旋转（只缩放和平移）
        verbose: 是否打印详细信息
        **kwargs: 忽略其他高级参数（为了与 confidence_blend 接口兼容）
        
    Returns:
        merged_recon: 合并后的 reconstruction（失败返回 None）
        info: 合并信息字典
    """
    info = {'success': False}
    
    # 兼容旧参数名 color_by_match_status
    if color_by_match_status is not None:
        color_by_source = color_by_match_status
    
    # 默认匹配半径
    if match_radii is None:
        match_radii = [3, 5, 10, 20, 50]
    
    if verbose:
        print("\n" + "=" * 60)
        print("Simple Confidence-based Reconstruction Merge")
        print("=" * 60)
        print(f"  R1: {len(recon1.images)} images, {len(recon1.points3D)} 3D points")
        print(f"  R2: {len(recon2.images)} images, {len(recon2.points3D)} 3D points")
    
    # 1. 找到共同影像
    common_images = find_common_images(recon1, recon2)
    info['num_common_images'] = len(common_images)
    
    if len(common_images) == 0:
        if verbose:
            print("  No common images found!")
        return None, info
    
    if verbose:
        print(f"\n  Step 1: Found {len(common_images)} common (overlap) images")
    
    # 2. 并行建立 2D-3D 对应关系和像素映射
    if verbose:
        print(f"  Step 2: Building 2D-3D correspondences...")
    corr_r1, corr_r2, pmap_r1, pmap_r2 = build_correspondences_parallel(
        recon1, recon2, common_images,
        include_track_pixels=False,
        verbose=False
    )
    
    # 3. 找到对应的 3D 点对
    if verbose:
        print(f"  Step 3: Finding corresponding 3D point pairs...")
    pts1, pts2, match_info = find_corresponding_3d_points(
        pmap_r1, pmap_r2, common_images, corr_r1, corr_r2,
        verbose=False
    )
    
    info['num_point_pairs'] = len(pts1)
    
    if len(pts1) < 3:
        if verbose:
            print(f"  Not enough corresponding points ({len(pts1)}) for alignment!")
        return None, info
    
    if verbose:
        print(f"    Found {len(pts1)} corresponding 3D point pairs")
    
    # 4. RANSAC 估计 Sim3 变换 (从 recon2 到 recon1)
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
    info['translation'] = t.tolist()
    
    if np.sum(inlier_mask) < 3:
        if verbose:
            print("  Too few inliers after RANSAC!")
        return None, info
    
    if verbose:
        print(f"    Inliers: {info['num_inliers']}, Scale: {scale:.6f}")
    
    # 5. 复制 recon2 并应用变换
    if verbose:
        print(f"  Step 5: Applying rotation ({rotation_mode}) + scale + translation to recon2...")
    recon2_aligned = copy.deepcopy(recon2)
    
    # 从完整旋转矩阵中提取欧拉角（ZYX 顺序）
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))  # clip 防止数值误差
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    # 根据 rotation_mode 构建旋转矩阵
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    if rotation_mode == 'full':
        # 使用完整旋转
        R_final = R.copy()
        rot_info = f"Yaw: {np.degrees(yaw):.2f}°, Pitch: {np.degrees(pitch):.2f}°, Roll: {np.degrees(roll):.2f}°"
    elif rotation_mode == 'yaw_roll':
        # Rz(yaw) @ Rx(roll)，pitch = 0
        R_final = np.array([
            [cy,  -sy * cr,  sy * sr],
            [sy,   cy * cr, -cy * sr],
            [0,    sr,       cr     ]
        ], dtype=np.float64)
        rot_info = f"Yaw: {np.degrees(yaw):.2f}°, Roll: {np.degrees(roll):.2f}°"
    elif rotation_mode == 'yaw_pitch':
        # Rz(yaw) @ Ry(pitch)，roll = 0
        R_final = np.array([
            [cy * cp, -sy, cy * sp],
            [sy * cp,  cy, sy * sp],
            [-sp,      0,  cp     ]
        ], dtype=np.float64)
        rot_info = f"Yaw: {np.degrees(yaw):.2f}°, Pitch: {np.degrees(pitch):.2f}°"
    elif rotation_mode == 'yaw':
        # Rz(yaw) only
        R_final = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [0,   0,  1]
        ], dtype=np.float64)
        rot_info = f"Yaw: {np.degrees(yaw):.2f}°"
    elif rotation_mode == 'none':
        # 不旋转
        R_final = np.eye(3, dtype=np.float64)
        rot_info = "No rotation"
    else:
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}. "
                        f"Choose from: 'full', 'yaw_roll', 'yaw_pitch', 'yaw', 'none'")
    
    # 重新计算平移：基于选定的旋转和质心对齐
    src_centroid = pts2.mean(axis=0)
    tgt_centroid = pts1.mean(axis=0)
    translation_final = tgt_centroid - scale * (R_final @ src_centroid)
    
    apply_sim3_to_reconstruction(recon2_aligned, R_final, translation_final, scale)
    
    if verbose:
        print(f"    {rot_info}, Scale: {scale:.6f}")
    
    # 输出保存变换后的 recon2（可选）
    if output_dir is not None:
        if start_idx is not None and end_idx is not None:
            subdir_name = f"{start_idx}_{end_idx}"
        else:
            subdir_name = f"common_{len(common_images)}"
        temp_path = Path(output_dir) / "temp_aligned_recon1" / subdir_name
        temp_path.mkdir(parents=True, exist_ok=True)
        recon2_aligned.write_text(str(temp_path))
        recon2_aligned.export_PLY(str(temp_path / "points3D.ply"))
        if verbose:
            print(f"    Saved aligned recon2 to: {temp_path}")
    
    # 6. 重新计算对齐后 recon2 的像素映射（只需共同影像）
    if verbose:
        print(f"  Step 6: Rebuilding pixel mappings for aligned recon2...")
    
    # recon1 未变化，直接复用 step 2 的 pmap_r1
    # 只需为对齐后的 recon2 的共同影像重建映射
    common_ids_r2 = list(common_images.values())
    corr_r2_aligned = build_2d_3d_correspondences(recon2_aligned, common_ids_r2, verbose=False)
    pmap_r2_aligned = build_pixel_to_3d_mapping(corr_r2_aligned)
    
    # 7. 基于简单置信度合并
    if verbose:
        mode_str = "weighted blending" if blend_mode == 'weighted' else "confidence selection"
        print(f"  Step 7: Merging based on {mode_str}...")
        print(f"    Match radii: {match_radii}")
        if blend_mode == 'weighted':
            print(f"    Blend weight: high={blend_weight:.0%}, low={1-blend_weight:.0%}")
        if color_by_source:
            print(f"    🎨 Debug coloring enabled:")
            print(f"       🔴 Red = matched/conflict points (selected)")
            print(f"       🔵 Blue = R1 only (unmatched)")
            print(f"       🟢 Green = R2 only (unmatched)")
            if blend_mode == 'weighted':
                print(f"       🟣 Magenta = blended points")
    
    merged_recon, merge_info = merge_by_simple_confidence(
        recon1,
        recon2_aligned,
        pmap_r1,           # 复用 step 2 的结果
        pmap_r2_aligned,   # 对齐后重建的映射
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
    
    info.update(merge_info)
    info['success'] = True
    
    if verbose:
        print(f"\n" + "=" * 60)
        print(f"  Merge completed!")
        print(f"    Input:  R1={len(recon1.points3D)} + R2={len(recon2.points3D)} = {len(recon1.points3D) + len(recon2.points3D)} points")
        print(f"    Output: {len(merged_recon.points3D)} points")
        print(f"    Reduction: {len(recon1.points3D) + len(recon2.points3D) - len(merged_recon.points3D)} points removed")
        print("=" * 60)
    
    return merged_recon, info


if __name__ == "__main__":
    print("=" * 70)
    print("Simple Confidence-based Reconstruction Merge Module")
    print("=" * 70)
    print("\n📍 Strategy: 2D Pixel Matching + Confidence Selection/Blending")
    print("\n✨ Features:")
    print("  • Multi-level radius matching using KD-Tree")
    print("  • Confidence-based point selection in overlap regions")
    print("  • Weighted blending mode for smoother transitions")
    print("  • All unmatched points preserved")
    print("\n📖 Usage:")
    print("  from merge_confidence import merge_two_reconstructions")
    print("  ")
    print("  # Mode 1: Selection (default) - pick higher confidence point")
    print("  merged, info = merge_two_reconstructions(")
    print("      recon1, recon2,")
    print("      prev_recon_conf=conf1,")
    print("      curr_recon_conf=conf2,")
    print("      image_name_to_idx=name_to_idx,")
    print("  )")
    print("  ")
    print("  # Mode 2: Weighted blending - blend points with fixed weight")
    print("  merged, info = merge_two_reconstructions(")
    print("      recon1, recon2,")
    print("      prev_recon_conf=conf1,")
    print("      curr_recon_conf=conf2,")
    print("      image_name_to_idx=name_to_idx,")
    print("      blend_mode='weighted',    # Enable weighted blending")
    print("      blend_weight=0.7,         # High conf=70%, Low conf=30%")
    print("  )")
    print("\n🎨 Debug Mode:")
    print("  # Enable color-coding for testing/visualization")
    print("  merged, info = merge_two_reconstructions(..., color_by_source=True)")
    print("  # Color coding:")
    print("  #   🔴 Red (255,0,0)     = Matched/conflict points (selected)")
    print("  #   🔵 Blue (0,0,255)    = R1 only (unmatched)")
    print("  #   🟢 Green (0,255,0)   = R2 only (unmatched)")
    print("  #   🟣 Magenta (255,0,255) = Blended points (weighted mode)")
    print("\n⚡ For more advanced features:")
    print("  Use merge_confidence_blend.py for:")
    print("  • Edge blending & smooth interpolation")
    print("  • Density equalization")
    print("=" * 70)

