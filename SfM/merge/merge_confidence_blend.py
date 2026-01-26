"""
重建合并工具 - 简化版

用于加载和分析两个 pycolmap Reconstruction，找到共同影像。
"""

import numpy as np
import pycolmap
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree

# 从体素降采样模块导入
# 兼容包导入和直接运行两种方式
try:
    from utils.voxel_downsample import voxel_downsample
except ImportError:
    try:
        from ..utils.voxel_downsample import voxel_downsample
    except ImportError:
        from point_cloud_utils import voxel_downsample


def _signed_distance_to_hull(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    """
    计算点到凸包边界的有符号距离（批量版本）
    
    正值 = 在凸包外，负值 = 在凸包内
    
    Args:
        points: 待检查的点 (N, 2)
        hull_points: 凸包顶点 (M, 2)
        
    Returns:
        signed_distances: 有符号距离数组 (N,)
    """
    from scipy.spatial import ConvexHull
    
    if len(hull_points) < 3:
        return np.zeros(len(points))
    
    try:
        hull = ConvexHull(hull_points)
        # 凸包的方程：A @ x + b <= 0 对于内部点
        # equations 是 (N_facets, D+1)，最后一列是 b
        equations = hull.equations
        
        # 计算点到每个面的有符号距离
        # distance = A @ point + b
        # 对于每个点，取最大距离（正值表示在外面，负值表示在里面）
        distances = points @ equations[:, :-1].T + equations[:, -1]
        signed_dists = np.max(distances, axis=1)
        
        return signed_dists
    except:
        return np.zeros(len(points))


def _smootherstep_vectorized(t: np.ndarray) -> np.ndarray:
    """向量化的 smootherstep 函数：6t^5 - 15t^4 + 10t^3"""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _compute_blend_factors(points: np.ndarray,
                           inner_hull_points: Optional[np.ndarray],
                           outer_hull_points: Optional[np.ndarray],
                           matched_arr: Optional[np.ndarray],
                           inner_shrink: float,
                           outer_expand: float,
                           use_smooth_curve: bool = True) -> np.ndarray:
    """
    基于到内外边界的有符号距离，计算 0~1 的融合因子（向量化版本）
    
    融合因子含义：
    - 0 = 完全在核心区边界（应该丢弃或低权重）
    - 1 = 完全在非重叠区边界（应该保留，高权重）
    - 0~1 之间 = 在融合带内，权重渐变
    
    改进：使用 smootherstep 曲线替代线性插值，实现更平滑的过渡
    【性能优化】完全向量化，消除逐点循环
    
    Args:
        points: 待检查的点 (N, 2)
        inner_hull_points: 内边界凸包顶点（收缩后）
        outer_hull_points: 外边界凸包顶点（扩展后）
        matched_arr: 匹配点数组（用于距离备用方案）
        inner_shrink: 内边界收缩量
        outer_expand: 外边界扩展量
        use_smooth_curve: 是否使用平滑曲线（默认 True）
        
    Returns:
        blend_factors: 融合因子数组 (N,)，范围 [0, 1]
    """
    n = len(points)
    if n == 0:
        return np.array([])
    
    blend_factors = np.ones(n, dtype=np.float64)  # 默认为1（保留）
    
    if inner_hull_points is not None and outer_hull_points is not None:
        # 双边界模式（最精确）- 向量化计算
        dist_inner = _signed_distance_to_hull(points, inner_hull_points)
        dist_outer = _signed_distance_to_hull(points, outer_hull_points)
        
        # 在内边界内（核心区）→ 0
        inner_mask = dist_inner <= 0
        blend_factors[inner_mask] = 0.0
        
        # 在外边界外（非重叠区）→ 1
        outer_mask = dist_outer >= 0
        blend_factors[outer_mask] = 1.0
        
        # 在融合带内 → 插值
        blend_mask = ~inner_mask & ~outer_mask
        if np.any(blend_mask):
            total = dist_inner[blend_mask] + np.abs(dist_outer[blend_mask])
            # 避免除零
            valid = total >= 1e-6
            linear_t = np.where(valid, dist_inner[blend_mask] / np.maximum(total, 1e-6), 0.5)
            
            if use_smooth_curve:
                blend_factors[blend_mask] = _smootherstep_vectorized(linear_t)
            else:
                blend_factors[blend_mask] = linear_t
                    
    elif matched_arr is not None and len(matched_arr) > 0:
        # 使用距离到匹配点的方案（备用）- 向量化计算
        tree = cKDTree(matched_arr)
        distances, _ = tree.query(points, k=1)
        distances = np.asarray(distances)
        
        total_width = inner_shrink + outer_expand
        inner_threshold = inner_shrink * 0.3
        
        # 在核心区 → 0
        inner_mask = distances < inner_threshold
        blend_factors[inner_mask] = 0.0
        
        # 在非重叠区 → 1
        outer_mask = distances > total_width
        blend_factors[outer_mask] = 1.0
        
        # 在融合带内 → 插值
        blend_mask = ~inner_mask & ~outer_mask
        if np.any(blend_mask):
            denom = total_width - inner_threshold
            linear_t = np.clip((distances[blend_mask] - inner_threshold) / denom, 0.0, 1.0)
            
            if use_smooth_curve:
                blend_factors[blend_mask] = _smootherstep_vectorized(linear_t)
            else:
                blend_factors[blend_mask] = linear_t
    
    return blend_factors


def _get_confidence_score(info: Dict, img_id: int, pixel_key: Tuple[int, int], 
                         conf_cache: Dict) -> float:
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
    return -info.get('error', 1.0)


def _confidence_to_weight(conf: float, min_weight: float = 0.01) -> float:
    """
    将置信度分数转换为正权重（用于加权平均）
    
    置信度分数可能为负（当使用重投影误差时），需要转换为正权重。
    使用 softplus 函数: weight = log(1 + exp(conf)) 确保权重为正
    
    Args:
        conf: 置信度分数（可为负）
        min_weight: 最小权重，防止权重为0
        
    Returns:
        正权重值
    """
    # 使用 softplus 确保权重为正且平滑
    # softplus(x) = log(1 + exp(x))
    if conf > 20:  # 防止数值溢出
        weight = conf
    elif conf < -20:
        weight = min_weight
    else:
        weight = np.log1p(np.exp(conf))
    return max(weight, min_weight)


def _blend_3d_points(xyz1: np.ndarray, xyz2: np.ndarray, 
                     conf1: float, conf2: float,
                     color1: np.ndarray, color2: np.ndarray,
                     error1: float, error2: float,
                     dominant_weight: float = 0.8) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    基于置信度加权融合两个3D点
    
    Args:
        xyz1, xyz2: 两个点的3D坐标
        conf1, conf2: 两个点的置信度分数
        color1, color2: 两个点的颜色
        error1, error2: 两个点的重投影误差
        dominant_weight: 高置信度点的权重比例，默认0.8（即80%），
                        低置信度点占 1-dominant_weight（即20%）
                        设为 None 时使用原始置信度加权
        
    Returns:
        blended_xyz: 融合后的3D坐标
        blended_color: 融合后的颜色
        blended_error: 融合后的误差（加权平均）
    """
    if dominant_weight is not None:
        # 使用固定权重分配：高置信度点占 dominant_weight
        if conf1 >= conf2:
            w1_norm = dominant_weight
            w2_norm = 1.0 - dominant_weight
        else:
            w1_norm = 1.0 - dominant_weight
            w2_norm = dominant_weight
    else:
        # 使用原始置信度加权
        w1 = _confidence_to_weight(conf1)
        w2 = _confidence_to_weight(conf2)
        total_weight = w1 + w2
        w1_norm = w1 / total_weight
        w2_norm = w2 / total_weight
    
    # 加权平均3D坐标
    blended_xyz = w1_norm * np.asarray(xyz1) + w2_norm * np.asarray(xyz2)
    
    # 加权平均颜色
    blended_color = (w1_norm * np.asarray(color1, dtype=np.float32) + 
                     w2_norm * np.asarray(color2, dtype=np.float32))
    blended_color = np.clip(blended_color, 0, 255).astype(np.uint8)
    
    # 加权平均误差
    blended_error = w1_norm * error1 + w2_norm * error2
    
    return blended_xyz, blended_color, blended_error


def _process_image_pair_for_merge(
    img_id1: int, 
    img_id2: int,
    pixel_map_recon1: Dict,
    pixel_map_recon2: Dict,
    match_radii: List[float]
) -> Optional[Tuple]:
    """
    处理单对影像的匹配（用于并行化），支持多级半径匹配
    
    使用多级半径策略：从小到大依次使用不同的匹配半径，
    已匹配的点不再参与后续匹配，从而实现更全面、更精确的覆盖。
    
    Args:
        img_id1: recon1 的图像 ID
        img_id2: recon2 的图像 ID
        pixel_map_recon1: recon1 的像素映射
        pixel_map_recon2: recon2 的像素映射
        match_radii: 匹配半径列表，从小到大排序
            - 小半径优先匹配高置信度的精确对应
            - 大半径补充匹配边缘和难以对齐的点
        
    Returns:
        匹配结果元组或 None
        (img_id1, img_id2, matches, matched_pixels_r1, matched_pixels_r2, pmap1, pmap2, radius_stats)
    """
    pmap1 = pixel_map_recon1.get(img_id1)
    pmap2 = pixel_map_recon2.get(img_id2)
    
    if pmap1 is None or pmap2 is None or len(pmap1) == 0 or len(pmap2) == 0:
        return None
    
    # 确保 match_radii 是排序的列表
    if isinstance(match_radii, (int, float)):
        match_radii = [match_radii]
    match_radii = sorted(match_radii)
    
    # 一次性转换为数组
    pixels2_list = list(pmap2.keys())
    pixels1_list = list(pmap1.keys())
    n1 = len(pixels1_list)
    n2 = len(pixels2_list)
    
    # 构建 KD-Tree（只需构建一次）
    tree2 = cKDTree(np.asarray(pixels2_list, dtype=np.float32))
    pixels1_arr = np.asarray(pixels1_list, dtype=np.float32)
    
    # 收集所有匹配结果
    matches = []
    matched_pixels_r1 = []
    matched_pixels_r2 = []
    seen_r1 = set()  # 已匹配的 recon1 3D 点 ID
    seen_r2 = set()  # 已匹配的 recon2 3D 点 ID
    seen_pixel_r1 = set()  # 已匹配的 recon1 像素
    seen_pixel_r2 = set()  # 已匹配的 recon2 像素
    
    # 记录每个半径的匹配统计
    radius_stats = {}
    
    # 使用最大半径进行一次查询，获取所有距离
    max_radius = match_radii[-1]
    all_distances, all_indices = tree2.query(
        pixels1_arr, 
        k=1, 
        distance_upper_bound=max_radius
    )
    
    # 多级半径匹配：从小到大依次处理
    for radius in match_radii:
        radius_matches = 0
        
        # 筛选在当前半径内的有效匹配
        valid_mask = (all_distances <= radius) & (all_indices < n2)
        valid_indices = np.where(valid_mask)[0]
        
        # 按距离排序，优先处理距离小的匹配（更可靠）
        sorted_valid = sorted(valid_indices, key=lambda i: all_distances[i])
        
        for i in sorted_valid:
            idx = all_indices[i]
            pixel_key1 = pixels1_list[i]
            pixel_key2 = pixels2_list[idx]
            
            # 检查像素是否已被匹配
            if pixel_key1 in seen_pixel_r1 or pixel_key2 in seen_pixel_r2:
                continue
            
            info1 = pmap1[pixel_key1]
            info2 = pmap2[pixel_key2]
            pt3d_id1 = info1['point3D_id']
            pt3d_id2 = info2['point3D_id']
            
            # 检查 3D 点是否已被匹配
            if pt3d_id1 in seen_r1 or pt3d_id2 in seen_r2:
                continue
            
            # 记录匹配
            seen_r1.add(pt3d_id1)
            seen_r2.add(pt3d_id2)
            seen_pixel_r1.add(pixel_key1)
            seen_pixel_r2.add(pixel_key2)
            
            matches.append((
                pixel_key1, pixel_key2, info1, info2, 
                pt3d_id1, pt3d_id2, all_distances[i], radius  # 添加 radius 信息
            ))
            matched_pixels_r1.append(pixel_key1)
            matched_pixels_r2.append(pixel_key2)
            radius_matches += 1
        
        radius_stats[radius] = radius_matches
    
    if len(matches) == 0:
        return None
    
    return (img_id1, img_id2, matches, matched_pixels_r1, matched_pixels_r2, pmap1, pmap2, radius_stats)


def _build_dual_boundary_delaunay(matched_pixels: List, 
                                   inner_shrink: float, 
                                   outer_expand: float) -> Tuple:
    """
    预计算双边界Delaunay三角剖分（用于平滑融合）
    
    三区域策略：
    1. 核心区（inner_delaunay 内）：丢弃未匹配点
    2. 融合带（inner_delaunay 外，outer_delaunay 内）：保留未匹配点（带插值权重）
    3. 非重叠区（outer_delaunay 外）：保留未匹配点
    
    Args:
        matched_pixels: 匹配成功的像素坐标列表
        inner_shrink: 内边界收缩量（像素），正值表示向内收缩
        outer_expand: 外边界扩展量（像素），正值表示向外扩展
    
    Returns:
        inner_delaunay: 内边界Delaunay对象（收缩后的凸包）
        outer_delaunay: 外边界Delaunay对象（扩展后的凸包）
        matched_arr: 匹配点数组（用于KD-Tree距离计算）
        inner_hull_points: 内边界凸包顶点（用于距离计算）
        outer_hull_points: 外边界凸包顶点（用于距离计算）
    """
    from scipy.spatial import ConvexHull, Delaunay
    
    if len(matched_pixels) == 0:
        return None, None, None, None, None
    
    matched_arr = np.array(matched_pixels, dtype=np.float64)
    
    if len(matched_pixels) < 3:
        return None, None, matched_arr, None, None
    
    try:
        hull = ConvexHull(matched_arr)
        hull_points = matched_arr[hull.vertices]
        center = hull_points.mean(axis=0)
        
        # 计算从中心到凸包顶点的方向向量（归一化）
        directions = hull_points - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # 避免除零
        unit_directions = directions / norms
        
        # 构建内边界（收缩）- 只有当收缩后凸包仍然有效时才使用
        inner_delaunay = None
        inner_hull_points = None
        if inner_shrink > 0:
            # 检查收缩后是否仍然有效（点不会越过中心）
            min_norm = norms.min()
            if inner_shrink < min_norm * 0.9:  # 留10%余量
                inner_hull_points = hull_points - inner_shrink * unit_directions
                try:
                    inner_delaunay = Delaunay(inner_hull_points)
                except:
                    inner_delaunay = None
                    inner_hull_points = None
        
        # 构建外边界（扩展）
        outer_hull_points = hull_points + outer_expand * unit_directions
        try:
            outer_delaunay = Delaunay(outer_hull_points)
        except:
            outer_delaunay = None
            outer_hull_points = None
        
        return inner_delaunay, outer_delaunay, matched_arr, inner_hull_points, outer_hull_points
    except:
        return None, None, matched_arr, None, None


def _classify_points_dual_boundary(points_to_check: List, 
                                    inner_delaunay, 
                                    outer_delaunay,
                                    matched_arr: Optional[np.ndarray],
                                    inner_hull_points: Optional[np.ndarray],
                                    outer_hull_points: Optional[np.ndarray],
                                    inner_shrink: float,
                                    outer_expand: float,
                                    fallback_distance: float = 50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用双边界对点进行分类（用于平滑融合），并计算插值权重
    
    分类逻辑：
    1. 在内边界内 → 丢弃（核心重叠区）
    2. 在外边界外 → 保留（非重叠区），权重=1
    3. 在内外边界之间 → 保留（融合带），权重按距离插值
    
    Args:
        points_to_check: list of (pixel_key, pt3d_id, info)
        inner_delaunay: 内边界Delaunay对象
        outer_delaunay: 外边界Delaunay对象
        matched_arr: 匹配点数组（用于距离计算备用方案）
        inner_hull_points: 内边界凸包顶点（用于精确距离计算）
        outer_hull_points: 外边界凸包顶点（用于精确距离计算）
        inner_shrink: 内边界收缩量
        outer_expand: 外边界扩展量
        fallback_distance: 当无法构建Delaunay时的备用距离阈值
        
    Returns:
        discard_mask: np.array of bool，True表示应该丢弃
        blend_mask: np.array of bool，True表示在融合带内
        blend_factors: np.array of float，融合权重 [0, 1]
    """
    n = len(points_to_check)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=np.float32)
    
    # 提取所有像素坐标
    pixels = np.array([p[0] for p in points_to_check], dtype=np.float64)
    
    # 初始化：默认都保留，权重为1
    discard_mask = np.zeros(n, dtype=bool)
    blend_mask = np.zeros(n, dtype=bool)
    blend_factors = np.ones(n, dtype=np.float32)
    
    if inner_delaunay is not None and outer_delaunay is not None:
        # 双边界模式（最佳情况）
        inside_inner = inner_delaunay.find_simplex(pixels) >= 0
        inside_outer = outer_delaunay.find_simplex(pixels) >= 0
        
        # 在内边界内 → 丢弃
        discard_mask = inside_inner
        # 在内边界外但在外边界内 → 融合带（保留）
        blend_mask = ~inside_inner & inside_outer
        
        # 计算融合带内点的插值权重
        if inner_hull_points is not None and outer_hull_points is not None:
            blend_factors = _compute_blend_factors(
                pixels, inner_hull_points, outer_hull_points,
                matched_arr, inner_shrink, outer_expand
            )
        
    elif outer_delaunay is not None:
        # 只有外边界（内边界构建失败，可能区域太小）
        # 使用距离匹配点的方式来判断核心区
        inside_outer = outer_delaunay.find_simplex(pixels) >= 0
        
        if matched_arr is not None and len(matched_arr) > 0:
            # 使用KD-Tree计算到最近匹配点的距离
            tree = cKDTree(matched_arr)
            distances, _ = tree.query(pixels, k=1)
            
            # 距离很近的点视为核心区（应该丢弃）
            # 距离较远但仍在外边界内的视为融合带
            close_to_matched = distances < fallback_distance * 0.5
            discard_mask = inside_outer & close_to_matched
            blend_mask = inside_outer & ~close_to_matched
            
            # 计算融合权重（基于距离）
            total_width = inner_shrink + outer_expand
            for i in range(n):
                if discard_mask[i]:
                    blend_factors[i] = 0.0
                elif blend_mask[i]:
                    # 线性插值
                    blend_factors[i] = min(max((distances[i] - fallback_distance * 0.3) / (fallback_distance * 0.7), 0), 1)
        else:
            # 没有匹配点信息，保守处理：都保留
            blend_mask = inside_outer
            
    elif matched_arr is not None and len(matched_arr) > 0:
        # 无法构建Delaunay，使用纯距离方案
        tree = cKDTree(matched_arr)
        distances, _ = tree.query(pixels, k=1)
        
        # 距离很近 → 丢弃
        # 距离中等 → 融合带
        # 距离很远 → 保留（非重叠区）
        discard_mask = distances < fallback_distance * 0.3
        blend_mask = (distances >= fallback_distance * 0.3) & (distances < fallback_distance)
        
        # 计算融合权重
        for i in range(n):
            if discard_mask[i]:
                blend_factors[i] = 0.0
            elif blend_mask[i]:
                blend_factors[i] = (distances[i] - fallback_distance * 0.3) / (fallback_distance * 0.7)
    
    # 其他情况：全部保留（discard_mask 和 blend_mask 都是 False，blend_factors = 1）
    
    return discard_mask, blend_mask, blend_factors


def _smoothstep(x: float) -> float:
    """
    Smoothstep 函数，实现平滑的 0-1 过渡
    
    公式: 3x^2 - 2x^3，在 x=0 和 x=1 处导数为 0
    """
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _smootherstep(x: float) -> float:
    """
    Smootherstep 函数，更平滑的过渡（二阶导数也为 0）
    
    公式: 6x^5 - 15x^4 + 10x^3
    """
    x = max(0.0, min(1.0, x))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _feather_blend_zone_boundary(
    keep_points: set,
    all_blend_weights: Dict,
    interpolated_xyz: Dict,
    blended_points: Dict,
    recon1: 'pycolmap.Reconstruction',
    recon2: 'pycolmap.Reconstruction',
    k_neighbors: int = 8,
    feather_strength: float = 0.3,
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    对融合带边界进行羽化处理，使过渡更加平滑（向量化版本）
    
    【性能优化】批量 KD-Tree 查询和向量化计算
    
    Args:
        keep_points: 保留的点集合
        all_blend_weights: 融合权重
        interpolated_xyz: 已插值的点坐标 {(source, pt3d_id): xyz}
        blended_points: 已融合点信息
        recon1, recon2: 两个 reconstruction
        k_neighbors: 用于平滑的近邻数
        feather_strength: 羽化强度 (0-1)，越大平滑效果越强
        
    Returns:
        feathered_xyz: 羽化后的坐标 {(source, pt3d_id): xyz}
    """
    if len(interpolated_xyz) == 0 or feather_strength <= 0:
        return interpolated_xyz
    
    # 收集所有点的坐标（用于构建 KD-Tree）
    all_xyz = []
    all_keys = []
    
    for key in keep_points:
        source, pt3d_id = key
        
        if key in interpolated_xyz:
            xyz = interpolated_xyz[key]
        elif key in blended_points:
            xyz = np.asarray(blended_points[key]['xyz'])
        elif source == 'r1' and pt3d_id in recon1.points3D:
            xyz = np.asarray(recon1.points3D[pt3d_id].xyz)
        elif source == 'r2' and pt3d_id in recon2.points3D:
            xyz = np.asarray(recon2.points3D[pt3d_id].xyz)
        else:
            continue
        
        all_xyz.append(xyz)
        all_keys.append(key)
    
    if len(all_xyz) < k_neighbors + 1:
        return interpolated_xyz
    
    all_xyz = np.array(all_xyz)
    tree = cKDTree(all_xyz)
    key_to_idx = {key: i for i, key in enumerate(all_keys)}
    
    # 收集需要羽化的点
    feather_keys = []
    feather_orig_xyz = []
    feather_blend_weights = []
    feather_indices = []
    
    for key, orig_xyz in interpolated_xyz.items():
        blend_weight = all_blend_weights.get(key, 1.0)
        idx = key_to_idx.get(key)
        
        if idx is not None and 0.1 <= blend_weight <= 0.95:
            feather_keys.append(key)
            feather_orig_xyz.append(orig_xyz)
            feather_blend_weights.append(blend_weight)
            feather_indices.append(idx)
    
    # 初始化结果（先复制不需要羽化的点）
    feathered_xyz = {}
    for key, orig_xyz in interpolated_xyz.items():
        blend_weight = all_blend_weights.get(key, 1.0)
        if blend_weight < 0.1 or blend_weight > 0.95:
            feathered_xyz[key] = orig_xyz
    
    if len(feather_keys) == 0:
        return interpolated_xyz
    
    # 转换为数组
    feather_orig_xyz = np.array(feather_orig_xyz)
    feather_blend_weights = np.array(feather_blend_weights)
    feather_indices = np.array(feather_indices)
    
    # 批量 KD-Tree 查询
    k = min(k_neighbors, len(all_xyz) - 1)
    all_distances, all_neighbor_indices = tree.query(feather_orig_xyz, k=k + 1)
    
    # 向量化计算
    eps = 1e-10
    
    for i, key in enumerate(feather_keys):
        idx = feather_indices[i]
        distances = all_distances[i]
        indices = all_neighbor_indices[i]
        
        # 排除自己
        neighbor_mask = indices != idx
        if not neighbor_mask.any():
            feathered_xyz[key] = feather_orig_xyz[i]
            continue
        
        neighbor_indices = indices[neighbor_mask][:k]
        neighbor_dists = distances[neighbor_mask][:k]
        
        if len(neighbor_indices) == 0:
            feathered_xyz[key] = feather_orig_xyz[i]
            continue
        
        # 计算权重
        weights = 1.0 / (neighbor_dists + eps)
        weights /= weights.sum()
        
        # 加权平均
        neighbor_xyz = all_xyz[neighbor_indices]
        avg_neighbor_xyz = np.sum(neighbor_xyz * weights[:, np.newaxis], axis=0)
        
        # 计算羽化强度
        blend_weight = feather_blend_weights[i]
        edge_factor = 1.0 - abs(2.0 * blend_weight - 1.0)
        actual_strength = feather_strength * edge_factor * _smootherstep(edge_factor)
        
        feathered_xyz[key] = feather_orig_xyz[i] * (1.0 - actual_strength) + avg_neighbor_xyz * actual_strength
    
    return feathered_xyz


def _compute_local_density(points_xyz: np.ndarray, k_neighbors: int = 10) -> np.ndarray:
    """
    计算每个点的局部密度（使用K近邻平均距离的倒数）
    
    密度定义：density = 1 / avg_knn_distance
    密度越高表示点越密集，距离越小
    
    Args:
        points_xyz: 点坐标数组 (N, 3)
        k_neighbors: 用于计算密度的近邻数
        
    Returns:
        densities: 每个点的局部密度 (N,)
    """
    n = len(points_xyz)
    if n < k_neighbors + 1:
        return np.ones(n)
    
    tree = cKDTree(points_xyz)
    # 查询 k+1 个近邻（因为第一个是点本身，距离为0）
    distances, _ = tree.query(points_xyz, k=k_neighbors + 1)
    
    # 排除自己（距离为0的点），计算平均距离
    avg_distances = distances[:, 1:].mean(axis=1)
    
    # 密度 = 1 / 平均距离（密度越高，距离越小）
    densities = 1.0 / (avg_distances + 1e-10)
    
    return densities


def _compute_avg_spacing(points_xyz: np.ndarray, k_neighbors: int = 10) -> float:
    """
    计算点云的平均点间距
    
    Args:
        points_xyz: 点坐标数组 (N, 3)
        k_neighbors: 用于计算的近邻数
        
    Returns:
        avg_spacing: 平均点间距
    """
    n = len(points_xyz)
    if n < k_neighbors + 1:
        return 1.0
    
    tree = cKDTree(points_xyz)
    distances, _ = tree.query(points_xyz, k=k_neighbors + 1)
    
    # 使用中位数而不是平均值，更鲁棒
    avg_spacing = np.median(distances[:, 1:])
    
    return avg_spacing


def _density_based_thinning(
    keep_points: set,
    point_source_type: Dict,
    blended_points: Dict,
    recon1: 'pycolmap.Reconstruction',
    recon2: 'pycolmap.Reconstruction',
    all_blend_weights: Optional[Dict] = None,
    k_neighbors: int = 10,
    target_density_percentile: float = 50.0,
    density_tolerance: float = 1.2,
    use_grid_thinning: bool = True,
    grid_resolution_factor: float = 1.0,
    use_blend_weight_decay: bool = True,
    distance_decay_factor: float = 0.5,
    min_points_for_analysis: int = 100,
    verbose: bool = True,
) -> Tuple[set, Dict]:
    """
    基于密度的稀疏化处理：实现从重叠区到非重叠区的密度渐变过渡（改进版）
    
    核心思想：
    - inner_blend_margin 区域：密度与重叠区一致（较稀疏）
    - outer_blend_margin 区域：密度接近原始状态（较密集）
    - 融合带内：密度从稀疏渐变到密集
    
    改进点：
    1. 使用 blend_weight 控制稀疏化强度（与 inner/outer_blend_margin 一致）
    2. 使用确定性的网格采样代替概率性丢弃
    3. 实现从内边界（强稀疏化）到外边界（无稀疏化）的渐变
    
    工作原理：
    1. 分析重叠区的点间距作为目标密度
    2. 基于 blend_weight 决定每个点的稀疏化强度：
       - blend_weight ≈ 0（靠近内边界）：按目标密度稀疏化
       - blend_weight ≈ 1（靠近外边界）：保持原始密度
    3. 对需要稀疏化的区域进行网格采样
    
    Args:
        keep_points: 当前保留的点集合 {('r1', pt3d_id), ('r2', pt3d_id), ...}
        point_source_type: 点来源类型映射 {(source, pt3d_id): type_str}
        blended_points: 已融合点信息 {(source, pt3d_id): {...}}
        recon1, recon2: 两个 reconstruction
        all_blend_weights: 融合权重字典 {(source, pt3d_id): weight}
            - 由 inner/outer_blend_margin 决定
            - weight=0: 靠近内边界，强稀疏化
            - weight=1: 靠近外边界，不稀疏化
        k_neighbors: 用于计算点间距的近邻数（默认 10）
        target_density_percentile: 使用重叠区密度的百分位数作为目标（默认 50）
        density_tolerance: 密度容差倍数（默认 1.2）
            - 网格大小 = target_spacing / density_tolerance
            - 值越大，稀疏化越强
        use_grid_thinning: 是否使用网格采样方法（默认 True）
        grid_resolution_factor: 网格分辨率因子（默认 1.0）
        use_blend_weight_decay: 是否使用 blend_weight 控制渐变（默认 True）
            - True: 基于 blend_weight 实现渐变（推荐）
            - False: 基于 3D 距离实现渐变（旧方法）
        distance_decay_factor: 距离衰减因子（当 use_blend_weight_decay=False 时使用）
        min_points_for_analysis: 最少需要多少个重叠区点才进行密度分析（默认 100）
        verbose: 是否打印详细信息
        
    Returns:
        updated_keep_points: 更新后的保留点集合
        stats: 密度均衡化统计信息
    """
    if all_blend_weights is None:
        all_blend_weights = {}
    
    stats = {
        'enabled': True,
        'overlap_points_count': 0,
        'non_overlap_points_count': 0,
        'target_spacing': 0.0,
        'grid_size': 0.0,
        'discarded_count': 0,
        'discard_by_source': {'r1': 0, 'r2': 0},
        'method': 'grid_thinning' if use_grid_thinning else 'probabilistic',
        'decay_method': 'blend_weight' if use_blend_weight_decay else 'distance',
    }
    
    # ========== Step 1: 收集重叠区的点（已融合的点）==========
    OVERLAP_TYPES = {
        'blended', 'blended_3d', 'blended_projected', 'blended_aggressive',
        'conflict', 'conflict_3d', 'conflict_projected', 'conflict_aggressive',
        'match_3d',
    }
    
    overlap_xyz_list = []
    overlap_keys = []
    
    for key in keep_points:
        source, pt3d_id = key
        source_type = point_source_type.get(key, 'only')
        
        if source_type in OVERLAP_TYPES or key in blended_points:
            if key in blended_points:
                xyz = np.asarray(blended_points[key]['xyz'])
            elif source == 'r1' and pt3d_id in recon1.points3D:
                xyz = np.asarray(recon1.points3D[pt3d_id].xyz)
            elif source == 'r2' and pt3d_id in recon2.points3D:
                xyz = np.asarray(recon2.points3D[pt3d_id].xyz)
            else:
                continue
            
            overlap_xyz_list.append(xyz)
            overlap_keys.append(key)
    
    stats['overlap_points_count'] = len(overlap_xyz_list)
    
    if len(overlap_xyz_list) < min_points_for_analysis:
        stats['enabled'] = False
        stats['reason'] = f'Not enough overlap points ({len(overlap_xyz_list)} < {min_points_for_analysis})'
        return keep_points, stats
    
    overlap_xyz = np.array(overlap_xyz_list)
    
    # ========== Step 2: 计算重叠区的点间距分布 ==========
    # 使用点间距而非密度，更直观
    overlap_tree = cKDTree(overlap_xyz)
    k = min(k_neighbors, len(overlap_xyz) - 1)
    overlap_dists, _ = overlap_tree.query(overlap_xyz, k=k + 1)
    overlap_spacings = overlap_dists[:, 1:].mean(axis=1)
    
    # 使用百分位数作为目标点间距
    target_spacing = np.percentile(overlap_spacings, target_density_percentile)
    stats['target_spacing'] = float(target_spacing)
    stats['overlap_spacing_stats'] = {
        'min': float(np.min(overlap_spacings)),
        'max': float(np.max(overlap_spacings)),
        'mean': float(np.mean(overlap_spacings)),
        'median': float(np.median(overlap_spacings)),
        'p25': float(np.percentile(overlap_spacings, 25)),
        'p75': float(np.percentile(overlap_spacings, 75)),
    }
    
    # ========== Step 3: 收集非重叠区的点（同时获取 blend_weight）==========
    NON_OVERLAP_TYPES = {'only', 'projected', 'blend'}
    
    non_overlap_xyz_list = []
    non_overlap_keys = []
    non_overlap_blend_weights = []  # 新增：记录每个点的 blend_weight
    
    for key in keep_points:
        source, pt3d_id = key
        source_type = point_source_type.get(key, 'only')
        
        if source_type in NON_OVERLAP_TYPES and key not in blended_points:
            if source == 'r1' and pt3d_id in recon1.points3D:
                xyz = np.asarray(recon1.points3D[pt3d_id].xyz)
            elif source == 'r2' and pt3d_id in recon2.points3D:
                xyz = np.asarray(recon2.points3D[pt3d_id].xyz)
            else:
                continue
            
            non_overlap_xyz_list.append(xyz)
            non_overlap_keys.append(key)
            # 获取 blend_weight，默认为 1（不稀疏化）
            bw = all_blend_weights.get(key, 1.0)
            non_overlap_blend_weights.append(bw)
    
    stats['non_overlap_points_count'] = len(non_overlap_xyz_list)
    
    if len(non_overlap_xyz_list) < k_neighbors + 1:
        stats['enabled'] = False
        stats['reason'] = f'Not enough non-overlap points ({len(non_overlap_xyz_list)})'
        return keep_points, stats
    
    non_overlap_xyz = np.array(non_overlap_xyz_list)
    non_overlap_blend_weights = np.array(non_overlap_blend_weights)
    
    # ========== Step 4: 计算非重叠区的点间距 ==========
    non_overlap_tree = cKDTree(non_overlap_xyz)
    k = min(k_neighbors, len(non_overlap_xyz) - 1)
    non_overlap_dists, _ = non_overlap_tree.query(non_overlap_xyz, k=k + 1)
    non_overlap_spacings = non_overlap_dists[:, 1:].mean(axis=1)
    
    stats['non_overlap_spacing_stats'] = {
        'min': float(np.min(non_overlap_spacings)),
        'max': float(np.max(non_overlap_spacings)),
        'mean': float(np.mean(non_overlap_spacings)),
        'median': float(np.median(non_overlap_spacings)),
    }
    
    # ========== Step 5: 计算稀疏化权重 ==========
    # 稀疏化权重（thinning_weight）：1 = 完全稀疏化，0 = 不稀疏化
    
    if use_blend_weight_decay and len(all_blend_weights) > 0:
        # ========== 基于 blend_weight 的渐变（推荐）==========
        # blend_weight = 0（靠近内边界）→ thinning_weight = 1（强稀疏化）
        # blend_weight = 1（靠近外边界）→ thinning_weight = 0（不稀疏化）
        thinning_weights = 1.0 - non_overlap_blend_weights
        
        # 统计融合带内点的分布
        blend_zone_count = np.sum((non_overlap_blend_weights > 0) & (non_overlap_blend_weights < 1))
        stats['blend_zone_points'] = int(blend_zone_count)
        stats['blend_weight_stats'] = {
            'min': float(np.min(non_overlap_blend_weights)),
            'max': float(np.max(non_overlap_blend_weights)),
            'mean': float(np.mean(non_overlap_blend_weights)),
            'in_blend_zone': int(blend_zone_count),
        }
    else:
        # ========== 基于 3D 距离的渐变（旧方法）==========
        dist_to_overlap, _ = overlap_tree.query(non_overlap_xyz, k=1)
        
        # 使用重叠区的特征尺寸作为参考
        overlap_extent = np.max(overlap_xyz.max(axis=0) - overlap_xyz.min(axis=0))
        normalized_dist = dist_to_overlap / (overlap_extent + 1e-10)
        # 渐变权重：距离越近权重越大（稀疏化越强）
        thinning_weights = np.exp(-distance_decay_factor * normalized_dist * 10)
        
        stats['distance_stats'] = {
            'min': float(np.min(dist_to_overlap)),
            'max': float(np.max(dist_to_overlap)),
            'mean': float(np.mean(dist_to_overlap)),
            'overlap_extent': float(overlap_extent),
        }
    
    # ========== Step 6: 执行稀疏化 ==========
    points_to_discard = set()
    
    if use_grid_thinning:
        # ========== 网格采样方法 ==========
        # 计算网格大小：基于目标点间距
        grid_size = target_spacing * grid_resolution_factor / density_tolerance
        stats['grid_size'] = float(grid_size)
        
        if grid_size < 1e-6:
            stats['enabled'] = False
            stats['reason'] = f'Grid size too small ({grid_size})'
            return keep_points, stats
        
        # 将点分配到网格单元
        # 对于每个网格单元，选择一个代表点（优先选择重投影误差小的或距离重叠区近的）
        grid_cells = {}  # {(gx, gy, gz): [(key, xyz, spacing, thinning_weight, idx), ...]}
        
        for i, (key, xyz, spacing, tw) in enumerate(zip(
            non_overlap_keys, non_overlap_xyz, non_overlap_spacings, thinning_weights
        )):
            # 计算网格坐标
            gx = int(np.floor(xyz[0] / grid_size))
            gy = int(np.floor(xyz[1] / grid_size))
            gz = int(np.floor(xyz[2] / grid_size))
            cell_key = (gx, gy, gz)
            
            if cell_key not in grid_cells:
                grid_cells[cell_key] = []
            
            # 获取重投影误差作为优先级
            source, pt3d_id = key
            if source == 'r1' and pt3d_id in recon1.points3D:
                error = recon1.points3D[pt3d_id].error
            elif source == 'r2' and pt3d_id in recon2.points3D:
                error = recon2.points3D[pt3d_id].error
            else:
                error = float('inf')
            
            grid_cells[cell_key].append((key, xyz, spacing, tw, error, i))
        
        # 对于每个网格单元，决定保留哪些点
        for cell_key, cell_points in grid_cells.items():
            if len(cell_points) <= 1:
                continue  # 只有一个点，不需要稀疏化
            
            # 计算该单元的平均稀疏化权重
            avg_tw = np.mean([p[3] for p in cell_points])
            
            # 根据稀疏化权重决定保留多少点
            # avg_tw 接近 1 时，只保留 1 个点
            # avg_tw 接近 0 时，保留所有点
            keep_ratio = 1.0 - avg_tw * (1.0 - 1.0 / len(cell_points))
            num_keep = max(1, int(np.ceil(len(cell_points) * keep_ratio)))
            
            # 按优先级排序：优先保留误差小的点
            sorted_points = sorted(cell_points, key=lambda p: p[4])
            
            # 标记要丢弃的点
            for p in sorted_points[num_keep:]:
                key = p[0]
                points_to_discard.add(key)
                source = key[0]
                stats['discard_by_source'][source] = stats['discard_by_source'].get(source, 0) + 1
    
    else:
        # ========== 概率性丢弃方法（旧方法，保留作为备选）==========
        target_density = 1.0 / (target_spacing + 1e-10)
        threshold_spacing = target_spacing / density_tolerance
        
        # 设置随机种子以获得可重复的结果
        np.random.seed(42)
        
        for i, (key, spacing, tw) in enumerate(zip(
            non_overlap_keys, non_overlap_spacings, thinning_weights
        )):
            # 只处理点间距小于目标的点（即密度过高的点）
            if spacing < threshold_spacing:
                # 计算基础丢弃概率
                base_prob = 1.0 - (spacing / target_spacing)
                base_prob = max(0.0, min(0.9, base_prob))
                
                # 应用距离衰减
                discard_prob = base_prob * tw
                
                if np.random.random() < discard_prob:
                    points_to_discard.add(key)
                    source = key[0]
                    stats['discard_by_source'][source] = stats['discard_by_source'].get(source, 0) + 1
    
    stats['discarded_count'] = len(points_to_discard)
    
    # ========== Step 7: 更新保留点集合 ==========
    updated_keep_points = keep_points - points_to_discard
    
    if verbose:
        print(f"\n  Density-based thinning ({stats['method']}, decay: {stats['decay_method']}):")
        print(f"    Overlap region points: {stats['overlap_points_count']}")
        print(f"    Non-overlap region points: {stats['non_overlap_points_count']}")
        print(f"    Target spacing (p{target_density_percentile:.0f}): {stats['target_spacing']:.4f}")
        print(f"    Overlap spacing: mean={stats['overlap_spacing_stats']['mean']:.4f}, "
              f"median={stats['overlap_spacing_stats']['median']:.4f}")
        print(f"    Non-overlap spacing: mean={stats['non_overlap_spacing_stats']['mean']:.4f}, "
              f"median={stats['non_overlap_spacing_stats']['median']:.4f}")
        if use_grid_thinning:
            print(f"    Grid size: {stats['grid_size']:.4f}")
            print(f"    Grid resolution factor: {grid_resolution_factor:.2f}")
        if use_blend_weight_decay and 'blend_weight_stats' in stats:
            bw_stats = stats['blend_weight_stats']
            print(f"    Blend weight decay: inner(bw=0)→sparse, outer(bw=1)→dense")
            print(f"      Points in blend zone (0<bw<1): {bw_stats['in_blend_zone']}")
            print(f"      Blend weight range: [{bw_stats['min']:.2f}, {bw_stats['max']:.2f}], mean={bw_stats['mean']:.2f}")
        else:
            print(f"    Distance decay factor: {distance_decay_factor:.2f}")
        print(f"    Discarded points: {stats['discarded_count']} "
              f"(R1: {stats['discard_by_source'].get('r1', 0)}, R2: {stats['discard_by_source'].get('r2', 0)})")
    
    return updated_keep_points, stats


def _interpolate_blend_zone_displacements(
    keep_points: set,
    point_source_type: Dict,
    all_blend_weights: Dict,
    blended_points: Dict,
    recon1: 'pycolmap.Reconstruction',
    recon2: 'pycolmap.Reconstruction',
    k_neighbors: int = 32,
    min_displacement_points: int = 5,
    use_smooth_transition: bool = True,
    smooth_power: float = 0.5,
    use_gaussian_weights: bool = True,
    sigma_factor: float = 2.0,
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    对融合带内的独有点进行3D坐标空间插值，实现平滑过渡（向量化版本）
    
    【性能优化】批量 KD-Tree 查询和向量化计算，大幅提升处理速度
    
    Args:
        keep_points: 保留的点集合 {('r1', pt3d_id), ('r2', pt3d_id), ...}
        point_source_type: 点来源类型映射 {(source, pt3d_id): type_str}
        all_blend_weights: 融合权重 {(source, pt3d_id): weight}，weight 在 [0, 1]
        blended_points: 已融合点对信息
        recon1, recon2: 两个 reconstruction（recon2 已对齐）
        k_neighbors: 用于插值的最近邻数量（默认 32）
        min_displacement_points: 最少需要多少个位移参考点才进行插值
        use_smooth_transition: 是否使用 smoothstep 函数实现更平滑的过渡
        smooth_power: 平滑过渡的力度参数（默认 0.5）
        use_gaussian_weights: 是否使用高斯加权（默认 True）
        sigma_factor: 高斯加权的 sigma 系数
        
    Returns:
        interpolated_xyz: {(source, pt3d_id): new_xyz} 插值后的3D坐标
    """
    interpolated_xyz = {}
    
    # ========== Step 1: 收集位移信息 ==========
    displacement_refs_r2 = []
    displacement_refs_r1 = []
    
    for key, blend_info in blended_points.items():
        source, pt3d_id = key
        blended_xyz = np.asarray(blend_info['xyz'])
        
        pt3d_id2 = blend_info.get('pt3d_id2')
        pt3d_id1 = blend_info.get('pt3d_id1')
        
        if pt3d_id2 is not None and pt3d_id2 in recon2.points3D:
            original_xyz_r2 = np.asarray(recon2.points3D[pt3d_id2].xyz)
            displacement_r2 = blended_xyz - original_xyz_r2
            displacement_refs_r2.append((original_xyz_r2, displacement_r2))
        
        if pt3d_id1 is not None and pt3d_id1 in recon1.points3D:
            original_xyz_r1 = np.asarray(recon1.points3D[pt3d_id1].xyz)
            displacement_r1 = blended_xyz - original_xyz_r1
            displacement_refs_r1.append((original_xyz_r1, displacement_r1))
    
    if len(displacement_refs_r2) < min_displacement_points:
        return interpolated_xyz
    
    # ========== Step 2: 构建 KD-Tree 和计算特征尺度 ==========
    ref_xyz_r2 = np.array([ref[0] for ref in displacement_refs_r2])
    ref_disp_r2 = np.array([ref[1] for ref in displacement_refs_r2])
    tree_r2 = cKDTree(ref_xyz_r2)
    
    k_scale = min(10, len(ref_xyz_r2) - 1)
    if k_scale > 0:
        scale_dists, _ = tree_r2.query(ref_xyz_r2, k=k_scale + 1)
        median_spacing_r2 = np.median(scale_dists[:, 1:].mean(axis=1))
    else:
        median_spacing_r2 = 1.0
    
    tree_r1 = None
    ref_disp_r1 = None
    median_spacing_r1 = 1.0
    if len(displacement_refs_r1) >= min_displacement_points:
        ref_xyz_r1 = np.array([ref[0] for ref in displacement_refs_r1])
        ref_disp_r1 = np.array([ref[1] for ref in displacement_refs_r1])
        tree_r1 = cKDTree(ref_xyz_r1)
        k_scale = min(10, len(ref_xyz_r1) - 1)
        if k_scale > 0:
            scale_dists, _ = tree_r1.query(ref_xyz_r1, k=k_scale + 1)
            median_spacing_r1 = np.median(scale_dists[:, 1:].mean(axis=1))
    
    # ========== Step 3: 收集需要插值的点（按 source 分组）==========
    points_r1_keys = []
    points_r1_xyz = []
    points_r1_bw = []
    points_r2_keys = []
    points_r2_xyz = []
    points_r2_bw = []
    
    for key in keep_points:
        if key in blended_points:
            continue
        
        blend_weight = all_blend_weights.get(key, 1.0)
        if blend_weight >= 1.0:
            continue
        
        source, pt3d_id = key
        if source == 'r1':
            if tree_r1 is None or ref_disp_r1 is None:
                continue
            if pt3d_id not in recon1.points3D:
                continue
            points_r1_keys.append(key)
            points_r1_xyz.append(np.asarray(recon1.points3D[pt3d_id].xyz))
            points_r1_bw.append(blend_weight)
        else:
            if pt3d_id not in recon2.points3D:
                continue
            points_r2_keys.append(key)
            points_r2_xyz.append(np.asarray(recon2.points3D[pt3d_id].xyz))
            points_r2_bw.append(blend_weight)
    
    # ========== Step 4: 批量处理 recon2 点 ==========
    if len(points_r2_xyz) > 0:
        points_r2_xyz = np.array(points_r2_xyz)
        points_r2_bw = np.array(points_r2_bw)
        
        # 批量 KD-Tree 查询
        k = min(k_neighbors, len(ref_disp_r2))
        all_distances, all_indices = tree_r2.query(points_r2_xyz, k=k)
        if k == 1:
            all_distances = all_distances[:, np.newaxis]
            all_indices = all_indices[:, np.newaxis]
        
        # 计算应用因子（向量化）
        apply_factors = 1.0 - points_r2_bw
        if use_smooth_transition:
            apply_factors = _smootherstep_vectorized(apply_factors)
            if smooth_power != 1.0:
                apply_factors = np.where(apply_factors > 0, apply_factors ** smooth_power, apply_factors)
        
        # 计算权重和位移（向量化）
        eps = 1e-10
        sigma = median_spacing_r2 * sigma_factor
        
        if use_gaussian_weights:
            gaussian_weights = np.exp(-0.5 * (all_distances / (sigma + eps)) ** 2)
            max_dists = all_distances[:, -1:] if k > 1 else all_distances
            distance_decay = np.clip(1.0 - (all_distances / (max_dists + eps)) ** 2, 0.1, 1.0)
            weights = gaussian_weights * distance_decay
        else:
            weights = 1.0 / (all_distances ** 1.5 + eps)
        
        # 归一化权重
        weight_sums = weights.sum(axis=1, keepdims=True)
        weights = weights / np.maximum(weight_sums, eps)
        
        # 计算加权位移（批量矩阵运算）
        # ref_disp_r2[all_indices] 形状: (n_points, k, 3)
        selected_disps = ref_disp_r2[all_indices]  # (n_points, k, 3)
        interpolated_disps = np.sum(weights[:, :, np.newaxis] * selected_disps, axis=1)  # (n_points, 3)
        
        # 应用位移
        new_xyz_r2 = points_r2_xyz + apply_factors[:, np.newaxis] * interpolated_disps
        
        # 存储结果
        valid_mask = apply_factors >= 0.005
        for i, key in enumerate(points_r2_keys):
            if valid_mask[i]:
                interpolated_xyz[key] = new_xyz_r2[i]
    
    # ========== Step 5: 批量处理 recon1 点 ==========
    if len(points_r1_xyz) > 0 and tree_r1 is not None:
        points_r1_xyz = np.array(points_r1_xyz)
        points_r1_bw = np.array(points_r1_bw)
        
        k = min(k_neighbors, len(ref_disp_r1))
        all_distances, all_indices = tree_r1.query(points_r1_xyz, k=k)
        if k == 1:
            all_distances = all_distances[:, np.newaxis]
            all_indices = all_indices[:, np.newaxis]
        
        apply_factors = 1.0 - points_r1_bw
        if use_smooth_transition:
            apply_factors = _smootherstep_vectorized(apply_factors)
            if smooth_power != 1.0:
                apply_factors = np.where(apply_factors > 0, apply_factors ** smooth_power, apply_factors)
        
        eps = 1e-10
        sigma = median_spacing_r1 * sigma_factor
        
        if use_gaussian_weights:
            gaussian_weights = np.exp(-0.5 * (all_distances / (sigma + eps)) ** 2)
            max_dists = all_distances[:, -1:] if k > 1 else all_distances
            distance_decay = np.clip(1.0 - (all_distances / (max_dists + eps)) ** 2, 0.1, 1.0)
            weights = gaussian_weights * distance_decay
        else:
            weights = 1.0 / (all_distances ** 1.5 + eps)
        
        weight_sums = weights.sum(axis=1, keepdims=True)
        weights = weights / np.maximum(weight_sums, eps)
        
        selected_disps = ref_disp_r1[all_indices]
        interpolated_disps = np.sum(weights[:, :, np.newaxis] * selected_disps, axis=1)
        
        new_xyz_r1 = points_r1_xyz + apply_factors[:, np.newaxis] * interpolated_disps
        
        valid_mask = apply_factors >= 0.005
        for i, key in enumerate(points_r1_keys):
            if valid_mask[i]:
                interpolated_xyz[key] = new_xyz_r1[i]
    
    return interpolated_xyz


def _process_unmatched_points(region_info: Dict, source_label: str, 
                               inner_shrink: float, outer_expand: float,
                               blend_strategy: str = 'interpolate') -> Dict:
    """
    处理单张影像中未匹配的点（双边界平滑融合版 + 插值过渡）
    
    三区域策略实现平滑过渡：
    1. 核心重叠区（内边界内）：丢弃未匹配点
       - 这些点在重叠区深处，如果质量好应该已经匹配了
    2. 融合带（内边界外，外边界内）：保留未匹配点，带插值权重
       - 边界附近的点，保留以实现平滑过渡
       - 权重从内边界(0)到外边界(1)线性插值
    3. 非重叠区（外边界外）：保留未匹配点，权重=1
       - 这些点不在重叠区内，应该保留
    
    Args:
        region_info: 区域信息字典，包含 'matched_pixels', 'pmap', 'processed_points'
        source_label: 来源标签 ('r1' 或 'r2')
        inner_shrink: 内边界收缩量（像素），控制核心丢弃区大小
        outer_expand: 外边界扩展量（像素），控制融合带宽度
        blend_strategy: 融合策略
            - 'interpolate': 使用插值权重（默认）
            - 'binary': 二值（兼容旧版本，不使用插值）
        
    Returns:
        结果字典 {
            'keep': [...], 
            'discard': [...], 
            'blend': [...], 
            'blend_weights': {pt3d_id: weight},  # 新增：每个点的插值权重
            'source': str
        }
    """
    matched_pixels = region_info['matched_pixels']
    pmap = region_info['pmap']
    processed_points = region_info['processed_points']
    
    # 收集所有需要检查的点
    points_to_check = [
        (pixel_key, info['point3D_id'], info)
        for pixel_key, info in pmap.items()
        if info['point3D_id'] not in processed_points
    ]
    
    if len(points_to_check) == 0:
        return {'keep': [], 'discard': [], 'blend': [], 'blend_weights': {}, 'source': source_label}
    
    # 预计算双边界Delaunay（每张图只算一次）
    inner_delaunay, outer_delaunay, matched_arr, inner_hull_pts, outer_hull_pts = _build_dual_boundary_delaunay(
        matched_pixels, inner_shrink, outer_expand
    )
    
    # 使用双边界分类（带插值权重计算）
    fallback_dist = max(inner_shrink + outer_expand, 50.0)
    discard_mask, blend_mask, blend_factors = _classify_points_dual_boundary(
        points_to_check, inner_delaunay, outer_delaunay, matched_arr,
        inner_hull_pts, outer_hull_pts, inner_shrink, outer_expand, fallback_dist
    )
    
    # 分类结果
    keep_list = []
    discard_list = []
    blend_list = []
    blend_weights = {}  # {pt3d_id: weight}
    
    for i, (pixel_key, pt3d_id, info) in enumerate(points_to_check):
        if discard_mask[i]:
            discard_list.append(pt3d_id)
        elif blend_mask[i]:
            blend_list.append(pt3d_id)
            keep_list.append(pt3d_id)  # 融合带的点也加入保留列表
            # 记录插值权重
            blend_weights[pt3d_id] = float(blend_factors[i])
        else:
            keep_list.append(pt3d_id)
            # 非重叠区的点权重为1
            blend_weights[pt3d_id] = 1.0
    
    return {
        'keep': keep_list, 
        'discard': discard_list, 
        'blend': blend_list, 
        'blend_weights': blend_weights,
        'source': source_label
    }


def _project_3d_to_2d(
    xyz: np.ndarray,
    image: 'pycolmap.Image',
    camera: 'pycolmap.Camera'
) -> Optional[Tuple[float, float]]:
    """
    将3D点投影到图像平面上
    
    Args:
        xyz: 3D点坐标 (3,)
        image: pycolmap Image 对象
        camera: pycolmap Camera 对象
        
    Returns:
        (u, v) 像素坐标，如果点在相机后面返回 None
    """
    # 获取相机外参
    R = image.cam_from_world.rotation.matrix()
    t = image.cam_from_world.translation
    
    # 变换到相机坐标系
    xyz_cam = R @ xyz + t
    
    # 检查点是否在相机前面
    if xyz_cam[2] <= 0:
        return None
    
    # 投影到归一化平面
    x_norm = xyz_cam[0] / xyz_cam[2]
    y_norm = xyz_cam[1] / xyz_cam[2]
    
    # 应用相机内参
    model = camera.model.name
    params = camera.params
    
    if model in ['SIMPLE_PINHOLE', 'PINHOLE']:
        if model == 'SIMPLE_PINHOLE':
            f = params[0]
            cx, cy = params[1], params[2]
            u = f * x_norm + cx
            v = f * y_norm + cy
        else:  # PINHOLE
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
            u = fx * x_norm + cx
            v = fy * y_norm + cy
    elif model in ['SIMPLE_RADIAL', 'RADIAL']:
        f = params[0]
        cx, cy = params[1], params[2]
        k1 = params[3] if len(params) > 3 else 0
        k2 = params[4] if len(params) > 4 else 0
        r2 = x_norm**2 + y_norm**2
        radial = 1 + k1 * r2 + k2 * r2**2
        u = f * x_norm * radial + cx
        v = f * y_norm * radial + cy
    else:
        # 对于其他模型，使用简化投影
        f = params[0] if len(params) > 0 else 1
        cx = params[1] if len(params) > 1 else camera.width / 2
        cy = params[2] if len(params) > 2 else camera.height / 2
        u = f * x_norm + cx
        v = f * y_norm + cy
    
    # 检查是否在图像范围内
    if 0 <= u < camera.width and 0 <= v < camera.height:
        return (u, v)
    return None


def _process_non_overlap_track_points(
    recon_src: pycolmap.Reconstruction,
    recon_dst: pycolmap.Reconstruction,
    common_img_ids_src: set,
    common_images: Dict[int, int],
    pixel_map_dst: Dict[int, Dict],
    matched_regions_dst: Dict[int, Dict],
    conf_cache_src: Dict,
    conf_cache_dst: Dict,
    kept_ids: set,
    discard_ids: set,
    max_match_radius: float,
    inner_shrink: float,
    outer_expand: float,
    keep_unmatched_overlap: bool,
    source_label: str,
    blend_mode: str = 'winner',
) -> Tuple[List, List, List, Dict]:
    """
    处理那些 track 中没有共同影像观测，但可能在空间上位于重叠区的 3D 点
    
    通过将这些点投影到共同影像上，检查是否在匹配区域内，
    如果是，则尝试在另一个 reconstruction 中找到对应的 3D 点。
    
    Args:
        recon_src: 源 reconstruction
        recon_dst: 目标 reconstruction（用于查找对应点）
        common_img_ids_src: 源 reconstruction 中共同影像的 ID 集合
        common_images: 共同影像映射 {src_img_id: dst_img_id}
        pixel_map_dst: 目标 reconstruction 的像素映射
        matched_regions_dst: 目标 reconstruction 的匹配区域信息
        conf_cache_src: 源 reconstruction 的置信度缓存
        conf_cache_dst: 目标 reconstruction 的置信度缓存
        kept_ids: 已保留的源点 ID 集合
        discard_ids: 已丢弃的源点 ID 集合
        max_match_radius: 最大匹配半径（像素），用于投影点的最近邻搜索
        inner_shrink: 内边界收缩量
        outer_expand: 外边界扩展量
        keep_unmatched_overlap: 是否保留重叠区所有未匹配点
        source_label: 源标签 ('r1' 或 'r2')
        blend_mode: 融合模式
        
    Returns:
        keep_list: 应该保留的点 ID 列表
        discard_list: 应该丢弃的点 ID 列表
        matched_list: 新匹配到的点对 [(src_pt3d_id, dst_pt3d_id, blend_info or None), ...]
        stats: 统计信息
    """
    keep_list = []
    discard_list = []
    matched_list = []
    stats = {
        'projected_points': 0,
        'valid_projections': 0,
        'in_region_points': 0,
        'new_matches': 0,
        'kept_outside': 0,
    }
    
    # 构建共同影像ID映射（用于快速查找）
    common_img_map = common_images if source_label == 'r1' else {v: k for k, v in common_images.items()}
    
    # 收集需要处理的点
    points_to_process = []
    for pt3d_id in recon_src.points3D:
        if pt3d_id in kept_ids or pt3d_id in discard_ids:
            continue
        
        pt3d = recon_src.points3D[pt3d_id]
        
        # 检查 track 中是否有共同影像的观测
        has_common = False
        for te in pt3d.track.elements:
            if te.image_id in common_img_ids_src:
                has_common = True
                break
        
        # 只处理没有共同影像观测的点
        if not has_common:
            points_to_process.append(pt3d_id)
    
    if len(points_to_process) == 0:
        return keep_list, discard_list, matched_list, stats
    
    stats['projected_points'] = len(points_to_process)
    
    # 遍历共同影像，构建投影并查找匹配
    for src_img_id, dst_img_id in common_img_map.items():
        if src_img_id not in recon_src.images:
            continue
        
        src_image = recon_src.images[src_img_id]
        src_camera = recon_src.cameras[src_image.camera_id]
        
        # 获取目标像素映射和匹配区域
        pmap_dst = pixel_map_dst.get(dst_img_id)
        region_info = matched_regions_dst.get(dst_img_id)
        
        if pmap_dst is None or len(pmap_dst) == 0:
            continue
        
        # 构建目标像素的 KD-Tree
        pixels_dst_list = list(pmap_dst.keys())
        tree_dst = cKDTree(np.asarray(pixels_dst_list, dtype=np.float32))
        
        # 预计算边界信息（如果有匹配区域）
        inner_delaunay = None
        outer_delaunay = None
        matched_arr = None
        
        if region_info is not None and len(region_info.get('matched_pixels', [])) >= 3:
            inner_delaunay, outer_delaunay, matched_arr, _, _ = _build_dual_boundary_delaunay(
                region_info['matched_pixels'], inner_shrink, outer_expand
            )
        
        # 投影每个点到该影像
        for pt3d_id in points_to_process:
            if pt3d_id in kept_ids or pt3d_id in discard_ids:
                continue
            
            pt3d = recon_src.points3D[pt3d_id]
            xyz = np.array(pt3d.xyz)
            
            # 投影到图像
            proj = _project_3d_to_2d(xyz, src_image, src_camera)
            if proj is None:
                continue
            
            stats['valid_projections'] += 1
            px, py = proj
            pixel_key = (int(round(px)), int(round(py)))
            
            # 检查投影是否在匹配区域内
            in_region = False
            in_core = False
            
            if outer_delaunay is not None:
                point_arr = np.array([[px, py]], dtype=np.float64)
                in_outer = outer_delaunay.find_simplex(point_arr)[0] >= 0
                if in_outer:
                    in_region = True
                    if inner_delaunay is not None:
                        in_core = inner_delaunay.find_simplex(point_arr)[0] >= 0
            elif matched_arr is not None and len(matched_arr) > 0:
                # 使用距离判断
                tree_matched = cKDTree(matched_arr)
                dist, _ = tree_matched.query([px, py], k=1)
                if dist < outer_expand:
                    in_region = True
                    if dist < inner_shrink:
                        in_core = True
            
            if not in_region:
                # 不在匹配区域内，保留
                if pt3d_id not in kept_ids:
                    keep_list.append(pt3d_id)
                    kept_ids.add(pt3d_id)
                    stats['kept_outside'] += 1
                continue
            
            stats['in_region_points'] += 1
            
            # 在匹配区域内，查找目标 reconstruction 中的对应点
            distances, indices = tree_dst.query([px, py], k=1, distance_upper_bound=max_match_radius)
            
            if distances <= max_match_radius and indices < len(pixels_dst_list):
                # 找到对应点
                dst_pixel_key = pixels_dst_list[indices]
                dst_info = pmap_dst[dst_pixel_key]
                dst_pt3d_id = dst_info['point3D_id']
                
                # 获取置信度
                src_info = {'error': pt3d.error, 'xyz': pt3d.xyz, 'color': pt3d.color}
                score_src = _get_confidence_score(src_info, src_img_id, pixel_key, conf_cache_src)
                score_dst = _get_confidence_score(dst_info, dst_img_id, dst_pixel_key, conf_cache_dst)
                
                if blend_mode == 'weighted':
                    # 加权融合
                    blended_xyz, blended_color, blended_error = _blend_3d_points(
                        pt3d.xyz, dst_info['xyz'],
                        score_src, score_dst,
                        pt3d.color, dst_info['color'],
                        pt3d.error, dst_info['error']
                    )
                    blend_info = {
                        'xyz': blended_xyz,
                        'color': blended_color,
                        'error': blended_error,
                        'conf_src': score_src,
                        'conf_dst': score_dst,
                    }
                    matched_list.append((pt3d_id, dst_pt3d_id, blend_info))
                else:
                    # 胜者通吃
                    if score_src >= score_dst:
                        matched_list.append((pt3d_id, dst_pt3d_id, None))
                    else:
                        discard_list.append(pt3d_id)
                
                discard_ids.add(pt3d_id)  # 标记为已处理
                stats['new_matches'] += 1
                
            elif keep_unmatched_overlap:
                # 在区域内但没找到匹配，根据参数决定
                if pt3d_id not in kept_ids:
                    keep_list.append(pt3d_id)
                    kept_ids.add(pt3d_id)
            elif in_core:
                # 在核心区域内，丢弃
                discard_list.append(pt3d_id)
                discard_ids.add(pt3d_id)
            else:
                # 在融合带内，保留
                if pt3d_id not in kept_ids:
                    keep_list.append(pt3d_id)
                    kept_ids.add(pt3d_id)
    
    # 处理还没有被处理的点（投影失败或不在任何共同影像的视野内）
    for pt3d_id in points_to_process:
        if pt3d_id not in kept_ids and pt3d_id not in discard_ids:
            keep_list.append(pt3d_id)
            kept_ids.add(pt3d_id)
    
    return keep_list, discard_list, matched_list, stats


def _match_unmatched_points_3d(
    recon1: pycolmap.Reconstruction,
    recon2: pycolmap.Reconstruction,
    matched_regions_r1: Dict[int, Dict],
    matched_regions_r2: Dict[int, Dict],
    conf_cache_r1: Dict,
    conf_cache_r2: Dict,
    distance_threshold_3d: float = 0.5,
    blend_mode: str = 'winner',
) -> Tuple[List, List, List, List, Dict]:
    """
    在 3D 空间中匹配那些在重叠影像中有 2D 观测但 2D 匹配失败的点
    
    这是对 2D 匹配的补充：有些点虽然在 2D 像素空间中没有匹配上
    （比如像素位置偏差超过 match_radius），但在 3D 空间中可能非常接近。
    
    关键改进：直接从 pmap（像素映射）中收集那些"有 2D 观测但 2D 匹配失败"的点，
    而不是从所有 points3D 中筛选。这样更精确地定位需要 3D 匹配的点。
    
    Args:
        recon1, recon2: 两个 reconstruction（recon2 已对齐到 recon1）
        matched_regions_r1: recon1 每张共同影像的匹配区域信息
            {img_id: {'matched_pixels': [...], 'pmap': {...}, 'processed_points': set}}
        matched_regions_r2: recon2 每张共同影像的匹配区域信息
        conf_cache_r1, conf_cache_r2: 置信度缓存
        distance_threshold_3d: 3D 空间距离阈值
        blend_mode: 融合模式 'winner' 或 'weighted'
        
    Returns:
        keep_r1: 应保留的 r1 点 ID 列表
        keep_r2: 应保留的 r2 点 ID 列表
        discard_r1: 应丢弃的 r1 点 ID 列表
        discard_r2: 应丢弃的 r2 点 ID 列表
        matched_pairs: 新匹配到的点对信息 {(src, pt3d_id): blend_info or None}
    """
    keep_r1 = []
    keep_r2 = []
    discard_r1 = []
    discard_r2 = []
    matched_pairs = {}
    
    stats = {
        'r1_candidates': 0,
        'r2_candidates': 0,
        'matched_3d': 0,
    }
    
    # ========== 从 pmap 中收集 2D 匹配失败的点 ==========
    # 这些点在重叠影像中有 2D 观测，但在 _process_image_pair_for_merge 中没有匹配成功
    
    # 收集 recon1 中 2D 匹配失败的点
    r1_unmatched = {}  # {pt3d_id: {'xyz': xyz, 'pt3d': pt3d, 'pixel_infos': [...]}}
    for img_id, region_info in matched_regions_r1.items():
        pmap = region_info['pmap']
        processed_points = region_info['processed_points']
        
        for pixel_key, info in pmap.items():
            pt3d_id = info['point3D_id']
            # 只收集那些没有被 2D 匹配处理的点
            if pt3d_id in processed_points:
                continue
            
            if pt3d_id not in recon1.points3D:
                continue
            
            if pt3d_id not in r1_unmatched:
                pt3d = recon1.points3D[pt3d_id]
                r1_unmatched[pt3d_id] = {
                    'xyz': np.array(pt3d.xyz),
                    'pt3d': pt3d,
                    'pixel_infos': []  # 记录该点在各影像上的 2D 信息
                }
            
            r1_unmatched[pt3d_id]['pixel_infos'].append({
                'img_id': img_id,
                'pixel_key': pixel_key,
                'error': info['error']
            })
    
    # 收集 recon2 中 2D 匹配失败的点
    r2_unmatched = {}
    for img_id, region_info in matched_regions_r2.items():
        pmap = region_info['pmap']
        processed_points = region_info['processed_points']
        
        for pixel_key, info in pmap.items():
            pt3d_id = info['point3D_id']
            if pt3d_id in processed_points:
                continue
            
            if pt3d_id not in recon2.points3D:
                continue
            
            if pt3d_id not in r2_unmatched:
                pt3d = recon2.points3D[pt3d_id]
                r2_unmatched[pt3d_id] = {
                    'xyz': np.array(pt3d.xyz),
                    'pt3d': pt3d,
                    'pixel_infos': []
                }
            
            r2_unmatched[pt3d_id]['pixel_infos'].append({
                'img_id': img_id,
                'pixel_key': pixel_key,
                'error': info['error']
            })
    
    stats['r1_candidates'] = len(r1_unmatched)
    stats['r2_candidates'] = len(r2_unmatched)
    
    if len(r1_unmatched) == 0 or len(r2_unmatched) == 0:
        return keep_r1, keep_r2, discard_r1, discard_r2, {'stats': stats, 'pairs': matched_pairs}
    
    # ========== 在 3D 空间中进行 KD-Tree 匹配 ==========
    # 构建 recon2 未匹配点的 KD-Tree
    r2_ids = list(r2_unmatched.keys())
    r2_xyzs = np.array([r2_unmatched[pid]['xyz'] for pid in r2_ids])
    tree_r2 = cKDTree(r2_xyzs)
    
    # 从 recon1 未匹配点查找 recon2 中的最近邻
    r1_ids = list(r1_unmatched.keys())
    r1_xyzs = np.array([r1_unmatched[pid]['xyz'] for pid in r1_ids])
    
    distances, indices = tree_r2.query(r1_xyzs, k=1, distance_upper_bound=distance_threshold_3d)
    
    # 处理匹配结果
    matched_r1 = set()
    matched_r2 = set()
    
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist > distance_threshold_3d or idx >= len(r2_ids):
            continue
        
        r1_pt3d_id = r1_ids[i]
        r2_pt3d_id = r2_ids[idx]
        
        # 避免重复匹配（一对一）
        if r1_pt3d_id in matched_r1 or r2_pt3d_id in matched_r2:
            continue
        
        matched_r1.add(r1_pt3d_id)
        matched_r2.add(r2_pt3d_id)
        
        r1_info = r1_unmatched[r1_pt3d_id]
        r2_info = r2_unmatched[r2_pt3d_id]
        r1_pt3d = r1_info['pt3d']
        r2_pt3d = r2_info['pt3d']
        
        # 获取置信度（使用重投影误差作为备用）
        score1 = -r1_pt3d.error  # 误差越小，置信度越高
        score2 = -r2_pt3d.error
        
        if blend_mode == 'weighted':
            # 加权融合
            blended_xyz, blended_color, blended_error = _blend_3d_points(
                r1_pt3d.xyz, r2_pt3d.xyz,
                score1, score2,
                r1_pt3d.color, r2_pt3d.color,
                r1_pt3d.error, r2_pt3d.error
            )
            
            if score1 >= score2:
                primary_key = ('r1', r1_pt3d_id)
                matched_pairs[primary_key] = {
                    'xyz': blended_xyz,
                    'color': blended_color,
                    'error': blended_error,
                    'pt3d_id1': r1_pt3d_id,
                    'pt3d_id2': r2_pt3d_id,
                    'match_type': '3d_blend',
                    'match_distance': dist,
                }
                discard_r2.append(r2_pt3d_id)
            else:
                primary_key = ('r2', r2_pt3d_id)
                matched_pairs[primary_key] = {
                    'xyz': blended_xyz,
                    'color': blended_color,
                    'error': blended_error,
                    'pt3d_id1': r1_pt3d_id,
                    'pt3d_id2': r2_pt3d_id,
                    'match_type': '3d_blend',
                    'match_distance': dist,
                }
                discard_r1.append(r1_pt3d_id)
        else:
            # 胜者通吃
            if score1 >= score2:
                keep_r1.append(r1_pt3d_id)
                discard_r2.append(r2_pt3d_id)
                matched_pairs[('r1', r1_pt3d_id)] = {
                    'pt3d_id1': r1_pt3d_id,
                    'pt3d_id2': r2_pt3d_id,
                    'match_type': '3d_winner',
                    'match_distance': dist,
                }
            else:
                keep_r2.append(r2_pt3d_id)
                discard_r1.append(r1_pt3d_id)
                matched_pairs[('r2', r2_pt3d_id)] = {
                    'pt3d_id1': r1_pt3d_id,
                    'pt3d_id2': r2_pt3d_id,
                    'match_type': '3d_winner',
                    'match_distance': dist,
                }
        
        stats['matched_3d'] += 1
    
    # 计算平均匹配距离
    if stats['matched_3d'] > 0:
        match_dists = [v.get('match_distance', 0) for v in matched_pairs.values() if 'match_distance' in v]
        stats['avg_3d_distance'] = float(np.mean(match_dists)) if match_dists else 0.0
    
    return keep_r1, keep_r2, discard_r1, discard_r2, {'stats': stats, 'pairs': matched_pairs}


def _final_aggressive_3d_matching(
    recon1: pycolmap.Reconstruction,
    recon2: pycolmap.Reconstruction,
    keep_points: set,
    discard_r1: set,
    discard_r2: set,
    point_source_type: Dict,
    conf_cache_r1: Dict,
    conf_cache_r2: Dict,
    distance_threshold: float = 2.0,
    blend_mode: str = 'winner',
) -> Tuple[set, set, set, set, Dict, Dict]:
    """
    最终阶段的激进 3D 匹配
    
    对于所有仍然未被匹配的独有点（包括 'only', 'projected', 'blend' 类型），
    在 3D 空间中尝试使用更宽松的阈值进行匹配。
    
    这一步骤可以显著减少重叠区内的蓝色/绿色独有点，
    将它们转换为紫红色的融合点。
    
    处理的点类型：
    - 'only': 在重叠影像 pmap 中存在但未匹配的点
    - 'projected': track 中没有共同影像观测，通过投影检查但未找到 2D 匹配的点
    - 'blend': 在融合带内的点（仅当 keep_unmatched_overlap=False）
    
    不处理的点类型（已经被匹配过）：
    - 'conflict*': 冲突解决后保留的点
    - 'blended*': 加权融合的点
    - 'match_3d': 3D 匹配的点
    
    Args:
        recon1, recon2: 两个 reconstruction
        keep_points: 当前保留的点集合 {('r1', pt3d_id), ...}
        discard_r1, discard_r2: 已丢弃的点 ID 集合
        point_source_type: 点来源类型映射
        conf_cache_r1, conf_cache_r2: 置信度缓存
        distance_threshold: 3D 距离阈值（宽松值）
        blend_mode: 融合模式
        
    Returns:
        updated_keep_points: 更新后的保留点集合
        updated_discard_r1, updated_discard_r2: 更新后的丢弃集合
        updated_point_source_type: 更新后的来源类型
        blended_points: 新的融合点信息
        stats: 统计信息
    """
    stats = {
        'r1_only_candidates': 0,
        'r2_only_candidates': 0,
        'new_matches': 0,
        'avg_distance': 0.0,
    }
    blended_points = {}
    
    # 定义需要进行激进匹配的点类型
    # 这些是"未被匹配"的独有点类型
    UNMATCHED_TYPES = {'only', 'projected', 'blend'}
    
    # 收集所有未被匹配的独有点
    r1_only_points = {}  # {pt3d_id: xyz}
    r2_only_points = {}
    
    for (source, pt3d_id) in keep_points:
        source_type = point_source_type.get((source, pt3d_id), 'only')
        # 处理所有未被匹配的点类型
        if source_type in UNMATCHED_TYPES:
            if source == 'r1' and pt3d_id in recon1.points3D:
                r1_only_points[pt3d_id] = np.array(recon1.points3D[pt3d_id].xyz)
            elif source == 'r2' and pt3d_id in recon2.points3D:
                r2_only_points[pt3d_id] = np.array(recon2.points3D[pt3d_id].xyz)
    
    stats['r1_only_candidates'] = len(r1_only_points)
    stats['r2_only_candidates'] = len(r2_only_points)
    
    if len(r1_only_points) == 0 or len(r2_only_points) == 0:
        return keep_points, discard_r1, discard_r2, point_source_type, blended_points, stats
    
    # 构建 r2 独有点的 KD-Tree
    r2_ids = list(r2_only_points.keys())
    r2_xyzs = np.array([r2_only_points[pid] for pid in r2_ids])
    tree_r2 = cKDTree(r2_xyzs)
    
    # 从 r1 独有点查找 r2 中的最近邻（先不设阈值，获取所有距离）
    r1_ids = list(r1_only_points.keys())
    r1_xyzs = np.array([r1_only_points[pid] for pid in r1_ids])
    
    # 先获取所有距离用于统计
    all_distances, all_indices = tree_r2.query(r1_xyzs, k=1)
    
    # 统计距离分布（用于诊断）
    valid_dists = all_distances[all_distances < np.inf]
    if len(valid_dists) > 0:
        stats['nn_distance_stats'] = {
            'min': float(np.min(valid_dists)),
            'max': float(np.max(valid_dists)),
            'mean': float(np.mean(valid_dists)),
            'median': float(np.median(valid_dists)),
            'p25': float(np.percentile(valid_dists, 25)),
            'p75': float(np.percentile(valid_dists, 75)),
            'p90': float(np.percentile(valid_dists, 90)),
            'within_threshold': int(np.sum(valid_dists <= distance_threshold)),
        }
    
    # 应用阈值筛选
    distances = np.where(all_distances <= distance_threshold, all_distances, np.inf)
    indices = all_indices
    
    # 处理匹配结果
    matched_r1 = set()
    matched_r2 = set()
    match_distances = []
    
    # 复制集合以进行更新
    updated_keep_points = keep_points.copy()
    updated_discard_r1 = discard_r1.copy()
    updated_discard_r2 = discard_r2.copy()
    updated_point_source_type = point_source_type.copy()
    
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist > distance_threshold or idx >= len(r2_ids):
            continue
        
        r1_pt3d_id = r1_ids[i]
        r2_pt3d_id = r2_ids[idx]
        
        # 避免重复匹配
        if r1_pt3d_id in matched_r1 or r2_pt3d_id in matched_r2:
            continue
        
        matched_r1.add(r1_pt3d_id)
        matched_r2.add(r2_pt3d_id)
        match_distances.append(dist)
        
        r1_pt3d = recon1.points3D[r1_pt3d_id]
        r2_pt3d = recon2.points3D[r2_pt3d_id]
        
        # 获取置信度（使用重投影误差）
        score1 = -r1_pt3d.error
        score2 = -r2_pt3d.error
        
        if blend_mode == 'weighted':
            # 加权融合
            blended_xyz, blended_color, blended_error = _blend_3d_points(
                r1_pt3d.xyz, r2_pt3d.xyz,
                score1, score2,
                r1_pt3d.color, r2_pt3d.color,
                r1_pt3d.error, r2_pt3d.error
            )
            
            if score1 >= score2:
                primary_key = ('r1', r1_pt3d_id)
                # 从 keep_points 中移除 r2 的点
                updated_keep_points.discard(('r2', r2_pt3d_id))
                updated_discard_r2.add(r2_pt3d_id)
            else:
                primary_key = ('r2', r2_pt3d_id)
                # 从 keep_points 中移除 r1 的点
                updated_keep_points.discard(('r1', r1_pt3d_id))
                updated_discard_r1.add(r1_pt3d_id)
            
            blended_points[primary_key] = {
                'xyz': blended_xyz,
                'color': blended_color,
                'error': blended_error,
                'primary_source': primary_key[0],
                'pt3d_id': primary_key[1],
                'pt3d_id1': r1_pt3d_id,
                'pt3d_id2': r2_pt3d_id,
                'match_type': 'aggressive_3d_blend',
                'match_distance': dist,
            }
            updated_point_source_type[primary_key] = 'blended_aggressive'
            
        else:
            # 胜者通吃
            if score1 >= score2:
                # r1 胜出，移除 r2 的点
                updated_keep_points.discard(('r2', r2_pt3d_id))
                updated_discard_r2.add(r2_pt3d_id)
                updated_point_source_type[('r1', r1_pt3d_id)] = 'conflict_aggressive'
            else:
                # r2 胜出，移除 r1 的点
                updated_keep_points.discard(('r1', r1_pt3d_id))
                updated_discard_r1.add(r1_pt3d_id)
                updated_point_source_type[('r2', r2_pt3d_id)] = 'conflict_aggressive'
        
        stats['new_matches'] += 1
    
    if len(match_distances) > 0:
        stats['avg_distance'] = float(np.mean(match_distances))
    
    return updated_keep_points, updated_discard_r1, updated_discard_r2, updated_point_source_type, blended_points, stats


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


def build_2d_3d_correspondences(
    reconstruction: pycolmap.Reconstruction,
    image_ids: List[int],
    verbose: bool = True,
    include_track_pixels: bool = False
) -> Dict[int, Dict]:
    """
    为指定影像建立 2D-3D 对应关系（优化版本）
    
    Args:
        reconstruction: pycolmap Reconstruction 对象
        image_ids: 要处理的影像 ID 列表
        verbose: 是否打印详细信息
        include_track_pixels: 是否包含 track 中每个观测的像素坐标（较慢）
        
    Returns:
        correspondences: {
            image_id: {
                'image_name': str,
                'points2D': [{point2D_idx, pixel_xy, point3D_id}, ...],
                'points3D': {point3D_id: {
                    'xyz': xyz, 
                    'color': color, 
                    'error': error,
                    'track_length': int,
                    'track': [(image_id, point2D_idx), ...] 或包含 pixel_xy
                }, ...}
            }
        }
    """
    correspondences = {}
    total_2d_points = 0
    total_valid_correspondences = 0
    
    # 缓存引用，避免重复属性访问
    points3D_map = reconstruction.points3D
    images_map = reconstruction.images
    
    for img_id in image_ids:
        if img_id not in images_map:
            continue
        
        image = images_map[img_id]
        image_name = image.name
        points2D = image.points2D
        num_points2d = len(points2D)
        total_2d_points += num_points2d
        
        # 预分配列表，使用局部变量加速
        valid_points2D = []
        valid_append = valid_points2D.append  # 缓存 append 方法
        points3D_info = {}
        
        for pt2d_idx in range(num_points2d):
            point2D = points2D[pt2d_idx]
            pt3d_id = point2D.point3D_id
            
            # 快速检查：-1 表示无效
            if pt3d_id == -1:
                continue
            
            # 检查 3D 点是否存在
            if pt3d_id not in points3D_map:
                continue
            
            point3D = points3D_map[pt3d_id]
            
            # 记录 2D 点信息
            valid_append({
                'point2D_idx': pt2d_idx,
                'pixel_xy': point2D.xy,
                'point3D_id': pt3d_id
            })
            
            # 记录 3D 点信息（去重）
            if pt3d_id not in points3D_info:
                track_elems = point3D.track.elements
                track_len = len(track_elems)
                
                if include_track_pixels:
                    # 完整 track（较慢）
                    track_data = []
                    track_append = track_data.append
                    for te in track_elems:
                        te_img_id = te.image_id
                        if te_img_id in images_map:
                            te_img = images_map[te_img_id]
                            track_append({
                                'image_id': te_img_id,
                                'image_name': te_img.name,
                                'point2D_idx': te.point2D_idx,
                                'pixel_xy': te_img.points2D[te.point2D_idx].xy
                            })
                else:
                    # 简化 track（快速）- 使用元组列表
                    track_data = [(te.image_id, te.point2D_idx) for te in track_elems]
                
                points3D_info[pt3d_id] = {
                    'xyz': point3D.xyz,
                    'color': point3D.color,
                    'error': point3D.error,
                    'track_length': track_len,
                    'track': track_data
                }
            
            total_valid_correspondences += 1
        
        correspondences[img_id] = {
            'image_name': image_name,
            'points2D': valid_points2D,
            'points3D': points3D_info,
            'num_valid': len(valid_points2D),
            'num_total_2d': num_points2d
        }
    
    if verbose:
        print(f"\n  2D-3D Correspondences built:")
        print(f"    Images processed: {len(correspondences)}")
        print(f"    Total 2D points: {total_2d_points}")
        print(f"    Valid correspondences: {total_valid_correspondences}")
        for img_id, data in correspondences.items():
            print(f"    Image {img_id} ({data['image_name']}): "
                  f"{data['num_valid']}/{data['num_total_2d']} valid 2D-3D pairs, "
                  f"{len(data['points3D'])} unique 3D points")
    
    return correspondences


def _process_single_image(
    img_id: int,
    images_map,
    points3D_map,
    include_track_pixels: bool
) -> Optional[Tuple[int, Dict]]:
    """处理单个影像的 2D-3D 对应（内部函数，用于并行处理）"""
    if img_id not in images_map:
        return None
    
    image = images_map[img_id]
    image_name = image.name
    points2D = image.points2D
    num_points2d = len(points2D)
    
    valid_points2D = []
    valid_append = valid_points2D.append
    points3D_info = {}
    
    for pt2d_idx in range(num_points2d):
        point2D = points2D[pt2d_idx]
        pt3d_id = point2D.point3D_id
        
        if pt3d_id == -1 or pt3d_id not in points3D_map:
            continue
        
        point3D = points3D_map[pt3d_id]
        
        valid_append({
            'point2D_idx': pt2d_idx,
            'pixel_xy': point2D.xy,
            'point3D_id': pt3d_id
        })
        
        if pt3d_id not in points3D_info:
            track_elems = point3D.track.elements
            track_len = len(track_elems)
            
            if include_track_pixels:
                track_data = []
                for te in track_elems:
                    te_img_id = te.image_id
                    if te_img_id in images_map:
                        te_img = images_map[te_img_id]
                        track_data.append({
                            'image_id': te_img_id,
                            'image_name': te_img.name,
                            'point2D_idx': te.point2D_idx,
                            'pixel_xy': te_img.points2D[te.point2D_idx].xy
                        })
            else:
                track_data = [(te.image_id, te.point2D_idx) for te in track_elems]
            
            points3D_info[pt3d_id] = {
                'xyz': point3D.xyz,
                'color': point3D.color,
                'error': point3D.error,
                'track_length': track_len,
                'track': track_data
            }
    
    return (img_id, {
        'image_name': image_name,
        'points2D': valid_points2D,
        'points3D': points3D_info,
        'num_valid': len(valid_points2D),
        'num_total_2d': num_points2d
    })


def build_2d_3d_correspondences_parallel(
    reconstruction: pycolmap.Reconstruction,
    image_ids: List[int],
    verbose: bool = True,
    include_track_pixels: bool = False,
    max_workers: int = 4
) -> Dict[int, Dict]:
    """
    并行版本：为指定影像建立 2D-3D 对应关系
    
    Args:
        reconstruction: pycolmap Reconstruction 对象
        image_ids: 要处理的影像 ID 列表
        verbose: 是否打印详细信息
        include_track_pixels: 是否包含 track 中每个观测的像素坐标
        max_workers: 最大并行线程数
        
    Returns:
        与 build_2d_3d_correspondences 相同的数据结构
    """
    correspondences = {}
    points3D_map = reconstruction.points3D
    images_map = reconstruction.images
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_image, 
                img_id, images_map, points3D_map, include_track_pixels
            ): img_id 
            for img_id in image_ids
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                img_id, data = result
                correspondences[img_id] = data
    
    if verbose:
        total_2d = sum(d['num_total_2d'] for d in correspondences.values())
        total_valid = sum(d['num_valid'] for d in correspondences.values())
        print(f"\n  2D-3D Correspondences built (parallel, {max_workers} workers):")
        print(f"    Images processed: {len(correspondences)}")
        print(f"    Total 2D points: {total_2d}")
        print(f"    Valid correspondences: {total_valid}")
        for img_id, data in correspondences.items():
            print(f"    Image {img_id} ({data['image_name']}): "
                  f"{data['num_valid']}/{data['num_total_2d']} valid 2D-3D pairs, "
                  f"{len(data['points3D'])} unique 3D points")
    
    return correspondences


def build_pixel_to_3d_mapping(
    correspondences: Dict[int, Dict]
) -> Dict[int, Dict[Tuple[int, int], Dict]]:
    """
    基于 2D-3D 对应关系，建立像素坐标到 3D 点的映射
    
    使用整数像素坐标作为键，方便后续匹配
    
    Args:
        correspondences: build_2d_3d_correspondences 的输出
        
    Returns:
        pixel_mapping: {
            image_id: {
                (pixel_x, pixel_y): {
                    'point3D_id': id,
                    'xyz': xyz,
                    'color': color,
                    'error': error,
                    'track_length': int,
                    'track': [...],  # track 观测信息
                    'exact_xy': exact pixel coordinates
                }
            }
        }
    """
    pixel_mapping = {}
    
    for img_id, data in correspondences.items():
        img_pixel_map = {}
        points3D_info = data['points3D']
        
        for pt2d_info in data['points2D']:
            pt3d_id = pt2d_info['point3D_id']
            pixel_xy = pt2d_info['pixel_xy']
            
            # 使用四舍五入的整数坐标作为键
            px = pixel_xy[0] if hasattr(pixel_xy, '__getitem__') else pixel_xy.x
            py = pixel_xy[1] if hasattr(pixel_xy, '__getitem__') else pixel_xy.y
            pixel_key = (int(round(px)), int(round(py)))
            
            # 获取 3D 点信息
            pt3d_info = points3D_info[pt3d_id]
            
            img_pixel_map[pixel_key] = {
                'point3D_id': pt3d_id,
                'xyz': pt3d_info['xyz'],
                'color': pt3d_info['color'],
                'error': pt3d_info['error'],
                'track_length': pt3d_info['track_length'],
                'track': pt3d_info['track'],
                'exact_xy': pixel_xy
            }
        
        pixel_mapping[img_id] = img_pixel_map
    
    return pixel_mapping


def build_correspondences_parallel(
    recon1: pycolmap.Reconstruction,
    recon2: pycolmap.Reconstruction,
    common_images: Dict[int, int],
    include_track_pixels: bool = False,
    verbose: bool = True
) -> Tuple[Dict[int, Dict], Dict[int, Dict], Dict[int, Dict], Dict[int, Dict]]:
    """
    并行构建两个 reconstruction 的 2D-3D 对应关系和像素映射
    
    Args:
        recon1: 第一个 reconstruction
        recon2: 第二个 reconstruction
        common_images: 共同影像映射 {recon1_image_id: recon2_image_id}
        include_track_pixels: 是否包含 track 中每个观测的像素坐标
        verbose: 是否打印详细信息
        
    Returns:
        correspondences_recon1: recon1 的 2D-3D 对应关系
        correspondences_recon2: recon2 的 2D-3D 对应关系
        pixel_map_recon1: recon1 的像素映射
        pixel_map_recon2: recon2 的像素映射
    """
    import time
    
    common_ids_recon1 = list(common_images.keys())
    common_ids_recon2 = list(common_images.values())
    
    if verbose:
        print(f"\nBuilding 2D-3D correspondences (parallel)...")
    
    t0 = time.time()
    
    # 并行构建 2D-3D 对应关系
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(
            build_2d_3d_correspondences, 
            recon1, common_ids_recon1, False, include_track_pixels
        )
        future2 = executor.submit(
            build_2d_3d_correspondences,
            recon2, common_ids_recon2, False, include_track_pixels
        )
        
        correspondences_recon1 = future1.result()
        correspondences_recon2 = future2.result()
    
    t1 = time.time()
    
    if verbose:
        print(f"  Recon1: {len(correspondences_recon1)} images")
        print(f"  Recon2: {len(correspondences_recon2)} images")
        print(f"  Time: {t1 - t0:.3f}s")
        print(f"\nBuilding pixel-to-3D mappings (parallel)...")
    
    # 并行构建像素映射
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(build_pixel_to_3d_mapping, correspondences_recon1)
        future2 = executor.submit(build_pixel_to_3d_mapping, correspondences_recon2)
        
        pixel_map_recon1 = future1.result()
        pixel_map_recon2 = future2.result()
    
    t2 = time.time()
    
    if verbose:
        print(f"  Time: {t2 - t1:.3f}s")
        print(f"  Total: {t2 - t0:.3f}s")
    
    return correspondences_recon1, correspondences_recon2, pixel_map_recon1, pixel_map_recon2


def find_corresponding_3d_points(
    pixel_map_recon1: Dict[int, Dict],
    pixel_map_recon2: Dict[int, Dict],
    common_images: Dict[int, int],
    correspondences_recon1: Dict[int, Dict],
    correspondences_recon2: Dict[int, Dict],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    基于共同像素位置找到两个 reconstruction 中对应的 3D 点对
    
    Args:
        pixel_map_recon1: recon1 的像素到 3D 点映射
        pixel_map_recon2: recon2 的像素到 3D 点映射
        common_images: 共同影像映射 {recon1_image_id: recon2_image_id}
        correspondences_recon1: recon1 的 2D-3D 对应关系
        correspondences_recon2: recon2 的 2D-3D 对应关系
        verbose: 是否打印详细信息
        
    Returns:
        pts1: recon1 中的 3D 点坐标 (N, 3)
        pts2: recon2 中的 3D 点坐标 (N, 3)
        match_info: 匹配详细信息列表
    """
    pts1_list = []
    pts2_list = []
    match_info = []
    
    # 用于去重：同一个 3D 点对只记录一次
    seen_pairs = set()
    
    for img_id1, img_id2 in common_images.items():
        if img_id1 not in pixel_map_recon1 or img_id2 not in pixel_map_recon2:
            continue
        
        pmap1 = pixel_map_recon1[img_id1]
        pmap2 = pixel_map_recon2[img_id2]
        
        # 找到共同的像素位置
        common_pixels = set(pmap1.keys()) & set(pmap2.keys())
        
        for pixel_key in common_pixels:
            info1 = pmap1[pixel_key]
            info2 = pmap2[pixel_key]
            
            pt3d_id1 = info1['point3D_id']
            pt3d_id2 = info2['point3D_id']
            
            # 去重：同一个 3D 点对只记录一次
            pair_key = (pt3d_id1, pt3d_id2)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            xyz1 = np.array(info1['xyz'])
            xyz2 = np.array(info2['xyz'])
            
            pts1_list.append(xyz1)
            pts2_list.append(xyz2)
            
            match_info.append({
                'pixel': pixel_key,
                'image_id1': img_id1,
                'image_id2': img_id2,
                'point3D_id1': pt3d_id1,
                'point3D_id2': pt3d_id2,
                'xyz1': xyz1,
                'xyz2': xyz2,
                'error1': info1['error'],
                'error2': info2['error']
            })
    
    pts1 = np.array(pts1_list) if pts1_list else np.zeros((0, 3))
    pts2 = np.array(pts2_list) if pts2_list else np.zeros((0, 3))
    
    if verbose:
        print(f"\nFound {len(pts1)} corresponding 3D point pairs")
        if len(pts1) > 0:
            # 计算点对之间的距离统计
            dists = np.linalg.norm(pts1 - pts2, axis=1)
            print(f"  Distance stats: min={dists.min():.4f}, max={dists.max():.4f}, "
                  f"mean={dists.mean():.4f}, std={dists.std():.4f}")
    
    return pts1, pts2, match_info


def estimate_sim3_transform(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    with_scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    估计从 src 到 dst 的 Sim3 变换: dst = scale * R @ src + t
    
    使用 Umeyama 算法
    
    Args:
        pts_src: 源点集 (N, 3)
        pts_dst: 目标点集 (N, 3)
        with_scale: 是否估计尺度
        
    Returns:
        R: 旋转矩阵 (3, 3)
        t: 平移向量 (3,)
        scale: 尺度因子
    """
    assert pts_src.shape == pts_dst.shape
    n = pts_src.shape[0]
    
    # 计算质心
    centroid_src = pts_src.mean(axis=0)
    centroid_dst = pts_dst.mean(axis=0)
    
    # 去质心
    src_centered = pts_src - centroid_src
    dst_centered = pts_dst - centroid_dst
    
    # 计算协方差矩阵
    H = src_centered.T @ dst_centered / n
    
    # SVD 分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = Vt.T @ U.T
    
    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算尺度
    if with_scale:
        var_src = np.sum(src_centered ** 2) / n
        scale = np.sum(S) / var_src if var_src > 1e-10 else 1.0
    else:
        scale = 1.0
    
    # 计算平移
    t = centroid_dst - scale * R @ centroid_src
    
    return R, t, scale


def estimate_sim3_ransac(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    max_iterations: int = 1000,
    inlier_threshold: float = 10,
    min_inliers: int = 5,
    min_sample_size: int = 3,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    使用 RANSAC 鲁棒估计 Sim3 变换
    
    Args:
        pts_src: 源点集 (N, 3)
        pts_dst: 目标点集 (N, 3)
        max_iterations: 最大迭代次数
        inlier_threshold: 内点阈值（距离）
        min_inliers: 最小内点数
        min_sample_size: 每次迭代采样的点数（最小 3，更大值更稳定但需要更多迭代）
        verbose: 是否打印信息
        
    Returns:
        R: 最佳旋转矩阵 (3, 3)
        t: 最佳平移向量 (3,)
        scale: 最佳尺度因子
        inlier_mask: 内点掩码 (N,)
    """
    n = pts_src.shape[0]
    min_sample_size = max(3, min(min_sample_size, n))  # 确保在有效范围内
    
    if n < min_sample_size:
        raise ValueError(f"Need at least {min_sample_size} points, got {n}")
    
    best_inliers = 0
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_scale = 1.0
    best_mask = np.zeros(n, dtype=bool)
    
    for _ in range(max_iterations):
        # 随机选择 min_sample_size 个点
        idx = np.random.choice(n, min_sample_size, replace=False)
        
        try:
            R, t, scale = estimate_sim3_transform(pts_src[idx], pts_dst[idx])
        except:
            continue
        
        # 计算所有点的变换误差
        pts_transformed = scale * (R @ pts_src.T).T + t
        errors = np.linalg.norm(pts_transformed - pts_dst, axis=1)
        
        # 统计内点
        inlier_mask = errors < inlier_threshold
        num_inliers = np.sum(inlier_mask)
        
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_R = R
            best_t = t
            best_scale = scale
            best_mask = inlier_mask
    
    # 使用所有内点重新估计
    if best_inliers >= min_inliers:
        best_R, best_t, best_scale = estimate_sim3_transform(
            pts_src[best_mask], pts_dst[best_mask]
        )
        # 更新内点
        pts_transformed = best_scale * (best_R @ pts_src.T).T + best_t
        errors = np.linalg.norm(pts_transformed - pts_dst, axis=1)
        best_mask = errors < inlier_threshold
    
    if verbose:
        print(f"\nSim3 RANSAC result:")
        print(f"  Inliers: {np.sum(best_mask)}/{n} ({100*np.sum(best_mask)/n:.1f}%)")
        print(f"  Scale: {best_scale:.6f}")
        if np.sum(best_mask) > 0:
            pts_transformed = best_scale * (best_R @ pts_src.T).T + best_t
            errors = np.linalg.norm(pts_transformed - pts_dst, axis=1)
            print(f"  Inlier error: mean={errors[best_mask].mean():.6f}, "
                  f"max={errors[best_mask].max():.6f}")
    
    return best_R, best_t, best_scale, best_mask


def apply_sim3_to_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    R: np.ndarray,
    t: np.ndarray,
    scale: float
) -> None:
    """
    将 Sim3 变换应用到 reconstruction（原地修改）
    
    变换公式: new_pos = scale * R @ old_pos + t
    
    Args:
        reconstruction: pycolmap Reconstruction 对象
        R: 旋转矩阵 (3, 3)
        t: 平移向量 (3,)
        scale: 尺度因子
    """
    # 变换所有 3D 点
    for pt3d_id in list(reconstruction.points3D.keys()):
        pt3d = reconstruction.points3D[pt3d_id]
        old_xyz = np.array(pt3d.xyz)
        new_xyz = scale * (R @ old_xyz) + t
        pt3d.xyz = new_xyz
    
    # 变换所有相机位姿
    for img_id in list(reconstruction.images.keys()):
        image = reconstruction.images[img_id]
        # 获取相机中心（世界坐标系）
        old_center = image.projection_center()
        new_center = scale * (R @ old_center) + t
        
        # 更新旋转：new_R_cam = old_R_cam @ R^T
        old_qvec = image.cam_from_world.rotation.quat
        old_R_cam = image.cam_from_world.rotation.matrix()
        new_R_cam = old_R_cam @ R.T
        
        # 从新的旋转矩阵和中心计算新的 cam_from_world
        new_tvec = -new_R_cam @ new_center
        
        # 更新 image 的位姿
        image.cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(new_R_cam),
            new_tvec
        )


def merge_reconstructions_by_confidence(
    recon1: pycolmap.Reconstruction,
    recon2: pycolmap.Reconstruction,
    pixel_map_recon1: Dict[int, Dict],
    pixel_map_recon2: Dict[int, Dict],
    common_images: Dict[int, int],
    prev_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    curr_recon_conf: Optional[Dict[int, np.ndarray]] = None,
    image_name_to_idx: Optional[Dict[str, int]] = None,
    color_by_source: bool = False,  # 新增：是否按来源着色用于可视化
    match_radii: Optional[List[float]] = None,  # 多级2D匹配搜索半径列表（像素）
    match_3d_threshold: float = 0.5,  # 3D空间匹配距离阈值（单位与点云坐标一致）
    aggressive_3d_threshold: float = 0.0,  # 最终阶段激进3D匹配阈值，0表示禁用
    inner_blend_margin: float = 150.0,  # 融合带向重叠区内部延伸的宽度（像素）【增大以扩大过渡区域】
    outer_blend_margin: float = 200.0,  # 融合带向外部延伸的宽度（像素）【增大以扩大过渡区域】
    blend_mode: str = 'winner',  # 融合模式：'winner' 或 'weighted'
    keep_unmatched_overlap: bool = False,  # 是否保留重叠区所有未匹配点
    spatial_blend_interpolation: bool = True,  # 是否启用融合带3D坐标空间插值
    spatial_blend_k_neighbors: int = 32,  # 空间插值使用的近邻数【增大以更稳定】
    spatial_blend_smooth_transition: bool = True,  # 是否使用 smoothstep 实现更平滑过渡
    spatial_blend_smooth_power: float = 0.5,  # 平滑过渡力度（<1更强，>1更弱）【减小以更强平滑】
    density_equalization: bool = False,  # 是否启用密度均衡化
    density_k_neighbors: int = 10,  # 密度计算使用的近邻数
    density_target_percentile: float = 50.0,  # 目标密度百分位数（重叠区）
    density_tolerance: float = 1.2,  # 密度容差倍数
    density_use_grid: bool = True,  # 是否使用网格采样（更稳定均匀）
    density_grid_resolution: float = 1.0,  # 网格分辨率因子（<1更精细，>1更粗糙）
    density_distance_decay: float = 0.5,  # 距离衰减因子（0=无衰减，1=强衰减）
    voxel_size: float = 0.0,  # 体素降采样大小，0表示不降采样
    verbose: bool = True,
) -> Tuple[pycolmap.Reconstruction, Dict]:
    """
    基于置信度合并两个 reconstruction，构建新的 reconstruction 对象（双边界平滑融合版）
    
    合并逻辑（使用多级半径匹配 + 双边界平滑融合）：
    - 对于重叠影像，使用多级 match_radii 策略进行2D点匹配：
      - 从小半径到大半径依次匹配
      - 小半径优先匹配高置信度的精确对应
      - 大半径补充匹配边缘和难以对齐的点
      - 已匹配的点不再参与后续匹配，保证一对一
    - 根据 blend_mode 处理匹配到的点对：
      - 'winner': 选择置信度更高的点（胜者通吃）
      - 'weighted': 基于置信度加权平均计算新的3D点位置（平滑融合）
    - 未匹配点处理（取决于 keep_unmatched_overlap 参数）：
      - keep_unmatched_overlap=False（默认）：三区域策略实现平滑过渡
        1. 核心重叠区（内边界内）：丢弃未匹配点
        2. 融合带（内边界外，外边界内）：保留未匹配点（平滑过渡）
        3. 非重叠区（外边界外）：保留未匹配点
      - keep_unmatched_overlap=True：保留所有未匹配点（无边缘处理）
    
    边界说明（以匹配点凸包为基准，仅当 keep_unmatched_overlap=False 时生效）：
                        outer boundary (外边界)
                            |
                            v
        +-------------------+-------------------+
        |   non-overlap     |   non-overlap     |  <- 保留
        |   (keep)          |   (keep)          |
        +-------------------+-------------------+
        |                   |                   |
        |   outer blend     |   outer blend     |  <- 融合带外层（保留）
        |   zone (keep)     |   zone (keep)     |      outer_blend_margin
        |                   |                   |
        +-------------------+-------------------+ <- 匹配点凸包边界
        |                   |                   |
        |   inner blend     |   inner blend     |  <- 融合带内层（保留）
        |   zone (keep)     |   zone (keep)     |      inner_blend_margin
        |                   |                   |
        +-------------------+-------------------+ <- inner boundary (内边界)
        |                                       |
        |          core discard zone            |  <- 核心丢弃区（丢弃）
        |          (discard unmatched)          |
        |                                       |
        +---------------------------------------+
    
    置信度来源（按优先级）：
    1. 像素级置信度图（prev_recon_conf / curr_recon_conf）
    2. 重投影误差（越小越好，作为回退）
    
    Args:
        recon1: 第一个 reconstruction（基准，prev_recon）
        recon2: 第二个 reconstruction（已对齐到 recon1，curr_recon）
        pixel_map_recon1: recon1 的像素映射 {img_id: {(x,y): {...}}}
        pixel_map_recon2: recon2 的像素映射
        common_images: 共同影像映射 {recon1_img_id: recon2_img_id}
        prev_recon_conf: recon1 的像素级置信度图 {global_img_idx: (H, W) array}
        curr_recon_conf: recon2 的像素级置信度图 {global_img_idx: (H, W) array}
        image_name_to_idx: 图像名称到全局索引的映射
        color_by_source: 是否按来源着色用于可视化测试
            - recon1 独有的点: 蓝色 (0, 0, 255)
            - recon2 独有的点: 绿色 (0, 255, 0)
            - 冲突后选择/融合的点: 红色 (255, 0, 0)
            - 融合带保留的点: 黄色/青色
        match_radii: 多级2D点匹配搜索半径列表（像素）
            - 默认 [3, 5, 10, 20, 50]
            - 从小到大依次匹配，小半径优先匹配精确对应
            - 大半径补充匹配边缘点，实现更全面的覆盖
        aggressive_3d_threshold: 最终阶段激进3D匹配阈值
            - 用于对重叠区内仍然独有的点（蓝色/绿色）进行最后一轮3D空间匹配
            - 使用比 match_3d_threshold 更宽松的阈值
            - 可以显著减少重叠区内的独有点，将它们融合
            - 默认 0 表示禁用
            - 建议值: 1.5-3.0（是 match_3d_threshold 的 1.5-3 倍）
        inner_blend_margin: 融合带向重叠区内部延伸的宽度（像素）
            - 控制融合带向匹配点凸包内部延伸多少
            - 值越大，核心丢弃区越小，保留更多重叠区内的点
            - 默认80像素，让边界附近的点有更好的过渡
            - 建议范围：50-150像素
            - 仅当 keep_unmatched_overlap=False 时生效
            - 同时控制空间插值的范围（融合带内的点会根据blend_weight进行3D坐标插值）
        outer_blend_margin: 融合带向外部（非重叠区方向）延伸的宽度（像素）
            - 控制融合带向匹配点凸包外部延伸多少
            - 值越大，融合带越宽，过渡越平滑
            - 默认100像素
            - 建议范围：50-150像素
            - 仅当 keep_unmatched_overlap=False 时生效
            - 同时控制空间插值的范围（融合带内的点会根据blend_weight进行3D坐标插值）
        blend_mode: 3D点融合模式
            - 'winner': 胜者通吃模式（默认）
              选择置信度更高的点，丢弃另一个
            - 'weighted': 加权平均模式
              基于置信度权重计算新的3D点位置: 
              xyz_new = (w1*xyz1 + w2*xyz2) / (w1+w2)
              实现更平滑的3D点云过渡
        keep_unmatched_overlap: 是否保留重叠区所有未匹配点
            - False（默认）：启用边缘处理，使用双边界平滑融合
            - True：禁用边缘处理，保留重叠区所有未匹配点
              适用于不需要边缘优化的场景，或想保留更多点云细节
        spatial_blend_interpolation: 是否启用融合带3D坐标空间插值
            - True（默认）：对融合带内的独有点进行3D坐标插值
              基于周围已融合点对的位移，使用 IDW 插值估算偏移
              插值强度由 blend_weight 决定（来自 inner/outer_blend_margin）
              实现从重叠区到非重叠区的平滑3D坐标过渡
            - False：禁用空间插值，融合带点保持原始坐标
        spatial_blend_k_neighbors: 空间插值使用的近邻数
            - 默认 16
            - 用于 IDW 插值的 K 近邻数量
            - 值越大，插值越平滑但计算量越大
        spatial_blend_smooth_transition: 是否使用 smoothstep 实现更平滑过渡
            - True（默认）：使用 smootherstep 函数，过渡更平滑自然
            - False：使用线性过渡
        spatial_blend_smooth_power: 平滑过渡的力度参数（默认 1.0）
            - < 1.0：过渡更"宽"，更多的点获得更大的位移，平滑效果更强
            - = 1.0：标准 smootherstep 过渡
            - > 1.0：过渡更"窄"，只有靠近核心区的点才有显著位移
            - 建议值：0.3-0.7 可获得更强的平滑效果
        density_equalization: 是否启用密度均衡化（默认 False）
            - True：分析重叠区点密度，对非重叠区进行稀疏化处理
              使非重叠区的点密度与重叠区一致，实现均匀的点云密度
            - False：不进行密度均衡化处理
        density_k_neighbors: 密度计算使用的 K 近邻数（默认 10）
            - 用于计算每个点的局部密度（1 / 平均近邻距离）
            - 值越大，密度估计越稳定但计算量越大
        density_target_percentile: 目标密度百分位数（默认 50，即中位数）
            - 使用重叠区点密度的该百分位数作为目标密度
            - 值越小（如 25），目标密度越低，会丢弃更多高密度区的点
            - 值越大（如 75），目标密度越高，保留更多点
        density_tolerance: 密度容差倍数（默认 1.2）
            - 只有密度 > target_density * tolerance 的点才考虑丢弃
            - 值越大越宽容，丢弃的点越少
            - 建议值：1.1-1.5
        voxel_size: 体素降采样大小（与点云坐标单位一致，通常为米）
            - 默认 0.0 表示不进行降采样
            - 正值时会在合并完成后对点云进行体素降采样
            - 每个体素保留一个代表点（质心），减少点云密度
            - 建议值: 0.5-2.0 米，根据点云精度需求调整
        verbose: 是否打印详细信息
        
    Returns:
        merged_recon: 合并后的 pycolmap.Reconstruction 对象
        info: 合并统计信息
    """
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
    match_radii = sorted(match_radii)  # 确保从小到大排序
    
    # ========== 预计算置信度缓存（避免重复查找）==========
    # {img_id: (conf_map, H, W)} 或 {img_id: None}
    _conf_cache_r1 = {}
    _conf_cache_r2 = {}
    
    for img_id in recon1.images:
        img_name = recon1.images[img_id].name
        global_idx = image_name_to_idx.get(img_name)
        if global_idx is not None and global_idx in prev_recon_conf:
            conf_map = prev_recon_conf[global_idx]
            _conf_cache_r1[img_id] = (conf_map, conf_map.shape[0], conf_map.shape[1])
        else:
            _conf_cache_r1[img_id] = None
    
    for img_id in recon2.images:
        img_name = recon2.images[img_id].name
        global_idx = image_name_to_idx.get(img_name)
        if global_idx is not None and global_idx in curr_recon_conf:
            conf_map = curr_recon_conf[global_idx]
            _conf_cache_r2[img_id] = (conf_map, conf_map.shape[0], conf_map.shape[1])
        else:
            _conf_cache_r2[img_id] = None
    
    # ========== 统计信息 ==========
    stats = {
        'pixels_only_r1': 0,
        'pixels_only_r2': 0,
        'pixels_both_choose_r1': 0,
        'pixels_both_choose_r2': 0,
        'pixels_neither': 0,
        'matched_pairs': 0,  # 通过邻域搜索匹配到的点对数
        'avg_match_distance': 0.0,  # 平均匹配距离
        'discarded_unmatched_r1': 0,  # 重叠区内丢弃的未匹配 r1 点
        'discarded_unmatched_r2': 0,  # 重叠区内丢弃的未匹配 r2 点
        'match_radii': match_radii,  # 使用的多级匹配半径
        'matches_per_radius': {},  # 每个半径匹配到的点对数
    }
    
    # ========== 第一步：为每个 3D 点决定是否保留 ==========
    # 记录要保留的 3D 点: {('r1', pt3d_id) or ('r2', pt3d_id)}
    keep_points = set()
    
    # 新增：记录点的来源类型（用于颜色可视化）
    # 'only': 独有的点, 'conflict': 冲突后选择的点
    point_source_type = {}  # (source, pt3d_id) -> 'only' | 'conflict'
    
    # 记录被淘汰的 3D 点（用于后续检查 track 冲突）
    discard_r1 = set()  # recon1 中被淘汰的点
    discard_r2 = set()  # recon2 中被淘汰的点
    
    # 遍历共同影像的所有像素
    common_img_ids_r1 = set(common_images.keys())
    common_img_ids_r2 = set(common_images.values())
    
    # 记录匹配距离用于统计
    match_distances = []
    
    # 新增：记录每张影像中匹配成功的2D点范围（用于判断边界外的点）
    # {img_id: {'matched_pixels': [(x,y), ...], 'hull': ConvexHull or None}}
    matched_regions_r1 = {}
    matched_regions_r2 = {}
    
    # 加权融合模式：存储融合后的点信息
    # {(pt3d_id1, pt3d_id2): {'xyz': blended_xyz, 'color': blended_color, 'error': blended_error,
    #                         'primary_source': 'r1'/'r2', 'pt3d_id': id, 'conf1': c1, 'conf2': c2}}
    blended_points = {}
    use_weighted_blend = (blend_mode == 'weighted')
    
    # 添加融合模式统计
    stats['blend_mode'] = blend_mode
    stats['weighted_blended_pairs'] = 0
    
    # 并行处理所有影像对
    image_pairs = list(common_images.items())
    
    # 预先初始化所有共同影像的 matched_regions（确保即使 2D 匹配失败，pmap 也会被处理）
    for img_id1, img_id2 in image_pairs:
        pmap1 = pixel_map_recon1.get(img_id1, {})
        pmap2 = pixel_map_recon2.get(img_id2, {})
        if len(pmap1) > 0:
            matched_regions_r1[img_id1] = {
                'matched_pixels': [],
                'pmap': pmap1,
                'processed_points': set()
            }
        if len(pmap2) > 0:
            matched_regions_r2[img_id2] = {
                'matched_pixels': [],
                'pmap': pmap2,
                'processed_points': set()
            }
    
    # 使用线程池并行处理（多级半径匹配）
    with ThreadPoolExecutor(max_workers=min(8, len(image_pairs) or 1)) as executor:
        results = list(executor.map(
            lambda p: _process_image_pair_for_merge(p[0], p[1], pixel_map_recon1, pixel_map_recon2, match_radii), 
            image_pairs
        ))
    
    # 合并结果并进行置信度比较/加权融合
    # 累计每个半径的匹配统计
    total_radius_stats = {r: 0 for r in match_radii}
    
    for result in results:
        if result is None:
            continue
        
        img_id1, img_id2, matches, matched_pixels_r1, matched_pixels_r2, pmap1, pmap2, radius_stats = result
        
        # 累计半径匹配统计
        for r, count in radius_stats.items():
            if r in total_radius_stats:
                total_radius_stats[r] += count
        
        processed_r1_points = set()
        processed_r2_points = set()
        
        for pixel_key1, pixel_key2, info1, info2, pt3d_id1, pt3d_id2, dist, match_radius in matches:
            # 获取置信度分数
            score1 = _get_confidence_score(info1, img_id1, pixel_key1, _conf_cache_r1)
            score2 = _get_confidence_score(info2, img_id2, pixel_key2, _conf_cache_r2)
            
            if use_weighted_blend:
                # ========== 加权平均模式 ==========
                # 计算融合后的3D点位置、颜色和误差
                blended_xyz, blended_color, blended_error = _blend_3d_points(
                    info1['xyz'], info2['xyz'],
                    score1, score2,
                    info1['color'], info2['color'],
                    info1['error'], info2['error']
                )
                
                # 选择置信度更高的点作为主源（用于获取track等信息）
                if score1 >= score2:
                    primary_source = 'r1'
                    primary_pt3d_id = pt3d_id1
                    stats['pixels_both_choose_r1'] += 1
                else:
                    primary_source = 'r2'
                    primary_pt3d_id = pt3d_id2
                    stats['pixels_both_choose_r2'] += 1
                
                # 存储融合信息（使用主源的点ID作为key）
                blend_key = (primary_source, primary_pt3d_id)
                blended_points[blend_key] = {
                    'xyz': blended_xyz,
                    'color': blended_color,
                    'error': blended_error,
                    'primary_source': primary_source,
                    'pt3d_id': primary_pt3d_id,
                    'conf1': score1,
                    'conf2': score2,
                    'pt3d_id1': pt3d_id1,
                    'pt3d_id2': pt3d_id2,
                }
                
                # 标记两个点都已处理
                keep_points.add(blend_key)
                point_source_type[blend_key] = 'blended'
                discard_r1.add(pt3d_id1)  # 标记为已处理
                discard_r2.add(pt3d_id2)  # 标记为已处理
                stats['weighted_blended_pairs'] += 1
                
            else:
                # ========== 胜者通吃模式 ==========
                if score1 >= score2:
                    key = ('r1', pt3d_id1)
                    keep_points.add(key)
                    point_source_type[key] = 'conflict'
                    discard_r2.add(pt3d_id2)
                    stats['pixels_both_choose_r1'] += 1
                else:
                    key = ('r2', pt3d_id2)
                    keep_points.add(key)
                    point_source_type[key] = 'conflict'
                    discard_r1.add(pt3d_id1)
                    stats['pixels_both_choose_r2'] += 1
            
            processed_r1_points.add(pt3d_id1)
            processed_r2_points.add(pt3d_id2)
            match_distances.append(dist)
            stats['matched_pairs'] += 1
        
        # 更新该影像的匹配区域信息（已在上面预先初始化）
        if img_id1 in matched_regions_r1:
            matched_regions_r1[img_id1]['matched_pixels'] = matched_pixels_r1
            matched_regions_r1[img_id1]['processed_points'].update(processed_r1_points)
        if img_id2 in matched_regions_r2:
            matched_regions_r2[img_id2]['matched_pixels'] = matched_pixels_r2
            matched_regions_r2[img_id2]['processed_points'].update(processed_r2_points)
    
    # ========== 1.5 步：3D 空间匹配（补充 2D 匹配）==========
    # 关键：对于那些在重叠影像的 pmap 中存在但 2D 匹配失败的点，尝试在 3D 空间中匹配
    # 这些点在 2D 像素空间中可能偏差较大（超过 match_radius），但在 3D 空间中可能很接近
    
    # 执行 3D 匹配（直接从 matched_regions 的 pmap 中收集未匹配的点）
    stats['match_3d'] = {'r1_candidates': 0, 'r2_candidates': 0, 'matched_3d': 0}
    
    if match_3d_threshold > 0:
        r1_keep_3d, r2_keep_3d, r1_discard_3d, r2_discard_3d, match_3d_result = _match_unmatched_points_3d(
            recon1, recon2,
            matched_regions_r1,  # 直接传入匹配区域信息
            matched_regions_r2,
            _conf_cache_r1, _conf_cache_r2,
            distance_threshold_3d=match_3d_threshold,
            blend_mode=blend_mode,
        )
        
        # 更新统计
        stats['match_3d'] = match_3d_result['stats']
        
        # 收集 3D 匹配处理的点（用于后续更新 matched_regions）
        processed_by_3d_r1 = set()
        processed_by_3d_r2 = set()
        
        # 处理 3D 匹配结果
        for pt3d_id in r1_keep_3d:
            key = ('r1', pt3d_id)
            keep_points.add(key)
            point_source_type[key] = 'match_3d'
            processed_by_3d_r1.add(pt3d_id)
        
        for pt3d_id in r2_keep_3d:
            key = ('r2', pt3d_id)
            keep_points.add(key)
            point_source_type[key] = 'match_3d'
            processed_by_3d_r2.add(pt3d_id)
        
        # 更新丢弃集合
        discard_r1.update(r1_discard_3d)
        discard_r2.update(r2_discard_3d)
        processed_by_3d_r1.update(r1_discard_3d)
        processed_by_3d_r2.update(r2_discard_3d)
        
        # 处理加权融合的点
        for key, blend_info in match_3d_result['pairs'].items():
            if 'xyz' in blend_info:  # 加权融合
                keep_points.add(key)
                blended_points[key] = {
                    'xyz': blend_info['xyz'],
                    'color': blend_info['color'],
                    'error': blend_info['error'],
                    'primary_source': key[0],
                    'pt3d_id': key[1],
                    'pt3d_id1': blend_info['pt3d_id1'],
                    'pt3d_id2': blend_info['pt3d_id2'],
                }
                point_source_type[key] = 'blended_3d'
                stats['weighted_blended_pairs'] += 1
                # 两边的点都标记为已处理
                processed_by_3d_r1.add(blend_info['pt3d_id1'])
                processed_by_3d_r2.add(blend_info['pt3d_id2'])
            else:
                # 胜者通吃已经在上面处理了
                point_source_type[key] = 'conflict_3d'
        
        # 更新 matched_regions 中的 processed_points（让后续 _process_unmatched_points 不再处理这些点）
        for region in matched_regions_r1.values():
            region['processed_points'].update(processed_by_3d_r1)
        for region in matched_regions_r2.values():
            region['processed_points'].update(processed_by_3d_r2)
    
    # ========== 第二遍：处理未匹配的点 ==========
    # 添加融合带统计
    stats['blend_zone_r1'] = 0
    stats['blend_zone_r2'] = 0
    stats['keep_unmatched_overlap'] = keep_unmatched_overlap
    
    # 并行处理 recon1 和 recon2 的未匹配点
    all_regions = [
        (region_info, 'r1') for region_info in matched_regions_r1.values()
    ] + [
        (region_info, 'r2') for region_info in matched_regions_r2.values()
    ]
    
    if len(all_regions) > 0:
        # 初始化插值权重字典（在所有分支之前）
        all_blend_weights = {}
        
        if keep_unmatched_overlap:
            # ========== 保留所有未匹配点（无边缘处理）==========
            # 直接收集所有未处理的点，不做任何丢弃
            r1_keep = []
            r2_keep = []
            
            for region_info, source in all_regions:
                pmap = region_info['pmap']
                processed_points = region_info['processed_points']
                
                for pixel_key, info in pmap.items():
                    pt3d_id = info['point3D_id']
                    if pt3d_id not in processed_points:
                        if source == 'r1':
                            r1_keep.append(pt3d_id)
                        else:
                            r2_keep.append(pt3d_id)
            
            # 更新统计
            stats['discarded_unmatched_r1'] = 0
            stats['discarded_unmatched_r2'] = 0
            
            # 更新 keep_points
            for pt3d_id in r1_keep:
                key = ('r1', pt3d_id)
                keep_points.add(key)
                if key not in point_source_type:
                    point_source_type[key] = 'only'
                all_blend_weights[key] = 1.0  # 无边缘处理时权重都为1
            stats['pixels_only_r1'] = len(r1_keep)
            
            for pt3d_id in r2_keep:
                key = ('r2', pt3d_id)
                keep_points.add(key)
                if key not in point_source_type:
                    point_source_type[key] = 'only'
                all_blend_weights[key] = 1.0  # 无边缘处理时权重都为1
            stats['pixels_only_r2'] = len(r2_keep)
            
        else:
            # ========== 双边界平滑融合（边缘处理）==========
            # 双边界策略实现平滑过渡：
            # - 内边界（收缩）：在此范围内的未匹配点被丢弃（核心重叠区）
            # - 外边界（扩展）：在此范围外的点被保留（非重叠区）
            # - 两边界之间：融合带，保留这些点以实现平滑过渡
            
            # 使用独立的内外边界参数（更直观的控制）
            inner_shrink = inner_blend_margin  # 内边界收缩量（向重叠区内部延伸）
            outer_expand = outer_blend_margin  # 外边界扩展量（向外部延伸）
            
            with ThreadPoolExecutor(max_workers=min(8, len(all_regions))) as executor:
                unmatched_results = list(executor.map(
                    lambda x: _process_unmatched_points(x[0], x[1], inner_shrink, outer_expand, 'interpolate'), 
                    all_regions
                ))
            
            # 批量合并结果（优化版：分离 r1 和 r2 结果，减少分支判断）
            r1_keep = []
            r1_discard = []
            r1_blend = []
            r1_blend_weights = {}  # 新增：收集插值权重
            r2_keep = []
            r2_discard = []
            r2_blend = []
            r2_blend_weights = {}  # 新增：收集插值权重
            
            for result in unmatched_results:
                if result['source'] == 'r1':
                    r1_keep.extend(result['keep'])
                    r1_discard.extend(result['discard'])
                    r1_blend.extend(result.get('blend', []))
                    r1_blend_weights.update(result.get('blend_weights', {}))
                else:
                    r2_keep.extend(result['keep'])
                    r2_discard.extend(result['discard'])
                    r2_blend.extend(result.get('blend', []))
                    r2_blend_weights.update(result.get('blend_weights', {}))
            
            # 批量更新 discard 集合
            discard_r1.update(r1_discard)
            discard_r2.update(r2_discard)
            stats['discarded_unmatched_r1'] = len(r1_discard)
            stats['discarded_unmatched_r2'] = len(r2_discard)
            stats['blend_zone_r1'] = len(r1_blend)
            stats['blend_zone_r2'] = len(r2_blend)
            
            # 批量更新 keep_points 和 point_source_type
            # 融合带的点标记为 'blend' 类型
            for pt3d_id in r1_keep:
                key = ('r1', pt3d_id)
                keep_points.add(key)
                if key not in point_source_type:
                    if pt3d_id in r1_blend:
                        point_source_type[key] = 'blend'
                    else:
                        point_source_type[key] = 'only'
            stats['pixels_only_r1'] = len(r1_keep)
            
            for pt3d_id in r2_keep:
                key = ('r2', pt3d_id)
                keep_points.add(key)
                if key not in point_source_type:
                    if pt3d_id in r2_blend:
                        point_source_type[key] = 'blend'
                    else:
                        point_source_type[key] = 'only'
            stats['pixels_only_r2'] = len(r2_keep)
            
            # 合并所有插值权重
            all_blend_weights = {}
            for pt3d_id, weight in r1_blend_weights.items():
                all_blend_weights[('r1', pt3d_id)] = weight
            for pt3d_id, weight in r2_blend_weights.items():
                all_blend_weights[('r2', pt3d_id)] = weight
    
    # 计算平均匹配距离
    if len(match_distances) > 0:
        stats['avg_match_distance'] = float(np.mean(match_distances))
    
    # 存储每个半径的匹配统计
    stats['matches_per_radius'] = total_radius_stats
    
    # ========== 第二步：处理 track 中没有共同影像观测的 3D 点 ==========
    # 这些点可能在空间上位于重叠区边界，需要通过投影来检查
    # 如果投影落在匹配区域内，尝试找到对应的3D点进行融合
    
    # 预提取已保留的点 ID
    kept_r1_ids = {pid for (src, pid) in keep_points if src == 'r1'}
    kept_r2_ids = {pid for (src, pid) in keep_points if src == 'r2'}
    
    # 初始化统计
    stats['non_overlap_track_r1'] = {
        'projected': 0, 'valid_proj': 0, 'in_region': 0, 'new_matches': 0, 'kept_outside': 0
    }
    stats['non_overlap_track_r2'] = {
        'projected': 0, 'valid_proj': 0, 'in_region': 0, 'new_matches': 0, 'kept_outside': 0
    }
    
    # 计算边界参数
    inner_shrink = inner_blend_margin if not keep_unmatched_overlap else 0
    outer_expand = outer_blend_margin if not keep_unmatched_overlap else float('inf')
    
    # 使用最大匹配半径进行投影点的匹配
    max_match_radius = max(match_radii)
    
    # 处理 recon1 中 track 没有共同影像观测的点
    r1_keep_proj, r1_discard_proj, r1_matched_proj, r1_proj_stats = _process_non_overlap_track_points(
        recon1, recon2,
        common_img_ids_r1, common_images,
        pixel_map_recon2, matched_regions_r2,
        _conf_cache_r1, _conf_cache_r2,
        kept_r1_ids.copy(), discard_r1.copy(),
        max_match_radius, inner_shrink, outer_expand,
        keep_unmatched_overlap, 'r1', blend_mode
    )
    
    # 处理 recon2 中 track 没有共同影像观测的点
    # 注意：common_images 的方向需要反转
    common_images_r2_to_r1 = {v: k for k, v in common_images.items()}
    r2_keep_proj, r2_discard_proj, r2_matched_proj, r2_proj_stats = _process_non_overlap_track_points(
        recon2, recon1,
        common_img_ids_r2, common_images_r2_to_r1,
        pixel_map_recon1, matched_regions_r1,
        _conf_cache_r2, _conf_cache_r1,
        kept_r2_ids.copy(), discard_r2.copy(),
        max_match_radius, inner_shrink, outer_expand,
        keep_unmatched_overlap, 'r2', blend_mode
    )
    
    # 更新统计
    stats['non_overlap_track_r1'] = r1_proj_stats
    stats['non_overlap_track_r2'] = r2_proj_stats
    
    # 更新 keep_points 和 discard 集合
    for pt3d_id in r1_keep_proj:
        key = ('r1', pt3d_id)
        if key not in keep_points:
            keep_points.add(key)
            point_source_type[key] = 'projected'  # 新类型：通过投影处理的点
    
    for pt3d_id in r2_keep_proj:
        key = ('r2', pt3d_id)
        if key not in keep_points:
            keep_points.add(key)
            point_source_type[key] = 'projected'
    
    discard_r1.update(r1_discard_proj)
    discard_r2.update(r2_discard_proj)
    
    # 处理新匹配到的点对
    for src_pt3d_id, dst_pt3d_id, blend_info in r1_matched_proj:
        if blend_info is not None:
            # 加权融合模式
            key = ('r1', src_pt3d_id)
            keep_points.add(key)
            blended_points[key] = {
                'xyz': blend_info['xyz'],
                'color': blend_info['color'],
                'error': blend_info['error'],
                'primary_source': 'r1',
                'pt3d_id': src_pt3d_id,
                'conf1': blend_info['conf_src'],
                'conf2': blend_info['conf_dst'],
                'pt3d_id1': src_pt3d_id,
                'pt3d_id2': dst_pt3d_id,
            }
            point_source_type[key] = 'blended_projected'
            stats['weighted_blended_pairs'] += 1
        else:
            # 胜者通吃模式，src 点胜出
            key = ('r1', src_pt3d_id)
            keep_points.add(key)
            point_source_type[key] = 'conflict_projected'
        discard_r2.add(dst_pt3d_id)
    
    for src_pt3d_id, dst_pt3d_id, blend_info in r2_matched_proj:
        if blend_info is not None:
            # 加权融合模式
            key = ('r2', src_pt3d_id)
            keep_points.add(key)
            blended_points[key] = {
                'xyz': blend_info['xyz'],
                'color': blend_info['color'],
                'error': blend_info['error'],
                'primary_source': 'r2',
                'pt3d_id': src_pt3d_id,
                'conf1': blend_info['conf_dst'],  # 注意：这里 conf 顺序
                'conf2': blend_info['conf_src'],
                'pt3d_id1': dst_pt3d_id,
                'pt3d_id2': src_pt3d_id,
            }
            point_source_type[key] = 'blended_projected'
            stats['weighted_blended_pairs'] += 1
        else:
            # 胜者通吃模式，src 点胜出
            key = ('r2', src_pt3d_id)
            keep_points.add(key)
            point_source_type[key] = 'conflict_projected'
        discard_r1.add(dst_pt3d_id)
    
    # ========== 2.5 步：最终阶段的激进 3D 匹配 ==========
    # 对于重叠区内仍然独有的点（蓝色/绿色），使用更宽松的阈值进行最后一轮3D空间匹配
    # 这可以显著减少重叠区内的独有点数量
    stats['aggressive_3d'] = {'r1_candidates': 0, 'r2_candidates': 0, 'new_matches': 0}
    
    if aggressive_3d_threshold > 0:
        (keep_points, discard_r1, discard_r2, point_source_type, 
         aggressive_blended, aggressive_stats) = _final_aggressive_3d_matching(
            recon1, recon2,
            keep_points, discard_r1, discard_r2,
            point_source_type,
            _conf_cache_r1, _conf_cache_r2,
            distance_threshold=aggressive_3d_threshold,
            blend_mode=blend_mode,
        )
        
        # 合并激进匹配产生的融合点
        blended_points.update(aggressive_blended)
        stats['aggressive_3d'] = aggressive_stats
        stats['weighted_blended_pairs'] += aggressive_stats['new_matches']
    
    # ========== 2.75 步：密度均衡化（可选）==========
    # 基于重叠区的点密度，对非重叠区进行稀疏化处理
    # 放在空间插值之前，这样：
    # 1. 先让点密度均匀化
    # 2. 在均匀密度的基础上进行空间插值，结果更自然
    # 3. 插值后的点不会被丢弃，效率更高
    stats['density_equalization'] = {'enabled': False}
    
    if density_equalization:
        keep_points, density_stats = _density_based_thinning(
            keep_points=keep_points,
            point_source_type=point_source_type,
            blended_points=blended_points,
            recon1=recon1,
            recon2=recon2,
            all_blend_weights=all_blend_weights,  # 传入 blend_weight 用于渐变
            k_neighbors=density_k_neighbors,
            target_density_percentile=density_target_percentile,
            density_tolerance=density_tolerance,
            use_grid_thinning=density_use_grid,
            grid_resolution_factor=density_grid_resolution,
            use_blend_weight_decay=True,  # 使用 blend_weight 控制渐变
            distance_decay_factor=density_distance_decay,
            min_points_for_analysis=100,
            verbose=verbose,
        )
        stats['density_equalization'] = density_stats
    
    # ========== 2.8 步：融合带空间插值 ==========
    # 对融合带内的独有点进行3D坐标插值，实现平滑过渡
    # 基于周围已融合点对的位移进行 IDW 插值
    # 插值范围完全由 inner_blend_margin 和 outer_blend_margin 控制（通过 blend_weight）
    # 注意：在密度均衡化之后进行，这样插值在均匀密度的基础上进行，效果更自然
    interpolated_xyz = {}
    stats['spatial_interpolation'] = {
        'enabled': spatial_blend_interpolation,
        'interpolated_count': 0,
        'avg_displacement': 0.0,
        'k_neighbors': spatial_blend_k_neighbors,
        'smooth_transition': spatial_blend_smooth_transition,
        'smooth_power': spatial_blend_smooth_power,
    }
    
    if spatial_blend_interpolation and len(blended_points) >= 5:  # 需要足够的参考点
        interpolated_xyz = _interpolate_blend_zone_displacements(
            keep_points=keep_points,
            point_source_type=point_source_type,
            all_blend_weights=all_blend_weights,
            blended_points=blended_points,
            recon1=recon1,
            recon2=recon2,
            k_neighbors=spatial_blend_k_neighbors,
            min_displacement_points=5,
            use_smooth_transition=spatial_blend_smooth_transition,
            smooth_power=spatial_blend_smooth_power,
            use_gaussian_weights=True,
            sigma_factor=2.0,
        )
        
        # ========== 边界羽化处理 ==========
        # 对融合带边缘进行额外平滑，消除局部不连续
        if len(interpolated_xyz) > 0:
            interpolated_xyz = _feather_blend_zone_boundary(
                keep_points=keep_points,
                all_blend_weights=all_blend_weights,
                interpolated_xyz=interpolated_xyz,
                blended_points=blended_points,
                recon1=recon1,
                recon2=recon2,
                k_neighbors=8,
                feather_strength=0.3,
            )
        
        stats['spatial_interpolation']['interpolated_count'] = len(interpolated_xyz)
        
        # 计算平均位移量
        if len(interpolated_xyz) > 0:
            displacements = []
            for key, new_xyz in interpolated_xyz.items():
                source, pt3d_id = key
                if source == 'r1' and pt3d_id in recon1.points3D:
                    old_xyz = recon1.points3D[pt3d_id].xyz
                elif source == 'r2' and pt3d_id in recon2.points3D:
                    old_xyz = recon2.points3D[pt3d_id].xyz
                else:
                    continue
                displacements.append(np.linalg.norm(new_xyz - np.asarray(old_xyz)))
            if displacements:
                stats['spatial_interpolation']['avg_displacement'] = float(np.mean(displacements))
                stats['spatial_interpolation']['max_displacement'] = float(np.max(displacements))
                stats['spatial_interpolation']['min_displacement'] = float(np.min(displacements))
    elif not spatial_blend_interpolation:
        stats['spatial_interpolation']['enabled'] = False
    else:
        stats['spatial_interpolation']['enabled'] = False
        stats['spatial_interpolation']['reason'] = 'Not enough reference points'
    
    # ========== 第三步：构建合并后的 Reconstruction ==========
    merged_recon = pycolmap.Reconstruction()
    
    # 确定重叠区使用哪个 recon 的相机参数（基于保留点数）
    r1_kept = sum(1 for (s, _) in keep_points if s == 'r1')
    r2_kept = sum(1 for (s, _) in keep_points if s == 'r2')
    use_r1_for_overlap = r1_kept >= r2_kept
    
    # 获取重叠影像名称
    overlap_img_names = {recon1.images[img_id].name for img_id in common_images.keys()}
    
    if verbose:
        print(f"  Overlap camera choice: {'R1' if use_r1_for_overlap else 'R2'} "
              f"(R1 kept: {r1_kept}, R2 kept: {r2_kept})")
    
    # 3.1 & 3.2 同时添加相机和影像（确保一一对应）
    camera_id_map_r1 = {}  # old_cam_id -> new_cam_id
    camera_id_map_r2 = {}
    image_id_map_r1 = {}  # recon1 old_img_id -> new_img_id
    image_id_map_r2 = {}  # recon2 old_img_id -> new_img_id
    image_name_to_new_id = {}
    
    new_cam_id = 1
    new_img_id = 1
    
    # 添加 recon1 的影像
    for img_id, img in recon1.images.items():
        is_overlap = img.name in overlap_img_names
        
        # 如果是重叠影像且应该用 r2 的相机，则跳过（后面由 r2 添加）
        if is_overlap and not use_r1_for_overlap:
            continue
        
        # 添加相机（如果还没有映射）
        old_cam_id = img.camera_id
        if old_cam_id not in camera_id_map_r1:
            cam = recon1.cameras[old_cam_id]
            new_camera = pycolmap.Camera(
                camera_id=new_cam_id,
                model=cam.model,
                width=cam.width,
                height=cam.height,
                params=cam.params
            )
            merged_recon.add_camera(new_camera)
            camera_id_map_r1[old_cam_id] = new_cam_id
            new_cam_id += 1
        
        # 添加影像
        new_points2D = [pycolmap.Point2D(pt.xy) for pt in img.points2D]
        new_image = pycolmap.Image(
            image_id=new_img_id,
            name=img.name,
            camera_id=camera_id_map_r1[old_cam_id],
            cam_from_world=img.cam_from_world,
            points2D=new_points2D
        )
        merged_recon.add_image(new_image)
        
        image_name_to_new_id[img.name] = new_img_id
        image_id_map_r1[img_id] = new_img_id
        new_img_id += 1
    
    # 添加 recon2 的影像
    for img_id, img in recon2.images.items():
        is_overlap = img.name in overlap_img_names
        
        if img.name in image_name_to_new_id:
            # 重叠影像已由 r1 添加，记录映射
            image_id_map_r2[img_id] = image_name_to_new_id[img.name]
            # 同时映射相机（指向 r1 的相机）
            r1_img_id = [k for k, v in common_images.items() 
                        if recon1.images[k].name == img.name][0]
            camera_id_map_r2[img.camera_id] = camera_id_map_r1[recon1.images[r1_img_id].camera_id]
        else:
            # 添加相机（如果还没有映射）
            old_cam_id = img.camera_id
            if old_cam_id not in camera_id_map_r2:
                cam = recon2.cameras[old_cam_id]
                new_camera = pycolmap.Camera(
                    camera_id=new_cam_id,
                    model=cam.model,
                    width=cam.width,
                    height=cam.height,
                    params=cam.params
                )
                merged_recon.add_camera(new_camera)
                camera_id_map_r2[old_cam_id] = new_cam_id
                new_cam_id += 1
            
            # 添加影像
            new_points2D = [pycolmap.Point2D(pt.xy) for pt in img.points2D]
            new_image = pycolmap.Image(
                image_id=new_img_id,
                name=img.name,
                camera_id=camera_id_map_r2[old_cam_id],
                cam_from_world=img.cam_from_world,
                points2D=new_points2D
            )
            merged_recon.add_image(new_image)
            
            image_name_to_new_id[img.name] = new_img_id
            image_id_map_r2[img_id] = new_img_id
            new_img_id += 1
    
    # 补充 recon1 中被跳过的重叠影像的映射（当 use_r1_for_overlap=False 时）
    if not use_r1_for_overlap:
        for r1_img_id, r2_img_id in common_images.items():
            img_name = recon1.images[r1_img_id].name
            if img_name in image_name_to_new_id:
                image_id_map_r1[r1_img_id] = image_name_to_new_id[img_name]
                # 映射相机
                camera_id_map_r1[recon1.images[r1_img_id].camera_id] = \
                    camera_id_map_r2[recon2.images[r2_img_id].camera_id]
    
    # 3.3 添加 3D 点并构建 track
    point3d_id_map = {}  # (source, old_id) -> new_id
    
    # 定义颜色（用于可视化）
    # BGR -> RGB: 注意 pycolmap 使用 RGB 格式
    COLOR_BLUE = np.array([0, 0, 255], dtype=np.uint8)    # recon1 独有
    COLOR_GREEN = np.array([0, 255, 0], dtype=np.uint8)   # recon2 独有
    COLOR_RED = np.array([255, 0, 0], dtype=np.uint8)     # 冲突后选择
    COLOR_YELLOW = np.array([255, 255, 0], dtype=np.uint8)  # 融合带保留 (r1)
    COLOR_CYAN = np.array([0, 255, 255], dtype=np.uint8)    # 融合带保留 (r2)
    COLOR_MAGENTA = np.array([255, 0, 255], dtype=np.uint8) # 加权融合点
    COLOR_GRAY = np.array([128, 128, 128], dtype=np.uint8)  # 渐变目标颜色（用于插值）
    COLOR_ORANGE_LIGHT = np.array([255, 200, 100], dtype=np.uint8)  # 空间插值点 (r1)
    COLOR_LIME = np.array([180, 255, 100], dtype=np.uint8)   # 空间插值点 (r2)
    
    for (source, old_pt3d_id) in keep_points:
        if source == 'r1':
            recon = recon1
            img_id_map = image_id_map_r1
        else:
            recon = recon2
            img_id_map = image_id_map_r2
        
        if old_pt3d_id not in recon.points3D:
            continue
        
        old_pt3d = recon.points3D[old_pt3d_id]
        key = (source, old_pt3d_id)
        
        # 检查是否是加权融合点
        is_blended = key in blended_points
        blend_info = blended_points.get(key) if is_blended else None
        
        # 创建新的 track，映射到新的 image_id
        new_track = pycolmap.Track()
        valid_elements = []
        
        # 对于加权融合点，尝试合并两个源点的 track
        if is_blended and blend_info is not None:
            # 从主源获取 track elements
            for te in old_pt3d.track.elements:
                old_img_id = te.image_id
                if old_img_id in img_id_map:
                    new_img_id_val = img_id_map[old_img_id]
                    pt2d_idx = te.point2D_idx
                    if new_img_id_val in merged_recon.images:
                        merged_img = merged_recon.images[new_img_id_val]
                        if pt2d_idx < len(merged_img.points2D):
                            valid_elements.append((new_img_id_val, pt2d_idx))
            
            # 也从另一个源获取 track elements（如果不冲突）
            other_source = 'r2' if source == 'r1' else 'r1'
            other_pt3d_id = blend_info['pt3d_id2'] if source == 'r1' else blend_info['pt3d_id1']
            other_recon = recon2 if source == 'r1' else recon1
            other_img_id_map = image_id_map_r2 if source == 'r1' else image_id_map_r1
            
            if other_pt3d_id in other_recon.points3D:
                other_pt3d = other_recon.points3D[other_pt3d_id]
                existing_pairs = set(valid_elements)
                for te in other_pt3d.track.elements:
                    old_img_id = te.image_id
                    if old_img_id in other_img_id_map:
                        new_img_id_val = other_img_id_map[old_img_id]
                        pt2d_idx = te.point2D_idx
                        pair = (new_img_id_val, pt2d_idx)
                        if pair not in existing_pairs:
                            if new_img_id_val in merged_recon.images:
                                merged_img = merged_recon.images[new_img_id_val]
                                if pt2d_idx < len(merged_img.points2D):
                                    valid_elements.append(pair)
                                    existing_pairs.add(pair)
        else:
            # 非融合点：只使用主源的 track
            for te in old_pt3d.track.elements:
                old_img_id = te.image_id
                if old_img_id in img_id_map:
                    new_img_id_val = img_id_map[old_img_id]
                    pt2d_idx = te.point2D_idx
                    if new_img_id_val in merged_recon.images:
                        merged_img = merged_recon.images[new_img_id_val]
                        if pt2d_idx < len(merged_img.points2D):
                            valid_elements.append((new_img_id_val, pt2d_idx))
        
        if len(valid_elements) == 0:
            continue
        
        # 确定 xyz、color 和 error
        if is_blended and blend_info is not None:
            # 使用融合后的值
            point_xyz = blend_info['xyz']
            point_error = blend_info['error']
            if color_by_source:
                point_color = COLOR_MAGENTA  # 加权融合点用洋红色
            else:
                point_color = blend_info['color']
        else:
            # 检查是否有空间插值的坐标
            if key in interpolated_xyz:
                # 使用插值后的坐标（融合带平滑过渡）
                point_xyz = interpolated_xyz[key]
                is_interpolated = True
            else:
                # 使用原始值
                point_xyz = old_pt3d.xyz
                is_interpolated = False
            point_error = old_pt3d.error
            
            # 确定颜色
            if color_by_source:
                source_type = point_source_type.get(key, 'only')
                
                # 优先检查是否是空间插值点
                if is_interpolated:
                    # 空间插值点：使用特殊颜色，并根据插值量渐变
                    blend_weight = all_blend_weights.get(key, 1.0)
                    # 基础颜色：浅橙色(r1) 或 青柠色(r2)
                    base_color = COLOR_ORANGE_LIGHT if source == 'r1' else COLOR_LIME
                    # 根据 blend_weight 渐变：越靠近核心区（weight 越低），颜色越接近洋红色
                    point_color = (blend_weight * base_color.astype(np.float32) + 
                                   (1 - blend_weight) * COLOR_MAGENTA.astype(np.float32))
                    point_color = np.clip(point_color, 0, 255).astype(np.uint8)
                elif source_type == 'conflict':
                    point_color = COLOR_RED
                elif source_type == 'conflict_projected':
                    # 通过投影新匹配到的冲突点（橙色）
                    point_color = np.array([255, 128, 0], dtype=np.uint8)
                elif source_type == 'conflict_3d' or source_type == 'match_3d':
                    # 通过 3D 匹配的点（深橙色/棕色）
                    point_color = np.array([200, 100, 0], dtype=np.uint8)
                elif source_type == 'blended_3d':
                    # 3D 匹配后加权融合的点（深洋红色）
                    point_color = np.array([200, 0, 200], dtype=np.uint8)
                elif source_type == 'blended_aggressive':
                    # 激进3D匹配后加权融合的点（亮洋红色）
                    point_color = np.array([255, 100, 255], dtype=np.uint8)
                elif source_type == 'conflict_aggressive':
                    # 激进3D匹配后胜者通吃的点（粉红色）
                    point_color = np.array([255, 150, 200], dtype=np.uint8)
                elif source_type == 'blend':
                    # 融合带内的点：使用插值权重实现颜色渐变
                    blend_weight = all_blend_weights.get(key, 1.0)
                    base_color = COLOR_YELLOW if source == 'r1' else COLOR_CYAN
                    # 权重越低（越靠近核心区），颜色越接近灰色；权重越高，颜色越接近基础色
                    point_color = (blend_weight * base_color.astype(np.float32) + 
                                   (1 - blend_weight) * COLOR_GRAY.astype(np.float32))
                    point_color = np.clip(point_color, 0, 255).astype(np.uint8)
                elif source_type == 'projected':
                    # 通过投影检查但没有匹配到的点（浅蓝/浅绿）- 也支持权重渐变
                    blend_weight = all_blend_weights.get(key, 1.0)
                    base_color = np.array([100, 150, 255], dtype=np.uint8) if source == 'r1' else np.array([100, 255, 150], dtype=np.uint8)
                    point_color = (blend_weight * base_color.astype(np.float32) + 
                                   (1 - blend_weight) * COLOR_GRAY.astype(np.float32))
                    point_color = np.clip(point_color, 0, 255).astype(np.uint8)
                elif source == 'r1':
                    # 独有点也可以根据权重渐变
                    blend_weight = all_blend_weights.get(key, 1.0)
                    if blend_weight < 1.0:
                        point_color = (blend_weight * COLOR_BLUE.astype(np.float32) + 
                                       (1 - blend_weight) * COLOR_GRAY.astype(np.float32))
                        point_color = np.clip(point_color, 0, 255).astype(np.uint8)
                    else:
                        point_color = COLOR_BLUE
                else:
                    # recon2 独有点
                    blend_weight = all_blend_weights.get(key, 1.0)
                    if blend_weight < 1.0:
                        point_color = (blend_weight * COLOR_GREEN.astype(np.float32) + 
                                       (1 - blend_weight) * COLOR_GRAY.astype(np.float32))
                        point_color = np.clip(point_color, 0, 255).astype(np.uint8)
                    else:
                        point_color = COLOR_GREEN
            else:
                point_color = old_pt3d.color
        
        # 添加 3D 点（先用空 track）
        new_pt3d_id = merged_recon.add_point3D(
            xyz=point_xyz,
            track=pycolmap.Track(),
            color=point_color
        )
        
        # 设置误差
        merged_recon.points3D[new_pt3d_id].error = point_error
        
        point3d_id_map[(source, old_pt3d_id)] = new_pt3d_id
        
        # 更新 track 和 points2D
        for new_img_id_val, pt2d_idx in valid_elements:
            # 添加到 track
            merged_recon.points3D[new_pt3d_id].track.add_element(new_img_id_val, pt2d_idx)
            
            # 更新 point2D 的 point3D_id
            merged_recon.images[new_img_id_val].points2D[pt2d_idx].point3D_id = new_pt3d_id
    
    # ========== 统计信息 ==========
    info = {
        'stats': stats,
        'num_points_r1_kept': sum(1 for (s, _) in keep_points if s == 'r1'),
        'num_points_r2_kept': sum(1 for (s, _) in keep_points if s == 'r2'),
        'total_merged_points': len(merged_recon.points3D),
        'total_merged_images': len(merged_recon.images),
        'total_merged_cameras': len(merged_recon.cameras),
    }
    
    if verbose:
        mode_desc = "Winner-Takes-All" if blend_mode == 'winner' else "Weighted Average"
        print(f"\n=== Merge by confidence results ({mode_desc}) ===")
        print(f"  Blend mode: {blend_mode}")
        if blend_mode == 'weighted':
            print(f"    Weighted blended pairs: {stats.get('weighted_blended_pairs', 0)}")
        print(f"  2D matching (multi-radius: {match_radii}):")
        print(f"    Total matched pairs: {stats['matched_pairs']}")
        print(f"    Avg match distance: {stats['avg_match_distance']:.2f}px")
        # 显示每个半径的匹配统计
        matches_per_r = stats.get('matches_per_radius', {})
        if matches_per_r:
            print(f"    Matches per radius:")
            for r in sorted(matches_per_r.keys()):
                count = matches_per_r[r]
                if count > 0:
                    print(f"      radius={r}px: {count} pairs")
        print(f"  3D matching (threshold={match_3d_threshold}):")
        match_3d_stats = stats.get('match_3d', {})
        print(f"    R1 candidates (2D unmatched in pmap): {match_3d_stats.get('r1_candidates', 0)}")
        print(f"    R2 candidates (2D unmatched in pmap): {match_3d_stats.get('r2_candidates', 0)}")
        print(f"    Matched in 3D space: {match_3d_stats.get('matched_3d', 0)}")
        if match_3d_stats.get('avg_3d_distance', 0) > 0:
            print(f"    Avg 3D match distance: {match_3d_stats.get('avg_3d_distance', 0):.4f}")
        
        # 激进 3D 匹配统计
        aggressive_stats = stats.get('aggressive_3d', {})
        if aggressive_3d_threshold > 0:
            print(f"  Aggressive 3D matching (threshold={aggressive_3d_threshold}):")
            print(f"    R1 unmatched candidates (only/projected/blend): {aggressive_stats.get('r1_only_candidates', 0)}")
            print(f"    R2 unmatched candidates (only/projected/blend): {aggressive_stats.get('r2_only_candidates', 0)}")
            print(f"    New matches found: {aggressive_stats.get('new_matches', 0)}")
            if aggressive_stats.get('avg_distance', 0) > 0:
                print(f"    Avg match distance: {aggressive_stats.get('avg_distance', 0):.4f}")
            
            # 显示距离分布诊断信息
            nn_stats = aggressive_stats.get('nn_distance_stats', {})
            if nn_stats:
                print(f"    [DIAGNOSTIC] Nearest neighbor distance distribution:")
                print(f"      Min: {nn_stats.get('min', 0):.4f}, Max: {nn_stats.get('max', 0):.4f}")
                print(f"      Mean: {nn_stats.get('mean', 0):.4f}, Median: {nn_stats.get('median', 0):.4f}")
                print(f"      P25: {nn_stats.get('p25', 0):.4f}, P75: {nn_stats.get('p75', 0):.4f}, P90: {nn_stats.get('p90', 0):.4f}")
                print(f"      Within threshold ({aggressive_3d_threshold}): {nn_stats.get('within_threshold', 0)}")
                remaining = aggressive_stats.get('r1_only_candidates', 0) - aggressive_stats.get('new_matches', 0)
                if remaining > 0 and nn_stats.get('p90', 0) > aggressive_3d_threshold:
                    print(f"      [TIP] Consider increasing aggressive_3d_threshold to {nn_stats.get('p90', 0):.1f} to match more points")
        
        print(f"  Unmatched points handling:")
        print(f"    Keep all unmatched: {keep_unmatched_overlap}")
        if not keep_unmatched_overlap:
            print(f"    Inner blend margin: {inner_blend_margin:.1f}px (toward overlap center)")
            print(f"    Outer blend margin: {outer_blend_margin:.1f}px (toward non-overlap)")
            print(f"    Total blend width: {inner_blend_margin + outer_blend_margin:.1f}px")
        print(f"  Overlap region processing:")
        print(f"    Both -> R1 (higher conf): {stats['pixels_both_choose_r1']}")
        print(f"    Both -> R2 (higher conf): {stats['pixels_both_choose_r2']}")
        print(f"    Core discarded (R1): {stats['discarded_unmatched_r1']}")
        print(f"    Core discarded (R2): {stats['discarded_unmatched_r2']}")
        if not keep_unmatched_overlap:
            print(f"    Blend zone kept (R1): {stats.get('blend_zone_r1', 0)}")
            print(f"    Blend zone kept (R2): {stats.get('blend_zone_r2', 0)}")
            # 统计插值权重分布
            if all_blend_weights:
                weights = list(all_blend_weights.values())
                weights_arr = np.array(weights)
                interpolated_count = np.sum((weights_arr > 0) & (weights_arr < 1))
                print(f"    Interpolated points: {interpolated_count} (0 < weight < 1)")
                if interpolated_count > 0:
                    interp_weights = weights_arr[(weights_arr > 0) & (weights_arr < 1)]
                    print(f"      Weight stats: min={interp_weights.min():.2f}, max={interp_weights.max():.2f}, mean={interp_weights.mean():.2f}")
        print(f"  Unmatched region (kept):")
        print(f"    Only R1: {stats['pixels_only_r1']}")
        print(f"    Only R2: {stats['pixels_only_r2']}")
        print(f"  Non-overlap track points (projected to check boundary):")
        r1_proj = stats.get('non_overlap_track_r1', {})
        r2_proj = stats.get('non_overlap_track_r2', {})
        print(f"    R1: {r1_proj.get('projected', 0)} projected, {r1_proj.get('new_matches', 0)} new matches, {r1_proj.get('kept_outside', 0)} kept outside")
        print(f"    R2: {r2_proj.get('projected', 0)} projected, {r2_proj.get('new_matches', 0)} new matches, {r2_proj.get('kept_outside', 0)} kept outside")
        
        # 空间插值统计
        spatial_stats = stats.get('spatial_interpolation', {})
        print(f"  Spatial blend interpolation:")
        print(f"    Enabled: {spatial_stats.get('enabled', False)}")
        if spatial_stats.get('enabled', False):
            print(f"    K neighbors: {spatial_stats.get('k_neighbors', 16)}")
            print(f"    Smooth transition: {spatial_stats.get('smooth_transition', True)}")
            print(f"    Smooth power: {spatial_stats.get('smooth_power', 1.0):.2f} (<1=stronger, >1=weaker)")
            print(f"    Interpolation range: controlled by inner/outer_blend_margin")
            print(f"    Points interpolated: {spatial_stats.get('interpolated_count', 0)}")
            if spatial_stats.get('interpolated_count', 0) > 0:
                print(f"    Displacement stats:")
                print(f"      Avg: {spatial_stats.get('avg_displacement', 0):.4f}")
                print(f"      Min: {spatial_stats.get('min_displacement', 0):.4f}")
                print(f"      Max: {spatial_stats.get('max_displacement', 0):.4f}")
        elif 'reason' in spatial_stats:
            print(f"    Reason: {spatial_stats['reason']}")
        
        print(f"  Merged reconstruction:")
        print(f"    Images: {info['total_merged_images']}")
        print(f"    3D Points: {info['total_merged_points']} "
              f"(R1: {info['num_points_r1_kept']}, R2: {info['num_points_r2_kept']})")
        print(f"    Cameras: {info['total_merged_cameras']}")
    
    # ========== 体素降采样（可选）==========
    if voxel_size > 0:
        original_points = len(merged_recon.points3D)
        
        if verbose:
            print(f"\n  Voxel downsampling (voxel_size={voxel_size})...")
        
        # 提取点云数据
        points_xyz = {}
        points_color = {}
        for pt3d_id, pt3d in merged_recon.points3D.items():
            points_xyz[pt3d_id] = np.array(pt3d.xyz)
            points_color[pt3d_id] = np.array(pt3d.color)
        
        # 执行体素降采样
        downsampled_xyz, downsampled_color, voxel_to_original = voxel_downsample(
            points_xyz, points_color, voxel_size, verbose=False
        )
        
        # 需要重建 reconstruction，因为无法直接删除/修改 points3D
        # 创建需要保留的原始点 ID 集合（每个体素保留第一个点）
        keep_original_ids = set()
        for new_id, original_ids in voxel_to_original.items():
            if original_ids:
                keep_original_ids.add(original_ids[0])  # 保留每个体素的第一个点
        
        # 删除不需要的点
        points_to_remove = [pt3d_id for pt3d_id in merged_recon.points3D.keys() 
                           if pt3d_id not in keep_original_ids]
        
        # pycolmap 中无效的 point3D_id 需要使用无符号整数最大值
        # 在 COLMAP 中这表示 kInvalidPoint3DId
        INVALID_POINT3D_ID = 2**64 - 1
        
        for pt3d_id in points_to_remove:
            # 需要先清除相关的 2D 点引用
            pt3d = merged_recon.points3D[pt3d_id]
            for elem in pt3d.track.elements:
                if elem.image_id in merged_recon.images:
                    img = merged_recon.images[elem.image_id]
                    if elem.point2D_idx < len(img.points2D):
                        # 使用无符号整数最大值表示无效 ID
                        img.points2D[elem.point2D_idx].point3D_id = INVALID_POINT3D_ID
            # 删除 3D 点
            del merged_recon.points3D[pt3d_id]
        
        # 更新保留点的坐标为体素质心
        for new_id, original_ids in voxel_to_original.items():
            if original_ids and original_ids[0] in merged_recon.points3D:
                pt3d = merged_recon.points3D[original_ids[0]]
                pt3d.xyz = downsampled_xyz[new_id]
                pt3d.color = downsampled_color[new_id]
        
        downsampled_points = len(merged_recon.points3D)
        info['voxel_downsampling'] = {
            'enabled': True,
            'voxel_size': voxel_size,
            'original_points': original_points,
            'downsampled_points': downsampled_points,
            'reduction_ratio': 1.0 - (downsampled_points / original_points) if original_points > 0 else 0.0,
        }
        info['total_merged_points'] = downsampled_points
        
        if verbose:
            print(f"    Original: {original_points} -> Downsampled: {downsampled_points}")
            print(f"    Reduction: {info['voxel_downsampling']['reduction_ratio']*100:.1f}%")
    else:
        info['voxel_downsampling'] = {'enabled': False}
    
    return merged_recon, info


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
    color_by_source: bool = False,  # 是否按来源着色
    match_radii: Optional[List[float]] = None,  # 多级2D点匹配搜索半径列表（像素）
    match_3d_threshold: float = 0.5,  # 3D空间匹配距离阈值
    aggressive_3d_threshold: float = 0.0,  # 最终阶段激进3D匹配阈值
    inner_blend_margin: float = 150.0,  # 融合带向重叠区内部延伸的宽度（像素）【优化】
    outer_blend_margin: float = 200.0,  # 融合带向外部延伸的宽度（像素）【优化】
    blend_mode: str = 'winner',  # 融合模式：'winner' 或 'weighted'
    keep_unmatched_overlap: bool = False,  # 是否保留重叠区所有未匹配点
    spatial_blend_interpolation: bool = True,  # 是否启用融合带3D坐标空间插值
    spatial_blend_k_neighbors: int = 32,  # 空间插值使用的近邻数【优化】
    spatial_blend_smooth_transition: bool = True,  # 是否使用 smoothstep 实现更平滑过渡
    spatial_blend_smooth_power: float = 0.5,  # 平滑过渡力度（<1更强，>1更弱）【优化】
    density_equalization: bool = False,  # 是否启用密度均衡化
    density_k_neighbors: int = 10,  # 密度计算使用的近邻数
    density_target_percentile: float = 50.0,  # 目标密度百分位数
    density_tolerance: float = 1.2,  # 密度容差倍数
    density_use_grid: bool = True,  # 是否使用网格采样
    density_grid_resolution: float = 1.0,  # 网格分辨率因子
    density_distance_decay: float = 0.5,  # 距离衰减因子
    voxel_size: float = 0.0,  # 体素降采样大小，0表示不降采样
    verbose: bool = True,
) -> Tuple[Optional[pycolmap.Reconstruction], Dict]:
    """
    完整的两个 reconstruction 合并流程
    （多级半径匹配 + 双边界平滑融合 + 3D匹配补充 + 激进3D匹配 + 融合带空间插值 + 密度均衡化）
    
    1. 找到共同影像
    2. 建立 2D-3D 对应关系
    3. 找到对应的 3D 点对
    4. RANSAC 估计 Sim3 变换
    5. 应用变换对齐 recon2
    6. 基于置信度合并点云（多级2D匹配 + 3D匹配 + 激进3D匹配）
    7. 融合带3D坐标空间插值（IDW插值实现平滑过渡）
    
    Args:
        recon1: 第一个 reconstruction（基准，不修改，对应 prev_recon）
        recon2: 第二个 reconstruction（会被变换，对应 curr_recon）
        inlier_threshold: RANSAC 内点阈值
        min_inliers: RANSAC 最小内点数
        min_sample_size: RANSAC 每次迭代采样的点数
        ransac_iterations: RANSAC 迭代次数
        prev_recon_conf: recon1 的像素级置信度图 {global_img_idx: (H, W) array}
        curr_recon_conf: recon2 的像素级置信度图 {global_img_idx: (H, W) array}
        image_name_to_idx: 图像名称到全局索引的映射
        output_dir: 可选的输出目录，如果提供则保存对齐后的 recon2
        start_idx: 起始图像索引，用于输出目录命名
        end_idx: 结束图像索引，用于输出目录命名
        color_by_source: 是否按来源着色（用于可视化测试）
            - recon1 独有: 蓝色, recon2 独有: 绿色
            - 冲突选择: 红色, 加权融合: 洋红色
            - 融合带: 黄色(r1) / 青色(r2)
            - 3D匹配: 深橙色, 3D融合: 深洋红色
            - 激进匹配: 亮洋红/粉红
            - 空间插值: 浅橙(r1) / 青柠(r2)
        match_radii: 多级2D点匹配搜索半径列表（像素）
            - 默认 [3, 5, 10, 20, 50]
            - 从小到大依次匹配，小半径优先匹配精确对应
            - 大半径补充匹配边缘点，实现更全面的覆盖
            - 也可以传入单个值，如 10.0，会自动转换为 [10.0]
        match_3d_threshold: 3D空间匹配距离阈值
            - 用于补充 2D 匹配失败的点
            - 单位与点云坐标一致（通常是米）
            - 默认 0.5，设为 0 禁用 3D 匹配
            - 建议值: 0.3-1.0
        aggressive_3d_threshold: 最终阶段激进3D匹配阈值
            - 对重叠区内仍然独有的点（蓝色/绿色）进行最后一轮3D空间匹配
            - 使用比 match_3d_threshold 更宽松的阈值
            - 可以显著减少重叠区内的独有点
            - 默认 0 表示禁用
            - 建议值: 1.5-3.0（是 match_3d_threshold 的 1.5-3 倍）
        inner_blend_margin: 融合带向重叠区内部延伸的宽度（像素）
            - 控制核心丢弃区的大小
            - 值越大，核心丢弃区越小，保留更多重叠区内的点
            - 默认80像素
            - 建议值: 50-150像素
            - 同时控制空间插值的范围
        outer_blend_margin: 融合带向外部延伸的宽度（像素）
            - 控制融合带向非重叠区延伸多少
            - 值越大，过渡越平滑
            - 默认100像素
            - 建议值: 50-150像素
            - 同时控制空间插值的范围
        blend_mode: 3D点融合模式
            - 'winner': 胜者通吃（默认），选择置信度更高的点
            - 'weighted': 加权平均，基于置信度计算加权位置
              实现更平滑的3D点云过渡
        keep_unmatched_overlap: 是否保留重叠区所有未匹配点
            - False（默认）：启用边缘处理，使用双边界平滑融合
            - True：禁用边缘处理，保留重叠区所有未匹配点
        spatial_blend_interpolation: 是否启用融合带3D坐标空间插值
            - True（默认）：对融合带内的独有点进行3D坐标插值
              基于周围已融合点对的位移，使用 IDW 插值估算偏移
              插值范围由 inner_blend_margin 和 outer_blend_margin 控制
              实现从重叠区到非重叠区的平滑3D坐标过渡
            - False：禁用空间插值，融合带点保持原始坐标
        spatial_blend_k_neighbors: 空间插值使用的近邻数
            - 默认 16
            - 用于 IDW 插值的 K 近邻数量
            - 值越大，插值越平滑但计算量越大
        spatial_blend_smooth_transition: 是否使用 smoothstep 实现更平滑过渡
            - True（默认）：使用 smootherstep 函数，过渡更平滑自然
            - False：使用线性过渡
        spatial_blend_smooth_power: 平滑过渡的力度参数（默认 1.0）
            - < 1.0：过渡更"宽"，更多的点获得更大的位移，平滑效果更强
            - = 1.0：标准 smootherstep 过渡
            - > 1.0：过渡更"窄"，只有靠近核心区的点才有显著位移
            - 建议值：0.3-0.7 可获得更强的平滑效果
        density_equalization: 是否启用密度均衡化（默认 False）
            - True：分析重叠区点密度，对非重叠区进行稀疏化
              使点云密度从重叠区到非重叠区均匀过渡
            - False：不进行密度均衡化处理
        density_k_neighbors: 密度计算使用的 K 近邻数（默认 10）
            - 用于计算每个点的局部密度
        density_target_percentile: 目标密度百分位数（默认 50）
            - 使用重叠区密度的该百分位数作为目标
        density_tolerance: 密度容差倍数（默认 1.2）
            - 只有密度 > target * tolerance 的点才可能被丢弃
        voxel_size: 体素降采样大小（与点云坐标单位一致，通常为米）
            - 默认 0.0 表示不进行降采样
            - 正值时会在合并完成后对点云进行体素降采样
            - 每个体素保留一个代表点（质心），减少点云密度
            - 建议值: 0.5-2.0 米，根据点云精度需求调整
        verbose: 是否打印详细信息
        
    Returns:
        merged_recon: 合并后的 reconstruction（如果失败返回 None）
        info: 合并信息字典
    """
    import copy
    
    info = {'success': False}
    
    # 1. 找到共同影像
    common_images = find_common_images(recon1, recon2)
    info['num_common_images'] = len(common_images)
    
    if len(common_images) == 0:
        if verbose:
            print("No common images found!")
        return None, info
    
    if verbose:
        print(f"\nFound {len(common_images)} common images")
    
    # 2. 并行建立 2D-3D 对应关系和像素映射
    corr_r1, corr_r2, pmap_r1, pmap_r2 = build_correspondences_parallel(
        recon1, recon2, common_images,
        include_track_pixels=False,
        verbose=verbose
    )
    
    # 3. 找到对应的 3D 点对
    pts1, pts2, match_info = find_corresponding_3d_points(
        pmap_r1, pmap_r2, common_images, corr_r1, corr_r2,
        verbose=verbose
    )
    
    info['num_point_pairs'] = len(pts1)
    
    if len(pts1) < 3:
        if verbose:
            print(f"Not enough corresponding points ({len(pts1)}) for alignment!")
        return None, info
    
    # 4. RANSAC 估计 Sim3 变换 (从 recon2 到 recon1)
    R, t, scale, inlier_mask = estimate_sim3_ransac(
        pts2, pts1,
        max_iterations=ransac_iterations,
        inlier_threshold=inlier_threshold,
        min_inliers=min_inliers,
        min_sample_size=min_sample_size,
        verbose=verbose
    )
    
    info['num_inliers'] = int(np.sum(inlier_mask))
    info['scale'] = float(scale)
    info['translation'] = t.tolist()
    
    if np.sum(inlier_mask) < 3:
        if verbose:
            print("Too few inliers after RANSAC!")
        return None, info
    
    # 5. 复制 recon2 并应用变换
    recon2_aligned = copy.deepcopy(recon2)
    apply_sim3_to_reconstruction(recon2_aligned, R, t, scale)

    # 6. 输出保存变换后的 recon2（可选）
    if output_dir is not None:
        # 使用 start_idx 和 end_idx 命名，如果没有则用共同影像数
        if start_idx is not None and end_idx is not None:
            subdir_name = f"{start_idx}_{end_idx}"
        else:
            subdir_name = f"common_{len(common_images)}"
        temp_path = output_dir / "temp_aligned_recon1" / subdir_name
        temp_path.mkdir(parents=True, exist_ok=True)
        recon2_aligned.write_text(str(temp_path))
        recon2_aligned.export_PLY(str(temp_path / "points3D.ply"))
        if verbose:
            print(f"  Saved aligned recon2 to: {temp_path}")
    
    if verbose:
        print(f"\nApplied Sim3 transformation to recon2")
        print(f"  Scale: {scale:.6f}")
    
    # 6. 重新计算对齐后的像素映射
    common_ids_r2 = list(common_images.values())
    corr_r2_aligned = build_2d_3d_correspondences(
        recon2_aligned, common_ids_r2, verbose=False
    )
    pmap_r2_aligned = build_pixel_to_3d_mapping(corr_r2_aligned)
    
    # 7. 基于置信度合并（使用 多级2D匹配 + 3D匹配 + 激进3D匹配 + 双边界平滑融合 + 空间插值 + 密度均衡化 + 体素降采样）
    merged_recon, merge_info = merge_reconstructions_by_confidence(
        recon1, 
        recon2_aligned, 
        pmap_r1, 
        pmap_r2_aligned, 
        common_images,
        prev_recon_conf=prev_recon_conf,
        curr_recon_conf=curr_recon_conf,
        image_name_to_idx=image_name_to_idx,
        color_by_source=color_by_source,  # 传递颜色参数
        match_radii=match_radii,  # 多级2D点匹配搜索半径列表
        match_3d_threshold=match_3d_threshold,  # 3D空间匹配距离阈值
        aggressive_3d_threshold=aggressive_3d_threshold,  # 激进3D匹配阈值
        inner_blend_margin=inner_blend_margin,  # 融合带向内延伸宽度（同时控制空间插值范围）
        outer_blend_margin=outer_blend_margin,  # 融合带向外延伸宽度（同时控制空间插值范围）
        blend_mode=blend_mode,  # 融合模式：'winner' 或 'weighted'
        keep_unmatched_overlap=keep_unmatched_overlap,  # 是否保留所有未匹配点
        spatial_blend_interpolation=spatial_blend_interpolation,  # 空间插值
        spatial_blend_k_neighbors=spatial_blend_k_neighbors,  # 近邻数
        spatial_blend_smooth_transition=spatial_blend_smooth_transition,  # 平滑过渡
        spatial_blend_smooth_power=spatial_blend_smooth_power,  # 平滑力度
        density_equalization=density_equalization,  # 密度均衡化
        density_k_neighbors=density_k_neighbors,  # 密度计算近邻数
        density_target_percentile=density_target_percentile,  # 目标密度百分位
        density_tolerance=density_tolerance,  # 密度容差
        density_use_grid=density_use_grid,  # 网格采样
        density_grid_resolution=density_grid_resolution,  # 网格分辨率
        density_distance_decay=density_distance_decay,  # 距离衰减
        voxel_size=voxel_size,  # 体素降采样大小
        verbose=verbose,
    )
        
    # 更新 info
    info['points_from_r1'] = merge_info['num_points_r1_kept']
    info['points_from_r2'] = merge_info['num_points_r2_kept']
    info['total_merged_points'] = merge_info['total_merged_points']
    info['merge_stats'] = merge_info['stats']
    info['success'] = True
    info['merged_images'] = merge_info['total_merged_images']
    info['merged_points3D'] = merge_info['total_merged_points']
    
    return merged_recon, info


def load_reconstructions(
    model_dir1: str,
    model_dir2: str,
    verbose: bool = True
) -> tuple:
    """
    加载两个 COLMAP 重建并找到共同影像
    
    Args:
        model_dir1: 第一个模型目录路径（作为基准）
        model_dir2: 第二个模型目录路径
        verbose: 是否打印详细信息
        
    Returns:
        recon1: 第一个 reconstruction
        recon2: 第二个 reconstruction
        common_images: 共同影像映射 {recon1_image_id: recon2_image_id}
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
    
    return recon1, recon2, common_images


def main():
    """
    示例用法 - 包含颜色可视化测试
    """
    # 定义路径
    base_dir = Path(__file__).parent.parent / "output" / "Ganluo_images" / "sparse_incremental_reconstruction" / "temp_aligned_to_prev_recon_overlay_image"
    
    model_dir1 = str(base_dir / "0_6")
    model_dir2 = str(base_dir / "4_10")
    
    print("=" * 60)
    print("Loading and analyzing COLMAP reconstructions")
    print("=" * 60)
    
    # 加载重建并找到共同影像
    recon1, recon2, common_images = load_reconstructions(
        model_dir1=model_dir1,
        model_dir2=model_dir2,
        verbose=True
    )
    
    if len(common_images) == 0:
        print("\n" + "=" * 60)
        print("Warning: No common images found between the two reconstructions!")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print("Merging reconstructions with COLOR VISUALIZATION")
    print("  (Using Dual-Boundary Smooth Blend + Spatial Interpolation)")
    print("  - Blue (蓝色): recon1 only (track has common images)")
    print("  - Green (绿色): recon2 only (track has common images)") 
    print("  - Red (红色): conflict resolved (matched via 2D)")
    print("  - Orange (橙色): conflict resolved (matched via projection)")
    print("  - Yellow (黄色): blend zone from recon1")
    print("  - Cyan (青色): blend zone from recon2")
    print("  - Gray gradient (灰色渐变): interpolated transition in blend zone")
    print("  - Light orange (浅橙): spatial interpolated from recon1")
    print("  - Lime (青柠): spatial interpolated from recon2")
    print("  - Magenta (洋红): weighted blend point")
    print("=" * 60)
    
    # 使用完整的合并流程，启用颜色可视化
    output_dir = base_dir.parent / "merged_color_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged_recon, info = merge_two_reconstructions(
        recon1=recon1,
        recon2=recon2,
        inlier_threshold=0.5,  # 根据实际尺度调整
        min_inliers=5,
        ransac_iterations=1000,
        output_dir=output_dir,
        start_idx=0,
        end_idx=10,
        color_by_source=True,  # 启用颜色可视化！
        match_radii=[3, 5, 10, 20, 50],  # 多级2D匹配半径（从小到大依次匹配）
        match_3d_threshold=1.0,  # 3D空间匹配阈值
        aggressive_3d_threshold=5.0,  # 启用激进3D匹配！根据点云尺度调整
        inner_blend_margin=200.0,  # 融合带向内延伸【增大：200像素】
        outer_blend_margin=300.0,  # 融合带向外延伸【增大：300像素】
        blend_mode='weighted',  # 使用加权融合模式
        keep_unmatched_overlap=False,  # 保持边缘处理
        spatial_blend_interpolation=True,  # 启用融合带3D坐标空间插值！
        spatial_blend_k_neighbors=32,  # 使用32个近邻进行高斯加权插值【增大】
        spatial_blend_smooth_transition=True,  # 使用 smoothstep 实现平滑过渡
        spatial_blend_smooth_power=0.4,  # 平滑力度【降低以更强平滑】
        density_equalization=True,  # 启用密度均衡化！使非重叠区密度与重叠区一致
        density_k_neighbors=10,  # 密度计算使用的近邻数
        density_target_percentile=50.0,  # 使用重叠区点间距中位数作为目标
        density_tolerance=1.2,  # 密度容差倍数（越大稀疏化越强）
        density_use_grid=True,  # 使用网格采样方法（更稳定均匀）
        density_grid_resolution=1.0,  # 网格分辨率因子（<1更精细，>1更粗糙）
        density_distance_decay=0.5,  # 距离衰减因子（0=无衰减，1=强衰减，越远越不稀疏化）
        voxel_size=0.0,  # 体素降采样大小，0表示不降采样（设为正值如1.0可启用）
        verbose=True,
    )
    
    if merged_recon is not None:
        # 保存合并结果
        merged_path = output_dir / "merged_colored"
        merged_path.mkdir(parents=True, exist_ok=True)
        merged_recon.write_text(str(merged_path))
        merged_recon.export_PLY(str(merged_path / "points3D.ply"))
        
        print("\n" + "=" * 60)
        print("COLOR VISUALIZATION LEGEND (2D + 3D Matching + Spatial Interpolation):")
        print("  🔵 Blue (蓝色): Points only in recon1 (non-overlap)")
        print("  🟢 Green (绿色): Points only in recon2 (non-overlap)")
        print("  🔴 Red (红色): Conflict resolved via 2D matching")
        print("  🟠 Orange (橙色): Conflict resolved via projection")
        print("  🟤 Brown (棕色): Conflict resolved via 3D matching")
        print("  🟡 Yellow (黄色): Blend zone from recon1")
        print("  🔷 Cyan (青色): Blend zone from recon2")
        print("  ⬜ Gray gradient (灰色渐变): Weight-based color transition")
        print("      - Points near inner boundary: more gray (low weight)")
        print("      - Points near outer boundary: more color (high weight)")
        print("  🟧 Light Orange (浅橙): Spatial interpolated from recon1")
        print("  💚 Lime (青柠): Spatial interpolated from recon2")
        print("      - 3D coordinates adjusted via IDW interpolation")
        print("      - Color transitions: blend weight -> magenta gradient")
        print("  💜 Magenta (洋红): Weighted blend (2D)")
        print("  💜 Dark Magenta: Weighted blend (3D)")
        print("=" * 60)
        print(f"\nMerged reconstruction saved to: {merged_path}")
        print(f"  Open points3D.ply in CloudCompare or MeshLab to visualize!")
        print("=" * 60)
    else:
        print("\nMerge failed!")
    
    # 以下是原有的详细分析代码（可选）
    print("\n" + "=" * 60)
    print("Detailed Analysis (optional)")
    print("=" * 60)
    
    # 并行构建 2D-3D 对应关系和像素映射
    correspondences_recon1, correspondences_recon2, pixel_map_recon1, pixel_map_recon2 = \
        build_correspondences_parallel(
            recon1, recon2, common_images,
            include_track_pixels=False,
            verbose=True
        )
    
    # 基于共同像素位置找到对应的 3D 点对
    pts1, pts2, match_info = find_corresponding_3d_points(
        pixel_map_recon1, pixel_map_recon2,
        common_images,
        correspondences_recon1, correspondences_recon2,
        verbose=True
    )
    
    # 使用 RANSAC 估计 Sim3 变换
    if len(pts1) >= 3:
        R, t, scale, inlier_mask = estimate_sim3_ransac(
            pts2, pts1,
            max_iterations=1000,
            inlier_threshold=0.05,
            verbose=True
        )
        
        print(f"\n  Transformation (recon2 -> recon1):")
        print(f"    Scale: {scale:.6f}")
        print(f"    Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
