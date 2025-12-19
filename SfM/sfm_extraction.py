"""
SfM reconstruction extraction utilities.

This module provides functions to extract sub-reconstructions from global sparse reconstructions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import pycolmap


def extract_sfm_reconstruction_from_global(
    global_sparse_reconstruction: pycolmap.Reconstruction,
    image_paths: List[Path],
    start_idx: int,
    end_idx: int,
    output_dir: Path,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    从全局稀疏重建中提取与指定批次影像相同的子重建
    
    Args:
        global_sparse_reconstruction: 全局稀疏重建对象
        image_paths: 所有影像路径列表
        start_idx: 起始影像索引
        end_idx: 结束影像索引（不包含）
        output_dir: 输出目录
        verbose: 是否输出详细信息
    
    Returns:
        包含子重建信息的字典，失败返回 None
        {
            'start_idx': int,
            'end_idx': int,
            'image_paths': List[Path],
            'image_name_mapping': Dict[str, str],
            'reconstruction': pycolmap.Reconstruction,
            'num_images': int,
            'num_points3D': int,
            'num_cameras': int,
            'source': str,
        }
    """
    try:
        if global_sparse_reconstruction is None:
            if verbose:
                print(f"  Warning: No global sparse reconstruction available")
            return None
        
        if verbose:
            print(f"  Extracting SfM reconstruction from global reconstruction for images {start_idx} to {end_idx-1}...")
        
        global_recon = global_sparse_reconstruction
        
        # ========== 优化1: 使用集合推导式一次性构建目标影像名称 ==========
        target_image_names = {image_paths[idx].name for idx in range(start_idx, end_idx)}
        
        # ========== 优化2: 缓存字典引用，减少属性查找 ==========
        global_images = global_recon.images
        global_cameras = global_recon.cameras
        global_points3D = global_recon.points3D
        
        # ========== 优化3: 一次遍历同时完成：匹配影像、收集3D点ID、收集相机ID ==========
        matched_image_ids = {}  # {global_image_id: local_image_id}
        required_point3D_ids: Set[int] = set()
        used_camera_ids: Set[int] = set()
        local_image_id = 1
        
        for global_image_id, image in global_images.items():
            if image.name in target_image_names:
                matched_image_ids[global_image_id] = local_image_id
                local_image_id += 1
                
                # 同时收集相机ID
                used_camera_ids.add(image.camera_id)
                
                # 同时收集3D点ID
                for point2D in image.points2D:
                    pt3d_id = point2D.point3D_id
                    if pt3d_id != -1:
                        required_point3D_ids.add(pt3d_id)
        
        if len(matched_image_ids) == 0:
            if verbose:
                print(f"  Warning: No matching images found in global reconstruction")
            return None
        
        if verbose:
            print(f"    Found {len(matched_image_ids)} matching images, {len(required_point3D_ids)} 3D points")
        
        # ========== 优化4: 创建子重建 ==========
        sub_recon = pycolmap.Reconstruction()
        
        # 添加相机（保持相同的camera_id）
        for camera_id in used_camera_ids:
            if camera_id in global_cameras:
                sub_recon.add_camera(global_cameras[camera_id])
        
        # ========== 优化5: 批量添加3D点，使用字典推导式预分配 ==========
        point3D_id_map = {}  # {global_point3D_id: new_point3D_id}
        
        # 预先过滤有效的3D点ID
        valid_point3D_ids = [pid for pid in required_point3D_ids if pid in global_points3D]
        
        for global_point3D_id in valid_point3D_ids:
            global_point = global_points3D[global_point3D_id]
            new_point3D_id = sub_recon.add_point3D(
                xyz=global_point.xyz,
                track=pycolmap.Track(),
                color=global_point.color
            )
            point3D_id_map[global_point3D_id] = new_point3D_id
        
        # ========== 优化6: 缓存 sub_recon.points3D 引用 ==========
        sub_points3D = sub_recon.points3D
        
        # ========== 优化7: 按 local_image_id 排序，使用列表推导式预排序 ==========
        sorted_image_pairs = sorted(matched_image_ids.items(), key=lambda x: x[1])
        
        for global_image_id, new_image_id in sorted_image_pairs:
            global_image = global_images[global_image_id]
            global_points2D = global_image.points2D
            
            # 预分配列表
            new_points2D = []
            
            # 使用局部变量缓存，减少字典查找
            point3D_id_map_get = point3D_id_map.get
            
            for point2D_idx, point2D in enumerate(global_points2D):
                pt3d_id = point2D.point3D_id
                new_point3D_id = point3D_id_map_get(pt3d_id) if pt3d_id != -1 else None
                
                if new_point3D_id is not None:
                    # 点在子重建中
                    new_points2D.append(pycolmap.Point2D(point2D.xy, new_point3D_id))
                    # 更新3D点的track信息
                    sub_points3D[new_point3D_id].track.add_element(new_image_id, point2D_idx)
                else:
                    # 点不在子重建中
                    new_points2D.append(pycolmap.Point2D(point2D.xy))
            
            # 创建新Image对象
            new_image = pycolmap.Image(
                image_id=new_image_id,
                name=global_image.name,
                camera_id=global_image.camera_id,
                cam_from_world=global_image.cam_from_world,
                points2D=new_points2D
            )
            sub_recon.add_image(new_image)
        
        # ========== 保存子重建结果 ==========
        temp_path = output_dir / "temp_sfm_extracted" / f"sfm_{start_idx}_{end_idx}"
        temp_path.mkdir(parents=True, exist_ok=True)
        sub_recon.write_text(str(temp_path))
        sub_recon.export_PLY(str(temp_path / "points3D.ply"))
        
        # ========== 构建结果字典 ==========
        image_paths_to_process = image_paths[start_idx:end_idx]
        image_name_to_path = {p.name: str(p) for p in image_paths_to_process}

        sfm_result = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'image_paths': image_paths_to_process,
            'image_name_mapping': image_name_to_path,
            'reconstruction': sub_recon,
            'num_images': len(sub_recon.images),
            'num_points3D': len(sub_recon.points3D),
            'num_cameras': len(sub_recon.cameras),
            'source': 'extracted_from_global',
        }
        
        if verbose:
            print(f"  ✓ SfM reconstruction extracted from global")
            print(f"    Number of images: {sfm_result['num_images']}")
            print(f"    Number of 3D points: {sfm_result['num_points3D']}")
            print(f"    Number of cameras: {sfm_result['num_cameras']}")
        
        return sfm_result
        
    except Exception as e:
        print(f"  Error extracting SfM reconstruction from global: {e}")
        import traceback
        traceback.print_exc()
        return None


