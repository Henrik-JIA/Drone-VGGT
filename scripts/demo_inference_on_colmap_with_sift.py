# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Demo script for SfM feature extraction and matching using pycolmap.

This script:
1. Extracts SIFT features using pycolmap
2. Performs feature matching
3. Builds 2D-2D track relationships from matches

Usage:
    python demo_inference_on_colmap_with_sfm_tracks.py \
        --colmap_path examples/Comprehensive_building_sel \
        --output_dir output/result \
        --verbose
"""

import argparse
import sys
import os
import copy

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from PIL import Image
import numpy as np
import torch
import pycolmap
from pycolmap import Rigid3d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))

from SfM.feature_matcher import FeatureMatcherSfM

from mapanything.models import MapAnything
from mapanything.utils.colmap import get_camera_matrix, qvec2rotmat, read_model
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import predictions_to_glb, predictions_to_ply, script_add_rerun_args

def build_2D_prior(
    image_path,
    output_dir,
    imgsz=2048,
    num_features=8192,
    match_mode="exhaustive",
    verbose=False
):
    """
    Run SfM feature extraction, matching, and 2D track building.
    
    Args:
        image_path: Path to images folder
        output_dir: Output directory for SfM results
        imgsz: Maximum image size for SIFT extraction (default: 2048)
        num_features: Number of SIFT features per image (default: 8192)
        match_mode: Feature matching mode - "exhaustive" or "spatial" (default: "exhaustive")
        verbose: Print detailed progress (default: False)
    
    Returns:
        pycolmap.Reconstruction: 2D prior reconstruction with tracks
    """
    print("Running SfM pipeline...")
    
    # Initialize feature matcher
    feature_matcher = FeatureMatcherSfM(
        input_dir=Path(image_path),
        output_dir=Path(output_dir),
        imgsz=imgsz,
        num_features=num_features,
        match_mode=match_mode,
        verbose=verbose,
    )
    
    # Run SfM pipeline steps
    if not feature_matcher.init_images():
        raise RuntimeError("Failed to initialize images")
    
    if not feature_matcher.init_crs():
        raise RuntimeError("Failed to initialize CRS")
    
    if not feature_matcher.init_pos():
        raise RuntimeError("Failed to initialize poses")
    
    if not feature_matcher.extract_features():
        raise RuntimeError("Failed to extract features")
    
    if not feature_matcher.match_features():
        raise RuntimeError("Failed to match features")
    
    print("✓ Feature extraction and matching completed")
    
    # Build 2D track relationships
    print("Building 2D-2D track relationships...")
    if not feature_matcher.build_2D_prior_recon():
        raise RuntimeError("Failed to build 2D prior reconstruction")
    
    prior_reconstruction = feature_matcher.rec_prior
    print(f"✓ Built 2D prior reconstruction")
    print(f"  Number of 2D tracks: {len(prior_reconstruction.points3D)}")
    print(f"  Number of images: {len(prior_reconstruction.images)}")
    
    # Export reconstruction
    if not feature_matcher.export_reconstruction():
        print("Warning: Failed to export 2D prior reconstruction")
    
    # # triangulation
    # if not feature_matcher.run_triangulation():
    #     print("Failed to run triangulation")
    # else:
    #     print("✓ Triangulation completed")

    return prior_reconstruction

def load_colmap_data(image_path, colmap_sparse_path, stride=1, verbose=False, ext=".bin"):
    """
    Load COLMAP format data for MapAnything inference.

    Args:
        image_path (str): Path to images folder
        colmap_sparse_path (str): Path to COLMAP sparse reconstruction folder
        stride (int): Load every nth image (default: 1)
        verbose (bool): Print progress messages
        ext (str): COLMAP file extension (".bin" or ".txt")

    Returns:
        list: List of view dictionaries for MapAnything inference
    """
    # Check that required folders exist
    if not os.path.exists(image_path):
        raise ValueError(f"Images folder not found at: {image_path}")
    if not os.path.exists(colmap_sparse_path):
        raise ValueError(f"COLMAP sparse folder not found at: {colmap_sparse_path}")

    if verbose:
        print(f"Images folder: {image_path}")
        print(f"Sparse folder: {colmap_sparse_path}")
        print(f"Using COLMAP file extension: {ext}")

    # Read COLMAP model
    try:
        cameras, images_colmap, points3D = read_model(colmap_sparse_path, ext=ext)
    except Exception as e:
        raise ValueError(f"Failed to read COLMAP model from {colmap_sparse_path}: {e}")

    if verbose:
        print(f"Loaded COLMAP model: {len(cameras)} cameras, {len(images_colmap)} images, {len(points3D)} 3D points")

    # Get list of available image files
    available_images = set()
    for f in os.listdir(image_path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            available_images.add(f)

    if not available_images:
        raise ValueError(f"No image files found in {image_path}")

    # Find posed and unposed images
    colmap_image_names = set(img_info.name for img_info in images_colmap.values())
    unposed_images = available_images - colmap_image_names

    if verbose:
        print(f"Posed images: {len(colmap_image_names)}")
        print(f"Unposed images: {len(unposed_images)}")

    views_example = []
    processed_count = 0

    # Process posed images (with COLMAP data)
    for img_id, img_info in images_colmap.items():
        if processed_count % stride != 0:
            processed_count += 1
            continue

        img_name = img_info.name
        img_file_path = os.path.join(image_path, img_name)
        
        if not os.path.exists(img_file_path):
            if verbose:
                print(f"Warning: Image file not found for {img_name}, skipping")
            processed_count += 1
            continue

        try:
            # Load image
            image = Image.open(img_file_path).convert("RGB")
            image_array = np.array(image).astype(np.uint8)

            # Get camera info
            cam_info = cameras[img_info.camera_id]
            K, _ = get_camera_matrix(camera_params=cam_info.params, camera_model=cam_info.model)

            # Get pose (world2cam -> cam2world)
            C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec
            world2cam_matrix = np.eye(4)
            world2cam_matrix[:3, :3] = C_R_G
            world2cam_matrix[:3, 3] = C_t_G
            pose_matrix = closed_form_pose_inverse(world2cam_matrix[None, :, :])[0]

            # Convert to tensors
            image_tensor = torch.from_numpy(image_array)
            intrinsics_tensor = torch.from_numpy(K.astype(np.float32))
            pose_tensor = torch.from_numpy(pose_matrix.astype(np.float32))

            view = {
                "img": image_tensor,
                "intrinsics": intrinsics_tensor,
                "camera_poses": pose_tensor,
                "is_metric_scale": torch.tensor([False]),
            }

            views_example.append(view)
            processed_count += 1

            if verbose:
                print(f"Loaded posed view {len(views_example) - 1}: {img_name} (shape: {image_array.shape})")

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {img_name}: {e}")
            processed_count += 1
            continue
    
    # Process unposed images (without COLMAP data)
    for img_name in unposed_images:
        if processed_count % stride != 0:
            processed_count += 1
            continue

        img_file_path = os.path.join(image_path, img_name)

        try:
            image = Image.open(img_file_path).convert("RGB")
            image_array = np.array(image).astype(np.uint8)
            image_tensor = torch.from_numpy(image_array)

            view = {"img": image_tensor}
            views_example.append(view)
            processed_count += 1

            if verbose:
                print(f"Loaded unposed view {len(views_example) - 1}: {img_name} (shape: {image_array.shape})")

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {img_name}: {e}")
            processed_count += 1
            continue

    if not views_example:
        raise ValueError("No valid images found")

    if verbose:
        print(f"Successfully loaded {len(views_example)} views with stride={stride}")

    return views_example

def load_and_preprocess_colmap_views(
    image_path,
    colmap_sparse_dir,
    stride=1,
    ext=".txt",
    verbose=False
):
    """
    Load COLMAP data and preprocess for MapAnything inference.
    
    Args:
        image_path: Path to images folder
        colmap_sparse_dir: Path to COLMAP sparse reconstruction folder
        stride: Load every nth image (default: 1)
        ext: COLMAP file extension ".bin" or ".txt" (default: ".txt")
        verbose: Print detailed progress (default: False)
    
    Returns:
        list: Preprocessed views ready for model.infer()
    """
    if verbose:
        print(f"Loading COLMAP data from: {colmap_sparse_dir}")
    
    # Load COLMAP data
    views_example = load_colmap_data(
        str(image_path),
        str(colmap_sparse_dir),
        stride=stride,
        verbose=verbose,
        ext=ext,
    )
    
    print(f"✓ Loaded {len(views_example)} views")
    
    # Preprocess inputs
    if verbose:
        print("Preprocessing inputs...")
    processed_views = preprocess_inputs(views_example, verbose=False)

    # ========== 计算每张影像的缩放信息 ==========
    scale_info_list = []
    
    for idx, (view_orig, view_proc) in enumerate(zip(views_example, processed_views)):
        # 原始尺寸: view_orig['img'] shape = (H, W, 3)
        orig_h, orig_w = view_orig['img'].shape[0], view_orig['img'].shape[1]
        
        # 处理后尺寸: view_proc['img'] shape = (1, 3, H', W')
        proc_h, proc_w = view_proc['img'].shape[2], view_proc['img'].shape[3]
        
        # 计算缩放比例
        scale_x = proc_w / orig_w
        scale_y = proc_h / orig_h
        
        scale_info = {
            'view_idx': idx,
            'original_size': (orig_w, orig_h),
            'output_size': (proc_w, proc_h),
            'scale_x': scale_x,
            'scale_y': scale_y,
        }
        scale_info_list.append(scale_info)
    
    if verbose:
        print(f"✓ Computed scale info for {len(scale_info_list)} views")
        if scale_info_list:
            info = scale_info_list[0]
            print(f"  Example (view 0):")
            print(f"    Original: {info['original_size']}")
            print(f"    Output: {info['output_size']}")
            print(f"    Scale: x={info['scale_x']:.4f}, y={info['scale_y']:.4f}")
    
    return processed_views, scale_info_list

def extract_3D_points_from_outputs(outputs, verbose=False):
    """
    从 MapAnything 推理输出中提取每个影像的 3D 点、图像和掩码信息。
    
    Args:
        outputs: List of prediction dictionaries from model.infer()
        verbose: Print detailed progress (default: False)
    
    Returns:
        tuple: (world_points, images, masks)
            - world_points: np.ndarray of shape (num_views, H, W, 3) - 每个像素的世界坐标
            - images: np.ndarray of shape (num_views, H, W, 3) - RGB 图像
            - masks: np.ndarray of shape (num_views, H, W) - 有效像素掩码
    """
    if verbose:
        print("Extracting 3D points from outputs...")
    
    world_points_list = []
    images_list = []
    masks_list = []
    
    # 处理每个视图
    for view_idx, pred in enumerate(outputs):
        # 获取深度图和相机参数
        depthmap_torch = pred["depth_z"][0].squeeze(-1)
        intrinsics_torch = pred["intrinsics"][0]
        camera_pose_torch = pred["camera_poses"][0]
        
        # 将深度图转换为世界坐标系下的 3D 点
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )
        
        # 获取掩码和图像
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # 结合有效深度掩码
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()
        
        # 存储数据
        world_points_list.append(pts3d_np)
        images_list.append(image_np)
        masks_list.append(mask)
        
        if verbose:
            print(f"  Processed view {view_idx + 1}/{len(outputs)}")
    
    # 堆叠所有视图
    world_points = np.stack(world_points_list, axis=0)
    images = np.stack(images_list, axis=0)
    masks = np.stack(masks_list, axis=0)
    
    if verbose:
        print(f"✓ Extracted 3D points for {len(outputs)} views")
        print(f"  Shape: world_points={world_points.shape}, images={images.shape}, masks={masks.shape}")
    
    return world_points, images, masks

def build_2D_3D_correspondences(
    prior_reconstruction,
    scale_info_list,
    outputs,
    image_path,
    output_dir,
    use_all_features=True,
    verbose=False
):
    """
    建立 2D 匹配点与 3D 世界坐标的对应关系，并保存到文件。
    
    Args:
        prior_reconstruction: pycolmap.Reconstruction object with 2D tracks
        scale_info_list: List of scale info dicts for each view
        world_points: np.ndarray of shape (N, H, W, 3) - 3D world coordinates
        images: np.ndarray of shape (N, H, W, 3) - RGB images
        masks: np.ndarray of shape (N, H, W) - valid pixel masks
        image_path: Path to image directory
        output_dir: Output directory for saving correspondences
        verbose: Print detailed progress (default: False)
    
    Returns:
        dict: all_correspondences - 每张图片的 2D-3D 对应关系
    """
    print("\n建立 2D 匹配点与 3D 世界坐标的对应关系...")
    
    # ========== 获取每个影像对应像素的三维点信息  ========== 

    print("\nExtracting 3D points from each view...")
    world_points, images, masks = extract_3D_points_from_outputs(
        outputs, 
        verbose=args.verbose
    )
    
    if prior_reconstruction is None or len(prior_reconstruction.images) == 0:
        print("警告: 没有可用的 prior_reconstruction 数据")
        return {}
    
    all_correspondences = {}
    total_2d_points = 0              # 提取的2D特征点总数
    total_2d_with_track = 0          # 有SfM track的2D点数
    total_2d_3d_correspondences = 0  # 成功建立的2D-3D对应

    for view_idx, (img_id, img) in enumerate(prior_reconstruction.images.items()):
        
        # 提取预测相机位姿 (cam2world)
        camera_pose = outputs[view_idx]["camera_poses"][0].cpu().numpy()  # [4, 4]
        # 转换为 COLMAP 格式 (world2cam)
        world2cam = np.linalg.inv(camera_pose)
        predict_cam_R = world2cam[:3, :3]
        predict_cam_t = world2cam[:3, 3]
        
        img_name = img.name
        img_path = str(Path(image_path) / img_name)
        
        scale = scale_info_list[view_idx]
        orig_w, orig_h = scale['original_size']
        out_w, out_h = scale['output_size']
        scale_x, scale_y = scale['scale_x'], scale['scale_y']
        
        # 提取2D点（原始尺寸坐标）
        points_2d_original = []
        track_ids = []
        
        if use_all_features:
            for point2D in img.points2D:
                points_2d_original.append(point2D.xy)
                track_ids.append(point2D.point3D_id)
            
            num_matched = sum(1 for tid in track_ids if tid != 18446744073709551615)
            if verbose:
                print(f"  {img_name}: 提取全部 {len(points_2d_original)} 个特征点 "
                      f"(其中 {num_matched} 个有 SfM track, {len(points_2d_original) - num_matched} 个无 SfM tracks)")
        else:            
            # 只使用已匹配的特征点
            for point2D in img.points2D:
                if point2D.point3D_id != 18446744073709551615:
                    points_2d_original.append(point2D.xy)
                    track_ids.append(point2D.point3D_id)
            if verbose:
                print(f"  {img_name}: 使用 {len(points_2d_original)} 个已匹配特征点")
        
        if len(points_2d_original) == 0:
            if verbose:
                print(f"  {img_name}: 没有特征点")
            continue
        
        points_2d_original = np.array(points_2d_original)
        total_2d_points += len(points_2d_original)

        # 坐标转换：原始尺寸 -> 输出尺寸
        points_2d_scaled = points_2d_original.copy()
        points_2d_scaled[:, 0] *= scale_x  # x 坐标
        points_2d_scaled[:, 1] *= scale_y  # y 坐标
        
        # 从 world_points 中获取对应的 3D 坐标
        matched_3d_points = []
        matched_colors = []
        valid_indices = []      # 存储的是：原始列表中的索引位置（0, 1, 2, 3, ...）
        valid_track_ids = []    # 存储的是：SfM track ID（可能是正数，也可能是 -1）
        
        for idx, (xy_scaled, track_id) in enumerate(zip(points_2d_scaled, track_ids)):
            x_int = int(np.round(xy_scaled[0]))
            y_int = int(np.round(xy_scaled[1]))
            
            # 边界检查和有效性检查
            if 0 <= x_int < out_w and 0 <= y_int < out_h:
                if masks[view_idx, y_int, x_int]:
                    point_3d = world_points[view_idx, y_int, x_int]
                    color = images[view_idx, y_int, x_int]
                    
                    matched_3d_points.append(point_3d)
                    matched_colors.append(color)
                    valid_indices.append(idx)
                    valid_track_ids.append(track_id)
        
        if len(matched_3d_points) > 0:
            # 统计有多少点有 track
            num_with_tracks = sum(1 for tid in valid_track_ids if tid != 18446744073709551615)
            num_without_tracks = len(valid_track_ids) - num_with_tracks
            total_2d_with_track += num_with_tracks
            total_2d_3d_correspondences += len(matched_3d_points)

            all_correspondences[img_name] = {
                'view_idx': view_idx,
                'img_path': img_path,
                'predict_cam_R': predict_cam_R,
                'predict_cam_t': predict_cam_t,
                'points_2d_original': points_2d_original[valid_indices],
                'points_2d_scaled': points_2d_scaled[valid_indices],
                'points_3d': np.array(matched_3d_points),
                'colors': np.array(matched_colors),
                'track_ids': [track_ids[i] for i in valid_indices],
                'num_points_2d_original': len(points_2d_original),
                'num_points_2d_scaled': len(points_2d_scaled),
                'num_points_3d': len(matched_3d_points),
                'num_colors': len(matched_colors),
                'num_track_ids': len(track_ids),
                'num_with_tracks': num_with_tracks,
                'num_without_tracks': num_without_tracks,
                'original_width': orig_w,
                'original_height': orig_h,
                'output_width': out_w,
                'output_height': out_h,
                'scale_info_x': scale_x,
                'scale_info_y': scale_y,
            }
            
            print(f"  ✓ {img_name}: {len(matched_3d_points)}/{len(points_2d_original)} 个 2D-3D 对应 "
                  f"(有track: {num_with_tracks}, 无track: {num_without_tracks})")
    
    # 打印总体统计
    print(f"\n总体统计:")
    print(f"  提取的2D特征点总数: {total_2d_points}")
    print(f"  成功建立的2D-3D对应: {total_2d_3d_correspondences} ({total_2d_3d_correspondences/total_2d_points*100:.1f}%)")
    print(f"  其中有 SfM track (多视图): {total_2d_with_track}")
    print(f"  其中无 SfM track (单视图): {total_2d_3d_correspondences - total_2d_with_track}")
    
    # 保存所有对应关系
    if all_correspondences:
        output_dir = Path(output_dir)
        correspondence_path = output_dir / "2d_3d_correspondences.npz"
        
        save_dict = {}
        image_names = []
        
        for img_name, data in all_correspondences.items():
            # 使用索引作为键
            idx = data['view_idx']
            save_dict[f'view_{idx}_img_path'] = data['img_path']
            save_dict[f'view_{idx}_predict_cam_R'] = data['predict_cam_R']
            save_dict[f'view_{idx}_predict_cam_t'] = data['predict_cam_t']
            save_dict[f'view_{idx}_points_2d_original'] = data['points_2d_original']
            save_dict[f'view_{idx}_points_2d_scaled'] = data['points_2d_scaled']
            save_dict[f'view_{idx}_points_3d'] = data['points_3d']
            save_dict[f'view_{idx}_colors'] = data['colors']
            save_dict[f'view_{idx}_track_ids'] = np.array(data['track_ids'])
            save_dict[f'view_{idx}_num_points_2d_original'] = data['num_points_2d_original']
            save_dict[f'view_{idx}_num_points_2d_scaled'] = data['num_points_2d_scaled']
            save_dict[f'view_{idx}_num_points_3d'] = data['num_points_3d']
            save_dict[f'view_{idx}_num_colors'] = data['num_colors']
            save_dict[f'view_{idx}_num_track_ids'] = data['num_track_ids']
            save_dict[f'view_{idx}_num_with_tracks'] = data['num_with_tracks']
            save_dict[f'view_{idx}_num_without_tracks'] = data['num_without_tracks']
            save_dict[f'view_{idx}_original_size'] = np.array([data['original_width'], data['original_height']])
            save_dict[f'view_{idx}_output_size'] = np.array([data['output_width'], data['output_height']])
            save_dict[f'view_{idx}_scale_info_x'] = data['scale_info_x']
            save_dict[f'view_{idx}_scale_info_y'] = data['scale_info_y']
            image_names.append(img_name)
        
        # 保存影像名称列表
        save_dict['image_names'] = np.array(image_names, dtype=object)
        save_dict['num_views'] = len(all_correspondences)
        
        np.savez(correspondence_path, **save_dict)
        print(f"\n✓ 保存 {len(all_correspondences)} 张影像的 2D-3D 对应关系到: {correspondence_path}")
    else:
        print("\n警告: 没有建立任何 2D-3D 对应关系")
    
    return all_correspondences

# ============================================================
# ============================================================


# def build_reconstruction_from_correspondences(
#     prior_reconstruction,
#     all_correspondences,
#     output_dir,
#     verbose=False
# ):
#     """
#     基于 2D-3D 对应关系构建新的 COLMAP Reconstruction。
    
#     该函数使用 MapAnything 预测的 3D 坐标替换传统 SfM 三角化的结果，
#     同时保留相机参数、图片信息和 2D-3D 观测关系。
    
#     Args:
#         prior_reconstruction: 原始的 pycolmap.Reconstruction（包含 2D tracks）
#         all_correspondences: 2D-3D 对应关系字典（从 build_2D_3D_correspondences 返回）
#         output_dir: 输出目录
#         verbose: 是否打印详细信息
    
#     Returns:
#         pycolmap.Reconstruction: 新的 reconstruction 对象
#     """

    
#     print("\n基于 2D-3D 对应关系构建新的 COLMAP Reconstruction...")
    
#     if not all_correspondences:
#         print("警告: 没有 2D-3D 对应关系，无法构建 Reconstruction")
#         return None
    
#     # 创建新的 Reconstruction 对象
#     new_reconstruction = pycolmap.Reconstruction()
    
#     # ========== 1. 复制相机信息 ==========
#     if verbose:
#         print("复制相机信息...")
    
#     for cam_id, camera in prior_reconstruction.cameras.items():
#         new_reconstruction.add_camera(camera)
    
#     if verbose:
#         print(f"  ✓ 复制了 {len(new_reconstruction.cameras)} 个相机")
    
#     # ========== 2. 收集所有唯一的 3D 点 ==========
#     if verbose:
#         print("收集 3D 点信息...")
    
#     # 使用字典来存储唯一的 3D 点
#     # key: track_id, value: {'xyz': 3D坐标, 'color': RGB颜色, 'observations': [(image_id, point2D_idx), ...]}
#     points_3d_data = {}
    
#     for img_name, corr_data in all_correspondences.items():
#         view_idx = corr_data['view_idx']
#         points_3d = corr_data['points_3d']
#         colors = corr_data['colors']
#         track_ids = corr_data['track_ids']
        
#         for i, track_id in enumerate(track_ids):
#             xyz = points_3d[i]
#             rgb = colors[i]
            
#             if track_id not in points_3d_data:
#                 # 首次遇到该 track，初始化
#                 points_3d_data[track_id] = {
#                     'xyz': xyz,
#                     'color': rgb,
#                     'xyz_list': [xyz],  # 用于后续求平均
#                     'color_list': [rgb],
#                     'count': 1
#                 }
#             else:
#                 # 该 track 已存在，累加坐标（后续求平均）
#                 points_3d_data[track_id]['xyz_list'].append(xyz)
#                 points_3d_data[track_id]['color_list'].append(rgb)
#                 points_3d_data[track_id]['count'] += 1
    
#     # 对每个 track 的多个观测求平均 3D 坐标和颜色
#     for track_id, data in points_3d_data.items():
#         if data['count'] > 1:
#             # 求平均坐标
#             avg_xyz = np.mean(data['xyz_list'], axis=0)
#             avg_color = np.mean(data['color_list'], axis=0)
#             data['xyz'] = avg_xyz
#             data['color'] = avg_color
    
#     if verbose:
#         print(f"  ✓ 收集了 {len(points_3d_data)} 个唯一的 3D 点")
    
#     # ========== 3. 添加图片信息 ==========
#     if verbose:
#         print("添加图片信息...")
    
#     # 首先需要建立 track_id 到新 point3D_id 的映射
#     track_id_to_point3d_id = {}
    
#     for img_id, img in prior_reconstruction.images.items():
#         img_name = img.name
        
#         # 检查该图片是否在 correspondences 中
#         if img_name not in all_correspondences:
#             if verbose:
#                 print(f"  警告: {img_name} 没有对应关系，跳过")
#             continue
        
#         corr_data = all_correspondences[img_name]
        
#         # 添加图片（使用原始相机参数和位姿）
#         new_img = pycolmap.Image(
#             id=img_id,
#             name=img_name,
#             camera_id=img.camera_id,
#             cam_from_world=img.cam_from_world  # 保留原始位姿
#         )
        
#         # 添加 2D 点观测
#         # 注意: prior_reconstruction 中的 points2D 已经包含了所有特征点
#         # 我们需要更新它们的 point3D_id
        
#         points2D_list = []
#         for point2D in img.points2D:
#             if point2D.point3D_id != -1 and point2D.point3D_id in corr_data['track_ids']:
#                 # 该 2D 点有对应的 3D 点
#                 # 记录 track_id 映射（稍后统一分配 point3D_id）
#                 track_id_to_point3d_id[point2D.point3D_id] = None  # 占位
            
#             points2D_list.append(point2D)
        
#         new_img.points2D = points2D_list
#         new_reconstruction.add_image(new_img)
    
#     if verbose:
#         print(f"  ✓ 添加了 {len(new_reconstruction.images)} 张图片")
    
#     # ========== 4. 添加 3D 点到 Reconstruction ==========
#     if verbose:
#         print("添加 3D 点到 Reconstruction...")
    
#     # 首先为每个 track_id 分配新的 point3D_id
#     for idx, track_id in enumerate(sorted(points_3d_data.keys())):
#         track_id_to_point3d_id[track_id] = idx
    
#     # 添加 3D 点
#     for track_id, data in points_3d_data.items():
#         point3d_id = track_id_to_point3d_id[track_id]
        
#         xyz = data['xyz']
#         rgb = data['color']
        
#         # 转换颜色到 [0, 255] 范围
#         if rgb.max() <= 1.0:
#             rgb_uint8 = (rgb * 255).astype(np.uint8)
#         else:
#             rgb_uint8 = rgb.astype(np.uint8)
        
#         # 创建 Point3D 对象
#         point3d = pycolmap.Point3D(
#             xyz=xyz,
#             color=rgb_uint8,
#             error=0.0,  # 可以设置为 reprojection error
#             track=pycolmap.Track()  # 稍后添加观测
#         )
        
#         new_reconstruction.add_point3D(point3d_id, point3d)
    
#     if verbose:
#         print(f"  ✓ 添加了 {len(new_reconstruction.points3D)} 个 3D 点")
    
#     # ========== 5. 建立 2D-3D 观测关系 ==========
#     if verbose:
#         print("建立 2D-3D 观测关系...")
    
#     # 更新图片中的 point3D_id
#     for img_id, img in new_reconstruction.images.items():
#         img_name = img.name
        
#         if img_name not in all_correspondences:
#             continue
        
#         corr_data = all_correspondences[img_name]
#         track_ids = corr_data['track_ids']
        
#         # 获取原始图片的 points2D
#         orig_img = prior_reconstruction.images[img_id]
        
#         # 创建新的 points2D 列表
#         new_points2D = []
#         point2D_idx = 0
        
#         for orig_point2D in orig_img.points2D:
#             if orig_point2D.point3D_id != -1 and orig_point2D.point3D_id in track_ids:
#                 # 该 2D 点有对应的 3D 点
#                 old_track_id = orig_point2D.point3D_id
#                 new_point3d_id = track_id_to_point3d_id[old_track_id]
                
#                 # 创建新的 Point2D，更新 point3D_id
#                 new_point2D = pycolmap.Point2D(
#                     xy=orig_point2D.xy,
#                     point3D_id=new_point3d_id
#                 )
                
#                 # 添加观测到 3D 点的 track
#                 if new_point3d_id in new_reconstruction.points3D:
#                     point3d = new_reconstruction.points3D[new_point3d_id]
#                     point3d.track.add_element(img_id, point2D_idx)
                
#                 new_points2D.append(new_point2D)
#             else:
#                 # 该 2D 点没有对应的 3D 点
#                 new_points2D.append(orig_point2D)
            
#             point2D_idx += 1
        
#         # 更新图片的 points2D
#         img.points2D = new_points2D
    
#     if verbose:
#         print("  ✓ 建立了 2D-3D 观测关系")
    
#     # ========== 6. 导出新的 Reconstruction ==========
#     output_dir = Path(output_dir)
#     sparse_output_dir = output_dir / "sparse_mapanything"
#     os.makedirs(sparse_output_dir, exist_ok=True)
    
#     # 导出为文本格式
#     new_reconstruction.write_text(str(sparse_output_dir))
#     print(f"\n✓ 新的 Reconstruction 已保存到: {sparse_output_dir}")
    
#     # 打印统计信息
#     print("\n新 Reconstruction 统计信息:")
#     print(f"  相机数量: {len(new_reconstruction.cameras)}")
#     print(f"  图片数量: {len(new_reconstruction.images)}")
#     print(f"  3D 点数量: {len(new_reconstruction.points3D)}")
    
#     # 计算平均 track 长度
#     if len(new_reconstruction.points3D) > 0:
#         avg_track_length = sum(len(p.track.elements) for p in new_reconstruction.points3D.values()) / len(new_reconstruction.points3D)
#         print(f"  平均 track 长度: {avg_track_length:.2f}")
    
#     return new_reconstruction

# =================================================
# =================================================
# =================================================
# def build_reconstruction_from_correspondences(
#     prior_reconstruction,
#     all_correspondences,
#     outputs,
#     scale_info_list,
#     output_dir,
#     verbose=False
# ):
#     """
#     基于 2D-3D 对应关系构建新的 COLMAP Reconstruction。
    
#     该函数创建一个基于缩放后影像尺寸的新 Reconstruction，
#     使用 MapAnything 预测的 3D 坐标和缩放后的相机内参。
    
#     Args:
#         prior_reconstruction: 原始的 pycolmap.Reconstruction（包含 2D tracks，原始尺寸）
#         all_correspondences: 2D-3D 对应关系字典
#         outputs: MapAnything 的推理输出（包含缩放后的相机内参）
#         scale_info_list: 每张影像的缩放信息列表
#         output_dir: 输出目录
#         verbose: 是否打印详细信息
    
#     Returns:
#         pycolmap.Reconstruction: 新的 reconstruction 对象（基于缩放后的尺寸）
#     """
    
#     print("\n基于 2D-3D 对应关系构建新的 COLMAP Reconstruction...")
    
#     if not all_correspondences:
#         print("警告: 没有 2D-3D 对应关系，无法构建 Reconstruction")
#         return None
    
#     # 创建新的 Reconstruction 对象
#     new_reconstruction = pycolmap.Reconstruction()
    
#     # ========== 1. 创建缩放后的相机 ==========
#     if verbose:
#         print("创建缩放后的相机...")
    
#     camera_id_mapping = {}
    
#     for view_idx, output in enumerate(outputs):
#         img_id_list = list(prior_reconstruction.images.keys())
#         if view_idx >= len(img_id_list):
#             continue
            
#         img_id = img_id_list[view_idx]
#         img = prior_reconstruction.images[img_id]
#         old_camera_id = img.camera_id
#         old_camera = prior_reconstruction.cameras[old_camera_id]
        
#         if old_camera_id in camera_id_mapping:
#             continue
        
#         # 从 output 中获取缩放后的相机内参
#         intrinsics = output["intrinsics"][0].cpu().numpy()
        
#         # 获取缩放后的影像尺寸
#         scale_info = scale_info_list[view_idx]
#         output_w, output_h = scale_info['output_size']
        
#         # 提取内参参数
#         fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#         cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
#         # 根据原相机模型创建新相机
#         if old_camera.model.name == "PINHOLE":
#             params = np.array([fx, fy, cx, cy])
#         elif old_camera.model.name == "SIMPLE_PINHOLE":
#             f = (fx + fy) / 2.0
#             params = np.array([f, cx, cy])
#         elif old_camera.model.name == "SIMPLE_RADIAL":
#             f = (fx + fy) / 2.0
#             params = np.array([f, cx, cy, 0.0])
#         elif old_camera.model.name == "RADIAL":
#             f = (fx + fy) / 2.0
#             params = np.array([f, cx, cy, 0.0, 0.0])
#         else:
#             params = np.array([fx, fy, cx, cy])
#             if verbose:
#                 print(f"  警告: 相机模型 {old_camera.model.name} 不常见，使用 PINHOLE")
        
#         new_camera = pycolmap.Camera(
#             model=old_camera.model.name if old_camera.model.name in ["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"] else "PINHOLE",
#             width=output_w,
#             height=output_h,
#             params=params,
#             camera_id=old_camera_id
#         )
        
#         new_reconstruction.add_camera(new_camera)
#         camera_id_mapping[old_camera_id] = old_camera_id
        
#         if verbose:
#             print(f"  相机 {old_camera_id}: {old_camera.width}x{old_camera.height} -> {output_w}x{output_h}")
    
#     if verbose:
#         print(f"  ✓ 创建了 {len(new_reconstruction.cameras)} 个缩放后的相机")
    
#     # ========== 2. 收集所有唯一的 3D 点并求平均 ==========
#     if verbose:
#         print("收集 3D 点信息...")
    
#     track_observations = {}
    
#     for img_name, corr_data in all_correspondences.items():
#         points_3d = corr_data['points_3d']
#         colors = corr_data['colors']
#         track_ids = corr_data['track_ids']
        
#         for i, track_id in enumerate(track_ids):
#             xyz = points_3d[i]
#             rgb = colors[i]
            
#             if track_id not in track_observations:
#                 track_observations[track_id] = {
#                     'xyz_list': [xyz],
#                     'color_list': [rgb],
#                 }
#             else:
#                 track_observations[track_id]['xyz_list'].append(xyz)
#                 track_observations[track_id]['color_list'].append(rgb)
    
#     # 对每个 track 的多个观测求平均 3D 坐标和颜色
#     for track_id, obs_data in track_observations.items():
#         xyz_list = obs_data['xyz_list']
#         color_list = obs_data['color_list']
        
#         avg_xyz = np.mean(xyz_list, axis=0)
#         avg_color = np.mean(color_list, axis=0)
        
#         if avg_color.max() <= 1.0:
#             rgb_uint8 = (avg_color * 255).astype(np.uint8)
#         else:
#             rgb_uint8 = avg_color.astype(np.uint8)
        
#         obs_data['xyz'] = avg_xyz
#         obs_data['color'] = rgb_uint8
#         obs_data['count'] = len(xyz_list)
    
#     if verbose:
#         print(f"  ✓ 收集了 {len(track_observations)} 个唯一的 3D 点")
#         multi_view_tracks = sum(1 for obs in track_observations.values() if obs['count'] > 1)
#         if multi_view_tracks > 0:
#             print(f"  ✓ 其中 {multi_view_tracks} 个点有多视图观测")
#             max_views = max(obs['count'] for obs in track_observations.values())
#             print(f"  ✓ 最多观测次数: {max_views}")
    
#     # ========== 3. 添加 3D 点到 Reconstruction 并建立映射 ==========
#     if verbose:
#         print("添加 3D 点...")
    
#     track_id_to_point3d_id = {}
    
#     for track_id, obs_data in track_observations.items():
#         avg_xyz = obs_data['xyz']
#         rgb_uint8 = obs_data['color']
        
#         # 创建空的 Track（稍后添加观测）
#         track = pycolmap.Track()
        
#         # add_point3D 返回新分配的 point3D_id
#         new_point3d_id = new_reconstruction.add_point3D(avg_xyz, track, rgb_uint8)
        
#         # 记录映射关系
#         track_id_to_point3d_id[track_id] = new_point3d_id
    
#     if verbose:
#         print(f"  ✓ 添加了 {len(new_reconstruction.points3D)} 个 3D 点")
    
#     # ========== 4. 添加图片信息（使用缩放后的相机和位姿） ==========
#     if verbose:
#         print("添加图片信息...")
    
#     for view_idx, output in enumerate(outputs):
#         img_id_list = list(prior_reconstruction.images.keys())
#         if view_idx >= len(img_id_list):
#             continue
            
#         img_id = img_id_list[view_idx]
#         img = prior_reconstruction.images[img_id]
#         img_name = img.name
        
#         if img_name not in all_correspondences:
#             if verbose:
#                 print(f"  警告: {img_name} 没有对应关系，跳过")
#             continue
        
#         # 从 output 中获取位姿
#         camera_pose = output["camera_poses"][0].cpu().numpy()
#         world2cam = np.linalg.inv(camera_pose)
#         R = world2cam[:3, :3]
#         t = world2cam[:3, 3]
        
#         cam_from_world = Rigid3d(rotation=R, translation=t)
        
#         # 创建新图片对象
#         new_img = pycolmap.Image(
#             id=img_id,
#             name=img_name,
#             camera_id=img.camera_id,
#             cam_from_world=cam_from_world
#         )
        
#         # 添加 2D 点观测（使用缩放后的坐标和新的 point3D_id）
#         corr_data = all_correspondences[img_name]
#         points_2d_scaled = corr_data['points_2d_scaled']
#         track_ids = corr_data['track_ids']
        
#         points2D_list = []
#         for i, (xy, old_track_id) in enumerate(zip(points_2d_scaled, track_ids)):
#             # 使用映射获取新的 point3D_id
#             new_point3d_id = track_id_to_point3d_id.get(old_track_id, -1)
            
#             point2D = pycolmap.Point2D(
#                 xy=xy,
#                 point3D_id=new_point3d_id
#             )
#             points2D_list.append(point2D)
        
#         new_img.points2D = points2D_list
#         new_reconstruction.add_image(new_img)
    
#     if verbose:
#         print(f"  ✓ 添加了 {len(new_reconstruction.images)} 张图片")
    
#     # ========== 5. 建立 Track 信息 ==========
#     if verbose:
#         print("建立 Track 信息...")
    
#     for img_id, img in new_reconstruction.images.items():
#         for point2D_idx, point2D in enumerate(img.points2D):
#             point3d_id = point2D.point3D_id
#             if point3d_id != -1 and point3d_id in new_reconstruction.points3D:
#                 point3d = new_reconstruction.points3D[point3d_id]
#                 point3d.track.add_element(img_id, point2D_idx)
    
#     if verbose:
#         print("  ✓ 建立了 Track 信息")
    
#     # ========== 5. Bundle Adjustment 优化 ==========
#     print("\n执行 Bundle Adjustment 优化...")
#     print(f"  优化前统计: {len(new_reconstruction.points3D)} 个 3D 点")
    
#     ba_options = pycolmap.BundleAdjustmentOptions()
#     ba_options.print_summary = True
    
#     try:
#         # 执行 BA 优化
#         summary = pycolmap.bundle_adjustment(new_reconstruction, ba_options)
#         print("✓ Bundle Adjustment 完成")
        
#         # 打印优化结果
#         if hasattr(summary, 'num_residuals'):
#             print(f"  残差数量: {summary.num_residuals}")
#         if hasattr(summary, 'initial_cost'):
#             print(f"  初始代价: {summary.initial_cost:.6f}")
#         if hasattr(summary, 'final_cost'):
#             print(f"  最终代价: {summary.final_cost:.6f}")
#             if summary.initial_cost > 0:
#                 improvement = (1 - summary.final_cost/summary.initial_cost) * 100
#                 print(f"  优化改进: {improvement:.2f}%")
                
#     except Exception as e:
#         print(f"⚠ Bundle Adjustment 失败: {e}")
#         print("  继续使用未优化的结果...")

#     # ========== 6. 导出新的 Reconstruction ==========
#     output_dir = Path(output_dir)
#     sparse_output_dir = output_dir / "sparse_mapanything"
#     os.makedirs(sparse_output_dir, exist_ok=True)
    
#     new_reconstruction.write_text(str(sparse_output_dir))
#     print(f"\n✓ 新的 Reconstruction 已保存到: {sparse_output_dir}")
    
#     # 打印统计信息
#     print("\n新 Reconstruction 统计信息:")
#     print(f"  相机数量: {len(new_reconstruction.cameras)}")
#     print(f"  图片数量: {len(new_reconstruction.images)}")
#     print(f"  注册图片数量: {new_reconstruction.num_reg_images()}")
#     print(f"  3D 点数量: {len(new_reconstruction.points3D)}")
    
#     if len(new_reconstruction.points3D) > 0:
#         track_lengths = [len(p.track.elements) for p in new_reconstruction.points3D.values()]
#         avg_track_length = sum(track_lengths) / len(track_lengths)
#         max_track_length = max(track_lengths)
#         min_track_length = min(track_lengths)
#         print(f"  平均 track 长度: {avg_track_length:.2f}")
#         print(f"  Track 长度范围: [{min_track_length}, {max_track_length}]")
    
#     return new_reconstruction

# ==========================================================================
# def build_reconstruction_from_correspondences(
#     prior_reconstruction,
#     all_correspondences,
#     outputs,
#     scale_info_list,
#     output_dir,
#     verbose=False
# ):
#     """
#     基于 2D-3D 对应关系构建新的 COLMAP Reconstruction。
    
#     参考 batch_np_matrix_to_pycolmap 的实现逻辑。
#     """
    
#     print("\n基于 2D-3D 对应关系构建新的 COLMAP Reconstruction...")
    
#     if not all_correspondences:
#         print("警告: 没有 2D-3D 对应关系，无法构建 Reconstruction")
#         return None
    
#     # 创建新的 Reconstruction 对象
#     new_reconstruction = pycolmap.Reconstruction()
    
#     # ========== 1. 收集所有唯一的 track_id 和对应的观测 ==========
#     if verbose:
#         print("收集唯一的 3D 点...")
    
#     # track_id -> {'xyz_list': [], 'color_list': [], 'img_observations': [(img_name, point_idx), ...]}
#     track_observations = {}
    
#     # 同时记录每张图片的信息
#     image_data = {}  # img_name -> {'points_2d_scaled': [], 'track_ids': [], 'points_3d': [], 'colors': []}
    
#     for img_name, corr_data in all_correspondences.items():
#         points_2d_scaled = corr_data['points_2d_scaled']
#         points_3d = corr_data['points_3d']
#         colors = corr_data['colors']
#         track_ids = corr_data['track_ids']
        
#         # 存储图片数据
#         image_data[img_name] = {
#             'points_2d_scaled': points_2d_scaled,
#             'track_ids': track_ids,
#             'points_3d': points_3d,
#             'colors': colors
#         }
        
#         # 收集每个 track 的观测
#         for point_idx, track_id in enumerate(track_ids):
#             xyz = points_3d[point_idx]
#             rgb = colors[point_idx]
            
#             if track_id not in track_observations:
#                 track_observations[track_id] = {
#                     'xyz_list': [xyz],
#                     'color_list': [rgb],
#                     'img_observations': [(img_name, point_idx)]
#                 }
#             else:
#                 track_observations[track_id]['xyz_list'].append(xyz)
#                 track_observations[track_id]['color_list'].append(rgb)
#                 track_observations[track_id]['img_observations'].append((img_name, point_idx))
    
#     # ========== 2. 筛选有效的 3D 点（至少在2张图中可见）==========
#     valid_tracks = {}
#     old_track_id_to_new_point3d_id = {}
    
#     for old_track_id, obs_data in track_observations.items():
#         num_observations = len(obs_data['img_observations'])
        
#         # 只保留至少在2张图中可见的点
#         if num_observations >= 2:
#             valid_tracks[old_track_id] = obs_data
    
#     if verbose:
#         print(f"  ✓ 收集了 {len(track_observations)} 个唯一的 track")
#         print(f"  ✓ 其中 {len(valid_tracks)} 个 track 在至少2张图中可见")
#         if len(valid_tracks) > 0:
#             obs_counts = [len(obs['img_observations']) for obs in valid_tracks.values()]
#             print(f"  ✓ 观测次数范围: [{min(obs_counts)}, {max(obs_counts)}]")
#             print(f"  ✓ 平均观测次数: {sum(obs_counts) / len(obs_counts):.2f}")
    
#     # ========== 3. 添加 3D 点（对多次观测求平均）==========
#     if verbose:
#         print("添加 3D 点到 Reconstruction...")
    
#     for old_track_id, obs_data in valid_tracks.items():
#         # 对多次观测求平均
#         avg_xyz = np.mean(obs_data['xyz_list'], axis=0)
#         avg_color = np.mean(obs_data['color_list'], axis=0)
        
#         # 确保颜色是 uint8 格式
#         if avg_color.max() <= 1.0:
#             rgb_uint8 = (avg_color * 255).astype(np.uint8)
#         else:
#             rgb_uint8 = avg_color.astype(np.uint8)
        
#         # 创建空 track（稍后会添加观测）
#         track = pycolmap.Track()
        
#         # 添加 3D 点，返回新分配的 point3D_id
#         new_point3d_id = new_reconstruction.add_point3D(avg_xyz, track, rgb_uint8)
        
#         # 记录映射关系
#         old_track_id_to_new_point3d_id[old_track_id] = new_point3d_id
    
#     if verbose:
#         print(f"  ✓ 添加了 {len(new_reconstruction.points3D)} 个 3D 点")
    
#     # ========== 4. 创建缩放后的相机 ==========
#     if verbose:
#         print("创建缩放后的相机...")

#     # 使用 set 跟踪已添加的相机 ID
#     added_camera_ids = set()

#     for view_idx, output in enumerate(outputs):
#         img_id_list = list(prior_reconstruction.images.keys())
#         if view_idx >= len(img_id_list):
#             continue
            
#         img_id = img_id_list[view_idx]
#         img = prior_reconstruction.images[img_id]
#         img_name = img.name
#         old_camera_id = img.camera_id
#         old_camera = prior_reconstruction.cameras[old_camera_id]
        
#         # 如果该相机已经创建过，跳过
#         if old_camera_id in added_camera_ids:
#             continue
        
#         # 从 output 中获取缩放后的相机内参
#         intrinsics = output["intrinsics"][0].cpu().numpy()
        
#         # 获取缩放后的影像尺寸
#         scale_info = scale_info_list[view_idx]
#         output_w, output_h = scale_info['output_size']
        
#         # 提取内参参数
#         fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#         cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
#         # 根据原相机模型创建新相机参数
#         if old_camera.model.name == "PINHOLE":
#             params = np.array([fx, fy, cx, cy])
#             model_name = "PINHOLE"
#         elif old_camera.model.name == "SIMPLE_PINHOLE":
#             f = (fx + fy) / 2.0
#             params = np.array([f, cx, cy])
#             model_name = "SIMPLE_PINHOLE"
#         elif old_camera.model.name == "SIMPLE_RADIAL":
#             f = (fx + fy) / 2.0
#             params = np.array([f, cx, cy, 0.0])
#             model_name = "SIMPLE_RADIAL"
#         elif old_camera.model.name == "RADIAL":
#             f = (fx + fy) / 2.0
#             params = np.array([f, cx, cy, 0.0, 0.0])
#             model_name = "RADIAL"
#         else:
#             params = np.array([fx, fy, cx, cy])
#             model_name = "PINHOLE"
#             if verbose:
#                 print(f"  警告: 相机模型 {old_camera.model.name} 不常见，使用 PINHOLE")
        
#         new_camera = pycolmap.Camera(
#             model=model_name,
#             width=output_w,
#             height=output_h,
#             params=params,
#             camera_id=old_camera_id
#         )
        
#         new_reconstruction.add_camera(new_camera)
#         added_camera_ids.add(old_camera_id)  # 记录已添加的相机 ID
        
#         if verbose:
#             print(f"  相机 {old_camera_id}: {old_camera.width}x{old_camera.height} -> {output_w}x{output_h}")

#     if verbose:
#         print(f"  ✓ 创建了 {len(new_reconstruction.cameras)} 个缩放后的相机")
        
#     # ========== 5. 添加图片和 2D 观测（类似 batch_np_matrix_to_pycolmap）==========
#     if verbose:
#         print("添加图片和 2D 观测...")

#     use_sfm_poses = True 

#     for view_idx, output in enumerate(outputs):
#         img_id_list = list(prior_reconstruction.images.keys())
#         if view_idx >= len(img_id_list):
#             continue
            
#         img_id = img_id_list[view_idx]
#         img = prior_reconstruction.images[img_id]
#         img_name = img.name
        
#         if img_name not in image_data:
#             if verbose:
#                 print(f"  警告: {img_name} 没有对应关系，跳过")
#             continue
        
#         if use_sfm_poses:
#             # 使用 SfM 优化的位姿（推荐）
#             cam_from_world = img.cam_from_world
#         else:
#             # 使用 MapAnything 预测的位姿
#             camera_pose = output["camera_poses"][0].cpu().numpy()
#             world2cam = np.linalg.inv(camera_pose)
#             R = world2cam[:3, :3]
#             t = world2cam[:3, 3]
#             cam_from_world = Rigid3d(rotation=R, translation=t)
        

#         # 创建新图片对象（使用原始的 camera_id）
#         new_img = pycolmap.Image(
#             id=img_id,
#             name=img_name,
#             camera_id=img.camera_id,  # 使用原始的 camera_id
#             cam_from_world=cam_from_world
#         )
        
#         # 添加 2D 点观测
#         img_data = image_data[img_name]
#         points_2d_scaled = img_data['points_2d_scaled']
#         track_ids = img_data['track_ids']
        
#         points2D_list = []
#         point2D_idx = 0
        
#         for i, old_track_id in enumerate(track_ids):
#             # 只添加有效的 track（在至少2张图中可见）
#             if old_track_id in old_track_id_to_new_point3d_id:
#                 new_point3d_id = old_track_id_to_new_point3d_id[old_track_id]
#                 xy = points_2d_scaled[i]
                
#                 # 创建 Point2D 并关联到 3D 点
#                 point2D = pycolmap.Point2D(xy=xy, point3D_id=new_point3d_id)
#                 points2D_list.append(point2D)
                
#                 # 同时更新 3D 点的 track 信息
#                 if new_point3d_id in new_reconstruction.points3D:
#                     track = new_reconstruction.points3D[new_point3d_id].track
#                     track.add_element(img_id, point2D_idx)
                
#                 point2D_idx += 1
        
#         # 设置图片的 2D 点
#         new_img.points2D = points2D_list
#         new_reconstruction.add_image(new_img)
    
#     if verbose:
#         print(f"  ✓ 添加了 {len(new_reconstruction.images)} 张图片")
    
#     # ========== 6. 清理没有观测的 3D 点 ==========
#     points_to_remove = [
#         point3D_id for point3D_id in new_reconstruction.points3D
#         if len(new_reconstruction.points3D[point3D_id].track.elements) == 0
#     ]
#     for point3D_id in points_to_remove:
#         del new_reconstruction.points3D[point3D_id]
    
#     if points_to_remove and verbose:
#         print(f"  清理了 {len(points_to_remove)} 个没有观测的 3D 点")
    
#     # # ========== 7. Bundle Adjustment 优化 ==========
#     # print("\n执行 Bundle Adjustment 优化...")
#     # print(f"  优化前统计: {len(new_reconstruction.points3D)} 个 3D 点")
    
#     # ba_options = pycolmap.BundleAdjustmentOptions()
#     # ba_options.print_summary = True
    
#     # try:
#     #     summary = pycolmap.bundle_adjustment(new_reconstruction, ba_options)
#     #     print("✓ Bundle Adjustment 完成")
        
#     #     if hasattr(summary, 'num_residuals'):
#     #         print(f"  残差数量: {summary.num_residuals}")
#     #     if hasattr(summary, 'initial_cost'):
#     #         print(f"  初始代价: {summary.initial_cost:.6f}")
#     #     if hasattr(summary, 'final_cost'):
#     #         print(f"  最终代价: {summary.final_cost:.6f}")
#     #         if summary.initial_cost > 0:
#     #             improvement = (1 - summary.final_cost/summary.initial_cost) * 100
#     #             print(f"  优化改进: {improvement:.2f}%")
                
#     # except Exception as e:
#     #     print(f"⚠ Bundle Adjustment 失败: {e}")
#     #     print("  继续使用未优化的结果...")

#     # ========== 8. 导出新的 Reconstruction ==========
#     output_dir = Path(output_dir)
#     sparse_output_dir = output_dir / "sparse_mapanything"
#     os.makedirs(sparse_output_dir, exist_ok=True)
    
#     new_reconstruction.write_text(str(sparse_output_dir))
#     new_reconstruction.export_PLY(sparse_output_dir / "sparse_mapanything_points.ply")
#     print(f"\n✓ 新的 Reconstruction 已保存到: {sparse_output_dir}")
    
#     # 打印统计信息
#     print("\n新 Reconstruction 统计信息:")
#     print(f"  相机数量: {len(new_reconstruction.cameras)}")
#     print(f"  图片数量: {len(new_reconstruction.images)}")
#     print(f"  注册图片数量: {new_reconstruction.num_reg_images()}")
#     print(f"  3D 点数量: {len(new_reconstruction.points3D)}")
    
#     if len(new_reconstruction.points3D) > 0:
#         track_lengths = [len(p.track.elements) for p in new_reconstruction.points3D.values()]
#         avg_track_length = sum(track_lengths) / len(track_lengths)
#         max_track_length = max(track_lengths)
#         min_track_length = min(track_lengths)
#         print(f"  平均 track 长度: {avg_track_length:.2f}")
#         print(f"  Track 长度范围: [{min_track_length}, {max_track_length}]")
    
#     return new_reconstruction

def build_reconstruction_from_correspondences_simple(
    prior_reconstruction,
    all_correspondences,
    output_dir,
    verbose=False
):
    """
    基于 2D-3D 对应关系构建新的 COLMAP Reconstruction（简化版）。
    
    策略：为每个2D-3D对应创建独立的3D点，不考虑track关系。
    
    Args:
        prior_reconstruction: 原始的pycolmap Reconstruction（包含相机和位姿）
        all_correspondences: 2D-3D对应关系字典（包含scale_info和3D点）
        output_dir: 输出目录
        verbose: 是否打印详细信息
    
    Returns:
        new_reconstruction: 新的pycolmap.Reconstruction对象
    """
    
    print("\n基于 2D-3D 对应关系构建新的 COLMAP Reconstruction（简化版）...")
    print("  策略: 为每个2D-3D对应创建独立的3D点")
    
    if not all_correspondences:
        print("警告: 没有 2D-3D 对应关系，无法构建 Reconstruction")
        return None
    
    # 创建新的 Reconstruction 对象
    new_reconstruction = pycolmap.Reconstruction()
    
    # ========== 1. 创建缩放后的相机 ==========
    if verbose:
        print("创建缩放后的相机...")

    added_camera_ids = set()
    
    # 直接遍历 prior_reconstruction 的图片
    for img_id, img in prior_reconstruction.images.items():
        img_name = img.name
        
        # 检查该图片是否在 correspondences 中
        if img_name not in all_correspondences:
            continue
        
        old_camera_id = img.camera_id
        
        # 如果该相机已经创建过，跳过
        if old_camera_id in added_camera_ids:
            continue
        
        old_camera = prior_reconstruction.cameras[old_camera_id]
        corr_data = all_correspondences[img_name]
        
        # 从 corr_data 中获取缩放信息
        output_w, output_h = corr_data['output_width'], corr_data['output_height']
        scale_x, scale_y = corr_data['scale_info_x'], corr_data['scale_info_y']
        
        # 缩放原始相机内参
        old_params = old_camera.params
        
        if old_camera.model.name == "PINHOLE":
            fx = old_params[0] * scale_x
            fy = old_params[1] * scale_y
            cx = old_params[2] * scale_x
            cy = old_params[3] * scale_y
            params = np.array([fx, fy, cx, cy])
            model_name = "PINHOLE"
            
        elif old_camera.model.name == "SIMPLE_PINHOLE":
            f_x = old_params[0] * scale_x
            f_y = old_params[0] * scale_y
            f = (f_x + f_y) / 2.0
            cx = old_params[1] * scale_x
            cy = old_params[2] * scale_y
            params = np.array([f, cx, cy])
            model_name = "SIMPLE_PINHOLE"
            
        elif old_camera.model.name == "SIMPLE_RADIAL":
            f_x = old_params[0] * scale_x
            f_y = old_params[0] * scale_y
            f = (f_x + f_y) / 2.0
            cx = old_params[1] * scale_x
            cy = old_params[2] * scale_y
            k = old_params[3]  # 畸变系数不缩放
            params = np.array([f, cx, cy, k])
            model_name = "SIMPLE_RADIAL"
            
        elif old_camera.model.name == "RADIAL":
            f_x = old_params[0] * scale_x
            f_y = old_params[0] * scale_y
            f = (f_x + f_y) / 2.0
            cx = old_params[1] * scale_x
            cy = old_params[2] * scale_y
            k1, k2 = old_params[3], old_params[4]
            params = np.array([f, cx, cy, k1, k2])
            model_name = "RADIAL"
            
        else:
            if len(old_params) >= 4:
                fx = old_params[0] * scale_x
                fy = old_params[1] * scale_y
                cx = old_params[2] * scale_x
                cy = old_params[3] * scale_y
                params = np.array([fx, fy, cx, cy])
            else:
                params = old_params
            model_name = "PINHOLE"
            if verbose:
                print(f"  警告: 相机模型 {old_camera.model.name} 不常见，使用 PINHOLE")
        
        new_camera = pycolmap.Camera(
            model=model_name,
            width=output_w,
            height=output_h,
            params=params,
            camera_id=old_camera_id
        )
        
        new_reconstruction.add_camera(new_camera)
        added_camera_ids.add(old_camera_id)
        
        if verbose:
            print(f"  相机 {old_camera_id} ({img_name}): {old_camera.width}x{old_camera.height} -> {output_w}x{output_h}")

    if verbose:
        print(f"  ✓ 创建了 {len(new_reconstruction.cameras)} 个缩放后的相机")
    
    # ========== 2. 为所有2D-3D对应创建独立的3D点 ==========
    if verbose:
        print("创建独立的3D点...")
    
    # 记录每张图片的2D点和对应的3D点ID
    image_point_mapping = {}
    
    for img_name, corr_data in all_correspondences.items():
        points_2d_scaled = corr_data['points_2d_scaled']
        points_3d = corr_data['points_3d']
        colors = corr_data['colors']
        
        image_point_mapping[img_name] = {
            'points_2d': [],
            'point3D_ids': []
        }
        
        # 为每个2D-3D对应创建一个独立的3D点
        for i in range(len(points_2d_scaled)):
            xyz = points_3d[i]
            rgb = colors[i]
            xy_2d = points_2d_scaled[i]
            
            # 确保颜色是 uint8 格式
            if rgb.max() <= 1.0:
                rgb_uint8 = (rgb * 255).astype(np.uint8)
            else:
                rgb_uint8 = rgb.astype(np.uint8)
            
            # 创建空 track
            track = pycolmap.Track()
            
            # 添加 3D 点
            point3D_id = new_reconstruction.add_point3D(xyz, track, rgb_uint8)
            
            # 记录映射
            image_point_mapping[img_name]['points_2d'].append(xy_2d)
            image_point_mapping[img_name]['point3D_ids'].append(point3D_id)
    
    if verbose:
        print(f"  ✓ 创建了 {len(new_reconstruction.points3D)} 个独立的3D点")
    
    # ========== 3. 添加图片和建立2D-3D关联 ==========
    if verbose:
        print("添加图片和建立2D-3D关联...")
    
    # 直接遍历 prior_reconstruction 的图片
    for img_id, img in prior_reconstruction.images.items():
        img_name = img.name
        
        if img_name not in image_point_mapping:
            if verbose:
                print(f"  警告: {img_name} 没有对应关系，跳过")
            continue
        
        # # 使用 prior_reconstruction 中的 SfM 位姿
        # cam_from_world = img.cam_from_world

        # 使用MapAnything的预测位姿
        predict_cam_R = all_correspondences[img_name]['predict_cam_R']
        predict_cam_t = all_correspondences[img_name]['predict_cam_t']
        quat_xyzw = Rotation.from_matrix(predict_cam_R).as_quat()  # [x, y, z, w]
        quat_wxyz = np.roll(quat_xyzw, 1)  # 转换为 [w, x, y, z]
        cam_from_world = pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(quat_wxyz),
            translation=predict_cam_t
        )
        
        # 创建新图片对象（保持原始的 img_id 和 camera_id）
        new_img = pycolmap.Image(
            id=img_id,
            name=img_name,
            camera_id=img.camera_id,
            cam_from_world=cam_from_world
        )
        
        # 添加 2D 点观测
        points_2d = image_point_mapping[img_name]['points_2d']
        point3D_ids = image_point_mapping[img_name]['point3D_ids']
        
        points2D_list = []
        
        for point2D_idx, (xy, point3D_id) in enumerate(zip(points_2d, point3D_ids)):
            # 创建 Point2D 并关联到 3D 点
            point2D = pycolmap.Point2D(xy=xy, point3D_id=point3D_id)
            points2D_list.append(point2D)
            
            # 更新 3D 点的 track 信息
            if point3D_id in new_reconstruction.points3D:
                track = new_reconstruction.points3D[point3D_id].track
                track.add_element(img_id, point2D_idx)
        
        # 设置图片的 2D 点
        new_img.points2D = points2D_list
        new_reconstruction.add_image(new_img)
        
        if verbose:
            print(f"  {img_name}: 添加了 {len(points2D_list)} 个 2D-3D 对应")
    
    if verbose:
        print(f"  ✓ 添加了 {len(new_reconstruction.images)} 张图片")
    
    # ========== 4. 导出新的 Reconstruction ==========
    output_dir = Path(output_dir)
    sparse_output_dir = output_dir / "sparse_mapanything"
    os.makedirs(sparse_output_dir, exist_ok=True)
    
    new_reconstruction.write_text(str(sparse_output_dir))
    new_reconstruction.export_PLY(sparse_output_dir / "sparse_mapanything_points.ply")
    print(f"\n✓ 新的 Reconstruction 已保存到: {sparse_output_dir}")
    
    # 打印统计信息
    print("\n新 Reconstruction 统计信息:")
    print(f"  相机数量: {len(new_reconstruction.cameras)}")
    print(f"  图片数量: {len(new_reconstruction.images)}")
    print(f"  注册图片数量: {new_reconstruction.num_reg_images()}")
    print(f"  3D 点数量: {len(new_reconstruction.points3D)}")
    
    if len(new_reconstruction.points3D) > 0:
        track_lengths = [len(p.track.elements) for p in new_reconstruction.points3D.values()]
        single_view_count = sum(1 for l in track_lengths if l == 1)
        multi_view_count = sum(1 for l in track_lengths if l > 1)
        print(f"  单视图3D点: {single_view_count}")
        print(f"  多视图3D点: {multi_view_count}")
        if multi_view_count > 0:
            avg_track_length = sum(track_lengths) / len(track_lengths)
            max_track_length = max(track_lengths)
            print(f"  平均 track 长度: {avg_track_length:.2f}")
            print(f"  最大 track 长度: {max_track_length}")
    
    return new_reconstruction

def align_and_restore_reconstruction_scale(
    new_reconstruction,
    prior_reconstruction,
    output_dir=None,
    method='proj_centers',  # 'points', 'reprojections', 或 'proj_centers'
    verbose=False
):
    """
    使用 pycolmap 的对齐函数恢复 new_reconstruction 的尺度
    
    Args:
        new_reconstruction: 需要对齐的重建对象
        prior_reconstruction: 原始参考重建对象
        output_dir: 输出目录，用于保存对齐后的重建结果（可选）
        method: 对齐方法 ('points', 'reprojections', 'proj_centers')
        verbose: 是否输出详细信息
    
    Returns:
        sim3d: Sim3d 变换对象，如果对齐失败则返回 None
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("使用 pycolmap 对齐重建")
        print("=" * 80)
    
    # 选择对齐方法
    sim3d = None
    
    if method == 'points':
        if verbose:
            print("使用方法: align_reconstructions_via_points")
        sim3d = pycolmap.align_reconstructions_via_points(
            src_reconstruction=new_reconstruction,
            tgt_reconstruction=prior_reconstruction,
            min_common_observations=3,
            max_error=0.01,  # 可以调整
            min_inlier_ratio=0.3  # 可以调整
        )
    elif method == 'reprojections':
        if verbose:
            print("使用方法: align_reconstructions_via_reprojections")
        sim3d = pycolmap.align_reconstructions_via_reprojections(
            src_reconstruction=new_reconstruction,
            tgt_reconstruction=prior_reconstruction,
            min_inlier_observations=0.3,
            max_reproj_error=8.0
        )
    elif method == 'proj_centers':
        if verbose:
            print("使用方法: align_reconstructions_via_proj_centers")
        sim3d = pycolmap.align_reconstructions_via_proj_centers(
            src_reconstruction=new_reconstruction,
            tgt_reconstruction=prior_reconstruction,
            max_proj_center_error=0.12
        )
    else:
        raise ValueError(f"未知的对齐方法: {method}")
    
    if sim3d is None:
        if verbose:
            print("⚠ 警告: 对齐失败，返回 None")
        return None, None
    
    if verbose:
        print(f"✓ 对齐成功!")
        print(f"  - 尺度因子: {sim3d.scale:.6f}")
        print(f"  - 旋转矩阵: {sim3d.rotation}")
        print(f"  - 平移向量: {sim3d.translation}")
    
    # 应用变换到 new_reconstruction
    if verbose:
        print(f"\n应用 Sim3d 变换到重建...")
    
    new_reconstruction.transform(sim3d)
    
    if verbose:
        print("✓ 变换应用完成!")
        print(f"  - 变换了 {len(new_reconstruction.points3D)} 个3D点")
        print(f"  - 变换了 {len(new_reconstruction.images)} 个相机位姿")
    
    # 导出对齐后的重建结果
    if output_dir is not None:
        output_dir = Path(output_dir)
        sparse_output_dir = output_dir / "sparse_mapanything_aligned"
        os.makedirs(sparse_output_dir, exist_ok=True)
        
        if verbose:
            print(f"\n导出对齐后的重建结果...")
        
        # 保存为文本格式
        new_reconstruction.write_text(str(sparse_output_dir))
        
        # 导出PLY点云
        ply_path = sparse_output_dir / "sparse_mapanything_aligned_points.ply"
        new_reconstruction.export_PLY(str(ply_path))
        
        if verbose:
            print(f"✓ 对齐后的 Reconstruction 已保存到: {sparse_output_dir}")
            print(f"  - COLMAP 文本文件: {sparse_output_dir}")
            print(f"  - 点云文件: {ply_path}")
    
    if verbose:
        print("=" * 80)
    
    return new_reconstruction, sim3d

# def restore_reconstruction_to_original_scale(
#     reconstruction,
#     all_correspondences,
#     image_path,
#     resample_colors_from_original=False,
#     verbose=False
# ):
#     """
#     将对齐后的 reconstruction 恢复到原始图像尺寸。
    
#     由于 MapAnything 会对图像进行缩放处理，reconstruction 中的相机内参和 2D 点坐标
#     都是基于缩放后的尺寸。此函数将它们恢复到原始图像尺寸。
    
#     Args:
#         reconstruction: 对齐后的 pycolmap.Reconstruction 对象
#         all_correspondences: 2D-3D 对应关系字典（包含缩放信息）
#         image_path: 原始图像路径
#         resample_colors_from_original: 是否从原始图像重新采样 3D 点的颜色
#         verbose: 是否打印详细信息
    
#     Returns:
#         reconstruction: 恢复到原始尺寸的 reconstruction 对象
#     """
    
#     if verbose:
#         print("\n" + "=" * 80)
#         print("恢复 Reconstruction 到原始图像尺寸")
#         print("=" * 80)
    
#     # 用于收集需要删除的3D点ID（如果坐标超出原始图像范围）
#     points_to_remove = set()
    
#     # ========== 1. 处理每张图片的相机和2D点 ==========
#     for img_id, img in reconstruction.images.items():
#         img_name = img.name
        
#         # 检查该图片是否在 correspondences 中
#         if img_name not in all_correspondences:
#             if verbose:
#                 print(f"  警告: {img_name} 不在 correspondences 中，跳过")
#             continue
        
#         corr_data = all_correspondences[img_name]
        
#         # 获取缩放信息
#         orig_w = corr_data['original_width']
#         orig_h = corr_data['original_height']
#         output_w = corr_data['output_width']
#         output_h = corr_data['output_height']
#         scale_x = corr_data['scale_info_x']
#         scale_y = corr_data['scale_info_y']
        
#         # 计算逆缩放比例
#         inv_scale_x = 1.0 / scale_x
#         inv_scale_y = 1.0 / scale_y
        
#         if verbose:
#             print(f"\n处理图片: {img_name}")
#             print(f"  缩放尺寸: {output_w}x{output_h} -> 原始尺寸: {orig_w}x{orig_h}")
#             print(f"  逆缩放比例: x={inv_scale_x:.4f}, y={inv_scale_y:.4f}")
        
#         # ========== 1.1 恢复相机内参 ==========
#         camera = reconstruction.cameras[img.camera_id]
#         old_params = camera.params.copy()
        
#         if camera.model.name == "PINHOLE":
#             # PINHOLE: [fx, fy, cx, cy]
#             fx = old_params[0] * inv_scale_x
#             fy = old_params[1] * inv_scale_y
#             cx = old_params[2] * inv_scale_x
#             cy = old_params[3] * inv_scale_y
#             camera.params = np.array([fx, fy, cx, cy])
            
#         elif camera.model.name == "SIMPLE_PINHOLE":
#             # SIMPLE_PINHOLE: [f, cx, cy]
#             # 对于各向异性缩放，需要计算平均
#             f_x = old_params[0] * inv_scale_x
#             f_y = old_params[0] * inv_scale_y
#             f = (f_x + f_y) / 2.0
#             cx = old_params[1] * inv_scale_x
#             cy = old_params[2] * inv_scale_y
#             camera.params = np.array([f, cx, cy])
            
#         elif camera.model.name == "SIMPLE_RADIAL":
#             # SIMPLE_RADIAL: [f, cx, cy, k]
#             f_x = old_params[0] * inv_scale_x
#             f_y = old_params[0] * inv_scale_y
#             f = (f_x + f_y) / 2.0
#             cx = old_params[1] * inv_scale_x
#             cy = old_params[2] * inv_scale_y
#             k = old_params[3]  # 畸变系数不缩放
#             camera.params = np.array([f, cx, cy, k])
            
#         elif camera.model.name == "RADIAL":
#             # RADIAL: [f, cx, cy, k1, k2]
#             f_x = old_params[0] * inv_scale_x
#             f_y = old_params[0] * inv_scale_y
#             f = (f_x + f_y) / 2.0
#             cx = old_params[1] * inv_scale_x
#             cy = old_params[2] * inv_scale_y
#             k1, k2 = old_params[3], old_params[4]
#             camera.params = np.array([f, cx, cy, k1, k2])
            
#         else:
#             # 其他相机模型，假设前4个参数是 [fx, fy, cx, cy]
#             if len(old_params) >= 4:
#                 new_params = old_params.copy()
#                 new_params[0] *= inv_scale_x  # fx
#                 new_params[1] *= inv_scale_y  # fy
#                 new_params[2] *= inv_scale_x  # cx
#                 new_params[3] *= inv_scale_y  # cy
#                 camera.params = new_params
#             if verbose:
#                 print(f"  警告: 相机模型 {camera.model.name} 不常见")
        
#         # 更新相机尺寸
#         camera.width = orig_w
#         camera.height = orig_h
        
#         if verbose:
#             print(f"  ✓ 相机内参已恢复: {output_w}x{output_h} -> {orig_w}x{orig_h}")
        
#         # ========== 1.2 恢复2D点坐标 ==========
#         # 如果需要重新采样颜色，加载原始图像
#         original_image_np = None
#         if resample_colors_from_original:
#             original_image_path = os.path.join(image_path, img_name)
#             try:
#                 original_image = Image.open(original_image_path).convert("RGB")
#                 original_image_np = np.array(original_image)
#                 if verbose:
#                     print(f"  ✓ 加载原始图像用于颜色重采样")
#             except Exception as e:
#                 if verbose:
#                     print(f"  警告: 加载原始图像失败: {e}")
        
#         # 遍历所有2D点
#         num_points_out_of_bounds = 0
#         num_points_resampled = 0
        
#         for point2D in img.points2D:
#             # 恢复2D点坐标到原始尺寸
#             point2D.xy = np.array([
#                 point2D.xy[0] * inv_scale_x,
#                 point2D.xy[1] * inv_scale_y
#             ])
            
#             # 检查是否有关联的3D点
#             if point2D.point3D_id != -1 and point2D.point3D_id != 18446744073709551615:
#                 x, y = int(point2D.xy[0]), int(point2D.xy[1])
                
#                 # 检查坐标是否在原始图像范围内
#                 is_in_bounds = (0 <= x < orig_w and 0 <= y < orig_h)
                
#                 if is_in_bounds:
#                     # 在范围内：如果需要，更新颜色
#                     if original_image_np is not None:
#                         try:
#                             color = original_image_np[y, x]
#                             point3D = reconstruction.points3D[point2D.point3D_id]
#                             point3D.color = color.astype(np.uint8)
#                             num_points_resampled += 1
#                         except:
#                             pass
#                 else:
#                     # 超出范围：标记为需要删除
#                     points_to_remove.add(point2D.point3D_id)
#                     num_points_out_of_bounds += 1
        
#         if verbose:
#             print(f"  ✓ 2D点坐标已恢复: {len(img.points2D)} 个点")
#             if num_points_out_of_bounds > 0:
#                 print(f"  ⚠ {num_points_out_of_bounds} 个点超出原始图像范围，将被删除")
#             if num_points_resampled > 0:
#                 print(f"  ✓ {num_points_resampled} 个3D点颜色已重新采样")
    
#     # ========== 2. 删除超出范围的3D点 ==========
#     if points_to_remove:
#         if verbose:
#             print(f"\n删除 {len(points_to_remove)} 个超出原始图像范围的3D点...")
        
#         for point_id in points_to_remove:
#             if point_id in reconstruction.points3D:
#                 del reconstruction.points3D[point_id]
        
#         if verbose:
#             print(f"✓ 删除完成，剩余 {len(reconstruction.points3D)} 个3D点")
    
#     if verbose:
#         print("=" * 80)
#         print("✓ Reconstruction 已恢复到原始图像尺寸")
#         print("=" * 80)
    
#     return reconstruction

def restore_reconstruction_to_original_scale(
    reconstruction,
    all_correspondences,
    image_path,
    output_dir=None,  # 新增参数：输出目录
    resample_colors_from_original=False,
    verbose=False
):
    """
    将对齐后的 reconstruction 恢复到原始图像尺寸。
    
    由于 MapAnything 会对图像进行缩放处理，reconstruction 中的相机内参和 2D 点坐标
    都是基于缩放后的尺寸。此函数将它们恢复到原始图像尺寸。
    
    Args:
        reconstruction: 对齐后的 pycolmap.Reconstruction 对象
        all_correspondences: 2D-3D 对应关系字典（包含缩放信息）
        image_path: 原始图像路径
        output_dir: 输出目录，用于保存恢复后的重建结果（可选）
        resample_colors_from_original: 是否从原始图像重新采样 3D 点的颜色
        verbose: 是否打印详细信息
    
    Returns:
        reconstruction: 恢复到原始尺寸的 reconstruction 对象
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("恢复 Reconstruction 到原始图像尺寸")
        print("=" * 80)
    
    # 用于收集需要删除的3D点ID
    points_to_remove = set()
    
    # ========== 关键修改：跟踪已处理的相机 ==========
    processed_cameras = set()  # 记录已经处理过的相机ID
    
    # ========== 1. 处理每张图片的相机和2D点 ==========
    for img_id, img in reconstruction.images.items():
        img_name = img.name
        
        # 检查该图片是否在 correspondences 中
        if img_name not in all_correspondences:
            if verbose:
                print(f"  警告: {img_name} 不在 correspondences 中，跳过")
            continue
        
        corr_data = all_correspondences[img_name]
        
        # 获取缩放信息
        orig_w = corr_data['original_width']
        orig_h = corr_data['original_height']
        output_w = corr_data['output_width']
        output_h = corr_data['output_height']
        scale_x = corr_data['scale_info_x']
        scale_y = corr_data['scale_info_y']
        
        # 计算逆缩放比例
        inv_scale_x = 1.0 / scale_x
        inv_scale_y = 1.0 / scale_y
        
        if verbose:
            print(f"\n处理图片: {img_name}")
            print(f"  缩放尺寸: {output_w}x{output_h} -> 原始尺寸: {orig_w}x{orig_h}")
            print(f"  逆缩放比例: x={inv_scale_x:.4f}, y={inv_scale_y:.4f}")
        
        # ========== 1.1 恢复相机内参（只处理一次）==========
        camera_id = img.camera_id
        
        # ========== 关键修改：检查相机是否已处理 ==========
        if camera_id not in processed_cameras:
            camera = reconstruction.cameras[camera_id]
            old_params = camera.params.copy()
            
            if verbose:
                print(f"  处理相机 {camera_id} (首次)")
            
            if camera.model.name == "PINHOLE":
                # PINHOLE: [fx, fy, cx, cy]
                fx = old_params[0] * inv_scale_x
                fy = old_params[1] * inv_scale_y
                cx = old_params[2] * inv_scale_x
                cy = old_params[3] * inv_scale_y
                camera.params = np.array([fx, fy, cx, cy])
                
            elif camera.model.name == "SIMPLE_PINHOLE":
                # SIMPLE_PINHOLE: [f, cx, cy]
                f_x = old_params[0] * inv_scale_x
                f_y = old_params[0] * inv_scale_y
                f = (f_x + f_y) / 2.0
                cx = old_params[1] * inv_scale_x
                cy = old_params[2] * inv_scale_y
                camera.params = np.array([f, cx, cy])
                
            elif camera.model.name == "SIMPLE_RADIAL":
                # SIMPLE_RADIAL: [f, cx, cy, k]
                f_x = old_params[0] * inv_scale_x
                f_y = old_params[0] * inv_scale_y
                f = (f_x + f_y) / 2.0
                cx = old_params[1] * inv_scale_x
                cy = old_params[2] * inv_scale_y
                k = old_params[3]  # 畸变系数不缩放
                camera.params = np.array([f, cx, cy, k])
                
            elif camera.model.name == "RADIAL":
                # RADIAL: [f, cx, cy, k1, k2]
                f_x = old_params[0] * inv_scale_x
                f_y = old_params[0] * inv_scale_y
                f = (f_x + f_y) / 2.0
                cx = old_params[1] * inv_scale_x
                cy = old_params[2] * inv_scale_y
                k1, k2 = old_params[3], old_params[4]
                camera.params = np.array([f, cx, cy, k1, k2])
                
            else:
                # 其他相机模型
                if len(old_params) >= 4:
                    new_params = old_params.copy()
                    new_params[0] *= inv_scale_x  # fx
                    new_params[1] *= inv_scale_y  # fy
                    new_params[2] *= inv_scale_x  # cx
                    new_params[3] *= inv_scale_y  # cy
                    camera.params = new_params
                if verbose:
                    print(f"  警告: 相机模型 {camera.model.name} 不常见")
            
            # 更新相机尺寸
            camera.width = orig_w
            camera.height = orig_h
            
            # 标记该相机已处理
            processed_cameras.add(camera_id)
            
            if verbose:
                print(f"  ✓ 相机内参已恢复: {output_w}x{output_h} -> {orig_w}x{orig_h}")
        else:
            if verbose:
                print(f"  跳过相机 {camera_id} (已处理)")
        
        # ========== 1.2 恢复2D点坐标（每张图片都要处理）==========
        # 如果需要重新采样颜色，加载原始图像
        original_image_np = None
        if resample_colors_from_original:
            original_image_path = os.path.join(image_path, img_name)
            try:
                original_image = Image.open(original_image_path).convert("RGB")
                original_image_np = np.array(original_image)
                if verbose:
                    print(f"  ✓ 加载原始图像用于颜色重采样")
            except Exception as e:
                if verbose:
                    print(f"  警告: 加载原始图像失败: {e}")
        
        # 遍历所有2D点
        num_points_out_of_bounds = 0
        num_points_resampled = 0
        
        for point2D in img.points2D:
            # 恢复2D点坐标到原始尺寸
            point2D.xy = np.array([
                point2D.xy[0] * inv_scale_x,
                point2D.xy[1] * inv_scale_y
            ])
            
            # 检查是否有关联的3D点
            if point2D.point3D_id != -1 and point2D.point3D_id != 18446744073709551615:
                x, y = int(point2D.xy[0]), int(point2D.xy[1])
                
                # 检查坐标是否在原始图像范围内
                is_in_bounds = (0 <= x < orig_w and 0 <= y < orig_h)
                
                if is_in_bounds:
                    # 在范围内：如果需要，更新颜色
                    if original_image_np is not None:
                        try:
                            color = original_image_np[y, x]
                            point3D = reconstruction.points3D[point2D.point3D_id]
                            point3D.color = color.astype(np.uint8)
                            num_points_resampled += 1
                        except:
                            pass
                else:
                    # 超出范围：标记为需要删除
                    points_to_remove.add(point2D.point3D_id)
                    num_points_out_of_bounds += 1
        
        if verbose:
            print(f"  ✓ 2D点坐标已恢复: {len(img.points2D)} 个点")
            if num_points_out_of_bounds > 0:
                print(f"  ⚠ {num_points_out_of_bounds} 个点超出原始图像范围，将被删除")
            if num_points_resampled > 0:
                print(f"  ✓ {num_points_resampled} 个3D点颜色已重新采样")
    
    # ========== 2. 删除超出范围的3D点 ==========
    if points_to_remove:
        if verbose:
            print(f"\n删除 {len(points_to_remove)} 个超出原始图像范围的3D点...")
        
        for point_id in points_to_remove:
            if point_id in reconstruction.points3D:
                del reconstruction.points3D[point_id]
        
        if verbose:
            print(f"✓ 删除完成，剩余 {len(reconstruction.points3D)} 个3D点")
    
    if verbose:
        print(f"\n✓ 处理了 {len(processed_cameras)} 个唯一相机")
        print("=" * 80)
        print("✓ Reconstruction 已恢复到原始图像尺寸")
        print("=" * 80)
    
    # ========== 3. 保存恢复后的重建结果（新增）==========
    if output_dir is not None:
        output_dir = Path(output_dir)
        restored_output_dir = output_dir / "sparse_mapanything_restored"
        os.makedirs(restored_output_dir, exist_ok=True)
        
        if verbose:
            print(f"\n保存恢复后的重建结果...")
        
        # 保存为文本格式
        reconstruction.write_text(str(restored_output_dir))
        
        # 导出PLY点云
        ply_path = restored_output_dir / "restored_points.ply"
        reconstruction.export_PLY(str(ply_path))
        
        print(f"✓ 恢复到原始尺寸的 Reconstruction 已保存到: {restored_output_dir}")
        if verbose:
            print(f"  - COLMAP 文本文件: {restored_output_dir}")
            print(f"  - 点云文件: {ply_path}")
    
    return reconstruction

def export_dense_point_cloud(
    outputs, 
    output_dir, 
    export_format="both",  # "ply", "glb", or "both"
    verbose=False
):
    """
    Export dense point cloud from MapAnything outputs.
    
    Args:
        outputs: List of prediction dictionaries from model.infer()
        output_dir: Output directory path (Path or str)
        export_format: Export format - "ply", "glb", or "both" (default: "both")
        verbose: Print detailed progress (default: False)
    
    Returns:
        dict: Dictionary with paths to exported files {"ply": path, "glb": path}
    """
    if verbose:
        print("Exporting dense point cloud...")
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取 3D 点信息
    world_points, images, masks = extract_3D_points_from_outputs(outputs, verbose=verbose)
    
    # 创建预测字典
    predictions = {
        "world_points": world_points,
        "images": images,
        "final_masks": masks,
    }
    
    exported_files = {}
    
    # 导出 PLY
    if export_format in ["ply", "both"]:
        scene_3d_ply = predictions_to_ply(predictions, as_mesh=False)
        ply_output_path = output_dir / "point_cloud.ply"
        scene_3d_ply.export(str(ply_output_path), encoding='binary')
        exported_files["ply"] = ply_output_path
        print(f"✓ Saved PLY file: {ply_output_path}")
    
    # 导出 GLB
    if export_format in ["glb", "both"]:
        scene_3d_glb = predictions_to_glb(predictions, as_mesh=False)
        glb_output_path = output_dir / "point_cloud.glb"
        scene_3d_glb.export(str(glb_output_path))
        exported_files["glb"] = glb_output_path
        print(f"✓ Saved GLB file: {glb_output_path}")
    
    return exported_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="SfM Feature Matching + MapAnything Inference Demo"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to folder containing images",
    )
    parser.add_argument(
        "--output_2D_prior",
        type=str,
        required=True,
        help="Output directory for 2D prior (SfM features and tracks)",
    )
    parser.add_argument(
        "--output_3D_prior",
        type=str,
        required=True,
        help="Output directory for 3D prior (MapAnything depth predictions)",
    )
    
    # SfM parameters
    parser.add_argument(
        "--imgsz",
        type=int,
        default=2048,
        help="Maximum image size for SIFT feature extraction",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=8192,
        help="Number of SIFT features to extract per image",
    )
    parser.add_argument(
        "--match_mode",
        type=str,
        default="exhaustive",
        choices=["exhaustive", "spatial"],
        help="Feature matching mode",
    )
    
    # COLMAP loading parameters
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Load every nth image (default: 1)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".txt",
        choices=[".bin", ".txt"],
        help="COLMAP file extension (default: .txt)",
    )
    
    # MapAnything parameters
    parser.add_argument(
        "--apache",
        action="store_true",
        default=False,
        help="Use Apache 2.0 licensed model instead of CC-BY-NC 4.0",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference",
    )
    parser.add_argument(
        "--ignore_calibration_inputs",
        action="store_true",
        default=False,
        help="Ignore COLMAP calibration data",
    )
    parser.add_argument(
        "--ignore_pose_inputs",
        action="store_true",
        default=False,
        help="Ignore COLMAP pose data",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose printouts",
    )
    
    return parser.parse_args()


def main(args):
    """Main function for SfM feature matching and MapAnything inference."""
    print("=" * 80)
    print("SfM Feature Matching + MapAnything Pipeline")
    print("=" * 80)
    print("Arguments:", vars(args))
    print()
    
    # Setup output directories
    output_2D_prior_dir = Path(args.output_2D_prior)
    output_3D_prior_dir = Path(args.output_3D_prior)
    
    print(f"Image path: {args.image_path}")
    print(f"2D prior output: {output_2D_prior_dir}")
    print(f"3D prior output: {output_3D_prior_dir}\n")
    
    # ========== Step 1: SfM Feature Extraction ==========
    print("=" * 80)
    print("STEP 1: 2D Prior - SfM Feature Extraction and Matching")
    print("=" * 80)
    
    prior_reconstruction = build_2D_prior(
        image_path=args.image_path,
        output_dir=output_2D_prior_dir,
        imgsz=args.imgsz,
        num_features=args.num_features,
        match_mode=args.match_mode,
        verbose=args.verbose
    )
    


    # ========== 统计匹配点信息 ==========
    if prior_reconstruction is not None:
        print("\n" + "=" * 80)
        print("2D Prior Reconstruction 匹配点统计")
        print("=" * 80)
        
        num_images = len(prior_reconstruction.images)
        num_points3D = len(prior_reconstruction.points3D)
        num_registered = prior_reconstruction.num_reg_images()
        
        print(f"图片总数: {num_images}")
        print(f"注册图片数: {num_registered}")
        print(f"3D 点总数: {num_points3D}")
        
        if num_images > 0:
            # 统计每张图片的 2D 点数
            points_per_image = []
            matched_points_per_image = []
            
            for img_id, img in prior_reconstruction.images.items():
                num_points2D = len(img.points2D)
                num_matched = sum(1 for p in img.points2D if p.point3D_id != -1)
                points_per_image.append(num_points2D)
                matched_points_per_image.append(num_matched)
            
            print(f"\n每张图片的特征点数:")
            print(f"  总 2D 点数 - 平均: {np.mean(points_per_image):.1f}, 范围: [{min(points_per_image)}, {max(points_per_image)}]")
            print(f"  已匹配点数 - 平均: {np.mean(matched_points_per_image):.1f}, 范围: [{min(matched_points_per_image)}, {max(matched_points_per_image)}]")
            print(f"  匹配率: {np.mean(matched_points_per_image) / np.mean(points_per_image) * 100:.2f}%")
        
        if num_points3D > 0:
            # 统计 track 长度
            track_lengths = [len(p.track.elements) for p in prior_reconstruction.points3D.values()]
            
            print(f"\n3D 点的观测统计:")
            print(f"  平均观测次数: {np.mean(track_lengths):.2f}")
            print(f"  观测次数范围: [{min(track_lengths)}, {max(track_lengths)}]")
            print(f"  观测次数 ≥ 3 的点: {sum(1 for t in track_lengths if t >= 3)} ({sum(1 for t in track_lengths if t >= 3) / len(track_lengths) * 100:.1f}%)")
            print(f"  观测次数 ≥ 5 的点: {sum(1 for t in track_lengths if t >= 5)} ({sum(1 for t in track_lengths if t >= 5) / len(track_lengths) * 100:.1f}%)")
            print(f"  观测次数 ≥ 10 的点: {sum(1 for t in track_lengths if t >= 10)} ({sum(1 for t in track_lengths if t >= 10) / len(track_lengths) * 100:.1f}%)")
        
        print("=" * 80)
    else:
        print("\n⚠ 警告: prior_reconstruction 为 None，无法统计匹配点信息")
    
    print()
    
    # ========== Step 2: MapAnything Inference ==========
    print("=" * 80)
    print("STEP 2: 3D Prior - MapAnything Depth Prediction")
    print("=" * 80)
    
    os.makedirs(output_3D_prior_dir, exist_ok=True)
    
    # Load and preprocess COLMAP data
    print("Loading and preprocessing data...")
    processed_views, scale_info_list = load_and_preprocess_colmap_views(
        image_path=args.image_path,
        colmap_sparse_dir=output_2D_prior_dir / "sparse",
        stride=args.stride,
        ext=args.ext,
        verbose=args.verbose
    )
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    print(f"Loading model: {model_name}...")
    model = MapAnything.from_pretrained(model_name).to(device)
    print("✓ Model loaded")

    # Run inference
    print("Running MapAnything inference...")
    outputs = model.infer(
        processed_views,
        memory_efficient_inference=args.memory_efficient_inference,
        ignore_calibration_inputs=args.ignore_calibration_inputs,
        ignore_depth_inputs=True,
        ignore_pose_inputs=args.ignore_pose_inputs,
        ignore_depth_scale_inputs=True,
        ignore_pose_scale_inputs=True,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    print("✓ Inference completed!")


    # ========== 建立 2D-3D 对应关系 ==========
    all_correspondences = build_2D_3D_correspondences(
        prior_reconstruction=prior_reconstruction,
        scale_info_list=scale_info_list,
        outputs=outputs,
        image_path=args.image_path,
        output_dir=output_3D_prior_dir,
        verbose=args.verbose
    )

    # ========== 基于 2D-3D 对应关系构建新的Colmap的Reconstration类 ==========
    # new_reconstruction = build_reconstruction_from_correspondences(
    #     prior_reconstruction=prior_reconstruction,
    #     all_correspondences=all_correspondences,
    #     output_dir=output_3D_prior_dir,
    #     verbose=args.verbose
    # )

    # new_reconstruction = build_reconstruction_from_correspondences(
    #     prior_reconstruction=prior_reconstruction,
    #     all_correspondences=all_correspondences,
    #     outputs=outputs,
    #     scale_info_list=scale_info_list,
    #     output_dir=output_3D_prior_dir,
    #     verbose=args.verbose
    # )

    new_reconstruction = build_reconstruction_from_correspondences_simple(
        prior_reconstruction=prior_reconstruction,
        all_correspondences=all_correspondences,
        output_dir=output_3D_prior_dir,
        verbose=args.verbose
    )

    # ========== 使用 pycolmap 对齐重建 ==========
    new_reconstruction, sim3d = align_and_restore_reconstruction_scale(
        new_reconstruction=new_reconstruction,
        prior_reconstruction=prior_reconstruction,
        output_dir=output_3D_prior_dir,
        method='proj_centers',  # 推荐使用 'points'，也可以尝试 'reprojections'
        verbose=args.verbose
    )
    if sim3d is None:
        print("警告: 重建对齐失败!")
    else:
        print(f"对齐成功，尺度因子: {sim3d.scale:.6f}")

    # ========== 恢复到原始图像尺寸 ==========
    new_reconstruction = restore_reconstruction_to_original_scale(
        reconstruction=new_reconstruction,
        all_correspondences=all_correspondences,
        image_path=args.image_path,
        output_dir = output_3D_prior_dir,
        resample_colors_from_original=True,  # 可选：重新采样颜色
        verbose=args.verbose
    )

    # ========== Export 3D Point Cloud ==========
    print("\nExporting 3D point cloud...")
    exported_files = export_dense_point_cloud(
        outputs=outputs,
        output_dir=output_3D_prior_dir,
        export_format="ply",  # 可以改为 "ply" 或 "glb"
        verbose=args.verbose
    )
    
    print()
    print("=" * 80)
    print("Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"2D Prior (SfM): {output_2D_prior_dir}")
    print(f"3D Prior (Depth): {output_3D_prior_dir}")
    print(f"  - Point cloud: {exported_files}")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    args = parse_args()
    main(args)



    