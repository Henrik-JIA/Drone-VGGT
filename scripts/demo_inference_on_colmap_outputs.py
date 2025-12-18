# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything demo using COLMAP reconstructions as input

This script demonstrates MapAnything inference on COLMAP format data.
By default MapAnything uses the calibration and poses from COLMAP as input.

The data is expected to be organized in a folder with subfolders:
- images/: containing image files (.jpg, .jpeg, .png)
- sparse/: containing COLMAP reconstruction files (.bin or .txt format)
  - cameras.bin/txt
  - images.bin/txt
  - points3D.bin/txt

Usage:
    python demo_inference_on_colmap_outputs.py --help
"""

import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import rerun as rr
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不显示窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mapanything.models import MapAnything
from mapanything.utils.colmap import get_camera_matrix, qvec2rotmat, read_model
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import predictions_to_glb, predictions_to_ply, script_add_rerun_args


def load_colmap_data(colmap_path, stride=1, verbose=False, ext=".bin"):
    """
    Load COLMAP format data for MapAnything inference.

    Expected folder structure:
    colmap_path/
      images/
        img1.jpg
        img2.jpg
        ...
      sparse/
        cameras.bin/txt
        images.bin/txt
        points3D.bin/txt

    Args:
        colmap_path (str): Path to the main folder containing images/ and sparse/ subfolders
        stride (int): Load every nth image (default: 50)
        verbose (bool): Print progress messages
        ext (str): COLMAP file extension (".bin" or ".txt")

    Returns:
        list: List of view dictionaries for MapAnything inference
    """
    # Define paths
    images_folder = os.path.join(colmap_path, "images")
    sparse_folder = os.path.join(colmap_path, "sparse")

    # Check that required folders exist
    if not os.path.exists(images_folder):
        raise ValueError(f"Required folder 'images' not found at: {images_folder}")
    if not os.path.exists(sparse_folder):
        raise ValueError(f"Required folder 'sparse' not found at: {sparse_folder}")

    if verbose:
        print(f"Loading COLMAP data from: {colmap_path}")
        print(f"Images folder: {images_folder}")
        print(f"Sparse folder: {sparse_folder}")
        print(f"Using COLMAP file extension: {ext}")

    # Read COLMAP model
    try:
        cameras, images_colmap, points3D = read_model(sparse_folder, ext=ext)
    except Exception as e:
        raise ValueError(f"Failed to read COLMAP model from {sparse_folder}: {e}")

    if verbose:
        print(
            f"Loaded COLMAP model with {len(cameras)} cameras, {len(images_colmap)} images, {len(points3D)} 3D points"
        )

    # Get list of available image files
    available_images = set()
    for f in os.listdir(images_folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            available_images.add(f)

    if not available_images:
        raise ValueError(f"No image files found in {images_folder}")

    views_example = []
    processed_count = 0

    # Get a list of all colmap image names
    colmap_image_names = set(img_info.name for img_info in images_colmap.values())
    # Find the unposed images (in images/ but not in colmap)
    unposed_images = available_images - colmap_image_names

    if verbose:
        print(f"Found {len(unposed_images)} images without COLMAP poses")

    # Process images in COLMAP order
    for img_id, img_info in images_colmap.items():
        # Apply stride
        if processed_count % stride != 0:
            processed_count += 1
            continue

        img_name = img_info.name

        # Check if image file exists
        image_path = os.path.join(images_folder, img_name)
        if not os.path.exists(image_path):
            if verbose:
                print(f"Warning: Image file not found for {img_name}, skipping")
            processed_count += 1
            continue

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image).astype(np.uint8)  # (H, W, 3) - [0, 255]

            # Get camera info
            cam_info = cameras[img_info.camera_id]
            cam_params = cam_info.params

            # Get intrinsic matrix
            K, _ = get_camera_matrix(
                camera_params=cam_params, camera_model=cam_info.model
            )

            # Get pose (COLMAP provides world2cam, we need cam2world)
            # COLMAP: world2cam rotation and translation
            C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec

            # Create 4x4 world2cam pose matrix
            world2cam_matrix = np.eye(4)
            world2cam_matrix[:3, :3] = C_R_G
            world2cam_matrix[:3, 3] = C_t_G

            # Convert to cam2world using closed form pose inverse
            pose_matrix = closed_form_pose_inverse(world2cam_matrix[None, :, :])[0]

            # Convert to tensors
            image_tensor = torch.from_numpy(image_array)  # (H, W, 3)
            intrinsics_tensor = torch.from_numpy(K.astype(np.float32))  # (3, 3)
            pose_tensor = torch.from_numpy(pose_matrix.astype(np.float32))  # (4, 4)

            # Create view dictionary for MapAnything inference
            view = {
                "img": image_tensor,  # (H, W, 3) - [0, 255]
                "intrinsics": intrinsics_tensor,  # (3, 3)
                "camera_poses": pose_tensor,  # (4, 4) in OpenCV cam2world convention
                "is_metric_scale": torch.tensor([False]),  # COLMAP data is non-metric
            }

            views_example.append(view)
            processed_count += 1

            if verbose:
                print(
                    f"Loaded view {len(views_example) - 1}: {img_name} (shape: {image_array.shape})"
                )
                print(f"  - Camera ID: {img_info.camera_id}")
                print(f"  - Camera Model: {cam_info.model}")
                print(f"  - Image ID: {img_id}")

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load data for {img_name}: {e}")
            processed_count += 1
            continue

    # process unposed images (without COLMAP poses)
    for img_name in unposed_images:
        # Apply stride
        if processed_count % stride != 0:
            processed_count += 1
            continue

        image_path = os.path.join(images_folder, img_name)

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image).astype(np.uint8)  # (H, W, 3) - [0, 255]

            # Convert to tensor
            image_tensor = torch.from_numpy(image_array)  # (H, W, 3)

            view = {
                "img": image_tensor,  # (H, W, 3) - [0, 255]
                # No intrinsics or pose available
            }

            views_example.append(view)
            processed_count += 1

            if verbose:
                print(
                    f"Loaded unposed view {len(views_example) - 1}: {img_name} (shape: {image_array.shape})"
                )

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load data for {img_name}: {e}")
            processed_count += 1
            continue

    if not views_example:
        raise ValueError("No valid images found")

    if verbose:
        print(f"Successfully loaded {len(views_example)} views with stride={stride}")

    return views_example


def log_data_to_rerun(
    image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None
):
    """Log visualization data to Rerun"""
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.1,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )

    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]

    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
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

def recover_colmap_scale(outputs, original_poses, ignore_pose_inputs=False, metric_scale_value=None):
    """
    Recover COLMAP scale by comparing input and output camera poses.
    
    Args:
        outputs: List of model output dictionaries
        original_poses: List of original COLMAP pose tensors (4x4 matrices)
        ignore_pose_inputs: Whether pose inputs were ignored during inference
        metric_scale_value: Optional float, if provided, use this instead of computing scale_ratio
        
    Returns:
        outputs: Modified outputs with corrected scale
        scale_ratio: The used scale ratio (or None if not computed)
    """
    # 如果提供了 metric_scale_value，直接使用它
    if metric_scale_value is not None:
        scale_ratio = metric_scale_value
        print(f"Using provided metric_scale_value: {scale_ratio:.6f}")
        print(f"Applying scale correction to outputs...")
    elif len(original_poses) == 0 or ignore_pose_inputs:
        print("Skipping scale recovery (no input poses or pose inputs ignored)")
        return outputs, None
    else:
        # Compute scale factor from pose comparison
        # Compare camera positions from input vs output
        device = outputs[0]["camera_poses"].device
        input_positions = torch.stack([p[:3, 3].to(device) for p in original_poses])  # (N, 3)
        output_positions = torch.stack([outputs[i]["camera_poses"][0, :3, 3] for i in range(len(outputs))])  # (N, 3)
        
        # Compute pairwise distances
        # 计算所有两两之间的距离
        # 计算COLMAP中每两个相机之间的距离（真实距离）
        # 计算MapAnything中每两个相机之间的距离（模型距离）
        input_dists = torch.cdist(input_positions, input_positions)  # (N, N)
        output_dists = torch.cdist(output_positions, output_positions)  # (N, N)
        
        # Get valid pairs (non-zero distances)
        valid_mask = input_dists > 1e-6
        
        if valid_mask.sum() == 0:
            print("Warning: Could not compute scale ratio (insufficient camera movement)")
            return outputs, None
        
        # Compute scale ratio
        # 计算真实距离 ÷ 模型距离，得到缩放比例
        # 使用中位数（median）而不是平均值，因为更稳定，不容易被异常值影响
        scale_ratio = (input_dists[valid_mask] / (output_dists[valid_mask] + 1e-8)).median()
        
        print(f"Detected scale ratio COLMAP/MapAnything: {scale_ratio.item():.6f}")
        print(f"Applying scale correction to outputs...")
    
    # Apply scale correction to all outputs
    # 深度图乘以缩放比例、相机位置也乘以缩放比例、3D点的坐标也要缩放
    for view_idx, pred in enumerate(outputs):
        # Clone tensors to allow modification (inference tensors are read-only)
        # Scale depth
        pred["depth_z"] = pred["depth_z"].clone() * scale_ratio
        if "depth_along_ray" in pred:
            pred["depth_along_ray"] = pred["depth_along_ray"].clone() * scale_ratio
        
        # Scale camera positions
        camera_poses_scaled = pred["camera_poses"].clone()
        camera_poses_scaled[0, :3, 3] = camera_poses_scaled[0, :3, 3] * scale_ratio
        pred["camera_poses"] = camera_poses_scaled
        
        if "cam_trans" in pred:
            pred["cam_trans"] = pred["cam_trans"].clone() * scale_ratio
        
        # Scale 3D points in camera frame
        if "pts3d_cam" in pred:
            pred["pts3d_cam"] = pred["pts3d_cam"].clone() * scale_ratio

        # Scale 3D points in world frame
        if "pts3d" in pred:
            pred["pts3d"] = pred["pts3d"].clone() * scale_ratio

    return outputs, scale_ratio

def align_poses_procrustes(source_poses, target_poses, scale_fixed=False):
    """
    使用Umeyama算法对齐两组相机poses
    
    这个函数计算一个相似变换（旋转R、平移t、缩放s），使得：
    target ≈ s * R @ source + t
    
    Args:
        source_poses: List of source pose tensors (预测的poses, 4x4 cam2world)
        target_poses: List of target pose tensors (真实的COLMAP poses, 4x4 cam2world)
        scale_fixed: If True, only compute rotation and translation (no scaling)
        
    Returns:
        aligned_poses: List of aligned pose tensors
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        s: Scale factor (scalar)
    """
    if len(source_poses) != len(target_poses):
        raise ValueError(f"Number of poses must match: {len(source_poses)} vs {len(target_poses)}")
    
    if len(source_poses) < 3:
        print("Warning: Need at least 3 poses for reliable alignment")
    
    # 提取相机位置（translation部分）
    source_positions = []
    target_positions = []
    
    for src_pose, tgt_pose in zip(source_poses, target_poses):
        if isinstance(src_pose, torch.Tensor):
            src_pos = src_pose[:3, 3].cpu().numpy()
        else:
            src_pos = src_pose[:3, 3]
        
        if isinstance(tgt_pose, torch.Tensor):
            tgt_pos = tgt_pose[:3, 3].cpu().numpy()
        else:
            tgt_pos = tgt_pose[:3, 3]
        
        source_positions.append(src_pos)
        target_positions.append(tgt_pos)
    
    source_positions = np.array(source_positions)  # (N, 3)
    target_positions = np.array(target_positions)  # (N, 3)
    
    # 使用Umeyama算法计算相似变换
    # 注意：umeyama_alignment计算的是 dst = s * R @ src + t
    # 所以我们传入 src=source_positions, dst=target_positions
    s, R, t = umeyama_alignment(source_positions, target_positions, with_scale=not scale_fixed)
    
    print("=" * 80)
    print("Umeyama Pose Alignment Results")
    print("=" * 80)
    print(f"Number of poses aligned: {len(source_poses)}")
    print(f"Scale factor (s): {s:.6f}")
    print(f"Rotation matrix (R):")
    print(R)
    print(f"Translation vector (t): {t}")
    
    # 计算对齐误差
    source_aligned_pos = (s * (R @ source_positions.T)).T + t  # (N, 3)
    alignment_errors = np.linalg.norm(source_aligned_pos - target_positions, axis=1)
    print(f"\nAlignment Error Statistics:")
    print(f"  - Mean Error:   {np.mean(alignment_errors):.6f}")
    print(f"  - Median Error: {np.median(alignment_errors):.6f}")
    print(f"  - Max Error:    {np.max(alignment_errors):.6f}")
    print(f"  - Min Error:    {np.min(alignment_errors):.6f}")
    print("=" * 80)
    
    # 应用变换到所有poses
    aligned_poses = []
    for src_pose in source_poses:
        if isinstance(src_pose, torch.Tensor):
            pose_np = src_pose.cpu().numpy()
            is_tensor = True
            device = src_pose.device
        else:
            pose_np = src_pose.copy()
            is_tensor = False
        
        # 创建新的pose矩阵
        aligned_pose = np.eye(4)
        
        # 变换位置: p_new = s * R @ p_old + t
        old_position = pose_np[:3, 3]
        new_position = s * (R @ old_position) + t
        aligned_pose[:3, 3] = new_position
        
        # 变换旋转: R_new = R @ R_old
        old_rotation = pose_np[:3, :3]
        new_rotation = R @ old_rotation
        aligned_pose[:3, :3] = new_rotation
        
        # 转换回原始格式
        if is_tensor:
            aligned_pose = torch.from_numpy(aligned_pose.astype(np.float32)).to(device)
        
        aligned_poses.append(aligned_pose)
    
    return aligned_poses, R, t, s


def align_camera_poses_to_colmap(outputs, original_poses):
    """
    将MapAnything预测的camera poses对齐到COLMAP的坐标系
    
    Args:
        outputs: List of model output dictionaries
        original_poses: List of original COLMAP pose tensors
        
    Returns:
        aligned_poses: List of aligned pose tensors (4x4)
        alignment_info: Dictionary containing R, t, s
    """
    if len(outputs) == 0 or len(original_poses) == 0:
        print("Warning: Cannot align - missing pose data")
        return None, None
    
    # 提取预测的poses
    predicted_poses = []
    for pred in outputs:
        predicted_poses.append(pred["camera_poses"][0])
    
    # 执行Umeyama对齐
    aligned_poses, R, t, s = align_poses_procrustes(
        source_poses=predicted_poses,
        target_poses=original_poses,
        scale_fixed=False  # 允许缩放
    )
    
    alignment_info = {
        "rotation": R,
        "translation": t,
        "scale": s
    }
    
    print("\nCamera pose alignment complete!")
    print(f"  - Aligned {len(aligned_poses)} camera poses")
    print(f"  - Scale: {s:.6f}, Translation: {t}")
    
    return aligned_poses, alignment_info

def plot_camera_trajectories(original_poses, predicted_poses, scale_ratio=None, output_path=None):
    """
    Plot and compare camera trajectories between COLMAP (ground truth) and MapAnything predictions.
    
    Args:
        original_poses: List of original COLMAP pose tensors (4x4 matrices, cam2world)
        predicted_poses: List of predicted pose tensors from outputs (4x4 matrices, cam2world)
        scale_ratio: Optional scale ratio applied to predictions
        output_path: Optional path to save the plot image
    """
    if len(original_poses) == 0 or len(predicted_poses) == 0:
        print("Warning: Cannot plot trajectories - missing pose data")
        return
    
    # Extract camera positions (translation vectors)
    original_positions = []
    for pose in original_poses:
        if isinstance(pose, torch.Tensor):
            pos = pose[:3, 3].cpu().numpy()
        else:
            pos = pose[:3, 3]
        original_positions.append(pos)
    original_positions = np.array(original_positions)  # (N, 3)
    
    predicted_positions = []
    for pose in predicted_poses:
        if isinstance(pose, torch.Tensor):
            pos = pose[:3, 3].cpu().numpy()
        else:
            pos = pose[:3, 3]
        predicted_positions.append(pos)
    predicted_positions = np.array(predicted_positions)  # (N, 3)
    
    # Calculate statistics
    position_errors = np.linalg.norm(original_positions - predicted_positions, axis=1)
    mean_error = np.mean(position_errors)
    max_error = np.max(position_errors)
    median_error = np.median(position_errors)
    
    print("=" * 80)
    print("Camera Trajectory Comparison Statistics")
    print("=" * 80)
    print(f"Number of cameras: {len(original_poses)}")
    if scale_ratio is not None:
        print(f"Applied scale ratio: {scale_ratio:.6f}")
    print(f"Position Error Statistics:")
    print(f"  - Mean Error:   {mean_error:.4f}")
    print(f"  - Median Error: {median_error:.4f}")
    print(f"  - Max Error:    {max_error:.4f}")
    print("=" * 80)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(original_positions[:, 0], original_positions[:, 1], original_positions[:, 2], 
             'b-o', label='COLMAP (Ground Truth)', linewidth=2, markersize=6)
    ax1.plot(predicted_positions[:, 0], predicted_positions[:, 1], predicted_positions[:, 2], 
             'r--s', label='MapAnything (Predicted)', linewidth=2, markersize=4, alpha=0.7)
    
    # Mark start and end points
    ax1.scatter(original_positions[0, 0], original_positions[0, 1], original_positions[0, 2], 
                c='green', s=200, marker='*', label='Start', edgecolors='black', linewidths=2)
    ax1.scatter(original_positions[-1, 0], original_positions[-1, 1], original_positions[-1, 2], 
                c='orange', s=200, marker='X', label='End', edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Camera Trajectories Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Top view (X-Y plane)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(original_positions[:, 0], original_positions[:, 1], 'b-o', 
             label='COLMAP', linewidth=2, markersize=6)
    ax2.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'r--s', 
             label='MapAnything', linewidth=2, markersize=4, alpha=0.7)
    ax2.scatter(original_positions[0, 0], original_positions[0, 1], 
                c='green', s=200, marker='*', label='Start', edgecolors='black', linewidths=2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (X-Y Plane)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Side view (X-Z plane)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(original_positions[:, 0], original_positions[:, 2], 'b-o', 
             label='COLMAP', linewidth=2, markersize=6)
    ax3.plot(predicted_positions[:, 0], predicted_positions[:, 2], 'r--s', 
             label='MapAnything', linewidth=2, markersize=4, alpha=0.7)
    ax3.scatter(original_positions[0, 0], original_positions[0, 2], 
                c='green', s=200, marker='*', label='Start', edgecolors='black', linewidths=2)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Side View (X-Z Plane)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Front view (Y-Z plane)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(original_positions[:, 1], original_positions[:, 2], 'b-o', 
             label='COLMAP', linewidth=2, markersize=6)
    ax4.plot(predicted_positions[:, 1], predicted_positions[:, 2], 'r--s', 
             label='MapAnything', linewidth=2, markersize=4, alpha=0.7)
    ax4.scatter(original_positions[0, 1], original_positions[0, 2], 
                c='green', s=200, marker='*', label='Start', edgecolors='black', linewidths=2)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('Front View (Y-Z Plane)')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')
    
    # Position error over camera index
    ax5 = fig.add_subplot(2, 3, 5)
    camera_indices = np.arange(len(position_errors))
    ax5.plot(camera_indices, position_errors, 'g-o', linewidth=2, markersize=6)
    ax5.axhline(y=mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.4f}')
    ax5.axhline(y=median_error, color='b', linestyle='--', label=f'Median: {median_error:.4f}')
    ax5.set_xlabel('Camera Index')
    ax5.set_ylabel('Position Error')
    ax5.set_title('Position Error per Camera')
    ax5.legend()
    ax5.grid(True)
    
    # Error histogram
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(position_errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax6.axvline(x=mean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
    ax6.axvline(x=median_error, color='b', linestyle='--', linewidth=2, label=f'Median: {median_error:.4f}')
    ax6.set_xlabel('Position Error')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Position Error Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Camera Trajectory Comparison\n'
                 f'Scale Ratio: {scale_ratio:.6f} | Mean Error: {mean_error:.4f} | Max Error: {max_error:.4f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot (no display)
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:  # 如果有目录路径（不是当前目录）
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to: {output_path}")
    else:
        # 如果没有指定路径，生成默认文件名
        default_path = "trajectory_comparison.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to: {default_path}")

    plt.close()

def get_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="MapAnything demo using COLMAP reconstructions as input"
    )
    parser.add_argument(
        "--colmap_path",
        type=str,
        required=True,
        help="Path to folder containing images/ and sparse/ subfolders",
    )
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
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable visualization with Rerun",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose printouts for loading",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save reconstruction as GLB file",
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        default=False,
        help="Save reconstruction as PLY file",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="colmap_mapanything_output",
        help="Output directory for GLB file and input images",
    )
    parser.add_argument(
        "--save_input_images",
        action="store_true",
        default=False,
        help="Save input images alongside GLB output (requires --save_glb)",
    )
    parser.add_argument(
        "--ignore_calibration_inputs",
        action="store_true",
        default=False,
        help="Ignore COLMAP calibration inputs (use only images and poses)",
    )
    parser.add_argument(
        "--ignore_pose_inputs",
        action="store_true",
        default=False,
        help="Ignore COLMAP pose inputs (use only images and calibration)",
    )
    parser.add_argument(
        "--scale_mode",
        type=str,
        default="metric",
        choices=["metric", "compute", "none"],
        help=(
            "Scale recovery mode: "
            "'metric' - use metric_scaling_factor from model output (default), "
            "'compute' - compute scale_ratio from camera poses, "
            "'none' - no scaling (scale=1.0)"
        ),
    )
    parser.add_argument(
        "--plot_trajectories",
        action="store_true",
        default=False,
        help="Plot and compare camera trajectories between COLMAP and MapAnything",
    )
    parser.add_argument(
        "--trajectory_plot_path",
        type=str,
        default=None,
        help="Path to save trajectory comparison plot (if not specified, will save to output_directory)",
    )

    return parser


def main():
    # Parser for arguments and Rerun
    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model from HuggingFace
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    model = MapAnything.from_pretrained(model_name).to(device)

    # Load COLMAP data
    print(f"Loading COLMAP data from: {args.colmap_path}")
    views_example = load_colmap_data(
        args.colmap_path,
        stride=args.stride,
        verbose=args.verbose,
        ext=args.ext,
    )
    print(f"Loaded {len(views_example)} views")

    # Preprocess inputs to the expected format
    print("Preprocessing COLMAP inputs...")
    processed_views = preprocess_inputs(views_example, verbose=False)

    # Run model inference
    print("Running MapAnything inference on COLMAP data...")
    outputs = model.infer(
        processed_views,
        memory_efficient_inference=args.memory_efficient_inference,
        # Control which COLMAP inputs to use/ignore
        ignore_calibration_inputs=args.ignore_calibration_inputs,  # Whether to use COLMAP calibration or not
        ignore_depth_inputs=True,  # COLMAP doesn't provide depth (can recover from sparse points but convoluted)
        ignore_pose_inputs=args.ignore_pose_inputs,  # Whether to use COLMAP poses or not
        ignore_depth_scale_inputs=True,  # No depth data
        ignore_pose_scale_inputs=True,  # COLMAP poses are non-metric
        # Use amp for better performance
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    print("COLMAP inference complete!")

    # Prepare lists for GLB export if needed
    world_points_list = []
    images_list = []
    masks_list = []

    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        viz_string = "MapAnything_COLMAP_Inference_Visualization"
        rr.script_setup(args, viz_string)
        rr.set_time("stable_time", sequence=0)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

    # Store original COLMAP poses for scale recovery
    original_poses = []
    for view in views_example:
        if "camera_poses" in view:
            original_poses.append(view["camera_poses"].clone())
    
    # Recover COLMAP scale, scale_ratio has used in this function process
    metric_scale_value = None
    if args.scale_mode == "metric":
        # 模式1: 使用 metric_scaling_factor（如果存在）
        if len(outputs) > 0 and "metric_scaling_factor" in outputs[0]:
            metric_scale_tensor = outputs[0]["metric_scaling_factor"]
            if metric_scale_tensor is not None:
                # 从 tensor([[7.8367]], device='cuda:0') 提取出 7.8367
                metric_scale_value = metric_scale_tensor.item()
                print(f"[Scale Mode: metric] Extracted metric_scaling_factor: {metric_scale_value}")
            else:
                print("[Scale Mode: metric] metric_scaling_factor is None, will compute scale_ratio instead")
        else:
            print("[Scale Mode: metric] No metric_scaling_factor found in outputs, will compute scale_ratio instead")
    elif args.scale_mode == "compute":
        # 模式2: 强制计算 scale_ratio（不使用 metric_scaling_factor）
        print("[Scale Mode: compute] Will compute scale_ratio from camera poses")
        metric_scale_value = None
    elif args.scale_mode == "none":
        # 模式3: 不做任何缩放
        print("[Scale Mode: none] No scaling will be applied (scale=1.0)")
        metric_scale_value = 1.0
        
    outputs, scale_ratio = recover_colmap_scale(
        outputs, 
        original_poses, 
        ignore_pose_inputs=args.ignore_pose_inputs,
        metric_scale_value=metric_scale_value
    )

    if args.plot_trajectories:
        # 使用Umeyama算法进行完整的位姿配准（包括旋转和平移）
        if len(original_poses) > 0 and not args.ignore_pose_inputs:
            print("\n" + "=" * 80)
            print("Aligning predicted poses to COLMAP coordinate system using Umeyama algorithm")
            print("=" * 80)
            aligned_poses, alignment_info = align_camera_poses_to_colmap(outputs, original_poses)
            print()
        else:
            print("\nSkipping pose alignment (no COLMAP poses available)")
            alignment_info = None

        # Plot camera trajectories comparison if requested
        print("\n" + "=" * 80)
        print("Generating Camera Trajectory Comparison Plot")
        print("=" * 80)
        
        # Extract predicted poses from outputs (after scale correction)
        if aligned_poses is not None:
            predicted_poses_to_plot = aligned_poses
            print("Using aligned poses for trajectory comparison")
        else:
            # 提取原始预测poses
            predicted_poses_to_plot = []
            for pred in outputs:
                pred_pose = pred["camera_poses"][0]  # (4, 4)
                predicted_poses_to_plot.append(pred_pose)
            print("Using original predicted poses for trajectory comparison")

        # Determine output path for plot
        if args.trajectory_plot_path:
            plot_output_path = args.trajectory_plot_path
        else:
            os.makedirs(args.output_directory, exist_ok=True)
            plot_output_path = os.path.join(
                args.output_directory, 
                f"{os.path.basename(args.output_directory)}_trajectory_comparison.png"
            )
        
        # Plot trajectories
        plot_camera_trajectories(
            original_poses=original_poses,
            predicted_poses=predicted_poses_to_plot,
            scale_ratio=scale_ratio * alignment_info["scale"],
            output_path=plot_output_path
        )
        print()

    # Loop through the outputs
    for view_idx, pred in enumerate(outputs):
        # 如果没有pts3d，则需要重新计算（通常不会发生）
        depthmap_torch = pred["depth_z"][0].squeeze(-1)
        intrinsics_torch = pred["intrinsics"][0]
        camera_pose_torch = pred["camera_poses"][0]
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Convert to numpy arrays
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()

        # Store data for GLB export if needed
        if args.save_glb or args.save_ply:
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)

        # Log to Rerun if visualization is enabled
        if args.viz:
            log_data_to_rerun(
                image=image_np,
                depthmap=depthmap_torch.cpu().numpy(),
                pose=camera_pose_torch.cpu().numpy(),
                intrinsics=intrinsics_torch.cpu().numpy(),
                pts3d=pts3d_np,
                mask=mask,
                base_name=f"mapanything/view_{view_idx}",
                pts_name=f"mapanything/pointcloud_view_{view_idx}",
                viz_mask=mask,
            )

    # Convey that the visualization is complete
    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")

    # Prepare data for export (if GLB or PLY export is requested)
    if args.save_glb or args.save_ply:
        # Stack all views
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)

        # Create predictions dict for export
        predictions = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }

        # Convert to 3D scene
        if args.save_glb:
            scene_3d_glb = predictions_to_glb(predictions, as_mesh=False)
        if args.save_ply:
            scene_3d_ply = predictions_to_ply(predictions, as_mesh=False)

    # Export GLB if requested
    if args.save_glb:
        # Create output directory structure
        scene_output_dir = args.output_directory
        os.makedirs(scene_output_dir, exist_ok=True)
        scene_prefix = os.path.basename(scene_output_dir)

        glb_output_path = os.path.join(
            scene_output_dir, f"{scene_prefix}_mapanything_colmap_output.glb"
        )
        print(f"Saving GLB file to: {glb_output_path}")

        # Save processed input images if requested
        if args.save_input_images:
            # Create processed images directory
            processed_images_dir = os.path.join(
                scene_output_dir, f"{scene_prefix}_input_images"
            )
            os.makedirs(processed_images_dir, exist_ok=True)
            print(f"Saving processed input images to: {processed_images_dir}")

            # Save each processed input image from outputs
            for view_idx, pred in enumerate(outputs):
                # Get processed image (RGB, 0-255)
                processed_image = (
                    pred["img_no_norm"][0].cpu().numpy() * 255
                )  # (H, W, 3)

                # Convert to PIL Image and save as PNG
                img_pil = Image.fromarray(processed_image.astype(np.uint8))
                img_path = os.path.join(processed_images_dir, f"view_{view_idx}.png")
                img_pil.save(img_path)

            print(
                f"Saved {len(outputs)} processed input images to: {processed_images_dir}"
            )

        # Save GLB file
        scene_3d_glb.export(glb_output_path)
        print(f"Successfully saved GLB file: {glb_output_path}")
    else:
        if args.save_input_images:
            print("Warning: --save_input_images has no effect without --save_glb")

    # Export PLY if requested
    if args.save_ply:
        # Create output directory structure
        scene_output_dir = args.output_directory
        os.makedirs(scene_output_dir, exist_ok=True)
        scene_prefix = os.path.basename(scene_output_dir)

        ply_output_path = os.path.join(
            scene_output_dir, f"{scene_prefix}_mapanything_colmap_output.ply"
        )
        print(f"Saving PLY file to: {ply_output_path}")

        # Save PLY file
        # scene_3d_ply.export(ply_output_path, encoding='ascii')
        scene_3d_ply.export(ply_output_path, encoding='binary')
        print(f"Successfully saved PLY file: {ply_output_path}")

    if not args.save_glb and not args.save_ply:
        print("Skipping export (--save_glb or --save_ply not specified)")
    else:
        # Show output directory when any export is performed
        print(f"All outputs saved to: {args.output_directory}")

if __name__ == "__main__":
    main()
