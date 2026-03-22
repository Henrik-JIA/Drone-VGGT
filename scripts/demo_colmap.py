# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Demo script to get MapAnything outputs in COLMAP format. Optionally can also run BA on outputs.

Reference: VGGT (https://github.com/facebookresearch/vggt/blob/main/demo_colmap.py)
"""

import argparse
import copy
import glob
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torchvision import transforms as tvf

from mapanything.models import MapAnything
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_model
from mapanything.third_party.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from mapanything.third_party.track_predict import predict_tracks
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from mapanything.utils.image import rgb
from mapanything.utils.misc import seed_everything
from mapanything.utils.viz import predictions_to_glb
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_args():
    parser = argparse.ArgumentParser(description="MapAnything COLMAP Demo")
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Directory containing the scene images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (GLB file and sparse folder). Defaults to scene_dir if not specified.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--conf_thres_value",
        type=float,
        default=0.0,
        help="Confidence threshold value for depth filtering (used only without BA)",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save dense reconstruction (without BA) as GLB file",
    )
    parser.add_argument(
        "--filter_black_bg",
        action="store_true",
        default=False,
        help="Filter out black background points (RGB sum < 16)",
    )
    parser.add_argument(
        "--filter_white_bg",
        action="store_true",
        default=False,
        help="Filter out white background points (all RGB > 240)",
    )
    parser.add_argument(
        "--use_ba", action="store_true", default=False, help="Use BA for reconstruction"
    )
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=8.0,
        help="Maximum reprojection error for reconstruction",
    )
    parser.add_argument(
        "--shared_camera",
        action="store_true",
        default=False,
        help="Use shared camera for all images",
    )
    parser.add_argument(
        "--camera_type",
        type=str,
        default="SIMPLE_PINHOLE",
        help="Camera type for reconstruction",
    )
    parser.add_argument(
        "--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks"
    )
    parser.add_argument(
        "--query_frame_num", type=int, default=8, help="Number of frames to query"
    )
    parser.add_argument(
        "--max_query_pts", type=int, default=4096, help="Maximum number of query points"
    )
    parser.add_argument(
        "--fine_tracking",
        action="store_true",
        default=True,
        help="Use fine tracking (slower but more accurate)",
    )
    parser.add_argument(
        "--max_points3D_val",
        type=float,
        default=3000.0,
        help="Maximum absolute coordinate value for valid 3D points (checks each coordinate component relative to 0). "
             "Points are kept only if abs(x) < threshold AND abs(y) < threshold AND abs(z) < threshold. "
             "This defines a cubic region centered at origin, NOT a spherical distance filter. "
             "Use larger values (e.g., 10000) for large-scale scenes, smaller values (e.g., 500) for indoor scenes. "
             "Default: 3000.0",
    )
    parser.add_argument(
        "--min_inlier_per_frame",
        type=int,
        default=32,
        help="Minimum number of valid inlier points required per frame for BA. "
            "Frames with fewer inliers will cause BA to skip. "
            "Lower values (e.g., 16) allow more frames but may reduce accuracy. "
            "Higher values (e.g., 64) ensure better quality but may reject more frames. "
            "Default: 32",
    )
    return parser.parse_args()


def load_and_preprocess_images_square(
    image_path_list, target_size=1024, data_norm_type=None
):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 1024.
        data_norm_type (str, optional): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys. Defaults to None (no normalization).

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty or if an invalid data_norm_type is provided
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive

    # Set up normalization based on data_norm_type
    if data_norm_type is None:
        # No normalization, just convert to tensor
        img_transform = tvf.ToTensor()
    elif data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
        # Use the specified normalization
        img_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
        img_transform = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {data_norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize(
            (target_size, target_size), Image.Resampling.BICUBIC
        )

        # Convert to tensor and apply normalization
        img_tensor = img_transform(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords

def load_and_preprocess_images_square_with_mode(
    image_path_list, 
    target_size=1024, 
    data_norm_type=None,
    scale_factor=1.0,
    mode="edge"
):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Uses smart padding to avoid black borders, or stretch mode to avoid padding entirely.
    
    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 1024.
        data_norm_type (str, optional): Image normalization type. Defaults to None (no normalization).
        scale_factor (float, optional): Downsample factor before processing. 1.0 means no downsampling. Defaults to 1.0.
        mode (str, optional): Padding/resize mode. Options:
            - "stretch": Stretch image to square (changes aspect ratio, no padding)
            - "black": Black padding (original behavior)
            - "white": White padding
            - "gray": Gray padding (128, 128, 128)
            - "edge": Use average edge color (smart padding, recommended)
            - "no_padding": Keep original aspect ratio, pad to common size for batching
            Defaults to "edge".
    
    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)
                         For most modes: H=W=target_size
                         For "no_padding": H and W are the max dimensions across all images
            torch.Tensor: Array of shape (N, 6) containing [x1, y1, x2, y2, width, height] for each image
        )
    
    Raises:
        ValueError: If the input list is empty or if an invalid data_norm_type is provided
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []
    processed_pil_images = []  # Store PIL images before converting to tensor (for no_padding mode)

    # Set up normalization based on data_norm_type
    if data_norm_type is None:
        img_transform = tvf.ToTensor()
    elif data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
        img_transform = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {data_norm_type}. "
            f"Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    for image_path in image_path_list:
        # Open image using PIL
        img = Image.open(image_path)
        
        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        
        # Convert to RGB
        img = img.convert("RGB")
        
        # Optional downsampling before padding/stretching
        if scale_factor != 1.0:
            orig_w, orig_h = img.size
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            img = img.resize((new_w, new_h), Image.Resampling.AREA)
        
        # Get original dimensions
        width, height = img.size
        
        # Handle different modes
        if mode == "stretch":
            # ========== STRETCH MODE: 直接拉伸到正方形 ==========
            square_img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
            x1 = 0
            y1 = 0
            x2 = target_size
            y2 = target_size
            original_coords.append(np.array([x1, y1, x2, y2, width, height]))
            # ====================================================
        elif mode == "no_padding":
            # ========== NO PADDING MODE: 保持原始宽高比 ==========
            # Calculate coordinate transformation info (as if it would be padded to square)
            max_dim = max(width, height)
            left = (max_dim - width) // 2
            top = (max_dim - height) // 2
            scale = target_size / max_dim
            
            x1 = left * scale
            y1 = top * scale
            x2 = (left + width) * scale
            y2 = (top + height) * scale
            
            original_coords.append(np.array([x1, y1, x2, y2, width, height]))
            
            # 保持原始图像，不做padding（稍后统一处理）
            square_img = img
            processed_pil_images.append(img)
            # ====================================================
        else:
            # ========== PADDING MODES: 保持宽高比 + 填充 ==========
            max_dim = max(width, height)
            left = (max_dim - width) // 2
            top = (max_dim - height) // 2
            scale = target_size / max_dim
            
            x1 = left * scale
            y1 = top * scale
            x2 = (left + width) * scale
            y2 = (top + height) * scale
            
            original_coords.append(np.array([x1, y1, x2, y2, width, height]))
            
            # Determine padding color based on mode
            if mode == "black":
                pad_color = (0, 0, 0)
            elif mode == "white":
                pad_color = (255, 255, 255)
            elif mode == "gray":
                pad_color = (128, 128, 128)
            elif mode == "edge":
                # Smart padding: use average edge color
                img_array = np.array(img)
                top_edge = img_array[0, :, :]
                bottom_edge = img_array[-1, :, :]
                left_edge = img_array[:, 0, :]
                right_edge = img_array[:, -1, :]
                edge_pixels = np.vstack([top_edge, bottom_edge, left_edge, right_edge])
                pad_color = tuple(edge_pixels.mean(axis=0).astype(int).tolist())
            else:
                raise ValueError(
                    f"Invalid mode: {mode}. "
                    f"Options: 'stretch', 'black', 'white', 'gray', 'edge', 'no_padding'"
                )
            
            # Create square image with padding
            square_img = Image.new("RGB", (max_dim, max_dim), pad_color)
            square_img.paste(img, (left, top))
            square_img = square_img.resize(
                (target_size, target_size), Image.Resampling.BICUBIC
            )
            # ====================================================
        
        # Convert to tensor (except for no_padding mode, handled later)
        if mode != "no_padding":
            img_tensor = img_transform(square_img)
            images.append(img_tensor)

    # Convert coordinates to tensor
    original_coords = torch.from_numpy(np.array(original_coords)).float()
    
    # Special handling for no_padding mode: pad all images to same size for batching
    if mode == "no_padding":
        # Find max width and height across all images
        max_width = max(img.size[0] for img in processed_pil_images)
        max_height = max(img.size[1] for img in processed_pil_images)
        
        # Pad each image to (max_height, max_width) with black padding
        for img in processed_pil_images:
            width, height = img.size
            # Create canvas with max dimensions
            padded_img = Image.new("RGB", (max_width, max_height), (0, 0, 0))
            # Paste original image at top-left (or center if you prefer)
            padded_img.paste(img, (0, 0))
            # Convert to tensor
            img_tensor = img_transform(padded_img)
            images.append(img_tensor)
    
    # Stack all images
    images = torch.stack(images)
    
    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)
    
    return images, original_coords

def load_and_downsample_images_rgb_with_coords(
    image_paths, 
    target_width=518, 
    scale_factor=1.0,
    data_norm_type=None
):
    """
    Load, optionally downsample images in RGB format, and calculate coordinate transformation info in one pass.
    
    Args:
        image_paths (List[Path] or List[str]): List of paths to image files
        target_width (int): Target width for coordinate calculation. Defaults to 518.
        scale_factor (float): Factor to downsample images. 1.0 means no downsampling. Defaults to 1.0.
        data_norm_type (str, optional): Image normalization type. Defaults to None (no normalization).
    
    Returns:
        tuple: (
            torch.Tensor: Batched tensor of images with shape (N, 3, H, W) where H and W 
                         are the max height and width across all images (after downsampling).
            torch.Tensor: Array of shape (N, 6) containing [x1, y1, x2, y2, width, height] for each image.
                         width and height are from the (possibly downsampled) image.
        )
    """
    # Set up normalization based on data_norm_type
    if data_norm_type is None:
        img_transform = tvf.ToTensor()
    elif data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
        img_transform = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {data_norm_type}. "
            f"Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )
    
    pil_images = []
    coords_data = []
    
    for image_path in image_paths:
        # Load image using PIL
        img = Image.open(image_path)
        
        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        
        # Convert to RGB
        img = img.convert("RGB")
        
        # Downsample if scale_factor is not 1.0
        if scale_factor != 1.0:
            orig_w, orig_h = img.size
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            img = img.resize((new_w, new_h), Image.Resampling.AREA)
        
        pil_images.append(img)
        
        # Get dimensions for coordinate calculation
        width, height = img.size
        
        # Calculate coordinate transformation info
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        scale = target_width / max_dim
        
        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale
        
        # Store image coordinates and scale
        coords_data.append([x1, y1, x2, y2, width, height])
    
    # Find max width and height across all images for batching
    max_width = max(img.size[0] for img in pil_images)
    max_height = max(img.size[1] for img in pil_images)
    
    # Pad all images to same size with black padding and convert to tensors
    images = []
    for img in pil_images:
        width, height = img.size
        # Create canvas with max dimensions (black background)
        padded_img = Image.new("RGB", (max_width, max_height), (0, 0, 0))
        # Paste original image at top-left
        padded_img.paste(img, (0, 0))
        # Convert to tensor with normalization
        img_tensor = img_transform(padded_img)
        images.append(img_tensor)
    
    # Stack into batched tensor
    images = torch.stack(images)
    
    # Convert coordinates to tensor
    coords_data = torch.tensor(coords_data, dtype=torch.float32)
    
    # Add additional dimension if single image
    if len(image_paths) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            coords_data = coords_data.unsqueeze(0)
    
    return images, coords_data

def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # randomly pick which True positions to keep
    sampled_indices = np.random.choice(
        true_indices, size=max_trues, replace=False
    )  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
            - y_coords (numpy.ndarray): Array of y coordinates for all frames
            - x_coords (numpy.ndarray): Array of x coordinates for all frames
            - f_coords (numpy.ndarray): Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf


def run_mapanything(
    model,
    images,
    dtype,
    resolution=518,
    image_normalization_type="dinov2",
    memory_efficient_inference=False,
):
    # Images: [V, 3, H, W]
    # Check image shape
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # Hard-coded to use 518 for MapAnything
    images = F.interpolate(
        images, size=(resolution, resolution), mode="bilinear", align_corners=False
    )

    # Run inference
    views = []
    for view_idx in range(images.shape[0]):
        view = {
            "img": images[view_idx][None],  # Add batch dimension
            "data_norm_type": [image_normalization_type],
        }
        views.append(view)
    predictions = model.infer(
        views, memory_efficient_inference=memory_efficient_inference
    )

    # Process predictions
    (
        all_extrinsics,
        all_intrinsics,
        all_depth_maps,
        all_depth_confs,
        all_pts3d,
        all_img_no_norm,
        all_masks,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for pred in predictions:
        # Compute 3D points from depth, intrinsics, and camera pose
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
        pts3d, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Extract mask from predictions and combine with valid depth mask
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask

        # Convert tensors to numpy arrays
        extrinsic = (
            closed_form_pose_inverse(pred["camera_poses"])[0].cpu().numpy()
        )  # c2w -> w2c
        intrinsic = intrinsics_torch.cpu().numpy()
        depth_map = depthmap_torch.cpu().numpy()
        depth_conf = pred["conf"][0].cpu().numpy()
        pts3d = pts3d.cpu().numpy()
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()  # Denormalized image

        # Collect results
        all_extrinsics.append(extrinsic)
        all_intrinsics.append(intrinsic)
        all_depth_maps.append(depth_map)
        all_depth_confs.append(depth_conf)
        all_pts3d.append(pts3d)
        all_img_no_norm.append(img_no_norm)
        all_masks.append(mask)

    # Stack results into arrays
    all_extrinsics = np.stack(all_extrinsics)
    all_intrinsics = np.stack(all_intrinsics)
    all_depth_maps = np.stack(all_depth_maps)
    all_depth_confs = np.stack(all_depth_confs)
    all_pts3d = np.stack(all_pts3d)
    all_img_no_norm = np.stack(all_img_no_norm)
    all_masks = np.stack(all_masks)

    return (
        all_extrinsics,
        all_intrinsics,
        all_depth_maps,
        all_depth_confs,
        all_pts3d,
        all_img_no_norm,
        all_masks,
    )


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Set device and dtype
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Init model
    print("Loading MapAnything model from huggingface ...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    # # 添加配置字典
    # high_level_config = {
    #     "path": "configs/train.yaml",
    #     "hf_model_name": "facebook/map-anything",
    #     "model_str": "mapanything",
    #     "config_overrides": [
    #         "machine=aws",
    #         "model=mapanything",
    #         "model/task=images_only",
    #         "model.encoder.uses_torch_hub=false",
    #     ],
    #     "checkpoint_name": "model.safetensors",
    #     "config_name": "config.json",
    #     "trained_with_amp": True,
    #     "trained_with_amp_dtype": "bf16",
    #     "data_norm_type": "dinov2",
    #     "patch_size": 14,
    #     "resolution": 518,
    # }
    # # 使用 initialize_mapanything_model 而不是 from_pretrained
    # model = initialize_mapanything_model(high_level_config, torch.device(device))
    # model.eval()

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running MapAnything with 518
    mapanything_fixed_resolution = 518
    img_load_resolution = 1024

    # this function preprocesses images to the model's input format while preserving original information for later restoration.
    # images, original_coords = load_and_preprocess_images_square(
    #     image_path_list, img_load_resolution, model.encoder.data_norm_type
    # )
    # padding_mode: "no_padding", "stretch", "black", "white", "gray", "edge"
    images, original_coords = load_and_preprocess_images_square_with_mode(
        image_path_list, img_load_resolution, model.encoder.data_norm_type, mode="black"
    )
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # # 快速保存并展示第一张处理后的图像
    # if len(images) > 0:
    #     import matplotlib.pyplot as plt
        
    #     img_to_show = images[0].cpu()
        
    #     # 反归一化
    #     if model.encoder.data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
    #         img_norm = IMAGE_NORMALIZATION_DICT[model.encoder.data_norm_type]
    #         mean = torch.tensor(img_norm.mean).view(3, 1, 1)
    #         std = torch.tensor(img_norm.std).view(3, 1, 1)
    #         img_to_show = img_to_show * std + mean
        
    #     img_to_show = img_to_show.permute(1, 2, 0).numpy().clip(0, 1)
        
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(img_to_show)
    #     plt.title(f'Preprocessed Image', fontsize=14)
    #     plt.axis('off')
    #     # 显示窗口（非阻塞模式，程序继续运行）
    #     plt.show(block=False)
    #     print("✓ Image window displayed (you can continue working)")

    # Determine output directory
    output_dir = args.output_dir if args.output_dir is not None else args.scene_dir
    print(f"Output directory: {output_dir}")

    # Run MapAnything to estimate camera and depth
    # Run with 518 x 518 images
    extrinsic, intrinsic, depth_map, depth_conf, points_3d, img_no_norm, masks = (
        run_mapanything(
            model,
            images,
            dtype,
            mapanything_fixed_resolution,
            model.encoder.data_norm_type,
            memory_efficient_inference=args.memory_efficient_inference,
        )
    )

    # Prepare lists for GLB export if needed
    world_points_list = []
    images_list = []
    masks_list = []

    if args.save_glb:
        for i in range(img_no_norm.shape[0]):
            # Use the already denormalized images from predictions
            images_list.append(img_no_norm[i])

            # Add world points and masks from predictions
            world_points_list.append(points_3d[i])
            masks_list.append(masks[i])  # Use masks from predictions

    if args.use_ba: # 使用BA
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / mapanything_fixed_resolution
        shared_camera = args.shared_camera

        with torch.amp.autocast("cuda", dtype=dtype):
            # Predicting Tracks
            # Uses VGGSfM tracker
            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = (
                predict_tracks(
                    images,
                    conf=depth_conf,
                    points_3d=points_3d,
                    max_query_pts=args.max_query_pts,
                    query_frame_num=args.query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=args.fine_tracking,
                )
            )

            torch.cuda.empty_cache()

        # Rescale the intrinsic matrix from 518 to 1024
        # MapAnything 在 518×518 分辨率下预测了相机内参，但后续的点跟踪（track prediction）是在 1024×1024 的图像上进行的，图像分辨率变化时，焦距和主点坐标都需要等比例缩放。
        intrinsic[:, :2, :] *= scale
        # 创建可见性掩码，过滤低置信度的跟踪点
        track_mask = pred_vis_scores > args.vis_thresh


        # 深度学习模型输出          batch_np_matrix_to_pycolmap          COLMAP格式
        # ├─ NumPy数组      ─────────────────────────────────────►     ├─ Cameras
        # ├─ 3D点云                    【翻译+质检】                    ├─ Images  
        # ├─ 相机参数                                                  ├─ Points3D
        # └─ 特征轨迹                                                  └─ Tracks
        #                                   ↓
        #                              Bundle Adjustment
        #                              （捆集调整优化）
        #                                   ↓
        #                             优化后的重建结果
        # Init pycolmap reconstruction
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error, # small than 8.0 pixel are retained
            max_points3D_val=args.max_points3D_val,
            shared_camera=shared_camera, # 如果 shared_camera=False，每帧创建新的相机；如果 shared_camera=True，只在第一帧创建，后续帧复用。
            camera_type=args.camera_type,
            points_rgb=points_rgb,
            min_inlier_per_frame=args.min_inlier_per_frame,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = (
            False  # in the feedforward manner, we do not support shared camera
        )
        camera_type = (
            "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera
        )

        image_size = np.array(
            [mapanything_fixed_resolution, mapanything_fixed_resolution]
        )
        num_frames, height, width, _ = points_3d.shape

        # Denormalize images before computing RGB values
        points_rgb_images = F.interpolate(
            images,
            size=(mapanything_fixed_resolution, mapanything_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )

        # Convert normalized images back to RGB [0,1] range using the rgb function
        points_rgb_list = []
        for i in range(points_rgb_images.shape[0]):
            # rgb function expects single image tensor and returns numpy array in [0,1] range
            rgb_img = rgb(points_rgb_images[i], model.encoder.data_norm_type)
            points_rgb_list.append(rgb_img)

        # Stack and convert to uint8
        points_rgb = np.stack(points_rgb_list)  # Shape: (N, H, W, 3)
        points_rgb = (points_rgb * 255).astype(np.uint8)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        # ========== 新增：创建padding区域的mask ==========
        # 将original_coords转为numpy（如果还是tensor）
        original_coords_np = original_coords.cpu().numpy() if isinstance(original_coords, torch.Tensor) else original_coords
        
        # 创建一个mask，标记哪些点在有效区域内
        padding_mask = np.ones((num_frames, height, width), dtype=bool)
        
        for frame_idx in range(num_frames):
            # 获取该帧的有效区域边界 [x1, y1, x2, y2, width, height]
            x1, y1, x2, y2 = original_coords_np[frame_idx, :4]
            
            # 创建该帧的padding mask
            # 只有在 [x1, y1] 到 [x2, y2] 范围内的点才是有效的
            y_coords, x_coords = np.indices((height, width))
            
            # 检查每个点是否在有效区域内
            valid_x = (x_coords >= x1) & (x_coords <= x2)
            valid_y = (y_coords >= y1) & (y_coords <= y2)
            padding_mask[frame_idx] = valid_x & valid_y
        
        print(f"Padding mask created: {padding_mask.sum()} valid points out of {padding_mask.size} total points")
        # ====================================================

        # 应用confidence和padding mask
        conf_mask = depth_conf >= conf_thres_value
        
        # 合并两个mask
        combined_mask = conf_mask & padding_mask
        
        # ========== 新增：黑色/白色背景过滤 ==========
        # 添加命令行参数控制（需要在 parse_args() 中添加）
        filter_black_bg = args.filter_black_bg  # 或者从 args.filter_black_bg 获取
        filter_white_bg = args.filter_white_bg  # 或者从 args.filter_white_bg 获取

        if filter_black_bg or filter_white_bg:
            # points_rgb 的形状是 (num_frames, height, width, 3)，值在 [0, 255]
            
            # 创建背景过滤mask
            bg_mask = np.ones((num_frames, height, width), dtype=bool)
            
            if filter_black_bg:
                # 过滤黑色背景：RGB总和 < 16 的像素
                rgb_sum = points_rgb.sum(axis=-1)  # (num_frames, height, width)
                black_bg_mask = rgb_sum >= 1
                bg_mask = bg_mask & black_bg_mask
                print(f"Black background filtering: removed {(~black_bg_mask).sum()} points")
            
            if filter_white_bg:
                # 过滤白色背景：所有RGB通道 > 240 的像素
                white_bg_mask = ~(
                    (points_rgb[:, :, :, 0] > 240) &
                    (points_rgb[:, :, :, 1] > 240) &
                    (points_rgb[:, :, :, 2] > 240)
                )
                bg_mask = bg_mask & white_bg_mask
                print(f"White background filtering: removed {(~white_bg_mask).sum()} points")
            
            # 合并背景过滤mask
            combined_mask = combined_mask & bg_mask

        # ====================================================

        # At most writing max_points_for_colmap 3d points to colmap reconstruction object
        combined_mask = randomly_limit_trues(combined_mask, max_points_for_colmap)

        # 应用mask过滤点云
        points_3d = points_3d[combined_mask]
        points_xyf = points_xyf[combined_mask]
        points_rgb = points_rgb[combined_mask]
        
        print(f"After filtering: {len(points_3d)} points remaining")
        print(f"  - Confidence filtering: removed {(~conf_mask).sum()} points")
        print(f"  - Padding filtering: removed {(~padding_mask).sum()} points")
        if filter_black_bg or filter_white_bg:
            print(f"  - Background filtering: removed {(~bg_mask).sum()} points")
        print(f"  - Random sampling: limited to {max_points_for_colmap} points")

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
            max_points3D_val=args.max_points3D_val,
        )

        reconstruction_resolution = mapanything_fixed_resolution

    # 重命名和缩放相机参数
    # 将基于处理后图像的重建结果，还原到原始图像尺寸
    # 处理流程中，原始图像被缩放到1024*1024尺寸，之后模型推理尺寸是518*518，重建结果（相机参数基于处理后的图像尺寸）需要缩放到原始图像尺寸。
    if args.use_ba:
        reconstruction = rename_colmap_recons_and_rescale_camera(
            reconstruction,
            base_image_path_list,
            original_coords.cpu().numpy(),
            img_size=reconstruction_resolution,
            shift_point2d_to_original_res=True,
            shared_camera=shared_camera,
            resample_colors_from_original=True,   
            original_images_dir=image_dir,
            remove_padding_points=True,
        )
    else:
        reconstruction = rename_colmap_recons_and_rescale_camera(
            reconstruction,
            base_image_path_list,
            original_coords.cpu().numpy(),
            img_size=reconstruction_resolution,
            shift_point2d_to_original_res=True,
            shared_camera=shared_camera,
            resample_colors_from_original=False,   
            original_images_dir=image_dir,
            remove_padding_points=True,
        )

    print(f"Saving reconstruction to {output_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write_text(sparse_reconstruction_dir)

    # 导出二进制ply
    reconstruction.export_PLY(os.path.join(output_dir, "sparse/points.ply")) # 二进制ply

    # 额外导出ASCII版本（使用trimesh重新保存）
    # 从reconstruction中提取点云数据
    points = []
    colors = []
    for point3D_id in reconstruction.points3D:
        point3D = reconstruction.points3D[point3D_id]
        points.append(point3D.xyz)
        colors.append(point3D.color)
    
    if len(points) > 0:
        points = np.array(points)
        colors = np.array(colors)
        trimesh.PointCloud(points, colors=colors).export(
            os.path.join(output_dir, "sparse/points_ascii.ply"),
            encoding='ascii'  # ASCII格式
        )
        print(f"✓ Exported ASCII PLY: {output_dir}/sparse/points_ascii.ply")
    else:
        print("Warning: No valid points to save after filtering")

    # Export GLB if requested
    if args.save_glb:
        glb_output_path = os.path.join(output_dir, "dense_mesh.glb")
        print(f"Saving GLB file to: {glb_output_path}")

        # Stack all views
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)

        # Create predictions dict for GLB export
        predictions = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }

        # Convert to GLB scene
        scene_3d = predictions_to_glb(predictions, as_mesh=False)

        # Save GLB file
        scene_3d.export(glb_output_path)
        print(f"Successfully saved GLB file: {glb_output_path}")

    return True


# def rename_colmap_recons_and_rescale_camera(
#     reconstruction,
#     image_paths,
#     original_coords,
#     img_size,
#     shift_point2d_to_original_res=False,
#     shared_camera=False,
#     resample_colors_from_original=False,  
#     original_images_dir=None,              
# ):
#     rescale_camera = True
    
#     # 用于收集需要删除的3D点ID
#     points_to_remove = set()

#     for pyimageid in reconstruction.images:
#         # Reshaped the padded & resized image to the original size
#         # Rename the images to the original names
#         pyimage = reconstruction.images[pyimageid]
#         pycamera = reconstruction.cameras[pyimage.camera_id]
#         pyimage.name = image_paths[pyimageid - 1]

#         if rescale_camera:
#             # Rescale the camera parameters
#             pred_params = copy.deepcopy(pycamera.params)

#             real_image_size = original_coords[pyimageid - 1, -2:]
#             resize_ratio = max(real_image_size) / img_size
#             pred_params = pred_params * resize_ratio
#             real_pp = real_image_size / 2
#             pred_params[-2:] = real_pp  # center of the image

#             pycamera.params = pred_params
#             pycamera.width = real_image_size[0]
#             pycamera.height = real_image_size[1]

#         if shift_point2d_to_original_res:
#             # Also shift the point2D to original resolution
#             top_left = original_coords[pyimageid - 1, :2]

#             for point2D in pyimage.points2D:
#                 point2D.xy = (point2D.xy - top_left) * resize_ratio

#         if resample_colors_from_original and original_images_dir:
#             # 构建完整路径
#             original_image_path = os.path.join(original_images_dir, pyimage.name)
#             try:
#                 original_image = Image.open(original_image_path).convert("RGB")
#                 original_image_np = np.array(original_image)
                
#                 # 遍历这张图像看到的所有3D点
#                 for point2D in pyimage.points2D:
#                     if point2D.point3D_id != -1:  # 有效的3D点
#                         x, y = int(point2D.xy[0]), int(point2D.xy[1])
                        
#                         # 检查坐标是否在原始图像范围内
#                         if 0 <= x < original_image_np.shape[1] and 0 <= y < original_image_np.shape[0]:
#                             # 在范围内：更新颜色
#                             color = original_image_np[y, x]
#                             point3D = reconstruction.points3D[point2D.point3D_id]
#                             point3D.color = color
#                         else:
#                             # 超出范围：标记为需要删除
#                             points_to_remove.add(point2D.point3D_id)
                            
#             except Exception as e:
#                 print(f"Warning: Failed to resample colors from {pyimage.name}: {e}")

#         if shared_camera:
#             # If shared_camera, all images share the same camera
#             # No need to rescale any more
#             rescale_camera = False

#     # 删除所有标记的3D点
#     if resample_colors_from_original and points_to_remove:
#         print(f"Removing {len(points_to_remove)} 3D points that fall outside original image boundaries")
        
#         for point_id in points_to_remove:
#             if point_id in reconstruction.points3D:
#                 del reconstruction.points3D[point_id]

#     return reconstruction

def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
    resample_colors_from_original=False,  
    original_images_dir=None,
    remove_padding_points=True,  # 新增参数：是否移除padding区域的点
):
    rescale_camera = True
    
    # 用于收集需要删除的3D点ID
    points_to_remove = set()

    for pyimageid in reconstruction.images:
        # Reshaped the padded & resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        # ========== 改进：统一处理padding区域的点 ==========
        if remove_padding_points or resample_colors_from_original:
            original_image_np = None
            
            # 如果需要重采样颜色，则加载原始图像
            if resample_colors_from_original and original_images_dir:
                original_image_path = os.path.join(original_images_dir, pyimage.name)
                try:
                    original_image = Image.open(original_image_path).convert("RGB")
                    original_image_np = np.array(original_image)
                except Exception as e:
                    print(f"Warning: Failed to load original image {pyimage.name}: {e}")
            
            # 遍历这张图像看到的所有3D点
            for point2D in pyimage.points2D:
                if point2D.point3D_id != -1:  # 有效的3D点
                    x, y = int(point2D.xy[0]), int(point2D.xy[1])
                    
                    # 获取原始图像的实际尺寸
                    if original_image_np is not None:
                        img_height, img_width = original_image_np.shape[:2]
                    else:
                        # 如果没有加载图像，使用original_coords中存储的尺寸
                        img_width = int(original_coords[pyimageid - 1, -2])
                        img_height = int(original_coords[pyimageid - 1, -1])
                    
                    # 检查坐标是否在原始图像范围内
                    is_in_bounds = (0 <= x < img_width and 0 <= y < img_height)
                    
                    if is_in_bounds:
                        # 在范围内：如果需要，更新颜色
                        if original_image_np is not None:
                            color = original_image_np[y, x]
                            point3D = reconstruction.points3D[point2D.point3D_id]
                            point3D.color = color
                    else:
                        # 超出范围（在padding区域）：标记为需要删除
                        points_to_remove.add(point2D.point3D_id)

        if shared_camera:
            # If shared_camera, all images share the same camera
            # No need to rescale any more
            rescale_camera = False

    # 删除所有标记的3D点
    if points_to_remove:
        print(f"Removing {len(points_to_remove)} 3D points that fall outside original image boundaries (padding regions)")
        
        for point_id in points_to_remove:
            if point_id in reconstruction.points3D:
                del reconstruction.points3D[point_id]

    return reconstruction

if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
