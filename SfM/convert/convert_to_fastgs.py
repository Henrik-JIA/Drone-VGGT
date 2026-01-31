#!/usr/bin/env python3
"""
Convert incremental_feature_matcher output to FastGS training format.

FastGS (3D Gaussian Splatting) expects the following directory structure:
    output_path/
    ├── images/                   # undistorted images
    └── sparse/
        └── 0/
            ├── cameras.bin/txt   # camera intrinsics
            ├── images.bin/txt    # camera poses
            └── points3D.bin/txt  # 3D points (sparse or dense)
            └── points3D.ply      # PLY with normals (required by FastGS)

This script converts COLMAP output (binary or text format) from incremental_feature_matcher
to the FastGS expected format using pycolmap library.

Usage:
    python convert_to_fastgs.py \
        --colmap_model_dir path/to/temp_merged_final_to_global_sfm \
        --images_dir path/to/original/images \
        --output_dir path/to/fastgs_output \
        [--copy_images]  # Copy images instead of symlink (default: symlink)
        [--resize]       # Create resized image versions (images_2, images_4, images_8)

Example:
    python convert_to_fastgs.py \
        --colmap_model_dir ../output/Ganluo_images/sparse_incremental_reconstruction/temp_merged_final_to_global_sfm \
        --images_dir ../examples/Ganluo_images/images \
        --output_dir ./fastgs_Ganluo
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Set, Optional, Tuple

import numpy as np

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False
    print("Warning: pycolmap not available. Install with: pip install pycolmap")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def store_ply_with_normals(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """
    Store point cloud as PLY file with normals (required by FastGS).
    
    FastGS requires PLY files to have nx, ny, nz fields.
    We add zero normals since they are not available from COLMAP.
    
    Args:
        path: Output PLY file path
        xyz: (N, 3) array of 3D point coordinates
        rgb: (N, 3) array of RGB colors (0-255)
    """
    n_points = len(xyz)
    
    if n_points == 0:
        logger.warning("No points to write to PLY file")
        return
    
    # Use plyfile library for correct PLY generation (same as FastGS)
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        logger.error("plyfile package not installed. Install with: pip install plyfile")
        raise
    
    # Define dtype matching FastGS's storePly exactly
    # IMPORTANT: Field names must be 'red', 'green', 'blue' (not 'r', 'g', 'b')
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    # Create zero normals
    normals = np.zeros_like(xyz, dtype=np.float32)
    
    # Ensure correct dtypes
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)
    
    # Create structured array
    elements = np.empty(n_points, dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(str(path))


def filter_outliers(
    xyz: np.ndarray, 
    rgb: np.ndarray, 
    std_ratio: float = 2.0,
    max_coord: float = 1000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    移除异常值点云，包括距离质心过远的点和坐标值异常的点。
    
    Args:
        xyz: (N, 3) 3D点坐标数组
        rgb: (N, 3) RGB颜色数组
        std_ratio: 标准差倍数阈值，距离超过 mean + std_ratio * std 的点将被移除
        max_coord: 坐标绝对值最大阈值，超过此值的点将被移除
        
    Returns:
        filtered_xyz: 过滤后的3D点坐标
        filtered_rgb: 过滤后的RGB颜色
    """
    if len(xyz) == 0:
        return xyz, rgb
    
    n_original = len(xyz)
    
    # 1. 过滤坐标值异常的点（绝对值过大）
    coord_mask = np.all(np.abs(xyz) < max_coord, axis=1)
    
    # 2. 基于距离的异常值过滤
    # 计算质心
    centroid = np.mean(xyz[coord_mask], axis=0) if coord_mask.sum() > 0 else np.mean(xyz, axis=0)
    
    # 计算每个点到质心的距离
    distances = np.linalg.norm(xyz - centroid, axis=1)
    
    # 计算距离阈值：mean + std_ratio * std
    mean_dist = np.mean(distances[coord_mask]) if coord_mask.sum() > 0 else np.mean(distances)
    std_dist = np.std(distances[coord_mask]) if coord_mask.sum() > 0 else np.std(distances)
    threshold = mean_dist + std_ratio * std_dist
    
    # 距离掩码
    distance_mask = distances < threshold
    
    # 组合所有掩码
    final_mask = coord_mask & distance_mask
    
    filtered_xyz = xyz[final_mask]
    filtered_rgb = rgb[final_mask]
    
    n_filtered = len(filtered_xyz)
    n_removed = n_original - n_filtered
    
    if n_removed > 0:
        logger.info(f"  Outlier filtering: removed {n_removed} points "
                   f"({n_removed/n_original*100:.1f}%), kept {n_filtered} points")
        logger.info(f"    - Distance threshold: {threshold:.2f} (mean={mean_dist:.2f}, std={std_dist:.2f})")
        logger.info(f"    - Max coordinate threshold: {max_coord}")
    
    return filtered_xyz, filtered_rgb


def extract_points_from_reconstruction(reconstruction: "pycolmap.Reconstruction") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 3D points and colors from a pycolmap Reconstruction object.
    
    This follows the same pattern as vggt/demo_colmap.py for extracting point cloud data.
    
    Args:
        reconstruction: pycolmap.Reconstruction object
        
    Returns:
        xyz: (N, 3) array of 3D point coordinates
        rgb: (N, 3) array of RGB colors (0-255)
    """
    if len(reconstruction.points3D) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    
    xyz_list = []
    rgb_list = []
    
    for point3D_id, point3D in reconstruction.points3D.items():
        xyz_list.append(point3D.xyz)
        rgb_list.append(point3D.color)
    
    xyz = np.array(xyz_list, dtype=np.float32)
    rgb = np.array(rgb_list, dtype=np.uint8)
    
    return xyz, rgb


def get_image_names_from_reconstruction(reconstruction: "pycolmap.Reconstruction") -> Set[str]:
    """
    Extract image names from a pycolmap Reconstruction object.
    
    Args:
        reconstruction: pycolmap.Reconstruction object
        
    Returns:
        Set of image filenames used in the reconstruction
    """
    image_names = set()
    for image_id, image in reconstruction.images.items():
        image_names.add(image.name)
    return image_names


def find_image_files(images_dir: Path, image_names: Set[str]) -> dict:
    """
    Find image files in the source directory that match the reconstruction.
    
    Args:
        images_dir: Directory containing source images
        image_names: Set of image names from COLMAP reconstruction
        
    Returns:
        Dict mapping image name to full path
    """
    image_paths = {}
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    for img_file in images_dir.iterdir():
        if img_file.suffix in supported_formats:
            if img_file.name in image_names:
                image_paths[img_file.name] = img_file
    
    return image_paths


def create_fastgs_structure(
    colmap_model_dir: Path,
    images_dir: Path,
    output_dir: Path,
    copy_images: bool = False,
    resize: bool = False,
    magick_executable: str = "magick",
    output_format: str = "binary",
    filter_outliers_enabled: bool = True,
    outlier_std_ratio: float = 2.0,
    outlier_max_coord: float = 1000.0
) -> bool:
    """
    Create FastGS training directory structure from COLMAP output.
    
    Uses pycolmap library to read reconstruction (supports both binary and text format),
    similar to the approach in vggt/demo_colmap.py.
    
    Args:
        colmap_model_dir: Directory containing COLMAP files (cameras.bin/txt, images.bin/txt, points3D.bin/txt)
        images_dir: Directory containing source images
        output_dir: Output directory for FastGS format
        copy_images: If True, copy images; if False, create symlinks
        resize: If True, create resized versions (images_2, images_4, images_8)
        magick_executable: Path to ImageMagick executable
        output_format: Output format for COLMAP files ("binary" or "text")
        filter_outliers_enabled: If True, filter outlier points from point cloud
        outlier_std_ratio: Standard deviation ratio for outlier filtering (default: 2.0)
        outlier_max_coord: Maximum coordinate value threshold (default: 1000.0)
        
    Returns:
        True if successful, False otherwise
    """
    if not PYCOLMAP_AVAILABLE:
        logger.error("pycolmap is required. Install with: pip install pycolmap")
        return False
    
    # Validate input paths
    if not colmap_model_dir.exists():
        logger.error(f"COLMAP model directory does not exist: {colmap_model_dir}")
        return False
    
    if not images_dir.exists():
        logger.error(f"Images directory does not exist: {images_dir}")
        return False
    
    # Load reconstruction using pycolmap (supports both binary and text format)
    logger.info(f"Loading COLMAP reconstruction from: {colmap_model_dir}")
    try:
        reconstruction = pycolmap.Reconstruction(str(colmap_model_dir))
        logger.info(f"  Loaded: {len(reconstruction.images)} images, "
                   f"{len(reconstruction.points3D)} 3D points, "
                   f"{len(reconstruction.cameras)} cameras")
    except Exception as e:
        logger.error(f"Failed to load COLMAP reconstruction: {e}")
        return False
    
    if len(reconstruction.images) == 0:
        logger.error("Reconstruction has no images")
        return False
    
    # Create output directory structure
    logger.info(f"Creating FastGS directory structure at: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write reconstruction to output directory using pycolmap
    # This follows the same pattern as vggt/demo_colmap.py: reconstruction.write(sparse_reconstruction_dir)
    logger.info(f"Writing COLMAP model files to sparse/0/ ({output_format} format)...")
    try:
        if output_format == "binary":
            reconstruction.write_binary(str(sparse_dir))
        else:
            reconstruction.write_text(str(sparse_dir))
        logger.info(f"  Written: cameras, images, points3D ({output_format})")
    except Exception as e:
        logger.error(f"Failed to write reconstruction: {e}")
        return False
    
    # Generate PLY file with normals (required by FastGS)
    # Extract points from reconstruction and save with normals
    logger.info("Generating points3D.ply with normals (required by FastGS)...")
    ply_output = sparse_dir / "points3D.ply"
    
    try:
        xyz, rgb = extract_points_from_reconstruction(reconstruction)
        n_original = len(xyz)
        
        # Apply outlier filtering if enabled
        if filter_outliers_enabled and len(xyz) > 0:
            logger.info(f"  Applying outlier filtering (std_ratio={outlier_std_ratio}, max_coord={outlier_max_coord})...")
            xyz, rgb = filter_outliers(xyz, rgb, std_ratio=outlier_std_ratio, max_coord=outlier_max_coord)
        
        if len(xyz) > 0:
            store_ply_with_normals(ply_output, xyz, rgb)
            logger.info(f"  Generated: points3D.ply ({len(xyz)} points with normals)")
        else:
            logger.warning("  No 3D points found in reconstruction!")
    except Exception as e:
        logger.error(f"  Failed to generate PLY with normals: {e}")
        import traceback
        traceback.print_exc()
    
    # Get image names from reconstruction
    logger.info("Getting image names from reconstruction...")
    image_names = get_image_names_from_reconstruction(reconstruction)
    logger.info(f"  Found {len(image_names)} images in reconstruction")
    
    # Find corresponding image files
    image_paths = find_image_files(images_dir, image_names)
    logger.info(f"  Found {len(image_paths)} matching image files")
    
    if len(image_paths) < len(image_names):
        missing = image_names - set(image_paths.keys())
        logger.warning(f"  Missing {len(missing)} images: {list(missing)[:5]}...")
    
    # Copy or link images
    logger.info(f"{'Copying' if copy_images else 'Linking'} images to images/...")
    for img_name, img_path in image_paths.items():
        dst = images_output_dir / img_name
        
        if copy_images:
            shutil.copy2(img_path, dst)
        else:
            # Create symlink (works on Windows with developer mode or admin)
            try:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(img_path.resolve())
            except OSError:
                # Fallback to copy if symlink fails
                logger.warning(f"  Symlink failed for {img_name}, copying instead")
                shutil.copy2(img_path, dst)
    
    logger.info(f"  Processed {len(image_paths)} images")
    
    # Create resized versions if requested
    if resize:
        logger.info("Creating resized image versions...")
        
        for scale, suffix in [(2, "images_2"), (4, "images_4"), (8, "images_8")]:
            resize_dir = output_dir / suffix
            resize_dir.mkdir(parents=True, exist_ok=True)
            
            resize_percent = 100 // scale
            
            for img_name in image_paths.keys():
                src = images_output_dir / img_name
                dst = resize_dir / img_name
                
                # Copy then resize
                shutil.copy2(src, dst)
                
                resize_cmd = f'{magick_executable} mogrify -resize {resize_percent}% "{dst}"'
                exit_code = os.system(resize_cmd)
                
                if exit_code != 0:
                    logger.warning(f"  Resize failed for {img_name} at {resize_percent}%")
            
            logger.info(f"  Created {suffix}/ ({resize_percent}% size)")
    
    # Create a metadata file with conversion info
    metadata_path = output_dir / "conversion_info.txt"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("# FastGS Conversion Info\n")
        f.write(f"Source COLMAP model: {colmap_model_dir.resolve()}\n")
        f.write(f"Source images: {images_dir.resolve()}\n")
        f.write(f"Number of images: {len(image_paths)}\n")
        f.write(f"Number of 3D points (original): {len(reconstruction.points3D)}\n")
        f.write(f"Number of cameras: {len(reconstruction.cameras)}\n")
        f.write(f"Copy mode: {'copy' if copy_images else 'symlink'}\n")
        f.write(f"Output format: {output_format}\n")
        f.write(f"Resized versions: {'yes' if resize else 'no'}\n")
        f.write(f"Outlier filtering: {'enabled' if filter_outliers_enabled else 'disabled'}\n")
        if filter_outliers_enabled:
            f.write(f"  - std_ratio: {outlier_std_ratio}\n")
            f.write(f"  - max_coord: {outlier_max_coord}\n")
    
    logger.info(f"\n✓ Conversion complete!")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Total images: {len(image_paths)}")
    logger.info(f"  Total 3D points: {len(reconstruction.points3D)}")
    logger.info(f"\nTo train FastGS, run:")
    logger.info(f"  python train.py -s {output_dir}")
    
    return True


def main():
    parser = ArgumentParser(description="Convert incremental_feature_matcher output to FastGS format")
    
    parser.add_argument(
        "--colmap_model_dir", "-m",
        type=str,
        required=True,
        help="Path to COLMAP model directory (supports both binary and text format)"
    )
    parser.add_argument(
        "--images_dir", "-i",
        type=str,
        required=True,
        help="Path to original images directory"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory for FastGS format"
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images instead of creating symlinks (default: symlink)"
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Create resized image versions (images_2, images_4, images_8)"
    )
    parser.add_argument(
        "--magick_executable",
        type=str,
        default="magick",
        help="Path to ImageMagick executable (default: magick)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="binary",
        choices=["binary", "text"],
        help="Output format for COLMAP files (default: binary)"
    )
    
    args = parser.parse_args()
    
    success = create_fastgs_structure(
        colmap_model_dir=Path(args.colmap_model_dir),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        copy_images=args.copy_images,
        resize=args.resize,
        magick_executable=args.magick_executable,
        output_format=args.output_format
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # ==================== 调试模式：直接指定路径 ====================
    # 设置 DEBUG = True 可直接运行，无需命令行参数
    DEBUG = True
    
    if DEBUG:
        # 获取脚本所在目录，确保相对路径正确
        SCRIPT_DIR = Path(__file__).parent.resolve()
        
        # 直接指定输入输出路径（方便调试）
        # 使用绝对路径，避免工作目录问题
        # 支持从 binary 或 text 格式的 COLMAP 重建读取
        COLMAP_MODEL_DIR = SCRIPT_DIR / "../../output/Ganluo_images/sparse_incremental_reconstruction/temp_merged/merged_8"
        IMAGES_DIR = SCRIPT_DIR / "../../examples/Ganluo_images/images"
        OUTPUT_DIR = SCRIPT_DIR / "./fastgs_Ganluo"
        COPY_IMAGES = True      # True: 复制图片, False: 创建符号链接
        RESIZE = False          # 是否创建缩小版本 (images_2, images_4, images_8)
        OUTPUT_FORMAT = "binary"  # 输出格式: "binary" 或 "text"
        
        # ==================== 离散点过滤参数 ====================
        FILTER_OUTLIERS = False      # 是否启用离散点过滤
        OUTLIER_STD_RATIO = 4.5    # 标准差倍数（越小过滤越严格，推荐2.0-4.0）
        OUTLIER_MAX_COORD = 1000.0  # 坐标绝对值最大阈值（超过此值的点会被移除）
        
        success = create_fastgs_structure(
            colmap_model_dir=COLMAP_MODEL_DIR.resolve(),
            images_dir=IMAGES_DIR.resolve(),
            output_dir=OUTPUT_DIR.resolve(),
            copy_images=COPY_IMAGES,
            resize=RESIZE,
            output_format=OUTPUT_FORMAT,
            filter_outliers_enabled=FILTER_OUTLIERS,
            outlier_std_ratio=OUTLIER_STD_RATIO,
            outlier_max_coord=OUTLIER_MAX_COORD,
        )
        sys.exit(0 if success else 1)
    else:
        # 命令行模式
        main()

