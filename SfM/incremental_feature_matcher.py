#!/usr/bin/env python3
"""
Incremental feature extraction and matching for SfM using pycolmap.
Process images one by one: extract features, match with previous images, 
build tracks, and triangulate.
"""

import os
import sys
import copy
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image
import laspy
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import subprocess
import cv2

# UTM coordinate conversion imports (conditional)
try:
    import pymap3d as pm
    import pyproj
    UTM_EXPORT_AVAILABLE = True
except ImportError:
    UTM_EXPORT_AVAILABLE = False
    print("Warning: pymap3d or pyproj not available. UTM export will not work. Install with: pip install pymap3d pyproj")

current_dir = Path(__file__).parent
project_root = current_dir.parent  # drone-map-anything 根目录
third_dir = project_root / "third" / "vggt"  # 指向 third/vggt 目录（vggt项目根目录）
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(third_dir) not in sys.path:
    sys.path.insert(0, str(third_dir))

from feature_matcher import FeatureMatcherSfM
from merge_full_pipeline import merge_reconstructions
from merge_confidence_blend import (
    merge_two_reconstructions as merge_by_confidence_blend,
    find_common_images,
    build_correspondences_parallel,
    find_corresponding_3d_points,
    estimate_sim3_ransac
)
from merge_confidence import merge_two_reconstructions as merge_by_confidence
from sfm_extraction import extract_sfm_reconstruction_from_global
from sfm_visualizer import SfMVisualizer
from reconstruction_alignment import (
    rescale_reconstruction_to_original_size,
    find_single_images_pair_matches,
    estimate_sim3_transform,
)
from reconstruction_rename import rename_colmap_recons_and_rescale_camera
from utils.gps import extract_gps_from_image, lat_lon_to_enu
from utils.xmp import parse_xmp_tags
from mapanything.utils.image import preprocess_inputs
from mapanything.third_party.projection import project_3D_points_np, project_3D_points
from mapanything.models import MapAnything
from mapanything.third_party.track_predict import predict_tracks
from mapanything.third_party.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from mapanything.utils.image import rgb

# VGGT model imports (conditional)
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    VGGT_AVAILABLE = True
except ImportError:
    VGGT_AVAILABLE = False
    print("Warning: VGGT model not available. Install vggt package to use VGGT model.")

def cam_from_enu_transform(roll, pitch, yaw):
    """
    Returns the transformation matrix from ENU to camera coordinates.
    
    Args:
        roll: Gimbal roll angle in degrees
        pitch: Gimbal pitch angle in degrees
        yaw: Gimbal yaw angle in degrees
    
    Returns:
        3x3 rotation matrix from ENU to camera coordinates
    """
    # ENU to NED
    ned_from_enu = R.align_vectors(
        a=[[0, 1, 0], [1, 0, 0], [0, 0, -1]], 
        b=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )[0].as_matrix()

    # Gimbal rotation in NED (ZYX order)
    ned_from_gimbal = R.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()
    gimbal_from_ned = ned_from_gimbal.T

    # Camera from NED
    cam_from_ned = R.align_vectors(
        a=[[0, 0, 1], [1, 0, 0], [0, 1, 0]], 
        b=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )[0].as_matrix()

    cam_from_enu = cam_from_ned @ gimbal_from_ned @ ned_from_enu
    return cam_from_enu

class IncrementalFeatureMatcherSfM:
    """Incremental feature extraction and matching using pycolmap.
    
    This class processes images one by one and stores their intrinsic and extrinsic parameters.
    """

    def __init__(
        self,
        output_dir: Path,
        reconstruction_type: str = 'dense_feature_points',  # 'dense_feature_points' | 'each_pixel_feature_points'
        model_type: str = 'mapanything',  # 'mapanything' | 'vggt'
        model_path: Optional[str] = None,  # 模型权重路径（VGGT需要）
        global_sparse_reconstruction: Optional[pycolmap.Reconstruction] = None,
        min_images_for_scale: int = 2,
        overlap: int = 1,
        max_reproj_error: float = 10.0,
        max_points3D_val: int = 5000,
        min_inlier_per_frame: int = 32,
        pred_vis_scores_thres_value: float = 0.3, 
        filter_edge_margin: float = 10.0,  # 边缘过滤范围（像素），默认10，设为0禁用
        merge_voxel_size: float = 1.0,  # 点云合并时的体素大小（米）
        merge_boundary_filter: bool = True,  # 是否启用边界过滤
        merge_statistical_filter: bool = False,  # 是否启用统计过滤
        merge_method: str = 'confidence_blend',  # 'full' | 'confidence' | 'confidence_blend' 合并方式
        enable_visualization: bool = True,
        visualization_mode: str = 'merged',  # 'aligned' | 'merged'，点云可视化模式
        verbose: bool = False,
    ):
        """Initialize incremental feature matcher.
        
        Args:
            output_dir: Directory for output files
            reconstruction_type: Type of reconstruction to use, 'dense_feature_points' | 'each_pixel_feature_points'
            model_type: Type of model to use, 'mapanything' | 'vggt'
            model_path: Path to model weights (required for VGGT, optional for MapAnything)
            min_images_for_scale: Minimum number of images before calculating scale.
                                  2 = calculate from 2nd image (default)
                                  3 = calculate from 3rd image
                                  4 = calculate from 4th image
                                  etc.
            overlap: Number of overlapping images between consecutive reconstructions
            max_reproj_error: Maximum reprojection error (in pixels) for filtering tracks
            max_points3D_val: Per-component absolute-value threshold for 3D points (a point is kept only if |x|, |y|, and |z| are all less than this value).
            min_inlier_per_frame: Minimum inlier count per frame for valid BA
            pred_vis_scores_thres_value: Visibility confidence threshold for tracks
            filter_edge_margin: Edge margin for filtering points (in pixels), default 10, set to 0 to disable
            merge_voxel_size: Voxel size for point cloud merging (in meters), default 1.0
            merge_boundary_filter: Whether to enable boundary filtering during merge, default True
            merge_statistical_filter: Whether to enable statistical filtering during merge, default False
            merge_method: Merge method selection:
                'full' - use merge_full_pipeline.py with full pipeline
                'confidence' - use merge_confidence.py with simple confidence-based selection
                'confidence_blend' - use merge_confidence_blend.py with confidence-based selection 
                                     and smooth edge blending/interpolation (default)
            enable_visualization: Whether to start viser server for visualization
            visualization_mode: Point cloud visualization mode, 'aligned' (per batch) or 'merged' (unified)
            verbose: Enable verbose logging
        """
        # Model type validation
        if model_type not in ['mapanything', 'vggt']:
            raise ValueError(f"model_type must be 'mapanything' or 'vggt', got: {model_type}")
        if model_type == 'vggt' and not VGGT_AVAILABLE:
            raise ValueError("VGGT model is not available. Please install the vggt package.")
        
        self.model_type = model_type
        self.model_path = model_path
        
        # Model (lazy loading)
        self.model = None
        self.device = None
        self.dtype = None  # For VGGT mixed precision

        self.output_dir = Path(output_dir)
        
        if reconstruction_type not in ['dense_feature_points', 'each_pixel_feature_points']:
            raise ValueError(f"reconstruction_type must be 'dense_feature_points' or 'each_pixel_feature_points', current is: {reconstruction_type}")
        if reconstruction_type == 'dense_feature_points':
            self.reconstruction_type = 'dense_feature_points'
        else:
            self.reconstruction_type = 'each_pixel_feature_points'
        
        self.global_sparse_reconstruction = global_sparse_reconstruction
        self.verbose = verbose
        self.min_images_for_scale = max(2, min_images_for_scale)
        self.overlap = overlap       
        self.pred_vis_scores_thres_value = pred_vis_scores_thres_value
        self.max_reproj_error = max_reproj_error
        self.max_points3D_val = max_points3D_val
        self.min_inlier_per_frame = min_inlier_per_frame
        self.filter_edge_margin = filter_edge_margin
        self.merge_voxel_size = merge_voxel_size
        self.merge_boundary_filter = merge_boundary_filter
        self.merge_statistical_filter = merge_statistical_filter
        self.merge_method = merge_method  # 'full', 'confidence', or 'confidence_blend'
        
        # Visualization mode: 'aligned' (每个batch单独点云) or 'merged' (合并后整体点云)
        if visualization_mode not in ['aligned', 'merged']:
            raise ValueError(f"visualization_mode must be 'aligned' or 'merged', got: {visualization_mode}")
        self.visualization_mode = visualization_mode

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.next_image_id: int = 1
        
        # Coordinate system
        self.enu_origin: Optional[np.ndarray] = None  # [lat, lon, alt]

        # Store image paths
        self.image_paths: List[Path] = []
        
        # Store intrinsic and extrinsic parameters
        self.ori_extrinsic: List[Dict] = []
        self.ori_intrinsic: List[Dict] = []
        self.input_views: List[Dict] = []
        self.preprocessed_views: List[Dict] = []
        self.scale_info: List[Dict] = []
        self.inference_outputs: List[Dict] = []
        self.batch_tracks: List[Dict] = [] 
        self.image_tracks: List[Dict] = []  # 存储每个影像的跟踪信息
        self.inference_reconstructions: List[Dict] = []  # 存储推理结果构建的 pycolmap 重建结果
        self.sfm_reconstructions: List[Dict] = []  # 存储传统SfM重建结果
        self.merged_reconstruction: Optional[pycolmap.Reconstruction] = None # 每次合并后更新的重建结果
        self.recovered_inference_outputs: List[Dict] = []
        
        # 像素级置信度图存储在 inference_reconstructions[i]['pixel_3d_mapping'][global_idx]['conf'] 中
        # 格式: {global_image_idx: {'pts3d': (H,W,3), 'conf': (H,W), 'valid_mask': (H,W)}}
        # conf 已缩放到原图尺寸，与 reconstruction 中的 2D 点坐标一致

        # Georeferenced coordinate system (for export_georeferenced)
        self.geo_center: Optional[np.ndarray] = None  # [x, y, alt] center offset in target CRS
        self.output_epsg_code: Optional[int] = None  # EPSG code for output CRS
        self.rec_georef: Optional[pycolmap.Reconstruction] = None  # Reconstruction in georeferenced coordinates
        self.rec_georef_dir: Optional[Path] = None  # Output directory for georeferenced reconstruction

        self.enable_visualization = enable_visualization
        
        # Setup visualization using SfMVisualizer
        self.visualizer: Optional[SfMVisualizer] = None
        if self.enable_visualization:
            self.visualizer = SfMVisualizer(
                visualization_mode=self.visualization_mode,
                verbose=self.verbose,
            )
            self.visualizer.setup()

    def _load_model(self):
        """Load model (lazy loading).
        
        Supports both MapAnything and VGGT models.
        
        Returns:
            Loaded model
        """
        if self.model is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.verbose:
                print(f"Using device: {self.device}")
            
            if self.model_type == 'mapanything':
                model_name = "facebook/map-anything"
                if self.verbose:
                    print(f"Loading MapAnything model: {model_name}...")
                self.model = MapAnything.from_pretrained(model_name).to(self.device)
                if self.verbose:
                    print("✓ MapAnything model loaded")
            
            elif self.model_type == 'vggt':
                if self.verbose:
                    print("Loading VGGT model...")
                
                # Determine dtype for mixed precision
                if torch.cuda.is_available():
                    self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                else:
                    self.dtype = torch.float32
                
                self.model = VGGT()
                
                # Load weights if provided
                if self.model_path:
                    # 处理相对路径：基于项目根目录解析
                    model_path = Path(self.model_path)
                    if not model_path.is_absolute():
                        model_path = project_root / model_path
                    
                    if self.verbose:
                        print(f"Loading weights from: {model_path}")
                    state_dict = torch.load(str(model_path), map_location='cpu')
                    self.model.load_state_dict(state_dict)
                else:
                    # Try to load from default location or huggingface
                    try:
                        # Try loading from huggingface hub
                        self.model = VGGT.from_pretrained("facebook/vggt")
                        if self.verbose:
                            print("Loaded VGGT model from Hugging Face Hub")
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Could not load pretrained weights: {e}")
                            print("Using randomly initialized VGGT model")
                
                self.model.to(self.device)
                if self.verbose:
                    print("✓ VGGT model loaded")
        
        return self.model

    def _release_model(self):
        """Release model from memory to free GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None
            self.dtype = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.verbose:
                print("✓ Model released from memory")

    def _cleanup_intermediate_data(self, keep_last_n: int = 2):
        """清理中间数据以释放内存，只保留最新的 N 个批次。
        
        Args:
            keep_last_n: 保留最新的批次数量，默认为2（用于重叠处理）
        """
        import gc
        
        cleaned_items = []
        
        # 1. 清理 batch_tracks - 只保留最新的 keep_last_n 个批次
        if len(self.batch_tracks) > keep_last_n:
            num_to_remove = len(self.batch_tracks) - keep_last_n
            for i in range(num_to_remove):
                batch = self.batch_tracks[i]
                # 显式删除大型 numpy 数组
                for key in ['pred_tracks', 'pred_vis_scores', 'pred_confs', 'points_3d', 'points_rgb']:
                    if key in batch and batch[key] is not None:
                        del batch[key]
            self.batch_tracks = self.batch_tracks[-keep_last_n:]
            cleaned_items.append(f"batch_tracks: 删除 {num_to_remove} 个")
        
        # 2. 清理 image_tracks - 只保留最新的 keep_last_n * min_images_for_scale 个
        max_image_tracks = keep_last_n * self.min_images_for_scale
        if len(self.image_tracks) > max_image_tracks:
            num_to_remove = len(self.image_tracks) - max_image_tracks
            for i in range(num_to_remove):
                track_info = self.image_tracks[i]
                # 显式删除大型数组
                for key in ['tracks_2d', 'vis_scores', 'confs', 'points_3d', 'points_rgb']:
                    if key in track_info and track_info[key] is not None:
                        del track_info[key]
            self.image_tracks = self.image_tracks[-max_image_tracks:]
            cleaned_items.append(f"image_tracks: 删除 {num_to_remove} 个")
        
        # 3. 清理 inference_reconstructions 中的 pixel_3d_mapping（保留 reconstruction 对象）
        if len(self.inference_reconstructions) > keep_last_n:
            for i in range(len(self.inference_reconstructions) - keep_last_n):
                recon_data = self.inference_reconstructions[i]
                if 'pixel_3d_mapping' in recon_data and recon_data['pixel_3d_mapping']:
                    # 清空 pixel_3d_mapping 字典
                    for global_idx in list(recon_data['pixel_3d_mapping'].keys()):
                        mapping = recon_data['pixel_3d_mapping'][global_idx]
                        for key in ['pts3d', 'conf', 'valid_mask']:
                            if key in mapping:
                                del mapping[key]
                    recon_data['pixel_3d_mapping'] = {}
            cleaned_items.append(f"pixel_3d_mapping: 清理 {len(self.inference_reconstructions) - keep_last_n} 个批次")
        
        # 4. 清理 recovered_inference_outputs - 只保留最新的
        max_recovered = keep_last_n * self.min_images_for_scale
        if len(self.recovered_inference_outputs) > max_recovered:
            num_to_remove = len(self.recovered_inference_outputs) - max_recovered
            for i in range(num_to_remove):
                output = self.recovered_inference_outputs[i]
                # 清理 tensor 数据
                for key in ['pts3d', 'pts3d_cam', 'camera_poses', 'cam_trans', 'cam_quats', 'conf']:
                    if key in output and output[key] is not None:
                        del output[key]
            self.recovered_inference_outputs = self.recovered_inference_outputs[-max_recovered:]
            cleaned_items.append(f"recovered_inference_outputs: 删除 {num_to_remove} 个")
        
        # 5. 清理 inference_outputs 中的大型 tensor（保留元数据）
        if len(self.inference_outputs) > keep_last_n * self.min_images_for_scale:
            keep_from_idx = len(self.inference_outputs) - keep_last_n * self.min_images_for_scale
            cleaned_count = 0
            for i in range(keep_from_idx):
                output = self.inference_outputs[i]
                # 清理 current_output 中的 tensor
                if 'current_output' in output and output['current_output'] is not None:
                    for key in ['pts3d', 'pts3d_cam', 'conf', 'camera_poses', 'intrinsics']:
                        if key in output['current_output']:
                            del output['current_output'][key]
                    output['current_output'] = None
                # 清理 outputs 列表
                if 'outputs' in output and output['outputs'] is not None:
                    output['outputs'] = None
                cleaned_count += 1
            if cleaned_count > 0:
                cleaned_items.append(f"inference_outputs: 清理 {cleaned_count} 个的 tensor")
        
        # 6. 清理 input_views 和 preprocessed_views 中的图像 tensor
        keep_views_from = max(0, len(self.input_views) - keep_last_n * self.min_images_for_scale)
        views_cleaned = 0
        for i in range(keep_views_from):
            if i < len(self.input_views) and 'img' in self.input_views[i]:
                self.input_views[i]['img'] = None
                views_cleaned += 1
            if i < len(self.preprocessed_views) and 'img' in self.preprocessed_views[i]:
                self.preprocessed_views[i]['img'] = None
        if views_cleaned > 0:
            cleaned_items.append(f"input/preprocessed_views: 清理 {views_cleaned} 个图像 tensor")
        
        # 7. 强制垃圾回收
        gc.collect()
        
        # 8. 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.verbose and cleaned_items:
            print(f"  ✓ 内存清理完成: {', '.join(cleaned_items)}")

    def add_image(self, image_path: Path) -> bool:
        """Add a new image and store its intrinsic and extrinsic parameters.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            True if successful, False otherwise
        """
        # Store image path
        self.image_paths.append(image_path)

        # Extract GPS and XMP metadata
        gps_data = extract_gps_from_image(image_path)
        if not gps_data:
            print(f"Error: No GPS data found in {image_path}")
            return False
        
        xmp_data = parse_xmp_tags(image_path)
        if not xmp_data:
            print(f"Error: No XMP data found in {image_path}")
            return False

        # Check if this is the first image
        if len(self.ori_extrinsic) == 0:
            # Set ENU origin only for the first image
            self.enu_origin = np.array(gps_data)
            if self.verbose:
                print(f"ENU origin set to: lat={gps_data[0]:.6f}, lon={gps_data[1]:.6f}, alt={gps_data[2]:.1f}")
            # Initialize with first image (ENU position is origin)
            enu_pos = np.array([0.0, 0.0, 0.0])
        else:
            # Convert GPS to ENU for subsequent images
            enu_pos = lat_lon_to_enu(
                gps_data[0], gps_data[1], gps_data[2],
                self.enu_origin[0], self.enu_origin[1], self.enu_origin[2]
            )

        # Process the image with computed ENU position
        success = self._initialize_image(image_path, gps_data, enu_pos, xmp_data)

        # Run inference
        inference_success = self._run_inference(image_path, self.preprocessed_views)

        # ==================== 批量恢复原始位姿 ====================
        # # 检查是否达到批量恢复的条件
        # num_images = len(self.inference_outputs)
        # # num_recovered = len(self.recovered_inference_outputs)
        # num_reconstructed = len(self.inference_reconstructions)
        # # 计算还未恢复的图像数量
        # # num_unrecovered = num_images - num_recovered
        # num_unreconstructed = num_images - num_reconstructed
        # # 当未恢复的图像数量达到 min_images_for_scale 时，批量恢复这批图像
        # if num_unreconstructed >= self.min_images_for_scale:

        # 检查是否达到批量恢复的条件
        num_images = len(self.inference_outputs)
        num_reconstructed = len(self.inference_reconstructions)

        # 计算下一批次应该处理的范围
        overlap = self.overlap  # 每次重叠1张影像

        if num_reconstructed == 0:
            # 第一次：从0开始
            start_idx = 0
            end_idx = self.min_images_for_scale  # 例如：0到3，处理[0,1,2]
        else:
            # 后续批次：从上一批的倒数第overlap张开始
            last_batch = self.inference_reconstructions[-1]
            start_idx = last_batch['end_idx'] - overlap  # 例如：3-1=2
            end_idx = start_idx + self.min_images_for_scale  # 例如：2+3=5，处理[2,3,4]

        # 检查是否有足够的图像来构建这一批次
        if num_images >= end_idx:  # ← 关键：检查是否已经有足够的图像
            # ==================== 从全局重建中提取传统SfM结果（新增）====================
            if self.global_sparse_reconstruction is not None:
                sfm_result = extract_sfm_reconstruction_from_global(
                    global_sparse_reconstruction=self.global_sparse_reconstruction,
                    image_paths=self.image_paths,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    output_dir=self.output_dir,
                    verbose=self.verbose,
                )
                if sfm_result is not None:
                    self.sfm_reconstructions.append(sfm_result)
                    sfm_extract_success = True
                else:
                    sfm_extract_success = False
                    if self.verbose:
                        print("  Skipping SfM extraction (no global reconstruction available)")

            if self.reconstruction_type == 'dense_feature_points':
                # ==================== 预测tracks（在推理坐标系） ====================
                track_predict_success = self._predict_tracks_for_batch(
                    start_idx=start_idx,
                    end_idx=end_idx
                )

                # ==================== 构建pycolmap重建 ====================
                pycolmap_success = self._build_pycolmap_reconstruction(
                    start_idx=start_idx,
                    end_idx=end_idx
                )

                # ==================== 合并reconstruction中间结果 ====================
                merge_reconstruction_success = self._merge_reconstruction_intermediate_results()

                # # ==================== 批量恢复位姿和3D点到真实坐标系 ====================
                # batch_recover_success = self._batch_recover_original_poses(
                #     image_path=image_path,
                #     start_idx=num_recovered,
                #     end_idx=num_images,
                #     transform_tracks=True  # 同时变换tracks
                # )
            elif self.reconstruction_type == 'each_pixel_feature_points':
                num_frames = self.min_images_for_scale
                height = self.inference_outputs[-1]['current_output']['pts3d'].shape[1]
                width = self.inference_outputs[-1]['current_output']['pts3d'].shape[2]

                conf_thres_value = 0.3
                max_points_for_colmap = 100000
                shared_camera = False
                camera_type = "PINHOLE"
                img_height, img_width = height, width
                image_size = np.array([height, width])

                latest_inference = self.inference_outputs[-1]
                latest_outputs = latest_inference['outputs']
                num_images = len(self.inference_outputs)
                num_outputs = len(latest_outputs)
                latest_start_idx = num_images - num_outputs
                latest_end_idx = num_images
                
                use_latest_outputs = (latest_start_idx == start_idx and latest_end_idx == end_idx)
                
                # 统一数据源，消除代码重复
                if use_latest_outputs:
                    outputs_list = latest_outputs
                    indices = list(range(start_idx, start_idx + len(latest_outputs)))
                else:
                    if self.verbose:
                        print(f"  Warning: Using outputs from different inference batches, points may be in different coordinate systems")
                    outputs_list = [self.inference_outputs[idx]['current_output'] for idx in range(start_idx, end_idx)]
                    indices = list(range(start_idx, end_idx))
                
                n = len(outputs_list)
                
                # ==================== GPU全流程优化 ====================
                # 核心思想: 尽可能在GPU上完成计算，减少CPU-GPU传输
                
                # 步骤1: 在GPU上stack所有tensor（零拷贝）
                pts3d_gpu = torch.stack([output['pts3d'][0] for output in outputs_list])  # (n, H, W, 3)
                conf_gpu = torch.stack([output['conf'][0] for output in outputs_list])    # (n, H, W)
                cam_gpu = torch.stack([output['camera_poses'][0] for output in outputs_list])  # (n, 4, 4)
                K_gpu = torch.stack([output['intrinsics'][0] for output in outputs_list])      # (n, 3, 3)
                
                device = pts3d_gpu.device
                use_gpu = pts3d_gpu.is_cuda
                
                # 步骤2: 在GPU上进行置信度过滤（避免传输全部数据到CPU）
                conf_mask_gpu = conf_gpu >= conf_thres_value  # (n, H, W)
                
                # 步骤3: GPU上随机采样（如果点数超过限制）
                true_count = conf_mask_gpu.sum().item()
                if true_count > max_points_for_colmap:
                    # 在GPU上高效随机采样
                    flat_mask = conf_mask_gpu.view(-1)
                    true_indices = torch.nonzero(flat_mask, as_tuple=True)[0]
                    # 随机选择 max_points_for_colmap 个索引
                    perm = torch.randperm(true_indices.size(0), device=device)[:max_points_for_colmap]
                    sampled_indices = true_indices[perm]
                    # 重建mask
                    combined_mask_gpu = torch.zeros_like(flat_mask)
                    combined_mask_gpu[sampled_indices] = True
                    combined_mask_gpu = combined_mask_gpu.view(conf_mask_gpu.shape)
                else:
                    combined_mask_gpu = conf_mask_gpu
                
                # 步骤4: 提取过滤后的3D点（仍在GPU上）
                points_3d_filtered_gpu = pts3d_gpu[combined_mask_gpu]  # (N_filtered, 3)
                
                # 步骤5: 在GPU上构建extrinsic（利用旋转矩阵正交性）
                R_gpu = cam_gpu[:, :3, :3]  # (n, 3, 3)
                t_gpu = cam_gpu[:, :3, 3:4]  # (n, 3, 1)
                R_inv_gpu = R_gpu.transpose(-1, -2)  # R^T
                t_inv_gpu = -torch.bmm(R_inv_gpu, t_gpu)  # -R^T @ t
                extrinsic_gpu = torch.cat([R_inv_gpu, t_inv_gpu], dim=2)  # (n, 3, 4)
                
                # 步骤6: 在GPU上执行投影（核心加速点）
                if use_gpu and points_3d_filtered_gpu.size(0) > 0:
                    projected_points2d_gpu, points_cam_gpu = project_3D_points(
                        points_3d_filtered_gpu,  # (N, 3)
                        extrinsic_gpu,           # (n, 3, 4)
                        K_gpu,                   # (n, 3, 3)
                    )
                    
                    # 步骤7: 在GPU上计算可见性mask
                    depths_gpu = points_cam_gpu[:, 2, :]  # (n, N)
                    proj_x_gpu = projected_points2d_gpu[:, :, 0]
                    proj_y_gpu = projected_points2d_gpu[:, :, 1]
                    
                    visible_mask_gpu = (
                        (depths_gpu > 0) &
                        (proj_x_gpu >= 0) & (proj_x_gpu < img_width) &
                        (proj_y_gpu >= 0) & (proj_y_gpu < img_height)
                    )  # (n, N)
                    
                    # 步骤8: 在GPU上筛选有效点
                    points_visible_count_gpu = visible_mask_gpu.sum(dim=0)  # (N,)
                    valid_points_mask_gpu = points_visible_count_gpu > 0
                    valid_point_indices_gpu = torch.nonzero(valid_points_mask_gpu, as_tuple=True)[0]
                    num_tracks = valid_point_indices_gpu.size(0)
                    
                    # 步骤9: 只传输需要的数据到CPU（大幅减少传输量）
                    if num_tracks > 0:
                        # 提取有效点的数据并传输到CPU
                        tracks_gpu = projected_points2d_gpu[:, valid_point_indices_gpu, :].float()
                        masks_gpu = visible_mask_gpu[:, valid_point_indices_gpu]
                        points3d_valid_gpu = points_3d_filtered_gpu[valid_point_indices_gpu]
                        
                        # 设置不可见位置为NaN（在GPU上完成）
                        tracks_gpu[~masks_gpu] = float('nan')
                        
                        # 一次性传输到CPU
                        tracks = tracks_gpu.cpu().numpy()
                        masks = masks_gpu.cpu().numpy()
                        all_points_3d = points_3d_filtered_gpu.cpu().numpy()
                        points3d_for_tracks = points3d_valid_gpu.double().cpu().numpy()
                        valid_point_indices = valid_point_indices_gpu.cpu().numpy()
                        
                        # extrinsic和intrinsic也传输（用于COLMAP）
                        extrinsic = extrinsic_gpu.cpu().numpy().astype(np.float32)
                        intrinsic = K_gpu.cpu().numpy().astype(np.float32)
                else:
                    # CPU fallback（数据不在GPU或无有效点）
                    pts3d_batch = pts3d_gpu.cpu().numpy()
                    conf_batch = conf_gpu.cpu().numpy()
                    cam_batch = cam_gpu.cpu().numpy()
                    K_batch = K_gpu.cpu().numpy()
                    
                    points_3d = pts3d_batch.astype(np.float32)
                    depth_conf = conf_batch.astype(np.float32)
                    intrinsic = K_batch.astype(np.float32)
                    
                    R_batch = cam_batch[:, :3, :3]
                    t_batch = cam_batch[:, :3, 3:4]
                    R_inv = np.transpose(R_batch, (0, 2, 1))
                    t_inv = -np.matmul(R_inv, t_batch)
                    extrinsic = np.concatenate([R_inv, t_inv], axis=2)
                    
                    conf_mask = depth_conf >= conf_thres_value
                    combined_mask = self.randomly_limit_trues(conf_mask, max_points_for_colmap)
                    points_3d_filtered = points_3d[combined_mask]
                    all_points_3d = points_3d_filtered
                    
                    if len(all_points_3d) > 0:
                        projected_points2d, points_cam = project_3D_points_np(
                            all_points_3d, extrinsic, intrinsic
                        )
                        depths = points_cam[:, 2, :]
                        proj_x = projected_points2d[:, :, 0]
                        proj_y = projected_points2d[:, :, 1]
                        visible_mask = (
                            (depths > 0) & (proj_x >= 0) & (proj_x < img_width) &
                            (proj_y >= 0) & (proj_y < img_height)
                        )
                        points_visible_count = visible_mask.sum(axis=0)
                        valid_point_indices = np.flatnonzero(points_visible_count > 0)
                        num_tracks = len(valid_point_indices)
                        
                        if num_tracks > 0:
                            tracks = projected_points2d[:, valid_point_indices, :].astype(np.float32, copy=True)
                            masks = visible_mask[:, valid_point_indices].copy()
                            tracks[~masks] = np.nan
                            points3d_for_tracks = all_points_3d[valid_point_indices].astype(np.float64)
                    else:
                        num_tracks = 0
                
                # ==================== 图像RGB处理 ====================
                # 处理图像和 original_coords
                if self.model_type == 'vggt':
                    vggt_images = []
                    original_coords = np.empty((n, 6), dtype=np.float32)
                    
                    for i, (output, idx) in enumerate(zip(outputs_list, indices)):
                        vggt_img = output.get('vggt_image', None)
                        if vggt_img is not None:
                            vggt_images.append(vggt_img)
                        elif idx < len(self.preprocessed_views):
                            img = self.preprocessed_views[idx]['img']
                            if img.dim() == 4 and img.shape[0] == 1:
                                img = img.squeeze(0)
                            vggt_images.append(img)
                        
                        scale_info = self.scale_info[idx]
                        ori_w, ori_h = scale_info['original_size']
                        original_coords[i] = [0, 0, ori_w, ori_h, ori_w, ori_h]
                    
                    if len(vggt_images) > 0:
                        vggt_images_tensor = torch.stack(vggt_images)
                        if vggt_images_tensor.shape[2:] != (height, width):
                            vggt_images_tensor = F.interpolate(
                                vggt_images_tensor, size=(height, width),
                                mode="bilinear", align_corners=False,
                            )
                        points_rgb_np = np.ascontiguousarray(
                            vggt_images_tensor.cpu().numpy().transpose(0, 2, 3, 1)
                        )
                        if points_rgb_np.max() <= 1.0:
                            points_rgb = (points_rgb_np * 255).astype(np.uint8)
                        else:
                            points_rgb = points_rgb_np.astype(np.uint8)
                    else:
                        raise ValueError("No VGGT images available for color extraction")
                else:
                    batch_images = []
                    original_coords = np.empty((n, 6), dtype=np.float32)
                    
                    for i, idx in enumerate(indices):
                        if idx < len(self.preprocessed_views):
                            img = self.preprocessed_views[idx]['img']
                            if img.dim() == 4 and img.shape[0] == 1:
                                img = img.squeeze(0)
                            batch_images.append(img)
                        
                        scale_info = self.scale_info[idx]
                        ori_w, ori_h = scale_info['original_size']
                        original_coords[i] = [0, 0, ori_w, ori_h, ori_w, ori_h]
                    
                    images = torch.stack(batch_images)
                    points_rgb_images = F.interpolate(
                        images, size=(height, width),
                        mode="bilinear", align_corners=False,
                    )
                    model = self._load_model()
                    points_rgb_float = rgb(points_rgb_images, model.encoder.data_norm_type)
                    points_rgb = (points_rgb_float * 255).astype(np.uint8)
                
                # ==================== 提取RGB颜色并构建COLMAP ====================
                if num_tracks == 0:
                    print("  Warning: No valid tracks found, skipping COLMAP conversion")
                    reconstruction = None
                    valid_track_mask = None
                else:
                    # 获取过滤点对应的RGB（使用combined_mask_gpu或combined_mask）
                    if use_gpu:
                        combined_mask_np = combined_mask_gpu.cpu().numpy()
                    else:
                        combined_mask_np = combined_mask
                    points_rgb_filtered = points_rgb[combined_mask_np]
                    points_rgb_for_tracks = points_rgb_filtered[valid_point_indices]

                    print("Converting to COLMAP format")
                    # 使用batch_np_matrix_to_pycolmap替代batch_np_matrix_to_pycolmap_wo_track
                    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                        points3d=points3d_for_tracks,  # (P, 3)
                        extrinsics=extrinsic,          # (N, 3, 4)
                        intrinsics=intrinsic,          # (N, 3, 3)
                        tracks=tracks,                 # (N, P, 2)
                        image_size=image_size,         # (2,)
                        masks=masks,                   # (N, P)
                        max_reproj_error=self.max_reproj_error,  # 重投影误差阈值
                        max_points3D_val=3000,
                        shared_camera=shared_camera,
                        camera_type=camera_type,
                        min_inlier_per_frame=self.min_inlier_per_frame,
                        points_rgb=points_rgb_for_tracks,  # (P, 3)
                    )

                if reconstruction is None:
                    print("  Warning: Failed to build pycolmap reconstruction")
                
                # 准备 image_paths 列表（文件名）
                image_paths_list = []
                for idx in range(start_idx, end_idx):
                    image_paths_list.append(self.image_paths[idx].name)
                
                # 获取预处理后的图像尺寸（img_size）
                proc_w = self.scale_info[start_idx]['output_size'][0]
                proc_h = self.scale_info[start_idx]['output_size'][1]
                
                # 重命名和缩放相机参数
                reconstruction = rename_colmap_recons_and_rescale_camera(
                    reconstruction=reconstruction,
                    image_paths=image_paths_list,
                    original_coords=original_coords,
                    img_size=(proc_w, proc_h),
                    shift_point2d_to_original_res=True,
                    shared_camera=shared_camera,
                )

                # 对齐到原始图像尺寸（基本对齐），对齐到已知的影像pose位置
                if self.global_sparse_reconstruction is not None and len(self.sfm_reconstructions) > 0:
                    reconstruction = rescale_reconstruction_to_original_size(
                        reconstruction=reconstruction,
                        ori_extrinsics=self.ori_extrinsic,
                        sfm_reconstructions=self.sfm_reconstructions,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        alignment_mode='pcl_alignment',
                        verbose=self.verbose,
                    )
                else:
                    reconstruction = rescale_reconstruction_to_original_size(
                        reconstruction=reconstruction,
                        ori_extrinsics=self.ori_extrinsic,
                        sfm_reconstructions=self.sfm_reconstructions,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        alignment_mode='image_alignment',
                        image_alignment_max_error=10.0,
                        image_alignment_min_inlier_ratio=0.3,
                    )

                # 保存重建结果（对齐到原始 SfM 后）
                temp_path = self.output_dir / "temp_aligned_to_original_sfm" / f"{start_idx}_{end_idx}"
                temp_path.mkdir(parents=True, exist_ok=True)
                reconstruction.write_text(str(temp_path))
                reconstruction.export_PLY(str(temp_path / "points3D.ply"))
                
                # 对齐到前一个重建 reconstruction：支持平移和缩放，不旋转，基于重叠影像的相机位置对齐。
                # 注意：对于 merge_method == 'confidence'，跳过此步骤，由 merge_by_confidence 一步到位完成更精确的对齐
                if self.merge_method == 'confidence':
                    # confidence 模式：跳过预对齐，直接使用 reconstruction
                    # merge_by_confidence 会通过 RANSAC + 3D点匹配 进行更精确的对齐
                    aligned_recon = reconstruction
                    if self.verbose:
                        print(f"  ✓ 重建已保存到: {temp_path}")
                        print(f"  (merge_method='confidence': 跳过预对齐，由 merge_by_confidence 一步到位)")
                elif len(self.inference_reconstructions) < 1:
                    # 第一个 batch，无需对齐
                    aligned_recon = reconstruction
                    if self.verbose:
                        print(f"  ✓ 重建已保存到: {temp_path}")
                else:
                    # 其他 merge_method：执行预对齐（基于重叠影像的相机位置）
                    prev_recon_data = self.inference_reconstructions[-1]  # 列表中最后一个
                    prev_recon = prev_recon_data['reconstruction']
                    aligned_recon = reconstruction  # 当前正在处理的，还未添加到列表
                    
                    # 获取重叠区域的影像ID（Reconstruction中影像ID是从1开始的）
                    # prev_recon中倒数overlap个影像
                    prev_overlap_image_ids = list(range(
                        len(prev_recon.images) - self.overlap + 1,
                        len(prev_recon.images) + 1
                    ))
                    # curr_recon中前overlap个影像
                    curr_overlap_image_ids = list(range(
                        1, self.overlap + 1
                    ))

                    # 从 prev_recon 和 aligned_recon 的重叠影像中提取相机位置
                    src_locations = []  # aligned_recon 的相机位置（源）
                    tgt_locations = []  # prev_recon 的相机位置（目标）
                    
                    for i, prev_img_id in enumerate(prev_overlap_image_ids):
                        curr_img_id = curr_overlap_image_ids[i]
                        
                        if prev_img_id in prev_recon.images and curr_img_id in aligned_recon.images:
                            # prev_recon 相机位置（目标）
                            prev_image = prev_recon.images[prev_img_id]
                            R_prev = np.array(prev_image.cam_from_world.rotation.matrix())
                            t_prev = np.array(prev_image.cam_from_world.translation)
                            camera_center_prev = -R_prev.T @ t_prev
                            tgt_locations.append(camera_center_prev)
                            
                            # aligned_recon 相机位置（源）
                            curr_image = aligned_recon.images[curr_img_id]
                            R_curr = np.array(curr_image.cam_from_world.rotation.matrix())
                            t_curr = np.array(curr_image.cam_from_world.translation)
                            camera_center_curr = -R_curr.T @ t_curr
                            src_locations.append(camera_center_curr)

                    if len(src_locations) >= 3:
                        # 有足够的点，估计 Sim3 变换
                        src_points = np.array(src_locations, dtype=np.float64)
                        tgt_points = np.array(tgt_locations, dtype=np.float64)
                        sim3d = estimate_sim3_transform(src_points, tgt_points)
                        if sim3d is not None:
                            # 只使用平移和缩放，不旋转
                            # 先缩放源点，再计算平移
                            scale = sim3d.scale
                            src_centroid = src_points.mean(axis=0)
                            tgt_centroid = tgt_points.mean(axis=0)
                            # 平移 = 目标质心 - 缩放后的源质心
                            translation = tgt_centroid - scale * src_centroid
                            
                            identity_rotation = pycolmap.Rotation3d(np.eye(3))
                            sim3_scale_translate = pycolmap.Sim3d(scale, identity_rotation, translation)
                            aligned_recon.transform(sim3_scale_translate)
                            
                            if self.verbose:
                                print(f"    Alignment: scale={scale:.4f}, translation={translation}")
                        else:
                            if self.verbose:
                                print("  ⚠ Sim3 estimation failed")
                    elif len(src_locations) >= 1:
                        # 点数不足，只使用平移对齐（不旋转不缩放）
                        src_points = np.array(src_locations, dtype=np.float64)
                        tgt_points = np.array(tgt_locations, dtype=np.float64)
                        # 计算质心平移
                        translation_only = tgt_points.mean(axis=0) - src_points.mean(axis=0)
                        identity_rotation = pycolmap.Rotation3d(np.eye(3))
                        sim3_translation_only = pycolmap.Sim3d(1.0, identity_rotation, translation_only)
                        aligned_recon.transform(sim3_translation_only)
                    else:
                        if self.verbose:
                            print(f"  ⚠ 没有有效的重叠影像，无法对齐")

                    # 保存对齐后的重建结果（对齐到前一个重建后）
                    temp_path = self.output_dir / "temp_aligned_to_prev_recon_overlay_image" / f"{start_idx}_{end_idx}"
                    temp_path.mkdir(parents=True, exist_ok=True)
                    aligned_recon.write_text(str(temp_path))
                    aligned_recon.export_PLY(str(temp_path / "points3D.ply"))
                    if self.verbose:
                        print(f"  ✓ 对齐后的重建已保存到: {temp_path}")

                # ==================== 提取逐像素3D点对应关系（缩放到原图尺寸）====================
                # 从 outputs_list 中提取逐像素的 pts3d，与 reconstruction 中的稀疏点不同，
                # 这里是密集的逐像素对应，可用于密集点云融合、语义投影等
                # 数据结构：{global_idx: {'pts3d': (H,W,3), 'conf': (H,W), 'valid_mask': (H,W)}}
                batch_pixel_3d_mapping = self._extract_pixel_to_3d_mapping_for_batch(
                    outputs_list=outputs_list,
                    global_indices=indices,
                    conf_threshold=1.0,  # 可调整置信度阈值
                    verbose=True
                )
                
                # 将 aligned_recon 添加到列表（包含该批次的 pixel_3d_mapping）
                # 注意：对于 merge_method='confidence'，aligned_recon 未经预对齐，由 merge_by_confidence 一步到位完成
                image_paths = [str(self.image_paths[idx]) for idx in range(start_idx, end_idx)]
                self.inference_reconstructions.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'image_paths': image_paths,
                    'reconstruction': aligned_recon,
                    'valid_track_mask': valid_track_mask,
                    'pixel_3d_mapping': batch_pixel_3d_mapping,  # 逐像素3D对应 {global_idx: {'pts3d', 'conf', 'valid_mask'}}，已缩放到原图尺寸
                })

                # ==================== 填充 recovered_inference_outputs 用于可视化 ====================
                # 使用 aligned_recon.points3D 中的整体稀疏点云，而不是每个影像单独的密集点云
                
                # 获取 scale_ratio（从最新的 inference_outputs 中获取）
                latest_inference_data = self.inference_outputs[-1]
                scale_ratio = latest_inference_data.get('scale_ratio', 1.0)
                if self.verbose:
                    print(f"  Using scale_ratio: {scale_ratio:.6f} for visualization")
                
                # ==================== 1. 从 aligned_recon 提取整体稀疏点云（增量添加）====================
                if len(aligned_recon.points3D) > 0:
                    num_points = len(aligned_recon.points3D)
                    unified_points = np.empty((num_points, 3), dtype=np.float32)
                    unified_colors = np.empty((num_points, 3), dtype=np.uint8)
                    
                    for i, (point3D_id, point3D) in enumerate(aligned_recon.points3D.items()):
                        unified_points[i] = point3D.xyz
                        unified_colors[i] = point3D.color  # RGB
                    
                    # 将当前 batch 的整体点云添加到可视化器
                    if self.visualizer is not None:
                        self.visualizer.add_batch_point_cloud(unified_points, unified_colors)
                    
                    if self.verbose:
                        batch_count = len(self.visualizer.unified_point_clouds) if self.visualizer else 0
                        print(f"  ✓ Added unified point cloud for batch {batch_count}: {num_points} points")
                
                # ==================== 2. 为每个图像存储相机位姿信息（用于显示 frustum）====================
                for local_idx, global_idx in enumerate(indices):
                    # 获取推理输出
                    output = outputs_list[local_idx]
                    
                    # 获取原图尺寸
                    real_w = int(original_coords[local_idx, 4])
                    real_h = int(original_coords[local_idx, 5])
                    
                    # 从 aligned_recon 获取对齐后的相机位姿
                    colmap_image_id = local_idx + 1  # COLMAP ID 从1开始
                    if colmap_image_id in aligned_recon.images:
                        pyimage = aligned_recon.images[colmap_image_id]
                        pycamera = aligned_recon.cameras[pyimage.camera_id]
                        
                        # 从 pycolmap 提取相机位姿 (cam_from_world 是 world2cam)
                        R_w2c = np.array(pyimage.cam_from_world.rotation.matrix())
                        t_w2c = np.array(pyimage.cam_from_world.translation)
                        
                        # 构建 world2cam 变换矩阵 (4x4)
                        T_cam_world_aligned = np.eye(4, dtype=np.float32)
                        T_cam_world_aligned[:3, :3] = R_w2c
                        T_cam_world_aligned[:3, 3] = t_w2c
                        
                        # 构建 cam2world 变换矩阵
                        T_world_cam_aligned = np.linalg.inv(T_cam_world_aligned).astype(np.float32)
                        
                        # 获取内参
                        K_aligned = pycamera.calibration_matrix().astype(np.float32)
                    else:
                        # 如果图像不在重建中，使用原始推理输出的位姿
                        T_world_cam_infer = output['camera_poses'][0].cpu().numpy()
                        T_world_cam_aligned = T_world_cam_infer.copy()
                        K_aligned = output['intrinsics'][0].cpu().numpy()
                    
                    # 创建 recovered_inference_output 字典（只包含相机信息，不包含点云）
                    device = output['pts3d'].device if hasattr(output['pts3d'], 'device') else 'cpu'
                    recovered_inference_output = {
                        'image_path': str(self.image_paths[global_idx]),
                        'image_width': real_w,
                        'image_height': real_h,
                        'camera_K': K_aligned,  # (3, 3)
                        'camera_poses': torch.from_numpy(T_world_cam_aligned).unsqueeze(0).float().to(device),  # (1, 4, 4)
                        'scale_ratio': scale_ratio,  # 保存尺度信息
                        # 注意：不再存储单独的 pts3d，使用整体点云代替
                    }
                    
                    # 添加到列表
                    self.recovered_inference_outputs.append(recovered_inference_output)
                
                if self.verbose:
                    print(f"  ✓ Added {len(indices)} images to recovered_inference_outputs for visualization")

                # ==================== 合并reconstruction中间结果 ====================
                merge_reconstruction_success = self._merge_reconstruction_intermediate_results()

            # Viser visualization
            if self.enable_visualization and self.visualizer is not None:
                self.visualizer.update(
                    recovered_inference_outputs=self.recovered_inference_outputs,
                    merged_reconstruction=self.merged_reconstruction,
                    input_views=self.input_views,
                    image_paths=self.image_paths,
                )
        
        if not success:
            print(f"Failed to process image: {image_path}")
            return False

        return True

    def _initialize_image(
        self, 
        image_path: Path, 
        gps_data: Tuple[float, float, float],
        enu_pos: np.ndarray,
        xmp_data: Dict
    ) -> bool:
        """Initialize and store intrinsic and extrinsic parameters for an image.
        
        This is a general function that can process any image.
        
        Args:
            image_path: Path to the image
            gps_data: GPS coordinates (lat, lon, alt)
            enu_pos: Position in ENU coordinates
            xmp_data: XMP metadata including gimbal pose and camera parameters
            
        Returns:
            True if successful, False otherwise
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Initializing image {len(self.ori_extrinsic) + 1}: {image_path.name}")
            print(f"{'='*60}")
        
        #  initialize the image
        # ================================================
        # Get image ID
        image_id = self.next_image_id
        self.next_image_id += 1
        image_name = image_path.name
        
        # Get image dimensions from xmp_data
        width_height = xmp_data.get("width_height", [0, 0])
        width, height = width_height[0], width_height[1]
        # Extract intrinsic parameters from XMP data
        dewarp_data = xmp_data.get("dewarp_data", [])
        # Extract camera parameters (fx, fy, cx, cy, k1, k2, p1, p2)
        params = dewarp_data[:8]
        # Construct intrinsics matrix
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Extract extrinsic parameters
        roll = xmp_data.get("roll")
        pitch = xmp_data.get("pitch")
        yaw = xmp_data.get("yaw")
        # Compute camera rotation matrix
        R_camera = cam_from_enu_transform(roll=roll, pitch=pitch, yaw=yaw)
        # Compute translation vector
        tvec = -R_camera @ enu_pos
        
        # Store intrinsic parameters
        intrinsic_info = {
            'image_id': image_id,
            'image_name': image_name,
            'image_path': str(image_path),
            'model': "OPENCV",
            'width': int(width),
            'height': int(height),
            'params': params,
            'K': K.tolist(),
        }
        self.ori_intrinsic.append(intrinsic_info)
        
        # Store extrinsic parameters
        extrinsic_info = {
            'image_id': image_id,
            'image_name': image_name,
            'image_path': str(image_path),
            'R_camera': R_camera.tolist(),  # Convert to list for JSON serialization
            'tvec': tvec.tolist(),  # Convert to list for JSON serialization
            'gps': gps_data,
            'enu': enu_pos,
        }
        self.ori_extrinsic.append(extrinsic_info)

        # Preprocess the image
        # ================================================
        # Create input view
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image).astype(np.uint8)
        image_tensor = torch.from_numpy(image_array)  # [H, W, 3]

        # Construct pose matrix (cam2world)
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = R_camera.T  # world to camera is R_camera, so cam to world is R_camera.T
        pose_matrix[:3, 3] = -R_camera.T @ tvec  # camera position in world coordinates
        pose_tensor = torch.from_numpy(pose_matrix)

        # Create view dict (similar to load_colmap_data)
        input_view = {
            # 'image_id': image_id,
            # 'image_name': image_name,
            # 'image_path': str(image_path),
            'img': image_tensor,  # [H, W, 3], uint8  # (H, W, 3) - [0, 255]
            'intrinsics': torch.from_numpy(K),  # [3, 3]
            'camera_poses': pose_tensor,  # [4, 4] in OpenCV cam2world convention
            'is_metric_scale': torch.tensor([False]),  # COLMAP data is non-metric
        }

        self.input_views.append(input_view)

        # Preprocess this single view
        preprocessed_view = preprocess_inputs(
            [input_view],
            resize_mode="fixed_mapping",
            resolution_set=518,
            verbose=False
        )[0]  # Get the single preprocessed view

        # Store preprocessed view
        self.preprocessed_views.append(preprocessed_view)

        # Calculate scale info
        orig_h, orig_w = image_array.shape[0], image_array.shape[1]
        proc_h, proc_w = preprocessed_view['img'].shape[2], preprocessed_view['img'].shape[3]
        
        scale_info = {
            'image_id': image_id,
            'image_name': image_name,
            'original_size': (orig_w, orig_h),
            'output_size': (proc_w, proc_h),
            'scale_x': proc_w / orig_w,
            'scale_y': proc_h / orig_h,
        }
        self.scale_info.append(scale_info)

        if self.verbose:
            print(f"  Original size: {orig_w}x{orig_h}")
            print(f"  Preprocessed size: {proc_w}x{proc_h}")
            print(f"  Scale: x={scale_info['scale_x']:.4f}, y={scale_info['scale_y']:.4f}")
            print(f"  Intrinsics stored: {width}x{height}, fx={params[0]:.2f}, fy={params[1]:.2f}")
            print(f"  Extrinsics stored:")
            print(f"    ENU position: [{enu_pos[0]:.2f}, {enu_pos[1]:.2f}, {enu_pos[2]:.2f}]")
            print(f"    Roll/Pitch/Yaw: [{roll:.2f}, {pitch:.2f}, {yaw:.2f}]")
        
        if self.verbose:
            print(f"✓ Image initialized: {image_name} (ID: {image_id})")
        
        return True

    def _run_inference(self, image_path: Path, preprocessed_view: Dict) -> bool:
        """Run inference on the image.
        
        Args:
            image_path: Path to the image
            preprocessed_view: Single preprocessed view (will be ignored, we'll use stored views)
        
        Returns:
            True if successful, False otherwise
        """
        # Load model
        model = self._load_model()

        # 判断是第几张图像
        num_images = len(self.preprocessed_views)

        if self.model_type == 'mapanything':
            outputs = self._run_mapanything_inference(model, num_images)
        elif self.model_type == 'vggt':
            outputs = self._run_vggt_inference(model, num_images)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # ==================== 计算 scale_ratio =====================
        scale_ratio = 1.0
        if num_images >= 2:  # 从第2张开始计算scale
            # 确定参与scale计算的图像数量
            num_infer = min(num_images, self.min_images_for_scale)
            
            # 提取所有参与推理的图像的原始相机位置
            orig_positions = []
            for i in range(-num_infer, 0):  # -num_infer, -num_infer+1, ..., -1
                ext = self.ori_extrinsic[i]
                R_cam = np.array(ext['R_camera'])
                t_cam = np.array(ext['tvec'])
                cam_pos = -R_cam.T @ t_cam
                orig_positions.append(cam_pos)
            
            # 转换为 torch tensor (N, 3)
            orig_positions = torch.from_numpy(np.stack(orig_positions)).float()
            
            # 提取推理的相机位置（在同一坐标系中）
            infer_positions = torch.stack([
                outputs[i]['camera_poses'][0, :3, 3].cpu()
                for i in range(len(outputs))
            ])  # (N, 3)
            
            # 使用 torch.cdist 计算距离矩阵
            orig_dists = torch.cdist(orig_positions, orig_positions)  # (N, N)
            infer_dists = torch.cdist(infer_positions, infer_positions)  # (N, N)
            
            # 获取有效的非零距离对（参考 demo_inference_on_colmap_outputs.py）
            valid_mask = orig_dists > 1e-6
            
            if valid_mask.sum() > 0:
                # 计算缩放比例 = 真实距离 / 推理距离（使用中位数）
                scale_ratio = (orig_dists[valid_mask] / (infer_dists[valid_mask] + 1e-8)).median().item()
                
                if self.verbose:
                    print(f"  ================================ Computed Scale Ratio (COLMAP/{self.model_type}): {scale_ratio:.6f}")
                    print(f"  Based on {valid_mask.sum().item()} camera pair distances from {num_infer} images")
            else:
                if self.verbose:
                    print("  Warning: Could not compute scale ratio (insufficient camera movement)")
        else:
            if self.verbose:
                print(f"  No scale calculation (first image)")

        if self.verbose:
            print("✓ Inference completed!")

        # 从outputs中提取预测的尺度比例
        predicted_scale_ratio = outputs[-1].get('metric_scaling_factor', torch.tensor(1.0))
        if isinstance(predicted_scale_ratio, torch.Tensor):
            predicted_scale_ratio = predicted_scale_ratio.item()
        
        # 只存储当前图像（最后一张）的输出
        current_output = outputs[-1] if num_images >= 2 else outputs[0]

        # Store inference outputs
        inference_outputs = {
            'image_path': str(image_path),
            'current_output': current_output,
            'outputs': outputs,
            'scale_ratio': scale_ratio,
            'predicted_scale_ratio': predicted_scale_ratio,
        }
        self.inference_outputs.append(inference_outputs)

        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    def _run_mapanything_inference(self, model, num_images: int) -> List[Dict]:
        """Run MapAnything model inference.
        
        Args:
            model: Loaded MapAnything model
            num_images: Number of images to process
            
        Returns:
            List of output dictionaries in unified format
        """
        if self.verbose:
            print("Running MapAnything inference...")

        if num_images == 1:
            # 第一张图像，只推理单张
            views_to_infer = [self.preprocessed_views[0]]
            if self.verbose:
                print("  Inferring first image (no scale calculation)")
        elif num_images < self.min_images_for_scale:
            # 图像数量不足 min_images_for_scale，推理从第一张到当前张的所有图像
            views_to_infer = self.preprocessed_views[:]  # 所有图像
            if self.verbose:
                print(f"  Inferring images 1 to {num_images} ({num_images} images, building up to {self.min_images_for_scale})")
        else:
            # 图像数量已达到 min_images_for_scale，使用滑动窗口
            views_to_infer = self.preprocessed_views[-self.min_images_for_scale:]
            start_idx = num_images - self.min_images_for_scale + 1
            if self.verbose:
                print(f"  Inferring images {start_idx} to {num_images} ({self.min_images_for_scale} images, sliding window)")

        outputs = model.infer(
            views_to_infer,
            memory_efficient_inference=False,
            ignore_calibration_inputs=False,
            ignore_depth_inputs=True,
            ignore_pose_inputs=False,
            ignore_depth_scale_inputs=True,
            ignore_pose_scale_inputs=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )
        
        return outputs

    def _run_vggt_inference(self, model, num_images: int) -> List[Dict]:
        """Run VGGT model inference.
        
        Args:
            model: Loaded VGGT model
            num_images: Number of images to process
            
        Returns:
            List of output dictionaries in unified format (compatible with MapAnything output)
        """
        if self.verbose:
            print("Running VGGT inference...")

        # Determine which images to infer
        if num_images == 1:
            image_paths_to_infer = [self.image_paths[0]]
            view_indices = [0]
            if self.verbose:
                print("  Inferring first image (no scale calculation)")
        elif num_images < self.min_images_for_scale:
            image_paths_to_infer = self.image_paths[:]
            view_indices = list(range(num_images))
            if self.verbose:
                print(f"  Inferring images 1 to {num_images} ({num_images} images, building up to {self.min_images_for_scale})")
        else:
            image_paths_to_infer = self.image_paths[-self.min_images_for_scale:]
            view_indices = list(range(num_images - self.min_images_for_scale, num_images))
            start_idx = num_images - self.min_images_for_scale + 1
            if self.verbose:
                print(f"  Inferring images {start_idx} to {num_images} ({self.min_images_for_scale} images, sliding window)")

        # Preprocess images for VGGT
        image_paths_str = [str(p) for p in image_paths_to_infer]
        images = load_and_preprocess_images(image_paths_str).to(self.device)
        
        # Get image size for pose decoding
        _, _, H, W = images.shape  # [S, 3, H, W]
        image_size_hw = (H, W)

        # Run inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = model(images)

        # Convert VGGT output to unified format
        # VGGT outputs: world_points [B, S, H, W, 3], pose_enc [B, S, 9], etc.
        outputs = self._convert_vggt_output_to_unified_format(
            predictions, 
            image_size_hw, 
            view_indices,
            images  # 传递原始图像用于颜色提取
        )
        
        return outputs

    def _convert_vggt_output_to_unified_format(
        self, 
        predictions: Dict, 
        image_size_hw: Tuple[int, int],
        view_indices: List[int],
        images: torch.Tensor = None  # VGGT 预处理后的图像 [S, 3, H, W]
    ) -> List[Dict]:
        """Convert VGGT predictions to unified output format compatible with MapAnything.
        
        Args:
            predictions: VGGT model predictions
            image_size_hw: Tuple of (height, width) of preprocessed images
            view_indices: List of view indices in original image list
            images: VGGT preprocessed images tensor [S, 3, H, W] for color extraction
            
        Returns:
            List of dictionaries with unified output format:
                - pts3d: [1, H, W, 3] - 3D points
                - conf: [1, H, W] - confidence
                - camera_poses: [1, 4, 4] - camera pose (cam2world)
                - intrinsics: [1, 3, 3] - camera intrinsics
                - metric_scaling_factor: scalar - metric scaling factor
                - vggt_image: [3, H, W] - VGGT preprocessed image for color (VGGT only)
        """
        # Extract predictions
        world_points = predictions['world_points']  # [B, S, H, W, 3]
        world_points_conf = predictions['world_points_conf']  # [B, S, H, W]
        pose_enc = predictions['pose_enc']  # [B, S, 9]
        
        # Convert pose encoding to extrinsics and intrinsics
        # extrinsics: [B, S, 3, 4] (cam from world)
        # intrinsics: [B, S, 3, 3]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, 
            image_size_hw=image_size_hw,
            pose_encoding_type="absT_quaR_FoV",
            build_intrinsics=True
        )
        
        # Convert extrinsics (cam from world) to camera poses (cam2world / world from cam)
        # cam2world = inverse of extrinsics
        B, S = extrinsics.shape[:2]
        camera_poses = torch.zeros(B, S, 4, 4, device=extrinsics.device, dtype=extrinsics.dtype)
        
        for b in range(B):
            for s in range(S):
                # extrinsics is [R|t] where camera_coords = R @ world_coords + t
                # cam2world: world_coords = R.T @ (camera_coords - t) = R.T @ camera_coords - R.T @ t
                R = extrinsics[b, s, :3, :3]
                t = extrinsics[b, s, :3, 3]
                
                R_inv = R.T
                t_inv = -R_inv @ t
                
                camera_poses[b, s, :3, :3] = R_inv
                camera_poses[b, s, :3, 3] = t_inv
                camera_poses[b, s, 3, 3] = 1.0
        
        # Build unified output list
        outputs = []
        for s in range(S):
            output = {
                'pts3d': world_points[:, s, :, :, :],  # [B, H, W, 3] -> keep B dim for consistency
                'conf': world_points_conf[:, s, :, :],  # [B, H, W]
                'camera_poses': camera_poses[:, s, :, :],  # [B, 4, 4]
                'intrinsics': intrinsics[:, s, :, :],  # [B, 3, 3]
                'metric_scaling_factor': torch.tensor(1.0),  # VGGT doesn't output this, use 1.0
                # Additional VGGT-specific outputs
                'depth': predictions.get('depth', None),
                'depth_conf': predictions.get('depth_conf', None),
                'view_index': view_indices[s],
                # 保存 VGGT 预处理后的图像用于颜色提取
                'vggt_image': images[s] if images is not None else None,  # [3, H, W]
            }
            outputs.append(output)
        
        return outputs


    def _batch_recover_original_poses(self, image_path: Path, start_idx: int, end_idx: int, transform_tracks: bool = True) -> bool:
        """Batch recover original poses for a range of images.
        
        Args:
            image_path: Path to the image
            start_idx: Starting index in self.inference_outputs
            end_idx: Ending index (exclusive) in self.inference_outputs
            transform_tracks: Whether to also transform the tracks' 3D points
        
        Returns:
            True if successful, False otherwise
        """
        if self.verbose:
            print(f"  Recovering poses for {end_idx - start_idx} images (indices {start_idx} to {end_idx-1})")
        
        # 计算这批图像的中位数尺度（用于恢复）
        all_scale_ratios = [data['scale_ratio'] for data in self.inference_outputs[:end_idx]]
        all_predicted_scale_ratios = [data['predicted_scale_ratio'] for data in self.inference_outputs[:end_idx]]
        
        # 最新尺度
        scale_ratio = all_scale_ratios[-1]
        predicted_scale_ratio = all_predicted_scale_ratios[-1]
        
        # 对这批图像逐一恢复
        for image_idx in range(start_idx, end_idx):
            success = self._recover_single_image_pose(image_path, image_idx, scale_ratio, predicted_scale_ratio)
            if not success:
                print(f"Failed to recover pose for image index {image_idx}")
                return False
        
        # ========== 新增：变换 tracks 的 3D 点到真实坐标系 ==========
        if transform_tracks and len(self.batch_tracks) > 0:
            success = self._transform_batch_tracks_to_real_coords(
                start_idx, end_idx, scale_ratio
            )
            if not success:
                print(f"Failed to transform tracks to real coordinates")
                return False

        return True

    def _recover_single_image_pose(self, image_path: Path, image_idx: int, scale_ratio: float, predicted_scale_ratio: float) -> bool:
        """Recover original pose for the current image.
        当前的三维点云是没有恢复尺度的，只存了最新的尺度，需要之后基于最新的尺度恢复所有尺度的三维点云。

        Args:
            image_path: Path to the image
            idx: Index of the image
            scale_ratio: Scale ratio
            predicted_scale_ratio: Predicted scale ratio

        Returns:
            True if successful, False otherwise
        """
        inference_data = self.inference_outputs[-1]
        inference_output = inference_data['current_output']
        image_path = Path(inference_data['image_path'])
        ori_extrinsic = self.ori_extrinsic[image_idx]
        ori_intrinsic = self.ori_intrinsic[image_idx]

        # Get original pose (cam2world)
        R_camera_orig = np.array(ori_extrinsic['R_camera']) # (w2c)
        tvec_orig = np.array(ori_extrinsic['tvec']) # (w2c)
        # Construct original cam2world transformation matrix
        T_world_cam_orig = np.eye(4, dtype=np.float32)
        T_world_cam_orig[:3, :3] = R_camera_orig.T  # cam to world
        T_world_cam_orig[:3, 3] = -R_camera_orig.T @ tvec_orig  # camera position in world
        
        # inference_output['camera_poses'] is a tensor of shape (B, 4, 4)
        T_world_cam_infer = inference_output['camera_poses'][0].cpu().numpy()  # (4, 4)
        # Compute transformation from inference world to original world
        # T_orig_infer = T_world_cam_orig @ inv(T_world_cam_infer)
        T_cam_world_infer = np.linalg.inv(T_world_cam_infer)
        T_transform = T_world_cam_orig @ T_cam_world_infer

        # Create recovered output dictionary
        recovered_inference_output = {
            'image_path': str(image_path),
            'image_width': ori_intrinsic['width'],
            'image_height': ori_intrinsic['height'],
            'camera_K': ori_intrinsic['K'],
            'transformation_matrix': T_transform,
            'scale_ratio': scale_ratio, # 最新的尺度
            'predicted_scale_ratio': predicted_scale_ratio, # 预测的尺度
            'conf': inference_output['conf'], # 置信度
        }
        
        # ==================== Transform pts3d =====================
        # Transform pts3d (world coordinates)
        pts3d_infer = inference_output['pts3d']  # (1, H, W, 3) on cuda
        # Convert to numpy and get shape
        pts3d_np = pts3d_infer[0].cpu().numpy()  # (H, W, 3)
        H, W, _ = pts3d_np.shape
        # Reshape to (H*W, 3) for transformation
        pts_flat = pts3d_np.reshape(-1, 3)
        # Apply scale ratio
        pts_flat_scaled = pts_flat * scale_ratio # 应用缩放比例
        # Convert to homogeneous coordinates (H*W, 4)
        pts_homo = np.concatenate([pts_flat_scaled, np.ones((pts_flat_scaled.shape[0], 1), dtype=np.float32)], axis=1)
        # Apply transformation (from inference world to original world)
        pts_recovered_homo = (T_transform @ pts_homo.T).T  # (H*W, 4)
        # Convert back to 3D (H*W, 3)
        pts_recovered = pts_recovered_homo[:, :3]
        # Reshape back to (H, W, 3)
        pts_recovered = pts_recovered.reshape(H, W, 3)
        # Store as tensor on the same device
        device = inference_output['pts3d'].device
        recovered_inference_output['pts3d'] = torch.from_numpy(pts_recovered).unsqueeze(0).to(device)
    
        # ==================== Transform pts3d_cam =====================
        # Transform pts3d_cam (camera coordinates)
        pts3d_cam_infer = inference_output['pts3d_cam']  # (1, H, W, 3) on cuda
        # Convert to numpy
        pts3d_cam_np = pts3d_cam_infer[0].cpu().numpy()  # (H, W, 3)
        H, W, _ = pts3d_cam_np.shape
        # Reshape to (H*W, 3)
        pts_cam_flat = pts3d_cam_np.reshape(-1, 3)
        # Apply scale ratio
        pts_cam_flat_scaled = pts_cam_flat * scale_ratio # 应用缩放比例
        # Convert to homogeneous coordinates
        pts_cam_homo = np.concatenate([pts_cam_flat_scaled, np.ones((pts_cam_flat_scaled.shape[0], 1), dtype=np.float32)], axis=1)
        # Transform from inference camera frame to original camera frame
        # T_cam_orig_cam_infer = inv(T_world_cam_orig) @ T_world_cam_infer
        T_cam_orig_world = np.linalg.inv(T_world_cam_orig)
        T_cam_orig_cam_infer = T_cam_orig_world @ T_world_cam_infer
        pts_cam_recovered_homo = (T_cam_orig_cam_infer @ pts_cam_homo.T).T
        # Convert back to 3D
        pts_cam_recovered = pts_cam_recovered_homo[:, :3]
        # Reshape back to (H, W, 3)
        pts_cam_recovered = pts_cam_recovered.reshape(H, W, 3)
        # Store as tensor
        device = inference_output['pts3d_cam'].device
        recovered_inference_output['pts3d_cam'] = torch.from_numpy(pts_cam_recovered).unsqueeze(0).to(device)
    
        # ==================== Update camera pose to original pose =====================
        # Update camera position to original position
        cam_pos_orig = T_world_cam_orig[:3, 3]
        device = inference_output['cam_trans'].device
        recovered_inference_output['cam_trans'] = torch.from_numpy(cam_pos_orig).unsqueeze(0).to(device)

        # Camera rotation in original world coordinates
        R_recovered = T_world_cam_orig[:3, :3]
        quat_recovered = R.from_matrix(R_recovered).as_quat()  # [x, y, z, w]
        device = inference_output['cam_quats'].device
        recovered_inference_output['cam_quats'] = torch.from_numpy(quat_recovered).unsqueeze(0).float().to(device)
    
        # Camera Pose to original pose
        device = inference_output['camera_poses'].device
        recovered_inference_output['camera_poses'] = torch.from_numpy(T_world_cam_orig).unsqueeze(0).to(device)

        # Store recovered inference outputs
        self.recovered_inference_outputs.append(recovered_inference_output)

        if self.verbose:
            print(f"✓ Pose recovered for image: {ori_extrinsic['image_name']}")
            print(f"  Original camera position (ENU): [{ori_extrinsic['enu'][0]:.2f}, {ori_extrinsic['enu'][1]:.2f}, {ori_extrinsic['enu'][2]:.2f}]")
            if 'pts3d' in recovered_inference_output:
                pts_shape = recovered_inference_output['pts3d'].shape
                print(f"  pts3d transformed: shape {pts_shape}")
            if 'pts3d_cam' in recovered_inference_output:
                pts_cam_shape = recovered_inference_output['pts3d_cam'].shape
                print(f"  pts3d_cam transformed: shape {pts_cam_shape}")
            print(f"  Camera pose updated to original coordinates")

        return True

    def _transform_batch_tracks_to_real_coords(self, start_idx: int, end_idx: int, scale_ratio: float) -> bool:
        """Transform batch tracks' 3D points from inference to real coordinate system.
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            scale_ratio: Scale ratio to apply
            
        Returns:
            True if successful
        """
        try:
            if self.verbose:
                print(f"  Transforming tracks 3D points to real coordinate system...")
            
            # 获取最新的 batch_tracks
            if len(self.batch_tracks) == 0:
                return True
            
            latest_batch = self.batch_tracks[-1]
            points_3d = latest_batch['points_3d']  # (P, 3) - 推理坐标系
            
            if points_3d is None:
                return True
            
            # 使用第一张图像的变换矩阵作为参考
            # （因为所有图像的 world 坐标系变换是一致的）
            inference_data = self.inference_outputs[start_idx]
            inference_output = inference_data['current_output']
            ori_extrinsic = self.ori_extrinsic[start_idx]
            
            # 获取变换矩阵
            R_camera_orig = np.array(ori_extrinsic['R_camera'])
            tvec_orig = np.array(ori_extrinsic['tvec'])
            T_world_cam_orig = np.eye(4, dtype=np.float32)
            T_world_cam_orig[:3, :3] = R_camera_orig.T
            T_world_cam_orig[:3, 3] = -R_camera_orig.T @ tvec_orig
            
            T_world_cam_infer = inference_output['camera_poses'][0].cpu().numpy()
            T_cam_world_infer = np.linalg.inv(T_world_cam_infer)
            T_transform = T_world_cam_orig @ T_cam_world_infer
            
            # 应用尺度和变换
            points_3d_scaled = points_3d * scale_ratio  # (P, 3)
            points_3d_homo = np.concatenate([
                points_3d_scaled, 
                np.ones((points_3d_scaled.shape[0], 1), dtype=np.float32)
            ], axis=1)  # (P, 4)
            
            points_3d_transformed = (T_transform @ points_3d_homo.T).T[:, :3]  # (P, 3)
            
            # 更新 batch_tracks 中的 3D 点
            latest_batch['points_3d'] = points_3d_transformed
            latest_batch['points_3d_transformed'] = True  # 标记已变换
            
            if self.verbose:
                print(f"  ✓ Transformed {points_3d.shape[0]} 3D points to real coordinate system")
                print(f"    Scale ratio applied: {scale_ratio:.6f}")
            
            return True
            
        except Exception as e:
            print(f"  Error transforming tracks: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _predict_tracks_for_batch(self, start_idx: int, end_idx: int) -> bool:
        """Predict tracks for a batch of images.
        
        Args:
            start_idx: Starting index in self.inference_outputs (未恢复的起始索引)
            end_idx: Ending index (exclusive) in self.inference_outputs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.verbose:
                print(f"\n  Predicting tracks for batch (indices {start_idx} to {end_idx-1})...")
            
            # 检查最新的 outputs 是否正好覆盖我们需要的范围
            latest_inference = self.inference_outputs[-1]
            latest_outputs = latest_inference['outputs']
            num_images = len(self.inference_outputs)
            num_outputs = len(latest_outputs)
            latest_start_idx = num_images - num_outputs
            latest_end_idx = num_images

            use_latest_outputs = (latest_start_idx == start_idx and latest_end_idx == end_idx)

            # 准备这批图像的数据
            batch_image_paths = []
            batch_images = []
            batch_confs = []
            batch_points_3d = []

            if use_latest_outputs:
                # 直接从最新的outputs列表中提取数据
                for i, output in enumerate(latest_outputs):
                    idx = start_idx + i      
                    # 收集图像路径
                    batch_image_paths.append(self.inference_outputs[idx]['image_path'])
                    
                    # 从 preprocessed_views 获取预处理后的图像（与推理输出尺寸匹配）
                    if idx < len(self.preprocessed_views):
                        img = self.preprocessed_views[idx]['img']  # [1, 3, H, W] tensor
                        # 去掉批次维度，得到 [3, H, W]
                        if img.dim() == 4 and img.shape[0] == 1:
                            img = img.squeeze(0)  # [3, H, W]
                        batch_images.append(img)

                    # 从 outputs[i] 中直接获取3D点（未恢复尺度的）
                    if 'pts3d' in output:
                        pts3d = output['pts3d']  # (1, H, W, 3) tensor
                        batch_points_3d.append(pts3d[0])

                    # 获取置信度
                    if 'conf' in output:
                        conf = output['conf']  # (1, H, W)
                        batch_confs.append(conf[0])
            else:
                # 范围不匹配，从各个 inference_outputs[idx]['current_output'] 获取
                for idx in range(start_idx, end_idx):
                    # 从 inference_outputs 获取推理输出（而不是 recovered_inference_outputs）
                    inference_data = self.inference_outputs[idx]
                    inference_output = inference_data['current_output']
                    # 收集图像路径
                    batch_image_paths.append(inference_data['image_path'])
                    
                    # 从 preprocessed_views 获取预处理后的图像（与推理输出尺寸匹配）
                    if idx < len(self.preprocessed_views):
                        img = self.preprocessed_views[idx]['img']  # [1, 3, H, W] tensor
                        # 去掉批次维度，得到 [3, H, W]
                        if img.dim() == 4 and img.shape[0] == 1:
                            img = img.squeeze(0)  # [3, H, W]
                        batch_images.append(img)
                    
                    # 从 inference_outputs 获取3D点（未恢复尺度的）
                    if 'pts3d' in inference_output:
                        pts3d = inference_output['pts3d']  # (1, H, W, 3) tensor
                        batch_points_3d.append(pts3d[0])
                    
                    # 获取置信度
                    if 'conf' in inference_output:
                        conf = inference_output['conf']  # (1, H, W)
                        batch_confs.append(conf[0])
            
            # 将列表转换为tensor
            if len(batch_images) > 0:
                # batch_images 中每个元素是 [3, H, W]
                batch_images_tensor = torch.stack(batch_images)  # (B, 3, H, W)
                
                # 确保在 CUDA 上
                if not batch_images_tensor.is_cuda:
                    batch_images_tensor = batch_images_tensor.cuda()
                
                # 准备置信度和3D点 - 转为 numpy 数组再转回 tensor（断开梯度）
                # 这样可以确保不携带计算图
                if batch_confs:
                    batch_confs_np = torch.stack(batch_confs).detach().cpu().numpy()
                    # 转回 tensor（不带梯度）并放到 CUDA
                    batch_confs_tensor = torch.from_numpy(batch_confs_np).cuda()
                else:
                    batch_confs_tensor = None
                    
                if batch_points_3d:
                    batch_points_3d_np = torch.stack(batch_points_3d).detach().cpu().numpy()
                    # 转回 tensor（不带梯度）并放到 CUDA
                    batch_points_3d_tensor = torch.from_numpy(batch_points_3d_np).cuda()
                else:
                    batch_points_3d_tensor = None
                
                # # 【关键修改】准备置信度和3D点 - 直接使用 numpy 数组，不转回 CUDA tensor
                # if batch_confs:
                #     batch_confs_np = torch.stack(batch_confs).detach().cpu().numpy()
                #     batch_confs_tensor = batch_confs_np  # 直接使用 numpy，不转回 CUDA
                # else:
                #     batch_confs_tensor = None
                    
                # if batch_points_3d:
                #     batch_points_3d_np = torch.stack(batch_points_3d).detach().cpu().numpy()
                #     batch_points_3d_tensor = batch_points_3d_np  # 直接使用 numpy，不转回 CUDA
                # else:
                #     batch_points_3d_tensor = None

                # 准备原始图像列表
                batch_original_images = []
                for idx in range(start_idx, end_idx):
                    if idx < len(self.input_views):
                        # input_views 中存储的是原始图像 tensor [H, W, 3], uint8
                        img_orig = self.input_views[idx]['img'].cpu().numpy()
                        batch_original_images.append(img_orig)

                # 运行特征点跟踪
                # 传入不带梯度的 tensor
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                            batch_images_tensor,
                            conf=batch_confs_tensor,
                            points_3d=batch_points_3d_tensor,
                            max_query_pts=4096,
                            query_frame_num=3,
                            keypoint_extractor="aliked+sp+sift",
                            fine_tracking=True,
                            original_images=batch_original_images,
                        )
                
                self.batch_tracks.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'image_indices': list(range(start_idx, end_idx)),
                    'image_paths': batch_image_paths,
                    'pred_tracks': pred_tracks,  # 已经是 numpy array
                    'pred_vis_scores': pred_vis_scores,  # 已经是 numpy array
                    'pred_confs': pred_confs,  # numpy array 或 None
                    'points_3d': points_3d,  # numpy array 或 None
                    'points_rgb': points_rgb,  # numpy array 或 None
                })
                
                num_frames = len(batch_image_paths)
                for frame_idx in range(num_frames):
                    image_idx = start_idx + frame_idx
                    
                    # 为每个影像创建跟踪信息
                    image_track_info = {
                        'image_idx': image_idx,
                        'image_path': batch_image_paths[frame_idx],
                        # 当前帧的track坐标（2D）
                        'tracks_2d': pred_tracks[frame_idx],# 该帧上的所有跟踪点 (num_points, 2)
                        'vis_scores': pred_vis_scores[frame_idx],  # 该帧上的可见性分数 (num_points,)
                        # 匹配信息
                        'matched_images_indices': list(range(start_idx, end_idx)),  # 与哪些影像匹配（索引）
                        'matched_images_paths': batch_image_paths,  # 与哪些影像匹配（路径）
                        'num_tracks': pred_tracks.shape[1],  # 跟踪点数量
                    }
                    
                    # 如果有置信度和3D点信息，也可以添加
                    # 注意：pred_confs 和 points_3d 是按查询帧组织的，可能需要特殊处理
                    if pred_confs is not None:
                        image_track_info['confs'] = pred_confs
                    if points_3d is not None:
                        image_track_info['points_3d'] = points_3d
                    if points_rgb is not None:
                        image_track_info['points_rgb'] = points_rgb
                    
                    self.image_tracks.append(image_track_info)

                if self.verbose:
                    print(f"  ✓ Track prediction completed")
                    print(f"    Tracks shape: {pred_tracks.shape}")
                    print(f"    Number of tracked points: {pred_tracks.shape[1]}")
                
                torch.cuda.empty_cache()
                return True
            else:
                print("  Warning: No images available for track prediction")
                return False
                
        except Exception as e:
            print(f"  Error during track prediction: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _build_pycolmap_reconstruction(self, start_idx: int, end_idx: int, use_recovered: bool = False) -> bool:
        """Build pycolmap reconstruction from predicted tracks.
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (exclusive)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.verbose:
                print(f"\n  Building pycolmap reconstruction for images {start_idx} to {end_idx-1}...")
            
            # 获取最新的batch_tracks
            latest_batch = self.batch_tracks[-1]
            
            # 准备数据
            image_paths = latest_batch['image_paths']
            pred_tracks = latest_batch['pred_tracks']  # (N, P, 2)
            pred_vis_scores = latest_batch['pred_vis_scores']  # (N, P)
            points_3d = latest_batch['points_3d']  # (P, 3) 或 None
            points_rgb = latest_batch['points_rgb']  # (P, 3) 或 None
            
            # 检查是否可以使用最新的 outputs（与 _predict_tracks_for_batch 保持一致）
            use_latest_outputs = False
            if not use_recovered:  # 只在不使用恢复位姿时才优化
                latest_inference = self.inference_outputs[-1]
                latest_outputs = latest_inference['outputs']
                num_images = len(self.inference_outputs)
                num_outputs = len(latest_outputs)
                latest_start_idx = num_images - num_outputs
                latest_end_idx = num_images
                use_latest_outputs = (latest_start_idx == start_idx and latest_end_idx == end_idx)
                
                if self.verbose and use_latest_outputs:
                    print(f"  ✓ Using latest inference outputs for camera parameters")
            
            # 准备 extrinsics (N, 3, 4)
            extrinsics = []
            if use_latest_outputs:
                # 直接从最新的 outputs 列表中获取
                for i, output in enumerate(latest_outputs):
                    cam2world = output['camera_poses'][0].cpu().numpy()
                    # 转为 world2cam (3, 4)
                    world2cam = np.linalg.inv(cam2world)[:3, :]  # (3, 4)
                    extrinsics.append(world2cam)
            else:
                # 原有逻辑：从各个 inference_outputs 或 recovered_inference_outputs 获取
                for idx in range(start_idx, end_idx):
                    if use_recovered:
                        recovered = self.recovered_inference_outputs[idx]
                        cam2world = recovered['camera_poses'][0].cpu().numpy()
                    else:
                        inference_data = self.inference_outputs[idx]
                        inference_output = inference_data['current_output']
                        cam2world = inference_output['camera_poses'][0].cpu().numpy()
                    # 转为 world2cam (3, 4)
                    world2cam = np.linalg.inv(cam2world)[:3, :]  # (3, 4)
                    extrinsics.append(world2cam)
            extrinsics = np.stack(extrinsics)  # (N, 3, 4)
            
            # 准备 intrinsics (N, 3, 3)
            intrinsics = []
            if use_latest_outputs:
                # 直接从最新的 outputs 列表中获取
                for i, output in enumerate(latest_outputs):
                    K = output['intrinsics'][0].cpu().numpy()  # (3, 3)
                    intrinsics.append(K)
            else:
                # 原有逻辑
                for idx in range(start_idx, end_idx):
                    if use_recovered:
                        # 如果使用恢复的位姿，仍使用原始内参
                        K = np.array(self.ori_intrinsic[idx]['K'])
                    else:
                        # 使用推理内参
                        inference_data = self.inference_outputs[idx]
                        inference_output = inference_data['current_output']
                        K = inference_output['intrinsics'][0].cpu().numpy()  # (3, 3)
                    intrinsics.append(K)
            intrinsics = np.stack(intrinsics)  # (N, 3, 3)

            # 准备 image_size (2,) - 使用原始图像尺寸
            width = self.ori_intrinsic[start_idx]['width']
            height = self.ori_intrinsic[start_idx]['height']
            image_size = np.array([width, height])
            
            # 准备 masks - 从可见性分数转换
            # 可见性阈值可以调整
            masks = pred_vis_scores > self.pred_vis_scores_thres_value  # (N, P)
            
            # 调用 batch_np_matrix_to_pycolmap
            reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                points3d=points_3d,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                tracks=pred_tracks,
                image_size=image_size,
                masks=masks,
                max_reproj_error=self.max_reproj_error,  # 可以作为参数配置
                max_points3D_val=self.max_points3D_val,
                shared_camera=False,  # 每个相机独立
                camera_type="PINHOLE",  # 使用OPENCV模型
                points_rgb=points_rgb,
                min_inlier_per_frame=self.min_inlier_per_frame,
            )
            
            if reconstruction is None:
                print("  Warning: Failed to build pycolmap reconstruction")
                return False

            # # Bundle Adjustment
            # ba_options = pycolmap.BundleAdjustmentOptions()
            # pycolmap.bundle_adjustment(reconstruction, ba_options)

            # 准备 image_paths 列表（文件名）
            image_paths_list = [Path(path).name for path in image_paths]

            # 准备 original_coords (N, 6) 数组
            # 格式: [x1, y1, x2, y2, width, height]
            # 其中 x1, y1 是裁剪左上角，x2, y2 是右下角，width, height 是原始尺寸
            original_coords_list = []
            for idx in range(start_idx, end_idx):
                scale_info = self.scale_info[idx]
                ori_w, ori_h = scale_info['original_size']
                # 由于没有裁剪，x1=0, y1=0, x2=ori_w, y2=ori_h
                original_coords_list.append(np.array([0, 0, ori_w, ori_h, ori_w, ori_h], dtype=np.float32))
            original_coords = np.stack(original_coords_list)  # (N, 6)

            # 获取预处理后的图像尺寸（img_size）
            proc_w = self.scale_info[start_idx]['output_size'][0]
            proc_h = self.scale_info[start_idx]['output_size'][1]

            # 调用函数
            reconstruction = rename_colmap_recons_and_rescale_camera(
                reconstruction=reconstruction,
                image_paths=image_paths_list,
                original_coords=original_coords,
                img_size=(proc_w, proc_h),
                shift_point2d_to_original_res=True,
                shared_camera=False,
            )

            # 步骤1：先缩放到原始图像尺寸（基本对齐），对齐到已知的影像pose位置
            if self.global_sparse_reconstruction is not None and len(self.sfm_reconstructions) > 0:
                reconstruction = rescale_reconstruction_to_original_size(
                    reconstruction=reconstruction,
                    ori_extrinsics=self.ori_extrinsic,
                    sfm_reconstructions=self.sfm_reconstructions,
                    start_idx=start_idx, 
                    end_idx=end_idx,
                    alignment_mode='pcl_alignment',
                    verbose=self.verbose,
                )
            else:
                reconstruction = rescale_reconstruction_to_original_size(
                    reconstruction=reconstruction,
                    ori_extrinsics=self.ori_extrinsic,
                    sfm_reconstructions=self.sfm_reconstructions,
                    start_idx=start_idx, 
                    end_idx=end_idx,
                    alignment_mode='image_alignment',
                    image_alignment_max_error=10.0,
                    image_alignment_min_inlier_ratio=0.3,
                    verbose=self.verbose,
                )
            # 保存重建结果
            temp_path = self.output_dir / "temp_rescale" / f"{start_idx}_{end_idx}"
            temp_path.mkdir(parents=True, exist_ok=True)
            reconstruction.write_text(str(temp_path))
            reconstruction.export_PLY(str(temp_path / "points3D.ply"))

            # # 步骤2：如果存在merged_reconstruction，与它对齐
            # if self.merged_reconstruction is not None:
            #     aligned_recon = self._align_current_reconstruction_to_merged(
            #         reconstruction,
            #     )
            # else:
            #     aligned_recon = self._align_current_reconstruction_by_point_cloud(
            #         reconstruction,
            #     )

            aligned_recon = self._align_current_reconstruction_by_point_cloud(
                    reconstruction,
                    match_type='use_bidirectional',
                )
            # 保存重建结果
            temp_path = self.output_dir / "temp_aligned" / f"{start_idx}_{end_idx}"
            temp_path.mkdir(parents=True, exist_ok=True)
            aligned_recon.write_text(str(temp_path))
            aligned_recon.export_PLY(str(temp_path / "points3D.ply"))

            # 先缩放到原始图像尺寸（基本对齐）
            aligned_recon=reconstruction 

            # 提取逐像素3D点对应关系（包含 conf）
            # 从最新的 inference_outputs 中获取 outputs_list
            latest_inference = self.inference_outputs[-1]
            outputs_list = latest_inference.get('outputs', [])
            global_indices = list(range(start_idx, end_idx))
            
            batch_pixel_3d_mapping = self._extract_pixel_to_3d_mapping_for_batch(
                outputs_list=outputs_list,
                global_indices=global_indices,
                conf_threshold=1.0,
                verbose=True
            )
            
            self.inference_reconstructions.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'image_paths': image_paths,
                'reconstruction': aligned_recon,
                'valid_track_mask': valid_track_mask,
                'pixel_3d_mapping': batch_pixel_3d_mapping,  # 逐像素3D对应 {global_idx: {'pts3d', 'conf', 'valid_mask'}}，已缩放到原图尺寸
            })
            
            if self.verbose:
                print(f"  ✓ PyColmap reconstruction built")
                print(f"    Number of 3D points: {len(aligned_recon.points3D)}")
                print(f"    Number of cameras: {len(aligned_recon.cameras)}")
                print(f"    Number of images: {len(aligned_recon.images)}")
            
            return True
            
        except Exception as e:
            print(f"  Error building pycolmap reconstruction: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_and_resize_pts3d_for_batch(
        self,
        outputs_list: List[Dict],
        global_indices: List[int],
        verbose: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        从 outputs_list 中提取逐像素 3D 点并缩放到原始图像尺寸
        
        与置信度图不同，pts3d 是 3D 坐标，需要使用最近邻插值以保持几何一致性。
        
        数据结构说明：
        - 输入 pts3d: (1, H_infer, W_infer, 3) - 推理尺寸下的密集 3D 点云
        - 每个像素 (u, v) 对应一个 3D 点 (x, y, z)
        - 输出: (H_orig, W_orig, 3) - 原图尺寸下的密集 3D 点云
        
        Args:
            outputs_list: 推理输出列表，每个元素包含 'pts3d' key
            global_indices: 每个输出对应的全局图像索引
            verbose: 是否打印详细信息
            
        Returns:
            batch_pts3d_maps: {global_idx: (H_orig, W_orig, 3) numpy array}
                             已缩放到原图尺寸的逐像素 3D 点云字典
        """
        batch_pts3d_maps = {}
        
        for i, output in enumerate(outputs_list):
            if i >= len(global_indices):
                break
            
            global_idx = global_indices[i]
            
            if 'pts3d' not in output:
                continue
            
            # 提取 pts3d: (1, H_infer, W_infer, 3) -> (H_infer, W_infer, 3)
            pts3d = output['pts3d']
            if hasattr(pts3d, 'cpu'):
                pts3d = pts3d.cpu().numpy()
            if pts3d.ndim == 4:
                pts3d = pts3d[0]  # (H_infer, W_infer, 3)
            
            infer_h, infer_w, _ = pts3d.shape
            
            # 获取原图尺寸
            if global_idx < len(self.scale_info):
                s_info = self.scale_info[global_idx]
                orig_w, orig_h = s_info['original_size']
                
                # 如果尺寸不同，进行最近邻重采样
                if infer_w != orig_w or infer_h != orig_h:
                    # 使用最近邻插值（保持 3D 坐标的精确性）
                    # cv2.resize 对多通道支持良好
                    pts3d_resized = cv2.resize(
                        pts3d, 
                        (orig_w, orig_h), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    pts3d_resized = pts3d
            else:
                pts3d_resized = pts3d
            
            batch_pts3d_maps[global_idx] = pts3d_resized
        
        if verbose and self.verbose:
            print(f"  ✓ Extracted {len(batch_pts3d_maps)} dense pts3d maps to original resolution")
        
        return batch_pts3d_maps

    def _extract_pixel_to_3d_mapping_for_batch(
        self,
        outputs_list: List[Dict],
        global_indices: List[int],
        conf_threshold: float = 1.0,
        verbose: bool = True
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        建立完整的逐像素到 3D 点的映射关系
        
        这个方法同时提取 pts3d 和 conf，并建立原图尺寸下的完整映射。
        比 reconstruction 中的稀疏匹配更密集，可用于：
        - 密集点云融合
        - 语义标签投影
        - 深度图生成
        
        Args:
            outputs_list: 推理输出列表，包含 'pts3d' 和 'conf'
            global_indices: 每个输出对应的全局图像索引
            conf_threshold: 置信度阈值，低于此值的点可以标记为无效
            verbose: 是否打印详细信息
            
        Returns:
            pixel_3d_mapping: {
                global_idx: {
                    'pts3d': (H_orig, W_orig, 3) numpy array,  # 3D 坐标
                    'conf': (H_orig, W_orig) numpy array,       # 置信度
                    'valid_mask': (H_orig, W_orig) bool array,  # 有效性掩码
                    'infer_size': (H_infer, W_infer),          # 推理尺寸
                    'orig_size': (H_orig, W_orig),             # 原图尺寸
                }
            }
        """
        pixel_3d_mapping = {}
        
        for i, output in enumerate(outputs_list):
            if i >= len(global_indices):
                break
            
            global_idx = global_indices[i]
            
            if 'pts3d' not in output:
                continue
            
            # 提取 pts3d
            pts3d = output['pts3d']
            if hasattr(pts3d, 'cpu'):
                pts3d = pts3d.cpu().numpy()
            if pts3d.ndim == 4:
                pts3d = pts3d[0]  # (H_infer, W_infer, 3)
            
            infer_h, infer_w, _ = pts3d.shape
            
            # 提取 conf
            conf = None
            if 'conf' in output:
                conf = output['conf']
                if hasattr(conf, 'cpu'):
                    conf = conf.cpu().numpy()
                if conf.ndim == 3:
                    conf = conf[0]  # (H_infer, W_infer)
            
            # 获取原图尺寸并重采样
            if global_idx < len(self.scale_info):
                s_info = self.scale_info[global_idx]
                orig_w, orig_h = s_info['original_size']
                
                if infer_w != orig_w or infer_h != orig_h:
                    # pts3d 使用最近邻（保持 3D 坐标精确性）
                    pts3d_resized = cv2.resize(
                        pts3d, (orig_w, orig_h), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    # conf 可以使用双线性插值（连续值）
                    if conf is not None:
                        conf_resized = cv2.resize(
                            conf, (orig_w, orig_h), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        conf_resized = np.ones((orig_h, orig_w), dtype=np.float32)
                else:
                    pts3d_resized = pts3d
                    conf_resized = conf if conf is not None else np.ones((infer_h, infer_w), dtype=np.float32)
            else:
                pts3d_resized = pts3d
                conf_resized = conf if conf is not None else np.ones((infer_h, infer_w), dtype=np.float32)
                orig_w, orig_h = infer_w, infer_h
            
            # 构建有效性掩码（基于置信度和坐标有效性）
            valid_mask = conf_resized >= conf_threshold
            # 过滤无效的 3D 坐标（NaN、Inf、极端值）
            valid_mask &= np.isfinite(pts3d_resized).all(axis=-1)
            valid_mask &= np.abs(pts3d_resized).max(axis=-1) < 1e6
            
            pixel_3d_mapping[global_idx] = {
                'pts3d': pts3d_resized,           # (H_orig, W_orig, 3)
                'conf': conf_resized,             # (H_orig, W_orig)
                'valid_mask': valid_mask,         # (H_orig, W_orig)
                'infer_size': (infer_h, infer_w),
                'orig_size': (orig_h, orig_w),
            }
        
        if verbose and self.verbose:
            total_pixels = sum(m['valid_mask'].sum() for m in pixel_3d_mapping.values())
            print(f"  ✓ Built pixel-to-3D mapping for {len(pixel_3d_mapping)} images, "
                  f"{total_pixels:,} valid pixels total")
        
        return pixel_3d_mapping

    def _align_current_reconstruction_by_point_cloud(
        self,
        reconstruction: pycolmap.Reconstruction,
        match_type: str = 'use_bidirectional',
    ) -> bool:
        """
        Align current reconstruction to previous reconstruction using point cloud correspondences.
        
        Args:
            reconstruction: Current reconstruction to be aligned
            match_type: 匹配类型，支持：
                        - 'use_bidirectional': 双向匹配（更准确，较慢）
                        - 'use_unidirectional': 单向匹配（更快）
                        - 'use_single_images_pair': 只使用一对重叠影像进行匹配（默认使用中间那一对）
            
        Returns:
            对齐后的 reconstruction；若失败或不存在前一重建则返回 False 或原 reconstruction
        """
        if len(self.inference_reconstructions) < 1:
            return reconstruction
        
        # 校验参数
        valid_match_types = {'use_bidirectional', 'use_unidirectional', 'use_single_images_pair'}
        if match_type not in valid_match_types:
            raise ValueError(f"match_type 必须是 {valid_match_types} 之一，当前为: {match_type}")

        # 获取前一个reconstruction（因为当前的还没添加，所以前一个是最后一个）
        prev_recon_data = self.inference_reconstructions[-1]
        prev_recon = prev_recon_data['reconstruction']
        curr_recon = reconstruction  # 当前的就是传入的参数

        # 1. 获取重叠区域的影像ID，Reconstruction中影像ID是从1开始的。
        prev_overlap_image_ids = list(range(
            len(prev_recon.images) - self.overlap + 1,
            len(prev_recon.images) + 1
        ))
        curr_overlap_image_ids = list(range(1, self.overlap + 1))

        # 2. 建立3D点对应关系
        point_correspondences = []  # [(prev_point3D_id, curr_point3D_id, dist)]
        pixel_threshold = 1.0  # 0.5像素阈值

        # 如果是仅使用单一影像对，则只选择一对（默认使用中间对）
        if match_type == 'use_single_images_pair':
            pairs = list(zip(prev_overlap_image_ids, curr_overlap_image_ids))
            if not pairs:
                if self.verbose:
                    print("  Warning: No overlapping image pairs")
                return False

            # 选择一对影像：可切换为 'first' 或 'last'
            selection_mode = 'first'
            if selection_mode == 'first':
                selected_pair = pairs[0]
            elif selection_mode == 'last':
                selected_pair = pairs[-1]
            else:  # 'middle'
                selected_pair = pairs[len(pairs) // 2]

            prev_image_id, curr_image_id = selected_pair

            prev_image_object = prev_recon.images[prev_image_id]
            curr_image_object = curr_recon.images[curr_image_id]

            # 为curr图像建立空间索引
            curr_spatial_index = {}
            for point2D in curr_image_object.points2D:
                if point2D.point3D_id != -1:
                    grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
                    if grid_key not in curr_spatial_index:
                        curr_spatial_index[grid_key] = []
                    # 保存浮点坐标便于距离计算
                    curr_spatial_index[grid_key].append(
                        (int(point2D.point3D_id), np.asarray(point2D.xy, dtype=np.float64))
                    )

            correspondences = find_single_images_pair_matches(
                prev_image_object, 
                curr_image_object, 
                curr_spatial_index, 
                pixel_threshold, 
            )
            point_correspondences.extend(correspondences)

        else:
            for prev_image_id, curr_image_id in zip(prev_overlap_image_ids, curr_overlap_image_ids):
                if prev_image_id not in prev_recon.images or curr_image_id not in curr_recon.images:
                    continue
                
                prev_image = prev_recon.images[prev_image_id]
                curr_image = curr_recon.images[curr_image_id]
                
                # 为curr图像建立空间索引
                curr_spatial_index = {}
                for point2D in curr_image.points2D:
                    if point2D.point3D_id != -1:
                        grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
                        if grid_key not in curr_spatial_index:
                            curr_spatial_index[grid_key] = []
                        curr_spatial_index[grid_key].append((point2D.point3D_id, point2D.xy))
                
                # 根据 match_type 选择匹配策略
                if match_type == 'use_bidirectional':
                    correspondences = self._find_bidirectional_matches(
                        prev_image, curr_image, curr_spatial_index, pixel_threshold
                    )
                elif match_type == 'use_unidirectional':
                    correspondences = self._find_unidirectional_matches(
                        prev_image, curr_image, curr_spatial_index, pixel_threshold
                    )
                    
                point_correspondences.extend(correspondences)

        if len(point_correspondences) == 0:
            print("  Warning: No point correspondences found between overlapping regions")
            return False

        if self.verbose:
            print(f"    Found {len(point_correspondences)} point correspondences in overlap region ({match_type})")

        # 3. 根据对应关系计算变换
        prev_pts3d = []
        curr_pts3d = []
        for prev_pt3d_id, curr_pt3d_id, _ in point_correspondences:
            if prev_pt3d_id in prev_recon.points3D and curr_pt3d_id in curr_recon.points3D:
                prev_pts3d.append(prev_recon.points3D[prev_pt3d_id].xyz)
                curr_pts3d.append(curr_recon.points3D[curr_pt3d_id].xyz)
        
        if len(prev_pts3d) < 3:
            print(f"  Warning: Not enough point correspondences ({len(prev_pts3d)}) for alignment")
            return False

        prev_pts3d = np.array(prev_pts3d)
        curr_pts3d = np.array(curr_pts3d)

        # 使用Umeyama算法计算Sim3变换
        sim3_transform = estimate_sim3_transform(curr_pts3d, prev_pts3d)
        if sim3_transform is None:
            if self.verbose:
                print("  Warning: Failed to estimate Sim3 transform")
            return False
            
        # 应用变换到curr_recon
        curr_recon_aligned = pycolmap.Reconstruction(curr_recon)
        curr_recon_aligned.transform(sim3_transform)

        return curr_recon_aligned

    def _find_unidirectional_matches(
        self,
        prev_image,
        curr_image,
        curr_spatial_index: dict,
        pixel_threshold: float,
    ) -> list:
        """
        单向配对：从prev到curr，选择最近的匹配点
        
        Args:
            prev_image: 前一个重建的影像
            curr_image: 当前重建的影像
            curr_spatial_index: 当前影像的空间索引
            pixel_threshold: 像素阈值
            
        Returns:
            点对应关系列表 [(prev_point3D_id, curr_point3D_id, dist)]
        """
        correspondences = []
        
        for point2D in prev_image.points2D:
            if point2D.point3D_id != -1:
                prev_xy = point2D.xy
                center_grid = (int(round(prev_xy[0])), int(round(prev_xy[1])))
                
                # 收集所有候选匹配点
                candidates = []
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        search_grid = (center_grid[0] + dx, center_grid[1] + dy)
                        if search_grid in curr_spatial_index:
                            for curr_pt3d_id, curr_xy in curr_spatial_index[search_grid]:
                                dist = np.linalg.norm(prev_xy - curr_xy)
                                if dist < pixel_threshold:
                                    candidates.append((curr_pt3d_id, dist))
                
                # 选择距离最近的匹配
                if len(candidates) > 0:
                    best_match = min(candidates, key=lambda x: x[1])
                    curr_pt3d_id, dist = best_match
                    correspondences.append((point2D.point3D_id, curr_pt3d_id, dist))
        
        return correspondences

    def _find_bidirectional_matches(
        self,
        prev_image,
        curr_image,
        curr_spatial_index: dict,
        pixel_threshold: float,
    ) -> list:
        """
        双向配对：prev→curr 和 curr→prev，只保留互相匹配的点
        
        Args:
            prev_image: 前一个重建的影像
            curr_image: 当前重建的影像
            curr_spatial_index: 当前影像的空间索引
            pixel_threshold: 像素阈值
            
        Returns:
            点对应关系列表 [(prev_point3D_id, curr_point3D_id, dist)]
        """
        # 第一步：prev → curr 的匹配
        prev_to_curr = {}  # {prev_point3D_id: (curr_point3D_id, dist)}
        
        for point2D in prev_image.points2D:
            if point2D.point3D_id != -1:
                prev_xy = point2D.xy
                center_grid = (int(round(prev_xy[0])), int(round(prev_xy[1])))
                
                candidates = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        search_grid = (center_grid[0] + dx, center_grid[1] + dy)
                        if search_grid in curr_spatial_index:
                            for curr_pt3d_id, curr_xy in curr_spatial_index[search_grid]:
                                dist = np.linalg.norm(prev_xy - curr_xy)
                                if dist < pixel_threshold:
                                    candidates.append((curr_pt3d_id, dist))
                
                if len(candidates) > 0:
                    best_match = min(candidates, key=lambda x: x[1])
                    prev_to_curr[point2D.point3D_id] = best_match
        
        # 第二步：建立prev图像的空间索引（用于反向匹配）
        prev_spatial_index = {}
        for point2D in prev_image.points2D:
            if point2D.point3D_id != -1:
                grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
                if grid_key not in prev_spatial_index:
                    prev_spatial_index[grid_key] = []
                prev_spatial_index[grid_key].append((point2D.point3D_id, point2D.xy))
        
        # 第三步：curr → prev 的匹配
        curr_to_prev = {}  # {curr_point3D_id: (prev_point3D_id, dist)}
        
        for point2D in curr_image.points2D:
            if point2D.point3D_id != -1:
                curr_xy = point2D.xy
                center_grid = (int(round(curr_xy[0])), int(round(curr_xy[1])))
                
                candidates = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        search_grid = (center_grid[0] + dx, center_grid[1] + dy)
                        if search_grid in prev_spatial_index:
                            for prev_pt3d_id, prev_xy in prev_spatial_index[search_grid]:
                                dist = np.linalg.norm(curr_xy - prev_xy)
                                if dist < pixel_threshold:
                                    candidates.append((prev_pt3d_id, dist))
                
                if len(candidates) > 0:
                    best_match = min(candidates, key=lambda x: x[1])
                    curr_to_prev[point2D.point3D_id] = best_match
        
        # 第四步：只保留互相匹配的点对
        correspondences = []
        for prev_pt3d_id, (curr_pt3d_id, dist_forward) in prev_to_curr.items():
            # 检查反向匹配是否存在且一致
            if curr_pt3d_id in curr_to_prev:
                matched_prev_id, dist_backward = curr_to_prev[curr_pt3d_id]
                if matched_prev_id == prev_pt3d_id:
                    # 互相匹配，取平均距离
                    avg_dist = (dist_forward + dist_backward) / 2.0
                    correspondences.append((prev_pt3d_id, curr_pt3d_id, avg_dist))
        
        return correspondences

    def _merge_reconstruction_intermediate_results(self) -> bool:
        """
        合并reconstruction中间结果
        
        将最新的reconstruction与之前已合并的reconstruction合并，
        通过重叠影像进行对齐。合并结果存储在 self.merged_reconstruction 中。
        
        支持三种合并方式 (由 self.merge_method 控制):
        - 'full': 使用 merge_full_pipeline.py 完整流程
        - 'confidence': 使用 merge_confidence.py 简单的置信度选择合并
        - 'confidence_blend': 使用 merge_confidence_blend.py 基于置信度选择 + 重叠区边缘平滑插值过渡
        
        Returns:
            True if successful, False otherwise
        """
        
        # 如果这是第一个reconstruction，直接设置为merged
        if len(self.inference_reconstructions) == 1:
            self.merged_reconstruction = self.inference_reconstructions[0]['reconstruction']
            merged_recon = self.merged_reconstruction
            # 提取 merged 点云用于可视化
            if self.visualizer is not None:
                self.visualizer.update_merged_point_cloud(merged_recon)
            # 保存merged_reconstruction
            temp_path = self.output_dir / "temp_merged" / f"merged_{len(self.inference_reconstructions)}"
            temp_path.mkdir(parents=True, exist_ok=True)
            merged_recon.write_text(str(temp_path))
            merged_recon.export_PLY(str(temp_path / "points3D.ply"))
            self.export_reconstruction_to_las(merged_recon, str(temp_path / "points3D.las"))
            return True

        # 获取当前reconstruction信息
        curr_recon_data = self.inference_reconstructions[-1]
        
        # 根据 merge_method 选择合并方式
        if self.merge_method == 'confidence_blend':
            # 使用基于置信度 + 边缘平滑插值的合并方式
            merged_recon = self._merge_by_confidence_blend(
                self.merged_reconstruction,
                curr_recon_data['reconstruction'],
                self.output_dir
            )
        elif self.merge_method == 'confidence':
            # 使用简单的置信度选择合并方式
            merged_recon = self._merge_by_confidence(
                self.merged_reconstruction,
                curr_recon_data['reconstruction'],
                self.output_dir
            )
        else:
            # 使用完整的 merge_full_pipeline 流程 (merge_method == 'full')
            merged_recon = self._merge_by_full_pipeline(
                self.merged_reconstruction,
                curr_recon_data['reconstruction'],
                self.output_dir
            )
        
        # 检查合并结果
        if merged_recon is None:
            if self.verbose:
                print(f"  ✗ 合并失败")
            return False
        
        # 更新merged_reconstruction
        self.merged_reconstruction = merged_recon
        
        # 提取 merged 点云用于可视化
        if self.visualizer is not None:
            self.visualizer.update_merged_point_cloud(merged_recon)
        
        # 保存merged_reconstruction
        temp_path = self.output_dir / "temp_merged" / f"merged_{len(self.inference_reconstructions)}"
        temp_path.mkdir(parents=True, exist_ok=True)
        merged_recon.write_text(str(temp_path))
        merged_recon.export_PLY(str(temp_path / "points3D.ply"))
        self.export_reconstruction_to_las(merged_recon, str(temp_path / "points3D.las"))
        
        if self.verbose:
            print(f"  ✓ 合并完成:")
            print(f"    总影像数: {len(merged_recon.images)}")
            print(f"    总3D点数: {len(merged_recon.points3D)}")
            print(f"    结果保存到: {temp_path}")

        # 清理不再需要的中间数据以释放内存
        self._cleanup_intermediate_data(keep_last_n=2)

        return True
    
    def _merge_by_confidence_blend(
        self,
        prev_recon: pycolmap.Reconstruction,
        curr_recon: pycolmap.Reconstruction,
        output_dir: Path
    ) -> Optional[pycolmap.Reconstruction]:
        """
        使用基于置信度 + 边缘平滑插值的方式合并两个 reconstruction
        
        该方法结合了多种技术实现高质量的重建合并：
        
        1. 基于置信度的点选择：
           - 使用神经网络输出的 conf 作为置信度依据
           - 对于相同像素位置，选择置信度更高的 3D 点
           - 非重叠区域的点全部保留
        
        2. 重叠区边缘平滑插值过渡：
           - 使用多级 2D 匹配半径进行渐进式匹配
           - 融合带内使用加权平均混合 3D 坐标
           - 空间插值使用 smoothstep 实现平滑过渡
           - 避免合并边界处出现明显的不连续
        
        3. 密度均衡化：
           - 使非重叠区点云密度与重叠区一致
           - 支持网格采样和距离衰减
        
        置信度存储结构：
        - self.inference_reconstructions[i]['pixel_3d_mapping']: 每个批次的逐像素3D对应
          {global_idx: {'pts3d': (H,W,3), 'conf': (H,W), 'valid_mask': (H,W)}}
        - conf 已经缩放到原图尺寸，与 reconstruction 中的 2D 点坐标一致
        
        Args:
            prev_recon: 之前已合并的 reconstruction
            curr_recon: 当前要合并的 reconstruction
            output_dir: 输出目录
            
        Returns:
            合并后的 reconstruction，失败返回 None
        """
        # 从 pixel_3d_mapping 中提取置信度图
        # prev_recon_conf: 之前所有批次的 conf（batches 0 to N-2）
        # curr_recon_conf: 当前批次的 conf（batch N-1）
        prev_recon_conf: Dict[int, np.ndarray] = {}
        curr_recon_conf: Dict[int, np.ndarray] = {}
        
        num_batches = len(self.inference_reconstructions)
        
        # 从 pixel_3d_mapping 提取 conf 的辅助函数
        def extract_conf_from_pixel_3d_mapping(pixel_3d_mapping: Dict) -> Dict[int, np.ndarray]:
            """从 pixel_3d_mapping 提取 conf 信息，转换为 {global_idx: (H, W) array} 格式"""
            conf_maps = {}
            for global_idx, data in pixel_3d_mapping.items():
                if 'conf' in data:
                    conf_maps[global_idx] = data['conf']
            return conf_maps
        
        # 前 N-1 个批次的 conf 属于 prev_recon
        for i in range(num_batches - 1):
            pixel_3d_mapping = self.inference_reconstructions[i].get('pixel_3d_mapping', {})
            batch_conf_maps = extract_conf_from_pixel_3d_mapping(pixel_3d_mapping)
            prev_recon_conf.update(batch_conf_maps)
        
        # 最后一个批次的 conf 属于 curr_recon
        if num_batches > 0:
            pixel_3d_mapping = self.inference_reconstructions[-1].get('pixel_3d_mapping', {})
            curr_recon_conf = extract_conf_from_pixel_3d_mapping(pixel_3d_mapping)
        
        if self.verbose:
            print(f"\n=== 使用 confidence_blend 合并 (置信度选择 + 边缘平滑插值) ===")
            print(f"    prev_recon: {len(prev_recon.images)} images, {len(prev_recon.points3D)} 3D points")
            print(f"    curr_recon: {len(curr_recon.images)} images, {len(curr_recon.points3D)} 3D points")
            print(f"    overlap: {self.overlap} images")
            print(f"    num_batches: {num_batches}")
            
            # 显示每个批次的 pixel_3d_mapping 信息
            for i, recon_data in enumerate(self.inference_reconstructions):
                pixel_3d_mapping = recon_data.get('pixel_3d_mapping', {})
                batch_keys = sorted(pixel_3d_mapping.keys()) if pixel_3d_mapping else []
                print(f"    batch {i} [{recon_data['start_idx']}-{recon_data['end_idx']}]: {len(pixel_3d_mapping)} pixel_3d_mapping, keys={batch_keys}")
            
            # 显示分离后的结果
            prev_keys = sorted(prev_recon_conf.keys())
            curr_keys = sorted(curr_recon_conf.keys())
            print(f"    prev_recon_conf: {len(prev_recon_conf)} images, keys={prev_keys}")
            print(f"    curr_recon_conf: {len(curr_recon_conf)} images, keys={curr_keys}")
        
        # 构建 image_name -> global_idx 的映射（用于像素级置信度查询）
        image_name_to_idx = {ext['image_name']: idx for idx, ext in enumerate(self.ori_extrinsic)}

        # 获取当前批次的 start_idx 和 end_idx（用于输出目录命名）
        if num_batches > 0:
            start_idx = self.inference_reconstructions[-1].get('start_idx')
            end_idx = self.inference_reconstructions[-1].get('end_idx')
        else:
            start_idx = None
            end_idx = None
        
        # 调用 reconstruction_merger 中的合并函数
        # 分别传入 prev_recon 和 curr_recon 的置信度图
        # 使用多级半径匹配 + 加权平均模式 + 3D匹配补充 + 激进3D匹配 + 边缘位置平滑 + 融合带空间插值
        merged_recon, info = merge_by_confidence_blend(
            prev_recon,
            curr_recon,
            inlier_threshold=10,
            min_inliers=5,
            min_sample_size=5,
            ransac_iterations=1000,
            prev_recon_conf=prev_recon_conf,
            curr_recon_conf=curr_recon_conf,
            image_name_to_idx=image_name_to_idx,
            output_dir=output_dir,
            start_idx=start_idx,
            end_idx=end_idx,
            color_by_source=True,
            match_radii=[1, 2, 3, 5, 8, 10, 20, 30, 40, 50],  # 多级2D匹配半径（从小到大依次匹配）
            match_3d_threshold=10.0,  # 3D空间匹配阈值（单位与点云坐标一致）
            aggressive_3d_threshold=3.0,  # 最终阶段激进3D匹配阈值，用于减少重叠区独有点
            inner_blend_margin=150.0,  # 融合带向内延伸（像素），同时控制空间插值范围
            outer_blend_margin=200.0,  # 融合带向外延伸（像素），同时控制空间插值范围
            blend_mode='weighted',   # 加权平均模式，基于置信度计算加权位置
            keep_unmatched_overlap=True,  # 保留重叠区未匹配点
            spatial_blend_interpolation=True,  # 启用融合带3D坐标空间插值
            spatial_blend_k_neighbors=32,  # 空间插值使用的近邻数
            spatial_blend_smooth_transition=True,  # 使用 smoothstep 实现更平滑过渡
            spatial_blend_smooth_power=0.7,  # 平滑力度：<1更强效果（建议0.3-0.7）
            density_equalization=True,  # 启用密度均衡化，使非重叠区密度与重叠区一致
            density_k_neighbors=10,  # 密度计算使用的近邻数
            density_target_percentile=50.0,  # 使用重叠区点间距中位数作为目标
            density_tolerance=1.2,  # 密度容差倍数
            density_use_grid=True,  # 使用网格采样（更稳定均匀）
            density_grid_resolution=1.0,  # 网格分辨率因子
            density_distance_decay=0.5,  # 距离衰减因子（远离重叠区的点保留更多）
            voxel_size=self.merge_voxel_size,  # 体素降采样大小
            verbose=self.verbose,
        )
        
        return merged_recon
    
    def _merge_by_confidence(
        self,
        prev_recon: pycolmap.Reconstruction,
        curr_recon: pycolmap.Reconstruction,
        output_dir: Path
    ) -> Optional[pycolmap.Reconstruction]:
        """
        使用简单的置信度选择方式合并两个 reconstruction
        
        该方法基于置信度进行点选择：
        - 对于相同像素位置，选择置信度更高的 3D 点
        - 非重叠区域的点全部保留
        
        与 confidence_blend 相比，此方法更简单直接，不包含边缘平滑插值等高级功能。
        
        Args:
            prev_recon: 之前已合并的 reconstruction
            curr_recon: 当前要合并的 reconstruction
            output_dir: 输出目录
            
        Returns:
            合并后的 reconstruction，失败返回 None
        """
        # 从 pixel_3d_mapping 中提取置信度图
        prev_recon_conf: Dict[int, np.ndarray] = {}
        curr_recon_conf: Dict[int, np.ndarray] = {}
        
        num_batches = len(self.inference_reconstructions)
        
        def extract_conf_from_pixel_3d_mapping(pixel_3d_mapping: Dict) -> Dict[int, np.ndarray]:
            """从 pixel_3d_mapping 提取 conf 信息"""
            conf_maps = {}
            for global_idx, data in pixel_3d_mapping.items():
                if 'conf' in data:
                    conf_maps[global_idx] = data['conf']
            return conf_maps
        
        # 前 N-1 个批次的 conf 属于 prev_recon
        for i in range(num_batches - 1):
            pixel_3d_mapping = self.inference_reconstructions[i].get('pixel_3d_mapping', {})
            batch_conf_maps = extract_conf_from_pixel_3d_mapping(pixel_3d_mapping)
            prev_recon_conf.update(batch_conf_maps)
        
        # 最后一个批次的 conf 属于 curr_recon
        if num_batches > 0:
            pixel_3d_mapping = self.inference_reconstructions[-1].get('pixel_3d_mapping', {})
            curr_recon_conf = extract_conf_from_pixel_3d_mapping(pixel_3d_mapping)
        
        if self.verbose:
            print(f"\n=== 使用 confidence 合并 (简单置信度选择) ===")
            print(f"    prev_recon: {len(prev_recon.images)} images, {len(prev_recon.points3D)} 3D points")
            print(f"    curr_recon: {len(curr_recon.images)} images, {len(curr_recon.points3D)} 3D points")
            print(f"    overlap: {self.overlap} images")
        
        # 构建 image_name -> global_idx 的映射
        image_name_to_idx = {ext['image_name']: idx for idx, ext in enumerate(self.ori_extrinsic)}
        
        # 获取当前批次的 start_idx 和 end_idx
        if num_batches > 0:
            start_idx = self.inference_reconstructions[-1].get('start_idx')
            end_idx = self.inference_reconstructions[-1].get('end_idx')
        else:
            start_idx = None
            end_idx = None
        
        # 调用简单置信度合并函数
        merged_recon, info = merge_by_confidence(
            prev_recon,
            curr_recon,
            inlier_threshold=10,
            min_inliers=5,
            min_sample_size=5,
            ransac_iterations=1000,
            prev_recon_conf=prev_recon_conf,
            curr_recon_conf=curr_recon_conf,
            image_name_to_idx=image_name_to_idx,
            output_dir=output_dir,
            start_idx=start_idx,
            end_idx=end_idx,
            match_radii=[1, 2, 3, 5, 8, 10, 20, 30, 40, 50],
            k_neighbors=10,  # 查询多个近邻以提高匹配率
            color_by_match_status=False, # 设为 True 可启用调试着色
            blend_mode='weighted',
            blend_weight=0.8,
            verbose=self.verbose,
        )
        
        return merged_recon
    
    def _merge_by_full_pipeline(
        self,
        prev_recon: pycolmap.Reconstruction,
        curr_recon: pycolmap.Reconstruction,
        output_dir: Path
    ) -> Optional[pycolmap.Reconstruction]:
        """
        使用完整的 merge_construction 流程合并
        
        Args:
            prev_recon: 之前已合并的 reconstruction
            curr_recon: 当前要合并的 reconstruction
            output_dir: 输出目录
            
        Returns:
            合并后的 reconstruction，失败返回 None
        """
        # 临时保存两个reconstruction到磁盘
        temp_base = self.output_dir / "temp_merge_input"
        temp_base.mkdir(parents=True, exist_ok=True)
        
        # 保存prev_recon (已合并的)
        prev_dir = temp_base / "prev"
        prev_dir.mkdir(parents=True, exist_ok=True)
        prev_recon.write_text(str(prev_dir))
        
        # 保存curr_recon (当前的)
        curr_dir = temp_base / "curr"
        curr_dir.mkdir(parents=True, exist_ok=True)
        curr_recon.write_text(str(curr_dir))
        
        if self.verbose:
            print(f"\n=== 使用 merge_reconstructions 进行合并 ===")
            print(f"  prev_dir: {prev_dir}")
            print(f"  curr_dir: {curr_dir}")
            print(f"  output_dir: {output_dir}")
        
        # 调用 merge_reconstructions 函数
        merged_recon = merge_reconstructions(
            model_dir1=str(prev_dir),
            model_dir2=str(curr_dir),
            output_dir=str(output_dir),
            overlap_count=self.overlap,
            translation_only=True,  # 初始对齐只做平移
            use_ransac=False,
            # 点云融合参数
            point_fusion=True,
            fusion_method="2d_matching",
            cell_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 80, 160, 320, 640, 1280],
            keep_unmatched_overlap=True,  # 保留重叠区未匹配点
            spatial_dedup_threshold=0.1,
            # 精化对齐参数
            refine_alignment=True,
            refine_cell_range=(1, 3),
            refine_stages=[
                (None, "translation"),       # 第1阶段：不筛选距离，只平移
                (10.0, "scale_translation"), # 第2阶段：dist<=10m
                (5.0, "scale_translation"),  # 第3阶段：dist<=5m
                (2.0, "scale_translation"),  # 第4阶段：dist<=2m
                (1.0, "scale_translation"),  # 第5阶段：dist<=1m
                (0.5, "scale_translation"),  # 第6阶段：dist<=0.5m
            ],
            voxel_size=self.merge_voxel_size,
            statistical_filter=self.merge_statistical_filter,
            min_track_length=2,
            boundary_filter=self.merge_boundary_filter,
            filter_edge_margin=self.filter_edge_margin,
            verbose=self.verbose
        )
        
        return merged_recon  

    def _align_merged_reconstruction_to_gps_poses(
        self,
        reconstruction: pycolmap.Reconstruction
    ) -> pycolmap.Reconstruction:
        """
        将merged_reconstruction对齐到已知的GPS poses
        
        这与_rescale_reconstruction_to_original_size类似，但用于merged_reconstruction：
        1. 收集merged中所有影像对应的GPS位置
        2. 使用RANSAC对齐到GPS坐标系
        3. 纠正累积误差，防止reconstruction弯曲
        
        Args:
            reconstruction: merged_reconstruction
            
        Returns:
            对齐后的reconstruction
        """
        if self.verbose:
            print(f"    Aligning merged reconstruction to GPS poses...")
        
        try:
            # 1. 收集所有已处理的影像索引
            # 从所有inference_reconstructions中提取影像范围
            all_image_indices = []
            for recon_data in self.inference_reconstructions:
                start_idx = recon_data['start_idx']
                end_idx = recon_data['end_idx']
                # 注意重叠：每个reconstruction的前overlap张可能已经在前一个中了
                if len(all_image_indices) == 0:
                    # 第一个reconstruction，全部添加
                    all_image_indices.extend(range(start_idx, end_idx))
                else:
                    # 后续reconstruction，跳过重叠部分
                    all_image_indices.extend(range(start_idx + self.overlap, end_idx))
            
            if self.verbose:
                print(f"      Total unique image indices: {len(all_image_indices)}")
                print(f"      Reconstruction has {len(reconstruction.images)} images")
            
            # 2. 为每个reconstruction中的影像收集GPS位置
            tgt_image_names = []
            tgt_locations = []
            
            # merged_reconstruction中的image_id是连续的1-based
            for merged_image_id, image in sorted(reconstruction.images.items()):
                # merged_image_id从1开始，映射到all_image_indices
                if merged_image_id <= len(all_image_indices):
                    orig_idx = all_image_indices[merged_image_id - 1]
                    
                    if orig_idx < len(self.ori_extrinsic):
                        extrinsic = self.ori_extrinsic[orig_idx]
                        R_camera = np.array(extrinsic['R_camera'])
                        tvec = np.array(extrinsic['tvec'])
                        camera_center = -R_camera.T @ tvec
                        
                        tgt_image_names.append(image.name)
                        tgt_locations.append(camera_center)
            
            if len(tgt_image_names) < 3:
                print(f"      Warning: Not enough images ({len(tgt_image_names)}) for GPS alignment")
                return reconstruction
            
            tgt_locations = np.array(tgt_locations, dtype=np.float64)
            
            if self.verbose:
                print(f"      Collected {len(tgt_image_names)} GPS positions for alignment")
            
            # 3. 使用RANSAC对齐
            ransac_options = pycolmap.RANSACOptions()
            ransac_options.max_error = 5.0  # 5米误差阈值（可以根据GPS精度调整）
            ransac_options.min_inlier_ratio = 0.3  # 60%内点
            
            sim3d = pycolmap.align_reconstruction_to_locations(
                src=reconstruction,
                tgt_image_names=tgt_image_names,
                tgt_locations=tgt_locations,
                min_common_points=max(3, len(tgt_image_names) // 4),  # 至少用1/4的点
                ransac_options=ransac_options
            )
            
            if sim3d is not None:
                # 应用变换
                reconstruction.transform(sim3d)
                
                if self.verbose:
                    print(f"      ✓ Merged reconstruction aligned to GPS poses")
                    print(f"        Scale: {sim3d.scale:.6f}")
                    print(f"        Aligned images: {len(tgt_image_names)}")
            else:
                print("      Warning: Failed to align merged reconstruction to GPS")
                
        except Exception as e:
            print(f"      Error aligning merged reconstruction to GPS: {e}")
            import traceback
            traceback.print_exc()
        
        return reconstruction    

    def create_pixel_coordinate_grid(self, num_frames, height, width):
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

    def randomly_limit_trues(self, mask: np.ndarray, max_trues: int) -> np.ndarray:
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

    def get_statistics(self) -> Dict:
        """Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_images': len(self.ori_extrinsic),
            'num_intrinsics': len(self.ori_intrinsic),
            'num_extrinsics': len(self.ori_extrinsic),
        }
        
        return stats
    
    def export_reconstruction_to_las(self, reconstruction: pycolmap.Reconstruction, output_path: str):
        """
        将pycolmap Reconstruction导出为LAS格式
        
        Args:
            reconstruction: pycolmap重建对象
            output_path: 输出的.las文件路径
        """
        # 提取所有3D点的坐标和颜色
        points = []
        colors = []
        
        for point3D_id, point3D in reconstruction.points3D.items():
            points.append(point3D.xyz)
            colors.append(point3D.color)
        
        points = np.array(points)
        colors = np.array(colors)
        
        # 创建LAS文件
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = np.min(points, axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])  # 1mm精度
        
        las = laspy.LasData(header)
        
        # 设置坐标
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        
        # 设置颜色 (LAS使用16位颜色值)
        las.red = (colors[:, 0] * 256).astype(np.uint16)
        las.green = (colors[:, 1] * 256).astype(np.uint16)
        las.blue = (colors[:, 2] * 256).astype(np.uint16)
        
        # 写入文件
        las.write(output_path)

    def export_georeferenced(self, reconstruction: Optional[pycolmap.Reconstruction] = None, 
                              output_dir: Optional[Path] = None,
                              target_crs: str = "auto_utm",
                              gps_prior: float = 5.0) -> bool:
        """Export the reconstruction and poses in a target coordinate system.

        This method:
        1. Extracts camera positions from reconstruction and matches with original GPS data
        2. Converts original GPS lat/lon/alt to target CRS coordinates using pyproj
        3. Aligns the reconstruction to target coordinates using pycolmap model_aligner
        4. Exports the transformed reconstruction

        Args:
            reconstruction: pycolmap重建对象，默认使用 self.merged_reconstruction
            output_dir: 输出目录，默认使用 self.output_dir / "georeferenced"
            target_crs: 目标坐标系，支持以下选项：
                - "auto_utm": 自动检测UTM区域（默认）
                - "EPSG:3857": Web Mercator（适合网页地图可视化）
                - "EPSG:4326": WGS84 经纬度坐标
                - "EPSG:XXXX": 任意EPSG代码，如 "EPSG:32648" (UTM Zone 48N)
            gps_prior: GPS先验误差（米），用于alignment_max_error

        Returns:
            True if successful, False otherwise
        """
        if not UTM_EXPORT_AVAILABLE:
            print("Error: pymap3d or pyproj not available. Install with: pip install pymap3d pyproj")
            return False

        # 使用默认值
        if reconstruction is None:
            reconstruction = self.merged_reconstruction
        if output_dir is None:
            output_dir = self.output_dir / "temp_merged_reconstruction_georeferenced"

        if reconstruction is None:
            print("Error: No reconstruction available.")
            return False
        
        if len(self.ori_extrinsic) == 0:
            print("Error: No original extrinsic data available.")
            return False

        try:
            # 1. 建立 image_name -> GPS 的映射（从原始数据）
            image_name_to_gps = {}
            for ext in self.ori_extrinsic:
                img_name = ext['image_name']
                gps = ext['gps']  # [lat, lon, alt]
                image_name_to_gps[img_name] = gps
            
            # 2. 从 reconstruction 中提取图像名称，并匹配 GPS
            image_names = []
            lats = []
            lons = []
            alts = []
            
            for img_id, img in reconstruction.images.items():
                img_name = img.name
                if img_name in image_name_to_gps:
                    gps = image_name_to_gps[img_name]
                    image_names.append(img_name)
                    lats.append(gps[0])
                    lons.append(gps[1])
                    alts.append(gps[2])
                else:
                    if self.verbose:
                        print(f"  Warning: No GPS data for image {img_name}, skipping")
            
            if len(image_names) < 3:
                print(f"Error: Not enough images with GPS data ({len(image_names)}). Need at least 3.")
                return False
            
            lats = np.array(lats)
            lons = np.array(lons)
            alts = np.array(alts)
            
            if self.verbose:
                print(f"  Matched {len(image_names)} images with GPS data")

            # 3. Determine target CRS
            try:
                if target_crs.lower() == "auto_utm":
                    # 自动检测UTM区域
                    utm_crs_info = pyproj.database.query_utm_crs_info(
                        datum_name="WGS 84",
                        area_of_interest=pyproj.aoi.AreaOfInterest(
                            west_lon_degree=float(lons[0]) - 0.01,
                            south_lat_degree=float(lats[0]) - 0.01,
                            east_lon_degree=float(lons[0]) + 0.01,
                            north_lat_degree=float(lats[0]) + 0.01,
                        ),
                    )
                    if not utm_crs_info:
                        print("Error: Could not determine UTM zone for the given coordinates")
                        return False
                    target_crs_obj = pyproj.CRS.from_epsg(utm_crs_info[0].code)
                    self.output_epsg_code = utm_crs_info[0].code
                elif target_crs.upper().startswith("EPSG:"):
                    # 使用指定的EPSG代码
                    epsg_code = int(target_crs.split(":")[1])
                    target_crs_obj = pyproj.CRS.from_epsg(epsg_code)
                    self.output_epsg_code = epsg_code
                else:
                    # 尝试直接解析为CRS
                    target_crs_obj = pyproj.CRS.from_user_input(target_crs)
                    self.output_epsg_code = target_crs_obj.to_epsg() if target_crs_obj.to_epsg() else None
                
                if self.verbose:
                    crs_name = target_crs_obj.name if hasattr(target_crs_obj, 'name') else str(target_crs_obj)
                    print(f"  Using CRS: {crs_name} (EPSG:{self.output_epsg_code})")
                    
            except Exception as e:
                print(f"Error determining target CRS: {e}")
                return False

            # Create transformer from WGS84 to target CRS
            crs_transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs_obj, always_xy=True)

            # Convert to target CRS coordinates
            target_x, target_y = crs_transformer.transform(lons, lats)
            target_coords = np.column_stack([target_x, target_y, alts])

            # 4. Use the mean as center to offset the whole scene (避免坐标值过大)
            geo_center = np.mean(target_coords, axis=0)
            geo_center[2] = 0.0  # Set altitude to 0 for center
            target_coords_offset = target_coords - geo_center

            # 5. Get image names for alignment
            tgt_image_names = image_names
            min_common_images = min(len(tgt_image_names), 3)

            # 保存ENU重建到临时目录（用于colmap model_aligner输入）
            enu_temp_dir = output_dir.parent / "enu_temp"
            enu_temp_dir.mkdir(parents=True, exist_ok=True)
            reconstruction.write_text(str(enu_temp_dir))

            # Write reference images file for colmap model_aligner
            ref_images_path = output_dir.parent / "georef_locations.txt"
            with open(ref_images_path, "w") as f:
                for img_name, (x, y, z) in zip(tgt_image_names, target_coords_offset):
                    f.write(f"{img_name} {x} {y} {z}\n")

            if self.verbose:
                print(f"  Reference images written to: {ref_images_path}")
                print(f"  Geo center (offset): [{geo_center[0]:.3f}, {geo_center[1]:.3f}, {geo_center[2]:.3f}]")

            # 6. Use colmap model_aligner to align reconstruction to target CRS
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "colmap",
                "model_aligner",
                "--log_to_stderr", "1" if self.verbose else "0",
                "--input_path", str(enu_temp_dir),
                "--output_path", str(output_dir),
                "--ref_images_path", str(ref_images_path),
                "--alignment_type", "custom",
                "--min_common_images", str(min_common_images),
                "--alignment_max_error", str(gps_prior * 3.0),
                "--ref_is_gps", "0",
            ]

            try:
                if self.verbose:
                    print(f"  Running COLMAP model_aligner...")
                subprocess.run(cmd, check=True, capture_output=not self.verbose)
            except subprocess.CalledProcessError as e:
                print(f"COLMAP model_aligner failed: {e}")
                return False
            except FileNotFoundError:
                print("Error: COLMAP not found. Please install COLMAP and add it to PATH.")
                return False

            # 7. Load the aligned model
            try:
                rec_georef = pycolmap.Reconstruction(str(output_dir))
            except Exception as e:
                print(f"Failed to load aligned georeferenced reconstruction: {e}")
                return False

            # 8. Export as PLY and LAS
            rec_georef.export_PLY(str(output_dir / "sparse_points.ply"))
            self.export_reconstruction_to_las(rec_georef, str(output_dir / "sparse_points.las"))

            # Store georeferenced data for later use
            self.geo_center = geo_center
            self.rec_georef = rec_georef
            self.rec_georef_dir = output_dir

            if self.verbose:
                print(f"  ✓ Georeferenced export completed:")
                print(f"    EPSG code: {self.output_epsg_code}")
                print(f"    Geo center (offset): [{geo_center[0]:.3f}, {geo_center[1]:.3f}, {geo_center[2]:.3f}]")
                print(f"    Output dir: {output_dir}")
                print(f"    Total images: {len(rec_georef.images)}")
                print(f"    Total 3D points: {len(rec_georef.points3D)}")

            return True

        except Exception as e:
            print(f"Error during georeferenced export: {e}")
            import traceback
            traceback.print_exc()
            return False

    def export_utm(self, reconstruction: Optional[pycolmap.Reconstruction] = None, 
                   output_dir: Optional[Path] = None,
                   gps_prior: float = 5.0) -> bool:
        """Export the reconstruction in auto-detected UTM coordinates.
        
        This is a convenience wrapper for export_georeferenced with target_crs="auto_utm".
        For more options (EPSG:3857, EPSG:4326, etc.), use export_georeferenced directly.

        Args:
            reconstruction: pycolmap重建对象，默认使用 self.merged_reconstruction
            output_dir: 输出目录
            gps_prior: GPS先验误差（米）

        Returns:
            True if successful, False otherwise
        """
        return self.export_georeferenced(
            reconstruction=reconstruction,
            output_dir=output_dir,
            target_crs="auto_utm",
            gps_prior=gps_prior
        )

def run_incremental_feature_matching(
    image_paths: List[Path],
    output_dir: Path,
    reconstruction_type: str = 'each_pixel_feature_points', # 'dense_feature_points' | 'each_pixel_feature_points'
    model_type: str = 'mapanything',  # 'mapanything' | 'vggt'
    model_path: Optional[str] = None,  # 模型权重路径（VGGT需要）
    image_interval: int = 2,
    min_images_for_scale: int = 6,
    overlap: int = 2,
    pred_vis_scores_thres_value: float = 0.8, 
    max_reproj_error: float = 5.0,
    max_points3D_val: int = 1000000,
    min_inlier_per_frame: int = 32,
    run_global_sfm_first: bool = True,
    filter_edge_margin: float = 100.0, # 边缘过滤范围（像素），默认50，设为0禁用
    merge_voxel_size: float = 2.5,  # 点云合并时的体素大小（米）
    merge_boundary_filter: bool = True,  # 是否启用边界过滤
    merge_statistical_filter: bool = False,  # 是否启用统计过滤
    export_georef: bool = True,  # 是否导出地理坐标系的重建结果
    target_crs: str = "auto_utm",  # 目标坐标系: "auto_utm", "EPSG:3857", "EPSG:4326", 等
    merge_method: str = 'full',  # 'full' | 'confidence' | 'confidence_blend' 合并方式
    enable_visualization: bool = True,
    visualization_mode: str = 'merged',  # 'aligned' | 'merged'
    verbose: bool = False,
) -> bool:
    """Run incremental image initialization pipeline.
    
    Args:
        image_paths: List of image file paths in processing order
        output_dir: Directory for output files
        reconstruction_type: Type of reconstruction, 'dense_feature_points' or 'each_pixel_feature_points'
        model_type: Type of model to use, 'mapanything' or 'vggt'
        model_path: Path to model weights (required for VGGT, optional for MapAnything)
        image_interval: Interval for selecting images (1=all, 2=every 2nd, etc.)
        min_images_for_scale: Minimum number of images required for scale estimation
        overlap: Number of overlapping images between consecutive reconstructions
        pred_vis_scores_thres_value: Minimum visibility threshold for feature tracking
        max_reproj_error: Maximum reprojection error for feature matching
        max_points3D_val: Maximum number of 3D points in the reconstruction
        min_inlier_per_frame: Minimum number of inliers per frame for feature matching
        run_global_sfm_first: Whether to run global SfM first
        filter_edge_margin: Edge margin for filtering points (in pixels), default 10, set to 0 to disable
        merge_voxel_size: Voxel size for point cloud merging (in meters), default 1.0
        merge_boundary_filter: Whether to enable boundary filtering during merge, default True
        merge_statistical_filter: Whether to enable statistical filtering during merge, default False
        export_georef: Whether to export reconstruction in georeferenced coordinates
        target_crs: Target coordinate system for georeferenced export:
            - "auto_utm": 自动检测UTM区域（默认）
            - "EPSG:3857": Web Mercator（适合网页地图可视化）
            - "EPSG:4326": WGS84 经纬度坐标
            - "EPSG:XXXX": 任意EPSG代码
        merge_method: Merge method selection: 
            'full' - use merge_full_pipeline.py with full pipeline
            'confidence' - use merge_confidence.py with simple confidence-based selection
            'confidence_blend' - use merge_confidence_blend.py with confidence-based selection
                                 and smooth edge blending/interpolation (default)
        enable_visualization: Whether to start viser server for visualization
        visualization_mode: Point cloud visualization mode, 'aligned' (per batch) or 'merged' (unified)
        verbose: Enable verbose logging
    
    Returns:
        True if successful, False otherwise
    """
    # Process images one by one with interval control
    selected_image_paths = image_paths[::image_interval]

    # Run global SfM first (optional but recommended), aim to  get sparse reconstruction of the whole dataset
    global_sparse_reconstruction = None

    if run_global_sfm_first:
        # Get input directory (all images should be in the same directory)
        input_dir = selected_image_paths[0].parent
        # Create output directory for global SfM
        global_sfm_output_dir = output_dir / "global_sfm"
        global_sfm_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if SfM results already exist (可直接读取已有结果，避免重复计算)
        # 增量SfM输出到 enu 目录（与直接三角化保持一致）
        sparse_model_dir = global_sfm_output_dir / "enu"
        cameras_file = sparse_model_dir / "cameras.txt"
        images_file = sparse_model_dir / "images.txt"
        points3D_file = sparse_model_dir / "points3D.txt"
        
        if cameras_file.exists() and images_file.exists() and points3D_file.exists():
            # 已有SfM结果，直接读取
            if verbose:
                print(f"  Found existing SfM results in: {sparse_model_dir}")
                print(f"  Loading reconstruction from files...")
            try:
                global_sparse_reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))
                if verbose:
                    print(f"  Loaded: {len(global_sparse_reconstruction.images)} images, "
                          f"{len(global_sparse_reconstruction.points3D)} 3D points, "
                          f"{len(global_sparse_reconstruction.cameras)} cameras")
            except Exception as e:
                print(f"  Warning: Failed to load existing SfM results: {e}")
                print(f"  Will run SfM from scratch...")
                global_sparse_reconstruction = None
        else:
            global_sparse_reconstruction = None
        
        # 如果没有已有结果，运行SfM
        if global_sparse_reconstruction is None:
            if verbose:
                print(f"  No existing SfM results found, running SfM...")
            # Create FeatureMatcherSfM instance
            # 增加 num_neighbors 以捕获跨航带匹配（对于转弯场景很重要）
            global_sparse_matcher = FeatureMatcherSfM(
                input_dir=input_dir,
                output_dir=global_sfm_output_dir,
                imgsz=2048,
                num_features=8192,
                match_mode="sequential",
                num_neighbors=20, 
                max_distance=1000.0,
                sfm_mode="incremental",  # Options: "direct", "direct_ba", "incremental"
                verbose=verbose,
            )        
            success = global_sparse_matcher.run_pipeline()
            if success:
                global_sparse_reconstruction = global_sparse_matcher.rec_prior
            else:
                print("  Warning: Failed to run global SfM")
                return False

    matcher = IncrementalFeatureMatcherSfM(
        output_dir=output_dir,
        reconstruction_type=reconstruction_type,
        model_type=model_type,
        model_path=model_path,
        global_sparse_reconstruction=global_sparse_reconstruction,
        min_images_for_scale=min_images_for_scale,
        overlap=overlap,
        pred_vis_scores_thres_value=pred_vis_scores_thres_value,
        max_reproj_error=max_reproj_error,
        max_points3D_val=max_points3D_val,
        min_inlier_per_frame=min_inlier_per_frame,
        filter_edge_margin=filter_edge_margin,
        merge_voxel_size=merge_voxel_size,
        merge_boundary_filter=merge_boundary_filter,
        merge_statistical_filter=merge_statistical_filter,
        merge_method=merge_method,
        enable_visualization=enable_visualization,
        visualization_mode=visualization_mode,
        verbose=verbose,
    )

    # Process images one by one
    for i, image_path in enumerate(selected_image_paths):
        success = matcher.add_image(image_path)
        
        if not success:
            print(f"Failed to process image: {image_path}")
            return False

    # Release model
    matcher._release_model()

    # Export georeferenced coordinates if requested
    if export_georef and matcher.merged_reconstruction is not None:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Exporting to georeferenced coordinates (target_crs: {target_crs})...")
            print(f"{'='*60}")
        georef_success = matcher.export_georeferenced(target_crs=target_crs)
        if not georef_success:
            print("  Warning: Georeferenced export failed, but ENU reconstruction is still available")

    if verbose:
        stats = matcher.get_statistics()
        print(f"\n{'='*60}")
        print("Final Statistics:")
        print(f"{'='*60}")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    return True


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # ==================== 配置参数 ====================
    # 输入输出目录
    # input_dir = Path(r"drone-map-anything\examples\Comprehensive_building_sel\images")
    # output_dir = Path(r"drone-map-anything\output\Comprehensive_building_sel\sparse_incremental_reconstruction")
    
    # input_dir = Path(r"drone-map-anything\examples\Ganluo_images\images")
    # output_dir = Path(r"drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction")
    
    # input_dir = Path(r"drone-map-anything\examples\Tazishan\images")
    # output_dir = Path(r"drone-map-anything\output\Tazishan\sparse_incremental_reconstruction")

    # input_dir = Path(r"drone-map-anything\examples\SWJTU_gongdi\images")
    # output_dir = Path(r"drone-map-anything\output\SWJTU_gongdi\sparse_incremental_reconstruction")

    # input_dir = Path(r"drone-map-anything\examples\SWJTU_7th_teaching_building\images")
    # output_dir = Path(r"drone-map-anything\output\SWJTU_7th_teaching_building\sparse_incremental_reconstruction")
    
    # input_dir = Path(r"drone-map-anything\examples\HuaPo\images")
    # output_dir = Path(r"drone-map-anything\output\HuaPo\sparse_incremental_reconstruction")

    input_dir = Path(r"drone-map-anything\examples\WenChuan\images")
    output_dir = Path(r"drone-map-anything\output\WenChuan\sparse_incremental_reconstruction")

    # 模型选择: 'mapanything' 或 'vggt'
    MODEL_TYPE = 'vggt'  # 切换模型类型
    MODEL_PATH = "weights/vggt/model.pt"  # VGGT 模型权重路径，如果使用 VGGT 需要设置，例如: "weights/model.pt"
    
    # ================================================

    # Get all image files and sort them
    supported_formats = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = sorted([
        f for f in input_dir.glob("*") 
        if f.suffix in supported_formats
    ])
    
    print(f"Found {len(image_files)} images")
    print(f"Using model: {MODEL_TYPE}")
    
    # Run incremental initialization
    success = run_incremental_feature_matching(
        image_paths=image_files,
        output_dir=output_dir,
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        verbose=True,
    )
    
    if success:
        print("\n✓ Image initialization completed successfully")
    else:
        print("\n✗ Image initialization failed")