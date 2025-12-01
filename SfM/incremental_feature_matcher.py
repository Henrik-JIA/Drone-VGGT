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
from PIL import Image
from scipy.spatial.transform import Rotation as R
import viser
import viser.transforms as tf

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from feature_matcher import FeatureMatcherSfM
from utils.gps import extract_gps_from_image, lat_lon_to_enu
from utils.xmp import parse_xmp_tags
from mapanything.utils.image import preprocess_inputs
from mapanything.models import MapAnything
from mapanything.third_party.track_predict import predict_tracks
from mapanything.third_party.np_to_pycolmap import batch_np_matrix_to_pycolmap
            
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
        global_sparse_reconstruction: Optional[pycolmap.Reconstruction] = None,
        verbose: bool = False,
        min_images_for_scale: int = 2,
        overlap: int = 1,
        max_reproj_error: float = 10.0,
        max_points3D_val: int = 5000,
        min_inlier_per_frame: int = 32,
        pred_vis_scores: float = 0.3, 
        enable_visualization: bool = True
    ):
        """Initialize incremental feature matcher.
        
        Args:
            output_dir: Directory for output files
            verbose: Enable verbose logging
            min_images_for_scale: Minimum number of images before calculating scale.
                                  2 = calculate from 2nd image (default)
                                  3 = calculate from 3rd image
                                  4 = calculate from 4th image
                                  etc.
            overlap: Number of overlapping images between consecutive reconstructions
            max_reproj_error: Maximum reprojection error (in pixels) for filtering tracks
            max_points3D_val: Per-component absolute-value threshold for 3D points (a point is kept only if |x|, |y|, and |z| are all less than this value).
            min_inlier_per_frame: Minimum inlier count per frame for valid BA
            pred_vis_scores: Visibility confidence threshold for tracks
            enable_visualization: Whether to start viser server for visualization
        """
        # Model (lazy loading)
        self.model = None
        self.device = None

        self.output_dir = Path(output_dir)
        self.global_sparse_reconstruction = global_sparse_reconstruction
        self.verbose = verbose
        self.min_images_for_scale = max(2, min_images_for_scale)
        self.overlap = overlap       
        self.pred_vis_scores = pred_vis_scores
        self.max_reproj_error = max_reproj_error
        self.max_points3D_val = max_points3D_val
        self.min_inlier_per_frame = min_inlier_per_frame

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

        self.enable_visualization = enable_visualization
        self.viser_server = None
        self.viser_frustum_handles = []
        self.viser_point_handles = []

        # Setup visualization if enabled
        if self.enable_visualization:
            self._setup_visualization()

    def _setup_visualization(self):
        """Setup viser visualization server."""
        self.viser_server = viser.ViserServer()

        # 添加客户端连接回调，设置相机参数
        @self.viser_server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            # 设置更大的渲染距离，防止点云在远距离消失
            # 这些参数会影响深度裁剪
            client.camera.far = 5000.0  # 远裁剪平面，默认可能只有100
            client.camera.near = 0.1      # 近裁剪平面

        self.viser_server.scene.add_frame(
            "/reconstruction",
            wxyz=tf.SO3.from_x_radians(0.0).wxyz,
            position=(0, 0, 0),
            show_axes=True,
        )
        
        # Add GUI controls
        with self.viser_server.gui.add_folder("Visualization"):
            self.gui_point_size = self.viser_server.gui.add_slider(
                "Point size",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=2.0,
            )
            self.gui_show_frustums = self.viser_server.gui.add_checkbox(
                "Show frustums", True
            )
            self.gui_show_points = self.viser_server.gui.add_checkbox(
                "Show point clouds", True
            )
            self.gui_frustum_scale = self.viser_server.gui.add_slider(
                "Frustum scale",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=3.0,
            )
        
        # Initialize storage for scene handles
        self.viser_frustum_handles = []
        self.viser_point_handles = []
        
        # Add callbacks for real-time updates
        @self.gui_show_frustums.on_update
        def _(_):
            for handle in self.viser_frustum_handles:
                handle.visible = self.gui_show_frustums.value

        @self.gui_point_size.on_update
        def _(_):
            for handle in self.viser_point_handles:
                handle.point_size = self.gui_point_size.value
        
        @self.gui_show_points.on_update
        def _(_):
            for handle in self.viser_point_handles:
                handle.visible = self.gui_show_points.value
        
        @self.gui_frustum_scale.on_update
        def _(_):
            self._update_visualization()
        
        if self.verbose:
            print("✓ Viser visualization server started")
            print(f"  Open browser at: http://localhost:8080")
        
        return self.viser_server

    # def _update_visualization(self):
    #     """Update viser visualization with latest data.
        
    #     This function:
    #     1. Uses the latest scale_ratio to rescale ALL point clouds
    #     2. Updates/adds camera frustums
    #     3. Updates/adds point clouds
    #     """
    #     if not hasattr(self, 'viser_server') or self.viser_server is None:
    #         return
        
    #     # Get number of images to visualize
    #     num_images = len(self.recovered_inference_outputs)
        
    #     # Clear old handles if we're re-scaling
    #     if len(self.viser_point_handles) > 0:
    #         for handle in self.viser_frustum_handles:
    #             handle.remove() # 移除所有相机视锥体
    #         for handle in self.viser_point_handles:
    #             handle.remove() # 移除所有点云
    #         self.viser_frustum_handles = [] # 清空列表
    #         self.viser_point_handles = [] # 清空列表
        
    #     # Visualize each image
    #     for i in range(num_images):
    #         recovered_output = self.recovered_inference_outputs[i]

    #         pts3d_recovered = recovered_output['pts3d']  # (1, H, W, 3) tensor
    #         pts3d_np = pts3d_recovered[0].cpu().numpy()  # (H, W, 3)

    #         H, W, _ = pts3d_np.shape
            
    #         pts_recovered = pts3d_np.reshape(-1, 3)
            
    #         # Get colors from original image (from input_views)
    #         if i < len(self.input_views):
    #             img_array = self.input_views[i]['img'].cpu().numpy()  # (H_orig, W_orig, 3)
    #             # Resize colors to match point cloud resolution
    #             img_resized = np.array(Image.fromarray(img_array).resize((W, H), Image.BILINEAR))
    #             colors = img_resized.reshape(-1, 3).astype(np.uint8)
    #         else:
    #             # Default gray color if no image available
    #             colors = np.full((pts_recovered.shape[0], 3), 128, dtype=np.uint8)
            
    #         # ==================== Add camera frustum =====================
    #         camera_pose = recovered_output['camera_poses'][0].cpu().numpy()  # (4, 4)
    #         K = np.array(recovered_output['camera_K'])  # (3, 3)
    #         width = recovered_output['image_width']
    #         height = recovered_output['image_height']
            
    #         # Calculate FOV from intrinsics
    #         fx = K[0, 0]
    #         fy = K[1, 1]
    #         fov_y = 2 * np.arctan2(height / 2, fy)
    #         aspect = width / height
            
    #         # Camera orientation and position
    #         R_cam = camera_pose[:3, :3]
    #         t_cam = camera_pose[:3, 3]
            
    #         if self.gui_show_frustums.value:
    #             # Get thumbnail image
    #             if i < len(self.input_views):
    #                 img_thumbnail = self.input_views[i]['img'].cpu().numpy()
    #                 # Downsample for display
    #                 downsample = 4
    #                 img_thumbnail = img_thumbnail[::downsample, ::downsample]
    #             else:
    #                 img_thumbnail = None
                
    #             frustum_handle = self.viser_server.scene.add_camera_frustum(
    #                 f"/reconstruction/camera_{i:03d}",
    #                 fov=fov_y,
    #                 aspect=aspect,
    #                 scale=self.gui_frustum_scale.value,
    #                 image=img_thumbnail,
    #                 wxyz=tf.SO3.from_matrix(R_cam).wxyz,
    #                 position=t_cam,
    #             )
    #             self.viser_frustum_handles.append(frustum_handle)
            
    #         # ==================== Add point cloud =====================
    #         if self.gui_show_points.value:
    #             point_handle = self.viser_server.scene.add_point_cloud(
    #                 f"/reconstruction/points_{i:03d}",
    #                 points=pts_recovered,
    #                 colors=colors,
    #                 point_size=self.gui_point_size.value,
    #                 point_shape="rounded",
    #             )
    #             self.viser_point_handles.append(point_handle)
        
    #     if self.verbose:
    #         print(f"  ✓ Visualization updated: {num_images} cameras and point clouds")
    #         print(f"    Total points: {sum([h.points.shape[0] for h in self.viser_point_handles])}")
            
    def _update_visualization(self):
        """Update viser visualization with latest data.
        
        This function adds only NEW images that haven't been visualized yet.
        """
        if not hasattr(self, 'viser_server') or self.viser_server is None:
            return
        
        # Get number of images to visualize
        num_images = len(self.recovered_inference_outputs)
        num_visualized = len(self.viser_point_handles)  # 已经可视化的数量
        
        # 只添加新的图像（增量添加）
        if num_images > num_visualized:
            if self.verbose:
                print(f"  Adding {num_images - num_visualized} new images to visualization...")
            
            # Visualize only NEW images
            for i in range(num_visualized, num_images):
                recovered_output = self.recovered_inference_outputs[i]

                pts3d_recovered = recovered_output['pts3d']  # (1, H, W, 3) tensor
                pts3d_np = pts3d_recovered[0].cpu().numpy()  # (H, W, 3)

                H, W, _ = pts3d_np.shape
                
                pts_recovered = pts3d_np.reshape(-1, 3)
                
                # Get colors from original image (from input_views)
                if i < len(self.input_views):
                    img_array = self.input_views[i]['img'].cpu().numpy()  # (H_orig, W_orig, 3)
                    # Resize colors to match point cloud resolution
                    img_resized = np.array(Image.fromarray(img_array).resize((W, H), Image.BILINEAR))
                    colors = img_resized.reshape(-1, 3).astype(np.uint8)
                else:
                    # Default gray color if no image available
                    colors = np.full((pts_recovered.shape[0], 3), 128, dtype=np.uint8)
                
                # ==================== Add camera frustum =====================
                camera_pose = recovered_output['camera_poses'][0].cpu().numpy()  # (4, 4)
                K = np.array(recovered_output['camera_K'])  # (3, 3)
                width = recovered_output['image_width']
                height = recovered_output['image_height']
                
                # Calculate FOV from intrinsics
                fx = K[0, 0]
                fy = K[1, 1]
                fov_y = 2 * np.arctan2(height / 2, fy)
                aspect = width / height
                
                # Camera orientation and position
                R_cam = camera_pose[:3, :3]
                t_cam = camera_pose[:3, 3]
                
                if self.gui_show_frustums.value:
                    # Get thumbnail image
                    if i < len(self.input_views):
                        img_thumbnail = self.input_views[i]['img'].cpu().numpy()
                        # Downsample for display
                        downsample = 4
                        img_thumbnail = img_thumbnail[::downsample, ::downsample]
                    else:
                        img_thumbnail = None
                    
                    frustum_handle = self.viser_server.scene.add_camera_frustum(
                        f"/reconstruction/camera_{i:03d}",
                        fov=fov_y,
                        aspect=aspect,
                        scale=self.gui_frustum_scale.value,
                        image=img_thumbnail,
                        wxyz=tf.SO3.from_matrix(R_cam).wxyz,
                        position=t_cam,
                    )
                    self.viser_frustum_handles.append(frustum_handle)
                
                # ==================== Add point cloud =====================
                if self.gui_show_points.value:
                    point_handle = self.viser_server.scene.add_point_cloud(
                        f"/reconstruction/points_{i:03d}",
                        points=pts_recovered,
                        colors=colors,
                        point_size=self.gui_point_size.value,
                        point_shape="rounded",
                    )
                    self.viser_point_handles.append(point_handle)
            
            if self.verbose:
                print(f"  ✓ Visualization updated: {num_images} cameras and point clouds total")
                print(f"    Total points: {sum([h.points.shape[0] for h in self.viser_point_handles])}")
        else:
            if self.verbose:
                print(f"  No new images to visualize")

    def _prepare_visualization_data(self, image_path: Path) -> bool:
        """Prepare and update visualization after adding a new image.
        
        This is called after _recover_original_pose for each new image.
        """
        # Setup visualization server on first call
        if not hasattr(self, 'viser_server'):
            self._setup_visualization()
        
        # Update visualization with all current data
        if hasattr(self, 'viser_server') and self.viser_server is not None:
            self._update_visualization()
        
        return True

    def _load_model(self):
        """Load MapAnything model (lazy loading).
        
        Returns:
            Loaded model
        """
        if self.model is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.verbose:
                print(f"Using device: {self.device}")
            model_name = "facebook/map-anything"
            if self.verbose:
                print(f"Loading model: {model_name}...")
            self.model = MapAnything.from_pretrained(model_name).to(self.device)
            if self.verbose:
                print("✓ Model loaded")
        
        return self.model

    def _release_model(self):
        """Release model from memory to free GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.verbose:
                print("✓ Model released from memory")

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
                sfm_extract_success = self._extract_sfm_reconstruction_from_global(
                    start_idx=start_idx,
                    end_idx=end_idx
                )
            else:
                if self.verbose:
                    print("  Skipping SfM extraction (no global reconstruction available)")

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

            # Viser visualization
            # prepare data for visualization (这里会自动更新可视化)
            if self.enable_visualization:
                prepare_visualization_data_success = self._prepare_visualization_data(image_path)
        
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

        # Run inference
        if self.verbose:
            print("Running MapAnything inference...")

        # 判断是第几张图像
        num_images = len(self.preprocessed_views)

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
                    print(f"  ================================ Computed Scale Ratio (COLMAP/MapAnything): {scale_ratio:.6f}")
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
        predicted_scale_ratio = outputs[-1]['metric_scaling_factor'].item()
        
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

    # def _predict_tracks_for_batch(self, start_idx: int, end_idx: int) -> bool:
    #     """Predict tracks for a batch of images.
        
    #     Args:
    #         start_idx: Starting index in self.recovered_inference_outputs
    #         end_idx: Ending index (exclusive) in self.recovered_inference_outputs
            
    #     Returns:
    #         True if successful, False otherwise
    #     """
    #     try:
    #         if self.verbose:
    #             print(f"\n  Predicting tracks for batch (indices {start_idx} to {end_idx-1})...")
            
    #         # 准备这批图像的数据
    #         batch_images = []
    #         batch_depths = []
    #         batch_confs = []
    #         batch_points_3d = []
            
    #         for idx in range(start_idx, end_idx):
    #             # 获取恢复后的输出
    #             recovered_output = self.recovered_inference_outputs[idx]
                
    #             # 从 input_views 获取图像
    #             if idx < len(self.input_views):
    #                 img = self.input_views[idx]['img']  # (H, W, 3) numpy array or tensor
    #                 batch_images.append(img)
                
    #             # 从 recovered_inference_outputs 获取深度
    #             if 'pts3d' in recovered_output:
    #                 pts3d = recovered_output['pts3d']  # (1, H, W, 3) tensor
    #                 batch_points_3d.append(pts3d[0])
    #                 # 计算深度（Z值）
    #                 depth = pts3d[0, :, :, 2]  # (H, W)
    #                 batch_depths.append(depth)
                
    #             # 获取置信度
    #             if 'conf' in recovered_output:
    #                 conf = recovered_output['conf']  # (1, H, W)
    #                 batch_confs.append(conf[0])
            
    #         # 将列表转换为tensor
    #         if len(batch_images) > 0:
    #             # 确保所有图像是浮点tensor格式并归一化到 [0, 1]
    #             processed_images = []
    #             for img in batch_images:
    #                 if isinstance(img, np.ndarray):
    #                     # numpy array: (H, W, 3), uint8 [0-255]
    #                     img_tensor = torch.from_numpy(img).float() / 255.0  # 转为 [0, 1]
    #                     img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
    #                 else:
    #                     # 已经是 tensor
    #                     if img.dtype == torch.uint8:
    #                         img_tensor = img.float() / 255.0
    #                     else:
    #                         img_tensor = img
    #                     # 确保格式为 (3, H, W)
    #                     if img_tensor.dim() == 3 and img_tensor.shape[-1] == 3:
    #                         img_tensor = img_tensor.permute(2, 0, 1)
                    
    #                 processed_images.append(img_tensor)
                
    #             batch_images_tensor = torch.stack(processed_images).cuda()  # (B, 3, H, W)
                
    #             # 准备深度、置信度和3D点
    #             batch_depths_tensor = torch.stack(batch_depths).cuda() if batch_depths else None
    #             batch_confs_tensor = torch.stack(batch_confs).cuda() if batch_confs else None
    #             batch_points_3d_tensor = torch.stack(batch_points_3d).cuda() if batch_points_3d else None
                
    #             # 运行特征点跟踪
    #             with torch.amp.autocast("cuda", dtype=torch.float16):
    #                 pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
    #                     batch_images_tensor,
    #                     conf=batch_confs_tensor,
    #                     points_3d=batch_points_3d_tensor,  # 可以传入已有的3D点
    #                     max_query_pts=4096,  # 可以作为参数配置
    #                     query_frame_num=2,   # 查询帧数量
    #                     keypoint_extractor="aliked+sp",
    #                     fine_tracking=True,
    #                 )
                
    #             # 存储跟踪结果
    #             if not hasattr(self, 'batch_tracks'):
    #                 self.batch_tracks = []
                
    #             self.batch_tracks.append({
    #                 'start_idx': start_idx,
    #                 'end_idx': end_idx,
    #                 'pred_tracks': pred_tracks.cpu(),
    #                 'pred_vis_scores': pred_vis_scores.cpu(),
    #                 'pred_confs': pred_confs.cpu(),
    #                 'points_3d': points_3d.cpu() if points_3d is not None else None,
    #                 'points_rgb': points_rgb.cpu() if points_rgb is not None else None,
    #             })
                
    #             if self.verbose:
    #                 print(f"  ✓ Track prediction completed")
    #                 print(f"    Tracks shape: {pred_tracks.shape}")
    #                 print(f"    Number of tracked points: {pred_tracks.shape[1]}")
                
    #             torch.cuda.empty_cache()
    #             return True
    #         else:
    #             print("  Warning: No images available for track prediction")
    #             return False
                
    #     except Exception as e:
    #         print(f"  Error during track prediction: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return False

    # def _extract_sfm_reconstruction_from_global(self, start_idx: int, end_idx: int) -> bool:
    #     """
    #     从全局稀疏重建中按顺序提取与当前批次影像相同的子重建

    #     Args:
    #         start_idx: 起始影像索引
    #         end_idx: 结束影像索引（不包含）

    #     Returns:
    #         True if successful, False otherwise
    #     """
    #     try:
    #         if self.global_sparse_reconstruction is None:
    #             if self.verbose:
    #                 print(f"  Warning: No global sparse reconstruction available")
    #             return False

    #         if self.verbose:
    #             print(f"  Extracting SfM reconstruction from global reconstruction for images {start_idx} to {end_idx-1}...")

    #         global_recon = self.global_sparse_reconstruction

    #         # 1. 按原始顺序收集需要提取的影像名称
    #         target_image_names_list = [self.image_paths[idx].name for idx in range(start_idx, end_idx)]

    #         # 2. 构建全局 name -> image_id 的映射
    #         global_name_to_id = {img.name: image_id for image_id, img in global_recon.images.items()}

    #         # 3. 按原始顺序建立 (global_image_id, new_image_id, orig_name) 列表
    #         matched_pairs = []  # [(gid, new_id, name)]
    #         new_id = 1
    #         for name in target_image_names_list:
    #             gid = global_name_to_id.get(name, None)
    #             if gid is not None:
    #                 matched_pairs.append((gid, new_id, name))
    #                 new_id += 1

    #         if len(matched_pairs) == 0:
    #             if self.verbose:
    #                 print(f"  Warning: No matching images found in global reconstruction")
    #             return False

    #         if self.verbose:
    #             print(f"    Found {len(matched_pairs)} matching images in global reconstruction (ordered)")

    #         # 无效 3D 点 ID 常量（uint64(-1)）
    #         INVALID_POINT3D_ID = (1 << 64) - 1

    #         def _valid_pid(pid):
    #             try:
    #                 pid_int = int(pid)
    #             except Exception:
    #                 return False
    #             return pid_int != -1 and pid_int != INVALID_POINT3D_ID

    #         # 4. 创建新的子重建
    #         sub_recon = pycolmap.Reconstruction()

    #         # 5. 先添加相机（保持相同的 camera_id），仅加入被使用到的相机
    #         used_camera_ids = set(global_recon.images[gid].camera_id for gid, _, _ in matched_pairs)
    #         for camera_id in used_camera_ids:
    #             if camera_id in global_recon.cameras:
    #                 sub_recon.add_camera(global_recon.cameras[camera_id])

    #         # 6. 收集需要的3D点ID（被匹配影像观测到的点）
    #         required_point3D_ids = set()
    #         for gid, _, _ in matched_pairs:
    #             global_image = global_recon.images[gid]
    #             for p2d in global_image.points2D:
    #                 if _valid_pid(p2d.point3D_id):
    #                     required_point3D_ids.add(int(p2d.point3D_id))

    #         if self.verbose:
    #             print(f"    Extracting {len(required_point3D_ids)} 3D points")

    #         # 7. 创建3D点ID映射并添加3D点（先不写入 track）
    #         point3D_id_map = {}  # {global_point3D_id: new_point3D_id}
    #         for global_point3D_id in required_point3D_ids:
    #             if global_point3D_id in global_recon.points3D:
    #                 global_point = global_recon.points3D[global_point3D_id]
    #                 new_point3D_id = sub_recon.add_point3D(
    #                     xyz=global_point.xyz,
    #                     track=pycolmap.Track(),
    #                     color=global_point.color
    #                 )
    #                 point3D_id_map[global_point3D_id] = new_point3D_id

    #         # 8. 添加影像并更新 track 信息（按 new_id 递增顺序）
    #         for gid, new_image_id, orig_name in matched_pairs:
    #             global_image = global_recon.images[gid]

    #             # 重建 points2D 列表，更新 point3D_id
    #             new_points2D = []

    #             for p2d in global_image.points2D:
    #                 if _valid_pid(p2d.point3D_id):
    #                     gpid = int(p2d.point3D_id)
    #                     if gpid in point3D_id_map:
    #                         new_pid = point3D_id_map[gpid]
    #                         new_points2D.append(pycolmap.Point2D(p2d.xy, new_pid))
    #                         # 更新 3D 点的 track，point2D 索引为当前刚加入的点的下标
    #                         track = sub_recon.points3D[new_pid].track
    #                         track.add_element(new_image_id, len(new_points2D) - 1)
    #                     else:
    #                         # 该 3D 点未被纳入子重建
    #                         new_points2D.append(pycolmap.Point2D(p2d.xy))
    #                 else:
    #                     # 无效 3D 点
    #                     new_points2D.append(pycolmap.Point2D(p2d.xy))

    #             # 使用统一的命名格式，且与 new_image_id 对齐
    #             unified_image_name = f"image_{new_image_id}"

    #             new_image = pycolmap.Image(
    #                 image_id=new_image_id,
    #                 name=unified_image_name,
    #                 camera_id=global_image.camera_id,
    #                 cam_from_world=global_image.cam_from_world,
    #                 points2D=new_points2D
    #             )
    #             sub_recon.add_image(new_image)

    #         # 9. 保存子重建结果（可选）
    #         temp_path = self.output_dir / "temp_sfm_extracted" / f"sfm_{start_idx}_{end_idx}"
    #         temp_path.mkdir(parents=True, exist_ok=True)
    #         sub_recon.write_text(str(temp_path))
    #         sub_recon.export_PLY(str(temp_path / "points3D.ply"))

    #         # 10. 存储到 sfm_reconstructions，构建映射
    #         # 先构建 原始名 -> 路径 的字典
    #         orig_name_to_path = {self.image_paths[idx].name: str(self.image_paths[idx]) for idx in range(start_idx, end_idx)}

    #         image_name_to_path = {}
    #         orig_name_to_new_id = {}
    #         for _, new_image_id, orig_name in matched_pairs:
    #             unified = f"image_{new_image_id}"
    #             if orig_name in orig_name_to_path:
    #                 image_name_to_path[unified] = orig_name_to_path[orig_name]
    #             orig_name_to_new_id[orig_name] = new_image_id

    #         sfm_result = {
    #             'start_idx': start_idx,
    #             'end_idx': end_idx,
    #             'image_paths': [orig_name_to_path[name] for _, _, name in matched_pairs if name in orig_name_to_path],
    #             'image_name_mapping': image_name_to_path,   # unified -> file path
    #             'orig_name_to_new_id': orig_name_to_new_id, # 原始文件名 -> new_image_id（可用于对齐时的稳健映射）
    #             'reconstruction': sub_recon,
    #             'num_images': len(sub_recon.images),
    #             'num_points3D': len(sub_recon.points3D),
    #             'num_cameras': len(sub_recon.cameras),
    #             'source': 'extracted_from_global',
    #         }
    #         self.sfm_reconstructions.append(sfm_result)

    #         if self.verbose:
    #             print(f"  ✓ SfM reconstruction extracted from global")
    #             print(f"    Number of images: {sfm_result['num_images']}")
    #             print(f"    Number of 3D points: {sfm_result['num_points3D']}")
    #             print(f"    Number of cameras: {sfm_result['num_cameras']}")

    #         return True

    #     except Exception as e:
    #         print(f"  Error extracting SfM reconstruction from global: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return False

    def _extract_sfm_reconstruction_from_global(self, start_idx: int, end_idx: int) -> bool:
        """
        从全局稀疏重建中提取与当前批次影像相同的子重建
        
        Args:
            start_idx: 起始影像索引
            end_idx: 结束影像索引（不包含）
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.global_sparse_reconstruction is None:
                if self.verbose:
                    print(f"  Warning: No global sparse reconstruction available")
                return False
            
            if self.verbose:
                print(f"  Extracting SfM reconstruction from global reconstruction for images {start_idx} to {end_idx-1}...")
            
            global_recon = self.global_sparse_reconstruction
            
            # 1. 收集需要提取的影像名称
            target_image_names = set()
            for idx in range(start_idx, end_idx):
                image_path = self.image_paths[idx]
                target_image_names.add(image_path.name)
            
            # 2. 在全局重建中查找匹配的影像ID
            matched_image_ids = {}  # {global_image_id: local_image_id}
            local_image_id = 1
            
            for global_image_id, image in global_recon.images.items():
                if image.name in target_image_names:
                    matched_image_ids[global_image_id] = local_image_id
                    local_image_id += 1
            
            if len(matched_image_ids) == 0:
                if self.verbose:
                    print(f"  Warning: No matching images found in global reconstruction")
                return False
            
            if self.verbose:
                print(f"    Found {len(matched_image_ids)} matching images in global reconstruction")
            
            # 3. 创建新的子重建
            sub_recon = pycolmap.Reconstruction()
            
            # 4. 收集需要的3D点ID（被匹配影像观测到的点）
            required_point3D_ids = set()
            for global_image_id in matched_image_ids.keys():
                global_image = global_recon.images[global_image_id]
                for point2D in global_image.points2D:
                    if point2D.point3D_id != -1:
                        required_point3D_ids.add(point2D.point3D_id)
            
            if self.verbose:
                print(f"    Extracting {len(required_point3D_ids)} 3D points")
            
            # 5. 添加相机模型
            # 收集被使用的相机ID
            used_camera_ids = set()
            for global_image_id in matched_image_ids.keys():
                used_camera_ids.add(global_recon.images[global_image_id].camera_id)
            
            # 添加相机（保持相同的camera_id）
            for camera_id in used_camera_ids:
                if camera_id in global_recon.cameras:
                    sub_recon.add_camera(global_recon.cameras[camera_id])
            
            # 6. 创建3D点ID映射并添加3D点
            point3D_id_map = {}  # {global_point3D_id: new_point3D_id}
            
            for global_point3D_id in required_point3D_ids:
                if global_point3D_id in global_recon.points3D:
                    global_point = global_recon.points3D[global_point3D_id]
                    
                    # 添加3D点（注意：先不添加track信息）
                    new_point3D_id = sub_recon.add_point3D(
                        xyz=global_point.xyz,
                        track=pycolmap.Track(),
                        color=global_point.color
                    )
                    point3D_id_map[global_point3D_id] = new_point3D_id
            
            # 7. 添加影像并更新track信息
            for global_image_id, new_image_id in sorted(matched_image_ids.items(), key=lambda x: x[1]):
                global_image = global_recon.images[global_image_id]
                
                # 重建points2D列表，更新point3D_id
                new_points2D = []
                point2D_idx = 0
                
                for point2D in global_image.points2D:
                    if point2D.point3D_id != -1 and point2D.point3D_id in point3D_id_map:
                        # 点在子重建中
                        new_point3D_id = point3D_id_map[point2D.point3D_id]
                        new_points2D.append(pycolmap.Point2D(point2D.xy, new_point3D_id))
                        
                        # 更新3D点的track信息
                        track = sub_recon.points3D[new_point3D_id].track
                        track.add_element(new_image_id, point2D_idx)
                        point2D_idx += 1
                    else:
                        # 点不在子重建中，不传递point3D_id参数（使用默认无效值）
                        new_points2D.append(pycolmap.Point2D(point2D.xy))
                        point2D_idx += 1
                
                # ✅ 修改这里：使用统一的命名格式，而不是真实文件名
                unified_image_name = f"image_{new_image_id}"  # 使用 image_1, image_2, ... 格式
                
                # 创建新Image对象
                new_image = pycolmap.Image(
                    image_id=new_image_id,
                    name=global_image.name,
                    camera_id=global_image.camera_id,
                    cam_from_world=global_image.cam_from_world,
                    points2D=new_points2D
                )
                
                sub_recon.add_image(new_image)
            
            # 8. 保存子重建结果
            temp_path = self.output_dir / "temp_sfm_extracted" / f"sfm_{start_idx}_{end_idx}"
            temp_path.mkdir(parents=True, exist_ok=True)
            sub_recon.write_text(str(temp_path))
            sub_recon.export_PLY(str(temp_path / "points3D.ply"))
            
            # 9. 存储到 sfm_reconstructions
            image_paths_to_process = [self.image_paths[idx] for idx in range(start_idx, end_idx)]

            # image_name_to_path = {}
            # for idx, image_path in enumerate(image_paths_to_process):
            #     unified_name = f"image_{idx + 1}"
            #     image_name_to_path[unified_name] = str(image_path)

            # 关键：用原始文件名做映射键，保证与 sub_recon.images 的 name 一致
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
                'source': 'extracted_from_global',  # 标记来源
            }
            self.sfm_reconstructions.append(sfm_result)
            
            if self.verbose:
                print(f"  ✓ SfM reconstruction extracted from global")
                print(f"    Number of images: {sfm_result['num_images']}")
                print(f"    Number of 3D points: {sfm_result['num_points3D']}")
                print(f"    Number of cameras: {sfm_result['num_cameras']}")
            
            return True
            
        except Exception as e:
            print(f"  Error extracting SfM reconstruction from global: {e}")
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
            masks = pred_vis_scores > self.pred_vis_scores  # (N, P)
            
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
            reconstruction = self.rename_colmap_recons_and_rescale_camera(
                reconstruction=reconstruction,
                image_paths=image_paths_list,
                original_coords=original_coords,
                img_size=(proc_w, proc_h),
                shift_point2d_to_original_res=True,
                shared_camera=False,
            )

            # 步骤1：先缩放到原始图像尺寸（基本对齐），对齐到已知的影像pose位置
            reconstruction = self._rescale_reconstruction_to_original_size(
                reconstruction, 
                start_idx, 
                end_idx
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

            self.inference_reconstructions.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'image_paths': image_paths,
                'reconstruction': aligned_recon,
                'valid_track_mask': valid_track_mask,
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

    def rename_colmap_recons_and_rescale_camera(
        self,
        reconstruction,
        image_paths,
        original_coords,
        img_size,
        shift_point2d_to_original_res=False,
        shared_camera=False,
    ):
        # 规范化 original_coords 到 numpy
        if "torch" in sys.modules and isinstance(original_coords, torch.Tensor):
            original_coords_np = original_coords.detach().cpu().numpy()
        else:
            original_coords_np = np.asarray(original_coords)
            
        if isinstance(img_size, (list, tuple, np.ndarray)) and len(img_size) == 2:
            proc_w = float(img_size[0])
            proc_h = float(img_size[1])
        else:
            # 兼容旧逻辑：若传入标量，则假定等比缩放
            proc_w = float(img_size)
            proc_h = float(img_size)
            
        rescale_camera = True

        for pyimageid in reconstruction.images:
            # Reshaped the padded & resized image to the original size
            # Rename the images to the original names
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = image_paths[pyimageid - 1]

            # 读取该帧原图尺寸
            real_image_size = original_coords_np[pyimageid - 1, -2:]  # [W_orig, H_orig]
            real_w = float(real_image_size[0])
            real_h = float(real_image_size[1])

            # 宽高方向缩放因子
            # 注意：如果 preprocess 是非等比缩放，这两个值不同
            sx = real_w / max(1e-8, proc_w)
            sy = real_h / max(1e-8, proc_h)

            if rescale_camera:
                pred_params = copy.deepcopy(pycamera.params)
                num_params = len(pred_params)
                # 尝试读取模型名（可能无此属性）
                model_name = getattr(pycamera, "model", "UNKNOWN")

                # 优先处理 PINHOLE（我们在构建时使用 camera_type="PINHOLE"）
                #   PINHOLE: [fx, fy, cx, cy]
                if str(model_name) == "PINHOLE" or num_params == 4:
                    pred_params[0] *= sx  # fx
                    pred_params[1] *= sy  # fy
                    pred_params[2] *= sx  # cx
                    pred_params[3] *= sy  # cy
                else:
                    # 其他模型做近似处理：
                    #   - 前两位当 fx、fy
                    #   - 若存在最后两位，当 cx、cy
                    if num_params >= 2:
                        pred_params[0] *= sx
                        pred_params[1] *= sy
                    if num_params >= 4:
                        pred_params[-2] *= sx
                        pred_params[-1] *= sy
                    # 畸变参数保持不变

                pycamera.params = pred_params
                pycamera.width = real_image_size[0]
                pycamera.height = real_image_size[1]

            if shift_point2d_to_original_res:
                # 将 points2D 从 (裁剪+缩放) 坐标恢复到原图坐标
                top_left = original_coords_np[pyimageid - 1, :2].astype(np.float32)  # [x1, y1]
                for p2d in pyimage.points2D:
                    # 先去掉裁剪偏移，再按轴缩放
                    shifted = np.array([p2d.xy[0] - top_left[0],
                                        p2d.xy[1] - top_left[1]], dtype=np.float32)
                    p2d.xy = shifted * np.array([sx, sy], dtype=np.float32)

            if shared_camera:
                # If shared_camera, all images share the same camera
                # No need to rescale any more
                rescale_camera = False

        return reconstruction

    def _rescale_reconstruction_to_original_size(
        self,
        reconstruction: pycolmap.Reconstruction,
        start_idx: int,
        end_idx: int,
    ) -> pycolmap.Reconstruction:
        """
        将reconstruction对齐到已知的影像pose位置
        
        Args:
            reconstruction: pycolmap重建结果
            start_idx: 起始影像索引
            end_idx: 结束影像索引
        
        Returns:
            对齐后的reconstruction
        """
        alignment_success = False

        # 方法1：直接通过3D点云配准到最新的SfM重建（简化版）
        if len(self.sfm_reconstructions) > 0:
            if self.verbose:
                print("  Attempting alignment to latest SfM via 3D point cloud (Sim3)...")

            sfm_result = self.sfm_reconstructions[-1]  # 直接使用最新的重建
            tgt_reconstruction = sfm_result['reconstruction']
            src_reconstruction = reconstruction
            num_tgt_images = len(tgt_reconstruction.images)
            num_src_images = len(src_reconstruction.images)
            sel_image_idx = 0
            if num_tgt_images != 0 and num_src_images != 0 and num_tgt_images == num_src_images:
                if num_tgt_images <=2:
                    sel_image_idx = list(range(1, num_tgt_images + 1))
                elif num_tgt_images == 3:
                    sel_image_idx = [1, 2, 3]
                else:
                    sel_image_idx = [1]
                    sel_image_idx.append((num_tgt_images + 1) // 2)
                    sel_image_idx. append(num_tgt_images)

            point_correspondences = []
            pixel_threshold = 3.0
            if sel_image_idx != 0:
                for index in sel_image_idx:
                    tgt_image_obj = tgt_reconstruction.images[index]
                    src_image_obj = src_reconstruction.images[index]

                    # 为src图像建立空间索引
                    src_spatial_index = {}
                    for point2D in src_image_obj.points2D:
                        if point2D.point3D_id != -1:
                            grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
                            if grid_key not in src_spatial_index:
                                src_spatial_index[grid_key] = []
                            # 保存浮点坐标便于距离计算
                            src_spatial_index[grid_key].append(
                                (int(point2D.point3D_id), np.asarray(point2D.xy, dtype=np.float64))
                            )

                    correspondences = self._find_single_images_pair_matches(
                        tgt_image_obj, 
                        src_image_obj, 
                        src_spatial_index, 
                        pixel_threshold, 
                    )
                    point_correspondences.extend(correspondences)

            if len(point_correspondences) == 0:
                print("  Warning: No point correspondences found between overlapping regions")
                return False

            tgt_pts3d = []
            src_pts3d = []
            for tgt_pt3d_id, src_pt3d_id, _ in point_correspondences:
                if tgt_pt3d_id in tgt_reconstruction.points3D and src_pt3d_id in src_reconstruction.points3D:
                    tgt_pts3d.append(tgt_reconstruction.points3D[tgt_pt3d_id].xyz)
                    src_pts3d.append(src_reconstruction.points3D[src_pt3d_id].xyz)
            
            # 需要至少3个点来估计Sim3变换
            if len(src_pts3d) >= 3 and len(tgt_pts3d) >= 3:
                # 取两个点云中较少的数量，进行配准
                min_points = min(len(src_pts3d), len(tgt_pts3d))
                src_pts3d = np.asarray(src_pts3d[:min_points], dtype=np.float64)
                tgt_pts3d = np.asarray(tgt_pts3d[:min_points], dtype=np.float64)

                # 估计 Sim3（src → tgt）
                try:
                    sim3_transform = self._estimate_sim3_transform(src_pts3d, tgt_pts3d)
                    if sim3_transform is not None:
                        reconstruction.transform(sim3_transform)
                        alignment_success = True
                        if self.verbose:
                            print(f"  ✓ Reconstruction aligned to latest SfM via 3D point cloud")
                            print(f"    Scale: {sim3_transform.scale:.6f}")
                            print(f"    Used 3D points: {len(src_pts3d)}")
                    else:
                        if self.verbose:
                            print("  Warning: Failed to compute Sim3 transform")
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Sim3 estimation failed: {e}")
                        import traceback; traceback.print_exc()
            else:
                if self.verbose:
                    print(f"  Warning: Not enough 3D points for Sim3 estimation (src={len(src_pts3d)}, tgt={len(tgt_pts3d)})")

        if not alignment_success and self.verbose:
            print("  No successful alignment with SfM reconstructions, falling back to GPS-based alignment...")

        # 方法2：如果重投影对齐失败或不可用，使用GPS位置对齐（原有方法）
        if not alignment_success:
            if self.verbose:
                print(f"  Using GPS-based alignment...")
        
            # 1. 提取reconstruction中的影像名称
            tgt_image_names = []
            for image_id, image in reconstruction.images.items():
                tgt_image_names.append(image.name)
            
            # 2. 从已知pose中提取对应的相机位置（世界坐标）
            tgt_locations = []
            valid_names = []

            for fidx, idx in enumerate(range(start_idx, end_idx)):
                extrinsic_info = self.ori_extrinsic[idx]
                # image_name = extrinsic_info['image_name']  # ← 不再使用这个实际文件名

                # 使用 reconstruction 中的影像名称格式
                image_name_in_reconstruction = f"image_{fidx + 1}"

                # 计算相机在世界坐标系中的位置
                R_camera = np.array(extrinsic_info['R_camera'])
                tvec = np.array(extrinsic_info['tvec'])

                # 相机位置 = -R^T @ t
                camera_center = -R_camera.T @ tvec
                
                tgt_locations.append(camera_center)
                valid_names.append(image_name_in_reconstruction)  
            
            if len(valid_names) == 0:
                print("  Warning: No matching images found for alignment")
                return reconstruction
            
            tgt_locations = np.array(tgt_locations, dtype=np.float64)

            # 添加 ransac_options 定义
            ransac_options = pycolmap.RANSACOptions()
            ransac_options.max_error = 5.0  # 最大误差阈值（米）
            ransac_options.min_inlier_ratio = 0.3  # 最小内点比例

            try:
                sim3d = pycolmap.align_reconstruction_to_locations(
                    src=reconstruction,
                    tgt_image_names=valid_names,
                    tgt_locations=tgt_locations,
                    min_common_points=3,  # ✓ 参数名是正确的
                    ransac_options=ransac_options  # ✓ 添加这个必需的参数
                )
                if sim3d is not None:
                    # 应用变换
                    reconstruction.transform(sim3d)
                    
                    if self.verbose:
                        print(f"  ✓ Reconstruction aligned to known poses")
                        print(f"    Scale: {sim3d.scale}")
                        print(f"    Number of aligned images: {len(valid_names)}")
                else:
                    print("  Warning: Failed to align reconstruction")
                    
            except Exception as e:
                print(f"  Error aligning reconstruction: {e}")
                import traceback
                traceback.print_exc()
            
        return reconstruction

    # def _rescale_reconstruction_to_original_size(
    #     self, 
    #     reconstruction: pycolmap.Reconstruction,
    #     start_idx: int,
    #     end_idx: int,
    # ) -> pycolmap.Reconstruction:
    #     """
    #     将reconstruction对齐到已知的影像pose位置
        
    #     Args:
    #         reconstruction: pycolmap重建结果
    #         start_idx: 起始影像索引
    #         end_idx: 结束影像索引
        
    #     Returns:
    #         对齐后的reconstruction
    #     """

    #     alignment_success = False

    #     # # 方法1：使用中间影像的像素位置对应进行对齐
    #     # if len(self.sfm_reconstructions) > 0:
    #     #     if self.verbose:
    #     #         print(f"  Attempting alignment with SfM reconstructions via middle image pixel correspondence...")

    #     #     INVALID_POINT3D_ID = (1 << 64) - 1
            
    #     #     def _valid_pid(pid):
    #     #         """检查3D点ID是否有效"""
    #     #         try:
    #     #             pid_int = int(pid)
    #     #         except Exception:
    #     #             return False
    #     #         return pid_int != -1 and pid_int != INVALID_POINT3D_ID

    #     #     # 找到与当前批次有重叠的SfM重建结果
    #     #     for sfm_result in self.sfm_reconstructions:
    #     #         sfm_start = sfm_result['start_idx']
    #     #         sfm_end = sfm_result['end_idx']

    #     #         # 检查是否有重叠区域
    #     #         has_overlap = not (end_idx <= sfm_start or start_idx >= sfm_end)
                
    #     #         if not has_overlap:
    #     #             continue
                
    #     #         tgt_reconstruction = sfm_result['reconstruction']
                
    #     #         if self.verbose:
    #     #             print(f"\n  === Aligning to SfM reconstruction [{sfm_start}:{sfm_end}] ===")
                
    #     #         # 计算重叠的图像索引范围
    #     #         overlap_start = max(start_idx, sfm_start)
    #     #         overlap_end = min(end_idx, sfm_end)
    #     #         overlap_count = overlap_end - overlap_start
                
    #     #         if overlap_count < 3:
    #     #             if self.verbose:
    #     #                 print(f"  Warning: Insufficient overlap ({overlap_count} images)")
    #     #             continue
                
    #     #         # ==================== 找到中间影像 ====================
    #     #         middle_idx = overlap_start + overlap_count // 2
    #     #         src_image_id = middle_idx - start_idx + 1
    #     #         tgt_image_id = middle_idx - sfm_start + 1
                
    #     #         if self.verbose:
    #     #             print(f"  Using middle image: global_idx={middle_idx}, src_id={src_image_id}, tgt_id={tgt_image_id}")
                
    #     #         # 检查中间影像是否存在
    #     #         if src_image_id not in reconstruction.images or tgt_image_id not in tgt_reconstruction.images:
    #     #             if self.verbose:
    #     #                 print(f"  Warning: Middle image not found in one of the reconstructions")
    #     #             continue
                
    #     #         src_image = reconstruction.images[src_image_id]
    #     #         tgt_image = tgt_reconstruction.images[tgt_image_id]
                
    #     #         if self.verbose:
    #     #             print(f"  Source image: {src_image.name} ({len(src_image.points2D)} 2D points)")
    #     #             print(f"  Target image: {tgt_image.name} ({len(tgt_image.points2D)} 2D points)")
                
    #     #         # ==================== 建立像素位置桶：grid(int,int) -> [(pt3d_id, xy_float)] ====================
    #     #         src_pixel_to_pt3d = {}
    #     #         tgt_pixel_to_pt3d = {}
    #     #         src_valid_count = 0
    #     #         tgt_valid_count = 0

    #     #         # 源影像：按整数像素网格分桶，但保存真实浮点坐标以做距离判断
    #     #         for p2d in src_image.points2D:
    #     #             if not _valid_pid(p2d.point3D_id):
    #     #                 continue
    #     #             grid_key = (int(round(p2d.xy[0])), int(round(p2d.xy[1])))
    #     #             if grid_key not in src_pixel_to_pt3d:
    #     #                 src_pixel_to_pt3d[grid_key] = []
    #     #             src_pixel_to_pt3d[grid_key].append((int(p2d.point3D_id), np.asarray(p2d.xy, dtype=np.float64)))
    #     #             src_valid_count += 1

    #     #         # 目标影像：同样分桶
    #     #         for p2d in tgt_image.points2D:
    #     #             if not _valid_pid(p2d.point3D_id):
    #     #                 continue
    #     #             grid_key = (int(round(p2d.xy[0])), int(round(p2d.xy[1])))
    #     #             if grid_key not in tgt_pixel_to_pt3d:
    #     #                 tgt_pixel_to_pt3d[grid_key] = []
    #     #             tgt_pixel_to_pt3d[grid_key].append((int(p2d.point3D_id), np.asarray(p2d.xy, dtype=np.float64)))
    #     #             tgt_valid_count += 1

    #     #         if self.verbose:
    #     #             print(f"  Source image valid 3D points: {src_valid_count}")
    #     #             print(f"  Target image valid 3D points: {tgt_valid_count}")
                
    #     #         # ==================== 在像素范围内找到3D点对应 ====================
    #     #         pixel_search_radius = 2.0  # 可调：1.0~3.0 常用
    #     #         window = int(np.ceil(pixel_search_radius))
    #     #         point_correspondences = []  # [(src_pt3d_id, tgt_pt3d_id)]
    #     #         used_tgt_ids = set()  # 可选：避免一个tgt点被多个src点重复匹配

    #     #         # 遍历源图像中的所有像素位置桶
    #     #         for src_pixel_key, src_pt_list in src_pixel_to_pt3d.items():
    #     #             sx, sy = src_pixel_key

    #     #             # 源桶内每个 2D-3D 点
    #     #             for src_pt3d_id, src_xy in src_pt_list:
    #     #                 best_match = None
    #     #                 best_dist = pixel_search_radius

    #     #                 # 在目标图像中，以整数像素为中心，搜索一个 (2*window+1)^2 的邻域
    #     #                 for dx in range(-window, window + 1):
    #     #                     for dy in range(-window, window + 1):
    #     #                         search_key = (sx + dx, sy + dy)
    #     #                         if search_key not in tgt_pixel_to_pt3d:
    #     #                             continue

    #     #                         # 遍历该邻域格子中的所有 2D-3D 点
    #     #                         for tgt_pt3d_id, tgt_xy in tgt_pixel_to_pt3d[search_key]:
    #     #                             # 若希望一一匹配，可跳过已被占用的目标点
    #     #                             if tgt_pt3d_id in used_tgt_ids:
    #     #                                 continue

    #     #                             # 真实像素距离判断与就近选择
    #     #                             dist = np.linalg.norm(src_xy - tgt_xy)
    #     #                             if dist < best_dist:
    #     #                                 best_dist = dist
    #     #                                 best_match = tgt_pt3d_id

    #     #                 if best_match is not None:
    #     #                     point_correspondences.append((src_pt3d_id, best_match))
    #     #                     used_tgt_ids.add(best_match)

    #     #         if self.verbose:
    #     #             print(f"  Found {len(point_correspondences)} point correspondences (search radius: {pixel_search_radius}px)")
                
    #     #         if len(point_correspondences) < 3:
    #     #             if self.verbose:
    #     #                 print(f"  Warning: Insufficient point correspondences ({len(point_correspondences)}) for alignment")
    #     #             continue
                
    #     #         # ==================== 提取对应的3D点坐标 ====================
    #     #         src_pts3d = []
    #     #         tgt_pts3d = []
    #     #         used_pairs = set()
    #     #         miss_src = 0
    #     #         miss_tgt = 0
                
    #     #         for src_pt3d_id, tgt_pt3d_id in point_correspondences:
    #     #             key = (src_pt3d_id, tgt_pt3d_id)
    #     #             if key in used_pairs:
    #     #                 continue
    #     #             used_pairs.add(key)

    #     #             # 尝试读取源3D点
    #     #             try:
    #     #                 src_pt = reconstruction.points3D[int(src_pt3d_id)]
    #     #             except Exception:
    #     #                 miss_src += 1
    #     #                 continue

    #     #             # 尝试读取目标3D点
    #     #             try:
    #     #                 tgt_pt = tgt_reconstruction.points3D[int(tgt_pt3d_id)]
    #     #             except Exception:
    #     #                 miss_tgt += 1
    #     #                 continue

    #     #             src_pts3d.append(np.asarray(src_pt.xyz, dtype=np.float64))
    #     #             tgt_pts3d.append(np.asarray(tgt_pt.xyz, dtype=np.float64))
                
    #     #         if self.verbose:
    #     #             print(f"  3D pair stats: total={len(point_correspondences)}, kept={len(src_pts3d)}, miss_src={miss_src}, miss_tgt={miss_tgt}")
                
    #     #         if len(src_pts3d) < 3:
    #     #             if self.verbose:
    #     #                 print(f"  Warning: Not enough valid 3D points ({len(src_pts3d)}) for alignment")
    #     #             continue
                
    #     #         src_pts3d = np.array(src_pts3d)
    #     #         tgt_pts3d = np.array(tgt_pts3d)
                
    #     #         # ==================== 计算Sim3变换 ====================
    #     #         try:
    #     #             # 使用Umeyama算法计算Sim3变换（src → tgt）
    #     #             sim3_transform = self._estimate_sim3_transform(src_pts3d, tgt_pts3d)
                    
    #     #             if sim3_transform is not None:
    #     #                 # 应用变换
    #     #                 reconstruction.transform(sim3_transform)
    #     #                 alignment_success = True
                        
    #     #                 if self.verbose:
    #     #                     print(f"  ✓ Reconstruction aligned via middle image pixel correspondence")
    #     #                     print(f"    Scale: {sim3_transform.scale:.6f}")
    #     #                     print(f"    Used {len(src_pts3d)} point pairs from middle image")
    #     #                     print(f"    Aligned to SfM reconstruction [{sfm_start}:{sfm_end}]")
                        
    #     #                 break  # 对齐成功，退出循环
    #     #             else:
    #     #                 if self.verbose:
    #     #                     print(f"  Warning: Failed to compute Sim3 transform")
                
    #     #         except Exception as e:
    #     #             if self.verbose:
    #     #                 print(f"  Warning: Sim3 computation failed: {e}")
    #     #                 import traceback
    #     #                 traceback.print_exc()

    #     #     if not alignment_success and self.verbose:
    #     #         print(f"  No successful alignment with SfM reconstructions, falling back to GPS-based alignment...")
    #     def build_grid_index(correspondences, grid_size=1.0):
    #         """
    #         构建网格索引以加速2D点查询
            
    #         Args:
    #             correspondences: 对应点列表
    #             grid_size: 网格单元大小（像素）
            
    #         Returns:
    #             grid_dict: 网格字典，key为(grid_x, grid_y)，value为该网格内的对应点列表
    #             bounds: 坐标范围 (min_x, min_y, max_x, max_y)
    #         """
    #         grid_dict = {}
            
    #         if not correspondences:
    #             return grid_dict, (0, 0, 0, 0)
            
    #         # 计算坐标范围
    #         all_xy = np.array([corr['xy'] for corr in correspondences])
    #         min_x, min_y = all_xy.min(axis=0)
    #         max_x, max_y = all_xy.max(axis=0)
            
    #         # 将每个点放入对应的网格
    #         for corr in correspondences:
    #             x, y = corr['xy']
    #             grid_x = int(np.floor(x / grid_size))
    #             grid_y = int(np.floor(y / grid_size))
    #             grid_key = (grid_x, grid_y)
                
    #             if grid_key not in grid_dict:
    #                 grid_dict[grid_key] = []
    #             grid_dict[grid_key].append(corr)
            
    #         return grid_dict, (min_x, min_y, max_x, max_y)


    #     def find_nearby_points_in_grid(tgt_xy, src_grid_dict, search_radius=1.0, grid_size=1.0):
    #         """
    #         在网格中查找目标点周围的源点
            
    #         Args:
    #             tgt_xy: 目标点坐标 [x, y]
    #             src_grid_dict: 源点网格索引
    #             search_radius: 搜索半径（像素）
    #             grid_size: 网格单元大小（像素）
            
    #         Returns:
    #             nearby_points: 在搜索范围内的源点列表
    #         """
    #         nearby_points = []
            
    #         # 计算需要检查的网格范围
    #         grid_x = int(np.floor(tgt_xy[0] / grid_size))
    #         grid_y = int(np.floor(tgt_xy[1] / grid_size))
            
    #         # 根据搜索半径确定需要检查的网格数量
    #         grid_range = int(np.ceil(search_radius / grid_size)) + 1
            
    #         # 检查周围网格
    #         for dx in range(-grid_range, grid_range + 1):
    #             for dy in range(-grid_range, grid_range + 1):
    #                 grid_key = (grid_x + dx, grid_y + dy)
    #                 if grid_key in src_grid_dict:
    #                     nearby_points.extend(src_grid_dict[grid_key])
            
    #         return nearby_points

    #     # 方法1：直接通过3D点云配准到最新的SfM重建（简化版）
    #     if len(self.sfm_reconstructions) > 0:
    #         if self.verbose:
    #             print("  Attempting alignment to latest SfM via 3D point cloud (Sim3)...")

    #         sfm_result = self.sfm_reconstructions[-1]  # 直接使用最新的重建
    #         tgt_reconstruction = sfm_result['reconstruction']
    #         src_reconstruction = reconstruction
    #         num_tgt_images = len(tgt_reconstruction.images)
    #         num_src_images = len(src_reconstruction.images)
    #         middle_tgt_image_idx = (num_tgt_images + 1) // 2
    #         middle_tgt_image = tgt_reconstruction.images[middle_tgt_image_idx]
    #         middle_src_image_idx = (num_src_images + 1) // 2
    #         middle_src_image = src_reconstruction.images[middle_src_image_idx]

    #         tgt_correspondences = []
    #         tgt_valid_point2d_count = 0
    #         src_correspondences = []
    #         src_valid_point2d_count = 0

    #         if middle_tgt_image_idx == middle_src_image_idx:
    #             for point2d_idx, point2d in enumerate(middle_tgt_image.points2D):
    #                 # 检查该2D点是否关联到3D点
    #                 if point2d.point3D_id != -1 and point2d.point3D_id != 18446744073709551615:
    #                     point3d_id = point2d.point3D_id
    #                     point3d = tgt_reconstruction.points3D[point3d_id]
    #                     correspondence = {
    #                         'point2d_id': point2d_idx,
    #                         'xy': point2d.xy.tolist(),  # 修改：使用 point2d.xy 而不是 xy
    #                         'point3d_id': point3d_id,
    #                         'xyz': point3d.xyz.tolist(),
    #                         'color': point3d.color.tolist() if point3d.color is not None else None,
    #                     }
    #                     tgt_correspondences.append(correspondence)
    #                     tgt_valid_point2d_count += 1
                
    #             for point2d_idx, point2d in enumerate(middle_src_image.points2D):
    #                 if point2d.point3D_id != -1 and point2d.point3D_id != 18446744073709551615:
    #                     point3d_id = point2d.point3D_id
    #                     point3d = src_reconstruction.points3D[point3d_id]
    #                     correspondence = {
    #                         'point2d_id': point2d_idx,
    #                         'xy': point2d.xy.tolist(),  # 修改：使用 point2d.xy 而不是 xy
    #                         'point3d_id': point3d_id,
    #                         'xyz': point3d.xyz.tolist(),
    #                         'color': point3d.color.tolist() if point3d.color is not None else None,
    #                     }
    #                     src_correspondences.append(correspondence)
    #                     src_valid_point2d_count += 1

    #         matched_correspondences = []
    #         search_radius = 5.0
    #         src_grid_dict, src_bounds = build_grid_index(src_correspondences, grid_size=1.0)
            
    #         for tgt_corr in tgt_correspondences:
    #             tgt_xy = np.array(tgt_corr['xy'])
                
    #             # 从网格中获取候选点
    #             candidate_src_corrs = find_nearby_points_in_grid(
    #                 tgt_xy, src_grid_dict, search_radius=search_radius, grid_size=1.0
    #             )

    #             # 计算所有候选点的距离
    #             distances_and_corrs = []
    #             for src_corr in candidate_src_corrs:
    #                 src_xy = np.array(src_corr['xy'])
    #                 distance = np.linalg.norm(tgt_xy - src_xy)
                    
    #                 if distance <= search_radius:
    #                     distances_and_corrs.append((distance, src_corr))
                
    #             # 如果有候选点，选择距离最小的那个
    #             if distances_and_corrs:
    #                 # 按距离排序，获取最小的
    #                 best_distance, best_src_corr = min(distances_and_corrs, key=lambda x: x[0])
                    
    #                 best_match = {
    #                     'tgt_point2d_id': tgt_corr['point2d_id'],
    #                     'tgt_xy': tgt_corr['xy'],
    #                     'tgt_point3d_id': tgt_corr['point3d_id'],
    #                     'tgt_xyz': tgt_corr['xyz'],
    #                     'src_point2d_id': best_src_corr['point2d_id'],
    #                     'src_xy': best_src_corr['xy'],
    #                     'src_point3d_id': best_src_corr['point3d_id'],
    #                     'src_xyz': best_src_corr['xyz'],
    #                     'distance': best_distance,
    #                 }
    #                 matched_correspondences.append(best_match)

    #         # 基于 matched_correspondences 提取成对的 3D 点（用于 Sim3）
    #         src_pts3d = []
    #         tgt_pts3d = []
    #         for m in matched_correspondences:
    #             src_pts3d.append(np.asarray(m['src_xyz'], dtype=np.float64))
    #             tgt_pts3d.append(np.asarray(m['tgt_xyz'], dtype=np.float64))

    #         # 需要至少3个点来估计Sim3变换
    #         if len(src_pts3d) >= 3 and len(tgt_pts3d) >= 3:
    #             # 取两个点云中较少的数量，进行配准
    #             min_points = min(len(src_pts3d), len(tgt_pts3d))
    #             src_pts3d = np.asarray(src_pts3d[:min_points], dtype=np.float64)
    #             tgt_pts3d = np.asarray(tgt_pts3d[:min_points], dtype=np.float64)

    #             # 估计 Sim3（src → tgt）
    #             try:
    #                 sim3_transform = self._estimate_sim3_transform(src_pts3d, tgt_pts3d)
    #                 if sim3_transform is not None:
    #                     reconstruction.transform(sim3_transform)
    #                     alignment_success = True
    #                     if self.verbose:
    #                         print(f"  ✓ Reconstruction aligned to latest SfM via 3D point cloud")
    #                         print(f"    Scale: {sim3_transform.scale:.6f}")
    #                         print(f"    Used 3D points: {len(src_pts3d)}")
    #                 else:
    #                     if self.verbose:
    #                         print("  Warning: Failed to compute Sim3 transform")
    #             except Exception as e:
    #                 if self.verbose:
    #                     print(f"  Warning: Sim3 estimation failed: {e}")
    #                     import traceback; traceback.print_exc()
    #         else:
    #             if self.verbose:
    #                 print(f"  Warning: Not enough 3D points for Sim3 estimation (src={len(src_pts3d)}, tgt={len(tgt_pts3d)})")

    #     if not alignment_success and self.verbose:
    #         print("  No successful alignment with SfM reconstructions, falling back to GPS-based alignment...")

    #     # 方法2：如果重投影对齐失败或不可用，使用GPS位置对齐（原有方法）
    #     if not alignment_success:
    #         if self.verbose:
    #             print(f"  Using GPS-based alignment...")
        
    #         # 1. 提取reconstruction中的影像名称
    #         tgt_image_names = []
    #         for image_id, image in reconstruction.images.items():
    #             tgt_image_names.append(image.name)
            
    #         # 2. 从已知pose中提取对应的相机位置（世界坐标）
    #         tgt_locations = []
    #         valid_names = []

    #         for fidx, idx in enumerate(range(start_idx, end_idx)):
    #             extrinsic_info = self.ori_extrinsic[idx]
    #             # image_name = extrinsic_info['image_name']  # ← 不再使用这个实际文件名

    #             # 使用 reconstruction 中的影像名称格式
    #             image_name_in_reconstruction = f"image_{fidx + 1}"

    #             # 计算相机在世界坐标系中的位置
    #             R_camera = np.array(extrinsic_info['R_camera'])
    #             tvec = np.array(extrinsic_info['tvec'])

    #             # 相机位置 = -R^T @ t
    #             camera_center = -R_camera.T @ tvec
                
    #             tgt_locations.append(camera_center)
    #             valid_names.append(image_name_in_reconstruction)  
            
    #         if len(valid_names) == 0:
    #             print("  Warning: No matching images found for alignment")
    #             return reconstruction
            
    #         tgt_locations = np.array(tgt_locations, dtype=np.float64)

    #         # 添加 ransac_options 定义
    #         ransac_options = pycolmap.RANSACOptions()
    #         ransac_options.max_error = 5.0  # 最大误差阈值（米）
    #         ransac_options.min_inlier_ratio = 0.3  # 最小内点比例

    #         try:
    #             sim3d = pycolmap.align_reconstruction_to_locations(
    #                 src=reconstruction,
    #                 tgt_image_names=valid_names,
    #                 tgt_locations=tgt_locations,
    #                 min_common_points=3,  # ✓ 参数名是正确的
    #                 ransac_options=ransac_options  # ✓ 添加这个必需的参数
    #             )
    #             if sim3d is not None:
    #                 # 应用变换
    #                 reconstruction.transform(sim3d)
                    
    #                 if self.verbose:
    #                     print(f"  ✓ Reconstruction aligned to known poses")
    #                     print(f"    Scale: {sim3d.scale}")
    #                     print(f"    Number of aligned images: {len(valid_names)}")
    #             else:
    #                 print("  Warning: Failed to align reconstruction")
                    
    #         except Exception as e:
    #             print(f"  Error aligning reconstruction: {e}")
    #             import traceback
    #             traceback.print_exc()
            
    #     return reconstruction

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

            correspondences = self._find_single_images_pair_matches(
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
        sim3_transform = self._estimate_sim3_transform(curr_pts3d, prev_pts3d)
        if sim3_transform is None:
            if self.verbose:
                print("  Warning: Failed to estimate Sim3 transform")
            return False
            
        # 应用变换到curr_recon
        curr_recon_aligned = pycolmap.Reconstruction(curr_recon)
        curr_recon_aligned.transform(sim3_transform)

        return curr_recon_aligned

    # def _align_current_reconstruction_by_point_cloud(
    #     self,
    #     reconstruction: pycolmap.Reconstruction,
    # ) -> bool:
    #     """
    #     Align current reconstruction to previous reconstruction using point cloud correspondences.
        
    #     Args:
    #         reconstruction: Current reconstruction to be aligned
            
    #     Returns:
    #         Aligned current reconstruction, or original reconstruction if alignment fails or no previous reconstruction exists
    #     """
    #     if len(self.inference_reconstructions) < 1:
    #         return reconstruction

    #     # 获取前一个reconstruction（因为当前的还没添加，所以前一个是最后一个）
    #     prev_recon_data = self.inference_reconstructions[-1]
    #     prev_recon = prev_recon_data['reconstruction']
    #     curr_recon = reconstruction  # 当前的就是传入的参数

    #     # 1. 获取重叠区域的影像ID，Reconstruction中影像ID是从1开始的。
    #     # prev_recon中倒数overlap个影像
    #     prev_overlap_image_ids = list(range(
    #         len(prev_recon.images) - self.overlap + 1,
    #         len(prev_recon.images) + 1
    #     ))
    #     # curr_recon中前overlap个影像
    #     curr_overlap_image_ids = list(range(
    #         1, self.overlap + 1
    #     ))

    #     # 2. 建立3D点对应关系
    #     # 在prev中找特征点，然后在curr的对应位置（1像素范围内）查找匹配
    #     point_correspondences = []  # [(prev_point3D_id, curr_point3D_id)]
    #     pixel_threshold = 0.5  # 0.5像素阈值

    #     # for prev_image_id, curr_image_id in zip(prev_overlap_image_ids, curr_overlap_image_ids):
    #     #     if prev_image_id not in prev_recon.images or curr_image_id not in curr_recon.images:
    #     #         continue
            
    #     #     prev_image = prev_recon.images[prev_image_id]
    #     #     curr_image = curr_recon.images[curr_image_id]
            
    #     #     # 为curr图像建立空间索引（使用取整后的坐标作为键）
    #     #     curr_spatial_index = {}  # {(int_x, int_y): [(point3D_id, xy)]}
    #     #     for point2D in curr_image.points2D:
    #     #         if point2D.point3D_id != -1:
    #     #             # 使用取整后的坐标作为网格索引
    #     #             grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
    #     #             if grid_key not in curr_spatial_index:
    #     #                 curr_spatial_index[grid_key] = []
    #     #             curr_spatial_index[grid_key].append((point2D.point3D_id, point2D.xy))
            
    #     #     # 对prev图像的每个特征点，在curr的对应位置附近查找匹配
    #     #     for point2D in prev_image.points2D:
    #     #         if point2D.point3D_id != -1:
    #     #             prev_xy = point2D.xy
    #     #             # 查找prev_xy附近1像素范围内的点
    #     #             center_grid = (int(round(prev_xy[0])), int(round(prev_xy[1])))
                    
    #     #             # 在3x3网格范围内搜索（覆盖1像素范围）
    #     #             found_match = False
    #     #             for dx in [-1, 0, 1]:
    #     #                 if found_match:
    #     #                     break
    #     #                 for dy in [-1, 0, 1]:
    #     #                     search_grid = (center_grid[0] + dx, center_grid[1] + dy)
    #     #                     if search_grid in curr_spatial_index:
    #     #                         for curr_pt3d_id, curr_xy in curr_spatial_index[search_grid]:
    #     #                             # 计算实际像素距离
    #     #                             dist = np.linalg.norm(prev_xy - curr_xy)
    #     #                             if dist < pixel_threshold:  # pixel_threshold像素阈值
    #     #                                 point_correspondences.append((point2D.point3D_id, curr_pt3d_id, dist))
    #     #                                 found_match = True
    #     #                                 break  # 找到一个匹配就停止

    #     for prev_image_id, curr_image_id in zip(prev_overlap_image_ids, curr_overlap_image_ids):
    #         if prev_image_id not in prev_recon.images or curr_image_id not in curr_recon.images:
    #             continue
            
    #         prev_image = prev_recon.images[prev_image_id]
    #         curr_image = curr_recon.images[curr_image_id]
            
    #         # 为curr图像建立空间索引
    #         curr_spatial_index = {}
    #         for point2D in curr_image.points2D:
    #             if point2D.point3D_id != -1:
    #                 grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
    #                 if grid_key not in curr_spatial_index:
    #                     curr_spatial_index[grid_key] = []
    #                 curr_spatial_index[grid_key].append((point2D.point3D_id, point2D.xy))
            
    #         # 对prev图像的每个特征点，查找最近的匹配
    #         for point2D in prev_image.points2D:
    #             if point2D.point3D_id != -1:
    #                 prev_xy = point2D.xy
    #                 center_grid = (int(round(prev_xy[0])), int(round(prev_xy[1])))
                    
    #                 # 【改进】收集所有候选匹配点
    #                 candidates = []  # [(curr_pt3d_id, dist)]
                    
    #                 # 在3x3网格范围内搜索
    #                 for dx in [-1, 0, 1]:
    #                     for dy in [-1, 0, 1]:
    #                         search_grid = (center_grid[0] + dx, center_grid[1] + dy)
    #                         if search_grid in curr_spatial_index:
    #                             for curr_pt3d_id, curr_xy in curr_spatial_index[search_grid]:
    #                                 # 计算实际像素距离
    #                                 dist = np.linalg.norm(prev_xy - curr_xy)
    #                                 if dist < pixel_threshold:
    #                                     candidates.append((curr_pt3d_id, dist))
                    
    #                 # 【关键】如果有候选点，选择距离最近的那个
    #                 if len(candidates) > 0:
    #                     # 按距离排序，选择最近的
    #                     best_match = min(candidates, key=lambda x: x[1])
    #                     curr_pt3d_id, dist = best_match
    #                     point_correspondences.append((point2D.point3D_id, curr_pt3d_id, dist))

    #     if len(point_correspondences) == 0:
    #         print("  Warning: No point correspondences found between overlapping regions")
    #         return False

    #     if self.verbose:
    #         print(f"    Found {len(point_correspondences)} point correspondences in overlap region")

    #     # 3. 根据对应关系计算变换
    #     # 收集对应的3D点坐标
    #     prev_pts3d = []
    #     curr_pts3d = []
    #     for prev_pt3d_id, curr_pt3d_id, _ in point_correspondences:
    #         if prev_pt3d_id in prev_recon.points3D and curr_pt3d_id in curr_recon.points3D:
    #             prev_pts3d.append(prev_recon.points3D[prev_pt3d_id].xyz)
    #             curr_pts3d.append(curr_recon.points3D[curr_pt3d_id].xyz)
        
    #     if len(prev_pts3d) < 3:
    #         print(f"  Warning: Not enough point correspondences ({len(prev_pts3d)}) for alignment")
    #         return False

    #     prev_pts3d = np.array(prev_pts3d)
    #     curr_pts3d = np.array(curr_pts3d)

    #     # 使用Umeyama算法计算Sim3变换
    #     sim3_transform = self._estimate_sim3_transform(curr_pts3d, prev_pts3d)
        
    #     # 应用变换到curr_recon
    #     curr_recon_aligned = pycolmap.Reconstruction(curr_recon)
    #     curr_recon_aligned.transform(sim3_transform)

    #     return curr_recon_aligned

    # def _align_current_reconstruction_by_point_cloud(
    #     self,
    #     reconstruction: pycolmap.Reconstruction,
    # ) -> bool:
    #     """
    #     Align current reconstruction to previous reconstruction using point cloud correspondences.
    #     使用双向匹配检查确保匹配质量。
        
    #     Args:
    #         reconstruction: Current reconstruction to be aligned
            
    #     Returns:
    #         Aligned current reconstruction, or original reconstruction if alignment fails or no previous reconstruction exists
    #     """
    #     if len(self.inference_reconstructions) < 1:
    #         return reconstruction

    #     # 获取前一个reconstruction
    #     prev_recon_data = self.inference_reconstructions[-1]
    #     prev_recon = prev_recon_data['reconstruction']
    #     curr_recon = reconstruction

    #     # 1. 获取重叠区域的影像ID
    #     prev_overlap_image_ids = list(range(
    #         len(prev_recon.images) - self.overlap + 1,
    #         len(prev_recon.images) + 1
    #     ))
    #     curr_overlap_image_ids = list(range(
    #         1, self.overlap + 1
    #     ))

    #     # 2. 建立双向3D点对应关系
    #     pixel_threshold = 1.0  # 可以适当放宽阈值
        
    #     # 存储双向匹配结果
    #     prev_to_curr_matches = {}  # {prev_pt3d_id: (curr_pt3d_id, dist)}
    #     curr_to_prev_matches = {}  # {curr_pt3d_id: (prev_pt3d_id, dist)}

    #     for prev_image_id, curr_image_id in zip(prev_overlap_image_ids, curr_overlap_image_ids):
    #         if prev_image_id not in prev_recon.images or curr_image_id not in curr_recon.images:
    #             continue
            
    #         prev_image = prev_recon.images[prev_image_id]
    #         curr_image = curr_recon.images[curr_image_id]
            
    #         # ========== 第一遍：prev → curr 找最近邻 ==========
    #         # 为curr图像建立空间索引
    #         curr_spatial_index = {}
    #         for point2D in curr_image.points2D:
    #             if point2D.point3D_id != -1:
    #                 grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
    #                 if grid_key not in curr_spatial_index:
    #                     curr_spatial_index[grid_key] = []
    #                 curr_spatial_index[grid_key].append((point2D.point3D_id, point2D.xy))
            
    #         # 对每个prev点找最近的curr点
    #         for point2D in prev_image.points2D:
    #             if point2D.point3D_id != -1:
    #                 prev_xy = point2D.xy
    #                 center_grid = (int(round(prev_xy[0])), int(round(prev_xy[1])))
                    
    #                 candidates = []
    #                 for dx in [-1, 0, 1]:
    #                     for dy in [-1, 0, 1]:
    #                         search_grid = (center_grid[0] + dx, center_grid[1] + dy)
    #                         if search_grid in curr_spatial_index:
    #                             for curr_pt3d_id, curr_xy in curr_spatial_index[search_grid]:
    #                                 dist = np.linalg.norm(prev_xy - curr_xy)
    #                                 if dist < pixel_threshold:
    #                                     candidates.append((curr_pt3d_id, dist))
                    
    #                 if len(candidates) > 0:
    #                     best_match = min(candidates, key=lambda x: x[1])
    #                     prev_pt3d_id = point2D.point3D_id
                        
    #                     # 如果这个prev点已经有匹配了，保留距离更小的
    #                     if prev_pt3d_id not in prev_to_curr_matches or best_match[1] < prev_to_curr_matches[prev_pt3d_id][1]:
    #                         prev_to_curr_matches[prev_pt3d_id] = best_match
            
    #         # ========== 第二遍：curr → prev 找最近邻 ==========
    #         # 为prev图像建立空间索引
    #         prev_spatial_index = {}
    #         for point2D in prev_image.points2D:
    #             if point2D.point3D_id != -1:
    #                 grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
    #                 if grid_key not in prev_spatial_index:
    #                     prev_spatial_index[grid_key] = []
    #                 prev_spatial_index[grid_key].append((point2D.point3D_id, point2D.xy))
            
    #         # 对每个curr点找最近的prev点
    #         for point2D in curr_image.points2D:
    #             if point2D.point3D_id != -1:
    #                 curr_xy = point2D.xy
    #                 center_grid = (int(round(curr_xy[0])), int(round(curr_xy[1])))
                    
    #                 candidates = []
    #                 for dx in [-1, 0, 1]:
    #                     for dy in [-1, 0, 1]:
    #                         search_grid = (center_grid[0] + dx, center_grid[1] + dy)
    #                         if search_grid in prev_spatial_index:
    #                             for prev_pt3d_id, prev_xy in prev_spatial_index[search_grid]:
    #                                 dist = np.linalg.norm(curr_xy - prev_xy)
    #                                 if dist < pixel_threshold:
    #                                     candidates.append((prev_pt3d_id, dist))
                    
    #                 if len(candidates) > 0:
    #                     best_match = min(candidates, key=lambda x: x[1])
    #                     curr_pt3d_id = point2D.point3D_id
                        
    #                     # 如果这个curr点已经有匹配了，保留距离更小的
    #                     if curr_pt3d_id not in curr_to_prev_matches or best_match[1] < curr_to_prev_matches[curr_pt3d_id][1]:
    #                         curr_to_prev_matches[curr_pt3d_id] = best_match

    #     # ========== 第三步：只保留双向一致的匹配 ==========
    #     point_correspondences = []  # [(prev_point3D_id, curr_point3D_id, dist)]
        
    #     for prev_pt3d_id, (curr_pt3d_id, dist) in prev_to_curr_matches.items():
    #         if curr_pt3d_id in curr_to_prev_matches:
    #             matched_prev_id, _ = curr_to_prev_matches[curr_pt3d_id]
    #             if matched_prev_id == prev_pt3d_id:  # 双向一致
    #                 point_correspondences.append((prev_pt3d_id, curr_pt3d_id, dist))

    #     if len(point_correspondences) == 0:
    #         print("  Warning: No point correspondences found between overlapping regions")
    #         return False

    #     if self.verbose:
    #         total_prev_matches = len(prev_to_curr_matches)
    #         total_curr_matches = len(curr_to_prev_matches)
    #         bidirectional_matches = len(point_correspondences)
    #         print(f"    Found {total_prev_matches} prev→curr matches, {total_curr_matches} curr→prev matches")
    #         print(f"    Bidirectional consistent matches: {bidirectional_matches} (threshold: {pixel_threshold}px)")

    #     # 3. 根据对应关系计算变换
    #     prev_pts3d = []
    #     curr_pts3d = []
    #     for prev_pt3d_id, curr_pt3d_id, _ in point_correspondences:
    #         if prev_pt3d_id in prev_recon.points3D and curr_pt3d_id in curr_recon.points3D:
    #             prev_pts3d.append(prev_recon.points3D[prev_pt3d_id].xyz)
    #             curr_pts3d.append(curr_recon.points3D[curr_pt3d_id].xyz)
        
    #     if len(prev_pts3d) < 3:
    #         print(f"  Warning: Not enough point correspondences ({len(prev_pts3d)}) for alignment")
    #         return False

    #     prev_pts3d = np.array(prev_pts3d)
    #     curr_pts3d = np.array(curr_pts3d)

    #     # 使用Umeyama算法计算Sim3变换
    #     sim3_transform = self._estimate_sim3_transform(curr_pts3d, prev_pts3d)
        
    #     # 应用变换到curr_recon
    #     curr_recon_aligned = pycolmap.Reconstruction(curr_recon)
    #     curr_recon_aligned.transform(sim3_transform)

    #     return curr_recon_aligned

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

    def _find_single_images_pair_matches(
        self,
        prev_image,
        curr_image,
        curr_spatial_index: dict,
        pixel_threshold: float,
    ) -> list:
        """
        在给定的 prev_image 与 curr_image 之间建立 3D 点对应关系（单对影像）。
        - 使用 curr_spatial_index（以整数像素为键，值为 [(point3D_id, xy_float), ...]）
        - 搜索邻域随像素阈值自适应 (window = ceil(pixel_threshold))
        - 使用 used_curr_ids 避免多个 prev 点匹配到同一个 curr 3D 点

        Returns:
            [(prev_point3D_id, curr_point3D_id, dist)]
        """
        correspondences = []

        search_radius = float(pixel_threshold)
        window = int(np.ceil(search_radius))
        used_curr_ids = set()  # 避免 curr 侧重复匹配

        for point2D in prev_image.points2D:
            prev_pt3d_id = int(point2D.point3D_id)
            if prev_pt3d_id == -1:
                continue

            prev_xy = np.asarray(point2D.xy, dtype=np.float64)
            cx, cy = int(round(prev_xy[0])), int(round(prev_xy[1]))

            best_curr_id = None
            best_dist = search_radius

            # 在 curr 图像索引中按窗口搜索邻域
            for dx in range(-window, window + 1):
                for dy in range(-window, window + 1):
                    search_key = (cx + dx, cy + dy)
                    if search_key not in curr_spatial_index:
                        continue

                    for curr_pt3d_id, curr_xy in curr_spatial_index[search_key]:
                        curr_pt3d_id = int(curr_pt3d_id)
                        if curr_pt3d_id in used_curr_ids:
                            continue

                        dist = float(np.linalg.norm(prev_xy - np.asarray(curr_xy, dtype=np.float64)))
                        if dist < best_dist and dist < search_radius:
                            best_dist = dist
                            best_curr_id = curr_pt3d_id

            if best_curr_id is not None:
                correspondences.append((prev_pt3d_id, best_curr_id, best_dist))
                used_curr_ids.add(best_curr_id)

        return correspondences
        
    def _estimate_sim3_transform(self, src_points: np.ndarray, tgt_points: np.ndarray) -> Optional[pycolmap.Sim3d]:
        """
        使用Umeyama算法估计Sim3变换
        
        Args:
            src_points: 源点云 (N, 3)
            tgt_points: 目标点云 (N, 3)
        
        Returns:
            pycolmap.Sim3d对象，如果失败则返回None
        """
        try:
            # 使用 Umeyama 算法计算相似变换
            scale, R, t = self._umeyama_alignment(src_points, tgt_points, with_scale=True)
            
            # 创建pycolmap.Sim3d对象
            rotation = pycolmap.Rotation3d(R)
            sim3d = pycolmap.Sim3d(scale, rotation, t)
            
            return sim3d
            
        except Exception as e:
            print(f"  Error estimating Sim3 transform: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _umeyama_alignment(self, src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
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
    
    def _merge_reconstruction_intermediate_results(self) -> bool:
        """
        合并reconstruction中间结果
        
        将最新的reconstruction与之前已合并的reconstruction合并，
        通过重叠影像进行对齐。合并结果存储在 self.merged_reconstruction 中。
        
        Returns:
            True if successful, False otherwise
        """
        # 如果这是第一个reconstruction，直接设置为merged
        if len(self.inference_reconstructions) == 1:
            self.merged_reconstruction = self.inference_reconstructions[0]['reconstruction']
            merged_recon = self.merged_reconstruction
            # 保存merged_reconstruction
            temp_path = self.output_dir / "temp_merged" / f"merged_reconstruction_{len(self.inference_reconstructions)}"
            temp_path.mkdir(parents=True, exist_ok=True)
            merged_recon.write_text(str(temp_path))
            merged_recon.export_PLY(str(temp_path / "points3D.ply"))
            return True

        # 获取前一个已合并的reconstruction和当前新的reconstruction
        prev_recon = self.merged_reconstruction
        curr_recon_data = self.inference_reconstructions[-1]
        curr_recon = curr_recon_data['reconstruction']

        # 1. 获取重叠区域的影像ID（Reconstruction中影像ID是从1开始的）
        # prev_recon中倒数overlap个影像
        prev_overlap_image_ids = list(range(
            len(prev_recon.images) - self.overlap + 1,
            len(prev_recon.images) + 1
        ))
        # curr_recon中前overlap个影像
        curr_overlap_image_ids = list(range(
            1, self.overlap + 1
        ))

        # 检查重叠影像数量是否匹配
        if len(prev_overlap_image_ids) != len(curr_overlap_image_ids):
            if self.verbose:
                print(f"  Warning: Overlapping image count mismatch")
            return False

        # 2. 在重叠区域建立3D点对应关系
        # 使用与_align_current_reconstruction_by_point_cloud类似的方法
        point_correspondences = {}  # {prev_point3D_id: [curr_point3D_id, ...]}
        
        for prev_image_id, curr_image_id in zip(prev_overlap_image_ids, curr_overlap_image_ids):
            if prev_image_id not in prev_recon.images or curr_image_id not in curr_recon.images:
                continue
            
            prev_image = prev_recon.images[prev_image_id]
            curr_image = curr_recon.images[curr_image_id]
            
            # 为curr图像建立空间索引（使用取整后的坐标作为键）
            curr_spatial_index = {}  # {(int_x, int_y): [(point3D_id, xy)]}
            for point2D in curr_image.points2D:
                if point2D.point3D_id != -1:
                    grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
                    if grid_key not in curr_spatial_index:
                        curr_spatial_index[grid_key] = []
                    curr_spatial_index[grid_key].append((point2D.point3D_id, point2D.xy))
            
            # 对prev图像的每个特征点，在curr的对应位置附近查找匹配
            for point2D in prev_image.points2D:
                if point2D.point3D_id != -1:
                    prev_xy = point2D.xy
                    center_grid = (int(round(prev_xy[0])), int(round(prev_xy[1])))
                    
                    # 在3x3网格范围内搜索（覆盖1像素范围）
                    found_match = False
                    for dx in [-1, 0, 1]:
                        if found_match:
                            break
                        for dy in [-1, 0, 1]:
                            search_grid = (center_grid[0] + dx, center_grid[1] + dy)
                            if search_grid in curr_spatial_index:
                                for curr_pt3d_id, curr_xy in curr_spatial_index[search_grid]:
                                    # 计算实际像素距离
                                    dist = np.linalg.norm(prev_xy - curr_xy)
                                    if dist < 1.0:  # 1像素阈值
                                        prev_pt3d_id = point2D.point3D_id
                                        if prev_pt3d_id not in point_correspondences:
                                            point_correspondences[prev_pt3d_id] = []
                                        point_correspondences[prev_pt3d_id].append(curr_pt3d_id)
                                        found_match = True
                                        break

        if len(point_correspondences) == 0:
            print("  Warning: No point correspondences found between overlapping regions")
            return False

        if self.verbose:
            print(f"    Found {len(point_correspondences)} 3D point correspondences in overlap region")

        # 3. 计算curr到prev的Sim3变换
        prev_pts3d = []
        curr_pts3d = []
        for prev_pt3d_id, curr_pt3d_ids in point_correspondences.items():
            if prev_pt3d_id in prev_recon.points3D:
                for curr_pt3d_id in curr_pt3d_ids:
                    if curr_pt3d_id in curr_recon.points3D:
                        prev_pts3d.append(prev_recon.points3D[prev_pt3d_id].xyz)
                        curr_pts3d.append(curr_recon.points3D[curr_pt3d_id].xyz)
                        break  # 只取第一个匹配的点
        
        if len(prev_pts3d) < 3:
            print(f"  Warning: Not enough point correspondences ({len(prev_pts3d)}) for alignment")
            return False

        prev_pts3d = np.array(prev_pts3d)
        curr_pts3d = np.array(curr_pts3d)

        # 3. 创建merged_reconstruction
        merged_recon = pycolmap.Reconstruction()

        # 3.1 添加相机模型
        for camera_id, camera in prev_recon.cameras.items():
            merged_recon.add_camera(camera)
        for camera_id, camera in curr_recon.cameras.items():
            if camera_id not in merged_recon.cameras:
                merged_recon.add_camera(camera)

        # 3.2 创建3D点ID映射
        prev_to_merged_point3D_map = {}
        curr_to_merged_point3D_map = {}
        curr_merged_point3D_ids = set()
        
        # 3.3 先添加所有3D点
        
        # 3.3.1 添加prev_recon的所有3D点，并合并重叠区域的对应点
        for prev_pt3d_id, prev_pt3d in prev_recon.points3D.items():
            if prev_pt3d_id in point_correspondences:
                for curr_pt3d_id in point_correspondences[prev_pt3d_id]:
                    if curr_pt3d_id in curr_recon.points3D:
                        curr_pt3d = curr_recon.points3D[curr_pt3d_id]
                        
                        merged_xyz = (prev_pt3d.xyz + curr_pt3d.xyz) / 2.0
                        merged_rgb = ((prev_pt3d.color.astype(np.float32) + 
                                    curr_pt3d.color.astype(np.float32)) / 2.0).astype(np.uint8)
                        
                        merged_pt3d_id = merged_recon.add_point3D(
                            xyz=merged_xyz,
                            track=pycolmap.Track(),
                            color=merged_rgb
                        )
                        
                        prev_to_merged_point3D_map[prev_pt3d_id] = merged_pt3d_id
                        curr_to_merged_point3D_map[curr_pt3d_id] = merged_pt3d_id
                        curr_merged_point3D_ids.add(curr_pt3d_id)
                        break
            else:
                merged_pt3d_id = merged_recon.add_point3D(
                    xyz=prev_pt3d.xyz,
                    track=pycolmap.Track(),
                    color=prev_pt3d.color
                )
                prev_to_merged_point3D_map[prev_pt3d_id] = merged_pt3d_id
        
        # 3.3.2 添加curr_recon中未被合并的3D点
        for curr_pt3d_id, curr_pt3d in curr_recon.points3D.items():
            if curr_pt3d_id not in curr_merged_point3D_ids:
                merged_pt3d_id = merged_recon.add_point3D(
                    xyz=curr_pt3d.xyz,
                    track=pycolmap.Track(),
                    color=curr_pt3d.color
                )
                curr_to_merged_point3D_map[curr_pt3d_id] = merged_pt3d_id
        
        # 3.4 添加影像并更新track信息（关键步骤！）
        
        # 3.4.1 添加prev_recon的所有影像
        merged_image_id = 1
        for image_id in sorted(prev_recon.images.keys()):
            prev_image = prev_recon.images[image_id]
            
            # 重新创建points2D，更新point3D_id映射
            new_points2D = []
            point2D_idx = 0  # 记录2D点在列表中的索引
            
            for point2D in prev_image.points2D:
                if point2D.point3D_id != -1 and point2D.point3D_id in prev_to_merged_point3D_map:
                    new_point3D_id = prev_to_merged_point3D_map[point2D.point3D_id]
                    new_points2D.append(pycolmap.Point2D(point2D.xy, new_point3D_id))
                    
                    # 【关键】更新3D点的track信息
                    track = merged_recon.points3D[new_point3D_id].track
                    track.add_element(merged_image_id, point2D_idx)
                    point2D_idx += 1
                else:
                    # 没有对应的3D点
                    new_points2D.append(pycolmap.Point2D(point2D.xy, -1))
                    point2D_idx += 1
            
            # 创建新Image对象
            new_image = pycolmap.Image(
                image_id=merged_image_id,
                name=prev_image.name,
                camera_id=prev_image.camera_id,
                cam_from_world=prev_image.cam_from_world,
                points2D=new_points2D
            )
            # 不设置 registered 属性
            merged_recon.add_image(new_image)
            merged_image_id += 1
        
        # 3.4.2 添加curr_recon的非重叠部分影像
        curr_non_overlap_image_ids = list(range(self.overlap + 1, len(curr_recon.images) + 1))
        for image_id in curr_non_overlap_image_ids:
            if image_id in curr_recon.images:
                curr_image = curr_recon.images[image_id]
                
                # 重新创建points2D，更新point3D_id映射
                new_points2D = []
                point2D_idx = 0
                
                for point2D in curr_image.points2D:
                    if point2D.point3D_id != -1 and point2D.point3D_id in curr_to_merged_point3D_map:
                        new_point3D_id = curr_to_merged_point3D_map[point2D.point3D_id]
                        new_points2D.append(pycolmap.Point2D(point2D.xy, new_point3D_id))
                        
                        # 【关键】更新3D点的track信息
                        track = merged_recon.points3D[new_point3D_id].track
                        track.add_element(merged_image_id, point2D_idx)
                        point2D_idx += 1
                    else:
                        new_points2D.append(pycolmap.Point2D(point2D.xy, -1))
                        point2D_idx += 1
                
                # 创建新Image对象
                new_image = pycolmap.Image(
                    image_id=merged_image_id,
                    name=curr_image.name,
                    camera_id=curr_image.camera_id,
                    cam_from_world=curr_image.cam_from_world,
                    points2D=new_points2D
                )
                # 不设置 registered 属性
                merged_recon.add_image(new_image)
                merged_image_id += 1

        # 4. 更新merged_reconstruction
        self.merged_reconstruction = merged_recon

        # if len(self.inference_reconstructions) > 6:  # 从第二次合并开始对齐
        #     merged_recon = self._align_merged_reconstruction_to_gps_poses(merged_recon)
        #     ba_options = pycolmap.BundleAdjustmentOptions()
        #     # ba_options.print_summary = False
        #     # ba_options.refine_extrinsics = False  # 保持GPS对齐的外参
        #     pycolmap.bundle_adjustment(merged_recon, ba_options)
        #     self.merged_reconstruction = merged_recon

        # 4.1 保存merged_reconstruction
        temp_path = self.output_dir / "temp_merged" / f"merged_reconstruction_{len(self.inference_reconstructions)}"
        temp_path.mkdir(parents=True, exist_ok=True)
        merged_recon.write_text(str(temp_path))
        merged_recon.export_PLY(str(temp_path / "points3D.ply"))

        if self.verbose:
            print(f"    Merged reconstruction: {len(merged_recon.images)} images, {len(merged_recon.points3D)} 3D points")
        
        return True

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

def run_incremental_feature_matching(
    image_paths: List[Path],
    output_dir: Path,
    image_interval: int = 8,
    min_images_for_scale: int = 6,
    overlap: int = 2,
    pred_vis_scores: float = 0.3, 
    max_reproj_error: float = 5.0,
    max_points3D_val: int = 5000,
    min_inlier_per_frame: int = 32,
    run_global_sfm_first: bool = True,
    verbose: bool = False,
) -> bool:
    """Run incremental image initialization pipeline.
    
    Args:
        image_paths: List of image file paths in processing order
        output_dir: Directory for output files
        image_interval: Interval for selecting images (1=all, 2=every 2nd, etc.)
        min_images_for_scale: Minimum number of images required for scale estimation
        overlap: Number of overlapping images between consecutive reconstructions
        pred_vis_scores: Minimum visibility threshold for feature tracking
        max_reproj_error: Maximum reprojection error for feature matching
        max_points3D_val: Maximum number of 3D points in the reconstruction
        min_inlier_per_frame: Minimum number of inliers per frame for feature matching
        run_global_sfm_first: Whether to run global SfM first
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
        # Create FeatureMatcherSfM instance
        global_sparse_matcher = FeatureMatcherSfM(
            input_dir=input_dir,
            output_dir=global_sfm_output_dir,
            imgsz=2048,
            num_features=8192,
            match_mode="spatial",
            num_neighbors=10,
            max_distance=500.0,
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
        global_sparse_reconstruction=global_sparse_reconstruction,
        min_images_for_scale=min_images_for_scale,
        overlap=overlap,
        pred_vis_scores=pred_vis_scores,
        max_reproj_error=max_reproj_error,
        max_points3D_val=max_points3D_val,
        min_inlier_per_frame=min_inlier_per_frame,
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
    
    # input_dir = Path(r"drone-map-anything\examples\Comprehensive_building_sel\images")
    # output_dir = Path(r"drone-map-anything\output\Comprehensive_building_sel\sparse_incremental_reconstruction")
    
    # input_dir = Path(r"drone-map-anything\examples\Ganluo_images\images")
    # output_dir = Path(r"drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction")
    
    # input_dir = Path(r"drone-map-anything\examples\Tazishan\images")
    # output_dir = Path(r"drone-map-anything\output\Tazishan\sparse_incremental_reconstruction")

    # input_dir = Path(r"drone-map-anything\examples\SWJTU_gongdi\images")
    # output_dir = Path(r"drone-map-anything\output\SWJTU_gongdi\sparse_incremental_reconstruction")

    input_dir = Path(r"drone-map-anything\examples\SWJTU_7th_teaching_building\images")
    output_dir = Path(r"drone-map-anything\output\SWJTU_7th_teaching_building\sparse_incremental_reconstruction")
    
    # Get all image files and sort them
    supported_formats = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = sorted([
        f for f in input_dir.glob("*") 
        if f.suffix in supported_formats
    ])
    
    print(f"Found {len(image_files)} images")
    
    # Run incremental initialization
    success = run_incremental_feature_matching(
        image_paths=image_files,
        output_dir=output_dir,
        verbose=True,
    )
    
    if success:
        print("\n✓ Image initialization completed successfully")
    else:
        print("\n✗ Image initialization failed")