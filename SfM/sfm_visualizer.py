#!/usr/bin/env python3
"""
SfM Visualization module using viser.
Provides real-time 3D visualization of camera poses and point clouds.
"""

import numpy as np
import pycolmap
import viser
import viser.transforms as tf
from typing import Optional, List, Dict
from pathlib import Path


class SfMVisualizer:
    """Viser-based 3D visualizer for SfM reconstruction.
    
    Provides real-time visualization of:
    - Camera frustums with thumbnails
    - Point clouds (aligned or merged mode)
    - GUI controls for point size, frustum scale, visibility
    """
    
    def __init__(
        self,
        visualization_mode: str = 'merged',  # 'aligned' | 'merged'
        verbose: bool = False,
    ):
        """Initialize the SfM visualizer.
        
        Args:
            visualization_mode: Point cloud visualization mode
                - 'aligned': Show point clouds per batch (incremental)
                - 'merged': Show unified merged point cloud
            verbose: Enable verbose logging
        """
        self.visualization_mode = visualization_mode
        self.verbose = verbose
        
        # Viser server and handles
        self.viser_server: Optional[viser.ViserServer] = None
        self.viser_frustum_handles: List = []
        self.viser_point_handles: List = []  # For 'aligned' mode
        
        # Point cloud data storage
        self.unified_point_clouds: List[Dict] = []  # For 'aligned' mode
        self.merged_point_cloud_handle = None  # For 'merged' mode
        self.merged_point_cloud: Optional[Dict] = None  # {'points': ndarray, 'colors': ndarray}
        self.merged_point_cloud_version: int = 0
        self._visualized_merged_version: int = 0
        
        # GUI controls (initialized in setup)
        self.gui_point_size = None
        self.gui_show_frustums = None
        self.gui_show_points = None
        self.gui_frustum_scale = None

    def setup(self) -> viser.ViserServer:
        """Setup viser visualization server.
        
        Returns:
            The viser server instance
        """
        self.viser_server = viser.ViserServer()

        # 添加客户端连接回调，设置相机参数
        @self.viser_server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            # 设置更大的渲染距离，防止点云在远距离消失
            client.camera.far = 5000.0  # 远裁剪平面
            client.camera.near = 0.1     # 近裁剪平面

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
            # 根据 visualization_mode 更新相应点云的大小
            if self.visualization_mode == 'aligned':
                for handle in self.viser_point_handles:
                    handle.point_size = self.gui_point_size.value
            else:  # merged
                if self.merged_point_cloud_handle is not None:
                    self.merged_point_cloud_handle.point_size = self.gui_point_size.value
        
        @self.gui_show_points.on_update
        def _(_):
            show = self.gui_show_points.value
            if self.visualization_mode == 'aligned':
                for handle in self.viser_point_handles:
                    handle.visible = show
            else:  # merged
                if self.merged_point_cloud_handle is not None:
                    self.merged_point_cloud_handle.visible = show
        
        @self.gui_frustum_scale.on_update
        def _(_):
            new_scale = self.gui_frustum_scale.value
            for handle in self.viser_frustum_handles:
                try:
                    handle.scale = new_scale
                except:
                    pass
        
        if self.verbose:
            print("✓ Viser visualization server started")
            print(f"  Open browser at: http://localhost:8080")
        
        return self.viser_server

    def add_batch_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        """Add a batch point cloud for 'aligned' mode visualization.
        
        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3) as uint8
        """
        self.unified_point_clouds.append({
            'points': points,
            'colors': colors,
        })

    def update_merged_point_cloud(self, reconstruction: pycolmap.Reconstruction):
        """Update merged point cloud from reconstruction (for 'merged' mode).
        
        Args:
            reconstruction: pycolmap Reconstruction object
        """
        if reconstruction is None or len(reconstruction.points3D) == 0:
            return
        
        num_points = len(reconstruction.points3D)
        merged_points = np.empty((num_points, 3), dtype=np.float32)
        merged_colors = np.empty((num_points, 3), dtype=np.uint8)
        
        for i, (point3D_id, point3D) in enumerate(reconstruction.points3D.items()):
            merged_points[i] = point3D.xyz
            merged_colors[i] = point3D.color  # RGB
        
        self.merged_point_cloud = {
            'points': merged_points,
            'colors': merged_colors,
        }
        self.merged_point_cloud_version += 1
        
        if self.verbose:
            print(f"  ✓ Updated merged point cloud for visualization: {num_points} points")

    def update_aligned_mode(self):
        """Update visualization in 'aligned' mode (incremental batch point clouds)."""
        if self.viser_server is None:
            return
            
        show_points = self.gui_show_points.value
        num_point_clouds = len(self.unified_point_clouds)
        num_visualized_point_clouds = len(self.viser_point_handles)
        
        if num_point_clouds > num_visualized_point_clouds:
            if self.verbose:
                print(f"  Adding {num_point_clouds - num_visualized_point_clouds} new batch point clouds...")
            
            for i in range(num_visualized_point_clouds, num_point_clouds):
                unified_pc = self.unified_point_clouds[i]
                points = unified_pc['points']
                colors = unified_pc['colors']
                
                if len(points) > 0:
                    point_handle = self.viser_server.scene.add_point_cloud(
                        f"/reconstruction/points_batch_{i:03d}",
                        points=points,
                        colors=colors,
                        point_size=self.gui_point_size.value,
                        point_shape="rounded",
                    )
                    point_handle.visible = show_points
                    self.viser_point_handles.append(point_handle)
                    
                    if self.verbose:
                        print(f"  ✓ Added point cloud for batch {i}: {len(points)} points")

    def update_merged_mode(self):
        """Update visualization in 'merged' mode (unified point cloud)."""
        if self.viser_server is None or self.merged_point_cloud is None:
            return
            
        show_points = self.gui_show_points.value
        current_version = self.merged_point_cloud_version
        
        if current_version > self._visualized_merged_version:
            points = self.merged_point_cloud['points']
            colors = self.merged_point_cloud['colors']
            
            if len(points) > 0:
                # 移除旧的 merged 点云（如果存在）
                if self.merged_point_cloud_handle is not None:
                    try:
                        self.merged_point_cloud_handle.remove()
                    except:
                        pass
                
                # 添加新的 merged 点云
                self.merged_point_cloud_handle = self.viser_server.scene.add_point_cloud(
                    "/reconstruction/points_merged",
                    points=points,
                    colors=colors,
                    point_size=self.gui_point_size.value,
                    point_shape="rounded",
                )
                self.merged_point_cloud_handle.visible = show_points
                
                if self.verbose:
                    print(f"  ✓ Merged point cloud updated: {len(points)} points")
                
                self._visualized_merged_version = current_version

    def add_camera_frustum_from_pose(
        self,
        index: int,
        camera_pose: np.ndarray,  # (4, 4) cam2world
        K: np.ndarray,  # (3, 3) intrinsics
        width: int,
        height: int,
        image_thumbnail: Optional[np.ndarray] = None,
        color: tuple = (255, 0, 0),
    ):
        """Add a camera frustum from pose matrix.
        
        Args:
            index: Camera index for naming
            camera_pose: 4x4 cam2world transformation matrix
            K: 3x3 intrinsic matrix
            width: Image width
            height: Image height
            image_thumbnail: Optional thumbnail image for frustum
            color: Frustum line color (RGB)
        """
        if self.viser_server is None or not self.gui_show_frustums.value:
            return
        
        fx = K[0, 0]
        fy = K[1, 1]
        fov_y = 2 * np.arctan2(height / 2, fy)
        aspect = width / height
        
        R_cam = camera_pose[:3, :3]
        t_cam = camera_pose[:3, 3]
        
        frustum_handle = self.viser_server.scene.add_camera_frustum(
            f"/reconstruction/camera_{index:03d}",
            fov=fov_y,
            aspect=aspect,
            scale=self.gui_frustum_scale.value,
            color=color,
            image=image_thumbnail,
            wxyz=tf.SO3.from_matrix(R_cam).wxyz,
            position=t_cam,
        )
        self.viser_frustum_handles.append(frustum_handle)

    def add_camera_frustum_from_pycolmap(
        self,
        pyimage: pycolmap.Image,
        pycamera: pycolmap.Camera,
        image_thumbnail: Optional[np.ndarray] = None,
        color: tuple = (255, 0, 0),
    ):
        """Add a camera frustum from pycolmap image and camera.
        
        Args:
            pyimage: pycolmap Image object
            pycamera: pycolmap Camera object
            image_thumbnail: Optional thumbnail image
            color: Frustum line color (RGB)
        """
        if self.viser_server is None or not self.gui_show_frustums.value:
            return
        
        # 从 pycolmap 提取相机位姿 (cam_from_world 是 world2cam)
        R_w2c = np.array(pyimage.cam_from_world.rotation.matrix())
        t_w2c = np.array(pyimage.cam_from_world.translation)
        
        # 转换为 cam2world (viser 需要 cam2world)
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        
        # 获取相机内参
        width = pycamera.width
        height = pycamera.height
        fx = pycamera.focal_length_x
        fy = pycamera.focal_length_y
        
        fov_y = 2 * np.arctan2(height / 2, fy)
        aspect = width / height
        
        image_id = pyimage.image_id
        
        frustum_handle = self.viser_server.scene.add_camera_frustum(
            f"/reconstruction/camera_merged_{image_id:03d}",
            fov=fov_y,
            aspect=aspect,
            scale=self.gui_frustum_scale.value,
            color=color,
            image=image_thumbnail,
            wxyz=tf.SO3.from_matrix(R_c2w).wxyz,
            position=t_c2w,
        )
        self.viser_frustum_handles.append(frustum_handle)

    def clear_frustums(self):
        """Clear all camera frustums."""
        for handle in self.viser_frustum_handles:
            try:
                handle.remove()
            except:
                pass
        self.viser_frustum_handles = []

    def update(
        self,
        recovered_inference_outputs: Optional[List[Dict]] = None,
        merged_reconstruction: Optional[pycolmap.Reconstruction] = None,
        input_views: Optional[List[Dict]] = None,
        image_paths: Optional[List[Path]] = None,
    ):
        """Update visualization with latest data.
        
        This function handles both 'aligned' and 'merged' visualization modes.
        
        Args:
            recovered_inference_outputs: List of recovered inference outputs (for aligned mode frustums)
            merged_reconstruction: Merged pycolmap reconstruction (for merged mode)
            input_views: List of input view dictionaries (for thumbnails)
            image_paths: List of image paths (for thumbnail lookup)
        """
        if self.viser_server is None:
            return
        
        # ==================== 1. Update point clouds ====================
        if self.visualization_mode == 'aligned':
            self.update_aligned_mode()
        elif self.visualization_mode == 'merged':
            self.update_merged_mode()
        
        # ==================== 2. Update camera frustums ====================
        if self.visualization_mode == 'aligned' and recovered_inference_outputs is not None:
            self._update_frustums_aligned_mode(recovered_inference_outputs, input_views)
        elif self.visualization_mode == 'merged' and merged_reconstruction is not None:
            self._update_frustums_merged_mode(merged_reconstruction, input_views, image_paths)
        else:
            if self.verbose:
                print(f"  No new cameras to visualize")

    def _update_frustums_aligned_mode(
        self,
        recovered_inference_outputs: List[Dict],
        input_views: Optional[List[Dict]] = None,
    ):
        """Update camera frustums in aligned mode."""
        num_images = len(recovered_inference_outputs)
        num_visualized = len(self.viser_frustum_handles)
        
        if num_images > num_visualized:
            if self.verbose:
                print(f"  Adding {num_images - num_visualized} new camera frustums (aligned mode)...")
            
            for i in range(num_visualized, num_images):
                recovered_output = recovered_inference_outputs[i]
                
                camera_pose = recovered_output['camera_poses'][0].cpu().numpy()  # (4, 4)
                K = np.array(recovered_output['camera_K'])  # (3, 3)
                width = recovered_output['image_width']
                height = recovered_output['image_height']
                
                # Get thumbnail
                img_thumbnail = None
                if input_views is not None and i < len(input_views):
                    img_thumbnail = input_views[i]['img'].cpu().numpy()
                    downsample = 4
                    img_thumbnail = img_thumbnail[::downsample, ::downsample]
                
                self.add_camera_frustum_from_pose(
                    index=i,
                    camera_pose=camera_pose,
                    K=K,
                    width=width,
                    height=height,
                    image_thumbnail=img_thumbnail,
                )
            
            if self.verbose:
                total_points = sum(len(pc['points']) for pc in self.unified_point_clouds)
                print(f"  ✓ Visualization updated: {num_images} cameras, {len(self.unified_point_clouds)} batches, {total_points} points (aligned mode)")

    def _update_frustums_merged_mode(
        self,
        merged_reconstruction: pycolmap.Reconstruction,
        input_views: Optional[List[Dict]] = None,
        image_paths: Optional[List[Path]] = None,
    ):
        """Update camera frustums in merged mode."""
        num_images = len(merged_reconstruction.images)
        num_visualized = len(self.viser_frustum_handles)
        
        if num_images > num_visualized:
            if self.verbose:
                print(f"  Adding {num_images - num_visualized} new camera frustums (merged mode)...")
            
            # 清除旧的 frustum handles（因为 merged 后 image_id 可能重新编号）
            self.clear_frustums()
            
            # 重新添加所有 merged_reconstruction 中的相机
            for image_id, pyimage in sorted(merged_reconstruction.images.items()):
                pycamera = merged_reconstruction.cameras[pyimage.camera_id]
                
                # 尝试通过图像名获取缩略图
                img_thumbnail = None
                if input_views is not None and image_paths is not None:
                    image_name = pyimage.name
                    for idx, path in enumerate(image_paths):
                        if path.name == image_name and idx < len(input_views):
                            img_thumbnail = input_views[idx]['img'].cpu().numpy()
                            downsample = 4
                            img_thumbnail = img_thumbnail[::downsample, ::downsample]
                            break
                
                self.add_camera_frustum_from_pycolmap(
                    pyimage=pyimage,
                    pycamera=pycamera,
                    image_thumbnail=img_thumbnail,
                )
            
            if self.verbose:
                total_points = len(self.merged_point_cloud['points']) if self.merged_point_cloud else 0
                print(f"  ✓ Visualization updated: {num_images} cameras (unique), {total_points} points (merged mode)")

    def is_ready(self) -> bool:
        """Check if visualizer is ready (server is running)."""
        return self.viser_server is not None

    def get_statistics(self) -> Dict:
        """Get visualization statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'mode': self.visualization_mode,
            'num_frustums': len(self.viser_frustum_handles),
        }
        
        if self.visualization_mode == 'aligned':
            stats['num_point_batches'] = len(self.unified_point_clouds)
            stats['total_points'] = sum(len(pc['points']) for pc in self.unified_point_clouds)
        else:
            stats['merged_points'] = len(self.merged_point_cloud['points']) if self.merged_point_cloud else 0
            stats['merged_version'] = self.merged_point_cloud_version
        
        return stats

