#!/usr/bin/env python3
"""
SfM Reconstruction Visualization for Publication.

用于论文出图的 SfM 重建可视化工具。
使用 pycolmap 读取 COLMAP 重建结果，使用 Open3D 渲染点云和相机视锥体。

Features:
- 高质量点云渲染
- 相机视锥体可视化
- 支持多种视角预设（正交、俯视、等轴测等）
- 高分辨率图像导出
- 可自定义的美观配色

Author: Zhihao Jia
Date: 2025
"""

# ============================================================================
# 导入库
# ============================================================================
# 标准库
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# 第三方库
import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image


# ============================================================================
# 颜色主题配置（适合论文出图）
# ============================================================================
class ColorTheme:
    """颜色主题配置"""

    # 论文友好的颜色主题
    PAPER_LIGHT = {
        "background": [1.0, 1.0, 1.0],  # 白色背景
        "frustum_line": [0.2, 0.4, 0.8],  # 蓝色视锥体
        "frustum_fill": [0.6, 0.8, 1.0, 0.3],  # 淡蓝色填充
        "axis_x": [0.9, 0.2, 0.2],
        "axis_y": [0.2, 0.8, 0.2],
        "axis_z": [0.2, 0.2, 0.9],
    }

    PAPER_DARK = {
        "background": [0.05, 0.05, 0.1],  # 深色背景
        "frustum_line": [0.9, 0.6, 0.2],  # 橙色视锥体
        "frustum_fill": [1.0, 0.8, 0.4, 0.3],
        "axis_x": [1.0, 0.3, 0.3],
        "axis_y": [0.3, 1.0, 0.3],
        "axis_z": [0.3, 0.3, 1.0],
    }

    # 现代科技感主题
    TECH_BLUE = {
        "background": [0.02, 0.02, 0.06],
        "frustum_line": [0.0, 0.8, 1.0],  # 青色
        "frustum_fill": [0.0, 0.6, 0.8, 0.2],
        "axis_x": [1.0, 0.2, 0.4],
        "axis_y": [0.2, 1.0, 0.4],
        "axis_z": [0.2, 0.6, 1.0],
    }


# ============================================================================
# 核心可视化类
# ============================================================================
class SfMVisualizerO3D:
    """
    使用 Open3D 进行 SfM 重建可视化的类。

    支持:
    - 读取 COLMAP text/binary 格式
    - 渲染点云（保持原始颜色或统一颜色）
    - 渲染相机视锥体
    - 多种视角预设
    - 高分辨率截图导出
    """

    def __init__(
        self,
        reconstruction_path: str,
        theme: str = "paper_light",
        point_size: float = 2.0,
        frustum_scale: float = 1.0,
        frustum_color: Optional[List[float]] = None,
        frustum_line_width: float = 2.0,
        frustum_line_thickness: int = 0,
        show_frustums: bool = True,
        show_points: bool = True,
        viewpoint_file: Optional[str] = None,
        projection_mode: str = "perspective",
        ortho_zoom: float = 0.5,
        transparent_background: bool = False,
        verbose: bool = True,
    ):
        """
        初始化可视化器。

        Args:
            reconstruction_path: COLMAP 重建结果路径
            theme: 颜色主题 ('paper_light', 'paper_dark', 'tech_blue')
            point_size: 点的大小
            frustum_scale: 视锥体缩放比例
            frustum_color: 视锥体颜色 [R, G, B]，取值 0.0~1.0 (可选，默认使用主题颜色)
            frustum_line_width: 视锥体线宽 (在现代 OpenGL 上可能不生效)
            frustum_line_thickness: 多线叠加厚度 (0=禁用, 1~5=模拟粗线效果)
            show_frustums: 是否显示视锥体
            show_points: 是否显示点云
            viewpoint_file: 视角保存文件路径 (可选，默认保存到重建结果文件夹)
            projection_mode: 投影模式 ('perspective' 透视 | 'orthographic' 正射)
            ortho_zoom: 正射投影的缩放因子 (数值越小场景显示越大)
            transparent_background: 是否使用透明背景 (仅对 PNG 格式有效)
            verbose: 是否输出详细信息
        """
        self.reconstruction_path = Path(reconstruction_path)
        self.point_size = point_size
        self.frustum_scale = frustum_scale
        self.frustum_color = frustum_color  # 自定义颜色，None 则使用主题颜色
        self.frustum_line_width = frustum_line_width
        self.frustum_line_thickness = frustum_line_thickness
        self.show_frustums = show_frustums
        self.show_points = show_points
        self.projection_mode = projection_mode  # 'perspective' or 'orthographic'
        self.ortho_zoom = ortho_zoom
        self.transparent_background = transparent_background
        self.verbose = verbose
        
        # 透明背景使用的特殊颜色 (用于后期替换为透明)
        # 使用一个极少出现的颜色：纯品红 (magenta)
        self.chroma_key_color = [1.0, 0.0, 1.0]  # RGB: (255, 0, 255)

        # 设置颜色主题
        theme_map = {
            "paper_light": ColorTheme.PAPER_LIGHT,
            "paper_dark": ColorTheme.PAPER_DARK,
            "tech_blue": ColorTheme.TECH_BLUE,
        }
        self.theme = theme_map.get(theme, ColorTheme.PAPER_LIGHT)

        # 数据存储
        self.reconstruction: Optional[pycolmap.Reconstruction] = None
        self.point_cloud: Optional[o3d.geometry.PointCloud] = None
        self.frustum_geometries: List[o3d.geometry.LineSet] = []
        self.all_geometries: List = []

        # 场景边界
        self.scene_center: np.ndarray = np.zeros(3)
        self.scene_extent: float = 1.0
        
        # 保存的视角文件路径（可自定义）
        if viewpoint_file:
            self.saved_viewpoint_path = Path(viewpoint_file)
        else:
            self.saved_viewpoint_path = self.reconstruction_path / "saved_viewpoint.json"
        
        # 已保存的视角列表
        self.saved_viewpoints: Dict[str, dict] = {}
        self._load_saved_viewpoints()

    def _load_saved_viewpoints(self):
        """加载已保存的视角配置。"""
        if self.saved_viewpoint_path.exists():
            try:
                with open(self.saved_viewpoint_path, 'r', encoding='utf-8') as f:
                    self.saved_viewpoints = json.load(f)
                if self.verbose:
                    print(f"  ✓ 已加载 {len(self.saved_viewpoints)} 个保存的视角")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ 无法加载保存的视角: {e}")
                self.saved_viewpoints = {}
    
    def _save_viewpoints_to_file(self):
        """将视角保存到文件。"""
        try:
            with open(self.saved_viewpoint_path, 'w', encoding='utf-8') as f:
                json.dump(self.saved_viewpoints, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"  ✗ 保存视角失败: {e}")
            return False

    def save_current_viewpoint(self, vis, name: str = "custom", is_ortho: bool = False) -> bool:
        """
        保存当前视角。
        
        Args:
            vis: Open3D Visualizer 对象
            name: 视角名称
            is_ortho: 当前是否为正射投影模式
            
        Returns:
            是否保存成功
        """
        try:
            ctr = vis.get_view_control()
            
            # 如果是正射投影模式，先临时切换到透视模式再保存
            if is_ortho:
                for _ in range(50):
                    ctr.change_field_of_view(step=5.0)
                vis.poll_events()
                vis.update_renderer()
            
            # 获取当前视角参数
            cam_params = ctr.convert_to_pinhole_camera_parameters()
            
            # 提取关键参数
            viewpoint = {
                "extrinsic": cam_params.extrinsic.tolist(),
                "intrinsic_width": cam_params.intrinsic.width,
                "intrinsic_height": cam_params.intrinsic.height,
                "intrinsic_matrix": cam_params.intrinsic.intrinsic_matrix.tolist(),
                "was_ortho": is_ortho,  # 记录保存时的投影模式
            }
            
            # 如果之前是正射投影，恢复
            if is_ortho:
                for _ in range(50):
                    ctr.change_field_of_view(step=-5.0)
                vis.poll_events()
                vis.update_renderer()
            
            self.saved_viewpoints[name] = viewpoint
            self._save_viewpoints_to_file()
            
            print(f"\n  ✓ 视角 '{name}' 已保存到: {self.saved_viewpoint_path}")
            return True
            
        except Exception as e:
            print(f"\n  ✗ 保存视角失败: {e}")
            return False
    
    def load_viewpoint(self, vis, name: str = "custom", is_ortho: bool = False) -> bool:
        """
        加载已保存的视角。
        
        Args:
            vis: Open3D Visualizer 对象
            name: 视角名称
            is_ortho: 当前是否处于正射投影模式
            
        Returns:
            是否加载成功
        """
        if name not in self.saved_viewpoints:
            print(f"\n  ⚠ 视角 '{name}' 不存在")
            return False
        
        try:
            viewpoint = self.saved_viewpoints[name]
            
            ctr = vis.get_view_control()
            
            # 如果当前是正射投影模式，先切换到透视模式
            if is_ortho:
                # 增大 FOV 切换到透视模式
                for _ in range(50):
                    ctr.change_field_of_view(step=5.0)
                vis.poll_events()
                vis.update_renderer()
            
            # 获取当前相机参数作为基础
            cam_params = ctr.convert_to_pinhole_camera_parameters()
            
            # 设置外参矩阵
            cam_params.extrinsic = np.array(viewpoint["extrinsic"])
            
            # 应用参数
            ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            vis.poll_events()
            vis.update_renderer()
            
            # 如果之前是正射投影，恢复
            if is_ortho:
                # 减小 FOV 切换回正射模式
                for _ in range(50):
                    ctr.change_field_of_view(step=-5.0)
                vis.poll_events()
                vis.update_renderer()
            
            print(f"\n  ✓ 视角 '{name}' 已加载")
            return True
            
        except Exception as e:
            print(f"\n  ✗ 加载视角失败: {e}")
            return False

    def get_saved_viewpoint_names(self) -> List[str]:
        """获取所有已保存的视角名称。"""
        return list(self.saved_viewpoints.keys())

    def _make_background_transparent(self, image_path: str, tolerance: int = 10) -> bool:
        """
        将截图中的特殊背景色替换为透明。
        
        Args:
            image_path: 图像文件路径
            tolerance: 颜色匹配容差 (0-255)
            
        Returns:
            是否成功处理
        """
        try:
            img = Image.open(image_path)
            img = img.convert("RGBA")
            data = np.array(img)
            
            # 目标颜色 (chroma key color)
            target_color = np.array([255, 0, 255])  # magenta
            
            # 计算每个像素与目标颜色的距离
            diff = np.abs(data[:, :, :3].astype(np.int16) - target_color.astype(np.int16))
            mask = np.all(diff <= tolerance, axis=2)
            
            # 将匹配的像素设为透明
            data[mask, 3] = 0
            
            # 保存为 PNG (支持透明通道)
            result = Image.fromarray(data, 'RGBA')
            result.save(image_path)
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ 透明背景处理失败: {e}")
            return False

    def load_reconstruction(self) -> bool:
        """
        加载 COLMAP 重建结果。

        Returns:
            是否成功加载
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"加载重建结果: {self.reconstruction_path}")
            print(f"{'=' * 60}")

        try:
            # 尝试使用 pycolmap 加载
            self.reconstruction = pycolmap.Reconstruction(str(self.reconstruction_path))

            if self.verbose:
                print(f"  ✓ 相机数量: {len(self.reconstruction.cameras)}")
                print(f"  ✓ 图像数量: {len(self.reconstruction.images)}")
                print(f"  ✓ 3D 点数量: {len(self.reconstruction.points3D)}")

            return True

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")

            # 尝试直接读取 PLY 文件
            ply_path = self.reconstruction_path / "points3D.ply"
            if ply_path.exists():
                print(f"  尝试直接读取 PLY 文件: {ply_path}")
                self.point_cloud = o3d.io.read_point_cloud(str(ply_path))
                if self.verbose:
                    print(f"  ✓ 从 PLY 加载点云: {len(self.point_cloud.points)} 点")
                return True

            return False

    def _extract_point_cloud(self) -> o3d.geometry.PointCloud:
        """从 pycolmap 重建中提取点云。"""
        if self.point_cloud is not None:
            return self.point_cloud

        if self.reconstruction is None:
            return o3d.geometry.PointCloud()

        num_points = len(self.reconstruction.points3D)
        points = np.empty((num_points, 3), dtype=np.float64)
        colors = np.empty((num_points, 3), dtype=np.float64)

        for i, (point_id, point3D) in enumerate(self.reconstruction.points3D.items()):
            points[i] = point3D.xyz
            colors[i] = point3D.color / 255.0  # 归一化到 [0, 1]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def _create_camera_frustum(
        self,
        R: np.ndarray,
        t: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        scale: float = 1.0,
        color: Optional[List[float]] = None,
    ) -> o3d.geometry.LineSet:
        """
        创建相机视锥体的线框几何体。

        Args:
            R: 3x3 旋转矩阵 (cam2world)
            t: 3x1 平移向量 (cam2world)
            fx, fy: 焦距
            cx, cy: 主点
            width, height: 图像尺寸
            scale: 视锥体缩放
            color: 线条颜色 [R, G, B]

        Returns:
            Open3D LineSet 几何体
        """
        if color is None:
            # 优先使用自定义颜色，否则使用主题颜色
            color = self.frustum_color if self.frustum_color else self.theme["frustum_line"]

        # 计算视锥体的四个角点（在相机坐标系中，z=1 平面）
        # 归一化图像坐标
        z = scale  # 视锥体深度

        # 图像四角在相机坐标系中的方向
        corners_img = np.array(
            [
                [0, 0],  # 左上
                [width, 0],  # 右上
                [width, height],  # 右下
                [0, height],  # 左下
            ],
            dtype=np.float64,
        )

        # 反投影到相机坐标系
        corners_cam = np.zeros((4, 3))
        for i, (u, v) in enumerate(corners_img):
            corners_cam[i] = [(u - cx) / fx * z, (v - cy) / fy * z, z]

        # 相机中心
        cam_center = np.array([0, 0, 0])

        # 所有点：相机中心 + 4个角点
        points_cam = np.vstack([cam_center, corners_cam])

        # 转换到世界坐标系
        points_world = (R @ points_cam.T).T + t

        # 定义线段
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],  # 从中心到四角
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],  # 四角连线
        ]

        # 创建 LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])

        # 多线叠加模拟粗线效果
        if self.frustum_line_thickness > 0:
            # 计算偏移量（基于视锥体大小的比例）
            offset_base = scale * 0.002 * self.frustum_line_thickness
            
            # 在多个方向上添加偏移的线段
            all_points = [points_world]
            offsets = [
                np.array([1, 0, 0]), np.array([-1, 0, 0]),
                np.array([0, 1, 0]), np.array([0, -1, 0]),
                np.array([0, 0, 1]), np.array([0, 0, -1]),
                np.array([1, 1, 0]) / np.sqrt(2), np.array([-1, -1, 0]) / np.sqrt(2),
                np.array([1, -1, 0]) / np.sqrt(2), np.array([-1, 1, 0]) / np.sqrt(2),
            ]
            
            for offset_dir in offsets[:self.frustum_line_thickness * 2]:
                offset_points = points_world + offset_dir * offset_base
                all_points.append(offset_points)
            
            # 合并所有点和线段
            combined_points = np.vstack(all_points)
            combined_lines = []
            combined_colors = []
            num_original_points = len(points_world)
            
            for i, pts in enumerate(all_points):
                base_idx = i * num_original_points
                for line in lines:
                    combined_lines.append([line[0] + base_idx, line[1] + base_idx])
                    combined_colors.append(color)
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(combined_points)
            line_set.lines = o3d.utility.Vector2iVector(combined_lines)
            line_set.colors = o3d.utility.Vector3dVector(combined_colors)

        return line_set

    def _extract_frustums(self) -> List[o3d.geometry.LineSet]:
        """从重建中提取所有相机视锥体。"""
        if self.reconstruction is None:
            return []

        frustums = []

        for image_id, image in self.reconstruction.images.items():
            camera = self.reconstruction.cameras[image.camera_id]

            # 获取 world2cam 变换
            R_w2c = np.array(image.cam_from_world.rotation.matrix())
            t_w2c = np.array(image.cam_from_world.translation)

            # 转换为 cam2world
            R_c2w = R_w2c.T
            t_c2w = -R_w2c.T @ t_w2c

            # 获取相机内参
            params = camera.params
            # 获取相机模型名称（pycolmap 使用 camera.model.name）
            model_name = camera.model.name if hasattr(camera.model, 'name') else str(camera.model)
            
            if model_name in ["PINHOLE", "SIMPLE_PINHOLE"]:
                if model_name == "SIMPLE_PINHOLE":
                    fx = fy = params[0]
                    cx, cy = params[1], params[2]
                else:  # PINHOLE
                    fx, fy = params[0], params[1]
                    cx, cy = params[2], params[3]
            elif model_name in ["SIMPLE_RADIAL", "RADIAL"]:
                # SIMPLE_RADIAL: [f, cx, cy, k]
                # RADIAL: [f, cx, cy, k1, k2]
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_name == "OPENCV":
                # OPENCV: [fx, fy, cx, cy, k1, k2, p1, p2]
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            else:
                # 其他模型，使用默认值
                fx = fy = params[0] if len(params) > 0 else camera.width
                cx, cy = camera.width / 2, camera.height / 2

            frustum = self._create_camera_frustum(
                R_c2w,
                t_c2w,
                fx,
                fy,
                cx,
                cy,
                camera.width,
                camera.height,
                scale=self.frustum_scale,
            )
            frustums.append(frustum)

        return frustums

    def prepare_scene(self) -> List:
        """
        准备所有场景几何体。

        Returns:
            几何体列表
        """
        self.all_geometries = []

        # 1. 加载点云
        if self.show_points:
            self.point_cloud = self._extract_point_cloud()
            if len(self.point_cloud.points) > 0:
                self.all_geometries.append(self.point_cloud)

                # 计算场景边界
                points = np.asarray(self.point_cloud.points)
                self.scene_center = points.mean(axis=0)
                self.scene_extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

                if self.verbose:
                    print(f"\n场景信息:")
                    print(f"  中心: {self.scene_center}")
                    print(f"  范围: {self.scene_extent:.2f}")

        # 2. 创建视锥体
        if self.show_frustums and self.reconstruction is not None:
            self.frustum_geometries = self._extract_frustums()
            self.all_geometries.extend(self.frustum_geometries)
            if self.verbose:
                print(f"  视锥体数量: {len(self.frustum_geometries)}")

        # 3. 添加坐标轴（可选）
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=self.scene_extent * 0.1, origin=self.scene_center
        # )
        # self.all_geometries.append(coord_frame)

        return self.all_geometries

    def visualize_interactive(
        self, 
        load_viewpoint: Optional[str] = None, 
        screenshot_dir: Optional[str] = None,
        window_size: Tuple[int, int] = (1280, 720),
        screenshot_dpi: int = 300,
    ):
        """
        启动交互式可视化窗口。

        使用鼠标可以旋转、缩放和平移视角。
        
        快捷键 (组合键模式):
        - S → 1-9: 保存视角到 'custom_1' ~ 'custom_9' (先按S，再按数字)
        - L → 1-9: 加载视角 'custom_1' ~ 'custom_9' (先按L，再按数字)
        - P: 截图保存当前渲染结果
        - Q/ESC: 退出
        
        Args:
            load_viewpoint: 启动时加载的视角名称 (可选)
            screenshot_dir: 截图保存目录 (可选，默认使用输出目录)
            window_size: 窗口尺寸 (width, height)
            screenshot_dpi: 截图 DPI (默认 300，用于论文打印)
        """
        if not self.all_geometries:
            self.prepare_scene()

        if not self.all_geometries:
            print("没有可视化的几何体！")
            return

        if self.verbose:
            print(f"\n{'=' * 60}")
            print("启动交互式可视化...")
            print("  鼠标左键: 旋转")
            print("  鼠标右键: 平移")
            print("  滚轮: 缩放")
            print("  ─────────────────────────────────")
            print("  🔭 O: 切换投影模式 (透视/正射)")
            print("  ─────────────────────────────────")
            print("  💾 保存视角 (组合键):")
            print("     S → 1-9: 保存到 'custom_1' ~ 'custom_9'")
            print("     (先按 S，然后按数字键)")
            print("  📂 加载视角 (组合键):")
            print("     L → 1-9: 加载 'custom_1' ~ 'custom_9'")
            print("     (先按 L，然后按数字键)")
            print("  ─────────────────────────────────")
            print("  📷 P: 截图保存当前渲染")
            print("  ❌ Q/ESC: 退出")
            print(f"{'=' * 60}")
            print(f"  🔭 当前投影模式: {self.projection_mode}")
            
            # 显示已保存的视角
            if self.saved_viewpoints:
                print(f"  📌 已保存的视角: {', '.join(self.saved_viewpoints.keys())}")
            print()

        # 创建可视化器（使用带回调的版本）
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            window_name="SfM Reconstruction Visualization",
            width=window_size[0],
            height=window_size[1],
        )

        # 添加几何体
        for geom in self.all_geometries:
            vis.add_geometry(geom)

        # 设置渲染选项
        opt = vis.get_render_option()
        # 如果需要透明背景，使用 chroma key 颜色；否则使用主题背景色
        if self.transparent_background:
            opt.background_color = np.array(self.chroma_key_color)
        else:
            opt.background_color = np.array(self.theme["background"])
        opt.point_size = self.point_size
        opt.line_width = self.frustum_line_width

        # 保存 self 引用供回调使用
        visualizer_self = self
        
        # ========== 状态机：用于组合键检测 ==========
        # 模式: None (正常), 'save' (等待保存数字), 'load' (等待加载数字)
        key_state = {"mode": None}
        
        # 截图保存目录
        if screenshot_dir:
            screenshot_path = Path(screenshot_dir)
        else:
            screenshot_path = self.saved_viewpoint_path.parent
        screenshot_path.mkdir(parents=True, exist_ok=True)
        
        # 截图计数器
        screenshot_counter = {"count": 0}
        
        def reset_mode(vis):
            """重置模式"""
            key_state["mode"] = None
            return False
        
        def enter_save_mode(vis):
            """S键: 进入保存模式，等待数字键"""
            key_state["mode"] = "save"
            print("\n  ⌨️  [保存模式] 请按 1-9 选择槽位...")
            return False
        
        def enter_load_mode(vis):
            """L键: 进入加载模式，等待数字键"""
            key_state["mode"] = "load"
            print("\n  ⌨️  [加载模式] 请按 1-9 选择槽位...")
            return False
        
        def take_screenshot(vis):
            """P键: 截图保存 (带 DPI 设置和可选透明背景)"""
            screenshot_counter["count"] += 1
            filename = screenshot_path / f"screenshot_{screenshot_counter['count']:03d}.png"
            vis.capture_screen_image(str(filename), do_render=True)
            
            # 处理透明背景
            if visualizer_self.transparent_background:
                visualizer_self._make_background_transparent(str(filename))
            
            # 使用 PIL 设置 DPI
            try:
                img = Image.open(str(filename))
                img.save(str(filename), dpi=(screenshot_dpi, screenshot_dpi))
                transparency_info = " (透明背景)" if visualizer_self.transparent_background else ""
                print(f"\n  📷 截图已保存: {filename} ({screenshot_dpi} DPI){transparency_info}")
            except Exception as e:
                print(f"\n  📷 截图已保存: {filename} (DPI 设置失败: {e})")
            
            return False
        
        # 投影状态追踪 (用于保存/加载视角时正确处理正射投影)
        projection_state = {"is_ortho": self.projection_mode == "orthographic"}
        
        def make_number_callback(num):
            """创建数字键回调"""
            def callback(vis):
                mode = key_state["mode"]
                if mode == "save":
                    visualizer_self.save_current_viewpoint(vis, f"custom_{num}", is_ortho=projection_state["is_ortho"])
                    key_state["mode"] = None
                elif mode == "load":
                    visualizer_self.load_viewpoint(vis, f"custom_{num}", is_ortho=projection_state["is_ortho"])
                    key_state["mode"] = None
                else:
                    # 非组合键模式下，数字键不做任何操作
                    pass
                return False
            return callback
        
        # 注册 S/s 键 (ASCII 83, 115) - 进入保存模式
        vis.register_key_callback(83, enter_save_mode)
        vis.register_key_callback(115, enter_save_mode)
        
        # 注册 L/l 键 (ASCII 76, 108) - 进入加载模式
        vis.register_key_callback(76, enter_load_mode)
        vis.register_key_callback(108, enter_load_mode)
        
        # 注册 P/p 键 (ASCII 80, 112) - 截图
        vis.register_key_callback(80, take_screenshot)
        vis.register_key_callback(112, take_screenshot)
        
        # 注册数字键 1-9 (ASCII 49-57)
        for i in range(1, 10):
            vis.register_key_callback(48 + i, make_number_callback(i))
        
        # ESC 键 (ASCII 27) 也用于取消当前模式
        def escape_callback(vis):
            if key_state["mode"] is not None:
                print("\n  ❌ 操作已取消")
                key_state["mode"] = None
                return False
            return False
        vis.register_key_callback(27, escape_callback)
        
        # O/o 键 (ASCII 79, 111) - 切换投影模式
        def toggle_projection(vis):
            """O键: 切换透视/正射投影"""
            ctr = vis.get_view_control()
            projection_state["is_ortho"] = not projection_state["is_ortho"]
            
            if projection_state["is_ortho"]:
                # 切换到正射投影（通过将 FOV 减小到最小值来模拟）
                # Open3D 的正射投影通过将 FOV 设置为很小的值来实现
                for _ in range(50):  # 多次减小 FOV
                    ctr.change_field_of_view(step=-5.0)
                print("\n  🔭 已切换到: 正射投影 (Orthographic)")
            else:
                # 切换到透视投影（恢复默认 FOV）
                for _ in range(50):  # 多次增大 FOV
                    ctr.change_field_of_view(step=5.0)
                print("\n  🔭 已切换到: 透视投影 (Perspective)")
            
            vis.update_renderer()
            return False
        
        vis.register_key_callback(79, toggle_projection)   # O
        vis.register_key_callback(111, toggle_projection)  # o

        # 设置初始视角
        ctr = vis.get_view_control()
        
        # 如果指定了要加载的视角
        if load_viewpoint and load_viewpoint in self.saved_viewpoints:
            # 需要先渲染一帧才能设置相机参数
            vis.poll_events()
            vis.update_renderer()
            self.load_viewpoint(vis, load_viewpoint)
        else:
            ctr.set_zoom(0.8)
        
        # 设置初始投影模式
        if self.projection_mode == "orthographic":
            # 切换到正射投影
            for _ in range(50):
                ctr.change_field_of_view(step=-5.0)
            if self.verbose:
                print("  🔭 初始投影: 正射投影 (Orthographic)")

        # 运行
        vis.run()
        vis.destroy_window()

    def capture_views(
        self,
        output_dir: str,
        views: Optional[List[str]] = None,
        resolution: Tuple[int, int] = (1920, 1080),
        format: str = "png",
        dpi: int = 300,
    ):
        """
        捕获多个视角的高分辨率截图。

        Args:
            output_dir: 输出目录
            views: 视角列表，支持:
                   - 预设视角: 'front', 'top', 'iso', 'side', 'iso2'
                   - 保存的自定义视角: 'custom', 'custom_1', 'custom_2', ...
                   - 如果为 None，使用默认预设视角
            resolution: 输出分辨率 (width, height)
            format: 输出格式 ('png', 'jpg')
            dpi: 输出 DPI (默认 300，用于论文打印)
        """
        if not self.all_geometries:
            self.prepare_scene()

        # 预设视角列表
        preset_views = ["front", "top", "iso", "side", "iso2"]
        
        if views is None:
            views = preset_views

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"生成高分辨率截图...")
            print(f"  输出目录: {output_path}")
            print(f"  分辨率: {resolution[0]}x{resolution[1]}")
            print(f"  视角: {views}")
            if self.saved_viewpoints:
                print(f"  📌 可用的自定义视角: {', '.join(self.saved_viewpoints.keys())}")
            print(f"{'=' * 60}\n")

        for view_name in views:
            # 创建离屏渲染器
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name="Offscreen",
                width=resolution[0],
                height=resolution[1],
                visible=False,
            )

            # 添加几何体
            for geom in self.all_geometries:
                vis.add_geometry(geom)

            # 设置渲染选项
            opt = vis.get_render_option()
            # 如果需要透明背景，使用 chroma key 颜色
            if self.transparent_background:
                opt.background_color = np.array(self.chroma_key_color)
            else:
                opt.background_color = np.array(self.theme["background"])
            opt.point_size = self.point_size
            opt.line_width = self.frustum_line_width

            # 设置视角
            ctr = vis.get_view_control()
            
            # 先渲染一帧（某些视角设置需要）
            vis.poll_events()
            vis.update_renderer()
            
            # 判断是预设视角还是自定义视角
            if view_name in preset_views:
                self._set_view(ctr, view_name)
            elif view_name in self.saved_viewpoints:
                # 使用保存的自定义视角
                self._apply_saved_viewpoint(vis, view_name)
            else:
                print(f"  ⚠ 未知视角 '{view_name}'，使用默认视角")
                ctr.set_zoom(0.7)

            # 渲染
            vis.poll_events()
            vis.update_renderer()

            # 保存
            output_file = output_path / f"reconstruction_{view_name}.{format}"
            vis.capture_screen_image(str(output_file), do_render=True)
            
            # 处理透明背景
            if self.transparent_background and format.lower() == "png":
                self._make_background_transparent(str(output_file))
            
            # 设置 DPI
            try:
                img = Image.open(str(output_file))
                img.save(str(output_file), dpi=(dpi, dpi))
            except Exception:
                pass  # 静默忽略 DPI 设置失败

            if self.verbose:
                transparency_info = " (透明背景)" if self.transparent_background else ""
                print(f"  ✓ 保存: {output_file} ({dpi} DPI){transparency_info}")

            vis.destroy_window()
    
    def _apply_saved_viewpoint(self, vis, name: str) -> bool:
        """
        应用已保存的视角（用于离屏渲染）。
        
        Args:
            vis: Open3D Visualizer 对象
            name: 视角名称
            
        Returns:
            是否成功应用
        """
        if name not in self.saved_viewpoints:
            return False
        
        try:
            viewpoint = self.saved_viewpoints[name]
            
            # 恢复相机参数
            ctr = vis.get_view_control()
            cam_params = ctr.convert_to_pinhole_camera_parameters()
            
            # 设置外参矩阵
            cam_params.extrinsic = np.array(viewpoint["extrinsic"])
            
            # 应用参数
            ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ 应用视角 '{name}' 失败: {e}")
            return False

    def _set_view(self, view_control, view_name: str):
        """
        设置预定义视角。

        Args:
            view_control: Open3D ViewControl 对象
            view_name: 视角名称
        """
        # 先重置到默认视角
        view_control.set_zoom(0.7)

        if view_name == "front":
            # 正视图
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])
        elif view_name == "top":
            # 俯视图
            view_control.set_front([0, -1, 0])
            view_control.set_up([0, 0, -1])
        elif view_name == "side":
            # 侧视图
            view_control.set_front([-1, 0, 0])
            view_control.set_up([0, -1, 0])
        elif view_name == "iso":
            # 等轴测视图
            view_control.set_front([-0.577, -0.577, -0.577])
            view_control.set_up([0, -1, 0])
        elif view_name == "iso2":
            # 另一个等轴测视图
            view_control.set_front([0.577, -0.577, -0.577])
            view_control.set_up([0, -1, 0])

        view_control.set_lookat(self.scene_center)

    def save_point_cloud(self, output_path: str, format: str = "ply"):
        """
        保存点云到文件。

        Args:
            output_path: 输出路径
            format: 格式 ('ply', 'pcd', 'xyz')
        """
        if self.point_cloud is None:
            self.point_cloud = self._extract_point_cloud()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        o3d.io.write_point_cloud(str(output_file), self.point_cloud)
        if self.verbose:
            print(f"  ✓ 点云已保存: {output_file}")


def create_custom_visualization(
    reconstruction_path: str,
    output_dir: Optional[str] = None,
    theme: str = "paper_light",
    point_size: float = 2.0,
    frustum_scale: float = 1.0,
    interactive: bool = True,
    capture_views: bool = True,
    resolution: Tuple[int, int] = (1920, 1080),
):
    """
    快捷函数：创建并运行可视化。

    Args:
        reconstruction_path: COLMAP 重建路径
        output_dir: 截图输出目录 (可选)
        theme: 颜色主题
        point_size: 点大小
        frustum_scale: 视锥体大小
        interactive: 是否启动交互式窗口
        capture_views: 是否自动捕获多视角截图
        resolution: 截图分辨率
    """
    visualizer = SfMVisualizerO3D(
        reconstruction_path=reconstruction_path,
        theme=theme,
        point_size=point_size,
        frustum_scale=frustum_scale,
        show_frustums=True,
        show_points=True,
        verbose=True,
    )

    if not visualizer.load_reconstruction():
        print("无法加载重建结果！")
        return

    visualizer.prepare_scene()

    # 捕获截图
    if capture_views and output_dir:
        visualizer.capture_views(
            output_dir=output_dir,
            views=["front", "top", "iso", "side", "iso2"],
            resolution=resolution,
        )

    # 交互式可视化
    if interactive:
        visualizer.visualize_interactive()

    return visualizer


# ============================================================================
# 主程序入口
# ============================================================================
# 工作流程：
# 保存视角 → 按 S/1-9 → 写入到 viewpoints.json
# 加载视角 → 按 L 或设置 LOAD_VIEWPOINT_ON_START → 从 viewpoints.json 读取
# 截图时使用自定义视角 → 设置 VIEWS_TO_CAPTURE = ["custom"] → 也从 viewpoints.json 读取
# ============================================================================
# ⚙️ 直接配置区域 - 方便调试时直接修改参数
# ============================================================================
# 设置为 True 时，使用下面的配置；设置为 False 时，使用命令行参数
USE_DIRECT_CONFIG = True

# 📁 重建结果路径（包含 cameras.txt, images.txt, points3D.txt/ply 的文件夹）
# RECONSTRUCTION_PATH = r"D:\Github_code\drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction\temp_merged_reconstruction_georeferenced"
# RECONSTRUCTION_PATH = r"D:\Github_code\drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction\temp_merged\merged_2"
# RECONSTRUCTION_PATH = r"D:\Github_code\drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction\global_sfm1\enu"
# RECONSTRUCTION_PATH = r"D:\Github_code\drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction\recon_2_2_6_vggt74_dense_feature_points\temp_merged\merged_1"
RECONSTRUCTION_PATH = r"D:\Github_code\drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction\recon_2_2_6_vggt74_dense_feature_points\temp_merged_reconstruction_georeferenced"


# 📂 截图输出目录（设为 None 则保存到重建路径下的 visualization 文件夹）
OUTPUT_DIR = r"D:\Github_code\drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction\temp_paper_figures"

# 🎨 颜色主题: "paper_light" (白底) | "paper_dark" (深色) | "tech_blue" (科技蓝)
THEME = "paper_light"

# 🔘 点云点大小 (推荐: 1.0 ~ 3.0)
POINT_SIZE = 2.5

# 📷 视锥体缩放比例 (根据场景大小调整，推荐: 1.0 ~ 10.0)
FRUSTUM_SCALE = 35.0

# 🎨 视锥体颜色 [R, G, B]，取值 0.0~1.0 (设为 None 则使用主题默认颜色)
# FRUSTUM_COLOR = None  # 使用主题颜色
# FRUSTUM_COLOR = [0.2, 0.4, 0.8]   # 蓝色
FRUSTUM_COLOR = [1.0, 0.0, 0.0]   # 正红色
# FRUSTUM_COLOR = [0.9, 0.3, 0.2]   # 暗红色
# FRUSTUM_COLOR = [0.2, 0.8, 0.3]   # 绿色
# FRUSTUM_COLOR = [1.0, 0.6, 0.0]   # 橙色

# ✏️ 视锥体线宽 (注意: Open3D 的 line_width 在现代 OpenGL 上不生效)
FRUSTUM_LINE_WIDTH = 5.0

# 🔧 多线叠加模拟粗线 (设为 0 禁用，设为 1~5 增加线条密度来模拟粗线效果)
FRUSTUM_LINE_THICKNESS = 10  # 推荐 0~5，数值越大越粗

# 📐 窗口/截图尺寸 (width, height)
WINDOW_SIZE = (1280, 720)   # 小窗口，方便调试
# WINDOW_SIZE = (1920, 1080)  # 1080p
# WINDOW_SIZE = (3840, 2160)  # 4K

# ✅ 开关选项
SHOW_FRUSTUMS = True       # 是否显示相机视锥体
SHOW_POINTS = True         # 是否显示点云

# 🔭 投影模式: "perspective" (透视投影) | "orthographic" (正射投影)
# PROJECTION_MODE = "perspective"
PROJECTION_MODE = "orthographic"

# 📐 正射投影时的缩放因子 (数值越小，场景显示越大)
ORTHO_ZOOM = 0.5

# 🎯 启动时加载的视角 (设为 None 使用默认视角，设为 "custom_1" 加载保存的视角)
LOAD_VIEWPOINT_ON_START = None
# LOAD_VIEWPOINT_ON_START = "custom"  # 加载保存的 'custom' 视角

# 📍 视角保存文件路径 (设为 None 则保存到重建结果文件夹中的 saved_viewpoint.json)
# VIEWPOINT_FILE = None
VIEWPOINT_FILE = r"D:\Github_code\drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction\temp_paper_figures\viewpoints.json"

# 📷 交互式截图保存目录 (按 P 键时保存的位置，设为 None 则使用 OUTPUT_DIR)
SCREENSHOT_DIR = None  # 默认使用 OUTPUT_DIR

# 🖨️ 截图 DPI (用于论文打印，推荐 300 DPI)
SCREENSHOT_DPI = 300

# 🔲 透明背景 (截图时将背景替换为透明，仅对 PNG 格式有效)
TRANSPARENT_BACKGROUND = True


def main():
    """主程序入口"""
    
    if USE_DIRECT_CONFIG:
        # ==================== 使用直接配置 ====================
        print("\n" + "=" * 60)
        print("🔧 使用直接配置模式 (USE_DIRECT_CONFIG = True)")
        print("=" * 60)
        
        # 创建可视化器
        visualizer = SfMVisualizerO3D(
            reconstruction_path=RECONSTRUCTION_PATH,
            theme=THEME,
            point_size=POINT_SIZE,
            frustum_scale=FRUSTUM_SCALE,
            frustum_color=FRUSTUM_COLOR,
            frustum_line_width=FRUSTUM_LINE_WIDTH,
            frustum_line_thickness=FRUSTUM_LINE_THICKNESS,
            show_frustums=SHOW_FRUSTUMS,
            show_points=SHOW_POINTS,
            viewpoint_file=VIEWPOINT_FILE,
            projection_mode=PROJECTION_MODE,
            ortho_zoom=ORTHO_ZOOM,
            transparent_background=TRANSPARENT_BACKGROUND,
            verbose=True,
        )

        # 加载数据
        if not visualizer.load_reconstruction():
            print("无法加载重建结果！")
            sys.exit(1)

        # 准备场景
        visualizer.prepare_scene()

        # 交互式可视化 (按 P 键截图)
        screenshot_save_dir = SCREENSHOT_DIR or OUTPUT_DIR or str(Path(RECONSTRUCTION_PATH) / "visualization")
        visualizer.visualize_interactive(
            load_viewpoint=LOAD_VIEWPOINT_ON_START,
            screenshot_dir=screenshot_save_dir,
            window_size=WINDOW_SIZE,
            screenshot_dpi=SCREENSHOT_DPI,
        )
            
    else:
        # ==================== 使用命令行参数 ====================
        parser = argparse.ArgumentParser(
            description="SfM 重建可视化工具 - 用于论文出图",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  # 交互式可视化
  python visualize_sfm_reconstruction.py --input /path/to/reconstruction

  # 生成高分辨率截图
  python visualize_sfm_reconstruction.py --input /path/to/reconstruction --output ./figures --capture

  # 使用深色主题
  python visualize_sfm_reconstruction.py --input /path/to/reconstruction --theme paper_dark

  # 自定义分辨率和点大小
  python visualize_sfm_reconstruction.py --input /path/to/reconstruction --resolution 3840 2160 --point-size 3.0
            """,
        )

        parser.add_argument(
            "--input",
            "-i",
            type=str,
            required=True,
            help="COLMAP 重建结果路径（包含 cameras.txt, images.txt, points3D.txt/ply 的文件夹）",
        )

        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="截图输出目录",
        )

        parser.add_argument(
            "--theme",
            "-t",
            type=str,
            default="paper_light",
            choices=["paper_light", "paper_dark", "tech_blue"],
            help="颜色主题 (默认: paper_light)",
        )

        parser.add_argument(
            "--point-size",
            type=float,
            default=2.0,
            help="点云点大小 (默认: 2.0)",
        )

        parser.add_argument(
            "--frustum-scale",
            type=float,
            default=1.0,
            help="视锥体缩放比例 (默认: 1.0)",
        )
        
        parser.add_argument(
            "--frustum-color",
            type=float,
            nargs=3,
            default=None,
            metavar=("R", "G", "B"),
            help="视锥体颜色，RGB 取值 0.0~1.0 (如: 0.2 0.4 0.8)",
        )
        
        parser.add_argument(
            "--frustum-line-width",
            type=float,
            default=2.0,
            help="视锥体线宽 (默认: 2.0，在现代 OpenGL 上可能不生效)",
        )
        
        parser.add_argument(
            "--frustum-thickness",
            type=int,
            default=0,
            help="多线叠加厚度 (0=禁用, 1~5=模拟粗线效果)",
        )

        parser.add_argument(
            "--no-frustums",
            action="store_true",
            help="不显示相机视锥体",
        )

        parser.add_argument(
            "--no-points",
            action="store_true",
            help="不显示点云",
        )
        
        parser.add_argument(
            "--load-viewpoint",
            type=str,
            default=None,
            help="启动时加载的视角名称 (如 'custom_1', 'custom_2')",
        )
        
        parser.add_argument(
            "--viewpoint-file",
            type=str,
            default=None,
            help="视角保存文件路径 (默认保存到重建结果文件夹)",
        )
        
        parser.add_argument(
            "--window-size",
            type=int,
            nargs=2,
            default=[1280, 720],
            metavar=("WIDTH", "HEIGHT"),
            help="窗口/截图尺寸 (默认: 1280 720)",
        )
        
        parser.add_argument(
            "--projection",
            type=str,
            default="perspective",
            choices=["perspective", "orthographic"],
            help="投影模式: perspective (透视) | orthographic (正射)",
        )
        
        parser.add_argument(
            "--ortho-zoom",
            type=float,
            default=0.5,
            help="正射投影的缩放因子 (默认: 0.5)",
        )
        
        parser.add_argument(
            "--dpi",
            type=int,
            default=300,
            help="截图 DPI (默认: 300，用于论文打印)",
        )
        
        parser.add_argument(
            "--transparent",
            action="store_true",
            help="使用透明背景 (仅对 PNG 格式有效)",
        )

        args = parser.parse_args()

        # 创建可视化器
        visualizer = SfMVisualizerO3D(
            reconstruction_path=args.input,
            theme=args.theme,
            point_size=args.point_size,
            frustum_scale=args.frustum_scale,
            frustum_color=args.frustum_color,
            frustum_line_width=args.frustum_line_width,
            frustum_line_thickness=args.frustum_thickness,
            show_frustums=not args.no_frustums,
            show_points=not args.no_points,
            viewpoint_file=args.viewpoint_file,
            projection_mode=args.projection,
            ortho_zoom=args.ortho_zoom,
            transparent_background=args.transparent,
            verbose=True,
        )

        # 加载数据
        if not visualizer.load_reconstruction():
            print("无法加载重建结果！")
            sys.exit(1)

        # 准备场景
        visualizer.prepare_scene()

        # 交互式可视化 (按 P 键截图)
        screenshot_dir = args.output or str(Path(args.input) / "visualization")
        visualizer.visualize_interactive(
            load_viewpoint=args.load_viewpoint,
            screenshot_dir=screenshot_dir,
            window_size=tuple(args.window_size),
            screenshot_dpi=args.dpi,
        )


if __name__ == "__main__":
    main()

