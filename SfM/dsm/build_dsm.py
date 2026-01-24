#!/usr/bin/env python3
"""
从密集点云生成数字表面模型 (DSM) 的工具模块。

支持从 PLY 或 LAS 格式的点云文件生成 GeoTIFF 格式的 DSM。

主要功能:
1. 读取点云数据（PLY/LAS 格式）
2. 创建规则网格并进行栅格化
3. 空洞填充（多种插值方法）
4. 输出为 GeoTIFF 格式

使用示例:
    python build_dsm.py --input points.ply --output dsm.tif --resolution 0.1
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import warnings

# 尝试导入可选依赖
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    warnings.warn("laspy not available. LAS file support disabled. Install with: pip install laspy")

try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False
    warnings.warn("plyfile not available. PLY file support disabled. Install with: pip install plyfile")

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn("rasterio not available. GeoTIFF export disabled. Install with: pip install rasterio")

try:
    from scipy import ndimage
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Advanced interpolation disabled. Install with: pip install scipy")

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False


class DSMBuilder:
    """
    数字表面模型 (DSM) 构建器
    
    从点云数据生成栅格化的数字表面模型。
    
    Attributes:
        resolution: 输出 DSM 的分辨率（单位与输入点云坐标一致，通常为米）
        nodata_value: 无数据值（用于标记空洞）
        interpolation_method: 空洞填充方法
        epsg_code: 输出 DSM 的 EPSG 坐标系代码
    """
    
    def __init__(
        self,
        resolution: float = 0.1,
        nodata_value: float = -9999.0,
        interpolation_method: str = "nearest",
        epsg_code: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        初始化 DSM 构建器
        
        Args:
            resolution: 输出 DSM 的分辨率（米），默认 0.1m (10cm)
            nodata_value: 无数据值，默认 -9999.0
            interpolation_method: 空洞填充方法，可选:
                - "nearest": 最近邻插值（快速，推荐）
                - "linear": 线性插值
                - "cubic": 三次插值（更平滑，但较慢）
                - "idw": 反距离加权插值
                - "none": 不进行空洞填充
            epsg_code: 输出 DSM 的 EPSG 坐标系代码，默认 None
            verbose: 是否输出详细日志
        """
        self.resolution = resolution
        self.nodata_value = nodata_value
        self.interpolation_method = interpolation_method
        self.epsg_code = epsg_code
        self.verbose = verbose
        
        # 验证插值方法
        valid_methods = ["nearest", "linear", "cubic", "idw", "none"]
        if interpolation_method not in valid_methods:
            raise ValueError(f"interpolation_method 必须是 {valid_methods} 之一")
        
        if interpolation_method in ["linear", "cubic"] and not SCIPY_AVAILABLE:
            warnings.warn(f"scipy 不可用，将使用 'nearest' 替代 '{interpolation_method}'")
            self.interpolation_method = "nearest"

    def load_point_cloud(
        self, 
        input_path: Union[str, Path]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        加载点云数据
        
        Args:
            input_path: 点云文件路径（支持 .ply 和 .las/.laz 格式）
            
        Returns:
            xyz: (N, 3) 点云坐标数组
            colors: (N, 3) RGB 颜色数组（如果有），否则为 None
        """
        input_path = Path(input_path)
        suffix = input_path.suffix.lower()
        
        if suffix == ".ply":
            return self._load_ply(input_path)
        elif suffix in [".las", ".laz"]:
            return self._load_las(input_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}. 支持 .ply, .las, .laz")
    
    def _load_ply(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """加载 PLY 格式点云"""
        if not PLYFILE_AVAILABLE:
            raise ImportError("plyfile 库不可用。请安装: pip install plyfile")
        
        if self.verbose:
            print(f"  正在加载 PLY 文件: {path}")
        
        plydata = PlyData.read(str(path))
        vertex = plydata['vertex']
        
        # 提取坐标
        x = np.asarray(vertex['x'])
        y = np.asarray(vertex['y'])
        z = np.asarray(vertex['z'])
        xyz = np.column_stack([x, y, z])
        
        # 尝试提取颜色
        colors = None
        try:
            r = np.asarray(vertex['red'])
            g = np.asarray(vertex['green'])
            b = np.asarray(vertex['blue'])
            colors = np.column_stack([r, g, b])
        except (ValueError, KeyError):
            pass
        
        if self.verbose:
            print(f"    加载了 {len(xyz):,} 个点")
        
        return xyz, colors
    
    def _load_las(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """加载 LAS/LAZ 格式点云"""
        if not LASPY_AVAILABLE:
            raise ImportError("laspy 库不可用。请安装: pip install laspy")
        
        if self.verbose:
            print(f"  正在加载 LAS 文件: {path}")
        
        las = laspy.read(str(path))
        
        # 提取坐标
        xyz = np.column_stack([las.x, las.y, las.z])
        
        # 尝试提取颜色
        colors = None
        try:
            # LAS 使用 16 位颜色，转换为 8 位
            r = (las.red / 256).astype(np.uint8)
            g = (las.green / 256).astype(np.uint8)
            b = (las.blue / 256).astype(np.uint8)
            colors = np.column_stack([r, g, b])
        except AttributeError:
            pass
        
        if self.verbose:
            print(f"    加载了 {len(xyz):,} 个点")
        
        return xyz, colors
    
    def build_dsm(
        self, 
        xyz: np.ndarray,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        aggregation: str = "max",
        auto_crop: bool = True,
        crop_margin: int = 5,
        boundary_mask: bool = True,
        boundary_alpha: float = 0.0,
        boundary_buffer: int = 10,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        从点云构建 DSM 栅格
        
        Args:
            xyz: (N, 3) 点云坐标数组
            bounds: 可选的边界范围 (xmin, ymin, xmax, ymax)，默认使用点云边界
            aggregation: 单元格内多点的聚合方式:
                - "max": 取最大值（DSM 通常使用）
                - "min": 取最小值（DTM 可能使用）
                - "mean": 取平均值
                - "median": 取中位数
                - "count": 点计数
            auto_crop: 是否自动裁剪到有效数据范围（包围盒裁剪），默认 True
            crop_margin: 裁剪时保留的边缘像素数，默认 5
            boundary_mask: 是否使用点云边界轮廓裁剪（非矩形），默认 True
            boundary_alpha: 边界轮廓参数:
                - 0: 使用凸包（convex hull）
                - >0: 使用凹包（concave hull），值越大边界越紧密贴合点云
            boundary_buffer: 边界向外扩展的像素数，默认 10
            
        Returns:
            dsm_array: (H, W) DSM 栅格数组
            metadata: 元数据字典，包含变换参数等
        """
        if self.verbose:
            print(f"\n  构建 DSM...")
            print(f"    分辨率: {self.resolution} 米")
            print(f"    聚合方式: {aggregation}")
            print(f"    插值方法: {self.interpolation_method}")
        
        # 确定边界
        if bounds is None:
            xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
            xmax, ymax = xyz[:, 0].max(), xyz[:, 1].max()
        else:
            xmin, ymin, xmax, ymax = bounds
        
        # 稍微扩展边界以包含边缘点
        padding = self.resolution
        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding
        
        # 计算栅格尺寸
        width = int(np.ceil((xmax - xmin) / self.resolution))
        height = int(np.ceil((ymax - ymin) / self.resolution))
        
        if self.verbose:
            print(f"    边界: X[{xmin:.2f}, {xmax:.2f}], Y[{ymin:.2f}, {ymax:.2f}]")
            print(f"    栅格尺寸: {width} x {height} 像素")
        
        # 初始化 DSM 数组
        dsm_array = np.full((height, width), self.nodata_value, dtype=np.float32)
        
        # 计算每个点所属的栅格单元
        col_indices = ((xyz[:, 0] - xmin) / self.resolution).astype(np.int32)
        row_indices = ((ymax - xyz[:, 1]) / self.resolution).astype(np.int32)  # Y 轴翻转
        
        # 过滤越界点
        valid_mask = (
            (col_indices >= 0) & (col_indices < width) &
            (row_indices >= 0) & (row_indices < height)
        )
        col_indices = col_indices[valid_mask]
        row_indices = row_indices[valid_mask]
        z_values = xyz[valid_mask, 2]
        
        if self.verbose:
            print(f"    有效点数: {valid_mask.sum():,} / {len(xyz):,}")
        
        # 使用不同的聚合方式
        if aggregation == "max":
            # 最大值聚合（标准 DSM）
            # 使用 numpy 高级索引进行快速聚合
            linear_indices = row_indices * width + col_indices
            unique_indices, inverse = np.unique(linear_indices, return_inverse=True)
            
            # 对每个唯一单元格计算最大 Z 值
            max_z = np.full(len(unique_indices), -np.inf, dtype=np.float32)
            np.maximum.at(max_z, inverse, z_values)
            
            # 填充到 DSM
            row_idx = unique_indices // width
            col_idx = unique_indices % width
            dsm_array[row_idx, col_idx] = max_z
            
        elif aggregation == "min":
            linear_indices = row_indices * width + col_indices
            unique_indices, inverse = np.unique(linear_indices, return_inverse=True)
            min_z = np.full(len(unique_indices), np.inf, dtype=np.float32)
            np.minimum.at(min_z, inverse, z_values)
            row_idx = unique_indices // width
            col_idx = unique_indices % width
            dsm_array[row_idx, col_idx] = min_z
            
        elif aggregation == "mean":
            linear_indices = row_indices * width + col_indices
            unique_indices, inverse = np.unique(linear_indices, return_inverse=True)
            sum_z = np.zeros(len(unique_indices), dtype=np.float64)
            count = np.zeros(len(unique_indices), dtype=np.int32)
            np.add.at(sum_z, inverse, z_values)
            np.add.at(count, inverse, 1)
            mean_z = (sum_z / count).astype(np.float32)
            row_idx = unique_indices // width
            col_idx = unique_indices % width
            dsm_array[row_idx, col_idx] = mean_z
            
        elif aggregation == "count":
            linear_indices = row_indices * width + col_indices
            unique_indices, inverse = np.unique(linear_indices, return_inverse=True)
            count = np.zeros(len(unique_indices), dtype=np.int32)
            np.add.at(count, inverse, 1)
            row_idx = unique_indices // width
            col_idx = unique_indices % width
            dsm_array[row_idx, col_idx] = count.astype(np.float32)
        
        # 应用点云边界轮廓掩码（非矩形裁剪）
        if boundary_mask:
            edge_mask = self._create_boundary_mask_from_points(
                xyz, xmin, ymin, xmax, ymax, width, height,
                alpha=boundary_alpha,
                buffer_pixels=boundary_buffer
            )
            # 将边界外的区域设为 nodata
            dsm_array[~edge_mask] = self.nodata_value
        
        # 统计空洞
        nodata_mask = dsm_array == self.nodata_value
        hole_count = nodata_mask.sum()
        hole_ratio = hole_count / (width * height) * 100
        
        if self.verbose:
            print(f"    空洞像素: {hole_count:,} ({hole_ratio:.1f}%)")
        
        # 空洞填充（只填充边界内的空洞）
        if self.interpolation_method != "none" and hole_count > 0:
            # 如果使用了边界掩码，只填充边界内的空洞
            if boundary_mask:
                # 只对边界内的空洞进行填充
                interior_holes = nodata_mask & edge_mask
                if interior_holes.any():
                    dsm_array = self._fill_holes(dsm_array, interior_holes)
            else:
                dsm_array = self._fill_holes(dsm_array, nodata_mask)
        
        # 自动裁剪到有效数据范围（包围盒裁剪）
        if auto_crop:
            dsm_array, xmin, ymin, xmax, ymax, width, height = self._auto_crop_dsm(
                dsm_array, xmin, ymin, xmax, ymax, crop_margin
            )
        
        # 构建元数据
        metadata = {
            'bounds': (xmin, ymin, xmax, ymax),
            'resolution': self.resolution,
            'width': width,
            'height': height,
            'nodata': self.nodata_value,
            'crs_epsg': self.epsg_code,
        }
        
        return dsm_array, metadata
    
    def _fill_holes(
        self, 
        dsm: np.ndarray, 
        nodata_mask: np.ndarray
    ) -> np.ndarray:
        """
        填充 DSM 中的空洞
        
        Args:
            dsm: DSM 栅格数组
            nodata_mask: 空洞掩码（True 表示空洞）
            
        Returns:
            填充后的 DSM 数组
        """
        if self.verbose:
            print(f"  正在填充空洞 (方法: {self.interpolation_method})...")
        
        if self.interpolation_method == "nearest":
            # 使用 scipy 的最近邻填充
            if SCIPY_AVAILABLE:
                # 距离变换 + 最近邻
                valid_mask = ~nodata_mask
                indices = ndimage.distance_transform_edt(
                    nodata_mask, 
                    return_distances=False, 
                    return_indices=True
                )
                dsm_filled = dsm[tuple(indices)]
            else:
                # 简单的迭代膨胀
                dsm_filled = self._iterative_dilation_fill(dsm, nodata_mask)
                
        elif self.interpolation_method in ["linear", "cubic"]:
            if not SCIPY_AVAILABLE:
                warnings.warn("scipy 不可用，使用最近邻插值")
                return self._fill_holes_nearest(dsm, nodata_mask)
            
            # 使用 griddata 进行插值
            valid_mask = ~nodata_mask
            rows, cols = np.where(valid_mask)
            values = dsm[valid_mask]
            
            # 创建插值网格
            all_rows, all_cols = np.where(nodata_mask)
            
            if len(all_rows) > 0 and len(rows) > 3:
                points = np.column_stack([cols, rows])
                xi = np.column_stack([all_cols, all_rows])
                
                # 使用 griddata 插值
                filled_values = griddata(
                    points, values, xi, 
                    method=self.interpolation_method,
                    fill_value=self.nodata_value
                )
                
                dsm_filled = dsm.copy()
                dsm_filled[all_rows, all_cols] = filled_values
            else:
                dsm_filled = dsm.copy()
                
        elif self.interpolation_method == "idw":
            # 反距离加权插值
            dsm_filled = self._idw_interpolation(dsm, nodata_mask)
        else:
            dsm_filled = dsm.copy()
        
        return dsm_filled
    
    def _auto_crop_dsm(
        self,
        dsm: np.ndarray,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        margin: int = 5,
    ) -> Tuple[np.ndarray, float, float, float, float, int, int]:
        """
        自动裁剪 DSM 到有效数据范围（包围盒裁剪）
        
        识别有数据覆盖的区域边界，裁剪掉纯空洞区域。
        
        Args:
            dsm: DSM 栅格数组
            xmin, ymin, xmax, ymax: 当前边界坐标
            margin: 边缘保留的像素数
            
        Returns:
            cropped_dsm: 裁剪后的 DSM 数组
            new_xmin, new_ymin, new_xmax, new_ymax: 新的边界坐标
            new_width, new_height: 新的尺寸
        """
        height, width = dsm.shape
        
        # 找到有效数据的掩码（非 nodata 值）
        valid_mask = dsm != self.nodata_value
        
        if not valid_mask.any():
            # 没有有效数据，返回原始数据
            if self.verbose:
                print("    ⚠ 没有有效数据，跳过裁剪")
            return dsm, xmin, ymin, xmax, ymax, width, height
        
        # 找到有效数据的边界行和列
        valid_rows = np.any(valid_mask, axis=1)
        valid_cols = np.any(valid_mask, axis=0)
        
        row_min = np.argmax(valid_rows)
        row_max = len(valid_rows) - np.argmax(valid_rows[::-1]) - 1
        col_min = np.argmax(valid_cols)
        col_max = len(valid_cols) - np.argmax(valid_cols[::-1]) - 1
        
        # 添加边缘余量
        row_min = max(0, row_min - margin)
        row_max = min(height - 1, row_max + margin)
        col_min = max(0, col_min - margin)
        col_max = min(width - 1, col_max + margin)
        
        # 计算新的边界坐标
        # 注意：行索引与 Y 坐标方向相反（Y 轴翻转）
        new_xmin = xmin + col_min * self.resolution
        new_xmax = xmin + (col_max + 1) * self.resolution
        new_ymax = ymax - row_min * self.resolution
        new_ymin = ymax - (row_max + 1) * self.resolution
        
        # 裁剪 DSM
        cropped_dsm = dsm[row_min:row_max + 1, col_min:col_max + 1].copy()
        new_height, new_width = cropped_dsm.shape
        
        if self.verbose:
            original_size = height * width
            new_size = new_height * new_width
            reduction = (1 - new_size / original_size) * 100
            print(f"    包围盒裁剪: {width}x{height} -> {new_width}x{new_height} (减少 {reduction:.1f}%)")
        
        return cropped_dsm, new_xmin, new_ymin, new_xmax, new_ymax, new_width, new_height
    
    def _create_boundary_mask_from_points(
        self,
        xyz: np.ndarray,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        width: int,
        height: int,
        alpha: float = 0.0,
        buffer_pixels: int = 5,
    ) -> np.ndarray:
        """
        基于点云边缘轮廓创建掩码
        
        使用点云的凸包或凹包（alpha shape）来生成边界掩码，
        只保留点云实际覆盖区域内的 DSM 值。
        
        Args:
            xyz: (N, 3) 点云坐标
            xmin, ymin, xmax, ymax: DSM 边界坐标
            width, height: DSM 栅格尺寸
            alpha: Alpha shape 参数:
                - 0: 使用凸包（convex hull）
                - >0: 使用凹包（concave hull / alpha shape），值越大越紧密
            buffer_pixels: 边界向外扩展的像素数
            
        Returns:
            mask: (height, width) bool 数组，True 表示在边界内
        """
        try:
            from scipy.spatial import ConvexHull, Delaunay
            import cv2
            HAS_CV2 = True
        except ImportError:
            HAS_CV2 = False
            if self.verbose:
                print("    ⚠ 需要 opencv-python 和 scipy 来创建边界掩码")
            return np.ones((height, width), dtype=bool)
        
        if len(xyz) < 3:
            return np.ones((height, width), dtype=bool)
        
        # 提取 2D 点（只使用 X, Y）
        points_2d = xyz[:, :2].copy()
        
        if self.verbose:
            print(f"    创建边界掩码 (alpha={alpha}, buffer={buffer_pixels}px)...")
        
        # 将世界坐标转换为像素坐标
        pixel_x = ((points_2d[:, 0] - xmin) / self.resolution).astype(np.float32)
        pixel_y = ((ymax - points_2d[:, 1]) / self.resolution).astype(np.float32)  # Y 轴翻转
        
        # 过滤越界点
        valid = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
        pixel_x = pixel_x[valid]
        pixel_y = pixel_y[valid]
        
        if len(pixel_x) < 3:
            return np.ones((height, width), dtype=bool)
        
        pixel_points = np.column_stack([pixel_x, pixel_y])
        
        if alpha == 0:
            # 使用凸包
            try:
                hull = ConvexHull(pixel_points)
                hull_points = pixel_points[hull.vertices]
            except Exception:
                # 凸包计算失败，返回全部区域
                return np.ones((height, width), dtype=bool)
        else:
            # 使用凹包 (alpha shape)
            hull_points = self._compute_alpha_shape(pixel_points, alpha)
            if hull_points is None or len(hull_points) < 3:
                # 回退到凸包
                try:
                    hull = ConvexHull(pixel_points)
                    hull_points = pixel_points[hull.vertices]
                except Exception:
                    return np.ones((height, width), dtype=bool)
        
        # 使用 OpenCV 绘制多边形掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        hull_points_int = hull_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [hull_points_int], 1)
        
        # 应用膨胀以扩展边界
        if buffer_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (buffer_pixels * 2 + 1, buffer_pixels * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel)
        
        if self.verbose:
            valid_ratio = mask.sum() / (height * width) * 100
            print(f"    边界掩码覆盖率: {valid_ratio:.1f}%")
        
        return mask.astype(bool)
    
    def _compute_alpha_shape(
        self, 
        points: np.ndarray, 
        alpha: float
    ) -> Optional[np.ndarray]:
        """
        计算点集的 alpha shape（凹包）
        
        Args:
            points: (N, 2) 2D 点坐标
            alpha: alpha 值，控制凹包的紧密程度
            
        Returns:
            边界点的坐标数组，按顺序排列
        """
        try:
            from scipy.spatial import Delaunay
            from collections import defaultdict
        except ImportError:
            return None
        
        if len(points) < 4:
            return points
        
        try:
            tri = Delaunay(points)
        except Exception:
            return None
        
        # 计算每个三角形的外接圆半径
        edges = defaultdict(int)
        
        for ia, ib, ic in tri.simplices:
            pa, pb, pc = points[ia], points[ib], points[ic]
            
            # 计算外接圆半径
            a = np.linalg.norm(pb - pc)
            b = np.linalg.norm(pa - pc)
            c = np.linalg.norm(pa - pb)
            s = (a + b + c) / 2.0
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
            
            if area > 0:
                circum_r = a * b * c / (4.0 * area)
            else:
                circum_r = np.inf
            
            # 如果外接圆半径小于 1/alpha，保留这个三角形的边
            if circum_r < 1.0 / alpha:
                for i, j in [(ia, ib), (ib, ic), (ic, ia)]:
                    edge = tuple(sorted([i, j]))
                    edges[edge] += 1
        
        # 只保留出现一次的边（边界边）
        boundary_edges = [e for e, count in edges.items() if count == 1]
        
        if not boundary_edges:
            return None
        
        # 将边连接成有序的多边形
        boundary_points = self._order_boundary_edges(boundary_edges, points)
        
        return boundary_points
    
    def _order_boundary_edges(
        self, 
        edges: list, 
        points: np.ndarray
    ) -> np.ndarray:
        """
        将边界边连接成有序的多边形顶点
        """
        from collections import defaultdict
        
        if not edges:
            return np.array([])
        
        # 构建邻接表
        adj = defaultdict(list)
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # 从任意点开始遍历
        start = edges[0][0]
        ordered = [start]
        visited = {start}
        current = start
        
        while True:
            neighbors = adj[current]
            next_node = None
            for n in neighbors:
                if n not in visited:
                    next_node = n
                    break
            
            if next_node is None:
                break
            
            ordered.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return points[ordered]
    
    def _iterative_dilation_fill(
        self, 
        dsm: np.ndarray, 
        nodata_mask: np.ndarray, 
        max_iterations: int = 100
    ) -> np.ndarray:
        """使用迭代膨胀填充空洞"""
        dsm_filled = dsm.copy()
        remaining = nodata_mask.copy()
        
        # 3x3 邻域核
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        
        for _ in range(max_iterations):
            if not remaining.any():
                break
            
            # 找到边缘空洞（与有效值相邻）
            # 简化版本：使用卷积检测边缘
            valid_neighbor_count = np.zeros_like(dsm_filled)
            valid_sum = np.zeros_like(dsm_filled)
            
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    # 滚动数组
                    rolled = np.roll(np.roll(dsm_filled, di, axis=0), dj, axis=1)
                    rolled_valid = np.roll(np.roll(~remaining, di, axis=0), dj, axis=1)
                    
                    valid_neighbor_count += rolled_valid.astype(np.float32)
                    valid_sum += np.where(rolled_valid, rolled, 0)
            
            # 填充有有效邻居的空洞
            can_fill = remaining & (valid_neighbor_count > 0)
            if not can_fill.any():
                break
            
            dsm_filled[can_fill] = valid_sum[can_fill] / valid_neighbor_count[can_fill]
            remaining[can_fill] = False
        
        return dsm_filled
    
    def _idw_interpolation(
        self, 
        dsm: np.ndarray, 
        nodata_mask: np.ndarray,
        power: float = 2.0,
        k_neighbors: int = 8
    ) -> np.ndarray:
        """反距离加权插值"""
        if not SCIPY_AVAILABLE:
            return self._iterative_dilation_fill(dsm, nodata_mask)
        
        from scipy.spatial import cKDTree
        
        valid_mask = ~nodata_mask
        valid_rows, valid_cols = np.where(valid_mask)
        valid_values = dsm[valid_mask]
        
        if len(valid_rows) == 0:
            return dsm.copy()
        
        hole_rows, hole_cols = np.where(nodata_mask)
        
        if len(hole_rows) == 0:
            return dsm.copy()
        
        # 构建 KD-Tree
        tree = cKDTree(np.column_stack([valid_cols, valid_rows]))
        
        # 查询最近邻
        distances, indices = tree.query(
            np.column_stack([hole_cols, hole_rows]), 
            k=min(k_neighbors, len(valid_rows))
        )
        
        # 处理单个最近邻的情况
        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        
        # 计算权重
        weights = 1.0 / (distances ** power + 1e-10)
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_normalized = weights / weights_sum
        
        # 计算插值值
        neighbor_values = valid_values[indices]
        interpolated = (neighbor_values * weights_normalized).sum(axis=1)
        
        # 填充
        dsm_filled = dsm.copy()
        dsm_filled[hole_rows, hole_cols] = interpolated.astype(np.float32)
        
        return dsm_filled
    
    def save_geotiff(
        self, 
        dsm_array: np.ndarray, 
        metadata: Dict[str, Any],
        output_path: Union[str, Path],
        compress: str = "lzw",
    ) -> bool:
        """
        保存 DSM 为 GeoTIFF 格式
        
        Args:
            dsm_array: DSM 栅格数组
            metadata: 元数据字典
            output_path: 输出文件路径
            compress: 压缩方式 ("lzw", "deflate", "none")
            
        Returns:
            成功返回 True
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio 库不可用。请安装: pip install rasterio")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\n  保存 GeoTIFF: {output_path}")
        
        xmin, ymin, xmax, ymax = metadata['bounds']
        height, width = dsm_array.shape
        
        # 创建仿射变换
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
        
        # 设置 CRS
        if metadata.get('crs_epsg'):
            crs = CRS.from_epsg(metadata['crs_epsg'])
        else:
            crs = None
        
        # 写入文件
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': width,
            'height': height,
            'count': 1,
            'crs': crs,
            'transform': transform,
            'nodata': metadata.get('nodata', self.nodata_value),
            'compress': compress if compress != "none" else None,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }
        
        with rasterio.open(str(output_path), 'w', **profile) as dst:
            dst.write(dsm_array, 1)
        
        if self.verbose:
            print(f"    ✓ 保存成功")
            print(f"    尺寸: {width} x {height}")
            print(f"    分辨率: {metadata['resolution']} 米")
            if metadata.get('crs_epsg'):
                print(f"    坐标系: EPSG:{metadata['crs_epsg']}")
        
        return True
    
    def build_from_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        bounds: Optional[Tuple[float, float, float, float]] = None,
        aggregation: str = "max",
        compress: str = "lzw",
        auto_crop: bool = True,
        crop_margin: int = 5,
        boundary_mask: bool = True,
        boundary_alpha: float = 0.0,
        boundary_buffer: int = 10,
    ) -> bool:
        """
        从点云文件构建 DSM 并保存为 GeoTIFF
        
        这是一个便捷方法，组合了加载、构建和保存步骤。
        
        Args:
            input_path: 输入点云文件路径
            output_path: 输出 GeoTIFF 文件路径
            bounds: 可选的边界范围
            aggregation: 聚合方式
            compress: 压缩方式
            auto_crop: 是否自动裁剪到有效数据范围（包围盒）
            crop_margin: 裁剪时保留的边缘像素数
            boundary_mask: 是否使用点云边界轮廓裁剪
            boundary_alpha: 边界轮廓参数 (0=凸包, >0=凹包)
            boundary_buffer: 边界向外扩展的像素数
            
        Returns:
            成功返回 True
        """
        # 加载点云
        xyz, colors = self.load_point_cloud(input_path)
        
        # 构建 DSM
        dsm_array, metadata = self.build_dsm(
            xyz, bounds, aggregation, 
            auto_crop=auto_crop, 
            crop_margin=crop_margin,
            boundary_mask=boundary_mask,
            boundary_alpha=boundary_alpha,
            boundary_buffer=boundary_buffer,
        )
        
        # 保存
        return self.save_geotiff(dsm_array, metadata, output_path, compress)


def build_dsm_from_reconstruction(
    point_cloud_path: Union[str, Path],
    output_path: Union[str, Path],
    resolution: float = 0.1,
    epsg_code: Optional[int] = None,
    interpolation_method: str = "nearest",
    aggregation: str = "max",
    auto_crop: bool = True,
    crop_margin: int = 5,
    boundary_mask: bool = True,
    boundary_alpha: float = 0.0,
    boundary_buffer: int = 10,
    verbose: bool = True,
) -> bool:
    """
    从 incremental_feature_matcher.py 输出的点云构建 DSM
    
    这是主要的入口函数，用于从密集点云生成 DSM.tif。
    
    Args:
        point_cloud_path: 点云文件路径（.ply 或 .las 格式）
        output_path: 输出 DSM 文件路径（.tif）
        resolution: DSM 分辨率（米），默认 0.1m
        epsg_code: EPSG 坐标系代码，如果知道的话
        interpolation_method: 空洞填充方法 ("nearest", "linear", "cubic", "idw", "none")
        aggregation: 聚合方式 ("max", "min", "mean")
        auto_crop: 是否自动裁剪到有效数据范围（包围盒），默认 True
        crop_margin: 裁剪时保留的边缘像素数，默认 5
        boundary_mask: 是否使用点云边界轮廓裁剪（非矩形），默认 True
        boundary_alpha: 边界轮廓参数:
            - 0: 使用凸包（convex hull）
            - >0: 使用凹包（concave hull），值越大边界越紧密
        boundary_buffer: 边界向外扩展的像素数，默认 10
        verbose: 是否输出详细日志
        
    Returns:
        成功返回 True，失败返回 False
    """
    try:
        if verbose:
            print(f"\n{'='*60}")
            print("构建数字表面模型 (DSM)")
            print(f"{'='*60}")
            print(f"  输入: {point_cloud_path}")
            print(f"  输出: {output_path}")
            if boundary_mask:
                mask_type = "凸包" if boundary_alpha == 0 else f"凹包(alpha={boundary_alpha})"
                print(f"  边界掩码: {mask_type}, 缓冲={boundary_buffer}px")
        
        builder = DSMBuilder(
            resolution=resolution,
            epsg_code=epsg_code,
            interpolation_method=interpolation_method,
            verbose=verbose,
        )
        
        success = builder.build_from_file(
            input_path=point_cloud_path,
            output_path=output_path,
            aggregation=aggregation,
            auto_crop=auto_crop,
            crop_margin=crop_margin,
            boundary_mask=boundary_mask,
            boundary_alpha=boundary_alpha,
            boundary_buffer=boundary_buffer,
        )
        
        if success and verbose:
            print(f"\n✓ DSM 构建完成: {output_path}")
        
        return success
        
    except Exception as e:
        print(f"✗ DSM 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="从点云生成数字表面模型 (DSM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 PLY 文件生成 DSM（10cm 分辨率）
  python build_dsm.py --input merged.ply --output dsm.tif --resolution 0.1
  
  # 从 LAS 文件生成 DSM，使用线性插值填充空洞
  python build_dsm.py --input merged.las --output dsm.tif --interpolation linear
  
  # 指定坐标系
  python build_dsm.py --input merged.ply --output dsm.tif --epsg 32648
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入点云文件路径 (.ply 或 .las)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出 DSM 文件路径 (.tif)"
    )
    
    parser.add_argument(
        "--resolution", "-r",
        type=float,
        default=0.1,
        help="DSM 分辨率（米），默认 0.1m"
    )
    
    parser.add_argument(
        "--epsg",
        type=int,
        default=None,
        help="EPSG 坐标系代码（可选）"
    )
    
    parser.add_argument(
        "--interpolation",
        type=str,
        default="nearest",
        choices=["nearest", "linear", "cubic", "idw", "none"],
        help="空洞填充方法，默认 nearest"
    )
    
    parser.add_argument(
        "--aggregation",
        type=str,
        default="max",
        choices=["max", "min", "mean", "count"],
        help="单元格聚合方式，默认 max（DSM 使用最高点）"
    )
    
    parser.add_argument(
        "--compress",
        type=str,
        default="lzw",
        choices=["lzw", "deflate", "none"],
        help="GeoTIFF 压缩方式，默认 lzw"
    )
    
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="禁用包围盒裁剪（默认会裁剪到有效数据范围）"
    )
    
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=5,
        help="包围盒裁剪时保留的边缘像素数，默认 5"
    )
    
    parser.add_argument(
        "--no-boundary-mask",
        action="store_true",
        help="禁用点云边界轮廓裁剪（默认会使用凸包/凹包裁剪）"
    )
    
    parser.add_argument(
        "--boundary-alpha",
        type=float,
        default=0.0,
        help="边界轮廓参数: 0=凸包, >0=凹包（值越大边界越紧密），默认 0"
    )
    
    parser.add_argument(
        "--boundary-buffer",
        type=int,
        default=10,
        help="边界向外扩展的像素数，默认 10"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不输出日志"
    )
    
    args = parser.parse_args()
    
    # 构建 DSM
    success = build_dsm_from_reconstruction(
        point_cloud_path=args.input,
        output_path=args.output,
        resolution=args.resolution,
        epsg_code=args.epsg,
        interpolation_method=args.interpolation,
        aggregation=args.aggregation,
        auto_crop=not args.no_crop,
        crop_margin=args.crop_margin,
        boundary_mask=not args.no_boundary_mask,
        boundary_alpha=args.boundary_alpha,
        boundary_buffer=args.boundary_buffer,
        verbose=not args.quiet,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    # ==================== 测试代码 ====================
    # 如果想使用命令行参数，将 TEST_MODE 设为 False
    TEST_MODE = True
    
    if TEST_MODE:
        from pathlib import Path
        
        # 获取项目根目录（从当前文件位置推断）
        # 当前文件: drone-map-anything/SfM/dsm/build_dsm.py
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        
        # ===== 配置测试参数 =====
        # 输入点云路径（修改为您的点云文件）
        INPUT_PLY = PROJECT_ROOT / "output/Ganluo_images/sparse_incremental_reconstruction/temp_merged_points_only/merged_4.ply"
        
        # 输出 DSM 路径
        OUTPUT_DSM = PROJECT_ROOT / "output/Ganluo_images/sparse_incremental_reconstruction/temp_dsm/dsm_test.tif"
        
        # DSM 参数
        RESOLUTION = 0.5          # 分辨率（米），测试时用 0.5 较快，生产用 0.1
        INTERPOLATION = "nearest" # 插值方法: nearest, linear, cubic, idw, none
        AGGREGATION = "max"       # 聚合方式: max, min, mean
        
        # 边界裁剪参数
        BOUNDARY_MASK = True      # 是否使用点云边界轮廓裁剪
        BOUNDARY_ALPHA = 0.005    # 0=凸包, >0=凹包（值越大越紧密，建议 0.001-0.01）
        BOUNDARY_BUFFER = 2       # 边界向外扩展的像素数（减小可更贴合点云）
        
        # EPSG 坐标系（如果知道的话，否则设为 None）
        EPSG_CODE = None
        # ===== 配置结束 =====
        
        print("\n" + "="*60)
        print("DSM 构建测试模式")
        print("="*60)
        
        # 检查输入文件
        if not INPUT_PLY.exists():
            print(f"❌ 输入文件不存在: {INPUT_PLY}")
            print("\n请修改 INPUT_PLY 变量指向有效的点云文件")
            exit(1)
        
        print(f"📥 输入: {INPUT_PLY}")
        print(f"📤 输出: {OUTPUT_DSM}")
        print(f"📏 分辨率: {RESOLUTION} 米")
        print(f"🔧 插值方法: {INTERPOLATION}")
        print(f"📊 聚合方式: {AGGREGATION}")
        if BOUNDARY_MASK:
            mask_type = "凸包" if BOUNDARY_ALPHA == 0 else f"凹包(alpha={BOUNDARY_ALPHA})"
            print(f"✂️  边界掩码: {mask_type}, 缓冲={BOUNDARY_BUFFER}px")
        
        # 执行构建
        success = build_dsm_from_reconstruction(
            point_cloud_path=INPUT_PLY,
            output_path=OUTPUT_DSM,
            resolution=RESOLUTION,
            epsg_code=EPSG_CODE,
            interpolation_method=INTERPOLATION,
            aggregation=AGGREGATION,
            boundary_mask=BOUNDARY_MASK,
            boundary_alpha=BOUNDARY_ALPHA,
            boundary_buffer=BOUNDARY_BUFFER,
            verbose=True,
        )
        
        if success:
            print(f"\n✅ 测试成功！DSM 已保存到: {OUTPUT_DSM}")
            # 显示文件大小
            if OUTPUT_DSM.exists():
                size_mb = OUTPUT_DSM.stat().st_size / (1024 * 1024)
                print(f"   文件大小: {size_mb:.2f} MB")
        else:
            print("\n❌ 测试失败！")
        
        exit(0 if success else 1)
    else:
        # 使用命令行参数模式
        exit(main())

