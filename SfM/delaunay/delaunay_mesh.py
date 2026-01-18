#!/usr/bin/env python3
"""
从点云 PLY 文件生成网格或 DSM (数字表面模型)
"""

from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import pymeshlab

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def create_mesh(input_path: Path, output_path: Path, method: str = "delaunay_2d", voxel_size: float = 0.0) -> bool:
    """
    从点云 PLY 文件创建网格。
    
    参数:
        input_path: 输入点云文件 (.ply) 或包含 points3D.ply 的目录
        output_path: 输出网格文件路径 (.ply)
        method: 网格化方法:
            - "delaunay_2d": 2D Delaunay 三角剖分，无空洞（默认，适合地形）
            - "ball_pivoting", "poisson", "alpha_shape"
        voxel_size: 下采样体素大小，0 表示不下采样
    
    返回:
        成功返回 True，否则返回 False。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 如果输入是目录，自动查找 points3D.ply
    if input_path.is_dir():
        ply_file = input_path / "points3D.ply"
        if ply_file.exists():
            input_path = ply_file
        else:
            print(f"错误: 目录中未找到 points3D.ply: {input_path}")
            return False
    
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return False
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 确保 .ply 扩展名
    if output_path.suffix.lower() != ".ply":
        output_path = output_path.with_suffix(".ply")
    
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"网格化方法: {method}")
    
    try:
        # 读取点云
        print("正在读取点云...")
        pcd = o3d.io.read_point_cloud(str(input_path))
        print(f"  原始点数量: {len(pcd.points)}")
        
        # 下采样（减少点数，让三角形更大）
        if voxel_size > 0:
            print(f"  下采样体素大小: {voxel_size}")
            pcd = pcd.voxel_down_sample(voxel_size)
            print(f"  下采样后点数量: {len(pcd.points)}")
        
        # 根据方法生成网格
        if method == "delaunay_2d":
            print("正在进行 2D Delaunay 三角剖分 (无空洞)...")
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            
            # 在 XY 平面上做 Delaunay 三角剖分
            points_2d = points[:, :2]
            tri = Delaunay(points_2d)
            
            # 创建网格
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
            
            # 设置顶点颜色
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            # 计算法线
            mesh.compute_vertex_normals()
            
        elif method == "ball_pivoting":
            # 检查是否有法线
            if not pcd.has_normals():
                print("正在估计法线...")
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
                pcd.orient_normals_consistent_tangent_plane(k=15)
            print("正在进行 Ball Pivoting 网格化...")
            # 计算合适的半径 (增大倍数以减少空洞)
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            # 使用更大的半径范围来填充空洞
            radii = [avg_dist * 2.0, avg_dist * 4.0, avg_dist * 8.0, avg_dist * 16.0, avg_dist * 32.0, avg_dist * 64.0, avg_dist * 128.0]
            print(f"  平均点距: {avg_dist:.4f}, 使用半径: {radii}")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            
            # 使用 PyMeshLab 填充孔洞
            print("正在填充孔洞...")
            temp_mesh_path = output_path.parent / "temp_mesh_bp.ply"
            o3d.io.write_triangle_mesh(str(temp_mesh_path), mesh)
            
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(temp_mesh_path))
            # 填充所有孔洞
            ms.meshing_close_holes(maxholesize=100000)  # 大数值填充所有孔洞
            ms.save_current_mesh(str(temp_mesh_path))
            
            # 重新读取填充后的网格
            mesh = o3d.io.read_triangle_mesh(str(temp_mesh_path))
            temp_mesh_path.unlink()  # 删除临时文件
            print("  孔洞填充完成")
            
        elif method == "poisson":
            if not pcd.has_normals():
                print("正在估计法线...")
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
                pcd.orient_normals_consistent_tangent_plane(k=15)
            print("正在进行 Poisson 网格化...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=5, width=0, scale=1.2, linear_fit=False
            )
            densities = np.asarray(densities)
            vertices_to_remove = densities < np.quantile(densities, 0.005)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        elif method == "alpha_shape":
            if not pcd.has_normals():
                print("正在估计法线...")
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            print("正在进行 Alpha Shape 网格化...")
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            alpha = avg_dist * 10.0
            print(f"  平均点距: {avg_dist:.4f}, alpha: {alpha:.4f}")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
        else:
            print(f"错误: 未知的网格化方法: {method}")
            return False
        
        print(f"  顶点数量: {len(mesh.vertices)}")
        print(f"  三角面数量: {len(mesh.triangles)}")
        
        # 保存网格
        print(f"正在保存网格到: {output_path}")
        o3d.io.write_triangle_mesh(str(output_path), mesh)
        
        print(f"✓ 网格已保存到: {output_path}")
        return True
        
    except Exception as e:
        import traceback
        print(f"网格化出错: {e}")
        traceback.print_exc()
        return False


def create_dsm(input_path: Path, output_path: Path, resolution: float = 0.1, 
               interpolation: str = "linear", nodata: float = -9999.0) -> bool:
    """
    从点云 PLY 文件生成 DSM (数字表面模型)。
    
    参数:
        input_path: 输入点云文件 (.ply) 或包含 points3D.ply 的目录
        output_path: 输出 DSM 文件路径 (.tif)
        resolution: DSM 分辨率（单位与点云坐标一致，通常为米）
        interpolation: 插值方法 ("nearest", "linear", "cubic")
        nodata: 无数据值
    
    返回:
        成功返回 True，否则返回 False。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 如果输入是目录，自动查找 points3D.ply
    if input_path.is_dir():
        ply_file = input_path / "points3D.ply"
        if ply_file.exists():
            input_path = ply_file
        else:
            print(f"错误: 目录中未找到 points3D.ply: {input_path}")
            return False
    
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return False
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 确保 .tif 扩展名
    if output_path.suffix.lower() not in [".tif", ".tiff"]:
        output_path = output_path.with_suffix(".tif")
    
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"DSM 分辨率: {resolution}")
    print(f"插值方法: {interpolation}")
    
    try:
        # 读取点云
        print("正在读取点云...")
        pcd = o3d.io.read_point_cloud(str(input_path))
        points = np.asarray(pcd.points)
        print(f"  点数量: {len(points)}")
        
        # 获取点云范围
        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)
        print(f"  X 范围: [{x_min:.2f}, {x_max:.2f}]")
        print(f"  Y 范围: [{y_min:.2f}, {y_max:.2f}]")
        print(f"  Z 范围: [{z_min:.2f}, {z_max:.2f}]")
        
        # 创建规则网格
        x_grid = np.arange(x_min, x_max + resolution, resolution)
        y_grid = np.arange(y_min, y_max + resolution, resolution)
        width = len(x_grid)
        height = len(y_grid)
        print(f"  DSM 尺寸: {width} x {height}")
        
        # 创建网格坐标
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # 插值生成 DSM
        print("正在插值生成 DSM...")
        z_interp = griddata(
            points[:, :2],  # XY 坐标
            points[:, 2],   # Z 值
            grid_points,
            method=interpolation,
            fill_value=nodata
        )
        
        # 重塑为 2D 数组
        dsm = z_interp.reshape(height, width)
        # 翻转 Y 轴（栅格从上到下）
        dsm = np.flipud(dsm)
        
        # 保存为 GeoTIFF
        print(f"正在保存 DSM 到: {output_path}")
        
        if RASTERIO_AVAILABLE:
            # 使用 rasterio 保存（带地理参考）
            transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
            
            with rasterio.open(
                str(output_path),
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=dsm.dtype,
                crs=None,  # 无 CRS（本地坐标系）
                transform=transform,
                nodata=nodata,
            ) as dst:
                dst.write(dsm, 1)
        else:
            # 使用 numpy 保存为简单格式
            np.save(output_path.with_suffix('.npy'), dsm)
            print(f"  警告: rasterio 未安装，保存为 .npy 格式")
            
            # 同时保存为图像（可视化）
            from PIL import Image
            # 归一化到 0-255
            dsm_valid = dsm[dsm != nodata]
            if len(dsm_valid) > 0:
                dsm_norm = (dsm - dsm_valid.min()) / (dsm_valid.max() - dsm_valid.min() + 1e-8)
                dsm_norm[dsm == nodata] = 0
                dsm_img = (dsm_norm * 255).astype(np.uint8)
                Image.fromarray(dsm_img).save(output_path.with_suffix('.png'))
                print(f"  同时保存可视化图像: {output_path.with_suffix('.png')}")
        
        print(f"✓ DSM 已保存到: {output_path}")
        return True
        
    except Exception as e:
        import traceback
        print(f"DSM 生成出错: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 默认测试路径 (直接指定 PLY 文件)
    # DEFAULT_INPUT = Path(r"D:\Github_code\drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction\recon_2_2_6_vggt\temp_merged\merged_8\points3D.ply")
    # DEFAULT_INPUT = Path(r"D:\Github_code\drone-map-anything\output\SWJTU_7th_teaching_building\sparse_incremental_reconstruction\recon_8_6_2_vggt\temp_merged\merged_2\points3D.ply")
    DEFAULT_INPUT = Path(r"D:\Github_code\drone-map-anything\output\WenChuan\sparse_incremental_reconstruction\recon2_2_6_vggt\temp_merged\merged_19\points3D.ply")
    DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "mesh.ply"
    
    import sys
    
    if len(sys.argv) >= 3:
        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
        method = sys.argv[3] if len(sys.argv) > 3 else "delaunay_2d"
        voxel_size = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    else:
        input_path = DEFAULT_INPUT
        output_path = DEFAULT_OUTPUT
        # 可选方法: "delaunay_2d"(无空洞), "ball_pivoting", "poisson", "alpha_shape"
        method = "alpha_shape"
        # 下采样体素大小: 0=不下采样, 越大越粗糙但速度快
        voxel_size = 1
    
    success = create_mesh(input_path, output_path, method, voxel_size)
    sys.exit(0 if success else 1)
