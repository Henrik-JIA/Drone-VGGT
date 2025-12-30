"""
Reconstruction utility functions for COLMAP reconstruction processing.
"""

import sys
import numpy as np

# 延迟导入 torch（可能未安装）
torch = None


def _ensure_torch():
    """延迟加载 torch 模块"""
    global torch
    if torch is None and "torch" in sys.modules:
        import torch as _torch
        torch = _torch
    return torch


def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    """
    重命名 COLMAP 重建中的图像，并将相机参数和特征点坐标缩放到原始分辨率。
    
    Args:
        reconstruction: pycolmap.Reconstruction 对象
        image_paths: 图像路径列表
        original_coords: 原始坐标数组，形状 (N, 4+) 包含 [x1, y1, ..., W_orig, H_orig]
        img_size: 处理时的图像尺寸 (width, height) 或单一值
        shift_point2d_to_original_res: 是否将 2D 点坐标变换到原始分辨率
        shared_camera: 是否使用共享相机（只需处理一次相机参数）
    
    Returns:
        更新后的 reconstruction 对象
    """
    # 规范化 original_coords 到 numpy
    _torch = _ensure_torch()
    if _torch is not None and isinstance(original_coords, _torch.Tensor):
        original_coords_np = original_coords.detach().cpu().numpy()
    else:
        original_coords_np = np.asarray(original_coords)
        
    if isinstance(img_size, (list, tuple, np.ndarray)) and len(img_size) == 2:
        proc_w = float(img_size[0])
        proc_h = float(img_size[1])
    else:
        proc_w = float(img_size)
        proc_h = float(img_size)
    
    # ========== 优化1: 预计算所有帧参数（向量化）==========
    real_sizes = original_coords_np[:, -2:].astype(np.float64)  # (N, 2) [W_orig, H_orig]
    inv_proc_w = 1.0 / max(1e-8, proc_w)
    inv_proc_h = 1.0 / max(1e-8, proc_h)
    scale_factors = real_sizes * np.array([inv_proc_w, inv_proc_h])  # (N, 2) [sx, sy]
    
    # 预提取 top_left 并转换为 float32（用于 shift_point2d）
    if shift_point2d_to_original_res:
        top_lefts = original_coords_np[:, :2].astype(np.float32)  # (N, 2) [x1, y1]
        scale_factors_f32 = scale_factors.astype(np.float32)  # (N, 2)
    
    rescale_camera = True
    processed_camera_ids = set()
    
    # ========== 优化2: 缓存字典引用 ==========
    images_dict = reconstruction.images
    cameras_dict = reconstruction.cameras

    # ========== 优化3: 使用 items() 迭代避免额外字典查找 ==========
    for pyimageid, pyimage in images_dict.items():
        idx = pyimageid - 1
        
        # Rename image
        pyimage.name = image_paths[idx]
        
        # 获取预计算的缩放因子（使用局部变量缓存）
        scale_xy = scale_factors[idx]
        sx, sy = scale_xy[0], scale_xy[1]
        real_w, real_h = real_sizes[idx]

        # ========== 优化4: 相机参数处理 ==========
        camera_id = pyimage.camera_id
        if rescale_camera and camera_id not in processed_camera_ids:
            pycamera = cameras_dict[camera_id]
            pred_params = list(pycamera.params)
            num_params = len(pred_params)
            model_name = getattr(pycamera, "model", "UNKNOWN")

            if str(model_name) == "PINHOLE" or num_params == 4:
                pred_params[0] *= sx  # fx
                pred_params[1] *= sy  # fy
                pred_params[2] *= sx  # cx
                pred_params[3] *= sy  # cy
            else:
                if num_params >= 2:
                    pred_params[0] *= sx
                    pred_params[1] *= sy
                if num_params >= 4:
                    pred_params[-2] *= sx
                    pred_params[-1] *= sy

            pycamera.params = pred_params
            pycamera.width = int(real_w)
            pycamera.height = int(real_h)
            
            processed_camera_ids.add(camera_id)
            
            if shared_camera:
                rescale_camera = False

        # ========== 优化5: points2D 坐标变换（仅在需要时执行）==========
        if shift_point2d_to_original_res:
            points2D_list = pyimage.points2D
            num_points = len(points2D_list)
            
            if num_points > 0:
                # 使用预计算的参数
                top_left = top_lefts[idx]
                scale_xy_f32 = scale_factors_f32[idx]
                
                # ========== 优化6: 列表推导提取坐标（比显式循环更快）==========
                # 直接使用 np.array + 列表推导，CPython 对列表推导有优化
                coords = np.array([p.xy for p in points2D_list], dtype=np.float32)
                
                # 向量化变换（融合为单次操作）
                coords = (coords - top_left) * scale_xy_f32
                
                # ========== 优化7: 使用 zip 迭代（避免索引开销）==========
                for p, xy in zip(points2D_list, coords):
                    p.xy = xy

    return reconstruction

