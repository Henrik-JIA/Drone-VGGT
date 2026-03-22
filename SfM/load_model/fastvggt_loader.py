#!/usr/bin/env python3
"""
FastVGGT 模型加载器

处理 FastVGGT 模型的加载和推理。
注意：FastVGGT 和 VGGT 都有 vggt 包，需要特殊处理避免冲突。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseModelLoader

# 获取项目根目录
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # drone-map-anything 根目录

# FastVGGT 模块（延迟初始化）
FASTVGGT_AVAILABLE = False
FastVGGT = None
fastvggt_pose_encoding_to_extri_intri = None
unproject_depth_map_to_point_map = None


def _load_images_rgb(image_paths: List[Path]) -> List[np.ndarray]:
    """加载图像为 RGB numpy 数组"""
    images = []
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images


def _get_vgg_input_imgs(images: np.ndarray):
    """预处理图像用于 FastVGGT 模型输入"""
    from torchvision import transforms as TF
    to_tensor = TF.ToTensor()
    vgg_input_images = []
    final_width = None
    final_height = None

    for image in images:
        img = Image.fromarray(image, mode="RGB")
        width, height = img.size
        # 调整图像大小，保持宽高比，确保高度是 14 的倍数
        new_width = 518
        new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # 转换为张量 (0, 1)

        # 如果高度超过 518，进行中心裁剪
        if new_height > 518:
            start_y = (new_height - 518) // 2
            img = img[:, start_y : start_y + 518, :]
            final_height = 518
        else:
            final_height = new_height

        final_width = new_width
        vgg_input_images.append(img)

    vgg_input_images = torch.stack(vgg_input_images)

    # 计算 patch 维度 (除以 14 得到 patch 大小)
    patch_width = final_width // 14  # 518 // 14 = 37
    patch_height = final_height // 14  # 动态计算

    return vgg_input_images, patch_width, patch_height


def _compute_original_coords(image_path_list: List[Path], new_width: int = 518) -> torch.Tensor:
    """计算 original_coords 用于将预测映射回原始图像坐标"""
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    original_coords = []
    for image_path in image_path_list:
        img = Image.open(image_path)
        img = img.convert("RGB")

        width, height = img.size
        max_dim = max(width, height)

        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        scale = new_width / max_dim

        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        original_coords.append(
            np.array([x1, y1, x2, y2, width, height], dtype=np.float32)
        )

    original_coords = torch.from_numpy(np.stack(original_coords, axis=0)).float()
    return original_coords


def _init_fastvggt() -> bool:
    """
    延迟初始化 FastVGGT 模块，避免与 vggt 包冲突
    
    策略：将 fastvggt 的模块以 'fastvggt_vggt' 前缀保存到 sys.modules，
    避免与原有的 vggt 包冲突。
    
    Returns:
        True if successful, False otherwise
    """
    global FASTVGGT_AVAILABLE, FastVGGT, fastvggt_pose_encoding_to_extri_intri
    global unproject_depth_map_to_point_map
    
    if FASTVGGT_AVAILABLE:
        return True  # 已经初始化过
    
    try:
        # fastvggt 位于 third/fastvggt 目录
        fastvggt_dir = project_root / "third" / "fastvggt"
        
        if not fastvggt_dir.exists():
            print(f"Warning: FastVGGT directory not found: {fastvggt_dir}")
            return False
        
        # 备份当前的 vggt 模块（如果存在）
        vggt_modules_backup = {}
        for key in list(sys.modules.keys()):
            if key == 'vggt' or key.startswith('vggt.'):
                vggt_modules_backup[key] = sys.modules.pop(key)
        
        # 将 fastvggt 目录添加到 sys.path 最前面
        fastvggt_dir_str = str(fastvggt_dir)
        if fastvggt_dir_str in sys.path:
            sys.path.remove(fastvggt_dir_str)
        sys.path.insert(0, fastvggt_dir_str)
        
        import_success = False
        try:
            # 导入 fastvggt 核心模块
            from vggt.models.vggt import VGGT as _FastVGGT
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri as _fastvggt_pose_enc
            from vggt.utils.geometry import unproject_depth_map_to_point_map as _unproject_depth
            
            # 保存导入的模块引用到全局变量
            FastVGGT = _FastVGGT
            fastvggt_pose_encoding_to_extri_intri = _fastvggt_pose_enc
            unproject_depth_map_to_point_map = _unproject_depth
            
            # 将 fastvggt 的模块以新前缀保存，避免后续被覆盖
            fastvggt_modules = {}
            for key in list(sys.modules.keys()):
                if key == 'vggt' or key.startswith('vggt.'):
                    fastvggt_modules['fastvggt_' + key] = sys.modules[key]
            sys.modules.update(fastvggt_modules)
            
            import_success = True
            FASTVGGT_AVAILABLE = True
            print("✓ FastVGGT modules loaded successfully")
            
        finally:
            # 从 sys.path 移除 fastvggt 目录
            if fastvggt_dir_str in sys.path:
                sys.path.remove(fastvggt_dir_str)
            
            # 清理临时的 vggt 模块（fastvggt 的）
            for key in list(sys.modules.keys()):
                if (key == 'vggt' or key.startswith('vggt.')) and not key.startswith('fastvggt_'):
                    sys.modules.pop(key, None)
            
            # 恢复原来的 vggt 模块
            sys.modules.update(vggt_modules_backup)
        
        return FASTVGGT_AVAILABLE
        
    except Exception as e:
        FASTVGGT_AVAILABLE = False
        print(f"Warning: FastVGGT model not available: {e}")
        import traceback
        traceback.print_exc()
        return False


class FastVGGTLoader(BaseModelLoader):
    """FastVGGT 模型加载器"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = False,
        # FastVGGT 特有参数
        merging: int = 0,
        merge_ratio: float = 0.9,
        depth_conf_thresh: float = 3.0,
    ):
        """
        初始化 FastVGGT 模型加载器
        
        Args:
            model_path: 模型权重路径（必需）
            device: 设备类型
            verbose: 是否输出详细日志
            merging: Token merging 参数，0=禁用
            merge_ratio: Token merge ratio (0.0-1.0)
            depth_conf_thresh: 深度置信度阈值
        """
        # 初始化 FastVGGT 模块
        if not _init_fastvggt():
            raise ImportError("FastVGGT model is not available. Please check the fastvggt installation.")
        
        super().__init__(model_path, device, verbose)
        
        # FastVGGT 特有参数
        self.merging = merging
        self.merge_ratio = merge_ratio
        self.depth_conf_thresh = depth_conf_thresh
    
    def load_model(self):
        """加载 FastVGGT 模型"""
        if self.model is None:
            if self.verbose:
                print("Loading FastVGGT model...")
                print(f"  merging: {self.merging}")
                print(f"  merge_ratio: {self.merge_ratio}")
            
            # 探测 GPU matmul 是否可用
            from .vggt_loader import _probe_cuda_matmul, _is_jetson_tegra
            self._gpu_matmul_ok = False
            if torch.cuda.is_available():
                self._gpu_matmul_ok = _probe_cuda_matmul(self.device, verbose=self.verbose)
            
            if not self._gpu_matmul_ok:
                self.device = 'cpu'
                self.dtype = torch.float32
                if self.verbose:
                    print(f"  Using CPU inference (dtype: float32)")
            else:
                cap = torch.cuda.get_device_capability()[0]
                is_tegra = _is_jetson_tegra()
                if is_tegra and cap < 9:
                    self.dtype = torch.float16
                else:
                    self.dtype = torch.bfloat16 if cap >= 8 else torch.float16
                if self.verbose:
                    print(f"  Using GPU inference (dtype: {self.dtype})")
            
            # 初始化 FastVGGT 模型
            self.model = FastVGGT(
                merging=self.merging,
                merge_ratio=self.merge_ratio,
                vis_attn_map=False
            )
            
            # 重要：当 merging=0 时，必须显式禁用 global_merging
            if self.merging == 0:
                self.model.aggregator.global_merging = False
                if self.verbose:
                    print("  global_merging disabled (merging=0)")
            
            # 加载权重
            if self.model_path:
                model_path = Path(self.model_path)
                if not model_path.is_absolute():
                    model_path = project_root / model_path
                
                if self.verbose:
                    print(f"  Loading weights from: {model_path}")
                ckpt = torch.load(str(model_path), map_location='cpu')
                self.model.load_state_dict(ckpt, strict=False)
            else:
                raise ValueError("FastVGGT requires model_path to be specified")
            
            # 移动到目标设备并设置 eval 模式
            self.model = self.model.to(self.device).eval()
            self.model = self.model.to(self.dtype)
            
            if self.verbose:
                print(f"✓ FastVGGT model loaded (device: {self.device})")
        
        return self.model
    
    def run_inference(
        self,
        preprocessed_views: List[Dict],
        image_paths: List[Path],
        num_images: int,
        min_images_for_scale: int,
    ) -> List[Dict]:
        """
        运行 FastVGGT 推理
        
        Args:
            preprocessed_views: 预处理后的视图列表（未使用，FastVGGT 有自己的预处理）
            image_paths: 图像路径列表
            num_images: 总图像数量
            min_images_for_scale: 尺度估计所需的最小图像数
            
        Returns:
            统一格式的输出列表
        """
        if self.verbose:
            print("Running FastVGGT inference...")
        
        # 加载模型
        model = self.load_model()
        
        # 确定需要推理的图像路径
        paths_to_infer, view_indices, message = self._get_image_paths_to_infer(
            image_paths, num_images, min_images_for_scale
        )
        
        if self.verbose:
            print(f"  {message}")
        
        # 转换路径为字符串
        image_paths_str = [str(p) for p in paths_to_infer]
        base_image_path_list = [p.name for p in paths_to_infer]
        
        # 使用 FastVGGT 的预处理流程
        # 1. 计算 original_coords
        original_coords = _compute_original_coords(paths_to_infer).to(self.device)
        
        # 2. 加载图像
        images_rgb = _load_images_rgb(paths_to_infer)
        if not images_rgb or len(images_rgb) < 1:
            raise ValueError(f"Failed to load images from {image_paths_str}")
        
        images_array = np.stack(images_rgb)
        
        # 3. 预处理图像
        vgg_input, patch_width, patch_height = _get_vgg_input_imgs(images_array)
        
        if self.verbose:
            print(f"  Image patch dimensions: {patch_width}x{patch_height}")
        
        # 4. 更新模型的 patch 维度
        model.update_patch_dimensions(patch_width, patch_height)
        
        # 5. 运行推理
        is_cuda = 'cuda' in str(self.device)
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            if is_cuda and self.dtype in (torch.float16, torch.bfloat16):
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    vgg_input_dev = vgg_input.cuda().to(self.dtype)
                    predictions = model(vgg_input_dev, image_paths=base_image_path_list)
            else:
                vgg_input_dev = vgg_input.to(self.device).float()
                predictions = model(vgg_input_dev, image_paths=base_image_path_list)
        
        if is_cuda:
            torch.cuda.synchronize()
        
        if self.verbose:
            if torch.cuda.is_available():
                max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                print(f"  Max GPU VRAM used: {max_mem_mb:.2f} MB")
        
        # 6. 解码位姿
        extrinsic, intrinsic = fastvggt_pose_encoding_to_extri_intri(
            predictions["pose_enc"], (vgg_input.shape[2], vgg_input.shape[3])
        )
        
        # 7. 处理深度图和置信度
        depth_tensor = predictions["depth"]  # [B, S, H, W]
        depth_conf = predictions["depth_conf"]  # [B, S, H, W]
        
        # 深度置信度过滤
        depth_np = depth_tensor[0].detach().float().cpu().numpy()  # [S, H, W]
        depth_conf_np = depth_conf[0].detach().float().cpu().numpy()  # [S, H, W]
        depth_mask = depth_conf_np >= self.depth_conf_thresh
        depth_filtered = depth_np.copy()
        depth_filtered[~depth_mask] = np.nan
        
        # 8. 反投影深度图到 3D 点云
        extrinsic_np = extrinsic[0].detach().float().cpu().numpy()  # [S, 3, 4]
        intrinsic_np = intrinsic[0].detach().float().cpu().numpy()  # [S, 3, 3]
        
        # 使用 unproject_depth_map_to_point_map 获取世界坐标系下的 3D 点
        points_3d = unproject_depth_map_to_point_map(depth_filtered, extrinsic_np, intrinsic_np)  # [S, H, W, 3]
        
        # 9. 准备图像用于颜色提取
        _, _, grid_h, grid_w = vgg_input.shape
        points_rgb_tensor = F.interpolate(
            vgg_input,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )
        points_rgb_np = (points_rgb_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        points_rgb_np = points_rgb_np.transpose(0, 2, 3, 1)  # [S, H, W, 3]
        
        # 10. 转换输出为统一格式
        outputs = self._convert_output_to_unified_format(
            points_3d=points_3d,
            depth_conf=depth_conf_np,
            extrinsic=extrinsic_np,
            intrinsic=intrinsic_np,
            vgg_input=vgg_input,
            view_indices=view_indices,
            original_coords=original_coords,
        )
        
        return outputs
    
    def _convert_output_to_unified_format(
        self,
        points_3d: np.ndarray,  # [S, H, W, 3]
        depth_conf: np.ndarray,  # [S, H, W]
        extrinsic: np.ndarray,  # [S, 3, 4]
        intrinsic: np.ndarray,  # [S, 3, 3]
        vgg_input: torch.Tensor,  # [S, 3, H, W]
        view_indices: List[int],
        original_coords: torch.Tensor,  # [S, 6]
    ) -> List[Dict]:
        """
        将 FastVGGT 预测转换为统一的输出格式
        
        Args:
            points_3d: 世界坐标 3D 点 [S, H, W, 3]
            depth_conf: 深度置信度图 [S, H, W]
            extrinsic: 相机外参 (cam from world) [S, 3, 4]
            intrinsic: 相机内参 [S, 3, 3]
            vgg_input: 预处理后的图像 [S, 3, H, W]
            view_indices: 视图索引列表
            original_coords: 原始图像坐标 [S, 6]
            
        Returns:
            统一格式的输出列表
        """
        S = len(view_indices)
        outputs = []
        
        for s in range(S):
            # 构建 cam2world 变换矩阵 (从 extrinsic 反转)
            R = extrinsic[s, :3, :3]  # [3, 3]
            t = extrinsic[s, :3, 3]   # [3]
            
            R_inv = R.T
            t_inv = -R_inv @ t
            
            cam2world = np.eye(4, dtype=np.float32)
            cam2world[:3, :3] = R_inv
            cam2world[:3, 3] = t_inv
            
            # 转换为 tensor
            pts3d_tensor = torch.from_numpy(points_3d[s:s+1]).float()  # [1, H, W, 3]
            conf_tensor = torch.from_numpy(depth_conf[s:s+1]).float()  # [1, H, W]
            cam2world_tensor = torch.from_numpy(cam2world).unsqueeze(0).float()  # [1, 4, 4]
            intrinsic_tensor = torch.from_numpy(intrinsic[s:s+1]).float()  # [1, 3, 3]
            
            # 移动到 GPU
            if torch.cuda.is_available():
                pts3d_tensor = pts3d_tensor.cuda()
                conf_tensor = conf_tensor.cuda()
                cam2world_tensor = cam2world_tensor.cuda()
                intrinsic_tensor = intrinsic_tensor.cuda()
            
            output = {
                'pts3d': pts3d_tensor,  # [1, H, W, 3]
                'conf': conf_tensor,  # [1, H, W]
                'camera_poses': cam2world_tensor,  # [1, 4, 4] (cam2world)
                'intrinsics': intrinsic_tensor,  # [1, 3, 3]
                'metric_scaling_factor': torch.tensor(1.0),  # FastVGGT 输出 metric scale
                'view_index': view_indices[s],
                # FastVGGT 特有：保存预处理后的图像用于颜色提取
                'vggt_image': vgg_input[s] if vgg_input is not None else None,  # [3, H, W]
                'original_coords': original_coords[s] if original_coords is not None else None,  # [6]
            }
            outputs.append(output)
        
        return outputs


def is_fastvggt_available() -> bool:
    """检查 FastVGGT 是否可用"""
    return FASTVGGT_AVAILABLE or _init_fastvggt()

