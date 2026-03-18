#!/usr/bin/env python3
"""
VGGT 模型加载器

处理 VGGT 模型的加载和推理。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .base import BaseModelLoader


def _is_jetson_tegra() -> bool:
    """检测是否运行在 Jetson/Tegra 平台上"""
    try:
        with open('/proc/version', 'r') as f:
            if 'tegra' in f.read().lower():
                return True
    except (FileNotFoundError, PermissionError):
        pass
    return os.path.exists('/proc/device-tree/compatible') and os.path.exists('/sys/devices/soc0/family')


def _probe_cuda_matmul(device: str, verbose: bool = False) -> bool:
    """测试 GPU matmul 是否可用（cuBLAS 可能不支持新架构如 sm_110）"""
    try:
        a = torch.randn(8, 8, device=device, dtype=torch.float32)
        _ = torch.matmul(a, a.transpose(-1, -2))
        torch.cuda.synchronize()
        del a
        if verbose:
            print(f"  cuBLAS matmul probe: OK")
        return True
    except RuntimeError as e:
        if verbose:
            print(f"  cuBLAS matmul probe: FAILED — GPU matmul unavailable")
            print(f"    ({e.__class__.__name__}: {str(e)[:120]})")
            print(f"    Falling back to CPU inference")
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        return False

# VGGT 模型导入（条件导入）
VGGT_AVAILABLE = False
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    VGGT_AVAILABLE = True
except ImportError:
    pass

# 获取项目根目录
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # drone-map-anything 根目录


class VGGTLoader(BaseModelLoader):
    """VGGT 模型加载器"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        初始化 VGGT 模型加载器
        
        Args:
            model_path: 模型权重路径（可选，默认从 HuggingFace 加载）
            device: 设备类型
            verbose: 是否输出详细日志
        """
        if not VGGT_AVAILABLE:
            raise ImportError("VGGT model is not available. Please install the vggt package.")
        
        super().__init__(model_path, device, verbose)
    
    def load_model(self):
        """加载 VGGT 模型"""
        if self.model is None:
            if self.verbose:
                print("Loading VGGT model...")
            
            # 探测 GPU matmul 是否可用
            # 新架构 GPU（如 NVIDIA Thor sm_110）可能 cuBLAS 尚无支持
            self._gpu_matmul_ok = False
            if 'cuda' in str(self.device):
                self._gpu_matmul_ok = _probe_cuda_matmul(self.device, verbose=self.verbose)
            
            # 如果 GPU matmul 不可用，回退到 CPU
            if not self._gpu_matmul_ok:
                self.device = 'cpu'
                self.dtype = torch.float32
                if self.verbose:
                    print(f"  Using CPU inference (dtype: float32)")
            else:
                # GPU 可用，选择半精度类型
                cap = torch.cuda.get_device_capability()[0]
                is_tegra = _is_jetson_tegra()
                if is_tegra and cap < 9:
                    self.dtype = torch.float16
                else:
                    self.dtype = torch.bfloat16 if cap >= 8 else torch.float16
                if self.verbose:
                    print(f"  Using GPU inference (dtype: {self.dtype})")
            
            self.model = VGGT()
            
            # 加载权重
            if self.model_path:
                model_path = Path(self.model_path)
                if not model_path.is_absolute():
                    model_path = project_root / model_path
                
                if self.verbose:
                    print(f"Loading weights from: {model_path}")
                    
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                state_dict = torch.load(str(model_path), map_location=device)
                self.model.load_state_dict(state_dict)
            else:
                try:
                    self.model = VGGT.from_pretrained("facebook/vggt")
                    if self.verbose:
                        print("Loaded VGGT model from Hugging Face Hub")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not load pretrained weights: {e}")
                        print("Using randomly initialized VGGT model")
            
            self.model.to(self.device)
            
            if self.verbose:
                print(f"✓ VGGT model loaded (device: {self.device})")
        
        return self.model
    
    def run_inference(
        self,
        preprocessed_views: List[Dict],
        image_paths: List[Path],
        num_images: int,
        min_images_for_scale: int,
    ) -> List[Dict]:
        """
        运行 VGGT 推理
        
        Args:
            preprocessed_views: 预处理后的视图列表（未使用，VGGT 有自己的预处理）
            image_paths: 图像路径列表
            num_images: 总图像数量
            min_images_for_scale: 尺度估计所需的最小图像数
            
        Returns:
            统一格式的输出列表
        """
        if self.verbose:
            print("Running VGGT inference...")
        
        # 加载模型
        model = self.load_model()
        
        # 确定需要推理的图像路径
        paths_to_infer, view_indices, message = self._get_image_paths_to_infer(
            image_paths, num_images, min_images_for_scale
        )
        
        if self.verbose:
            print(f"  {message}")
        
        # 预处理图像
        image_paths_str = [str(p) for p in paths_to_infer]
        images = load_and_preprocess_images(image_paths_str).to(self.device)
        
        # 获取图像尺寸用于位姿解码
        _, _, H, W = images.shape  # [S, 3, H, W]
        image_size_hw = (H, W)
        
        # 推理前清理 CUDA 缓存
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 运行推理
        if self.verbose:
            print(f"  Running on {self.device} with dtype {self.dtype}")
        
        with torch.no_grad():
            if self.dtype in (torch.float16, torch.bfloat16) and 'cuda' in str(self.device):
                with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                    predictions = model(images)
            else:
                predictions = model(images.float())
        
        # 转换为统一格式
        outputs = self._convert_output_to_unified_format(
            predictions, 
            image_size_hw, 
            view_indices,
            images  # 传递原始图像用于颜色提取
        )
        
        return outputs
    
    def _convert_output_to_unified_format(
        self, 
        predictions: Dict, 
        image_size_hw: Tuple[int, int],
        view_indices: List[int],
        images: torch.Tensor = None  # VGGT 预处理后的图像 [S, 3, H, W]
    ) -> List[Dict]:
        """
        将 VGGT 预测转换为统一的输出格式
        
        Args:
            predictions: VGGT 模型预测结果
            image_size_hw: 预处理图像的 (高度, 宽度)
            view_indices: 视图索引列表
            images: VGGT 预处理后的图像张量 [S, 3, H, W]
            
        Returns:
            统一格式的输出列表:
                - pts3d: [1, H, W, 3] - 3D 点
                - conf: [1, H, W] - 置信度
                - camera_poses: [1, 4, 4] - 相机位姿 (cam2world)
                - intrinsics: [1, 3, 3] - 相机内参
                - metric_scaling_factor: 尺度因子
                - vggt_image: [3, H, W] - VGGT 预处理图像（用于颜色提取）
        """
        # 提取预测结果
        world_points = predictions['world_points']  # [B, S, H, W, 3]
        world_points_conf = predictions['world_points_conf']  # [B, S, H, W]
        pose_enc = predictions['pose_enc']  # [B, S, 9]
        
        # 将位姿编码转换为外参和内参
        # extrinsics: [B, S, 3, 4] (cam from world)
        # intrinsics: [B, S, 3, 3]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, 
            image_size_hw=image_size_hw,
            pose_encoding_type="absT_quaR_FoV",
            build_intrinsics=True
        )
        
        # 将外参 (cam from world) 转换为相机位姿 (cam2world / world from cam)
        # cam2world = inverse of extrinsics
        B, S = extrinsics.shape[:2]
        camera_poses = torch.zeros(B, S, 4, 4, device=extrinsics.device, dtype=extrinsics.dtype)
        
        for b in range(B):
            for s in range(S):
                # extrinsics 是 [R|t]，其中 camera_coords = R @ world_coords + t
                # cam2world: world_coords = R.T @ (camera_coords - t) = R.T @ camera_coords - R.T @ t
                R = extrinsics[b, s, :3, :3]
                t = extrinsics[b, s, :3, 3]
                
                R_inv = R.T
                t_inv = -R_inv @ t
                
                camera_poses[b, s, :3, :3] = R_inv
                camera_poses[b, s, :3, 3] = t_inv
                camera_poses[b, s, 3, 3] = 1.0
        
        # 构建统一输出列表
        outputs = []
        for s in range(S):
            output = {
                'pts3d': world_points[:, s, :, :, :],  # [B, H, W, 3] -> 保持 B 维度以保持一致性
                'conf': world_points_conf[:, s, :, :],  # [B, H, W]
                'camera_poses': camera_poses[:, s, :, :],  # [B, 4, 4]
                'intrinsics': intrinsics[:, s, :, :],  # [B, 3, 3]
                'metric_scaling_factor': torch.tensor(1.0),  # VGGT 不输出此值，使用 1.0
                # VGGT 特定输出
                'depth': predictions.get('depth', None),
                'depth_conf': predictions.get('depth_conf', None),
                'view_index': view_indices[s],
                # 保存 VGGT 预处理后的图像用于颜色提取
                'vggt_image': images[s] if images is not None else None,  # [3, H, W]
            }
            outputs.append(output)
        
        return outputs


def is_vggt_available() -> bool:
    """检查 VGGT 是否可用"""
    return VGGT_AVAILABLE

