#!/usr/bin/env python3
"""
模型加载器基类

定义模型加载和推理的统一接口。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch


class BaseModelLoader(ABC):
    """模型加载器基类，定义统一的接口"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        初始化模型加载器
        
        Args:
            model_path: 模型权重路径
            device: 设备类型 ('cuda' 或 'cpu')
            verbose: 是否输出详细日志
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model = None
        self.dtype = None  # 用于混合精度推理
    
    @abstractmethod
    def load_model(self):
        """加载模型（延迟加载）"""
        pass
    
    @abstractmethod
    def run_inference(
        self,
        preprocessed_views: List[Dict],
        image_paths: List[Path],
        num_images: int,
        min_images_for_scale: int,
    ) -> List[Dict]:
        """
        运行推理
        
        Args:
            preprocessed_views: 预处理后的视图列表
            image_paths: 图像路径列表
            num_images: 总图像数量
            min_images_for_scale: 尺度估计所需的最小图像数
            
        Returns:
            统一格式的输出列表，每个字典包含:
                - pts3d: [1, H, W, 3] - 3D点
                - conf: [1, H, W] - 置信度
                - camera_poses: [1, 4, 4] - 相机位姿 (cam2world)
                - intrinsics: [1, 3, 3] - 内参
                - metric_scaling_factor: 尺度因子
        """
        pass
    
    def release_model(self):
        """释放模型内存"""
        if self.model is not None:
            del self.model
            self.model = None
            self.dtype = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.verbose:
                print("✓ Model released from memory")
    
    def _get_views_to_infer(
        self,
        preprocessed_views: List[Dict],
        num_images: int,
        min_images_for_scale: int,
    ) -> Tuple[List[Dict], List[int], str]:
        """
        确定需要推理的视图
        
        Args:
            preprocessed_views: 预处理后的视图列表
            num_images: 总图像数量
            min_images_for_scale: 尺度估计所需的最小图像数
            
        Returns:
            views_to_infer: 需要推理的视图列表
            view_indices: 视图索引列表
            message: 日志消息
        """
        if num_images == 1:
            views_to_infer = [preprocessed_views[0]]
            view_indices = [0]
            message = "Inferring first image (no scale calculation)"
        elif num_images < min_images_for_scale:
            views_to_infer = preprocessed_views[:]
            view_indices = list(range(num_images))
            message = f"Inferring images 1 to {num_images} ({num_images} images, building up to {min_images_for_scale})"
        else:
            views_to_infer = preprocessed_views[-min_images_for_scale:]
            view_indices = list(range(num_images - min_images_for_scale, num_images))
            start_idx = num_images - min_images_for_scale + 1
            message = f"Inferring images {start_idx} to {num_images} ({min_images_for_scale} images, sliding window)"
        
        return views_to_infer, view_indices, message
    
    def _get_image_paths_to_infer(
        self,
        image_paths: List[Path],
        num_images: int,
        min_images_for_scale: int,
    ) -> Tuple[List[Path], List[int], str]:
        """
        确定需要推理的图像路径
        
        Args:
            image_paths: 图像路径列表
            num_images: 总图像数量
            min_images_for_scale: 尺度估计所需的最小图像数
            
        Returns:
            paths_to_infer: 需要推理的图像路径列表
            view_indices: 视图索引列表
            message: 日志消息
        """
        if num_images == 1:
            paths_to_infer = [image_paths[0]]
            view_indices = [0]
            message = "Inferring first image (no scale calculation)"
        elif num_images < min_images_for_scale:
            paths_to_infer = image_paths[:]
            view_indices = list(range(num_images))
            message = f"Inferring images 1 to {num_images} ({num_images} images, building up to {min_images_for_scale})"
        else:
            paths_to_infer = image_paths[-min_images_for_scale:]
            view_indices = list(range(num_images - min_images_for_scale, num_images))
            start_idx = num_images - min_images_for_scale + 1
            message = f"Inferring images {start_idx} to {num_images} ({min_images_for_scale} images, sliding window)"
        
        return paths_to_infer, view_indices, message

