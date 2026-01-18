#!/usr/bin/env python3
"""
MapAnything 模型加载器

处理 MapAnything 模型的加载和推理。
"""

from pathlib import Path
from typing import Dict, List, Optional
import torch

from .base import BaseModelLoader


class MapAnythingLoader(BaseModelLoader):
    """MapAnything 模型加载器"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        初始化 MapAnything 模型加载器
        
        Args:
            model_path: 模型权重路径（可选，默认从 HuggingFace 加载）
            device: 设备类型
            verbose: 是否输出详细日志
        """
        super().__init__(model_path, device, verbose)
    
    def load_model(self):
        """加载 MapAnything 模型"""
        if self.model is None:
            from mapanything.models import MapAnything
            
            model_name = "facebook/map-anything"
            if self.verbose:
                print(f"Loading MapAnything model: {model_name}...")
            
            self.model = MapAnything.from_pretrained(model_name).to(self.device)
            
            if self.verbose:
                print("✓ MapAnything model loaded")
        
        return self.model
    
    def run_inference(
        self,
        preprocessed_views: List[Dict],
        image_paths: List[Path],
        num_images: int,
        min_images_for_scale: int,
    ) -> List[Dict]:
        """
        运行 MapAnything 推理
        
        Args:
            preprocessed_views: 预处理后的视图列表
            image_paths: 图像路径列表（未使用，MapAnything 直接使用预处理视图）
            num_images: 总图像数量
            min_images_for_scale: 尺度估计所需的最小图像数
            
        Returns:
            统一格式的输出列表
        """
        if self.verbose:
            print("Running MapAnything inference...")
        
        # 加载模型
        model = self.load_model()
        
        # 确定需要推理的视图
        views_to_infer, view_indices, message = self._get_views_to_infer(
            preprocessed_views, num_images, min_images_for_scale
        )
        
        if self.verbose:
            print(f"  {message}")
        
        # 运行推理
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
        
        # 添加视图索引到输出（与其他模型保持一致）
        for i, output in enumerate(outputs):
            output['view_index'] = view_indices[i]
        
        return outputs

