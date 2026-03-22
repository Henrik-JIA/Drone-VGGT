#!/usr/bin/env python3
"""
模型加载器模块

支持的模型:
- MapAnything: 通用的地图重建模型
- VGGT: Visual Geometry Grounded Transformers
- FastVGGT: 快速版本的 VGGT
"""

from .base import BaseModelLoader
from .mapanything_loader import MapAnythingLoader
from .vggt_loader import VGGTLoader, is_vggt_available
from .fastvggt_loader import FastVGGTLoader, is_fastvggt_available

__all__ = [
    # 基类
    'BaseModelLoader',
    # 加载器
    'MapAnythingLoader',
    'VGGTLoader',
    'FastVGGTLoader',
    # 检查函数
    'is_vggt_available',
    'is_fastvggt_available',
    # 工厂函数
    'create_model_loader',
]


def create_model_loader(
    model_type: str,
    model_path: str = None,
    device: str = None,
    verbose: bool = False,
    **kwargs
) -> BaseModelLoader:
    """
    创建模型加载器的工厂函数
    
    Args:
        model_type: 模型类型 ('mapanything', 'vggt', 'fastvggt')
        model_path: 模型权重路径
        device: 设备类型
        verbose: 是否输出详细日志
        **kwargs: 其他模型特定参数
            - FastVGGT 特有: merging, merge_ratio, depth_conf_thresh
            
    Returns:
        BaseModelLoader: 对应的模型加载器实例
        
    Raises:
        ValueError: 未知的模型类型或模型不可用
    """
    model_type = model_type.lower()
    
    if model_type == 'mapanything':
        return MapAnythingLoader(
            model_path=model_path,
            device=device,
            verbose=verbose,
        )
    
    elif model_type == 'vggt':
        if not is_vggt_available():
            raise ValueError("VGGT model is not available. Please install the vggt package.")
        return VGGTLoader(
            model_path=model_path,
            device=device,
            verbose=verbose,
        )
    
    elif model_type == 'fastvggt':
        if not is_fastvggt_available():
            raise ValueError("FastVGGT model is not available. Please check the fastvggt installation.")
        return FastVGGTLoader(
            model_path=model_path,
            device=device,
            verbose=verbose,
            merging=kwargs.get('fastvggt_merging', 0),
            merge_ratio=kwargs.get('fastvggt_merge_ratio', 0.9),
            depth_conf_thresh=kwargs.get('fastvggt_depth_conf_thresh', 3.0),
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'mapanything', 'vggt', 'fastvggt'")

