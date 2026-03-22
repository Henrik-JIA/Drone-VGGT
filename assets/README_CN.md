<div align="center">

# Drone-VGGT

**基于即插即用视觉几何变换器模型的无人机影像模块化增量式密集三维重建**

[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://drone-vggt.github.io/)
[![English](https://img.shields.io/badge/English-README-orange.svg)](../README.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

## 概述

Drone-VGGT 是一个专为无人机航拍影像设计的增量式 3D 重建系统。系统支持多种深度学习模型（MapAnything、VGGT、FastVGGT），实现了高效的增量式点云重建和合并，并支持实时可视化和地理坐标导出。

![记录03推理密集点云Ganluo](./记录03推理密集点云Ganluo.gif)

### 主要特性

- 🚀 **增量式处理**：逐张添加影像，实时更新重建结果
- 🧠 **多模型支持**：MapAnything、VGGT、FastVGGT 三种模型可选
- 🔗 **智能点云合并**：四种合并策略（full、confidence、confidence_blend、points_only）
- 🌍 **地理坐标导出**：自动检测 UTM 区域，支持多种坐标系导出
- 👁️ **实时可视化**：基于 Viser 的 3D 点云和相机位姿可视化
- 📊 **GPS/XMP 元数据**：自动从无人机影像中提取位姿信息

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [核心功能](#核心功能)
  - [增量式重建流程](#增量式重建流程)
  - [模型选择](#模型选择)
  - [点云合并方法](#点云合并方法)
- [API 参考](#api-参考)
- [配置参数](#配置参数)
- [输出格式](#输出格式)
- [示例数据集](#示例数据集)
- [致谢](#致谢)

## 安装

### 环境要求

- Python 3.10 - 3.11
- PyTorch 2.3.1
- CUDA 11.8
- [Pixi](https://pixi.sh/) (推荐) 或 Conda

### 方式一：使用 Pixi 安装（推荐）

[Pixi](https://pixi.sh/) 是一个现代化的包管理器，能够自动处理 Conda 和 PyPI 依赖。

```bash
# 1. 安装 Pixi (如果尚未安装)
# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex

# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# 2. 克隆仓库
git clone https://github.com/Henrik-JIA/Drone-VGGT.git
cd Drone-VGGT

# 3. 一键安装所有依赖并激活环境
pixi install

# 4. 完整安装（包括本地包和第三方子模块）
pixi run setup

# 5. 激活环境
pixi shell
```

#### Pixi 常用命令

```bash
# 查看环境信息（PyTorch 版本、CUDA 状态等）
pixi run info

# 运行增量 SfM 重建
pixi run run-sfm

# 运行 Delaunay 网格化
pixi run run-mesh

# 生成 DSM
pixi run run-dsm

# 代码格式检查
pixi run lint
pixi run format
```

#### Pixi 环境说明

项目提供了三种预配置环境：

| 环境 | 说明 | 激活命令 |
|------|------|---------|
| `default` | 完整环境，包含所有功能 | `pixi shell` |
| `dev` | 开发环境，包含测试工具 | `pixi shell -e dev` |
| `minimal` | 最小环境，仅核心依赖 | `pixi shell -e minimal` |

### 方式二：使用 Conda/Pip 安装

```bash
# 克隆仓库
git clone https://github.com/Henrik-JIA/Drone-VGGT.git
cd Drone-VGGT

# 创建 conda 环境
conda create -n drone-vggt python=3.11 -y
conda activate drone-vggt

# 安装 PyTorch (CUDA 11.8)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -e .

# 安装可选依赖 (可视化、地理坐标导出等)
pip install -e ".[all]"

# 安装第三方子模块
pip install -e third/vggt --no-deps
pip install -e third/fastvggt --no-deps
pip install git+https://github.com/cvg/LightGlue.git --no-deps
```

### 模型权重

根据选择的模型类型，下载相应的权重：

| 模型 | 权重路径 | 下载地址 |
|------|---------|---------|
| MapAnything | 自动从 HuggingFace 下载 | [facebook/map-anything](https://huggingface.co/facebook/map-anything) |
| VGGT | `weights/vggt/model.pt` | [VGGT 官方仓库](https://github.com/facebookresearch/vggt) |
| FastVGGT | `weights/fastvggt/model_tracker_fixed_e20.pt` | 联系作者获取 |

## 快速开始

### 基础用法

```python
from pathlib import Path
from SfM.incremental_feature_matcher import run_incremental_feature_matching

# 配置输入输出路径
input_dir = Path("examples/your_drone_images/images")
output_dir = Path("output/your_project")

# 获取所有影像文件
image_files = sorted(input_dir.glob("*.JPG"))

# 运行增量式重建
success = run_incremental_feature_matching(
    image_paths=image_files,
    output_dir=output_dir,
    model_type='mapanything',      # 'mapanything' | 'vggt' | 'fastvggt'
    min_images_for_scale=6,        # 每批次处理的影像数量
    overlap=2,                     # 批次间的重叠影像数
    merge_method='confidence',     # 点云合并方法
    enable_visualization=True,     # 启用实时可视化
    export_georef=True,            # 导出地理坐标
    verbose=True,
)
```

### 命令行运行

```bash
# 直接运行主脚本
python SfM/incremental_feature_matcher.py
```

修改脚本底部的配置参数来处理你自己的数据集：

```python
# 在 if __name__ == "__main__": 部分修改
input_dir = Path(r"your/image/folder/images")
output_dir = Path(r"your/output/folder")
MODEL_TYPE = 'vggt'  # 选择模型
```

## 核心功能

### 增量式重建流程

系统采用增量式处理策略，将大量影像分批处理，显著降低内存消耗：

```
┌─────────────────────────────────────────────────────────────┐
│                     增量式重建流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Batch 0: [img_0, img_1, img_2, img_3, img_4, img_5]        │
│           └── 推理 → 重建 → 初始点云                         │
│                                                             │
│  Batch 1: [img_4, img_5, img_6, img_7, img_8, img_9]        │
│           └── 推理 → 重建 → Sim3 对齐 → 合并到主点云          │
│                  ↑                                          │
│              重叠影像                                        │
│                                                             │
│  Batch 2: [img_8, img_9, img_10, img_11, ...]               │
│           └── 推理 → 重建 → Sim3 对齐 → 合并到主点云          │
│                                                             │
│  ...                                                        │
│                                                             │
│  最终输出: 完整的合并点云 + 地理坐标 Reconstruction           │
└─────────────────────────────────────────────────────────────┘
```

### 模型选择

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| **MapAnything** | 端到端 Transformer，支持多种输入模态 | 通用场景，高精度需求 |
| **VGGT** | 视觉几何基础模型 | 大规模场景，需要稳定性 |
| **FastVGGT** | VGGT 加速版本，支持 Token Merging | 实时处理，内存受限场景 |

### 点云合并方法

系统提供四种点云合并策略：

#### 1. `points_only` (推荐用于大规模数据)

```python
merge_method='points_only'
```

- ✅ 最轻量级，仅维护点云数组
- ✅ 内存效率最高
- ✅ 支持增量式体素去重
- ❌ 不维护 pycolmap Reconstruction 结构

**合并流程**：
```
1. 通过公共影像估计 Sim3 变换
2. 使用 2D 像素匹配识别重叠区
3. 只添加非重叠区的点（重叠区已存在于累积点云中）
4. 可选：延迟体素去重（点数超过阈值时执行）
```

#### 2. `confidence` (推荐用于高质量输出)

```python
merge_method='confidence'
```

- ✅ 基于置信度选择最优点
- ✅ 维护完整的 Reconstruction 结构
- ✅ 支持精确的像素级匹配
- ❌ 内存消耗较高

#### 3. `confidence_blend` (最高质量)

```python
merge_method='confidence_blend'
```

- ✅ 置信度加权混合
- ✅ 融合带平滑插值
- ✅ 密度均衡化
- ❌ 计算开销最大

#### 4. `full` (完整流程)

```python
merge_method='full'
```

- ✅ 使用完整的 merge_full_pipeline
- ✅ 支持多阶段精化对齐
- ❌ 最慢

### 重建类型与 3DGS 支持

系统支持两种重建类型，决定了点云的生成方式和后续应用：

| 重建类型 | 输出格式 | 3DGS 兼容性 | 适用场景 |
|---------|---------|-------------|---------|
| `each_pixel_feature_points` | 密集点云（每个像素一个3D点） | ❌ 不支持 | 纯点云可视化、DSM生成 |
| `dense_feature_points` | COLMAP Reconstruction + 密集点云 | ✅ 支持 | 3DGS训练、传统MVS流程 |

#### 为什么需要 `dense_feature_points` 才能支持 3DGS？

3D Gaussian Splatting (3DGS) 训练需要标准的 COLMAP 输出格式，包括：
- `cameras.txt` / `cameras.bin` - 相机内参
- `images.txt` / `images.bin` - 相机位姿及 2D-3D 对应关系
- `points3D.txt` / `points3D.bin` - 3D 点云及其观测信息

只有 `dense_feature_points` 模式会通过特征跟踪建立多视图间的 2D-3D 对应关系，生成完整的 COLMAP Reconstruction 结构。

#### 配置 3DGS 兼容输出

要生成可用于 3DGS 训练的输出，需要正确配置以下参数：

```python
success = run_incremental_feature_matching(
    image_paths=image_files,
    output_dir=output_dir,
    
    # 关键参数：必须使用 dense_feature_points
    reconstruction_type='dense_feature_points',
    
    # 重要：query_frame_num 应与 min_images_for_scale 一致
    # 这确保每张影像都参与特征跟踪，满足 3DGS 的观测要求
    min_images_for_scale=6,
    query_frame_num=6,  # 必须 >= min_images_for_scale
    
    # 其他参数
    model_type='vggt',
    merge_method='confidence',
    verbose=True,
)
```

> ⚠️ **重要提示**：`query_frame_num` 参数控制特征跟踪时的查询帧数量。为确保每张影像都能建立 2D-3D 对应关系，且每个 3D 点至少被 3 个 2D 点观测到（3DGS 的基本要求），`query_frame_num` 必须与 `min_images_for_scale` 保持一致。

#### 输出目录结构（3DGS 兼容）

使用 `dense_feature_points` 模式后，输出目录将包含：

```
output/
├── temp_merged/
│   └── merged_N/
│       ├── cameras.txt      # 相机内参 ← 3DGS 需要
│       ├── images.txt       # 影像位姿 + 2D-3D 对应 ← 3DGS 需要
│       ├── points3D.txt     # 3D 点云 + 观测信息 ← 3DGS 需要
│       └── points3D.ply     # PLY 格式点云
└── temp_merged_reconstruction_georeferenced/
    └── ...                  # 地理坐标版本
```

可直接用于 3DGS 训练：
```bash
# 使用 gaussian-splatting 官方实现
python train.py -s output/temp_merged/merged_N/
```

## API 参考

### `IncrementalFeatureMatcherSfM` 类

核心类，管理增量式重建的完整生命周期。

```python
from SfM.incremental_feature_matcher import IncrementalFeatureMatcherSfM

matcher = IncrementalFeatureMatcherSfM(
    output_dir=Path("output"),
    reconstruction_type='each_pixel_feature_points',  # 或 'dense_feature_points'
    model_type='mapanything',
    min_images_for_scale=6,
    overlap=2,
    merge_method='confidence',
    enable_visualization=True,
    verbose=True,
)
```

#### 主要方法

| 方法 | 描述 |
|------|------|
| `add_image(image_path)` | 添加单张影像并处理 |
| `export_georeferenced(target_crs)` | 导出地理坐标系下的重建结果 |
| `export_utm()` | 导出 UTM 坐标（自动检测区域） |
| `get_statistics()` | 获取当前统计信息 |

#### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `merged_reconstruction` | `pycolmap.Reconstruction` | 合并后的重建结果 |
| `merged_points_xyz` | `np.ndarray` | 合并后的点云坐标 (N, 3) |
| `merged_points_colors` | `np.ndarray` | 合并后的点云颜色 (N, 3) |
| `inference_reconstructions` | `List[Dict]` | 每个批次的重建结果 |

### `run_incremental_feature_matching` 函数

便捷函数，封装完整的增量式重建流程。

```python
from SfM.incremental_feature_matcher import run_incremental_feature_matching

success = run_incremental_feature_matching(
    image_paths=image_files,        # 影像路径列表
    output_dir=output_dir,          # 输出目录
    
    # 重建类型：'each_pixel_feature_points'（纯密集点云）或 'dense_feature_points'（支持3DGS）
    reconstruction_type='dense_feature_points',
    
    model_type='vggt',              # 'mapanything' | 'vggt' | 'fastvggt'
    model_path=None,                # 模型权重路径
    image_interval=1,               # 影像间隔
    min_images_for_scale=6,         # 每批次影像数
    overlap=2,                      # 批次重叠数
    
    # 特征跟踪参数（仅 dense_feature_points 模式）
    # query_frame_num 应与 min_images_for_scale 一致以支持 3DGS
    query_frame_num=6,
    max_query_pts=12288,
    
    pred_vis_scores_thres_value=0.7,
    max_reproj_error=5.0,
    run_global_sfm_first=True,      # 先运行全局 SfM
    merge_method='confidence',
    merge_voxel_size=0.5,           # 体素大小（米）
    enable_visualization=True,
    visualization_mode='merged',    # 'aligned' | 'merged'
    export_georef=True,
    target_crs='auto_utm',
    verbose=True,
)
```

## 配置参数

### 核心参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `reconstruction_type` | 'each_pixel_feature_points' | 重建类型：`'each_pixel_feature_points'`（纯密集点云）或 `'dense_feature_points'`（COLMAP格式，支持3DGS） |
| `min_images_for_scale` | 6 | 每批次处理的影像数量 |
| `overlap` | 2 | 相邻批次间的重叠影像数 |
| `merge_method` | 'confidence' | 点云合并方法 |
| `merge_voxel_size` | 1.5 | 体素去重大小（米） |

### 模型参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `model_type` | 'mapanything' | 模型类型 |
| `model_path` | None | 模型权重路径（MapAnything 自动下载） |

### FastVGGT 专用参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `fastvggt_merging` | 0 | Token Merging 参数（0=禁用） |
| `fastvggt_merge_ratio` | 0.9 | Token Merge 比例 (0.0-1.0) |
| `fastvggt_depth_conf_thresh` | 3.0 | 深度置信度阈值 |

### 特征跟踪参数（仅 `dense_feature_points` 模式）

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `max_query_pts` | 12288 | 每个查询帧最大特征点数 |
| `query_frame_num` | 6 | 查询帧数量，**建议与 `min_images_for_scale` 保持一致** |

> 💡 **3DGS 兼容性提示**：为确保生成的 COLMAP 输出满足 3DGS 训练要求（每个 3D 点至少被 3 个 2D 点观测到），`query_frame_num` 必须 ≥ `min_images_for_scale`。推荐设置两者相等。

### 重建质量参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `max_reproj_error` | 5.0 | 最大重投影误差（像素） |
| `max_points3D_val` | 1000000 | 3D 点坐标最大绝对值 |
| `pred_vis_scores_thres_value` | 0.8 | 特征点可见性阈值 |
| `filter_edge_margin` | 100.0 | 边缘过滤范围（像素） |

### 可视化参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `enable_visualization` | True | 启用 Viser 可视化 |
| `visualization_mode` | 'merged' | 可视化模式：'aligned' 或 'merged' |

### 地理坐标导出参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `export_georef` | True | 是否导出地理坐标 |
| `target_crs` | 'auto_utm' | 目标坐标系 |

支持的坐标系：
- `auto_utm`: 自动检测 UTM 区域
- `EPSG:3857`: Web Mercator
- `EPSG:4326`: WGS84 经纬度
- `EPSG:XXXX`: 任意 EPSG 代码

## 输出格式

运行完成后，输出目录结构如下：

```
output/
├── global_sfm/                    # 全局 SfM 结果
│   └── enu/
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
├── temp_aligned/                  # 每批次对齐后的重建
│   ├── 0_6/
│   ├── 4_10/
│   └── ...
├── temp_merged/                   # 合并后的重建
│   ├── merged_1/
│   ├── merged_2/
│   └── ...
│       ├── cameras.txt
│       ├── images.txt
│       ├── points3D.txt
│       ├── points3D.ply
│       └── points3D.las
├── temp_merged_points_only/       # points_only 模式输出
│   ├── merged_1.ply
│   ├── merged_1.las
│   └── ...
└── temp_merged_reconstruction_georeferenced/  # 地理坐标输出
    ├── cameras.txt
    ├── images.txt
    ├── points3D.txt
    ├── sparse_points.ply
    └── sparse_points.las
```

### 输出文件格式

| 格式 | 描述 |
|------|------|
| `.txt` | COLMAP 文本格式 |
| `.bin` | COLMAP 二进制格式 |
| `.ply` | 标准 PLY 点云格式 |
| `.las` | LAS 点云格式（支持 GIS 软件） |

## 示例数据集

项目提供了多个示例数据集用于测试：

```
examples/
├── Ganluo_images/          # 甘洛数据集
├── Tazishan/               # 塔子山数据集
├── SWJTU_gongdi/           # 西南交大工地数据集
├── SWJTU_7th_teaching_building/  # 西南交大七教数据集
├── HuaPo/                  # 滑坡数据集
├── WenChuan/               # 汶川数据集
└── Comprehensive_building_sel/   # 综合建筑数据集
```

## 可视化

系统支持基于 [Viser](https://github.com/nerfstudio-project/viser) 的实时 3D 可视化：

```python
# 启用可视化（默认开启）
matcher = IncrementalFeatureMatcherSfM(
    ...,
    enable_visualization=True,
    visualization_mode='merged',  # 显示合并后的点云
)
```

启动后，在浏览器中访问 `http://localhost:8080` 查看：

- 📍 相机位姿（frustum 显示）
- 🔵 3D 点云
- 🖼️ 影像缩略图

## 致谢

本项目基于以下优秀的开源项目：

- [MapAnything](https://github.com/facebookresearch/map-anything) - Meta 的端到端 3D 重建模型
- [VGGT](https://github.com/facebookresearch/vggt) - 视觉几何基础模型
- [COLMAP](https://colmap.github.io/) - 经典 SfM/MVS 框架
- [pycolmap](https://github.com/colmap/pycolmap) - COLMAP Python 绑定
- [Viser](https://github.com/nerfstudio-project/viser) - 3D 可视化工具
- [laspy](https://github.com/laspy/laspy) - LAS 文件读写库

## 引用

如果您使用了本项目，请引用以下论文：

```bibtex
@article{jia2025dronevggt,
  title={Modular Incremental Dense 3D Reconstruction from UAV Imagery with Plug-and-Play Visual Geometry Transformer Models},
  author={Jia, Zhihao and Ding, Yulin and Hu, Han},
  journal={},
  year={2025},
  url={https://github.com/Henrik-JIA/Drone-VGGT}
}

@inproceedings{wang2025vggt,
  title={Vggt: Visual geometry grounded transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5294--5306},
  year={2025}
}

@article{shen2025fastvggt,
  title={Fastvggt: Training-free acceleration of visual geometry transformer},
  author={Shen, You and Zhang, Zhipeng and Qu, Yansong and Zheng, Xiawu and Ji, Jiayi and Zhang, Shengchuan and Cao, Liujuan},
  journal={arXiv preprint arXiv:2509.02560},
  year={2025}
}

@article{keetha2025mapanything,
  title={Mapanything: Universal feed-forward metric 3d reconstruction},
  author={Keetha, Nikhil and M{\"u}ller, Norman and Sch{\"o}nberger, Johannes and Porzi, Lorenzo and Zhang, Yuchen and Fischer, Tobias and Knapitsch, Arno and Zauss, Duncan and Weber, Ethan and Antunes, Nelson and others},
  journal={arXiv preprint arXiv:2509.13414},
  year={2025}
}
```

## 许可证

本项目采用 [Apache 2.0 许可证](LICENSE)。

---

<div align="center">

**如有问题或建议，欢迎提交 Issue 或 Pull Request！**

</div>
