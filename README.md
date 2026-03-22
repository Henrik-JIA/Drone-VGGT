<div align="center">

# Drone-VGGT

**Modular Incremental Dense 3D Reconstruction from UAV Imagery with Plug-and-Play Visual Geometry Transformer Models**

[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://drone-vggt.github.io/)
[![中文文档](https://img.shields.io/badge/中文-README-orange.svg)](assets/README_CN.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

## Overview

Drone-VGGT is an incremental 3D reconstruction system designed specifically for UAV aerial imagery. The system supports multiple deep learning models (MapAnything, VGGT, FastVGGT), enabling efficient incremental point cloud reconstruction and merging, with real-time visualization and georeferenced coordinate export.

![Incremental Dense Point Cloud Reconstruction](./assets/记录03推理密集点云Ganluo.gif)

### Key Features

- 🚀 **Incremental Processing**: Frame-by-frame image addition with real-time reconstruction updates
- 🧠 **Multi-Model Support**: MapAnything, VGGT, and FastVGGT models available
- 🔗 **Intelligent Point Cloud Merging**: Four merging strategies (full, confidence, confidence_blend, points_only)
- 🌍 **Georeferenced Export**: Automatic UTM zone detection with multi-coordinate system support
- 👁️ **Real-time Visualization**: Viser-based 3D point cloud and camera pose visualization
- 📊 **GPS/XMP Metadata**: Automatic pose extraction from UAV imagery

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
  - [Incremental Reconstruction Pipeline](#incremental-reconstruction-pipeline)
  - [Model Selection](#model-selection)
  - [Point Cloud Merging Methods](#point-cloud-merging-methods)
- [API Reference](#api-reference)
- [Configuration Parameters](#configuration-parameters)
- [Output Format](#output-format)
- [Example Datasets](#example-datasets)
- [Acknowledgements](#acknowledgements)

## Installation

### Requirements

- Python 3.10 - 3.11
- PyTorch 2.3.1
- CUDA 11.8
- [Pixi](https://pixi.sh/) (recommended) or Conda

### Option 1: Install with Pixi (Recommended)

[Pixi](https://pixi.sh/) is a modern package manager that automatically handles Conda and PyPI dependencies.

```bash
# 1. Install Pixi (if not already installed)
# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex

# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone the repository
git clone https://github.com/Henrik-JIA/Drone-VGGT.git
cd Drone-VGGT

# 3. Install all dependencies and activate environment
pixi install

# 4. Complete installation (including local packages and third-party submodules)
pixi run setup

# 5. Activate environment
pixi shell
```

#### Common Pixi Commands

```bash
# View environment info (PyTorch version, CUDA status, etc.)
pixi run info

# Run incremental SfM reconstruction
pixi run run-sfm

# Run Delaunay meshing
pixi run run-mesh

# Generate DSM
pixi run run-dsm

# Code linting
pixi run lint
pixi run format
```

#### Pixi Environment Description

The project provides three pre-configured environments:

| Environment | Description | Activation Command |
|-------------|-------------|-------------------|
| `default` | Full environment with all features | `pixi shell` |
| `dev` | Development environment with testing tools | `pixi shell -e dev` |
| `minimal` | Minimal environment with core dependencies only | `pixi shell -e minimal` |

### Option 2: Install with Conda/Pip

```bash
# Clone the repository
git clone https://github.com/Henrik-JIA/Drone-VGGT.git
cd Drone-VGGT

# Create conda environment
conda create -n drone-vggt python=3.11 -y
conda activate drone-vggt

# Install PyTorch (CUDA 11.8)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -e .

# Install optional dependencies (visualization, georeferenced export, etc.)
pip install -e ".[all]"

# Install third-party submodules
pip install -e third/vggt --no-deps
pip install -e third/fastvggt --no-deps
pip install git+https://github.com/cvg/LightGlue.git --no-deps
```

### Model Weights

Download the corresponding weights based on your chosen model:

| Model | Weight Path | Download Link |
|-------|-------------|---------------|
| MapAnything | Auto-download from HuggingFace | [facebook/map-anything](https://huggingface.co/facebook/map-anything) |
| VGGT | `weights/vggt/model.pt` | [VGGT Official Repository](https://github.com/facebookresearch/vggt) |
| FastVGGT | `weights/fastvggt/model_tracker_fixed_e20.pt` | Contact authors |

## Quick Start

### Basic Usage

```python
from pathlib import Path
from SfM.incremental_feature_matcher import run_incremental_feature_matching

# Configure input/output paths
input_dir = Path("examples/your_drone_images/images")
output_dir = Path("output/your_project")

# Get all image files
image_files = sorted(input_dir.glob("*.JPG"))

# Run incremental reconstruction
success = run_incremental_feature_matching(
    image_paths=image_files,
    output_dir=output_dir,
    model_type='mapanything',      # 'mapanything' | 'vggt' | 'fastvggt'
    min_images_for_scale=6,        # Number of images per batch
    overlap=2,                     # Overlap images between batches
    merge_method='confidence',     # Point cloud merging method
    enable_visualization=True,     # Enable real-time visualization
    export_georef=True,            # Export georeferenced coordinates
    verbose=True,
)
```

### Command Line Execution

```bash
# Run the main script directly
python SfM/incremental_feature_matcher.py
```

Modify the configuration parameters at the bottom of the script to process your own dataset:

```python
# Modify in the if __name__ == "__main__": section
input_dir = Path(r"your/image/folder/images")
output_dir = Path(r"your/output/folder")
MODEL_TYPE = 'vggt'  # Select model
```

## Core Features

### Incremental Reconstruction Pipeline

The system employs an incremental processing strategy, processing large numbers of images in batches to significantly reduce memory consumption:

```
┌─────────────────────────────────────────────────────────────┐
│              Incremental Reconstruction Pipeline             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Batch 0: [img_0, img_1, img_2, img_3, img_4, img_5]        │
│           └── Inference → Reconstruction → Initial Cloud    │
│                                                             │
│  Batch 1: [img_4, img_5, img_6, img_7, img_8, img_9]        │
│           └── Inference → Reconstruction → Sim3 Align → Merge│
│                  ↑                                          │
│              Overlap Images                                 │
│                                                             │
│  Batch 2: [img_8, img_9, img_10, img_11, ...]               │
│           └── Inference → Reconstruction → Sim3 Align → Merge│
│                                                             │
│  ...                                                        │
│                                                             │
│  Final Output: Complete Merged Point Cloud + Georef Recon   │
└─────────────────────────────────────────────────────────────┘
```

### Model Selection

| Model | Features | Use Cases |
|-------|----------|-----------|
| **MapAnything** | End-to-end Transformer, supports multiple input modalities | General scenarios, high precision requirements |
| **VGGT** | Visual Geometry Grounded Transformer | Large-scale scenes, stability required |
| **FastVGGT** | Accelerated VGGT with Token Merging | Real-time processing, memory-constrained scenarios |

### Point Cloud Merging Methods

The system provides four point cloud merging strategies:

#### 1. `points_only` (Recommended for large-scale data)

```python
merge_method='points_only'
```

- ✅ Most lightweight, maintains only point cloud arrays
- ✅ Highest memory efficiency
- ✅ Supports incremental voxel deduplication
- ❌ Does not maintain pycolmap Reconstruction structure

**Merging Process**:
```
1. Estimate Sim3 transformation via common images
2. Identify overlapping regions using 2D pixel matching
3. Add only non-overlapping points (overlapping regions already exist in accumulated cloud)
4. Optional: Delayed voxel deduplication (executed when point count exceeds threshold)
```

#### 2. `confidence` (Recommended for high-quality output)

```python
merge_method='confidence'
```

- ✅ Selects optimal points based on confidence
- ✅ Maintains complete Reconstruction structure
- ✅ Supports precise pixel-level matching
- ❌ Higher memory consumption

#### 3. `confidence_blend` (Highest quality)

```python
merge_method='confidence_blend'
```

- ✅ Confidence-weighted blending
- ✅ Fusion band with smooth interpolation
- ✅ Density equalization
- ❌ Highest computational overhead

#### 4. `full` (Complete pipeline)

```python
merge_method='full'
```

- ✅ Uses complete merge_full_pipeline
- ✅ Supports multi-stage refinement alignment
- ❌ Slowest

### Reconstruction Types and 3DGS Support

The system supports two reconstruction types, determining point cloud generation and downstream applications:

| Reconstruction Type | Output Format | 3DGS Compatibility | Use Cases |
|--------------------|---------------|---------------------|-----------|
| `each_pixel_feature_points` | Dense point cloud (one 3D point per pixel) | ❌ Not supported | Pure point cloud visualization, DSM generation |
| `dense_feature_points` | COLMAP Reconstruction + Dense point cloud | ✅ Supported | 3DGS training, traditional MVS pipeline |

#### Why is `dense_feature_points` required for 3DGS?

3D Gaussian Splatting (3DGS) training requires standard COLMAP output format, including:
- `cameras.txt` / `cameras.bin` - Camera intrinsics
- `images.txt` / `images.bin` - Camera poses and 2D-3D correspondences
- `points3D.txt` / `points3D.bin` - 3D point cloud with observation information

Only `dense_feature_points` mode establishes 2D-3D correspondences across multiple views through feature tracking, generating a complete COLMAP Reconstruction structure.

#### Configuring 3DGS-Compatible Output

To generate output usable for 3DGS training, configure the following parameters correctly:

```python
success = run_incremental_feature_matching(
    image_paths=image_files,
    output_dir=output_dir,
    
    # Key parameter: must use dense_feature_points
    reconstruction_type='dense_feature_points',
    
    # Important: query_frame_num should match min_images_for_scale
    # This ensures every image participates in feature tracking, meeting 3DGS observation requirements
    min_images_for_scale=6,
    query_frame_num=6,  # Must be >= min_images_for_scale
    
    # Other parameters
    model_type='vggt',
    merge_method='confidence',
    verbose=True,
)
```

> ⚠️ **Important Note**: The `query_frame_num` parameter controls the number of query frames during feature tracking. To ensure every image establishes 2D-3D correspondences and each 3D point is observed by at least 3 2D points (basic 3DGS requirement), `query_frame_num` must match `min_images_for_scale`.

#### Output Directory Structure (3DGS Compatible)

When using `dense_feature_points` mode, the output directory will contain:

```
output/
├── temp_merged/
│   └── merged_N/
│       ├── cameras.txt      # Camera intrinsics ← Required by 3DGS
│       ├── images.txt       # Image poses + 2D-3D correspondences ← Required by 3DGS
│       ├── points3D.txt     # 3D point cloud + observation info ← Required by 3DGS
│       └── points3D.ply     # PLY format point cloud
└── temp_merged_reconstruction_georeferenced/
    └── ...                  # Georeferenced version
```

Can be used directly for 3DGS training:
```bash
# Using official gaussian-splatting implementation
python train.py -s output/temp_merged/merged_N/
```

## API Reference

### `IncrementalFeatureMatcherSfM` Class

Core class managing the complete lifecycle of incremental reconstruction.

```python
from SfM.incremental_feature_matcher import IncrementalFeatureMatcherSfM

matcher = IncrementalFeatureMatcherSfM(
    output_dir=Path("output"),
    reconstruction_type='each_pixel_feature_points',  # or 'dense_feature_points'
    model_type='mapanything',
    min_images_for_scale=6,
    overlap=2,
    merge_method='confidence',
    enable_visualization=True,
    verbose=True,
)
```

#### Main Methods

| Method | Description |
|--------|-------------|
| `add_image(image_path)` | Add and process a single image |
| `export_georeferenced(target_crs)` | Export reconstruction in georeferenced coordinate system |
| `export_utm()` | Export UTM coordinates (auto-detect zone) |
| `get_statistics()` | Get current statistics |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `merged_reconstruction` | `pycolmap.Reconstruction` | Merged reconstruction result |
| `merged_points_xyz` | `np.ndarray` | Merged point cloud coordinates (N, 3) |
| `merged_points_colors` | `np.ndarray` | Merged point cloud colors (N, 3) |
| `inference_reconstructions` | `List[Dict]` | Reconstruction results for each batch |

### `run_incremental_feature_matching` Function

Convenience function encapsulating the complete incremental reconstruction pipeline.

```python
from SfM.incremental_feature_matcher import run_incremental_feature_matching

success = run_incremental_feature_matching(
    image_paths=image_files,        # List of image paths
    output_dir=output_dir,          # Output directory
    
    # Reconstruction type: 'each_pixel_feature_points' (pure dense cloud) or 'dense_feature_points' (3DGS support)
    reconstruction_type='dense_feature_points',
    
    model_type='vggt',              # 'mapanything' | 'vggt' | 'fastvggt'
    model_path=None,                # Model weights path
    image_interval=1,               # Image interval
    min_images_for_scale=6,         # Images per batch
    overlap=2,                      # Batch overlap count
    
    # Feature tracking parameters (dense_feature_points mode only)
    # query_frame_num should match min_images_for_scale for 3DGS support
    query_frame_num=6,
    max_query_pts=12288,
    
    pred_vis_scores_thres_value=0.7,
    max_reproj_error=5.0,
    run_global_sfm_first=True,      # Run global SfM first
    merge_method='confidence',
    merge_voxel_size=0.5,           # Voxel size (meters)
    enable_visualization=True,
    visualization_mode='merged',    # 'aligned' | 'merged'
    export_georef=True,
    target_crs='auto_utm',
    verbose=True,
)
```

## Configuration Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reconstruction_type` | 'each_pixel_feature_points' | Reconstruction type: `'each_pixel_feature_points'` (pure dense cloud) or `'dense_feature_points'` (COLMAP format, 3DGS support) |
| `min_images_for_scale` | 6 | Number of images per batch |
| `overlap` | 2 | Number of overlapping images between adjacent batches |
| `merge_method` | 'confidence' | Point cloud merging method |
| `merge_voxel_size` | 1.5 | Voxel deduplication size (meters) |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | 'mapanything' | Model type |
| `model_path` | None | Model weights path (MapAnything auto-downloads) |

### FastVGGT-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fastvggt_merging` | 0 | Token Merging parameter (0=disabled) |
| `fastvggt_merge_ratio` | 0.9 | Token Merge ratio (0.0-1.0) |
| `fastvggt_depth_conf_thresh` | 3.0 | Depth confidence threshold |

### Feature Tracking Parameters (`dense_feature_points` mode only)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_query_pts` | 12288 | Maximum feature points per query frame |
| `query_frame_num` | 6 | Number of query frames, **recommended to match `min_images_for_scale`** |

> 💡 **3DGS Compatibility Tip**: To ensure generated COLMAP output meets 3DGS training requirements (each 3D point observed by at least 3 2D points), `query_frame_num` must be ≥ `min_images_for_scale`. Setting them equal is recommended.

### Reconstruction Quality Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_reproj_error` | 5.0 | Maximum reprojection error (pixels) |
| `max_points3D_val` | 1000000 | Maximum absolute value for 3D point coordinates |
| `pred_vis_scores_thres_value` | 0.8 | Feature point visibility threshold |
| `filter_edge_margin` | 100.0 | Edge filtering range (pixels) |

### Visualization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_visualization` | True | Enable Viser visualization |
| `visualization_mode` | 'merged' | Visualization mode: 'aligned' or 'merged' |

### Georeferenced Export Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `export_georef` | True | Whether to export georeferenced coordinates |
| `target_crs` | 'auto_utm' | Target coordinate system |

Supported coordinate systems:
- `auto_utm`: Automatic UTM zone detection
- `EPSG:3857`: Web Mercator
- `EPSG:4326`: WGS84 lat/lon
- `EPSG:XXXX`: Any EPSG code

## Output Format

After execution, the output directory structure is as follows:

```
output/
├── global_sfm/                    # Global SfM results
│   └── enu/
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
├── temp_aligned/                  # Aligned reconstruction per batch
│   ├── 0_6/
│   ├── 4_10/
│   └── ...
├── temp_merged/                   # Merged reconstruction
│   ├── merged_1/
│   ├── merged_2/
│   └── ...
│       ├── cameras.txt
│       ├── images.txt
│       ├── points3D.txt
│       ├── points3D.ply
│       └── points3D.las
├── temp_merged_points_only/       # points_only mode output
│   ├── merged_1.ply
│   ├── merged_1.las
│   └── ...
└── temp_merged_reconstruction_georeferenced/  # Georeferenced output
    ├── cameras.txt
    ├── images.txt
    ├── points3D.txt
    ├── sparse_points.ply
    └── sparse_points.las
```

### Output File Formats

| Format | Description |
|--------|-------------|
| `.txt` | COLMAP text format |
| `.bin` | COLMAP binary format |
| `.ply` | Standard PLY point cloud format |
| `.las` | LAS point cloud format (GIS software compatible) |

## Example Datasets

The project provides several example datasets for testing:

```
examples/
├── Ganluo_images/          # Ganluo dataset
├── Tazishan/               # Tazishan dataset
├── SWJTU_gongdi/           # SWJTU construction site dataset
├── SWJTU_7th_teaching_building/  # SWJTU 7th teaching building dataset
├── HuaPo/                  # Landslide dataset
├── WenChuan/               # Wenchuan dataset
└── Comprehensive_building_sel/   # Comprehensive building dataset
```

## Visualization

The system supports real-time 3D visualization based on [Viser](https://github.com/nerfstudio-project/viser):

```python
# Enable visualization (enabled by default)
matcher = IncrementalFeatureMatcherSfM(
    ...,
    enable_visualization=True,
    visualization_mode='merged',  # Display merged point cloud
)
```

After starting, access `http://localhost:8080` in your browser to view:

- 📍 Camera poses (frustum display)
- 🔵 3D point cloud
- 🖼️ Image thumbnails

## Acknowledgements

This project is built upon the following excellent open-source projects:

- [MapAnything](https://github.com/facebookresearch/map-anything) - Meta's end-to-end 3D reconstruction model
- [VGGT](https://github.com/facebookresearch/vggt) - Visual Geometry Grounded Transformer
- [COLMAP](https://colmap.github.io/) - Classic SfM/MVS framework
- [pycolmap](https://github.com/colmap/pycolmap) - COLMAP Python bindings
- [Viser](https://github.com/nerfstudio-project/viser) - 3D visualization tool
- [laspy](https://github.com/laspy/laspy) - LAS file I/O library

## Citation

If you use this project, please cite the following papers:

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

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

<div align="center">

**If you have questions or suggestions, feel free to submit an Issue or Pull Request!**

</div>
