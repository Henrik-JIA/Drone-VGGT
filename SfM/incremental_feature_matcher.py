#!/usr/bin/env python3
"""
Incremental feature extraction and matching for SfM using pycolmap.
Process images one by one: extract features, match with previous images, 
build tracks, and triangulate.
"""

import os
import sys
import copy
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image
import laspy
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import subprocess
import cv2


class TimingTracker:
    """用于跟踪代码各步骤耗时的工具类"""
    
    def __init__(self, output_dir: Path = None):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.current_step: Dict[str, float] = {}
        self.total_start_time: float = None
        self.output_dir = output_dir or Path("temp_time")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def start_total(self):
        """开始总计时"""
        self.total_start_time = time.time()
        
    def start(self, step_name: str):
        """开始某个步骤的计时"""
        self.current_step[step_name] = time.time()
        
    def end(self, step_name: str):
        """结束某个步骤的计时并记录"""
        if step_name in self.current_step:
            elapsed = time.time() - self.current_step[step_name]
            self.timings[step_name].append(elapsed)
            del self.current_step[step_name]
            return elapsed
        return 0.0
    
    @contextmanager
    def track(self, step_name: str):
        """上下文管理器用于跟踪步骤耗时"""
        self.start(step_name)
        try:
            yield
        finally:
            self.end(step_name)
    
    def get_total_time(self) -> float:
        """获取总耗时"""
        if self.total_start_time is None:
            return 0.0
        return time.time() - self.total_start_time
    
    def get_step_summary(self, step_name: str) -> Dict:
        """获取某个步骤的统计信息"""
        times = self.timings.get(step_name, [])
        if not times:
            return {"count": 0, "total": 0, "mean": 0, "min": 0, "max": 0}
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }
    
    def export_to_file(self, filename: str = None) -> Path:
        """导出计时信息到文本文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"timing_report_{timestamp}.txt"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("                    Incremental Dense Reconstruction Timing Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Runtime: {self.get_total_time():.2f} seconds ({self.get_total_time()/60:.2f} minutes)\n")
            f.write("=" * 80 + "\n\n")
            
            # 按类别分组输出
            categories = {
                "Initialization": ["global_sfm", "model_loading"],
                "Per-Image Processing": ["add_image", "initialize_image", "run_inference"],
                "Batch Processing": ["predict_tracks", "build_pycolmap", "merge_reconstruction"],
                "Export": ["export_georef", "export_dsm", "export_dsm_georef", "export_fastgs"],
            }
            
            all_listed_steps = set()
            for cat_name, steps in categories.items():
                f.write(f"\n{'─' * 40}\n")
                f.write(f"  {cat_name}\n")
                f.write(f"{'─' * 40}\n")
                
                for step in steps:
                    all_listed_steps.add(step)
                    summary = self.get_step_summary(step)
                    if summary["count"] > 0:
                        f.write(f"\n  [{step}]\n")
                        f.write(f"    调用次数: {summary['count']}\n")
                        f.write(f"    总耗时:   {summary['total']:.3f}s ({summary['total']/60:.2f}min)\n")
                        f.write(f"    平均耗时: {summary['mean']:.3f}s\n")
                        f.write(f"    最小耗时: {summary['min']:.3f}s\n")
                        f.write(f"    最大耗时: {summary['max']:.3f}s\n")
            
            # 输出其他未分类的步骤
            other_steps = set(self.timings.keys()) - all_listed_steps
            if other_steps:
                f.write(f"\n{'─' * 40}\n")
                f.write(f"  Other Steps\n")
                f.write(f"{'─' * 40}\n")
                for step in sorted(other_steps):
                    summary = self.get_step_summary(step)
                    if summary["count"] > 0:
                        f.write(f"\n  [{step}]\n")
                        f.write(f"    调用次数: {summary['count']}\n")
                        f.write(f"    总耗时:   {summary['total']:.3f}s ({summary['total']/60:.2f}min)\n")
                        f.write(f"    平均耗时: {summary['mean']:.3f}s\n")
            
            # 总结表格
            f.write(f"\n\n{'=' * 80}\n")
            f.write("                              Summary Table\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Step':<30} {'Count':>8} {'Total(s)':>12} {'Mean(s)':>10} {'%':>8}\n")
            f.write("-" * 80 + "\n")
            
            total_time = self.get_total_time()
            sorted_steps = sorted(
                self.timings.items(),
                key=lambda x: sum(x[1]),
                reverse=True
            )
            
            for step_name, times in sorted_steps:
                step_total = sum(times)
                step_mean = step_total / len(times) if times else 0
                pct = (step_total / total_time * 100) if total_time > 0 else 0
                f.write(f"{step_name:<30} {len(times):>8} {step_total:>12.3f} {step_mean:>10.3f} {pct:>7.1f}%\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"{'TOTAL':<30} {'-':>8} {total_time:>12.3f} {'-':>10} {'100.0':>7}%\n")
            f.write("=" * 80 + "\n")
        
        print(f"Timing report saved to: {output_path}")
        return output_path


# 全局计时器实例（在 run_incremental_feature_matching 中初始化）
_timing_tracker: Optional[TimingTracker] = None


def get_timing_tracker() -> Optional[TimingTracker]:
    """获取全局计时器实例"""
    return _timing_tracker


class GPUMemoryTracker:
    """用于跟踪GPU显存使用情况的工具类
    
    桌面GPU：使用 pynvml 获取真实的GPU显存占用（与任务管理器一致）
    Tegra设备（统一内存）：PyTorch CUDA stats 在 Tegra 上不准确，
        改用进程级内存监控（/proc/self/status VmRSS/VmHWM）
    """
    
    def __init__(self, output_dir: Path = None):
        self.memory_records: List[Dict] = []
        self.output_dir = output_dir or Path("temp_memory")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.peak_gpu_memory: float = 0.0
        self.initial_gpu_memory: float = 0.0
        self.nvml_initialized: bool = False
        self.nvml_handle = None
        self.is_tegra: bool = False
        
        self._detect_platform()
        
        if not self.is_tegra:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_initialized = True
            except Exception as e:
                print(f"Warning: pynvml not available, using torch.cuda for memory tracking: {e}")
                self.nvml_initialized = False
    
    def _detect_platform(self):
        """检测是否运行在 Tegra 平台（统一内存架构）"""
        if not torch.cuda.is_available():
            return
        try:
            gpu_name = torch.cuda.get_device_properties(0).name.lower()
            tegra_keywords = ['tegra', 'orin', 'xavier', 'thor', 'jetson']
            self.is_tegra = any(kw in gpu_name for kw in tegra_keywords)
            if not self.is_tegra:
                import platform
                self.is_tegra = 'tegra' in platform.release().lower()
        except Exception:
            self.is_tegra = False
        
        if self.is_tegra:
            print(f"[GPUMemoryTracker] Tegra platform detected — using process RSS for memory tracking")

    @staticmethod
    def _read_proc_status_mb() -> Dict[str, float]:
        """从 /proc/self/status 读取进程内存（适用于 Linux/Tegra）"""
        result = {"VmRSS": 0.0, "VmHWM": 0.0, "VmSize": 0.0}
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    for key in result:
                        if line.startswith(key + ':'):
                            result[key] = int(line.split()[1]) / 1024.0
        except Exception:
            pass
        return result
    
    def _get_gpu_memory_mb(self) -> Dict[str, float]:
        """获取GPU/内存使用情况（MB）"""
        result = {
            "real_used": 0.0,
            "total": 0.0,
            "torch_allocated": 0.0,
            "torch_reserved": 0.0,
            "proc_rss": 0.0,
            "proc_hwm": 0.0,
        }
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["torch_allocated"] = torch.cuda.memory_allocated() / (1024 ** 2)
            result["torch_reserved"] = torch.cuda.memory_reserved() / (1024 ** 2)
            result["total"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        
        if self.is_tegra:
            proc = self._read_proc_status_mb()
            result["proc_rss"] = proc["VmRSS"]
            result["proc_hwm"] = proc["VmHWM"]
            result["real_used"] = proc["VmRSS"]
        elif self.nvml_initialized and self.nvml_handle:
            try:
                import pynvml
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                result["real_used"] = mem_info.used / (1024 ** 2)
                result["total"] = mem_info.total / (1024 ** 2)
            except Exception:
                result["real_used"] = result["torch_reserved"]
        else:
            result["real_used"] = result["torch_reserved"]
        
        return result
    
    def start_monitoring(self):
        """开始监控，记录初始状态"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        mem_info = self._get_gpu_memory_mb()
        self.initial_gpu_memory = mem_info["real_used"]
        self.peak_gpu_memory = mem_info["real_used"]
        self.record("initial_state", "开始监控", 0)
    
    def record(self, step_name: str, description: str = "", image_idx: int = 0):
        """记录当前内存使用状态"""
        mem_info = self._get_gpu_memory_mb()
        
        current_used = mem_info["real_used"]
        self.peak_gpu_memory = max(self.peak_gpu_memory, current_used)
        
        # Tegra: 同时用 VmHWM 作为峰值的补充（OS级别的峰值跟踪，不会遗漏）
        if self.is_tegra and mem_info["proc_hwm"] > 0:
            self.peak_gpu_memory = max(self.peak_gpu_memory, mem_info["proc_hwm"])
        
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_idx": image_idx,
            "step_name": step_name,
            "description": description,
            "gpu_real_used_mb": current_used,
            "torch_allocated_mb": mem_info["torch_allocated"],
            "torch_reserved_mb": mem_info["torch_reserved"],
            "proc_rss_mb": mem_info["proc_rss"],
            "proc_hwm_mb": mem_info["proc_hwm"],
            "gpu_total_mb": mem_info["total"],
            "gpu_usage_percent": (current_used / mem_info["total"] * 100) if mem_info["total"] > 0 else 0,
        }
        self.memory_records.append(record)
    
    def export_to_file(self, filename: str = None) -> Path:
        """导出内存使用信息到文本文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gpu_memory_report_{timestamp}.txt"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("                         GPU Memory Usage Report for Model Inference\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            
            # GPU信息
            f.write("─" * 60 + "\n")
            f.write("  GPU / Platform Information\n")
            f.write("─" * 60 + "\n")
            
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                f.write(f"  GPU Device:     {gpu_props.name}\n")
                f.write(f"  GPU Total Mem:  {gpu_props.total_memory / (1024**3):.2f} GB\n")
            else:
                f.write("  GPU: Not available\n")
            
            if self.is_tegra:
                f.write(f"  Platform:       Tegra (Unified Memory Architecture)\n")
                f.write(f"  Tracking Mode:  Process RSS (/proc/self/status VmRSS)\n")
                f.write(f"  Note:           On Tegra, GPU shares system RAM. torch.cuda stats\n")
                f.write(f"                  are unreliable; process RSS is the true indicator.\n")
            else:
                mode = "pynvml" if self.nvml_initialized else "torch.cuda"
                f.write(f"  Platform:       Discrete GPU\n")
                f.write(f"  Tracking Mode:  {mode}\n")
            f.write("\n")
            
            # 峰值使用
            f.write("─" * 60 + "\n")
            f.write("  Peak Memory Usage\n")
            f.write("─" * 60 + "\n")
            f.write(f"  Initial:  {self.initial_gpu_memory:>10.2f} MB  ({self.initial_gpu_memory/1024:.2f} GB)\n")
            f.write(f"  Peak:     {self.peak_gpu_memory:>10.2f} MB  ({self.peak_gpu_memory/1024:.2f} GB)\n")
            increase = self.peak_gpu_memory - self.initial_gpu_memory
            f.write(f"  Increase: {increase:>10.2f} MB  ({increase/1024:.2f} GB)\n")
            
            if self.is_tegra:
                # 取最后一条记录的 VmHWM 作为 OS 报告的历史峰值
                if self.memory_records:
                    last_hwm = self.memory_records[-1].get("proc_hwm_mb", 0)
                    f.write(f"\n  [OS Peak (VmHWM)]: {last_hwm:.2f} MB ({last_hwm/1024:.2f} GB)\n")
            f.write("\n")
            
            # ==================== 按步骤类型统计 ====================
            f.write("=" * 110 + "\n")
            f.write("                              Average Memory by Step Type\n")
            f.write("=" * 110 + "\n\n")
            
            step_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"real": [], "torch_alloc": []})
            for record in self.memory_records:
                step_stats[record["step_name"]]["real"].append(record["gpu_real_used_mb"])
                step_stats[record["step_name"]]["torch_alloc"].append(record["torch_allocated_mb"])
            
            col_label = "RSS Avg(MB)" if self.is_tegra else "Real Avg(MB)"
            col_label2 = "RSS Max(MB)" if self.is_tegra else "Real Max(MB)"
            f.write(f"{'Step':<30} {'Count':>6} {col_label:>14} {col_label2:>14} {'Torch Avg(MB)':>14} {'Torch Max(MB)':>14}\n")
            f.write("-" * 110 + "\n")
            
            for step_name in sorted(step_stats.keys()):
                real_values = step_stats[step_name]["real"]
                torch_values = step_stats[step_name]["torch_alloc"]
                count = len(real_values)
                real_avg = sum(real_values) / count if count > 0 else 0
                real_max = max(real_values) if real_values else 0
                torch_avg = sum(torch_values) / count if count > 0 else 0
                torch_max = max(torch_values) if torch_values else 0
                f.write(f"{step_name:<30} {count:>6} {real_avg:>14.2f} {real_max:>14.2f} {torch_avg:>14.2f} {torch_max:>14.2f}\n")
            
            f.write("-" * 110 + "\n\n")
            
            # ==================== 关键步骤增量统计 ====================
            f.write("─" * 60 + "\n")
            delta_label = "Process RSS Delta" if self.is_tegra else "Real GPU Memory Delta"
            f.write(f"  Key Step Memory Delta ({delta_label})\n")
            f.write("─" * 60 + "\n\n")
            
            def calc_delta_stats(before_key, after_key, label):
                before_data = step_stats.get(before_key, {"real": []})
                after_data = step_stats.get(after_key, {"real": []})
                before_real = before_data["real"]
                after_real = after_data["real"]
                
                if before_real and after_real:
                    num = min(len(before_real), len(after_real))
                    real_deltas = [after_real[i] - before_real[i] for i in range(num)]
                    f.write(f"  [{label}] (Calls: {num})\n")
                    f.write(f"    Delta - Avg: {sum(real_deltas)/num:+.2f} MB, "
                            f"Min: {min(real_deltas):+.2f} MB, Max: {max(real_deltas):+.2f} MB\n\n")
            
            calc_delta_stats("model_loading_before", "model_loading_after", "Model Loading")
            calc_delta_stats("model_inference_before", "model_inference_after", "Model Inference")
            calc_delta_stats("cuda_cache_before", "cuda_cache_after", "CUDA Cache Cleanup")
            
            # ==================== 详细记录表格 ====================
            if self.is_tegra:
                f.write("\n" + "=" * 140 + "\n")
                f.write("                                    Detailed Memory Records\n")
                f.write("  Note: Tegra unified memory — 'RSS' = process resident memory, 'HWM' = peak RSS (OS-tracked)\n")
                f.write("=" * 140 + "\n")
                f.write(f"{'Timestamp':<20} {'Img':>4} {'Step':<28} {'RSS(MB)':>12} {'HWM(MB)':>12} {'Torch Alloc':>12} {'Torch Rsv':>12} {'Usage%':>8}\n")
                f.write("-" * 140 + "\n")
                
                for record in self.memory_records:
                    f.write(f"{record['timestamp']:<20} "
                           f"{record['image_idx']:>4} "
                           f"{record['step_name'][:28]:<28} "
                           f"{record['gpu_real_used_mb']:>12.2f} "
                           f"{record.get('proc_hwm_mb', 0):>12.2f} "
                           f"{record['torch_allocated_mb']:>12.2f} "
                           f"{record['torch_reserved_mb']:>12.2f} "
                           f"{record['gpu_usage_percent']:>7.1f}%\n")
                
                f.write("=" * 140 + "\n")
            else:
                f.write("\n" + "=" * 120 + "\n")
                f.write("                                    Detailed Memory Records\n")
                f.write("  Note: 'Real Used' matches Task Manager, 'Torch Alloc/Reserved' are PyTorch internal values\n")
                f.write("=" * 120 + "\n")
                f.write(f"{'Timestamp':<20} {'Img':>4} {'Step':<28} {'Real Used(MB)':>14} {'Torch Alloc':>12} {'Torch Rsv':>12} {'Usage%':>8}\n")
                f.write("-" * 120 + "\n")
                
                for record in self.memory_records:
                    f.write(f"{record['timestamp']:<20} "
                           f"{record['image_idx']:>4} "
                           f"{record['step_name'][:28]:<28} "
                           f"{record['gpu_real_used_mb']:>14.2f} "
                           f"{record['torch_allocated_mb']:>12.2f} "
                           f"{record['torch_reserved_mb']:>12.2f} "
                           f"{record['gpu_usage_percent']:>7.1f}%\n")
                
                f.write("=" * 120 + "\n")
        
        print(f"GPU Memory report saved to: {output_path}")
        return output_path


# 全局GPU显存跟踪器实例
_gpu_memory_tracker: Optional[GPUMemoryTracker] = None


def get_gpu_memory_tracker() -> Optional[GPUMemoryTracker]:
    """获取全局GPU显存跟踪器实例"""
    return _gpu_memory_tracker


# UTM coordinate conversion imports (conditional)
try:
    import pymap3d as pm
    import pyproj
    UTM_EXPORT_AVAILABLE = True
except ImportError:
    UTM_EXPORT_AVAILABLE = False
    print("Warning: pymap3d or pyproj not available. UTM export will not work. Install with: pip install pymap3d pyproj")

current_dir = Path(__file__).parent
project_root = current_dir.parent  # drone-map-anything 根目录
third_dir = project_root / "third" / "vggt"  # 指向 third/vggt 目录（vggt项目根目录）
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(third_dir) not in sys.path:
    sys.path.insert(0, str(third_dir))

from feature_matcher import FeatureMatcherSfM
from merge.merge_full_pipeline import merge_reconstructions
from merge.merge_confidence_blend import (
    merge_two_reconstructions as merge_by_confidence_blend,
)
from merge.merge_confidence import merge_two_reconstructions as merge_by_confidence
from merge.merge_points_only import merge_all_reconstructions_points_only, save_ply_binary
from convert.convert_to_fastgs import create_fastgs_structure
from utils.voxel_downsample import _voxel_dedup, voxel_dedup
from sfm_visualizer import SfMVisualizer
from reconstruction_alignment import (
    rescale_reconstruction_to_original_size,
    align_reconstruction_by_overlap,
    find_single_images_pair_matches,
    estimate_sim3_transform,
)
from reconstruction_rename import rename_colmap_recons_and_rescale_camera
from utils.gps import extract_gps_from_image, lat_lon_to_enu
from utils.xmp import parse_xmp_tags
from utils.las_export import (
    export_reconstruction_to_las,
    export_points_to_las,
)
from utils.georef import (
    export_reconstruction_georeferenced,
    GeoreferencedExporter,
)
from mapanything.utils.image import preprocess_inputs
from mapanything.third_party.projection import project_3D_points_np, project_3D_points
from mapanything.models import MapAnything
from mapanything.third_party.track_predict import predict_tracks
from mapanything.third_party.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
    batch_np_matrix_to_pycolmap_with_rename,
)
from mapanything.utils.image import rgb

# Model loader imports
from load_model import (
    create_model_loader,
    is_vggt_available,
    is_fastvggt_available,
)

# Check model availability
VGGT_AVAILABLE = is_vggt_available()
FASTVGGT_AVAILABLE = is_fastvggt_available()

def cam_from_enu_transform(roll, pitch, yaw):
    """
    Returns the transformation matrix from ENU to camera coordinates.
    
    Args:
        roll: Gimbal roll angle in degrees
        pitch: Gimbal pitch angle in degrees
        yaw: Gimbal yaw angle in degrees
    
    Returns:
        3x3 rotation matrix from ENU to camera coordinates
    """
    # ENU to NED
    ned_from_enu = R.align_vectors(
        a=[[0, 1, 0], [1, 0, 0], [0, 0, -1]], 
        b=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )[0].as_matrix()

    # Gimbal rotation in NED (ZYX order)
    ned_from_gimbal = R.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()
    gimbal_from_ned = ned_from_gimbal.T

    # Camera from NED
    cam_from_ned = R.align_vectors(
        a=[[0, 0, 1], [1, 0, 0], [0, 1, 0]], 
        b=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )[0].as_matrix()

    cam_from_enu = cam_from_ned @ gimbal_from_ned @ ned_from_enu
    return cam_from_enu

class IncrementalFeatureMatcherSfM:
    """Incremental feature extraction and matching using pycolmap.
    
    This class processes images one by one and stores their intrinsic and extrinsic parameters.
    """

    def __init__(
        self,
        output_dir: Path,
        reconstruction_type: str = 'dense_feature_points',  # 'dense_feature_points' | 'each_pixel_feature_points'
        model_type: str = 'mapanything',  # 'mapanything' | 'vggt' | 'fastvggt'
        model_path: Optional[str] = None,  # 模型权重路径（VGGT/FastVGGT需要）
        min_images_for_scale: int = 2,
        overlap: int = 1,
        max_reproj_error: float = 10.0,
        max_points3D_val: int = 100000,  # 增大默认值，5000 对航拍场景太小
        min_inlier_per_frame: int = 32,
        pred_vis_scores_thres_value: float = 0.3,
        min_visible_frames: int = 2,  # 3D点至少在多少帧中可见才保留（仅 each_pixel_feature_points 模式）
        max_sampled_points: int = 100000,  # 每帧最大采样点数（仅 each_pixel_feature_points 模式）
        filter_edge_margin: float = 10.0,  # 边缘过滤范围（像素），默认10，设为0禁用
        merge_voxel_size: float = 1.0,  # 点云合并时的体素大小（米）
        merge_boundary_filter: bool = True,  # 是否启用边界过滤
        merge_statistical_filter: bool = False,  # 是否启用统计过滤
        merge_method: str = 'confidence_blend',  # 'full' | 'confidence' | 'confidence_blend' | 'points_only' 合并方式
        points_merge_mode: str = 'fast',  # 'fast' | 'quality'，points_only 模式下的合并策略
        enable_visualization: bool = True,
        visualization_mode: str = 'merged',  # 'aligned' | 'merged'，点云可视化模式
        # 特征点跟踪参数（仅 reconstruction_type='dense_feature_points' 模式有效）
        max_query_pts: int = 4096,  # 每个查询帧最大特征点数 4096 8192 12288
        query_frame_num: int = 3,    # 查询帧数量（建议 >= min_images_for_scale）
        # FastVGGT 特有参数
        fastvggt_merging: int = 0,  # FastVGGT token merging 参数
        fastvggt_merge_ratio: float = 0.9,  # FastVGGT token merge ratio (0.0-1.0)
        fastvggt_depth_conf_thresh: float = 3.0,  # FastVGGT 深度置信度阈值
        # 内存优化参数（边缘设备可设为 1 以降低峰值内存）
        memory_keep_batches: int = 2,  # 保留的批次数，1=最省内存，2=默认（重叠处理）
        batch_sfm_coord_mode: str = 'global_enu',  # 'global_enu' | 'wgs84'，batch SfM 坐标系模式
        all_image_paths: Optional[List[Path]] = None,  # 全部原始影像路径（间隔选取前）
        image_interval: int = 1,  # 影像选取间隔，用于在 batch SfM 时补充中间影像
        enable_batch_sfm: bool = True,  # 是否对每个 batch 运行传统 SfM 构建稀疏点云（用于对齐参考）
        verbose: bool = False,
    ):
        """Initialize incremental feature matcher.
        
        Args:
            output_dir: Directory for output files
            reconstruction_type: Type of reconstruction to use, 'dense_feature_points' | 'each_pixel_feature_points'
            model_type: Type of model to use, 'mapanything' | 'vggt'
            model_path: Path to model weights (required for VGGT, optional for MapAnything)
            min_images_for_scale: Minimum number of images before calculating scale.
                                  2 = calculate from 2nd image (default)
                                  3 = calculate from 3rd image
                                  4 = calculate from 4th image
                                  etc.
            overlap: Number of overlapping images between consecutive reconstructions
            max_reproj_error: Maximum reprojection error (in pixels) for filtering tracks
            max_points3D_val: Per-component absolute-value threshold for 3D points (a point is kept only if |x|, |y|, and |z| are all less than this value).
            min_inlier_per_frame: Minimum inlier count per frame for valid BA
            pred_vis_scores_thres_value: Visibility confidence threshold for tracks
            filter_edge_margin: Edge margin for filtering points (in pixels), default 10, set to 0 to disable
            merge_voxel_size: Voxel size for point cloud merging (in meters), default 1.0
            merge_boundary_filter: Whether to enable boundary filtering during merge, default True
            merge_statistical_filter: Whether to enable statistical filtering during merge, default False
            merge_method: Merge method selection:
                'full' - use merge_full_pipeline.py with full pipeline
                'confidence' - use merge_confidence.py with simple confidence-based selection
                'confidence_blend' - use merge_confidence_blend.py with confidence-based selection 
                                     and smooth edge blending/interpolation (default)
                'points_only' - use merge_points_only.py for lightweight point cloud only merge
                                (no pycolmap Reconstruction, faster, outputs xyz/colors only)
            enable_visualization: Whether to start viser server for visualization
            visualization_mode: Point cloud visualization mode, 'aligned' (per batch) or 'merged' (unified)
            memory_keep_batches: Number of batches to keep for overlap (1=min memory for edge, 2=default)
            enable_batch_sfm: Whether to run traditional SfM for each batch to build sparse point cloud
                True - run SfM for pcl_alignment (default)
                False - skip SfM, fall back to image_alignment mode (faster, suitable when GPS is accurate)
            batch_sfm_coord_mode: Batch SfM coordinate mode:
                'global_enu' - 平移到全局 ENU 坐标系（默认，推荐用于完整对齐流程）
                'wgs84' - 将坐标转换为 WGS84 经纬度高程 (lat, lon, alt)，
                          每个点通过 pymap3d.enu2geodetic() 从局部 ENU 转到地理坐标。
                          注意：WGS84 模式下坐标为（度, 度, 米），基于度量的对齐精度可能受影响。
            verbose: Enable verbose logging
        """
        # Model type validation
        if model_type not in ['mapanything', 'vggt', 'fastvggt']:
            raise ValueError(f"model_type must be 'mapanything', 'vggt', or 'fastvggt', got: {model_type}")
        if model_type == 'vggt' and not VGGT_AVAILABLE:
            raise ValueError("VGGT model is not available. Please install the vggt package.")
        if model_type == 'fastvggt' and not FASTVGGT_AVAILABLE:
            raise ValueError("FastVGGT model is not available. Please check the fastvggt installation.")
        
        self.model_type = model_type
        self.model_path = model_path
        
        # FastVGGT 特有参数
        self.fastvggt_merging = fastvggt_merging
        self.fastvggt_merge_ratio = fastvggt_merge_ratio
        self.fastvggt_depth_conf_thresh = fastvggt_depth_conf_thresh
        
        # Model loader (lazy loading)
        self.model_loader = None
        self.device = None
        self.dtype = None  # For VGGT/FastVGGT mixed precision

        self.output_dir = Path(output_dir)
        
        if reconstruction_type not in ['dense_feature_points', 'each_pixel_feature_points']:
            raise ValueError(f"reconstruction_type must be 'dense_feature_points' or 'each_pixel_feature_points', current is: {reconstruction_type}")
        if reconstruction_type == 'dense_feature_points':
            self.reconstruction_type = 'dense_feature_points'
        else:
            self.reconstruction_type = 'each_pixel_feature_points'
        
        self.verbose = verbose
        self.min_images_for_scale = max(2, min_images_for_scale)
        self.overlap = overlap       
        self.pred_vis_scores_thres_value = pred_vis_scores_thres_value
        self.min_visible_frames = max(1, min_visible_frames)  # 至少为1
        self.max_sampled_points = max_sampled_points
        self.max_reproj_error = max_reproj_error
        self.max_points3D_val = max_points3D_val
        self.min_inlier_per_frame = min_inlier_per_frame
        self.filter_edge_margin = filter_edge_margin
        self.merge_voxel_size = merge_voxel_size
        self.merge_boundary_filter = merge_boundary_filter
        self.merge_statistical_filter = merge_statistical_filter
        self.merge_method = merge_method
        self.points_merge_mode = points_merge_mode
        self.memory_keep_batches = max(1, memory_keep_batches)
        
        if batch_sfm_coord_mode not in ('global_enu', 'wgs84'):
            raise ValueError(f"batch_sfm_coord_mode must be 'global_enu' or 'wgs84', got: {batch_sfm_coord_mode}")
        self.batch_sfm_coord_mode = batch_sfm_coord_mode
        
        # 全部原始影像路径和间隔参数（用于 batch SfM 补充中间影像）
        self.all_image_paths = list(all_image_paths) if all_image_paths else []
        self.image_interval = max(1, image_interval)
        self.enable_batch_sfm = enable_batch_sfm
        
        # 特征点跟踪参数
        self.max_query_pts = max_query_pts
        self.query_frame_num = query_frame_num
        
        # Visualization mode: 'aligned' (每个batch单独点云) or 'merged' (合并后整体点云)
        if visualization_mode not in ['aligned', 'merged']:
            raise ValueError(f"visualization_mode must be 'aligned' or 'merged', got: {visualization_mode}")
        self.visualization_mode = visualization_mode

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.next_image_id: int = 1
        
        # Coordinate system
        self.enu_origin: Optional[np.ndarray] = None  # [lat, lon, alt]

        # Store image paths
        self.image_paths: List[Path] = []
        
        # Store intrinsic and extrinsic parameters
        self.ori_extrinsic: List[Dict] = []
        self.ori_intrinsic: List[Dict] = []
        self.input_views: List[Dict] = []
        self.preprocessed_views: List[Dict] = []
        self.scale_info: List[Dict] = []
        self.inference_outputs: List[Dict] = []
        self.batch_tracks: List[Dict] = [] 
        self.image_tracks: List[Dict] = []  # 存储每个影像的跟踪信息
        self.inference_reconstructions: List[Dict] = []  # 存储推理结果构建的 pycolmap 重建结果
        self.sfm_reconstructions: List[Dict] = []  # 存储传统SfM重建结果
        self.merged_reconstruction: Optional[pycolmap.Reconstruction] = None # 每次合并后更新的重建结果
        self.merged_reconstruction_path: Optional[str] = None # 每次合并后更新的重建结果路径
        self.recovered_inference_outputs: List[Dict] = []
        
        # points_only 模式专用：存储合并后的点云（不维护 Reconstruction 结构）
        self.merged_points_xyz: Optional[np.ndarray] = None  # (N, 3) 合并后的点云坐标
        self.merged_points_colors: Optional[np.ndarray] = None  # (N, 3) 合并后的颜色 (uint8)
        
        # points_only 模式：保存前一个已对齐的 reconstruction（用于增量式对齐）
        self._prev_aligned_recon: Optional[pycolmap.Reconstruction] = None
        
        # 像素级置信度图存储在 inference_reconstructions[i]['pixel_3d_mapping'][global_idx]['conf'] 中
        # 格式: {global_image_idx: {'pts3d': (H,W,3), 'conf': (H,W), 'valid_mask': (H,W)}}
        # conf 已缩放到原图尺寸，与 reconstruction 中的 2D 点坐标一致

        # Georeferenced coordinate system (for export_georeferenced)
        self.geo_center: Optional[np.ndarray] = None  # [x, y, alt] center offset in target CRS
        self.output_epsg_code: Optional[int] = None  # EPSG code for output CRS
        self.rec_georef: Optional[pycolmap.Reconstruction] = None  # Reconstruction in georeferenced coordinates
        self.rec_georef_dir: Optional[Path] = None  # Output directory for georeferenced reconstruction

        self.enable_visualization = enable_visualization
        
        # Setup visualization using SfMVisualizer
        self.visualizer: Optional[SfMVisualizer] = None
        if self.enable_visualization:
            self.visualizer = SfMVisualizer(
                visualization_mode=self.visualization_mode,
                verbose=self.verbose,
            )
            self.visualizer.setup()

    def _load_model(self):
        """Load model (lazy loading).
        
        使用 load_model 模块的工厂函数创建模型加载器。
        
        Returns:
            ModelLoader 实例
        """
        if self.model_loader is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.verbose:
                print(f"Using device: {self.device}")
            
            # 使用工厂函数创建模型加载器
            self.model_loader = create_model_loader(
                model_type=self.model_type,
                model_path=self.model_path,
                device=self.device,
                verbose=self.verbose,
                # FastVGGT 特有参数
                fastvggt_merging=self.fastvggt_merging,
                fastvggt_merge_ratio=self.fastvggt_merge_ratio,
                fastvggt_depth_conf_thresh=self.fastvggt_depth_conf_thresh,
            )
            
            # 预加载模型（可能会 fallback 到 CPU）
            self.model_loader.load_model()
            self.dtype = self.model_loader.dtype
            # 同步 device（load_model 可能因 cuBLAS 不可用而回退到 CPU）
            self.device = self.model_loader.device
        
        return self.model_loader

    def _release_model(self):
        """Release model from memory to free GPU resources."""
        if self.model_loader is not None:
            self.model_loader.release_model()
            self.model_loader = None
            self.dtype = None
            if self.verbose:
                print("✓ Model released from memory")

    def _cleanup_intermediate_data(self, keep_last_n: Optional[int] = None):
        """清理中间数据以释放内存，只保留最新的 N 个批次。
        
        Args:
            keep_last_n: 保留最新的批次数量，None 时使用 self.memory_keep_batches
        """
        import gc
        
        if keep_last_n is None:
            keep_last_n = self.memory_keep_batches
        keep_last_n = max(1, keep_last_n)
        
        cleaned_items = []
        
        # 1. 清理 batch_tracks - 只保留最新的 keep_last_n 个批次
        if len(self.batch_tracks) > keep_last_n:
            num_to_remove = len(self.batch_tracks) - keep_last_n
            for i in range(num_to_remove):
                batch = self.batch_tracks[i]
                # 显式删除大型 numpy 数组
                for key in ['pred_tracks', 'pred_vis_scores', 'pred_confs', 'points_3d', 'points_rgb']:
                    if key in batch and batch[key] is not None:
                        del batch[key]
            self.batch_tracks = self.batch_tracks[-keep_last_n:]
            cleaned_items.append(f"batch_tracks: 删除 {num_to_remove} 个")
        
        # 2. 清理 image_tracks - 只保留最新的 keep_last_n * min_images_for_scale 个
        max_image_tracks = keep_last_n * self.min_images_for_scale
        if len(self.image_tracks) > max_image_tracks:
            num_to_remove = len(self.image_tracks) - max_image_tracks
            for i in range(num_to_remove):
                track_info = self.image_tracks[i]
                # 显式删除大型数组
                for key in ['tracks_2d', 'vis_scores', 'confs', 'points_3d', 'points_rgb']:
                    if key in track_info and track_info[key] is not None:
                        del track_info[key]
            self.image_tracks = self.image_tracks[-max_image_tracks:]
            cleaned_items.append(f"image_tracks: 删除 {num_to_remove} 个")
        
        # 3. 清理 inference_reconstructions 中的 pixel_3d_mapping（保留 reconstruction 对象）
        if len(self.inference_reconstructions) > keep_last_n:
            for i in range(len(self.inference_reconstructions) - keep_last_n):
                recon_data = self.inference_reconstructions[i]
                if 'pixel_3d_mapping' in recon_data and recon_data['pixel_3d_mapping']:
                    # 清空 pixel_3d_mapping 字典
                    for global_idx in list(recon_data['pixel_3d_mapping'].keys()):
                        mapping = recon_data['pixel_3d_mapping'][global_idx]
                        for key in ['pts3d', 'conf', 'valid_mask']:
                            if key in mapping:
                                del mapping[key]
                    recon_data['pixel_3d_mapping'] = {}
            cleaned_items.append(f"pixel_3d_mapping: 清理 {len(self.inference_reconstructions) - keep_last_n} 个批次")
        
        # 4. 清理 recovered_inference_outputs - 只保留最新的
        max_recovered = keep_last_n * self.min_images_for_scale
        if len(self.recovered_inference_outputs) > max_recovered:
            num_to_remove = len(self.recovered_inference_outputs) - max_recovered
            for i in range(num_to_remove):
                output = self.recovered_inference_outputs[i]
                # 清理 tensor 数据
                for key in ['pts3d', 'pts3d_cam', 'camera_poses', 'cam_trans', 'cam_quats', 'conf']:
                    if key in output and output[key] is not None:
                        del output[key]
            self.recovered_inference_outputs = self.recovered_inference_outputs[-max_recovered:]
            cleaned_items.append(f"recovered_inference_outputs: 删除 {num_to_remove} 个")
        
        # 5. 清理 inference_outputs 中的大型 tensor（保留元数据）
        if len(self.inference_outputs) > keep_last_n * self.min_images_for_scale:
            keep_from_idx = len(self.inference_outputs) - keep_last_n * self.min_images_for_scale
            cleaned_count = 0
            for i in range(keep_from_idx):
                output = self.inference_outputs[i]
                # 清理 current_output 中的 tensor
                if 'current_output' in output and output['current_output'] is not None:
                    for key in ['pts3d', 'pts3d_cam', 'conf', 'camera_poses', 'intrinsics']:
                        if key in output['current_output']:
                            del output['current_output'][key]
                    output['current_output'] = None
                # 清理 outputs 列表
                if 'outputs' in output and output['outputs'] is not None:
                    output['outputs'] = None
                cleaned_count += 1
            if cleaned_count > 0:
                cleaned_items.append(f"inference_outputs: 清理 {cleaned_count} 个的 tensor")
        
        # 6. 清理 input_views 和 preprocessed_views 中的图像 tensor
        keep_views_from = max(0, len(self.input_views) - keep_last_n * self.min_images_for_scale)
        views_cleaned = 0
        for i in range(keep_views_from):
            if i < len(self.input_views) and 'img' in self.input_views[i]:
                self.input_views[i]['img'] = None
                views_cleaned += 1
            if i < len(self.preprocessed_views) and 'img' in self.preprocessed_views[i]:
                self.preprocessed_views[i]['img'] = None
        if views_cleaned > 0:
            cleaned_items.append(f"input/preprocessed_views: 清理 {views_cleaned} 个图像 tensor")
        
        # 7. 清理 sfm_reconstructions（rescale 仅用最后一个，保留最近 keep_last_n 个即可）
        if len(self.sfm_reconstructions) > keep_last_n:
            num_remove = len(self.sfm_reconstructions) - keep_last_n
            self.sfm_reconstructions = self.sfm_reconstructions[-keep_last_n:]
            cleaned_items.append(f"sfm_reconstructions: 删除 {num_remove} 个")
        
        # 8. points_only 模式：清理旧批次的 reconstruction 对象（仅保留最后一个用于下次合并）
        if self.merge_method == 'points_only' and len(self.inference_reconstructions) > 1:
            for i in range(len(self.inference_reconstructions) - 1):
                recon_data = self.inference_reconstructions[i]
                if recon_data.get('reconstruction') is not None:
                    recon_data['reconstruction'] = None
            cleaned_items.append(f"inference_reconstructions: 清理 points_only 旧 reconstruction")
        
        # 9. 强制垃圾回收
        gc.collect()
        
        # 10. 清理 CUDA 缓存（synchronize 确保 GPU 操作完成后再释放）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        if self.verbose and cleaned_items:
            print(f"  ✓ 内存清理完成: {', '.join(cleaned_items)}")

    def add_image(self, image_path: Path) -> bool:
        """Add a new image and store its intrinsic and extrinsic parameters.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            True if successful, False otherwise
        """
        # Store image path
        self.image_paths.append(image_path)

        # Extract GPS and XMP metadata
        gps_data = extract_gps_from_image(image_path)
        if not gps_data:
            print(f"Error: No GPS data found in {image_path}")
            return False
        
        xmp_data = parse_xmp_tags(image_path)
        if not xmp_data:
            print(f"Error: No XMP data found in {image_path}")
            return False

        # Check if this is the first image
        if len(self.ori_extrinsic) == 0:
            # Set ENU origin only for the first image
            self.enu_origin = np.array(gps_data)
            if self.verbose:
                print(f"ENU origin set to: lat={gps_data[0]:.6f}, lon={gps_data[1]:.6f}, alt={gps_data[2]:.1f}")
            # Initialize with first image (ENU position is origin)
            enu_pos = np.array([0.0, 0.0, 0.0])
        else:
            # Convert GPS to ENU for subsequent images
            enu_pos = lat_lon_to_enu(
                gps_data[0], gps_data[1], gps_data[2],
                self.enu_origin[0], self.enu_origin[1], self.enu_origin[2]
            )

        # Process the image with computed ENU position
        success = self._initialize_image(image_path, gps_data, enu_pos, xmp_data)

        # ==================== 延迟推理：仅在批次边界执行 ====================
        # 优化原理：模型（VGGT/FastVGGT）是多视图联合推理，每次 _run_inference 会对
        # 滑动窗口内的所有图像（min_images_for_scale 张）一起送入模型。
        #
        # 但 batch 处理只在边界触发，中间的推理结果不会被任何 batch 使用：
        #   例：min_images_for_scale=6, overlap=2
        #   - 图1~5 添加时各调用一次推理 → 结果全部浪费（batch 未触发）
        #   - 图6 添加时推理 [1,2,3,4,5,6] → Batch 1 使用此结果 ✓
        #   - 图7~9 添加时各调用一次推理 → 结果全部浪费（batch 未触发）
        #   - 图10 添加时推理 [5,6,7,8,9,10] → Batch 2 使用此结果 ✓
        #   原来 10 次推理只有 2 次有效，8 次浪费。
        #
        # 因此：只在 batch 边界（_n_views >= _next_end）才执行推理，
        # 中间图像仅追加轻量占位符保持索引一致。
        # 注意：overlap 图像（如图5,6）在两个 batch 中都会被推理，这是必要的，
        # 因为模型对同一图像在不同窗口上下文中的预测不同，overlap 用于 batch 间对齐。
        _n_views = len(self.preprocessed_views)
        _n_recon = len(self.inference_reconstructions)
        if _n_recon == 0:
            _next_end = self.min_images_for_scale
        else:
            _last_batch = self.inference_reconstructions[-1]
            _next_end = _last_batch['end_idx'] - self.overlap + self.min_images_for_scale

        if _n_views >= _next_end:
            # 到达批次边界，执行真正的模型推理（对窗口内所有图像联合推理）
            inference_success = self._run_inference(image_path, self.preprocessed_views)
        else:
            # 未到批次边界，跳过推理，追加占位符保持 len(inference_outputs) == len(preprocessed_views)
            self.inference_outputs.append({
                'image_path': str(image_path),
                'current_output': None,
                'outputs': None,
                'scale_ratio': 1.0,
                'predicted_scale_ratio': 1.0,
            })
            inference_success = True
            if self.verbose:
                print(f"  [Skip] Deferred inference (image {_n_views}/{_next_end})")

        # ==================== 批量恢复原始位姿 ====================
        # # 检查是否达到批量恢复的条件
        # num_images = len(self.inference_outputs)
        # # num_recovered = len(self.recovered_inference_outputs)
        # num_reconstructed = len(self.inference_reconstructions)
        # # 计算还未恢复的图像数量
        # # num_unrecovered = num_images - num_recovered
        # num_unreconstructed = num_images - num_reconstructed
        # # 当未恢复的图像数量达到 min_images_for_scale 时，批量恢复这批图像
        # if num_unreconstructed >= self.min_images_for_scale:

        # 检查是否达到批量恢复的条件
        num_images = len(self.inference_outputs)
        num_reconstructed = len(self.inference_reconstructions)

        # 计算下一批次应该处理的范围
        overlap = self.overlap  # 每次重叠1张影像

        if num_reconstructed == 0:
            # 第一次：从0开始
            start_idx = 0
            end_idx = self.min_images_for_scale  # 例如：0到3，处理[0,1,2]
        else:
            # 后续批次：从上一批的倒数第overlap张开始
            last_batch = self.inference_reconstructions[-1]
            start_idx = last_batch['end_idx'] - overlap  # 例如：3-1=2
            end_idx = start_idx + self.min_images_for_scale  # 例如：2+3=5，处理[2,3,4]

        # 检查是否有足够的图像来构建这一批次
        if num_images >= end_idx:  # ← 关键：检查是否已经有足够的图像
            # ==================== 对当前批次运行SfM（可选） ====================
            sfm_extract_success = False
            if self.enable_batch_sfm:
                import shutil as _shutil
                aligned_sfm_dir = self.output_dir / "temp_aligned_batch_sfm" / f"sfm_{start_idx}_{end_idx}"
                batch_images_dir = aligned_sfm_dir / "images"
                batch_images_dir.mkdir(parents=True, exist_ok=True)

                batch_image_names = set()
                for _img_path in self.image_paths[start_idx:end_idx]:
                    _dst = batch_images_dir / _img_path.name
                    if not _dst.exists():
                        _shutil.copy2(str(_img_path), str(_dst))
                    batch_image_names.add(_img_path.name)

                if self.image_interval > 1 and len(self.all_image_paths) > 0:
                    orig_start = start_idx * self.image_interval
                    orig_end = min((end_idx - 1) * self.image_interval + 1, len(self.all_image_paths))
                    intermediate_count = 0
                    for _img_path in self.all_image_paths[orig_start:orig_end]:
                        _dst = batch_images_dir / _img_path.name
                        if not _dst.exists():
                            _shutil.copy2(str(_img_path), str(_dst))
                            intermediate_count += 1
                    if self.verbose and intermediate_count > 0:
                        print(f"  Added {intermediate_count} intermediate images for SfM "
                              f"(original range [{orig_start}:{orig_end}], "
                              f"total SfM images: {len(batch_image_names) + intermediate_count})")

                sfm_result = self._run_batch_sfm(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    images_dir=batch_images_dir,
                    output_dir=aligned_sfm_dir,
                    batch_image_names=batch_image_names,
                )
                if sfm_result is not None:
                    if len(self.sfm_reconstructions) > 0:
                        prev_recon = self.sfm_reconstructions[-1]['reconstruction']
                        curr_recon = sfm_result['reconstruction']
                        if self.verbose:
                            print(f"  Aligning batch SfM [{start_idx}:{end_idx}] "
                                  f"to previous batch via overlapping images...")
                        align_reconstruction_by_overlap(
                        curr_recon=curr_recon,
                        prev_recon=prev_recon,
                        pixel_threshold=5.0,
                        verbose=self.verbose,
                    )[0]

                    sfm_result['reconstruction'].write_text(str(aligned_sfm_dir))
                    sfm_result['reconstruction'].export_PLY(str(aligned_sfm_dir / "points3D.ply"))
                    if self.verbose:
                        print(f"  ✓ Aligned batch SfM saved to: {aligned_sfm_dir}")

                    self.sfm_reconstructions.append(sfm_result)
                    sfm_extract_success = True
                else:
                    if self.verbose:
                        print("  Warning: Batch SfM failed, alignment will use image_alignment mode")
            else:
                if self.verbose:
                    print("  [Skip] Batch SfM disabled (enable_batch_sfm=False), "
                          "alignment will use image_alignment mode")

            if self.reconstruction_type == 'dense_feature_points':
                # ==================== 预测tracks（在推理坐标系） ====================
                track_predict_success = self._predict_tracks_for_batch(
                    start_idx=start_idx,
                    end_idx=end_idx
                )

                # ==================== 构建pycolmap重建 ====================
                pycolmap_success = self._build_pycolmap_reconstruction(
                    start_idx=start_idx,
                    end_idx=end_idx
                )

                # ==================== 合并reconstruction中间结果 ====================
                # 只有当 pycolmap 重建成功后才进行合并
                if pycolmap_success:
                    merge_reconstruction_success = self._merge_reconstruction_intermediate_results()
                else:
                    merge_reconstruction_success = False
                    if self.verbose:
                        print(f"  [Skip] Merge skipped: pycolmap reconstruction failed")

                # # ==================== 批量恢复位姿和3D点到真实坐标系 ====================
                # batch_recover_success = self._batch_recover_original_poses(
                #     image_path=image_path,
                #     start_idx=num_recovered,
                #     end_idx=num_images,
                #     transform_tracks=True  # 同时变换tracks
                # )
            elif self.reconstruction_type == 'each_pixel_feature_points':
                num_frames = self.min_images_for_scale
                height = self.inference_outputs[-1]['current_output']['pts3d'].shape[1]
                width = self.inference_outputs[-1]['current_output']['pts3d'].shape[2]

                conf_thres_value = 0.3
                shared_camera = False
                camera_type = "PINHOLE"
                img_height, img_width = height, width
                image_size = np.array([height, width])

                latest_inference = self.inference_outputs[-1]
                latest_outputs = latest_inference['outputs']
                num_images = len(self.inference_outputs)
                num_outputs = len(latest_outputs)
                latest_start_idx = num_images - num_outputs
                latest_end_idx = num_images
                
                use_latest_outputs = (latest_start_idx == start_idx and latest_end_idx == end_idx)
                
                # 统一数据源，消除代码重复
                if use_latest_outputs:
                    outputs_list = latest_outputs
                    indices = list(range(start_idx, start_idx + len(latest_outputs)))
                else:
                    if self.verbose:
                        print(f"  Warning: Using outputs from different inference batches, points may be in different coordinate systems")
                    outputs_list = [self.inference_outputs[idx]['current_output'] for idx in range(start_idx, end_idx)]
                    indices = list(range(start_idx, end_idx))
                
                n = len(outputs_list)
                
                # ==================== GPU全流程优化 ====================
                # 核心思想: 尽可能在GPU上完成计算，减少CPU-GPU传输
                
                # 步骤1: 在GPU上stack所有tensor（零拷贝）
                pts3d_gpu = torch.stack([output['pts3d'][0] for output in outputs_list])  # (n, H, W, 3)
                conf_gpu = torch.stack([output['conf'][0] for output in outputs_list])    # (n, H, W)
                cam_gpu = torch.stack([output['camera_poses'][0] for output in outputs_list])  # (n, 4, 4)
                K_gpu = torch.stack([output['intrinsics'][0] for output in outputs_list])      # (n, 3, 3)
                
                device = pts3d_gpu.device
                use_gpu = pts3d_gpu.is_cuda
                
                # 步骤2: 在GPU上进行置信度过滤（避免传输全部数据到CPU）
                conf_mask_gpu = conf_gpu >= conf_thres_value  # (n, H, W)
                
                # 步骤3: GPU上随机采样（如果点数超过限制）
                true_count = conf_mask_gpu.sum().item()
                if true_count > self.max_sampled_points:
                    # 在GPU上高效随机采样
                    flat_mask = conf_mask_gpu.view(-1)
                    true_indices = torch.nonzero(flat_mask, as_tuple=True)[0]
                    # 随机选择 self.max_sampled_points 个索引
                    perm = torch.randperm(true_indices.size(0), device=device)[:self.max_sampled_points]
                    sampled_indices = true_indices[perm]
                    # 重建mask
                    combined_mask_gpu = torch.zeros_like(flat_mask)
                    combined_mask_gpu[sampled_indices] = True
                    combined_mask_gpu = combined_mask_gpu.view(conf_mask_gpu.shape)
                else:
                    combined_mask_gpu = conf_mask_gpu
                
                # 步骤4: 提取过滤后的3D点（仍在GPU上）
                points_3d_filtered_gpu = pts3d_gpu[combined_mask_gpu]  # (N_filtered, 3)
                
                # 步骤5: 在GPU上构建extrinsic（利用旋转矩阵正交性）
                R_gpu = cam_gpu[:, :3, :3]  # (n, 3, 3)
                t_gpu = cam_gpu[:, :3, 3:4]  # (n, 3, 1)
                R_inv_gpu = R_gpu.transpose(-1, -2)  # R^T
                t_inv_gpu = -torch.bmm(R_inv_gpu, t_gpu)  # -R^T @ t
                extrinsic_gpu = torch.cat([R_inv_gpu, t_inv_gpu], dim=2)  # (n, 3, 4)
                
                # 步骤6: 在GPU上执行投影（核心加速点）
                if use_gpu and points_3d_filtered_gpu.size(0) > 0:
                    projected_points2d_gpu, points_cam_gpu = project_3D_points(
                        points_3d_filtered_gpu,  # (N, 3)
                        extrinsic_gpu,           # (n, 3, 4)
                        K_gpu,                   # (n, 3, 3)
                    )
                    
                    # # 步骤7: 在GPU上计算可见性mask
                    # depths_gpu = points_cam_gpu[:, 2, :]  # (n, N)
                    # proj_x_gpu = projected_points2d_gpu[:, :, 0]
                    # proj_y_gpu = projected_points2d_gpu[:, :, 1]
                    
                    # visible_mask_gpu = (
                    #     (depths_gpu > 0) &
                    #     (proj_x_gpu >= 0) & (proj_x_gpu < img_width) &
                    #     (proj_y_gpu >= 0) & (proj_y_gpu < img_height)
                    # )  # (n, N)
                    
                    # 步骤7: 在GPU上计算可见性mask（优化版）
                    visible_mask_gpu = torch.logical_and(
                        points_cam_gpu[:, 2, :] > 0,
                        torch.logical_and(
                            torch.logical_and(projected_points2d_gpu[:, :, 0] >= 0, 
                                            projected_points2d_gpu[:, :, 0] < img_width),
                            torch.logical_and(projected_points2d_gpu[:, :, 1] >= 0, 
                                            projected_points2d_gpu[:, :, 1] < img_height)
                        )
                    )

                    # # 步骤8: 在GPU上筛选有效点
                    # points_visible_count_gpu = visible_mask_gpu.sum(dim=0)  # (N,)
                    # valid_points_mask_gpu = points_visible_count_gpu >= self.min_visible_frames  # 至少在 min_visible_frames 帧中可见
                    # valid_point_indices_gpu = torch.nonzero(valid_points_mask_gpu, as_tuple=True)[0]
                    # num_tracks = valid_point_indices_gpu.size(0)

                    # 步骤8: 在GPU上筛选有效点
                    valid_points_mask_gpu = visible_mask_gpu.sum(dim=0) >= self.min_visible_frames
                    valid_point_indices_gpu = valid_points_mask_gpu.nonzero(as_tuple=True)[0]
                    num_tracks = valid_point_indices_gpu.size(0)
                    
                    # 步骤9: 只传输需要的数据到CPU（大幅减少传输量）
                    if num_tracks > 0:
                        # 提取有效点的数据并传输到CPU（合并操作）
                        tracks_gpu = projected_points2d_gpu[:, valid_point_indices_gpu, :].float()
                        masks_gpu = visible_mask_gpu[:, valid_point_indices_gpu]
                        
                        # 设置不可见位置为NaN（在GPU上完成）
                        tracks_gpu[~masks_gpu] = float('nan')
                        
                        # 一次性传输到CPU（只传输必要的数据）
                        tracks = tracks_gpu.cpu().numpy()
                        masks = masks_gpu.cpu().numpy()
                        points3d_for_tracks = points_3d_filtered_gpu[valid_point_indices_gpu].double().cpu().numpy()
                        valid_point_indices = valid_point_indices_gpu.cpu().numpy()
                        
                        # extrinsic和intrinsic传输（用于COLMAP）
                        extrinsic = extrinsic_gpu.cpu().numpy().astype(np.float32)
                        intrinsic = K_gpu.cpu().numpy().astype(np.float32)
                        
                        # 统一变量名，便于后续使用
                        combined_mask_np = combined_mask_gpu.cpu().numpy()
                else:
                    # CPU fallback（数据不在GPU或无有效点）
                    pts3d_batch = pts3d_gpu.cpu().numpy()
                    conf_batch = conf_gpu.cpu().numpy()
                    cam_batch = cam_gpu.cpu().numpy()
                    K_batch = K_gpu.cpu().numpy()
                    
                    points_3d = pts3d_batch.astype(np.float32)
                    depth_conf = conf_batch.astype(np.float32)
                    intrinsic = K_batch.astype(np.float32)
                    
                    R_batch = cam_batch[:, :3, :3]
                    t_batch = cam_batch[:, :3, 3:4]
                    R_inv = np.transpose(R_batch, (0, 2, 1))
                    t_inv = -np.matmul(R_inv, t_batch)
                    extrinsic = np.concatenate([R_inv, t_inv], axis=2)
                    
                    conf_mask = depth_conf >= conf_thres_value
                    combined_mask = self.randomly_limit_trues(conf_mask, self.max_sampled_points)
                    points_3d_filtered = points_3d[combined_mask]
                    all_points_3d = points_3d_filtered
                    
                    if len(all_points_3d) > 0:
                        projected_points2d, points_cam = project_3D_points_np(
                            all_points_3d, extrinsic, intrinsic
                        )
                        depths = points_cam[:, 2, :]
                        proj_x = projected_points2d[:, :, 0]
                        proj_y = projected_points2d[:, :, 1]
                        visible_mask = (
                            (depths > 0) & (proj_x >= 0) & (proj_x < img_width) &
                            (proj_y >= 0) & (proj_y < img_height)
                        )
                        points_visible_count = visible_mask.sum(axis=0)
                        valid_point_indices = np.flatnonzero(points_visible_count > 0)
                        num_tracks = len(valid_point_indices)
                        
                        if num_tracks > 0:
                            tracks = projected_points2d[:, valid_point_indices, :].astype(np.float32, copy=True)
                            masks = visible_mask[:, valid_point_indices].copy()
                            tracks[~masks] = np.nan
                            points3d_for_tracks = all_points_3d[valid_point_indices].astype(np.float64)
                        
                        # 统一变量名，便于后续使用
                        combined_mask_np = combined_mask
                    else:
                        num_tracks = 0
                
                # ==================== 图像RGB处理 ====================
                # 处理图像和 original_coords
                if self.model_type in ['vggt', 'fastvggt']:
                    # VGGT 和 FastVGGT 都使用 vggt_image 输出
                    vggt_images = []
                    original_coords = np.empty((n, 6), dtype=np.float32)
                    
                    for i, (output, idx) in enumerate(zip(outputs_list, indices)):
                        vggt_img = output.get('vggt_image', None)
                        if vggt_img is not None:
                            vggt_images.append(vggt_img)
                        elif idx < len(self.preprocessed_views):
                            img = self.preprocessed_views[idx]['img']
                            if img.dim() == 4 and img.shape[0] == 1:
                                img = img.squeeze(0)
                            vggt_images.append(img)
                        
                        scale_info = self.scale_info[idx]
                        ori_w, ori_h = scale_info['original_size']
                        original_coords[i] = [0, 0, ori_w, ori_h, ori_w, ori_h]
                    
                    if len(vggt_images) > 0:
                        vggt_images_tensor = torch.stack(vggt_images)
                        if vggt_images_tensor.shape[2:] != (height, width):
                            vggt_images_tensor = F.interpolate(
                                vggt_images_tensor, size=(height, width),
                                mode="bilinear", align_corners=False,
                            )
                        points_rgb_np = np.ascontiguousarray(
                            vggt_images_tensor.cpu().numpy().transpose(0, 2, 3, 1)
                        )
                        if points_rgb_np.max() <= 1.0:
                            points_rgb = (points_rgb_np * 255).astype(np.uint8)
                        else:
                            points_rgb = points_rgb_np.astype(np.uint8)
                    else:
                        raise ValueError("No VGGT/FastVGGT images available for color extraction")
                else:
                    batch_images = []
                    original_coords = np.empty((n, 6), dtype=np.float32)
                    
                    for i, idx in enumerate(indices):
                        if idx < len(self.preprocessed_views):
                            img = self.preprocessed_views[idx]['img']
                            if img.dim() == 4 and img.shape[0] == 1:
                                img = img.squeeze(0)
                            batch_images.append(img)
                        
                        scale_info = self.scale_info[idx]
                        ori_w, ori_h = scale_info['original_size']
                        original_coords[i] = [0, 0, ori_w, ori_h, ori_w, ori_h]
                    
                    images = torch.stack(batch_images)
                    points_rgb_images = F.interpolate(
                        images, size=(height, width),
                        mode="bilinear", align_corners=False,
                    )
                    model_loader = self._load_model()
                    # MapAnything 模型需要访问 encoder.data_norm_type
                    points_rgb_float = rgb(points_rgb_images, model_loader.model.encoder.data_norm_type)
                    points_rgb = (points_rgb_float * 255).astype(np.uint8)
                
                # ==================== 提取RGB颜色并构建COLMAP ====================
                if num_tracks == 0:
                    print("  Warning: No valid tracks found, skipping COLMAP conversion")
                    reconstruction = None
                    valid_track_mask = None
                else:
                    # 获取过滤点对应的RGB（使用统一的 combined_mask_np）
                    points_rgb_filtered = points_rgb[combined_mask_np]
                    points_rgb_for_tracks = points_rgb_filtered[valid_point_indices]

                    # 准备 image_paths 列表（文件名）
                    image_paths_list = [self.image_paths[idx].name for idx in range(start_idx, end_idx)]

                    print("Converting to COLMAP format (with rename and rescale)")
                    # 使用合并优化版本，直接构建带正确图像名和缩放参数的 reconstruction
                    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap_with_rename(
                        points3d=points3d_for_tracks,  # (P, 3)
                        extrinsics=extrinsic,          # (N, 3, 4)
                        intrinsics=intrinsic,          # (N, 3, 3)
                        tracks=tracks,                 # (N, P, 2)
                        image_size=image_size,         # (2,)
                        masks=masks,                   # (N, P)
                        max_reproj_error=self.max_reproj_error,  # 重投影误差阈值
                        max_points3D_val=self.max_points3D_val,  # 使用实例参数而不是硬编码
                        shared_camera=shared_camera,
                        camera_type=camera_type,
                        min_inlier_per_frame=self.min_inlier_per_frame,
                        points_rgb=points_rgb_for_tracks,  # (P, 3)
                        # 新增参数：直接在构建时完成重命名和缩放
                        image_paths=image_paths_list,
                        original_coords=original_coords,
                        shift_point2d_to_original_res=True,
                    )

                if reconstruction is None:
                    print("  Warning: Failed to build pycolmap reconstruction")

                # 对齐策略（互斥，只执行一种）：
                # 1. confidence 模式或第一个 batch：使用 rescale_reconstruction_to_original_size 对齐到原始 SfM/GPS
                # 2. 后续 batch（非 confidence）：直接基于重叠影像的相机位置对齐到前一个重建（跳过 rescale，避免重复对齐）
                overlap_matched_pairs = []
                if self.merge_method == 'confidence' or len(self.inference_reconstructions) < 1:
                    # confidence 模式：rescale 后由 merge_by_confidence 通过 RANSAC+3D点匹配 进一步精确对齐
                    # 第一个 batch：没有前一个重建可参考，只能对齐到原始 SfM/GPS
                    print(" align to sfm original size")
                    if len(self.sfm_reconstructions) > 0:
                        reconstruction = rescale_reconstruction_to_original_size(
                            reconstruction=reconstruction,
                            ori_extrinsics=self.ori_extrinsic,
                            sfm_reconstructions=self.sfm_reconstructions,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            alignment_mode='pcl_alignment',
                            verbose=self.verbose,
                        )
                    else:
                        reconstruction = rescale_reconstruction_to_original_size(
                            reconstruction=reconstruction,
                            ori_extrinsics=self.ori_extrinsic,
                            sfm_reconstructions=self.sfm_reconstructions,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            alignment_mode='image_alignment',
                            image_alignment_max_error=10.0,
                            image_alignment_min_inlier_ratio=0.3,
                        )
                    aligned_recon = reconstruction

                    temp_path = self.output_dir / "temp_aligned_to_original_sfm" / f"{start_idx}_{end_idx}"
                    temp_path.mkdir(parents=True, exist_ok=True)
                    aligned_recon.write_text(str(temp_path))
                    aligned_recon.export_PLY(str(temp_path / "points3D.ply"))
                    if self.verbose:
                        if self.merge_method == 'confidence':
                            print(f"  ✓ 重建已对齐到原始SfM并保存到: {temp_path}，后续由 merge_by_confidence 进一步精确对齐")
                        else:
                            print(f"  ✓ 第一个batch，重建已对齐到原始SfM并保存到: {temp_path}")
                else:
                    # 后续 batch（非 confidence）：跳过 rescale，直接基于重叠影像对齐到前一个重建
                    # 使用两阶段对齐：Stage 1 相机中心粗对齐 + Stage 2 3D-3D 像素匹配精修
                    prev_recon_data = self.inference_reconstructions[-1]
                    prev_recon = prev_recon_data['reconstruction']
                    aligned_recon = reconstruction

                    if self.verbose:
                        print(f"  Aligning batch [{start_idx}:{end_idx}] to previous "
                              f"inference reconstruction via overlapping images...")
                    _, overlap_matched_pairs = align_reconstruction_by_overlap(
                        curr_recon=aligned_recon,
                        prev_recon=prev_recon,
                        pixel_threshold=5.0,
                        max_correspondences=10,
                        verbose=self.verbose,
                    )

                    # temp_path = self.output_dir / "temp_aligned_to_prev_recon_overlay_image" / f"{start_idx}_{end_idx}"
                    # temp_path.mkdir(parents=True, exist_ok=True)
                    # aligned_recon.write_text(str(temp_path))
                    # aligned_recon.export_PLY(str(temp_path / "points3D.ply"))
                    # if self.verbose:
                    #     print(f"  ✓ 对齐后的重建已保存到: {temp_path}")

                # ==================== 提取逐像素3D点对应关系（缩放到原图尺寸）====================
                # points_only+fast 模式不使用逐像素映射，跳过以节省 GPU→CPU 传输和 resize 开销
                if self.merge_method == 'points_only' and self.points_merge_mode == 'fast':
                    batch_pixel_3d_mapping = {}
                else:
                    # 从 outputs_list 中提取逐像素的 pts3d，与 reconstruction 中的稀疏点不同，
                    # 这里是密集的逐像素对应，可用于密集点云融合、语义投影等
                    # 数据结构：{global_idx: {'pts3d': (H,W,3), 'conf': (H,W), 'valid_mask': (H,W)}}
                    batch_pixel_3d_mapping = self._extract_pixel_to_3d_mapping_for_batch(
                        outputs_list=outputs_list,
                        global_indices=indices,
                        conf_threshold=1.0,
                        verbose=True
                    )
                
                # 将 aligned_recon 添加到列表（包含该批次的 pixel_3d_mapping）
                # 对于 merge_method='confidence'，aligned_recon 未经预对齐，由 merge_by_confidence 一步到位完成
                image_paths = [str(self.image_paths[idx]) for idx in range(start_idx, end_idx)]
                self.inference_reconstructions.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'image_paths': image_paths,
                    'reconstruction': aligned_recon,
                    'valid_track_mask': valid_track_mask,
                    'pixel_3d_mapping': batch_pixel_3d_mapping,
                    'overlap_matched_pairs': overlap_matched_pairs,
                })

                # ==================== 可视化数据准备 ====================
                scale_ratio = self.inference_outputs[-1].get('scale_ratio', 1.0)
                if self.verbose:
                    print(f"  Using scale_ratio: {scale_ratio:.6f} for visualization")

                # 1. 从 aligned_recon 提取稀疏点云并添加到可视化器
                # points_only 模式：merged 可视化在 _merge_by_points_only 中更新，跳过 per-batch 提取
                if self.merge_method != 'points_only':
                    points3D_dict = aligned_recon.points3D
                    num_points = len(points3D_dict)
                    if num_points > 0:
                        unified_points = np.empty((num_points, 3), dtype=np.float32)
                        unified_colors = np.empty((num_points, 3), dtype=np.uint8)
                        for i, pt3d in enumerate(points3D_dict.values()):
                            unified_points[i] = pt3d.xyz
                            unified_colors[i] = pt3d.color

                        if self.visualizer is not None:
                            self.visualizer.add_batch_point_cloud(unified_points, unified_colors)

                        if self.verbose:
                            batch_count = len(self.visualizer.unified_point_clouds) if self.visualizer else 0
                            print(f"  ✓ Added unified point cloud for batch {batch_count}: {num_points} points")

                # 2. 为每个图像存储相机位姿信息（用于 frustum 可视化）
                #    先收集所有位姿到 numpy 数组，最后一次性批量传输到 GPU
                sample_pts3d = outputs_list[0]['pts3d']
                device = sample_pts3d.device if isinstance(sample_pts3d, torch.Tensor) else 'cpu'
                aligned_images = aligned_recon.images
                aligned_cameras = aligned_recon.cameras
                all_image_paths = self.image_paths

                n_images = len(indices)
                real_ws = original_coords[:n_images, 4].astype(np.int32)
                real_hs = original_coords[:n_images, 5].astype(np.int32)
                all_T_world_cam = np.empty((n_images, 4, 4), dtype=np.float32)
                all_K = [None] * n_images
                eye4 = np.eye(4, dtype=np.float32)

                for local_idx, global_idx in enumerate(indices):
                    colmap_image_id = local_idx + 1  # COLMAP ID 从1开始
                    if colmap_image_id in aligned_images:
                        pyimage = aligned_images[colmap_image_id]
                        # 刚体变换解析求逆：cam2world = [R.T | -R.T @ t]
                        R_w2c = np.array(pyimage.cam_from_world.rotation.matrix(), dtype=np.float32)
                        t_w2c = np.array(pyimage.cam_from_world.translation, dtype=np.float32)
                        R_c2w = R_w2c.T
                        T = eye4.copy()
                        T[:3, :3] = R_c2w
                        T[:3, 3] = -(R_c2w @ t_w2c)
                        all_T_world_cam[local_idx] = T
                        all_K[local_idx] = aligned_cameras[pyimage.camera_id].calibration_matrix().astype(np.float32)
                    else:
                        output = outputs_list[local_idx]
                        all_T_world_cam[local_idx] = output['camera_poses'][0].cpu().numpy()
                        all_K[local_idx] = output['intrinsics'][0].cpu().numpy()

                # 批量 CPU→GPU 传输（1 次代替 N 次 .to(device)）
                all_poses_torch = torch.from_numpy(all_T_world_cam).unsqueeze(1).to(device)  # (N, 1, 4, 4)

                for local_idx, global_idx in enumerate(indices):
                    self.recovered_inference_outputs.append({
                        'image_path': str(all_image_paths[global_idx]),
                        'image_width': int(real_ws[local_idx]),
                        'image_height': int(real_hs[local_idx]),
                        'camera_K': all_K[local_idx],
                        'camera_poses': all_poses_torch[local_idx],  # (1, 4, 4)
                        'scale_ratio': scale_ratio,
                    })

                if self.verbose:
                    print(f"  ✓ Added {n_images} images to recovered_inference_outputs for visualization")

                # 合并前释放临时 GPU 张量
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                # ==================== 合并reconstruction中间结果 ====================
                merge_reconstruction_success = self._merge_reconstruction_intermediate_results()

            # Viser visualization
            if self.enable_visualization and self.visualizer is not None:
                self.visualizer.update(
                    recovered_inference_outputs=self.recovered_inference_outputs,
                    merged_reconstruction=self.merged_reconstruction,
                    input_views=self.input_views,
                    image_paths=self.image_paths,
                )
        
        if not success:
            print(f"Failed to process image: {image_path}")
            return False

        return True

    def _initialize_image(
        self, 
        image_path: Path, 
        gps_data: Tuple[float, float, float],
        enu_pos: np.ndarray,
        xmp_data: Dict
    ) -> bool:
        """Initialize and store intrinsic and extrinsic parameters for an image.
        
        This is a general function that can process any image.
        
        Args:
            image_path: Path to the image
            gps_data: GPS coordinates (lat, lon, alt)
            enu_pos: Position in ENU coordinates
            xmp_data: XMP metadata including gimbal pose and camera parameters
            
        Returns:
            True if successful, False otherwise
        """
        tracker = get_timing_tracker()
        if tracker:
            tracker.start("initialize_image")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Initializing image {len(self.ori_extrinsic) + 1}: {image_path.name}")
            print(f"{'='*60}")
        
        #  initialize the image
        # ================================================
        # Get image ID
        image_id = self.next_image_id
        self.next_image_id += 1
        image_name = image_path.name
        
        # Get image dimensions from xmp_data
        width_height = xmp_data.get("width_height", [0, 0])
        width, height = width_height[0], width_height[1]
        # Extract intrinsic parameters from XMP data
        dewarp_data = xmp_data.get("dewarp_data", [])
        # Extract camera parameters (fx, fy, cx, cy, k1, k2, p1, p2)
        params = dewarp_data[:8]
        # Construct intrinsics matrix
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Extract extrinsic parameters
        roll = xmp_data.get("roll")
        pitch = xmp_data.get("pitch")
        yaw = xmp_data.get("yaw")
        # Compute camera rotation matrix
        R_camera = cam_from_enu_transform(roll=roll, pitch=pitch, yaw=yaw)
        # Compute translation vector
        tvec = -R_camera @ enu_pos
        
        # Store intrinsic parameters
        intrinsic_info = {
            'image_id': image_id,
            'image_name': image_name,
            'image_path': str(image_path),
            'model': "OPENCV",
            'width': int(width),
            'height': int(height),
            'params': params,
            'K': K.tolist(),
        }
        self.ori_intrinsic.append(intrinsic_info)
        
        # Store extrinsic parameters
        extrinsic_info = {
            'image_id': image_id,
            'image_name': image_name,
            'image_path': str(image_path),
            'R_camera': R_camera.tolist(),  # Convert to list for JSON serialization
            'tvec': tvec.tolist(),  # Convert to list for JSON serialization
            'gps': gps_data,
            'enu': enu_pos,
        }
        self.ori_extrinsic.append(extrinsic_info)

        # Preprocess the image
        # ================================================
        # Create input view
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image).astype(np.uint8)
        image_tensor = torch.from_numpy(image_array)  # [H, W, 3]

        # Construct pose matrix (cam2world)
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = R_camera.T  # world to camera is R_camera, so cam to world is R_camera.T
        pose_matrix[:3, 3] = -R_camera.T @ tvec  # camera position in world coordinates
        pose_tensor = torch.from_numpy(pose_matrix)

        # Create view dict (similar to load_colmap_data)
        input_view = {
            # 'image_id': image_id,
            # 'image_name': image_name,
            # 'image_path': str(image_path),
            'img': image_tensor,  # [H, W, 3], uint8  # (H, W, 3) - [0, 255]
            'intrinsics': torch.from_numpy(K),  # [3, 3]
            'camera_poses': pose_tensor,  # [4, 4] in OpenCV cam2world convention
            'is_metric_scale': torch.tensor([False]),  # COLMAP data is non-metric
        }

        self.input_views.append(input_view)

        # Preprocess this single view
        # 将图像缩放到固定分辨率（resolution_set=518），以适配神经网络的输入要求。
        preprocessed_view = preprocess_inputs(
            [input_view],
            resize_mode="fixed_mapping",
            resolution_set=518,
            verbose=False
        )[0]  # Get the single preprocessed view

        # Store preprocessed view
        self.preprocessed_views.append(preprocessed_view)

        # Calculate scale info
        orig_h, orig_w = image_array.shape[0], image_array.shape[1]
        proc_h, proc_w = preprocessed_view['img'].shape[2], preprocessed_view['img'].shape[3]
        
        scale_info = {
            'image_id': image_id,
            'image_name': image_name,
            'original_size': (orig_w, orig_h),
            'output_size': (proc_w, proc_h),
            'scale_x': proc_w / orig_w,
            'scale_y': proc_h / orig_h,
        }
        self.scale_info.append(scale_info)

        if self.verbose:
            print(f"  Original size: {orig_w}x{orig_h}")
            print(f"  Preprocessed size: {proc_w}x{proc_h}")
            print(f"  Scale: x={scale_info['scale_x']:.4f}, y={scale_info['scale_y']:.4f}")
            print(f"  Intrinsics stored: {width}x{height}, fx={params[0]:.2f}, fy={params[1]:.2f}")
            print(f"  Extrinsics stored:")
            print(f"    ENU position: [{enu_pos[0]:.2f}, {enu_pos[1]:.2f}, {enu_pos[2]:.2f}]")
            print(f"    Roll/Pitch/Yaw: [{roll:.2f}, {pitch:.2f}, {yaw:.2f}]")
        
        if self.verbose:
            print(f"✓ Image initialized: {image_name} (ID: {image_id})")
        
        tracker = get_timing_tracker()
        if tracker:
            tracker.end("initialize_image")
        return True

    def _run_inference(self, image_path: Path, preprocessed_view: Dict) -> bool:
        """Run inference on the image.
        
        Args:
            image_path: Path to the image
            preprocessed_view: Single preprocessed view (will be ignored, we'll use stored views)
        
        Returns:
            True if successful, False otherwise
        """
        # 获取全局计时器和GPU显存跟踪器
        tracker = get_timing_tracker()
        gpu_tracker = get_gpu_memory_tracker()
        
        # 判断是第几张图像
        num_images = len(self.preprocessed_views)
        
        if tracker:
            tracker.start("run_inference")
        
        # ==================== 模型加载 ====================
        if tracker:
            tracker.start("model_loading")
        
        # 记录模型加载前的GPU显存
        if gpu_tracker:
            gpu_tracker.record("model_loading_before", "模型加载前", num_images)
        
        model_loader = self._load_model()
        
        # 记录模型加载后的GPU显存
        if gpu_tracker:
            gpu_tracker.record("model_loading_after", "模型加载后", num_images)
        
        if tracker:
            tracker.end("model_loading")

        # ==================== 模型推理 ====================
        # 记录推理前的GPU显存
        if gpu_tracker:
            gpu_tracker.record("model_inference_before", f"模型推理前", num_images)

        # 使用模型加载器运行推理
        outputs = model_loader.run_inference(
            preprocessed_views=self.preprocessed_views,
            image_paths=self.image_paths,
            num_images=num_images,
            min_images_for_scale=self.min_images_for_scale,
        )
        
        # 记录推理后的GPU显存
        if gpu_tracker:
            gpu_tracker.record("model_inference_after", f"模型推理后", num_images)

        # ==================== 计算 scale_ratio =====================
        scale_ratio = 1.0
        if num_images >= 2:  # 从第2张开始计算scale
            # 确定参与scale计算的图像数量
            num_infer = min(num_images, self.min_images_for_scale)
            
            # 提取所有参与推理的图像的原始相机位置
            orig_positions = []
            for i in range(-num_infer, 0):  # -num_infer, -num_infer+1, ..., -1
                ext = self.ori_extrinsic[i]
                R_cam = np.array(ext['R_camera'])
                t_cam = np.array(ext['tvec'])
                cam_pos = -R_cam.T @ t_cam
                orig_positions.append(cam_pos)
            
            # 转换为 torch tensor (N, 3)
            orig_positions = torch.from_numpy(np.stack(orig_positions)).float()
            
            # 提取推理的相机位置（在同一坐标系中）
            infer_positions = torch.stack([
                outputs[i]['camera_poses'][0, :3, 3].cpu()
                for i in range(len(outputs))
            ])  # (N, 3)
            
            # 使用 torch.cdist 计算距离矩阵
            orig_dists = torch.cdist(orig_positions, orig_positions)  # (N, N)
            infer_dists = torch.cdist(infer_positions, infer_positions)  # (N, N)
            
            # 获取有效的非零距离对（参考 demo_inference_on_colmap_outputs.py）
            valid_mask = orig_dists > 1e-6
            
            if valid_mask.sum() > 0:
                # 计算缩放比例 = 真实距离 / 推理距离（使用中位数）
                scale_ratio = (orig_dists[valid_mask] / (infer_dists[valid_mask] + 1e-8)).median().item()
                
                if self.verbose:
                    print(f"  ================================ Computed Scale Ratio (COLMAP/{self.model_type}): {scale_ratio:.6f}")
                    print(f"  Based on {valid_mask.sum().item()} camera pair distances from {num_infer} images")
            else:
                if self.verbose:
                    print("  Warning: Could not compute scale ratio (insufficient camera movement)")
        else:
            if self.verbose:
                print(f"  No scale calculation (first image)")

        if self.verbose:
            print("✓ Inference completed!")

        # 从outputs中提取预测的尺度比例
        predicted_scale_ratio = outputs[-1].get('metric_scaling_factor', torch.tensor(1.0))
        if isinstance(predicted_scale_ratio, torch.Tensor):
            predicted_scale_ratio = predicted_scale_ratio.item()
        
        # 只存储当前图像（最后一张）的输出
        current_output = outputs[-1] if num_images >= 2 else outputs[0]

        # Store inference outputs
        inference_outputs = {
            'image_path': str(image_path),
            'current_output': current_output,
            'outputs': outputs,
            'scale_ratio': scale_ratio,
            'predicted_scale_ratio': predicted_scale_ratio,
        }
        self.inference_outputs.append(inference_outputs)

        # ==================== 清理CUDA缓存 ====================
        # 记录清理前的GPU显存
        if gpu_tracker:
            gpu_tracker.record("cuda_cache_before", "CUDA缓存清理前", num_images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # 记录清理后的GPU显存
        if gpu_tracker:
            gpu_tracker.record("cuda_cache_after", "CUDA缓存清理后", num_images)

        # 结束计时
        tracker = get_timing_tracker()
        if tracker:
            tracker.end("run_inference")

        return True

    def _batch_recover_original_poses(self, image_path: Path, start_idx: int, end_idx: int, transform_tracks: bool = True) -> bool:
        """Batch recover original poses for a range of images.
        
        Args:
            image_path: Path to the image
            start_idx: Starting index in self.inference_outputs
            end_idx: Ending index (exclusive) in self.inference_outputs
            transform_tracks: Whether to also transform the tracks' 3D points
        
        Returns:
            True if successful, False otherwise
        """
        if self.verbose:
            print(f"  Recovering poses for {end_idx - start_idx} images (indices {start_idx} to {end_idx-1})")
        
        # 计算这批图像的中位数尺度（用于恢复）
        all_scale_ratios = [data['scale_ratio'] for data in self.inference_outputs[:end_idx]]
        all_predicted_scale_ratios = [data['predicted_scale_ratio'] for data in self.inference_outputs[:end_idx]]
        
        # 最新尺度
        scale_ratio = all_scale_ratios[-1]
        predicted_scale_ratio = all_predicted_scale_ratios[-1]
        
        # 对这批图像逐一恢复
        for image_idx in range(start_idx, end_idx):
            success = self._recover_single_image_pose(image_path, image_idx, scale_ratio, predicted_scale_ratio)
            if not success:
                print(f"Failed to recover pose for image index {image_idx}")
                return False
        
        # ========== 新增：变换 tracks 的 3D 点到真实坐标系 ==========
        if transform_tracks and len(self.batch_tracks) > 0:
            success = self._transform_batch_tracks_to_real_coords(
                start_idx, end_idx, scale_ratio
            )
            if not success:
                print(f"Failed to transform tracks to real coordinates")
                return False

        return True

    def _recover_single_image_pose(self, image_path: Path, image_idx: int, scale_ratio: float, predicted_scale_ratio: float) -> bool:
        """Recover original pose for the current image.
        当前的三维点云是没有恢复尺度的，只存了最新的尺度，需要之后基于最新的尺度恢复所有尺度的三维点云。

        Args:
            image_path: Path to the image
            idx: Index of the image
            scale_ratio: Scale ratio
            predicted_scale_ratio: Predicted scale ratio

        Returns:
            True if successful, False otherwise
        """
        inference_data = self.inference_outputs[-1]
        inference_output = inference_data['current_output']
        image_path = Path(inference_data['image_path'])
        ori_extrinsic = self.ori_extrinsic[image_idx]
        ori_intrinsic = self.ori_intrinsic[image_idx]

        # Get original pose (cam2world)
        R_camera_orig = np.array(ori_extrinsic['R_camera']) # (w2c)
        tvec_orig = np.array(ori_extrinsic['tvec']) # (w2c)
        # Construct original cam2world transformation matrix
        T_world_cam_orig = np.eye(4, dtype=np.float32)
        T_world_cam_orig[:3, :3] = R_camera_orig.T  # cam to world
        T_world_cam_orig[:3, 3] = -R_camera_orig.T @ tvec_orig  # camera position in world
        
        # inference_output['camera_poses'] is a tensor of shape (B, 4, 4)
        T_world_cam_infer = inference_output['camera_poses'][0].cpu().numpy()  # (4, 4)
        # Compute transformation from inference world to original world
        # T_orig_infer = T_world_cam_orig @ inv(T_world_cam_infer)
        T_cam_world_infer = np.linalg.inv(T_world_cam_infer)
        T_transform = T_world_cam_orig @ T_cam_world_infer

        # Create recovered output dictionary
        recovered_inference_output = {
            'image_path': str(image_path),
            'image_width': ori_intrinsic['width'],
            'image_height': ori_intrinsic['height'],
            'camera_K': ori_intrinsic['K'],
            'transformation_matrix': T_transform,
            'scale_ratio': scale_ratio, # 最新的尺度
            'predicted_scale_ratio': predicted_scale_ratio, # 预测的尺度
            'conf': inference_output['conf'], # 置信度
        }
        
        # ==================== Transform pts3d =====================
        # Transform pts3d (world coordinates)
        pts3d_infer = inference_output['pts3d']  # (1, H, W, 3) on cuda
        # Convert to numpy and get shape
        pts3d_np = pts3d_infer[0].cpu().numpy()  # (H, W, 3)
        H, W, _ = pts3d_np.shape
        # Reshape to (H*W, 3) for transformation
        pts_flat = pts3d_np.reshape(-1, 3)
        # Apply scale ratio
        pts_flat_scaled = pts_flat * scale_ratio # 应用缩放比例
        # Convert to homogeneous coordinates (H*W, 4)
        pts_homo = np.concatenate([pts_flat_scaled, np.ones((pts_flat_scaled.shape[0], 1), dtype=np.float32)], axis=1)
        # Apply transformation (from inference world to original world)
        pts_recovered_homo = (T_transform @ pts_homo.T).T  # (H*W, 4)
        # Convert back to 3D (H*W, 3)
        pts_recovered = pts_recovered_homo[:, :3]
        # Reshape back to (H, W, 3)
        pts_recovered = pts_recovered.reshape(H, W, 3)
        # Store as tensor on the same device
        device = inference_output['pts3d'].device
        recovered_inference_output['pts3d'] = torch.from_numpy(pts_recovered).unsqueeze(0).to(device)
    
        # ==================== Transform pts3d_cam =====================
        # Transform pts3d_cam (camera coordinates)
        pts3d_cam_infer = inference_output['pts3d_cam']  # (1, H, W, 3) on cuda
        # Convert to numpy
        pts3d_cam_np = pts3d_cam_infer[0].cpu().numpy()  # (H, W, 3)
        H, W, _ = pts3d_cam_np.shape
        # Reshape to (H*W, 3)
        pts_cam_flat = pts3d_cam_np.reshape(-1, 3)
        # Apply scale ratio
        pts_cam_flat_scaled = pts_cam_flat * scale_ratio # 应用缩放比例
        # Convert to homogeneous coordinates
        pts_cam_homo = np.concatenate([pts_cam_flat_scaled, np.ones((pts_cam_flat_scaled.shape[0], 1), dtype=np.float32)], axis=1)
        # Transform from inference camera frame to original camera frame
        # T_cam_orig_cam_infer = inv(T_world_cam_orig) @ T_world_cam_infer
        T_cam_orig_world = np.linalg.inv(T_world_cam_orig)
        T_cam_orig_cam_infer = T_cam_orig_world @ T_world_cam_infer
        pts_cam_recovered_homo = (T_cam_orig_cam_infer @ pts_cam_homo.T).T
        # Convert back to 3D
        pts_cam_recovered = pts_cam_recovered_homo[:, :3]
        # Reshape back to (H, W, 3)
        pts_cam_recovered = pts_cam_recovered.reshape(H, W, 3)
        # Store as tensor
        device = inference_output['pts3d_cam'].device
        recovered_inference_output['pts3d_cam'] = torch.from_numpy(pts_cam_recovered).unsqueeze(0).to(device)
    
        # ==================== Update camera pose to original pose =====================
        # Update camera position to original position
        cam_pos_orig = T_world_cam_orig[:3, 3]
        device = inference_output['cam_trans'].device
        recovered_inference_output['cam_trans'] = torch.from_numpy(cam_pos_orig).unsqueeze(0).to(device)

        # Camera rotation in original world coordinates
        R_recovered = T_world_cam_orig[:3, :3]
        quat_recovered = R.from_matrix(R_recovered).as_quat()  # [x, y, z, w]
        device = inference_output['cam_quats'].device
        recovered_inference_output['cam_quats'] = torch.from_numpy(quat_recovered).unsqueeze(0).float().to(device)
    
        # Camera Pose to original pose
        device = inference_output['camera_poses'].device
        recovered_inference_output['camera_poses'] = torch.from_numpy(T_world_cam_orig).unsqueeze(0).to(device)

        # Store recovered inference outputs
        self.recovered_inference_outputs.append(recovered_inference_output)

        if self.verbose:
            print(f"✓ Pose recovered for image: {ori_extrinsic['image_name']}")
            print(f"  Original camera position (ENU): [{ori_extrinsic['enu'][0]:.2f}, {ori_extrinsic['enu'][1]:.2f}, {ori_extrinsic['enu'][2]:.2f}]")
            if 'pts3d' in recovered_inference_output:
                pts_shape = recovered_inference_output['pts3d'].shape
                print(f"  pts3d transformed: shape {pts_shape}")
            if 'pts3d_cam' in recovered_inference_output:
                pts_cam_shape = recovered_inference_output['pts3d_cam'].shape
                print(f"  pts3d_cam transformed: shape {pts_cam_shape}")
            print(f"  Camera pose updated to original coordinates")

        return True

    def _transform_batch_tracks_to_real_coords(self, start_idx: int, end_idx: int, scale_ratio: float) -> bool:
        """Transform batch tracks' 3D points from inference to real coordinate system.
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            scale_ratio: Scale ratio to apply
            
        Returns:
            True if successful
        """
        try:
            if self.verbose:
                print(f"  Transforming tracks 3D points to real coordinate system...")
            
            # 获取最新的 batch_tracks
            if len(self.batch_tracks) == 0:
                return True
            
            latest_batch = self.batch_tracks[-1]
            points_3d = latest_batch['points_3d']  # (P, 3) - 推理坐标系
            
            if points_3d is None:
                return True
            
            # 使用第一张图像的变换矩阵作为参考
            # （因为所有图像的 world 坐标系变换是一致的）
            inference_data = self.inference_outputs[start_idx]
            inference_output = inference_data['current_output']
            ori_extrinsic = self.ori_extrinsic[start_idx]
            
            # 获取变换矩阵
            R_camera_orig = np.array(ori_extrinsic['R_camera'])
            tvec_orig = np.array(ori_extrinsic['tvec'])
            T_world_cam_orig = np.eye(4, dtype=np.float32)
            T_world_cam_orig[:3, :3] = R_camera_orig.T
            T_world_cam_orig[:3, 3] = -R_camera_orig.T @ tvec_orig
            
            T_world_cam_infer = inference_output['camera_poses'][0].cpu().numpy()
            T_cam_world_infer = np.linalg.inv(T_world_cam_infer)
            T_transform = T_world_cam_orig @ T_cam_world_infer
            
            # 应用尺度和变换
            points_3d_scaled = points_3d * scale_ratio  # (P, 3)
            points_3d_homo = np.concatenate([
                points_3d_scaled, 
                np.ones((points_3d_scaled.shape[0], 1), dtype=np.float32)
            ], axis=1)  # (P, 4)
            
            points_3d_transformed = (T_transform @ points_3d_homo.T).T[:, :3]  # (P, 3)
            
            # 更新 batch_tracks 中的 3D 点
            latest_batch['points_3d'] = points_3d_transformed
            latest_batch['points_3d_transformed'] = True  # 标记已变换
            
            if self.verbose:
                print(f"  ✓ Transformed {points_3d.shape[0]} 3D points to real coordinate system")
                print(f"    Scale ratio applied: {scale_ratio:.6f}")
            
            return True
            
        except Exception as e:
            print(f"  Error transforming tracks: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _filter_reconstruction_to_images(
        self,
        full_recon: pycolmap.Reconstruction,
        keep_image_names: set,
    ) -> pycolmap.Reconstruction:
        """
        从完整的 SfM 重建中提取仅包含指定影像的子重建。

        全量影像（含中间帧）参与 SfM 可获得更准确的位姿，
        但下游流程只需要 batch 影像。此方法将完整重建裁剪为
        仅包含 keep_image_names 指定影像的子集，同时保留
        这些影像可观测到的 3D 点及其 track 信息。

        Args:
            full_recon: 包含全部影像的完整 SfM 重建
            keep_image_names: 需要保留的影像文件名集合

        Returns:
            仅包含指定影像的新 Reconstruction 对象
        """
        # 快速路径：所有影像都需要保留
        keep_ids = {
            img_id for img_id, img in full_recon.images.items()
            if img.name in keep_image_names
        }
        if len(keep_ids) == len(full_recon.images):
            return full_recon

        filtered = pycolmap.Reconstruction()

        # 1. 复制相机
        for cam_id, cam in full_recon.cameras.items():
            filtered.add_camera(cam)

        # 2. 复制目标影像（重新编号为连续 ID）
        old_to_new_id = {}
        new_id = 1
        for old_id in sorted(keep_ids):
            old_img = full_recon.images[old_id]
            new_img = pycolmap.Image(
                image_id=new_id,
                name=old_img.name,
                camera_id=old_img.camera_id,
            )
            new_img.cam_from_world = old_img.cam_from_world
            new_img.points2D = pycolmap.ListPoint2D(
                [pycolmap.Point2D(pt.xy) for pt in old_img.points2D]
            )
            filtered.add_image(new_img)
            old_to_new_id[old_id] = new_id
            new_id += 1

        # 3. 复制在保留影像中有至少 2 个观测的 3D 点
        for pt3d_id, pt3d in full_recon.points3D.items():
            new_elements = []
            for elem in pt3d.track.elements:
                if elem.image_id in old_to_new_id:
                    new_elements.append(
                        pycolmap.TrackElement(old_to_new_id[elem.image_id], elem.point2D_idx)
                    )
            if len(new_elements) >= 2:
                new_track = pycolmap.Track()
                new_track.elements = new_elements
                new_pt3d_id = filtered.add_point3D(pt3d.xyz, new_track, pt3d.color)
                for elem in new_elements:
                    img = filtered.images[elem.image_id]
                    pt2d = img.points2D[elem.point2D_idx]
                    pt2d.point3D_id = new_pt3d_id

        if self.verbose:
            print(f"    Filtered reconstruction: {len(full_recon.images)} -> {len(filtered.images)} images, "
                  f"{len(full_recon.points3D)} -> {len(filtered.points3D)} 3D points")

        return filtered

    def _run_batch_sfm(self, start_idx: int, end_idx: int,
                       images_dir: Path, output_dir: Path,
                       batch_image_names: Optional[set] = None) -> Optional[Dict]:
        """
        对当前批次的影像直接运行SfM，得到稀疏重建结果。

        使用 FeatureMatcherSfM 对 images_dir 中的全部影像执行特征提取、
        匹配和增量SfM。如果提供了 batch_image_names，SfM 完成后会将
        重建过滤为仅包含这些影像的子集（间隔选取的推理影像），从而在
        保留密集 SfM 精度的同时，输出与下游 batch 流程兼容的结构。

        Args:
            start_idx: 起始影像索引（在 self.image_paths 中）
            end_idx: 结束影像索引（不包含）
            images_dir: 影像所在目录（可能包含中间帧）
            output_dir: SfM 输出目录
            batch_image_names: 需要保留的影像文件名集合。若为 None 则保留全部。

        Returns:
            包含子重建信息的字典，失败返回 None
        """
        batch_image_paths = self.image_paths[start_idx:end_idx]
        if len(batch_image_paths) < 2:
            if self.verbose:
                print(f"  Warning: Not enough images ({len(batch_image_paths)}) for batch SfM, need at least 2")
            return None

        try:
            # images_dir 可能包含中间帧，统计实际影像数量
            supported_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            sfm_image_count = sum(1 for f in images_dir.iterdir() if f.suffix in supported_exts)

            if self.verbose:
                print(f"  Running batch SfM for images {start_idx} to {end_idx - 1} "
                      f"(batch: {len(batch_image_paths)}, SfM total: {sfm_image_count})...")

            batch_matcher = FeatureMatcherSfM(
                input_dir=images_dir,
                output_dir=output_dir,
                imgsz=2048,
                num_features=8192,
                match_mode="sequential", # mode: "exhaustive", "spatial", "sequential"
                num_neighbors=min(20, sfm_image_count),
                max_distance=1000.0,
                sfm_mode="incremental", # mode: "direct", "direct_ba", "incremental"
                verbose=self.verbose,
            )
            success = batch_matcher.run_pipeline()

            if not success or batch_matcher.rec_prior is None:
                if self.verbose:
                    print(f"  Warning: Batch SfM failed for images {start_idx} to {end_idx - 1}")
                return None

            batch_recon = batch_matcher.rec_prior

            # FeatureMatcherSfM._align_to_enu() 将 batch 的第一张影像置于 (0,0,0)，
            # 需要将局部 ENU 坐标系转换到目标坐标系。
            if self.batch_sfm_coord_mode == 'global_enu':
                # 模式1：平移到全局 ENU 坐标系
                # 平移量 = batch 第一张影像在全局 ENU 中的实际位置
                global_enu_offset = np.array(self.ori_extrinsic[start_idx]['enu'], dtype=np.float64)
                if np.linalg.norm(global_enu_offset) > 1e-6:
                    sim3_translate = pycolmap.Sim3d(
                        1.0,
                        pycolmap.Rotation3d(),
                        global_enu_offset.reshape(3, 1)
                    )
                    batch_recon.transform(sim3_translate)
                    if self.verbose:
                        print(f"    Translated batch SfM to global ENU: "
                              f"offset=[{global_enu_offset[0]:.2f}, {global_enu_offset[1]:.2f}, {global_enu_offset[2]:.2f}]")

            elif self.batch_sfm_coord_mode == 'wgs84':
                # 模式2：将局部 ENU 坐标转换为 WGS84 经纬度高程 (lat, lon, alt)
                # 局部 ENU 原点对应 batch 第一张影像的 GPS 位置
                if not UTM_EXPORT_AVAILABLE:
                    raise RuntimeError("batch_sfm_coord_mode='wgs84' requires pymap3d. "
                                       "Install with: pip install pymap3d")

                batch_gps = self.ori_extrinsic[start_idx]['gps']  # [lat, lon, alt]
                lat0, lon0, alt0 = batch_gps[0], batch_gps[1], batch_gps[2]

                for pt3d_id in list(batch_recon.points3D.keys()):
                    e, n, u = batch_recon.points3D[pt3d_id].xyz
                    lat, lon, alt = pm.enu2geodetic(e, n, u, lat0, lon0, alt0)
                    batch_recon.points3D[pt3d_id].xyz = np.array([lat, lon, alt])

                for img_id in batch_recon.images:
                    image = batch_recon.images[img_id]
                    R = image.cam_from_world.rotation.matrix()
                    t = image.cam_from_world.translation
                    center_enu = -R.T @ t
                    lat, lon, alt = pm.enu2geodetic(
                        center_enu[0], center_enu[1], center_enu[2],
                        lat0, lon0, alt0
                    )
                    center_wgs84 = np.array([lat, lon, alt])
                    t_new = -R @ center_wgs84
                    image.cam_from_world = pycolmap.Rigid3d(
                        rotation=image.cam_from_world.rotation,
                        translation=t_new
                    )

                if self.verbose:
                    print(f"    Converted batch SfM to WGS84: "
                          f"origin=({lat0:.6f}, {lon0:.6f}, {alt0:.1f})")

            # 如果提供了 batch_image_names，过滤重建为仅包含 batch 影像
            if batch_image_names and len(batch_recon.images) > len(batch_image_names):
                if self.verbose:
                    print(f"    Filtering reconstruction to batch images only...")
                batch_recon = self._filter_reconstruction_to_images(batch_recon, batch_image_names)

            image_name_to_path = {p.name: str(p) for p in batch_image_paths}

            sfm_result = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'image_paths': batch_image_paths,
                'image_name_mapping': image_name_to_path,
                'reconstruction': batch_recon,
                'num_images': len(batch_recon.images),
                'num_points3D': len(batch_recon.points3D),
                'num_cameras': len(batch_recon.cameras),
                'source': 'batch_sfm',
            }

            if self.verbose:
                print(f"  ✓ Batch SfM completed")
                print(f"    Number of images: {sfm_result['num_images']}")
                print(f"    Number of 3D points: {sfm_result['num_points3D']}")
                print(f"    Number of cameras: {sfm_result['num_cameras']}")

            return sfm_result

        except Exception as e:
            print(f"  Error running batch SfM: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _predict_tracks_for_batch(self, start_idx: int, end_idx: int) -> bool:
        """Predict tracks for a batch of images.
        
        Args:
            start_idx: Starting index in self.inference_outputs (未恢复的起始索引)
            end_idx: Ending index (exclusive) in self.inference_outputs
            
        Returns:
            True if successful, False otherwise
        """
        tracker = get_timing_tracker()
        if tracker:
            tracker.start("predict_tracks")
        
        try:
            if self.verbose:
                print(f"\n  Predicting tracks for batch (indices {start_idx} to {end_idx-1})...")
            
            # 检查最新的 outputs 是否正好覆盖我们需要的范围
            latest_inference = self.inference_outputs[-1]
            latest_outputs = latest_inference['outputs']
            num_images = len(self.inference_outputs)
            num_outputs = len(latest_outputs)
            latest_start_idx = num_images - num_outputs
            latest_end_idx = num_images

            use_latest_outputs = (latest_start_idx == start_idx and latest_end_idx == end_idx)

            # 准备这批图像的数据
            batch_image_paths = []
            batch_images = []
            batch_confs = []
            batch_points_3d = []

            if use_latest_outputs:
                # 直接从最新的outputs列表中提取数据
                for i, output in enumerate(latest_outputs):
                    idx = start_idx + i      
                    # 收集图像路径
                    batch_image_paths.append(self.inference_outputs[idx]['image_path'])
                    
                    # 从 preprocessed_views 获取预处理后的图像（与推理输出尺寸匹配）
                    if idx < len(self.preprocessed_views):
                        img = self.preprocessed_views[idx]['img']  # [1, 3, H, W] tensor
                        # 去掉批次维度，得到 [3, H, W]
                        if img.dim() == 4 and img.shape[0] == 1:
                            img = img.squeeze(0)  # [3, H, W]
                        batch_images.append(img)

                    # 从 outputs[i] 中直接获取3D点（未恢复尺度的）
                    if 'pts3d' in output:
                        pts3d = output['pts3d']  # (1, H, W, 3) tensor
                        batch_points_3d.append(pts3d[0])

                    # 获取置信度
                    if 'conf' in output:
                        conf = output['conf']  # (1, H, W)
                        batch_confs.append(conf[0])
            else:
                # 范围不匹配，从各个 inference_outputs[idx]['current_output'] 获取
                for idx in range(start_idx, end_idx):
                    # 从 inference_outputs 获取推理输出（而不是 recovered_inference_outputs）
                    inference_data = self.inference_outputs[idx]
                    inference_output = inference_data['current_output']
                    # 收集图像路径
                    batch_image_paths.append(inference_data['image_path'])
                    
                    # 从 preprocessed_views 获取预处理后的图像（与推理输出尺寸匹配）
                    if idx < len(self.preprocessed_views):
                        img = self.preprocessed_views[idx]['img']  # [1, 3, H, W] tensor
                        # 去掉批次维度，得到 [3, H, W]
                        if img.dim() == 4 and img.shape[0] == 1:
                            img = img.squeeze(0)  # [3, H, W]
                        batch_images.append(img)
                    
                    # 从 inference_outputs 获取3D点（未恢复尺度的）
                    if 'pts3d' in inference_output:
                        pts3d = inference_output['pts3d']  # (1, H, W, 3) tensor
                        batch_points_3d.append(pts3d[0])
                    
                    # 获取置信度
                    if 'conf' in inference_output:
                        conf = inference_output['conf']  # (1, H, W)
                        batch_confs.append(conf[0])
            
            # 将列表转换为tensor
            if len(batch_images) > 0:
                # batch_images 中每个元素是 [3, H, W]
                batch_images_tensor = torch.stack(batch_images)  # (B, 3, H, W)
                
                # 确保在 CUDA 上
                if not batch_images_tensor.is_cuda:
                    batch_images_tensor = batch_images_tensor.cuda()
                
                # 准备置信度和3D点 - 转为 numpy 数组再转回 tensor（断开梯度）
                # 这样可以确保不携带计算图
                if batch_confs:
                    batch_confs_np = torch.stack(batch_confs).detach().cpu().numpy()
                    # 转回 tensor（不带梯度）并放到 CUDA
                    batch_confs_tensor = torch.from_numpy(batch_confs_np).cuda()
                else:
                    batch_confs_tensor = None
                    
                if batch_points_3d:
                    batch_points_3d_np = torch.stack(batch_points_3d).detach().cpu().numpy()
                    # 转回 tensor（不带梯度）并放到 CUDA
                    batch_points_3d_tensor = torch.from_numpy(batch_points_3d_np).cuda()
                else:
                    batch_points_3d_tensor = None
                
                # # 【关键修改】准备置信度和3D点 - 直接使用 numpy 数组，不转回 CUDA tensor
                # if batch_confs:
                #     batch_confs_np = torch.stack(batch_confs).detach().cpu().numpy()
                #     batch_confs_tensor = batch_confs_np  # 直接使用 numpy，不转回 CUDA
                # else:
                #     batch_confs_tensor = None
                    
                # if batch_points_3d:
                #     batch_points_3d_np = torch.stack(batch_points_3d).detach().cpu().numpy()
                #     batch_points_3d_tensor = batch_points_3d_np  # 直接使用 numpy，不转回 CUDA
                # else:
                #     batch_points_3d_tensor = None

                # 准备原始图像列表
                batch_original_images = []
                for idx in range(start_idx, end_idx):
                    if idx < len(self.input_views):
                        # input_views 中存储的是原始图像 tensor [H, W, 3], uint8
                        img_orig = self.input_views[idx]['img'].cpu().numpy()
                        batch_original_images.append(img_orig)

                # 运行特征点跟踪
                # 传入不带梯度的 tensor
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                            batch_images_tensor,
                            conf=batch_confs_tensor,
                            points_3d=batch_points_3d_tensor,
                            max_query_pts=self.max_query_pts,
                            query_frame_num=self.query_frame_num,
                            keypoint_extractor="aliked+sp+sift",
                            fine_tracking=True,
                            original_images=batch_original_images,
                        )
                
                self.batch_tracks.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'image_indices': list(range(start_idx, end_idx)),
                    'image_paths': batch_image_paths,
                    'pred_tracks': pred_tracks,  # 已经是 numpy array
                    'pred_vis_scores': pred_vis_scores,  # 已经是 numpy array
                    'pred_confs': pred_confs,  # numpy array 或 None
                    'points_3d': points_3d,  # numpy array 或 None
                    'points_rgb': points_rgb,  # numpy array 或 None
                })
                
                num_frames = len(batch_image_paths)
                for frame_idx in range(num_frames):
                    image_idx = start_idx + frame_idx
                    
                    # 为每个影像创建跟踪信息
                    image_track_info = {
                        'image_idx': image_idx,
                        'image_path': batch_image_paths[frame_idx],
                        # 当前帧的track坐标（2D）
                        'tracks_2d': pred_tracks[frame_idx],# 该帧上的所有跟踪点 (num_points, 2)
                        'vis_scores': pred_vis_scores[frame_idx],  # 该帧上的可见性分数 (num_points,)
                        # 匹配信息
                        'matched_images_indices': list(range(start_idx, end_idx)),  # 与哪些影像匹配（索引）
                        'matched_images_paths': batch_image_paths,  # 与哪些影像匹配（路径）
                        'num_tracks': pred_tracks.shape[1],  # 跟踪点数量
                    }
                    
                    # 如果有置信度和3D点信息，也可以添加
                    # 注意：pred_confs 和 points_3d 是按查询帧组织的，可能需要特殊处理
                    if pred_confs is not None:
                        image_track_info['confs'] = pred_confs
                    if points_3d is not None:
                        image_track_info['points_3d'] = points_3d
                    if points_rgb is not None:
                        image_track_info['points_rgb'] = points_rgb
                    
                    self.image_tracks.append(image_track_info)

                if self.verbose:
                    print(f"  ✓ Track prediction completed")
                    print(f"    Tracks shape: {pred_tracks.shape}")
                    print(f"    Number of tracked points: {pred_tracks.shape[1]}")
                
                torch.cuda.empty_cache()
                tracker = get_timing_tracker()
                if tracker:
                    tracker.end("predict_tracks")
                return True
            else:
                print("  Warning: No images available for track prediction")
                tracker = get_timing_tracker()
                if tracker:
                    tracker.end("predict_tracks")
                return False
                
        except Exception as e:
            print(f"  Error during track prediction: {e}")
            import traceback
            traceback.print_exc()
            tracker = get_timing_tracker()
            if tracker:
                tracker.end("predict_tracks")
            return False

    def _build_pycolmap_reconstruction(self, start_idx: int, end_idx: int, use_recovered: bool = False) -> bool:
        """Build pycolmap reconstruction from predicted tracks.
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (exclusive)
            
        Returns:
            True if successful, False otherwise
        """
        tracker = get_timing_tracker()
        if tracker:
            tracker.start("build_pycolmap")
        
        try:
            if self.verbose:
                print(f"\n  Building pycolmap reconstruction for images {start_idx} to {end_idx-1}...")
            
            # 获取最新的batch_tracks
            latest_batch = self.batch_tracks[-1]
            
            # 准备数据
            image_paths = latest_batch['image_paths']
            pred_tracks = latest_batch['pred_tracks']  # (N, P, 2)
            pred_vis_scores = latest_batch['pred_vis_scores']  # (N, P)
            points_3d = latest_batch['points_3d']  # (P, 3) 或 None
            points_rgb = latest_batch['points_rgb']  # (P, 3) 或 None
            
            # 检查是否可以使用最新的 outputs（与 _predict_tracks_for_batch 保持一致）
            use_latest_outputs = False
            if not use_recovered:  # 只在不使用恢复位姿时才优化
                latest_inference = self.inference_outputs[-1]
                latest_outputs = latest_inference['outputs']
                num_images = len(self.inference_outputs)
                num_outputs = len(latest_outputs)
                latest_start_idx = num_images - num_outputs
                latest_end_idx = num_images
                use_latest_outputs = (latest_start_idx == start_idx and latest_end_idx == end_idx)
                
                if self.verbose and use_latest_outputs:
                    print(f"  ✓ Using latest inference outputs for camera parameters")
            
            # 准备 extrinsics (N, 3, 4)
            extrinsics = []
            if use_latest_outputs:
                # 直接从最新的 outputs 列表中获取
                for i, output in enumerate(latest_outputs):
                    cam2world = output['camera_poses'][0].cpu().numpy()
                    # 转为 world2cam (3, 4)
                    world2cam = np.linalg.inv(cam2world)[:3, :]  # (3, 4)
                    extrinsics.append(world2cam)
            else:
                # 原有逻辑：从各个 inference_outputs 或 recovered_inference_outputs 获取
                for idx in range(start_idx, end_idx):
                    if use_recovered:
                        recovered = self.recovered_inference_outputs[idx]
                        cam2world = recovered['camera_poses'][0].cpu().numpy()
                    else:
                        inference_data = self.inference_outputs[idx]
                        inference_output = inference_data['current_output']
                        cam2world = inference_output['camera_poses'][0].cpu().numpy()
                    # 转为 world2cam (3, 4)
                    world2cam = np.linalg.inv(cam2world)[:3, :]  # (3, 4)
                    extrinsics.append(world2cam)
            extrinsics = np.stack(extrinsics)  # (N, 3, 4)
            
            # 准备 intrinsics (N, 3, 3)
            intrinsics = []
            if use_latest_outputs:
                # 直接从最新的 outputs 列表中获取
                for i, output in enumerate(latest_outputs):
                    K = output['intrinsics'][0].cpu().numpy()  # (3, 3)
                    intrinsics.append(K)
            else:
                # 原有逻辑
                for idx in range(start_idx, end_idx):
                    if use_recovered:
                        # 如果使用恢复的位姿，仍使用原始内参
                        K = np.array(self.ori_intrinsic[idx]['K'])
                    else:
                        # 使用推理内参
                        inference_data = self.inference_outputs[idx]
                        inference_output = inference_data['current_output']
                        K = inference_output['intrinsics'][0].cpu().numpy()  # (3, 3)
                    intrinsics.append(K)
            intrinsics = np.stack(intrinsics)  # (N, 3, 3)

            # 准备 image_size (2,) - 使用原始图像尺寸
            width = self.ori_intrinsic[start_idx]['width']
            height = self.ori_intrinsic[start_idx]['height']
            image_size = np.array([width, height])
            
            # 准备 masks - 从可见性分数转换
            # 可见性阈值可以调整
            masks = pred_vis_scores > self.pred_vis_scores_thres_value  # (N, P)
            
            # === 调试输出：点云过滤统计 ===
            if self.verbose:
                total_points = points_3d.shape[0] if points_3d is not None else 0
                vis_filtered_per_frame = masks.sum(axis=1)  # 每帧通过可见性过滤的点数
                vis_filtered_total = masks.any(axis=0).sum()  # 至少在一帧可见的点数
                vis_filtered_2frames = (masks.sum(axis=0) >= 2).sum()  # 至少在2帧可见的点数
                print(f"  [Debug] 点云过滤统计:")
                print(f"    原始点数: {total_points}")
                print(f"    可见性阈值: {self.pred_vis_scores_thres_value}")
                print(f"    每帧通过可见性过滤的点数: {vis_filtered_per_frame}")
                print(f"    至少在1帧可见的点数: {vis_filtered_total}")
                print(f"    至少在2帧可见的点数（将保留）: {vis_filtered_2frames}")
                
                # 检查3D点坐标范围
                if points_3d is not None:
                    pts_min = points_3d.min(axis=0)
                    pts_max = points_3d.max(axis=0)
                    pts_in_range = (np.abs(points_3d).max(axis=1) < self.max_points3D_val).sum()
                    print(f"    3D点坐标范围: min={pts_min}, max={pts_max}")
                    print(f"    坐标范围阈值: {self.max_points3D_val}")
                    print(f"    坐标在范围内的点数: {pts_in_range}/{total_points}")
            
            # 调用 batch_np_matrix_to_pycolmap
            reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                points3d=points_3d,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                tracks=pred_tracks,
                image_size=image_size,
                masks=masks,
                max_reproj_error=self.max_reproj_error,  # 可以作为参数配置
                max_points3D_val=self.max_points3D_val,
                shared_camera=False,  # 每个相机独立
                camera_type="PINHOLE",  # 使用OPENCV模型
                points_rgb=points_rgb,
                min_inlier_per_frame=self.min_inlier_per_frame,
            )
            
            if reconstruction is None:
                print("  Warning: Failed to build pycolmap reconstruction")
                return False
            
            # === 调试输出：重建后的点数 ===
            if self.verbose:
                print(f"  [Debug] batch_np_matrix_to_pycolmap 完成:")
                print(f"    重建后3D点数: {len(reconstruction.points3D)}")
                print(f"    有效track mask 中 True 的数量: {valid_track_mask.sum() if valid_track_mask is not None else 'N/A'}")

            # # Bundle Adjustment
            # ba_options = pycolmap.BundleAdjustmentOptions()
            # pycolmap.bundle_adjustment(reconstruction, ba_options)

            # 准备 image_paths 列表（文件名）
            image_paths_list = [Path(path).name for path in image_paths]

            # 准备 original_coords (N, 6) 数组
            # 格式: [x1, y1, x2, y2, width, height]
            # 其中 x1, y1 是裁剪左上角，x2, y2 是右下角，width, height 是原始尺寸
            original_coords_list = []
            for idx in range(start_idx, end_idx):
                scale_info = self.scale_info[idx]
                ori_w, ori_h = scale_info['original_size']
                # 由于没有裁剪，x1=0, y1=0, x2=ori_w, y2=ori_h
                original_coords_list.append(np.array([0, 0, ori_w, ori_h, ori_w, ori_h], dtype=np.float32))
            original_coords = np.stack(original_coords_list)  # (N, 6)

            # 获取预处理后的图像尺寸（img_size）
            proc_w = self.scale_info[start_idx]['output_size'][0]
            proc_h = self.scale_info[start_idx]['output_size'][1]

            # 调用函数
            reconstruction = rename_colmap_recons_and_rescale_camera(
                reconstruction=reconstruction,
                image_paths=image_paths_list,
                original_coords=original_coords,
                img_size=(proc_w, proc_h),
                shift_point2d_to_original_res=True,
                shared_camera=False,
            )

            # 步骤1：先缩放到原始图像尺寸（基本对齐），对齐到已知的影像pose位置
            if len(self.sfm_reconstructions) > 0:
                reconstruction = rescale_reconstruction_to_original_size(
                    reconstruction=reconstruction,
                    ori_extrinsics=self.ori_extrinsic,
                    sfm_reconstructions=self.sfm_reconstructions,
                    start_idx=start_idx, 
                    end_idx=end_idx,
                    alignment_mode='pcl_alignment',
                    verbose=self.verbose,
                )
            else:
                reconstruction = rescale_reconstruction_to_original_size(
                    reconstruction=reconstruction,
                    ori_extrinsics=self.ori_extrinsic,
                    sfm_reconstructions=self.sfm_reconstructions,
                    start_idx=start_idx, 
                    end_idx=end_idx,
                    alignment_mode='image_alignment',
                    image_alignment_max_error=10.0,
                    image_alignment_min_inlier_ratio=0.3,
                    verbose=self.verbose,
                )
            # === 调试输出：rescale 后的点数 ===
            if self.verbose:
                print(f"  [Debug] rescale_reconstruction_to_original_size 完成:")
                print(f"    rescale后3D点数: {len(reconstruction.points3D)}")
            
            # 保存重建结果
            temp_path = self.output_dir / "temp_rescale" / f"{start_idx}_{end_idx}"
            temp_path.mkdir(parents=True, exist_ok=True)
            reconstruction.write_text(str(temp_path))
            reconstruction.export_PLY(str(temp_path / "points3D.ply"))

            # # 步骤2：如果存在merged_reconstruction，与它对齐
            # if self.merged_reconstruction is not None:
            #     aligned_recon = self._align_current_reconstruction_to_merged(
            #         reconstruction,
            #     )
            # else:
            #     aligned_recon = self._align_current_reconstruction_by_point_cloud(
            #         reconstruction,
            #     )

            aligned_recon = self._align_current_reconstruction_by_point_cloud(
                    reconstruction,
                    match_type='use_bidirectional',
                )
            
            # === 调试输出：对齐结果检查 ===
            if self.verbose:
                if aligned_recon is False:
                    print(f"  [Debug] ⚠ _align_current_reconstruction_by_point_cloud 返回 False（对齐失败）")
                    print(f"    将使用原始 reconstruction（未对齐）")
                    aligned_recon = reconstruction  # 回退到原始 reconstruction
                elif isinstance(aligned_recon, pycolmap.Reconstruction):
                    print(f"  [Debug] _align_current_reconstruction_by_point_cloud 完成:")
                    print(f"    对齐后3D点数: {len(aligned_recon.points3D)}")
                else:
                    print(f"  [Debug] ⚠ 意外的返回类型: {type(aligned_recon)}")
                    aligned_recon = reconstruction
            else:
                # 非 verbose 模式下也需要处理 False 返回值
                if aligned_recon is False:
                    aligned_recon = reconstruction
            
            # 保存重建结果
            temp_path = self.output_dir / "temp_aligned" / f"{start_idx}_{end_idx}"
            temp_path.mkdir(parents=True, exist_ok=True)
            aligned_recon.write_text(str(temp_path))
            aligned_recon.export_PLY(str(temp_path / "points3D.ply"))

            # 先缩放到原始图像尺寸（基本对齐）
            aligned_recon=reconstruction 

            # 提取逐像素3D点对应关系（包含 conf）
            # 从最新的 inference_outputs 中获取 outputs_list
            latest_inference = self.inference_outputs[-1]
            outputs_list = latest_inference.get('outputs', [])
            global_indices = list(range(start_idx, end_idx))
            
            if self.merge_method == 'points_only' and self.points_merge_mode == 'fast':
                batch_pixel_3d_mapping = {}
            else:
                batch_pixel_3d_mapping = self._extract_pixel_to_3d_mapping_for_batch(
                    outputs_list=outputs_list,
                    global_indices=global_indices,
                    conf_threshold=1.0,
                    verbose=True
                )
            
            self.inference_reconstructions.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'image_paths': image_paths,
                'reconstruction': aligned_recon,
                'valid_track_mask': valid_track_mask,
                'pixel_3d_mapping': batch_pixel_3d_mapping,
            })
            
            if self.verbose:
                print(f"  ✓ PyColmap reconstruction built")
                print(f"    Number of 3D points: {len(aligned_recon.points3D)}")
                print(f"    Number of cameras: {len(aligned_recon.cameras)}")
                print(f"    Number of images: {len(aligned_recon.images)}")
            
            tracker = get_timing_tracker()
            if tracker:
                tracker.end("build_pycolmap")
            return True
            
        except Exception as e:
            print(f"  Error building pycolmap reconstruction: {e}")
            import traceback
            traceback.print_exc()
            tracker = get_timing_tracker()
            if tracker:
                tracker.end("build_pycolmap")
            return False

    def _extract_and_resize_pts3d_for_batch(
        self,
        outputs_list: List[Dict],
        global_indices: List[int],
        verbose: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        从 outputs_list 中提取逐像素 3D 点并缩放到原始图像尺寸
        
        与置信度图不同，pts3d 是 3D 坐标，需要使用最近邻插值以保持几何一致性。
        
        数据结构说明：
        - 输入 pts3d: (1, H_infer, W_infer, 3) - 推理尺寸下的密集 3D 点云
        - 每个像素 (u, v) 对应一个 3D 点 (x, y, z)
        - 输出: (H_orig, W_orig, 3) - 原图尺寸下的密集 3D 点云
        
        Args:
            outputs_list: 推理输出列表，每个元素包含 'pts3d' key
            global_indices: 每个输出对应的全局图像索引
            verbose: 是否打印详细信息
            
        Returns:
            batch_pts3d_maps: {global_idx: (H_orig, W_orig, 3) numpy array}
                             已缩放到原图尺寸的逐像素 3D 点云字典
        """
        batch_pts3d_maps = {}
        
        for i, output in enumerate(outputs_list):
            if i >= len(global_indices):
                break
            
            global_idx = global_indices[i]
            
            if 'pts3d' not in output:
                continue
            
            # 提取 pts3d: (1, H_infer, W_infer, 3) -> (H_infer, W_infer, 3)
            pts3d = output['pts3d']
            if hasattr(pts3d, 'cpu'):
                pts3d = pts3d.cpu().numpy()
            if pts3d.ndim == 4:
                pts3d = pts3d[0]  # (H_infer, W_infer, 3)
            
            infer_h, infer_w, _ = pts3d.shape
            
            # 获取原图尺寸
            if global_idx < len(self.scale_info):
                s_info = self.scale_info[global_idx]
                orig_w, orig_h = s_info['original_size']
                
                # 如果尺寸不同，进行最近邻重采样
                if infer_w != orig_w or infer_h != orig_h:
                    # 使用最近邻插值（保持 3D 坐标的精确性）
                    # cv2.resize 对多通道支持良好
                    pts3d_resized = cv2.resize(
                        pts3d, 
                        (orig_w, orig_h), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    pts3d_resized = pts3d
            else:
                pts3d_resized = pts3d
            
            batch_pts3d_maps[global_idx] = pts3d_resized
        
        if verbose and self.verbose:
            print(f"  ✓ Extracted {len(batch_pts3d_maps)} dense pts3d maps to original resolution")
        
        return batch_pts3d_maps

    def _extract_pixel_to_3d_mapping_for_batch(
        self,
        outputs_list: List[Dict],
        global_indices: List[int],
        conf_threshold: float = 1.0,
        verbose: bool = True
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        建立完整的逐像素到 3D 点的映射关系
        
        这个方法同时提取 pts3d 和 conf，并建立原图尺寸下的完整映射。
        比 reconstruction 中的稀疏匹配更密集，可用于：
        - 密集点云融合
        - 语义标签投影
        - 深度图生成
        
        Args:
            outputs_list: 推理输出列表，包含 'pts3d' 和 'conf'
            global_indices: 每个输出对应的全局图像索引
            conf_threshold: 置信度阈值，低于此值的点可以标记为无效
            verbose: 是否打印详细信息
            
        Returns:
            pixel_3d_mapping: {
                global_idx: {
                    'pts3d': (H_orig, W_orig, 3) numpy array,  # 3D 坐标
                    'conf': (H_orig, W_orig) numpy array,       # 置信度
                    'valid_mask': (H_orig, W_orig) bool array,  # 有效性掩码
                    'infer_size': (H_infer, W_infer),          # 推理尺寸
                    'orig_size': (H_orig, W_orig),             # 原图尺寸
                }
            }
        """
        pixel_3d_mapping = {}
        total_valid_pixels = 0
        n = min(len(outputs_list), len(global_indices))

        if n == 0:
            return pixel_3d_mapping

        scale_info = self.scale_info
        n_scale = len(scale_info)

        for i in range(n):
            output = outputs_list[i]
            global_idx = global_indices[i]

            pts3d_raw = output.get('pts3d')
            if pts3d_raw is None:
                continue

            conf_raw = output.get('conf')

            # GPU→CPU 转移：先在 GPU 上 squeeze batch 维度再传输，减少传输数据量
            # .detach() 防止 autograd 图保留
            if isinstance(pts3d_raw, torch.Tensor):
                pts3d_t = pts3d_raw.detach()
                if pts3d_t.ndim == 4:
                    pts3d_t = pts3d_t[0]  # GPU 上切片（view，无拷贝）

                if conf_raw is not None:
                    conf_t = conf_raw.detach()
                    if conf_t.ndim == 3:
                        conf_t = conf_t[0]
                    # pts3d (H,W,3) 和 conf (H,W) 拼接为 (H,W,4)，一次 CPU 传输
                    combined = torch.cat([pts3d_t, conf_t.unsqueeze(-1)], dim=-1)  # (H, W, 4)
                    combined_np = combined.cpu().numpy()
                    pts3d_np = combined_np[..., :3]
                    conf_np = combined_np[..., 3]
                else:
                    pts3d_np = pts3d_t.cpu().numpy()
                    conf_np = None
            else:
                pts3d_np = pts3d_raw[0] if pts3d_raw.ndim == 4 else pts3d_raw
                if conf_raw is not None:
                    conf_np = conf_raw[0] if conf_raw.ndim == 3 else conf_raw
                else:
                    conf_np = None

            infer_h, infer_w = pts3d_np.shape[:2]

            # 确定原图尺寸和是否需要 resize
            if global_idx < n_scale:
                orig_w, orig_h = scale_info[global_idx]['original_size']
            else:
                orig_w, orig_h = infer_w, infer_h

            if infer_w != orig_w or infer_h != orig_h:
                pts3d_out = cv2.resize(pts3d_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                conf_out = cv2.resize(conf_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) if conf_np is not None else None
            else:
                pts3d_out = pts3d_np
                conf_out = conf_np

            # 合并有效性检查：NaN 经 abs→max 后仍为 NaN，NaN < 1e6 为 False；
            # Inf 经 abs→max 后仍为 Inf，Inf < 1e6 为 False。
            # 因此 abs().max() < 1e6 已隐含 isfinite 检查，无需单独的 np.isfinite 调用
            coord_valid = np.abs(pts3d_out).max(axis=-1) < 1e6

            if conf_out is not None:
                valid_mask = (conf_out >= conf_threshold) & coord_valid
            else:
                valid_mask = coord_valid
                conf_out = np.ones((orig_h, orig_w), dtype=np.float32)

            pixel_3d_mapping[global_idx] = {
                'pts3d': pts3d_out,
                'conf': conf_out,
                'valid_mask': valid_mask,
                'infer_size': (infer_h, infer_w),
                'orig_size': (orig_h, orig_w),
            }

            total_valid_pixels += np.count_nonzero(valid_mask)

        if verbose and self.verbose:
            print(f"  ✓ Built pixel-to-3D mapping for {len(pixel_3d_mapping)} images, "
                  f"{total_valid_pixels:,} valid pixels total")

        return pixel_3d_mapping

    def _align_current_reconstruction_by_point_cloud(
        self,
        reconstruction: pycolmap.Reconstruction,
        match_type: str = 'use_bidirectional',
    ) -> bool:
        """
        Align current reconstruction to previous reconstruction using point cloud correspondences.
        
        Args:
            reconstruction: Current reconstruction to be aligned
            match_type: 匹配类型，支持：
                        - 'use_bidirectional': 双向匹配（更准确，较慢）
                        - 'use_unidirectional': 单向匹配（更快）
                        - 'use_single_images_pair': 只使用一对重叠影像进行匹配（默认使用中间那一对）
            
        Returns:
            对齐后的 reconstruction；若失败或不存在前一重建则返回 False 或原 reconstruction
        """
        if len(self.inference_reconstructions) < 1:
            return reconstruction
        
        # 校验参数
        valid_match_types = {'use_bidirectional', 'use_unidirectional', 'use_single_images_pair'}
        if match_type not in valid_match_types:
            raise ValueError(f"match_type 必须是 {valid_match_types} 之一，当前为: {match_type}")

        # 获取前一个reconstruction（因为当前的还没添加，所以前一个是最后一个）
        prev_recon_data = self.inference_reconstructions[-1]
        prev_recon = prev_recon_data['reconstruction']
        curr_recon = reconstruction  # 当前的就是传入的参数

        # 1. 获取重叠区域的影像ID，Reconstruction中影像ID是从1开始的。
        prev_overlap_image_ids = list(range(
            len(prev_recon.images) - self.overlap + 1,
            len(prev_recon.images) + 1
        ))
        curr_overlap_image_ids = list(range(1, self.overlap + 1))

        # 2. 建立3D点对应关系
        point_correspondences = []  # [(prev_point3D_id, curr_point3D_id, dist)]
        pixel_threshold = 1.0  # 0.5像素阈值

        # 如果是仅使用单一影像对，则只选择一对（默认使用中间对）
        if match_type == 'use_single_images_pair':
            pairs = list(zip(prev_overlap_image_ids, curr_overlap_image_ids))
            if not pairs:
                if self.verbose:
                    print("  Warning: No overlapping image pairs")
                return False

            # 选择一对影像：可切换为 'first' 或 'last'
            selection_mode = 'first'
            if selection_mode == 'first':
                selected_pair = pairs[0]
            elif selection_mode == 'last':
                selected_pair = pairs[-1]
            else:  # 'middle'
                selected_pair = pairs[len(pairs) // 2]

            prev_image_id, curr_image_id = selected_pair

            prev_image_object = prev_recon.images[prev_image_id]
            curr_image_object = curr_recon.images[curr_image_id]

            # 为curr图像建立空间索引
            curr_spatial_index = {}
            for point2D in curr_image_object.points2D:
                if point2D.point3D_id != -1:
                    grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
                    if grid_key not in curr_spatial_index:
                        curr_spatial_index[grid_key] = []
                    # 保存浮点坐标便于距离计算
                    curr_spatial_index[grid_key].append(
                        (int(point2D.point3D_id), np.asarray(point2D.xy, dtype=np.float64))
                    )

            correspondences = find_single_images_pair_matches(
                prev_image_object, 
                curr_image_object, 
                curr_spatial_index, 
                pixel_threshold, 
            )
            point_correspondences.extend(correspondences)

        else:
            for prev_image_id, curr_image_id in zip(prev_overlap_image_ids, curr_overlap_image_ids):
                if prev_image_id not in prev_recon.images or curr_image_id not in curr_recon.images:
                    continue
                
                prev_image = prev_recon.images[prev_image_id]
                curr_image = curr_recon.images[curr_image_id]
                
                # 为curr图像建立空间索引
                curr_spatial_index = {}
                for point2D in curr_image.points2D:
                    if point2D.point3D_id != -1:
                        grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
                        if grid_key not in curr_spatial_index:
                            curr_spatial_index[grid_key] = []
                        curr_spatial_index[grid_key].append((point2D.point3D_id, point2D.xy))
                
                # 根据 match_type 选择匹配策略
                if match_type == 'use_bidirectional':
                    correspondences = self._find_bidirectional_matches(
                        prev_image, curr_image, curr_spatial_index, pixel_threshold
                    )
                elif match_type == 'use_unidirectional':
                    correspondences = self._find_unidirectional_matches(
                        prev_image, curr_image, curr_spatial_index, pixel_threshold
                    )
                    
                point_correspondences.extend(correspondences)

        if len(point_correspondences) == 0:
            print("  Warning: No point correspondences found between overlapping regions")
            return False

        if self.verbose:
            print(f"    Found {len(point_correspondences)} point correspondences in overlap region ({match_type})")

        # 3. 根据对应关系计算变换
        prev_pts3d = []
        curr_pts3d = []
        for prev_pt3d_id, curr_pt3d_id, _ in point_correspondences:
            if prev_pt3d_id in prev_recon.points3D and curr_pt3d_id in curr_recon.points3D:
                prev_pts3d.append(prev_recon.points3D[prev_pt3d_id].xyz)
                curr_pts3d.append(curr_recon.points3D[curr_pt3d_id].xyz)
        
        if len(prev_pts3d) < 3:
            print(f"  Warning: Not enough point correspondences ({len(prev_pts3d)}) for alignment")
            return False

        prev_pts3d = np.array(prev_pts3d)
        curr_pts3d = np.array(curr_pts3d)

        # 使用Umeyama算法计算Sim3变换
        sim3_transform = estimate_sim3_transform(curr_pts3d, prev_pts3d)
        if sim3_transform is None:
            if self.verbose:
                print("  Warning: Failed to estimate Sim3 transform")
            return False
            
        # 应用变换到curr_recon
        # curr_recon_aligned = pycolmap.Reconstruction(curr_recon)
        curr_recon_aligned = copy.deepcopy(curr_recon)
        curr_recon_aligned.transform(sim3_transform)

        return curr_recon_aligned

    def _find_unidirectional_matches(
        self,
        prev_image,
        curr_image,
        curr_spatial_index: dict,
        pixel_threshold: float,
    ) -> list:
        """
        单向配对：从prev到curr，选择最近的匹配点
        
        Args:
            prev_image: 前一个重建的影像
            curr_image: 当前重建的影像
            curr_spatial_index: 当前影像的空间索引
            pixel_threshold: 像素阈值
            
        Returns:
            点对应关系列表 [(prev_point3D_id, curr_point3D_id, dist)]
        """
        correspondences = []
        
        for point2D in prev_image.points2D:
            if point2D.point3D_id != -1:
                prev_xy = point2D.xy
                center_grid = (int(round(prev_xy[0])), int(round(prev_xy[1])))
                
                # 收集所有候选匹配点
                candidates = []
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        search_grid = (center_grid[0] + dx, center_grid[1] + dy)
                        if search_grid in curr_spatial_index:
                            for curr_pt3d_id, curr_xy in curr_spatial_index[search_grid]:
                                dist = np.linalg.norm(prev_xy - curr_xy)
                                if dist < pixel_threshold:
                                    candidates.append((curr_pt3d_id, dist))
                
                # 选择距离最近的匹配
                if len(candidates) > 0:
                    best_match = min(candidates, key=lambda x: x[1])
                    curr_pt3d_id, dist = best_match
                    correspondences.append((point2D.point3D_id, curr_pt3d_id, dist))
        
        return correspondences

    def _find_bidirectional_matches(
        self,
        prev_image,
        curr_image,
        curr_spatial_index: dict,
        pixel_threshold: float,
    ) -> list:
        """
        双向配对：prev→curr 和 curr→prev，只保留互相匹配的点
        
        Args:
            prev_image: 前一个重建的影像
            curr_image: 当前重建的影像
            curr_spatial_index: 当前影像的空间索引
            pixel_threshold: 像素阈值
            
        Returns:
            点对应关系列表 [(prev_point3D_id, curr_point3D_id, dist)]
        """
        # 第一步：prev → curr 的匹配
        prev_to_curr = {}  # {prev_point3D_id: (curr_point3D_id, dist)}
        
        for point2D in prev_image.points2D:
            if point2D.point3D_id != -1:
                prev_xy = point2D.xy
                center_grid = (int(round(prev_xy[0])), int(round(prev_xy[1])))
                
                candidates = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        search_grid = (center_grid[0] + dx, center_grid[1] + dy)
                        if search_grid in curr_spatial_index:
                            for curr_pt3d_id, curr_xy in curr_spatial_index[search_grid]:
                                dist = np.linalg.norm(prev_xy - curr_xy)
                                if dist < pixel_threshold:
                                    candidates.append((curr_pt3d_id, dist))
                
                if len(candidates) > 0:
                    best_match = min(candidates, key=lambda x: x[1])
                    prev_to_curr[point2D.point3D_id] = best_match
        
        # 第二步：建立prev图像的空间索引（用于反向匹配）
        prev_spatial_index = {}
        for point2D in prev_image.points2D:
            if point2D.point3D_id != -1:
                grid_key = (int(round(point2D.xy[0])), int(round(point2D.xy[1])))
                if grid_key not in prev_spatial_index:
                    prev_spatial_index[grid_key] = []
                prev_spatial_index[grid_key].append((point2D.point3D_id, point2D.xy))
        
        # 第三步：curr → prev 的匹配
        curr_to_prev = {}  # {curr_point3D_id: (prev_point3D_id, dist)}
        
        for point2D in curr_image.points2D:
            if point2D.point3D_id != -1:
                curr_xy = point2D.xy
                center_grid = (int(round(curr_xy[0])), int(round(curr_xy[1])))
                
                candidates = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        search_grid = (center_grid[0] + dx, center_grid[1] + dy)
                        if search_grid in prev_spatial_index:
                            for prev_pt3d_id, prev_xy in prev_spatial_index[search_grid]:
                                dist = np.linalg.norm(curr_xy - prev_xy)
                                if dist < pixel_threshold:
                                    candidates.append((prev_pt3d_id, dist))
                
                if len(candidates) > 0:
                    best_match = min(candidates, key=lambda x: x[1])
                    curr_to_prev[point2D.point3D_id] = best_match
        
        # 第四步：只保留互相匹配的点对
        correspondences = []
        for prev_pt3d_id, (curr_pt3d_id, dist_forward) in prev_to_curr.items():
            # 检查反向匹配是否存在且一致
            if curr_pt3d_id in curr_to_prev:
                matched_prev_id, dist_backward = curr_to_prev[curr_pt3d_id]
                if matched_prev_id == prev_pt3d_id:
                    # 互相匹配，取平均距离
                    avg_dist = (dist_forward + dist_backward) / 2.0
                    correspondences.append((prev_pt3d_id, curr_pt3d_id, avg_dist))
        
        return correspondences

    def _merge_reconstruction_intermediate_results(self) -> bool:
        """
        合并reconstruction中间结果
        
        将最新的reconstruction与之前已合并的reconstruction合并，
        通过重叠影像进行对齐。合并结果存储在 self.merged_reconstruction 中。
        
        支持四种合并方式 (由 self.merge_method 控制):
        - 'full': 使用 merge_full_pipeline.py 完整流程
        - 'confidence': 使用 merge_confidence.py 简单的置信度选择合并
        - 'confidence_blend': 使用 merge_confidence_blend.py 基于置信度选择 + 重叠区边缘平滑插值过渡
        - 'points_only': 使用 merge_points_only.py 仅合并点云（不维护 Reconstruction 结构）
        
        Returns:
            True if successful, False otherwise
        """
        tracker = get_timing_tracker()
        if tracker:
            tracker.start("merge_reconstruction")
        
        # ========== 检查是否有 reconstruction 可合并 ==========
        if len(self.inference_reconstructions) == 0:
            if self.verbose:
                print("  [Skip] No reconstruction to merge yet")
            if tracker:
                tracker.end("merge_reconstruction")
            return True  # 返回 True 表示正常，只是还没有数据
        
        # ========== points_only 模式：特殊处理 ==========
        # 每次新增 batch 时，将所有已有的 reconstruction 一起合并
        if self.merge_method == 'points_only':
            result = self._merge_by_points_only()
            tracker = get_timing_tracker()
            if tracker:
                tracker.end("merge_reconstruction")
            return result
        
        # ========== 其他模式：两两递增合并 ==========
        # 如果这是第一个reconstruction，直接设置为merged
        if len(self.inference_reconstructions) == 1:
            self.merged_reconstruction = self.inference_reconstructions[0]['reconstruction']
            merged_recon = self.merged_reconstruction
            # 提取 merged 点云用于可视化
            if self.visualizer is not None:
                self.visualizer.update_merged_point_cloud(merged_recon)
            # 保存merged_reconstruction
            temp_path = self.output_dir / "temp_merged" / f"merged_{len(self.inference_reconstructions)}"
            self.merged_reconstruction_path = str(temp_path)
            temp_path.mkdir(parents=True, exist_ok=True)
            merged_recon.write_text(str(temp_path))
            merged_recon.export_PLY(str(temp_path / "points3D.ply"))
            export_reconstruction_to_las(merged_recon, str(temp_path / "points3D.las"), verbose=self.verbose)
            tracker = get_timing_tracker()
            if tracker:
                tracker.end("merge_reconstruction")
            return True

        # 获取当前reconstruction信息
        curr_recon_data = self.inference_reconstructions[-1]
        
        # 根据 merge_method 选择合并方式
        if self.merge_method == 'confidence_blend':
            # 使用基于置信度 + 边缘平滑插值的合并方式
            merged_recon = self._merge_by_confidence_blend(
                self.merged_reconstruction,
                curr_recon_data['reconstruction'],
                self.output_dir
            )
        elif self.merge_method == 'confidence':
            # 使用简单的置信度选择合并方式
            merged_recon = self._merge_by_confidence(
                self.merged_reconstruction,
                curr_recon_data['reconstruction'],
                self.output_dir
            )
        else:
            # 使用完整的 merge_full_pipeline 流程 (merge_method == 'full')
            merged_recon = self._merge_by_full_pipeline(
                self.merged_reconstruction,
                curr_recon_data['reconstruction'],
                self.output_dir
            )
        
        # 检查合并结果
        if merged_recon is None:
            if self.verbose:
                print(f"  ✗ 合并失败")
            tracker = get_timing_tracker()
            if tracker:
                tracker.end("merge_reconstruction")
            return False
        
        # 更新merged_reconstruction
        self.merged_reconstruction = merged_recon
        
        # 提取 merged 点云用于可视化
        if self.visualizer is not None:
            self.visualizer.update_merged_point_cloud(merged_recon)
        
        # 保存merged_reconstruction
        temp_path = self.output_dir / "temp_merged" / f"merged_{len(self.inference_reconstructions)}"
        self.merged_reconstruction_path = str(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        merged_recon.write_text(str(temp_path))
        merged_recon.export_PLY(str(temp_path / "points3D.ply"))
        export_reconstruction_to_las(merged_recon, str(temp_path / "points3D.las"), verbose=self.verbose)
        
        if self.verbose:
            print(f"  ✓ 合并完成:")
            print(f"    总影像数: {len(merged_recon.images)}")
            print(f"    总3D点数: {len(merged_recon.points3D)}")
            print(f"    结果保存到: {temp_path}")

        # 清理不再需要的中间数据以释放内存
        self._cleanup_intermediate_data()

        tracker = get_timing_tracker()
        if tracker:
            tracker.end("merge_reconstruction")
        return True
    
    def _merge_by_confidence_blend(
        self,
        prev_recon: pycolmap.Reconstruction,
        curr_recon: pycolmap.Reconstruction,
        output_dir: Path
    ) -> Optional[pycolmap.Reconstruction]:
        """
        使用基于置信度 + 边缘平滑插值的方式合并两个 reconstruction
        
        该方法结合了多种技术实现高质量的重建合并：
        
        1. 基于置信度的点选择：
           - 使用神经网络输出的 conf 作为置信度依据
           - 对于相同像素位置，选择置信度更高的 3D 点
           - 非重叠区域的点全部保留
        
        2. 重叠区边缘平滑插值过渡：
           - 使用多级 2D 匹配半径进行渐进式匹配
           - 融合带内使用加权平均混合 3D 坐标
           - 空间插值使用 smoothstep 实现平滑过渡
           - 避免合并边界处出现明显的不连续
        
        3. 密度均衡化：
           - 使非重叠区点云密度与重叠区一致
           - 支持网格采样和距离衰减
        
        置信度存储结构：
        - self.inference_reconstructions[i]['pixel_3d_mapping']: 每个批次的逐像素3D对应
          {global_idx: {'pts3d': (H,W,3), 'conf': (H,W), 'valid_mask': (H,W)}}
        - conf 已经缩放到原图尺寸，与 reconstruction 中的 2D 点坐标一致
        
        Args:
            prev_recon: 之前已合并的 reconstruction
            curr_recon: 当前要合并的 reconstruction
            output_dir: 输出目录
            
        Returns:
            合并后的 reconstruction，失败返回 None
        """
        # 从 pixel_3d_mapping 中提取置信度图
        # prev_recon_conf: 之前所有批次的 conf（batches 0 to N-2）
        # curr_recon_conf: 当前批次的 conf（batch N-1）
        prev_recon_conf: Dict[int, np.ndarray] = {}
        curr_recon_conf: Dict[int, np.ndarray] = {}
        
        num_batches = len(self.inference_reconstructions)
        
        # 从 pixel_3d_mapping 提取 conf 的辅助函数
        def extract_conf_from_pixel_3d_mapping(pixel_3d_mapping: Dict) -> Dict[int, np.ndarray]:
            """从 pixel_3d_mapping 提取 conf 信息，转换为 {global_idx: (H, W) array} 格式"""
            conf_maps = {}
            for global_idx, data in pixel_3d_mapping.items():
                if 'conf' in data:
                    conf_maps[global_idx] = data['conf']
            return conf_maps
        
        # 前 N-1 个批次的 conf 属于 prev_recon
        for i in range(num_batches - 1):
            pixel_3d_mapping = self.inference_reconstructions[i].get('pixel_3d_mapping', {})
            batch_conf_maps = extract_conf_from_pixel_3d_mapping(pixel_3d_mapping)
            prev_recon_conf.update(batch_conf_maps)
        
        # 最后一个批次的 conf 属于 curr_recon
        if num_batches > 0:
            pixel_3d_mapping = self.inference_reconstructions[-1].get('pixel_3d_mapping', {})
            curr_recon_conf = extract_conf_from_pixel_3d_mapping(pixel_3d_mapping)
        
        if self.verbose:
            print(f"\n=== 使用 confidence_blend 合并 (置信度选择 + 边缘平滑插值) ===")
            print(f"    prev_recon: {len(prev_recon.images)} images, {len(prev_recon.points3D)} 3D points")
            print(f"    curr_recon: {len(curr_recon.images)} images, {len(curr_recon.points3D)} 3D points")
            print(f"    overlap: {self.overlap} images")
            print(f"    num_batches: {num_batches}")
            
            # 显示每个批次的 pixel_3d_mapping 信息
            for i, recon_data in enumerate(self.inference_reconstructions):
                pixel_3d_mapping = recon_data.get('pixel_3d_mapping', {})
                batch_keys = sorted(pixel_3d_mapping.keys()) if pixel_3d_mapping else []
                print(f"    batch {i} [{recon_data['start_idx']}-{recon_data['end_idx']}]: {len(pixel_3d_mapping)} pixel_3d_mapping, keys={batch_keys}")
            
            # 显示分离后的结果
            prev_keys = sorted(prev_recon_conf.keys())
            curr_keys = sorted(curr_recon_conf.keys())
            print(f"    prev_recon_conf: {len(prev_recon_conf)} images, keys={prev_keys}")
            print(f"    curr_recon_conf: {len(curr_recon_conf)} images, keys={curr_keys}")
        
        # 构建 image_name -> global_idx 的映射（用于像素级置信度查询）
        image_name_to_idx = {ext['image_name']: idx for idx, ext in enumerate(self.ori_extrinsic)}

        # 获取当前批次的 start_idx 和 end_idx（用于输出目录命名）
        if num_batches > 0:
            start_idx = self.inference_reconstructions[-1].get('start_idx')
            end_idx = self.inference_reconstructions[-1].get('end_idx')
        else:
            start_idx = None
            end_idx = None
        
        # 调用 reconstruction_merger 中的合并函数
        # 分别传入 prev_recon 和 curr_recon 的置信度图
        # 使用多级半径匹配 + 加权平均模式 + 3D匹配补充 + 激进3D匹配 + 边缘位置平滑 + 融合带空间插值
        merged_recon, info = merge_by_confidence_blend(
            prev_recon,
            curr_recon,
            inlier_threshold=10,
            min_inliers=5,
            min_sample_size=5,
            ransac_iterations=1000,
            prev_recon_conf=prev_recon_conf,
            curr_recon_conf=curr_recon_conf,
            image_name_to_idx=image_name_to_idx,
            output_dir=output_dir,
            start_idx=start_idx,
            end_idx=end_idx,
            color_by_source=False, 
            match_radii=[1, 2, 3, 5, 8, 10, 20, 30, 40, 50],  # 多级2D匹配半径（从小到大依次匹配）
            match_3d_threshold=10.0,  # 3D空间匹配阈值（单位与点云坐标一致）
            aggressive_3d_threshold=3.0,  # 最终阶段激进3D匹配阈值，用于减少重叠区独有点
            inner_blend_margin=150.0,  # 融合带向内延伸（像素），同时控制空间插值范围
            outer_blend_margin=200.0,  # 融合带向外延伸（像素），同时控制空间插值范围
            blend_mode='weighted',   # 加权平均模式，基于置信度计算加权位置
            keep_unmatched_overlap=False,  # 保留重叠区未匹配点
            spatial_blend_interpolation=True,  # 启用融合带3D坐标空间插值
            spatial_blend_k_neighbors=32,  # 空间插值使用的近邻数
            spatial_blend_smooth_transition=True,  # 使用 smoothstep 实现更平滑过渡
            spatial_blend_smooth_power=0.7,  # 平滑力度：<1更强效果（建议0.3-0.7）
            density_equalization=True,  # 启用密度均衡化，使非重叠区密度与重叠区一致
            density_k_neighbors=10,  # 密度计算使用的近邻数
            density_target_percentile=50.0,  # 使用重叠区点间距中位数作为目标
            density_tolerance=1.2,  # 密度容差倍数
            density_use_grid=True,  # 使用网格采样（更稳定均匀）
            density_grid_resolution=1.0,  # 网格分辨率因子
            density_distance_decay=0.5,  # 距离衰减因子（远离重叠区的点保留更多）
            voxel_size=self.merge_voxel_size,  # 体素降采样大小
            verbose=self.verbose,
        )
        
        return merged_recon
    
    def _merge_by_confidence(
        self,
        prev_recon: pycolmap.Reconstruction,
        curr_recon: pycolmap.Reconstruction,
        output_dir: Path
    ) -> Optional[pycolmap.Reconstruction]:
        """
        使用简单的置信度选择方式合并两个 reconstruction
        
        该方法基于置信度进行点选择：
        - 对于相同像素位置，选择置信度更高的 3D 点
        - 非重叠区域的点全部保留
        
        与 confidence_blend 相比，此方法更简单直接，不包含边缘平滑插值等高级功能。
        
        Args:
            prev_recon: 之前已合并的 reconstruction
            curr_recon: 当前要合并的 reconstruction
            output_dir: 输出目录
            
        Returns:
            合并后的 reconstruction，失败返回 None
        """
        # 从 pixel_3d_mapping 中提取置信度图
        prev_recon_conf: Dict[int, np.ndarray] = {}
        curr_recon_conf: Dict[int, np.ndarray] = {}
        
        num_batches = len(self.inference_reconstructions)
        
        def extract_conf_from_pixel_3d_mapping(pixel_3d_mapping: Dict) -> Dict[int, np.ndarray]:
            """从 pixel_3d_mapping 提取 conf 信息"""
            conf_maps = {}
            for global_idx, data in pixel_3d_mapping.items():
                if 'conf' in data:
                    conf_maps[global_idx] = data['conf']
            return conf_maps
        
        # 前 N-1 个批次的 conf 属于 prev_recon
        for i in range(num_batches - 1):
            pixel_3d_mapping = self.inference_reconstructions[i].get('pixel_3d_mapping', {})
            batch_conf_maps = extract_conf_from_pixel_3d_mapping(pixel_3d_mapping)
            prev_recon_conf.update(batch_conf_maps)
        
        # 最后一个批次的 conf 属于 curr_recon
        if num_batches > 0:
            pixel_3d_mapping = self.inference_reconstructions[-1].get('pixel_3d_mapping', {})
            curr_recon_conf = extract_conf_from_pixel_3d_mapping(pixel_3d_mapping)
        
        if self.verbose:
            print(f"\n=== 使用 confidence 合并 (简单置信度选择) ===")
            print(f"    prev_recon: {len(prev_recon.images)} images, {len(prev_recon.points3D)} 3D points")
            print(f"    curr_recon: {len(curr_recon.images)} images, {len(curr_recon.points3D)} 3D points")
            print(f"    overlap: {self.overlap} images")
        
        # 构建 image_name -> global_idx 的映射
        image_name_to_idx = {ext['image_name']: idx for idx, ext in enumerate(self.ori_extrinsic)}
        
        # 获取当前批次的 start_idx 和 end_idx
        if num_batches > 0:
            start_idx = self.inference_reconstructions[-1].get('start_idx')
            end_idx = self.inference_reconstructions[-1].get('end_idx')
        else:
            start_idx = None
            end_idx = None
        
        # 调用简单置信度合并函数
        merged_recon, info = merge_by_confidence(
            prev_recon,
            curr_recon,
            inlier_threshold=10,
            min_inliers=5,
            min_sample_size=5,
            ransac_iterations=1000,
            prev_recon_conf=prev_recon_conf,
            curr_recon_conf=curr_recon_conf,
            image_name_to_idx=image_name_to_idx,
            output_dir=output_dir,
            start_idx=start_idx,
            end_idx=end_idx,
            match_radii=[1, 2, 3, 5, 8, 10, 20, 30, 40, 50, 60],
            k_neighbors=15,  # 查询多个近邻以提高匹配率
            color_by_match_status=False, # 设为 True 可启用调试着色
            blend_mode='weighted',
            blend_weight=0.7,
            rotation_mode='full',
            verbose=self.verbose,
        )
        
        return merged_recon
    
    def _merge_by_full_pipeline(
        self,
        prev_recon: pycolmap.Reconstruction,
        curr_recon: pycolmap.Reconstruction,
        output_dir: Path
    ) -> Optional[pycolmap.Reconstruction]:
        """
        使用完整的 merge_construction 流程合并
        
        Args:
            prev_recon: 之前已合并的 reconstruction
            curr_recon: 当前要合并的 reconstruction
            output_dir: 输出目录
            
        Returns:
            合并后的 reconstruction，失败返回 None
        """
        # 临时保存两个reconstruction到磁盘
        temp_base = self.output_dir / "temp_merge_input"
        temp_base.mkdir(parents=True, exist_ok=True)
        
        # 保存prev_recon (已合并的)
        prev_dir = temp_base / "prev"
        prev_dir.mkdir(parents=True, exist_ok=True)
        prev_recon.write_text(str(prev_dir))
        
        # 保存curr_recon (当前的)
        curr_dir = temp_base / "curr"
        curr_dir.mkdir(parents=True, exist_ok=True)
        curr_recon.write_text(str(curr_dir))
        
        if self.verbose:
            print(f"\n=== 使用 merge_reconstructions 进行合并 ===")
            print(f"  prev_dir: {prev_dir}")
            print(f"  curr_dir: {curr_dir}")
            print(f"  output_dir: {output_dir}")
        
        # 调用 merge_reconstructions 函数
        merged_recon = merge_reconstructions(
            model_dir1=str(prev_dir),
            model_dir2=str(curr_dir),
            output_dir=str(output_dir),
            overlap_count=self.overlap,
            translation_only=True,  # 初始对齐只做平移
            use_ransac=False,
            # 点云融合参数
            point_fusion=True,
            fusion_method="2d_matching",
            cell_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 80, 160, 320, 640, 1280],
            keep_unmatched_overlap=True,  # 保留重叠区未匹配点
            spatial_dedup_threshold=0.1,
            # 精化对齐参数
            refine_alignment=True,
            refine_cell_range=(1, 3),
            refine_stages=[
                (None, "translation"),       # 第1阶段：不筛选距离，只平移
                (10.0, "scale_translation"), # 第2阶段：dist<=10m
                (5.0, "scale_translation"),  # 第3阶段：dist<=5m
                (2.0, "scale_translation"),  # 第4阶段：dist<=2m
                (1.0, "scale_translation"),  # 第5阶段：dist<=1m
                (0.5, "scale_translation"),  # 第6阶段：dist<=0.5m
            ],
            voxel_size=self.merge_voxel_size,
            statistical_filter=self.merge_statistical_filter,
            min_track_length=2,
            boundary_filter=self.merge_boundary_filter,
            filter_edge_margin=self.filter_edge_margin,
            verbose=self.verbose
        )
        
        return merged_recon  

    def _extract_points_from_reconstruction(
        self, 
        recon: pycolmap.Reconstruction
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 reconstruction 中提取点云坐标和颜色
        
        Args:
            recon: pycolmap.Reconstruction 对象
            
        Returns:
            xyz: (N, 3) 点云坐标
            colors: (N, 3) RGB 颜色 (uint8)
        """
        n = len(recon.points3D)
        if n == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        xyz = np.empty((n, 3), dtype=np.float32)
        colors = np.empty((n, 3), dtype=np.uint8)
        for i, pt3d in enumerate(recon.points3D.values()):
            xyz[i] = pt3d.xyz
            colors[i] = pt3d.color
        return xyz, colors

    def _append_recon_points(self, recon: pycolmap.Reconstruction):
        """将 reconstruction 的所有点追加到 merged_points（回退路径的公共逻辑）"""
        curr_xyz, curr_colors = self._extract_points_from_reconstruction(recon)
        if len(curr_xyz) > 0:
            self.merged_points_xyz = np.concatenate([self.merged_points_xyz, curr_xyz])
            self.merged_points_colors = np.concatenate([self.merged_points_colors, curr_colors])

    def _merge_by_points_only(self) -> bool:
        """
        增量式点云合并（points_only 模式）
        
        实现增量式合并流程：
        1. 第一个 batch：直接提取点云作为基础，保存 reconstruction 用于后续对齐
        2. 后续 batch：
           a. 与前一个已对齐的 reconstruction 对齐（通过公共影像估计 Sim3 变换）
           b. 将变换后的点云合并到已有的 merged_points
           c. 更新 _prev_aligned_recon 用于下一次对齐
        
        性能优化：
        - 消除 copy.deepcopy，改用原地 Sim3 变换 + 引用传递
        - 向量化重叠点过滤（np.isin 替代 Python 循环）
        - PLY/LAS 并行写入
        - 公共回退逻辑提取为 _append_recon_points
        
        Returns:
            True if successful, False otherwise
        """
        if len(self.inference_reconstructions) == 0:
            if self.verbose:
                print("  ✗ 没有 reconstruction 可合并")
            return False
        
        curr_recon_data = self.inference_reconstructions[-1]
        curr_recon = curr_recon_data['reconstruction']
        
        num_batches = len(self.inference_reconstructions)
        output_dir = self.output_dir / "temp_merged_points_only"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        is_first_batch = (self.merged_points_xyz is None or self._prev_aligned_recon is None)
        
        # ==================== 第一个 batch：直接提取点云 ====================
        if is_first_batch:
            if self.verbose:
                print(f"\n=== points_only 模式: 第一个 batch (batch {num_batches-1}) ===")
                print(f"    [{curr_recon_data['start_idx']}-{curr_recon_data['end_idx']}]: "
                      f"{len(curr_recon.images)} images, {len(curr_recon.points3D)} 3D points")
            
            self.merged_points_xyz, self.merged_points_colors = self._extract_points_from_reconstruction(curr_recon)
            # 直接引用，无需 deepcopy（第一个 batch 无变换，cleanup 仅将旧 recon 置 None 不影响此引用）
            self._prev_aligned_recon = curr_recon
        
        # ==================== 后续 batch：根据 points_merge_mode 选择合并策略 ====================
        else:
            if self.verbose:
                mode_label = 'fast(3D近邻)' if self.points_merge_mode == 'fast' else 'quality(2D匹配+置信度)'
                print(f"\n=== points_only 模式: 增量合并 batch {num_batches-1} [{mode_label}] ===")
                print(f"    当前 batch [{curr_recon_data['start_idx']}-{curr_recon_data['end_idx']}]: "
                      f"{len(curr_recon.images)} images, {len(curr_recon.points3D)} 3D points")
                print(f"    已合并点数: {len(self.merged_points_xyz)}")

            from scipy.spatial import cKDTree

            n_pts = len(curr_recon.points3D)
            all_xyz = np.empty((n_pts, 3), dtype=np.float32)
            all_colors = np.empty((n_pts, 3), dtype=np.uint8)
            for i, (pid, pt3d) in enumerate(curr_recon.points3D.items()):
                all_xyz[i] = pt3d.xyz
                all_colors[i] = pt3d.color

            if self.points_merge_mode == 'fast':
                # ======== fast 模式：纯 3D 近邻，直接向量化混合 ========
                overlap_threshold = self.merge_voxel_size if self.merge_voxel_size > 0 else 0.05
                tree_merged = cKDTree(self.merged_points_xyz)
                dists, nn_idx = tree_merged.query(
                    all_xyz, k=1, distance_upper_bound=overlap_threshold)

                overlap_mask = np.isfinite(dists)
                new_mask = ~overlap_mask
                n_overlap = int(overlap_mask.sum())
                n_new = int(new_mask.sum())

                if n_overlap > 0:
                    ov_nn = nn_idx[overlap_mask]
                    self.merged_points_xyz[ov_nn] = (
                        0.5 * self.merged_points_xyz[ov_nn]
                        + 0.5 * all_xyz[overlap_mask]
                    )
                    self.merged_points_colors[ov_nn] = (
                        (self.merged_points_colors[ov_nn].astype(np.uint16)
                         + all_colors[overlap_mask].astype(np.uint16)) >> 1
                    ).astype(np.uint8)

                if n_new > 0:
                    self.merged_points_xyz = np.concatenate(
                        [self.merged_points_xyz, all_xyz[new_mask]])
                    self.merged_points_colors = np.concatenate(
                        [self.merged_points_colors, all_colors[new_mask]])

                if self.verbose:
                    print(f"    3D近邻: 重叠{n_overlap}点(融合), "
                          f"新增{n_new}点(追加), 合并后总点数: {len(self.merged_points_xyz)}")

            else:
                # ======== quality 模式：2D 像素匹配 + 置信度加权 + 3D 兜底 ========
                curr_pt3d_items = list(curr_recon.points3D.items())
                curr_id_to_idx = {pid: i for i, (pid, _) in enumerate(curr_pt3d_items)}

                prev_recon = self._prev_aligned_recon
                matched_pairs = curr_recon_data.get('overlap_matched_pairs', [])
                matched_pairs = [
                    (p_id, c_id) for p_id, c_id in matched_pairs
                    if p_id in prev_recon.points3D and c_id in curr_recon.points3D
                ]
                n_matched = len(matched_pairs)

                n_updated = 0
                matched_curr_idx_set = set()

                if n_matched > 0:
                    prev_matched_xyz = np.array(
                        [prev_recon.points3D[pid].xyz for pid, _ in matched_pairs],
                        dtype=np.float32
                    )
                    tree_merged = cKDTree(self.merged_points_xyz)
                    m_dists, m_nn = tree_merged.query(prev_matched_xyz, k=1)

                    num_recon = len(self.inference_reconstructions)
                    prev_pixel_3d = {}
                    if num_recon >= 2:
                        prev_pixel_3d = self.inference_reconstructions[-2].get('pixel_3d_mapping', {})
                    curr_pixel_3d = self.inference_reconstructions[-1].get('pixel_3d_mapping', {})
                    img_name_to_global = {
                        ext['image_name']: idx for idx, ext in enumerate(self.ori_extrinsic)
                    }

                    def _pt_max_conf(recon, pt3d_id, p3d_map):
                        if not p3d_map or pt3d_id not in recon.points3D:
                            return 1.0
                        best = 0.0
                        for elem in recon.points3D[pt3d_id].track.elements:
                            if elem.image_id not in recon.images:
                                continue
                            img = recon.images[elem.image_id]
                            gidx = img_name_to_global.get(img.name)
                            if gidx is None or gidx not in p3d_map:
                                continue
                            conf_map = p3d_map[gidx].get('conf')
                            if conf_map is None:
                                continue
                            pt2d = img.points2D[elem.point2D_idx]
                            py, px = int(round(pt2d.xy[1])), int(round(pt2d.xy[0]))
                            if 0 <= py < conf_map.shape[0] and 0 <= px < conf_map.shape[1]:
                                best = max(best, float(conf_map[py, px]))
                        return best if best > 0 else 1.0

                    max_merge_dist = (self.merge_voxel_size * 3) if self.merge_voxel_size > 0 else 1.0

                    upd_merged_idx = []
                    upd_curr_idx = []
                    upd_w_prev = []
                    upd_w_curr = []

                    for k, (prev_pid, curr_pid) in enumerate(matched_pairs):
                        if m_dists[k] > max_merge_dist:
                            continue
                        curr_arr_idx = curr_id_to_idx[curr_pid]
                        matched_curr_idx_set.add(curr_arr_idx)
                        upd_merged_idx.append(m_nn[k])
                        upd_curr_idx.append(curr_arr_idx)
                        upd_w_prev.append(_pt_max_conf(prev_recon, prev_pid, prev_pixel_3d))
                        upd_w_curr.append(_pt_max_conf(curr_recon, curr_pid, curr_pixel_3d))

                    n_updated = len(upd_merged_idx)

                    if n_updated > 0:
                        mi = np.array(upd_merged_idx)
                        ci = np.array(upd_curr_idx)
                        wp = np.array(upd_w_prev, dtype=np.float32)
                        wc = np.array(upd_w_curr, dtype=np.float32)
                        total = wp + wc
                        total[total < 1e-8] = 2.0
                        a_prev = (wp / total)[:, None]
                        a_curr = (wc / total)[:, None]

                        self.merged_points_xyz[mi] = (
                            a_prev * self.merged_points_xyz[mi] + a_curr * all_xyz[ci]
                        )
                        self.merged_points_colors[mi] = (
                            a_prev * self.merged_points_colors[mi].astype(np.float32)
                            + a_curr * all_colors[ci].astype(np.float32)
                        ).astype(np.uint8)

                # 未匹配的 curr 点 → 3D 近邻兜底 + 追加新点
                unmatched_mask = np.ones(n_pts, dtype=bool)
                if matched_curr_idx_set:
                    unmatched_mask[np.array(list(matched_curr_idx_set))] = False

                unmatched_xyz = all_xyz[unmatched_mask]
                unmatched_colors = all_colors[unmatched_mask]
                n_unmatched = len(unmatched_xyz)
                n_fallback_merged = 0
                n_new = 0

                if n_unmatched > 0:
                    overlap_threshold = self.merge_voxel_size if self.merge_voxel_size > 0 else 0.05
                    tree_fb = cKDTree(self.merged_points_xyz)
                    fb_dists, fb_nn = tree_fb.query(unmatched_xyz, k=1)

                    overlap_3d = fb_dists <= overlap_threshold
                    truly_new = ~overlap_3d
                    n_fallback_merged = int(overlap_3d.sum())
                    n_new = int(truly_new.sum())

                    if n_fallback_merged > 0:
                        ov_nn = fb_nn[overlap_3d]
                        ov_xyz = unmatched_xyz[overlap_3d]
                        ov_colors = unmatched_colors[overlap_3d]

                        unique_nn, inv = np.unique(ov_nn, return_inverse=True)
                        n_u = len(unique_nn)
                        sum_xyz = np.zeros((n_u, 3), dtype=np.float64)
                        sum_col = np.zeros((n_u, 3), dtype=np.float64)
                        cnt = np.zeros(n_u, dtype=np.int32)
                        np.add.at(sum_xyz, inv, ov_xyz.astype(np.float64))
                        np.add.at(sum_col, inv, ov_colors.astype(np.float64))
                        np.add.at(cnt, inv, 1)
                        avg_xyz = (sum_xyz / cnt[:, None]).astype(np.float32)
                        avg_col = sum_col / cnt[:, None]

                        w_exist, w_inc = 0.8, 0.2
                        self.merged_points_xyz[unique_nn] = (
                            w_exist * self.merged_points_xyz[unique_nn] + w_inc * avg_xyz
                        )
                        self.merged_points_colors[unique_nn] = (
                            w_exist * self.merged_points_colors[unique_nn].astype(np.float32)
                            + w_inc * avg_col
                        ).astype(np.uint8)

                    if n_new > 0:
                        self.merged_points_xyz = np.concatenate(
                            [self.merged_points_xyz, unmatched_xyz[truly_new]])
                        self.merged_points_colors = np.concatenate(
                            [self.merged_points_colors, unmatched_colors[truly_new]])

                if self.verbose:
                    print(f"    重叠匹配(复用对齐阶段): {n_matched}组, 置信度融合{n_updated}点")
                    print(f"    3D近邻兜底: {n_fallback_merged}点(融合), "
                          f"新增{n_new}点(追加), 合并后总点数: {len(self.merged_points_xyz)}")

            self._prev_aligned_recon = curr_recon
            
            # ==================== 延迟体素去重（每 N 个 batch 执行一次，减少开销）====================
            DEDUP_THRESHOLD = 100
            DEDUP_INTERVAL = 3
            if (self.merge_voxel_size > 0
                    and len(self.merged_points_xyz) > DEDUP_THRESHOLD
                    and num_batches % DEDUP_INTERVAL == 0):
                before_count = len(self.merged_points_xyz)
                self.merged_points_xyz, self.merged_points_colors = _voxel_dedup(
                    self.merged_points_xyz, 
                    self.merged_points_colors,
                    voxel_size=self.merge_voxel_size,
                    verbose=self.verbose
                )
                if self.verbose:
                    print(f"    体素去重: {before_count} -> {len(self.merged_points_xyz)} 点")
        
        # ==================== 并行写入 PLY + LAS ====================
        output_path = output_dir / f"merged_{num_batches}.ply"
        las_path = output_dir / f"merged_{num_batches}.las"
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_ply = executor.submit(
                save_ply_binary, output_path, self.merged_points_xyz,
                self.merged_points_colors, include_normals=True
            )
            fut_las = executor.submit(
                export_points_to_las, self.merged_points_xyz,
                self.merged_points_colors, str(las_path), verbose=self.verbose
            )
            fut_ply.result()
            fut_las.result()
        
        if self.visualizer is not None:
            self.visualizer.update_merged_point_cloud_from_arrays(
                self.merged_points_xyz, self.merged_points_colors
            )
        
        if self.verbose:
            status_msg = "第一个 batch 点云提取完成" if is_first_batch else "增量合并完成"
            print(f"  ✓ {status_msg}:")
            print(f"    总3D点数: {len(self.merged_points_xyz)}")
            print(f"    PLY 保存到: {output_path}")
            print(f"    LAS 保存到: {las_path}")
        
        if not is_first_batch:
            self._cleanup_intermediate_data()
        
        return True
    
    def _find_matched_points(
        self,
        pixel_map_prev: Dict,
        pixel_map_curr: Dict,
        common_images: Dict[int, int],
        match_radius: float = 60.0,
    ) -> set:
        """
        使用 2D 像素匹配识别 curr 中与 prev 重叠的 3D 点（向量化版本）
        
        Args:
            pixel_map_prev: prev_recon 的像素映射
            pixel_map_curr: curr_recon 的像素映射
            common_images: 公共影像映射 {prev_img_id: curr_img_id}
            match_radius: 像素匹配半径（默认 60 像素）
            
        Returns:
            matched_curr_pt3d_ids: curr 中与 prev 匹配的 point3D_id 集合
        """
        from scipy.spatial import cKDTree
        
        matched_curr_pt3d_ids = set()
        
        for prev_img_id, curr_img_id in common_images.items():
            pmap_prev = pixel_map_prev.get(prev_img_id)
            pmap_curr = pixel_map_curr.get(curr_img_id)
            
            if not pmap_prev or not pmap_curr:
                continue
            
            pixels_curr_list = list(pmap_curr.keys())
            pixels_prev_list = list(pmap_prev.keys())
            
            if not pixels_curr_list or not pixels_prev_list:
                continue
            
            # 预构建 curr 侧的 pt3d_id 数组，配合向量化索引一次性收集结果
            pt3d_ids_curr = np.array([pmap_curr[k]['point3D_id'] for k in pixels_curr_list])
            
            tree_curr = cKDTree(np.asarray(pixels_curr_list, dtype=np.float32))
            pixels_prev_arr = np.asarray(pixels_prev_list, dtype=np.float32)
            distances, indices = tree_curr.query(pixels_prev_arr, k=1, distance_upper_bound=match_radius)
            
            # 向量化过滤：一次性筛选所有有效匹配，避免 Python 逐元素循环
            valid = (distances <= match_radius) & (indices < len(pixels_curr_list))
            if valid.any():
                matched_curr_pt3d_ids.update(pt3d_ids_curr[indices[valid]].tolist())
        
        return matched_curr_pt3d_ids
    
    def _align_merged_reconstruction_to_gps_poses(
        self,
        reconstruction: pycolmap.Reconstruction
    ) -> pycolmap.Reconstruction:
        """
        将merged_reconstruction对齐到已知的GPS poses
        
        这与_rescale_reconstruction_to_original_size类似，但用于merged_reconstruction：
        1. 收集merged中所有影像对应的GPS位置
        2. 使用RANSAC对齐到GPS坐标系
        3. 纠正累积误差，防止reconstruction弯曲
        
        Args:
            reconstruction: merged_reconstruction
            
        Returns:
            对齐后的reconstruction
        """
        if self.verbose:
            print(f"    Aligning merged reconstruction to GPS poses...")
        
        try:
            # 1. 收集所有已处理的影像索引
            # 从所有inference_reconstructions中提取影像范围
            all_image_indices = []
            for recon_data in self.inference_reconstructions:
                start_idx = recon_data['start_idx']
                end_idx = recon_data['end_idx']
                # 注意重叠：每个reconstruction的前overlap张可能已经在前一个中了
                if len(all_image_indices) == 0:
                    # 第一个reconstruction，全部添加
                    all_image_indices.extend(range(start_idx, end_idx))
                else:
                    # 后续reconstruction，跳过重叠部分
                    all_image_indices.extend(range(start_idx + self.overlap, end_idx))
            
            if self.verbose:
                print(f"      Total unique image indices: {len(all_image_indices)}")
                print(f"      Reconstruction has {len(reconstruction.images)} images")
            
            # 2. 为每个reconstruction中的影像收集GPS位置
            tgt_image_names = []
            tgt_locations = []
            
            # merged_reconstruction中的image_id是连续的1-based
            for merged_image_id, image in sorted(reconstruction.images.items()):
                # merged_image_id从1开始，映射到all_image_indices
                if merged_image_id <= len(all_image_indices):
                    orig_idx = all_image_indices[merged_image_id - 1]
                    
                    if orig_idx < len(self.ori_extrinsic):
                        extrinsic = self.ori_extrinsic[orig_idx]
                        R_camera = np.array(extrinsic['R_camera'])
                        tvec = np.array(extrinsic['tvec'])
                        camera_center = -R_camera.T @ tvec
                        
                        tgt_image_names.append(image.name)
                        tgt_locations.append(camera_center)
            
            if len(tgt_image_names) < 3:
                print(f"      Warning: Not enough images ({len(tgt_image_names)}) for GPS alignment")
                return reconstruction
            
            tgt_locations = np.array(tgt_locations, dtype=np.float64)
            
            if self.verbose:
                print(f"      Collected {len(tgt_image_names)} GPS positions for alignment")
            
            # 3. 使用RANSAC对齐
            ransac_options = pycolmap.RANSACOptions()
            ransac_options.max_error = 5.0  # 5米误差阈值（可以根据GPS精度调整）
            ransac_options.min_inlier_ratio = 0.3  # 60%内点
            
            sim3d = pycolmap.align_reconstruction_to_locations(
                src=reconstruction,
                tgt_image_names=tgt_image_names,
                tgt_locations=tgt_locations,
                min_common_points=max(3, len(tgt_image_names) // 4),  # 至少用1/4的点
                ransac_options=ransac_options
            )
            
            if sim3d is not None:
                # 应用变换
                reconstruction.transform(sim3d)
                
                if self.verbose:
                    print(f"      ✓ Merged reconstruction aligned to GPS poses")
                    print(f"        Scale: {sim3d.scale:.6f}")
                    print(f"        Aligned images: {len(tgt_image_names)}")
            else:
                print("      Warning: Failed to align merged reconstruction to GPS")
                
        except Exception as e:
            print(f"      Error aligning merged reconstruction to GPS: {e}")
            import traceback
            traceback.print_exc()
        
        return reconstruction    

    def _center_reconstruction_at_first_camera(
        self,
        reconstruction: pycolmap.Reconstruction
    ) -> pycolmap.Reconstruction:
        """
        将重建平移使第一个相机位置为世界坐标系原点。
        
        这与 feature_matcher.py 中 _align_to_enu 的行为一致：
        对齐后确保第一个相机在原点 (0, 0, 0)。
        
        Args:
            reconstruction: 要调整的重建
            
        Returns:
            调整后的reconstruction（第一个相机在原点）
        """
        if len(self.image_paths) == 0:
            if self.verbose:
                print("    Warning: No image paths available, cannot center at first camera")
            return reconstruction
        
        try:
            # 获取第一个图像的名称
            first_img_name = self.image_paths[0].name
            first_camera_center = None
            
            # 在重建中找到第一个图像
            for img in reconstruction.images.values():
                if img.name == first_img_name:
                    cam_from_world = img.cam_from_world
                    R = cam_from_world.rotation.matrix()
                    t = cam_from_world.translation
                    # 计算相机中心: C = -R^T @ t
                    first_camera_center = -R.T @ t
                    break
            
            if first_camera_center is None:
                if self.verbose:
                    print(f"    Warning: First image '{first_img_name}' not found in reconstruction")
                return reconstruction
            
            # 计算将第一个相机移动到原点所需的偏移
            offset = -first_camera_center
            
            if self.verbose:
                print(f"    First camera position: [{first_camera_center[0]:.4f}, {first_camera_center[1]:.4f}, {first_camera_center[2]:.4f}]")
                print(f"    Applying translation to center at origin: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
            
            # 应用纯平移变换（scale=1, R=identity）
            self._apply_translation_to_reconstruction(reconstruction, offset)
            
            if self.verbose:
                # 验证第一个相机现在是否在原点
                for img in reconstruction.images.values():
                    if img.name == first_img_name:
                        cam_from_world = img.cam_from_world
                        R = cam_from_world.rotation.matrix()
                        t = cam_from_world.translation
                        new_center = -R.T @ t
                        print(f"    First camera final position: [{new_center[0]:.6f}, {new_center[1]:.6f}, {new_center[2]:.6f}]")
                        break
            
            return reconstruction
            
        except Exception as e:
            print(f"    Error centering reconstruction at first camera: {e}")
            import traceback
            traceback.print_exc()
            return reconstruction

    def _apply_translation_to_reconstruction(
        self,
        reconstruction: pycolmap.Reconstruction,
        translation: np.ndarray
    ):
        """
        对重建应用纯平移变换。
        
        将所有3D点和相机位置平移指定的偏移量。
        
        Args:
            reconstruction: 要变换的重建
            translation: 3D平移向量
        """
        # 平移所有3D点
        for point3D_id in list(reconstruction.points3D.keys()):
            point3D = reconstruction.points3D[point3D_id]
            point3D.xyz = point3D.xyz + translation
        
        # 平移所有相机位置
        for image_id in reconstruction.images:
            image = reconstruction.images[image_id]
            cam_from_world = image.cam_from_world
            
            # 获取旧的旋转和平移
            R_old = cam_from_world.rotation.matrix()
            t_old = cam_from_world.translation
            
            # 计算旧的相机中心: C_old = -R_old^T @ t_old
            C_old = -R_old.T @ t_old
            
            # 新的相机中心: C_new = C_old + translation
            C_new = C_old + translation
            
            # 旋转不变: R_new = R_old
            # 新的平移: t_new = -R_old @ C_new
            t_new = -R_old @ C_new
            
            # 更新相机位姿
            new_cam_from_world = pycolmap.Rigid3d(
                rotation=cam_from_world.rotation,  # 保持原旋转
                translation=t_new
            )
            image.cam_from_world = new_cam_from_world

    def create_pixel_coordinate_grid(self, num_frames, height, width):
        """
        Creates a grid of pixel coordinates and frame indices for all frames.
        Returns:
            tuple: A tuple containing:
                - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                                with x, y coordinates and frame indices
                - y_coords (numpy.ndarray): Array of y coordinates for all frames
                - x_coords (numpy.ndarray): Array of x coordinates for all frames
                - f_coords (numpy.ndarray): Array of frame indices for all frames
        """
        # Create coordinate grids for a single frame
        y_grid, x_grid = np.indices((height, width), dtype=np.float32)
        x_grid = x_grid[np.newaxis, :, :]
        y_grid = y_grid[np.newaxis, :, :]

        # Broadcast to all frames
        x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
        y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

        # Create frame indices and broadcast
        f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
        f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

        # Stack coordinates and frame indices
        points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

        return points_xyf

    def randomly_limit_trues(self, mask: np.ndarray, max_trues: int) -> np.ndarray:
        """
        If mask has more than max_trues True values,
        randomly keep only max_trues of them and set the rest to False.
        """
        # 1D positions of all True entries
        true_indices = np.flatnonzero(mask)  # shape = (N_true,)

        # if already within budget, return as-is
        if true_indices.size <= max_trues:
            return mask

        # randomly pick which True positions to keep
        sampled_indices = np.random.choice(
            true_indices, size=max_trues, replace=False
        )  # shape = (max_trues,)

        # build new flat mask: True only at sampled positions
        limited_flat_mask = np.zeros(mask.size, dtype=bool)
        limited_flat_mask[sampled_indices] = True

        # restore original shape
        return limited_flat_mask.reshape(mask.shape)

    def get_statistics(self) -> Dict:
        """Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_images': len(self.ori_extrinsic),
            'num_intrinsics': len(self.ori_intrinsic),
            'num_extrinsics': len(self.ori_extrinsic),
        }
        
        return stats
    
    def export_georeferenced(self, reconstruction: Optional[pycolmap.Reconstruction] = None, 
                              output_dir: Optional[Path] = None,
                              target_crs: str = "auto_utm",
                              gps_prior: float = 5.0,
                              align_to_global_sfm: bool = True,
                              center_at_first_camera: bool = True) -> bool:
        """Export the reconstruction and poses in a target coordinate system.

        This method:
        0. (Optional) Aligns merged reconstruction to global sparse SfM to correct drift
        1. Extracts camera positions from reconstruction and matches with original GPS data
        2. Converts original GPS lat/lon/alt to target CRS coordinates using pyproj
        3. Aligns the reconstruction to target coordinates using pycolmap model_aligner
        4. Exports the transformed reconstruction

        Args:
            reconstruction: pycolmap重建对象，默认使用 self.merged_reconstruction
            output_dir: 输出目录，默认使用 self.output_dir / "georeferenced"
            target_crs: 目标坐标系，支持以下选项：
                - "auto_utm": 自动检测UTM区域（默认）
                - "EPSG:3857": Web Mercator（适合网页地图可视化）
                - "EPSG:4326": WGS84 经纬度坐标
                - "EPSG:XXXX": 任意EPSG代码，如 "EPSG:32648" (UTM Zone 48N)
            gps_prior: GPS先验误差（米），用于alignment_max_error
            align_to_global_sfm: 已弃用，保留参数以兼容接口
            center_at_first_camera: 已弃用，保留参数以兼容接口

        Returns:
            True if successful, False otherwise
        """
        if not UTM_EXPORT_AVAILABLE:
            print("Error: pymap3d or pyproj not available. Install with: pip install pymap3d pyproj")
            return False

        # 使用默认值
        if reconstruction is None:
            reconstruction = self.merged_reconstruction
        if output_dir is None:
            output_dir = self.output_dir / "temp_merged_reconstruction_georeferenced"

        if reconstruction is None:
            print("Error: No reconstruction available.")
            return False
        
        if len(self.ori_extrinsic) == 0:
            print("Error: No original extrinsic data available.")
            return False

        try:
            # 0. 对齐步骤（已通过per-batch SfM在各批次中完成对齐，此处跳过全局SfM对齐）
            temp_input_dir = None
            if self.verbose:
                print(f"\n  Step 0: Skipping alignment to global SfM (using per-batch SfM alignment)")
            
            # 1. 建立 image_name -> GPS 的映射（从原始数据）
            image_name_to_gps = {}
            for ext in self.ori_extrinsic:
                img_name = ext['image_name']
                gps = ext['gps']  # [lat, lon, alt]
                image_name_to_gps[img_name] = gps
            
            # 2. 使用 utils.georef 进行地理参考导出
            success, result = export_reconstruction_georeferenced(
                reconstruction=reconstruction,
                image_name_to_gps=image_name_to_gps,
                output_dir=output_dir,
                target_crs=target_crs,
                gps_prior=gps_prior,
                temp_input_dir=temp_input_dir,
                verbose=self.verbose
            )
            
            if success and result:
                # 存储地理参考数据以供后续使用
                self.geo_center = result['geo_center']
                self.output_epsg_code = result['epsg_code']
                self.rec_georef = result['reconstruction']
                self.rec_georef_dir = result['output_dir']
                return True
            else:
                return False

        except Exception as e:
            print(f"Error during georeferenced export: {e}")
            import traceback
            traceback.print_exc()
            return False

    def export_dsm(
        self,
        output_path: Optional[Path] = None,
        resolution: float = 0.1,
        interpolation_method: str = "nearest",
        aggregation: str = "max",
        boundary_mask: bool = True,
        boundary_alpha: float = 0.005,
        boundary_buffer: int = 2,
    ) -> bool:
        """Export a Digital Surface Model (DSM) from the merged point cloud.
        
        从合并后的密集点云生成 DSM.tif 文件。
        
        Args:
            output_path: 输出 DSM 文件路径，默认为 output_dir / "temp_dsm" / "dsm.tif"
            resolution: DSM 分辨率（米），默认 0.1m (10cm)
            interpolation_method: 空洞填充方法，可选:
                - "nearest": 最近邻插值（快速，推荐）
                - "linear": 线性插值
                - "cubic": 三次插值（更平滑）
                - "idw": 反距离加权插值
                - "none": 不进行空洞填充
            aggregation: 单元格聚合方式:
                - "max": 取最大值（标准 DSM）
                - "min": 取最小值
                - "mean": 取平均值
            boundary_mask: 是否使用点云边界轮廓裁剪（非矩形），默认 True
            boundary_alpha: 边界轮廓参数:
                - 0: 使用凸包（convex hull）
                - >0: 使用凹包（concave hull），值越大边界越紧密
            boundary_buffer: 边界向外扩展的像素数，默认 10
                
        Returns:
            True if successful, False otherwise
        """
        # 导入 DSM 导出模块
        try:
            from SfM.dsm import export_dsm_from_point_cloud
        except ImportError:
            try:
                from dsm import export_dsm_from_point_cloud
            except ImportError:
                print("Error: DSM 模块不可用。请确保 SfM/dsm/ 存在。")
                return False
        
        # 确定输入点云路径
        point_cloud_path = self._find_merged_point_cloud()
        
        if point_cloud_path is None:
            print("Error: 未找到有效的点云文件。请先运行重建流程。")
            return False
        
        # 设置输出路径
        if output_path is None:
            output_path = self.output_dir / "temp_dsm" / "dsm.tif"
        
        # 使用通用导出函数
        return export_dsm_from_point_cloud(
            point_cloud_path=point_cloud_path,
            output_path=output_path,
            resolution=resolution,
            epsg_code=self.output_epsg_code,
            interpolation_method=interpolation_method,
            aggregation=aggregation,
            boundary_mask=boundary_mask,
            boundary_alpha=boundary_alpha,
            boundary_buffer=boundary_buffer,
            verbose=self.verbose,
        )

    def export_dsm_georeferenced(
        self,
        output_path: Optional[Path] = None,
        resolution: float = 0.1,
        interpolation_method: str = "nearest",
        aggregation: str = "max",
        boundary_mask: bool = True,
        boundary_alpha: float = 0.005,
        boundary_buffer: int = 2,
    ) -> bool:
        """Export a Digital Surface Model (DSM) from the georeferenced point cloud.
        
        从地理参考后的点云生成 DSM.tif 文件。
        需要先调用 export_georeferenced() 生成地理参考点云。
        
        Args:
            output_path: 输出 DSM 文件路径，默认为 output_dir / "temp_dsm_georeferenced" / "dsm.tif"
            resolution: DSM 分辨率（米），默认 0.1m (10cm)
            interpolation_method: 空洞填充方法
            aggregation: 单元格聚合方式
            boundary_mask: 是否使用点云边界轮廓裁剪
            boundary_alpha: 边界轮廓参数
            boundary_buffer: 边界向外扩展的像素数
                
        Returns:
            True if successful, False otherwise
        """
        # 导入 DSM 导出模块
        try:
            from SfM.dsm import export_dsm_from_point_cloud, find_point_cloud_in_directory
        except ImportError:
            try:
                from dsm import export_dsm_from_point_cloud, find_point_cloud_in_directory
            except ImportError:
                print("Error: DSM 模块不可用。请确保 SfM/dsm/ 存在。")
                return False
        
        # 检查是否已有地理参考导出
        if not hasattr(self, 'rec_georef_dir') or self.rec_georef_dir is None:
            print("Error: 未找到地理参考导出结果。请先调用 export_georeferenced()。")
            return False
        
        if self.verbose:
            print(f"\n  [Georeferenced DSM] 地理参考目录: {self.rec_georef_dir}")
            print(f"  [Georeferenced DSM] 目录存在: {self.rec_georef_dir.exists() if hasattr(self.rec_georef_dir, 'exists') else 'N/A'}")
        
        # 查找地理参考点云
        point_cloud_path = find_point_cloud_in_directory(
            self.rec_georef_dir, 
            preferred_names=["sparse_points", "points3D"]
        )
        
        if point_cloud_path is None:
            print(f"Error: 未找到地理参考点云文件")
            print(f"  搜索目录: {self.rec_georef_dir}")
            # 列出目录内容以便调试
            if hasattr(self.rec_georef_dir, 'exists') and self.rec_georef_dir.exists():
                print(f"  目录内容: {list(self.rec_georef_dir.iterdir())}")
            else:
                print(f"  目录不存在!")
            return False
        
        # 设置输出路径
        if output_path is None:
            output_path = self.output_dir / "temp_dsm" / "dsm_georeferenced.tif"
        
        if self.verbose:
            print(f"  [Georeferenced DSM] 使用点云: {point_cloud_path}")
            print(f"  [Georeferenced DSM] 点云存在: {point_cloud_path.exists()}")
            print(f"  [Georeferenced DSM] EPSG: {self.output_epsg_code}")
        
        # 使用通用导出函数
        return export_dsm_from_point_cloud(
            point_cloud_path=point_cloud_path,
            output_path=output_path,
            resolution=resolution,
            epsg_code=self.output_epsg_code,
            interpolation_method=interpolation_method,
            aggregation=aggregation,
            boundary_mask=boundary_mask,
            boundary_alpha=boundary_alpha,
            boundary_buffer=boundary_buffer,
            verbose=self.verbose,
        )

    def _find_merged_point_cloud(self) -> Optional[Path]:
        """查找合并后的点云文件路径。"""
        # 优先使用 points_only 模式的输出
        if self.merge_method == 'points_only' and self.merged_points_xyz is not None:
            num_batches = len(self.inference_reconstructions)
            ply_path = self.output_dir / "temp_merged_points_only" / f"merged_{num_batches}.ply"
            if ply_path.exists():
                return ply_path
        
        # 如果没有找到，尝试使用 merged_reconstruction
        if self.merged_reconstruction is not None:
            num_batches = len(self.inference_reconstructions)
            ply_path = self.output_dir / "temp_merged" / f"merged_{num_batches}" / "points3D.ply"
            if ply_path.exists():
                return ply_path
        
        return None

    def export_fastgs(
        self,
        reconstruction: Optional[pycolmap.Reconstruction] = None,
        images_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        copy_images: bool = True,
        resize: bool = False,
        output_format: str = "binary",
        filter_outliers_enabled: bool = False,
        outlier_std_ratio: float = 2.5,
        outlier_max_coord: float = 1000.0,
        use_georef: bool = False,
    ) -> bool:
        """
        将重建结果导出为 FastGS (3D Gaussian Splatting) 训练格式。
        
        FastGS 期望的目录结构:
            output_dir/
            ├── images/                   # 去畸变图像
            └── sparse/
                └── 0/
                    ├── cameras.bin/txt   # 相机内参
                    ├── images.bin/txt    # 相机位姿
                    ├── points3D.bin/txt  # 3D点（稀疏或密集）
                    └── points3D.ply      # 带法向量的PLY文件（FastGS需要）
        
        Args:
            reconstruction: pycolmap 重建对象（已弃用，使用 use_georef 参数）
            images_dir: 原始图像目录，默认从 self.image_paths 推断
            output_dir: 输出目录，默认为 self.output_dir / "temp_convert_fastgs"
            copy_images: 是否复制图像（True）或创建符号链接（False）
            resize: 是否创建缩小版本（images_2, images_4, images_8）
            output_format: COLMAP 文件输出格式，"binary" 或 "text"
            filter_outliers_enabled: 是否启用离散点过滤
            outlier_std_ratio: 标准差倍数阈值（越小过滤越严格）
            outlier_max_coord: 坐标绝对值最大阈值
            use_georef: 是否使用地理参考坐标系（默认 False 使用本地坐标系）
            
        Returns:
            True if successful, False otherwise
        """
        
        # 根据 use_georef 选择使用哪个已保存的重建路径
        if use_georef:
            if self.rec_georef_dir is None:
                print("Error: 地理参考重建路径不存在。请先调用 export_georeferenced()。")
                return False
            colmap_model_dir = Path(self.rec_georef_dir)
            coord_system = "georeferenced"
        else:
            if self.merged_reconstruction_path is None:
                print("Error: 合并重建路径不存在。请先完成重建合并。")
                return False
            colmap_model_dir = Path(self.merged_reconstruction_path)
            coord_system = "local"
        
        # 检查路径是否存在
        if not colmap_model_dir.exists():
            print(f"Error: COLMAP 模型目录不存在: {colmap_model_dir}")
            return False
        
        # 推断图像目录
        if images_dir is None:
            if len(self.image_paths) > 0:
                images_dir = self.image_paths[0].parent
            else:
                print("Error: Cannot determine images directory. Please provide images_dir.")
                return False
        
        # 设置输出目录
        if output_dir is None:
            output_dir = self.output_dir / "temp_convert_fastgs"
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Exporting to FastGS format...")
            print(f"{'='*60}")
            print(f"  COLMAP model dir: {colmap_model_dir} ({coord_system})")
            print(f"  Images directory: {images_dir}")
            print(f"  Output directory: {output_dir}")
            print(f"  Copy images: {copy_images}")
            print(f"  Output format: {output_format}")
            print(f"  Filter outliers: {filter_outliers_enabled}")
        
        # 调用 create_fastgs_structure（直接使用已保存的重建路径）
        success = create_fastgs_structure(
            colmap_model_dir=colmap_model_dir,
            images_dir=images_dir,
            output_dir=output_dir,
            copy_images=copy_images,
            resize=resize,
            output_format=output_format,
            filter_outliers_enabled=filter_outliers_enabled,
            outlier_std_ratio=outlier_std_ratio,
            outlier_max_coord=outlier_max_coord,
        )
        
        if success:
            if self.verbose:
                print(f"\n✓ FastGS export complete!")
                print(f"  Output directory: {output_dir}")
                print(f"\nTo train FastGS, run:")
                print(f"  python train.py -s {output_dir}")
        else:
            print("Error: FastGS export failed.")
        
        return success


def run_incremental_feature_matching(
    image_paths: List[Path],
    output_dir: Path,
    # ==================== 重建类型 ====================
    reconstruction_type: str = 'each_pixel_feature_points',  # 'dense_feature_points' | 'each_pixel_feature_points'
    # ==================== 模型参数 ====================
    model_type: str = 'vggt',  # 'mapanything' | 'vggt' | 'fastvggt'
    model_path: Optional[str] = None,  # 模型权重路径（VGGT/FastVGGT需要）
    # ==================== 影像处理参数 ====================
    image_interval: int = 2,  # 影像选取间隔（1=全部, 2=每隔1张, etc.）
    min_images_for_scale: int = 6,  # 每批次处理的影像数量
    overlap: int = 2,  # 相邻批次间的重叠影像数
    # ==================== 重建质量参数 ====================
    pred_vis_scores_thres_value: float = 0.7,  # 特征点可见性阈值
    min_visible_frames: int = 2,  # 3D点至少在多少帧中可见才保留（仅 each_pixel_feature_points 模式）
    max_reproj_error: float = 5.0,  # 最大重投影误差（像素）
    max_points3D_val: int = 1000000,  # 3D点坐标最大绝对值
    max_sampled_points: int = 100000,  # 每帧最大采样点数（仅 each_pixel_feature_points 模式）
    min_inlier_per_frame: int = 32,  # 每帧最少内点数
    filter_edge_margin: float = 100.0,  # 边缘过滤范围（像素），设为0禁用
    # ==================== 特征点跟踪参数（仅 dense_feature_points 模式）====================
    max_query_pts: int = 4096,  # 每个查询帧最大特征点数 4096 8192 12288
    query_frame_num: int = 3,  # 查询帧数量（建议 >= min_images_for_scale）
    # ==================== 点云合并参数 ====================
    merge_method: str = 'confidence',  # 'full' | 'confidence' | 'confidence_blend' | 'points_only'
    points_merge_mode: str = 'fast',  # 'fast' | 'quality'，points_only 模式下的合并策略
    merge_voxel_size: float = 0.5,  # 点云合并时的体素大小（米）
    merge_boundary_filter: bool = False,  # 是否启用边界过滤
    merge_statistical_filter: bool = False,  # 是否启用统计过滤
    # ==================== 导出参数 - 地理坐标 ====================
    export_georef: bool = True,  # 是否导出地理坐标系的重建结果
    target_crs: str = "auto_utm",  # 目标坐标系: "auto_utm", "EPSG:3857", "EPSG:4326", 等
    # ==================== 导出参数 - DSM ====================
    export_dsm: bool = True,  # 是否导出 DSM (数字表面模型)
    dsm_resolution: float = 1.0,  # DSM 分辨率（米），默认 100cm
    # ==================== 导出参数 - FastGS ====================
    export_fastgs: bool = True,  # 是否导出 FastGS 格式（3D Gaussian Splatting）
    fastgs_output_dir: Optional[Path] = None,  # FastGS 输出目录
    fastgs_copy_images: bool = True,  # 是否复制图像（True）或创建符号链接（False）
    fastgs_filter_outliers: bool = False,  # 是否在 FastGS 导出时过滤离散点
    fastgs_use_georef: bool = False,  # 是否使用地理坐标系（注意：可能影响训练精度）
    # ==================== 可视化参数 ====================
    enable_visualization: bool = True,  # 是否启用 Viser 可视化
    visualization_mode: str = 'merged',  # 'aligned' | 'merged'
    # ==================== FastVGGT 特有参数 ====================
    fastvggt_merging: int = 0,  # Token merging 参数（0=禁用）
    fastvggt_merge_ratio: float = 0.9,  # Token merge ratio (0.0-1.0)
    fastvggt_depth_conf_thresh: float = 3.0,  # 深度置信度阈值
    # ==================== 内存优化参数 ====================
    memory_keep_batches: int = 2,  # 保留批次数（1=边缘设备最省内存，2=默认）
    # ==================== Batch SfM 参数 ====================
    enable_batch_sfm: bool = False,  # 是否对每个 batch 运行传统 SfM 构建稀疏点云（用于对齐参考）
    batch_sfm_coord_mode: str = 'global_enu',  # 'global_enu' | 'wgs84'
    # ==================== 日志参数 ====================
    verbose: bool = False,
    # ==================== 计时参数 ====================
    enable_timing: bool = True,  # 是否启用计时统计
    # ==================== GPU显存监控参数 ====================
    enable_gpu_memory_tracking: bool = True,  # 是否启用GPU显存监控
) -> bool:
    """Run incremental image initialization pipeline.
    
    Args:
        image_paths: List of image file paths in processing order
        output_dir: Directory for output files
        
        # 重建类型
        reconstruction_type: 重建类型
            - 'each_pixel_feature_points': 纯密集点云（默认）
            - 'dense_feature_points': COLMAP格式输出，支持3DGS训练
        
        # 模型参数
        model_type: 模型类型 'mapanything' | 'vggt' | 'fastvggt'
        model_path: 模型权重路径（VGGT/FastVGGT需要，MapAnything自动下载）
        
        # 影像处理参数
        image_interval: 影像选取间隔（1=全部, 2=每隔1张, etc.）
        min_images_for_scale: 每批次处理的影像数量
        overlap: 相邻批次间的重叠影像数
        
        # 重建质量参数
        pred_vis_scores_thres_value: 特征点可见性阈值
        min_visible_frames: 3D点至少在多少帧中可见才保留（仅 each_pixel_feature_points 模式）
        max_reproj_error: 最大重投影误差（像素）
        max_points3D_val: 3D点坐标最大绝对值
        min_inlier_per_frame: 每帧最少内点数
        filter_edge_margin: 边缘过滤范围（像素），设为0禁用
        
        # 特征点跟踪参数（仅 dense_feature_points 模式有效）
        max_query_pts: 每个查询帧最大特征点数
        query_frame_num: 查询帧数量（建议 >= min_images_for_scale 以支持3DGS）
        
        # 点云合并参数
        merge_method: 合并方式
            - 'full': 完整流程
            - 'confidence': 简单置信度选择
            - 'confidence_blend': 置信度选择+边缘平滑插值（默认）
            - 'points_only': 仅合并点云（无COLMAP结构，更快）
        points_merge_mode: points_only 模式下的合并策略
            - 'fast': 3D 近邻 + 等权重平均（速度快，适合对齐精度高的场景）
            - 'quality': 2D 像素匹配 + 置信度加权 + 3D 兜底（质量好，减少分层）
        merge_voxel_size: 点云合并时的体素大小（米）
        merge_boundary_filter: 是否启用边界过滤
        merge_statistical_filter: 是否启用统计过滤
        
        # 导出参数 - 地理坐标
        export_georef: 是否导出地理坐标系的重建结果
        target_crs: 目标坐标系
            - "auto_utm": 自动检测UTM区域（默认）
            - "EPSG:3857": Web Mercator
            - "EPSG:4326": WGS84 经纬度
            - "EPSG:XXXX": 任意EPSG代码
        
        # 导出参数 - DSM
        export_dsm: 是否导出DSM（数字表面模型）
        dsm_resolution: DSM分辨率（米），默认0.1m (10cm)
        
        # 导出参数 - FastGS (3D Gaussian Splatting)
        export_fastgs: 是否导出FastGS训练格式
        fastgs_output_dir: FastGS输出目录，默认 output_dir / "temp_convert_fastgs"
        fastgs_copy_images: 是否复制图像（True）或创建符号链接（False）
        fastgs_filter_outliers: 是否在FastGS导出时过滤离散点
        fastgs_use_georef: 是否使用地理坐标系导出（默认False）
            - False: 使用本地坐标系（推荐，训练更稳定）
            - True: 使用地理坐标系（需先启用export_georef，坐标值较大可能影响精度）
        
        # 可视化参数
        enable_visualization: 是否启用Viser可视化
        visualization_mode: 可视化模式 'aligned' | 'merged'
        
        # FastVGGT 特有参数
        fastvggt_merging: Token merging参数（0=禁用）
        fastvggt_merge_ratio: Token merge ratio (0.0-1.0)
        fastvggt_depth_conf_thresh: 深度置信度阈值
        
        # 内存优化参数
        memory_keep_batches: 保留批次数（1=边缘设备最省内存，2=默认）
        
        # Batch SfM 参数
        enable_batch_sfm: 是否对每个 batch 运行传统 SfM 构建稀疏点云
            - True: 运行 SfM 获取稀疏重建，用于 pcl_alignment 对齐（默认）
            - False: 跳过 SfM，对齐回退到 image_alignment 模式（更快，适合 GPS 精度较高的场景）
        batch_sfm_coord_mode: Batch SfM 坐标系模式
            - 'global_enu': 平移到全局 ENU 坐标系（默认）
            - 'wgs84': 转换为 WGS84 经纬度高程 (lat, lon, alt)
        
        # 日志参数
        verbose: 是否启用详细日志
        
        # 计时参数
        enable_timing: 是否启用计时统计
            - True: 记录各步骤耗时并在完成后导出报告到 output_dir/temp_time/ 目录
            - False: 不记录计时信息（默认True）
        
        # GPU显存监控参数
        enable_gpu_memory_tracking: 是否启用GPU显存监控
            - True: 记录模型加载、推理、缓存清理等步骤的GPU显存使用情况
            - False: 不记录GPU显存信息（默认True）
    
    Returns:
        True if successful, False otherwise
    """
    # ==================== 初始化计时器 ====================
    global _timing_tracker
    if enable_timing:
        timing_output_dir = output_dir / "temp_time"
        _timing_tracker = TimingTracker(output_dir=timing_output_dir)
        _timing_tracker.start_total()
        if verbose:
            print(f"\n{'='*60}")
            print(f"Timing enabled. Report will be saved to: {timing_output_dir}")
            print(f"{'='*60}\n")
    else:
        _timing_tracker = None
    
    # ==================== 初始化GPU显存跟踪器 ====================
    global _gpu_memory_tracker
    if enable_gpu_memory_tracking:
        gpu_memory_output_dir = output_dir / "temp_memory"
        _gpu_memory_tracker = GPUMemoryTracker(output_dir=gpu_memory_output_dir)
        _gpu_memory_tracker.start_monitoring()
        if verbose:
            print(f"\n{'='*60}")
            print(f"GPU Memory tracking enabled. Report will be saved to: {gpu_memory_output_dir}")
            print(f"{'='*60}\n")
    else:
        _gpu_memory_tracker = None
    
    # Process images one by one with interval control
    selected_image_paths = image_paths[::image_interval]

    matcher = IncrementalFeatureMatcherSfM(
        output_dir=output_dir,
        # 重建类型
        reconstruction_type=reconstruction_type,
        # 模型参数
        model_type=model_type,
        model_path=model_path,
        # 影像处理参数
        min_images_for_scale=min_images_for_scale,
        overlap=overlap,
        # 重建质量参数
        pred_vis_scores_thres_value=pred_vis_scores_thres_value,
        min_visible_frames=min_visible_frames,
        max_reproj_error=max_reproj_error,
        max_points3D_val=max_points3D_val,
        max_sampled_points=max_sampled_points,
        min_inlier_per_frame=min_inlier_per_frame,
        filter_edge_margin=filter_edge_margin,
        # 特征点跟踪参数
        max_query_pts=max_query_pts,
        query_frame_num=query_frame_num,
        # 点云合并参数
        merge_method=merge_method,
        points_merge_mode=points_merge_mode,
        merge_voxel_size=merge_voxel_size,
        merge_boundary_filter=merge_boundary_filter,
        merge_statistical_filter=merge_statistical_filter,
        # 可视化参数
        enable_visualization=enable_visualization,
        visualization_mode=visualization_mode,
        # FastVGGT 参数
        fastvggt_merging=fastvggt_merging,
        fastvggt_merge_ratio=fastvggt_merge_ratio,
        fastvggt_depth_conf_thresh=fastvggt_depth_conf_thresh,
        # 内存优化参数
        memory_keep_batches=memory_keep_batches,
        # Batch SfM 参数
        enable_batch_sfm=enable_batch_sfm,
        batch_sfm_coord_mode=batch_sfm_coord_mode,
        # 全部原始影像和间隔参数（用于 batch SfM 补充中间帧）
        all_image_paths=image_paths,
        image_interval=image_interval,
        # 日志参数
        verbose=verbose,
    )

    # Process images one by one
    for i, image_path in enumerate(selected_image_paths):
        if _timing_tracker:
            _timing_tracker.start("add_image")
        
        success = matcher.add_image(image_path)
        
        if _timing_tracker:
            _timing_tracker.end("add_image")
        
        if not success:
            print(f"Failed to process image: {image_path}")
            return False

    # Release model
    matcher._release_model()

    # Export georeferenced coordinates if requested
    georef_export_success = False
    if export_georef:
        if matcher.merged_reconstruction is not None:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Exporting to georeferenced coordinates (target_crs: {target_crs})...")
                print(f"{'='*60}")
            
            if _timing_tracker:
                _timing_tracker.start("export_georef")
            
            georef_export_success = matcher.export_georeferenced(
                target_crs=target_crs,
            )
            
            if _timing_tracker:
                _timing_tracker.end("export_georef")
            
            if not georef_export_success:
                print("  Warning: Georeferenced export failed, but ENU reconstruction is still available")
        elif merge_method == 'points_only' and matcher.merged_points_xyz is not None:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Note: Georeferenced export is not supported in 'points_only' mode.")
                print(f"Point cloud has been saved to: {output_dir / 'temp_merged_points_only'}")
                print(f"{'='*60}")
        else:
            if verbose:
                print(f"\n  [Skip] Georeferenced export: merged_reconstruction is None")

    # Export DSM if requested
    if export_dsm:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Exporting Digital Surface Model (DSM)...")
            print(f"{'='*60}")
        
        if _timing_tracker:
            _timing_tracker.start("export_dsm")
        
        dsm_success = matcher.export_dsm(
            resolution=dsm_resolution,
            interpolation_method="nearest",
            aggregation="max",
        )
        
        if _timing_tracker:
            _timing_tracker.end("export_dsm")
        
        if not dsm_success:
            print("  Warning: DSM export failed")
        else:
            if verbose:
                print(f"  ✓ DSM exported to: {output_dir / 'temp_dsm' / 'dsm.tif'}")
        
        # 如果已进行地理参考导出，也基于地理参考点云导出 DSM
        if export_georef and georef_export_success and matcher.rec_georef_dir is not None:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Exporting DSM from georeferenced point cloud...")
                print(f"{'='*60}")
            
            if _timing_tracker:
                _timing_tracker.start("export_dsm_georef")
            
            georef_dsm_success = matcher.export_dsm_georeferenced(
                resolution=dsm_resolution,
                interpolation_method="nearest",
                aggregation="max",
            )
            
            if _timing_tracker:
                _timing_tracker.end("export_dsm_georef")
            
            if not georef_dsm_success:
                print("  Warning: Georeferenced DSM export failed")
            else:
                if verbose:
                    print(f"  ✓ Georeferenced DSM exported to: {output_dir / 'temp_dsm' / 'dsm_georeferenced.tif'}")

    # Export FastGS format if requested
    if export_fastgs:
        if matcher.merged_reconstruction is not None:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Exporting to FastGS (3D Gaussian Splatting) format...")
                print(f"{'='*60}")
            
            if _timing_tracker:
                _timing_tracker.start("export_fastgs")
            
            # 确定 FastGS 输出目录
            if fastgs_output_dir is None:
                fastgs_output_dir = output_dir / "temp_convert_fastgs"
            
            # 获取图像目录
            images_dir = selected_image_paths[0].parent if len(selected_image_paths) > 0 else None
            
            fastgs_success = matcher.export_fastgs(
                images_dir=images_dir,
                output_dir=fastgs_output_dir,
                copy_images=fastgs_copy_images,
                resize=False,
                output_format="binary",
                filter_outliers_enabled=fastgs_filter_outliers,
                use_georef=fastgs_use_georef,
            )
            
            if _timing_tracker:
                _timing_tracker.end("export_fastgs")
            
            if not fastgs_success:
                print("  Warning: FastGS export failed")
            else:
                if verbose:
                    coord_system = "georeferenced" if fastgs_use_georef else "local"
                    print(f"  ✓ FastGS format exported to: {fastgs_output_dir} ({coord_system} coordinates)")
                    print(f"\n  To train FastGS, run:")
                    print(f"    python train.py -s {fastgs_output_dir}")
        elif merge_method == 'points_only':
            if verbose:
                print(f"\n{'='*60}")
                print(f"Note: FastGS export requires 'dense_feature_points' reconstruction type")
                print(f"      with merge_method != 'points_only' to maintain COLMAP structure.")
                print(f"{'='*60}")
        else:
            if verbose:
                print(f"\n  [Skip] FastGS export: merged_reconstruction is None")

    if verbose:
        stats = matcher.get_statistics()
        print(f"\n{'='*60}")
        print("Final Statistics:")
        print(f"{'='*60}")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # ==================== 导出计时报告 ====================
    if _timing_tracker:
        print(f"\n{'='*60}")
        print("Exporting Timing Report...")
        print(f"{'='*60}")
        timing_report_path = _timing_tracker.export_to_file()
        print(f"Total runtime: {_timing_tracker.get_total_time():.2f} seconds ({_timing_tracker.get_total_time()/60:.2f} minutes)")
    
    # ==================== 导出GPU显存报告 ====================
    if _gpu_memory_tracker:
        print(f"\n{'='*60}")
        print("Exporting GPU Memory Report...")
        print(f"{'='*60}")
        # 记录最终显存状态
        _gpu_memory_tracker.record("final_state", "处理完成", 0)
        # 导出报告
        gpu_memory_report_path = _gpu_memory_tracker.export_to_file()
        print(f"Peak GPU Memory: {_gpu_memory_tracker.peak_gpu_memory:.2f} MB ({_gpu_memory_tracker.peak_gpu_memory/1024:.2f} GB)")
    
    return True


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # ==================== 配置参数 ====================
    # 输入输出目录
    # input_dir = Path(r"drone-map-anything\examples\Comprehensive_building_sel\images")
    # output_dir = Path(r"drone-map-anything\output\Comprehensive_building_sel\sparse_incremental_reconstruction")
    
    # windows path
    input_dir = Path(r"drone-map-anything\examples\Ganluo_images\images")
    output_dir = Path(r"drone-map-anything\output\Ganluo_images\sparse_incremental_reconstruction")

    # linux path
    # input_dir = Path("examples/Ganluo_images/images")
    # output_dir = Path("output/Ganluo_images/sparse_incremental_reconstruction")

    # input_dir = Path(r"drone-map-anything\examples\Tazishan\images")
    # output_dir = Path(r"drone-map-anything\output\Tazishan\sparse_incremental_reconstruction")

    # input_dir = Path(r"drone-map-anything\examples\SWJTU_gongdi\images")
    # output_dir = Path(r"drone-map-anything\output\SWJTU_gongdi\sparse_incremental_reconstruction")

    # input_dir = Path(r"drone-map-anything\examples\SWJTU_7th_teaching_building\images")
    # output_dir = Path(r"drone-map-anything\output\SWJTU_7th_teaching_building\sparse_incremental_reconstruction")
    
    # input_dir = Path(r"drone-map-anything\examples\HuaPo\images")
    # output_dir = Path(r"drone-map-anything\output\HuaPo\sparse_incremental_reconstruction")

    # input_dir = Path(r"drone-map-anything\examples\WenChuan\images")
    # output_dir = Path(r"drone-map-anything\output\WenChuan\sparse_incremental_reconstruction")

    # ==================== 模型参数 ====================
    # 模型选择: 'mapanything', 'vggt', 或 'fastvggt'
    MODEL_TYPE = 'vggt'  # 切换模型类型: 'mapanything' | 'vggt' | 'fastvggt'
    
    # 模型权重路径
    # - VGGT: "weights/vggt/model.pt"
    # - FastVGGT: "weights/fastvggt/model_tracker_fixed_e20.pt" (参考 eval_custom_colmap.py 默认路径)
    MODEL_PATH = "weights/vggt/model.pt"
    
    # ==================== FastVGGT 特有参数 ====================
    # FastVGGT 特有参数（仅当 MODEL_TYPE='fastvggt' 时生效）
    FASTVGGT_MERGING = 0  # Token merging 参数（0=禁用）
    FASTVGGT_MERGE_RATIO = 0.5  # Token merge ratio (0.0-1.0)
    FASTVGGT_DEPTH_CONF_THRESH = 0.5  # 深度置信度阈值
    
    # ================================================

    # Get all image files and sort them
    supported_formats = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = sorted([
        f for f in input_dir.glob("*") 
        if f.suffix in supported_formats
    ])
    
    print(f"Found {len(image_files)} images")
    print(f"Using model: {MODEL_TYPE}")
    
    # Run incremental initialization
    success = run_incremental_feature_matching(
        image_paths=image_files,
        output_dir=output_dir,
        # 模型参数
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        # FastVGGT 参数
        fastvggt_merging=FASTVGGT_MERGING,
        fastvggt_merge_ratio=FASTVGGT_MERGE_RATIO,
        fastvggt_depth_conf_thresh=FASTVGGT_DEPTH_CONF_THRESH,
        # 日志参数
        verbose=True,
    )
    
    if success:
        print("\n✓ Image initialization completed successfully")
    else:
        print("\n✗ Image initialization failed")