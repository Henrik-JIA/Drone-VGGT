#!/usr/bin/env python3
"""
LAS file export utilities for point clouds.

This module provides functions for exporting point clouds to LAS format,
supporting both pycolmap Reconstruction objects and raw numpy arrays.
"""

from pathlib import Path
from typing import Union

import numpy as np

# Optional imports
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False


def check_laspy_available() -> bool:
    """Check if laspy is available."""
    return LASPY_AVAILABLE


def export_points_to_las(
    xyz: np.ndarray,
    colors: np.ndarray,
    output_path: Union[str, Path],
    verbose: bool = False
) -> bool:
    """
    Export point cloud arrays to LAS format.
    
    Args:
        xyz: (N, 3) array of point coordinates
        colors: (N, 3) array of RGB colors (uint8)
        output_path: Output .las file path
        verbose: Print progress info
        
    Returns:
        True if successful, False otherwise
    """
    if not LASPY_AVAILABLE:
        print("Error: laspy not available. Install with: pip install laspy")
        return False
    
    try:
        if len(xyz) == 0:
            if verbose:
                print("Warning: No points to export")
            return False
        
        # Create LAS file
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = np.min(xyz, axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision
        
        las = laspy.LasData(header)
        
        # Set coordinates
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]
        
        # Set colors (LAS uses 16-bit color values)
        las.red = (colors[:, 0].astype(np.uint16) * 256)
        las.green = (colors[:, 1].astype(np.uint16) * 256)
        las.blue = (colors[:, 2].astype(np.uint16) * 256)
        
        # Write file
        las.write(str(output_path))
        
        if verbose:
            print(f"  Exported {len(xyz)} points to LAS: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error exporting to LAS: {e}")
        return False


def export_reconstruction_to_las(
    reconstruction: "pycolmap.Reconstruction",
    output_path: Union[str, Path],
    verbose: bool = False
) -> bool:
    """
    Export pycolmap Reconstruction to LAS format.
    
    Args:
        reconstruction: pycolmap Reconstruction object
        output_path: Output .las file path
        verbose: Print progress info
        
    Returns:
        True if successful, False otherwise
    """
    if not LASPY_AVAILABLE:
        print("Error: laspy not available. Install with: pip install laspy")
        return False
    
    if not PYCOLMAP_AVAILABLE:
        print("Error: pycolmap not available.")
        return False
    
    try:
        # Extract all 3D points and colors
        points = []
        colors = []
        
        for point3D_id, point3D in reconstruction.points3D.items():
            points.append(point3D.xyz)
            colors.append(point3D.color)
        
        if len(points) == 0:
            if verbose:
                print("Warning: No 3D points in reconstruction")
            return False
        
        points = np.array(points, dtype=np.float64)
        colors = np.array(colors, dtype=np.uint8)
        
        # Use the common export function
        return export_points_to_las(points, colors, output_path, verbose)
        
    except Exception as e:
        print(f"Error exporting reconstruction to LAS: {e}")
        return False

