#!/usr/bin/env python3
"""
Georeferencing utilities for SfM reconstruction.

This module provides functions for:
- Converting reconstructions to real-world coordinate systems (UTM, etc.)
- Using COLMAP model_aligner for coordinate system alignment
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import subprocess

import numpy as np

# Import LAS export functions from las_export module
from .las_export import (
    export_points_to_las,
    export_reconstruction_to_las,
    check_laspy_available,
)

# Optional imports
try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False


def check_dependencies() -> Tuple[bool, str]:
    """Check if all required dependencies are available.
    
    Returns:
        (is_available, error_message)
    """
    missing = []
    if not PYCOLMAP_AVAILABLE:
        missing.append("pycolmap")
    if not check_laspy_available():
        missing.append("laspy")
    if not PYPROJ_AVAILABLE:
        missing.append("pyproj")
    
    if missing:
        return False, f"Missing dependencies: {', '.join(missing)}. Install with: pip install {' '.join(missing)}"
    return True, ""


def determine_target_crs(
    lats: np.ndarray,
    lons: np.ndarray,
    target_crs: str = "auto_utm",
    verbose: bool = False
) -> Tuple[Optional["pyproj.CRS"], Optional[int]]:
    """
    Determine the target coordinate reference system.
    
    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        target_crs: Target CRS specification:
            - "auto_utm": Automatically detect UTM zone
            - "EPSG:XXXX": Specific EPSG code
            - Other valid CRS string
        verbose: Print progress info
        
    Returns:
        (crs_object, epsg_code) or (None, None) on failure
    """
    if not PYPROJ_AVAILABLE:
        print("Error: pyproj not available. Install with: pip install pyproj")
        return None, None
    
    try:
        if target_crs.lower() == "auto_utm":
            # Automatically detect UTM zone
            utm_crs_info = pyproj.database.query_utm_crs_info(
                datum_name="WGS 84",
                area_of_interest=pyproj.aoi.AreaOfInterest(
                    west_lon_degree=float(lons[0]) - 0.01,
                    south_lat_degree=float(lats[0]) - 0.01,
                    east_lon_degree=float(lons[0]) + 0.01,
                    north_lat_degree=float(lats[0]) + 0.01,
                ),
            )
            if not utm_crs_info:
                print("Error: Could not determine UTM zone for the given coordinates")
                return None, None
            target_crs_obj = pyproj.CRS.from_epsg(utm_crs_info[0].code)
            epsg_code = utm_crs_info[0].code
            
        elif target_crs.upper().startswith("EPSG:"):
            # Use specified EPSG code
            epsg_code = int(target_crs.split(":")[1])
            target_crs_obj = pyproj.CRS.from_epsg(epsg_code)
            
        else:
            # Try to parse as CRS directly
            target_crs_obj = pyproj.CRS.from_user_input(target_crs)
            epsg_code = target_crs_obj.to_epsg() if target_crs_obj.to_epsg() else None
        
        if verbose:
            crs_name = target_crs_obj.name if hasattr(target_crs_obj, 'name') else str(target_crs_obj)
            print(f"  Using CRS: {crs_name} (EPSG:{epsg_code})")
        
        return target_crs_obj, epsg_code
        
    except Exception as e:
        print(f"Error determining target CRS: {e}")
        return None, None


def convert_gps_to_target_crs(
    lats: np.ndarray,
    lons: np.ndarray,
    alts: np.ndarray,
    target_crs_obj: "pyproj.CRS",
    center_coords: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert GPS coordinates to target CRS.
    
    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        alts: Array of altitudes
        target_crs_obj: Target pyproj.CRS object
        center_coords: Whether to center coordinates at mean (avoid large values)
        
    Returns:
        (target_coords, geo_center) where:
            - target_coords: (N, 3) array in target CRS (possibly centered)
            - geo_center: (3,) array of center offset (zeros if not centered)
    """
    if not PYPROJ_AVAILABLE:
        raise ImportError("pyproj not available")
    
    # Create transformer from WGS84 to target CRS
    crs_transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs_obj, always_xy=True)
    
    # Convert to target CRS coordinates
    target_x, target_y = crs_transformer.transform(lons, lats)
    target_coords = np.column_stack([target_x, target_y, alts])
    
    if center_coords:
        # Use the mean as center to offset the whole scene
        geo_center = np.mean(target_coords, axis=0)
        geo_center[2] = 0.0  # Set altitude to 0 for center
        target_coords_offset = target_coords - geo_center
        return target_coords_offset, geo_center
    else:
        return target_coords, np.zeros(3)


def run_colmap_model_aligner(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    ref_images_path: Union[str, Path],
    alignment_max_error: float = 15.0,
    min_common_images: int = 3,
    verbose: bool = False
) -> bool:
    """
    Run COLMAP model_aligner to align reconstruction to reference coordinates.
    
    Args:
        input_path: Path to input reconstruction (COLMAP format)
        output_path: Path to output aligned reconstruction
        ref_images_path: Path to reference images file (image_name x y z per line)
        alignment_max_error: Maximum alignment error in meters
        min_common_images: Minimum number of common images required
        verbose: Print progress info
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "colmap",
        "model_aligner",
        "--log_to_stderr", "1" if verbose else "0",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--ref_images_path", str(ref_images_path),
        "--alignment_type", "custom",
        "--min_common_images", str(min_common_images),
        "--alignment_max_error", str(alignment_max_error),
        "--ref_is_gps", "0",
    ]
    
    try:
        if verbose:
            print(f"  Running COLMAP model_aligner...")
        subprocess.run(cmd, check=True, capture_output=not verbose)
        return True
    except subprocess.CalledProcessError as e:
        print(f"COLMAP model_aligner failed: {e}")
        return False
    except FileNotFoundError:
        print("Error: COLMAP not found. Please install COLMAP and add it to PATH.")
        return False


class GeoreferencedExporter:
    """
    A class for exporting reconstructions with georeferencing.
    
    This class handles the full pipeline of:
    1. Converting GPS coordinates to target CRS
    2. Running COLMAP model_aligner for alignment
    3. Exporting results in various formats
    """
    
    def __init__(
        self,
        reconstruction: "pycolmap.Reconstruction",
        image_name_to_gps: Dict[str, List[float]],
        verbose: bool = False
    ):
        """
        Initialize the exporter.
        
        Args:
            reconstruction: pycolmap Reconstruction object
            image_name_to_gps: Dict mapping image names to [lat, lon, alt]
            verbose: Print progress info
        """
        self.reconstruction = reconstruction
        self.image_name_to_gps = image_name_to_gps
        self.verbose = verbose
        
        # Results
        self.geo_center = None
        self.output_epsg_code = None
        self.rec_georef = None
        self.rec_georef_dir = None
    
    def export(
        self,
        output_dir: Union[str, Path],
        target_crs: str = "auto_utm",
        gps_prior: float = 5.0,
        temp_input_dir: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Export the reconstruction with georeferencing.
        
        Args:
            output_dir: Output directory for georeferenced reconstruction
            target_crs: Target coordinate reference system
            gps_prior: GPS prior error in meters (used for alignment_max_error)
            temp_input_dir: Temporary directory for input reconstruction
                           (if None, will create one)
                           
        Returns:
            True if successful, False otherwise
        """
        # Check dependencies
        is_available, error_msg = check_dependencies()
        if not is_available:
            print(f"Error: {error_msg}")
            return False
        
        output_dir = Path(output_dir)
        
        try:
            # 1. Match reconstruction images with GPS data
            image_names = []
            lats = []
            lons = []
            alts = []
            
            for img_id, img in self.reconstruction.images.items():
                img_name = img.name
                if img_name in self.image_name_to_gps:
                    gps = self.image_name_to_gps[img_name]
                    image_names.append(img_name)
                    lats.append(gps[0])
                    lons.append(gps[1])
                    alts.append(gps[2])
                else:
                    if self.verbose:
                        print(f"  Warning: No GPS data for image {img_name}, skipping")
            
            if len(image_names) < 3:
                print(f"Error: Not enough images with GPS data ({len(image_names)}). Need at least 3.")
                return False
            
            lats = np.array(lats)
            lons = np.array(lons)
            alts = np.array(alts)
            
            if self.verbose:
                print(f"  Matched {len(image_names)} images with GPS data")
            
            # 2. Determine target CRS
            target_crs_obj, epsg_code = determine_target_crs(
                lats, lons, target_crs, verbose=self.verbose
            )
            if target_crs_obj is None:
                return False
            self.output_epsg_code = epsg_code
            
            # 3. Convert GPS to target CRS
            target_coords_offset, geo_center = convert_gps_to_target_crs(
                lats, lons, alts, target_crs_obj, center_coords=True
            )
            self.geo_center = geo_center
            
            # 4. Prepare directories
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if temp_input_dir is None:
                temp_input_dir = output_dir.parent / "temp_georef_input"
            temp_input_dir = Path(temp_input_dir)
            temp_input_dir.mkdir(parents=True, exist_ok=True)
            
            # Save reconstruction for model_aligner input
            self.reconstruction.write_text(str(temp_input_dir))
            
            # 5. Write reference images file
            ref_images_path = output_dir / "georef_locations.txt"
            with open(ref_images_path, "w") as f:
                for img_name, (x, y, z) in zip(image_names, target_coords_offset):
                    f.write(f"{img_name} {x} {y} {z}\n")
            
            if self.verbose:
                print(f"  Reference images written to: {ref_images_path}")
                print(f"  Geo center (offset): [{geo_center[0]:.3f}, {geo_center[1]:.3f}, {geo_center[2]:.3f}]")
            
            # 6. Run COLMAP model_aligner
            min_common_images = min(len(image_names), 3)
            success = run_colmap_model_aligner(
                input_path=temp_input_dir,
                output_path=output_dir,
                ref_images_path=ref_images_path,
                alignment_max_error=gps_prior * 3.0,
                min_common_images=min_common_images,
                verbose=self.verbose
            )
            
            if not success:
                return False
            
            # 7. Load aligned model
            try:
                self.rec_georef = pycolmap.Reconstruction(str(output_dir))
            except Exception as e:
                print(f"Failed to load aligned georeferenced reconstruction: {e}")
                return False
            
            # 8. Export as PLY and LAS
            self.rec_georef.export_PLY(str(output_dir / "points3D.ply"))
            export_reconstruction_to_las(
                self.rec_georef, 
                output_dir / "points3D.las",
                verbose=self.verbose
            )
            
            self.rec_georef_dir = output_dir
            
            if self.verbose:
                print(f"  ✓ Georeferenced export completed:")
                print(f"    EPSG code: {self.output_epsg_code}")
                print(f"    Geo center (offset): [{geo_center[0]:.3f}, {geo_center[1]:.3f}, {geo_center[2]:.3f}]")
                print(f"    Output dir: {output_dir}")
                print(f"    Total images: {len(self.rec_georef.images)}")
                print(f"    Total 3D points: {len(self.rec_georef.points3D)}")
            
            return True
            
        except Exception as e:
            print(f"Error during georeferenced export: {e}")
            import traceback
            traceback.print_exc()
            return False


def export_reconstruction_georeferenced(
    reconstruction: "pycolmap.Reconstruction",
    image_name_to_gps: Dict[str, List[float]],
    output_dir: Union[str, Path],
    target_crs: str = "auto_utm",
    gps_prior: float = 5.0,
    temp_input_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False
) -> Tuple[bool, Optional[Dict]]:
    """
    Convenience function for exporting a reconstruction with georeferencing.
    
    Args:
        reconstruction: pycolmap Reconstruction object
        image_name_to_gps: Dict mapping image names to [lat, lon, alt]
        output_dir: Output directory for georeferenced reconstruction
        target_crs: Target coordinate reference system:
            - "auto_utm": Automatically detect UTM zone (default)
            - "EPSG:3857": Web Mercator
            - "EPSG:4326": WGS84
            - "EPSG:XXXX": Any EPSG code
        gps_prior: GPS prior error in meters
        temp_input_dir: Temporary directory for input reconstruction
        verbose: Print progress info
        
    Returns:
        (success, result_dict) where result_dict contains:
            - 'geo_center': (3,) center offset
            - 'epsg_code': EPSG code of output CRS
            - 'reconstruction': georeferenced reconstruction
            - 'output_dir': output directory path
    """
    exporter = GeoreferencedExporter(
        reconstruction=reconstruction,
        image_name_to_gps=image_name_to_gps,
        verbose=verbose
    )
    
    success = exporter.export(
        output_dir=output_dir,
        target_crs=target_crs,
        gps_prior=gps_prior,
        temp_input_dir=temp_input_dir
    )
    
    if success:
        return True, {
            'geo_center': exporter.geo_center,
            'epsg_code': exporter.output_epsg_code,
            'reconstruction': exporter.rec_georef,
            'output_dir': exporter.rec_georef_dir,
        }
    else:
        return False, None

