#!/usr/bin/env python3
"""
Feature extraction and matching for SfM using pycolmap.
Provides image loading, feature extraction, and 2D-2D matching capabilities.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List

import networkx as nx
import numpy as np
import pycolmap
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from utils.gps import extract_gps_from_image, lat_lon_to_enu
from utils.xmp import parse_xmp_tags


def cam_from_enu_transform(roll, pitch, yaw):
    """
    Returns the transformation matrix from ENU to camera coordinates,
    applying NED-from-ENU, camera-from-NED, and gimbal rotation.
    
    Args:
        roll: Gimbal roll angle in degrees
        pitch: Gimbal pitch angle in degrees
        yaw: Gimbal yaw angle in degrees
    
    Returns:
        3x3 rotation matrix from ENU to camera coordinates
    """
    # ENU to NED, align_vectors a=Tb
    ned_from_enu = R.align_vectors(
        a=[[0, 1, 0], 
           [1, 0, 0], 
           [0, 0, -1]], 
        b=[[1, 0, 0], 
           [0, 1, 0], 
           [0, 0, 1]]
    )[0].as_matrix()

    # Gimbal rotation in NED (ZYX order)
    ned_from_gimbal = R.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()
    gimbal_from_ned = ned_from_gimbal.T

    # Camera from NED (fixed mapping)
    # NED -> ZXY
    cam_from_ned = R.align_vectors(
        a=[[0, 0, 1], 
           [1, 0, 0], 
           [0, 1, 0]], 
        b=[[1, 0, 0], 
           [0, 1, 0], 
           [0, 0, 1]]
    )[0].as_matrix()

    cam_from_enu = cam_from_ned @ gimbal_from_ned @ ned_from_enu

    # Full chain: ENU -> NED -> Gimbal -> Camera
    return cam_from_enu


def image_ids_to_pair_id(image_id1, image_id2):
    """Convert two image IDs to a unique pair ID."""
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def pair_id_to_image_ids(pair_id):
    """Convert pair ID back to two image IDs."""
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2


class FeatureMatcherSfM:
    """Feature extraction and matching using pycolmap.
    
    This class provides functionality for:
    1. Loading images and extracting GPS/XMP metadata
    2. Initializing coordinate systems (ENU)
    3. Extracting SIFT features using pycolmap
    4. Performing feature matching (exhaustive or spatial)
    5. Creating prior reconstruction with initial poses
    
    The implementation is designed to work with DJI drone images containing
    GPS and gimbal pose information in XMP metadata.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        imgsz: int = 2048,
        num_features: int = 8192,
        match_mode: str = "exhaustive",  # "exhaustive" or "spatial"
        num_neighbors: int = 10,
        max_distance: float = 500.0,  # for spatial matching
        min_track_length: int = 2, 
        min_num_matches: int = 15, 
        verbose: bool = False,
    ):
        """Initialize feature matcher.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output files (database, etc.)
            imgsz: Maximum image size (longest edge) for feature extraction
            num_features: Number of features to extract per image
            match_mode: Feature matching mode ("exhaustive" or "spatial")
            num_neighbors: Number of neighbors for spatial matching
            max_distance: Maximum distance for spatial matching (meters)
            min_track_length: Minimum track length for valid tracks (used in build_2D_prior_recon)
            min_num_matches: Minimum number of matches for valid tracks (used in build_2D_prior_recon)
            verbose: Enable verbose logging
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.imgsz = imgsz
        self.num_features = num_features
        self.match_mode = match_mode
        self.num_neighbors = num_neighbors
        self.max_distance = max_distance
        self.min_track_length = min_track_length
        self.min_num_matches = min_num_matches
        self.verbose = verbose

        # Class member variables
        self.image_paths: List[Path] = []
        self.database_path: Optional[Path] = None
        
        self.image_gps_locations: Optional[np.ndarray] = None  # Shape: (N, 3)
        self.enu_origin: Optional[np.ndarray] = None  # Shape: (3,) [lat, lon, alt]
        self.image_enu_locations: Optional[np.ndarray] = None  # Shape: (N, 3)
        
        self.rec_prior: Optional[pycolmap.Reconstruction] = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pose = {
            "CameraName": {},
            "GpsLatitude": {},
            "GpsLongitude": {},
            "GpsAbsoluteAltitude": {},
            "GimbalRollDegree": {},
            "GimbalPitchDegree": {},
            "GimbalYawDegree": {},
            "CamReverse": {},
            "GimbalReverse": {},
            "DewarpFlag": {},
            "DewarpData": {},
        }

    def init_images(self) -> bool:
        """Load images and GPS data.
        
        Returns:
            True if successful, False otherwise
        """
        if self.verbose:
            print("Extracting GPS data from images...")

        image_paths = []
        gps_locations = []
        supported_formats = {".jpg", ".jpeg", ".png", ".tiff", ".tif", 
                           ".JPG", ".JPEG", ".PNG", ".TIFF", ".TIF"}

        for img_path in self.input_dir.glob("*"):
            if img_path.suffix in supported_formats:
                gps_data = extract_gps_from_image(img_path)
                if gps_data:
                    image_paths.append(img_path)
                    gps_locations.append(gps_data)

        if len(image_paths) < 2:
            print("Error: Need at least 2 images with GPS data for SFM")
            return False

        # Convert to NumPy array
        self.image_gps_locations = np.array(gps_locations)

        # Sort images by name
        indices = np.argsort([img.name for img in image_paths])
        self.image_paths = [image_paths[i] for i in indices]
        self.image_gps_locations = self.image_gps_locations[indices]

        if self.verbose:
            print(f"Found {len(self.image_paths)} images with GPS data")

        self.database_path = self.output_dir / "database.db"

        return True
    
    def init_crs(self) -> bool:
        """Initialize ENU coordinate system and convert GPS to ENU.
        
        Returns:
            True if successful, False otherwise
        """
        if self.image_gps_locations is None:
            print("Error: No GPS data loaded. Call init_images() first.")
            return False

        # Use the first image as ENU origin (same as COLMAP)
        origin = self.image_gps_locations[0]
        self.enu_origin = origin
        if self.verbose:
            print(f"ENU origin: lat={origin[0]:.6f}, lon={origin[1]:.6f}, alt={origin[2]:.1f}")

        # Compute ENU coordinates for all images
        self.image_enu_locations = np.array(
            [
                lat_lon_to_enu(gps[0], gps[1], gps[2], origin[0], origin[1], origin[2])
                for gps in self.image_gps_locations
            ]
        )
        return True

    def init_pos(self) -> bool:
        """Initialize gimbal pose angles from XMP metadata.
        
        Returns:
            True if successful, False otherwise
        """
        # Read XMP metadata from images
        success_count = 0

        for img_path in self.image_paths:
            try:
                xmp_data = parse_xmp_tags(img_path)

                if xmp_data is None:
                    if self.verbose:
                        print(f"Warning: No XMP data found in {img_path}")
                    continue

                # Parse XMP for gimbal pose data
                img_name = img_path.name

                # Store pose data for this image
                self.pose["CameraName"][img_name] = xmp_data.get("camera_name")
                self.pose["GpsLatitude"][img_name] = xmp_data.get("latitude")
                self.pose["GpsLongitude"][img_name] = xmp_data.get("longitude")
                self.pose["GpsAbsoluteAltitude"][img_name] = xmp_data.get("altitude")
                self.pose["GimbalRollDegree"][img_name] = xmp_data.get("roll")
                self.pose["GimbalPitchDegree"][img_name] = xmp_data.get("pitch")
                self.pose["GimbalYawDegree"][img_name] = xmp_data.get("yaw")
                self.pose["CamReverse"][img_name] = xmp_data.get("cam_reverse", 0)
                self.pose["GimbalReverse"][img_name] = xmp_data.get("gimbal_reverse", 0)
                self.pose["DewarpFlag"][img_name] = xmp_data.get("dewarp_flag", 0)
                self.pose["DewarpData"][img_name] = xmp_data.get("dewarp_data", [])
                success_count += 1

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to extract pose data from {img_path}: {e}")
                continue

        if self.verbose:
            print(f"Successfully extracted pose data from {success_count}/{len(self.image_paths)} images")
        return success_count > 0

    def extract_features(self) -> bool:
        """Extract SIFT features from images using pycolmap.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.image_paths:
            print("Error: No images loaded. Call init_images() first.")
            return False

        if self.verbose:
            print("Extracting SIFT features using pycolmap...")
        
        # # Delete existing database if it exists
        # if self.database_path and self.database_path.exists():
        #     if self.verbose:
        #         print(f"Removing existing database: {self.database_path}")
        #     os.remove(self.database_path)

        image_list_path = self.output_dir / "image_list.txt"
        with open(image_list_path, "w") as f:
            for img in self.image_paths:
                f.write(f"{img.name}\n")
        
        try:
            # Configure feature extraction options
            sift_options = pycolmap.SiftExtractionOptions()
            sift_options.max_image_size = self.imgsz
            sift_options.max_num_features = self.num_features

            # Extract features
            pycolmap.extract_features(
                database_path=str(self.database_path),
                image_path=str(self.input_dir),
                sift_options=sift_options,
                device=pycolmap.Device.auto
            )

            if self.verbose:
                print("Feature extraction completed successfully")
            return True

        except Exception as e:
            print(f"Error: Feature extraction failed: {e}")
            return False

    def match_features(self) -> bool:
        """Match features between images using pycolmap.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.database_path or not self.database_path.exists():
            print("Error: Database not found. Call extract_features() first.")
            return False

        if self.verbose:
            print(f"Matching features using {self.match_mode} mode...")

        try:
            if self.match_mode == "exhaustive":
                # Exhaustive matching
                pycolmap.match_exhaustive(
                    database_path=str(self.database_path)
                )
            elif self.match_mode == "spatial" and self.image_gps_locations is not None:
                # Spatial matching using GPS data
                sift_options = pycolmap.SiftMatchingOptions()

                spatial_options = pycolmap.SpatialMatchingOptions()
                spatial_options.max_distance = self.max_distance
                spatial_options.max_num_neighbors = self.num_neighbors
                spatial_options.ignore_z = True

                verification_options = pycolmap.TwoViewGeometryOptions()
            
                pycolmap.match_spatial(
                    database_path=str(self.database_path),
                    sift_options=sift_options,
                    matching_options=spatial_options,
                    verification_options=verification_options,
                    device=pycolmap.Device.auto
                )
                
            else:
                # Fallback to exhaustive matching
                if self.verbose:
                    print("Warning: Spatial matching requested but no GPS data available. Using exhaustive matching.")
                pycolmap.match_exhaustive(
                    database_path=str(self.database_path)
                )

            if self.verbose:
                print("Feature matching completed successfully")
            return True

        except Exception as e:
            print(f"Error: Feature matching failed: {e}")
            return False

    def build_2D_prior_recon(self) -> bool:
        """Create and save a prior reconstruction using initial poses from XMP metadata.
        
        Note:
            - Gimbal is in NED coordinates with angles in ZYX order
            - GPS is transformed to ENU coordinates
            - Image frame is x-right, y-down, z-forward (camera frame)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a new reconstruction object
            prior_recon = pycolmap.Reconstruction()

            # Create a shared camera model (using parameters from first image)
            camera_id = 1
            camera_model = "OPENCV"

            # Get camera parameters from first image
            first_img_name = self.image_paths[0].name

            dewarp_data = self.pose["DewarpData"].get(first_img_name, [])

            # Default parameters if DewarpData is missing
            if not dewarp_data:
                raise ValueError("DewarpData missing for first image")
            else:
                # Extract parameters from DewarpData
                # Get width and height from the loaded image size
                with PILImage.open(self.image_paths[0]) as img:
                    width, height = img.size
                params = dewarp_data[:8]  # fx, fy, cx, cy, k1, k2, p1, p2

            camera = pycolmap.Camera(
                camera_id=camera_id, 
                model=camera_model, 
                width=int(width), 
                height=int(height), 
                params=params
            )
            prior_recon.add_camera(camera)

            # Add images with their initial poses
            self._add_images_with_poses(prior_recon, camera_id)

            # Initialize point tracks from the correspondence graph
            self._initialize_point_tracks(prior_recon)

            self.rec_prior = prior_recon

            if self.verbose:
                print(f"Built 2D prior reconstruction with {len(prior_recon.images)} images")
            
            return True

        except Exception as e:
            print(f"Error: Failed to save prior reconstruction: {e}")
            return False


    def _add_images_with_poses(self, reconstruction: pycolmap.Reconstruction, camera_id: int):
        """Add images with their initial poses to the reconstruction.

        Args:
            reconstruction: The pycolmap reconstruction object
            camera_id: The camera ID to associate with images
        """
        for i, image_path in enumerate(self.image_paths):
            image_name = image_path.name
            image_id = i + 1

            # Skip images without complete pose data
            if (
                image_name not in self.pose["GimbalRollDegree"]
                or image_name not in self.pose["GimbalYawDegree"]
                or image_name not in self.pose["GimbalPitchDegree"]
            ):
                print(f"Skipping {image_name} - incomplete pose data")
                continue

            # Get ENU position for this image
            pos_enu = self.image_enu_locations[i]

            # Get gimbal orientation
            roll = self.pose["GimbalRollDegree"][image_name]
            yaw = self.pose["GimbalYawDegree"][image_name]
            pitch = self.pose["GimbalPitchDegree"][image_name]

            # Compute camera rotation matrix
            # world enu -> ned -> gimbal -> camera
            R_camera = cam_from_enu_transform(roll=roll, pitch=pitch, yaw=yaw)

            # Compute translation vector
            tvec = -R_camera @ pos_enu

            # Create full transformation matrix (cam_from_world)
            cam_from_world = pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d(R_camera),
                translation=tvec,
            )

            # Add image to reconstruction
            image = pycolmap.Image(
                image_id=image_id, name=image_name, cam_from_world=cam_from_world, camera_id=camera_id
            )
            reconstruction.add_image(image)


    def _initialize_point_tracks(self, reconstruction: pycolmap.Reconstruction):
        """Initialize point tracks for each image using the correspondence graph from the database.

        This method extracts correspondences from the COLMAP database and initializes
        the 2D point tracks for each image in the reconstruction. These tracks are
        essential for triangulation and bundle adjustment.

        Args:
            reconstruction: The pycolmap reconstruction object
        """
        if not self.database_path or not self.database_path.exists():
            print("Database not found, skipping point track initialization")
            return

        # Open the database and load the correspondence graph
        database = pycolmap.Database(str(self.database_path))
        image_names_set = set(img.name for img in reconstruction.images.values())
        dbcache = pycolmap.DatabaseCache.create(
            database, 
            min_num_matches=self.min_num_matches, 
            ignore_watermarks=True, 
            image_names=image_names_set
        )
        images = dbcache.images
        correspondence_graph = dbcache.correspondence_graph
        num_all_matches = correspondence_graph.num_correspondences_between_all_images()
        graph = nx.Graph()

        for pair_id, num_matches in num_all_matches.items():
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            if num_matches < self.min_num_matches:
                print(
                    f"Image pair ({image_id1}, {image_id2}) has too few matches ({num_matches}), skipping"
                )
                continue

            # (m, 2) ndarray of uint32
            matches = correspondence_graph.find_correspondences_between_images(image_id1, image_id2)
            # add nodes and edges to the graph
            if matches is None:
                print(f"No matches found between images {image_id1} and {image_id2}, skipping")
                continue
            # Add all edges from matches at once for efficiency
            edges = [((image_id1, int(idx1)), (image_id2, int(idx2))) for idx1, idx2 in matches]
            graph.add_edges_from(edges)

        # 3. 查找连通分量
        components = list(nx.connected_components(graph))
        print(f"Found {len(components)} connected components in correspondence graph")

        # 4. 为每个连通分量创建3D点
        try:
            point3D_id = 0
            tracks = []

            db_name_to_image = {img.name: img for img in images.values()}
            for recon_img_id, recon_img in reconstruction.images.items():
                img_name = recon_img.name
                if img_name in db_name_to_image:
                    db_img = db_name_to_image[img_name]
                    kps = database.read_keypoints(db_img.image_id)
                    pts2d = [
                            pycolmap.Point2D(xy=np.array([float(x), float(y)], dtype=np.float64))
                            for x, y in kps[:, :2]
                        ]
                    reconstruction.images[recon_img_id].points2D = pts2d
                else:
                    print(f"No keypoints found for {img_name}")

            for comp_idx, component in enumerate(components):
                if len(component) < self.min_track_length: # 允许只在 2 张图中匹配的点
                    continue
                # 添加到重建
                xyz = np.zeros(3).astype(np.float64)  # Initialize 3D point at origin
                tracks = []
                point3D_id = reconstruction.add_point3D(xyz, pycolmap.Track())

                # 创建Track对象（包含所有观测）
                for node in component:
                    image_id, point2D_idx = node
                    tracks.append(pycolmap.TrackElement(image_id=image_id, point2D_idx=point2D_idx))
                    reconstruction.images[image_id].points2D[
                        point2D_idx
                    ].point3D_id = point3D_id  # Assign track ID to point2D
                reconstruction.points3D[point3D_id].track = pycolmap.Track(tracks)

        except Exception as e:
            print(f"Error initializing point tracks: {e}")
            return
        print(f"Successfully added {len(components)} 3D points to reconstruction")
        return True

    def export_reconstruction(self) -> bool:
        """Export reconstruction to text format.
        
        Returns:
            True if successful, False otherwise
        """
        if self.rec_prior is None:
            print("Error: No reconstruction available. Call save_prior_recon() first.")
            return False
        
        try:
            recon_output_dir = self.output_dir / "sparse"
            recon_output_dir.mkdir(exist_ok=True)
            self.rec_prior.write_text(str(recon_output_dir))
            self.rec_prior.export_PLY(recon_output_dir / "sparse_points.ply")
            
            if self.verbose:
                print(f"Reconstruction exported to {recon_output_dir}")
            return True
            
        except Exception as e:
            print(f"Error: Failed to export reconstruction: {e}")
            return False

    def run_triangulation(self) -> bool:
        """基于数据库匹配对已知位姿模型进行三角化，并做一次全局BA收尾以抑制分层."""
        if not self.database_path or not self.database_path.exists():
            print("Database not found. Call extract_features/match_features() first.")
            return False

        tri_dir = self.output_dir / "enu"
        tri_dir.mkdir(exist_ok=True)

        try:
            # 配置全局束调整 - 更严格的设置
            opts = pycolmap.IncrementalPipelineOptions()
            opts.ba_global_max_num_iterations = 75  # 增加全局BA最大迭代次数 (默认50)
            tri_opts = pycolmap.IncrementalTriangulatorOptions()
            tri_opts.ignore_two_view_tracks = True
            tri_opts.min_angle = 1.5
            tri_opts.complete_max_reproj_error = 4.0
            tri_opts.merge_max_reproj_error = 4.0
            opts.triangulation = tri_opts

            result_reconstruction = pycolmap.triangulate_points(
                reconstruction=self.rec_prior,
                database_path=str(self.database_path),
                image_path=str(self.input_dir),
                output_path=str(tri_dir),
                clear_points=True,
                options=opts,
                refine_intrinsics=True
            )
            print(f"Triangulation completed: {tri_dir}")

            if result_reconstruction is not None:
                obs = pycolmap.ObservationManager(result_reconstruction)
                obs.filter_all_points3D(max_reproj_error=4.0, min_tri_angle=1.5)

                result_reconstruction.write_text(str(tri_dir))
                self.rec_prior = result_reconstruction
                self.rec_prior_dir = tri_dir
                result_reconstruction.export_PLY(tri_dir / "sparse_points.ply")
                print(f"Triangulation completed: {len(result_reconstruction.images)} images, {len(result_reconstruction.points3D)} 3D points")
            else:
                print("Triangulation returned None")
                return False

        except Exception as e:
            print(f"Triangulation failed: {e}")
            return False

        return True

    def run_pipeline(self) -> bool:
        """Run the complete feature extraction and matching pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.verbose:
                print("Feature Matcher: Starting pipeline")
            
            if self.verbose:
                print("Feature Matcher: Initializing images...")
            if not self.init_images():
                return False
            if not self.init_crs():
                return False
            if not self.init_pos():
                return False
            
            if self.verbose:
                print("Feature Matcher: Extracting and matching features...")
            if not self.extract_features():
                return False
            if not self.match_features():
                return False
            
            if self.verbose:
                print("Feature Matcher: Saving prior reconstruction...")
            if not self.build_2D_prior_recon():
                return False
            
            if self.verbose:
                print("Feature Matcher: Exporting reconstruction...")
            if not self.export_reconstruction():
                return False
            
            if self.verbose:
                print("Feature Matcher: Running triangulation...")
            if not self.run_triangulation():
                return False

            if self.verbose:
                print("Feature Matcher: Pipeline completed successfully")
            return True

        except Exception as e:
            print(f"Error: Feature matcher pipeline failed: {e}")
            return False


def run_feature_matching(
    input_dir: Path,
    output_dir: Path,
    imgsz: int = 2048,
    num_features: int = 8192,
    match_mode: str = "exhaustive",
    num_neighbors: int = 10,
    max_distance: float = 500.0,
    verbose: bool = False,
) -> bool:
    """Run feature extraction and matching pipeline.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output files
        imgsz: Maximum image size (longest edge)
        num_features: Number of features to extract per image
        match_mode: Feature matching mode ("exhaustive" or "spatial")
        num_neighbors: Number of neighbors for spatial matching
        max_distance: Maximum distance for spatial matching (meters)
        verbose: Enable verbose logging
        
    Returns:
        True if successful, False otherwise
    """
    matcher = FeatureMatcherSfM(
        input_dir=input_dir,
        output_dir=output_dir,
        imgsz=imgsz,
        num_features=num_features,
        match_mode=match_mode,
        num_neighbors=num_neighbors,
        max_distance=max_distance,
        verbose=verbose,
    )
    return matcher.run_pipeline()


if __name__ == "__main__":
    # Example usage
    input_dir = Path(r"drone-map-anything\examples\Ganluo_images\images")
    output_dir = Path(r"drone-map-anything\output\Ganluo_images\sparse_reconstruction")
    
    success = run_feature_matching(
        input_dir=input_dir,
        output_dir=output_dir,
        imgsz=2048,
        num_features=8192,
        match_mode="spatial",
        num_neighbors=10,
        max_distance=500.0,
        verbose=True,
    )
    
    if success:
        print("Feature matching completed successfully")
    else:
        print("Feature matching failed")