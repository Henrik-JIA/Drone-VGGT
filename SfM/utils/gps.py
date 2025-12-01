#!/usr/bin/env python3
"""
GPS utilities for extracting and working with GPS data from images.
"""

from pathlib import Path
from typing import Optional, Union

import pymap3d as pm
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


def convert_to_degrees(value: Union[tuple, list]) -> Optional[float]:
    """Convert GPS coordinate tuple to decimal degrees.

    Args:
        value: GPS coordinate value in (degrees, minutes, seconds) format

    Returns:
        Decimal degrees or None if conversion fails
    """
    if not value:
        return None

    try:
        if isinstance(value[0], tuple):
            # Format: ((degrees, 1), (minutes, 1), (seconds, 1))
            d = float(value[0][0]) / float(value[0][1])
            m = float(value[1][0]) / float(value[1][1])
            s = float(value[2][0]) / float(value[2][1])
        else:
            # Format: (degrees, minutes, seconds)
            d, m, s = value
        return d + m / 60.0 + s / 3600.0
    except (ValueError, IndexError, TypeError):
        return None


def extract_gps_from_image(image_path: Path, include_altitude: bool = True) -> Optional[tuple]:
    """Extract GPS coordinates from image EXIF data.

    Args:
        image_path: Path to the image file
        include_altitude: Whether to include altitude in the result

    Returns:
        Tuple of (latitude, longitude) or (latitude, longitude, altitude) or None if no GPS data found
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None

            gps_data = {}
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "GPSInfo":
                    for gps_tag in value:
                        gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        gps_data[gps_tag_name] = value[gps_tag]

            if not gps_data:
                return None

            # Extract latitude
            lat = None
            if "GPSLatitude" in gps_data and "GPSLatitudeRef" in gps_data:
                lat_ref = gps_data["GPSLatitudeRef"]
                lat_coords = gps_data["GPSLatitude"]
                lat = convert_to_degrees(lat_coords)
                if lat is not None and lat_ref == "S":
                    lat = -lat

            # Extract longitude
            lon = None
            if "GPSLongitude" in gps_data and "GPSLongitudeRef" in gps_data:
                lon_ref = gps_data["GPSLongitudeRef"]
                lon_coords = gps_data["GPSLongitude"]
                lon = convert_to_degrees(lon_coords)
                if lon is not None and lon_ref == "W":
                    lon = -lon

            # Both lat and lon must be present
            if lat is None or lon is None:
                return None

            if not include_altitude:
                return lat, lon

            # Extract altitude (optional)
            altitude = 0.0
            if "GPSAltitude" in gps_data:
                alt_value = gps_data["GPSAltitude"]
                if isinstance(alt_value, tuple):
                    altitude = float(alt_value[0]) / float(alt_value[1])
                else:
                    altitude = float(alt_value)

                # Check altitude reference (0 = above sea level, 1 = below sea level)
                if gps_data.get("GPSAltitudeRef") == 1:
                    altitude = -altitude

            return lat, lon, altitude

    except Exception:
        return None


def lat_lon_to_enu(
    lat: float, lon: float, alt: float, ref_lat: float, ref_lon: float, ref_alt: float
) -> tuple[float, float, float]:
    """Convert latitude/longitude/altitude to ENU (East-North-Up) coordinates.

    Args:
        lat, lon, alt: Point coordinates in decimal degrees and meters
        ref_lat, ref_lon, ref_alt: Reference point coordinates in decimal degrees and meters

    Returns:
        ENU coordinates (east, north, up) in meters
    """
    # Use pymap3d for accurate geodetic coordinate conversion
    east, north, up = pm.geodetic2enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

    return east, north, up


def calculate_gps_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance between two GPS coordinates using pymap3d's haversine implementation.

    Args:
        lat1, lon1: First point coordinates in decimal degrees
        lat2, lon2: Second point coordinates in decimal degrees

    Returns:
        Distance in meters
    """
    # Use pymap3d for accurate distance calculation
    distance = pm.haversine(lat1, lon1, lat2, lon2)
    return distance


def get_gps_bounds(images_with_gps: list[tuple[Path, tuple[float, float, float]]]) -> tuple[float, float, float, float]:
    """Get the bounding box of GPS coordinates.

    Args:
        images_with_gps: List of (image_path, (lat, lon, alt)) tuples

    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon)
    """
    if not images_with_gps:
        return 0.0, 0.0, 0.0, 0.0

    lats = [gps[0] for _, gps in images_with_gps]
    lons = [gps[1] for _, gps in images_with_gps]

    return min(lats), max(lats), min(lons), max(lons)


def enu_to_lat_lon(
    east: float, north: float, up: float, ref_lat: float, ref_lon: float, ref_alt: float
) -> tuple[float, float, float]:
    """Convert ENU coordinates back to latitude/longitude/altitude.

    Args:
        east, north, up: ENU coordinates in meters
        ref_lat, ref_lon, ref_alt: Reference point coordinates in decimal degrees and meters

    Returns:
        Tuple of (lat, lon, alt) in decimal degrees and meters
    """
    # Use pymap3d for accurate coordinate conversion
    lat, lon, alt = pm.enu2geodetic(east, north, up, ref_lat, ref_lon, ref_alt)

    return lat, lon, alt


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the bearing between two GPS coordinates.

    Args:
        lat1, lon1: First point coordinates in decimal degrees
        lat2, lon2: Second point coordinates in decimal degrees

    Returns:
        Bearing in degrees (0-360)
    """
    # Use pymap3d for accurate bearing calculation
    bearing = pm.bearing(lat1, lon1, lat2, lon2)
    return bearing


def calculate_intermediate_point(
    lat1: float, lon1: float, lat2: float, lon2: float, fraction: float
) -> tuple[float, float]:
    """Calculate an intermediate point between two GPS coordinates.

    Args:
        lat1, lon1: First point coordinates in decimal degrees
        lat2, lon2: Second point coordinates in decimal degrees
        fraction: Fraction of the way between the points (0.0 to 1.0)

    Returns:
        Tuple of (lat, lon) for the intermediate point
    """
    # Use pymap3d for accurate interpolation
    lat_int, lon_int = pm.intermediate(lat1, lon1, lat2, lon2, fraction)
    return lat_int, lon_int
