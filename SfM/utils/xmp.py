#!/usr/bin/env python3
"""
XMP utilities for extracting and working with XMP data from images, including attitude conversion.
Updated to use piexif and Pillow instead of pyexiv2.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import re
import xml.etree.ElementTree as ET

import numpy as np
import piexif
from PIL import Image as PILImage
from PIL.ExifTags import TAGS
from scipy.spatial.transform import Rotation as R


def extract_xmp_from_image(image_path: Path) -> Optional[Dict[str, str]]:
    """Extract XMP metadata from image using Pillow.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing XMP metadata or None if extraction fails
    """
    try:
        with PILImage.open(str(image_path)) as img:
            # Try to get XMP data from info
            if hasattr(img, 'info') and 'xmp' in img.info:
                xmp_raw = img.info['xmp']
                return parse_xmp_data(xmp_raw)
            
            # Try to extract XMP from EXIF
            if hasattr(img, '_getexif'):
                exif_dict = img._getexif()
                if exif_dict:
                    for tag_id, value in exif_dict.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if isinstance(tag, str) and 'xmp' in tag.lower():
                            return parse_xmp_data(value)
            
            # Try using piexif to extract XMP
            try:
                exif_data = piexif.load(str(image_path))
                if "0th" in exif_data:
                    for tag, value in exif_data["0th"].items():
                        if tag == piexif.ImageIFD.XPComment or tag == piexif.ImageIFD.ImageDescription:
                            if isinstance(value, bytes):
                                value = value.decode('utf-8', errors='ignore')
                            if 'drone-dji' in str(value):
                                return parse_xmp_data(value)
            except Exception:
                pass
            
            # Alternative: Read file directly and search for XMP
            return extract_xmp_from_file_content(image_path)
            
    except Exception as e:
        print(f"Error extracting XMP from {image_path}: {e}")
        return None


def extract_xmp_from_file_content(image_path: Path) -> Optional[Dict[str, str]]:
    """Extract XMP by reading file content directly."""
    try:
        with open(image_path, 'rb') as f:
            content = f.read()
            
        # Look for XMP packet
        xmp_start = content.find(b'<?xpacket begin=')
        xmp_end = content.find(b'<?xpacket end=')
        
        if xmp_start != -1 and xmp_end != -1:
            # Find the actual end of the xpacket
            end_marker = content.find(b'?>', xmp_end)
            if end_marker != -1:
                xmp_content = content[xmp_start:end_marker + 2]
            else:
                xmp_content = content[xmp_start:xmp_end + 20]
            
            return parse_xmp_data(xmp_content)
            
        # Alternative: Look for DJI-specific tags
        dji_patterns = [
            b'drone-dji:GimbalRollDegree',
            b'drone-dji:GimbalPitchDegree',
            b'drone-dji:FlightRollDegree'
        ]
        
        for pattern in dji_patterns:
            if pattern in content:
                # Find surrounding XML-like content
                start_idx = max(0, content.find(pattern) - 1000)
                end_idx = min(len(content), content.find(pattern) + 1000)
                xml_section = content[start_idx:end_idx]
                
                parsed_data = parse_xmp_data(xml_section)
                if parsed_data:
                    return parsed_data
                    
    except Exception as e:
        print(f"Error reading file content from {image_path}: {e}")
        
    return None


def parse_xmp_data(xmp_content) -> Optional[Dict[str, str]]:
    """Parse XMP content into a dictionary."""
    try:
        # Convert bytes to string if necessary
        if isinstance(xmp_content, bytes):
            xmp_content = xmp_content.decode('utf-8', errors='ignore')
        
        # Try XML parsing first
        if '<x:xmpmeta' in xmp_content or '<rdf:RDF' in xmp_content:
            return parse_xmp_xml(xmp_content)
        else:
            return parse_xmp_data_simple(xmp_content)
    except Exception as e:
        print(f"Error in parse_xmp_data: {e}")
        return parse_xmp_data_simple(xmp_content)


def parse_xmp_xml(xmp_content: str) -> Optional[Dict[str, str]]:
    """Parse XMP XML content."""
    try:
        # Remove the XMP packet markers
        xmp_content = re.sub(r'<\?xpacket[^>]*\?>', '', xmp_content)
        
        # Find the rdf:Description tag and extract attributes
        desc_pattern = r'<rdf:Description([^>]*?)(?:/>|>.*?</rdf:Description>)'
        desc_match = re.search(desc_pattern, xmp_content, re.DOTALL)
        
        if not desc_match:
            return None
        
        attr_content = desc_match.group(1)
        
        # Parse attributes using regex
        result = {}
        
        # Pattern to match XML attributes: namespace:name="value" or namespace:name='value'
        attr_pattern = r'(\w+(?:-\w+)*):(\w+(?:\w+)*)=["\'](.*?)["\']\s*'
        matches = re.findall(attr_pattern, attr_content)
        
        for namespace, attr_name, value in matches:
            if namespace == 'drone-dji':
                key = f"Xmp.drone-dji.{attr_name}"
                # Clean up the value (remove + signs for positive numbers)
                clean_value = value.strip()
                if clean_value.startswith('+'):
                    clean_value = clean_value[1:]
                result[key] = clean_value
        
        return result if result else None
        
    except Exception as e:
        print(f"Error parsing XMP XML: {e}")
        return None

def parse_xmp_data_simple(content) -> Optional[Dict[str, str]]:
    """Simple pattern-based XMP parsing for DJI data."""
    # Convert bytes to string if necessary
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='ignore')
    
    result = {}
    
    # DJI-specific patterns
    patterns = {
        'Xmp.drone-dji.GimbalRollDegree': [
            r'drone-dji:GimbalRollDegree["\s]*=["\']\s*([^"\']+)',
            r'GimbalRollDegree["\s]*>([^<]+)',
            r'GimbalRollDegree["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.GimbalPitchDegree': [
            r'drone-dji:GimbalPitchDegree["\s]*=["\']\s*([^"\']+)',
            r'GimbalPitchDegree["\s]*>([^<]+)',
            r'GimbalPitchDegree["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.GimbalYawDegree': [
            r'drone-dji:GimbalYawDegree["\s]*=["\']\s*([^"\']+)',
            r'GimbalYawDegree["\s]*>([^<]+)',
            r'GimbalYawDegree["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.FlightRollDegree': [
            r'drone-dji:FlightRollDegree["\s]*=["\']\s*([^"\']+)',
            r'FlightRollDegree["\s]*>([^<]+)',
            r'FlightRollDegree["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.FlightPitchDegree': [
            r'drone-dji:FlightPitchDegree["\s]*=["\']\s*([^"\']+)',
            r'FlightPitchDegree["\s]*>([^<]+)',
            r'FlightPitchDegree["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.FlightYawDegree': [
            r'drone-dji:FlightYawDegree["\s]*=["\']\s*([^"\']+)',
            r'FlightYawDegree["\s]*>([^<]+)',
            r'FlightYawDegree["\s]*=["\']\s*([^"\']+)'
        ],
        # 添加速度相关的模式
        'Xmp.drone-dji.FlightXSpeed': [
            r'drone-dji:FlightXSpeed["\s]*=["\']\s*([^"\']+)',
            r'FlightXSpeed["\s]*>([^<]+)',
            r'FlightXSpeed["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.FlightYSpeed': [
            r'drone-dji:FlightYSpeed["\s]*=["\']\s*([^"\']+)',
            r'FlightYSpeed["\s]*>([^<]+)',
            r'FlightYSpeed["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.FlightZSpeed': [
            r'drone-dji:FlightZSpeed["\s]*=["\']\s*([^"\']+)',
            r'FlightZSpeed["\s]*>([^<]+)',
            r'FlightZSpeed["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.GpsLatitude': [
            r'drone-dji:GpsLatitude["\s]*=["\']\s*([^"\']+)',
            r'GpsLatitude["\s]*>([^<]+)',
            r'GpsLatitude["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.GpsLongtitude': [  # Note: DJI uses "Longtitude" (typo)
            r'drone-dji:GpsLongtitude["\s]*=["\']\s*([^"\']+)',
            r'GpsLongtitude["\s]*>([^<]+)',
            r'GpsLongtitude["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.AbsoluteAltitude': [
            r'drone-dji:AbsoluteAltitude["\s]*=["\']\s*([^"\']+)',
            r'AbsoluteAltitude["\s]*>([^<]+)',
            r'AbsoluteAltitude["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.RelativeAltitude': [
            r'drone-dji:RelativeAltitude["\s]*=["\']\s*([^"\']+)',
            r'RelativeAltitude["\s]*>([^<]+)',
            r'RelativeAltitude["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.CalibratedFocalLength': [
            r'drone-dji:CalibratedFocalLength["\s]*=["\']\s*([^"\']+)',
            r'CalibratedFocalLength["\s]*>([^<]+)',
            r'CalibratedFocalLength["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.CalibratedOpticalCenterX': [
            r'drone-dji:CalibratedOpticalCenterX["\s]*=["\']\s*([^"\']+)',
            r'CalibratedOpticalCenterX["\s]*>([^<]+)',
            r'CalibratedOpticalCenterX["\s]*=["\']\s*([^"\']+)'
        ],
        'Xmp.drone-dji.CalibratedOpticalCenterY': [
            r'drone-dji:CalibratedOpticalCenterY["\s]*=["\']\s*([^"\']+)',
            r'CalibratedOpticalCenterY["\s]*>([^<]+)',
            r'CalibratedOpticalCenterY["\s]*=["\']\s*([^"\']+)'
        ]
    }
    
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                clean_value = matches[0].strip()
                # Remove + sign from positive numbers
                if clean_value.startswith('+'):
                    clean_value = clean_value[1:]
                result[key] = clean_value
                break
    
    return result if result else None


def parse_dji_attitude(xmp_data: Dict[str, str]) -> Optional[Dict[str, Dict[str, float]]]:
    """Parse DJI attitude data from XMP metadata.

    Args:
        xmp_data: XMP metadata dictionary

    Returns:
        Dictionary containing gimbal and flight attitude data in degrees
        Format: {
            'gimbal': {'roll': float, 'pitch': float, 'yaw': float},
            'flight': {'roll': float, 'pitch': float, 'yaw': float}
        }
    """
    try:
        attitude_data = {}
        
        # Extract gimbal attitude
        gimbal_roll = float(xmp_data.get('Xmp.drone-dji.GimbalRollDegree', '0.0'))
        gimbal_pitch = float(xmp_data.get('Xmp.drone-dji.GimbalPitchDegree', '0.0'))
        gimbal_yaw = float(xmp_data.get('Xmp.drone-dji.GimbalYawDegree', '0.0'))
        
        attitude_data['gimbal'] = {
            'yaw': gimbal_yaw,
            'pitch': gimbal_pitch,
            'roll': gimbal_roll
        }
        
        # Extract flight attitude
        flight_roll = float(xmp_data.get('Xmp.drone-dji.FlightRollDegree', '0.0'))
        flight_pitch = float(xmp_data.get('Xmp.drone-dji.FlightPitchDegree', '0.0'))
        flight_yaw = float(xmp_data.get('Xmp.drone-dji.FlightYawDegree', '0.0'))
        
        attitude_data['flight'] = {
            'yaw': flight_yaw,
            'pitch': flight_pitch,
            'roll': flight_roll
        }
        
        return attitude_data
    except (ValueError, KeyError):
        return None


def parse_dji_gps(xmp_data: Dict[str, str]) -> Optional[Dict[str, float]]:
    """Parse DJI GPS data from XMP metadata.

    Args:
        xmp_data: XMP metadata dictionary

    Returns:
        Dictionary containing GPS data
        Format: {
            'latitude': float,
            'longitude': float,
            'absolute_altitude': float,
            'relative_altitude': float
        }
    """
    try:
        gps_data = {}
        
        gps_data['latitude'] = float(xmp_data.get('Xmp.drone-dji.GpsLatitude', '0.0'))
        gps_data['longitude'] = float(xmp_data.get('Xmp.drone-dji.GpsLongitude', '0.0'))
        gps_data['absolute_altitude'] = float(xmp_data.get('Xmp.drone-dji.AbsoluteAltitude', '0.0'))
        gps_data['relative_altitude'] = float(xmp_data.get('Xmp.drone-dji.RelativeAltitude', '0.0'))
        
        return gps_data
    except (ValueError, KeyError):
        return None


def parse_dji_camera_params(xmp_data: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Parse DJI camera parameters from XMP metadata and build intrinsic matrix K.

    Priority:
    1) Xmp.drone-dji.DewarpData: time;fx,fy,cx,cy,k1,k2,p1,p2,k3
       Note: cx, cy in DewarpData are offsets relative to the image center.
    2) Fallback to CalibratedFocalLength, CalibratedOpticalCenterX/Y (pixels)

    K = [[alpha, -alpha*cot(theta), cx],
         [0,     beta/sin(theta),   cy],
         [0,     0,                 1 ]]
    where alpha ~ fx(px), beta ~ fy(px) if f,k,l (metric) are not provided.
    """
    try:
        camera_params = {}

        focal_length = float(xmp_data.get('Xmp.drone-dji.CalibratedFocalLength', '0.0'))
        optical_center_x = float(xmp_data.get('Xmp.drone-dji.CalibratedOpticalCenterX', '0.0'))
        optical_center_y = float(xmp_data.get('Xmp.drone-dji.CalibratedOpticalCenterY', '0.0'))

        dewarp = xmp_data.get('Xmp.drone-dji.DewarpData', '') # 'Xmp.drone-dji.DewarpData' = time, fx, fy, cx, cy, k1, k2, p1, p2, k3  fx, fy - Calibrated focal length (unit: pixel)
        if dewarp:
            # DewarpData format: "YYYY-MM-DD;fx,fy,cx,cy,k1,k2,p1,p2,k3"
            vals_str = dewarp.split(';')[1] if ';' in dewarp else dewarp
            nums = [float(x.strip()) for x in vals_str.split(',') if x.strip() != '']
            if len(nums) >= 9:
                fx, fy, cx_off, cy_off, k1, k2, p1, p2, k3 = nums[:9]
                cx_abs = optical_center_x + cx_off
                cy_abs = optical_center_y + cy_off

                camera_params['fx_fy'] = [fx, fy]
                camera_params['optical_center_x_y'] = [optical_center_x, optical_center_y]
                camera_params['cx_off_cy_off'] = [cx_off, cy_off]
                camera_params['cx_cy'] = [cx_abs, cy_abs]
                camera_params['distortion'] = [k1, k2, p1, p2, k3]

                theta_deg = 90.0
                theta = np.deg2rad(theta_deg)
                sin_t = 1 # np.sin(theta) 
                cos_t = 0 # np.cos(theta)
                cot_t = (cos_t / sin_t) # 0
                K = np.array([
                    [fx,  -fx * cot_t, cx_abs ],
                    [0.0, fy / sin_t,  cy_abs ],
                    [0.0, 0.0,         1.0    ]
                ], dtype=float)

                camera_params['K'] = K
        
        return camera_params
    except (ValueError, KeyError):
        return None

def parse_dji_gimbal_direction(xmp_data: Dict[str, str]) -> Optional[Dict[str, float]]:
    """Parse DJI gimbal direction from XMP metadata.

    Args:
        xmp_data: XMP metadata dictionary

    Returns:
        Dictionary containing gimbal direction data
        Format: {
            'roll': float,
            'pitch': float,
            'yaw': float,
            'heading': float  # Gimbal heading angle (relative to true north)
        }
    """
    try:
        gimbal_data = {}
        
        gimbal_data['roll'] = float(xmp_data.get('Xmp.drone-dji.GimbalRollDegree', '0.0'))
        gimbal_data['pitch'] = float(xmp_data.get('Xmp.drone-dji.GimbalPitchDegree', '0.0'))
        gimbal_data['yaw'] = float(xmp_data.get('Xmp.drone-dji.GimbalYawDegree', '0.0'))
        
        # Calculate gimbal heading (aircraft heading + gimbal yaw angle)
        flight_yaw = float(xmp_data.get('Xmp.drone-dji.FlightYawDegree', '0.0'))
        gimbal_heading = (flight_yaw + gimbal_data['yaw']) % 360
        gimbal_data['heading'] = gimbal_heading
        
        return gimbal_data
    except (ValueError, KeyError):
        return None

def parse_dji_flight_direction(xmp_data: Dict[str, str]) -> Optional[Dict[str, float]]:
    """Parse DJI flight direction from XMP metadata.

    Args:
        xmp_data: XMP metadata dictionary

    Returns:
        Dictionary containing flight direction data
        Format: {
            'roll': float,
            'pitch': float,
            'yaw': float,
            'heading': float,  # Aircraft heading angle (relative to true north)
            'compass_heading': str  # Compass direction (e.g., "N", "NE", "E", etc.)
        }
    """
    try:
        flight_data = {}
        
        flight_data['roll'] = float(xmp_data.get('Xmp.drone-dji.FlightRollDegree', '0.0'))
        flight_data['pitch'] = float(xmp_data.get('Xmp.drone-dji.FlightPitchDegree', '0.0'))
        flight_data['yaw'] = float(xmp_data.get('Xmp.drone-dji.FlightYawDegree', '0.0'))
        flight_data['heading'] = flight_data['yaw']
        
        # Convert to compass direction
        flight_data['compass_heading'] = degrees_to_compass(flight_data['heading'])
        
        return flight_data
    except (ValueError, KeyError):
        return None


def parse_dji_flight_speed(xmp_data: Dict[str, str]) -> Optional[Dict[str, float]]:
    """Parse DJI flight speed from XMP metadata.

    Args:
        xmp_data: XMP metadata dictionary

    Returns:
        Dictionary containing flight speed data
        Format: {
            'x_speed': float,  # East-west velocity (m/s)
            'y_speed': float,  # North-south velocity (m/s)
            'z_speed': float,  # Vertical velocity (m/s)
            'horizontal_speed': float,  # Horizontal speed (m/s)
            'total_speed': float,  # Total speed (m/s)
            'speed_kmh': float  # Total speed (km/h)
        }
    """
    try:
        speed_data = {}
        
        # Extract speed components from XMP data (unit: cm/s, need to convert to m/s)
        x_speed_cms = float(xmp_data.get('Xmp.drone-dji.FlightXSpeed', '0.0'))
        y_speed_cms = float(xmp_data.get('Xmp.drone-dji.FlightYSpeed', '0.0'))
        z_speed_cms = float(xmp_data.get('Xmp.drone-dji.FlightZSpeed', '0.0'))
        
        # Convert to m/s
        speed_data['x_speed'] = x_speed_cms / 100.0
        speed_data['y_speed'] = y_speed_cms / 100.0
        speed_data['z_speed'] = z_speed_cms / 100.0
        
        # Calculate horizontal and total speed
        speed_data['horizontal_speed'] = np.sqrt(speed_data['x_speed']**2 + speed_data['y_speed']**2)
        speed_data['total_speed'] = np.sqrt(speed_data['x_speed']**2 + speed_data['y_speed']**2 + speed_data['z_speed']**2)
        
        # Convert to km/h
        speed_data['speed_kmh'] = speed_data['total_speed'] * 3.6
        
        return speed_data
    except (ValueError, KeyError):
        return None


def degrees_to_compass(degrees: float) -> str:
    """Convert degrees to compass direction.

    Args:
        degrees: Angle in degrees (0-360)

    Returns:
        Compass direction string
    """
    # Normalize angle to 0-360 range
    degrees = degrees % 360
    
    # 16-direction compass
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    
    # Each direction covers 22.5 degrees
    index = round(degrees / 22.5) % 16
    return directions[index]

def extract_exif_data(image_path: Path) -> Optional[Dict[str, Any]]:
    """Extract EXIF data using piexif.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing EXIF data or None if extraction fails
    """
    try:
        exif_dict = piexif.load(str(image_path))
        
        result = {}
        for ifd_name in ["0th", "Exif", "GPS", "1st"]:
            if ifd_name in exif_dict:
                ifd_data = {}
                for tag, value in exif_dict[ifd_name].items():
                    tag_name = piexif.TAGS[ifd_name].get(tag, {}).get("name", f"Unknown_{tag}")
                    ifd_data[tag_name] = value
                result[ifd_name] = ifd_data
        
        return result
    except Exception as e:
        print(f"Error extracting EXIF from {image_path}: {e}")
        return None


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """Convert Euler angles to rotation matrix using SciPy.

    Args:
        roll: Roll angle
        pitch: Pitch angle  
        yaw: Yaw angle
        degrees: Whether angles are in degrees (True) or radians (False)

    Returns:
        3x3 rotation matrix
    """
    # Create rotation object from Euler angles (intrinsic ZYX order)
    r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=degrees)
    return r.as_matrix()


def euler_to_quaternion(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """Convert Euler angles to quaternion using SciPy.

    Args:
        roll: Roll angle
        pitch: Pitch angle
        yaw: Yaw angle
        degrees: Whether angles are in degrees (True) or radians (False)

    Returns:
        Quaternion as numpy array [x, y, z, w] (SciPy convention)
    """
    # Create rotation object from Euler angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    return r.as_quat()  # Returns [x, y, z, w]


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion using SciPy.

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Quaternion as numpy array [x, y, z, w] (SciPy convention)
    """
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix using SciPy.

    Args:
        q: Quaternion as numpy array [x, y, z, w] (SciPy convention)

    Returns:
        3x3 rotation matrix
    """
    r = R.from_quat(q)
    return r.as_matrix()


def quaternion_to_euler(q: np.ndarray, degrees: bool = True) -> tuple[float, float, float]:
    """Convert quaternion to Euler angles using SciPy.

    Args:
        q: Quaternion as numpy array [x, y, z, w] (SciPy convention)
        degrees: Whether to return angles in degrees (True) or radians (False)

    Returns:
        Tuple of (roll, pitch, yaw) angles
    """
    r = R.from_quat(q)
    euler_angles = r.as_euler('xyz', degrees=degrees)
    return euler_angles[0], euler_angles[1], euler_angles[2]  # roll, pitch, yaw


def rotation_matrix_to_euler(rotation_matrix: np.ndarray, degrees: bool = True) -> tuple[float, float, float]:
    """Convert rotation matrix to Euler angles using SciPy.

    Args:
        rotation_matrix: 3x3 rotation matrix
        degrees: Whether to return angles in degrees (True) or radians (False)

    Returns:
        Tuple of (roll, pitch, yaw) angles
    """
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=degrees)
    return euler_angles[0], euler_angles[1], euler_angles[2]  # roll, pitch, yaw


def quaternion_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w, x, y, z] format to SciPy's [x, y, z, w] format.

    Args:
        q_wxyz: Quaternion in [w, x, y, z] format

    Returns:
        Quaternion in [x, y, z, w] format (SciPy convention)
    """
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def quaternion_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from SciPy's [x, y, z, w] format to [w, x, y, z] format.

    Args:
        q_xyzw: Quaternion in [x, y, z, w] format (SciPy convention)

    Returns:
        Quaternion in [w, x, y, z] format
    """
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def create_rotation_object(roll: float, pitch: float, yaw: float, degrees: bool = True) -> R:
    """Create a SciPy Rotation object from Euler angles.

    Args:
        roll: Roll angle
        pitch: Pitch angle
        yaw: Yaw angle
        degrees: Whether angles are in degrees (True) or radians (False)

    Returns:
        SciPy Rotation object
    """
    return R.from_euler('ZYX', [yaw, pitch, roll], degrees=degrees)

def parse_xmp_tags(image_path):
    """Parse XMP metadata and return key parameters using PIL"""
    meta = extract_img_metadata(image_path)
    if not meta:
        return None

    # parse camera name from exif
    exif_0th = (meta.get('exif') or {}).get('0th', {})
    make_bytes = exif_0th.get('Make', b'')
    model_bytes = exif_0th.get('Model', b'')
    make = make_bytes.decode('utf-8', errors='ignore').strip('\x00').strip() if isinstance(make_bytes, bytes) else str(make_bytes).strip()
    model = model_bytes.decode('utf-8', errors='ignore').strip('\x00').strip() if isinstance(model_bytes, bytes) else str(model_bytes).strip()
    camera_name = f"{make} {model}".strip() or None

    raw = meta.get('raw_xmp') or {}
    gps = meta.get('gps') or {}
    gimbal = meta.get('gimbal_direction') or {}
    cam = meta.get('camera') or {}
    width_height = meta.get('width_height', [0, 0]) 

    xmp_data = {
        "camera_name": camera_name,
        "width_height": width_height,
        "roll": float(gimbal.get("roll", 0.0)),
        "pitch": float(gimbal.get("pitch", 0.0)),
        "yaw": float(gimbal.get("yaw", 0.0)),
        "latitude": float(gps.get("latitude", 0.0)),
        "longitude": float(gps.get("longitude", 0.0)),
        "altitude": float(gps.get("absolute_altitude", 0.0)),
        "cam_reverse": int(raw.get("Xmp.drone-dji.CamReverse", 0)),
        "gimbal_reverse": int(raw.get("Xmp.drone-dji.GimbalReverse", 0)),
        "dewarp_flag": int(raw.get("Xmp.drone-dji.DewarpFlag", 0)),
        "dewarp_data": raw.get("Xmp.drone-dji.DewarpData", ""),
    }

    # parse dewarp_data to list of floats, get all the floats
    # e.g. 2025-03-10;3701.352827703056,3701.352827703056,50.905173817603,-26.805788363656,-6.142148585377,13.596736300008,-0.000047240461,-0.000131949051,-4.134061689205,-6.054688992297,13.094564369611,-3.110503344184
    # Ignore the beginning date (before the first semicolon)
    dewarp_str = xmp_data["dewarp_data"]
    if dewarp_str and ";" in dewarp_str:
        dewarp_str = dewarp_str.split(";", 1)[1]
        dewarp_data = dewarp_str.split(",")
        xmp_data["dewarp_data"] = [float(x) for x in dewarp_data if x.strip()]
    else:
        xmp_data["dewarp_data"] = []

    cx, cy = cam.get("cx_cy")[0], cam.get("cx_cy")[1]
    xmp_data["dewarp_data"][2] = float(cx)
    xmp_data["dewarp_data"][3] = float(cy)

    return xmp_data


def extract_img_metadata(image_path: Path) -> Optional[Dict[str, Any]]:
    """Extract all DJI data from image XMP metadata.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing all extracted DJI data or None if extraction fails
    """
    xmp_data = extract_xmp_from_image(image_path)
    if not xmp_data:
        print("No XMP data found")
        return None
    
    with PILImage.open(str(image_path)) as img:
        width, height = img.size
    
    result = {
        'raw_xmp': xmp_data,
        'width_height': [width, height],
        'gps': parse_dji_gps(xmp_data),
        'camera': parse_dji_camera_params(xmp_data),
        'gimbal_direction': parse_dji_gimbal_direction(xmp_data),  # Added: Gimbal direction
        'flight_direction': parse_dji_flight_direction(xmp_data),  # Added: Flight direction
        'gimbal_direction_attitude': parse_dji_attitude(xmp_data),
        'flight_direction_attitude': parse_dji_attitude(xmp_data),
        'flight_speed': parse_dji_flight_speed(xmp_data),  # Added: Flight speed
        'exif': extract_exif_data(image_path)
    }
    
    # Add rotation matrices and quaternions if attitude data is available
    if result['gimbal_direction_attitude']:
        for attitude_type in ['gimbal', 'flight']:
            if attitude_type in result['gimbal_direction_attitude']:
                att_data = result['gimbal_direction_attitude'][attitude_type]
                roll, pitch, yaw = att_data['roll'], att_data['pitch'], att_data['yaw']
                
                # Create SciPy Rotation object, reorder to ZYX order
                rotation_obj = create_rotation_object(roll, pitch, yaw)
                
                # Convert to rotation matrix and quaternion
                result['gimbal_direction_attitude'][attitude_type]['rotation_matrix'] = rotation_obj.as_matrix()
                result['gimbal_direction_attitude'][attitude_type]['quaternion_xyzw'] = rotation_obj.as_quat()  # SciPy format
                result['gimbal_direction_attitude'][attitude_type]['quaternion_wxyz'] = quaternion_xyzw_to_wxyz(rotation_obj.as_quat())  # Traditional format
                result['gimbal_direction_attitude'][attitude_type]['rotation_object'] = rotation_obj  # Store the rotation object itself
    
    # Add flight direction attitude data
    if result['flight_direction_attitude']:
        for attitude_type in ['flight']:
            if attitude_type in result['flight_direction_attitude']:
                att_data = result['flight_direction_attitude'][attitude_type]
                roll, pitch, yaw = att_data['roll'], att_data['pitch'], att_data['yaw']
                
                # Create SciPy Rotation object
                rotation_obj = create_rotation_object(roll, pitch, yaw)
                
                # Convert to rotation matrix and quaternion
                result['flight_direction_attitude'][attitude_type]['rotation_matrix'] = rotation_obj.as_matrix()
                result['flight_direction_attitude'][attitude_type]['quaternion_xyzw'] = quaternion_wxyz_to_xyzw(rotation_obj.as_quat())  # SciPy format
                result['flight_direction_attitude'][attitude_type]['quaternion_wxyz'] = rotation_obj.as_quat()  # Traditional format
                result['flight_direction_attitude'][attitude_type]['rotation_object'] = rotation_obj  # Store the rotation object itself

    return result

if __name__ == "__main__":
    # Extract all data
    image_path = Path(r"D:\Github_code\vggt\examples\test_data\001-20250715_151008\images\L1_0000.jpg")
    data = extract_img_metadata(image_path)
    
    if data:
        print("Successfully extracted data:")
        print(f"XMP data: {len(data.get('raw_xmp', {})) if data.get('raw_xmp') else 0} fields")
        print(f"GPS data: {'✓' if data.get('gps') else '✗'}")
        print(f"Camera parameters: {'✓' if data.get('camera') else '✗'}")
        print(f"Attitude data: {'✓' if data.get('gimbal_direction_attitude') else '✗'}")
        print(f"Gimbal direction: {'✓' if data.get('gimbal_direction') else '✗'}")
        print(f"Flight direction: {'✓' if data.get('flight_direction') else '✗'}")
        print(f"Flight speed: {'✓' if data.get('flight_speed') else '✗'}")
        print(f"EXIF data: {'✓' if data.get('exif') else '✗'}")
        print("\nDetailed data:")
        for key, value in data.items():
            if key != 'raw_xmp':  # Skip raw XMP data display
                print(f"{key}: {value}")
        
        # Display raw XMP data (for debugging)
        if data.get('raw_xmp'):
            print("\nRaw XMP data:")
            for key, value in data['raw_xmp'].items():
                print(f"  {key}: {value}")
    else:
        print("Unable to extract data")