#!/usr/bin/env python3
"""
预处理畸变矫正影像：基于 image_list.json 重命名 undistort 文件夹中的影像
同时将位置信息、相机姿态和相机参数写入影像的 EXIF/XMP 中
"""

import json
import re
import shutil
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import piexif 
from PIL import Image


def load_image_list(json_path: Path) -> List[Dict]:
    """
    加载 image_list.json 文件
    
    Args:
        json_path: image_list.json 文件路径
        
    Returns:
        图像信息列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除非法字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        清理后的文件名
    """
    # Windows 非法字符: < > : " / \ | ? *
    # 同时移除不可见字符和控制字符
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # 移除首尾空格和点
    filename = filename.strip(' .')
    # 如果文件名为空，使用默认名称
    if not filename:
        filename = 'unnamed'
    return filename


def build_id_to_metadata_mapping(image_list: List[Dict]) -> Dict[str, Dict]:
    """
    构建 id（hash）到完整元数据的映射
    
    Args:
        image_list: image_list.json 中的图像信息列表
        
    Returns:
        {id: metadata_dict} 的映射字典
    """
    mapping = {}
    for item in image_list:
        image_id = item.get('id', '')
        if image_id:
            mapping[image_id] = item
    return mapping


def decimal_to_dms(decimal_degrees: float) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    将十进制度数转换为度分秒格式 (用于 EXIF GPS)
    
    Args:
        decimal_degrees: 十进制度数
        
    Returns:
        ((degrees, 1), (minutes, 1), (seconds * 10000, 10000))
    """
    is_negative = decimal_degrees < 0
    decimal_degrees = abs(decimal_degrees)
    
    degrees = int(decimal_degrees)
    minutes_decimal = (decimal_degrees - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60
    
    # 使用高精度表示秒
    seconds_num = int(seconds * 10000)
    seconds_den = 10000
    
    return ((degrees, 1), (minutes, 1), (seconds_num, seconds_den))


def float_to_rational(value: float, precision: int = 1000000) -> Tuple[int, int]:
    """
    将浮点数转换为有理数格式 (用于 EXIF)
    
    Args:
        value: 浮点数值
        precision: 精度（分母）
        
    Returns:
        (numerator, denominator)
    """
    return (int(value * precision), precision)


def build_xmp_packet(metadata: Dict) -> bytes:
    """
    构建 XMP 数据包，包含 DJI 无人机格式的元数据
    
    Args:
        metadata: 元数据字典
        
    Returns:
        XMP 数据包的字节串
    """
    # 获取各种元数据
    gps = metadata.get('gps', {})
    orientation = metadata.get('orientation', [])
    pre_calib_param = metadata.get('pre_calib_param', [])
    width = metadata.get('width', 0)
    height = metadata.get('height', 0)
    relative_height = metadata.get('relative_height', 0)
    camera_maker = metadata.get('camera_maker', 'DJI')
    camera_model = metadata.get('camera_model', '')
    focal_length = metadata.get('focal_length', 0)
    
    # GPS 数据
    lat = gps.get('lat', 0)
    lng = gps.get('lng', 0)
    alt = gps.get('altitude', 0)
    
    # 姿态数据 [yaw, pitch, roll]
    yaw = orientation[0] if len(orientation) > 0 else 0
    pitch = orientation[1] if len(orientation) > 1 else 0
    roll = orientation[2] if len(orientation) > 2 else 0
    
    # 相机内参 [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    fx = pre_calib_param[0] if len(pre_calib_param) > 0 else 0
    fy = pre_calib_param[1] if len(pre_calib_param) > 1 else 0
    cx_offset = pre_calib_param[2] if len(pre_calib_param) > 2 else 0
    cy_offset = pre_calib_param[3] if len(pre_calib_param) > 3 else 0
    
    # 计算实际主点坐标（相对于图像中心的偏移量 -> 绝对像素坐标）
    cx_actual = width / 2 + cx_offset if width > 0 else cx_offset
    cy_actual = height / 2 + cy_offset if height > 0 else cy_offset
    
    # 畸变参数
    k1 = pre_calib_param[4] if len(pre_calib_param) > 4 else 0
    k2 = pre_calib_param[5] if len(pre_calib_param) > 5 else 0
    p1 = pre_calib_param[6] if len(pre_calib_param) > 6 else 0
    p2 = pre_calib_param[7] if len(pre_calib_param) > 7 else 0
    k3 = pre_calib_param[8] if len(pre_calib_param) > 8 else 0
    
    # 构建 DewarpData 字符串（DJI 格式：日期;参数数据）
    # 尝试从 capture_time 获取日期，否则使用当前日期
    capture_time = metadata.get('capture_time', 0)
    if capture_time and isinstance(capture_time, (int, float)) and capture_time > 0:
        date_str = datetime.fromtimestamp(capture_time).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
    dewarp_data = f"{date_str};{fx},{fy},{cx_offset},{cy_offset},{k1},{k2},{p1},{p2},{k3}"
    
    # 构建相机内参 JSON 字符串
    camera_params_json = json.dumps({
        "fx": fx, "fy": fy,
        "cx": cx_actual, "cy": cy_actual,
        "cx_offset": cx_offset, "cy_offset": cy_offset,
        "k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3,
        "width": width, "height": height
    }, ensure_ascii=False)
    
    # 构建 XMP XML
    xmp_content = f'''<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
      xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
      drone-dji:Make="{camera_maker}"
      drone-dji:Model="{camera_model}"
      drone-dji:GimbalYawDegree="{yaw:.6f}"
      drone-dji:GimbalPitchDegree="{pitch:.6f}"
      drone-dji:GimbalRollDegree="{roll:.6f}"
      drone-dji:FlightYawDegree="{yaw:.6f}"
      drone-dji:FlightPitchDegree="0.0"
      drone-dji:FlightRollDegree="0.0"
      drone-dji:AbsoluteAltitude="{alt:.6f}"
      drone-dji:RelativeAltitude="{relative_height:.6f}"
      drone-dji:GpsLatitude="{lat:.8f}"
      drone-dji:GpsLongitude="{lng:.8f}"
      drone-dji:DewarpData="{dewarp_data}"
      drone-dji:DewarpFlag="0"
      drone-dji:CalibratedFocalLength="{focal_length:.6f}"
      drone-dji:CalibratedOpticalCenterX="{cx_actual:.6f}"
      drone-dji:CalibratedOpticalCenterY="{cy_actual:.6f}"
      tiff:Make="{camera_maker}"
      tiff:Model="{camera_model}"
      tiff:ImageWidth="{width}"
      tiff:ImageLength="{height}"
      exif:FocalLength="{focal_length:.2f}"
      exif:GPSLatitude="{abs(lat):.8f}"
      exif:GPSLatitudeRef="{'N' if lat >= 0 else 'S'}"
      exif:GPSLongitude="{abs(lng):.8f}"
      exif:GPSLongitudeRef="{'E' if lng >= 0 else 'W'}"
      exif:GPSAltitude="{abs(alt):.6f}"
      exif:GPSAltitudeRef="{'0' if alt >= 0 else '1'}"
      crs:CameraIntrinsics="{camera_params_json}">
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
    
    return xmp_content.encode('utf-8')


def insert_xmp_into_jpeg(image_path: Path, xmp_data: bytes) -> bool:
    """
    将 XMP 数据插入到 JPEG 文件中
    
    Args:
        image_path: JPEG 文件路径
        xmp_data: XMP 数据字节串
        
    Returns:
        是否成功插入
    """
    try:
        with open(image_path, 'rb') as f:
            jpeg_data = f.read()
        
        # 检查是否为 JPEG 文件
        if jpeg_data[:2] != b'\xff\xd8':
            print(f"    ⚠️  不是有效的 JPEG 文件")
            return False
        
        # 查找现有的 XMP 段并移除
        new_jpeg_data = remove_existing_xmp(jpeg_data)
        
        # 构建 XMP APP1 段
        # APP1 标记: FF E1
        # 长度: 2 bytes (包含长度字段本身)
        # 命名空间: "http://ns.adobe.com/xap/1.0/\x00"
        xmp_namespace = b'http://ns.adobe.com/xap/1.0/\x00'
        xmp_segment_data = xmp_namespace + xmp_data
        segment_length = len(xmp_segment_data) + 2  # +2 for length field
        
        if segment_length > 65535:
            print(f"    ⚠️  XMP 数据过大，无法写入")
            return False
        
        # 构建 APP1 段
        app1_marker = b'\xff\xe1'
        length_bytes = struct.pack('>H', segment_length)
        xmp_segment = app1_marker + length_bytes + xmp_segment_data
        
        # 在 SOI 标记后插入 XMP 段（在其他 APP 段之前）
        # 找到第一个非 APP0/APP1 段的位置
        insert_pos = 2  # 跳过 SOI (FF D8)
        
        # 跳过现有的 APP0 (JFIF) 段
        while insert_pos < len(new_jpeg_data) - 1:
            if new_jpeg_data[insert_pos:insert_pos+2] == b'\xff\xe0':  # APP0
                segment_len = struct.unpack('>H', new_jpeg_data[insert_pos+2:insert_pos+4])[0]
                insert_pos += 2 + segment_len
            else:
                break
        
        # 插入 XMP 段
        final_jpeg_data = new_jpeg_data[:insert_pos] + xmp_segment + new_jpeg_data[insert_pos:]
        
        # 写回文件
        with open(image_path, 'wb') as f:
            f.write(final_jpeg_data)
        
        return True
        
    except Exception as e:
        print(f"    ⚠️  插入 XMP 失败: {e}")
        return False


def remove_existing_xmp(jpeg_data: bytes) -> bytes:
    """
    从 JPEG 数据中移除现有的 XMP 段
    
    Args:
        jpeg_data: JPEG 文件数据
        
    Returns:
        移除 XMP 后的 JPEG 数据
    """
    result = bytearray()
    pos = 0
    
    # 复制 SOI
    result.extend(jpeg_data[:2])
    pos = 2
    
    while pos < len(jpeg_data) - 1:
        # 检查是否为标记
        if jpeg_data[pos] != 0xFF:
            result.extend(jpeg_data[pos:])
            break
        
        marker = jpeg_data[pos:pos+2]
        
        # 检查是否到达图像数据
        if marker == b'\xff\xda':  # SOS (Start of Scan)
            result.extend(jpeg_data[pos:])
            break
        
        # 检查是否为 APP1 段
        if marker == b'\xff\xe1':
            segment_len = struct.unpack('>H', jpeg_data[pos+2:pos+4])[0]
            segment_data = jpeg_data[pos+4:pos+2+segment_len]
            
            # 检查是否为 XMP 数据
            if segment_data.startswith(b'http://ns.adobe.com/xap/1.0/'):
                # 跳过此 XMP 段
                pos += 2 + segment_len
                continue
        
        # 复制其他段
        if marker[1] in [0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9]:
            # 无长度字段的标记
            result.extend(marker)
            pos += 2
        else:
            # 有长度字段的标记
            segment_len = struct.unpack('>H', jpeg_data[pos+2:pos+4])[0]
            result.extend(jpeg_data[pos:pos+2+segment_len])
            pos += 2 + segment_len
    
    return bytes(result)


def write_exif_metadata(image_path: Path, metadata: Dict, verbose: bool = False) -> bool:
    """
    将元数据写入影像的 EXIF
    
    Args:
        image_path: 影像文件路径
        metadata: 元数据字典
        verbose: 是否输出详细信息
        
    Returns:
        是否成功写入
    """
    
    try:
        # 读取现有 EXIF 或创建新的
        try:
            exif_dict = piexif.load(str(image_path))
        except:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        # ============ 1. 写入 GPS 信息 ============
        gps = metadata.get('gps', {})
        lat = 0
        lng = 0
        alt = 0
        if gps:
            lat = gps.get('lat', 0)
            lng = gps.get('lng', 0)
            alt = gps.get('altitude', 0)
            
            # 纬度
            lat_dms = decimal_to_dms(lat)
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = lat_dms
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
            
            # 经度
            lng_dms = decimal_to_dms(lng)
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = lng_dms
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = 'E' if lng >= 0 else 'W'
            
            # 海拔
            if alt >= 0:
                exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = float_to_rational(alt)
                exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef] = 0  # 0 = 海平面以上
            else:
                exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = float_to_rational(abs(alt))
                exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef] = 1  # 1 = 海平面以下
        
        # ============ 2. 写入基本图像信息 ============
        width = metadata.get('width', 0)
        height = metadata.get('height', 0)
        if width > 0:
            exif_dict['0th'][piexif.ImageIFD.ImageWidth] = width
        if height > 0:
            exif_dict['0th'][piexif.ImageIFD.ImageLength] = height
        
        # 相机制造商和型号
        camera_maker = metadata.get('camera_maker', '')
        camera_model = metadata.get('camera_model', '')
        if camera_maker:
            exif_dict['0th'][piexif.ImageIFD.Make] = camera_maker.encode('utf-8')
        if camera_model:
            exif_dict['0th'][piexif.ImageIFD.Model] = camera_model.encode('utf-8')
        
        # ============ 3. 写入焦距信息 ============
        focal_length = metadata.get('focal_length', 0)
        focal_length_35mm = metadata.get('focal_length_in_35mm', 0)
        
        if focal_length > 0:
            exif_dict['Exif'][piexif.ExifIFD.FocalLength] = float_to_rational(focal_length)
        if focal_length_35mm > 0:
            exif_dict['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(focal_length_35mm)
        
        # ============ 4. 写入图像尺寸到 Exif ============
        if width > 0:
            exif_dict['Exif'][piexif.ExifIFD.PixelXDimension] = width
        if height > 0:
            exif_dict['Exif'][piexif.ExifIFD.PixelYDimension] = height
        
        # ============ 5. 写入用户注释（存储相机内参和姿态） ============
        # 将相机参数和姿态信息存储在 UserComment 中
        user_comment_data = {}
        
        # 相机内参 pre_calib_param: [fx, fy, cx_offset, cy_offset, k1, k2, p1, p2, k3]
        # 注意: cx, cy 是相对于图像中心的偏移量
        pre_calib_param = metadata.get('pre_calib_param', [])
        if pre_calib_param and len(pre_calib_param) >= 4:
            fx = pre_calib_param[0]
            fy = pre_calib_param[1]
            cx_offset = pre_calib_param[2]
            cy_offset = pre_calib_param[3]
            
            # 计算实际主点坐标
            cx_actual = width / 2 + cx_offset if width > 0 else cx_offset
            cy_actual = height / 2 + cy_offset if height > 0 else cy_offset
            
            user_comment_data['camera_intrinsics'] = {
                'fx': fx,
                'fy': fy,
                'cx': cx_actual,  # 实际主点 X 坐标（像素）
                'cy': cy_actual,  # 实际主点 Y 坐标（像素）
                'cx_offset': cx_offset,  # 相对于图像中心的偏移
                'cy_offset': cy_offset,
            }
            if len(pre_calib_param) >= 9:
                user_comment_data['distortion'] = {
                    'k1': pre_calib_param[4],
                    'k2': pre_calib_param[5],
                    'p1': pre_calib_param[6],
                    'p2': pre_calib_param[7],
                    'k3': pre_calib_param[8],
                }
        
        # 相机姿态 orientation: [yaw, pitch, roll]
        orientation = metadata.get('orientation', [])
        if orientation and len(orientation) >= 3:
            user_comment_data['orientation'] = {
                'yaw': orientation[0],
                'pitch': orientation[1],
                'roll': orientation[2],
                'type': metadata.get('orientation_type', 'YPR')
            }
        
        # 其他有用信息
        if metadata.get('relative_height'):
            user_comment_data['relative_height'] = metadata['relative_height']
        if metadata.get('capture_time'):
            user_comment_data['capture_time'] = metadata['capture_time']
        if metadata.get('rtk_flag'):
            user_comment_data['rtk_flag'] = metadata['rtk_flag']
        if metadata.get('pos_sigma'):
            user_comment_data['pos_sigma'] = metadata['pos_sigma']
        if metadata.get('rtk_std'):
            user_comment_data['rtk_std'] = metadata['rtk_std']
        
        # 将用户注释数据转为 JSON 并写入
        if user_comment_data:
            comment_json = json.dumps(user_comment_data, ensure_ascii=False)
            # UserComment 格式: charset code (8 bytes) + comment
            user_comment = b'ASCII\x00\x00\x00' + comment_json.encode('utf-8')
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = user_comment
        
        # ============ 6. 保存 EXIF ============
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(image_path))
        
        if verbose:
            print(f"    📍 GPS: ({lat:.6f}, {lng:.6f}, {alt:.1f}m)")
            if orientation:
                print(f"    🧭 姿态: Y={orientation[0]:.1f}° P={orientation[1]:.1f}° R={orientation[2]:.1f}°")
            if pre_calib_param and len(pre_calib_param) >= 4:
                cx_actual = width / 2 + pre_calib_param[2] if width > 0 else pre_calib_param[2]
                cy_actual = height / 2 + pre_calib_param[3] if height > 0 else pre_calib_param[3]
                print(f"    📷 内参: fx={pre_calib_param[0]:.2f} fy={pre_calib_param[1]:.2f} cx={cx_actual:.2f} cy={cy_actual:.2f}")
        
        return True
        
    except Exception as e:
        print(f"    ⚠️  写入 EXIF 失败: {e}")
        return False


def write_xmp_metadata(image_path: Path, metadata: Dict, verbose: bool = False) -> bool:
    """
    将元数据写入影像的 XMP（DJI 格式）
    
    Args:
        image_path: 影像文件路径
        metadata: 元数据字典
        verbose: 是否输出详细信息
        
    Returns:
        是否成功写入
    """
    try:
        # 只处理 JPEG 文件
        if image_path.suffix.lower() not in ['.jpg', '.jpeg']:
            if verbose:
                print(f"    ℹ️  跳过 XMP 写入（非 JPEG 文件）")
            return True
        
        # 构建 XMP 数据包
        xmp_data = build_xmp_packet(metadata)
        
        # 插入 XMP 到 JPEG 文件
        success = insert_xmp_into_jpeg(image_path, xmp_data)
        
        if verbose and success:
            orientation = metadata.get('orientation', [])
            pre_calib_param = metadata.get('pre_calib_param', [])
            if orientation or pre_calib_param:
                print(f"    📝 XMP: DJI 格式元数据已写入")
        
        return success
        
    except Exception as e:
        print(f"    ⚠️  写入 XMP 失败: {e}")
        return False


def rename_and_embed_metadata(
    undistort_dir: Path,
    output_dir: Path,
    id_to_metadata: Dict[str, Dict],
    verbose: bool = True
) -> Dict[str, str]:
    """
    重命名 undistort 文件夹中的影像，并将元数据嵌入到输出影像中
    
    Args:
        undistort_dir: undistort 影像所在目录
        output_dir: 重命名后的输出目录
        id_to_metadata: id 到完整元数据的映射
        verbose: 是否输出详细信息
        
    Returns:
        {原文件名: 新文件名} 的映射字典
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    renamed_mapping = {}
    unmatched_files = []
    
    # 获取所有支持的图像格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # 遍历 undistort 目录中的文件
    undistort_files = sorted([
        f for f in undistort_dir.iterdir() 
        if f.is_file() and f.suffix in supported_formats
    ])
    
    if verbose:
        print(f"📁 在 undistort 目录中找到 {len(undistort_files)} 个影像文件")
        print(f"📋 image_list.json 中有 {len(id_to_metadata)} 条记录")
    
    matched_count = 0
    exif_success_count = 0
    xmp_success_count = 0
    
    for undistort_file in undistort_files:
        # 获取文件名（不含扩展名）作为 id
        file_id = undistort_file.stem
        
        if file_id in id_to_metadata:
            metadata = id_to_metadata[file_id]
            
            # 获取原始文件名
            origin_path = metadata.get('origin_path', '') or metadata.get('path', '')
            if not origin_path:
                unmatched_files.append(undistort_file.name)
                continue
                
            original_filename = Path(origin_path).name
            
            # 保持原始扩展名或使用 undistort 文件的扩展名
            original_stem = Path(original_filename).stem
            # 清理文件名，移除非法字符
            clean_stem = sanitize_filename(original_stem)
            new_filename = clean_stem + undistort_file.suffix.lower()
            
            new_path = output_dir / new_filename
            
            # 复制文件（保留原始文件）
            try:
                # 使用 shutil.copy 而非 copy2，避免元数据复制问题
                shutil.copy(undistort_file, new_path)
            except OSError as e:
                print(f"    ⚠️  复制文件失败: {e}")
                print(f"       源文件: {undistort_file}")
                print(f"       目标文件: {new_path}")
                unmatched_files.append(undistort_file.name)
                continue
            
            if verbose:
                print(f"  ✓ {undistort_file.name} -> {new_filename}")
            
            # 注意：必须先写入 XMP，再写入 EXIF
            # 因为 XMP 写入会重构 JPEG 文件结构，如果先写 EXIF 会被覆盖
            
            # 1. 先写入 XMP 元数据（DJI 格式，包含姿态和相机内参）
            if write_xmp_metadata(new_path, metadata, verbose=verbose):
                xmp_success_count += 1
            
            # 2. 再写入 EXIF 元数据（piexif 会正确处理已有的 APP1 段）
            if write_exif_metadata(new_path, metadata, verbose=verbose):
                exif_success_count += 1
            
            renamed_mapping[undistort_file.name] = new_filename
            matched_count += 1
        else:
            unmatched_files.append(undistort_file.name)
            if verbose:
                print(f"  ✗ 未找到匹配: {undistort_file.name}")
    
    # 输出统计信息
    if verbose:
        print(f"\n📊 处理结果:")
        print(f"  - 成功匹配: {matched_count} 个文件")
        print(f"  - XMP 写入成功: {xmp_success_count} 个文件")
        print(f"  - EXIF 写入成功: {exif_success_count} 个文件")
        print(f"  - 未匹配: {len(unmatched_files)} 个文件")
        
        if unmatched_files:
            print(f"\n⚠️  未匹配的文件 (前10个):")
            for f in unmatched_files[:10]:
                print(f"    - {f}")
    
    return renamed_mapping


def save_rename_mapping(mapping: Dict[str, str], output_path: Path):
    """
    保存重命名映射到 JSON 文件
    
    Args:
        mapping: 重命名映射字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def save_metadata_summary(id_to_metadata: Dict[str, Dict], output_path: Path):
    """
    保存所有影像的元数据摘要到 JSON 文件
    
    Args:
        id_to_metadata: id 到元数据的映射
        output_path: 输出文件路径
    """
    summary = {}
    for image_id, metadata in id_to_metadata.items():
        origin_path = metadata.get('origin_path', '') or metadata.get('path', '')
        filename = Path(origin_path).name if origin_path else image_id
        
        summary[filename] = {
            'id': image_id,
            'gps': metadata.get('gps', {}),
            'orientation': metadata.get('orientation', []),
            'orientation_type': metadata.get('orientation_type', ''),
            'pre_calib_param': metadata.get('pre_calib_param', []),
            'focal_length': metadata.get('focal_length', 0),
            'width': metadata.get('width', 0),
            'height': metadata.get('height', 0),
            'relative_height': metadata.get('relative_height', 0),
            'capture_time': metadata.get('capture_time', 0),
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    """
    主函数：执行影像重命名预处理并嵌入元数据
    """
    # ============ 配置路径 ============
    # 获取脚本所在目录，然后定位到项目根目录
    script_dir = Path(__file__).resolve().parent  # SfM 目录
    project_root = script_dir.parent  # drone-map-anything 目录
    
    # 基础目录
    # preprocessing_dir = project_root / "examples" / "Ganluo_images" / "preprocessing"
    # preprocessing_dir = project_root / "examples" / "SWJTU_7th_teaching_building" / "preprocessing"
    # preprocessing_dir = project_root / "examples" / "HuaPo" / "preprocessing"
    preprocessing_dir = project_root / "examples" / "WenChuan" / "preprocessing"
    
    # image_list.json 路径
    image_list_path = preprocessing_dir / "undistort_images" / "image_list.json"
    
    # undistort 影像目录
    undistort_dir = preprocessing_dir / "undistort_images" / "undistort"
    
    # 重命名后的输出目录
    output_dir = preprocessing_dir / "undistort_images" / "rename"
    
    # ============ 执行处理 ============
    print("=" * 60)
    print("🔧 开始预处理：重命名畸变矫正影像并嵌入元数据")
    print("=" * 60)
    
    # 检查路径是否存在
    if not image_list_path.exists():
        print(f"❌ 错误: image_list.json 不存在: {image_list_path}")
        return
    
    if not undistort_dir.exists():
        print(f"❌ 错误: undistort 目录不存在: {undistort_dir}")
        return
    
    print(f"\n📂 image_list.json: {image_list_path}")
    print(f"📂 undistort 目录: {undistort_dir}")
    print(f"📂 输出目录: {output_dir}")
    print()
    
    # 1. 加载 image_list.json
    print("📖 加载 image_list.json...")
    image_list = load_image_list(image_list_path)
    print(f"   加载了 {len(image_list)} 条记录")
    
    # 2. 构建 id -> 元数据映射
    print("\n🔗 构建 ID 到元数据的映射...")
    id_to_metadata = build_id_to_metadata_mapping(image_list)
    print(f"   创建了 {len(id_to_metadata)} 个映射")
    
    # 3. 重命名影像并嵌入元数据
    print("\n📝 开始重命名影像并嵌入元数据...\n")
    renamed_mapping = rename_and_embed_metadata(
        undistort_dir=undistort_dir,
        output_dir=output_dir,
        id_to_metadata=id_to_metadata,
        verbose=True
    )
    
    # 4. 保存映射结果
    if renamed_mapping:
        mapping_output_path = output_dir / "rename_mapping.json"
        save_rename_mapping(renamed_mapping, mapping_output_path)
        print(f"\n💾 重命名映射已保存到: {mapping_output_path}")
        
        # 保存元数据摘要
        metadata_output_path = output_dir / "metadata_summary.json"
        save_metadata_summary(id_to_metadata, metadata_output_path)
        print(f"💾 元数据摘要已保存到: {metadata_output_path}")
    
    print("\n" + "=" * 60)
    print("✅ 预处理完成!")
    print("=" * 60)
    print("\n📋 嵌入的元数据包括:")
    print("   【EXIF 数据】")
    print("   - GPS 位置 (纬度、经度、海拔)")
    print("   - 相机信息 (制造商、型号)")
    print("   - 焦距信息 (实际焦距、等效35mm焦距)")
    print("   - 图像尺寸 (宽度、高度)")
    print("   - UserComment (JSON 格式的完整相机参数)")
    print("")
    print("   【XMP 数据 (DJI 格式)】")
    print("   - 云台姿态 (GimbalYaw, GimbalPitch, GimbalRoll)")
    print("   - GPS 位置 (GpsLatitude, GpsLongitude, AbsoluteAltitude)")
    print("   - 相对高度 (RelativeAltitude)")
    print("   - 相机内参 (CalibratedFocalLength, OpticalCenterX/Y)")
    print("   - 畸变参数 (DewarpData: fx, fy, cx, cy, k1, k2, p1, p2, k3)")
    print("")
    print("💡 提示: 使用 exiftool 或类似工具可查看完整元数据")


if __name__ == "__main__":
    main()
