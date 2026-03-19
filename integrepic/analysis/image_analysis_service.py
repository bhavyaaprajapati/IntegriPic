"""
Core image analysis service - handles metadata, ELA, and steganography detection
"""
from PIL import Image, ImageChops
from PIL.ExifTags import TAGS
import hashlib
import os
import platform
import tempfile
import logging
from stegano import lsb

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """Service class for core image analysis operations"""
    
    @staticmethod
    def calculate_sha256(image_file):
        """Calculate SHA256 hash of uploaded image file"""
        try:
            image_file.seek(0)  # Reset file pointer
            hash_obj = hashlib.sha256()
            for chunk in iter(lambda: image_file.read(4096), b""):
                hash_obj.update(chunk)
            image_file.seek(0)  # Reset file pointer again
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash: {e}")
            return None
    
    @staticmethod
    def calculate_sha256_from_path(file_path):
        """Calculate SHA256 hash from file path"""
        try:
            hash_obj = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash from path: {e}")
            return None
    
    @staticmethod
    def extract_metadata(image_path):
        """Extract EXIF metadata from image"""
        try:
            with Image.open(image_path) as image:
                exif_data = image._getexif()
                metadata = {}
                if exif_data:
                    for tag, value in exif_data.items():
                        tag_name = TAGS.get(tag, tag)
                        # Convert value to string for JSON serialization
                        metadata[str(tag_name)] = str(value)
                return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    @staticmethod
    def extract_detailed_exif(image_path):
        """Extract detailed EXIF data including GPS and device information"""
        try:
            with Image.open(image_path) as img:
                exif_dict = {}
                exif_data = img._getexif()
                
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        # Convert bytes to string if needed
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except:
                                value = str(value)
                        
                        exif_dict[tag] = value
                    
                    # Extract specific useful information
                    extracted = {
                        'camera_make': exif_dict.get('Make', 'Unknown'),
                        'camera_model': exif_dict.get('Model', 'Unknown'),
                        'datetime': exif_dict.get('DateTime', 'Unknown'),
                        'software': exif_dict.get('Software', 'Unknown'),
                        'gps_info': ImageAnalysisService._extract_gps_info(exif_dict.get('GPSInfo', {})),
                        'flash_used': exif_dict.get('Flash', 'Unknown'),
                        'focal_length': exif_dict.get('FocalLength', 'Unknown'),
                        'exposure_time': exif_dict.get('ExposureTime', 'Unknown'),
                        'f_number': exif_dict.get('FNumber', 'Unknown'),
                        'iso_speed': exif_dict.get('ISOSpeedRatings', 'Unknown'),
                        'white_balance': exif_dict.get('WhiteBalance', 'Unknown'),
                        'orientation': exif_dict.get('Orientation', 'Unknown')
                    }
                    
                    return extracted
                
                return {'message': 'No EXIF data found'}
                
        except Exception as e:
            logger.error(f"Error extracting detailed EXIF data: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _extract_gps_info(gps_info):
        """Extract GPS coordinates from EXIF GPS info"""
        try:
            if not gps_info:
                return {'status': 'No GPS data found'}
            
            def convert_to_degrees(value):
                """Convert GPS coordinates to decimal degrees"""
                if not value or len(value) != 3:
                    return 0
                d, m, s = value
                return float(d) + float(m)/60 + float(s)/3600
            
            gps_data = {}
            
            # Get latitude
            if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                lat = convert_to_degrees(gps_info['GPSLatitude'])
                if gps_info['GPSLatitudeRef'] == 'S':
                    lat = -lat
                gps_data['latitude'] = round(lat, 6)
            
            # Get longitude
            if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                lon = convert_to_degrees(gps_info['GPSLongitude'])
                if gps_info['GPSLongitudeRef'] == 'W':
                    lon = -lon
                gps_data['longitude'] = round(lon, 6)
            
            # Get altitude
            if 'GPSAltitude' in gps_info:
                altitude = float(gps_info['GPSAltitude'])
                if 'GPSAltitudeRef' in gps_info and gps_info['GPSAltitudeRef'] == 1:
                    altitude = -altitude
                gps_data['altitude'] = round(altitude, 2)
            
            # Get timestamp
            if 'GPSTimeStamp' in gps_info:
                gps_data['timestamp'] = str(gps_info['GPSTimeStamp'])
            
            if gps_data:
                return gps_data
            else:
                return {'status': 'GPS data present but could not be parsed'}
                
        except Exception as e:
            logger.error(f"Error extracting GPS info: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def perform_ela_analysis(image_path, quality=95):
        """Perform Error Level Analysis on image - returns analysis data instead of image files"""
        try:
            # Check if it's a JPEG image
            if not image_path.lower().endswith(('.jpg', '.jpeg')):
                logger.warning("ELA works best with JPEG images")
                return {"status": "skipped", "reason": "Not a JPEG image", "quality": quality}
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Step 1: Save the image at lower quality
                with Image.open(image_path) as original:
                    original = original.convert('RGB')
                    original.save(temp_filename, 'JPEG', quality=quality)
                
                # Step 2: Re-open and calculate differences
                with Image.open(temp_filename) as resaved:
                    diff = ImageChops.difference(original, resaved)
                
                # Step 3: Analyze differences
                extrema = diff.getextrema()
                max_diff = max([ex[1] for ex in extrema])
                avg_diff = sum([sum(ex) / len(ex) for ex in extrema]) / len(extrema)
                
                # Calculate statistics
                pixel_differences = []
                for pixel in diff.getdata():
                    if isinstance(pixel, tuple):
                        pixel_differences.append(sum(pixel) / len(pixel))
                    else:
                        pixel_differences.append(pixel)
                
                # Calculate percentage of pixels with significant differences
                significant_diff_threshold = 20  # Threshold for "significant" difference
                significant_pixels = sum(1 for p in pixel_differences if p > significant_diff_threshold)
                total_pixels = len(pixel_differences)
                significant_percentage = (significant_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                
                return {
                    "status": "completed",
                    "quality": quality,
                    "max_difference": max_diff,
                    "avg_difference": round(avg_diff, 2),
                    "significant_pixels_percentage": round(significant_percentage, 2),
                    "total_pixels": total_pixels,
                    "significant_pixels": significant_pixels,
                    "analysis_notes": f"ELA analysis completed. {significant_percentage:.1f}% of pixels show significant differences."
                }
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                    
        except Exception as e:
            logger.error(f"Error performing ELA analysis: {e}")
            return None
    
    @staticmethod
    def get_system_info():
        """Get system information"""
        try:
            return f"OS: {platform.system()} {platform.release()}"
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return "System information unavailable"
