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
try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

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
        """Extract EXIF metadata from image using multiple methods"""
        metadata = {}
        try:
            with Image.open(image_path) as image:
                # Try Piexif first (more robust for JPEG)
                if PIEXIF_AVAILABLE and image_path.lower().endswith(('.jpg', '.jpeg')):
                    try:
                        exif_dict = piexif.load(image_path)
                        for ifd_name in ("0th", "Exif", "GPS"):
                            if ifd_name in exif_dict:
                                for tag, value in exif_dict[ifd_name].items():
                                    tag_name = piexif.TAGS[ifd_name][tag]["name"]
                                    try:
                                        metadata[str(tag_name)] = str(value)
                                    except:
                                        metadata[str(tag_name)] = str(value)
                        return metadata
                    except Exception as piexif_error:
                        logger.debug(f"Piexif extraction failed, falling back to PIL: {piexif_error}")

                # Fallback to PIL _getexif
                exif_data = image._getexif()
                if exif_data:
                    for tag, value in exif_data.items():
                        tag_name = TAGS.get(tag, tag)
                        try:
                            metadata[str(tag_name)] = str(value)
                        except:
                            metadata[str(tag_name)] = repr(value)

                return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    @staticmethod
    def extract_detailed_exif(image_path):
        """Extract detailed EXIF data including GPS and device information"""
        try:
            extracted = {
                'camera_make': 'Unknown',
                'camera_model': 'Unknown',
                'datetime': 'Unknown',
                'software': 'Unknown',
                'gps_info': {'status': 'No GPS data found'},
                'flash_used': 'Unknown',
                'focal_length': 'Unknown',
                'exposure_time': 'Unknown',
                'f_number': 'Unknown',
                'iso_speed': 'Unknown',
                'white_balance': 'Unknown',
                'orientation': 'Unknown'
            }

            # Try Piexif first for JPEG (more reliable)
            if PIEXIF_AVAILABLE and image_path.lower().endswith(('.jpg', '.jpeg')):
                try:
                    exif_dict = piexif.load(image_path)

                    # Extract 0th IFD (main image)
                    if "0th" in exif_dict:
                        ifd_0th = exif_dict["0th"]
                        extracted['camera_make'] = str(ifd_0th.get(piexif.ImageIFD.Make, b'Unknown')).strip("b'\"")
                        extracted['camera_model'] = str(ifd_0th.get(piexif.ImageIFD.Model, b'Unknown')).strip("b'\"")
                        extracted['software'] = str(ifd_0th.get(piexif.ImageIFD.Software, b'Unknown')).strip("b'\"")
                        extracted['orientation'] = str(ifd_0th.get(piexif.ImageIFD.Orientation, 1))

                    # Extract Exif IFD
                    if "Exif" in exif_dict:
                        ifd_exif = exif_dict["Exif"]
                        extracted['datetime'] = str(ifd_exif.get(piexif.ExifIFD.DateTimeOriginal, b'Unknown')).strip("b'\"")
                        extracted['focal_length'] = str(ifd_exif.get(piexif.ExifIFD.FocalLength, b'Unknown')).strip("b'\"")
                        extracted['iso_speed'] = str(ifd_exif.get(piexif.ExifIFD.ISOSpeedRatings, b'Unknown')).strip("b'\"")

                    # Extract GPS data
                    if "GPS" in exif_dict:
                        gps_ifd = exif_dict["GPS"]
                        extracted['gps_info'] = ImageAnalysisService._extract_gps_info_piexif(gps_ifd)

                    return extracted
                except Exception as piexif_error:
                    logger.debug(f"Piexif detailed extraction failed, falling back to PIL: {piexif_error}")

            # Fallback to PIL method
            with Image.open(image_path) as img:
                exif_data = img._getexif()

                if exif_data:
                    exif_dict = {}
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except:
                                value = str(value)
                        exif_dict[tag] = value

                    extracted['camera_make'] = exif_dict.get('Make', 'Unknown')
                    extracted['camera_model'] = exif_dict.get('Model', 'Unknown')
                    extracted['datetime'] = exif_dict.get('DateTime', 'Unknown')
                    extracted['software'] = exif_dict.get('Software', 'Unknown')
                    extracted['gps_info'] = ImageAnalysisService._extract_gps_info(exif_dict.get('GPSInfo', {}))
                    extracted['flash_used'] = exif_dict.get('Flash', 'Unknown')
                    extracted['focal_length'] = exif_dict.get('FocalLength', 'Unknown')
                    extracted['exposure_time'] = exif_dict.get('ExposureTime', 'Unknown')
                    extracted['f_number'] = exif_dict.get('FNumber', 'Unknown')
                    extracted['iso_speed'] = exif_dict.get('ISOSpeedRatings', 'Unknown')
                    extracted['white_balance'] = exif_dict.get('WhiteBalance', 'Unknown')
                    extracted['orientation'] = exif_dict.get('Orientation', 'Unknown')

            return extracted

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
    def _extract_gps_info_piexif(gps_ifd):
        """Extract GPS coordinates from Piexif GPS IFD"""
        try:
            if not gps_ifd:
                return {'status': 'No GPS data found'}

            gps_data = {}

            # Helper to convert piexif GPS rational to degrees
            def rational_to_degree(rational):
                """Convert piexif rational to float"""
                if isinstance(rational, tuple):
                    return float(rational[0]) / float(rational[1])
                return float(rational)

            # Get latitude
            if piexif.GPSIFD.GPSLatitude in gps_ifd and piexif.GPSIFD.GPSLatitudeRef in gps_ifd:
                lat_data = gps_ifd[piexif.GPSIFD.GPSLatitude]
                lat = rational_to_degree(lat_data[0]) + rational_to_degree(lat_data[1])/60 + rational_to_degree(lat_data[2])/3600
                lat_ref = gps_ifd[piexif.GPSIFD.GPSLatitudeRef]
                if isinstance(lat_ref, bytes):
                    lat_ref = lat_ref.decode()
                if lat_ref == 'S':
                    lat = -lat
                gps_data['latitude'] = round(lat, 6)

            # Get longitude
            if piexif.GPSIFD.GPSLongitude in gps_ifd and piexif.GPSIFD.GPSLongitudeRef in gps_ifd:
                lon_data = gps_ifd[piexif.GPSIFD.GPSLongitude]
                lon = rational_to_degree(lon_data[0]) + rational_to_degree(lon_data[1])/60 + rational_to_degree(lon_data[2])/3600
                lon_ref = gps_ifd[piexif.GPSIFD.GPSLongitudeRef]
                if isinstance(lon_ref, bytes):
                    lon_ref = lon_ref.decode()
                if lon_ref == 'W':
                    lon = -lon
                gps_data['longitude'] = round(lon, 6)

            # Get altitude
            if piexif.GPSIFD.GPSAltitude in gps_ifd:
                alt_data = gps_ifd[piexif.GPSIFD.GPSAltitude]
                altitude = rational_to_degree(alt_data)
                if piexif.GPSIFD.GPSAltitudeRef in gps_ifd:
                    alt_ref = gps_ifd[piexif.GPSIFD.GPSAltitudeRef]
                    if alt_ref == 1:
                        altitude = -altitude
                gps_data['altitude'] = round(altitude, 2)

            if gps_data:
                return gps_data
            else:
                return {'status': 'GPS data present but could not be parsed'}

        except Exception as e:
            logger.error(f"Error extracting GPS info from piexif: {e}")
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
                # Step 1: Load and convert the original image
                with Image.open(image_path) as img:
                    original = img.convert('RGB')
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

                # Generate heatmap visualization
                heatmap_b64 = None
                try:
                    import numpy as np
                    import base64
                    from io import BytesIO

                    # Convert diff image to numpy array
                    diff_array = np.array(diff)  # Shape: (H, W, 3)
                    gray = diff_array.mean(axis=2)  # (H, W) grayscale
                    amplified = np.clip(gray * 10, 0, 255).astype(np.uint8)  # 10x amplification

                    # Build RGBA heatmap: red channel = amplified intensity
                    heatmap_array = np.zeros((amplified.shape[0], amplified.shape[1], 4), dtype=np.uint8)
                    heatmap_array[:, :, 0] = amplified          # R
                    heatmap_array[:, :, 3] = amplified          # Alpha

                    from PIL import Image as PILImage
                    heatmap_img = PILImage.fromarray(heatmap_array, 'RGBA')

                    # Resize to max 800px wide
                    max_width = 800
                    if heatmap_img.width > max_width:
                        ratio = max_width / heatmap_img.width
                        heatmap_img = heatmap_img.resize(
                            (max_width, int(heatmap_img.height * ratio)),
                            PILImage.LANCZOS
                        )

                    buf = BytesIO()
                    heatmap_img.save(buf, format='PNG')
                    heatmap_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                except Exception as e:
                    logger.warning(f"Could not generate ELA heatmap: {e}")

                return {
                    "status": "completed",
                    "quality": quality,
                    "max_difference": max_diff,
                    "avg_difference": round(avg_diff, 2),
                    "significant_pixels_percentage": round(significant_percentage, 2),
                    "total_pixels": total_pixels,
                    "significant_pixels": significant_pixels,
                    "analysis_notes": f"ELA analysis completed. {significant_percentage:.1f}% of pixels show significant differences.",
                    "heatmap_b64": heatmap_b64
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

    @staticmethod
    def extract_rgb_data(image_path):
        """Extract RGB histogram data, per-channel stats, and dominant colors.
        Returns a dict suitable for storage in rgb_histogram_data JSONField."""
        try:
            import numpy as np

            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                pixels = np.array(img)

                r_vals = pixels[:, :, 0].flatten()
                g_vals = pixels[:, :, 1].flatten()
                b_vals = pixels[:, :, 2].flatten()

                def channel_stats(arr):
                    return {
                        'mean': round(float(np.mean(arr)), 2),
                        'std': round(float(np.std(arr)), 2),
                        'min': int(np.min(arr)),
                        'max': int(np.max(arr)),
                    }

                r_hist = np.histogram(r_vals, bins=256, range=(0, 256))[0].tolist()
                g_hist = np.histogram(g_vals, bins=256, range=(0, 256))[0].tolist()
                b_hist = np.histogram(b_vals, bins=256, range=(0, 256))[0].tolist()

                # Dominant colors: reshape, get unique with counts, sort top 10
                pixels_flat = pixels.reshape(-1, 3)
                unique_colors, counts = np.unique(pixels_flat, axis=0, return_counts=True)
                top_indices = np.argsort(counts)[-10:][::-1]
                dominant_colors = [
                    {
                        'rgb': [int(unique_colors[i][0]), int(unique_colors[i][1]), int(unique_colors[i][2])],
                        'hex': '#{:02x}{:02x}{:02x}'.format(
                            int(unique_colors[i][0]), int(unique_colors[i][1]), int(unique_colors[i][2])
                        ),
                        'count': int(counts[i]),
                    }
                    for i in top_indices
                ]

                return {
                    'r_hist': r_hist,
                    'g_hist': g_hist,
                    'b_hist': b_hist,
                    'r_stats': channel_stats(r_vals),
                    'g_stats': channel_stats(g_vals),
                    'b_stats': channel_stats(b_vals),
                    'dominant_colors': dominant_colors,
                }
        except Exception as e:
            logger.error(f"Error extracting RGB data: {e}")
            return {}

    @staticmethod
    def compute_perceptual_hashes(image_path):
        """Compute pHash, dHash, aHash for near-duplicate detection.
        Returns dict with 'phash', 'dhash', 'ahash' string values."""
        try:
            import imagehash
            with Image.open(image_path) as img:
                return {
                    'phash': str(imagehash.phash(img)),
                    'dhash': str(imagehash.dhash(img)),
                    'ahash': str(imagehash.average_hash(img)),
                }
        except Exception as e:
            logger.error(f"Error computing perceptual hashes: {e}")
            return {'phash': None, 'dhash': None, 'ahash': None}

    @staticmethod
    def detect_timeline_inconsistencies(metadata):
        """Analyze EXIF timestamps for forensic inconsistencies.
        Returns a list of flag dicts: [{'severity': 'warning'|'critical', 'description': '...'}]"""
        flags = []
        try:
            from datetime import datetime

            def parse_exif_datetime(s):
                if not s or s == 'Unknown':
                    return None
                for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                    try:
                        return datetime.strptime(str(s).strip(), fmt)
                    except ValueError:
                        continue
                return None

            dt_original = parse_exif_datetime(metadata.get('DateTimeOriginal'))
            dt_modified = parse_exif_datetime(metadata.get('DateTime'))
            software = metadata.get('Software', '')

            # Flag 1: Modification date before original capture date
            if dt_original and dt_modified and dt_modified < dt_original:
                flags.append({
                    'severity': 'critical',
                    'field': 'DateTime vs DateTimeOriginal',
                    'description': (
                        f"File modification timestamp ({dt_modified}) is earlier than "
                        f"capture timestamp ({dt_original}). This is physically impossible "
                        "and strongly indicates metadata tampering."
                    )
                })

            # Flag 2: Software modification date present (implies post-processing)
            if software and software not in ('Unknown', ''):
                non_camera_software = ['photoshop', 'gimp', 'lightroom', 'affinity',
                                       'paint', 'snapseed', 'instagram', 'whatsapp']
                if any(s in software.lower() for s in non_camera_software):
                    flags.append({
                        'severity': 'warning',
                        'field': 'Software',
                        'description': (
                            f"Image was processed by editing software: '{software}'. "
                            "The original capture metadata may have been altered."
                        )
                    })

            # Flag 3: Capture date in the future
            now = datetime.utcnow()
            if dt_original and dt_original > now:
                flags.append({
                    'severity': 'critical',
                    'field': 'DateTimeOriginal',
                    'description': (
                        f"Capture date ({dt_original}) is in the future ({now}). "
                        "This indicates the date/time has been tampered with."
                    )
                })

        except Exception as e:
            logger.error(f"Error detecting timeline inconsistencies: {e}")

        return flags

    @staticmethod
    def detect_copy_move(image_path):
        """Detect copy-move forgery using SIFT feature matching.
        Returns a dict: {'detected': bool, 'match_count': int, 'notes': str}"""
        try:
            import cv2
            import numpy as np

            img = cv2.imread(image_path)
            if img is None:
                return {'detected': False, 'match_count': 0, 'notes': 'Could not load image'}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Use SIFT to find keypoints and descriptors
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < 10:
                return {'detected': False, 'match_count': 0, 'notes': 'Insufficient features'}

            # Match descriptors against themselves using BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(descriptors, descriptors)

            # Filter out self-matches (same keypoint) and find pairs with high spatial offset
            significant_matches = 0
            MIN_SPATIAL_DISTANCE = 20  # pixels

            for m in matches:
                if m.queryIdx == m.trainIdx:
                    continue  # skip self-matches
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                if dist > MIN_SPATIAL_DISTANCE:
                    significant_matches += 1

            detected = significant_matches >= 10  # threshold for suspicion
            return {
                'detected': detected,
                'match_count': significant_matches,
                'notes': (
                    f"Found {significant_matches} spatially-separated duplicate feature matches. "
                    "This may indicate copy-move forgery." if detected
                    else f"No significant copy-move patterns detected ({significant_matches} matches below threshold)."
                )
            }
        except Exception as e:
            logger.error(f"Error in copy-move detection: {e}")
            return {'detected': False, 'match_count': 0, 'notes': f'Error: {str(e)}'}

    @staticmethod
    def compute_ai_probability(ela_results, copy_move_result, metadata):
        """Compute AI generation probability using forensic heuristics.
        Returns tuple: (score: float 0-100, notes: str)

        Key Indicators:
        - NO EXIF + LOW ELA = VERY strong AI indicator (combination matters)
        - NO EXIF alone = moderate indicator (could be screenshot/edited)
        - LOW ELA alone = moderate indicator (could be clean photo)
        - Copy-move presence = indicates human tampering (reduces AI prob)
        - Metadata inconsistencies = indicates tampering
        """
        score = 0.0
        notes = []

        try:
            # Extract key forensic indicators
            ela_sig_pct = None
            if ela_results:
                ela_sig_pct = ela_results.get('significant_pixels_percentage')

            has_exif = metadata and bool(metadata.get('exif'))
            timeline_flags = metadata.get('timeline_flags', []) if metadata else []
            copy_move_count = copy_move_result.get('match_count', 0) if copy_move_result else 0

            # CRITICAL COMBINATION: No EXIF + Very Low ELA = Strong AI indicator
            if not has_exif and ela_sig_pct is not None and ela_sig_pct < 1.0:
                score += 50
                notes.append("No camera metadata + extremely clean compression profile (hallmark of AI)")
            else:
                # Individual indicators when not combined
                if not has_exif:
                    score += 20
                    notes.append("No EXIF metadata (not from camera or stripped)")

                if ela_sig_pct is not None:
                    if ela_sig_pct < 1.0:
                        score += 25
                        notes.append("Extremely low ELA activity (too clean for real camera)")
                    elif ela_sig_pct < 3.0:
                        score += 10
                        notes.append("Unusually low compression artifacts")

            # Metadata timestamp inconsistencies suggest tampering (not AI)
            if timeline_flags and len(timeline_flags) > 0:
                score += 10
                notes.append(f"Metadata timestamp inconsistencies detected ({len(timeline_flags)} flags)")

            # Copy-move forgery indicates HUMAN tampering (reduces AI probability)
            if copy_move_count > 5:
                score -= 15  # Stronger penalty: human copy-paste is opposite of AI
                notes.append("Copy-move forgery detected (indicates human tampering)")
            elif copy_move_count > 0:
                score -= 5
                notes.append(f"Minor copy-move regions ({copy_move_count} matches)")

            # Clamp to [0, 100]
            score = max(0.0, min(100.0, score))

            # Generate confidence description
            if not notes:
                notes_str = "Within normal forensic range"
            else:
                notes_str = "; ".join(notes)

            return score, notes_str

        except Exception as e:
            logger.error(f"Error computing AI probability: {e}")
            return 0.0, f"Error: {str(e)}"


class AuditService:
    """Service for logging audit trail for forensic chain of custody"""

    @staticmethod
    def get_client_ip(request):
        """Extract client IP address from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')

    @staticmethod
    def log(request, action_type, analysis=None, comparison=None, extra_data=None):
        """Create an audit log entry. Silently catches exceptions."""
        try:
            from .models import AuditLog
            import json

            result_hash = ''
            if analysis and analysis.ela_results:
                result_hash = hashlib.sha256(
                    json.dumps(analysis.ela_results, sort_keys=True).encode()
                ).hexdigest()

            AuditLog.objects.create(
                user=request.user if request.user.is_authenticated else None,
                action_type=action_type,
                analysis=analysis,
                comparison=comparison,
                ip_address=AuditService.get_client_ip(request),
                result_hash=result_hash,
                extra_data=extra_data or {},
            )
        except Exception as e:
            logger.error(f"Audit log error: {e}")
