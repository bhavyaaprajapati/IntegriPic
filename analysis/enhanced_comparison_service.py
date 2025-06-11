"""
Enhanced image comparison service with brightness, sharpness, GPS, and device analysis
"""
from PIL import Image, ImageChops
from PIL.ExifTags import TAGS
import math
import os
import time
import tempfile
import logging
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)


class ImageComparisonService:
    """Service class for enhanced image comparison operations"""
    
    @staticmethod
    def compare_images(image1_path, image2_path):
        """Compare two images and generate detailed comparison results"""
        try:
            with Image.open(image1_path) as img1, Image.open(image2_path) as img2:
                # Store original dimensions for metadata
                original_size1 = img1.size
                original_size2 = img2.size
                
                # Convert to RGB for consistent processing
                img1 = img1.convert('RGB')
                img2 = img2.convert('RGB')
                
                # Get image formats
                format1 = img1.format if hasattr(img1, 'format') else 'Unknown'
                format2 = img2.format if hasattr(img2, 'format') else 'Unknown'
                
                # Resize if needed (using LANCZOS for better quality)
                size_matched = original_size1 == original_size2
                if not size_matched:
                    logger.info("Images have different sizes. Resizing second image...")
                    img2 = img2.resize(original_size1, Image.Resampling.LANCZOS)
                
                # Create difference image
                diff = ImageChops.difference(img1, img2)
                
                # Check if images are identical
                bbox = diff.getbbox()
                are_identical = bbox is None
                
                # Calculate detailed similarity metrics
                similarity_metrics = ImageComparisonService._calculate_similarity_metrics(img1, img2, diff)
                
                # Generate color analysis
                color_analysis = ImageComparisonService._analyze_color_differences(img1, img2)
                
                # Extract enhanced properties for both images
                img1_properties = ImageComparisonService._extract_enhanced_image_properties(image1_path)
                img2_properties = ImageComparisonService._extract_enhanced_image_properties(image2_path)
                
                # Save difference image to memory
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as diff_temp:
                    diff.save(diff_temp.name, 'PNG')
                    
                    # Read the file content and create Django file
                    with open(diff_temp.name, 'rb') as f:
                        diff_content = f.read()
                    
                    os.unlink(diff_temp.name)  # Clean up temp file
                    
                    return {
                        'difference_image': ContentFile(diff_content, name=f'diff_{int(time.time())}.png'),
                        'are_identical': are_identical,
                        'similarity_score': similarity_metrics['similarity_percentage'],
                        'success': True,
                        'message': 'Images are identical.' if are_identical else f'Images have differences. Similarity: {similarity_metrics["similarity_percentage"]:.1f}%',
                        'details': {
                            'image1_info': {
                                'size': original_size1,
                                'format': format1,
                                'total_pixels': original_size1[0] * original_size1[1]
                            },
                            'image2_info': {
                                'size': original_size2,
                                'format': format2,
                                'total_pixels': original_size2[0] * original_size2[1]
                            },
                            'image1_properties': img1_properties,
                            'image2_properties': img2_properties,
                            'comparison_info': {
                                'size_matched': size_matched,
                                'different_pixels': similarity_metrics['different_pixels'],
                                'different_percentage': similarity_metrics['different_percentage'],
                                'max_difference': similarity_metrics['max_difference'],
                                'avg_difference': similarity_metrics['avg_difference'],
                                'structural_similarity': similarity_metrics.get('ssim', 0),
                            },
                            'color_analysis': color_analysis,
                            'difference_region': {
                                'bounding_box': bbox if bbox else None,
                                'has_differences': bbox is not None
                            },
                            'enhanced_comparison': {
                                'brightness_difference': abs(img1_properties['brightness'] - img2_properties['brightness']),
                                'sharpness_difference': abs(img1_properties['sharpness'] - img2_properties['sharpness']),
                                'contrast_difference': abs(img1_properties['contrast'] - img2_properties['contrast']),
                                'same_device': (img1_properties['exif_data'].get('camera_make') == img2_properties['exif_data'].get('camera_make') and 
                                              img1_properties['exif_data'].get('camera_model') == img2_properties['exif_data'].get('camera_model')),
                                'gps_comparison': ImageComparisonService._compare_gps_data(
                                    img1_properties['exif_data'].get('gps_info', {}),
                                    img2_properties['exif_data'].get('gps_info', {})
                                ),
                                'color_temperature_match': img1_properties['color_temperature'] == img2_properties['color_temperature'],
                                'quality_comparison': ImageComparisonService._compare_quality_metrics(
                                    img1_properties['quality_metrics'],
                                    img2_properties['quality_metrics']
                                )
                            }
                        }
                    }
                    
        except Exception as e:
            logger.error(f"Error comparing images: {e}")
            return {
                'difference_image': None,
                'are_identical': False,
                'similarity_score': 0,
                'success': False,
                'message': f'Error comparing images: {str(e)}',
                'details': {}
            }
    
    @staticmethod
    def _extract_enhanced_image_properties(image_path):
        """Extract enhanced properties including brightness, sharpness, EXIF data"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB for consistent analysis
                rgb_img = img.convert('RGB')
                
                # Calculate brightness (average luminance)
                grayscale = rgb_img.convert('L')
                pixels = list(grayscale.getdata())
                brightness = sum(pixels) / len(pixels)
                
                # Calculate sharpness using Laplacian variance
                sharpness = ImageComparisonService._calculate_sharpness(grayscale)
                
                # Calculate contrast
                contrast = ImageComparisonService._calculate_contrast(pixels)
                
                # Extract EXIF data for device and location information
                exif_data = ImageComparisonService._extract_detailed_exif(img)
                
                # Calculate color temperature estimation
                color_temp = ImageComparisonService._estimate_color_temperature(rgb_img)
                
                # Get image quality metrics
                quality_metrics = ImageComparisonService._assess_image_quality(rgb_img)
                
                return {
                    'brightness': round(brightness, 2),
                    'sharpness': round(sharpness, 2),
                    'contrast': round(contrast, 2),
                    'color_temperature': color_temp,
                    'quality_metrics': quality_metrics,
                    'exif_data': exif_data,
                    'size': img.size,
                    'format': img.format or 'Unknown',
                    'mode': img.mode
                }
                
        except Exception as e:
            logger.error(f"Error extracting enhanced image properties: {e}")
            return {
                'brightness': 0,
                'sharpness': 0,
                'contrast': 0,
                'color_temperature': 'Unknown',
                'quality_metrics': {},
                'exif_data': {},
                'size': (0, 0),
                'format': 'Unknown',
                'mode': 'Unknown'
            }
    
    @staticmethod
    def _calculate_sharpness(grayscale_image):
        """Calculate image sharpness using edge detection"""
        try:
            # Try advanced method with numpy if available
            try:
                import numpy as np
                from scipy import ndimage
                
                image_array = np.array(grayscale_image)
                laplacian = ndimage.laplace(image_array)
                variance = laplacian.var()
                return variance
                
            except ImportError:
                # Fallback method without scipy
                pixels = list(grayscale_image.getdata())
                width, height = grayscale_image.size
                
                edge_count = 0
                threshold = 30
                
                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        center = pixels[y * width + x]
                        
                        # Check horizontal gradient
                        left = pixels[y * width + (x - 1)]
                        right = pixels[y * width + (x + 1)]
                        h_gradient = abs(right - left)
                        
                        # Check vertical gradient
                        top = pixels[(y - 1) * width + x]
                        bottom = pixels[(y + 1) * width + x]
                        v_gradient = abs(bottom - top)
                        
                        if h_gradient > threshold or v_gradient > threshold:
                            edge_count += 1
                
                total_pixels = (width - 2) * (height - 2)
                sharpness = (edge_count / total_pixels) * 100 if total_pixels > 0 else 0
                return sharpness
                
        except Exception as e:
            logger.error(f"Error calculating sharpness: {e}")
            return 0
    
    @staticmethod
    def _calculate_contrast(pixels):
        """Calculate image contrast as standard deviation of pixel values"""
        try:
            if not pixels:
                return 0
                
            mean_brightness = sum(pixels) / len(pixels)
            variance = sum((p - mean_brightness) ** 2 for p in pixels) / len(pixels)
            contrast = variance ** 0.5
            return contrast
            
        except Exception as e:
            logger.error(f"Error calculating contrast: {e}")
            return 0
    
    @staticmethod
    def _extract_detailed_exif(img):
        """Extract detailed EXIF data including GPS and device information"""
        try:
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
                    'gps_info': ImageComparisonService._extract_gps_info(exif_dict.get('GPSInfo', {})),
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
            logger.error(f"Error extracting EXIF data: {e}")
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
    def _compare_gps_data(gps1, gps2):
        """Compare GPS data between two images"""
        try:
            if not gps1 or not gps2:
                return {'status': 'One or both images lack GPS data'}
            
            if 'error' in gps1 or 'error' in gps2:
                return {'status': 'GPS data parsing error'}
            
            if 'latitude' not in gps1 or 'latitude' not in gps2:
                return {'status': 'Incomplete GPS coordinates'}
            
            # Calculate distance between two GPS points using Haversine formula
            lat1, lon1 = gps1['latitude'], gps1['longitude']
            lat2, lon2 = gps2['latitude'], gps2['longitude']
            
            # Convert to radians
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            
            # Haversine formula
            a = (math.sin(delta_lat/2)**2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance_km = 6371 * c  # Earth's radius in km
            
            # Determine if they're at the same location (within 100m)
            same_location = distance_km < 0.1
            
            return {
                'same_location': same_location,
                'distance_km': round(distance_km, 3),
                'coordinates_1': f"{lat1}, {lon1}",
                'coordinates_2': f"{lat2}, {lon2}"
            }
            
        except Exception as e:
            logger.error(f"Error comparing GPS data: {e}")
            return {'status': 'Error comparing GPS data', 'error': str(e)}
    
    @staticmethod
    def _estimate_color_temperature(rgb_img):
        """Estimate color temperature of the image"""
        try:
            # Sample pixels from the image
            pixels = list(rgb_img.getdata())
            sample_size = min(10000, len(pixels))  # Sample max 10k pixels
            import random
            sampled_pixels = random.sample(pixels, sample_size)
            
            # Calculate average RGB values
            avg_r = sum(p[0] for p in sampled_pixels) / sample_size
            avg_g = sum(p[1] for p in sampled_pixels) / sample_size
            avg_b = sum(p[2] for p in sampled_pixels) / sample_size
            
            # Estimate color temperature based on RGB ratios
            if avg_r > avg_b * 1.2:
                if avg_r > avg_g * 1.1:
                    return "Warm (3000-4000K)"
                else:
                    return "Neutral-Warm (4000-5000K)"
            elif avg_b > avg_r * 1.2:
                return "Cool (6000-7000K)"
            else:
                return "Neutral (5000-6000K)"
                
        except Exception as e:
            logger.error(f"Error estimating color temperature: {e}")
            return "Unknown"
    
    @staticmethod
    def _assess_image_quality(rgb_img):
        """Assess various image quality metrics"""
        try:
            # Calculate noise estimation
            noise_level = ImageComparisonService._estimate_noise_level(rgb_img)
            
            # Calculate dynamic range
            pixels = list(rgb_img.getdata())
            all_values = []
            for pixel in pixels:
                all_values.extend(pixel)
            
            min_val = min(all_values)
            max_val = max(all_values)
            dynamic_range = max_val - min_val
            
            # Calculate color diversity
            unique_colors = len(set(pixels))
            total_pixels = len(pixels)
            color_diversity = (unique_colors / total_pixels) * 100
            
            return {
                'noise_level': round(noise_level, 2),
                'dynamic_range': dynamic_range,
                'color_diversity': round(color_diversity, 2)
            }
            
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return {
                'noise_level': 0,
                'dynamic_range': 0,
                'color_diversity': 0
            }
    
    @staticmethod
    def _estimate_noise_level(rgb_img):
        """Estimate noise level in the image"""
        try:
            # Convert to grayscale for noise analysis
            gray = rgb_img.convert('L')
            pixels = list(gray.getdata())
            width, height = gray.size
            
            # Calculate local variance to estimate noise
            noise_sum = 0
            count = 0
            
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    # Get 3x3 neighborhood
                    neighborhood = []
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            idx = (y + dy) * width + (x + dx)
                            neighborhood.append(pixels[idx])
                    
                    # Calculate variance of neighborhood
                    mean_val = sum(neighborhood) / len(neighborhood)
                    variance = sum((p - mean_val) ** 2 for p in neighborhood) / len(neighborhood)
                    noise_sum += variance
                    count += 1
            
            avg_noise = noise_sum / count if count > 0 else 0
            return avg_noise
            
        except Exception as e:
            logger.error(f"Error estimating noise level: {e}")
            return 0
    
    @staticmethod
    def _compare_quality_metrics(metrics1, metrics2):
        """Compare quality metrics between two images"""
        try:
            if not metrics1 or not metrics2:
                return {'status': 'Quality metrics unavailable'}
            
            return {
                'noise_difference': abs(metrics1.get('noise_level', 0) - metrics2.get('noise_level', 0)),
                'dynamic_range_difference': abs(metrics1.get('dynamic_range', 0) - metrics2.get('dynamic_range', 0)),
                'color_diversity_difference': abs(metrics1.get('color_diversity', 0) - metrics2.get('color_diversity', 0)),
                'similar_quality': (
                    abs(metrics1.get('noise_level', 0) - metrics2.get('noise_level', 0)) < 10 and
                    abs(metrics1.get('dynamic_range', 0) - metrics2.get('dynamic_range', 0)) < 50 and
                    abs(metrics1.get('color_diversity', 0) - metrics2.get('color_diversity', 0)) < 10
                )
            }
            
        except Exception as e:
            logger.error(f"Error comparing quality metrics: {e}")
            return {'status': 'Error comparing quality metrics'}
    
    @staticmethod
    def _calculate_similarity_metrics(img1, img2, diff):
        """Calculate detailed similarity metrics between two images"""
        try:
            # Try advanced method with numpy if available
            try:
                import numpy as np
                
                img1_array = np.array(img1)
                img2_array = np.array(img2)
                
                # Calculate pixel differences
                total_pixels = img1_array.size // 3  # RGB channels
                
                # Count different pixels (threshold for considering pixel "different")
                pixel_diff_threshold = 5  # Small threshold to account for compression artifacts
                pixel_differences = np.sqrt(np.sum((img1_array - img2_array) ** 2, axis=2))
                different_pixels = np.sum(pixel_differences > pixel_diff_threshold)
                different_percentage = (different_pixels / total_pixels) * 100
                
                # Calculate average and max differences
                avg_difference = np.mean(pixel_differences)
                max_difference = np.max(pixel_differences)
                
                # Calculate similarity percentage
                similarity_percentage = max(0, 100 - different_percentage)
                
                return {
                    'different_pixels': int(different_pixels),
                    'different_percentage': round(different_percentage, 2),
                    'similarity_percentage': round(similarity_percentage, 2),
                    'avg_difference': round(float(avg_difference), 2),
                    'max_difference': round(float(max_difference), 2),
                }
                
            except ImportError:
                # Fallback method without numpy
                return ImageComparisonService._basic_similarity_metrics(img1, img2, diff)
                
        except Exception as e:
            logger.error(f"Error calculating similarity metrics: {e}")
            return ImageComparisonService._basic_similarity_metrics(img1, img2, diff)
    
    @staticmethod
    def _basic_similarity_metrics(img1, img2, diff):
        """Basic similarity calculation without numpy"""
        try:
            # Get pixel data
            pixels1 = list(img1.getdata())
            pixels2 = list(img2.getdata())
            
            total_pixels = len(pixels1)
            different_pixels = 0
            total_difference = 0
            max_diff = 0
            
            for p1, p2 in zip(pixels1, pixels2):
                # Calculate RGB difference
                r_diff = abs(p1[0] - p2[0])
                g_diff = abs(p1[1] - p2[1])
                b_diff = abs(p1[2] - p2[2])
                pixel_diff = (r_diff + g_diff + b_diff) / 3
                
                if pixel_diff > 5:  # Threshold for "different"
                    different_pixels += 1
                
                total_difference += pixel_diff
                max_diff = max(max_diff, pixel_diff)
            
            different_percentage = (different_pixels / total_pixels) * 100
            similarity_percentage = max(0, 100 - different_percentage)
            avg_difference = total_difference / total_pixels
            
            return {
                'different_pixels': different_pixels,
                'different_percentage': round(different_percentage, 2),
                'similarity_percentage': round(similarity_percentage, 2),
                'avg_difference': round(avg_difference, 2),
                'max_difference': round(max_diff, 2),
            }
            
        except Exception as e:
            logger.error(f"Error in basic similarity calculation: {e}")
            return {
                'different_pixels': 0,
                'different_percentage': 0,
                'similarity_percentage': 0,
                'avg_difference': 0,
                'max_difference': 0,
            }
    
    @staticmethod
    def _analyze_color_differences(img1, img2):
        """Analyze color distribution differences between images"""
        try:
            # Get color histograms for each channel
            hist1_r = img1.histogram()[0:256]    # Red channel
            hist1_g = img1.histogram()[256:512]  # Green channel
            hist1_b = img1.histogram()[512:768]  # Blue channel
            
            hist2_r = img2.histogram()[0:256]
            hist2_g = img2.histogram()[256:512] 
            hist2_b = img2.histogram()[512:768]
            
            # Calculate histogram correlation for each channel
            def calculate_correlation(h1, h2):
                if sum(h1) == 0 or sum(h2) == 0:
                    return 0
                    
                # Normalize histograms
                h1_norm = [x / sum(h1) for x in h1]
                h2_norm = [x / sum(h2) for x in h2]
                
                # Calculate correlation coefficient
                mean1 = sum(h1_norm) / len(h1_norm)
                mean2 = sum(h2_norm) / len(h2_norm)
                
                numerator = sum((h1_norm[i] - mean1) * (h2_norm[i] - mean2) for i in range(len(h1_norm)))
                denominator1 = sum((h1_norm[i] - mean1) ** 2 for i in range(len(h1_norm)))
                denominator2 = sum((h2_norm[i] - mean2) ** 2 for i in range(len(h2_norm)))
                
                if denominator1 == 0 or denominator2 == 0:
                    return 0
                    
                return numerator / (denominator1 * denominator2) ** 0.5
            
            r_correlation = calculate_correlation(hist1_r, hist2_r)
            g_correlation = calculate_correlation(hist1_g, hist2_g)
            b_correlation = calculate_correlation(hist1_b, hist2_b)
            
            overall_color_similarity = (r_correlation + g_correlation + b_correlation) / 3
            
            return {
                'red_channel_similarity': round(r_correlation * 100, 2),
                'green_channel_similarity': round(g_correlation * 100, 2),
                'blue_channel_similarity': round(b_correlation * 100, 2),
                'overall_color_similarity': round(overall_color_similarity * 100, 2),
            }
            
        except Exception as e:
            logger.error(f"Error analyzing color differences: {e}")
            return {
                'red_channel_similarity': 0,
                'green_channel_similarity': 0,
                'blue_channel_similarity': 0,
                'overall_color_similarity': 0,
            }
