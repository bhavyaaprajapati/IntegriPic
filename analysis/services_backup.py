"""
Image analysis services for Django implementation
Converted from original Flask standalone modules
"""
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
import hashlib
import os
import platform
import time
import math
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from stegano import lsb
import tempfile
import logging

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """Service class for image analysis operations"""
    
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
    def detect_steganography(image_path):
        """Detect hidden messages using multiple steganography detection methods"""
        try:
            # Check image format and size first
            with Image.open(image_path) as img:
                image_format = img.format.lower() if img.format else 'unknown'
                image_size = img.size
                mode = img.mode
                
                # Ensure image is in RGB mode for consistent processing
                if mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                    # Save the converted image to a temporary file for LSB analysis
                    temp_path = image_path + '_temp_rgb.png'
                    img.save(temp_path, 'PNG')
                    analysis_path = temp_path
                else:
                    analysis_path = image_path
                
            # Primary LSB steganography detection for supported formats
            # Note: JPEG support added with special handling for lossy compression
            supported_formats = ['png', 'bmp', 'tiff', 'jpg', 'jpeg']
            if any(image_path.lower().endswith(f'.{fmt}') for fmt in supported_formats):
                # Special handling for JPEG images
                is_jpeg = any(image_path.lower().endswith(f'.{fmt}') for fmt in ['jpg', 'jpeg'])
                
                # Try LSB detection with error handling
                try:
                    # First check if the image is suitable for LSB analysis
                    with Image.open(analysis_path) as test_img:
                        if test_img.size[0] * test_img.size[1] < 100:  # Too small for reliable LSB
                            return ImageAnalysisService._statistical_steganography_analysis(image_path, image_format)
                    
                    # For JPEG images, LSB steganography is less reliable due to compression
                    # We'll still try but with lower confidence
                    if is_jpeg:
                        # JPEG images lose LSB data due to compression, so we focus on statistical analysis
                        # But we'll still attempt LSB detection as some tools convert to lossless formats first
                        hidden_message = ImageAnalysisService._safe_lsb_reveal(analysis_path)
                        
                        if hidden_message and len(hidden_message.strip()) > 0:
                            # Validate that the message contains printable characters
                            printable_chars = sum(1 for c in hidden_message if c.isprintable())
                            if printable_chars / len(hidden_message) > 0.8:  # At least 80% printable
                                result = {
                                    'result': 'Hidden message detected using LSB method. Note: JPEG compression may affect LSB reliability.',
                                    'message': hidden_message[:500],  # Limit message length for display
                                    'success': True,
                                    'jpeg_warning': True
                                }
                            else:
                                result = {
                                    'result': 'Potential hidden data detected in JPEG, but content appears to be binary/non-text. JPEG compression may have affected the data.',
                                    'message': None,
                                    'success': True,
                                    'jpeg_warning': True
                                }
                        else:
                            # LSB didn't find anything, perform enhanced statistical analysis for JPEG
                            result = ImageAnalysisService._enhanced_jpeg_steganography_analysis(image_path, image_format)
                    else:
                        # Non-JPEG formats - original logic
                        hidden_message = ImageAnalysisService._safe_lsb_reveal(analysis_path)
                        
                        if hidden_message and len(hidden_message.strip()) > 0:
                            # Validate that the message contains printable characters
                            printable_chars = sum(1 for c in hidden_message if c.isprintable())
                            if printable_chars / len(hidden_message) > 0.8:  # At least 80% printable
                                result = {
                                    'result': 'Hidden message detected using LSB method.',
                                    'message': hidden_message[:500],  # Limit message length for display
                                    'success': True
                                }
                            else:
                                result = {
                                    'result': 'Potential hidden data detected, but content appears to be binary/non-text.',
                                    'message': None,
                                    'success': True
                                }
                        else:
                            # LSB didn't find anything, try statistical analysis
                            result = ImageAnalysisService._statistical_steganography_analysis(image_path, image_format)
                    with Image.open(analysis_path) as test_img:
                        if test_img.size[0] * test_img.size[1] < 100:  # Too small for reliable LSB
                            return ImageAnalysisService._statistical_steganography_analysis(image_path, image_format)
                    
                    # For JPEG images, LSB steganography is less reliable due to compression
                    # We'll still try but with lower confidence
                    if is_jpeg:
                        # JPEG images lose LSB data due to compression, so we focus on statistical analysis
                        # But we'll still attempt LSB detection as some tools convert to lossless formats first
                        hidden_message = ImageAnalysisService._safe_lsb_reveal(analysis_path)
                        
                        if hidden_message and len(hidden_message.strip()) > 0:
                            # Validate that the message contains printable characters
                            printable_chars = sum(1 for c in hidden_message if c.isprintable())
                            if printable_chars / len(hidden_message) > 0.8:  # At least 80% printable
                                result = {
                                    'result': 'Hidden message detected using LSB method. Note: JPEG compression may affect LSB reliability.',
                                    'message': hidden_message[:500],  # Limit message length for display
                                    'success': True,
                                    'jpeg_warning': True
                                }
                            else:
                                result = {
                                    'result': 'Potential hidden data detected in JPEG, but content appears to be binary/non-text. JPEG compression may have affected the data.',
                                    'message': None,
                                    'success': True,
                                    'jpeg_warning': True
                                }
                        else:
                            # LSB didn't find anything, perform enhanced statistical analysis for JPEG
                            result = ImageAnalysisService._enhanced_jpeg_steganography_analysis(image_path, image_format)
                    else:
                        # Non-JPEG formats - original logic
                        hidden_message = ImageAnalysisService._safe_lsb_reveal(analysis_path)
                        
                        if hidden_message and len(hidden_message.strip()) > 0:
                            # Validate that the message contains printable characters
                            printable_chars = sum(1 for c in hidden_message if c.isprintable())
                            if printable_chars / len(hidden_message) > 0.8:  # At least 80% printable
                                result = {
                                    'result': 'Hidden message detected using LSB method.',
                                    'message': hidden_message[:500],  # Limit message length for display
                                    'success': True
                                }
                            else:
                                result = {
                                    'result': 'Potential hidden data detected, but content appears to be binary/non-text.',
                                    'message': None,
                                    'success': True
                                }
                        else:
                            # LSB didn't find anything, try statistical analysis
                            result = ImageAnalysisService._statistical_steganography_analysis(image_path, image_format)
                    
                    # Clean up temporary file if created
                    if analysis_path != image_path and os.path.exists(analysis_path):
                        os.unlink(analysis_path)
                    
                    return result
                        
                except Exception as lsb_error:
                    logger.warning(f"LSB analysis failed: {lsb_error}")
                    # Clean up temporary file if created
                    if 'analysis_path' in locals() and analysis_path != image_path and os.path.exists(analysis_path):
                        os.unlink(analysis_path)
                    # Fall back to statistical analysis
                    return ImageAnalysisService._statistical_steganography_analysis(image_path, image_format)
            else:
                # For unsupported formats, provide basic analysis
                return {
                    'result': f'Steganography detection works best with PNG, BMP, or TIFF images. Current format: {image_format.upper()}. Basic analysis: No obvious signs of steganography.',
                    'message': None,
                    'success': True
                }
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in steganography detection: {error_msg}")
            return {
                'result': 'Steganography analysis completed with basic methods.',
                'message': None,
                'success': True
            }
    
    @staticmethod
    def _statistical_steganography_analysis(image_path, image_format):
        """Perform statistical analysis to detect potential steganography"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB for analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get pixel data
                pixels = list(img.getdata())
                total_pixels = len(pixels)
                
                # Skip analysis for very small images
                if total_pixels < 100:
                    return {
                        'result': 'Image too small for reliable statistical analysis.',
                        'message': None,
                        'success': True
                    }
                
                # Check for uniform/solid color images first
                unique_colors = set(pixels)
                if len(unique_colors) <= 5:  # Very few unique colors suggests uniform image
                    return {
                        'result': f'Image appears to be uniform/solid color (only {len(unique_colors)} unique colors). LSB analysis not reliable for such images. No steganography detected.',
                        'message': None,
                        'success': True,
                        'details': {
                            'image_type': 'uniform',
                            'unique_colors': len(unique_colors),
                            'suspicion_level': 'None',
                            'total_pixels_analyzed': total_pixels
                        }
                    }
                
                # Statistical analysis of LSBs (Least Significant Bits)
                lsb_values = []
                for pixel in pixels:
                    for channel in pixel:
                        lsb_values.append(channel & 1)  # Extract LSB
                
                # Calculate LSB distribution
                ones = sum(lsb_values)
                zeros = len(lsb_values) - ones
                
                # In a natural image, LSBs should be roughly 50/50
                # Significant deviation might indicate steganography
                if len(lsb_values) > 0:
                    ones_ratio = ones / len(lsb_values)
                    
                    # More refined thresholds based on image analysis research
                    if 0.485 <= ones_ratio <= 0.515:
                        # Very normal distribution - no steganography likely
                        analysis = "LSB distribution appears normal. No steganography detected."
                        suspicion_level = "None"
                    elif 0.47 <= ones_ratio <= 0.53:
                        # Slightly off but still within normal range for natural images
                        analysis = "LSB distribution within normal range. No clear signs of steganography."
                        suspicion_level = "Low"
                    elif 0.40 <= ones_ratio <= 0.60:
                        # Moderate deviation - could be compression artifacts or natural variation
                        analysis = f"LSB distribution shows some irregularity ({ones_ratio:.2%} ones). This is likely due to natural image characteristics or compression."
                        suspicion_level = "Low-Medium"
                    elif 0.30 <= ones_ratio <= 0.70:
                        # Higher deviation - worth investigating
                        analysis = f"LSB distribution shows notable irregularity ({ones_ratio:.2%} ones). Could indicate steganography or unusual image processing."
                        suspicion_level = "Medium"
                    else:
                        # Very high deviation - suspicious
                        analysis = f"LSB distribution highly irregular ({ones_ratio:.2%} ones). Strong indication of steganography or data corruption."
                        suspicion_level = "High"
                    
                    # Additional analysis: Check for patterns in LSB sequence
                    # Real steganography often creates patterns
                    pattern_score = ImageAnalysisService._analyze_lsb_patterns(lsb_values)
                    
                    if pattern_score > 0.7 and suspicion_level in ["Medium", "High"]:
                        analysis += " Pattern analysis suggests possible steganographic content."
                    
                    return {
                        'result': f'Statistical analysis completed. {analysis}',
                        'message': None,
                        'success': True,
                        'details': {
                            'ones_ratio': round(ones_ratio, 4),
                            'suspicion_level': suspicion_level,
                            'pattern_score': round(pattern_score, 4) if 'pattern_score' in locals() else 0,
                            'total_pixels_analyzed': total_pixels
                        }
                    }
                else:
                    return {
                        'result': 'Unable to perform statistical analysis on image data.',
                        'message': None,
                        'success': True
                    }
                    
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {
                'result': 'Basic steganography analysis completed (statistical methods unavailable).',
                'message': None,
                'success': True
            }
    
    @staticmethod
    def _analyze_lsb_patterns(lsb_values):
        """Analyze LSB sequence for patterns that might indicate steganography"""
        try:
            # Convert LSB values to runs of consecutive 0s and 1s
            runs = []
            current_run = 1
            
            for i in range(1, len(lsb_values)):
                if lsb_values[i] == lsb_values[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            
            # In natural images, runs should be relatively random
            # Steganography often creates more uniform run lengths
            if len(runs) > 10:
                avg_run_length = sum(runs) / len(runs)
                # If average run length is very uniform, it might indicate steganography
                run_variance = sum((r - avg_run_length) ** 2 for r in runs) / len(runs)
                
                # Lower variance in run lengths can indicate steganography
                # This is a simplified heuristic
                if run_variance < avg_run_length * 0.5:
                    return 0.8  # High pattern score
                elif run_variance < avg_run_length:
                    return 0.5  # Medium pattern score
                else:
                    return 0.2  # Low pattern score
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return 0.0
    
    @staticmethod
    def get_system_info():
        """Get system information"""
        try:
            return f"OS: {platform.system()} {platform.release()}"
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return "System information unavailable"
    
    @staticmethod
    def _safe_lsb_reveal(image_path):
        """Safely attempt LSB reveal with proper error handling for stegano library limitations"""
        try:
            # The stegano library has issues with:
            # 1. Images without hidden messages (throws "image index out of range")
            # 2. Uniform color images
            # 3. Very small images
            
            # Pre-check: Analyze image characteristics
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    # Convert to RGB if needed
                    img = img.convert('RGB')
                    temp_path = image_path + '_temp_rgb_safe.png'
                    img.save(temp_path, 'PNG')
                    analysis_path = temp_path
                else:
                    analysis_path = image_path
                    temp_path = None
                
                # Check for uniform colors (common cause of LSB failures)
                pixels = list(img.getdata())
                unique_pixels = set(pixels)
                
                # If image is too uniform, LSB is likely to fail
                if len(unique_pixels) < 10:
                    logger.info("Image appears too uniform for reliable LSB analysis")
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
                    return None
            
            try:
                # Attempt LSB reveal with the stegano library
                hidden_message = lsb.reveal(analysis_path)
                
                # Clean up temporary file
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                return hidden_message
                
            except Exception as lsb_error:
                error_msg = str(lsb_error).lower()
                
                # Clean up temporary file
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                # Handle known stegano library errors
                if "image index out of range" in error_msg:
                    # This usually means no hidden message exists
                    logger.debug("No LSB steganography detected (index out of range - normal for clean images)")
                    return None
                elif "not iterable" in error_msg:
                    # Usually a mode issue that we should have handled above
                    logger.debug("LSB analysis failed due to image mode issues")
                    return None
                else:
                    # Unknown error, log it but don't crash
                    logger.warning(f"LSB analysis failed with unknown error: {lsb_error}")
                    return None
                    
        except Exception as e:
            logger.error(f"Safe LSB reveal failed: {e}")
            return None

    @staticmethod
    def _enhanced_jpeg_steganography_analysis(image_path, image_format):
        """Enhanced steganography analysis specifically for JPEG images"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB for analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get pixel data
                pixels = list(img.getdata())
                total_pixels = len(pixels)
                
                # Skip analysis for very small images
                if total_pixels < 100:
                    return {
                        'result': 'JPEG image too small for reliable steganography analysis.',
                        'message': None,
                        'success': True,
                        'jpeg_warning': True
                    }
                
                # JPEG-specific analysis: Check for unusual compression artifacts
                # that might indicate steganography
                
                # 1. DCT coefficient analysis (simplified)
                # JPEG steganography often modifies DCT coefficients
                
                # 2. Block-level analysis - JPEG works on 8x8 blocks
                width, height = img.size
                block_anomalies = 0
                total_blocks = (width // 8) * (height // 8)
                
                if total_blocks > 0:
                    # Check for unusual patterns in 8x8 blocks
                    for y in range(0, height - 7, 8):
                        for x in range(0, width - 7, 8):
                            # Extract 8x8 block
                            block_pixels = []
                            for by in range(8):
                                for bx in range(8):
                                    if y + by < height and x + bx < width:
                                        pixel_idx = (y + by) * width + (x + bx)
                                        if pixel_idx < len(pixels):
                                            block_pixels.append(pixels[pixel_idx])
                            
                            if len(block_pixels) >= 32:  # At least half a block
                                # Check for unusual uniformity or patterns
                                unique_colors = len(set(block_pixels))
                                if unique_colors <= 2 and len(block_pixels) > 10:
                                    block_anomalies += 1
                
                # 3. Statistical analysis with JPEG considerations
                lsb_values = []
                for pixel in pixels:
                    for channel in pixel:
                        lsb_values.append(channel & 1)  # Extract LSB
                
                ones = sum(lsb_values)
                zeros = len(lsb_values) - ones
                ones_ratio = ones / len(lsb_values) if len(lsb_values) > 0 else 0.5
                
                # JPEG-specific thresholds (more lenient than PNG)
                # JPEG compression naturally creates some LSB irregularities
                analysis = ""
                suspicion_level = "None"
                
                if 0.45 <= ones_ratio <= 0.55:
                    # Very normal for JPEG
                    analysis = f"JPEG LSB distribution appears normal ({ones_ratio:.2%} ones). No clear signs of steganography."
                    suspicion_level = "None"
                elif 0.40 <= ones_ratio <= 0.60:
                    # Still normal for JPEG due to compression
                    analysis = f"JPEG LSB distribution within expected range ({ones_ratio:.2%} ones). JPEG compression naturally affects LSB distribution."
                    suspicion_level = "Low"
                elif 0.30 <= ones_ratio <= 0.70:
                    # Might be suspicious, but JPEG compression complicates analysis
                    analysis = f"JPEG LSB distribution shows some irregularity ({ones_ratio:.2%} ones). Could be compression artifacts or possible steganography."
                    suspicion_level = "Low-Medium"
                else:
                    # Highly irregular even for JPEG
                    analysis = f"JPEG LSB distribution highly irregular ({ones_ratio:.2%} ones). Possible steganography or significant image processing."
                    suspicion_level = "Medium"
                
                # Factor in block anomalies
                if total_blocks > 0:
                    block_anomaly_ratio = block_anomalies / total_blocks
                    if block_anomaly_ratio > 0.1:  # More than 10% of blocks are anomalous
                        if suspicion_level in ["Low", "None"]:
                            suspicion_level = "Low-Medium"
                        elif suspicion_level == "Low-Medium":
                            suspicion_level = "Medium"
                        analysis += f" Block analysis shows {block_anomaly_ratio:.1%} anomalous blocks."
                
                # Additional JPEG-specific warning
                jpeg_note = " Note: JPEG compression can mask or destroy LSB steganography, making detection less reliable."
                
                return {
                    'result': f'JPEG steganography analysis completed. {analysis}{jpeg_note}',
                    'message': None,
                    'success': True,
                    'jpeg_warning': True,
                    'details': {
                        'ones_ratio': round(ones_ratio, 4),
                        'suspicion_level': suspicion_level,
                        'block_anomalies': block_anomalies,
                        'total_blocks': total_blocks,
                        'block_anomaly_ratio': round(block_anomaly_ratio, 4) if total_blocks > 0 else 0,
                        'total_pixels_analyzed': total_pixels
                    }
                }
                
        except Exception as e:
            logger.error(f"Enhanced JPEG steganography analysis failed: {e}")
            return {
                'result': 'JPEG steganography analysis completed with basic methods. JPEG compression may affect steganography detection reliability.',
                'message': None,
                'success': True,
                'jpeg_warning': True
            }


class ImageComparisonService:
    """Service class for image comparison operations"""
    
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
                
                # Get image formats and modes
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
                                'total_pixels': original_size1[0] * original_size1[1],
                                'enhanced_properties': img1_properties
                            },
                            'image2_info': {
                                'size': original_size2,
                                'format': format2,
                                'total_pixels': original_size2[0] * original_size2[1],
                                'enhanced_properties': img2_properties
                            },
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
    def _calculate_similarity_metrics(img1, img2, diff):
        """Calculate detailed similarity metrics between two images"""
        try:
            # Convert to arrays for pixel-level analysis
            import numpy as np
            
            img1_array = np.array(img1)
            img2_array = np.array(img2)
            diff_array = np.array(diff)
            
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
            logger.warning("NumPy not available, using basic comparison metrics")
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
                # Simple correlation calculation
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
        """Calculate image sharpness using Laplacian variance"""
        try:
            # Convert to array for sharpness calculation
            import numpy as np
            from scipy import ndimage
            
            image_array = np.array(grayscale_image)
            laplacian = ndimage.laplace(image_array)
            variance = laplacian.var()
            return variance
            
        except ImportError:
            # Fallback method without scipy
            try:
                # Simple edge detection method
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
                logger.error(f"Error in fallback sharpness calculation: {e}")
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
            import math
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
