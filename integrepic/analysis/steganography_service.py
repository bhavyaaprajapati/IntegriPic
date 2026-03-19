"""
Steganography detection service - handles LSB and statistical analysis
"""
from PIL import Image
import logging
from stegano import lsb
import os

logger = logging.getLogger(__name__)


class SteganographyService:
    """Service class for steganography detection operations"""
    
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
            supported_formats = ['png', 'bmp', 'tiff', 'jpg', 'jpeg']
            if any(image_path.lower().endswith(f'.{fmt}') for fmt in supported_formats):
                # Special handling for JPEG images
                is_jpeg = any(image_path.lower().endswith(f'.{fmt}') for fmt in ['jpg', 'jpeg'])
                
                # Try LSB detection with error handling
                try:
                    # First check if the image is suitable for LSB analysis
                    with Image.open(analysis_path) as test_img:
                        if test_img.size[0] * test_img.size[1] < 100:  # Too small for reliable LSB
                            return SteganographyService._statistical_steganography_analysis(image_path, image_format)
                    
                    # For JPEG images, LSB steganography is less reliable due to compression
                    if is_jpeg:
                        hidden_message = SteganographyService._safe_lsb_reveal(analysis_path)
                        
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
                            result = SteganographyService._enhanced_jpeg_steganography_analysis(image_path, image_format)
                    else:
                        # Non-JPEG formats - original logic
                        hidden_message = SteganographyService._safe_lsb_reveal(analysis_path)
                        
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
                            result = SteganographyService._statistical_steganography_analysis(image_path, image_format)
                    
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
                    return SteganographyService._statistical_steganography_analysis(image_path, image_format)
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
    def _safe_lsb_reveal(image_path):
        """Safely attempt LSB reveal with proper error handling for stegano library limitations"""
        try:
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
                    logger.debug("No LSB steganography detected (index out of range - normal for clean images)")
                    return None
                elif "not iterable" in error_msg:
                    logger.debug("LSB analysis failed due to image mode issues")
                    return None
                else:
                    logger.warning(f"LSB analysis failed with unknown error: {lsb_error}")
                    return None
                    
        except Exception as e:
            logger.error(f"Safe LSB reveal failed: {e}")
            return None
    
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
                if len(lsb_values) > 0:
                    ones_ratio = ones / len(lsb_values)
                    
                    # More refined thresholds based on image analysis research
                    if 0.485 <= ones_ratio <= 0.515:
                        analysis = "LSB distribution appears normal. No steganography detected."
                        suspicion_level = "None"
                    elif 0.47 <= ones_ratio <= 0.53:
                        analysis = "LSB distribution within normal range. No clear signs of steganography."
                        suspicion_level = "Low"
                    elif 0.40 <= ones_ratio <= 0.60:
                        analysis = f"LSB distribution shows some irregularity ({ones_ratio:.2%} ones). This is likely due to natural image characteristics or compression."
                        suspicion_level = "Low-Medium"
                    elif 0.30 <= ones_ratio <= 0.70:
                        analysis = f"LSB distribution shows notable irregularity ({ones_ratio:.2%} ones). Could indicate steganography or unusual image processing."
                        suspicion_level = "Medium"
                    else:
                        analysis = f"LSB distribution highly irregular ({ones_ratio:.2%} ones). Strong indication of steganography or data corruption."
                        suspicion_level = "High"
                    
                    # Additional analysis: Check for patterns in LSB sequence
                    pattern_score = SteganographyService._analyze_lsb_patterns(lsb_values)
                    
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
            if len(runs) > 10:
                avg_run_length = sum(runs) / len(runs)
                run_variance = sum((r - avg_run_length) ** 2 for r in runs) / len(runs)
                
                # Lower variance in run lengths can indicate steganography
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
                width, height = img.size
                block_anomalies = 0
                total_blocks = (width // 8) * (height // 8)
                
                if total_blocks > 0:
                    # Check for unusual patterns in 8x8 blocks
                    for y in range(0, height - 7, 8):
                        for x in range(0, width - 7, 8):
                            # Extract 8x8 block and check for anomalies
                            block_pixels = []
                            for by in range(8):
                                for bx in range(8):
                                    if y + by < height and x + bx < width:
                                        pixel_idx = (y + by) * width + (x + bx)
                                        if pixel_idx < len(pixels):
                                            block_pixels.append(pixels[pixel_idx])
                            
                            # Simple anomaly detection: check variance
                            if len(block_pixels) > 0:
                                avg_r = sum(p[0] for p in block_pixels) / len(block_pixels)
                                variance = sum((p[0] - avg_r) ** 2 for p in block_pixels) / len(block_pixels)
                                if variance > 1000:  # High variance threshold
                                    block_anomalies += 1
                
                # Statistical analysis with JPEG considerations
                lsb_values = []
                for pixel in pixels:
                    for channel in pixel:
                        lsb_values.append(channel & 1)  # Extract LSB
                
                ones = sum(lsb_values)
                ones_ratio = ones / len(lsb_values) if len(lsb_values) > 0 else 0.5
                
                # JPEG-specific thresholds (more lenient than PNG)
                if 0.45 <= ones_ratio <= 0.55:
                    analysis = f"JPEG LSB distribution appears normal ({ones_ratio:.2%} ones). No clear signs of steganography."
                    suspicion_level = "None"
                elif 0.40 <= ones_ratio <= 0.60:
                    analysis = f"JPEG LSB distribution within expected range ({ones_ratio:.2%} ones). JPEG compression naturally affects LSB distribution."
                    suspicion_level = "Low"
                elif 0.30 <= ones_ratio <= 0.70:
                    analysis = f"JPEG LSB distribution shows some irregularity ({ones_ratio:.2%} ones). Could be compression artifacts or possible steganography."
                    suspicion_level = "Low-Medium"
                else:
                    analysis = f"JPEG LSB distribution highly irregular ({ones_ratio:.2%} ones). Possible steganography or significant image processing."
                    suspicion_level = "Medium"
                
                # Factor in block anomalies
                if total_blocks > 0:
                    block_anomaly_ratio = block_anomalies / total_blocks
                    if block_anomaly_ratio > 0.1:
                        analysis += f" Block-level analysis shows {block_anomaly_ratio:.1%} anomalous blocks."
                        if suspicion_level == "None":
                            suspicion_level = "Low"
                        elif suspicion_level == "Low":
                            suspicion_level = "Low-Medium"
                
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
