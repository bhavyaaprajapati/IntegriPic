"""
Main services module - imports and consolidates all service classes
"""
from .image_analysis_service import ImageAnalysisService as BaseImageAnalysisService
from .steganography_service import SteganographyService
from .enhanced_comparison_service import ImageComparisonService as BaseImageComparisonService
import logging

logger = logging.getLogger(__name__)


class ImageAnalysisService(BaseImageAnalysisService):
    """Enhanced Image Analysis Service with steganography detection"""
    
    @staticmethod
    def detect_steganography(image_path):
        """Detect steganography using the dedicated service"""
        return SteganographyService.detect_steganography(image_path)


class ImageComparisonService(BaseImageComparisonService):
    """Enhanced Image Comparison Service"""
    pass
