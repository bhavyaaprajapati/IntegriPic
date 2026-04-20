from django.db import models
from django.contrib.auth.models import User
import uuid


def upload_to_analysis(instance, filename):
    """Generate upload path for analysis results"""
    ext = filename.split('.')[-1]
    filename = f"analysis_{uuid.uuid4().hex}.{ext}"
    return f'analysis/{filename}'


class ImageAnalysis(models.Model):
    """Model for image analysis results - no image storage"""
    ANALYSIS_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses')
    
    # Image information (stored as metadata, no file storage)
    original_filename = models.CharField(max_length=255)
    file_size = models.PositiveIntegerField(help_text="File size in bytes")
    image_format = models.CharField(max_length=10)
    image_width = models.PositiveIntegerField()
    image_height = models.PositiveIntegerField()
    sha256_hash = models.CharField(max_length=64)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=ANALYSIS_STATUS_CHOICES, default='pending')
    
    # Metadata fields
    metadata = models.JSONField(default=dict, blank=True)
    
    # ELA Analysis (store results as JSON/text, no image files)
    ela_analysis_performed = models.BooleanField(default=False)
    ela_quality = models.PositiveIntegerField(default=95)
    ela_results = models.JSONField(default=dict, blank=True)
    ela_heatmap_b64 = models.TextField(
        blank=True, null=True,
        help_text="Base64-encoded PNG of the ELA difference heatmap visualization"
    )

    # Steganography Analysis
    steganography_result = models.TextField(blank=True, null=True)
    steganography_message = models.TextField(blank=True, null=True)

    # RGB Color Analysis (stored as JSON)
    rgb_histogram_data = models.JSONField(default=dict, blank=True, help_text="RGB histogram bins for color analysis")

    # Perceptual Hash Fields (for near-duplicate detection)
    phash = models.CharField(max_length=64, blank=True, null=True, help_text="Perceptual hash for near-duplicate detection")
    dhash = models.CharField(max_length=64, blank=True, null=True, help_text="Difference hash for near-duplicate detection")
    ahash = models.CharField(max_length=64, blank=True, null=True, help_text="Average hash for near-duplicate detection")

    # Metadata Timeline Flags
    timeline_flags = models.JSONField(
        default=list, blank=True,
        help_text="List of detected metadata timestamp inconsistencies"
    )

    # Copy-Move Forgery Detection
    copy_move_result = models.JSONField(
        default=dict, blank=True,
        help_text="SIFT-based copy-move forgery detection results"
    )

    # Deepfake & AI Generation Detection
    deepfake_probability = models.FloatField(
        blank=True, null=True, 
        help_text="Confidence score (0.0 to 100.0) that the image is AI-generated"
    )
    deepfake_notes = models.TextField(
        blank=True, null=True,
        help_text="Explanation or heuristic reasoning for the AI detection score"
    )

    # System Information
    os_info = models.CharField(max_length=100, blank=True)
    analysis_duration = models.FloatField(blank=True, null=True, help_text="Duration in seconds")
    
    # Error handling
    error_message = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Image Analysis"
        verbose_name_plural = "Image Analyses"
    
    def __str__(self):
        return f"Analysis of {self.original_filename} - {self.status}"
    
    @property
    def file_size_mb(self):
        return round(self.file_size / (1024 * 1024), 2)


class ImageComparison(models.Model):
    """Model for image comparison results - no image storage"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comparisons')
    
    # Information about first image
    image1_filename = models.CharField(max_length=255)
    image1_hash = models.CharField(max_length=64)
    image1_size = models.PositiveIntegerField()
    
    # Information about second image
    image2_filename = models.CharField(max_length=255)
    image2_hash = models.CharField(max_length=64)
    image2_size = models.PositiveIntegerField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Comparison results (stored as data, no difference image files)
    are_identical = models.BooleanField(default=False)
    similarity_score = models.FloatField(default=0.0)
    comparison_results = models.JSONField(default=dict, blank=True)
    comparison_notes = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Image Comparison"
        verbose_name_plural = "Image Comparisons"
    
    def __str__(self):
        return f"Comparison: {self.image1_filename} vs {self.image2_filename}"


class AuditLog(models.Model):
    """Immutable audit log for chain of custody and forensic integrity"""
    ACTION_CHOICES = [
        ('upload', 'Image Uploaded'),
        ('view_analysis', 'Analysis Viewed'),
        ('report_generate', 'Report Generated'),
        ('report_download', 'Report Downloaded'),
        ('comparison_create', 'Comparison Created'),
        ('export_pdf', 'PDF Exported'),
    ]

    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, related_name='audit_logs'
    )
    action_type = models.CharField(max_length=30, choices=ACTION_CHOICES)
    analysis = models.ForeignKey(
        'ImageAnalysis', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='audit_logs'
    )
    comparison = models.ForeignKey(
        'ImageComparison', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='audit_logs'
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    result_hash = models.CharField(
        max_length=64, blank=True,
        help_text="SHA256 of the analysis result at time of action (for integrity)"
    )
    extra_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Audit Log Entry"
        verbose_name_plural = "Audit Log Entries"

    def __str__(self):
        return f"{self.user} - {self.action_type} - {self.timestamp}"
