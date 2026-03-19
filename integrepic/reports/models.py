from django.db import models
from django.contrib.auth.models import User
from analysis.models import ImageAnalysis, ImageComparison


class AnalysisReport(models.Model):
    """Model for generated analysis reports"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reports')
    analysis = models.OneToOneField(ImageAnalysis, on_delete=models.CASCADE, related_name='report')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Report metadata
    report_title = models.CharField(max_length=255)
    is_public = models.BooleanField(default=False)  # For future sharing features
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Analysis Report"
        verbose_name_plural = "Analysis Reports"
    
    def __str__(self):
        return f"Report for {self.analysis.original_filename}"


class ComparisonReport(models.Model):
    """Model for generated comparison reports"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comparison_reports')
    comparison = models.OneToOneField(ImageComparison, on_delete=models.CASCADE, related_name='report')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Report metadata
    report_title = models.CharField(max_length=255)
    is_public = models.BooleanField(default=False)  # For future sharing features
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Comparison Report"
        verbose_name_plural = "Comparison Reports"
    
    def __str__(self):
        return f"Comparison Report: {self.comparison.image1_filename} vs {self.comparison.image2_filename}"
