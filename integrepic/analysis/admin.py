from django.contrib import admin
from .models import ImageAnalysis, ImageComparison, AuditLog

@admin.register(ImageAnalysis)
class ImageAnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'original_filename', 'user', 'status', 'created_at', 'analysis_duration', 'file_size_mb')
    list_filter = ('status', 'created_at', 'user', 'image_format')
    search_fields = ('original_filename', 'user__username', 'sha256_hash')
    readonly_fields = ('created_at', 'updated_at', 'analysis_duration', 'sha256_hash', 'file_size', 'image_format', 'image_width', 'image_height')
    ordering = ('-created_at',)
    
    def file_size_mb(self, obj):
        return f"{obj.file_size_mb} MB"
    file_size_mb.short_description = 'File Size'

@admin.register(ImageComparison)
class ImageComparisonAdmin(admin.ModelAdmin):
    list_display = ('id', 'image1_filename', 'image2_filename', 'user', 'are_identical', 'similarity_score', 'created_at')
    list_filter = ('are_identical', 'created_at', 'user')
    search_fields = ('image1_filename', 'image2_filename', 'user__username', 'image1_hash', 'image2_hash')
    readonly_fields = ('created_at', 'image1_hash', 'image2_hash', 'comparison_results')
    ordering = ('-created_at',)

@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'action_type', 'analysis', 'timestamp', 'ip_address')
    list_filter = ('action_type', 'timestamp')
    search_fields = ('user__username', 'ip_address')
    readonly_fields = ('user', 'action_type', 'analysis', 'comparison', 'timestamp', 'ip_address', 'result_hash', 'extra_data')
    ordering = ('-timestamp',)

    # Prevent any modification of audit logs
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
