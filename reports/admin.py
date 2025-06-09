from django.contrib import admin
from .models import AnalysisReport, ComparisonReport

@admin.register(AnalysisReport)
class AnalysisReportAdmin(admin.ModelAdmin):
    list_display = ('report_title', 'user', 'analysis_image', 'created_at')
    list_filter = ('created_at', 'is_public', 'user')
    search_fields = ('report_title', 'user__username', 'analysis__image__original_filename')
    readonly_fields = ('created_at', 'updated_at')
    ordering = ('-created_at',)
    
    def analysis_image(self, obj):
        return obj.analysis.image.original_filename
    analysis_image.short_description = 'Analyzed Image'

@admin.register(ComparisonReport)
class ComparisonReportAdmin(admin.ModelAdmin):
    list_display = ('report_title', 'user', 'comparison_images', 'created_at')
    list_filter = ('created_at', 'is_public', 'user')
    search_fields = ('report_title', 'user__username')
    readonly_fields = ('created_at', 'updated_at')
    ordering = ('-created_at',)
    
    def comparison_images(self, obj):
        return f"{obj.comparison.image1.original_filename} vs {obj.comparison.image2.original_filename}"
    comparison_images.short_description = 'Compared Images'
