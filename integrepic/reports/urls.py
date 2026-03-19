from django.urls import path
from . import views

app_name = 'reports'

urlpatterns = [
    # User report URLs
    path('', views.user_reports, name='user_reports'),
    path('list/', views.user_reports, name='report_list'),  # Alias for compatibility
    path('generate/analysis/<int:analysis_pk>/', views.generate_analysis_report, name='generate_analysis_report'),
    path('generate/comparison/<int:comparison_pk>/', views.generate_comparison_report, name='generate_comparison_report'),
    path('view/<int:report_id>/', views.view_report, name='view_report'),
    path('download/<int:report_id>/', views.download_report, name='download_report'),
    
    # Admin URLs
    path('admin/', views.admin_reports, name='admin_reports'),
    path('admin/delete/<int:report_id>/', views.delete_report, name='delete_report'),
    path('admin/stats/', views.system_stats, name='system_stats'),
]
