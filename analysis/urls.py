from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_image, name='upload'),
    path('images/', views.image_list, name='image_list'),
    path('images/<int:pk>/', views.image_detail, name='image_detail'),
    path('images/<int:pk>/analyze/', views.analyze_image, name='analyze_image'),
    path('analyses/', views.analysis_list, name='analysis_list'),
    path('analyses/<int:pk>/', views.analysis_detail, name='analysis_detail'),
    path('compare/', views.compare_images, name='compare'),
    path('comparisons/', views.comparison_list, name='comparison_list'),
    path('comparisons/<int:pk>/', views.comparison_detail, name='comparison_detail'),
]
