from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, Http404
from django.core.exceptions import ValidationError
from django.urls import reverse
from django.utils import timezone
from django.db import transaction
from PIL import Image
import os
import time
import tempfile
import logging

from .models import ImageAnalysis, ImageComparison
from .services import ImageAnalysisService, ImageComparisonService
from .forms import ImageUploadForm, ImageComparisonForm

logger = logging.getLogger(__name__)


@login_required
def dashboard(request):
    """Main dashboard view"""
    # Calculate user statistics
    total_analyses = ImageAnalysis.objects.filter(user=request.user).count()
    total_comparisons = ImageComparison.objects.filter(user=request.user).count()
    
    # Count total unique images uploaded by user
    total_images = ImageAnalysis.objects.filter(user=request.user).values('sha256_hash').distinct().count()
    
    # Recent analyses
    recent_analyses = ImageAnalysis.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    # Count reports (assuming reports app is properly set up)
    try:
        from reports.models import AnalysisReport, ComparisonReport
        total_reports = (
            AnalysisReport.objects.filter(user=request.user).count() +
            ComparisonReport.objects.filter(user=request.user).count()
        )
    except ImportError:
        total_reports = 0
    
    stats = {
        'total_images': total_images,
        'total_analyses': total_analyses,
        'total_comparisons': total_comparisons,
        'total_reports': total_reports,
    }
    
    context = {
        'recent_analyses': recent_analyses,
        'stats': stats,
    }
    return render(request, 'analysis/dashboard.html', context)


@login_required
def upload_image(request):
    """Handle image upload and immediate analysis"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_file = form.cleaned_data['image']
                
                # Validate file size (10MB limit)
                if uploaded_file.size > 10 * 1024 * 1024:
                    messages.error(request, 'File size must be less than 10MB.')
                    return render(request, 'analysis/upload.html', {'form': form})
                
                # Save file temporarily for analysis
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    for chunk in uploaded_file.chunks():
                        temp_file.write(chunk)
                    temp_file_path = temp_file.name
                
                try:
                    # Get image information
                    with Image.open(temp_file_path) as img:
                        image_format = img.format
                        width, height = img.size
                    
                    # Calculate hash
                    file_hash = ImageAnalysisService.calculate_sha256_from_path(temp_file_path)
                    
                    # Perform analysis immediately
                    start_time = time.time()
                    
                    # Create analysis instance
                    analysis = ImageAnalysis.objects.create(
                        user=request.user,
                        original_filename=uploaded_file.name,
                        file_size=uploaded_file.size,
                        image_format=image_format,
                        image_width=width,
                        image_height=height,
                        sha256_hash=file_hash,
                        status='processing',
                        os_info=ImageAnalysisService.get_system_info()
                    )
                    
                    try:
                        # Extract metadata
                        metadata = ImageAnalysisService.extract_metadata(temp_file_path)
                        analysis.metadata = metadata
                        
                        # Perform ELA analysis for JPEG images
                        if image_format.upper() in ['JPEG', 'JPG']:
                            ela_result = ImageAnalysisService.perform_ela_analysis(temp_file_path)
                            analysis.ela_analysis_performed = True
                            analysis.ela_results = ela_result
                        
                        # Perform steganography analysis
                        stego_result = ImageAnalysisService.detect_steganography(temp_file_path)
                        analysis.steganography_result = stego_result['result']
                        analysis.steganography_message = stego_result['message']
                        
                        # Calculate analysis duration
                        analysis.analysis_duration = time.time() - start_time
                        analysis.status = 'completed'
                        analysis.save()
                        
                        messages.success(request, f'Image "{uploaded_file.name}" analyzed successfully!')
                        return redirect('analysis:analysis_detail', pk=analysis.pk)
                        
                    except Exception as e:
                        logger.error(f"Error during analysis: {e}")
                        analysis.status = 'failed'
                        analysis.error_message = str(e)
                        analysis.save()
                        messages.error(request, 'Analysis failed. Please try again.')
                        return redirect('analysis:analysis_detail', pk=analysis.pk)
                        
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                    
            except Exception as e:
                logger.error(f"Error uploading image: {e}")
                messages.error(request, 'An error occurred while uploading the image. Please try again.')
    else:
        form = ImageUploadForm()
    
    return render(request, 'analysis/upload.html', {'form': form})


@login_required
def analysis_detail(request, pk):
    """Display analysis results"""
    analysis = get_object_or_404(ImageAnalysis, pk=pk, user=request.user)
    
    context = {
        'analysis': analysis,
    }
    return render(request, 'analysis/analysis_detail.html', context)


@login_required
def analysis_list(request):
    """List all analyses for the user"""
    analyses = ImageAnalysis.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'analyses': analyses,
    }
    return render(request, 'analysis/analysis_list.html', context)


# For backward compatibility - redirect image_detail to analysis_list
@login_required
def image_detail(request, pk):
    """Redirect to analysis list - for backward compatibility"""
    return redirect('analysis:analysis_list')


# For backward compatibility - redirect image_list to analysis_list  
@login_required
def image_list(request):
    """Redirect to analysis list - for backward compatibility"""
    return redirect('analysis:analysis_list')


# For backward compatibility - redirect analyze_image to upload
@login_required
def analyze_image(request, pk=None):
    """Redirect to upload page - for backward compatibility"""
    return redirect('analysis:upload')


@login_required
def compare_images(request):
    """Handle image comparison"""
    if request.method == 'POST':
        form = ImageComparisonForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                image1_file = form.cleaned_data['image1']
                image2_file = form.cleaned_data['image2']
                
                # Save files temporarily for comparison
                temp_files = []
                try:
                    # Save first image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image1_file.name)[1]) as temp_file1:
                        for chunk in image1_file.chunks():
                            temp_file1.write(chunk)
                        temp_file1_path = temp_file1.name
                        temp_files.append(temp_file1_path)
                    
                    # Save second image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image2_file.name)[1]) as temp_file2:
                        for chunk in image2_file.chunks():
                            temp_file2.write(chunk)
                        temp_file2_path = temp_file2.name
                        temp_files.append(temp_file2_path)
                    
                    # Calculate hashes
                    hash1 = ImageAnalysisService.calculate_sha256_from_path(temp_file1_path)
                    hash2 = ImageAnalysisService.calculate_sha256_from_path(temp_file2_path)
                    
                    # Check if comparison already exists
                    existing_comparison = ImageComparison.objects.filter(
                        user=request.user,
                        image1_hash=hash1,
                        image2_hash=hash2
                    ).first()
                    
                    # Also check reverse comparison
                    if not existing_comparison:
                        existing_comparison = ImageComparison.objects.filter(
                            user=request.user,
                            image1_hash=hash2,
                            image2_hash=hash1
                        ).first()
                    
                    if existing_comparison:
                        messages.info(request, 'This comparison already exists.')
                        return redirect('analysis:comparison_detail', pk=existing_comparison.pk)
                    
                    # Perform comparison
                    comparison_result = ImageComparisonService.compare_images(
                        temp_file1_path, temp_file2_path
                    )
                    
                    if comparison_result['success']:
                        # Create comparison instance
                        comparison = ImageComparison.objects.create(
                            user=request.user,
                            image1_filename=image1_file.name,
                            image1_hash=hash1,
                            image1_size=image1_file.size,
                            image2_filename=image2_file.name,
                            image2_hash=hash2,
                            image2_size=image2_file.size,
                            are_identical=comparison_result['are_identical'],
                            similarity_score=comparison_result.get('similarity_score', 0),
                            comparison_results=comparison_result.get('details', {}),
                            comparison_notes=comparison_result['message']
                        )
                        
                        messages.success(request, 'Image comparison completed successfully!')
                        return redirect('analysis:comparison_detail', pk=comparison.pk)
                    else:
                        messages.error(request, f'Comparison failed: {comparison_result["message"]}')
                        
                finally:
                    # Clean up temporary files
                    for temp_file_path in temp_files:
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                        
            except Exception as e:
                logger.error(f"Error during comparison: {e}")
                messages.error(request, 'An error occurred during comparison.')
    else:
        form = ImageComparisonForm()
    
    context = {
        'form': form,
    }
    return render(request, 'analysis/compare.html', context)


@login_required
def comparison_detail(request, pk):
    """Display comparison results"""
    comparison = get_object_or_404(ImageComparison, pk=pk, user=request.user)
    
    context = {
        'comparison': comparison,
    }
    return render(request, 'analysis/comparison_detail.html', context)


@login_required
def comparison_list(request):
    """List all comparisons for the user"""
    comparisons = ImageComparison.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'comparisons': comparisons,
    }
    return render(request, 'analysis/comparison_list.html', context)
