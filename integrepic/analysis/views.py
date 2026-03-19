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




def _analyze_single_file(request, uploaded_file):
   """Analyze a single uploaded file. Returns the ImageAnalysis instance (completed or failed)."""
   temp_file_path = None
   try:
       with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
           for chunk in uploaded_file.chunks():
               temp_file.write(chunk)
           temp_file_path = temp_file.name


       with Image.open(temp_file_path) as img:
           image_format = img.format
           width, height = img.size


       file_hash = ImageAnalysisService.calculate_sha256_from_path(temp_file_path)
       start_time = time.time()


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
           metadata = ImageAnalysisService.extract_metadata(temp_file_path)
           analysis.metadata = metadata


           if image_format.upper() in ['JPEG', 'JPG']:
               ela_result = ImageAnalysisService.perform_ela_analysis(temp_file_path)
               analysis.ela_analysis_performed = True
               analysis.ela_results = ela_result


           stego_result = ImageAnalysisService.detect_steganography(temp_file_path)
           analysis.steganography_result = stego_result['result']
           analysis.steganography_message = stego_result['message']


           analysis.analysis_duration = time.time() - start_time
           analysis.status = 'completed'
           analysis.save()


       except Exception as e:
           logger.error(f"Error during analysis of {uploaded_file.name}: {e}")
           analysis.status = 'failed'
           analysis.error_message = str(e)
           analysis.save()


       return analysis


   except Exception as e:
       logger.error(f"Error processing file {uploaded_file.name}: {e}")
       return None


   finally:
       if temp_file_path and os.path.exists(temp_file_path):
           os.unlink(temp_file_path)




@login_required
def upload_image(request):
   """Handle image upload and immediate analysis (single or multiple files)"""
   if request.method == 'POST':
       uploaded_files = request.FILES.getlist('image')


       if not uploaded_files:
           form = ImageUploadForm()
           messages.error(request, 'Please select at least one image file.')
           return render(request, 'analysis/upload.html', {'form': form})


       # Per-file validation
       valid_files = []
       for f in uploaded_files:
           if f.size > 10 * 1024 * 1024:
               messages.error(request, f'"{f.name}" exceeds 10MB limit and was skipped.')
               continue
           if not f.content_type.startswith('image/'):
               messages.error(request, f'"{f.name}" is not a valid image and was skipped.')
               continue
           valid_files.append(f)


       if not valid_files:
           form = ImageUploadForm()
           return render(request, 'analysis/upload.html', {'form': form})


       # Process each valid file
       analysis_ids = []
       for f in valid_files:
           analysis = _analyze_single_file(request, f)
           if analysis:
               analysis_ids.append(analysis.pk)


       if not analysis_ids:
           messages.error(request, 'All uploads failed. Please try again.')
           return render(request, 'analysis/upload.html', {'form': ImageUploadForm()})


       if len(analysis_ids) == 1:
           messages.success(request, f'Image "{valid_files[0].name}" analyzed successfully!')
           return redirect('analysis:analysis_detail', pk=analysis_ids[0])


       messages.success(request, f'{len(analysis_ids)} image(s) analyzed successfully!')
       ids_param = ','.join(map(str, analysis_ids))
       return redirect(f"{reverse('analysis:batch_results')}?ids={ids_param}")


   form = ImageUploadForm()
   return render(request, 'analysis/upload.html', {'form': form})




@login_required
def batch_results(request):
   """Show results for a batch of uploaded images"""
   ids = [
       int(i) for i in request.GET.get('ids', '').split(',')
       if i.strip().isdigit()
   ]
   analyses = ImageAnalysis.objects.filter(pk__in=ids, user=request.user).order_by('created_at')
   completed = analyses.filter(status='completed').count()
   failed = analyses.filter(status='failed').count()
   context = {
       'analyses': analyses,
       'completed': completed,
       'failed': failed,
   }
   return render(request, 'analysis/batch_results.html', context)




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
