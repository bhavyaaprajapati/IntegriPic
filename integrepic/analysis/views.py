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
from .image_analysis_service import AuditService
from .forms import ImageUploadForm, ImageComparisonForm
from .visualization_service import VisualizationService


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

           # Detect metadata timeline inconsistencies
           timeline_flags = ImageAnalysisService.detect_timeline_inconsistencies(metadata)
           analysis.timeline_flags = timeline_flags

           # Extract RGB histogram data and color statistics
           rgb_data = ImageAnalysisService.extract_rgb_data(temp_file_path)
           analysis.rgb_histogram_data = rgb_data

           # Compute perceptual hashes for near-duplicate detection
           hashes = ImageAnalysisService.compute_perceptual_hashes(temp_file_path)
           analysis.phash = hashes.get('phash')
           analysis.dhash = hashes.get('dhash')
           analysis.ahash = hashes.get('ahash')

           if image_format.upper() in ['JPEG', 'JPG']:
               ela_result = ImageAnalysisService.perform_ela_analysis(temp_file_path)
               analysis.ela_analysis_performed = True
               if ela_result:
                   # Extract heatmap separately to avoid storing large base64 in JSONField
                   heatmap_b64 = ela_result.pop('heatmap_b64', None)
                   analysis.ela_results = ela_result
                   analysis.ela_heatmap_b64 = heatmap_b64


           stego_result = ImageAnalysisService.detect_steganography(temp_file_path)
           analysis.steganography_result = stego_result['result']
           analysis.steganography_message = stego_result['message']

           # Detect copy-move forgery
           copy_move_result = ImageAnalysisService.detect_copy_move(temp_file_path)
           analysis.copy_move_result = copy_move_result

           # Compute AI generation probability using forensic heuristics
           ai_prob, ai_notes = ImageAnalysisService.compute_ai_probability(
               analysis.ela_results,
               analysis.copy_move_result,
               analysis.metadata
           )
           analysis.deepfake_probability = ai_prob
           analysis.deepfake_notes = ai_notes

           analysis.analysis_duration = time.time() - start_time
           analysis.status = 'completed'
           analysis.save()

           # Log upload action
           AuditService.log(request, 'upload', analysis=analysis)

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


def _ela_confidence_label(ela_pct):
    """Convert ELA significant_pixels_percentage to a label"""
    if ela_pct is None:
        return "N/A"
    elif ela_pct < 1.0:
        return "Low"
    elif ela_pct < 5.0:
        return "Moderate"
    else:
        return "High"


@login_required
def analysis_detail(request, pk):
   """Display analysis results with interactive visualizations"""
   analysis = get_object_or_404(ImageAnalysis, pk=pk, user=request.user)

   # Log view action
   AuditService.log(request, 'view_analysis', analysis=analysis)

   # Generate visualizations
   viz_service = VisualizationService()

   charts = {
       'file_info_chart': viz_service.create_file_info_chart(
           analysis.file_size,
           analysis.image_format,
           analysis.image_width,
           analysis.image_height
       ),
       'properties_chart': viz_service.create_image_properties_chart(
           analysis.image_width,
           analysis.image_height,
           analysis.file_size
       ),
       'timeline_chart': viz_service.create_analysis_timeline(
           analysis.created_at,
           analysis.updated_at,
           analysis.analysis_duration
       ),
       'metadata_chart': viz_service.create_metadata_summary(analysis.metadata),
       'ela_chart': viz_service.create_ela_analysis_chart(analysis.ela_results) if analysis.ela_analysis_performed else None,
       'steganography_chart': viz_service.create_steganography_chart(analysis.steganography_result),
   }

   # Generate RGB histogram charts from pre-computed RGB data
   rgb_charts = {}
   try:
       rgb_data = analysis.rgb_histogram_data
       if rgb_data:
           rgb_charts = {
               'rgb_histogram': viz_service.create_rgb_histogram(rgb_data=rgb_data),
               'color_distribution': viz_service.create_color_distribution_chart(rgb_data=rgb_data),
               'color_stats': viz_service.create_color_space_analysis(rgb_data=rgb_data),
           }
       else:
           rgb_charts = {
               'rgb_histogram': None,
               'color_distribution': None,
               'color_stats': None,
           }
   except Exception as e:
       logger.error(f"Error generating RGB charts: {e}")
       rgb_charts = {
           'rgb_histogram': None,
           'color_distribution': None,
           'color_stats': None,
       }

   charts.update(rgb_charts)

   # Generate geolocation charts if GPS metadata is available
   gps_charts = {}
   try:
       if analysis.metadata and 'GPSInfo' in analysis.metadata:
           gps_charts = {
               'geolocation_map': viz_service.create_geolocation_map(
                   analysis.metadata,
                   analysis.original_filename
               ),
               'location_info_card': viz_service.create_location_info_card(analysis.metadata),
           }
       else:
           gps_charts = {
               'geolocation_map': None,
               'location_info_card': None,
           }
   except Exception as e:
       logger.error(f"Error generating geolocation charts: {e}")
       gps_charts = {
           'geolocation_map': None,
           'location_info_card': None,
       }

   charts.update(gps_charts)

   # Build confidence scores context
   ela_pct = analysis.ela_results.get('significant_pixels_percentage') if analysis.ela_results else None
   confidence = {
       'ela_tamper_confidence': round(ela_pct, 1) if ela_pct is not None else None,
       'ela_tamper_label': _ela_confidence_label(ela_pct),
       'ai_probability': round(analysis.deepfake_probability, 1) if analysis.deepfake_probability is not None else None,
       'copy_move_match_count': analysis.copy_move_result.get('match_count', 0) if analysis.copy_move_result else 0,
       'stego_detected': analysis.steganography_result and 'detected' in analysis.steganography_result.lower(),
   }

   context = {
       'analysis': analysis,
       'charts': charts,
       'confidence': confidence,
       'ela_heatmap_b64': analysis.ela_heatmap_b64,
   }
   return render(request, 'analysis/analysis_detail.html', context)


@login_required
def analysis_status(request, pk):
   """Return JSON status for a specific analysis (for AJAX polling)"""
   analysis = get_object_or_404(ImageAnalysis, pk=pk, user=request.user)
   return JsonResponse({
       'status': analysis.status,
       'progress_label': {
           'pending': 'Queued',
           'processing': 'Analyzing...',
           'completed': 'Complete',
           'failed': 'Failed',
       }.get(analysis.status, analysis.status),
       'error_message': analysis.error_message if analysis.status == 'failed' else None,
   })


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
                   # Clean up temporary files (with retry for Windows file locking)
                   for temp_file_path in temp_files:
                       if os.path.exists(temp_file_path):
                           try:
                               os.unlink(temp_file_path)
                           except OSError:
                               # File still locked on Windows - let OS clean it up
                               import gc
                               gc.collect()  # Force garbage collection to release file handles
                               try:
                                   os.unlink(temp_file_path)
                               except OSError:
                                   logger.warning(f"Could not delete temp file: {temp_file_path}")
                      
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

   viz_service = VisualizationService()
   charts = {}

   try:
       cr = comparison.comparison_results or {}

       # Similarity gauge chart
       charts['similarity_gauge'] = viz_service.create_similarity_gauge(comparison.similarity_score)

       # Channel similarity bar chart from color_analysis dict
       ca = cr.get('color_analysis', {})
       if ca:
           charts['channel_comparison'] = viz_service.create_channel_comparison_chart(ca)
       else:
           charts['channel_comparison'] = None

       # Difference region highlight if bounding box exists
       diff_region = cr.get('difference_region', {})
       charts['has_diff_region'] = diff_region.get('has_differences', False)
       charts['diff_bounding_box'] = diff_region.get('bounding_box')

   except Exception as e:
       logger.error(f"Error generating comparison charts: {e}")

   context = {
       'comparison': comparison,
       'charts': charts,
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

def hex_to_hash(hexstr):
    try:
        return int(hexstr, 16)
    except:
        return 0

def hamming_distance(h1, h2):
    if not h1 or not h2:
        return 999
    try:
        val1 = int(str(h1), 16)
        val2 = int(str(h2), 16)
        return bin(val1 ^ val2).count('1')
    except:
        return 999

@login_required
def network_graph(request):
    """Display vis.js network graph of perceptual hash matches"""
    import json
    analyses = ImageAnalysis.objects.filter(user=request.user, status='completed')
    
    nodes = []
    edges = []
    
    for a in analyses:
        nodes.append({
            'id': a.id,
            'label': a.original_filename,
            'title': f"Hash: {a.phash}",
            'group': str(a.created_at.date())
        })
        
    analysis_list = list(analyses)
    for i in range(len(analysis_list)):
        for j in range(i+1, len(analysis_list)):
            a1 = analysis_list[i]
            a2 = analysis_list[j]
            
            p_dist = hamming_distance(a1.phash, a2.phash)
            
            if p_dist < 15:
                edges.append({
                    'from': a1.id,
                    'to': a2.id,
                    'title': f"Hamming Distance: {p_dist}",
                    'value': max(1, 15 - p_dist)
                })
                
    context = {
        'nodes': json.dumps(nodes),
        'edges': json.dumps(edges),
    }
    return render(request, 'analysis/network_graph.html', context)

from django.views.decorators.csrf import csrf_exempt

@login_required
@csrf_exempt
def bot_ask_global_view(request):
    import json
    from django.http import JsonResponse
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query = data.get('query', '')
            history = data.get('history', [])
            analysis_id = data.get('analysis_id', None)
            
            if analysis_id:
                analysis = get_object_or_404(ImageAnalysis, pk=analysis_id, user=request.user)
            else:
                analysis = None
                
            from .bot_service import ForensicBotService
            reply = ForensicBotService.answer_query(query, analysis, history)
            
            return JsonResponse({'reply': reply})
        except Exception as e:
            return JsonResponse({'reply': f'Error processing query: {str(e)}'}, status=500)
    return JsonResponse({'reply': 'Method not allowed'}, status=405)
