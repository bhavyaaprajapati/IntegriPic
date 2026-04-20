from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse, Http404
from django.template.loader import render_to_string
from django.utils import timezone
from django.db.models import Q
from analysis.models import ImageAnalysis, ImageComparison
from .models import AnalysisReport, ComparisonReport
import itertools
import logging

logger = logging.getLogger(__name__)

# Try to import xhtml2pdf for PDF generation
try:
    from xhtml2pdf import pisa
    PDF_AVAILABLE = True
except ImportError as e:
    PDF_AVAILABLE = False
    logger.warning(f"PDF generation not available: {e}")


@login_required
def generate_analysis_report(request, analysis_pk):
    """Generate and save analysis report"""
    analysis = get_object_or_404(ImageAnalysis, pk=analysis_pk, user=request.user)
    
    # Get or create report
    report, created = AnalysisReport.objects.get_or_create(
        analysis=analysis,
        user=request.user,
        defaults={
            'report_title': f"Analysis Report - {analysis.original_filename}"
        }
    )
    
    if created:
        messages.success(request, 'Analysis report generated successfully!')
    else:
        messages.info(request, 'Report already exists. Displaying existing report.')
    
    return redirect('reports:view_report', report_type='analysis', report_id=report.id)


@login_required
def generate_comparison_report(request, comparison_pk):
    """Generate and save comparison report"""
    comparison = get_object_or_404(ImageComparison, pk=comparison_pk, user=request.user)
    
    # Get or create report
    report, created = ComparisonReport.objects.get_or_create(
        comparison=comparison,
        user=request.user,
        defaults={
            'report_title': f"Comparison Report - {comparison.image1_filename} vs {comparison.image2_filename}"
        }
    )
    
    if created:
        messages.success(request, 'Comparison report generated successfully!')
    else:
        messages.info(request, 'Report already exists. Displaying existing report.')
    
    return redirect('reports:view_report', report_type='comparison', report_id=report.id)


@login_required
def view_report(request, report_type, report_id):
    """View a specific report"""
    if report_type == 'analysis':
        report = AnalysisReport.objects.filter(id=report_id).first()
    elif report_type == 'comparison':
        report = ComparisonReport.objects.filter(id=report_id).first()
    else:
        report = None
    
    if not report:
        raise Http404("Report not found")
    
    # Check permissions
    if not (request.user == report.user or request.user.is_staff):
        raise Http404("Report not found")
    
    context = {
        'report': report,
    }
    
    return render(request, 'reports/view_report.html', context)


@login_required
def user_reports(request):
    """List all reports for the current user"""
    analysis_reports = AnalysisReport.objects.filter(user=request.user).order_by('-created_at')
    comparison_reports = ComparisonReport.objects.filter(user=request.user).order_by('-created_at')
    
    # Combine and sort all reports
    all_reports = list(itertools.chain(analysis_reports, comparison_reports))
    all_reports.sort(key=lambda x: x.created_at, reverse=True)
    
    context = {
        'reports': all_reports,
    }
    
    return render(request, 'reports/user_reports.html', context)


@staff_member_required
def admin_reports(request):
    """Admin view to see all reports"""
    analysis_reports = AnalysisReport.objects.all().select_related('user', 'analysis')
    comparison_reports = ComparisonReport.objects.all().select_related('user', 'comparison')
    
    # Combine and sort all reports
    all_reports = list(itertools.chain(analysis_reports, comparison_reports))
    all_reports.sort(key=lambda x: x.created_at, reverse=True)
    
    # Get statistics
    total_users = User.objects.count()
    total_analyses = ImageAnalysis.objects.count()
    total_comparisons = ImageComparison.objects.count()
    total_reports = len(all_reports)
    
    stats = {
        'total_users': total_users,
        'total_analyses': total_analyses,
        'total_comparisons': total_comparisons,
        'total_reports': total_reports,
    }
    
    # Get unique users for filter
    users = User.objects.filter(
        Q(reports__isnull=False) | Q(comparison_reports__isnull=False)
    ).distinct().order_by('username')
    
    context = {
        'reports': all_reports,
        'stats': stats,
        'users': users,
    }
    
    return render(request, 'reports/admin_reports.html', context)


@login_required
def download_report(request, report_type, report_id):
    """Download report as PDF or HTML file"""
    if report_type == 'analysis':
        report = AnalysisReport.objects.filter(id=report_id).first()
    elif report_type == 'comparison':
        report = ComparisonReport.objects.filter(id=report_id).first()
    else:
        report = None

    if not report:
        raise Http404("Report not found")

    # Check permissions
    if not (request.user == report.user or request.user.is_staff):
        raise Http404("Report not found")

    try:
        # Render the report template
        html_content = render_to_string('reports/view_report.html', {
            'report': report,
        }, request=request)

        # Try to generate PDF if xhtml2pdf is available
        if PDF_AVAILABLE:
            try:
                from io import BytesIO
                pdf_file = BytesIO()
                pisa.CreatePDF(html_content, pdf_file)
                pdf_bytes = pdf_file.getvalue()

                if pdf_bytes:
                    filename = f"IntegriPic_Report_{report.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    response = HttpResponse(pdf_bytes, content_type='application/pdf')
                    response['Content-Disposition'] = f'attachment; filename="{filename}"'
                    return response
            except Exception as pdf_error:
                logger.warning(f"PDF generation failed, falling back to HTML: {pdf_error}")
                # Fall through to HTML response

        # Fallback: Return HTML file
        response = HttpResponse(html_content, content_type='text/html; charset=utf-8')
        filename = f"IntegriPic_Report_{report.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.html"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        return response

    except Exception as e:
        logger.error(f"Error generating report download: {e}")
        raise Http404("Error generating report")


@staff_member_required
def delete_report(request, report_type, report_id):
    """Delete a report (admin only)"""
    if report_type == 'analysis':
        report = AnalysisReport.objects.filter(id=report_id).first()
    elif report_type == 'comparison':
        report = ComparisonReport.objects.filter(id=report_id).first()
    else:
        report = None
    
    if not report:
        messages.error(request, 'Report not found.')
        return redirect('reports:admin_reports')
    
    report_title = report.report_title
    report.delete()
    
    messages.success(request, f'Report "{report_title}" has been deleted.')
    return redirect('reports:admin_reports')


@staff_member_required
def system_stats(request):
    """Display system statistics"""
    # User statistics
    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    staff_users = User.objects.filter(is_staff=True).count()
    
    # Analysis statistics
    total_analyses = ImageAnalysis.objects.count()
    total_comparisons = ImageComparison.objects.count()
    
    # Report statistics
    total_analysis_reports = AnalysisReport.objects.count()
    total_comparison_reports = ComparisonReport.objects.count()
    total_reports = total_analysis_reports + total_comparison_reports
    
    # Recent activity (last 30 days)
    from datetime import timedelta
    thirty_days_ago = timezone.now() - timedelta(days=30)
    
    recent_analyses = ImageAnalysis.objects.filter(created_at__gte=thirty_days_ago).count()
    recent_comparisons = ImageComparison.objects.filter(created_at__gte=thirty_days_ago).count()
    recent_reports = (
        AnalysisReport.objects.filter(created_at__gte=thirty_days_ago).count() +
        ComparisonReport.objects.filter(created_at__gte=thirty_days_ago).count()
    )
    
    context = {
        'stats': {
            'total_users': total_users,
            'active_users': active_users,
            'staff_users': staff_users,
            'total_analyses': total_analyses,
            'total_comparisons': total_comparisons,
            'total_analysis_reports': total_analysis_reports,
            'total_comparison_reports': total_comparison_reports,
            'total_reports': total_reports,
            'recent_analyses': recent_analyses,
            'recent_comparisons': recent_comparisons,
            'recent_reports': recent_reports,
        }
    }
    
    return render(request, 'reports/system_stats.html', context)


@staff_member_required
def admin_comparison_report(request, comparison_pk):
    """Admin view for any comparison report"""
    comparison = get_object_or_404(ImageComparison, pk=comparison_pk)
    
    report, created = ComparisonReport.objects.get_or_create(
        comparison=comparison,
        defaults={
            'user': comparison.user,
            'report_title': f"Comparison Report - {comparison.image1_filename} vs {comparison.image2_filename}"
        }
    )
    
    context = {
        'comparison': comparison,
        'report': report,
        'generated_time': timezone.now(),
        'is_admin_view': True,
    }
    
    return render(request, 'reports/comparison_report.html', context)


@login_required
def export_analysis_pdf(request, analysis_pk):
    """Export analysis report as PDF"""
    if not PDF_AVAILABLE:
        messages.error(request, "PDF export is not available. Downloading as HTML instead.")
        return redirect('analysis:analysis_detail', pk=analysis_pk)

    try:
        from io import BytesIO
        analysis = get_object_or_404(ImageAnalysis, pk=analysis_pk, user=request.user)

        # Render HTML template
        html_string = render_to_string('reports/analysis_report_pdf.html', {
            'analysis': analysis,
        })

        # Convert to PDF using xhtml2pdf
        pdf_file = BytesIO()
        pisa.CreatePDF(html_string, pdf_file)
        pdf_bytes = pdf_file.getvalue()

        if pdf_bytes:
            # Create HTTP response
            response = HttpResponse(pdf_bytes, content_type='application/pdf')
            filename = f"IntegriPic_Analysis_{analysis.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'

            logger.info(f"User {request.user.username} exported PDF for analysis {analysis.id}")
            messages.success(request, f"Report exported successfully: {filename}")

            return response
        else:
            raise Exception("PDF generation returned empty")

    except Exception as e:
        logger.error(f"Error exporting PDF for analysis {analysis_pk}: {str(e)}")
        messages.error(request, f"Error exporting PDF: {str(e)}")
        return redirect('analysis:analysis_detail', pk=analysis_pk)


@login_required
def export_comparison_pdf(request, comparison_pk):
    """Export comparison report as PDF"""
    if not PDF_AVAILABLE:
        messages.error(request, "PDF export is not available. Downloading as HTML instead.")
        return redirect('analysis:comparison_detail', pk=comparison_pk)

    try:
        from io import BytesIO
        comparison = get_object_or_404(ImageComparison, pk=comparison_pk, user=request.user)

        # Create context for PDF template
        context = {
            'comparison': comparison,
        }

        # Render HTML template
        html_string = render_to_string('reports/comparison_report_pdf.html', context)

        # Convert to PDF using xhtml2pdf
        pdf_file = BytesIO()
        pisa.CreatePDF(html_string, pdf_file)
        pdf_bytes = pdf_file.getvalue()

        if pdf_bytes:
            # Create HTTP response
            response = HttpResponse(pdf_bytes, content_type='application/pdf')
            filename = f"IntegriPic_Comparison_{comparison.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'

            logger.info(f"User {request.user.username} exported PDF for comparison {comparison.id}")
            messages.success(request, f"Report exported successfully: {filename}")

            return response
        else:
            raise Exception("PDF generation returned empty")

    except Exception as e:
        logger.error(f"Error exporting PDF for comparison {comparison_pk}: {str(e)}")
        messages.error(request, f"Error exporting PDF: {str(e)}")
        return redirect('analysis:comparison_detail', pk=comparison_pk)


@login_required
def export_analysis_json(request, analysis_pk):
    """Export analysis data as JSON"""
    import json
    analysis = get_object_or_404(ImageAnalysis, pk=analysis_pk, user=request.user)

    data = {
        'id': analysis.pk,
        'filename': analysis.original_filename,
        'sha256_hash': analysis.sha256_hash,
        'status': analysis.status,
        'image_format': analysis.image_format,
        'image_width': analysis.image_width,
        'image_height': analysis.image_height,
        'file_size': analysis.file_size,
        'file_size_mb': analysis.file_size_mb,
        'created_at': analysis.created_at.isoformat(),
        'analysis_duration': analysis.analysis_duration,
        'ela_analysis_performed': analysis.ela_analysis_performed,
        'ela_results': analysis.ela_results,
        'steganography_result': analysis.steganography_result,
        'steganography_message': analysis.steganography_message,
        'copy_move_result': analysis.copy_move_result,
        'deepfake_probability': analysis.deepfake_probability,
        'deepfake_notes': analysis.deepfake_notes,
        'timeline_flags': analysis.timeline_flags,
        'metadata': analysis.metadata,
        'phash': analysis.phash,
        'dhash': analysis.dhash,
        'ahash': analysis.ahash,
    }
    response = HttpResponse(json.dumps(data, indent=2), content_type='application/json')
    filename = f"IntegriPic_Analysis_{analysis.pk}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.json"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    logger.info(f"User {request.user.username} exported JSON for analysis {analysis.pk}")
    return response


@login_required
def export_analysis_csv(request, analysis_pk):
    """Export flat analysis summary as CSV"""
    import csv
    from io import StringIO
    analysis = get_object_or_404(ImageAnalysis, pk=analysis_pk, user=request.user)

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(['Field', 'Value'])
    rows = [
        ('ID', analysis.pk),
        ('Filename', analysis.original_filename),
        ('SHA256', analysis.sha256_hash),
        ('Status', analysis.status),
        ('Format', analysis.image_format),
        ('Width', analysis.image_width),
        ('Height', analysis.image_height),
        ('File Size (bytes)', analysis.file_size),
        ('File Size (MB)', f"{analysis.file_size_mb:.2f}"),
        ('Created', analysis.created_at.isoformat()),
        ('Duration (seconds)', analysis.analysis_duration),
        ('ELA Performed', analysis.ela_analysis_performed),
        ('ELA Max Diff', analysis.ela_results.get('max_difference') if analysis.ela_results else ''),
        ('ELA Avg Diff', analysis.ela_results.get('avg_difference') if analysis.ela_results else ''),
        ('ELA Significant Pixels %', analysis.ela_results.get('significant_pixels_percentage') if analysis.ela_results else ''),
        ('Steganography Result', analysis.steganography_result or ''),
        ('Deepfake Probability', analysis.deepfake_probability or ''),
        ('pHash', analysis.phash or ''),
        ('dHash', analysis.dhash or ''),
        ('aHash', analysis.ahash or ''),
        ('Copy-Move Matches', analysis.copy_move_result.get('match_count') if analysis.copy_move_result else 0),
        ('Timeline Flags Count', len(analysis.timeline_flags) if analysis.timeline_flags else 0),
    ]
    writer.writerows(rows)
    response = HttpResponse(buf.getvalue(), content_type='text/csv')
    filename = f"IntegriPic_Analysis_{analysis.pk}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.csv"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    logger.info(f"User {request.user.username} exported CSV for analysis {analysis.pk}")
    return response
