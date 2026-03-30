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

# Try to import WeasyPrint for PDF generation
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    WEASYPRINT_AVAILABLE = False
    logger.warning(f"WeasyPrint not available: {e}")


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
    
    return redirect('reports:view_report', report_id=report.id)


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
    
    return redirect('reports:view_report', report_id=report.id)


@login_required
def view_report(request, report_id):
    """View a specific report"""
    # Try to find the report in both tables
    analysis_report = AnalysisReport.objects.filter(id=report_id).first()
    comparison_report = ComparisonReport.objects.filter(id=report_id).first()
    
    report = analysis_report or comparison_report
    
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
def download_report(request, report_id):
    """Download report as PDF or HTML file"""
    # Try to find the report in both tables
    analysis_report = AnalysisReport.objects.filter(id=report_id).first()
    comparison_report = ComparisonReport.objects.filter(id=report_id).first()

    report = analysis_report or comparison_report

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

        # Try to generate PDF if WeasyPrint is available
        if WEASYPRINT_AVAILABLE:
            try:
                pdf_bytes = HTML(string=html_content).write_pdf()
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
def delete_report(request, report_id):
    """Delete a report (admin only)"""
    # Try to find the report in both tables
    analysis_report = AnalysisReport.objects.filter(id=report_id).first()
    comparison_report = ComparisonReport.objects.filter(id=report_id).first()
    
    report = analysis_report or comparison_report
    
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
    if not WEASYPRINT_AVAILABLE:
        messages.error(request, "PDF export is not available. Please install WeasyPrint dependencies.")
        return redirect('analysis:analysis_detail', pk=analysis_pk)

    try:
        analysis = get_object_or_404(ImageAnalysis, pk=analysis_pk, user=request.user)

        # Render HTML template
        html_string = render_to_string('reports/analysis_report_pdf.html', {
            'analysis': analysis,
        })

        # Convert to PDF using WeasyPrint
        html = HTML(string=html_string)
        pdf = html.write_pdf()

        # Create HTTP response
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = f"IntegriPic_Analysis_{analysis.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        logger.info(f"User {request.user.username} exported PDF for analysis {analysis.id}")
        messages.success(request, f"Report exported successfully: {filename}")

        return response

    except Exception as e:
        logger.error(f"Error exporting PDF for analysis {analysis_pk}: {str(e)}")
        messages.error(request, f"Error exporting PDF: {str(e)}")
        return redirect('analysis:analysis_detail', pk=analysis_pk)


@login_required
def export_comparison_pdf(request, comparison_pk):
    """Export comparison report as PDF"""
    if not WEASYPRINT_AVAILABLE:
        messages.error(request, "PDF export is not available. Please install WeasyPrint dependencies.")
        return redirect('analysis:comparison_detail', pk=comparison_pk)

    try:
        comparison = get_object_or_404(ImageComparison, pk=comparison_pk, user=request.user)

        # Create context for PDF template
        context = {
            'comparison': comparison,
        }

        # Render HTML template (we'll create this next)
        html_string = render_to_string('reports/comparison_report_pdf.html', context)

        # Convert to PDF using WeasyPrint
        html = HTML(string=html_string)
        pdf = html.write_pdf()

        # Create HTTP response
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = f"IntegriPic_Comparison_{comparison.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        logger.info(f"User {request.user.username} exported PDF for comparison {comparison.id}")
        messages.success(request, f"Report exported successfully: {filename}")

        return response

    except Exception as e:
        logger.error(f"Error exporting PDF for comparison {comparison_pk}: {str(e)}")
        messages.error(request, f"Error exporting PDF: {str(e)}")
        return redirect('analysis:comparison_detail', pk=comparison_pk)
