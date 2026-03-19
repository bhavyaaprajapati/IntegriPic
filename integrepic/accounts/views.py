from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import AuthenticationForm, PasswordResetForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.views import (
    LoginView, LogoutView, PasswordResetView, PasswordResetConfirmView,
    PasswordResetCompleteView, PasswordResetDoneView, PasswordChangeView,
    PasswordChangeDoneView
)
from django.views.generic import CreateView
from .forms import UserRegistrationForm


class CustomLoginView(LoginView):
    """Custom login view"""
    template_name = 'accounts/login.html'
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return reverse_lazy('analysis:dashboard')


class CustomLogoutView(LogoutView):
    """Custom logout view"""
    next_page = reverse_lazy('accounts:login')


class RegisterView(CreateView):
    """User registration view"""
    form_class = UserRegistrationForm
    template_name = 'accounts/register.html'
    success_url = reverse_lazy('accounts:login')
    
    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, 'Account created successfully! You can now log in.')
        return response
    
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('analysis:dashboard')
        return super().dispatch(request, *args, **kwargs)


class CustomPasswordResetView(PasswordResetView):
    """Custom password reset view"""
    template_name = 'accounts/password_reset.html'
    email_template_name = 'accounts/password_reset_email.html'
    subject_template_name = 'accounts/password_reset_subject.txt'
    success_url = reverse_lazy('accounts:password_reset_done')


class CustomPasswordResetDoneView(PasswordResetDoneView):
    """Custom password reset done view"""
    template_name = 'accounts/password_reset_done.html'


class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    """Custom password reset confirm view"""
    template_name = 'accounts/password_reset_confirm.html'
    success_url = reverse_lazy('accounts:password_reset_complete')


class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    """Custom password reset complete view"""
    template_name = 'accounts/password_reset_complete.html'


class CustomPasswordChangeView(PasswordChangeView):
    """Custom password change view"""
    template_name = 'accounts/password_change.html'
    success_url = reverse_lazy('accounts:password_change_done')


class CustomPasswordChangeDoneView(PasswordChangeDoneView):
    """Custom password change done view"""
    template_name = 'accounts/password_change_done.html'


@login_required
def profile(request):
    """User profile view"""
    from analysis.models import ImageAnalysis, ImageComparison
    from .models import UserProfile
    
    # Get or create user profile
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    # Get user statistics
    total_analyses = ImageAnalysis.objects.filter(user=request.user).count()
    total_comparisons = ImageComparison.objects.filter(user=request.user).count()
    
    # Update profile statistics
    profile.total_analyses_count = total_analyses
    profile.total_comparisons_count = total_comparisons
    profile.save()
    
    # Get recent activity
    recent_analyses = ImageAnalysis.objects.filter(user=request.user).order_by('-created_at')[:5]
    recent_comparisons = ImageComparison.objects.filter(user=request.user).order_by('-created_at')[:3]
    
    # Calculate total uploads (analyses + comparisons * 2 since comparisons involve 2 images)
    total_uploads = total_analyses + (total_comparisons * 2)
    
    context = {
        'profile': profile,
        'total_analyses': total_analyses,
        'total_comparisons': total_comparisons,
        'total_uploads': total_uploads,
        'recent_analyses': recent_analyses,
        'recent_comparisons': recent_comparisons,
    }
    return render(request, 'accounts/profile.html', context)
