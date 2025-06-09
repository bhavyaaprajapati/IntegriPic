from django import forms
from django.core.validators import FileExtensionValidator


class ImageUploadForm(forms.Form):
    """Form for uploading images for analysis"""
    image = forms.ImageField(
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp'])],
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp',
            'id': 'imageUpload'
        }),
        help_text="Upload JPG, JPEG, PNG, BMP, TIFF, GIF, or WebP files (max 10MB)"
    )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Check file size (10MB limit)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError('File size must be less than 10MB.')
            
            # Check file type
            if not image.content_type.startswith('image/'):
                raise forms.ValidationError('Invalid file type. Please upload an image.')
        
        return image


class ImageComparisonForm(forms.Form):
    """Form for comparing two uploaded images"""
    image1 = forms.ImageField(
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp'])],
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp',
        }),
        label="First Image",
        help_text="Upload first image (JPG, JPEG, PNG, BMP, TIFF, GIF, or WebP, max 10MB)"
    )
    
    image2 = forms.ImageField(
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp'])],
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp',
        }),
        label="Second Image",
        help_text="Upload second image (JPG, JPEG, PNG, BMP, TIFF, GIF, or WebP, max 10MB)"
    )
    
    def clean_image1(self):
        image = self.cleaned_data.get('image1')
        if image and image.size > 10 * 1024 * 1024:
            raise forms.ValidationError('First image file size must be less than 10MB.')
        return image
    
    def clean_image2(self):
        image = self.cleaned_data.get('image2')
        if image and image.size > 10 * 1024 * 1024:
            raise forms.ValidationError('Second image file size must be less than 10MB.')
        return image
