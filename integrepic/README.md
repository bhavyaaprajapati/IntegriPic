# IntegriPic - Digital Forensics & Image Analysis Platform

<div align="center">

*A comprehensive web-based platform for image integrity analysis and digital forensics*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2.11-green.svg)](https://djangoproject.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 🔍 Overview

**IntegriPic** is a powerful Django-based web application designed for cybersecurity professionals, digital forensics investigators, and researchers. It provides comprehensive tools for analyzing image integrity, detecting tampering, and uncovering hidden content through advanced steganography detection techniques.

### 🎯 Purpose

IntegriPic helps users:
- **Verify image authenticity** and detect potential tampering
- **Analyze images for hidden steganographic content** using multiple detection methods
- **Compare images** to identify differences and modifications
- **Generate detailed forensic reports** for investigations and documentation

---

## ✨ Key Features

### 🔐 **Image Integrity Analysis**
- **SHA256 Hash Calculation** - Generate cryptographic hashes for image integrity verification
- **EXIF Metadata Extraction** - Extract and analyze embedded image metadata including camera settings, GPS data, and timestamps
- **Error Level Analysis (ELA)** - Detect potential image tampering through compression artifacts analysis
- **Multi-format Support** - Works with JPEG, PNG, BMP, TIFF, and other common image formats

### 🕵️ **Advanced Steganography Detection**
- **LSB (Least Significant Bit) Analysis** - Detect hidden messages embedded in image pixels
- **Statistical Pattern Analysis** - Identify suspicious LSB distributions that may indicate hidden content
- **Multi-method Detection** - Uses both direct extraction and statistical analysis for comprehensive detection
- **Robust Error Handling** - Gracefully handles various image types and edge cases

### 🔄 **Image Comparison & Difference Detection**
- **Visual Difference Highlighting** - Compare two images and generate difference maps
- **Pixel-level Analysis** - Detailed comparison with precise difference visualization
- **Tampering Identification** - Identify modifications between image versions
- **Side-by-side Comparison** - Clear visual presentation of differences

### 📊 **Comprehensive Reporting System**
- **Detailed Analysis Reports** - Professional forensic analysis documentation
- **HTML Report Generation** - Clean, printable reports with embedded images and technical details
- **Export Capabilities** - Generate reports suitable for legal proceedings and investigations
- **Analysis History** - Track and revisit previous analyses

### 👤 **User Management & Security**
- **Secure Authentication** - User registration, login, and profile management
- **Analysis History Tracking** - Complete audit trail of all performed analyses
- **Permission-based Access** - Secure access control for sensitive investigations
- **Statistics Dashboard** - Monitor usage patterns and analysis counts

---

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Backend Framework** | Django | 4.2.11 |
| **Image Processing** | Pillow (PIL) | 10.2.0 |
| **Steganography** | Stegano Library | 1.0.1 |
| **Database** | SQLite | (Built-in) |
| **Frontend** | HTML5, CSS3, JavaScript | - |
| **UI Framework** | Bootstrap 5 | via crispy-bootstrap5 |
| **Forms** | Django Crispy Forms | 2.0 |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd IntegriPic/IntegriPic
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Database Setup
```bash
python manage.py migrate
```

### 4. Create Superuser (Optional)
```bash
python manage.py createsuperuser
```

### 5. Run the Development Server
```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000`

---

## 📖 Usage Guide

### Getting Started

1. **Register/Login** - Create an account or log in to access the analysis features
2. **Upload Image** - Use the upload interface to submit images for analysis
3. **Review Results** - Examine the comprehensive analysis results including metadata, hash values, and steganography detection
4. **Compare Images** - Upload two images to detect differences and modifications
5. **Generate Reports** - Create professional forensic reports for your findings

### Steganography Detection

IntegriPic uses multiple methods to detect hidden content:

- **Direct LSB Detection**: Extracts hidden messages using the Least Significant Bit method
- **Statistical Analysis**: Analyzes LSB distribution patterns to identify potential steganography
- **Format-specific Handling**: Optimized detection for PNG, BMP, and TIFF formats

### Image Analysis Features

- **Metadata Analysis**: Extracts EXIF data including camera make/model, settings, and GPS coordinates
- **Hash Verification**: Calculates SHA256 hashes for integrity verification
- **ELA Analysis**: Performs Error Level Analysis on JPEG images to detect tampering

---

## 🧪 Testing & Quality Assurance

The project includes comprehensive testing for steganography functionality:

```bash
# Run steganography tests
python test_steganography.py

# Run enhanced steganography tests  
python test_enhanced_stego.py

# Run Django tests
python manage.py test
```

### Recent Fixes & Improvements

#### ✅ **Steganography Detection Issues Resolved**

**Problem**: The original steganography detection had several critical issues:
- "image index out of range" errors when analyzing clean images
- Poor error handling for edge cases
- Inaccurate statistical analysis thresholds
- Crashes on uniform color images

**Solution**: Implemented comprehensive fixes:
- **Safe LSB Reveal Method**: Added robust error handling for the stegano library's limitations
- **Improved Statistical Analysis**: Enhanced thresholds and pattern recognition
- **Better Format Support**: Improved handling of different image modes and formats
- **Graceful Fallbacks**: Statistical analysis fallback when direct LSB detection fails

**Results**:
- ✅ No more crashes or "image index out of range" errors
- ✅ Successfully detects hidden messages when present
- ✅ Gracefully handles images without steganographic content
- ✅ More accurate statistical analysis results
- ✅ Comprehensive error handling for all edge cases

---

## 🚨 Use Cases

### Digital Forensics
- **Evidence Analysis**: Analyze digital images submitted as evidence in legal cases
- **Tampering Detection**: Identify manipulated or altered images
- **Chain of Custody**: Maintain integrity verification through hash calculation

### Cybersecurity Research
- **Malware Analysis**: Detect steganographic malware hiding in images
- **Threat Intelligence**: Analyze suspicious images for hidden communication
- **Security Auditing**: Verify image integrity in security-sensitive applications

### Academic & Educational
- **Digital Forensics Training**: Hands-on learning platform for students
- **Research Platform**: Tool for steganography and image analysis research
- **Case Studies**: Generate educational materials and examples

---

## ⚠️ Important Notes

### Security Considerations
- **Authorized Use Only**: Ensure proper authorization before analyzing images
- **Data Privacy**: Be mindful of privacy laws when processing personal images
- **Evidence Handling**: Follow proper digital forensics procedures for legal evidence

### Limitations
- **Steganography Detection**: No single method can detect all steganography techniques
- **File Size Limits**: Large images may require additional processing time
- **Format Support**: Some advanced features work best with specific image formats

### Best Practices
- **Verify Results**: Cross-reference findings with multiple analysis methods
- **Document Process**: Maintain detailed logs of analysis procedures
- **Regular Updates**: Keep dependencies updated for security and performance

---

## 🔄 Recent Updates

### Version 1.1.0 (Current)
- ✅ **Fixed**: Steganography detection "image index out of range" errors
- ✅ **Improved**: Statistical analysis with better thresholds and pattern recognition
- ✅ **Enhanced**: Error handling for edge cases and uniform images
- ✅ **Added**: Comprehensive testing suite with multiple test scenarios
- ✅ **Updated**: Documentation with detailed technical information

### Key Technical Improvements:
- **Safe LSB Reveal**: Implements robust error handling for stegano library limitations
- **Enhanced Statistical Analysis**: More accurate LSB distribution analysis with refined thresholds
- **Better Format Handling**: Improved support for various image modes (RGB, RGBA, L, P)
- **Pattern Recognition**: Advanced pattern analysis for detecting steganographic signatures

---

<div align="center">

**IntegriPic** - *Empowering Digital Forensics Through Advanced Image Analysis*

Made with ❤️ for the cybersecurity and digital forensics community

</div>
