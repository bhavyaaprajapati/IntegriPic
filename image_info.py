from PIL import Image
from PIL.ExifTags import TAGS
import hashlib
import os
import sys
import platform
import time
from image_ela import perform_ela_analysis
from image_compare import compare_images
from stegano import lsb
import webbrowser

def get_first_jpg_image():
    for file in os.listdir():
        if file.lower().endswith('.jpg'):
            return file
    return None
def generate_report_from_template(image_name, hash_value, metadata, ela_img,
                                  diff_img=None, stego_result=None, os_info=None):
    with open("report.html", "r", encoding="utf-8") as f:
        template = f.read()

    compare_section = ""
    if diff_img:
        compare_section = f"""
        <div class="section">
            <h2>Image Comparison</h2>
            <img src="{diff_img}" alt="Difference Image">
        </div>"""

    stego_section = ""
    if stego_result:
        stego_section = f"""
        <div class="section">
            <h2>Steganography Detection</h2>
            <p>{stego_result}</p>
        </div>"""

    system_info_section = ""
    if os_info:
        system_info_section = f"""
        <div class="section">
            <h2>System Information</h2>
            <p>{os_info}</p>
        </div>"""

    # Replace placeholders
    html_report = template.replace("{{image_name}}", image_name)
    html_report = html_report.replace("{{hash_value}}", hash_value)
    html_report = html_report.replace("{{metadata}}", metadata)
    html_report = html_report.replace("{{ela_img}}", ela_img)
    html_report = html_report.replace("{{compare_section}}", compare_section)
    html_report = html_report.replace("{{stego_section}}", stego_section)
    html_report = html_report.replace("{{system_info_section}}", system_info_section)

    # Save it
    report_file = f"report_{image_name.split('.')[0]}.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_report)

    print(f"\n✅ HTML report created: {report_file}")
    webbrowser.open(report_file)

def extract_metadata(image_path):
    try:
        image = Image.open(image_path)
        info = image._getexif()
        metadata = {}
        if info:
            for tag, value in info.items():
                tag_name = TAGS.get(tag, tag)
                metadata[tag_name] = value
        return metadata
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return {}

def calculate_sha256(image_path):
    try:
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        print(f"Error calculating hash: {e}")
        return None
def detect_steganography(image_path):
    try:
        hidden_message = lsb.reveal(image_path)
        if hidden_message:
            print("\n--- Hidden Data Detected (LSB) ---")
            print(f"Message: {hidden_message}")
        else:
            print("\n--- No hidden message found using LSB method ---")
    except Exception as e:
        print(f"\nError checking steganography: {e}")

if __name__ == "__main__":
    path = get_first_jpg_image()
if not path:
    print("No .jpg image found in the current directory.")
    sys.exit()
else:
    print(f"Using image: {path}")

    if not os.path.isfile(path):
        print("File does not exist. Please check the path.")
    else:
        meta = extract_metadata(path)
        print("\n--- Metadata ---")
        if meta:
            for k, v in meta.items():
                print(f"{k}: {v}")
        else:
            print("No metadata found or image not readable.")
        
        hash_val = calculate_sha256(path)
        if hash_val:
            print(f"\nSHA256 Hash: {hash_val}")
        else:
            print("Could not calculate hash.")
        print("\n--- Image Information ---")
        try:
            with Image.open(path) as img:
                print(f"Format: {img.format}")
                print(f"Size: {img.size}")
                print(f"Mode: {img.mode}")
                print(f"Info: {img.info}")
                print("\n--- Running Error Level Analysis (ELA) ---")
                perform_ela_analysis(path)
        except Exception as e:
            print(f"Error opening image: {e}")
        print("\n--- End of Information ---")
        if input("Do you want to compare with another image? (y/n): ").lower() == 'y':
            second_path = input("Enter second image path: ").strip()
            if os.path.isfile(second_path):
                print("\n--- Comparing Images ---")
                compare_images(path, second_path)
            else:
                print("Second image file not found.")
                print("\n--- Checking for Steganography ---")
        detect_steganography(path)
        print("\n--- System Information ---")
        print(f"OS: {platform.system()} {platform.release()}")
        print("Thank you for using the image info tool.")