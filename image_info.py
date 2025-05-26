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

    # Creative CSS styling
    style = '''
    <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fb; color: #222; margin: 0; padding: 0; }
    .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 16px; box-shadow: 0 4px 24px #0001; padding: 32px; }
    h1 { color: #2a5298; text-align: center; margin-bottom: 0.5em; }
    .section { margin-bottom: 2em; padding: 1.5em; border-radius: 12px; background: #f0f4fa; box-shadow: 0 2px 8px #0001; }
    .section h2 { color: #1e3c72; margin-top: 0; }
    .meta-table { width: 100%; border-collapse: collapse; }
    .meta-table th, .meta-table td { padding: 8px 12px; border-bottom: 1px solid #e0e6ed; }
    .meta-table th { background: #e3ecfa; text-align: left; }
    .image-row { display: flex; gap: 32px; justify-content: center; align-items: flex-start; margin-top: 1em; }
    .image-col { flex: 1; text-align: center; }
    img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px #0002; }
    .hash { font-family: 'Consolas', monospace; background: #e3ecfa; padding: 6px 12px; border-radius: 6px; display: inline-block; }
    .footer { text-align: center; color: #888; margin-top: 2em; font-size: 0.95em; }
    </style>
    '''

    # Prepare metadata as a table if possible
    meta_html = "<table class='meta-table'>"
    for line in metadata.split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            meta_html += f"<tr><th>{k.strip()}</th><td>{v.strip()}</td></tr>"
    meta_html += "</table>"

    # ELA and comparison images
    ela_section = ""
    if ela_img:
        ela_section = f"""
        <div class='image-col'>
            <h3>ELA Image</h3>
            <img src='{ela_img}' alt='ELA Image'>
        </div>"""
    compare_section = ""
    if diff_img:
        compare_section = f"""
        <div class='image-col'>
            <h3>Comparison Image</h3>
            <img src='{diff_img}' alt='Difference Image'>
        </div>"""

    # Combine ELA and comparison images in a row
    images_row = ""
    if ela_section or compare_section:
        images_row = f"<div class='image-row'>{ela_section}{compare_section}</div>"

    stego_section = ""
    if stego_result:
        stego_section = f"""
        <div class='section'>
            <h2>Steganography Detection</h2>
            <p>{stego_result}</p>
        </div>"""

    system_info_section = ""
    if os_info:
        system_info_section = f"""
        <div class='section'>
            <h2>System Information</h2>
            <p>{os_info}</p>
        </div>"""

    # Build the creative HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Image Analysis Report - {image_name}</title>
        {style}
    </head>
    <body>
        <div class='container'>
            <h1>Image Analysis Report</h1>
            <div class='section'>
                <h2>Image Name</h2>
                <p><b>{image_name}</b></p>
                <h2>SHA256 Hash</h2>
                <p class='hash'>{hash_value}</p>
            </div>
            <div class='section'>
                <h2>Metadata</h2>
                {meta_html}
            </div>
            {images_row}
            {stego_section}
            {system_info_section}
            <div class='footer'>
                Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} | IntegriPic Tool
            </div>
        </div>
    </body>
    </html>
    """

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
            # Format metadata as a string for HTML
            meta_str = '\n'.join([f"{k}: {v}" for k, v in meta.items()])
        else:
            print("No metadata found or image not readable.")
            meta_str = "No metadata found or image not readable."
        
        hash_val = calculate_sha256(path)
        if hash_val:
            print(f"\nSHA256 Hash: {hash_val}")
        else:
            print("Could not calculate hash.")
        print("\n--- Image Information ---")
        ela_img = None
        try:
            with Image.open(path) as img:
                print(f"Format: {img.format}")
                print(f"Size: {img.size}")
                print(f"Mode: {img.mode}")
                print(f"Info: {img.info}")
                print("\n--- Running Error Level Analysis (ELA) ---")
                perform_ela_analysis(path)
                ela_img = os.path.splitext(path)[0] + '_ELA.png'
        except Exception as e:
            print(f"Error opening image: {e}")
        print("\n--- End of Information ---")
        diff_img = None
        if input("Do you want to compare with another image? (y/n): ").lower() == 'y':
            second_path = input("Enter second image path: ").strip()
            if os.path.isfile(second_path):
                print("\n--- Comparing Images ---")
                compare_images(path, second_path)
                diff_img = f"diff_{os.path.basename(path).split('.')[0]}_{os.path.basename(second_path).split('.')[0]}.png"
            else:
                print("Second image file not found.")
        print("\n--- Checking for Steganography ---")
        stego_result = None
        try:
            hidden_message = lsb.reveal(path)
            if hidden_message:
                stego_result = f"Hidden message: {hidden_message}"
            else:
                stego_result = "No hidden message found using LSB method."
        except Exception as e:
            stego_result = f"Error checking steganography: {e}"
        print("\n--- System Information ---")
        os_info = f"OS: {platform.system()} {platform.release()}"
        print(os_info)
        print("Thank you for using the image info tool.")
        # Generate dynamic HTML report
        generate_report_from_template(
            image_name=os.path.basename(path),
            hash_value=hash_val or "N/A",
            metadata=meta_str,
            ela_img=ela_img or "",
            diff_img=diff_img,
            stego_result=stego_result,
            os_info=os_info
        )