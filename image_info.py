from PIL import Image
from PIL.ExifTags import TAGS
import hashlib
import os
import sys
import platform
import time
from image_ela import perform_ela_analysis


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

if __name__ == "__main__":
    path = input("Enter image path: ").strip()
    
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
        print("Thank you for using the image info tool.")