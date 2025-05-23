from PIL import Image, ImageChops, ImageEnhance
import os

def perform_ela_analysis(image_path, quality=95):
    if not image_path.lower().endswith('.jpg') and not image_path.lower().endswith('.jpeg'):
        print("ELA works best with JPEG images. Please use a .jpg or .jpeg file.")
        return

    temp_filename = "resaved.jpg"

    try:
        # Step 1: Save the image at lower quality
        original = Image.open(image_path).convert('RGB')
        original.save(temp_filename, 'JPEG', quality=quality)

        # Step 2: Re-open and subtract
        resaved = Image.open(temp_filename)
        diff = ImageChops.difference(original, resaved)

        # Step 3: Enhance differences for visibility
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1

        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)

        # Save result
        ela_filename = os.path.splitext(image_path)[0] + '_ELA.png'
        diff.save(ela_filename)
        print(f"ELA image saved as: {ela_filename}")
    except Exception as e:
        print("Error performing ELA:", e)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
