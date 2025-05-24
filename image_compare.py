from PIL import Image, ImageChops
import os

def compare_images(img1_path, img2_path):
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if img1.size != img2.size:
            print("Images have different sizes. Resizing second image...")
            img2 = img2.resize(img1.size)

        diff = ImageChops.difference(img1, img2)
        diff_filename = f"diff_{os.path.basename(img1_path).split('.')[0]}_{os.path.basename(img2_path).split('.')[0]}.png"
        diff.save(diff_filename)

        print(f"Difference image saved as: {diff_filename}")

        # Optional: Check if images are the same
        bbox = diff.getbbox()
        if bbox is None:
            print("Images are identical.")
        else:
            print("Images have differences.")

    except Exception as e:
        print("Error comparing images:", e)
