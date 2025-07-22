import os

from PIL import Image

# === CONFIGURATION ===
src_folder = "dataset/test/oxeye_daisy"  # Folder with original 256x256 images
dst_folder = "dataset_small/test/oxeyer_daisy"  # Folder for resized 32x32 images
target_size = (32, 32)

# === CREATE DESTINATION FOLDER IF MISSING ===
os.makedirs(dst_folder, exist_ok=True)

# === RESIZE IMAGES ===
for filename in os.listdir(src_folder):
    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(dst_folder, filename)

    try:
        with Image.open(src_path) as img:
            img = img.resize(target_size)
            img.save(dst_path)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
