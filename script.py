import os
from pathlib import Path

from PIL import Image

# Input and output directories
input_dir = Path("dataset_new")
output_dir = Path("dataset_new_resized")

# Make sure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Desired image size
target_size = (64, 64)


def resize_and_pad(img: Image.Image, size=(128, 128)) -> Image.Image:
    """Resize image with aspect ratio preserved and pad with black."""
    img.thumbnail(size, Image.LANCZOS)  # Resize maintaining aspect ratio
    new_img = Image.new("RGB", size, (0, 0, 0))  # Black canvas
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    new_img.paste(img, offset)
    return new_img


for class_dir in input_dir.iterdir():
    if not class_dir.is_dir():
        continue
    output_class_dir = output_dir / class_dir.name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in class_dir.glob("*.jpg"):
        try:
            img = Image.open(img_path).convert("RGB")
            resized = resize_and_pad(img, target_size)
            resized.save(output_class_dir / img_path.name)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
