import cv2
import os
import glob
import numpy as np

input_folder = "dataset/dataset_old"
output_folder = "dataset"
target_size = (256, 256)

os.makedirs(output_folder, exist_ok=True)

image_paths_train = glob.glob(os.path.join(input_folder, "train", "**", "*.*"), recursive=True)

for img_path in image_paths_train:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    relative_path = os.path.relpath(img_path, os.path.join(input_folder, "train"))
    output_path = os.path.join(output_folder, "train", relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)

image_paths_test = glob.glob(os.path.join(input_folder, "test", "*.*"))

for img_path in image_paths_test:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    relative_path = os.path.relpath(img_path, os.path.join(input_folder, "test"))
    output_path = os.path.join(output_folder, "test", relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)

print("Готово! Все изображения приведены к", target_size)
