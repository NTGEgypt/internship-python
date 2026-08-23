"""
Dataset Preparation & Ingestion for YOLO Segmentation.
Supports the Kaggle dataset: nagwaahmed/egyptian-national-ids
Converts annotations into YOLO segmentation dataset format (data.yaml + polygon masks).
"""
import os
import sys
import glob
import shutil
import json
import random
import cv2
import numpy as np
from pathlib import Path

# Canonical YOLO Segmentation Classes for Egyptian National IDs
CLASSES = [
    "national_id",
    "name",
    "date_of_birth",
    "gender",
    "birth_governorate",
    "profession",
    "address",
    "barcode",
    "photo",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def download_kaggle_dataset(target_dir: str = "data/dataset") -> bool:
    """
    Attempt to download dataset from Kaggle if kaggle CLI is configured.
    """
    os.makedirs(target_dir, exist_ok=True)
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "nagwaahmed/egyptian-national-ids", "-p", target_dir, "--unzip"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print(f"[+] Successfully downloaded Kaggle dataset to {target_dir}")
            return True
        else:
            print(f"[!] Kaggle CLI note: {result.stderr.strip()}")
    except Exception as e:
        print(f"[!] Kaggle download skipped or unavailable: {e}")
    return False


def bbox_to_polygon_norm(x0: float, y0: float, x1: float, y1: float) -> list:
    """
    Convert normalized bounding box [x0, y0, x1, y1] (0.0 to 1.0)
    into 4-point polygon coordinates [x0,y0, x1,y0, x1,y1, x0,y1].
    """
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def bbox_xywh_to_polygon_norm(cx: float, cy: float, w: float, h: float) -> list:
    """
    Convert normalized center-based box [cx, cy, w, h] to 4-point polygon coordinates.
    """
    x0 = max(0.0, cx - w / 2.0)
    y0 = max(0.0, cy - h / 2.0)
    x1 = min(1.0, cx + w / 2.0)
    y1 = min(1.0, cy + h / 2.0)
    return bbox_to_polygon_norm(x0, y0, x1, y1)


def generate_synthetic_sample(img_w: int = 1600, img_h: int = 1009) -> tuple:
    """
    Generate a diverse synthetic Egyptian ID image and ground-truth polygon segmentation labels.
    Introduces realistic geometric jitter, varying line counts, and diverse layout positions
    so the model learns feature visual characteristics across different IDs rather than fixed coordinates.
    """
    # Background card with slight color/illumination variation
    bg_color = (
        random.randint(225, 245),
        random.randint(228, 248),
        random.randint(232, 252),
    )
    img = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)

    # Subtle texture noise
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    labels = []

    # 1. State Header (Olive green decoration, y: 0.03 - 0.22 with slight jitter)
    hy0 = random.uniform(0.03, 0.05)
    hy1 = random.uniform(0.18, 0.23)
    hx0 = random.uniform(0.36, 0.42)
    hx1 = random.uniform(0.90, 0.95)
    cv2.rectangle(img, (int(hx0 * img_w), int(hy0 * img_h)), (int(hx1 * img_w), int(hy1 * img_h)), (70 + random.randint(-10, 10), 130 + random.randint(-15, 15), 70 + random.randint(-10, 10)), -1)

    # 2. Photo (Left panel, varied size & position)
    px0 = random.uniform(0.035, 0.050)
    py0 = random.uniform(0.055, 0.080)
    px1 = px0 + random.uniform(0.28, 0.32)
    py1 = py0 + random.uniform(0.52, 0.58)
    cv2.rectangle(img, (int(px0 * img_w), int(py0 * img_h)), (int(px1 * img_w), int(py1 * img_h)), (110 + random.randint(-15, 15), 110 + random.randint(-15, 15), 110 + random.randint(-15, 15)), -1)
    poly = bbox_to_polygon_norm(px0, py0, px1, py1)
    labels.append(f"{CLASS_TO_IDX['photo']} " + " ".join(f"{p:.5f}" for p in poly))

    # 3. Name (Right side, 1-2 lines, y: 0.22 - 0.48)
    num_name_lines = random.choice([1, 2, 2])
    nx0 = random.uniform(0.34, 0.40)
    ny0 = random.uniform(0.22, 0.28)
    nx1 = random.uniform(0.88, 0.96)
    ny1 = ny0 + (0.08 if num_name_lines == 1 else random.uniform(0.14, 0.19))
    ny1 = min(0.48, ny1)
    
    # Draw simulated text lines
    for line_i in range(num_name_lines):
        line_y = ny0 + (line_i + 0.5) * (ny1 - ny0) / num_name_lines
        cv2.line(img, (int(nx0 * img_w), int(line_y * img_h)), (int((nx1 - random.uniform(0.02, 0.10)) * img_w), int(line_y * img_h)), (30, 30, 30), max(2, int(img_h * 0.008)))
    
    poly = bbox_to_polygon_norm(nx0, ny0, nx1, ny1)
    labels.append(f"{CLASS_TO_IDX['name']} " + " ".join(f"{p:.5f}" for p in poly))

    # 4. Address (Right side, y: 0.45 - 0.65 covering 2 lines cleanly)
    num_addr_lines = random.choice([2, 2, 3])
    ax0 = random.uniform(0.33, 0.38)
    ay0 = max(ny1 + random.uniform(0.010, 0.020), random.uniform(0.45, 0.49))
    ax1 = random.uniform(0.88, 0.96)
    ay1 = ay0 + (random.uniform(0.14, 0.18) if num_addr_lines == 2 else random.uniform(0.18, 0.20))
    ay1 = min(0.65, ay1)

    for line_i in range(num_addr_lines):
        line_y = ay0 + (line_i + 0.5) * (ay1 - ay0) / num_addr_lines
        cv2.line(img, (int(ax0 * img_w), int(line_y * img_h)), (int((ax1 - random.uniform(0.01, 0.08)) * img_w), int(line_y * img_h)), (40, 40, 40), max(2, int(img_h * 0.007)))

    poly = bbox_to_polygon_norm(ax0, ay0, ax1, ay1)
    labels.append(f"{CLASS_TO_IDX['address']} " + " ".join(f"{p:.5f}" for p in poly))

    # 5. National ID (Right side, 14-digit strip moved slightly up and extending till end of card, y: 0.67 - 0.985)
    ix0 = random.uniform(0.33, 0.38)
    iy0 = random.uniform(0.67, 0.70)
    ix1 = random.uniform(0.92, 0.97)
    iy1 = random.uniform(0.95, 0.985)

    cv2.line(img, (int(ix0 * img_w), int(((iy0 + iy1) / 2) * img_h)), (int(ix1 * img_w), int(((iy0 + iy1) / 2) * img_h)), (20, 20, 20), max(3, int(img_h * 0.012)))
    poly = bbox_to_polygon_norm(ix0, iy0, ix1, iy1)
    labels.append(f"{CLASS_TO_IDX['national_id']} " + " ".join(f"{p:.5f}" for p in poly))

    # 6. Date of Birth (Left hologram strip, y: 0.72 - 0.86)
    dx0 = px0
    dy0 = random.uniform(0.72, 0.75)
    dx1 = px1
    dy1 = min(0.86, dy0 + random.uniform(0.09, 0.12))
    cv2.line(img, (int(dx0 * img_w), int(((dy0 + dy1) / 2) * img_h)), (int((dx0 + 0.20) * img_w), int(((dy0 + dy1) / 2) * img_h)), (35, 35, 35), max(2, int(img_h * 0.007)))
    poly = bbox_to_polygon_norm(dx0, dy0, dx1, dy1)
    labels.append(f"{CLASS_TO_IDX['date_of_birth']} " + " ".join(f"{p:.5f}" for p in poly))

    # 7. Serial Number (Left bottom, y: 0.87 - 0.99)
    sx0 = px0
    sy0 = random.uniform(0.87, 0.90)
    sx1 = px1
    sy1 = min(0.99, sy0 + random.uniform(0.08, 0.10))
    cv2.line(img, (int(sx0 * img_w), int(((sy0 + sy1) / 2) * img_h)), (int((sx0 + 0.22) * img_w), int(((sy0 + sy1) / 2) * img_h)), (35, 35, 35), max(2, int(img_h * 0.007)))
    poly = bbox_to_polygon_norm(sx0, sy0, sx1, sy1)
    class_idx = CLASS_TO_IDX.get('serial_number', CLASS_TO_IDX['barcode'])
    labels.append(f"{class_idx} " + " ".join(f"{p:.5f}" for p in poly))

    return img, labels


def prepare_yolo_dataset(
    source_dir: str = "data/dataset",
    output_dir: str = "data/yolo_seg_dataset",
    val_split: float = 0.2,
    synthetic_fallback_count: int = 60
) -> str:
    """
    Prepare standard YOLO segmentation dataset directory structure and data.yaml.
    """
    images_train_dir = os.path.join(output_dir, "images", "train")
    images_val_dir = os.path.join(output_dir, "images", "val")
    labels_train_dir = os.path.join(output_dir, "labels", "train")
    labels_val_dir = os.path.join(output_dir, "labels", "val")

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(d, exist_ok=True)

    # Check for existing image and label files in source_dir
    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    found_images = []
    if os.path.exists(source_dir):
        for ext in image_exts:
            found_images.extend(glob.glob(os.path.join(source_dir, "**", ext), recursive=True))

    print(f"[*] Found {len(found_images)} source images in '{source_dir}'")

    samples = []
    if found_images:
        for img_path in found_images:
            base_name = Path(img_path).stem
            # Check for matching label file
            txt_candidates = glob.glob(os.path.join(source_dir, "**", f"{base_name}.txt"), recursive=True)
            if txt_candidates:
                with open(txt_candidates[0], "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                # If YOLO box format (5 numbers: class cx cy w h), convert to polygon format
                converted_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cls_id = parts[0]
                        cx, cy, w, h = map(float, parts[1:5])
                        poly = bbox_xywh_to_polygon_norm(cx, cy, w, h)
                        converted_lines.append(f"{cls_id} " + " ".join(f"{p:.5f}" for p in poly))
                    elif len(parts) >= 7:  # already polygon format
                        converted_lines.append(line)
                if converted_lines:
                    img = cv2.imread(img_path)
                    if img is not None:
                        samples.append((img, converted_lines))

    # If no valid annotated source images found, generate clean synthetic ground truth
    if len(samples) < 10:
        print(f"[*] Generating {synthetic_fallback_count} synthetic ground-truth segmentation samples...")
        for _ in range(synthetic_fallback_count):
            img, labels = generate_synthetic_sample()
            samples.append((img, labels))

    # Split into train and val
    random.seed(42)
    random.shuffle(samples)
    split_idx = max(1, int(len(samples) * (1.0 - val_split)))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    for idx, (img, labels) in enumerate(train_samples):
        img_filename = f"nid_sample_train_{idx:04d}.jpg"
        txt_filename = f"nid_sample_train_{idx:04d}.txt"
        cv2.imwrite(os.path.join(images_train_dir, img_filename), img)
        with open(os.path.join(labels_train_dir, txt_filename), "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

    for idx, (img, labels) in enumerate(val_samples):
        img_filename = f"nid_sample_val_{idx:04d}.jpg"
        txt_filename = f"nid_sample_val_{idx:04d}.txt"
        cv2.imwrite(os.path.join(images_val_dir, img_filename), img)
        with open(os.path.join(labels_val_dir, txt_filename), "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

    # Write data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_output = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""path: {abs_output}
train: images/train
val: images/val

names:
"""
    for idx, name in enumerate(CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[+] YOLO segmentation dataset ready at: {yaml_path}")
    print(f"[+] Train: {len(train_samples)} images, Val: {len(val_samples)} images")
    return yaml_path


if __name__ == "__main__":
    download_kaggle_dataset()
    prepare_yolo_dataset()
