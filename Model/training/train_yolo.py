"""
Train YOLO segmentation model for Egyptian National ID field localization.
Saves the trained weights to models/yolo_nid_seg.pt.
"""
import os
import sys
import shutil
import argparse
from pathlib import Path

# Ensure workspace root is on sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from training.prepare_dataset import prepare_yolo_dataset, download_kaggle_dataset


def train_yolo_segmentation(
    data_yaml: str = None,
    model_name: str = "yolov8n-seg.pt",
    epochs: int = 15,
    img_size: int = 640,
    batch_size: int = 8,
    device: str = "",
    output_model_path: str = "models/yolo_nid_seg.pt"
) -> str:
    """
    Train YOLO segmentation model and export weights.
    """
    from ultralytics import YOLO

    if not data_yaml or not os.path.exists(data_yaml):
        print("[*] Preparing dataset...")
        download_kaggle_dataset()
        data_yaml = prepare_yolo_dataset()

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    print(f"[*] Initializing YOLO segmentation model: {model_name}")
    model = YOLO(model_name)

    print(f"[*] Starting training on {data_yaml} for {epochs} epochs...")
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": img_size,
        "batch": batch_size,
        "project": "runs/segment",
        "name": "egyptian_nid_seg",
        "exist_ok": True,
        "verbose": True,
        "plots": True,
        "degrees": 10.0,
        "translate": 0.05,
        "scale": 0.1,
        "shear": 2.0,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 0.0,
    }
    if device:
        train_args["device"] = device

    results = model.train(**train_args)

    # Find best.pt recursively
    best_candidates = list(Path("runs").glob("**/weights/best.pt"))
    if not best_candidates:
        best_candidates = list(Path("runs").glob("**/weights/last.pt"))

    if best_candidates:
        best_weights = sorted(best_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        shutil.copy(str(best_weights), output_model_path)
        print(f"[+] Successfully saved trained YOLO model to: {output_model_path}")
    else:
        print(f"[!] Warning: weights file not found under runs/")

    return output_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Egyptian National ID Segmentation")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Base YOLO segmentation model")
    parser.add_argument("--device", type=str, default="", help="cuda / cpu")
    parser.add_argument("--out", type=str, default="models/yolo_nid_seg.pt", help="Output model path")
    args = parser.parse_args()

    train_yolo_segmentation(
        model_name=args.model,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        output_model_path=args.out
    )
