"""
Egyptian National ID — CLI Entrypoint for Single & Batch Image Processing.

Usage:
    python cli.py path/to/card.jpg
    python cli.py path/to/folder_with_cards/ --output results.json
    python cli.py card.jpg --localization-mode yolo_segmentation
"""
import argparse
import json
import os
import sys
from pathlib import Path
import cv2

from core import config
from core.pipeline import run_pipeline


def process_single_image(
    image_path: str,
    run_ocr: bool = True,
    force_side: str = None,
    localization_mode: str = config.LOCALIZATION_MODE_ANCHORS,
    yolo_model_path: str = None,
    full_debug: bool = False,
) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        return {"image_path": image_path, "error": f"Could not read image file: {image_path}"}

    result = run_pipeline(
        image,
        run_ocr=run_ocr,
        force_side=force_side,
        localization_mode=localization_mode,
        yolo_model_path=yolo_model_path,
    )

    data = result.as_full_dict() if full_debug else result.as_dict()
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Egyptian National ID OCR & Verification Pipeline CLI"
    )
    parser.add_argument("input_path", help="Path to an ID image or a directory containing images")
    parser.add_argument("-o", "--output", help="Optional output JSON file path to write results")
    parser.add_argument("--no-ocr", action="store_true", help="Run detection and localization only, skip OCR")
    parser.add_argument(
        "--side", choices=["auto", "front", "back"], default="auto",
        help="Force card side classification ('front', 'back', or 'auto')"
    )
    parser.add_argument(
        "--localization-mode", choices=["dynamic_anchors", "yolo_segmentation"],
        default="dynamic_anchors", help="Field localization engine"
    )
    parser.add_argument("--yolo-model", default=None, help="Custom YOLO model weights path")
    parser.add_argument("--full-debug", action="store_true", help="Output full internal diagnostic payload")
    args = parser.parse_args()

    force_side = None if args.side == "auto" else args.side
    input_path = Path(args.input_path)

    if not input_path.exists():
        sys.exit(f"Error: Input path '{args.input_path}' does not exist.")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if input_path.is_file():
        result = process_single_image(
            str(input_path),
            run_ocr=not args.no_ocr,
            force_side=force_side,
            localization_mode=args.localization_mode,
            yolo_model_path=args.yolo_model,
            full_debug=args.full_debug,
        )
        json_output = json.dumps(result, ensure_ascii=False, indent=2)
        print(json_output)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"\nResult saved to: {args.output}", file=sys.stderr)

    elif input_path.is_dir():
        image_files = [p for p in input_path.iterdir() if p.suffix.lower() in image_extensions]
        if not image_files:
            sys.exit(f"Error: No image files found in directory '{args.input_path}'.")

        batch_results = {}
        for idx, img_path in enumerate(sorted(image_files), start=1):
            print(f"[{idx}/{len(image_files)}] Processing {img_path.name}...", file=sys.stderr)
            res = process_single_image(
                str(img_path),
                run_ocr=not args.no_ocr,
                force_side=force_side,
                localization_mode=args.localization_mode,
                yolo_model_path=args.yolo_model,
                full_debug=args.full_debug,
            )
            batch_results[img_path.name] = res

        json_output = json.dumps(batch_results, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"\nProcessed {len(batch_results)} images. Saved to: {args.output}", file=sys.stderr)
        else:
            print(json_output)


if __name__ == "__main__":
    main()
