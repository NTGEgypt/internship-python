"""
Usage:
    python main.py path/to/id_card.jpg
"""

import json
import sys

import cv2

from core.image_quality import print_quality_result
from core.pipeline import process_image


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read {image_path}")

    result = process_image(image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
