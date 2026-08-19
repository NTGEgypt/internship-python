"""
Pre-flight image quality check. Runs before detection so the rest of
the pipeline can adapt (e.g. more aggressive NID preprocessing on a
BAD/BORDERLINE capture) and so a null result can be attributed to
capture quality instead of looking like an unexplained failure.
"""

import cv2
import numpy as np


def check_image_quality(
    image,
    min_width=500,
    min_height=300,
    blur_threshold=80,
    min_brightness=45,
    max_brightness=210,
    min_contrast=30,
    max_glare_ratio=0.05,
):
    if image is None:
        raise ValueError("Image is None.")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resolution_passed = width >= min_width and height >= min_height

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_passed = blur_score >= blur_threshold

    brightness = float(np.mean(gray))
    brightness_passed = min_brightness <= brightness <= max_brightness

    contrast = float(np.std(gray))
    contrast_passed = contrast >= min_contrast

    overexposed_pixels = np.sum(gray >= 245)
    total_pixels = gray.size
    glare_ratio = overexposed_pixels / total_pixels if total_pixels > 0 else 1.0
    glare_passed = glare_ratio <= max_glare_ratio

    checks = {
        "resolution": resolution_passed,
        "blur": blur_passed,
        "brightness": brightness_passed,
        "contrast": contrast_passed,
        "glare": glare_passed,
    }

    num_failed = sum(1 for passed in checks.values() if not passed)
    if num_failed == 0:
        status = "GOOD"
    elif num_failed <= 2:
        status = "BORDERLINE"
    else:
        status = "BAD"

    reasons = []
    if not resolution_passed:
        reasons.append(f"Image resolution is too low ({width}x{height}).")
    if not blur_passed:
        reasons.append(f"Image is blurry (sharpness score: {blur_score:.2f}).")
    if not brightness_passed:
        if brightness < min_brightness:
            reasons.append(f"Image is too dark (brightness: {brightness:.2f}).")
        else:
            reasons.append(f"Image is too bright (brightness: {brightness:.2f}).")
    if not contrast_passed:
        reasons.append(f"Image has low contrast (contrast: {contrast:.2f}).")
    if not glare_passed:
        reasons.append(
            f"Image contains too much glare/overexposure "
            f"({glare_ratio * 100:.2f}% of pixels)."
        )

    return {
        "passed": status == "GOOD",
        "status": status,
        "metrics": {
            "width": width,
            "height": height,
            "blur": round(float(blur_score), 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "glare_ratio": round(glare_ratio, 4),
            "glare_percentage": round(glare_ratio * 100, 2),
        },
        "checks": checks,
        "reasons": reasons,
    }


def print_quality_result(result, title="Image Quality"):
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)
    print(f"Status: {result['status']}")
    print(f"Passed: {result['passed']}")
    print("\nMetrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")
    print("\nChecks:")
    for key, passed in result["checks"].items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {key}")
    print("\nProblems:")
    if result["reasons"]:
        for reason in result["reasons"]:
            print(f"  - {reason}")
    else:
        print("  None")
    print("=" * 55)
