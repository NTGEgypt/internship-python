"""
Pixel-level preprocessing: card-wide enhancement, per-field OCR
variant generation (upscale/CLAHE/sharpen), and the field-specific
crop shapes (address needs extra vertical room, NID needs a tight
horizontal band, names need generous padding to avoid clipping the
first/last character).
"""

import cv2
import numpy as np


def enhance_card_for_ocr(image):
    denoised = cv2.bilateralFilter(image, 7, 55, 55)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lightness)
    contrast = cv2.cvtColor(cv2.merge([lightness, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.2)
    return cv2.addWeighted(contrast, 1.35, blur, -0.35, 0)


def build_variants(image):
    """General-purpose OCR variants for name/address/serial crops."""
    up = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 7, 7, 21)
    blur = cv2.GaussianBlur(denoised, (0, 0), 2)
    sharp = cv2.addWeighted(denoised, 1.65, blur, -0.65, 0)
    return {
        'original': up,
        'enhanced': cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        'sharp': cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),
    }


def nid_variants(image, aggressive=False):
    """Digit-focused variants for the NID crop. `aggressive=True` adds
    a gamma-correction variant, used when check_image_quality flags
    poor brightness/glare on the source capture."""
    large = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    clip = 4.5 if aggressive else 3.0
    enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)
    _, threshold = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants = {
        'raw': large,
        'enhanced': cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        'threshold': cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR),
    }
    if aggressive:
        gamma_corrected = np.power(gray / 255.0, 0.6) * 255
        variants['gamma'] = cv2.cvtColor(gamma_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return variants


def crop_field(image, box, padding=4):
    left, top, right, bottom = map(int, box)
    height, width = image.shape[:2]
    return image[
        max(0, top - padding):min(height, bottom + padding),
        max(0, left - padding):min(width, right + padding),
    ]


def narrow_address_crop(image, box):
    left, top, right, bottom = map(int, box)
    height, width = image.shape[:2]
    box_width, box_height = right - left, bottom - top
    return image[
        max(0, top - int(box_height * .2)):min(height, bottom + int(box_height * 1.35)),
        max(0, left - int(box_width * .12)):min(width, right + int(box_width * .12)),
    ]


def tight_nid_crop(image, box):
    left, top, right, bottom = map(int, box)
    height, width = image.shape[:2]
    center_y = (top + bottom) // 2
    box_height = bottom - top
    return image[
        max(0, center_y - int(box_height * .75)):min(height, center_y + int(box_height * .75)),
        max(0, left - 8):min(width, right + 8),
    ]
