"""
Field-specific preprocessing (Section 18).

Each function returns a dict of {variant_name: processed_image} so callers
can run OCR against several variants and let candidate ranking decide
(Section 21). Arabic text is deliberately NOT hit with aggressive
thresholding/sharpening by default -- that destroys dots and connecting
strokes (Section 18 warning).
"""
import cv2
import numpy as np
from . import config


def _upscale_if_small(img: np.ndarray, min_h: int = config.MIN_FIELD_CROP_HEIGHT_PX * 2) -> np.ndarray:
    h, w = img.shape[:2]
    if h < min_h:
        scale = min_h / max(h, 1)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def preprocess_nid(crop_bgr: np.ndarray) -> dict:
    """Numeric strip: high contrast, crisp digit strokes and dots, suppressed background."""
    img = _upscale_if_small(crop_bgr, 60)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Illumination division to remove background watermark and shading
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((15, 15), np.uint8))
    norm = cv2.divide(gray, bg, scale=255)

    # Unsharp mask to make zero dots (٠) and fine strokes crisp
    blur = cv2.GaussianBlur(norm, (0, 0), 2.0)
    sharpened = cv2.addWeighted(norm, 1.8, blur, -0.8, 0)
    variant_a = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    variant_b = clahe.apply(norm)

    _, variant_c = cv2.threshold(variant_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variant_d = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return {
        "illum_sharpened": variant_a,
        "clahe_norm": variant_b,
        "otsu_thresh": variant_c,
        "rgb_original": variant_d,
    }


def preprocess_name(crop_bgr: np.ndarray) -> dict:
    """Arabic name: background illumination division to eliminate pyramids while preserving dots & ligatures."""
    img = _upscale_if_small(crop_bgr, 60)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Background illumination division (erases yellow/orange pyramids and textured background)
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((19, 19), np.uint8))
    norm = cv2.divide(gray, bg, scale=255)

    # 2. Dark Ink Isolation Channel (pure black ink is dark across R,G,B; pyramids have high R,G)
    min_rgb = np.min(img, axis=2)
    bg_min = cv2.morphologyEx(min_rgb, cv2.MORPH_DILATE, np.ones((19, 19), np.uint8))
    norm_min = cv2.divide(min_rgb, bg_min, scale=255)

    # 3. Gentle denoised contrast
    denoised = cv2.fastNlMeansDenoising(norm, h=6)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variant_clahe = clahe.apply(norm)

    return {
        "illum_norm": norm,
        "ink_isolated": norm_min,
        "denoised_norm": denoised,
        "clahe_norm": variant_clahe,
        "rgb_minimal": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    }


def preprocess_address(crop_bgr: np.ndarray) -> dict:
    """Multi-line Arabic address: illumination division + background pyramid suppression."""
    return preprocess_name(crop_bgr)


def preprocess_dob(crop_bgr: np.ndarray) -> dict:
    return preprocess_nid(crop_bgr)


def preprocess_gender(crop_bgr: np.ndarray) -> dict:
    img = _upscale_if_small(crop_bgr, 40)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variant_a = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    variant_b = clahe.apply(gray)
    return {"contrast_norm": variant_a, "clahe_only": variant_b}


def preprocess_barcode(crop_bgr: np.ndarray) -> dict:
    """Barcode decoding wants sharp edges, not denoised softness."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    variant_a = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    variant_b = cv2.filter2D(variant_a, -1, kernel)
    return {"contrast_norm": variant_a, "sharpened": variant_b}


def preprocess_serial(crop_bgr: np.ndarray) -> dict:
    """Card factory serial (e.g. IJ4086874): crisp alphanumerics."""
    img = _upscale_if_small(crop_bgr, 40)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variant_a = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variant_b = clahe.apply(gray)
    _, variant_c = cv2.threshold(variant_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"contrast_norm": variant_a, "clahe_only": variant_b, "otsu_thresh": variant_c}


PREPROCESSORS = {
    "national_id": preprocess_nid,
    "name": preprocess_name,
    "address": preprocess_address,
    "date_of_birth": preprocess_dob,
    "gender": preprocess_gender,
    "serial_number": preprocess_serial,
    "barcode": preprocess_barcode,
}


def preprocess_field(field_name: str, crop_bgr: np.ndarray, field_type: str = "arabic_text") -> dict:
    fn = PREPROCESSORS.get(field_name)
    if fn:
        return fn(crop_bgr)
    # fallback by type
    if field_type == "numeric":
        return preprocess_nid(crop_bgr)
    if field_type == "barcode":
        return preprocess_barcode(crop_bgr)
    return preprocess_name(crop_bgr)
