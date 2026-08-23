"""
Front/back classification (Section 14).

Uses layout evidence on the CANONICAL card:
  - front: photo block in x in [3%, 31%], y in [5%, 67%] (high local chrominance variance)
  - back: PDF417-style barcode block in bottom panel (high horizontal Sobel gradient energy |dI/dx|)
"""
import cv2
import numpy as np
from . import config


def _photo_block_score(canonical_bgr: np.ndarray) -> float:
    h, w = canonical_bgr.shape[:2]
    # Photo region sits in x in [3%, 31%], y in [5%, 67%]
    region = canonical_bgr[int(0.05 * h):int(0.67 * h), int(0.03 * w):int(0.31 * w)]
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))
    return min((sat_std + val_std) / 110.0, 1.0)


def _barcode_block_score(canonical_bgr: np.ndarray) -> float:
    h, w = canonical_bgr.shape[:2]
    # Barcode region in bottom panel
    region = canonical_bgr[int(0.50 * h):int(0.98 * h), int(0.30 * w):int(0.98 * w)]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_density = float(np.mean(np.abs(sobelx) > 40))
    return min(edge_density * 3.0, 1.0)


def classify_side(canonical_bgr: np.ndarray) -> dict:
    photo_score = _photo_block_score(canonical_bgr)
    barcode_score = _barcode_block_score(canonical_bgr)

    front_signal = photo_score - 0.3 * barcode_score
    back_signal = barcode_score - 0.3 * photo_score

    if front_signal < 0.12 and back_signal < 0.12:
        return {
            "side": config.SIDE_UNKNOWN, "confidence": 0.0,
            "evidence": {"photo_score": photo_score, "barcode_score": barcode_score},
        }

    if front_signal >= back_signal:
        conf = min(0.5 + front_signal, 0.98)
        return {"side": config.SIDE_FRONT, "confidence": conf,
                "evidence": {"photo_score": photo_score, "barcode_score": barcode_score}}
    else:
        conf = min(0.5 + back_signal, 0.98)
        return {"side": config.SIDE_BACK, "confidence": conf,
                "evidence": {"photo_score": photo_score, "barcode_score": barcode_score}}
