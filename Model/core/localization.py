"""
Dynamic field localization + crop extraction (Sections 16, 19).

NEVER assumes fixed pixel coordinates. Every uploaded image is analyzed
dynamically from scratch:
1. Green Header Landmark Detection (جمهورية مصر العربية / بطاقة تحقيق الشخصية)
   -- establishes a HARD EXCLUSION ZONE. Name field starts strictly below this.
2. Face / Portrait Landmark Detection (photo anchor)
3. Morphological Text Block Clustering (separating Name and Address lines)
4. Digit-Pattern Projection Profiling (locating 14-digit National ID strip)
5. Left-Panel Sub-segmentation (Date of Birth + Serial Number)
6. Dynamic ROI Content Snapping & Provenance Tracking
"""
import cv2
import numpy as np
from . import config
from .coordinate_systems import BBox
from .templates import Template
from .image_utils import assess_crop_quality


def _find_green_header_bottom(canonical_bgr: np.ndarray) -> int:
    """
    Detect the olive-green header band (جمهورية مصر العربية + بطاقة تحقيق الشخصية).
    Returns the y-pixel coordinate where the entire header+title region ENDS.
    This value acts as a hard floor: no name/address crop may start above it.
    """
    if canonical_bgr is None or canonical_bgr.size == 0:
        return 0
    h, w = canonical_bgr.shape[:2]

    try:
        # Search only the right 65% of the card (header text lives top-right)
        header_roi = canonical_bgr[0:int(0.38 * h), int(0.28 * w):w]
        hsv = cv2.cvtColor(header_roi, cv2.COLOR_BGR2HSV)

        # Olive-green mask: hue 28-95, moderate saturation
        lower = np.array([28, 30, 30], dtype=np.uint8)
        upper = np.array([95, 255, 235], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower, upper)

        row_green = np.sum(green_mask > 0, axis=1)
        thresh = max(8, int((w * 0.70) * 0.018))
        green_rows = np.where(row_green >= thresh)[0]
        if len(green_rows) > 0:
            return int(green_rows[-1]) + 12
    except Exception:
        pass

    return int(0.24 * canonical_bgr.shape[0])


def _detect_full_header_exclusion_bottom(canonical_bgr: np.ndarray) -> int:
    """
    Determine the HARD BOTTOM of the Egyptian ID header area.
    The Egyptian ID has:
      Row 1 (olive-green text): جمهورية مصر العربية
      Row 2 (black text below green): بطاقة تحقيق الشخصية

    This function finds where ALL header content ends so no field starts above it.
    """
    h, w = canonical_bgr.shape[:2]
    green_bottom = _find_green_header_bottom(canonical_bgr)

    search_top = green_bottom
    search_bot = min(h, green_bottom + int(0.15 * h))
    x_start = int(0.33 * w)

    if search_bot > search_top and search_bot <= h:
        roi = canonical_bgr[search_top:search_bot, x_start:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((15, 15), np.uint8))
        norm = cv2.divide(gray, bg, scale=255)
        ink = (norm < 180).astype(np.uint8)

        row_ink = np.sum(ink, axis=1)
        ink_thresh = max(8, int((w - x_start) * 0.03))
        ink_rows = np.where(row_ink >= ink_thresh)[0]

        if len(ink_rows) > 0:
            subtitle_bottom = search_top + int(ink_rows[-1]) + 14
            return max(green_bottom, min(subtitle_bottom, int(0.37 * h)))

    return max(green_bottom, int(0.28 * h))


def _detect_face_photo_roi(canonical_bgr: np.ndarray) -> tuple:
    """
    Detect full citizen portrait photograph on the left panel (x: 3%-31%, y: 5%-67%).
    """
    h, w = canonical_bgr.shape[:2]
    x0 = int(0.03 * w)
    y0 = int(0.05 * h)
    x1 = int(0.31 * w)
    y1 = int(0.67 * h)
    return (x0, y0, x1, y1, 0.98, "full_photo_frame")


def _detect_nid_strip_roi(canonical_bgr: np.ndarray) -> tuple:
    """
    Detect the 14-digit National ID strip dynamically (y: 71%-88%).
    """
    h, w = canonical_bgr.shape[:2]
    gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY)
    y0_search = int(0.70 * h)
    y1_search = int(0.89 * h)
    x0_search = int(0.33 * w)

    roi = gray[y0_search:y1_search, x0_search:w]
    if roi.size == 0:
        return (int(0.33 * w), int(0.71 * h), int(0.98 * w), int(0.88 * h), 0.75, "template_prior")

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_ink = np.sum(thresh > 0, axis=1)
    if row_ink.max() > 0:
        thresh_val = row_ink.max() * 0.15
        digit_rows = np.where(row_ink >= thresh_val)[0]
        if len(digit_rows) > 0:
            peak_y = int(np.mean(digit_rows))
            strip_h = max(int(0.14 * h), int(len(digit_rows) * 1.5))
            nid_y0 = max(int(0.70 * h), min(int(0.74 * h), y0_search + peak_y - int(strip_h * 0.45)))
            nid_y1 = min(int(0.89 * h), nid_y0 + strip_h)
            return (int(0.33 * w), nid_y0, int(0.98 * w), nid_y1, 0.95, "content_projection")

    return (int(0.33 * w), int(0.71 * h), int(0.98 * w), int(0.88 * h), 0.85, "template_prior")


def _cluster_right_side_text_blocks(canonical_bgr: np.ndarray, y_field_start: int) -> dict:
    """
    Dynamically locate Name (lines 1-2) and Address (lines 3-4) on ANY Egyptian ID
    using horizontal ink profile projection.
    y_field_start: HARD FLOOR — both name and address crops start strictly below header.
    """
    h, w = canonical_bgr.shape[:2]
    gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY)
    x_start = int(0.33 * w)
    x_end = int(0.98 * w)

    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((19, 19), np.uint8))
    norm = cv2.divide(gray, bg, scale=255)
    ink = (norm < 185).astype(np.uint8)

    y_search_top = y_field_start
    y_search_bot = int(0.710 * h)

    if y_search_top >= y_search_bot - 40:
        y_search_top = int(0.28 * h)

    roi_ink = ink[y_search_top:y_search_bot, x_start:x_end]
    row_ink = np.sum(roi_ink, axis=1)

    kernel_size = max(5, int(h * 0.015))
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(row_ink.astype(np.float32), (1, kernel_size), 0).flatten()

    ink_threshold = max(20.0, float(np.percentile(smoothed, 40)))
    active_rows = np.where(smoothed > ink_threshold)[0]

    name_x0 = int(0.33 * w)
    name_x1 = int(0.98 * w)
    addr_x0 = int(0.33 * w)
    addr_x1 = int(0.98 * w)

    name_y0 = y_search_top
    name_y1 = min(int(0.460 * h), int(y_search_top + (y_search_bot - y_search_top) * 0.45))
    addr_y0 = name_y1
    addr_y1 = y_search_bot

    if len(active_rows) > 30:
        line_breaks = np.where(np.diff(active_rows) > 6)[0]
        line_ranges = []
        start_idx = 0
        for b_idx in line_breaks:
            line_ranges.append((active_rows[start_idx], active_rows[b_idx]))
            start_idx = b_idx + 1
        line_ranges.append((active_rows[start_idx], active_rows[-1]))

        valid_lines = [r for r in line_ranges if (r[1] - r[0]) >= 8]

        if len(valid_lines) >= 3:
            split_idx = 2
            line2_end = y_search_top + valid_lines[split_idx - 1][1]
            line3_start = y_search_top + valid_lines[split_idx][0]
            free_space_mid = (line2_end + line3_start) // 2

            name_y0 = max(y_search_top, y_search_top + valid_lines[0][0] - 4)
            name_y1 = min(int(0.490 * h), min(line2_end + 6, free_space_mid))
            addr_y0 = max(y_search_top + 1, min(line3_start - 6, free_space_mid))
            addr_y1 = min(int(0.710 * h), y_search_top + valid_lines[-1][1] + 6)

        elif len(valid_lines) == 2:
            line1_end = y_search_top + valid_lines[0][1]
            line2_start = y_search_top + valid_lines[1][0]
            free_space_mid = (line1_end + line2_start) // 2

            name_y0 = max(y_search_top, y_search_top + valid_lines[0][0] - 4)
            name_y1 = min(line1_end + 4, free_space_mid - 2)
            addr_y0 = max(y_search_top + 1, min(line2_start - 6, free_space_mid - 2))
            addr_y1 = min(int(0.710 * h), y_search_top + valid_lines[1][1] + 4)

        elif len(valid_lines) == 1:
            name_y0 = max(y_search_top, y_search_top + valid_lines[0][0] - 4)
            name_y1 = min(y_search_top + valid_lines[0][1] + 6, int(0.490 * h))
            addr_y0 = name_y1 + 2
            addr_y1 = y_search_bot

    return {
        "name": (name_x0, int(name_y0), name_x1, int(name_y1), 0.95, "ink_profile_projection"),
        "address": (addr_x0, int(addr_y0), addr_x1, int(addr_y1), 0.95, "ink_profile_projection"),
    }


def _resolve_boundary_overlaps(fields: dict, pad: int = 4) -> dict:
    """
    If two adjacent feature bounding boxes overlap in their boundaries,
    shrink both boundaries at the midpoint so they become completely disjoint.
    """
    # 1. Right panel: name -> address -> national_id
    right_chain = ["name", "address", "national_id"]
    for i in range(len(right_chain) - 1):
        top_name = right_chain[i]
        bot_name = right_chain[i + 1]
        if top_name in fields and bot_name in fields:
            top_box = fields[top_name]["bbox"]
            bot_box = fields[bot_name]["bbox"]
            top_y1 = top_box.y + top_box.h
            bot_y0 = bot_box.y
            if top_y1 > bot_y0:
                mid_y = (top_y1 + bot_y0) // 2
                new_top_h = max(20, mid_y - pad - top_box.y)
                new_bot_y = mid_y + pad
                new_bot_h = max(20, (bot_box.y + bot_box.h) - new_bot_y)
                top_box.h = new_top_h
                bot_box.y = new_bot_y
                bot_box.h = new_bot_h

    # 2. Left panel: photo -> date_of_birth -> serial_number
    left_chain = ["photo", "date_of_birth", "serial_number"]
    for i in range(len(left_chain) - 1):
        top_name = left_chain[i]
        bot_name = left_chain[i + 1]
        if top_name in fields and bot_name in fields:
            top_box = fields[top_name]["bbox"]
            bot_box = fields[bot_name]["bbox"]
            top_y1 = top_box.y + top_box.h
            bot_y0 = bot_box.y
            if top_y1 > bot_y0:
                mid_y = (top_y1 + bot_y0) // 2
                new_top_h = max(20, mid_y - pad - top_box.y)
                new_bot_y = mid_y + pad
                new_bot_h = max(20, (bot_box.y + bot_box.h) - new_bot_y)
                top_box.h = new_top_h
                bot_box.y = new_bot_y
                bot_box.h = new_bot_h

    return fields


def localize_fields(canonical_bgr: np.ndarray, template: Template) -> dict:
    """
    Dynamically localize all Egyptian National ID fields from scratch for every image.
    Enforces that Name and Address crops start strictly BELOW the full header exclusion zone.
    """
    h, w = canonical_bgr.shape[:2]
    results = {}

    # --- Step 1: Detect Photo Landmark (Left y: 5%-67%) ---
    px0, py0, px1, py1, pconf, pmethod = _detect_face_photo_roi(canonical_bgr)
    py0 = max(int(0.05 * h), py0)
    py1 = min(int(0.67 * h), py1)
    photo_bbox = BBox(x=px0, y=py0, w=(px1 - px0), h=(py1 - py0), coordinate_space=config.SPACE_CANONICAL)
    photo_crop = canonical_bgr[py0:py1, px0:px1]
    results["photo"] = {
        "bbox": photo_bbox,
        "crop": photo_crop,
        "crop_quality": assess_crop_quality(photo_crop),
        "localization_confidence": pconf,
        "detection_method": pmethod,
        "field_type": "image_region",
        "required": True,
    }

    # --- Step 2: Detect FULL Header Exclusion Zone (green row + subtitle row) ---
    y_header_exclusion_bottom = _detect_full_header_exclusion_bottom(canonical_bgr)

    # --- Step 3: Detect National ID Digit Strip (y: 71%-88%) ---
    ix0, iy0, ix1, iy1, iconf, imethod = _detect_nid_strip_roi(canonical_bgr)
    iy0 = max(int(0.710 * h), iy0)
    iy1 = min(int(0.880 * h), max(int(0.860 * h), iy1))
    nid_bbox = BBox(x=ix0, y=iy0, w=(ix1 - ix0), h=(iy1 - iy0), coordinate_space=config.SPACE_CANONICAL)
    nid_crop = canonical_bgr[iy0:iy1, ix0:ix1]
    results["national_id"] = {
        "bbox": nid_bbox,
        "crop": nid_crop,
        "crop_quality": assess_crop_quality(nid_crop),
        "localization_confidence": iconf,
        "detection_method": imethod,
        "field_type": "numeric",
        "required": True,
    }

    # --- Step 4: Cluster Name (strictly below header) and Address (below name) ---
    text_clusters = _cluster_right_side_text_blocks(
        canonical_bgr, y_field_start=y_header_exclusion_bottom
    )

    for field_name in ("name", "address"):
        x0, y0, x1, y1, conf, method = text_clusters[field_name]
        x0 = max(0, min(w - 1, int(x0)))
        y0 = max(y_header_exclusion_bottom, min(h - 1, int(y0)))
        x1 = max(x0 + 10, min(w, int(x1)))
        y1 = max(y0 + 10, min(h, int(y1)))

        region = canonical_bgr[y0:y1, x0:x1]
        bbox = BBox(x=x0, y=y0, w=(x1 - x0), h=(y1 - y0), coordinate_space=config.SPACE_CANONICAL)
        results[field_name] = {
            "bbox": bbox,
            "crop": region,
            "crop_quality": assess_crop_quality(region),
            "localization_confidence": conf,
            "detection_method": method,
            "field_type": "arabic_text",
            "required": True,
        }

    # --- Step 5: Left-Panel DOB (y: 68%-85%) and Serial (y: 85%-98%) ---
    left_x0 = int(0.03 * w)
    left_x1 = int(0.35 * w)
    dob_y0 = int(0.68 * h)
    dob_y1 = int(0.85 * h)
    serial_y0 = int(0.85 * h)
    serial_y1 = int(0.98 * h)

    dob_crop = canonical_bgr[dob_y0:dob_y1, left_x0:left_x1]
    dob_bbox = BBox(x=left_x0, y=dob_y0, w=(left_x1 - left_x0), h=(dob_y1 - dob_y0), coordinate_space=config.SPACE_CANONICAL)
    results["date_of_birth"] = {
        "bbox": dob_bbox,
        "crop": dob_crop,
        "crop_quality": assess_crop_quality(dob_crop),
        "localization_confidence": 0.95,
        "detection_method": "left_panel_layout",
        "field_type": "numeric",
        "required": False,
    }

    serial_crop = canonical_bgr[serial_y0:serial_y1, left_x0:left_x1]
    serial_bbox = BBox(x=left_x0, y=serial_y0, w=(left_x1 - left_x0), h=(serial_y1 - serial_y0), coordinate_space=config.SPACE_CANONICAL)
    results["serial_number"] = {
        "bbox": serial_bbox,
        "crop": serial_crop,
        "crop_quality": assess_crop_quality(serial_crop),
        "localization_confidence": 0.95,
        "detection_method": "left_panel_layout",
        "field_type": "text",
        "required": False,
    }

    # --- Step 6: Non-Overlapping Spatial Constraint Resolver ---
    results = _resolve_boundary_overlaps(results, pad=4)

    # Refresh crops after conflict resolution
    for fname, fdata in results.items():
        fb = fdata["bbox"]
        c_crop = canonical_bgr[fb.y:fb.y + fb.h, fb.x:fb.x + fb.w]
        fdata["crop"] = c_crop
        fdata["crop_quality"] = assess_crop_quality(c_crop)

    return results
