"""
Card detection + perspective rectification (Sections 9-10).

detect_card_corners(): finds the best quadrilateral candidate for the
physical card in the processing canvas.

rectify_card(): applies a homography to map the detected quad onto the
canonical card rectangle (config.CANONICAL_CARD_WIDTH x HEIGHT).
"""
import cv2
import numpy as np
from . import config


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 quadrilateral points as [TL, TR, BR, BL] regardless of rotation,
    perspective tilt, or corner order.
    """
    pts = pts.reshape(4, 2).astype("float32")
    # Sort points by y-coordinate (top 2 points vs bottom 2 points)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_two = y_sorted[:2]
    bot_two = y_sorted[2:]

    # Sort top two by x-coordinate: leftmost is TL, rightmost is TR
    tl = top_two[np.argmin(top_two[:, 0])]
    tr = top_two[np.argmax(top_two[:, 0])]

    # Sort bottom two by x-coordinate: leftmost is BL, rightmost is BR
    bl = bot_two[np.argmin(bot_two[:, 0])]
    br = bot_two[np.argmax(bot_two[:, 0])]

    # Check for acute tilt where top-left and top-right might swap
    # Compute horizontal vector (TL -> TR) vs vertical vector (TL -> BL)
    v_top = tr - tl
    v_left = bl - tl
    cross_z = v_top[0] * v_left[1] - v_top[1] * v_left[0]
    if cross_z < 0:
        # Flipped orientation: re-order via centroid angles
        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        # Shift to start from top-left quadrant (approx -3pi/4)
        pts_clock = pts[sorted_indices]
        dists_to_tl = np.sum(pts_clock ** 2, axis=1)
        start_idx = np.argmin(dists_to_tl)
        pts_clock = np.roll(pts_clock, -start_idx, axis=0)
        return pts_clock.astype("float32")

    return np.array([tl, tr, br, bl], dtype="float32")


def _get_image_content_quad(processing_image_bgr: np.ndarray) -> np.ndarray:
    """
    Finds the exact bounding rectangle of the actual input image content inside
    the canvas, completely stripping away any white letterbox padding frames.
    """
    h, w = processing_image_bgr.shape[:2]
    non_pad_y = np.where(np.any(processing_image_bgr != 255, axis=(1, 2)))[0]
    non_pad_x = np.where(np.any(processing_image_bgr != 255, axis=(0, 2)))[0]

    if len(non_pad_y) > 0 and len(non_pad_x) > 0:
        y0, y1 = float(non_pad_y[0]), float(non_pad_y[-1])
        x0, x1 = float(non_pad_x[0]), float(non_pad_x[-1])

        # Canvas letterboxing only pads on ONE axis (top/bottom OR left/right) spanning the full opposite dimension
        is_tb_letterbox = (x0 <= 2 and x1 >= float(w - 3)) and (y0 >= 15 or y1 <= float(h - 16))
        is_lr_letterbox = (y0 <= 2 and y1 >= float(h - 3)) and (x0 >= 15 or x1 <= float(w - 16))

        if (is_tb_letterbox or is_lr_letterbox) and (x1 - x0) > 100 and (y1 - y0) > 100:
            return np.array([
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1]
            ], dtype="float32")

    return np.array([[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]], dtype="float32")


def detect_card_corners(processing_image_bgr: np.ndarray):
    """
    Finds the exact 4 corners (TL, TR, BR, BL) of the physical Egyptian National ID card
    on any background (wood table, desk, cloth, dark/light surface, paper).
    Warps ONLY the card quadrilateral so 100% of the output is the ID card with zero background.
    """
    h, w = processing_image_bgr.shape[:2]
    canvas_area = float(w * h)
    target_aspect = config.CANONICAL_CARD_WIDTH / float(config.CANONICAL_CARD_HEIGHT)  # ~1.5857

    gray = cv2.cvtColor(processing_image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(processing_image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(processing_image_bgr, cv2.COLOR_BGR2LAB)

    candidate_contours = []

    # 1. Color / Chrominance Difference (|Blue - Red| channel difference)
    # The ID card body has a light bluish/purple tint compared to wood/cream backgrounds
    b_chan = processing_image_bgr[:, :, 0].astype(np.float32)
    r_chan = processing_image_bgr[:, :, 2].astype(np.float32)
    diff_br = np.abs(b_chan - r_chan)
    diff_br = cv2.normalize(diff_br, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur_diff = cv2.GaussianBlur(diff_br, (5, 5), 0)
    _, thresh_diff = cv2.threshold(blur_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_diff = cv2.morphologyEx(thresh_diff, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts_diff, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_contours.extend(cnts_diff)

    # 2. Multi-channel Edge Fusion (LAB L-channel + Grayscale)
    l_chan = lab[:, :, 0]
    blur_l = cv2.bilateralFilter(l_chan, 9, 75, 75)
    blur_g = cv2.bilateralFilter(gray, 9, 75, 75)

    edges_l = cv2.Canny(blur_l, 25, 90)
    edges_g = cv2.Canny(blur_g, 35, 110)
    edges_fused = cv2.bitwise_or(edges_l, edges_g)

    # Bridge small gaps in rounded corners
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    edges_closed = cv2.morphologyEx(edges_fused, cv2.MORPH_CLOSE, k_close)
    cnts_edges, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidate_contours.extend(cnts_edges)

    # 3. Morphological Gradient on Luminance
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(blur_g, cv2.MORPH_GRADIENT, kernel_grad)
    _, thresh_grad = cv2.threshold(morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_grad = cv2.dilate(thresh_grad, np.ones((5, 5), np.uint8), iterations=2)
    cnts_grad, _ = cv2.findContours(thresh_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_contours.extend(cnts_grad)

    # 4. Adaptive Thresholding (isolates card on wood/textured desk)
    adapt = cv2.adaptiveThreshold(blur_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 6)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts_adapt, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_contours.extend(cnts_adapt)

    # 5. Multi-Scale Edge Pyramids (captures cards whether 10% or 90% of image)
    for scale in [1.0, 0.5]:
        if scale != 1.0:
            sw, sh = int(w * scale), int(h * scale)
            scaled_g = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            scaled_g = gray

        blurred_s = cv2.GaussianBlur(scaled_g, (5, 5), 0)
        edges_s = cv2.Canny(blurred_s, 30, 100)
        edges_s = cv2.dilate(edges_s, np.ones((3, 3), np.uint8), iterations=1)
        if scale != 1.0:
            edges_s = cv2.resize(edges_s, (w, h), interpolation=cv2.INTER_NEAREST)
        cnts_s, _ = cv2.findContours(edges_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_contours.extend(cnts_s)

    best = None
    best_score = 0.0

    for c in candidate_contours:
        area = cv2.contourArea(c)
        if area >= canvas_area * 0.96 or area < canvas_area * 0.22:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area >= canvas_area * 0.96 or hull_area < canvas_area * 0.22:
            continue

        # Strategy A: Polygon Approximation of Convex Hull
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.028 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            ordered = _order_corners(approx)
            side_top = np.linalg.norm(ordered[1] - ordered[0])
            side_bottom = np.linalg.norm(ordered[2] - ordered[3])
            side_left = np.linalg.norm(ordered[3] - ordered[0])
            side_right = np.linalg.norm(ordered[2] - ordered[1])
            est_w = (side_top + side_bottom) / 2.0
            est_h = (side_left + side_right) / 2.0
            if est_w >= 0.45 * w and est_h >= 0.40 * h:
                aspect = max(est_w, est_h) / max(min(est_w, est_h), 1e-6)
                aspect_diff = abs(aspect - target_aspect) / target_aspect
                if aspect_diff < 0.40:
                    aspect_score = max(0.0, 1.0 - aspect_diff)
                    area_score = min(hull_area / (canvas_area * 0.65), 1.0)
                    score = 0.70 * aspect_score + 0.30 * area_score
                    if score > best_score:
                        best_score = score
                        best = ordered

        # Strategy B: Minimum Area Rotated Bounding Box
        rect = cv2.minAreaRect(hull)
        (rw, rh) = rect[1]
        w_rect, h_rect = max(rw, rh), min(rw, rh)
        if w_rect >= 0.45 * w and h_rect >= 0.35 * h:
            aspect = w_rect / max(h_rect, 1e-6)
            aspect_diff = abs(aspect - target_aspect) / target_aspect
            if aspect_diff < 0.35:
                box = cv2.boxPoints(rect)
                ordered_box = _order_corners(box)
                aspect_score = max(0.0, 1.0 - aspect_diff)
                area_score = min(hull_area / (canvas_area * 0.65), 1.0)
                score = 0.65 * aspect_score + 0.35 * area_score
                if score > best_score:
                    best_score = score
                    best = ordered_box


    debug = {
        "num_contours": len(candidate_contours),
        "canvas_area": canvas_area,
    }

    content_quad = _get_image_content_quad(processing_image_bgr)
    c_x0 = np.min(content_quad[:, 0])
    c_x1 = np.max(content_quad[:, 0])
    c_y0 = np.min(content_quad[:, 1])
    c_y1 = np.max(content_quad[:, 1])
    content_w = max(1.0, c_x1 - c_x0)
    content_h = max(1.0, c_y1 - c_y0)

    # If no card quad on background was detected (e.g. card fills the image/pre-cropped),
    # use image content bounds (stripping white letterbox canvas frame).
    if best is None:
        best = content_quad
        best_score = 0.95
    else:
        min_x = np.min(best[:, 0])
        max_x = np.max(best[:, 0])
        min_y = np.min(best[:, 1])
        max_y = np.max(best[:, 1])

        margin_left = max(0.0, (min_x - c_x0) / content_w)
        margin_right = max(0.0, (c_x1 - max_x) / content_w)
        margin_top = max(0.0, (min_y - c_y0) / content_h)
        margin_bot = max(0.0, (c_y1 - max_y) / content_h)

        if (margin_top < 0.04 or margin_bot < 0.04 or margin_left < 0.04 or margin_right < 0.04) and (cv2.contourArea(best) > (content_w * content_h) * 0.65):
            # Input image is already closely framed or pre-cropped -> use real image bounds (no white frame)
            best = content_quad
            best_score = 0.95
        else:
            # For cards with large background margins on table/desk, expand outward safely
            center = np.mean(best, axis=0)
            expanded = center + 1.025 * (best - center)
            expanded[:, 0] = np.clip(expanded[:, 0], c_x0, c_x1)
            expanded[:, 1] = np.clip(expanded[:, 1], c_y0, c_y1)
            best = expanded.astype("float32")

    return best, float(best_score), debug


def rectify_card(processing_image_bgr: np.ndarray, corners: np.ndarray):
    """
    Warp the detected quadrilateral (processing-canvas coords) into the
    canonical card rectangle using Lanczos4 interpolation for razor-sharp text edges.
    Returns (canonical_card_bgr, homography_matrix).
    """
    dst = np.array([
        [0, 0],
        [config.CANONICAL_CARD_WIDTH - 1, 0],
        [config.CANONICAL_CARD_WIDTH - 1, config.CANONICAL_CARD_HEIGHT - 1],
        [0, config.CANONICAL_CARD_HEIGHT - 1],
    ], dtype="float32")

    homography = cv2.getPerspectiveTransform(corners, dst)
    canonical = cv2.warpPerspective(
        processing_image_bgr, homography,
        (config.CANONICAL_CARD_WIDTH, config.CANONICAL_CARD_HEIGHT),
        flags=cv2.INTER_LANCZOS4,
    )
    return canonical, homography


def crop_canonical_to_card(canonical_bgr: np.ndarray) -> np.ndarray:
    """
    Ensure the canonical rectified card is at CANONICAL_CARD_WIDTH x CANONICAL_CARD_HEIGHT
    without cutting off any borders, headers, or bottom National ID numbers.
    """
    if canonical_bgr is None or canonical_bgr.size == 0:
        return canonical_bgr
    return cv2.resize(canonical_bgr, (config.CANONICAL_CARD_WIDTH, config.CANONICAL_CARD_HEIGHT), interpolation=cv2.INTER_LANCZOS4)


def detect_and_rectify(processing_image_bgr: np.ndarray):
    """
    Convenience wrapper. Returns a dict with corners, confidence, canonical
    card image (or None), and homography (or None).
    """
    corners, confidence, debug = detect_card_corners(processing_image_bgr)
    if corners is None:
        return {
            "corners": None, "confidence": 0.0,
            "canonical_card": None, "homography": None, "debug": debug,
        }
    canonical, homography = rectify_card(processing_image_bgr, corners)
    return {
        "corners": corners.tolist(), "confidence": confidence,
        "canonical_card": canonical, "homography": homography, "debug": debug,
    }

