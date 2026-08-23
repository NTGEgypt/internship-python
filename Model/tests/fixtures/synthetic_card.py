"""
Generates a synthetic "card-like" rectangle with a distinct border and
interior marks, embedded in a larger canvas with configurable margin,
scale, and (optionally) perspective skew. This lets us test the
detection -> rectification -> localization pipeline's scale invariance
(Section 45) WITHOUT needing a real ID photo, which we don't have in this
sandbox. It is not a substitute for testing on real cards (Section 53).
"""
import cv2
import numpy as np


def make_synthetic_card(card_w=856, card_h=540, margin_ratio=0.3,
                         canvas_scale=1.0, skew_px=0):
    """
    Returns a BGR image containing a light rectangular "card" with a dark
    border, on a cluttered gray background, at the given overall canvas
    scale. skew_px shifts the top edge horizontally to simulate mild
    perspective distortion.
    """
    margin = int(card_w * margin_ratio)
    canvas_w = int((card_w + 2 * margin) * canvas_scale)
    canvas_h = int((card_h + 2 * margin) * canvas_scale)

    canvas = np.full((canvas_h, canvas_w, 3), 180, dtype=np.uint8)
    # add background clutter/texture
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 40, size=canvas.shape, dtype=np.uint8)
    canvas = cv2.subtract(canvas, noise // 2)

    cw = int(card_w * canvas_scale)
    ch = int(card_h * canvas_scale)
    mx = int(margin * canvas_scale)
    my = int(margin * canvas_scale)
    sk = int(skew_px * canvas_scale)

    src_card = np.full((ch, cw, 3), 245, dtype=np.uint8)
    cv2.rectangle(src_card, (0, 0), (cw - 1, ch - 1), (30, 30, 30), thickness=max(2, cw // 200))
    # interior "field" marks so blur/quality scoring has texture to read
    cv2.rectangle(src_card, (int(0.05 * cw), int(0.1 * ch)), (int(0.3 * cw), int(0.7 * ch)), (120, 120, 120), -1)
    cv2.putText(src_card, "12345", (int(0.35 * cw), int(0.85 * ch)),
                cv2.FONT_HERSHEY_SIMPLEX, cw / 500, (10, 10, 10), max(1, cw // 300))

    dst_pts = np.float32([
        [mx + sk, my], [mx + cw + sk, my],
        [mx + cw, my + ch], [mx, my + ch],
    ])
    src_pts = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(src_card, H, (canvas_w, canvas_h),
                                  borderMode=cv2.BORDER_TRANSPARENT)

    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    canvas[mask > 0] = warped[mask > 0]

    return canvas
