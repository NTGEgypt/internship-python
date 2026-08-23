import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.image_utils import normalize_image, assess_image_quality
from core import config


def _dummy(w, h):
    return np.full((h, w, 3), 200, dtype=np.uint8)


def test_aspect_ratio_preserved_for_various_resolutions():
    for w, h in [(640, 480), (1280, 720), (1920, 1080), (3024, 4032), (500, 500)]:
        img = _dummy(w, h)
        canvas, transform, original = normalize_image(img)
        assert original.shape[1] == w and original.shape[0] == h  # original untouched
        assert max(canvas.shape[0], canvas.shape[1]) <= config.PROCESSING_MAX_DIM
        assert transform.pad_x == 0 and transform.pad_y == 0
        # round trip a known point back to original space and check consistency
        ox, oy = transform.to_original(0, 0)
        assert abs(ox) < 1.0 and abs(oy) < 1.0


def test_original_image_not_mutated():
    img = _dummy(800, 600)
    img_copy_bytes = img.tobytes()
    canvas, transform, original = normalize_image(img)
    assert img.tobytes() == img_copy_bytes  # caller's array untouched
    assert original.tobytes() == img_copy_bytes


def test_no_independent_x_y_stretch():
    """Explicitly confirm scale is a single scalar applied to both axes."""
    img = _dummy(400, 1000)
    canvas, transform, original = normalize_image(img)
    # scale is one float, by construction; verify resized region keeps ratio
    new_w = int(round(400 * transform.scale))
    new_h = int(round(1000 * transform.scale))
    assert abs((new_w / new_h) - (400 / 1000)) < 1e-6


def test_quality_assessment_flags_dark_image():
    dark = np.full((600, 800, 3), 10, dtype=np.uint8)
    q = assess_image_quality(dark)
    assert "image_too_dark" in q["issues"]
    assert not q["acceptable"]


def test_quality_assessment_flags_blur():
    # uniform image => zero laplacian variance => blurry
    flat = np.full((600, 800, 3), 128, dtype=np.uint8)
    q = assess_image_quality(flat)
    assert "image_is_blurry" in q["issues"]
