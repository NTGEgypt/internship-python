"""
Unit tests for YOLO Segmentation Field Localization & graceful fallback.
"""
import pytest
import numpy as np
import cv2
from core import config
from core.coordinate_systems import BBox
from core.yolo_localization import YOLOSegmentationLocalizer, FIELD_TYPES
from core.pipeline import run_pipeline


def _make_dummy_card() -> np.ndarray:
    """Create a minimal synthetic ID card image."""
    img = np.full((1009, 1600, 3), 235, dtype=np.uint8)
    # Fake photo
    cv2.rectangle(img, (50, 50), (450, 550), (120, 120, 120), -1)
    # Fake header
    cv2.rectangle(img, (500, 30), (1500, 220), (100, 140, 100), -1)
    # Fake name
    cv2.rectangle(img, (500, 270), (1500, 440), (20, 20, 20), -1)
    # Fake address
    cv2.rectangle(img, (500, 490), (1500, 670), (30, 30, 30), -1)
    # Fake NID
    cv2.rectangle(img, (550, 720), (1500, 860), (10, 10, 10), -1)
    return img


def test_yolo_localizer_reports_unavailable_when_model_missing():
    localizer = YOLOSegmentationLocalizer(model_path="nonexistent/models/yolo_nid_seg.pt")
    assert not localizer.is_available()


def test_yolo_localizer_raises_filenotfound_on_load_when_missing():
    localizer = YOLOSegmentationLocalizer(model_path="nonexistent/models/yolo_nid_seg.pt")
    with pytest.raises(FileNotFoundError):
        localizer.load()


def test_pipeline_graceful_fallback_when_yolo_model_missing():
    card = _make_dummy_card()
    result = run_pipeline(
        card,
        run_ocr=False,
        force_side=config.SIDE_FRONT,
        localization_mode=config.LOCALIZATION_MODE_YOLO,
        yolo_model_path="nonexistent/models/yolo_nid_seg.pt"
    )
    # Must not crash
    assert result is not None
    assert result.document.is_egyptian_id
    assert result.debug.get("localization_mode") in ("dynamic_anchors_fallback", "dynamic_anchors")
    assert "name" in result.fields
    assert "national_id" in result.fields


def test_field_types_schema():
    assert FIELD_TYPES["national_id"] == "numeric"
    assert FIELD_TYPES["name"] == "arabic_text"
    assert FIELD_TYPES["photo"] == "image_region"
    assert FIELD_TYPES["address"] == "arabic_text"


def test_address_boundary_expanded_upper_side():
    from core.templates import FRONT_TEMPLATE_V1
    addr_field = next(f for f in FRONT_TEMPLATE_V1.fields if f.name == "address")
    # Address field spans between Name (ends at ~0.46) and National ID (starts at ~0.71)
    assert addr_field.region.y <= 0.48, f"Address region y ({addr_field.region.y}) must start below name"
    assert addr_field.region.h >= 0.20, f"Address region h ({addr_field.region.h}) should cover address lines"
    assert addr_field.region.y + addr_field.region.h <= 0.72, "Address region must not spill into bottom NID strip"


def test_synthetic_dataset_layout_diversity():
    from training.prepare_dataset import generate_synthetic_sample
    coords = []
    for _ in range(5):
        _, labels = generate_synthetic_sample()
        coords.append(labels)
    # Samples should have distinct labels due to layout jitter, not identical
    assert len(set(["\n".join(c) for c in coords])) > 1, "Synthetic samples must be diverse and not identical"


def test_yolo_localizer_first_name_extraction():
    card = _make_dummy_card()
    result = run_pipeline(
        card,
        run_ocr=False,
        force_side=config.SIDE_FRONT,
        localization_mode=config.LOCALIZATION_MODE_YOLO
    )
    assert result is not None
    if "name" in result.fields:
        result.fields["name"].normalized = "أحمد محمد محمود علي"
        result.fields["name"].ocr_confidence = 0.95
    data = result.as_dict()
    assert data["full_name"] == "أحمد محمد محمود علي"
    assert data["first_name"] == "أحمد"
    assert data["last_name"] == "محمد محمود علي"
    assert data["confidence"]["first_name"] == 0.95



