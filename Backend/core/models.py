"""
Loads every model exactly once at import time: three YOLO detectors
(card, fields, digits) plus three OCR engines (PaddleOCR-ar,
PaddleOCR-en, EasyOCR). Import this module, don't reconstruct these
objects elsewhere — reloading weights per-request is expensive.
"""

import easyocr
import torch
from paddleocr import PaddleOCR
from ultralytics import YOLO

from config import DEVICE, MODEL_DIR

card_model = YOLO(MODEL_DIR / "detect_id_card.pt")
field_model = YOLO(MODEL_DIR / "detect_odjects.pt")
digit_model = YOLO(MODEL_DIR / "detect_id.pt")

paddle_ar = PaddleOCR(
    lang='ar',
    device='cpu',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
    cpu_threads=2,
)

paddle_en = PaddleOCR(
    lang='en',
    device='cpu',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
    cpu_threads=2,
)

easy_reader = easyocr.Reader(['ar', 'en'], gpu=torch.cuda.is_available())
