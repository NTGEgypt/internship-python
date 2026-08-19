"""
Runs every OCR engine (PaddleOCR-ar, PaddleOCR-en, EasyOCR,
Tesseract) over every preprocessing variant of a field crop, and
normalizes their outputs into one shape:
    {engine, variant, text, confidence, y, x}
The y/x centroid is what lets downstream field-selection order
multi-word/multi-line Arabic text correctly (see text_utils.order_rtl_lines).
"""

import json

import numpy as np
import pytesseract

from core.models import easy_reader, paddle_ar, paddle_en
from core.preprocessing import build_variants
from core.text_utils import clean_space


def paddle_payload(result):
    data = result.json
    if callable(data):
        data = data()
    if isinstance(data, str):
        data = json.loads(data)
    return data.get('res', data) if isinstance(data, dict) else {}


def paddle_texts(model, image, engine):
    output = []
    for result in model.predict(image):
        data = paddle_payload(result)
        polys = data.get('rec_polys', data.get('rec_boxes', []))
        for index, text in enumerate(data.get('rec_texts', [])):
            text = clean_space(text)
            if not text:
                continue
            scores = data.get('rec_scores', [])
            y_center, x_center = 0.0, 0.0
            if index < len(polys):
                pts = np.array(polys[index]).reshape(-1, 2)
                y_center, x_center = float(pts[:, 1].mean()), float(pts[:, 0].mean())
            output.append({
                'engine': engine,
                'text': text,
                'confidence': float(scores[index]) if index < len(scores) else 0.0,
                'y': y_center,
                'x': x_center,
            })
    return output


def ocr_ensemble(field_image):
    output = []
    for variant_name, variant in build_variants(field_image).items():
        for item in paddle_texts(paddle_ar, variant, 'PaddleOCR-ar') + paddle_texts(paddle_en, variant, 'PaddleOCR-en'):
            output.append({**item, 'variant': variant_name})

        for bbox, text, confidence in easy_reader.readtext(variant, detail=1, paragraph=False):
            text = clean_space(text)
            if text:
                pts = np.array(bbox)
                output.append({
                    'engine': 'EasyOCR',
                    'variant': variant_name,
                    'text': text,
                    'confidence': float(confidence),
                    'y': float(pts[:, 1].mean()),
                    'x': float(pts[:, 0].mean()),
                })

        tess = pytesseract.image_to_data(
            variant, lang='ara+eng', config='--oem 3 --psm 11',
            output_type=pytesseract.Output.DICT,
        )
        for index, text in enumerate(tess['text']):
            text = clean_space(text)
            if not text:
                continue
            try:
                confidence = max(0.0, float(tess['conf'][index]) / 100.0)
            except (TypeError, ValueError):
                confidence = 0.0
            output.append({
                'engine': 'Tesseract',
                'variant': variant_name,
                'text': text,
                'confidence': confidence,
                'y': float(tess['top'][index]),
                'x': float(tess['left'][index]),
            })
    return output
