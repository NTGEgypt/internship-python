"""
Top-level orchestration. This is the only file that should change
when you add a new field type or reorder pipeline stages — every
stage it calls (quality, detection, preprocessing, OCR, selection,
NID voting) lives in its own module and can be tested independently.
"""

from config import DEVICE
from core.field_selection import choose_address, choose_field
from core.image_quality import check_image_quality
from core.models import card_model, field_model
from core.nid_extraction import (
    add_nid_windows,
    tesseract_digit_text,
    vote_nid_candidates,
    yolo_digit_text,
)
from core.ocr_engines import ocr_ensemble
from core.preprocessing import (
    crop_field,
    enhance_card_for_ocr,
    narrow_address_crop,
    nid_variants,
    tight_nid_crop,
)
from core.constants import ADDRESS_STOPWORDS
from core.text_utils import clean_space

LABELS = {
    'firstname': 'first_name',
    'lastname': 'last_name',
    'address': 'address',
    'serial': 'serial',
    'nid': 'nid',
}


def detect_card(image):
    card_result = card_model(image, device=DEVICE, verbose=False)[0]
    if card_result.boxes is None or len(card_result.boxes) == 0:
        raise ValueError("NASO7Y did not detect an ID card.")
    best_card = int(card_result.boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, card_result.boxes.xyxy[best_card].cpu().numpy())
    card = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
    detection_meta = {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'confidence': float(card_result.boxes.conf[best_card].cpu().item()),
    }
    return card, detection_meta


def detect_fields(card):
    field_result = field_model(card, device=DEVICE, verbose=False)[0]
    if field_result.boxes is None or len(field_result.boxes) == 0:
        raise ValueError("NASO7Y did not detect fields.")

    field_detections = {}
    for box in field_result.boxes:
        raw_label = str(field_result.names[int(box.cls[0])])
        label = LABELS.get(raw_label.lower(), raw_label)
        candidate = {
            'box': box.xyxy[0].cpu().numpy(),
            'confidence': float(box.conf[0]),
            'source_label': raw_label,
        }
        if label not in field_detections or candidate['confidence'] > field_detections[label]['confidence']:
            field_detections[label] = candidate
    return field_detections


def run_field_ocr(ocr_card, field_detections):
    field_ocr = {}
    for label, detection in field_detections.items():
        if label == 'address':
            crop = narrow_address_crop(ocr_card, detection['box'])
        else:
            padding = 8 if label in ('nid', 'first_name', 'last_name') else 4
            crop = crop_field(ocr_card, detection['box'], padding)
        field_ocr[label] = ocr_ensemble(crop)
    return field_ocr


def run_nid_extraction(ocr_card, field_detections, image_quality):
    if 'nid' not in field_detections:
        return vote_nid_candidates([])

    nid_crop = tight_nid_crop(ocr_card, field_detections['nid']['box'])
    bad_quality = (
        image_quality['status'] == 'BAD'
        or not image_quality['checks']['glare']
        or not image_quality['checks']['brightness']
    )

    nid_candidates = []
    for variant_name, variant in nid_variants(nid_crop, aggressive=bad_quality).items():
        digits, confidence = yolo_digit_text(variant)
        add_nid_windows(digits, 'NASO7Y digit YOLO', variant_name, confidence, nid_candidates)

        tess = tesseract_digit_text(variant)
        add_nid_windows(tess, 'Tesseract-nid', variant_name, 0.60, nid_candidates)

    for item in ocr_ensemble(nid_crop):
        add_nid_windows(item['text'], item['engine'], item['variant'], item['confidence'], nid_candidates)

    return vote_nid_candidates(nid_candidates)


def process_image(image):
    image_quality = check_image_quality(image)

    card, card_meta = detect_card(image)
    ocr_card = enhance_card_for_ocr(card)

    field_detections = detect_fields(card)
    field_ocr = run_field_ocr(ocr_card, field_detections)

    first_name, first_confidence = choose_field(field_ocr.get('first_name', []))
    last_name, last_confidence = choose_field(field_ocr.get('last_name', []))
    address, address_confidence = choose_address(field_ocr.get('address', []))
    serial, serial_confidence = choose_field(
        field_ocr.get('serial', []), serial=True, min_engines=2, min_confidence=.55,
    )

    nid_result = run_nid_extraction(ocr_card, field_detections, image_quality)

    full_name = ' '.join(value for value in [first_name, last_name] if value) or None
    address = clean_space(address or '') or None
    address_review = (
        address is None
        or len(address.split()) < 2
        or len(address) < 12
        or bool(set(address.split()) & ADDRESS_STOPWORDS)
    )

    return {
        'national_id': nid_result['value'],
        'full_name': full_name,
        'first_name': first_name,
        'last_name': last_name,
        'birth_date': nid_result.get('birth_date'),
        'gender': nid_result.get('gender'),
        'governorate': nid_result.get('governorate'),
        'address': address,
        'serial_number': serial,
        'confidence': {
            'national_id': nid_result['confidence'],
            'first_name': round(first_confidence, 4),
            'last_name': round(last_confidence, 4),
            'address': round(address_confidence, 4),
            'serial_number': round(serial_confidence, 4),
            'field_detection': {
                label: round(item['confidence'], 4) for label, item in field_detections.items()
            },
        },
        'needs_review': {
            'national_id': nid_result['needs_review'],
            'full_name': full_name is None,
            'address': address_review,
            'serial_number': serial is None,
        },
        'capture_quality': image_quality['status'],
        'card_detection': card_meta,
    }
