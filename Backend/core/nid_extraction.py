"""
National ID is handled separately from other fields because it has a
verifiable structure (century digit, birth date, governorate code,
checksum-adjacent parity digit), so acceptance uses per-digit-position
majority voting across engines plus structural validation rather than
requiring an exact string match between engines.
"""

import re
from collections import defaultdict
from datetime import date

import numpy as np
import pytesseract

from config import DEVICE
from core.constants import GOVERNORATES_AR
from core.models import digit_model
from core.text_utils import numeric_text


def decode_nid(nid):
    if not re.fullmatch(r'[0-9]{14}', nid or '') or nid[0] not in {'2', '3'}:
        return None
    year = (1900 if nid[0] == '2' else 2000) + int(nid[1:3])
    try:
        birth_date = date(year, int(nid[3:5]), int(nid[5:7])).isoformat()
    except ValueError:
        return None
    if nid[7:9] not in GOVERNORATES_AR:
        return None
    return {
        'birth_date': birth_date,
        'gender': 'ذكر' if int(nid[12]) % 2 else 'أنثى',
        'governorate': GOVERNORATES_AR[nid[7:9]],
    }


def add_nid_windows(text, engine, variant, confidence, output):
    """Slides a 14-digit window across the digit string so a slightly
    long/short OCR read still yields a usable candidate rather than
    being discarded outright."""
    digits = numeric_text(text)
    for start in range(max(0, len(digits) - 13)):
        value = digits[start:start + 14]
        if len(value) == 14:
            output.append({'nid': value, 'engine': engine, 'variant': variant, 'confidence': float(confidence)})


def tesseract_digit_text(image):
    """Digit-only Tesseract pass used as a second independent reader
    on the NID crop, alongside the digit YOLO."""
    return pytesseract.image_to_string(
        image, lang='eng',
        config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
    )


def yolo_digit_text(image):
    result = digit_model(image, conf=.05, device=DEVICE, verbose=False)[0]
    if result.boxes is None:
        return '', 0.0
    digits = []
    for box in result.boxes:
        label = str(result.names[int(box.cls[0])])
        if label.isdigit():
            digits.append((float(box.xyxy[0][0]), label, float(box.conf[0])))
    digits.sort(key=lambda item: item[0])
    return ''.join(item[1] for item in digits), (float(np.mean([item[2] for item in digits])) if digits else 0.0)


def vote_nid_candidates(candidates):
    """Per-digit-position weighted majority vote across all candidates,
    then validates the resulting 14-digit string structurally. This is
    far more forgiving than requiring two engines to agree on the full
    string byte-for-byte, which rarely happens even on clean images."""
    if not candidates:
        return {'value': None, 'confidence': 0.0, 'needs_review': True, 'alternatives': []}

    position_votes = [defaultdict(float) for _ in range(14)]
    for item in candidates:
        weight = max(item['confidence'], 0.05)
        for i, digit in enumerate(item['nid']):
            position_votes[i][digit] += weight

    consensus = ''.join(max(position_votes[i], key=position_votes[i].get) for i in range(14))
    agreement = float(np.mean([
        position_votes[i][consensus[i]] / sum(position_votes[i].values()) for i in range(14)
    ]))

    decoded = decode_nid(consensus)
    near_engines = {
        item['engine'] for item in candidates
        if sum(a != b for a, b in zip(item['nid'], consensus)) <= 1
    }
    accepted = decoded is not None and (len(near_engines) >= 2 or agreement >= 0.5)

    alt_counts = defaultdict(int)
    for item in candidates:
        alt_counts[item['nid']] += 1
    alternatives = sorted(alt_counts.items(), key=lambda kv: -kv[1])[:8]

    return {
        'value': consensus if accepted else None,
        'confidence': round(agreement, 4) if accepted else 0.0,
        'needs_review': not accepted,
        'alternatives': [{'national_id': k, 'count': v} for k, v in alternatives],
        **(decoded if accepted else {}),
    }
