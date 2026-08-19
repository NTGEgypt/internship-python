import os

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_xdnn'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pytesseract
import torch

from paddleocr import PaddleOCR
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


# ============================================================
# DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# MODELS
# ============================================================

MODEL_DIR = ROOT / "models"

card_model = YOLO(MODEL_DIR / "detect_id_card.pt")
field_model = YOLO(MODEL_DIR / "detect_odjects.pt")
digit_model = YOLO(MODEL_DIR / "detect_id.pt")


# ============================================================
# OCR
# ============================================================

paddle_ar = PaddleOCR(
    lang='ar',
    device='cpu',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
    cpu_threads=2
)

paddle_en = PaddleOCR(
    lang='en',
    device='cpu',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
    cpu_threads=2
)

easy_reader = easyocr.Reader(
    ['ar', 'en'],
    gpu=torch.cuda.is_available()
)


# ============================================================
# CONSTANTS
# ============================================================

ARABIC_DIGITS = str.maketrans(
    '٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹',
    '01234567890123456789'
)

DIGIT_CONFUSIONS = {
    'O': '0',
    'o': '0',
    'D': '0',
    'I': '1',
    'l': '1',
    '|': '1',
    'Z': '2',
    'z': '2',
    'S': '5',
    's': '5',
    'G': '6',
    'g': '6',
    'B': '8',
    'A': '4',
    'E': '3',
    'b': '6',
    'q': '9'
}

ADDRESS_STOPWORDS = {
    'جمهورية',
    'مصر',
    'العربية',
    'بطاقة',
    'تحقيق',
    'الشخصية',
    'الرقم',
    'القومي',
    'وزارة',
    'الداخلية'
}

GOVERNORATES_AR = {
    '01': 'القاهرة',
    '02': 'الإسكندرية',
    '03': 'بورسعيد',
    '04': 'السويس',
    '11': 'دمياط',
    '12': 'الدقهلية',
    '13': 'الشرقية',
    '14': 'القليوبية',
    '15': 'كفر الشيخ',
    '16': 'الغربية',
    '17': 'المنوفية',
    '18': 'البحيرة',
    '19': 'الإسماعيلية',
    '21': 'الجيزة',
    '22': 'بني سويف',
    '23': 'الفيوم',
    '24': 'المنيا',
    '25': 'أسيوط',
    '26': 'سوهاج',
    '27': 'قنا',
    '28': 'أسوان',
    '29': 'الأقصر',
    '31': 'البحر الأحمر',
    '32': 'الوادي الجديد',
    '33': 'مطروح',
    '34': 'شمال سيناء',
    '35': 'جنوب سيناء',
    '88': 'خارج مصر'
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_space(text):
    return re.sub(r'\s+', ' ', str(text)).strip()


def arabic_normalize(text):
    text = str(text)

    for old, new in {
        'أ': 'ا',
        'إ': 'ا',
        'آ': 'ا',
        'ى': 'ي'
    }.items():
        text = text.replace(old, new)

    return clean_space(text)


def numeric_text(text):
    text = str(text).translate(ARABIC_DIGITS)

    for old, new in DIGIT_CONFUSIONS.items():
        text = text.replace(old, new)

    return re.sub(r'[^0-9]', '', text)


def enhance_card_for_ocr(image):
    denoised = cv2.bilateralFilter(
        image,
        7,
        55,
        55
    )

    lab = cv2.cvtColor(
        denoised,
        cv2.COLOR_BGR2LAB
    )

    lightness, a_channel, b_channel = cv2.split(lab)

    lightness = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    ).apply(lightness)

    contrast = cv2.cvtColor(
        cv2.merge([
            lightness,
            a_channel,
            b_channel
        ]),
        cv2.COLOR_LAB2BGR
    )

    blur = cv2.GaussianBlur(
        contrast,
        (0, 0),
        1.2
    )

    return cv2.addWeighted(
        contrast,
        1.35,
        blur,
        -0.35,
        0
    )


def build_variants(image):
    up = cv2.resize(
        image,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        up,
        cv2.COLOR_BGR2GRAY
    )

    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        7,
        7,
        21
    )

    blur = cv2.GaussianBlur(
        denoised,
        (0, 0),
        2
    )

    sharp = cv2.addWeighted(
        denoised,
        1.65,
        blur,
        -0.65,
        0
    )

    return {
        'original': up,
        'enhanced': cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2BGR
        ),
        'sharp': cv2.cvtColor(
            sharp,
            cv2.COLOR_GRAY2BGR
        )
    }


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

        polys = data.get(
            'rec_polys',
            data.get('rec_boxes', [])
        )

        for index, text in enumerate(
            data.get('rec_texts', [])
        ):

            text = clean_space(text)

            if text:

                scores = data.get(
                    'rec_scores',
                    []
                )

                y_center = 0.0
                x_center = 0.0

                if index < len(polys):

                    pts = np.array(
                        polys[index]
                    ).reshape(-1, 2)

                    y_center = float(
                        pts[:, 1].mean()
                    )

                    x_center = float(
                        pts[:, 0].mean()
                    )

                output.append({
                    'engine': engine,
                    'text': text,
                    'confidence': (
                        float(scores[index])
                        if index < len(scores)
                        else 0.0
                    ),
                    'y': y_center,
                    'x': x_center
                })

    return output


def ocr_ensemble(field_image):

    output = []

    for variant_name, variant in build_variants(field_image).items():

        for item in (
            paddle_texts(
                paddle_ar,
                variant,
                'PaddleOCR-ar'
            )
            +
            paddle_texts(
                paddle_en,
                variant,
                'PaddleOCR-en'
            )
        ):
            output.append({
                **item,
                'variant': variant_name
            })

        for bbox, text, confidence in easy_reader.readtext(
            variant,
            detail=1,
            paragraph=False
        ):

            text = clean_space(text)

            if text:

                pts = np.array(bbox)

                output.append({
                    'engine': 'EasyOCR',
                    'variant': variant_name,
                    'text': text,
                    'confidence': float(confidence),
                    'y': float(pts[:, 1].mean()),
                    'x': float(pts[:, 0].mean())
                })

        tess = pytesseract.image_to_data(
            variant,
            lang='ara+eng',
            config='--oem 3 --psm 11',
            output_type=pytesseract.Output.DICT
        )

        for index, text in enumerate(
            tess['text']
        ):

            text = clean_space(text)

            if text:

                try:
                    confidence = max(
                        0.0,
                        float(tess['conf'][index]) / 100.0
                    )
                except (TypeError, ValueError):
                    confidence = 0.0

                output.append({
                    'engine': 'Tesseract',
                    'variant': variant_name,
                    'text': text,
                    'confidence': confidence
                })

    return output


def decode_nid(nid):

    if not re.fullmatch(
        r'[0-9]{14}',
        nid or ''
    ) or nid[0] not in {'2', '3'}:
        return None

    year = (
        1900 if nid[0] == '2'
        else 2000
    ) + int(nid[1:3])

    try:
        birth_date = date(
            year,
            int(nid[3:5]),
            int(nid[5:7])
        ).isoformat()

    except ValueError:
        return None

    if nid[7:9] not in GOVERNORATES_AR:
        return None

    return {
        'birth_date': birth_date,
        'gender': 'ذكر' if int(nid[12]) % 2 else 'أنثى',
        'governorate': GOVERNORATES_AR[nid[7:9]]
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def check_image_quality(
    image,
    min_width=500,
    min_height=300,
    blur_threshold=80,
    min_brightness=45,
    max_brightness=210,
    min_contrast=30,
    max_glare_ratio=0.05
):
    """
    Evaluate image quality for OCR.

    Returns:
        {
            "passed": bool,
            "status": "GOOD" | "BORDERLINE" | "BAD",
            "metrics": {...},
            "checks": {...},
            "reasons": [...]
        }
    """

    if image is None:
        raise ValueError("Image is None.")

    height, width = image.shape[:2]

    # 1. Resolution
    resolution_passed = (
        width >= min_width and
        height >= min_height
    )

    # 2. Blur / sharpness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    blur_passed = blur_score >= blur_threshold

    # 3. Brightness
    brightness = float(np.mean(gray))

    brightness_passed = (
        min_brightness <= brightness <= max_brightness
    )

    # 4. Contrast
    contrast = float(np.std(gray))

    contrast_passed = contrast >= min_contrast

    # 5. Glare / overexposure
    overexposed_pixels = np.sum(gray >= 245)
    total_pixels = gray.size

    glare_ratio = (
        overexposed_pixels / total_pixels
        if total_pixels > 0 else 1.0
    )

    glare_passed = glare_ratio <= max_glare_ratio

    # Determine status
    checks = {
        "resolution": resolution_passed,
        "blur": blur_passed,
        "brightness": brightness_passed,
        "contrast": contrast_passed,
        "glare": glare_passed
    }

    failed_checks = [
        name for name, passed in checks.items()
        if not passed
    ]

    num_failed = len(failed_checks)

    if num_failed == 0:
        status = "GOOD"
    elif num_failed <= 2:
        status = "BORDERLINE"
    else:
        status = "BAD"

    # Human-readable reasons
    reasons = []

    if not resolution_passed:
        reasons.append(
            f"Image resolution is too low "
            f"({width}x{height})."
        )

    if not blur_passed:
        reasons.append(
            f"Image is blurry "
            f"(sharpness score: {blur_score:.2f})."
        )

    if not brightness_passed:
        if brightness < min_brightness:
            reasons.append(
                f"Image is too dark "
                f"(brightness: {brightness:.2f})."
            )
        else:
            reasons.append(
                f"Image is too bright "
                f"(brightness: {brightness:.2f})."
            )

    if not contrast_passed:
        reasons.append(
            f"Image has low contrast "
            f"(contrast: {contrast:.2f})."
        )

    if not glare_passed:
        reasons.append(
            f"Image contains too much glare/overexposure "
            f"({glare_ratio * 100:.2f}% of pixels)."
        )

    return {
        "passed": status == "GOOD",
        "status": status,

        "metrics": {
            "width": width,
            "height": height,
            "blur": round(float(blur_score), 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "glare_ratio": round(glare_ratio, 4),
            "glare_percentage": round(glare_ratio * 100, 2)
        },

        "checks": checks,

        "reasons": reasons
    }


def print_quality_result(result, title="Image Quality"):
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)

    print(f"Status: {result['status']}")
    print(f"Passed: {result['passed']}")

    print("\nMetrics:")

    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")

    print("\nChecks:")

    for key, passed in result["checks"].items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {key}")

    print("\nProblems:")

    if result["reasons"]:
        for reason in result["reasons"]:
            print(f"  - {reason}")
    else:
        print("  None")

    print("=" * 55)

def crop_field(image, box, padding=4):
    left, top, right, bottom = map(int, box)

    height, width = image.shape[:2]

    return image[
        max(0, top-padding):min(height, bottom+padding),
        max(0, left-padding):min(width, right+padding)
    ]


def narrow_address_crop(image, box):
    left, top, right, bottom = map(int, box)

    height, width = image.shape[:2]

    box_width = right - left
    box_height = bottom - top

    return image[
        max(0, top-int(box_height*.2)):
        min(height, bottom+int(box_height*1.35)),
        max(0, left-int(box_width*.12)):
        min(width, right+int(box_width*.12))
    ]

def choose_field(results, serial=False, min_engines=1, min_confidence=0.0):
    groups = defaultdict(list)
    for item in results:
        value = re.sub(r'[^A-Za-z0-9]', '', item['text']).upper() if serial else arabic_normalize(item['text'])
        if len(value) >= (5 if serial else 2): groups[value].append(item)
    if not groups: return None, 0.0
    ranked = []
    for value, items in groups.items():
        confidence = float(np.mean([item['confidence'] for item in items]))
        engines = len({item['engine'] for item in items})
        ranked.append((engines*10+len(items)*2+confidence*3, value, confidence, engines))
    ranked.sort(reverse=True, key=lambda item:item[0])
    _, value, confidence, engines = ranked[0]
    if engines < min_engines or confidence < min_confidence:
        return None, 0.0
    return value, confidence

def order_rtl_lines(items, line_tol=15):
    if not items: return []
    lines = []
    for item in sorted(items, key=lambda i: i.get('y', 0)):
        placed = False
        for line in lines:
            if abs(line[0].get('y', 0) - item.get('y', 0)) <= line_tol:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])
    lines.sort(key=lambda line: np.mean([i.get('y', 0) for i in line]))
    ordered = []
    for line in lines:
        line.sort(key=lambda i: -i.get('x', 0))
        ordered.extend(line)
    return ordered

def choose_address(results):
    grouped = defaultdict(list)
    for item in results:
        text = arabic_normalize(item['text'])
        if len(text) >= 2 and not (set(text.split()) & ADDRESS_STOPWORDS): grouped[(item['engine'],item['variant'])].append({**item,'text':text})
    candidates = []
    for (engine, variant), items in grouped.items():
        lines, seen = [], []
        for item in order_rtl_lines(items):
            key = re.sub(r'\s+', '', item['text'])
            if key not in seen: seen.append(key); lines.append(item['text'])
        value = clean_space(' '.join(line for line in lines if len(numeric_text(line)) < 3))
        if value:
            confidence = float(np.mean([item['confidence'] for item in items]))
            score = confidence*5 + min(len(value),60)/18 + (2 if len(value.split())>=3 else 0) + (2 if any(gov in value for gov in GOVERNORATES_AR.values()) else 0)
            candidates.append((score,value,confidence))
    if not candidates: return None, 0.0
    _, value, confidence = max(candidates, key=lambda item:item[0])
    return value, confidence

def nid_variants(image, aggressive=False):
    large = cv2.resize(image,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(large,cv2.COLOR_BGR2GRAY)
    clip = 4.5 if aggressive else 3.0
    enhanced = cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)).apply(gray)
    _, threshold = cv2.threshold(enhanced,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    variants = {'raw':large,'enhanced':cv2.cvtColor(enhanced,cv2.COLOR_GRAY2BGR),'threshold':cv2.cvtColor(threshold,cv2.COLOR_GRAY2BGR)}
    if aggressive:
        gamma_corrected = np.power(gray/255.0, 0.6)*255
        variants['gamma'] = cv2.cvtColor(gamma_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return variants

def add_nid_windows(text, engine, variant, confidence, output):
    digits = numeric_text(text)
    for start in range(max(0,len(digits)-13)):
        value = digits[start:start+14]
        if len(value)==14: output.append({'nid':value,'engine':engine,'variant':variant,'confidence':float(confidence)})

def yolo_digit_text(image):
    result = digit_model(image,conf=.05,device=DEVICE,verbose=False)[0]
    if result.boxes is None: return '',0.0
    digits=[]
    for box in result.boxes:
        label=str(result.names[int(box.cls[0])])
        if label.isdigit(): digits.append((float(box.xyxy[0][0]),label,float(box.conf[0])))
    digits.sort(key=lambda item:item[0])
    return ''.join(item[1] for item in digits), float(np.mean([item[2] for item in digits])) if digits else 0.0

def vote_nid_candidates(candidates):
    if not candidates: return {'value':None,'confidence':0.0,'needs_review':True,'alternatives':[]}
    position_votes = [defaultdict(float) for _ in range(14)]
    for item in candidates:
        weight = max(item['confidence'], 0.05)
        for i, digit in enumerate(item['nid']):
            position_votes[i][digit] += weight
    consensus = ''.join(max(position_votes[i], key=position_votes[i].get) for i in range(14))
    agreement = float(np.mean([position_votes[i][consensus[i]] / sum(position_votes[i].values()) for i in range(14)]))
    decoded = decode_nid(consensus)
    near_engines = {item['engine'] for item in candidates if sum(a != b for a, b in zip(item['nid'], consensus)) <= 1}
    accepted = decoded is not None and (len(near_engines) >= 2 or agreement >= 0.5)
    alt_counts = defaultdict(int)
    for item in candidates: alt_counts[item['nid']] += 1
    alternatives = sorted(alt_counts.items(), key=lambda kv: -kv[1])[:8]
    return {
        'value': consensus if accepted else None,
        'confidence': round(agreement, 4) if accepted else 0.0,
        'needs_review': not accepted,
        'alternatives': [{'national_id':k, 'count':v} for k,v in alternatives],
        **(decoded if accepted else {})
    }

def tight_nid_crop(image, box):
    left, top, right, bottom = map(int, box)

    height, width = image.shape[:2]

    center_y = (top + bottom) // 2
    box_height = bottom - top

    return image[
        max(0, center_y-int(box_height*.75)):
        min(height, center_y+int(box_height*.75)),
        max(0, left-8):
        min(width, right+8)
    ]

def process_image(image):
    # ---------------------------------------------------------
    # 1. Check image quality
    # ---------------------------------------------------------

    image_quality = check_image_quality(image)

    # ---------------------------------------------------------
    # 2. Detect ID card
    # ---------------------------------------------------------

    card_result = card_model(
        image,
        device=DEVICE,
        verbose=False
    )[0]

    if card_result.boxes is None or len(card_result.boxes) == 0:
        raise ValueError("NASO7Y did not detect an ID card.")

    best_card = int(card_result.boxes.conf.argmax())

    x1, y1, x2, y2 = map(
        int,
        card_result.boxes.xyxy[best_card].cpu().numpy()
    )

    card = image[
        max(0, y1):min(image.shape[0], y2),
        max(0, x1):min(image.shape[1], x2)
    ]

    OCR_CARD = enhance_card_for_ocr(card)

    # ---------------------------------------------------------
    # 3. Detect fields
    # ---------------------------------------------------------

    field_result = field_model(
        card,
        device=DEVICE,
        verbose=False
    )[0]

    if field_result.boxes is None or len(field_result.boxes) == 0:
        raise ValueError("NASO7Y did not detect fields.")

    LABELS = {
        'firstname': 'first_name',
        'lastname': 'last_name',
        'address': 'address',
        'serial': 'serial',
        'nid': 'nid'
    }

    field_detections = {}

    for box in field_result.boxes:

        raw_label = str(
            field_result.names[int(box.cls[0])]
        )

        label = LABELS.get(
            raw_label.lower(),
            raw_label
        )

        candidate = {
            'box': box.xyxy[0].cpu().numpy(),
            'confidence': float(box.conf[0]),
            'source_label': raw_label
        }

        if (
            label not in field_detections
            or candidate['confidence']
            > field_detections[label]['confidence']
        ):
            field_detections[label] = candidate

    # ---------------------------------------------------------
    # 4. OCR fields
    # ---------------------------------------------------------

    field_ocr = {}

    for label, detection in field_detections.items():

        crop = (
            narrow_address_crop(
                OCR_CARD,
                detection['box']
            )
            if label == 'address'
            else crop_field(
                OCR_CARD,
                detection['box'],
                8 if label in (
                    'nid',
                    'first_name',
                    'last_name'
                )
                else 4
            )
        )

        field_ocr[label] = ocr_ensemble(crop)

    first_name, first_confidence = choose_field(
        field_ocr.get('first_name', [])
    )

    last_name, last_confidence = choose_field(
        field_ocr.get('last_name', [])
    )

    address, address_confidence = choose_address(
        field_ocr.get('address', [])
    )

    serial, serial_confidence = choose_field(
        field_ocr.get('serial', []),
        serial=True
    )

    # ---------------------------------------------------------
    # 5. National ID extraction
    # ---------------------------------------------------------

    nid_candidates = []

    if 'nid' in field_detections:

        nid_crop = tight_nid_crop(
            OCR_CARD,
            field_detections['nid']['box']
        )

        bad_quality = (
            image_quality['status'] == 'BAD'
            or not image_quality['checks']['glare']
            or not image_quality['checks']['brightness']
        )

        for variant_name, variant in nid_variants(
            nid_crop,
            aggressive=bad_quality
        ).items():

            # YOLO digit model
            digits, confidence = yolo_digit_text(variant)

            add_nid_windows(
                digits,
                'NASO7Y digit YOLO',
                variant_name,
                confidence,
                nid_candidates
            )

            # Tesseract
            tess = pytesseract.image_to_string(
                variant,
                lang='eng',
                config='--oem 3 --psm 7 '
                       '-c tessedit_char_whitelist=0123456789'
            )

            add_nid_windows(
                tess,
                'Tesseract-nid',
                variant_name,
                0.60,
                nid_candidates
            )

        # OCR ensemble
        for item in ocr_ensemble(nid_crop):

            add_nid_windows(
                item['text'],
                item['engine'],
                item['variant'],
                item['confidence'],
                nid_candidates
            )

        # ---------------------------------------------------------
    # 6. Select final NID
    # ---------------------------------------------------------

    nid_result = vote_nid_candidates(nid_candidates)

    full_name = ' '.join(
        value
        for value in [first_name, last_name]
        if value
    ) or None

    address = clean_space(address or '') or None

    address_review = (
        address is None
        or len(address.split()) < 2
        or len(address) < 12
        or bool(set(address.split()) & ADDRESS_STOPWORDS)
    )

    # ---------------------------------------------------------
    # 7. Final result
    # ---------------------------------------------------------

    final_result = {
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
                label: round(
                    item['confidence'],
                    4
                )
                for label, item
                in field_detections.items()
            }
        },

        'needs_review': {
            'national_id': nid_result['needs_review'],
            'full_name': full_name is None,
            'address': address_review,
            'serial_number': serial is None
        },

        'capture_quality': image_quality['status'],
        'card_detection': {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'confidence': float(
                card_result.boxes.conf[best_card].cpu().item()
            )
        },
    }

    return final_result