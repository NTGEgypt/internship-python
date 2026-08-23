"""
Global configuration and constants.
Nothing here should be treated as ground truth about the *content* of a
specific card (no hard-coded per-image pixel coordinates). It defines the
canonical processing space and shared thresholds only.
"""

import os

# --- Canonical card space -----------------------------------------------
# Egyptian ID physical card is ID-1 format: 85.60mm x 53.98mm (ratio ~1.5857)
CANONICAL_CARD_WIDTH = 1600
CANONICAL_CARD_HEIGHT = 1009  # int(round(1600 / (85.60 / 53.98)))

# --- Processing canvas (pre-detection normalization) ---------------------
PROCESSING_MAX_DIM = 1600  # longest side after aspect-preserving resize

# --- Quality thresholds ---------------------------------------------------
MIN_CARD_AREA_RATIO = 0.05       # card must occupy at least 5% of processing canvas
BLUR_LAPLACIAN_MIN = 60.0        # below this -> flagged as blurry
MIN_FIELD_CROP_HEIGHT_PX = 20    # crops shorter than this get upscaled

# --- Confidence thresholds ------------------------------------------------
LOW_CONFIDENCE_OCR = 0.55
LOW_CONFIDENCE_LOCALIZATION = 0.5

# --- Coordinate space identifiers -----------------------------------------
SPACE_ORIGINAL = "original_image"
SPACE_PROCESSING = "processing_canvas"
SPACE_CANONICAL = "canonical_card"
SPACE_FIELD_CROP = "field_crop"

# --- Sides -----------------------------------------------------------------
SIDE_FRONT = "front"
SIDE_BACK = "back"
SIDE_UNKNOWN = "unknown"

# --- Status vocabulary (Section 52 distinction) ---------------------------
STATUS_DETECTED = "DETECTED"           # text found, not yet parsed
STATUS_EXTRACTED = "EXTRACTED"         # parsed into a field value, not verified
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
STATUS_VALIDATED = "VALIDATED"         # passed internal structural validation
STATUS_CROSS_VALIDATED = "CROSS_VALIDATED"  # agrees with an independent source
STATUS_VERIFIED = "VERIFIED"           # validated + cross-checked, strongest evidence
STATUS_MISMATCH = "MISMATCH"
STATUS_FAILED = "FAILED"

GOVERNORATE_CODES = {
    "01": "القاهرة",
    "02": "الإسكندرية",
    "03": "بورسعيد",
    "04": "السويس",
    "11": "دمياط",
    "12": "الدقهلية",
    "13": "الشرقية",
    "14": "القليوبية",
    "15": "كفر الشيخ",
    "16": "الغربية",
    "17": "المنوفية",
    "18": "البحيرة",
    "19": "الإسماعيلية",
    "21": "الجيزة",
    "22": "بني سويف",
    "23": "الفيوم",
    "24": "المنيا",
    "25": "أسيوط",
    "26": "سوهاج",
    "27": "قنا",
    "28": "أسوان",
    "29": "الأقصر",
    "31": "البحر الأحمر",
    "32": "الوادي الجديد",
    "33": "مطروح",
    "34": "شمال سيناء",
    "35": "جنوب سيناء",
    "88": "خارج مصر (مواليد خارج جمهورية مصر العربية)",
}

GOVERNORATE_CODES_EN = {
    "01": "Cairo",
    "02": "Alexandria",
    "03": "Port Said",
    "04": "Suez",
    "11": "Damietta",
    "12": "Dakahlia",
    "13": "Ash Sharqia",
    "14": "Kaliobeya",
    "15": "Kafr El Sheikh",
    "16": "Gharbia",
    "17": "Monoufia",
    "18": "Beheira",
    "19": "Ismailia",
    "21": "Giza",
    "22": "Beni Suef",
    "23": "Fayoum",
    "24": "Minya",
    "25": "Asyut",
    "26": "Sohag",
    "27": "Qena",
    "28": "Aswan",
    "29": "Luxor",
    "31": "Red Sea",
    "32": "New Valley",
    "33": "Matrouh",
    "34": "North Sinai",
    "35": "South Sinai",
    "88": "Born Abroad",
}

# --- Localization Modes ---
LOCALIZATION_MODE_ANCHORS = "dynamic_anchors"
LOCALIZATION_MODE_YOLO = "yolo_segmentation"

# --- YOLO Segmentation Model Configuration ---
_POSSIBLE_YOLO_PATHS = [
    "models/yolo_nid_seg.pt",
    "egyptian_id_ocr/models/yolo_nid_seg.pt",
    "egyptian_id_ocr/runs/segment/runs/segment/egyptian_nid_seg/weights/best.pt",
    "runs/segment/runs/segment/egyptian_nid_seg/weights/best.pt",
]

YOLO_MODEL_PATH = next((p for p in _POSSIBLE_YOLO_PATHS if os.path.exists(p)), "yolov8n-seg.pt")
YOLO_CONF_THRESHOLD = 0.15
YOLO_CLASS_NAMES = [
    "national_id",
    "name",
    "date_of_birth",
    "gender",
    "birth_governorate",
    "profession",
    "address",
    "barcode",
    "photo",
]
