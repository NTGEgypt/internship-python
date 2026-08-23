"""
OCR engine wrapper (Sections 5, 6, 42).

PaddleOCR is the primary baseline. Models are loaded once and cached
(module-level singleton) so Streamlit reruns don't reinstantiate them.

IMPORTANT: this module is written against the PaddleOCR API but is
defensive about import/availability -- if PaddleOCR/PaddlePaddle are not
installed in the current environment, `is_available()` returns False and
callers get a clear OCRUnavailableError instead of a crash, per Section 48
(never crash the whole app because one component is missing) combined with
Section 1 (never silently fake success).
"""
from dataclasses import dataclass
import numpy as np
import cv2

_PADDLE_IMPORT_ERROR = None
try:
    import paddleocr
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except Exception as e:  # pragma: no cover - environment dependent
    _PADDLE_AVAILABLE = False
    _PADDLE_IMPORT_ERROR = e


class OCRUnavailableError(RuntimeError):
    pass


@dataclass
class OCRCandidate:
    text: str
    confidence: float
    bbox: list  # polygon in the *input crop's* pixel coords
    engine: str
    variant: str


_ENGINE_CACHE = {}


def is_available() -> bool:
    return _PADDLE_AVAILABLE


def import_error() -> str:
    return str(_PADDLE_IMPORT_ERROR) if _PADDLE_IMPORT_ERROR else ""


def get_engine(lang: str = "ar", use_gpu: bool = None):
    """
    Singleton OCR recognition engine per language.
    """
    if not _PADDLE_AVAILABLE:
        raise OCRUnavailableError(
            "PaddleOCR is not installed in this environment. "
            f"Import error: {_PADDLE_IMPORT_ERROR}. "
            "Install paddlepaddle + paddleocr to enable OCR."
        )

    if use_gpu is None:
        use_gpu = _detect_gpu()

    cache_key = (lang, use_gpu)
    if cache_key not in _ENGINE_CACHE:
        engine = None
        last_err = None

        # First, try dedicated TextRecognition if available (fast, robust for cropped fields)
        if hasattr(paddleocr, "TextRecognition"):
            model_names = []
            if lang in ("ar", "arabic"):
                model_names = ["arabic_PP-OCRv5_mobile_rec", "arabic_PP-OCRv3_rec", "PP-OCRv5_mobile_rec"]
            else:
                model_names = ["PP-OCRv5_mobile_rec", "PP-OCRv4_mobile_rec"]

            for mname in model_names:
                try:
                    engine = paddleocr.TextRecognition(model_name=mname)
                    break
                except Exception as e:
                    last_err = e
                    continue

        # If not loaded, try PaddleOCR standard pipeline with various kwarg fallbacks
        if engine is None:
            attempts = [
                dict(lang=lang),
                dict(lang=lang, use_gpu=use_gpu),
                dict(lang=lang, use_angle_cls=True, use_gpu=use_gpu),
            ]
            for kwargs in attempts:
                try:
                    engine = PaddleOCR(**kwargs)
                    break
                except Exception as e:
                    last_err = e
                    continue

        if engine is None:
            raise OCRUnavailableError(f"Could not initialize PaddleOCR: {last_err}")
        _ENGINE_CACHE[cache_key] = engine
    return _ENGINE_CACHE[cache_key]


def _detect_gpu() -> bool:
    try:
        import paddle
        return bool(paddle.device.cuda.device_count())
    except Exception:
        return False


def _predict_single_crop(engine, img_bgr: np.ndarray) -> tuple:
    """Predict text and confidence on a single text line crop."""
    try:
        raw = list(engine.predict(img_bgr))
        if raw:
            item = raw[0]
            if isinstance(item, dict):
                text = item.get("rec_text", "")
                conf = float(item.get("rec_score", 0.0))
            else:
                text = getattr(item, "rec_text", "")
                conf = float(getattr(item, "rec_score", 0.0))
            return (text.strip() if text else ""), conf
    except Exception:
        pass
    return "", 0.0


def _find_horizontal_valley(gray_img: np.ndarray) -> int:
    """Find horizontal valley between 2 lines of text."""
    h = gray_img.shape[0]
    if h < 40:
        return h // 2
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8)
    row_sum = np.sum(thresh > 0, axis=1)

    mid_start, mid_end = int(0.30 * h), int(0.70 * h)
    if mid_end > mid_start:
        valley_y = mid_start + int(np.argmin(row_sum[mid_start:mid_end]))
        if 15 <= valley_y <= (h - 15):
            return valley_y
    return h // 2


def run_ocr(image: np.ndarray, lang: str = "ar", variant: str = "default") -> list:
    """
    Run OCR on a single preprocessed image (grayscale or BGR).
    Returns a list of OCRCandidate.
    """
    if not _PADDLE_AVAILABLE:
        raise OCRUnavailableError(
            "PaddleOCR is not installed; cannot run OCR. "
            f"Import error: {_PADDLE_IMPORT_ERROR}"
        )
    if image is None or image.size == 0:
        return []

    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image
    else:
        image_bgr = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image_bgr.shape[:2]
    engine = get_engine(lang=lang)
    candidates = []

    # 1. Full 2D Detection + Recognition (PaddleOCR standard pipeline)
    # Correctly detects 2 separate text lines (e.g. Line 1: "ملك", Line 2: "حمدى محمود محمد شاهين")
    if hasattr(engine, "ocr"):
        try:
            raw = engine.ocr(image_bgr, cls=True)
            if raw and raw[0] is not None:
                detected_lines = []
                for line in raw[0]:
                    poly, (text, conf) = line
                    if text and text.strip():
                        poly_y = np.mean([pt[1] for pt in poly])
                        poly_x = np.mean([pt[0] for pt in poly])
                        detected_lines.append((poly_y, poly_x, text.strip(), float(conf), poly))

                if detected_lines:
                    # Sort primarily top-to-bottom (Y), then right-to-left for Arabic within same line
                    detected_lines.sort(key=lambda item: (item[0] // 20, -item[1]))

                    # A. Multi-line combined candidate (crucial for 2-line name & multi-line address)
                    if len(detected_lines) >= 2:
                        combined_text = " ".join([item[2] for item in detected_lines])
                        avg_conf = float(np.mean([item[3] for item in detected_lines]))
                        candidates.append(OCRCandidate(
                            text=combined_text,
                            confidence=avg_conf,
                            bbox=[[0, 0], [w, 0], [w, h], [0, h]],
                            engine="paddleocr_multiline",
                            variant=variant,
                        ))

                    # B. Individual line candidates
                    for _, _, text, conf, poly in detected_lines:
                        candidates.append(OCRCandidate(
                            text=text, confidence=conf, bbox=poly,
                            engine="paddleocr", variant=variant,
                        ))
        except Exception:
            pass

    if candidates:
        return candidates

    # 2. TextRecognition / Single Crop Predict Fallback
    if hasattr(engine, "predict"):
        # Multi-line handling for tall text crops (e.g. 2-line name or address)
        if h >= 36 and (w / max(h, 1)) < 8.0:
            valley_y = _find_horizontal_valley(gray)
            line1_img = image_bgr[0:valley_y, :]
            line2_img = image_bgr[valley_y:h, :]

            t1, c1 = _predict_single_crop(engine, line1_img)
            t2, c2 = _predict_single_crop(engine, line2_img)

            if t1 and t2:
                candidates.append(OCRCandidate(
                    text=f"{t1} {t2}", confidence=(c1 + c2) / 2.0,
                    bbox=[[0, 0], [w, 0], [w, h], [0, h]],
                    engine="paddleocr_segmented", variant=variant,
                ))

        try:
            raw_pred = list(engine.predict(image_bgr))
            for item in raw_pred:
                if isinstance(item, dict):
                    if "rec_text" in item and item["rec_text"]:
                        candidates.append(OCRCandidate(
                            text=item["rec_text"].strip(),
                            confidence=float(item.get("rec_score", 0.0)),
                            bbox=[[0, 0], [w, 0], [w, h], [0, h]],
                            engine="paddleocr", variant=variant,
                        ))
                    elif "rec_texts" in item and item["rec_texts"]:
                        for t, s in zip(item["rec_texts"], item.get("rec_scores", [1.0] * len(item["rec_texts"]))):
                            if t and t.strip():
                                candidates.append(OCRCandidate(
                                    text=t.strip(), confidence=float(s),
                                    bbox=[[0, 0], [w, 0], [w, h], [0, h]],
                                    engine="paddleocr", variant=variant,
                                ))
        except Exception:
            pass

    return candidates


def run_ocr_multi_variant(variants: dict, lang: str = "ar") -> dict:
    """variants: {variant_name: image}. Returns {variant_name: [OCRCandidate,...]}"""
    results = {}
    for name, img in variants.items():
        try:
            results[name] = run_ocr(img, lang=lang, variant=name)
        except OCRUnavailableError:
            results[name] = []
    return results
