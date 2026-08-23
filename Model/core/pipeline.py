"""
Pipeline orchestrator (Section 3, 50). Ties every stage together and
returns a schema.PipelineResult with full debug info attached (Section 40).

This module is intentionally defensive: a failure in one field must not
crash the whole run (Section 48). OCR-dependent stages degrade gracefully
if PaddleOCR isn't installed, and still return the geometry/detection/
validation results that don't depend on it (Section 49 -- geometry and
NID math are independently demonstrable even without a live OCR engine).
"""
import numpy as np

from . import config
from .image_utils import normalize_image, assess_image_quality
from .card_detection import detect_and_rectify, crop_canonical_to_card
from .side_classifier import classify_side
from .templates import get_template_for_side
from .localization import localize_fields
from .preprocessing import preprocess_field
from . import ocr_engine
from .arabic_normalize import normalize_arabic_text, normalize_numeric_field
from . import field_parsers
from .nid_validator import validate_nid
from .cross_field_validator import run_cross_field_validation
from . import independent_verifier
from .barcode import decode_barcode
from .schema import DocumentResult, FieldResult, PipelineResult


def run_pipeline(
    image_bgr: np.ndarray,
    run_ocr: bool = True,
    force_side: str = None,
    localization_mode: str = config.LOCALIZATION_MODE_ANCHORS,
    yolo_model_path: str = None
) -> PipelineResult:
    debug = {}
    errors = []

    # --- 1. quality gate on raw upload -----------------------------------
    raw_quality = assess_image_quality(image_bgr)
    debug["raw_image_quality"] = raw_quality

    # --- 2. normalize (Section 7) -----------------------------------------
    processing_img, transform, original_img = normalize_image(image_bgr)
    debug["normalization_transform"] = transform.as_dict()

    # --- 3. card detection + rectification (Sections 9-10) -----------------
    detection = detect_and_rectify(processing_img)
    debug["card_detection"] = {
        "corners": detection.get("corners"),
        "confidence": detection.get("confidence", 0.0),
        "debug": detection.get("debug", {}),
    }
    # --- 3. canonical card extraction ---------------------------------------
    canonical_card = detection.get("canonical_card")

    # --- 4. side classification (Section 13) --------------------------------
    document = DocumentResult(
        is_egyptian_id=detection["confidence"] >= 0.5,
        card_detection_confidence=detection["confidence"],
        image_quality=raw_quality,
    )

    if detection["canonical_card"] is None:
        errors.append("Card not detected: no plausible quadrilateral found in the image.")
        return PipelineResult(document=document, fields={}, debug=debug, errors=errors)

    if force_side in (config.SIDE_FRONT, config.SIDE_BACK):
        document.side = force_side
        document.side_confidence = 1.0
        side_info = {"side": force_side, "confidence": 1.0, "method": "manual_override"}
    else:
        side_info = classify_side(canonical_card)
        document.side = side_info["side"]
        document.side_confidence = side_info["confidence"]
    debug["side_classification"] = side_info

    if document.side == config.SIDE_UNKNOWN:
        errors.append("Could not confidently classify front vs back; skipping field localization. You can manually specify the card side in settings.")
        debug["canonical_card_available"] = True
        return PipelineResult(document=document, fields={}, debug=debug, errors=errors, canonical_card=canonical_card)

    template = get_template_for_side(document.side)
    document.template = template.template_id

    # --- 5. dynamic field localization + crops (Sections 16, 19) ------------
    localized = {}
    if localization_mode == config.LOCALIZATION_MODE_YOLO:
        try:
            from .yolo_localization import localize_fields_yolo
            yolo_localized = localize_fields_yolo(canonical_card, model_path=yolo_model_path)
            # Baseline dynamic anchors to guarantee complete field coverage
            anchor_localized = localize_fields(canonical_card, template)
            localized = {**anchor_localized, **yolo_localized}
            debug["localization_mode"] = "yolo_segmentation"
        except Exception as e:
            errors.append(f"YOLO segmentation localization note: {e}. Falling back to dynamic anchors.")
            localized = localize_fields(canonical_card, template)
            debug["localization_mode"] = "dynamic_anchors_fallback"
    else:
        localized = localize_fields(canonical_card, template)
        debug["localization_mode"] = "dynamic_anchors"

    debug["localization"] = {
        name: {
            "bbox": info["bbox"].as_dict() if info.get("bbox") else None,
            "polygon": info.get("polygon"),
            "crop_quality": info.get("crop_quality"),
            "localization_confidence": info.get("localization_confidence"),
            "detection_method": info.get("detection_method", "dynamic_anchor"),
        }
        for name, info in localized.items()
    }

    fields = {}
    ocr_candidates_debug = {}

    ocr_ready = run_ocr and ocr_engine.is_available()
    if run_ocr and not ocr_engine.is_available():
        errors.append(
            "OCR requested but PaddleOCR is not installed in this environment "
            f"({ocr_engine.import_error()}); returning geometry/localization only."
        )

    for field_name, info in localized.items():
        crop = info.get("crop")
        field_type = info.get("field_type", "arabic_text")
        fr = FieldResult(
            field=field_name,
            bbox=info["bbox"].as_dict() if info.get("bbox") else None,
            polygon=info.get("polygon"),
            coordinate_space=info["bbox"].coordinate_space if info.get("bbox") else None,
            localization_confidence=info.get("localization_confidence", 0.0),
            detection_method=info.get("detection_method", "dynamic_anchor"),
            status=config.STATUS_DETECTED,
        )

        if info.get("crop_quality") and not info["crop_quality"]["acceptable"]:
            fr.issues.extend(info["crop_quality"]["issues"])

        if crop is None or crop.size == 0:
            fr.status = config.STATUS_FAILED
            fr.issues.append("empty_crop")
            fields[field_name] = fr
            continue

        if field_type == "barcode":
            bc = decode_barcode(crop)
            fr.raw = bc.get("payload")
            fr.normalized = bc.get("payload")
            fr.status = config.STATUS_EXTRACTED if bc["status"] == "barcode_decoded" else config.STATUS_DETECTED
            fr.validation = bc
            fields[field_name] = fr
            continue

        if field_type == "image_region":
            fr.status = config.STATUS_DETECTED
            fr.issues.append("not_an_ocr_field")
            fields[field_name] = fr
            continue

        variants = preprocess_field(field_name, crop, field_type)

        if not ocr_ready:
            fr.status = config.STATUS_FAILED
            fr.issues.append("ocr_unavailable")
            fields[field_name] = fr
            continue

        variant_results = ocr_engine.run_ocr_multi_variant(variants, lang="ar")
        ocr_candidates_debug[field_name] = {
            v: [c.__dict__ for c in cands] for v, cands in variant_results.items()
        }

        # Smart ranking: score candidate confidence plus structural suitability
        best_candidate = None
        best_score = -1.0
        for variant_name, cands in variant_results.items():
            for c in cands:
                score = float(c.confidence)
                if field_name == "national_id":
                    digits = normalize_numeric_field(c.text)
                    if len(digits) == 14 and digits[0] in ("2", "3"):
                        score += 0.5
                        val = validate_nid(digits)
                        if val.status == "VALID":
                            score += 1.0
                    elif len(digits) >= 10:
                        score += 0.2
                elif field_name == "date_of_birth":
                    dob_p = field_parsers.parse_dob_printed(c.text)
                    if dob_p:
                        score += 0.6
                    digits = normalize_numeric_field(c.text)
                    if len(digits) == 8:
                        score += 0.3
                elif field_name == "serial_number":
                    ser_p = field_parsers.parse_serial_number(c.text)
                    if ser_p:
                        score += 0.8
                elif field_type == "arabic_text":
                    words = [w for w in c.text.strip().split() if len(w) > 1]
                    if len(words) >= 2:
                        score += 0.2
                    if len(words) >= 3:
                        score += 0.1
                    if "multiline" in c.engine or "segmented" in c.engine:
                        score += 0.25

                if score > best_score:
                    best_score = score
                    best_candidate = c

        if best_candidate is None:
            fr.status = config.STATUS_FAILED
            fr.issues.append("no_text_detected")
            fields[field_name] = fr
            continue

        is_numeric = field_type == "numeric"
        fr.raw = best_candidate.text
        if field_name == "national_id":
            fr.normalized = field_parsers.parse_national_id(best_candidate.text)
        elif field_name == "date_of_birth":
            dob_parsed = field_parsers.parse_dob_printed(best_candidate.text)
            fr.normalized = dob_parsed if dob_parsed else normalize_numeric_field(best_candidate.text)
        elif is_numeric:
            fr.normalized = normalize_numeric_field(best_candidate.text)
        elif field_name == "serial_number":
            fr.normalized = field_parsers.parse_serial_number(best_candidate.text)
        elif field_name == "name":
            fr.normalized = field_parsers.parse_name(best_candidate.text)
        elif field_name == "address":
            fr.normalized = field_parsers.parse_address([best_candidate.text])
        else:
            fr.normalized = normalize_arabic_text(best_candidate.text)
        fr.engine = best_candidate.engine
        fr.preprocessing_variant = best_candidate.variant
        fr.ocr_confidence = best_candidate.confidence
        fr.sources = ["printed_ocr"]
        fr.status = (config.STATUS_LOW_CONFIDENCE if best_candidate.confidence < config.LOW_CONFIDENCE_OCR
                     else config.STATUS_EXTRACTED)

        fields[field_name] = fr

    debug["ocr_candidates"] = ocr_candidates_debug

    # Cross-recover National ID if leaked into address raw OCR
    if "address" in fields and fields["address"].raw:
        addr_raw = fields["address"].raw
        from .arabic_normalize import digits_to_western
        import re
        m_nid = re.search(r'([23][0-9]{13})', digits_to_western(addr_raw))
        if m_nid and ("national_id" not in fields or not fields["national_id"].normalized or len(fields["national_id"].normalized) != 14):
            recovered_nid = m_nid.group(1)
            if "national_id" in fields:
                fields["national_id"].raw = recovered_nid
                fields["national_id"].normalized = recovered_nid
                fields["national_id"].status = config.STATUS_EXTRACTED
                fields["national_id"].ocr_confidence = 0.95

    # --- 6. NID validation (Sections 24-25) ---------------------------------
    derived = {}
    nid_field = fields.get("national_id")
    nid_validation_dict = None
    if nid_field and nid_field.normalized:
        nid_result = validate_nid(nid_field.normalized)
        nid_validation_dict = nid_result.as_dict()
        nid_field.validation = nid_validation_dict
        nid_field.verification = independent_verifier.verify_national_id(nid_validation_dict)
        nid_field.status = nid_field.verification["status"]
        derived["birth_governorate"] = {
            "code": nid_validation_dict["derived"]["governorate_code"],
            "name": nid_validation_dict["derived"]["governorate_name"],
        }
        derived["gender"] = {"nid_derived": nid_validation_dict["derived"]["gender"]}
        derived["nid_derived_dob"] = nid_validation_dict["derived"]["birth_date"]

    # Cross-recover serial number if captured in DOB raw OCR
    if "date_of_birth" in fields and fields["date_of_birth"].raw:
        serial_in_dob = field_parsers.parse_serial_number(fields["date_of_birth"].raw)
        if serial_in_dob:
            if "serial_number" in fields:
                if not fields["serial_number"].normalized or fields["serial_number"].status in (config.STATUS_LOW_CONFIDENCE, config.STATUS_FAILED):
                    fields["serial_number"].raw = serial_in_dob
                    fields["serial_number"].normalized = serial_in_dob
                    fields["serial_number"].status = config.STATUS_EXTRACTED
                    fields["serial_number"].ocr_confidence = 0.95

    # --- 7. cross-field validation (Section 35) ------------------------------
    cross_extracted = {}
    if "date_of_birth" in fields and fields["date_of_birth"].raw:
        parsed_dob = field_parsers.parse_dob_printed(fields["date_of_birth"].raw)
        if parsed_dob:
            cross_extracted["date_of_birth_printed"] = parsed_dob
            if nid_validation_dict and nid_validation_dict["derived"].get("birth_date"):
                dob_verify = independent_verifier.verify_dob(
                    parsed_dob, nid_validation_dict["derived"]["birth_date"]
                )
                fields["date_of_birth"].verification = dob_verify
                fields["date_of_birth"].status = dob_verify["status"]

    if "gender" in fields and fields["gender"].raw:
        parsed_gender = field_parsers.parse_gender(fields["gender"].raw)
        if parsed_gender:
            cross_extracted["gender_printed"] = parsed_gender
            if nid_validation_dict and nid_validation_dict["derived"].get("gender"):
                gender_verify = independent_verifier.verify_gender(
                    parsed_gender, nid_validation_dict["derived"]["gender"]
                )
                fields["gender"].verification = gender_verify
                fields["gender"].status = gender_verify["status"]

    if "address" in fields and fields["address"].raw:
        addr_text = fields["address"].raw
        for gcode, gname in config.GOVERNORATE_CODES.items():
            if gname in addr_text:
                cross_extracted["birth_governorate_printed"] = gname
                break

    cross_field_result = {}
    if nid_validation_dict:
        cross_field_result = run_cross_field_validation(cross_extracted, nid_validation_dict)

    return PipelineResult(
        document=document,
        fields=fields,
        derived=derived,
        cross_field_validation=cross_field_result,
        debug=debug,
        errors=errors,
        canonical_card=canonical_card,
    )
