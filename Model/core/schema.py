"""
Structured result schema with full provenance (Section 34, 47).
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import numpy as np
from . import config


@dataclass
class FieldResult:
    field: str
    raw: str = None
    normalized: str = None
    sources: list = field(default_factory=list)   # e.g. ["printed_ocr"], ["printed_ocr","nid_derived"]
    engine: str = None
    preprocessing_variant: str = None
    bbox: dict = None
    polygon: list = None
    coordinate_space: str = None
    ocr_confidence: float = None
    localization_confidence: float = None
    detection_method: str = "dynamic_anchor"
    validation: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    status: str = config.STATUS_DETECTED
    issues: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class DocumentResult:
    is_egyptian_id: bool = False
    side: str = config.SIDE_UNKNOWN
    side_confidence: float = 0.0
    card_detection_confidence: float = 0.0
    template: str = None
    image_quality: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


def _json_safe(val: Any) -> Any:
    """Recursively convert objects to JSON-serializable primitives."""
    if isinstance(val, np.ndarray):
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, dict):
        return {k: _json_safe(v) for k, v in val.items() if not isinstance(v, np.ndarray)}
    if isinstance(val, (list, tuple)):
        return [_json_safe(x) for x in val if not isinstance(x, np.ndarray)]
    return val


@dataclass
class PipelineResult:
    document: DocumentResult
    fields: dict            # {field_name: FieldResult}
    derived: dict = field(default_factory=dict)
    cross_field_validation: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    canonical_card: Any = None  # np.ndarray or None (kept out of JSON dict)

    def as_dict(self):
        # 1. National ID parsing & derivation
        nid_field = self.fields.get("national_id")
        nid_value = None
        nid_confidence = 0.0
        nid_needs_review = True
        birth_date = None
        gender = None
        governorate = None

        if nid_field:
            nid_value = nid_field.normalized or nid_field.raw or None
            if nid_field.ocr_confidence is not None:
                nid_confidence = round(float(nid_field.ocr_confidence), 4)

            if nid_field.validation:
                v_status = nid_field.validation.get("status")
                nid_needs_review = (v_status != "VALID" or nid_field.status in (config.STATUS_LOW_CONFIDENCE, config.STATUS_FAILED))
                val_derived = nid_field.validation.get("derived", {})
                birth_date = val_derived.get("birth_date")
                gender = val_derived.get("gender")
                governorate = val_derived.get("governorate_name")
            else:
                nid_needs_review = (nid_field.status != config.STATUS_EXTRACTED or not nid_value or len(nid_value) != 14)

        if not birth_date:
            if "birth_date" in self.derived and isinstance(self.derived["birth_date"], dict):
                birth_date = self.derived["birth_date"].get("value")
            elif "nid_derived_dob" in self.derived:
                birth_date = self.derived["nid_derived_dob"]
            elif "date_of_birth" in self.fields and self.fields["date_of_birth"].normalized:
                birth_date = self.fields["date_of_birth"].normalized

        if not gender:
            if "gender" in self.derived:
                if isinstance(self.derived["gender"], dict):
                    gender = self.derived["gender"].get("value") or self.derived["gender"].get("nid_derived")
                else:
                    gender = self.derived["gender"]
            elif "gender" in self.fields and self.fields["gender"].normalized:
                gender = self.fields["gender"].normalized

        if not governorate:
            if "birth_governorate" in self.derived and isinstance(self.derived["birth_governorate"], dict):
                governorate = self.derived["birth_governorate"].get("name")
            elif "birth_governorate" in self.derived and isinstance(self.derived["birth_governorate"], str):
                governorate = self.derived["birth_governorate"]

        nid_result = {
            'value': nid_value,
            'confidence': nid_confidence,
            'birth_date': birth_date,
            'gender': gender,
            'governorate': governorate,
            'needs_review': nid_needs_review,
        }

        # 2. Name parsing
        name_field = self.fields.get("name")
        full_name = None
        first_confidence = 0.0
        last_confidence = 0.0

        if name_field:
            full_name = name_field.normalized or name_field.raw or None
            if name_field.ocr_confidence is not None:
                first_confidence = float(name_field.ocr_confidence)
                last_confidence = float(name_field.ocr_confidence)

        first_name = None
        last_name = None
        if full_name:
            words = [w for w in full_name.split() if w.strip()]
            if words:
                first_name = words[0]
                if len(words) > 1:
                    last_name = " ".join(words[1:])
        elif "first_name" in self.fields:
            fn_field = self.fields["first_name"]
            first_name = fn_field.normalized or fn_field.raw or None
            if fn_field.ocr_confidence is not None:
                first_confidence = float(fn_field.ocr_confidence)
            if "last_name" in self.fields:
                ln_field = self.fields["last_name"]
                last_name = ln_field.normalized or ln_field.raw or None
                if ln_field.ocr_confidence is not None:
                    last_confidence = float(ln_field.ocr_confidence)
            if first_name and last_name:
                full_name = f"{first_name} {last_name}"
            elif first_name:
                full_name = first_name

        # 3. Address parsing
        address_field = self.fields.get("address")
        address = None
        address_confidence = 0.0
        if address_field:
            address = address_field.normalized or address_field.raw or None
            if address_field.ocr_confidence is not None:
                address_confidence = float(address_field.ocr_confidence)
            address_review = (address is None or address_field.status in (config.STATUS_LOW_CONFIDENCE, config.STATUS_FAILED))
        else:
            address_review = True

        # 4. Serial number parsing
        serial_field = self.fields.get("serial_number")
        serial = None
        serial_confidence = 0.0
        if serial_field:
            serial = serial_field.normalized or serial_field.raw or None
            if serial_field.ocr_confidence is not None:
                serial_confidence = float(serial_field.ocr_confidence)

        # 5. Field detections
        field_detections = {}
        if self.debug and isinstance(self.debug, dict) and "localization" in self.debug:
            for label, loc_info in self.debug["localization"].items():
                if isinstance(loc_info, dict):
                    conf = loc_info.get("localization_confidence", 0.0)
                    field_detections[label] = {"confidence": float(conf) if conf is not None else 0.0}

        for label, fr in self.fields.items():
            if label not in field_detections:
                conf = getattr(fr, 'localization_confidence', None)
                if conf is None:
                    conf = getattr(fr, 'ocr_confidence', 0.0)
                field_detections[label] = {"confidence": float(conf) if conf is not None else 0.0}

        # 6. Capture quality
        image_quality = self.document.image_quality if (self.document and self.document.image_quality) else {}
        if "status" not in image_quality:
            issues = image_quality.get("issues", [])
            if len(issues) == 0:
                image_quality_status = "good"
            elif len(issues) == 1:
                image_quality_status = "acceptable"
            else:
                image_quality_status = "poor"
            image_quality = {**image_quality, "status": image_quality_status}

        # 7. Card detection
        card_detection_info = self.debug.get("card_detection", {}) if (self.debug and isinstance(self.debug, dict)) else {}
        card_meta = {
            "detected": bool(self.document.is_egyptian_id) if self.document else False,
            "confidence": round(float(self.document.card_detection_confidence), 4) if self.document else 0.0,
            "side": self.document.side if self.document else config.SIDE_UNKNOWN,
            "side_confidence": round(float(self.document.side_confidence), 4) if self.document else 0.0,
            "corners": card_detection_info.get("corners"),
        }

        # Return exact dictionary matching requested structure
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

    def as_full_dict(self):
        """Full internal diagnostic payload with provenance, bounding boxes, and intermediate steps."""
        return {
            "document": self.document.as_dict() if self.document else {},
            "fields": {k: v.as_dict() if hasattr(v, 'as_dict') else v for k, v in self.fields.items()},
            "derived": _json_safe(self.derived),
            "cross_field_validation": _json_safe(self.cross_field_validation),
            "debug": _json_safe(self.debug),
            "errors": self.errors,
        }
