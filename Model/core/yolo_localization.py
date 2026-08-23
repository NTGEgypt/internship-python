"""
YOLO Segmentation Field Localizer for Egyptian National ID.
Uses a trained YOLO segmentation model to dynamically detect, segment, and crop
ID fields without relying on fixed coordinates or template priors.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from . import config
from .coordinate_systems import BBox
from .image_utils import assess_crop_quality

# Field type mappings
FIELD_TYPES = {
    "national_id": "numeric",
    "name": "arabic_text",
    "date_of_birth": "numeric",
    "gender": "arabic_text",
    "birth_governorate": "arabic_text",
    "profession": "arabic_text",
    "address": "arabic_text",
    "barcode": "text",
    "photo": "image_region",
}


class YOLOSegmentationLocalizer:
    """
    Dynamically localizes Egyptian ID card fields using YOLO segmentation model.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self._model = None

    def _is_ultralytics_name(self) -> bool:
        """Return True if the model path is a named Ultralytics hub model (e.g. yolov8n-seg.pt)."""
        p = Path(self.model_path)
        return p.parent == Path('.') and p.suffix == '.pt'

    def is_available(self) -> bool:
        """Check if YOLO model is available (either on disk or auto-downloadable) and ultralytics is installed."""
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False
        if self._is_ultralytics_name():
            return True
        return os.path.exists(self.model_path)

    def load(self):
        """Lazy load YOLO model. Auto-downloads named Ultralytics models (e.g. yolov8n-seg.pt)."""
        if self._model is None:
            if not self._is_ultralytics_name() and not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"YOLO segmentation model not found at '{self.model_path}'. "
                    "Run 'python training/train_yolo.py' to train the model."
                )
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)

    def localize(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = None,
        multi_scale: bool = True
    ) -> dict:
        """
        Run YOLO segmentation on input image to locate and segment all fields.
        Supports multi-scale inference for detecting micro text and macro blocks.
        Returns dictionary of localized fields with polygon masks, bounding boxes, and crops.
        """
        if conf_threshold is None:
            conf_threshold = config.YOLO_CONF_THRESHOLD

        self.load()
        h, w = image_bgr.shape[:2]

        scales = [1.0, 1.25] if multi_scale and max(h, w) < 2000 else [1.0]
        all_detections = []

        for s in scales:
            if s != 1.0:
                cur_img = cv2.resize(image_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
            else:
                cur_img = image_bgr

            results_list = self._model.predict(
                source=cur_img,
                conf=conf_threshold,
                imgsz=640,
                verbose=False
            )

            if not results_list or len(results_list) == 0:
                continue

            yolo_res = results_list[0]
            boxes = yolo_res.boxes
            masks = yolo_res.masks
            names = yolo_res.names

            if boxes is None or len(boxes) == 0:
                continue

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(float)

                if s != 1.0:
                    xyxy /= s

                poly_pts = None
                if masks is not None and len(masks.xy) > i:
                    poly = masks.xy[i]
                    if len(poly) > 0:
                        if s != 1.0:
                            poly = poly / s
                        poly_pts = poly.astype(int).tolist()

                all_detections.append({
                    "cls_name": cls_name,
                    "conf": conf,
                    "xyxy": xyxy.astype(int),
                    "poly_pts": poly_pts,
                })

        detected_fields = {}
        if not all_detections:
            return detected_fields

        # Select highest-confidence detection per class
        best_per_class = {}
        for det in all_detections:
            cname = det["cls_name"]
            if cname not in best_per_class or det["conf"] > best_per_class[cname]["conf"]:
                best_per_class[cname] = det

        # Build Field Results with Dynamic Spatial Sanity Validation
        for raw_cls_name, det in best_per_class.items():
            x0, y0, x1, y1 = det["xyxy"]
            cy = (y0 + y1) / 2.0 / h
            cx = (x0 + x1) / 2.0 / w

            cls_name = raw_cls_name

            # Skip header decoration (Republic header at the extreme top-right)
            if cx > 0.30 and cy < 0.20:
                continue

            # Expand address boundary from the upper side so top-line text and diacritics are fully captured
            if cls_name == "address":
                pad_top = max(8, int(0.025 * h))
                y0 = max(0, int(y0 - pad_top))

            x0 = max(0, min(w - 1, int(x0)))
            y0 = max(0, min(h - 1, int(y0)))
            x1 = max(x0 + 1, min(w, int(x1)))
            y1 = max(y0 + 1, min(h, int(y1)))

            crop = image_bgr[y0:y1, x0:x1]
            bbox = BBox(x=x0, y=y0, w=(x1 - x0), h=(y1 - y0), coordinate_space=config.SPACE_CANONICAL)

            detected_fields[cls_name] = {
                "bbox": bbox,
                "polygon": det["poly_pts"],
                "crop": crop,
                "crop_quality": assess_crop_quality(crop),
                "localization_confidence": det["conf"],
                "detection_method": "yolo_segmentation",
                "field_type": FIELD_TYPES.get(cls_name, "arabic_text"),
                "required": cls_name in ("name", "national_id", "address", "photo"),
            }

        # Apply strict disjoint single-feature boundary resolution
        from .localization import _resolve_boundary_overlaps
        detected_fields = _resolve_boundary_overlaps(detected_fields, pad=4)

        # Refresh crops
        for fname, fdata in detected_fields.items():
            fb = fdata["bbox"]
            c_crop = image_bgr[fb.y:fb.y + fb.h, fb.x:fb.x + fb.w]
            fdata["crop"] = c_crop
            fdata["crop_quality"] = assess_crop_quality(c_crop)

        return detected_fields


# Global singleton instance
_yolo_localizer_instance = None


def get_yolo_localizer(model_path: str = None) -> YOLOSegmentationLocalizer:
    global _yolo_localizer_instance
    if _yolo_localizer_instance is None or (model_path and _yolo_localizer_instance.model_path != model_path):
        _yolo_localizer_instance = YOLOSegmentationLocalizer(model_path=model_path)
    return _yolo_localizer_instance


def localize_fields_yolo(
    image_bgr: np.ndarray,
    model_path: str = None,
    conf_threshold: float = None,
    multi_scale: bool = True,
) -> dict:
    """
    Functional helper to localize Egyptian ID fields using YOLO segmentation.
    """
    localizer = get_yolo_localizer(model_path=model_path)
    return localizer.localize(image_bgr, conf_threshold=conf_threshold, multi_scale=multi_scale)


def visualize_field_detections(
    image_bgr: np.ndarray,
    fields_dict: dict,
    card_corners: np.ndarray = None
) -> np.ndarray:
    """
    Model-only visualization debugger: renders detected card bounds, field bounding boxes,
    segmentation polygons, and confidence scores onto an annotated canvas.
    """
    vis = image_bgr.copy()
    colors = {
        "photo": (0, 165, 255),          # Orange
        "name": (0, 255, 0),             # Green
        "address": (255, 0, 255),        # Magenta
        "national_id": (255, 255, 0),    # Cyan/Yellow
        "date_of_birth": (0, 255, 255),  # Yellow
        "serial_number": (255, 128, 0),  # Blue-cyan
        "barcode": (255, 128, 0),
        "gender": (180, 105, 255),
        "profession": (200, 200, 0),
    }

    # Draw card quad if provided
    if card_corners is not None:
        pts = np.array(card_corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    # Draw field bounding boxes and labels
    for fname, fdata in fields_dict.items():
        bbox = fdata.get("bbox")
        if bbox is None:
            continue
        col = colors.get(fname, (0, 255, 0))
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)

        poly = fdata.get("polygon")
        if poly and len(poly) >= 3:
            poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly_np], col)
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        conf = fdata.get("localization_confidence", 0.0)
        label = f"{fname} ({conf:.2f})"
        cv2.putText(vis, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    return vis

