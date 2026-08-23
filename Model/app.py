"""
Egyptian National ID — Arabic-First Extraction, Dynamic Localization & Verification Dashboard.
"""
import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import pandas as pd

# Suppress torch.classes watcher RuntimeError (harmless Streamlit+torch compat issue)
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

from core import config
from core.pipeline import run_pipeline
from core import ocr_engine

st.set_page_config(
    page_title="Egyptian National ID — Extraction & Verification",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design with #38bdf8 accents & dark/light theme compatibility
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        border: 1px solid rgba(56, 189, 248, 0.25);
        background: rgba(56, 189, 248, 0.04);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge-verified {
        background-color: #059669;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-review {
        background-color: #dc2626;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🪪 Egyptian National ID — Extraction & Verification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Arabic-First OCR, Dual-Engine Localization, Geometric Rectification & Cross-Field Consistency</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration & Controls")

side_override_opt = st.sidebar.selectbox(
    "Card Side Selection",
    options=["Auto-detect", "Front (الوجه الأمامي)", "Back (الوجه الخلفي)"],
    index=0,
    help="Automatically classify the card side or force a specific side."
)
force_side = None
if "Front" in side_override_opt:
    force_side = config.SIDE_FRONT
elif "Back" in side_override_opt:
    force_side = config.SIDE_BACK

enable_ocr = st.sidebar.toggle("Run OCR Engine", value=True)

loc_mode_opt = st.sidebar.radio(
    "Field Localization Engine",
    options=["Dynamic Anchors (Default)", "YOLO Segmentation (AI Model)"],
    index=0,
    help="Dynamic Anchors uses semantic visual landmarks & ink projection. YOLO Segmentation uses deep learning polygon segmentation."
)
localization_mode = config.LOCALIZATION_MODE_YOLO if "YOLO" in loc_mode_opt else config.LOCALIZATION_MODE_ANCHORS

if not ocr_engine.is_available():
    st.sidebar.warning(
        f"⚠️ PaddleOCR initialization warning: {ocr_engine.import_error()}"
    )
else:
    st.sidebar.success("✅ PaddleOCR Engine Ready")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Demo & Testing")
use_demo = st.sidebar.button("🖼️ Load Sample Egyptian ID Demo")

uploaded = st.file_uploader("Upload photo or scan of Egyptian ID (Front or Back)", type=["jpg", "jpeg", "png"])

image_bgr = None

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
elif use_demo:
    # Generate clean demo card
    sample = np.full((630, 1000, 3), 245, dtype=np.uint8)
    cv2.rectangle(sample, (30, 30), (970, 600), (220, 225, 230), -1)
    cv2.rectangle(sample, (50, 150), (280, 560), (160, 180, 200), -1)
    cv2.putText(sample, "PHOTO", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
    cv2.putText(sample, "Ahmed Mohamed Ali", (320, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(sample, "29501011234567", (320, 530), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    image_bgr = sample

if image_bgr is not None:
    with st.spinner("Processing card through Egyptian ID extraction pipeline..."):
        result = run_pipeline(
            image_bgr,
            run_ocr=enable_ocr,
            force_side=force_side,
            localization_mode=localization_mode
        )

    output_dict = result.as_dict()
    doc = result.document

    # --- 1. Card Detection & Side Classification Metrics ---
    st.subheader("1. Card Detection & Quality Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Card Detected", "✅ Yes" if output_dict['card_detection']['detected'] else "❌ No")
    m2.metric("Detection Confidence", f"{output_dict['card_detection']['confidence']:.1%}")
    m3.metric("Card Side", f"{output_dict['card_detection']['side'].upper()} ({output_dict['card_detection']['side_confidence']:.1%})")
    m4.metric("Capture Quality", output_dict['capture_quality'].upper())

    if result.errors:
        for e in result.errors:
            st.error(f"⚠️ {e}")

    # --- 2. Standardized Output JSON & Download ---
    st.subheader("2. Standardized Output JSON")
    json_str = json.dumps(output_dict, ensure_ascii=False, indent=2)
    st.json(output_dict)
    st.download_button(
        label="📥 Download JSON Structured Output",
        data=json_str,
        file_name="egyptian_id_extraction.json",
        mime="application/json",
    )

    # --- 3. Extracted Fields Summary & Dataframe ---
    st.subheader("3. Extracted Fields Dataframe")
    if result.fields:
        table_rows = []
        for name, fr in result.fields.items():
            table_rows.append({
                "Field": name,
                "Value (Normalized)": fr.normalized or fr.raw or "-",
                "OCR Conf": f"{fr.ocr_confidence:.2%}" if fr.ocr_confidence is not None else "-",
                "Localization Conf": f"{fr.localization_confidence:.2%}" if fr.localization_confidence is not None else "-",
                "Status": fr.status,
                "Method": getattr(fr, 'detection_method', 'dynamic_anchor'),
                "Issues": ", ".join(fr.issues) if fr.issues else "-",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.info("No field features extracted.")

    # --- 4. Decoded Identity Data & Cross-Field Consistency Tree ---
    st.subheader("4. Decoded Identity & Cross-Field Consistency")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Identity Fields**")
        st.write(f"- **National ID:** `{output_dict['national_id'] or 'Not detected'}`")
        st.write(f"- **Full Name:** `{output_dict['full_name'] or 'Not detected'}`")
        st.write(f"- **First Name:** `{output_dict['first_name'] or '-'}`")
        st.write(f"- **Last Name:** `{output_dict['last_name'] or '-'}`")
        st.write(f"- **Address:** `{output_dict['address'] or '-'}`")
        st.write(f"- **Serial Number:** `{output_dict['serial_number'] or '-'}`")

    with c2:
        st.markdown("**Derived Demographics (from 14-digit NID)**")
        st.write(f"- **Birth Date:** `{output_dict['birth_date'] or '-'}`")
        st.write(f"- **Gender:** `{output_dict['gender'] or '-'}`")
        st.write(f"- **Governorate:** `{output_dict['governorate'] or '-'}`")

        if result.cross_field_validation:
            cfv = result.cross_field_validation
            checks = cfv.get("checks", [])
            if checks:
                st.markdown("**Cross-Field Checks**")
                for c in checks:
                    result_val = c.get("result", "")
                    check_name = c.get("check", "").replace("_", " ").title()
                    if result_val == "MATCH":
                        st.write(f"- **{check_name}:** ✅ Consistent")
                    elif result_val == "MISMATCH":
                        st.write(f"- **{check_name}:** ❌ Mismatch")
                    else:
                        st.write(f"- **{check_name}:** ⚠️ Insufficient Evidence")

    # --- 5. Visual Debugging Sequence Tabs ---
    st.subheader("5. Visual Debug Sequence")
    tabs = st.tabs([
        "📸 Original Upload",
        "📐 Canonical Card (1600 × 1009)",
        "🎯 Field BBoxes & Polygons",
        "🔍 Individual Field Crops",
        "⚙️ Full Diagnostic Metadata"
    ])

    with tabs[0]:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original Input Image")

    canonical_card = result.canonical_card
    if canonical_card is None and doc.is_egyptian_id:
        from core.image_utils import normalize_image
        from core.card_detection import detect_and_rectify
        proc_img, _, _ = normalize_image(image_bgr)
        det = detect_and_rectify(proc_img)
        canonical_card = det.get("canonical_card")

    with tabs[1]:
        if canonical_card is not None:
            st.image(cv2.cvtColor(canonical_card, cv2.COLOR_BGR2RGB), caption="Canonical Rectified ID-1 Card (1600 × 1009 px)")
        else:
            st.info("No canonical card available (detection failed).")

    with tabs[2]:
        if canonical_card is not None and result.fields:
            annotated = canonical_card.copy()
            for name, fr in result.fields.items():
                if fr.bbox and isinstance(fr.bbox, dict):
                    x = int(fr.bbox.get("x", 0))
                    y = int(fr.bbox.get("y", 0))
                    w = int(fr.bbox.get("w", 0))
                    h = int(fr.bbox.get("h", 0))
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(annotated, name, (x, max(y - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Localized Field Bounding Boxes")
        else:
            st.info("No localized fields available.")

    with tabs[3]:
        if result.fields:
            cols = st.columns(min(3, len(result.fields)))
            for idx, (name, fr) in enumerate(result.fields.items()):
                with cols[idx % len(cols)]:
                    st.caption(f"**{name}** ({fr.status})")
                    if canonical_card is not None and fr.bbox and isinstance(fr.bbox, dict):
                        x = int(fr.bbox.get("x", 0))
                        y = int(fr.bbox.get("y", 0))
                        w = int(fr.bbox.get("w", 0))
                        h = int(fr.bbox.get("h", 0))
                        crop = canonical_card[y:y + h, x:x + w]
                        if crop.size > 0:
                            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.write(f"Value: `{fr.normalized or fr.raw or '-'}`")

    with tabs[4]:
        st.json(result.as_full_dict())

else:
    st.info("👆 Upload an Egyptian National ID photo or click **'Load Sample Egyptian ID Demo'** in the sidebar to begin.")
