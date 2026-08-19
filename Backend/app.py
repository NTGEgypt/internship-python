import json

import cv2
import numpy as np
import streamlit as st

from core.pipeline import process_image
from core.database import save_ocr_result


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Egyptian ID OCR",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1b5e20, #2e7d32, #66bb6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .field-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .field-card.review {
        border-left: 4px solid #f59e0b;
        background: #fffbeb;
    }
    .field-card.ok {
        border-left: 4px solid #22c55e;
    }
    .field-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .field-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #111827;
        direction: rtl;
        text-align: right;
    }
    .field-value.ltr {
        direction: ltr;
        text-align: left;
        font-family: 'Courier New', monospace;
    }
    .field-value.empty {
        color: #9ca3af;
        font-style: italic;
        font-weight: 400;
    }
    .conf-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        float: right;
    }
    .conf-high { background: #dcfce7; color: #15803d; }
    .conf-mid  { background: #fef3c7; color: #b45309; }
    .conf-low  { background: #fee2e2; color: #b91c1c; }
    .quality-banner {
        padding: 12px 18px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .quality-good { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .quality-borderline { background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }
    .quality-bad { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #86efac;
        border-radius: 12px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def confidence_class(value):
    if value >= 0.8:
        return "conf-high"
    if value >= 0.5:
        return "conf-mid"
    return "conf-low"


def quality_class(status):
    return {"GOOD": "quality-good", "BORDERLINE": "quality-borderline", "BAD": "quality-bad"}.get(status, "quality-borderline")


def quality_icon(status):
    return {"GOOD": "✅", "BORDERLINE": "⚠️", "BAD": "🚫"}.get(status, "ℹ️")


def render_field(label, value, confidence=None, needs_review=False, ltr=False):
    card_class = "review" if needs_review else "ok"
    value_class = "empty" if not value else ("ltr" if ltr else "")
    display_value = value if value else "Not detected"

    conf_html = ""
    if confidence is not None:
        conf_html = f'<span class="conf-badge {confidence_class(confidence)}">{confidence*100:.0f}%</span>'

    review_flag = " 🔎 needs review" if needs_review else ""

    st.markdown(f"""
    <div class="field-card {card_class}">
        <div class="field-label">{label}{review_flag}{conf_html}</div>
        <div class="field-value {value_class}">{display_value}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🪪 About")
    st.write(
        "Upload a photo of an Egyptian National ID card. "
        "The system detects the card, locates each field, "
        "runs a multi-engine OCR ensemble, and validates the "
        "national ID number structurally before returning it."
    )
    st.markdown("---")
    st.markdown("### 📋 Tips for best results")
    st.markdown(
        "- Shoot flat, front-on — avoid angled shots\n"
        "- Even lighting, no glare across the card\n"
        "- Fill the frame with the card\n"
        "- Avoid covering any field with fingers"
    )
    st.markdown("---")
    st.caption("Fields extracted: name, address, national ID, serial number, birth date, gender, governorate.")


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">Egyptian National ID OCR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an ID image to automatically extract and validate its data</div>', unsafe_allow_html=True)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload ID Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:

    left_col, right_col = st.columns([1, 1.3], gap="large")

    with left_col:
        st.image(uploaded_file, caption="Uploaded ID", use_container_width=True)
        process_clicked = st.button("🔍 Process Image", use_container_width=True, type="primary")

    if process_clicked:
        try:
            upload_bytes = uploaded_file.getvalue()
            file_bytes = np.frombuffer(upload_bytes, dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Could not read the uploaded image.")
                st.stop()

            with st.spinner("Detecting card, reading fields, validating national ID..."):
                result = process_image(image)
                result_id = save_ocr_result(
                    result=result,
                    image_bytes=upload_bytes,
                    original_filename=uploaded_file.name,
                    image_mime_type=uploaded_file.type,
                )

            with right_col:
                st.caption(f"Saved as OCR submission #{result_id} (pending review).")
                quality = result.get("capture_quality", "UNKNOWN")
                st.markdown(
                    f'<div class="quality-banner {quality_class(quality)}">'
                    f'{quality_icon(quality)} Capture quality: {quality}</div>',
                    unsafe_allow_html=True,
                )

                conf = result.get("confidence", {})
                review = result.get("needs_review", {})

                st.markdown("#### Personal Information")
                render_field("Full Name", result.get("full_name"),
                             conf.get("first_name"), review.get("full_name"))
                render_field("National ID", result.get("national_id"),
                             conf.get("national_id"), review.get("national_id"), ltr=True)
                render_field("Birth Date", result.get("birth_date"))
                render_field("Gender", result.get("gender"))
                render_field("Governorate", result.get("governorate"))

                st.markdown("#### Address & Document")
                render_field("Address", result.get("address"),
                             conf.get("address"), review.get("address"))
                render_field("Serial Number", result.get("serial_number"),
                             conf.get("serial_number"), review.get("serial_number"), ltr=True)

                needs_any_review = any(review.values())
                if needs_any_review:
                    flagged = [k for k, v in review.items() if v]
                    st.warning(f"⚠️ Fields flagged for manual review: {', '.join(flagged)}")
                else:
                    st.success("✅ All fields passed validation.")

                st.download_button(
                    "⬇️ Download result as JSON",
                    data=json.dumps(result, ensure_ascii=False, indent=2),
                    file_name="id_ocr_result.json",
                    mime="application/json",
                    use_container_width=True,
                )

                with st.expander("🔧 Raw JSON / detection confidence"):
                    st.json(result)

        except Exception as e:
            st.error(f"Processing failed: {e}")

else:
    st.info("👆 Upload an image to get started.")
