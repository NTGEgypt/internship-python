"""
Template registry (Section 15).

Field regions are stored as NORMALIZED (fraction-of-card) coordinates so
the same template works regardless of input image resolution -- they are
converted to canonical-card pixel coordinates at runtime (Section 12/16).

NOTE ON ACCURACY: the exact region fractions below are a best-effort
starting layout based on the publicly known general structure of the
current-issue Egyptian National ID (front: photo + name/DOB/address block
on the right for RTL layout, NID number bottom strip; back: job/marital
fields + barcode strip). These fractions are almost certainly not pixel
perfect for every physical print run and MUST be recalibrated against real
sample cards (Section 53) -- that recalibration is exactly what
localization.py's semantic-anchor refinement step is for. Do not treat
these numbers as ground truth; treat them as a coarse prior that anchors
detection.
"""
from dataclasses import dataclass, field as dc_field
from .coordinate_systems import NormalizedBBox
from . import config


@dataclass
class FieldSpec:
    name: str
    region: NormalizedBBox
    field_type: str  # "arabic_text" | "numeric" | "arabic_numeric" | "barcode" | "image_region"
    required: bool = True


@dataclass
class Template:
    template_id: str
    side: str
    fields: list


FRONT_TEMPLATE_V1 = Template(
    template_id="egy_id_front_v1",
    side=config.SIDE_FRONT,
    fields=[
        # Photo: portrait on left (x: 3%-31%, y: 5%-67%)
        FieldSpec("photo",         NormalizedBBox(0.03, 0.050, 0.28, 0.620), "image_region"),
        # Name: citizen full name lines 1 & 2 (x: 33%-98%, y: 26%-46%)
        FieldSpec("name",          NormalizedBBox(0.33, 0.260, 0.65, 0.200), "arabic_text"),
        # Address: citizen address lines 3 & 4 (x: 33%-98%, y: 46%-71%)
        FieldSpec("address",       NormalizedBBox(0.33, 0.460, 0.65, 0.250), "arabic_text"),
        # National ID: 14-digit Arabic strip (x: 33%-98%, y: 71%-88%)
        FieldSpec("national_id",   NormalizedBBox(0.33, 0.710, 0.65, 0.170), "numeric"),
        # DOB: left hologram strip containing date of birth (x: 3%-35%, y: 68%-85%)
        FieldSpec("date_of_birth", NormalizedBBox(0.03, 0.680, 0.32, 0.170), "numeric", required=False),
        # Serial Number: black alphanumeric code at bottom-left (x: 5%-35%, y: 85%-98%)
        FieldSpec("serial_number", NormalizedBBox(0.05, 0.850, 0.30, 0.130), "text",    required=False),
    ],
)

BACK_TEMPLATE_V1 = Template(
    template_id="egy_id_back_v1",
    side=config.SIDE_BACK,
    fields=[
        FieldSpec("job_or_status", NormalizedBBox(0.05, 0.06, 0.55, 0.14), "arabic_text"),
        FieldSpec("gender", NormalizedBBox(0.05, 0.22, 0.30, 0.12), "arabic_text"),
        FieldSpec("religion", NormalizedBBox(0.38, 0.22, 0.30, 0.12), "arabic_text", required=False),
        FieldSpec("marital_status", NormalizedBBox(0.05, 0.36, 0.30, 0.12), "arabic_text", required=False),
        FieldSpec("husband_name", NormalizedBBox(0.05, 0.50, 0.60, 0.12), "arabic_text", required=False),
        FieldSpec("barcode", NormalizedBBox(0.55, 0.60, 0.42, 0.35), "barcode"),
    ],
)

TEMPLATE_REGISTRY = {
    FRONT_TEMPLATE_V1.template_id: FRONT_TEMPLATE_V1,
    BACK_TEMPLATE_V1.template_id: BACK_TEMPLATE_V1,
}


def get_template_for_side(side: str) -> Template:
    if side == config.SIDE_FRONT:
        return FRONT_TEMPLATE_V1
    if side == config.SIDE_BACK:
        return BACK_TEMPLATE_V1
    raise ValueError(f"No template registered for side={side!r}")
