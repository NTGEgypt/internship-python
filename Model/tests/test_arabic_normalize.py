import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.arabic_normalize import digits_to_western, normalize_arabic_text, normalize_numeric_field


def test_arabic_indic_digit_conversion():
    assert digits_to_western("٠١٢٣٤٥٦٧٨٩") == "0123456789"


def test_persian_indic_digit_conversion():
    assert digits_to_western("۰۱۲۳") == "0123"


def test_mixed_digits_untouched_letters():
    assert digits_to_western("محمد١٢٣") == "محمد123"


def test_tatweel_removed():
    text = "محمـــد"
    normalized = normalize_arabic_text(text)
    assert "\u0640" not in normalized


def test_numeric_field_strips_non_digits():
    assert normalize_numeric_field("77-047 4717 44177") == "77047471744177"


def test_numeric_field_handles_arabic_digits_with_noise():
    assert normalize_numeric_field("٧٧٠٤٧-٤٧١٧٤٤١٧٧") == "77047471744177"


def test_whitespace_collapsed():
    assert normalize_arabic_text("محمد    علي") == "محمد علي"
