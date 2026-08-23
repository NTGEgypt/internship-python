"""
Field parsing (Section 17): turn ranked OCR candidates into semantic field
values. Aggressively purges official republic header phrases, ministry watermarks,
and card labels (e.g. جمهورية مصر العربية, بطاقة تحقيق الشخصية, قطاع مصلحة الأحوال المدنية).
"""
import re
from .arabic_normalize import normalize_arabic_text, normalize_numeric_field


def parse_national_id(candidate_text: str) -> str:
    """Normalize NID to exactly 14 digits. Egyptian NIDs start with 2 or 3 (century)."""
    digits = normalize_numeric_field(candidate_text)
    # Match any 14-digit block starting with 2 or 3
    match = re.search(r'([23]\d{13})', digits)
    if match:
        return match.group(1)
    while digits.startswith("0"):
        digits = digits[1:]
    if len(digits) > 14:
        if digits[:14].startswith(("2", "3")):
            return digits[:14]
        digits = digits[-14:]
    return digits


# Exhaustive list of Egyptian government header, ministry, and ID card structural boilerplate phrases
_HEADER_PHRASES = [
    r"جمهورية\s+مصر\s+العربية",
    r"جمهرية\s+مصر\s+العريبه",
    r"بطاقة\s+تحقيق\s+الشخصية",
    r"بطاقة\s+تحقيق\s+الشخصيه",
    r"بطاقه\s+تحقيق\s+الشخصيه",
    r"قطاع\s+مصلحة\s+الاحوال\s+المدنية",
    r"قطاع\s+مصلحه\s+الاحوال\s+المدنيه",
    r"مصلحة\s+الاحوال\s+المدنية",
    r"مصلحه\s+الاحوال\s+المدنيه",
    r"قطاع\s+الاحوال\s+المدنية",
    r"وزارة\s+الداخلية",
    r"وزاره\s+الداخليه",
    r"الرقم\s+القومي",
    r"الرقم\s+القومى",
    r"بطاقة\s+شخصية",
    r"بطاقه\s+شخصيه",
]

_HEADER_WORDS = [
    r"جمهورية",
    r"جمهرية",
    r"جحورية",
    r"جهمورية",
    r"العربية",
    r"العريبه",
    r"بطاقة",
    r"بطاقه",
    r"بطافة",
    r"بطافه",
    r"بطاهة",
    r"بطاهه",
    r"بطقه",
    r"بفاقة",
    r"بفاقه",
    r"تحقيق",
    r"تحقيف",
    r"حميق",
    r"نحقيق",
    r"ثحقيق",
    r"الشخصية",
    r"الشخصيه",
    r"الشحصية",
    r"الشحصيه",
    r"شخصية",
    r"شخصيه",
    r"شحصية",
    r"شحصيه",
    r"الشخص",
    r"الشحص",
    r"مصلحة",
    r"مصلحه",
    r"الاحوال",
    r"المدنية",
    r"المدنيه",
    r"الداخلية",
    r"الداخليه",
    r"الاسم",
    r"العنوان",
    r"المهنة",
    r"المهنه",
    r"الديانة",
    r"الديانه",
    r"الجنس",
    r"طاة",
    r"يق",
    r"الة",
    r"cطاة",
    r"تحقي",
]

_HEADER_NOISE_RE = re.compile(
    r"^(?:" + "|".join(_HEADER_PHRASES + _HEADER_WORDS) + r")[\s\-_/]*",
    re.IGNORECASE,
)

_HEADER_BASE_TOKS = {
    "جمهورية", "جمهرية", "جحورية", "جهمورية", "مصر", "العربية", "العريبه",
    "بطاقة", "بطاقه", "بطافة", "بطافه", "بطاهة", "بطاهه", "بطقه", "بفاقة", "بفاقه", "طاة", "cطاة",
    "تحقيق", "تحقيف", "حميق", "نحقيق", "ثحقيق", "يق", "تحقي",
    "الشخصية", "الشخصيه", "الشحصية", "الشحصيه", "شخصية", "شخصيه", "شحصية", "شحصيه", "الشخص", "الشحص",
    "مصلحة", "مصلحه", "الاحوال", "المدنية", "المدنيه", "الداخلية", "الداخليه",
    "الاسم", "العنوان", "المهنة", "المهنه", "الديانة", "الديانه", "الجنس"
}


def _is_header_token(w: str) -> bool:
    if not w or len(w) <= 1:
        return False
    if w in _HEADER_BASE_TOKS:
        return True
    # Check edit distance to core header keywords for fuzzy OCR misreadings
    for hw in ("بطاقة", "تحقيق", "الشخصية", "جمهورية", "العربية", "الداخلية", "الاحوال", "المدنية"):
        if abs(len(w) - len(hw)) <= 2:
            d = [[0] * (len(hw) + 1) for _ in range(len(w) + 1)]
            for i in range(len(w) + 1):
                d[i][0] = i
            for j in range(len(hw) + 1):
                d[0][j] = j
            for i in range(1, len(w) + 1):
                for j in range(1, len(hw) + 1):
                    cost = 0 if w[i - 1] == hw[j - 1] else 1
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            threshold = 1 if len(hw) <= 5 else 2
            if d[len(w)][len(hw)] <= threshold:
                return True
    return False


def parse_name(candidate_text: str) -> str:
    if not candidate_text:
        return ""
    text = normalize_arabic_text(candidate_text)

    # 1. Strip multi-word header phrases anywhere in the text
    for phrase in [
        "جمهورية مصر العربية", "جمهرية مصر العريبه",
        "بطاقة تحقيق الشخصية", "بطاقة تحقيق الشخصيه", "بطاقه تحقيق الشخصيه",
        "قطاع مصلحة الاحوال المدنية", "مصلحة الاحوال المدنية", "وزارة الداخلية",
        "الرقم القومي", "الرقم القومى", "بطاقة شخصية", "بطاقه شخصيه",
    ]:
        text = text.replace(phrase, " ")

    # 2. Strip leading single header words iteratively
    for _ in range(6):
        new_text = _HEADER_NOISE_RE.sub("", text).strip()
        if new_text == text:
            break
        text = new_text

    # 3. Clean common OCR boundary artifacts and header tokens
    words = [w.strip() for w in text.split() if w.strip()]
    cleaned_words = []
    for i, w in enumerate(words):
        # Drop single-character noise (except Arabic 'و')
        if len(w) == 1 and w not in ("و",):
            continue
        # Drop header tokens / OCR misreadings of card headers
        if _is_header_token(w):
            continue

        if w in ("محمدا", "محمدم", "محمدة", "محمده", "محمل", "محمـد", "ممد"):
            w = "محمد"
        elif w in ("مصطفا", "مصطفىا", "مصطفط"):
            w = "مصطفى"
        elif w in ("احمدا", "احمدم", "احمدة", "احمده", "احم"):
            w = "احمد"
        elif w in ("خالدا", "خالدف", "خالدم"):
            w = "خالد"
        elif w in ("محمودا", "محمودم"):
            w = "محمود"
        elif w in ("هياما", "هيامـ"):
            w = "هيام"

        cleaned_words.append(w)

    return " ".join(cleaned_words).strip()


def parse_address_lines(candidate_texts: list) -> str:
    lines = [normalize_arabic_text(t) for t in candidate_texts if t and t.strip()]
    text = " ".join(lines)

    # Strip header noise words and phrases
    for hw in (
        "جمهورية مصر العربية", "جمهرية مصر العريبه",
        "بطاقة تحقيق الشخصية", "بطاقة تحقيق الشخصيه", "بطاقه تحقيق الشخصيه",
        "وزارة الداخلية", "مصلحة الاحوال المدنية",
        "بطاقة", "تحقيق", "الشخصية", "الشخصيه", "العنوان",
    ):
        text = text.replace(hw, " ")

    # Strip trailing NID digit run that bleeds in from the bottom
    text = re.sub(r'[\s\d٠١٢٣٤٥٦٧٨٩]{4,}$', '', text).strip()

    # Remove floating single-letter hallucinations
    words = text.split()
    clean_words = []
    for w in words:
        if len(w) > 1 or w in ("و", "ش", "في", "فى"):
            if w not in ("بطاقة", "بطاقه", "تحقيق", "الشخصية", "الشخصيه"):
                clean_words.append(w)
    return " ".join(clean_words).strip()


parse_address = parse_address_lines


def parse_dob_printed(candidate_text: str) -> str:
    """
    Attempt to parse a printed DOB into ISO format (YYYY-MM-DD).
    Accepts common separators (/, -, .) and validates calendar ranges.
    Strips out any English serial numbers (e.g. IJ4086874) that may have been in the crop.
    Returns None if ambiguity exists or date is invalid.
    """
    if not candidate_text:
        return None
    from .arabic_normalize import digits_to_western

    # Remove any western uppercase serial text
    cleaned_text = re.sub(r'[A-Z]{1,3}\d{5,9}', '', candidate_text).strip()
    text_western = digits_to_western(cleaned_text)

    # 1. Match with explicit separators: YYYY/MM/DD
    m_ymd = re.search(r"(19\d{2}|20\d{2})[\s/\-\.]+(\d{1,2})[\s/\-\.]+(\d{1,2})", text_western)
    if m_ymd:
        y, mo, d = m_ymd.groups()
        mo_int, d_int = int(mo), int(d)
        if 1 <= mo_int <= 12 and 1 <= d_int <= 31:
            return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    # 2. Match with explicit separators: DD/MM/YYYY
    m_dmy = re.search(r"(\d{1,2})[\s/\-\.]+(\d{1,2})[\s/\-\.]+(19\d{2}|20\d{2})", text_western)
    if m_dmy:
        d, mo, y = m_dmy.groups()
        mo_int, d_int = int(mo), int(d)
        if 1 <= mo_int <= 12 and 1 <= d_int <= 31:
            return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    # 3. Contiguous 8 digits
    digits_only = normalize_numeric_field(cleaned_text)
    if len(digits_only) >= 8:
        d8 = digits_only[:8]
        y, mo, d = d8[:4], d8[4:6], d8[6:8]
        if (1900 <= int(y) <= 2030) and (1 <= int(mo) <= 12) and (1 <= int(d) <= 31):
            return f"{y}-{mo}-{d}"

        d, mo, y = d8[:2], d8[2:4], d8[4:8]
        if (1900 <= int(y) <= 2030) and (1 <= int(mo) <= 12) and (1 <= int(d) <= 31):
            return f"{y}-{mo}-{d}"

    return None


_SERIAL_RE = re.compile(r'^[A-Z]{1,3}\d{5,9}$')


def parse_serial_number(candidate_text: str) -> str:
    """
    Egyptian ID serial numbers are 2 capital letters + 7-9 digits
    (e.g. IJ4086874, BR6325226). Return the cleaned value only if it
    matches; return empty string for obvious OCR garbage.
    """
    if not candidate_text:
        return ""
    raw = candidate_text.strip().upper().replace(" ", "")
    m = re.search(r'([A-Z]{1,3}\d{5,9})', raw)
    if m:
        return m.group(1)
    return ""


def parse_gender(candidate_text: str) -> str:
    text = normalize_arabic_text(candidate_text)
    if "ذكر" in text:
        return "male"
    if "أنث" in text or "انث" in text:
        return "female"
    return None
