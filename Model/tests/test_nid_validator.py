import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.nid_validator import validate_nid, _compute_check_digit, VALID, INVALID_FORMAT, \
    INVALID_DATE, INVALID_GOVERNORATE, INVALID_CHECKSUM, INVALID_STRUCTURE


def _make_valid_nid(century="2", yy="99", mm="04", dd="17", gov="01", serial="0011"):
    # century="2" -> 1900s, so yy="99" means 1999 (not a future date).
    body = century + yy + mm + dd + gov + serial
    check = _compute_check_digit(body)
    return body + str(check)


def test_valid_nid_round_trip():
    nid = _make_valid_nid()
    result = validate_nid(nid)
    assert result.status == VALID
    assert result.derived_birth_date == "1999-04-17"
    assert result.derived_governorate_code == "01"
    assert result.derived_gender == "male"  # serial ends in 1 -> odd -> male


def test_wrong_length():
    result = validate_nid("12345")
    assert result.status == INVALID_FORMAT


def test_arabic_indic_digits_normalized():
    nid = _make_valid_nid()
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_to_arabic = {str(i): arabic_digits[i] for i in range(10)}
    arabic_nid = "".join(western_to_arabic[c] for c in nid)
    result = validate_nid(arabic_nid)
    assert result.status == VALID
    assert result.normalized == nid


def test_invalid_month():
    body = "2" + "99" + "13" + "17" + "01" + "0011"  # month=13
    check = _compute_check_digit(body)
    result = validate_nid(body + str(check))
    assert result.status == INVALID_DATE


def test_invalid_day_for_month():
    body = "2" + "99" + "02" + "30" + "01" + "0011"  # Feb 30 invalid
    check = _compute_check_digit(body)
    result = validate_nid(body + str(check))
    assert result.status == INVALID_DATE


def test_invalid_governorate():
    body = "2" + "99" + "04" + "17" + "99" + "0011"  # 99 not a real gov code
    check = _compute_check_digit(body)
    result = validate_nid(body + str(check))
    assert result.status == INVALID_GOVERNORATE


def test_invalid_century_digit():
    body = "5" + "99" + "04" + "17" + "01" + "0011"  # century digit 5 invalid
    check = _compute_check_digit(body)
    result = validate_nid(body + str(check))
    assert result.status == INVALID_STRUCTURE


def test_checksum_mismatch_detected():
    nid = _make_valid_nid()
    tampered = nid[:-1] + str((int(nid[-1]) + 1) % 10)
    result = validate_nid(tampered)
    assert result.status == INVALID_CHECKSUM


def test_gender_derivation_even_serial_is_female():
    nid = _make_valid_nid(serial="0012")  # ends in 2 -> even -> female
    result = validate_nid(nid)
    assert result.derived_gender == "female"


def test_future_date_rejected():
    # century="3" -> 2000s, yy="99" -> 2099: genuinely in the future.
    body = "3" + "99" + "04" + "17" + "01" + "0011"
    check = _compute_check_digit(body)
    result = validate_nid(body + str(check))
    assert result.status == INVALID_DATE
    assert "future" in result.errors[0].lower()


def test_non_future_1999_date_is_fine():
    result = validate_nid(_make_valid_nid(yy="99"))
    assert result.status == VALID
    assert result.derived_birth_date == "1999-04-17"
