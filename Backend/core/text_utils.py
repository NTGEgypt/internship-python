"""
String-level helpers shared by field selection and NID extraction.
Nothing here touches an image or a model — pure text transforms,
so this file is trivial to unit test on its own.
"""

import re

import numpy as np

from core.constants import ARABIC_DIGITS, DIGIT_CONFUSIONS


def clean_space(text):
    return re.sub(r'\s+', ' ', str(text)).strip()


def arabic_normalize(text):
    text = str(text)
    for old, new in {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ى': 'ي'}.items():
        text = text.replace(old, new)
    return clean_space(text)


def numeric_text(text):
    text = str(text).translate(ARABIC_DIGITS)
    for old, new in DIGIT_CONFUSIONS.items():
        text = text.replace(old, new)
    return re.sub(r'[^0-9]', '', text)


def order_rtl_lines(items, line_tol=15):
    """
    Groups OCR word/line detections into visual rows by y-coordinate,
    orders rows top-to-bottom, then orders words within each row
    right-to-left by x-coordinate. Needed because OCR engines return
    detections in whatever order they found them, not reading order.
    """
    if not items:
        return []
    lines = []
    for item in sorted(items, key=lambda i: i.get('y', 0)):
        placed = False
        for line in lines:
            if abs(line[0].get('y', 0) - item.get('y', 0)) <= line_tol:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])
    lines.sort(key=lambda line: np.mean([i.get('y', 0) for i in line]))
    ordered = []
    for line in lines:
        line.sort(key=lambda i: -i.get('x', 0))
        ordered.extend(line)
    return ordered
