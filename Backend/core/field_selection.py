"""
Turns a flat list of OCR ensemble results for one field into a single
chosen value. `choose_field` handles single-line fields (name pieces,
serial); `choose_address` additionally reconstructs multi-word,
multi-line RTL reading order via order_rtl_lines.
"""

import re
from collections import defaultdict

import numpy as np

from core.constants import ADDRESS_STOPWORDS, GOVERNORATES_AR
from core.text_utils import arabic_normalize, clean_space, numeric_text, order_rtl_lines


def choose_field(results, serial=False, min_engines=1, min_confidence=0.0):
    groups = defaultdict(list)
    for item in results:
        value = re.sub(r'[^A-Za-z0-9]', '', item['text']).upper() if serial else arabic_normalize(item['text'])
        if len(value) >= (5 if serial else 2):
            groups[value].append(item)
    if not groups:
        return None, 0.0

    ranked = []
    for value, items in groups.items():
        confidence = float(np.mean([item['confidence'] for item in items]))
        engines = len({item['engine'] for item in items})
        ranked.append((engines * 10 + len(items) * 2 + confidence * 3, value, confidence, engines))
    ranked.sort(reverse=True, key=lambda item: item[0])

    _, value, confidence, engines = ranked[0]
    if engines < min_engines or confidence < min_confidence:
        return None, 0.0
    return value, confidence


def choose_address(results):
    grouped = defaultdict(list)
    for item in results:
        text = arabic_normalize(item['text'])
        if len(text) >= 2 and not (set(text.split()) & ADDRESS_STOPWORDS):
            grouped[(item['engine'], item['variant'])].append({**item, 'text': text})

    candidates = []
    for (engine, variant), items in grouped.items():
        lines, seen = [], []
        for item in order_rtl_lines(items):
            key = re.sub(r'\s+', '', item['text'])
            if key not in seen:
                seen.append(key)
                lines.append(item['text'])
        value = clean_space(' '.join(line for line in lines if len(numeric_text(line)) < 3))
        if value:
            confidence = float(np.mean([item['confidence'] for item in items]))
            score = (
                confidence * 5
                + min(len(value), 60) / 18
                + (2 if len(value.split()) >= 3 else 0)
                + (2 if any(gov in value for gov in GOVERNORATES_AR.values()) else 0)
            )
            candidates.append((score, value, confidence))

    if not candidates:
        return None, 0.0
    _, value, confidence = max(candidates, key=lambda item: item[0])
    return value, confidence
