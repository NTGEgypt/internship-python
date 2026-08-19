"""
Paths and runtime device configuration.
Keep this file free of heavy imports (torch/cv2 only) so it loads fast
and can be imported by any other module without side effects.
"""

import os

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_xdnn'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import platform
from pathlib import Path

import pytesseract
import torch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# TESSERACT_CMD env var wins if set; otherwise fall back to the default
# Windows install path (only on Windows, only if it exists). On Linux/Mac
# this block does nothing and pytesseract finds the binary on PATH.
_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if not _tesseract_cmd and platform.system() == "Windows":
    _default_windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _default_windows_path.exists():
        _tesseract_cmd = str(_default_windows_path)

if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd