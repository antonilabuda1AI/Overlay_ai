from __future__ import annotations

from typing import List

import numpy as np
import pytesseract


_LANGS_VALIDATED = False


def _validate_langs(langs: str) -> str:
    global _LANGS_VALIDATED
    if _LANGS_VALIDATED:
        return langs
    try:
        available = set(pytesseract.get_languages())
        req = set(langs.split("+"))
        if not req.issubset(available) and available:
            langs = "+".join([l for l in req if l in available]) or list(available)[0]
    except Exception:
        pass
    _LANGS_VALIDATED = True
    return langs


def choose_psm_for_roi(w: int, h: int, default_psm: int) -> int:
    # Narrow or short text lines benefit from PSM 7
    aspect = w / max(1, h)
    if w < 60 or h < 18 or aspect > 6.0:
        return 7
    return default_psm


def ocr_roi(img_rgb: np.ndarray, psm: int = 6, langs: str = "eng+pol", whitelist: str = "") -> str:
    langs = _validate_langs(langs)
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    txt = pytesseract.image_to_string(img_rgb, lang=langs, config=config)
    return txt.strip()


def batch_ocr(img_rgb: np.ndarray, rois: List[List[int]], psm: int, langs: str, whitelist: str = "") -> List[str]:
    results: List[str] = []
    for x, y, w, h in rois:
        crop = img_rgb[y : y + h, x : x + w]
        p = choose_psm_for_roi(w, h, psm)
        results.append(ocr_roi(crop, psm=p, langs=langs, whitelist=whitelist))
    return results
