"""QR decoding with OpenCV (no ROS): text, company and corners of every QR in a frame.

The boxes carry QR codes such as "EMEZON-03" (config/qr_companies.yaml, from the QR
teammate's reader): `companies` maps each text to its company. Like his reader, a
second attempt with a contrast boost (CLAHE) catches codes in poor light or glare.
"""
# Author: Raymundo Gomez (contrast fallback and company table: the QR teammate's qr_core)

import cv2
import numpy as np

_CLAHE = None


def load_companies(path):
    """{QR text (upper case): company (lower case)} from a flat YAML file."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {str(k).strip().upper(): str(v).strip().lower() for k, v in data.items()}


def _decode(gray, detector):
    """[(text, 4x2 corners)] of every QR in a gray image."""
    found = []
    try:
        ok, texts, points, _ = detector.detectAndDecodeMulti(gray)
    except cv2.error:
        ok, texts, points = False, (), None
    if ok and points is not None:
        for text, pts in zip(texts, points):
            text = (text or '').strip()
            if text:
                found.append((text, np.asarray(pts, np.float32).reshape(-1, 2)))
    if not found:
        try:
            text, pts, _ = detector.detectAndDecode(gray)
        except cv2.error:
            text, pts = '', None
        text = (text or '').strip()
        if text and pts is not None:
            found.append((text, np.asarray(pts, np.float32).reshape(-1, 2)))
    return found


def decode_qr(frame, detector, upscale=1.0, known=(), companies=None, contrast_retry=True):
    """All QR codes in a BGR frame -> [{'text', 'raw', 'corners'}] in frame pixels.

    raw: what the QR says. text: its company when `companies` is given (codes not in
    the table are dropped), else the raw text. known: if not empty, only these texts
    (after the company lookup) are reported.
    """
    global _CLAHE
    img = frame if upscale == 1.0 else cv2.resize(frame, None, fx=upscale, fy=upscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    found = _decode(gray, detector)
    if not found and contrast_retry:
        if _CLAHE is None:
            _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        found = _decode(_CLAHE.apply(gray), detector)
    out = []
    for raw, pts in found:
        text = raw
        if companies is not None:
            text = companies.get(raw.upper())
            if text is None:
                continue
        if known and text not in known:
            continue
        corners = (pts / upscale).tolist()
        out.append({'text': text, 'raw': raw,
                    'corners': [[round(u, 1), round(v, 1)] for u, v in corners]})
    return out
