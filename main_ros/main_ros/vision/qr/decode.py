"""QR decoding with OpenCV (no ROS): text and corners of every QR code in a frame."""
# Author: Raymundo Gomez

import cv2
import numpy as np


def decode_qr(frame, detector, upscale=1.0, known=()):
    """All QR codes in a BGR frame -> [{'text', 'corners'}] in frame pixels."""
    img = frame if upscale == 1.0 else cv2.resize(frame, None, fx=upscale, fy=upscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    try:
        ok, texts, points, _ = detector.detectAndDecodeMulti(gray)
    except cv2.error:
        ok, texts, points = False, (), None
    if ok and points is not None:
        for text, pts in zip(texts, points):
            text = (text or '').strip()
            if not text or (known and text not in known):
                continue
            corners = (np.asarray(pts, np.float32).reshape(-1, 2) / upscale).tolist()
            out.append({'text': text, 'corners': [[round(u, 1), round(v, 1)] for u, v in corners]})
    if not out:
        text, pts, _ = detector.detectAndDecode(gray)
        text = (text or '').strip()
        if text and pts is not None and (not known or text in known):
            corners = (np.asarray(pts, np.float32).reshape(-1, 2) / upscale).tolist()
            out.append({'text': text, 'corners': [[round(u, 1), round(v, 1)] for u, v in corners]})
    return out
