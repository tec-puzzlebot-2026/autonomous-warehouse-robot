"""QR decoding and the carried-box LiDAR mask (no ROS needed beyond imports)."""

import math

import cv2
import numpy as np

from main_ros.navigation.scan_mask import box_mask
from main_ros.restock import geometry as G
from main_ros.vision.qr.decode import decode_qr


def _qr(text, size=140):
    img = cv2.QRCodeEncoder.create().encode(text)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def test_qr_codes_are_decoded_with_their_corners():
    frame = np.full((480, 640, 3), 255, np.uint8)
    frame[100:240, 80:220] = cv2.cvtColor(_qr('walnut'), cv2.COLOR_GRAY2BGR)
    frame[100:240, 400:540] = cv2.cvtColor(_qr('emezon'), cv2.COLOR_GRAY2BGR)
    codes = decode_qr(frame, cv2.QRCodeDetector())
    texts = {c['text'] for c in codes}
    assert texts == {'walnut', 'emezon'}
    walnut = [c for c in codes if c['text'] == 'walnut'][0]
    us = [p[0] for p in walnut['corners']]
    assert 70 <= min(us) and max(us) <= 230
    # Only the code inside the target box's outline belongs to it.
    box = [(60, 80), (240, 80), (240, 260), (60, 260)]
    owned = [c['text'] for c in codes if G.qr_belongs_to_box(c['corners'], box, max_ratio=0.9)]
    assert owned == ['walnut']


def test_unknown_texts_are_ignored():
    frame = np.full((480, 640, 3), 255, np.uint8)
    frame[100:240, 80:220] = cv2.cvtColor(_qr('hello'), cv2.COLOR_GRAY2BGR)
    assert decode_qr(frame, cv2.QRCodeDetector(), known=('walnut',)) == []


def test_box_mask_hides_only_the_carried_box():
    n = 360
    angle_min, inc = -math.pi, 2 * math.pi / n
    ranges = np.full(n, 2.0)
    a = angle_min + inc * np.arange(n)
    front = np.abs(a) < math.radians(20)
    ranges[front] = 0.20                                    # the box, 20 cm ahead
    mask = box_mask(ranges, angle_min, inc, (0.0, 0.0, 0.0), 0.0, 0.32, 0.09)
    assert mask[np.abs(a) < math.radians(10)].all()          # the middle of the box
    assert not mask[~front].any()                            # walls 2 m away stay
    ranges[np.abs(a - math.radians(90)) < 0.05] = 0.15       # something beside the robot
    mask = box_mask(ranges, angle_min, inc, (0.0, 0.0, 0.0), 0.0, 0.32, 0.09)
    assert not mask[np.abs(a - math.radians(90)) < 0.05].any()
