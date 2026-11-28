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


def test_printed_codes_are_reported_as_their_company():
    companies = {'WALNUT-02': 'walnut', 'EMEZON-03': 'emezon'}
    frame = np.full((480, 640, 3), 255, np.uint8)
    frame[100:240, 80:220] = cv2.cvtColor(_qr('WALNUT-02'), cv2.COLOR_GRAY2BGR)
    frame[100:240, 400:540] = cv2.cvtColor(_qr('NOT-A-BOX'), cv2.COLOR_GRAY2BGR)
    codes = decode_qr(frame, cv2.QRCodeDetector(), companies=companies)
    assert [(c['text'], c['raw']) for c in codes] == [('walnut', 'WALNUT-02')]


def test_a_dim_low_contrast_code_is_read_on_the_second_try():
    frame = np.full((480, 640, 3), 255, np.uint8)
    frame[100:240, 80:220] = cv2.cvtColor(_qr('EMEZON-03'), cv2.COLOR_GRAY2BGR)
    dim = (frame.astype(np.float32) * 0.12 + 40).astype(np.uint8)     # dark, washed out
    codes = decode_qr(dim, cv2.QRCodeDetector(), companies={'EMEZON-03': 'emezon'})
    assert [c['text'] for c in codes] == ['emezon']


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


def test_face_center_offset_whole_and_cut_by_the_image_edge():
    cam = G.Camera(500.0, 500.0, 320.0, 240.0, (0.10, 0.0, 0.178, 0.0, 0.26, 0.0))
    Tinv = np.linalg.inv(cam.T)

    def uv(y, x=0.40, z=0.06 - 0.047):          # face mid-height, base_link frame
        p = Tinv @ np.array([x, y, z, 1.0])
        return cam.cx + cam.fx * p[0] / p[2], cam.cy + cam.fy * p[1] / p[2]
    for lat in (0.0, 0.05, -0.08):
        (ua, va), (ub, vb) = sorted((uv(lat + 0.06), uv(lat - 0.06)))
        v = 0.5 * (va + vb)
        y1, y2 = v - 50, v + 50
        whole = G.face_center_offset(cam, [ua, y1, ub, y2], 0.40, frame_w=640)
        assert abs(whole - lat) < 0.005, (lat, whole)
        left_cut = G.face_center_offset(cam, [0, y1, ub, y2], 0.40, frame_w=640)
        assert abs(left_cut - lat) < 0.005, (lat, left_cut)
        right_cut = G.face_center_offset(cam, [ua, y1, 640, y2], 0.40, frame_w=640)
        assert abs(right_cut - lat) < 0.005, (lat, right_cut)
