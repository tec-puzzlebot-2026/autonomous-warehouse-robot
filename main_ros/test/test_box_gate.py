"""The box detectors' gate (no model needed): a confident detection whose middle is
carton-coloured is a box; a white wall or an ArUco marker is not.

Run:  cd main_ros && python3 -m pytest test/test_box_gate.py -v
"""
# Author: Raymundo Gomez

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from main_ros.vision.boxes import near_box as nb  # noqa: E402

CARTON = (60, 140, 200)          # BGR: cardboard brown


def frame():
    img = np.full((480, 640, 3), 245, np.uint8)               # white wall
    img[200:340, 100:220] = CARTON                            # the box
    img[240:280, 140:180] = 0                                 # its QR code (black)
    img[100:180, 400:480] = 255                               # an ArUco: white ...
    img[115:165, 415:465] = 0                                 # ... and black
    return img


def detector(**s):
    det = nb.NearBoxDetector.__new__(nb.NearBoxDetector)
    det.s = dict(nb.DEFAULTS, **s)
    return det


def test_only_the_carton_passes_the_colour_gate():
    img = frame()
    dets = [{'cls': 0, 'conf': 0.95, 'box': [100, 200, 220, 340]},     # box (QR in the middle)
            {'cls': 0, 'conf': 0.95, 'box': [300, 50, 620, 460]},      # white wall
            {'cls': 0, 'conf': 0.95, 'box': [400, 100, 480, 180]}]     # ArUco
    kept = detector(min_color_fraction=0.15)._gate(img, dets)
    assert [d['box'][0] for d in kept] == [100]


def test_low_confidence_is_dropped_and_the_gate_can_be_off():
    img = frame()
    dets = [{'cls': 0, 'conf': 0.5, 'box': [100, 200, 220, 340]},
            {'cls': 0, 'conf': 0.95, 'box': [300, 50, 620, 460]}]
    assert detector(conf=0.9, min_color_fraction=0.15)._gate(img, dets) == []
    assert len(detector(conf=0.4, min_color_fraction=0.0)._gate(img, dets)) == 2
