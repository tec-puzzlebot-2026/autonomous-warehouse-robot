"""
landmarks.py — ArUco markers and company logos placed on the SLAM map (no ROS).

During mapping every sighting is turned into a map position with the robot's SLAM
pose at that moment. Sightings are noisy (detector, timing, pose error), so each
marker / logo keeps all of its sightings and its estimate is a robust average:
the median first, then the mean of the sightings close to it.
"""
# Author: Raymundo Gomez

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Landmark:
    """Robust position estimate from many sightings of the same thing."""
    key: object                                   # ArUco ID or company name
    points: list = field(default_factory=list)    # (x, y) per sighting
    yaws: list = field(default_factory=list)      # facing direction per sighting (or None)
    inlier_radius: float = 0.15

    def add(self, x, y, yaw=None):
        self.points.append((float(x), float(y)))
        self.yaws.append(None if yaw is None else float(yaw))

    @property
    def count(self):
        return len(self.points)

    def estimate(self):
        """(x, y, yaw or None, n_inliers, spread_m) or None without sightings."""
        if not self.points:
            return None
        p = np.array(self.points)
        med = np.median(p, axis=0)
        d = np.hypot(*(p - med).T)
        inl = d <= self.inlier_radius
        if not inl.any():
            inl = d <= d.min() + 1e-9
        x, y = p[inl].mean(axis=0)
        spread = float(np.hypot(*(p[inl] - (x, y)).T).std()) if inl.sum() > 1 else 0.0
        yaws = [a for a, ok in zip(self.yaws, inl) if ok and a is not None]
        yaw = None
        if yaws:
            yaw = math.atan2(sum(math.sin(a) for a in yaws), sum(math.cos(a) for a in yaws))
        return float(x), float(y), yaw, int(inl.sum()), spread


class LandmarkSet:
    """All landmarks of one kind (markers or logos), keyed by ID or name."""

    def __init__(self, min_count=5, inlier_radius=0.15):
        self.min_count = min_count
        self.inlier_radius = inlier_radius
        self.items = {}

    def add(self, key, x, y, yaw=None):
        lm = self.items.get(key)
        if lm is None:
            lm = self.items[key] = Landmark(key, inlier_radius=self.inlier_radius)
        lm.add(x, y, yaw)

    def confirmed(self):
        """{key: (x, y, yaw, n, spread)} of the landmarks seen often enough."""
        out = {}
        for k, lm in self.items.items():
            e = lm.estimate()
            if e is not None and e[3] >= self.min_count:
                out[k] = e
        return out

    def to_list(self):
        """For the mission folder's YAML files."""
        out = []
        for k, (x, y, yaw, n, spread) in sorted(self.confirmed().items(), key=lambda i: str(i[0])):
            out.append({'id': k, 'x': round(x, 4), 'y': round(y, 4),
                        'yaw': None if yaw is None else round(yaw, 4),
                        'sightings': n, 'spread_m': round(spread, 4)})
        return out


def marker_yaw_from_optical(q_xyzw, T_base_optical, robot_yaw):
    """Direction the marker's printed side faces, in the map frame.

    The detector gives the marker's orientation in the optical frame; its +z axis
    points out of the printed side (toward the camera). Rotated into base_link and
    then by the robot's heading, its horizontal direction is the facing yaw.
    """
    x, y, z, w = q_xyzw
    # Rotation matrix of the quaternion, third column = marker z axis.
    zc = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    zb = T_base_optical[:3, :3] @ zc
    return math.atan2(math.sin(robot_yaw + math.atan2(zb[1], zb[0])),
                      math.cos(robot_yaw + math.atan2(zb[1], zb[0])))
