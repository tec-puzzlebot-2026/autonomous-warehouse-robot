"""
zones.py — the area in front of every ArUco marker, labeled (no ROS)
=====================================================================
Every ArUco marker found on the map marks a square zone on the floor in front of
it (the side its printed face looks toward). The zone's label comes from the
marker's ID: by default ID 21 is the DROP ZONE (where restocking sets boxes down
when it does not use a company dock), every other marker is just "aruco_<id>".

The square (size_m, default 1 x 1 m) starts at the wall the marker hangs on and
reaches size_m into the arena. It is slid along the wall (and a little away from
it) until as much of it as possible lies on known free floor of the map, so it
stays inside the arena, e.g. when the marker is close to a corner.

    zones = zones_from_markers(grid, markers, ZoneConfig())
    zone.contains(x, y), zone.corners, zone.center, zone.to_dict()

Used by landmark_mapper_node (live, shown in Foxglove), the mission folder
(zones.yaml, mission_store.py) and restocking (restock_core.py: the drop zone).
"""
# Author: Raymundo Gomez

import math
from dataclasses import dataclass, field

import numpy as np

DROP_ZONE = 'drop_zone'


@dataclass
class ZoneConfig:
    """config/restock.yaml, section zones."""
    size_m: float = 1.0                                       # side of the square
    labels: dict = field(default_factory=lambda: {21: DROP_ZONE})   # ArUco id -> label
    max_wall_gap_m: float = 0.25        # the square may move this far off the wall

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        cfg = cls(**{k: v for k, v in d.items() if k in ('size_m', 'max_wall_gap_m')})
        if d.get('labels') is not None:
            cfg.labels = {int(k): str(v) for k, v in dict(d['labels']).items()}
        return cfg

    def label(self, marker_id):
        return self.labels.get(int(marker_id), f'aruco_{int(marker_id)}')


@dataclass
class Zone:
    label: str
    marker_id: int
    center: tuple                  # (x, y) on the map
    yaw: float                     # direction from the wall into the zone (the marker faces)
    size: float
    inside: float = 1.0            # fraction of the square on known free floor

    @property
    def corners(self):
        h = self.size / 2.0
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        out = []
        for a, b in ((-h, -h), (h, -h), (h, h), (-h, h)):
            out.append((self.center[0] + a * c - b * s, self.center[1] + a * s + b * c))
        return out

    def contains(self, x, y, margin=0.0):
        dx, dy = x - self.center[0], y - self.center[1]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        a, b = dx * c + dy * s, -dx * s + dy * c
        h = self.size / 2.0 - margin
        return abs(a) <= h and abs(b) <= h

    def to_dict(self):
        return {'label': self.label, 'marker_id': int(self.marker_id),
                'x': round(float(self.center[0]), 3), 'y': round(float(self.center[1]), 3),
                'yaw': round(float(self.yaw), 4), 'size': round(float(self.size), 3),
                'inside': round(float(self.inside), 3),
                'corners': [[round(float(x), 3), round(float(y), 3)] for x, y in self.corners]}

    @classmethod
    def from_dict(cls, d):
        return cls(str(d['label']), int(d['marker_id']), (float(d['x']), float(d['y'])),
                   float(d['yaw']), float(d['size']), float(d.get('inside', 1.0)))


def open_direction(grid, x, y, reach=0.5):
    """Direction the wall at (x, y) faces: toward the free floor around it."""
    free = grid.free()

    def count(a):
        n = 0
        for i in range(1, int(reach / grid.res) + 1):
            r, c = grid.world_to_cell(x + i * grid.res * math.cos(a),
                                      y + i * grid.res * math.sin(a))
            n += bool(grid.inside(r, c) and free[r, c])
        return n

    vx = vy = 0.0                    # free floor around, as a vector: points off the wall
    for k in range(32):
        a = k * math.pi / 16
        n = count(a)
        vx += n * math.cos(a)
        vy += n * math.sin(a)
    return math.atan2(vy, vx)


def _inside_fraction(grid, free, cx, cy, yaw, size):
    """Fraction of the square's cells that are known free floor of the map."""
    h = size / 2.0
    step = grid.res
    n = max(2, int(round(size / step)))
    c, s = math.cos(yaw), math.sin(yaw)
    good = total = 0
    for i in range(n):
        a = -h + (i + 0.5) * size / n
        for j in range(n):
            b = -h + (j + 0.5) * size / n
            r, col = grid.world_to_cell(cx + a * c - b * s, cy + a * s + b * c)
            total += 1
            good += bool(grid.inside(r, col) and free[r, col])
    return good / max(total, 1)


def place_zone(grid, marker_id, x, y, yaw, cfg=None):
    """The zone of one marker (x, y on the wall, yaw = the direction it faces)."""
    cfg = cfg or ZoneConfig()
    if yaw is None:
        yaw = open_direction(grid, x, y)
    size = float(cfg.size_m)
    nx, ny = math.cos(yaw), math.sin(yaw)          # out of the wall
    tx, ty = -ny, nx                               # along the wall
    free = grid.free()
    best = None
    step = max(grid.res, 0.05)
    gaps = np.arange(0.0, cfg.max_wall_gap_m + 1e-9, step)
    shifts = np.arange(-size / 2.0, size / 2.0 + 1e-9, step)
    for gap in gaps:
        for sh in shifts:
            cx = x + (gap + size / 2.0) * nx + sh * tx
            cy = y + (gap + size / 2.0) * ny + sh * ty
            f = _inside_fraction(grid, free, cx, cy, yaw, size)
            key = (round(f, 3), -abs(sh), -gap)      # most inside, then centered, then close
            if best is None or key > best[0]:
                best = (key, cx, cy, f)
    _, cx, cy, f = best
    return Zone(cfg.label(marker_id), int(marker_id), (cx, cy), float(yaw), size, f)


def zones_from_markers(grid, markers, cfg=None):
    """[Zone] for the markers [{'id', 'x', 'y', 'yaw'}] of the landmark mapper."""
    cfg = cfg or ZoneConfig()
    return [place_zone(grid, int(m['id']), float(m['x']), float(m['y']),
                       None if m.get('yaw') is None else float(m['yaw']), cfg)
            for m in markers]


def drop_zone(zones):
    """The drop zone among `zones` (the first one labeled drop_zone), or None."""
    return next((z for z in zones if z.label == DROP_ZONE), None)
