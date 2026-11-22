"""
grid.py — the occupancy map as the restocking planners see it (no ROS).

Holds a ROS-style occupancy grid (row 0 at the bottom, values -1 unknown / 0..100)
and answers the questions the planners ask: which cells are free, occupied or
unknown, where the robot fits, what is visible from where, and how far it is to
drive between two places.

Built from a nav_msgs/OccupancyGrid message (from_msg_fields), from the map files
Planisferio saves (from_files: map_server YAML + PGM), or from an array (tests).
"""
# Author: Raymundo Gomez

import heapq
import math
import os

import numpy as np

try:
    from scipy import ndimage as _ndi
except ImportError:                      # the Jetson may lack it: a slower fallback is used
    _ndi = None


def _read_pgm(path):
    """Binary PGM (P5, 8 bit) -> uint8 array, rows top-down as in the file."""
    with open(path, 'rb') as f:
        data = f.read()
    tokens, i = [], 0
    while len(tokens) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b'#':
            while data[i:i + 1] not in (b'\n', b''):
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    if tokens[0] != b'P5':
        raise ValueError(f'{path}: only binary PGM (P5) is supported')
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    if maxval > 255:
        raise ValueError(f'{path}: 16-bit PGM is not supported')
    i += 1                                  # the single whitespace after maxval
    return np.frombuffer(data[i:i + w * h], dtype=np.uint8).reshape(h, w)


def _distance_cells(mask):
    """Euclidean distance (in cells) from every cell to the nearest True cell of mask."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    if _ndi is not None:
        return _ndi.distance_transform_edt(~mask)
    # Fallback: two-pass chamfer (3-4) distance, accurate to a few percent.
    big = 10 ** 9
    d = np.where(mask, 0, big).astype(np.int64)
    h, w = d.shape
    for y in range(h):
        for x in range(w):
            v = d[y, x]
            if y > 0:
                v = min(v, d[y - 1, x] + 3)
                if x > 0:
                    v = min(v, d[y - 1, x - 1] + 4)
                if x < w - 1:
                    v = min(v, d[y - 1, x + 1] + 4)
            if x > 0:
                v = min(v, d[y, x - 1] + 3)
            d[y, x] = v
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            v = d[y, x]
            if y < h - 1:
                v = min(v, d[y + 1, x] + 3)
                if x > 0:
                    v = min(v, d[y + 1, x - 1] + 4)
                if x < w - 1:
                    v = min(v, d[y + 1, x + 1] + 4)
            if x < w - 1:
                v = min(v, d[y, x + 1] + 3)
            d[y, x] = v
    return d / 3.0


def _label(mask):
    """Connected components (8-connected) of a boolean mask -> (labels, count)."""
    if _ndi is not None:
        return _ndi.label(mask, structure=np.ones((3, 3), dtype=int))
    labels = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    h, w = mask.shape
    for sy, sx in zip(*np.nonzero(mask)):
        if labels[sy, sx]:
            continue
        n += 1
        labels[sy, sx] = n
        stack = [(sy, sx)]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not labels[yy, xx]:
                        labels[yy, xx] = n
                        stack.append((yy, xx))
    return labels, n


class OccupancyMap:
    """Occupancy grid with origin (x, y) of cell (0, 0)'s corner and resolution [m]."""

    def __init__(self, values, origin, resolution, occupied_min=65, free_max=25):
        self.values = np.asarray(values, dtype=np.int16)
        self.origin = (float(origin[0]), float(origin[1]))
        self.res = float(resolution)
        self.occupied_min = occupied_min
        self.free_max = free_max
        self.h, self.w = self.values.shape

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_msg_fields(cls, width, height, resolution, origin_x, origin_y, data, **kw):
        """From the fields of a nav_msgs/OccupancyGrid (data in row-major order)."""
        values = np.asarray(data, dtype=np.int16).reshape(int(height), int(width))
        return cls(values, (origin_x, origin_y), resolution, **kw)

    @classmethod
    def from_files(cls, yaml_path, **kw):
        """From a map_server map (YAML + PGM), as Planisferio saves it."""
        import yaml
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
        img = np.flipud(_read_pgm(img_path)).astype(np.int16)
        if int(meta.get('negate', 0)):
            img = 255 - img
        occ_t = float(meta.get('occupied_thresh', 0.65))
        free_t = float(meta.get('free_thresh', 0.25))
        p = (255.0 - img) / 255.0              # probability of being occupied
        values = np.full(img.shape, -1, dtype=np.int16)
        values[p >= occ_t] = 100
        values[p <= free_t] = 0
        return cls(values, meta['origin'][:2], float(meta['resolution']), **kw)

    # ---------------------------------------------------------------- cells
    def world_to_cell(self, x, y):
        """Map point -> (row, col); may be outside the grid."""
        return (int(math.floor((y - self.origin[1]) / self.res)),
                int(math.floor((x - self.origin[0]) / self.res)))

    def cell_to_world(self, row, col):
        """Center of a cell in map coordinates."""
        return (self.origin[0] + (col + 0.5) * self.res,
                self.origin[1] + (row + 0.5) * self.res)

    def inside(self, row, col):
        return 0 <= row < self.h and 0 <= col < self.w

    def occupied(self):
        return self.values >= self.occupied_min

    def free(self):
        return (self.values >= 0) & (self.values <= self.free_max)

    def unknown(self):
        return self.values < 0

    def value_at(self, x, y):
        r, c = self.world_to_cell(x, y)
        return int(self.values[r, c]) if self.inside(r, c) else -1

    # ---------------------------------------------------------------- derived maps
    def obstacles(self, min_cells=3):
        """Occupied cells, without isolated specks smaller than min_cells (LiDAR noise)."""
        occ = self.occupied()
        if min_cells <= 1:
            return occ
        labels, n = _label(occ)
        if n == 0:
            return occ
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_cells
        keep[0] = False
        return keep[labels]

    def clearance(self, treat_unknown_as_obstacle=True):
        """Distance [m] from each cell to the nearest obstacle (and unknown cell)."""
        blocked = self.occupied()
        if treat_unknown_as_obstacle:
            blocked = blocked | self.unknown()
        return _distance_cells(blocked) * self.res

    def traversable(self, robot_radius, margin=0.02):
        """Free cells where a round robot of robot_radius fits."""
        return self.free() & (self.clearance() > robot_radius + margin)

    # ---------------------------------------------------------------- geometry
    def ray_cells(self, r0, c0, r1, c1):
        """Cells on the segment between two cells (Bresenham), both ends included."""
        cells = []
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
        err = dc - dr
        r, c = r0, c0
        while True:
            cells.append((r, c))
            if r == r1 and c == c1:
                return cells
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

    def line_of_sight(self, a, b, blocked=None, ignore_last=1):
        """True if nothing blocks the straight line between map points a and b.

        blocked: boolean grid of blocking cells (default: occupied cells). The last
        `ignore_last` cells (the target itself) do not count as blocking.
        """
        if blocked is None:
            blocked = self.occupied()
        r0, c0 = self.world_to_cell(*a)
        r1, c1 = self.world_to_cell(*b)
        cells = self.ray_cells(r0, c0, r1, c1)
        if ignore_last:
            cells = cells[:-ignore_last]
        for r, c in cells[1:]:
            if not self.inside(r, c) or blocked[r, c]:
                return False
        return True

    def first_hit(self, origin, bearing, max_range):
        """First obstacle cell along a ray from origin (map point) -> (x, y) or None."""
        occ = self.occupied()
        x0, y0 = origin
        x1 = x0 + max_range * math.cos(bearing)
        y1 = y0 + max_range * math.sin(bearing)
        r0, c0 = self.world_to_cell(x0, y0)
        r1, c1 = self.world_to_cell(x1, y1)
        for r, c in self.ray_cells(r0, c0, r1, c1):
            if not self.inside(r, c):
                return None
            if occ[r, c]:
                return self.cell_to_world(r, c)
        return None

    # ---------------------------------------------------------------- paths
    def path_lengths(self, start, passable):
        """Shortest driving distance [m] from map point `start` to every cell (8-connected).

        Cells that are not passable are inf. Returns a float array like values.
        """
        dist = np.full(self.values.shape, np.inf)
        r, c = self.world_to_cell(*start)
        if not self.inside(r, c):
            return dist
        if not passable[r, c]:
            # Start from the nearest passable cell (the robot may stand close to a wall).
            rr, cc = np.nonzero(passable)
            if rr.size == 0:
                return dist
            k = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
            r, c = int(rr[k]), int(cc[k])
        dist[r, c] = 0.0
        heap = [(0.0, r, c)]
        diag = math.sqrt(2.0) * self.res
        steps = [(-1, 0, self.res), (1, 0, self.res), (0, -1, self.res), (0, 1, self.res),
                 (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)]
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc, w in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.h and 0 <= cc < self.w and passable[rr, cc]:
                    nd = d + w
                    if nd < dist[rr, cc]:
                        dist[rr, cc] = nd
                        heapq.heappush(heap, (nd, rr, cc))
        return dist

    def bounds(self):
        """(x_min, x_max, y_min, y_max) of the known part of the map."""
        known = ~self.unknown()
        if not known.any():
            return (self.origin[0], self.origin[0], self.origin[1], self.origin[1])
        rr, cc = np.nonzero(known)
        x0, y0 = self.cell_to_world(rr.min(), cc.min())
        x1, y1 = self.cell_to_world(rr.max(), cc.max())
        return (x0, x1, y0, y1)
