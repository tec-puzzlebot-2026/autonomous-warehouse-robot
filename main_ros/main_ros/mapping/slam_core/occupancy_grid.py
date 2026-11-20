"""
Probabilistic occupancy grid with log-odds updates.

Each cell c stores  L(c) = log( p(c) / (1 - p(c)) ).  With a 0.5 prior (L0 = 0)
the Bayesian update for a measurement z at pose x is additive:

    L_t(c) = L_{t-1}(c) + l_inv(c | z, x)

with the inverse sensor model
    l_inv = l_occ  (> 0)  for the cell containing a beam endpoint,
    l_inv = l_free (< 0)  for cells the beam passed through,
    l_inv = 0             otherwise.
Probability is recovered with p = 1 / (1 + exp(-L)).

Two kinds of grid use this class:
  * the GLOBAL map (clamp=True): values are clamped to [l_min, l_max] so the
    map can still change when an obstacle moves;
  * the LOCAL map of the front-end (clamp=False): a sliding window of recent
    keyframes. Without clamping the update is exactly reversible, so a keyframe
    that leaves the window is removed by applying the same update with the
    opposite sign.

Cell indices are anchored to the world (cell = floor(x / resolution)), so
growing the grid never changes which cell a point falls in. This makes
insert/remove pairs exact.

Each cell is updated at most once per scan, and a cell holding an endpoint of
this scan is never cleared by another beam of the same scan.
Indexing convention: grid[row, col] = grid[iy, ix].
"""

import math

import numpy as np


class OccupancyGrid:

    def __init__(self, resolution=0.05, initial_size_m=20.0,
                 l_free=-0.4, l_occ=0.85, l_min=-5.0, l_max=5.0,
                 occupied_threshold=0.6, grow_margin_m=5.0, clamp=True):
        self.res = float(resolution)
        self.l_free, self.l_occ = float(l_free), float(l_occ)
        self.l_min, self.l_max = float(l_min), float(l_max)
        self.occupied_threshold = float(occupied_threshold)
        self.grow_margin = int(math.ceil(grow_margin_m / self.res))
        self.clamp = bool(clamp)

        n = int(math.ceil(initial_size_m / self.res))
        self.log_odds = np.zeros((n, n), dtype=np.float32)
        self.observed = np.zeros((n, n), dtype=bool)
        # World cell index (ix, iy) of grid[0, 0].
        self.cell_origin = np.array([-(n // 2), -(n // 2)], dtype=np.int64)
        self.version = 0     # increments whenever the map changes

    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.log_odds.shape

    @property
    def origin(self):
        """World coordinates [m] of the lower-left corner of grid[0, 0]."""
        return self.cell_origin.astype(np.float64) * self.res

    def world_to_cell(self, pts):
        """(N, 2) world points -> (ix, iy) indices into this grid."""
        ij = np.floor(np.asarray(pts, dtype=np.float64) / self.res).astype(np.int64)
        return ij[:, 0] - self.cell_origin[0], ij[:, 1] - self.cell_origin[1]

    def _ensure_contains(self, lo, hi):
        h, w = self.shape
        c0 = np.floor(lo / self.res).astype(np.int64) - self.cell_origin
        c1 = np.floor(hi / self.res).astype(np.int64) - self.cell_origin
        pad_l = (self.grow_margin - c0[0]) if c0[0] < 0 else 0
        pad_b = (self.grow_margin - c0[1]) if c0[1] < 0 else 0
        pad_r = (c1[0] - (w - 1) + self.grow_margin) if c1[0] >= w else 0
        pad_t = (c1[1] - (h - 1) + self.grow_margin) if c1[1] >= h else 0
        if pad_l == pad_b == pad_r == pad_t == 0:
            return
        pads = ((int(pad_b), int(pad_t)), (int(pad_l), int(pad_r)))
        self.log_odds = np.pad(self.log_odds, pads)
        self.observed = np.pad(self.observed, pads)
        self.cell_origin = self.cell_origin - np.array([pad_l, pad_b], dtype=np.int64)

    # ------------------------------------------------------------------
    def integrate_scan(self, sensor_xy, hits_world, misses_world, sign=1.0):
        """
        Insert (sign=+1) or remove (sign=-1) one scan.
          sensor_xy    : (2,) LiDAR position in the map frame.
          hits_world   : (K, 2) endpoints that hit an obstacle.
          misses_world : (J, 2) ray ends with no obstacle (free space only).
        """
        hits_world = np.asarray(hits_world, dtype=np.float64).reshape(-1, 2)
        misses_world = np.asarray(misses_world, dtype=np.float64).reshape(-1, 2)
        sensor_xy = np.asarray(sensor_xy, dtype=np.float64)
        ends = np.vstack((hits_world, misses_world))
        if ends.shape[0] == 0:
            return

        all_pts = np.vstack((ends, sensor_xy[None, :]))
        self._ensure_contains(all_pts.min(axis=0), all_pts.max(axis=0))
        h, w = self.shape
        ox, oy = self.cell_origin

        # ---- Free cells: sample each ray at half-cell spacing ----
        # Only the samples each ray really needs are generated.
        d = ends - sensor_xy
        length = np.hypot(d[:, 0], d[:, 1])
        stop = length.copy()
        stop[: hits_world.shape[0]] -= self.res      # stop one cell short of a hit
        step = 0.5 * self.res
        counts = np.maximum(np.ceil(stop / step), 0).astype(np.int64)
        total = int(counts.sum())
        if total > 0:
            ray = np.repeat(np.arange(ends.shape[0]), counts)
            first = np.cumsum(counts) - counts
            t = (np.arange(total) - np.repeat(first, counts)) * step
            unit = d / np.maximum(length, 1e-9)[:, None]
            px = sensor_xy[0] + unit[ray, 0] * t
            py = sensor_xy[1] + unit[ray, 1] * t
            fx = np.floor(px / self.res).astype(np.int64) - ox
            fy = np.floor(py / self.res).astype(np.int64) - oy
            free_lin = fy * w + fx
        else:
            free_lin = np.empty(0, dtype=np.int64)

        # ---- Occupied cells ----
        if hits_world.shape[0]:
            hx, hy = self.world_to_cell(hits_world)
            hit_lin = hy * w + hx
            is_hit = np.zeros(h * w, dtype=bool)
            is_hit[hit_lin] = True
            free_lin = free_lin[~is_hit[free_lin]]
        else:
            hit_lin = np.empty(0, dtype=np.int64)

        # Fancy-index "+=" applies once per distinct cell even if the index
        # repeats, which is exactly the "update each cell once per scan" rule.
        lo = self.log_odds.reshape(-1)
        lo[free_lin] += np.float32(sign * self.l_free)
        lo[hit_lin] += np.float32(sign * self.l_occ)
        if self.clamp:
            touched = np.concatenate((free_lin, hit_lin))
            lo[touched] = np.clip(lo[touched], self.l_min, self.l_max)
        if sign > 0:
            ob = self.observed.reshape(-1)
            ob[free_lin] = True
            ob[hit_lin] = True
        self.version += 1

    # ------------------------------------------------------------------
    def occupied_window(self, center_xy, half_size_m):
        """
        Square window of the occupied mask centred on center_xy.
        Cells outside the grid are treated as not occupied.
        Returns (mask, window_origin_xy).
        """
        half = int(math.ceil(half_size_m / self.res))
        cx, cy = self.world_to_cell(np.asarray(center_xy, dtype=np.float64)[None, :])
        cx, cy = int(cx[0]), int(cy[0])
        x0, y0 = cx - half, cy - half
        size = 2 * half
        mask = np.zeros((size, size), dtype=bool)

        h, w = self.shape
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + size, w), min(y0 + size, h)
        if sx0 < sx1 and sy0 < sy1:
            mask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = (
                self.log_odds[sy0:sy1, sx0:sx1] > self.occupied_threshold)
        origin = (self.cell_origin + np.array([x0, y0])) * self.res
        return mask, origin.astype(np.float64)

    def occupied_count(self):
        return int(np.count_nonzero(self.log_odds > self.occupied_threshold))

    # ------------------------------------------------------------------
    def to_occupancy_values(self, crop=True):
        """
        Convert to ROS OccupancyGrid semantics:
        -1 unknown, 0..100 occupancy probability in percent.
        Returns (int8 array [rows, cols], origin_xy). Cropped to observed cells.
        """
        lo, ob, origin = self.log_odds, self.observed, self.origin
        if crop and ob.any():
            rows = np.flatnonzero(ob.any(axis=1))
            cols = np.flatnonzero(ob.any(axis=0))
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            lo, ob = lo[r0:r1, c0:c1], ob[r0:r1, c0:c1]
            origin = origin + np.array([c0, r0]) * self.res

        prob = 1.0 / (1.0 + np.exp(-lo.astype(np.float64)))
        values = np.round(prob * 100.0).astype(np.int8)
        values[~ob] = -1
        return values, origin

    def reset(self):
        self.log_odds[:] = 0.0
        self.observed[:] = False
        self.version += 1
