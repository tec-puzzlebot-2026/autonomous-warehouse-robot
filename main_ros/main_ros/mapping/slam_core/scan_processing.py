"""
LiDAR preprocessing.

Each beam i has angle  a_i = angle_min + i * angle_increment  and range r_i.
In the laser frame the endpoint is  p_l = (r_i cos a_i, r_i sin a_i).
The mounting transform (R, t) from the laser frame to base_link gives
    p_b = R p_l + t
so the rest of SLAM works in the robot frame and never assumes the LiDAR sits
at the robot origin.

Beams are split into three groups:
  * hits      – finite range inside [range_min, range_max): an obstacle was seen.
  * no-return – inf or >= range_max: nothing was hit, so the ray is free space
                (used only to clear cells, never to mark obstacles).
  * invalid   – NaN or < range_min: discarded.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PreparedScan:
    hits_base: np.ndarray        # (K, 2) all valid endpoints in base_link [m]
    map_hits_base: np.ndarray    # (K', 2) endpoints used for mapping
    map_misses_base: np.ndarray  # (J, 2) free-space ray ends for no-return beams
    match_base: np.ndarray       # (M, 2) downsampled endpoints for scan matching
    loop_base: np.ndarray        # (P, 2) voxel-downsampled endpoints for loop closure
    sensor_origin_base: np.ndarray  # (2,) LiDAR position in base_link
    n_total: int
    n_valid: int


class ScanPreprocessor:

    def __init__(self, map_beam_step=4, match_beams=180,
                 map_max_range=10.0, match_max_range=10.0,
                 use_no_return_free_space=True,
                 loop_voxel=0.08, loop_max_points=400):
        self.map_beam_step = max(1, int(map_beam_step))
        self.match_beams = int(match_beams)
        self.map_max_range = float(map_max_range)
        self.match_max_range = float(match_max_range)
        self.use_no_return_free_space = use_no_return_free_space
        self.loop_voxel = float(loop_voxel)
        self.loop_max_points = int(loop_max_points)

        # Laser -> base_link transform. Identity until set.
        self.R = np.eye(2)
        self.t = np.zeros(2)

        self._cache_key = None
        self._cos = None
        self._sin = None

    def set_mounting(self, rotation_2x2, translation_xy):
        """Set the laser -> base_link transform (2x2 rotation block + xy offset)."""
        self.R = np.asarray(rotation_2x2, dtype=np.float64)
        self.t = np.asarray(translation_xy, dtype=np.float64)

    def _to_base(self, pts_laser):
        return pts_laser @ self.R.T + self.t

    def prepare(self, ranges, angle_min, angle_increment, range_min, range_max):
        r = np.asarray(ranges, dtype=np.float64)
        n = r.size

        # Cache cos/sin tables: the beam geometry never changes between scans.
        key = (n, angle_min, angle_increment)
        if key != self._cache_key:
            angles = angle_min + angle_increment * np.arange(n)
            self._cos, self._sin = np.cos(angles), np.sin(angles)
            self._cache_key = key

        with np.errstate(invalid="ignore"):
            hit = np.isfinite(r) & (r >= range_min) & (r < range_max)
            no_return = (np.isposinf(r) | (np.isfinite(r) & (r >= range_max)))

        idx_hit = np.flatnonzero(hit)
        rh = r[idx_hit]
        hits_laser = np.column_stack((rh * self._cos[idx_hit], rh * self._sin[idx_hit]))
        hits_base = self._to_base(hits_laser)

        # ---- Mapping beams: every k-th beam, capped range ----
        sel = idx_hit[::self.map_beam_step]
        rm = r[sel]
        keep = rm <= self.map_max_range
        sel_hit, r_hit = sel[keep], rm[keep]
        map_hits = self._to_base(np.column_stack(
            (r_hit * self._cos[sel_hit], r_hit * self._sin[sel_hit])))

        # Long hits beyond map_max_range only clear space up to that range.
        sel_long = sel[~keep]
        miss_list = [sel_long]
        if self.use_no_return_free_space:
            miss_list.append(np.flatnonzero(no_return)[::self.map_beam_step])
        miss_idx = np.concatenate(miss_list)
        free_len = min(self.map_max_range, float(range_max))
        map_misses = self._to_base(np.column_stack(
            (free_len * self._cos[miss_idx], free_len * self._sin[miss_idx])))

        # ---- Matching beams: evenly spread subset of the valid hits ----
        close = idx_hit[rh <= self.match_max_range]
        if close.size > self.match_beams:
            pick = np.linspace(0, close.size - 1, self.match_beams).astype(int)
            close = close[pick]
        rc = r[close]
        match_base = self._to_base(np.column_stack(
            (rc * self._cos[close], rc * self._sin[close])))

        # ---- Loop-closure points: one point per voxel, evenly capped ----
        # Voxel downsampling keeps the shape of the scan but removes the
        # density bias near the sensor, so far walls count as much as near ones.
        loop_src = hits_base[np.hypot(hits_base[:, 0], hits_base[:, 1]) <= self.match_max_range]
        if loop_src.shape[0]:
            keys = np.floor(loop_src / self.loop_voxel).astype(np.int64)
            _, first = np.unique(keys, axis=0, return_index=True)
            loop_pts = loop_src[np.sort(first)]
            if loop_pts.shape[0] > self.loop_max_points:
                pick = np.linspace(0, loop_pts.shape[0] - 1, self.loop_max_points).astype(int)
                loop_pts = loop_pts[pick]
        else:
            loop_pts = np.zeros((0, 2))

        return PreparedScan(
            hits_base=hits_base,
            map_hits_base=map_hits,
            map_misses_base=map_misses,
            match_base=match_base,
            loop_base=loop_pts,
            sensor_origin_base=self.t.copy(),
            n_total=n,
            n_valid=int(idx_hit.size),
        )
