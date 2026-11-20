"""
Time alignment between LiDAR scans and wheel odometry.

Odometry arrives at a different rate than the LiDAR. For each scan we need the
odometry pose at the scan's own timestamp t_s, so we keep a short history and
interpolate between the two samples (t_a, p_a) and (t_b, p_b) that bracket t_s:

    s = (t_s - t_a) / (t_b - t_a)
    x = x_a + s (x_b - x_a),  y likewise
    theta = theta_a + s * wrap(theta_b - theta_a)

If the scan is slightly newer than the newest odometry sample we use the newest
sample (no extrapolation) as long as the gap is small; otherwise the scan is
rejected, because inserting it at a wrong pose would corrupt the map.
"""

from collections import deque
import bisect

import numpy as np

from .se2 import wrap_angle


class OdometryBuffer:

    def __init__(self, history_s=2.0, max_gap_s=0.15):
        self.history_s = history_s
        self.max_gap_s = max_gap_s
        self._t = deque()
        self._p = deque()

    def __len__(self):
        return len(self._t)

    def add(self, t, pose):
        # Drop out-of-order samples (e.g. after a /clock reset).
        if self._t and t <= self._t[-1]:
            if t < self._t[-1] - 1.0:   # time jumped backwards: simulation reset
                self.clear()
            else:
                return
        self._t.append(t)
        self._p.append(np.asarray(pose, dtype=np.float64))
        while self._t and self._t[-1] - self._t[0] > self.history_s:
            self._t.popleft()
            self._p.popleft()

    def clear(self):
        self._t.clear()
        self._p.clear()

    def latest(self):
        if not self._t:
            return None, None
        return self._t[-1], self._p[-1]

    def pose_at(self, t):
        """Interpolated odometry pose at time t, or None if unavailable."""
        if not self._t:
            return None
        ts = self._t
        if t >= ts[-1]:
            return self._p[-1].copy() if t - ts[-1] <= self.max_gap_s else None
        if t <= ts[0]:
            return self._p[0].copy() if ts[0] - t <= self.max_gap_s else None

        i = bisect.bisect_right(ts, t)
        ta, tb = ts[i - 1], ts[i]
        pa, pb = self._p[i - 1], self._p[i]
        if tb - ta > 4.0 * self.max_gap_s:   # hole in the odometry stream
            return None
        s = (t - ta) / (tb - ta)
        out = pa + s * (pb - pa)
        out[2] = float(wrap_angle(pa[2] + s * wrap_angle(pb[2] - pa[2])))
        return out
