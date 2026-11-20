"""
Correlative scan matching with branch-and-bound, for loop-closure verification.

Problem: given a reference point cloud (an old submap) and a query scan, find
the rigid transform T = (x, y, theta) that places the query on the reference,
searching a whole window around an initial guess (the drift can be large, so a
local optimizer that needs a good start is not enough).

Score (Olson, "Real-Time Correlative Scan Matching", ICRA 2009):
the reference points are rasterized and blurred into a lookup table

    S(c) = exp( -d(c)^2 / (2 sigma^2) )      d(c) = distance from cell c to the
                                                    nearest reference point
and a candidate pose is scored by the mean of S over the transformed query
points (1 = every point on a reference point, 0 = nothing matches).

Exhaustive search is exact but slow, so the translation search uses a two-level
branch-and-bound (Hess et al., "Real-Time Loop Closure in 2D LIDAR SLAM",
ICRA 2016):
  * a coarse table C(c) = max of S over an f x f block of cells is an UPPER
    BOUND of the score of every fine translation inside that block;
  * all coarse candidates are scored, sorted by bound, and only expanded to
    fine resolution while their bound can still beat the best fine score.
The result is identical to the exhaustive search at fine resolution.

The angular step follows Hess et al.: the smallest rotation that moves the
farthest point by about one cell, delta = acos(1 - r^2 / (2 d_max^2)).

Uncertainty: the scores of the poses around the best one are turned into
weights and their weighted covariance is the match covariance (Olson 2009).
In a long corridor the score barely changes along the corridor, so the
covariance becomes long in that direction and the loop edge is weak there,
which is exactly the aperture problem handled correctly.
"""

from dataclasses import dataclass
import math

import numpy as np

from .likelihood_field import bounded_distance_transform


@dataclass
class MatchResult:
    pose: np.ndarray          # (3,) transform query -> reference
    score: float              # mean score at the best pose, 0..1
    inlier_ratio: float       # fraction of points within 2 sigma of a reference point
    cov: np.ndarray           # (3, 3) covariance of the pose
    candidates_expanded: int


class CorrelativeScanMatcher:

    def __init__(self, resolution=0.05, sigma=0.05, coarse_factor=4,
                 cov_temperature=0.02):
        self.res = float(resolution)
        self.sigma = float(sigma)
        self.f = int(coarse_factor)
        self.cov_temperature = float(cov_temperature)

    # ------------------------------------------------------------------
    def _tables(self, ref, margin_m):
        res = self.res
        lo = ref.min(axis=0) - margin_m
        hi = ref.max(axis=0) + margin_m
        origin = np.floor(lo / res) * res
        w = int(math.ceil((hi[0] - origin[0]) / res)) + 2
        h = int(math.ceil((hi[1] - origin[1]) / res)) + 2
        occ = np.zeros((h, w), dtype=bool)
        ix = np.floor((ref[:, 0] - origin[0]) / res).astype(np.int64)
        iy = np.floor((ref[:, 1] - origin[1]) / res).astype(np.int64)
        occ[iy, ix] = True
        R = max(1, int(math.ceil(3.0 * self.sigma / res)))
        d = bounded_distance_transform(occ, R) * res
        fine = np.exp(-d ** 2 / (2.0 * self.sigma ** 2)).astype(np.float32)
        fine[d > 3.0 * self.sigma] = 0.0
        fine[0, :] = fine[-1, :] = 0.0       # zero border: clipped indices land here
        fine[:, 0] = fine[:, -1] = 0.0

        # coarse[y, x] = max(fine[y:y+f, x:x+f])  -> upper bound for a block
        f = self.f
        mx = fine.copy()
        for s in range(1, f):
            mx[:, :-s] = np.maximum(mx[:, :-s], fine[:, s:])
        coarse = mx.copy()
        for s in range(1, f):
            coarse[:-s, :] = np.maximum(coarse[:-s, :], mx[s:, :])
        return fine, coarse, origin

    # ------------------------------------------------------------------
    def match(self, ref_points, query_points, guess, lin_window, ang_window):
        """
        ref_points   : (R, 2) reference cloud in the reference frame.
        query_points : (M, 2) query scan in its own frame.
        guess        : (3,) initial transform query -> reference.
        lin_window   : search half-width in x and y [m].
        ang_window   : search half-width in theta [rad].
        Returns MatchResult, or None if there is nothing to match.
        """
        ref = np.asarray(ref_points, dtype=np.float64)
        q = np.asarray(query_points, dtype=np.float64)
        if ref.shape[0] < 10 or q.shape[0] < 10:
            return None
        res, f = self.res, self.f
        fine, coarse, origin = self._tables(ref, 3.0 * self.sigma + res)
        h, w = fine.shape
        fine_flat, coarse_flat = fine.ravel(), coarse.ravel()

        r_max = max(float(np.max(np.hypot(q[:, 0], q[:, 1]))), 4.0 * res)
        dth = math.acos(max(-1.0, 1.0 - res ** 2 / (2.0 * r_max ** 2)))
        dth = min(max(dth, math.radians(0.1)), math.radians(1.0))
        n_ang = int(math.ceil(ang_window / dth))
        angles = guess[2] + dth * np.arange(-n_ang, n_ang + 1)
        W = int(math.ceil(lin_window / res))
        coarse_off = np.arange(-W, W + 1, f)

        # Base cells of the query for every rotation (translation = guess).
        cells = []
        for th in angles:
            c, s = math.cos(th), math.sin(th)
            px = guess[0] + c * q[:, 0] - s * q[:, 1]
            py = guess[1] + s * q[:, 0] + c * q[:, 1]
            cells.append((np.floor((px - origin[0]) / res).astype(np.int64),
                          np.floor((py - origin[1]) / res).astype(np.int64)))

        def score(table, k, tx, ty):
            """Mean table value for rotation k and integer translations (arrays)."""
            cx, cy = cells[k]
            ix = np.clip(cx[None, :] + tx[:, None], 0, w - 1)
            iy = np.clip(cy[None, :] + ty[:, None], 0, h - 1)
            return table[iy * w + ix].mean(axis=1)

        # ---- Level 1: upper bounds of every coarse block ----
        gx, gy = np.meshgrid(coarse_off, coarse_off)
        gx, gy = gx.ravel(), gy.ravel()
        bounds = np.empty((len(angles), gx.size), dtype=np.float32)
        for k in range(len(angles)):
            bounds[k] = score(coarse_flat, k, gx, gy)
        order = np.argsort(bounds, axis=None)[::-1]

        # ---- Level 2: expand blocks while they can still win ----
        fo = np.arange(f)
        fx, fy = np.meshgrid(fo, fo)
        fx, fy = fx.ravel(), fy.ravel()
        best = (-1.0, 0, 0, 0)
        expanded = 0
        for flat in order:
            k, t = divmod(int(flat), gx.size)
            if bounds[k, t] <= best[0]:
                break
            tx, ty = gx[t] + fx, gy[t] + fy
            keep = (tx <= W) & (ty <= W)
            tx, ty = tx[keep], ty[keep]
            sc = score(fine_flat, k, tx, ty)
            i = int(np.argmax(sc))
            expanded += 1
            if sc[i] > best[0]:
                best = (float(sc[i]), k, int(tx[i]), int(ty[i]))

        s_best, kb, txb, tyb = best
        pose = np.array([guess[0] + txb * res, guess[1] + tyb * res, angles[kb]])

        # ---- Inliers at the best pose ----
        cx, cy = cells[kb]
        ix = np.clip(cx + txb, 0, w - 1)
        iy = np.clip(cy + tyb, 0, h - 1)
        vals = fine_flat[iy * w + ix]
        inlier = float(np.mean(vals >= math.exp(-2.0)))   # within 2 sigma

        # ---- Covariance from the score landscape around the best pose ----
        samples, weights = [], []
        dk = np.arange(max(0, kb - 3), min(len(angles), kb + 4))
        off = np.arange(-6, 7)
        ox, oy = np.meshgrid(off, off)
        ox, oy = ox.ravel() + txb, oy.ravel() + tyb
        for k in dk:
            sc = score(fine_flat, int(k), ox, oy)
            samples.append(np.column_stack((guess[0] + ox * res, guess[1] + oy * res,
                                            np.full(ox.size, angles[k]))))
            weights.append(sc)
        P = np.vstack(samples)
        wts = np.exp((np.concatenate(weights) - s_best) / self.cov_temperature)
        wts /= wts.sum()
        mean = wts @ P
        D = P - mean
        cov = (D * wts[:, None]).T @ D
        return MatchResult(pose, s_best, inlier, cov, expanded)
