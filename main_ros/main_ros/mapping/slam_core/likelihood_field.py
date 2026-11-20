"""
Likelihood-field sensor model (Thrun, Burgard & Fox, Probabilistic Robotics, ch. 6.4).

Instead of ray casting into the map for every particle, each beam endpoint is
projected into the map and scored by its distance d to the nearest occupied cell:

    p(z_j | x, m) = z_hit * exp(-d_j^2 / (2 sigma^2)) + z_rand

    log p(z | x, m) = sum_j log p(z_j | x, m)        (beams assumed independent)

The distances come from a Euclidean distance transform of the occupied cells,
computed once per map change (not once per particle), so scoring N particles
with M beams is just N*M table lookups.

The distance transform is our own bounded, exact, separable implementation:
  pass 1 (columns): g(x, y) = min |dy| such that cell (x, y+dy) is occupied
  pass 2 (rows):    D(x, y)^2 = min_dx  dx^2 + g(x+dx, y)^2
Distances are only needed up to R cells (beyond ~4 sigma the score is flat),
so each pass is 2R array shifts.
"""

import math

import numpy as np


def bounded_distance_transform(occupied, max_cells):
    """Exact Euclidean distance (in cells) to the nearest True cell, clipped at max_cells."""
    R = int(max_cells)
    far = float(R + 1)
    g = np.where(occupied, 0.0, far).astype(np.float32)

    for d in range(1, R + 1):
        dd = np.float32(d)
        g[:-d, :] = np.minimum(g[:-d, :], np.where(occupied[d:, :], dd, far))
        g[d:, :] = np.minimum(g[d:, :], np.where(occupied[:-d, :], dd, far))

    g2 = g * g
    D2 = g2.copy()
    for d in range(1, R + 1):
        dd = np.float32(d * d)
        D2[:, :-d] = np.minimum(D2[:, :-d], g2[:, d:] + dd)
        D2[:, d:] = np.minimum(D2[:, d:], g2[:, :-d] + dd)
    return np.minimum(np.sqrt(D2), far)


class LikelihoodField:

    def __init__(self, occupied_mask, origin_xy, resolution,
                 sigma_hit=0.1, z_hit=0.9, z_rand=0.1, max_dist=0.5):
        self.res = float(resolution)
        self.origin = np.asarray(origin_xy, dtype=np.float64)
        self.shape = occupied_mask.shape
        self.max_dist = float(max_dist)

        R = max(1, int(math.ceil(max_dist / self.res)))
        dist = bounded_distance_transform(occupied_mask, R) * self.res
        dist = np.minimum(dist, self.max_dist)
        self.dist = dist
        # Per-cell log-likelihood table.
        self.log_p = np.log(z_hit * np.exp(-dist ** 2 / (2.0 * sigma_hit ** 2)) + z_rand
                            ).astype(np.float32)
        self.log_p_far = float(np.log(
            z_hit * math.exp(-self.max_dist ** 2 / (2.0 * sigma_hit ** 2)) + z_rand))
        self.center = self.origin + np.array(self.shape[::-1]) * self.res / 2.0

    def _lookup(self, table, far_value, wx, wy):
        ix = np.floor((wx - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((wy - self.origin[1]) / self.res).astype(np.int64)
        h, w = self.shape
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        out = np.full(wx.shape, far_value, dtype=np.float32)
        out[inside] = table[iy[inside], ix[inside]]
        return out

    def score_particles(self, particles, points_base):
        """
        log p(z | x_i, m) for every particle.
          particles   : (N, 3) poses [x, y, theta] in the map frame.
          points_base : (M, 2) beam endpoints in base_link.
        Returns (N,) log-likelihoods.
        """
        c = np.cos(particles[:, 2])[:, None]
        s = np.sin(particles[:, 2])[:, None]
        px, py = points_base[None, :, 0], points_base[None, :, 1]
        wx = particles[:, 0:1] + c * px - s * py          # (N, M)
        wy = particles[:, 1:2] + s * px + c * py
        return self._lookup(self.log_p, self.log_p_far, wx, wy).sum(axis=1)

    def match_ratio(self, pose, points_base, tol):
        """Fraction of endpoints within tol metres of a mapped obstacle (quality metric)."""
        if points_base.shape[0] == 0:
            return 0.0
        c, s = math.cos(pose[2]), math.sin(pose[2])
        wx = pose[0] + c * points_base[:, 0] - s * points_base[:, 1]
        wy = pose[1] + s * points_base[:, 0] + c * points_base[:, 1]
        d = self._lookup(self.dist, self.max_dist, wx, wy)
        return float(np.mean(d <= tol))
