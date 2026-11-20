"""
Monte Carlo pose estimation (particle filter) against the evolving map.

Each particle i is a pose hypothesis x_i = [x, y, theta] with log-weight log w_i.

1. PREDICT  — sample_motion_model_odometry (Probabilistic Robotics, Table 5.6).
   The odometry increment between two odometry poses is decomposed into
       rot1  = atan2(dy, dx)         (turn towards the motion)
       trans = sqrt(dx^2 + dy^2)     (drive)
       rot2  = dtheta - rot1         (final turn)
   and every particle receives its own noisy copy:
       rot1^  = rot1  - N(0, a1 rot1^2  + a2 trans^2)
       trans^ = trans - N(0, a3 trans^2 + a4 (rot1^2 + rot2^2))
       rot2^  = rot2  - N(0, a1 rot2^2  + a2 trans^2)
   Backward motion is handled by flipping the sign of trans.

2. UPDATE   — log w_i += beta * log p(z | x_i, m)   (likelihood field).
   beta < 1 ("tempering") compensates for neighbouring beams not being truly
   independent; without it the weights become so peaked that a single particle
   takes all the mass after one scan.

3. RESAMPLE — systematic (low-variance) resampling, only when the effective
   sample size  N_eff = 1 / sum(w_i^2)  drops below a fraction of N.

4. ESTIMATE — weighted mean of x and y, circular mean of theta,
   weighted covariance of the particle cloud.

Limitation (stated on purpose): all particles share one map. This is an
approximate single-map particle-filter SLAM, not Rao-Blackwellized SLAM.
"""

import math

import numpy as np

from .se2 import relative, wrap_angle


class ParticleFilter:

    def __init__(self, num_particles=200, alphas=(0.05, 0.01, 0.05, 0.01),
                 min_trans_std=0.005, min_rot_std=0.005,
                 resample_ratio=0.5, beta=0.1, seed=None):
        self.n = int(num_particles)
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.min_trans_std = min_trans_std
        self.min_rot_std = min_rot_std
        self.resample_ratio = resample_ratio
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.particles = np.zeros((self.n, 3))
        self.log_w = np.full(self.n, -math.log(self.n))

    # ------------------------------------------------------------------
    def initialize(self, pose, std=(0.0, 0.0, 0.0)):
        self.particles = np.tile(np.asarray(pose, dtype=np.float64), (self.n, 1))
        self.particles += self.rng.normal(0.0, 1.0, (self.n, 3)) * np.asarray(std)
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])
        self.log_w[:] = -math.log(self.n)

    @property
    def weights(self):
        w = np.exp(self.log_w - self.log_w.max())
        return w / w.sum()

    # ------------------------------------------------------------------
    def predict(self, odom_prev, odom_now):
        d = relative(odom_prev, odom_now)
        dx, dy, dth = d
        trans = math.hypot(dx, dy)
        if trans < 1e-4:
            rot1 = 0.0
        else:
            rot1 = math.atan2(dy, dx)
            if dx < 0.0:  # driving backwards
                rot1 = float(wrap_angle(rot1 - math.pi))
                trans = -trans
        rot2 = float(wrap_angle(dth - rot1))

        sd_rot1 = math.sqrt(self.a1 * rot1 ** 2 + self.a2 * trans ** 2) + self.min_rot_std
        sd_trans = math.sqrt(self.a3 * trans ** 2 + self.a4 * (rot1 ** 2 + rot2 ** 2)) \
            + self.min_trans_std
        sd_rot2 = math.sqrt(self.a1 * rot2 ** 2 + self.a2 * trans ** 2) + self.min_rot_std

        n = self.n
        r1 = rot1 - self.rng.normal(0.0, sd_rot1, n)
        tr = trans - self.rng.normal(0.0, sd_trans, n)
        r2 = rot2 - self.rng.normal(0.0, sd_rot2, n)

        th = self.particles[:, 2]
        self.particles[:, 0] += tr * np.cos(th + r1)
        self.particles[:, 1] += tr * np.sin(th + r1)
        self.particles[:, 2] = wrap_angle(th + r1 + r2)

    # ------------------------------------------------------------------
    def update(self, log_likelihood):
        self.log_w += self.beta * log_likelihood
        self.log_w -= _logsumexp(self.log_w)

    def effective_sample_size(self):
        w = self.weights
        return 1.0 / float(np.sum(w * w))

    def resample_if_needed(self):
        if self.effective_sample_size() >= self.resample_ratio * self.n:
            return False
        w = self.weights
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        cumulative = np.cumsum(w)
        cumulative[-1] = 1.0
        idx = np.searchsorted(cumulative, positions)
        self.particles = self.particles[idx].copy()
        self.log_w[:] = -math.log(self.n)
        return True

    # ------------------------------------------------------------------
    def estimate(self):
        w = self.weights
        x = float(np.sum(w * self.particles[:, 0]))
        y = float(np.sum(w * self.particles[:, 1]))
        th = math.atan2(float(np.sum(w * np.sin(self.particles[:, 2]))),
                        float(np.sum(w * np.cos(self.particles[:, 2]))))
        mean = np.array([x, y, th])
        diff = self.particles - mean
        diff[:, 2] = wrap_angle(diff[:, 2])
        cov = (diff * w[:, None]).T @ diff
        return mean, cov


def _logsumexp(a):
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))
