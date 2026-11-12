"""Map-based localization EKF with known data association (Algorithm 6.1), no ROS.

Landmarks m_i = [mx, my] (ArUco centers in the map). Camera measurement:
    z_i,k = [ρ, α] = g(m_i, s_k) + r_i,k,  r ~ N(0, R_k)
    ρ = sqrt((mx − sx)² + (my − sy)²),   α = atan2(my − sy, mx − sx) − sθ

    G_k = [[−Δx/√p, −Δy/√p,  0],
           [ Δy/p,  −Δx/p,  −1]],   Δx = mx − ŝx, Δy = my − ŝy, p = Δx² + Δy²

All angle differences are wrapped to (−π, π] (wrapToPi).
"""
import math

import numpy as np

from main_ros.vision.aruco.paths import wrap

CHI2_2DOF_95 = 5.991
CHI2_2DOF_99 = 9.210


def observe(landmark, s):
    """ẑ = g(m_i, s) and G = ∇_s g. Raises ValueError if the robot is on top of the landmark."""
    dx = landmark[0] - s[0]
    dy = landmark[1] - s[1]
    p = dx * dx + dy * dy
    if p < 1e-9:                       # √p ~ 0: G is not defined
        raise ValueError('landmark at ~0 distance from the robot')
    rho = math.sqrt(p)
    z_hat = np.array([rho, wrap(math.atan2(dy, dx) - s[2])])
    G = np.array([[-dx / rho, -dy / rho, 0.0],
                  [dy / p, -dx / p, -1.0]])
    return z_hat, G


def range_bearing_from_base(p_base):
    """Measurement z = [ρ, α] from the ArUco center expressed in base_link."""
    x, y = float(p_base[0]), float(p_base[1])
    return np.array([math.hypot(x, y), math.atan2(y, x)])


def inflate_R(R, rho, v=0.0, w=0.0, landmark_sigma=0.0, latency_s=0.0, rho_min=0.05):
    """R_k plus two error sources that the camera test grid (robot standing still, ArUco
    position known exactly) does not contain:

    * landmark_sigma [m]: error of the ArUco position in arena_map.yaml (tape measure).
      A landmark displaced by e changes range by up to e and bearing by up to e/ρ.
    * latency_s [s]: delay between the image and its use in the filter. While the robot
      moves at (v, ω) the measurement is stale: range by up to |v|·τ, bearing by up to
      |ω|·τ + |v|·τ/ρ.
    Both are added as variances (independent errors). Returns a new 2x2 matrix.
    """
    R = np.array(R, dtype=float)
    rho = max(float(rho), rho_min)
    if landmark_sigma > 0.0:
        R[0, 0] += landmark_sigma ** 2
        R[1, 1] += (landmark_sigma / rho) ** 2
    if latency_s > 0.0:
        R[0, 0] += (abs(v) * latency_s) ** 2
        R[1, 1] += (abs(w) * latency_s) ** 2 + (abs(v) * latency_s / rho) ** 2
    return R


class EKF:
    """State μ = [x, y, θ] in the map frame, covariance Σ."""

    def __init__(self, motion_model, mu0=(0.0, 0.0, 0.0), sigma0=None, gate=CHI2_2DOF_99):
        self.mm = motion_model
        self.mu = np.array(mu0, dtype=float)
        self.mu[2] = wrap(self.mu[2])
        self.Sigma = np.zeros((3, 3)) if sigma0 is None else np.array(sigma0, dtype=float)
        self.gate = gate

    # Algorithm 6.1, lines 2-4
    def predict(self, wR, wL, dt):
        if dt <= 0:
            return
        v, w = self.mm.wheels_to_twist(wR, wL)
        H = self.mm.jacobian_state(self.mu, v, dt)
        Qk = self.mm.Q(self.mu, wR, wL, dt)
        self.mu = self.mm.predict_mean(self.mu, v, w, dt)
        self.Sigma = H @ self.Sigma @ H.T + Qk

    # Algorithm 6.1, lines 5-11
    def update(self, z, landmark, R):
        """Correction with a [ρ, α] measurement of the known landmark.

        Returns a dict with the innovation, NIS = νᵀ Z⁻¹ ν and whether it was accepted. A
        measurement with NIS above the gate (χ² 2 dof, 99 %) is rejected: with known
        association that indicates a detection error or an overly optimistic R.
        """
        z_hat, G = observe(landmark, self.mu)
        nu = np.array([z[0] - z_hat[0], wrap(z[1] - z_hat[1])])
        Z = G @ self.Sigma @ G.T + R
        try:
            Z_inv = np.linalg.inv(Z)
        except np.linalg.LinAlgError:
            return {'accepted': False, 'nis': float('inf'), 'nu': nu, 'Z': Z}
        nis = float(nu @ Z_inv @ nu)
        if self.gate is not None and nis > self.gate:
            return {'accepted': False, 'nis': nis, 'nu': nu, 'Z': Z}
        K = self.Sigma @ G.T @ Z_inv
        self.mu = self.mu + K @ nu
        self.mu[2] = wrap(self.mu[2])
        I_KG = np.eye(3) - K @ G
        # Joseph form: equivalent to (I − KG)Σ̂ but stays symmetric and > 0
        self.Sigma = I_KG @ self.Sigma @ I_KG.T + K @ R @ K.T
        return {'accepted': True, 'nis': nis, 'nu': nu, 'Z': Z, 'K': K}
