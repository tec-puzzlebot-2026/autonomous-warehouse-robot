"""Puzzlebot motion model (differential drive) and its noise, without ROS.

Model (state s = [sx, sy, sθ], input u = [v, ω]):

    sx,k = sx,k-1 + Δt·v·cos(sθ,k-1)
    sy,k = sy,k-1 + Δt·v·sin(sθ,k-1)
    sθ,k = sθ,k-1 + Δt·ω                    + q_k, q_k ~ N(0, Q_k)

with v = r(ωR + ωL)/2 and ω = r(ωR - ωL)/L from the wheel velocities measured
by the encoders (rad/s).

Noise (measured with analyze_motion_noise): The velocity of each wheel, in
stable state, has a dispersion (standard deviation of the velocity
distribution of the system) that increases with velocity:

    σ_wheel(|ω_wheel|) = σ0 + k·|ω_wheel|   [rad/s]

Q_k is obtained by propagating the input noise by the model:

    Σ_w = diag(σ_R², σ_L²)
    J = ∂h/∂[ωR, ωL] = (r·Δt/2)·[[cosθ, cosθ], [sinθ, sinθ], [2/L, -2/L]]
    Q_k = α · J Σ_w Jᵀ

α (consistency factor, by default 1) adjusts to the validation against measuring tape:
the encoder noise doesn't see wheel drifting, so it is normally α > 1.
"""

import math
import numpy as np
import yaml

def wrap(a):
    """Angle a (-π, π) (equivalent to MATLAB wrapToPi)."""
    return math.atan2(math.sin(a), math.cos(a))

class MotionModel:
    """Physical parameters + wheel noise model."""

    def __init__(self, wheel_radius=0.05, wheelbase=0.19, sigma0=(0.0, 0.0), 
    k =(0.0, 0.0), alpha=1.0, min_sigma=1e-3):
        if wheel_radius <= 0 or wheelbase <= 0:
            raise ValueError('wheel_radius and wheelbase must be > 0')
        self.r = float(wheel_radius)
        self.L = float(wheelbase)
        # (right, left)
        self.sigma0 = tuple(float(v) for v in sigma0)
        self.k = tuple(float(v) for v in k)
        self.alpha = float(alpha)
        self.min_sigma = float(min_sigma)

    # Kinematics
    def wheels_to_twist(self, wR, wL):
        v = self.r * (wR + wL) / 2.0
        w = self.r * (wR - wL) / self.L
        return v, w
    
    def twist_to_wheels(self, v, w):
        wR = (v + w * self.L / 2.0) / self.r
        wL = (v - w * self.L / 2.0) / self.r
        return wR, wL

    def predict_mean(self, s, v, w, dt):
        """h(s_{k-1}, u_k): Euler integration."""
        x, y, th = s
        return np.array([x + dt * v * math.cos(th),
        y + dt * v * math.sin(th), wrap(th + dt * w)])
    
    @staticmethod
    def jacobian_state(s, v, dt):
        """H_k = ∂h/∂s_{k-1}."""
        th = s[2]
        return np.array([[1.0, 0.0, -dt * v * math.sin(th)],
        [0.0, 1.0, dt * v * math.cos(th)], [0.0, 0.0, 1.0]])

    def jacobian_wheels(self, s, dt):
        th = s[2]
        c = self.r * dt / 2.0
        return c * np.array([[math.cos(th), math.cos(th)],
        [math.sin(th), math.sin(th)],
        [2.0 / self.L, -2.0 / self.L]])

    # Noise
    def wheel_sigma(self, wR, wL):
        sR = self.sigma0[0] + self.k[0] * abs(wR)
        sL = self.sigma0[1] + self.k[1] * abs(wL)
        return max(sR, self.min_sigma), max(sL, self.min_sigma)

    def Q(self, s, wR, wL, dt):
        """Q_k = α·J·diag(σR², σL²)·Jᵀ. With robot  kept still (ω = 0), σ = σ0."""
        sR, sL = self.wheel_sigma(wR, wL)
        J = self.jacobian_wheels(s, dt)
        return self.alpha * J @ np.diag([sR ** 2, sL ** 2]) @ J.T

    # Files
    def to_dict(self):
        return {'wheel_radius': self.r, 'wheelbase': self.L, 'sigma0_right': self.sigma0[0], 
        'self.sigma0_left': self.sigma0[1], 'k_right': self.k[0], 'k_left': self.k[1], 
        'alpha': self.alpha, 'min_sigma': self.min_sigma}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('wheel_radius', 0.05), d.get('wheelbase', 0.19),
        (d.get('sigma0_right', 0.0), d.get('sigma0_left', 0.0)),
        (d.get('k_right', 0.0), d.get('k_left', 0.0)),
        d.get('alpha', 1.0), d.get('min_sigma', 1e-3))

    @classmethod
    def load(cls, path):
        with open(str(path)) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d.get('motion_model', d))

def dead_reckon(model, t, wR, wL, s0=(0.0, 0.0, 0.0), P0=None):
    """Integrate encoder series (same length) and propagate Σ with Q_k.

    Returns (s_final, Σ_final, trajectory Nx3). Uses the real Δt between samples;
    Δt <= 0 or > 0.5 s (empty in the data) is skipped to not invent movement.
    """
    s = np.array(s0, dtype=float)
    P = np.zeros((3, 3) if P0 is None else np.array(P0, dtype=float))
    traj = [s.copy()]
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        if dt <= 0 or dt > 0.5:
            continue
        # Interval input = mean velocity at interval start
        v, w = model.wheels_to_twist(wR[i - 1], wL[i - 1])
        H = model.jacobian_state(s, v, dt)
        Qk = model.Q(s, wR[i - 1], wL[i - 1], dt)
        s = model.predict_mean(s, v, w, dt)
        P = H @ P @ H.T @ Qk
        traj.append(s.copy())
    return s, P, np.array(traj)