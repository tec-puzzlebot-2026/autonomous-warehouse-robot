"""ArUco measurement error model for the Kalman filter (no ROS).

analyze_aruco_errors writes <calib_dir>/aruco/error_model.yaml; this module
reads it and gives the covariance R of each detection as a function of distance and view angle:

    sigma_q(r, a) = (c0_q + c2_q * r^2) * f_q(a)

  * c0 + c2 r^2: the depth error of a monocular PnP grows ~ r^2
    (a one-pixel error at the corners is worth more meters the farther away it is).
  * f_q(a): foreshortening factor, interpolated from the measured table (1 = head-on).

How the filter decides between odometry and perception: the per-axis Kalman gain
is approximately K = P / (P + R). P (odometry uncertainty) grows with the
distance traveled; R (from this model) grows with the distance and angle to the ArUco.
  * R << P  -> K ~ 1: the correction trusts the camera.
  * R >> P  -> K ~ 0: odometry dominates (distant, very oblique or ambiguous ArUco).
"""
import math

import numpy as np
import yaml

from main_ros.vision.aruco.transforms import safe_div, wrap_angle

QUANTITIES = ('x', 'y', 'yaw', 'range', 'bearing')


def load(path):
    with open(str(path)) as f:
        return yaml.safe_load(f)


def angle_factor(model, q, view_angle_deg):
    tab = model['angle_factor'][q]
    angles, factors = tab['angles_deg'], tab['factor']
    if not angles:
        return 1.0
    a = abs(float(view_angle_deg)) if math.isfinite(view_angle_deg) else angles[-1]
    return float(np.interp(a, angles, factors))   # out of range: saturates at the end


def sigma(model, q, distance, view_angle_deg):
    c = model['sigma'][q]
    base = c['c0'] + c['c2'] * float(distance) ** 2
    return max(base * angle_factor(model, q, view_angle_deg), c.get('floor', 1e-6))


def is_valid(model, distance, view_angle_deg, ambiguity=None):
    """Gate: discard detections outside the range where the model is reliable.

    Pass `ambiguity` only if you will use the full pose (x, y, θ); for the
    range-bearing measurement the orientation ambiguity does not matter.
    """
    v = model['valid']
    if not (math.isfinite(distance) and math.isfinite(view_angle_deg)):
        return False
    if distance > v['max_distance'] or abs(view_angle_deg) > v['max_view_angle_deg']:
        return False
    if ambiguity is not None and math.isfinite(ambiguity) and ambiguity < v['min_ambiguity']:
        return False
    return True


def measurement_covariance(model, distance, view_angle_deg, yaw_ambiguous=False):
    """R (3x3) of the robot pose (x, y, yaw) expressed in the ArUco frame.

    If the pose is ambiguous (IPPE), the position is still useful but the yaw is not:
    its variance is inflated so the filter practically ignores it.
    """
    sx = sigma(model, 'x', distance, view_angle_deg)
    sy = sigma(model, 'y', distance, view_angle_deg)
    syaw = sigma(model, 'yaw', distance, view_angle_deg)
    if yaw_ambiguous:
        syaw = max(syaw, math.radians(45.0))
    return np.diag([sx ** 2, sy ** 2, syaw ** 2])


def range_bearing_covariance(model, distance, view_angle_deg):
    """R (2x2) of the (range, bearing) measurement to the ArUco center from base_link.

    This is the recommended EKF measurement: it does not depend on the estimated
    marker orientation, the noisiest part of a PnP with a single ArUco.
    """
    sr = sigma(model, 'range', distance, view_angle_deg)
    sb = sigma(model, 'bearing', distance, view_angle_deg)
    return np.diag([sr ** 2, sb ** 2])


def position_sigma_range_bearing(model, distance, view_angle_deg):
    """Equivalent position sigma (worst axis) of a range-bearing measurement."""
    sr = sigma(model, 'range', distance, view_angle_deg)
    sb = sigma(model, 'bearing', distance, view_angle_deg)
    return max(sr, float(distance) * sb)


def rotate_to_map(R_marker, marker_yaw_in_map):
    """Expresses the (x, y, yaw) covariance of the ArUco frame in the map frame."""
    c, s = math.cos(marker_yaw_in_map), math.sin(marker_yaw_in_map)
    Rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rot @ R_marker @ Rot.T


def odometry_covariance(dist_travelled, angle_travelled, k_pos=0.02, k_yaw=0.05,
                        k_yaw_per_m=0.01):
    """Accumulated odometry variance (random-walk model).

    k_pos [m/sqrt(m)], k_yaw [rad/sqrt(rad)], k_yaw_per_m [rad/sqrt(m)]: measure them on
    your robot (e.g. drive 2 m several times and measure the final spread).
    """
    s, th = abs(dist_travelled), abs(angle_travelled)
    var_pos = k_pos ** 2 * s
    var_yaw = k_yaw ** 2 * th + k_yaw_per_m ** 2 * s
    return np.diag([var_pos, var_pos, var_yaw])


def perception_weight(P_diag, R_diag):
    """Scalar gain per axis K = P / (P + R): 0 = odometry only, 1 = camera only."""
    return [safe_div(p, p + r, default=0.0) for p, r in zip(P_diag, R_diag)]


def ekf_update_pose(x, P, z, R, gate_chi2=11.34):
    """EKF update with a direct pose measurement z = (x, y, yaw), H = I.

    gate_chi2 = 11.34 is the 99th percentile of chi^2 with 3 degrees of freedom
    (Mahalanobis distance): outlier measurements are rejected.
    Returns (x, P, accepted, d2).
    """
    x = np.asarray(x, dtype=float)
    innov = np.asarray(z, dtype=float) - x
    innov[2] = wrap_angle(innov[2])
    S = P + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return x, P, False, float('inf')
    d2 = float(innov @ S_inv @ innov)
    if d2 > gate_chi2:
        return x, P, False, d2
    K = P @ S_inv
    x_new = x + K @ innov
    x_new[2] = wrap_angle(x_new[2])
    I_K = np.eye(3) - K
    P_new = I_K @ P @ I_K.T + K @ R @ K.T      # Joseph form: P stays symmetric and > 0
    return x_new, P_new, True, d2
