"""Estimation/validation of the camera-robot extrinsic (AX = XB) and of the calibration error.

Based on the RAS hand-eye calibration (ras_docker, MoveIt Calibration). There:
  * A = motion of the "effector" between two samples, B = motion of the target
    seen by the camera, X = unknown transform.
  * The calibration error margin is HandEyeSolverBase::getReprojectionError
    of moveit_calibration: for each pair of consecutive samples it compares AX with XB
    and returns the RMS of the relative rotation angle (rad) and the RMS of the
    translation error (average of ||t(AX)-t(XB)|| and ||t((AX)^-1)-t((XB)^-1)||), divided
    by the number of motions (samples - 1).

Adaptation to the Puzzlebot (differential drive): the camera is rigid on the chassis,
as "eye-in-hand" where the effector is base_link moving in the world W and the
target is the ArUco fixed on the wall:

    T_W_M = T_W_B(i) · X · T_C_M(i)        X = T_base_link_camera_optical
    A = T_W_B(i)^-1 · T_W_B(j)             B = T_C_M(i) · T_C_M(j)^-1      =>  A X = X B

Important physical limitation: the robot only moves in the plane and only rotates
about the vertical axis. With that, AX=XB determines x, y, yaw, roll and pitch of the
camera, but NOT the height z (the translation equation does not contain t_z). So
z is fixed with the tape-measure value. (The RAS guide says so:
"include rotations about at least two axes" - a planar robot cannot.)

Test-track frame W: origin on the wall, at the height of base_link (wheel axle)
and right below/above the ArUco center, x
pointing TOWARD the wall (the direction the robot faces when it looks at it
head-on), y to the left, z up. A grid sample is
described by d (distance from the wall to the wheel axle), y (lateral
offset, + = left) and yaw (robot rotation, + = counterclockwise).
"""
import itertools
import math

import numpy as np

from main_ros.vision.aruco import transforms as tf

ROT_WEIGHT = 0.2   # m/rad: 1° of rotation error weighs as 3.5 mm in the fit


# ---------------------------------------------------------------- geometry

def mount_to_X(x, y, z, roll, pitch, yaw):
    """Measured mount of camera_link in base_link -> T_base_link_camera_optical."""
    return tf.make_T(tf.rpy_to_matrix(roll, pitch, yaw) @ tf.R_LINK_OPTICAL, [x, y, z])


def X_to_mount(X):
    """Inverse of mount_to_X: (x, y, z, roll, pitch, yaw) of camera_link in base_link."""
    R_link = X[:3, :3] @ tf.R_LINK_OPTICAL.T
    roll, pitch, yaw = tf.matrix_to_rpy(R_link)
    return float(X[0, 3]), float(X[1, 3]), float(X[2, 3]), roll, pitch, yaw


def marker_in_world(marker_height):
    """T_W_M: ArUco on the wall (plane x = 0), facing -x (toward the track).

    marker_height: height of the ArUco center ABOVE base_link (see experiment_io.marker_z).
    """
    # marker axes in W: x_m = -y_W (right seen head-on), y_m = +z_W, z_m = -x_W
    R = np.array([[0.0, 0.0, -1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])
    return tf.make_T(R, [0.0, 0.0, marker_height])


def robot_in_world(d, y, yaw):
    """T_W_B of a grid position (d > 0 = distance to the wall)."""
    return tf.planar_pose_T(-d, y, yaw)


def robot_from_camera(T_W_M, T_C_M, X):
    """Robot pose in W according to the camera: T_W_B = T_W_M · T_C_M^-1 · X^-1."""
    return T_W_M @ tf.inv_T(T_C_M) @ tf.inv_T(X)


# ------------------------------------------------- MoveIt / RAS style error

def _motion_pairs(n, pairs):
    if pairs == 'consecutive':
        return [(i, i + 1) for i in range(n - 1)]
    return list(itertools.combinations(range(n), 2))


def motions(effector_wrt_world, object_wrt_sensor, pairs='consecutive'):
    """List of (A, B) eye-in-hand, same as moveit_calibration."""
    E, O = effector_wrt_world, object_wrt_sensor
    out = []
    for i, j in _motion_pairs(len(E), pairs):
        A = tf.inv_T(E[i]) @ E[j]
        B = O[i] @ tf.inv_T(O[j])
        out.append((A, B))
    return out


def reprojection_error(effector_wrt_world, object_wrt_sensor, X, pairs='consecutive'):
    """(rot_rms_rad, trans_rms_m) of AX vs XB - the calibration "error margin".

    Replica of HandEyeSolverBase::getReprojectionError (moveit_calibration). Returns
    (nan, nan) if samples are missing or the lists do not match, instead of dividing by zero.
    """
    if len(effector_wrt_world) != len(object_wrt_sensor) or len(effector_wrt_world) < 2:
        return float('nan'), float('nan')
    rot_sq = trans_sq = 0.0
    ms = motions(effector_wrt_world, object_wrt_sensor, pairs)
    for A, B in ms:
        AX, XB = A @ X, X @ B
        rot_sq += tf.rotation_angle(AX[:3, :3].T @ XB[:3, :3]) ** 2
        e1 = np.linalg.norm(AX[:3, 3] - XB[:3, 3])
        e2 = np.linalg.norm(tf.inv_T(AX)[:3, 3] - tf.inv_T(XB)[:3, 3])
        trans_sq += (0.5 * (e1 + e2)) ** 2
    n = len(ms)
    return math.sqrt(rot_sq / n), math.sqrt(trans_sq / n)


# ------------------------------------------------- planar solution of AX = XB

def _residuals(params, z_fixed, ms):
    x, y, yaw, roll, pitch = params
    X = mount_to_X(x, y, z_fixed, roll, pitch, yaw)
    res = []
    for A, B in ms:
        AX, XB = A @ X, X @ B
        E = tf.inv_T(AX) @ XB               # = identity if AX = XB
        res.extend(ROT_WEIGHT * tf.rotation_log(E[:3, :3]))
        res.extend(AX[:3, 3] - XB[:3, 3])
    return np.array(res)


def _levenberg_marquardt(fun, p0, iterations=60):
    """LM with numerical Jacobian (few parameters; avoids depending on scipy)."""
    p = np.asarray(p0, dtype=float)
    lam = 1e-3
    r = fun(p)
    cost = float(r @ r)
    J = np.zeros((len(r), len(p)))
    for _ in range(iterations):
        r = fun(p)
        for k in range(len(p)):
            dp = np.zeros(len(p))
            dp[k] = 1e-6
            J[:, k] = (fun(p + dp) - r) / 1e-6
        H, g = J.T @ J, J.T @ r
        improved, step = False, np.zeros(len(p))
        for _ in range(10):
            try:
                step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r_new = fun(p + step)
            new_cost = float(r_new @ r_new)
            if new_cost < cost:
                p, cost, lam, improved = p + step, new_cost, max(lam / 10, 1e-9), True
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-10:
            break
    sv = np.linalg.svd(J, compute_uv=False)
    return p, cost, sv


def solve_planar(effector_wrt_world, object_wrt_sensor, X0, iterations=60):
    """Solves AX = XB for (x, y, yaw, roll, pitch) with z fixed to that of X0.

    Uses all sample pairs. Returns a dict with X, the mount and an observability
    measure (singular values of the Jacobian). Note: B uses the ArUco ORIENTATION,
    the noisiest part of the pose (see solve_from_points).
    """
    ms = motions(effector_wrt_world, object_wrt_sensor, pairs='all')
    if len(ms) < 2:
        raise ValueError('At least 3 samples are needed to solve AX = XB')
    x0, y0, z0, r0, p0, yw0 = X_to_mount(X0)
    p, _cost, sv = _levenberg_marquardt(lambda q: _residuals(q, z0, ms),
                                        [x0, y0, yw0, r0, p0], iterations)
    X = mount_to_X(p[0], p[1], z0, p[3], p[4], p[2])
    return {'X': X, 'mount': X_to_mount(X), 'singular_values': sv,
            'well_conditioned': bool(sv[-1] > 1e-4 * max(sv[0], 1e-12))}


def solve_from_points(effector_wrt_world, marker_pos_in_sensor, marker_pos_in_world, X0,
                      iterations=60):
    """Full mount (6 DOF) using only the ArUco POSITION in each sample.

    Constraint per sample: T_W_B(i) · X · p_C(i) = p_W (ArUco center, known).
    The position from a PnP (range + bearing) is much more precise than its
    orientation, especially far away, so this estimate is more robust than
    AX = XB when the target is a single small marker.
    """
    if len(effector_wrt_world) < 3:
        raise ValueError('At least 3 samples are needed')
    pw = np.asarray(marker_pos_in_world, dtype=float)
    pcs = [np.append(np.asarray(pc, dtype=float), 1.0) for pc in marker_pos_in_sensor]

    def fun(q):
        X = mount_to_X(*q)
        return np.concatenate([(E @ X @ pc)[:3] - pw
                               for E, pc in zip(effector_wrt_world, pcs)])

    p, cost, sv = _levenberg_marquardt(fun, X_to_mount(X0), iterations)
    X = mount_to_X(*p)
    res = fun(p).reshape(-1, 3)
    return {'X': X, 'mount': X_to_mount(X), 'singular_values': sv,
            'rms_mm': 1000 * float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))),
            'well_conditioned': bool(sv[-1] > 1e-4 * max(sv[0], 1e-12))}


def absolute_X(effector_wrt_world, object_wrt_sensor, T_W_M):
    """Direct per-sample estimate: X_i = T_W_B(i)^-1 · T_W_M · T_C_M(i)^-1.

    Unlike AX = XB, it uses the known ArUco position (including its height),
    so it also estimates z. Returns the list of mounts (x, y, z, roll, pitch, yaw).
    """
    return [X_to_mount(tf.inv_T(E) @ T_W_M @ tf.inv_T(O))
            for E, O in zip(effector_wrt_world, object_wrt_sensor)]
