"""4x4 homogeneous transforms in plain numpy (no tf_transformations).

Convention: T_A_B maps points expressed in frame B to frame A,
p_A = T_A_B @ p_B. Angles in radians unless the name says _deg.
"""
import math

import numpy as np

EPS = 1e-12


def safe_div(num, den, default=float('nan')):
    """Guarded division: returns `default` if |den| ~ 0 (avoids ZeroDivisionError/inf)."""
    try:
        if abs(den) < EPS:
            return default
        return num / den
    except ZeroDivisionError:
        return default


def safe_acos(x):
    """acos with the argument clipped to [-1, 1] (numerical noise can give 1.0000001)."""
    return math.acos(max(-1.0, min(1.0, float(x))))


def per_marker_map(ids, values, what='value'):
    """{id: value} from two parallel lists (YAML: per_marker_ids / per_marker_<what>).

    Entries with id < 0 are placeholders (ROS needs a non-empty default list) and are
    ignored. Raises ValueError if the lists differ in length or an id repeats.
    """
    ids = [] if ids is None else list(ids)
    values = [] if values is None else list(values)
    if len(ids) != len(values):
        raise ValueError('per_marker_ids (%d) and per_marker_%s (%d) must have the same length'
                         % (len(ids), what, len(values)))
    out = {}
    for i, v in zip(ids, values):
        if int(i) < 0:
            continue
        if int(i) in out:
            raise ValueError('marker id %d repeated in per_marker_ids' % int(i))
        out[int(i)] = float(v)
    return out


def wrap_angle(a):
    """Wraps an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_matrix(roll, pitch, yaw):
    """ROS convention (fixed axes X-Y-Z): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R):
    """Inverse of rpy_to_matrix. Handles gimbal lock (pitch = ±90°) without dividing by zero."""
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: yaw and roll are not separable; yaw is set to 0
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < EPS:
        raise ValueError('Zero-norm quaternion')
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """Returns (x, y, z, w), with w >= 0. Uses the numerically stable branch."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q if w >= 0 else -q


def rotation_angle(R):
    """Angle (rad) of rotation R, via the trace, robust to numerical noise."""
    return safe_acos((np.trace(R) - 1.0) / 2.0)


def rotation_log(R):
    """Rotation vector (axis * angle), robust near 0 and pi."""
    ang = rotation_angle(R)
    if ang < 1e-9:
        return np.zeros(3)
    if abs(math.pi - ang) < 1e-6:
        # near 180°: axis = dominant column of (R + I)/2
        M = (R + np.eye(3)) / 2.0
        axis = M[:, int(np.argmax(np.diag(M)))]
        axis = axis / max(np.linalg.norm(axis), EPS)
        return axis * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (ang / (2.0 * math.sin(ang)))


def average_quaternions(quats):
    """Quaternion average (Markley method: eigenvector of sum(q q^T)).
    Sign-robust (q and -q represent the same rotation)."""
    Q = np.asarray(quats, dtype=float).reshape(-1, 4)
    if len(Q) == 0:
        raise ValueError('Empty quaternion list')
    M = Q.T @ Q
    _vals, vecs = np.linalg.eigh(M)
    q = vecs[:, -1]
    return q if q[3] >= 0 else -q


# Optical frame (z forward, x right, y down) with respect to camera_link
# (x forward, y left, z up): roll=-90°, pitch=0, yaw=-90° (REP 103/105).
R_LINK_OPTICAL = rpy_to_matrix(-math.pi / 2, 0.0, -math.pi / 2)


def planar_pose_T(x, y, yaw):
    """Planar robot pose (x, y, yaw) as a 4x4 T (z = 0)."""
    return make_T(rot_z(yaw), [x, y, 0.0])


def T_to_planar(T):
    """Extracts (x, y, yaw) from a 4x4 T, projecting the body x axis onto the plane."""
    yaw = math.atan2(T[1, 0], T[0, 0])
    return float(T[0, 3]), float(T[1, 3]), yaw
