"""
SE(2) rigid-body utilities.

A planar pose is stored as a NumPy array [x, y, theta]:

    T(x, y, theta) = | cos(theta)  -sin(theta)  x |
                     | sin(theta)   cos(theta)  y |
                     |     0            0       1 |

Composition a ⊕ b means "apply b expressed in the frame of a".
Only rotations and translations are allowed (no affine stretch/shear),
because a rigid robot cannot deform the environment.
"""

import math

import numpy as np


def wrap_angle(angle):
    """Wrap an angle (scalar or array) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def compose(a, b):
    """Return a ⊕ b."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([
        a[0] + c * b[0] - s * b[1],
        a[1] + s * b[0] + c * b[1],
        float(wrap_angle(a[2] + b[2])),
    ])


def inverse(a):
    """Return a⁻¹ such that a ⊕ a⁻¹ = identity."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([
        -c * a[0] - s * a[1],
        s * a[0] - c * a[1],
        float(wrap_angle(-a[2])),
    ])


def relative(a, b):
    """Return the pose of b expressed in the frame of a: a⁻¹ ⊕ b."""
    return compose(inverse(a), b)


def transform_points(pose, points):
    """Transform (N, 2) points from the pose's local frame into its parent frame."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = pose[0] + c * points[:, 0] - s * points[:, 1]
    out[:, 1] = pose[1] + s * points[:, 0] + c * points[:, 1]
    return out


def yaw_from_quaternion(x, y, z, w):
    """Yaw (rotation about Z) of a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw):
    """Quaternion (x, y, z, w) for a pure rotation about Z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def rotation_matrix_from_quaternion(x, y, z, w):
    """3x3 rotation matrix of a unit quaternion."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ----------------------------------------------------------------------
# Vectorized versions (N poses at once), used by the pose graph back-end.
# ----------------------------------------------------------------------
def compose_many(a, b):
    """a ⊕ b for one pose a (3,) and many poses b (N, 3)."""
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    c, s = math.cos(a[2]), math.sin(a[2])
    out = np.empty_like(b)
    out[:, 0] = a[0] + c * b[:, 0] - s * b[:, 1]
    out[:, 1] = a[1] + s * b[:, 0] + c * b[:, 1]
    out[:, 2] = wrap_angle(a[2] + b[:, 2])
    return out


def relative_many(a, b):
    """a⁻¹ ⊕ b for one pose a (3,) and many poses b (N, 3)."""
    return compose_many(inverse(a), b)
