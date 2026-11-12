"""Configuration and samples file of the error experiment (no ROS)."""
import csv
import math
import os
import pathlib

import numpy as np
import yaml

from main_ros.vision.aruco import handeye as he
from main_ros.vision.aruco import transforms as tf
from main_ros.vision.aruco.paths import DEFAULT_CALIB_DIR, expand

CONFIG_NAME = 'aruco_localization.yaml'

COLUMNS = [
    'sample', 'time', 'marker_id', 'gt_d', 'gt_y', 'gt_yaw_deg', 'n_frames',
    'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw',                 # ArUco in the optical frame (mean)
    'std_x', 'std_y', 'std_yaw_deg', 'std_dist',              # jitter between frames
    'distance', 'view_angle_deg', 'reproj_err_px', 'ambiguity_min',
    'mount_x', 'mount_y', 'mount_z', 'mount_roll', 'mount_pitch', 'mount_yaw',
    'note',
]


def default_config_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        p = pathlib.Path(get_package_share_directory('main_ros')) / 'config' / CONFIG_NAME
        if p.is_file():
            return p
    except Exception:  # noqa: BLE001 - ROS not sourced: look in the source tree
        pass
    return pathlib.Path(__file__).resolve().parents[3] / 'config' / CONFIG_NAME


def default_samples_path():
    return expand(DEFAULT_CALIB_DIR) / 'aruco' / 'samples.csv'


def load_config(path=None):
    path = expand(path) if path else default_config_path()
    with open(str(path)) as f:
        raw = yaml.safe_load(f)

    def block(name):
        return (raw.get(name) or {}).get('ros__parameters', {})

    m = block('camera_mount')
    return {
        'path': path,
        'mount': tuple(float(m.get(k, 0.0)) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')),
        'detector': block('aruco_detector'),
        'experiment': block('aruco_experiment'),
    }


def marker_height_floor(experiment, marker_id=None):
    """Height of the CENTER of this ArUco above the floor (m): per-marker value if listed
    in per_marker_ids / per_marker_height, otherwise `marker_height`."""
    per = tf.per_marker_map(experiment.get('per_marker_ids'),
                            experiment.get('per_marker_height'), 'height')
    if marker_id is not None and int(marker_id) in per:
        return per[int(marker_id)]
    return float(experiment.get('marker_height', 0.09))


def marker_z(experiment, marker_id=None):
    """Height of the center of ArUco `marker_id` above the base_link plane (wheel axle).

    The track frame W has its origin at the height of base_link, not on the floor,
    because robot poses are expressed as base_link with z = 0.
    """
    return marker_height_floor(experiment, marker_id) - \
        float(experiment.get('base_link_height', 0.055))


def marker_size(detector, marker_id=None):
    """Side of the black square of this ArUco (m): per_marker_size if listed, else marker_size."""
    per = tf.per_marker_map(detector.get('per_marker_ids'),
                            detector.get('per_marker_size'), 'size')
    if marker_id is not None and int(marker_id) in per:
        return per[int(marker_id)]
    return float(detector.get('marker_size', float('nan')))


def append_row(path, row):
    path = expand(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or os.path.getsize(str(path)) == 0
    with open(str(path), 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def read_rows(path):
    with open(str(expand(path)), newline='') as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        conv = {}
        for k, v in r.items():
            if k == 'note':
                conv[k] = v or ''
                continue
            try:
                conv[k] = float(v)
            except (TypeError, ValueError):
                conv[k] = float('nan')
        out.append(conv)
    return out


def remove_last_row(path):
    path = expand(path)
    lines = path.read_text().splitlines(keepends=True)
    if len(lines) <= 1:
        return False
    path.write_text(''.join(lines[:-1]))
    return True


def T_C_M_from_row(row):
    R = tf.quat_to_matrix(row['qx'], row['qy'], row['qz'], row['qw'])
    return tf.make_T(R, [row['tx'], row['ty'], row['tz']])


def evaluate(row, X, marker_height):
    """Compares the camera with the grid ground truth for one sample.

    Returns a dict with the real pose and the camera-measured robot pose in W (x, y, yaw),
    the real and measured range/bearing/depth, and the errors (measured - real).
    """
    T_W_M = he.marker_in_world(marker_height)
    yaw_real = math.radians(row['gt_yaw_deg'])
    T_W_B = he.robot_in_world(row['gt_d'], row['gt_y'], yaw_real)
    T_C_M = T_C_M_from_row(row)
    xc, yc, yawc = tf.T_to_planar(he.robot_from_camera(T_W_M, T_C_M, X))

    # Expected ArUco in base_link and in the optical frame (for Zreal vs Zcam of step C)
    T_B_M_real = tf.inv_T(T_W_B) @ T_W_M
    T_C_M_real = tf.inv_T(X) @ T_B_M_real
    T_B_M_cam = X @ T_C_M
    p_real, p_cam = T_B_M_real[:3, 3], T_B_M_cam[:3, 3]
    range_real = math.hypot(p_real[0], p_real[1])
    range_cam = math.hypot(p_cam[0], p_cam[1])
    bearing_real = math.atan2(p_real[1], p_real[0])   # atan2: no division by zero
    bearing_cam = math.atan2(p_cam[1], p_cam[0])
    return {
        'x_real': -row['gt_d'], 'y_real': row['gt_y'], 'yaw_real': yaw_real,
        'x_cam': xc, 'y_cam': yc, 'yaw_cam': yawc,
        'err_x': xc - (-row['gt_d']), 'err_y': yc - row['gt_y'],
        'err_yaw': tf.wrap_angle(yawc - yaw_real),
        'z_real': float(T_C_M_real[2, 3]), 'z_cam': float(T_C_M[2, 3]),
        'err_z': float(T_C_M[2, 3] - T_C_M_real[2, 3]),
        'range_real': range_real, 'range_cam': range_cam, 'err_range': range_cam - range_real,
        'bearing_real': bearing_real, 'bearing_cam': bearing_cam,
        'err_bearing': tf.wrap_angle(bearing_cam - bearing_real),
        'T_W_B': T_W_B, 'T_C_M': T_C_M,
    }


def summarize_frames(dets, X, marker_height):
    """Averages N detections of the same marker -> row fields + jitter."""
    if not dets:
        raise ValueError('No detections')
    ts = np.array([d['t'] for d in dets], dtype=float)
    q = tf.average_quaternions([d['q'] for d in dets])
    T_W_M = he.marker_in_world(marker_height)
    poses = []
    for d in dets:
        T = tf.make_T(tf.quat_to_matrix(*d['q']), d['t'])
        poses.append(tf.T_to_planar(he.robot_from_camera(T_W_M, T, X)))
    poses = np.array(poses)
    # yaw deviation with circular mean (avoids jumps at ±180°)
    yaw_mean = math.atan2(np.mean(np.sin(poses[:, 2])), np.mean(np.cos(poses[:, 2])))
    yaw_dev = np.array([tf.wrap_angle(a - yaw_mean) for a in poses[:, 2]])
    amb = [d['ambiguity'] for d in dets if d.get('ambiguity') is not None]

    def mean_of(key):
        vals = [d[key] for d in dets if d.get(key) is not None]
        return float(np.mean(vals)) if vals else float('nan')

    t = ts.mean(axis=0)
    return {
        'n_frames': len(dets),
        'tx': t[0], 'ty': t[1], 'tz': t[2],
        'qx': q[0], 'qy': q[1], 'qz': q[2], 'qw': q[3],
        'std_x': float(np.std(poses[:, 0])), 'std_y': float(np.std(poses[:, 1])),
        'std_yaw_deg': math.degrees(float(np.sqrt(np.mean(yaw_dev ** 2)))),
        'std_dist': float(np.std(np.linalg.norm(ts, axis=1))),
        'distance': mean_of('distance'), 'view_angle_deg': mean_of('view_angle_deg'),
        'reproj_err_px': mean_of('reproj_err_px'),
        'ambiguity_min': float(min(amb)) if amb else float('nan'),
    }
