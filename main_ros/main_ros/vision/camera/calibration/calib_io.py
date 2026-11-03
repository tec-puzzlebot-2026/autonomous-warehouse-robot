"""Calibration file helpers shared by the scripts and nodes (no ROS imports).

File layout under ``calib_dir`` (default: main_ros/vision/camera/calibration in src):

    images/                    checkerboard captures (raw, PNG)
    whiteimages/               white wall / paper captures (raw, PNG)
    data/camera_info.yaml      intrinsics, ROS camera_calibration_parsers format
                               (written by calibrate_intrinsics.py OR by the
                               image_pipeline cameracalibrator "COMMIT" button)
    data/intrinsicCalibration.npz   MAS507-style intrinsics (mtx, dist, ...)
    data/colorCalibration.npz       MAS507-style per-pixel colour gains (arr_0)
"""
import os
import pathlib

import cv2
import numpy as np
import yaml

DEFAULT_CAMERA_CALIB_DIR = '~/ros2_ws/src/main_ros/main_ros/vision/camera/calibration'
CAMERA_INFO_FILE = 'data/camera_info.yaml'
INTRINSICS_NPZ_FILE = 'data/intrinsicCalibration.npz'
COLOR_NPZ_FILE = 'data/colorCalibration.npz'
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

def expand(path):
    return pathlib.Path(os.path.expandvars(os.path.expanduser(str(path))))


def resolve(calib_dir, path_or_empty, default_rel):
    """Return explicit path if given, otherwise calib_dir/default_rel."""
    if path_or_empty:
        return expand(path_or_empty)
    return expand(calib_dir) / default_rel


def list_images(folder):
    folder = expand(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def corners_clear_of_border(corners, width, height, margin=10):
    """False if any detected corner is within `margin` px of the image border.

    A board that runs off the image edge can still be "found" with the wrong
    corners; such detections poison the calibration, so they are rejected.
    """
    p = corners.reshape(-1, 2)
    return bool(p[:, 0].min() >= margin and p[:, 1].min() >= margin
                and p[:, 0].max() <= width - 1 - margin
                and p[:, 1].max() <= height - 1 - margin)

# ---------------------------------------------------------------- camera_info

def _mat(rows, cols, data):
    return {'rows': rows, 'cols': cols,
            'data': [float(x) for x in np.asarray(data, dtype=float).ravel()]}


def write_camera_info_yaml(path, width, height, K, D, R=None, P=None,
                           camera_name='pi_camera_v2', distortion_model='plumb_bob'):
    """Write intrinsics in the format used by camera_calibration_parsers."""
    K = np.asarray(K, dtype=float).reshape(3, 3)
    D = np.asarray(D, dtype=float).ravel()
    R = np.eye(3) if R is None else np.asarray(R, dtype=float).reshape(3, 3)
    if P is None:
        P = np.hstack([K, np.zeros((3, 1))])
    P = np.asarray(P, dtype=float).reshape(3, 4)
    content = {
        'image_width': int(width),
        'image_height': int(height),
        'camera_name': str(camera_name),
        'camera_matrix': _mat(3, 3, K),
        'distortion_model': str(distortion_model),
        'distortion_coefficients': _mat(1, len(D), D),
        'rectification_matrix': _mat(3, 3, R),
        'projection_matrix': _mat(3, 4, P),
    }
    path = expand(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'w') as f:
        yaml.safe_dump(content, f, sort_keys=False, default_flow_style=None)
    return path


def read_camera_info_yaml(path):
    """Return dict with width, height, name, K, D, R, P, distortion_model."""
    with open(str(expand(path))) as f:
        y = yaml.safe_load(f)

    def m(key, shape):
        return np.array(y[key]['data'], dtype=float).reshape(shape)

    d = np.array(y['distortion_coefficients']['data'], dtype=float).ravel()
    return {
        'width': int(y['image_width']),
        'height': int(y['image_height']),
        'camera_name': y.get('camera_name', 'camera'),
        'distortion_model': y.get('distortion_model', 'plumb_bob'),
        'K': m('camera_matrix', (3, 3)),
        'D': d,
        'R': m('rectification_matrix', (3, 3)),
        'P': m('projection_matrix', (3, 4)),
    }


def load_intrinsics(path):
    """Load intrinsics from a camera_info .yaml or a MAS507 .npz.

    Returns dict(K, D, R, P, width, height); P/width/height may be None for
    npz files written by the original MAS507 script.
    """
    path = expand(path)
    if path.suffix.lower() in ('.yaml', '.yml'):
        return read_camera_info_yaml(path)
    data = np.load(str(path))
    size = data['image_size'] if 'image_size' in data.files else None
    return {
        'K': np.asarray(data['mtx'], dtype=float),
        'D': np.asarray(data['dist'], dtype=float).ravel(),
        'R': np.eye(3),
        'P': None,
        'width': int(size[0]) if size is not None else None,
        'height': int(size[1]) if size is not None else None,
    }

# ------------------------------------------------------------- rectification

def new_camera_matrix(K, D, size, alpha):
    """alpha=0: crop to valid pixels only; alpha=1: keep all source pixels."""
    newK, _roi = cv2.getOptimalNewCameraMatrix(K, D, size, alpha, size)
    return newK


def build_rectify_maps(K, D, size, R=None, P=None, alpha=None):
    """Precompute remap tables. size=(width, height).

    If alpha is None and P is given, P[:3,:3] is used (same as image_proc);
    otherwise the new camera matrix is computed from alpha (default 0).
    Returns (map1, map2, newK).
    """
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).ravel()
    R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
    if alpha is None and P is not None:
        newK = np.asarray(P, dtype=np.float64)[:3, :3]
    else:
        newK = new_camera_matrix(K, D, size, 0.0 if alpha is None else float(alpha))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, R, newK, size, cv2.CV_16SC2)
    return map1, map2, newK

# ----------------------------------------------------------- colour gains

def load_color_gains(path):
    data = np.load(str(expand(path)))
    key = 'arr_0' if 'arr_0' in data.files else data.files[0]
    gains = np.asarray(data[key], dtype=np.float32)
    if gains.ndim != 3 or gains.shape[2] != 3:
        raise ValueError('Colour gains must be HxWx3, got %s' % (gains.shape,))
    return gains


def apply_color_gains(img, gains):
    """Per-pixel, per-channel gain (MAS507 colour calibration), saturating."""
    return cv2.multiply(img, gains, dtype=cv2.CV_8U)