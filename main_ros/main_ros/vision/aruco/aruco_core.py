"""ArUco detection and pose estimation (no ROS), compatible with OpenCV 4.1 ... 5.x.

The marker used on the test track (photo "ID 21") belongs to DICT_ARUCO_ORIGINAL: the
photo was tested against every OpenCV dictionary and only this one decodes it, with
0 corrected bits (5x5 inner grid).

Marker frame (OpenCV convention): origin at the center, x to the right,
y up (seen from the front), z coming out of the face toward the camera.
"""
import math

import cv2
import numpy as np

from main_ros.vision.aruco.transforms import safe_acos, safe_div

DEFAULT_DICTIONARY = 'DICT_ARUCO_ORIGINAL'


def has_aruco():
    return hasattr(cv2, 'aruco')


def get_dictionary(name):
    if not has_aruco():
        raise RuntimeError('This OpenCV has no aruco module (install opencv-contrib '
                           'or use OpenCV >= 4.7).')
    if not hasattr(cv2.aruco, name):
        raise ValueError('Unknown dictionary: %s' % name)
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


class MarkerDetector:
    """Wraps the two aruco APIs: ArucoDetector (>= 4.7) and detectMarkers (< 4.7)."""

    def __init__(self, dictionary_name=DEFAULT_DICTIONARY):
        self.dictionary = get_dictionary(dictionary_name)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):     # OpenCV < 4.7
            self.params = cv2.aruco.DetectorParameters_create()
        else:
            self.params = cv2.aruco.DetectorParameters()
        # Sub-pixel corner refinement: greatly improves the pose at long range.
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = None
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self._detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, gray):
        """Returns a list of (id, 4x2 float32 corners in TL, TR, BR, BL order)."""
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary,
                                                       parameters=self.params)
        if ids is None:
            return []
        return [(int(i), c.reshape(4, 2).astype(np.float32))
                for i, c in zip(ids.ravel(), corners)]


def marker_object_points(size):
    """3D corners of the marker in its frame (order required by SOLVEPNP_IPPE_SQUARE)."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def _reproj_rms(obj, img, rvec, tvec, K, D):
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, 1))))


def estimate_pose(corners, size, K, D):
    """Pose of the marker in the camera optical frame.

    Uses IPPE_SQUARE, which gives TWO solutions for a planar square. The ratio
    error(2nd)/error(1st) measures the ambiguity: close to 1 means the
    orientation is unreliable (distant marker, or seen almost head-on with
    few pixels). Returns a dict with R (3x3), t (3,), reproj_err_px and ambiguity.
    """
    obj = marker_object_points(size)
    img = corners.reshape(4, 1, 2).astype(np.float64)
    K = np.asarray(K, dtype=np.float64)
    D = np.zeros(5) if D is None else np.asarray(D, dtype=np.float64).ravel()
    flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)

    ambiguity = float('nan')
    if hasattr(cv2, 'solvePnPGeneric') and flag != cv2.SOLVEPNP_ITERATIVE:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, D, flags=flag)
        if n < 1:
            return None
        sols = sorted(((_reproj_rms(obj, img, r, t, K, D), r, t)
                       for r, t in zip(rvecs, tvecs)), key=lambda s: s[0])
        err, rvec, tvec = sols[0]
        if len(sols) > 1:
            ambiguity = safe_div(sols[1][0], max(err, 1e-6), default=float('inf'))
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=flag)
        if not ok:
            return None
        err = _reproj_rms(obj, img, rvec, tvec, K, D)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3).astype(float)
    if t[2] <= 0:            # the marker cannot be behind the camera
        return None
    return {'R': R, 't': t, 'rvec': rvec.reshape(3), 'reproj_err_px': err,
            'ambiguity': ambiguity}


def viewing_angle(R, t):
    """Angle (rad) between the marker normal and the camera->marker line of sight.
    0 = marker seen head-on. Returns NaN if the distance is ~0."""
    dist = float(np.linalg.norm(t))
    if dist < 1e-9:
        return float('nan')
    normal = R[:, 2]                    # marker z, points toward the camera
    to_cam = -np.asarray(t) / dist
    return safe_acos(float(np.dot(normal, to_cam)))


def marker_area_px(corners):
    return float(cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.float32)))


def describe_detection(marker_id, corners, size, K, D):
    """Full detection: pose + quality metrics used by the error model."""
    pose = estimate_pose(corners, size, K, D)
    if pose is None:
        return None
    R, t = pose['R'], pose['t']
    dist = float(np.linalg.norm(t))
    return {
        'id': marker_id,
        'R': R,
        't': t,
        'distance': dist,
        'view_angle_deg': math.degrees(viewing_angle(R, t)),
        'reproj_err_px': pose['reproj_err_px'],
        'ambiguity': pose['ambiguity'],
        'area_px': marker_area_px(corners),
        'corners': corners,
        'rvec': pose['rvec'],
    }
