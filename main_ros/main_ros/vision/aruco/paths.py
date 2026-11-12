"""Single place for the folders and the motion-model import used by the ArUco/EKF code.

Repository layout:
main_ros/vision/camera/calibration/calib_io.py -> DEFAULT_CAMERA_CALIB_DIR 
main_ros/navigation/motion_model.py -> MotionModel, dead_reckon, wrap 
main_ros/navigation/calibration/motion/ -> motion_model.yaml, runs.csv, ...

Results written by this code: 
<ARUCO_DIR>/samples.csv, report.md, error_model.yaml, *.png 
<EKF_DIR>/ekf_<date>.csv, ekf_<date>.md
"""
import pathlib

from main_ros.vision.camera.calibration.calib_io import (
    DEFAULT_CAMERA_CALIB_DIR as DEFAULT_CALIB_DIR)

from main_ros.navigation.motion_model import MotionModel, dead_reckon, wrap
    _NAV = True

def expand(p):
    return pathlib.Path(str(p)).expanduser()


def calib_dir(path=None):
    return expand(path if path else DEFAULT_CALIB_DIR)


def motion_dir():
    """Folder with motion_model.yaml (analyze_motion_noise output)."""
    if _NAV:
        import main_ros
        d = pathlib.Path(main_ros.__file__).resolve().parent / 'navigation' / 'calibration' / 'motion'
        if d.is_dir():
            return d
    return calib_dir() / 'motion'


def aruco_dir():
    return calib_dir() / 'aruco'


def ekf_dir():
    return calib_dir() / 'ekf'
