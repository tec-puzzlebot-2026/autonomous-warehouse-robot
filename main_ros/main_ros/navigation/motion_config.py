"""Robot configuration and motion experiment directories (without ROS)."""

import os
import pathlib
import yaml

def expand(path):
    return pathlib.Path(os.path.expandvars(os.path.expanduser(str(path))))

CONFIG_NAME = 'puzzlebot.yaml'
DEFAULT_NAV_CALIB_DIR = '~/ros2_ws/src/main_ros/main_ros/navigation/calibration'
RUNS_COLUMNS = ['run', 'type', 'target', 'speed', 'csv', 'odom_x', 'odom_y', 'odom_theta_deg',
'odom_dist', 'odom_turn_deg', 'real_x', 'real_y', 'real_theta_deg', 'note']

def default_config_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        p = pathlib.Path(get_package_share_directory('main_ros')) / 'config' / CONFIG_NAME
        if p.is_file():
            return p
    except Exception:
        pass
    return pathlib.Path(__file__).resolve().parents[2] / 'config' / CONFIG_NAME

def load_config(path=None):
    path = expand(path) if path else default_config_path()
    with open(str(path)) as f:
        raw = yaml.safe_load(f)

    def block(name):
        return (raw.get(name) or {}).get('ros__parameters', {})
    return {'path': path, 'robot': block('puzzlebot'), 'test': block('motion_noise_test')}

def motion_dir():
    return expand(DEFAULT_NAV_CALIB_DIR) / 'motion'