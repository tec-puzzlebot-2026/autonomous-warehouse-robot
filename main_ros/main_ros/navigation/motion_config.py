"""Robot configuration and motion experiment directories (without ROS)."""

import os
import pathlib
import yaml


def expand(path):
    return pathlib.Path(os.path.expandvars(os.path.expanduser(str(path))))


CONFIG_NAME = 'puzzlebot.yaml'
_INSTALLED = ('install', 'build', 'site-packages', 'dist-packages')


def _default_nav_calib_dir():
    """<repo>/main_ros/main_ros/navigation/calibration, with no hard-coded path.

    1) env var MAIN_ROS_NAV_CALIB_DIR; 2) the folder next to this file (git clone, Docker,
    colcon --symlink-install); 3) if this is an installed copy (install/, build/, site-packages:
    a rebuild would erase the data) the source tree found above the current directory;
    4) ~/main_ros_nav_calibration.
    """
    env = os.environ.get('MAIN_ROS_NAV_CALIB_DIR')
    if env:
        return env
    here = pathlib.Path(__file__).resolve().parent            # .../main_ros/navigation
    if not any(part in _INSTALLED for part in here.parts):
        return str(here / 'calibration')
    cwd = pathlib.Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        for cand in (p, p / 'navigation', p / 'main_ros' / 'navigation',
                     p / 'main_ros' / 'main_ros' / 'navigation'):
            if (cand / 'motion_config.py').is_file() \
                    and not any(part in _INSTALLED for part in cand.parts):
                return str(cand / 'calibration')
    return '~/main_ros_nav_calibration'


DEFAULT_NAV_CALIB_DIR = _default_nav_calib_dir()

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