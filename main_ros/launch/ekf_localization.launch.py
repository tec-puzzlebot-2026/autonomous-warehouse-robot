"""EKF localization with an ArUco map: camera + detector + EKF.

    ros2 launch main_ros ekf_localization.launch.py
    ros2 launch main_ros ekf_localization.launch.py start_aruco:=false   # if already running

Parameters: config/arena_map.yaml (landmarks, start-up mode: fixed pose or auto), config/puzzlebot.yaml
(r, L, encoder topics) and camera_mount from config/aruco_localization.yaml.
Error models: motion_model.yaml (navigation/calibration/motion/ or <calib_dir>/motion/)
and <calib_dir>/aruco/error_model.yaml.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from main_ros.vision.aruco.paths import DEFAULT_CALIB_DIR  # noqa: E402

PKG = 'main_ros'


def _block(path, name):
    with open(path) as f:
        return (yaml.safe_load(f).get(name) or {}).get('ros__parameters', {})


def _setup(context):
    share = get_package_share_directory(PKG)
    cfg = os.path.join(share, 'config')
    robot = _block(os.path.join(cfg, 'puzzlebot.yaml'), 'puzzlebot')
    m = _block(LaunchConfiguration('aruco_config').perform(context), 'camera_mount')
    params = {
        'calib_dir': LaunchConfiguration('calib_dir').perform(context),
        'wheel_radius': float(robot.get('wheel_radius', 0.05)),
        'wheelbase': float(robot.get('wheelbase', 0.19)),
        'enc_right_topic': robot.get('enc_right_topic', '/VelocityEncR'),
        'enc_left_topic': robot.get('enc_left_topic', '/VelocityEncL'),
        'enc_right_sign': float(robot.get('enc_right_sign', 1.0)),
        'enc_left_sign': float(robot.get('enc_left_sign', 1.0)),
        'camera_mount': [float(m.get(k, 0.0)) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')],
        'aruco_config': LaunchConfiguration('aruco_config').perform(context),   # per-ID heights
    }
    ekf = Node(package=PKG, executable='ekf_localization', name='ekf_localization',
               output='screen',
               parameters=[LaunchConfiguration('map').perform(context), params])
    return [ekf]


def generate_launch_description():
    share = get_package_share_directory(PKG)
    return LaunchDescription([
        DeclareLaunchArgument('calib_dir', default_value=str(DEFAULT_CALIB_DIR)),
        DeclareLaunchArgument('map', default_value=os.path.join(share, 'config',
                                                                'arena_map.yaml')),
        DeclareLaunchArgument('aruco_config', default_value=os.path.join(
            share, 'config', 'aruco_localization.yaml')),
        DeclareLaunchArgument('start_aruco', default_value='true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch',
                                                       'aruco_localization.launch.py')),
            launch_arguments={'calib_dir': LaunchConfiguration('calib_dir'),
                              'config': LaunchConfiguration('aruco_config')}.items(),
            condition=IfCondition(LaunchConfiguration('start_aruco'))),
        OpaqueFunction(function=_setup),
    ])
