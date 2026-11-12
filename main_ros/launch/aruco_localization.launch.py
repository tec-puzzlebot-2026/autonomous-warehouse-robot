"""Static TF base_link -> camera_link -> camera_optical_frame + ArUco detector.

    ros2 launch main_ros aruco_localization.launch.py                 # includes vision.launch.py
    ros2 launch main_ros aruco_localization.launch.py start_vision:=false

The mount (x, y, z, roll, pitch, yaw) and the detector parameters are read from
config/aruco_localization.yaml, the same file used by take_aruco_sample and
aruco_error_analysis (a single source of truth).
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


def _setup(context):
    config = LaunchConfiguration('config').perform(context)
    with open(config) as f:
        m = yaml.safe_load(f)['camera_mount']['ros__parameters']

    def num(k):
        return str(float(m.get(k, 0.0)))

    # base_link (center of the wheel axle) -> camera_link (optical center, x forward)
    base_to_camera = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='base_to_camera_link',
        arguments=['--x', num('x'), '--y', num('y'), '--z', num('z'),
                   '--roll', num('roll'), '--pitch', num('pitch'), '--yaw', num('yaw'),
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link'],
    )
    # camera_link -> camera_optical_frame (z forward, x right, y down; REP 103)
    camera_to_optical = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='camera_link_to_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '-1.5707963267948966', '--pitch', '0',
                   '--yaw', '-1.5707963267948966',
                   '--frame-id', 'camera_link', '--child-frame-id', 'camera_optical_frame'],
    )
    detector = Node(
        package=PKG, executable='aruco_detector', name='aruco_detector', output='screen',
        parameters=[config],
    )
    return [base_to_camera, camera_to_optical, detector]


def generate_launch_description():
    share = get_package_share_directory(PKG)
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(
            share, 'config', 'aruco_localization.yaml')),
        DeclareLaunchArgument('start_vision', default_value='true',
                              description='also launch camera + rectification'),
        # Forwarded to vision.launch.py: they must match the calibration
        DeclareLaunchArgument('calib_dir', default_value=str(DEFAULT_CALIB_DIR)),
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='480'),
        DeclareLaunchArgument('flip_method', default_value='2'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'vision.launch.py')),
            launch_arguments={k: LaunchConfiguration(k) for k in
                              ('calib_dir', 'width', 'height', 'flip_method')}.items(),
            condition=IfCondition(LaunchConfiguration('start_vision'))),
        OpaqueFunction(function=_setup),
    ])
