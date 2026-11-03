"""Camera + image_pipeline cameracalibrator (GUI, needs a display).

    ros2 launch main_ros calibration_image_pipeline.launch.py

Move the board until X, Y, Size and Skew bars are green, press CALIBRATE,
then COMMIT: the calibrator calls /camera/set_camera_info and camera_node
writes <calib_dir>/data/camera_info.yaml. SAVE also stores the images and
ost.yaml in /tmp/calibrationdata.tar.gz.

Headless Jetson: run only camera.launch.py on the Jetson and start the
calibrator on a laptop on the same network / ROS_DOMAIN_ID with the command
from the README.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Edit the default folder in main_ros/vision/calibration/calib_io.py
from main_ros.vision.camera.calibration.calib_io import DEFAULT_CAMERA_CALIB_DIR  # noqa: E402


def generate_launch_description():
    args = [
        DeclareLaunchArgument('calib_dir', default_value=DEFAULT_CAMERA_CALIB_DIR),
        DeclareLaunchArgument('size', default_value='7x5',
                              description='INNER corners (8x6 squares -> 7x5)'),
        DeclareLaunchArgument('square', default_value='0.03', description='metres'),
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='480'),
    ]
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('main_ros'), 'launch', 'camera.launch.py')),
        launch_arguments={'calib_dir': LaunchConfiguration('calib_dir'),
                          'width': LaunchConfiguration('width'),
                          'height': LaunchConfiguration('height')}.items(),
    )
    calibrator = Node(
        package='camera_calibration', executable='cameracalibrator', output='screen',
        arguments=['--size', LaunchConfiguration('size'),
                   '--square', LaunchConfiguration('square'),
                   '--camera_name', 'pi_camera_v2'],
        remappings=[('image', '/camera/image_raw'),
                    ('camera/set_camera_info', '/camera/set_camera_info')],
    )
    return LaunchDescription(args + [camera, calibrator])
