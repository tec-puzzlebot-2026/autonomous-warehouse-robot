"""Camera + ImageSaver for collecting calibration images (custom OpenCV route).

Checkerboard images (-> <calib_dir>/images):
    ros2 launch main_ros calibration_capture.launch.py
White images for colour calibration (-> <calib_dir>/whiteimages):
    ros2 launch main_ros calibration_capture.launch.py mode:=white
Then, in another terminal, once per pose:
    ros2 topic pub --once /save_image std_msgs/msg/Bool "{data: true}"
Watch /calibration/preview (viewer:=true opens rqt_image_view on it).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Edit the default folder in main_ros/vision/calibration/calib_io.py
from main_ros.vision.camera.calibration.calib_io import DEFAULT_CAMERA_CALIB_DIR  # noqa: E402


def generate_launch_description():
    calib_dir = LaunchConfiguration('calib_dir')
    args = [
        DeclareLaunchArgument('calib_dir', default_value=DEFAULT_CAMERA_CALIB_DIR),
        DeclareLaunchArgument('mode', default_value='checkerboard',
                              description='checkerboard | white'),
        DeclareLaunchArgument('pattern_cols', default_value='7',
                              description='inner corners per row (8 squares -> 7)'),
        DeclareLaunchArgument('pattern_rows', default_value='5',
                              description='inner corners per column (6 squares -> 5)'),
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='480'),
        DeclareLaunchArgument('viewer', default_value='false',
                              description='open rqt_image_view on /calibration/preview'),
    ]

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('main_ros'), 'launch', 'camera.launch.py')),
        launch_arguments={'calib_dir': calib_dir,
                          'width': LaunchConfiguration('width'),
                          'height': LaunchConfiguration('height')}.items(),
    )
    saver = Node(
        package='main_ros', executable='image_saver', name='image_saver', output='screen',
        parameters=[{
            'calib_dir': ParameterValue(calib_dir, value_type=str),
            'mode': ParameterValue(LaunchConfiguration('mode'), value_type=str),
            'pattern_cols': ParameterValue(LaunchConfiguration('pattern_cols'), value_type=int),
            'pattern_rows': ParameterValue(LaunchConfiguration('pattern_rows'), value_type=int),
        }],
    )
    viewer = Node(
        package='rqt_image_view', executable='rqt_image_view',
        arguments=['/calibration/preview'],
        condition=IfCondition(LaunchConfiguration('viewer')),
    )
    return LaunchDescription(args + [camera, saver, viewer])
