"""Pi Camera v2 driver -> /camera/image_raw, /camera/camera_info.

    ros2 launch main_ros camera.launch.py
    ros2 launch main_ros camera.launch.py width:=1280 height:=960 flip_method:=2
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Edit the default folder in main_ros/vision/calibration/calib_io.py
from main_ros.vision.camera.calibration.calib_io import DEFAULT_CAMERA_CALIB_DIR  # noqa: E402


def generate_launch_description():
    args = [
        DeclareLaunchArgument('calib_dir', default_value=DEFAULT_CAMERA_CALIB_DIR,
                              description='Folder with images/, whiteimages/, data/'),
        DeclareLaunchArgument('capture_width', default_value='1640'),
        DeclareLaunchArgument('capture_height', default_value='1232'),
        DeclareLaunchArgument('width', default_value='640',
                              description='Published width (calibration is tied to it)'),
        DeclareLaunchArgument('height', default_value='480'),
        DeclareLaunchArgument('framerate', default_value='30'),
        DeclareLaunchArgument('flip_method', default_value='2',
                              description='nvvidconv flip-method (2 = rotate 180)'),
        DeclareLaunchArgument('argus_extra', default_value='',
                              description='extra nvarguscamerasrc properties'),
    ]

    def lc(name, t=None):
        v = LaunchConfiguration(name)
        return ParameterValue(v, value_type=t) if t else v

    camera = Node(
        package='main_ros', executable='camera_node', name='camera_node',
        namespace='camera', output='screen',
        parameters=[{
            'calib_dir': lc('calib_dir', str),
            'capture_width': lc('capture_width', int),
            'capture_height': lc('capture_height', int),
            'width': lc('width', int),
            'height': lc('height', int),
            'framerate': lc('framerate', int),
            'flip_method': lc('flip_method', int),
            'argus_extra': lc('argus_extra', str),
        }],
    )
    return LaunchDescription(args + [camera])
