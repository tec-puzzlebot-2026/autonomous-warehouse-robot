"""Runtime vision pipeline: camera -> colour correction -> rectification.

    ros2 launch main_ros vision.launch.py                         # OpenCV route
    ros2 launch main_ros vision.launch.py rectifier:=image_proc   # image_pipeline route
    ros2 launch main_ros vision.launch.py apply_color:=false

Topics: /camera/image_raw, /camera/camera_info, /camera/image_cc (colour
corrected), /camera/image_rect (colour corrected + rectified).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue

# Edit the default folder in main_ros/vision/calibration/calib_io.py
from main_ros.vision.camera.calibration.calib_io import DEFAULT_CAMERA_CALIB_DIR  # noqa: E402


def generate_launch_description():
    calib_dir = LaunchConfiguration('calib_dir')
    rectifier = LaunchConfiguration('rectifier')
    apply_color = LaunchConfiguration('apply_color')

    use_opencv = PythonExpression(["'", rectifier, "' == 'opencv'"])
    use_image_proc = PythonExpression(["'", rectifier, "' == 'image_proc'"])
    color_on = PythonExpression(["'", apply_color, "'.lower() == 'true'"])
    run_processing = PythonExpression(
        ["'", rectifier, "' == 'opencv' or '", apply_color, "'.lower() == 'true'"])
    rectify_input = PythonExpression(
        ["'/camera/image_cc' if '", apply_color, "'.lower() == 'true' else '/camera/image_raw'"])

    args = [
        DeclareLaunchArgument('calib_dir', default_value=DEFAULT_CAMERA_CALIB_DIR),
        DeclareLaunchArgument('rectifier', default_value='opencv',
                              description='opencv (main_ros node) | image_proc (image_pipeline)'),
        DeclareLaunchArgument('apply_color', default_value='true'),
        DeclareLaunchArgument('alpha', default_value='-1.0',
                              description='OpenCV route: <0 use P from camera_info.yaml, '
                                          '0 crop, 1 keep all pixels'),
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='480'),
        DeclareLaunchArgument('flip_method', default_value='2'),
    ]

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('main_ros'), 'launch', 'camera.launch.py')),
        launch_arguments={'calib_dir': calib_dir,
                          'width': LaunchConfiguration('width'),
                          'height': LaunchConfiguration('height'),
                          'flip_method': LaunchConfiguration('flip_method')}.items(),
    )

    processing = Node(
        package='main_ros', executable='image_processing', name='image_processing',
        output='screen', condition=IfCondition(run_processing),
        parameters=[{
            'calib_dir': ParameterValue(calib_dir, value_type=str),
            'apply_color': ParameterValue(color_on, value_type=bool),
            'apply_rectify': ParameterValue(use_opencv, value_type=bool),
            'alpha': ParameterValue(LaunchConfiguration('alpha'), value_type=float),
        }],
    )

    # image_pipeline route: image_proc::RectifyNode reads `image` and the sibling
    # camera_info (/camera/camera_info, published by camera_node with the same stamp).
    image_proc = ComposableNodeContainer(
        name='image_proc_container', namespace='camera',
        package='rclcpp_components', executable='component_container', output='screen',
        condition=IfCondition(use_image_proc),
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc', plugin='image_proc::RectifyNode',
                name='rectify_node', namespace='camera',
                remappings=[('image', rectify_input), ('image_rect', '/camera/image_rect')],
                parameters=[{'interpolation': 1}],
            ),
        ],
    )
    return LaunchDescription(args + [camera, processing, image_proc])
