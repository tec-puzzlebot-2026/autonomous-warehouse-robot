"""
yolo_webcam_test.launch.py - test the logo detector with the laptop webcam.

    ros2 launch main_ros yolo_webcam_test.launch.py
    ros2 launch main_ros yolo_webcam_test.launch.py camera_index:=2 conf:=0.4
    ros2 launch main_ros yolo_webcam_test.launch.py model_name:=other_model.pt
    ros2 launch main_ros yolo_webcam_test.launch.py viewer:=true      # also open rqt_image_view
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument("model_name", default_value="",
                              description="model file in vision/models/ ('' = MODEL_NAME in the script)"),
        DeclareLaunchArgument("camera_index", default_value="0",
                              description="webcam index (/dev/videoN)"),
        DeclareLaunchArgument("conf", default_value="0.5",
                              description="minimum detection confidence"),
        DeclareLaunchArgument("show_window", default_value="true",
                              description="show an OpenCV window with the result"),
        DeclareLaunchArgument("viewer", default_value="false",
                              description="also start rqt_image_view"),
    ]

    detector = Node(
        package="main_ros",
        executable="yolo_webcam_test",
        name="yolo_webcam_test",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "model_name": ParameterValue(lc("model_name"), value_type=str),
            "camera_index": ParameterValue(lc("camera_index"), value_type=int),
            "conf": ParameterValue(lc("conf"), value_type=float),
            "show_window": ParameterValue(lc("show_window"), value_type=bool),
        }],
    )

    viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        arguments=["/logo_detections/image"],
        condition=IfCondition(lc("viewer")),
    )

    return LaunchDescription(args + [detector, viewer])