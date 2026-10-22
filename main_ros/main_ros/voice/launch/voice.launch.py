"""
voice.launch.py - starts the voice command system.

    ros2 launch main_ros voice.launch.py
    ros2 launch main_ros voice.launch.py mic_name:=USB threshold:=0.5 verbose:=true

Output topics:
    /voice_command  (std_msgs/String)  recognised word, e.g. "stop"
    /voice_status   (std_msgs/String)  starting | calibrating | listening | error: ...
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("model_path", default_value="",
                              description="models.pkl path ('' = voice/scripts/models.pkl)"),
        DeclareLaunchArgument("threshold", default_value="",
                              description="rejection threshold ('' = value from training)"),
        DeclareLaunchArgument("stop_bonus", default_value="1.0",
                              description="lower the threshold for 'stop' by this much"),
        DeclareLaunchArgument("device", default_value="-1",
                              description="microphone index (-1 = mic_name or default)"),
        DeclareLaunchArgument("mic_name", default_value="",
                              description="part of the microphone name, e.g. USB"),
        DeclareLaunchArgument("cooldown", default_value="1.0",
                              description="seconds before the same word can be published again"),
        DeclareLaunchArgument("verbose", default_value="false"),
        DeclareLaunchArgument("command_topic", default_value="/voice_command"),
        DeclareLaunchArgument("status_topic", default_value="/voice_status"),
    ]

    lc = LaunchConfiguration
    manager = Node(
        package="main_ros",
        executable="voice_program_manager",
        name="voice_program_manager",
        output="screen",
        emulate_tty=True,          # show logs immediately, with colours
        respawn=True,              # restart automatically if it crashes
        respawn_delay=2.0,
        parameters=[{
            "model_path": ParameterValue(lc("model_path"), value_type=str),
            "threshold": ParameterValue(lc("threshold"), value_type=str),
            "stop_bonus": ParameterValue(lc("stop_bonus"), value_type=float),
            "device": ParameterValue(lc("device"), value_type=int),
            "mic_name": ParameterValue(lc("mic_name"), value_type=str),
            "cooldown": ParameterValue(lc("cooldown"), value_type=float),
            "verbose": ParameterValue(lc("verbose"), value_type=bool),
        }],
        remappings=[
            ("voice_command", lc("command_topic")),
            ("voice_status", lc("status_topic")),
        ],
    )

    return LaunchDescription(args + [manager])