"""host.launch.py — the Team Zero warehouse robot (host PC side).

Opens the mission monitor (running on the robot side, shown here through
ssh) and Foxglove Studio connected to the robot's Foxglove bridge. The host
does not join the robot's ROS 2 network, so it works with a Humble laptop
and a Jazzy robot, for the simulation and the real robot.

    ros2 launch main_ros host.launch.py                      # simulation
    ros2 launch main_ros host.launch.py mode:=real           # the robot
    ros2 launch main_ros host.launch.py visualization:=none  # monitor only

Ctrl+C closes what this launch opened; the robot keeps running.
"""
# Author: Raymundo Gomez

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """The host manager as a plain process (no ROS node on this PC)."""
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='sim', choices=['sim', 'real']),
        DeclareLaunchArgument('visualization', default_value='foxglove',
                              choices=['foxglove', 'rviz', 'none']),
        DeclareLaunchArgument('monitor', default_value='true'),
        DeclareLaunchArgument('config_file', default_value=''),
        ExecuteProcess(
            cmd=['ros2', 'run', 'main_ros', 'host_manager',
                 '--mode', LaunchConfiguration('mode'),
                 '--visualization', LaunchConfiguration('visualization'),
                 '--monitor', LaunchConfiguration('monitor'),
                 '--config-file', LaunchConfiguration('config_file')],
            name='host_manager',
            output='screen',
            emulate_tty=True,
            sigterm_timeout='20',
            sigkill_timeout='5',
        ),
    ])
