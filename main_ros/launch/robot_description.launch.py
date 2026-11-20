"""robot_description.launch.py — publishes the PuzzleBot URDF.

robot_state_publisher puts the URDF on /robot_description (Foxglove draws
the robot from it) and the fixed joints on /tf_static (base_link ->
laser_frame, used by SLAM). Started by the Program Manager (step
"robot_description"); can also be run by hand:

    ros2 launch main_ros robot_description.launch.py use_sim_time:=true
    ros2 launch main_ros robot_description.launch.py urdf:=/path/robot.urdf
"""
# Author: Raymundo Gomez

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _publisher(context):
    path = LaunchConfiguration('urdf').perform(context)
    with open(os.path.expanduser(path)) as f:
        urdf = f.read()
    sim = LaunchConfiguration('use_sim_time').perform(context).lower() in ('true', '1')
    return [Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf, 'use_sim_time': sim}],
    )]


def generate_launch_description():
    default = os.path.join(get_package_share_directory('main_ros'), 'config', 'puzzlebot.urdf')
    return LaunchDescription([
        DeclareLaunchArgument('urdf', default_value=default,
                              description='URDF file (default: main_ros/config/puzzlebot.urdf)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=_publisher),
    ])
