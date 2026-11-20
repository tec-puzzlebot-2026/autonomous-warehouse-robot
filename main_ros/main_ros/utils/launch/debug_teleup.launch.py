#!/usr/bin/env python3

"""
Launch the full PuzzleBot bringup from the HOST (laptop): connects to the
Jetson over SSH and starts the micro-ROS agent (motors, encoders, LiDAR) and
the camera/EKF launch there, then the velocity relay and keyboard teleop here.

Run from the HOST PC:

    ros2 launch main_ros debug_teleup.launch.py
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():

    return LaunchDescription([

        ExecuteProcess(
            cmd=[
                "ros2",
                "run",
                "main_ros",
                "debug_teleup_script",
            ],
            output="screen",
            emulate_tty=True,
            # give the script time to stop the Jetson processes cleanly
            sigterm_timeout="20",
            sigkill_timeout="20",
        ),

    ])