#!/usr/bin/env python3

"""
Launch PuzzleBot debug teleoperation.

Run from HOST PC:

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
        ),

    ])