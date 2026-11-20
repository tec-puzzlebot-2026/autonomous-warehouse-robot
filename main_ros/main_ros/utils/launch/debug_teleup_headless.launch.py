#!/usr/bin/env python3

"""
Launch PuzzleBot teleoperation WITHOUT a terminal window (Docker / no screen).

Run from the HOST PC (container):

    ros2 launch main_ros debug_teleup_headless.launch.py

Then, in a second shell:

    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in
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
                "debug_teleup_headless",
            ],
            output="screen",
        ),

    ])