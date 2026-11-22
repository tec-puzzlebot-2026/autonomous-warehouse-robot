#!/usr/bin/env python3

"""
Launch the full PuzzleBot bringup LOCALLY on the Jetson:
micro-ROS agent (motors, encoders), RPLidar, camera/EKF launch and velocity relay.
(The Program Manager starts the same script itself in mode:=real.)

Run on the JETSON:

    ros2 launch main_ros jetson_bringup.launch.py

Drive from the laptop or a second SSH terminal:

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
                "jetson_bringup_script",
            ],
            output="screen",
            emulate_tty=True,
            # give the script time to stop the camera launch and the agent cleanly
            sigterm_timeout="20",
            sigkill_timeout="20",
        ),

    ])