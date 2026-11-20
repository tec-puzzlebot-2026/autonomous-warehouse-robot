"""ROS 2 side of PuzzleBot simulation (Isaac Sim runs separately on host)."""

import os
import shutil

from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, OpaqueFunction


def open_teleop(context, *args, **kwargs):
    if not os.environ.get("TMUX"):
        return [LogInfo(msg="Teleop not started: first enter tmux (tmux new -s puzzlebot).")]
    if shutil.which("tmux") is None:
        return [LogInfo(msg="Teleop not started: tmux is not installed.")]

    distro = os.environ.get("ROS_DISTRO", "jazzy")
    domain = os.environ.get("ROS_DOMAIN_ID", "0")
    command = (
        f"export ROS_DOMAIN_ID={domain}; "
        f"source /opt/ros/{distro}/setup.bash; "
        "if [ -f /ws/install/setup.bash ]; then source /ws/install/setup.bash; fi; "
        "exec ros2 run teleop_twist_keyboard teleop_twist_keyboard "
        "--ros-args -r cmd_vel:=/cmd_vel"
    )
    return [
        ExecuteProcess(
            cmd=["tmux", "new-window", "-n", "teleop", "bash", "-lc", command],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg="Start Isaac Sim on the DGX Spark host and press Play separately."),
        ExecuteProcess(
            cmd=["ros2", "topic", "hz", "/clock"],
            name="clock_monitor",
            output="screen",
        ),
        OpaqueFunction(function=open_teleop),
        # Later: Node(...) entries for encoder odometry, TF, and custom SLAM.
    ])