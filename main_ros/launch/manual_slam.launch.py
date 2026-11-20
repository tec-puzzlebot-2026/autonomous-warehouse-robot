from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        Node(
            package="main_ros",
            executable="manual_program_manager",
            name="program_manager",
            output="screen"
        )

    ])