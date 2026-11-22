"""main.launch.py — the Team Zero warehouse robot (robot / simulation side).

Starts only the Program Manager. The Program Manager starts every other
script itself (odometry, SLAM, Sendero, Frontera, drivers on the real robot),
waits until each is ready, and runs the mission. Nothing moves until the
operator sends S twice: the first S brings the system up, the second starts
exploring.

    ros2 launch main_ros main.launch.py                 # simulation (Isaac Sim)
    ros2 launch main_ros main.launch.py mode:=real      # the robot (Jetson)
    ros2 launch main_ros main.launch.py autostart:=true # bring up right away

Missions:
    mission:=mapping      Phase 1: explore, map, find the ArUco markers, logos and docks;
                          the end of mapping saves a mission folder (default)
    mission:=restocking   Phase 2 on the latest mission folder (or map:=<folder>)
    mission:=full         Phase 1, then R in the monitor starts Phase 2 on the new map

The monitor is opened on the host PC by host.launch.py.
Ctrl+C here halts the robot and closes every script. X in the monitor does
the same and then ends this launch (exit_on_stop in program_manager.yaml).
"""
# Author: Raymundo Gomez

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Program Manager with its three arguments; the launch ends with it."""
    manager = Node(
        package='main_ros',
        executable='program_manager',
        name='program_manager',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'mode': LaunchConfiguration('mode'),
            'config_file': LaunchConfiguration('config_file'),
            'autostart': ParameterValue(LaunchConfiguration('autostart'), value_type=bool),
            'mission': LaunchConfiguration('mission'),
            'map': ParameterValue(LaunchConfiguration('map'), value_type=str),
        }],
        # Give the Program Manager time to halt the robot and close every
        # script after Ctrl+C before launch escalates to SIGTERM / SIGKILL.
        sigterm_timeout='40',
        sigkill_timeout='10',
    )
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='sim', choices=['sim', 'real'],
                              description='sim (Isaac Sim on the Spark) or real (Jetson)'),
        DeclareLaunchArgument('config_file', default_value='',
                              description='program_manager.yaml (default: main_ros/config)'),
        DeclareLaunchArgument('autostart', default_value='false',
                              description='true = bring the system up without waiting for S'),
        DeclareLaunchArgument('mission', default_value='mapping',
                              choices=['mapping', 'restocking', 'full'],
                              description='mapping (Phase 1), restocking (Phase 2) or full'),
        DeclareLaunchArgument('map', default_value='',
                              description='restocking: mission folder name or path '
                                          '(default: <maps_dir>/missions/latest)'),
        manager,
        # The Program Manager closed (after X, or it crashed): end the launch.
        RegisterEventHandler(OnProcessExit(
            target_action=manager,
            on_exit=[EmitEvent(event=Shutdown(reason='Program Manager closed'))])),
    ])
