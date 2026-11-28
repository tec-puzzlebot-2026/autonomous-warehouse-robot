"""main.launch.py — the Team Zero warehouse robot (robot / simulation side).

Starts only the Program Manager. The Program Manager starts every other
script itself, stage by stage, waits until each is ready, and runs the mission.
Nothing moves until the operator sends S twice: the first S brings the system
up, the second starts mapping.

    ros2 launch main_ros main.launch.py                 # simulation (Isaac Sim)
    ros2 launch main_ros main.launch.py mode:=real      # the robot (Jetson)
    ros2 launch main_ros main.launch.py autostart:=true # bring up right away

One run, four stages; after each one the monitor waits for a key (S next, K skip):
    1 mapping      Frontera explores, Planisferio maps (no camera: light)
    2 map quality  optional: the robot drives round the map with the camera on, closes
                   loops and places the ArUco markers and logos on the map
    3 route        the restocking patrol route on the final map
    4 restocking   boxes to the docks
    map:=latest | <mission folder>   start from a saved map: stage 1 is skipped
    map:=restock                     the newest saved map that has a drop zone
    start_pose:=origin | x,y,yaw     with map:=, the robot's pose is sent for you
    restock_only:=true               with map:=, straight on to stages 3 and 4

    Test restocking only (Isaac stopped and played, so the robot is at its start):
      ros2 launch main_ros main.launch.py mode:=sim map:=restock start_pose:=origin \
          restock_only:=true autostart:=true

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
            'map': ParameterValue(LaunchConfiguration('map'), value_type=str),
            'start_pose': ParameterValue(LaunchConfiguration('start_pose'), value_type=str),
            'restock_only': ParameterValue(LaunchConfiguration('restock_only'),
                                           value_type=bool),
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
        DeclareLaunchArgument('map', default_value='',
                              description='start from a saved map: latest, or a mission '
                                          'folder name or path in <maps_dir>/missions'),
        DeclareLaunchArgument('start_pose', default_value='',
                              description="with map:=, the robot's pose on that map, sent "
                                          'for you: x,y,yaw or origin (the mapping start)'),
        DeclareLaunchArgument('restock_only', default_value='false',
                              description='with map:=, once localized: skip stage 2, plan '
                                          'the route and start restocking by itself'),
        manager,
        # The Program Manager closed (after X, or it crashed): end the launch.
        RegisterEventHandler(OnProcessExit(
            target_action=manager,
            on_exit=[EmitEvent(event=Shutdown(reason='Program Manager closed'))])),
    ])
