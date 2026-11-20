import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'main_ros'


setup(
    name=package_name,
    version='0.0.0',

    # Include main_ros and all nested Python packages
    packages=find_packages(
        include=[
            package_name,
            f'{package_name}.*'
        ],
        exclude=['test']
    ),

    data_files=[
        # ==========================================
        # ROS 2 PACKAGE REGISTRATION
        # ==========================================
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),

        (
            'share/' + package_name,
            ['package.xml']
        ),

        # ==========================================
        # LAUNCH FILES
        # ==========================================

        # General launch files
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')
        ),

        # Utils launch files
        (
            os.path.join('share', package_name, 'utils', 'launch'),
            glob('main_ros/utils/launch/*.launch.py')
        ),

        # Vision launch files
        (
            os.path.join('share', package_name, 'vision', 'launch'),
            glob('main_ros/vision/launch/*.launch.py')
        ),

        # Voice launch files
        (
            os.path.join('share', package_name, 'voice', 'launch'),
            glob('main_ros/voice/launch/*.launch.py')
        ),

        # ==========================================
        # CONFIGURATION FILES (parameters and the robot URDF)
        # ==========================================
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.urdf')
        ),

        # ==========================================
        # RVIZ CONFIGURATIONS
        # ==========================================
        (
            os.path.join('share', package_name, 'rviz'),
            glob('main_ros/mapping/rviz/*.rviz')
        ),
    ],

    install_requires=[
        'setuptools',
    ],

    zip_safe=True,

    maintainer='goma',
    maintainer_email='a01285451@tec.mx',

    description='ROS 2 package for the autonomous warehouse robot.',

    license='MIT',

    tests_require=[
        'pytest',
    ],

    entry_points={
        'console_scripts': [

            # ==========================================
            # SIMULATION: MANUAL SLAM
            # ==========================================

            # Starts manual teleop, wheel odometry and SLAM
            'manual_program_manager = main_ros.simulation.scripts.manual_sim.program_manager:main',

            # Keyboard teleoperation
            'manual_teleop = main_ros.simulation.scripts.manual_sim.manual_teleop:main',

            # ==========================================
            # MAPPING: CUSTOM SLAM
            # ==========================================

            # Our custom LiDAR SLAM algorithm
            'slam = main_ros.mapping.slam:main',

            # RViz launcher (runs on laptop)
            'slam_rviz = main_ros.mapping.slam_rviz:main',

            # ==========================================
            # NAVIGATION: ODOMETRY
            # ==========================================

            # Custom differential-drive wheel odometry
            'wheel_odometry = main_ros.navigation.wheel_odometry:main',

            # Motion noise measurement
            'motion_noise_test = main_ros.navigation.motion_noise_test:main',

            # Motion noise analysis
            'analyze_motion_noise = main_ros.navigation.analyze_motion_noise:main',

            # ==========================================
            # UTILS
            # ==========================================

            'debug_teleup_script = main_ros.utils.scripts.debug_teleup_script:main',

            'differential_drive = main_ros.utils.scripts.differential_drive:main',

            # ==========================================
            # VOICE
            # ==========================================

            'voice_program_manager = main_ros.voice.voice_program_manager:main',

            # ==========================================
            # VISION: CAMERA
            # ==========================================

            'camera_node = main_ros.vision.camera.camera_node:main',

            'image_processing = main_ros.vision.camera.image_processing_node:main',

            'image_saver = main_ros.vision.camera.calibration.image_saver:main',

            'calibrate_intrinsics = main_ros.vision.camera.calibration.calibrate_intrinsics:main',

            'color_calibrate = main_ros.vision.camera.calibration.color_calibrate:main',

            # ==========================================
            # VISION: TESTING
            # ==========================================

            'yolo_webcam_test = main_ros.vision.testing.yolo_webcam_test:main',

            # ==========================================
            # CONTROL: PROGRAM MANAGER (robot side) AND HOST SIDE
            # ==========================================

            'program_manager = main_ros.control.program_manager:main',
            'mission_monitor = main_ros.control.mission_monitor:main',
            'host_manager = main_ros.control.host_manager:main',
        ],
    },
)
