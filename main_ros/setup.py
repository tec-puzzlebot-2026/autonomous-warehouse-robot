import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'main_ros'


setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),

        # General launch files
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')
        ),

        # Infrastructure launch files
        (
            os.path.join('share', package_name, 'infrastructure', 'launch'),
            glob('main_ros/infrastructure/launch/*.launch.py')
        ),

        # Vision launch files
        (
            os.path.join('share', package_name, 'vision', 'launch'),
            glob('main_ros/vision/launch/*.launch.py')
        ),

        # Voice launch files (if you keep voice.launch.py inside voice/launch/)
        (
            os.path.join('share', package_name, 'voice', 'launch'),
            glob('main_ros/voice/launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'), 
            glob('config/*.yaml')
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

            # Infrastructure: Debug teleoperation
            'debug_teleup_script = main_ros.infrastructure.scripts.debug_teleup_script:main',

            # Infrastructure: Differential drive
            'differential_drive = main_ros.infrastructure.scripts.differential_drive:main',

            # Navigation: Motion tests to measure noise for EKF odometry prediction model
            'motion_noise_test = main_ros.navigation.motion_noise_test:main',
            # Navigation: Analysis for EKF odometry prediction error
            'analyze_motion_noise = main_ros.navigation.analyze_motion_noise:main',

            # Voice: command recognition (HMM) -> /voice_command
            'voice_program_manager = main_ros.voice.voice_program_manager:main',

            # Vision: YOLO webcam test
            'yolo_webcam_test = main_ros.vision.testing.yolo_webcam_test:main',
        ],
    },
)