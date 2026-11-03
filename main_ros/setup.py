import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'main_ros'


setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(include=[package_name, f'{package_name}.*'], exclude=['test']),

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

            # Utils: Debug teleoperation
            'debug_teleup_script = main_ros.utils.scripts.debug_teleup_script:main',

            # Utils: Differential drive
            'differential_drive = main_ros.utils.scripts.differential_drive:main',

            # Navigation: Motion tests to measure noise for EKF odometry prediction model
            'motion_noise_test = main_ros.navigation.motion_noise_test:main',
            # Navigation: Analysis for EKF odometry prediction error
            'analyze_motion_noise = main_ros.navigation.analyze_motion_noise:main',

            # Voice: command recognition (HMM) -> /voice_command
            'voice_program_manager = main_ros.voice.voice_program_manager:main',

            # Vision: Raspberry Pi Camera v2 untouched sensor image and info publishing
            'camera_node = main_ros.vision.camera.camera_node:main',
            # Vision: Color corrects and rectifies sensor image
            'image_processing = main_ros.vision.camera.image_processing_node:main',
            # Vision: Latest *raw* camera frame as image saving
            'image_saver = main_ros.vision.camera.calibration.image_saver:main',
            # Vision: Intrinsic camera calibration with OpenCV
            'calibrate_intrinsics = main_ros.vision.camera.calibration.calibrate_intrinsics:main',
            # Vision: Color calibration
            'color_calibrate = main_ros.vision.camera.calibration.color_calibrate:main',

            # Vision: YOLO webcam test
            'yolo_webcam_test = main_ros.vision.testing.yolo_webcam_test:main',
        ],
    },
)