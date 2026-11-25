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
            # NAVIGATION: ODOMETRY
            # ==========================================

            # Custom differential-drive wheel odometry
            'wheel_odometry = main_ros.navigation.wheel_odometry:main',

            # Motion tests to measure noise for the EKF odometry prediction model
            'motion_noise_test = main_ros.navigation.motion_noise_test:main',

            # Analysis of the EKF odometry prediction error
            'motion_noise_analysis = main_ros.navigation.motion_noise_analysis:main',

            # ==========================================
            # NAVIGATION: EKF LOCALIZATION
            # ==========================================

            # EKF localization node
            'ekf_localization = main_ros.navigation.ekf_localization_node:main',

            # EKF consistency analysis
            'ekf_consistency_analysis = main_ros.navigation.ekf_consistency_analysis:main',

            # ==========================================
            # UTILS
            # ==========================================

            'debug_teleup_script = main_ros.utils.scripts.debug_teleup_script:main',

            'debug_teleup_headless = main_ros.utils.scripts.debug_teleup_headless:main',

            'go_to_pose = main_ros.navigation.go_to_pose:main',

            'differential_drive = main_ros.utils.scripts.differential_drive:main',
            
            'jetson_bringup_script = main_ros.utils.scripts.jetson_bringup_script:main',

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
            # VISION: ARUCO
            # ==========================================

            # ArUco marker detection and pose estimation
            'aruco_detector = main_ros.vision.aruco.aruco_detector_node:main',

            # ArUco sample taking
            'take_aruco_sample = main_ros.vision.aruco.take_aruco_sample:main',

            # ArUco error analysis
            'aruco_error_analysis = main_ros.vision.aruco.aruco_error_analysis:main',

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

            # ==========================================
            # PHASE 2: RESTOCKING
            # ==========================================

            # Hides the carried box from the LiDAR: /laser_scan -> /scan_nav
            'scan_filter = main_ros.navigation.scan_filter:main',

            # Puts ArUco markers and company logos on the SLAM map (Phase 1)
            'landmark_mapper = main_ros.restock.landmark_mapper_node:main',

            # Detectors: company logos, boxes (near: box_target, far: box_target as box_far), QR
            'logo_detector = main_ros.vision.logos.logo_detector_node:main',
            'box_target = main_ros.vision.boxes.box_target_node:main',
            'qr_reader = main_ros.vision.qr.qr_reader_node:main',

            # PickBox / SetDownBox placeholder until the forklift node exists
            'forklift_placeholder = main_ros.manipulation.forklift_placeholder:main',

            # Forklift control (the teammate's control_forklift: Tang Nano 9K lift + IR)
            #   forklift_controller  owns the Tang's serial port (start it first)
            #   forklift_operate     ROS interface: forklift/target_height_mm, IR readings
            #   forklift_pick_back   supervised Pick & Back ("Ready Pick Up" -> Done/Error)
            'forklift_controller = main_ros.manipulation.control_forklift.controller:main',
            'forklift_operate = main_ros.manipulation.control_forklift.operate:main',
            'forklift_pick_back = main_ros.manipulation.control_forklift.pick_back:main',

            # Isaac Sim /joint_states -> /VelocityEncL, /VelocityEncR (for the EKF)
            'encoder_bridge = main_ros.simulation.encoder_bridge:main',
        ],
    },
)
