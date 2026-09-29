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
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')
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
        ],
    },
)