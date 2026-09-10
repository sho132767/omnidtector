import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'omni_follower'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
     
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sho',
    maintainer_email='sho@todo.todo',
    description='Camera-based target follower for omni-wheel robot',
    license='TODO-License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'target_detector = omni_follower.target_detector:main',
            'follower_controller = omni_follower.follower_controller:main',
            'stm32_bridge = omni_follower.stm32_bridge:main',
        ],
    },
)