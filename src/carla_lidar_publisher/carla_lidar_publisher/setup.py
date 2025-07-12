from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'carla_lidar_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'models'), glob(os.path.join(package_name, package_name, 'VoltarisSim.pt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bugrakaya',
    maintainer_email='bugrakaya020@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_publisher = carla_lidar_publisher.lidar_publisher:main',
            'static_tf_publisher = carla_lidar_publisher.static_tf_publisher:main',
            'dynamic_tf_publisher = carla_lidar_publisher.dynamic_tf_publisher:main',
            'occupancy_grid_publisher = carla_lidar_publisher.occupancy_grid_publisher:main',
            'traffic_light_detector = carla_lidar_publisher.traffic_light_detector:main',
        ],
    },
)
