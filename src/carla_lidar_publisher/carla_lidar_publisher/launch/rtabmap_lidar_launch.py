import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                '_subscribe_depth': True,
                '_subscribe_scan': True,
                '_frame_id': 'base_link',
                '_laser_scan_topic': '/lidar_topic',
            }]
        ),
    ])
