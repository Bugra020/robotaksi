import os

from launch_ros.actions import IncludeLaunchDescription, Node

from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import FindPackageShare


def generate_launch_description():

    lidarslam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("lidarslam"), "launch", "lidarslam.launch.py"
                )
            ]
        ),
        launch_arguments=[
            ("use_odom", "true"),  # or false if you're doing pure scan matching
            ("publish_tf", "true"),
            ("set_initial_pose", "false"),
        ],
    )

    return LaunchDescription(
        [
            # Static TF publisher (e.g. base_link -> lidar_frame)
            Node(
                package="carla_lidar_publisher",
                executable="static_tf_publisher",
                name="static_tf_publisher",
                output="screen",
            ),
            # Dynamic TF publisher (e.g. odom -> base_link)
            Node(
                package="carla_lidar_publisher",
                executable="dynamic_tf_publisher",
                name="dynamic_tf_publisher",
                output="screen",
            ),
            lidarslam_launch,
        ]
    )
