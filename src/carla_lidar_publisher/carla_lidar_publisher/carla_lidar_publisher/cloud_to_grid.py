import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class CloudToGrid(Node):
    def __init__(self):
        super().__init__('cloud_to_grid')
        self.subcription = self.create_subscription(PointCloud)
