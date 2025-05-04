import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import struct

class OccupancyGridPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_grid_publisher')

        self.map_width = 100
        self.map_height = 100
        self.resolution = 1 
        self.origin_x = -20
        self.origin_y = -20

        self.subscription = self.create_subscription(PointCloud2, '/lidar_topic', self.pointcloud_callback, 10)
        self.publisher = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)

        self.get_logger().info("Listening to /lidar_topic and publishing to /occupancy_grid")

    def pointcloud_callback(self, msg: PointCloud2):
        grid = [0] * (self.map_width * self.map_height)

        point_step = msg.point_step
        x_offset = next(f.offset for f in msg.fields if f.name == 'x')
        y_offset = next(f.offset for f in msg.fields if f.name == 'y')
        z_offset = next(f.offset for f in msg.fields if f.name == 'z')

        for i in range(0, len(msg.data), point_step):
            x = struct.unpack_from('f', msg.data, i + x_offset)[0]
            y = struct.unpack_from('f', msg.data, i + y_offset)[0]
            z = struct.unpack_from('f', msg.data, i + z_offset)[0]

            if z < 0.5 or z > 2.0:
                continue

            mx = int((x - self.origin_x) / self.resolution)
            my = int((y - self.origin_y) / self.resolution)

            if 0 <= mx < self.map_width and 0 <= my < self.map_height:
                idx = my * self.map_width + mx
                grid[idx] = 100

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = msg.header.stamp
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.map_width
        grid_msg.info.height = self.map_height
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = grid

        self.publisher.publish(grid_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
