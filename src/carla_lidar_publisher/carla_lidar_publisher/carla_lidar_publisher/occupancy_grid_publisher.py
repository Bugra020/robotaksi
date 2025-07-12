import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
import struct

class OccupancyGridPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_grid_publisher')

        self.declare_parameter('map_width', 100)
        self.declare_parameter('map_height', 100)
        self.declare_parameter('resolution', 1.0) 
        self.declare_parameter('origin_x', -50.0)
        self.declare_parameter('origin_y', -50.0)

        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.resolution = self.get_parameter('resolution').value
        self.origin_x = self.get_parameter('origin_x').value
        self.origin_y = self.get_parameter('origin_y').value

        self.declare_parameter('min_z_filter', 0.2)
        self.declare_parameter('max_z_filter', 2.0)
        self.min_z_filter = self.get_parameter('min_z_filter').value
        self.max_z_filter = self.get_parameter('max_z_filter').value

        self.subscription = self.create_subscription(
            PointCloud2,
            '/lidar_topic',
            self.pointcloud_callback,
            10
        )
        self.publisher = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)

        self.get_logger().info(f"Occupancy Grid Publisher initialized:")
        self.get_logger().info(f"  Map Dimensions: {self.map_width}x{self.map_height} cells")
        self.get_logger().info(f"  Resolution: {self.resolution} m/cell")
        self.get_logger().info(f"  Origin: ({self.origin_x:.2f}, {self.origin_y:.2f})")
        self.get_logger().info(f"  Z-filter: [{self.min_z_filter:.2f}, {self.max_z_filter:.2f}]")
        self.get_logger().info(f"Listening to /lidar_topic and publishing to /occupancy_grid")

    def pointcloud_callback(self, msg: PointCloud2):
        grid = [0] * (self.map_width * self.map_height) 
        
        x_offset = -1
        y_offset = -1
        z_offset = -1

        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
        
        if x_offset == -1 or y_offset == -1 or z_offset == -1:
            self.get_logger().error("PointCloud2 message missing 'x', 'y', or 'z' fields!")
            return

        point_step = msg.point_step
        total_points = len(msg.data) // point_step
        self.get_logger().debug(f"Processing PointCloud2 with {total_points} points.")

        points_processed = 0
        points_filtered_z = 0
        points_outside_map = 0
        points_occupied = 0

        for i in range(0, len(msg.data), point_step):
            points_processed += 1
            try:
                x = struct.unpack_from('f', msg.data, i + x_offset)[0]
                y = struct.unpack_from('f', msg.data, i + y_offset)[0]
                z = struct.unpack_from('f', msg.data, i + z_offset)[0]
            except struct.error as e:
                self.get_logger().error(f"Error unpacking point data at index {i}: {e}")
                continue

            if z < self.min_z_filter or z > self.max_z_filter:
                points_filtered_z += 1
                continue

            mx = int((x - self.origin_x) / self.resolution)
            my = int((y - self.origin_y) / self.resolution)

            if 0 <= mx < self.map_width and 0 <= my < self.map_height:
                idx = my * self.map_width + mx
                if grid[idx] != 100:
                    grid[idx] = 100
                    points_occupied += 1
            else:
                points_outside_map += 1

        self.get_logger().info(f"PointCloud processing summary:")
        self.get_logger().info(f"  Total points received: {total_points}")
        self.get_logger().info(f"  Points processed (iterated): {points_processed}")
        self.get_logger().info(f"  Points filtered by Z-axis: {points_filtered_z}")
        self.get_logger().info(f"  Points outside map bounds: {points_outside_map}")
        self.get_logger().info(f"  Occupied cells marked: {points_occupied}")
        
        if points_occupied == 0 and total_points > 0:
            self.get_logger().warn("No occupied cells were marked. Check Z-filter, map origin/dimensions, or input point cloud data range.")

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = msg.header.stamp
        grid_msg.header.frame_id = 'base_link'

        grid_msg.info.resolution = float(self.resolution)
        grid_msg.info.width = self.map_width
        grid_msg.info.height = self.map_height

        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position = Point(x=float(self.origin_x), y=float(self.origin_y), z=0.0)
        grid_msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        grid_msg.data = grid

        self.publisher.publish(grid_msg)
        self.get_logger().debug("Published OccupancyGrid message.")


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped cleanly.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
