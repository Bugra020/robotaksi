import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from builtin_interfaces.msg import Time

class StaticTFPublisher(Node):
    def __init__(self):
        super().__init__('static_tf_publisher')
        
        self.br = StaticTransformBroadcaster(self)
        
        static_transform = TransformStamped()
        
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'base_link'
        static_transform.child_frame_id = 'lidar'
        
        static_transform.transform.translation.x = 0.0  # No movement in the x-axis
        static_transform.transform.translation.y = 0.0  # No movement in the y-axis
        static_transform.transform.translation.z = 0.2  # 0.2 meters above base_link
        
        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0  # Identity quaternion (no rotation)
        
        self.br.sendTransform(static_transform)
        
        self.get_logger().info('Broadcasting static transform from base_link to lidar')

def main(args=None):
    rclpy.init(args=args)
    node = StaticTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
