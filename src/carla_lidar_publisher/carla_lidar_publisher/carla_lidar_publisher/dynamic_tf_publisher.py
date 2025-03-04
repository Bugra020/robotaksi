import carla
import geometry_msgs.msg
import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node


class DynamicTFPublisher(Node):
    def __init__(self, vehicle):
        super().__init__("carla_tf_publisher")
        self.vehicle = vehicle
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.publish_tf)  # Publish at 20Hz

    def publish_tf(self):
        transform = self.vehicle.get_transform()

        tf_msg = geometry_msgs.msg.TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_link"

        tf_msg.transform.translation.x = transform.location.x
        tf_msg.transform.translation.y = transform.location.y
        tf_msg.transform.translation.z = transform.location.z

        yaw = np.deg2rad(transform.rotation.yaw)
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = np.sin(yaw / 2)
        tf_msg.transform.rotation.w = np.cos(yaw / 2)

        self.tf_broadcaster.sendTransform(tf_msg)


def main():
    rclpy.init()

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.get_world()
    vehicle = world.get_actors().filter("vehicle.*")[0]  # Assuming 1 vehicle

    node = CarlaTFPublisher(vehicle)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
