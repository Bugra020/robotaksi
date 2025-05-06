import carla
import geometry_msgs.msg
import numpy as np
import rclpy
import tf2_ros
import math
from rclpy.node import Node


class DynamicTFPublisher(Node):
    def __init__(self, vehicle):
        super().__init__("dynamic_tf_publisher")
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

        roll = math.radians(transform.rotation.roll)
        pitch = math.radians(transform.rotation.pitch)
        yaw = math.radians(transform.rotation.yaw)

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy

        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf_msg)


def main():
    rclpy.init()

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.get_world()
    vehicle = world.get_actors().filter("vehicle.*")[0]

    node = DynamicTFPublisher(vehicle)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
