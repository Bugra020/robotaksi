import math

import carla
import geometry_msgs.msg
import rclpy
import tf2_ros
from rclpy.node import Node


class DynamicTFPublisher(Node):
    def __init__(self, vehicle: carla.Vehicle):
        super().__init__("dynamic_tf_publisher")
        self.vehicle = vehicle
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.publish_tf)  # 20 Hz
        self.get_logger().info("DynamicTFPublisher node initialized.")

    def publish_tf(self):
        try:
            transform = self.vehicle.get_transform()

            tf_msg = geometry_msgs.msg.TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = "map"
            tf_msg.child_frame_id = "base_link"

            tf_msg.transform.translation.x = -transform.location.x
            tf_msg.transform.translation.y = -transform.location.y
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
        except Exception as e:
            self.get_logger().error(f"Failed to publish transform: {e}")


def main():
    rclpy.init()

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(5.0)

        world = client.get_world()
        vehicles = world.get_actors().filter("vehicle.*")

        if not vehicles:  
            print("No vehicles found in the CARLA world.")
            return

        vehicle = vehicles[0]
        print(f"Publishing TF for vehicle ID: {vehicle.id}")

        node = DynamicTFPublisher(vehicle)
        rclpy.spin(node)

    except Exception as e:
        print(f"Exception during initialization: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
