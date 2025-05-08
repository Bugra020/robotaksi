import math
import time

import carla
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

DEBUG_MODE = True


def debug(debug_msg):
    if DEBUG_MODE:
        print(debug_msg)


class LidarPublisher(Node):
    def __init__(self):
        super().__init__("lidar_publisher")
        self.publisher = self.create_publisher(PointCloud2, "lidar_topic", 10)
        debug("publisher created")

        self.client = None
        self.world = None
        self.blueprint_lib = None
        self.spectator = None
        self.connect_carla()
        debug("compleated connecting to carla")

        self.car = None
        self.lidar_sensor = None
        self.spawn_car_with_lidar()

    def connect_carla(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(15.0)
        debug("connected to carla")
        self.world = self.client.get_world()
        debug("got the world from carla")
        self.blueprint_lib = self.world.get_blueprint_library()
        debug("got the blueprint lib from carla")
        self.spectator = self.world.get_spectator()

    def spawn_car_with_lidar(self):
        debug("spawing car with lidar")
        lidar_bp = self.blueprint_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("points_per_second", "100000")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar_bp.set_attribute("range", "50")

        try:
            self.car = self.world.spawn_actor(
                self.blueprint_lib.filter("model3")[0],
                self.world.get_map().get_spawn_points()[1],
            )
            debug("car spawned")

            lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5))
            self.lidar_sensor = self.world.spawn_actor(
                lidar_bp, lidar_transform, attach_to=self.car
            )
            debug("lidar spawned on top of the car")

            self.lidar_sensor.listen(self.lidar_callback)

            self.car.set_autopilot(False)
            debug("car with lidar spawned successfully")

            while True:
                self.world.tick()
                self.update_spectator_view()
                # time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nkeyboardinterrupt killing the program")
        finally:
            if self.car is not None:
                self.car.destroy()
                print("destroyed the car")
            if self.lidar_sensor is not None:
                self.lidar_sensor.destroy()
                print("destroyed the lidar sensor")

    def update_spectator_view(self):
        if self.car is None or self.spectator is None:
            return

        transform = self.car.get_transform()
        location = transform.location
        rotation = transform.rotation


        yaw_rad = math.radians(rotation.yaw)

        distance_behind = 10
        height_above = 20

        offset_x = -distance_behind * math.cos(yaw_rad)
        offset_y = -distance_behind * math.sin(yaw_rad)

        camera_location = carla.Location(
            x=location.x + offset_x,
            y=location.y + offset_y,
            z=location.z + height_above,
        )

        camera_rotation = carla.Rotation(pitch=-30, yaw=rotation.yaw, roll=0)

        self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))

    def lidar_callback(self, data):
        # np.frombuffer turns the raw data to a float32 array and reshaping makes it nx4 matrix. x,y,z and intensity
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)

        """ 
        std_msgs/Header header
            uint32 height
            uint32 width
        sensor_msgs/PointField[] fields
            bool is_bigendian
            uint32 point_step
            uint32 row_step
            uint8[] data
            bool is_dense
        """

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar_frame"

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
        ]

        msg.point_step = 16  # 4 fields * 4 bytes each = 16 bytes total
        msg.row_step = msg.point_step * len(points)  # total size of the msg
        msg.height = 1
        msg.width = len(points)

        msg.is_dense = True  # true for ignoring nan and inf nums
        msg.is_bigendian = False
        msg.data = points.astype(
            np.float32
        ).tobytes()  # converting numpy array to raw byte array

        self.publisher.publish(msg)
        # debug(f"published {msg.row_step} bytes of lidar data!")


def main():
    rclpy.init()
    node = LidarPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
