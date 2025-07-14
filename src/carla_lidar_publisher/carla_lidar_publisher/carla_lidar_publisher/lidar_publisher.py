import math
import os
import sys
import threading
import time
from queue import Queue

import carla
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from ultralytics import YOLO

DEBUG_MODE = True


def debug(msg):
    if DEBUG_MODE:
        print(msg)


class CarlaConnector:
    def __init__(self, host: str = "localhost", port: int = 2000):
        self.client = None
        self.world = None
        self.blueprint_lib = None
        self.spectator = None
        self.host = host
        self.port = port
        self.connect()

    def connect(self):
        connected = False
        while not connected:
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(30.0)
                #self.world = self.client.load_world("Town01")
                self.world = self.client.get_world()
                self.blueprint_lib = self.world.get_blueprint_library()
                self.spectator = self.world.get_spectator()
                connected = True
                debug("Connected to CARLA simulator.")
            except RuntimeError as e:
                debug(f"CARLA not ready yet: {e}")
                time.sleep(2)


class LidarPublisher(Node):
    def __init__(self, connector: CarlaConnector):
        super().__init__("lidar_publisher")
        self.lidar_publisher = self.create_publisher(PointCloud2, "lidar_topic", 10)
        self.connector = connector
        self.world = connector.world
        self.spectator = connector.spectator
        self.bp_lib = connector.blueprint_lib
        self.car = None
        self.lidar_sensor = None

        while self.world.get_map() is None:
            debug("Waiting for map to load...")
            time.sleep(1)
        self.spawn_lidar_vehicle()

    def spawn_lidar_vehicle(self):
        lidar_bp = self.bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("points_per_second", "100000")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar_bp.set_attribute("range", "50")

        try:
            self.car = self.world.spawn_actor(
                self.bp_lib.filter("model3")[0],
                self.world.get_map().get_spawn_points()[1],
            )

            lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5))
            self.lidar_sensor = self.world.spawn_actor(
                lidar_bp, lidar_transform, attach_to=self.car
            )

            self.world.tick()
            time.sleep(1)
            self.world.tick()

            self.lidar_sensor.listen(self.lidar_callback)
            self.car.set_autopilot(True)
            tm = self.connector.client.get_trafficmanager()
            tm.set_global_distance_to_leading_vehicle(2.0)
            tm.vehicle_percentage_speed_difference(self.car, 90.0)

            while rclpy.ok():
                self.world.wait_for_tick(15.0)
                self.update_spectator_view()

        except RuntimeError as e:
            print(f"CARLA error: {e}")
        finally:
            if self.car:
                self.car.destroy()
            if self.lidar_sensor:
                self.lidar_sensor.destroy()
            if hasattr(self, "gnss_sensor"):
                self.gnss_sensor.destroy()

    def lidar_callback(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
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
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.data = points.astype(np.float32).tobytes()
        self.lidar_publisher.publish(msg)

    def update_spectator_view(self):
        if self.car is None or self.spectator is None:
            return
        transform = self.car.get_transform()
        location = transform.location
        rotation = transform.rotation
        yaw_rad = math.radians(rotation.yaw)
        offset_x = -10 * math.cos(yaw_rad)
        offset_y = -10 * math.sin(yaw_rad)
        camera_location = carla.Location(
            x=location.x + offset_x,
            y=location.y + offset_y,
            z=location.z + 20,
        )
        camera_rotation = carla.Rotation(pitch=-30, yaw=rotation.yaw, roll=0)
        self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))


class CameraPublisher(Node):
    def __init__(self, connector: CarlaConnector):
        super().__init__("camera_publisher")
        self.camera_publisher = self.create_publisher(Image, "camera_topic", 10)
        self.connector = connector
        self.world = connector.world
        self.bp_lib = connector.blueprint_lib

        script_exec_path = sys.argv[0]
        install_base_dir = os.path.abspath(
            os.path.join(os.path.dirname(script_exec_path), "..", "..")
        )
        self.model_path = os.path.join(
            install_base_dir,
            "share",
            "carla_lidar_publisher",
            "models",
            "VoltarisSim.pt",
        )
        self.model = YOLO(self.model_path)
        self.frame_queue = Queue(maxsize=2)

        while self.world.get_map() is None:
            debug("Waiting for map to load...")
            time.sleep(1)

        self.spawn_camera_vehicle()
        threading.Thread(target=self.process_frames, daemon=True).start()

    def spawn_camera_vehicle(self):
        try:
            self.car = self.world.spawn_actor(
                self.bp_lib.filter("model3")[0],
                self.world.get_map().get_spawn_points()[1],
            )

            gnss_bp = self.bp_lib.find("sensor.other.gnss")
            gnss_sensor = self.world.spawn_actor(
                gnss_bp, carla.Transform(), attach_to=self.car
            )

            def gnss_callback(data):
                print(f"Origin GNSS: lat={data.latitude}, lon={data.longitude}")

            gnss_sensor.listen(gnss_callback)
            self.gnss_sensor = gnss_sensor

            camera_bp = self.bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("sensor_tick", "0.1")
            camera_bp.set_attribute("image_size_x", "480")
            camera_bp.set_attribute("image_size_y", "360")
            camera_bp.set_attribute("fov", "90")
            # camera_bp.set_attribute("sensor_tick", "0.05")
            camera_transform = carla.Transform(carla.Location(x=1.6, y=0, z=1.7))

            self.camera_sensor = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.car
            )
            self.camera_sensor.listen(self.camera_callback)
            self.car.set_autopilot(True)
            tm = self.connector.client.get_trafficmanager()
            tm.set_global_distance_to_leading_vehicle(2.0)
            tm.vehicle_percentage_speed_difference(self.car, 70.0)

        except RuntimeError as e:
            print(f"CARLA error (camera): {e}")

    def camera_callback(self, image_data):
        try:
            img_array = np.frombuffer(image_data.raw_data, dtype=np.uint8).reshape(
                (image_data.height, image_data.width, 4)
            )
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)

            if not self.frame_queue.full():
                self.frame_queue.put(rgb_image)

            img_msg = Image()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "rgb_front_camera_frame"
            img_msg.height = image_data.height
            img_msg.width = image_data.width
            img_msg.encoding = "bgra8"
            img_msg.is_bigendian = False
            img_msg.step = image_data.width * 4
            img_msg.data = image_data.raw_data
            self.camera_publisher.publish(img_msg)

        except Exception as e:
            print(f"Camera callback error: {e}")

    def process_frames(self):
        while True:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # results = self.model(frame)[0]
            results = self.model(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls_id = int(box.cls[0])
                label = f"{self.model.names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            cv2.imshow("Traffic Detection", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


def main():
    rclpy.init()
    connector = CarlaConnector()
    node_type = "camera"
    if node_type == "camera":
        node = CameraPublisher(connector)
    else:
        node = LidarPublisher(connector)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
