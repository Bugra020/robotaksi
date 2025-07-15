import math
import os
import sys
import threading
import time
from queue import Empty, Queue

import carla
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

DEBUG_MODE = True
LOG_PATH = "carla_log.txt"
SAVE_DIR = "/home/ubuntu/carla_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

def debug(msg):
    if DEBUG_MODE:
        print(msg)


def log_event(car, extra_msg=""):
    transform = car.get_transform()
    velocity = car.get_velocity()
    traffic_light = car.get_traffic_light()
    traffic_state = traffic_light.get_state() if traffic_light else "Unknown"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with open(LOG_PATH, "a") as f:
        f.write(
            f"[{timestamp}] Pos: ({transform.location.x:.2f}, {transform.location.y:.2f}), "
            f"Speed: {math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2):.2f} m/s, "
            f"Yaw: {transform.rotation.yaw:.2f}, "
            f"Traffic Light: {traffic_state}, {extra_msg}\n"
        )


class CarlaConnector:
    def __init__(self, host: str = "51.20.6.84", port: int = 2000):
        self.client = None
        self.world = None
        self.blueprint_lib = None
        self.spectator = None
        self.host = host
        self.port = port
        self.connect()

    def connect(self):
        while True:
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(30.0)
                self.world = self.client.load_world("Town01")
                self.blueprint_lib = self.world.get_blueprint_library()
                self.spectator = self.world.get_spectator()
                debug("Connected to CARLA simulator.")
                break
            except RuntimeError as e:
                debug(f"CARLA not ready yet: {e}")
                time.sleep(2)


class CameraPublisher(Node):
    def __init__(self, connector: CarlaConnector):
        super().__init__("camera_publisher")

        self.frame_index = 0  # inside CameraPublisher __init__ after creating SAVE_DIR
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
        self.frame_to_show = None
        self.frame_lock = threading.Lock()

        while self.world.get_map() is None:
            debug("Waiting for map to load...")
            time.sleep(1)

        self.spawn_camera_vehicle()

        threading.Thread(target=self.process_frames, daemon=True).start()
        #threading.Thread(target=self.display_loop, daemon=True).start()
        threading.Thread(target=self.log_loop, daemon=True).start()

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
            gnss_sensor.listen(lambda data: None)
            self.gnss_sensor = gnss_sensor

            camera_bp = self.bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("sensor_tick", "0.033")
            camera_bp.set_attribute("image_size_x", "1920")
            camera_bp.set_attribute("image_size_y", "1080")
            camera_bp.set_attribute("fov", "90")
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
            bgr_image = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

            if not self.frame_queue.full():
                self.frame_queue.put(bgr_image)

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
            except Empty:
                continue

            results = self.model(frame)[0]

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

            # Save frame to disk
            filename = os.path.join(SAVE_DIR, f"frame_{self.frame_index:06d}.png")
            cv2.imwrite(filename, frame)
            self.frame_index += 1

    def display_loop(self):
        while True:
            with self.frame_lock:
                if self.frame_to_show is not None:
                    cv2.imshow("Traffic Detection", self.frame_to_show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)

    def log_loop(self):
        while rclpy.ok():
            if hasattr(self, "car") and self.car is not None:
                log_event(self.car)
            time.sleep(0.5)


def main():
    rclpy.init()
    connector = CarlaConnector()
    node = CameraPublisher(connector)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
