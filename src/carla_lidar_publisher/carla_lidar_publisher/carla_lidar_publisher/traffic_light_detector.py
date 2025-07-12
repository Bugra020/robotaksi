import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import sys

import torch
from ultralytics import YOLO

from std_msgs.msg import Bool

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector_node')
        self.bridge = CvBridge()

        self.declare_parameter('camera_image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('target_class_id_red_light', 7) # Changed to 7
        self.declare_parameter('annotate_image', True)
        self.declare_parameter('display_window', True)

        self.camera_image_topic = self.get_parameter('camera_image_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.target_class_id_red_light = self.get_parameter('target_class_id_red_light').value
        self.annotate_image = self.get_parameter('annotate_image').value
        self.display_window = self.get_parameter('display_window').value

        script_exec_path = sys.argv[0] 
        install_base_dir = os.path.abspath(os.path.join(os.path.dirname(script_exec_path), '..', '..'))
        
        package_name = 'carla_lidar_publisher'
        self.model_filename = 'VoltarisSim.pt'
        self.model_path = os.path.join(install_base_dir, 'share', package_name, 'models', self.model_filename)

        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"Successfully loaded PyTorch model from {self.model_path}")
            
            self.class_names = self.model.names
            self.get_logger().info(f"Model class names: {self.class_names}")

            if self.target_class_id_red_light == -1:
                self.get_logger().warn("target_class_id_red_light is not set. Please set this parameter correctly based on your model's labels.")
                self.get_logger().warn("Example: If 'red light' is the 2nd class in your labels, set target_class_id_red_light to 1 (0-indexed).")
            elif self.target_class_id_red_light >= len(self.class_names) or self.target_class_id_red_light < 0:
                 self.get_logger().warn(f"Target red light class ID {self.target_class_id_red_light} is out of bounds for model's {len(self.class_names)} classes.")
            else:
                self.get_logger().info(f"Targeting class '{self.class_names[self.target_class_id_red_light]}' for red light detection.")

        except Exception as e:
            self.get_logger().error(f"Failed to load PyTorch model from {self.model_path}: {e}")
            rclpy.shutdown()
            return

        self.subscription = self.create_subscription(
            Image,
            self.camera_image_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f"Subscribed to {self.camera_image_topic}")

        self.image_publisher = self.create_publisher(Image, '/traffic_light/image_with_detections', 10)
        self.get_logger().info(f"Publishing annotated images to /traffic_light/image_with_detections")
        
        self.red_light_publisher = self.create_publisher(Bool, '/traffic_light/is_red', 10)
        self.get_logger().info(f"Publishing red light status to /traffic_light/is_red")

        if self.display_window:
            cv2.namedWindow("Annotated Camera Feed", cv2.WINDOW_AUTOSIZE)
            self.get_logger().info("OpenCV display window created.")


    def image_callback(self, msg: Image):
        try:
            cv_image_bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
            cv_image = cv2.cvtColor(cv_image_bgra, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        is_red_light_detected = False
        
        annotated_image = cv_image.copy()

        try:
            results = self.model(cv_image, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)

            detections_found = False
            for r in results:
                if len(r.boxes) > 0:
                    detections_found = True
                for box in r.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    xyxy = box.xyxy[0].tolist()

                    if cls_id == self.target_class_id_red_light:
                        is_red_light_detected = True
                        self.get_logger().info(f"Red light detected! Conf: {conf:.2f}, Box: {xyxy}")

                    if self.annotate_image and annotated_image is not None:
                        x1, y1, x2, y2 = map(int, xyxy)
                        label = self.class_names[cls_id] if self.class_names else f"Class {cls_id}"
                        color = (0, 255, 0)
                        if cls_id == self.target_class_id_red_light:
                            color = (0, 0, 255)

                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if self.annotate_image and not detections_found:
                self.get_logger().debug("No objects detected in this frame.")


        except Exception as e:
            self.get_logger().error(f"Error during model inference or processing: {e}")
            is_red_light_detected = False 
            if annotated_image is None:
                 annotated_image = cv_image


        red_light_msg = Bool()
        red_light_msg.data = is_red_light_detected
        self.red_light_publisher.publish(red_light_msg)

        if annotated_image is not None:
            try:
                self.image_publisher.publish(self.bridge.cv2_to_imgmsg(annotated_image, "bgr8"))
            except Exception as e:
                self.get_logger().error(f"Failed to convert and publish annotated image: {e}")
        else:
            self.get_logger().debug("No image to publish (annotate_image and display_window are False).")


        if self.display_window and annotated_image is not None:
            cv2.imshow("Annotated Camera Feed", annotated_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Traffic Light Detector node stopped cleanly.')
    finally:
        cv2.destroyAllWindows() 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

