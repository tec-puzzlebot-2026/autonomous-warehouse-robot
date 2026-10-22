"""
yolo_webcam_test.py - TEST the logo detector on your laptop with the laptop webcam.

    webcam -> YOLO (GPU if available) -> boxes drawn -> window + ROS 2 topics

Run:
    ros2 launch main_ros yolo_webcam_test.launch.py
    ros2 launch main_ros yolo_webcam_test.launch.py camera_index:=2 conf:=0.4

Published topics:
    /logo_detections/image             sensor_msgs/Image           annotated video (raw)
    /logo_detections/image/compressed  sensor_msgs/CompressedImage annotated video (JPEG)
    /logo_detections                   std_msgs/String             JSON list of detections

View the video in another terminal:  ros2 run rqt_image_view rqt_image_view
See the detections:                  ros2 topic echo /logo_detections

To test another model: copy it into vision/models/ and change MODEL_NAME below.
"""
import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

# =============================================================================
# >>> CHANGE THESE TO TEST ANOTHER MODEL / SETTINGS <<<
# =============================================================================
MODEL_NAME = "logos_yolo11n.pt"          # file inside vision/models/  (.pt or .onnx)
CONF_THRESHOLD = 0.5                     # minimum confidence to show a detection (0-1)
IMG_SIZE = 640                           # same size used in training

# Colour calibration of the ROBOT camera. Keep False for the laptop webcam:
# the gain map belongs to the robot camera, applying it to another camera
# would make the colours worse, not better.
APPLY_COLOR_CORRECTION = False
COLOR_CALIBRATION_FILE = "colorCalibration.npz"   # inside vision/training/calibration/
# =============================================================================

VISION_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))   # testing/ -> vision/
MODELS_DIR = os.path.join(VISION_DIR, "models")
CALIB_DIR = os.path.join(VISION_DIR, "training", "calibration")


def load_gain_map(path):
    """Same logic as training/scripts/color_correct.py: per-pixel B,G,R gains."""
    data = np.load(path)
    keys = [k for k in data.files if data[k].ndim == 3 and 3 in data[k].shape]
    keys += [k for k in data.files if data[k].shape == (3,)]
    g = data[keys[0]].astype(np.float32)
    if g.ndim == 3 and g.shape[0] == 3 and g.shape[-1] != 3:
        g = np.transpose(g, (1, 2, 0))
    return g


class YoloWebcamTest(Node):
    def __init__(self):
        super().__init__("yolo_webcam_test")
        self.declare_parameter("model_name", "")        # "" = use MODEL_NAME above
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("conf", CONF_THRESHOLD)
        self.declare_parameter("show_window", True)
        self.declare_parameter("publish_raw", True)

        model_name = self.get_parameter("model_name").value or MODEL_NAME
        self.conf = float(self.get_parameter("conf").value)
        self.show = bool(self.get_parameter("show_window").value)
        self.publish_raw = bool(self.get_parameter("publish_raw").value)
        cam_index = int(self.get_parameter("camera_index").value)

        # ---- model ----------------------------------------------------------
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.isfile(model_path):
            available = sorted(os.listdir(MODELS_DIR)) if os.path.isdir(MODELS_DIR) else []
            raise SystemExit(f"Model not found: {model_path}\nAvailable in {MODELS_DIR}: {available}")
        from ultralytics import YOLO          # imported here so errors above are clear
        import torch
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path, task="detect")
        self.get_logger().info(
            f"Model: {model_name} | classes: {self.model.names} | device: "
            f"{torch.cuda.get_device_name(0) if self.device == 0 else 'CPU'}")

        # ---- colour correction (robot camera only) ----------------------------
        self.gain = None
        if APPLY_COLOR_CORRECTION:
            self.gain = load_gain_map(os.path.join(CALIB_DIR, COLOR_CALIBRATION_FILE))
            self.get_logger().info("Colour correction ON")

        # ---- camera ------------------------------------------------------------
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera index {cam_index}. "
                             "List cameras with: ls /dev/video*")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ---- publishers ----------------------------------------------------------
        self.pub_raw = self.create_publisher(Image, "logo_detections/image", 1)
        self.pub_jpg = self.create_publisher(CompressedImage, "logo_detections/image/compressed", 1)
        self.pub_det = self.create_publisher(String, "logo_detections", 10)

        self.fps = 0.0
        self.last_log = 0.0
        self.timer = self.create_timer(0.0, self.step)     # run as fast as the camera allows
        self.get_logger().info("Running. Press 'q' in the video window (or Ctrl+C) to stop.")

    def step(self):
        t0 = time.time()
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Camera frame not received", throttle_duration_sec=2.0)
            return

        if self.gain is not None:
            g = self.gain
            if g.ndim == 3 and g.shape[:2] != frame.shape[:2]:
                g = cv2.resize(g, (frame.shape[1], frame.shape[0]))
            frame = np.clip(frame.astype(np.float32) * g, 0, 255).astype(np.uint8)

        # ---- inference ----------------------------------------------------------
        res = self.model.predict(frame, imgsz=IMG_SIZE, conf=self.conf,
                                 device=self.device, verbose=False)[0]
        annotated = res.plot()                       # boxes + names + confidence drawn

        dets = []
        for box in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            dets.append({"logo": self.model.names[int(box.cls)],
                         "conf": round(float(box.conf), 3),
                         "box": [x1, y1, x2, y2]})

        # ---- FPS overlay ----------------------------------------------------------
        dt = time.time() - t0
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-6))
        cv2.putText(annotated, f"{self.fps:.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ---- publish ----------------------------------------------------------------
        stamp = self.get_clock().now().to_msg()
        if self.publish_raw:
            img = Image()
            img.header.stamp, img.header.frame_id = stamp, "webcam"
            img.height, img.width = annotated.shape[:2]
            img.encoding, img.step = "bgr8", annotated.shape[1] * 3
            img.data = annotated.tobytes()
            self.pub_raw.publish(img)
        jpg = CompressedImage()
        jpg.header.stamp, jpg.header.frame_id = stamp, "webcam"
        jpg.format = "jpeg"
        jpg.data = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self.pub_jpg.publish(jpg)
        self.pub_det.publish(String(data=json.dumps(dets)))

        # ---- terminal log (once per second) --------------------------------------------
        if time.time() - self.last_log > 1.0:
            self.last_log = time.time()
            found = ", ".join(f"{d['logo']} {d['conf']:.2f}" for d in dets) or "none"
            self.get_logger().info(f"{self.fps:5.1f} FPS | detections: {found}")

        # ---- local window ----------------------------------------------------------------
        if self.show:
            cv2.imshow("YOLO logo test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloWebcamTest()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit) as e:
        if isinstance(e, SystemExit) and e.code not in (None, 0):
            print(e.code)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()