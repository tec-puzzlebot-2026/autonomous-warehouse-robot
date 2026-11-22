#!/usr/bin/env python3
"""
box_target_node.py - ROS 2 (Humble) node: camera image -> near_box.py -> the ONE box to manipulate.

Runtime copy of training_m2/scripts/box_target_node.py (installed as `box_target`), with
two additions for the Program Manager: an on/off switch (~/enable, std_msgs/Bool, and the
`enabled` parameter), so only one network runs at a time on the Jetson, and the box's
outline (`polygon`, image pixels) in the JSON, used to accept only the QR code that lies
inside the target box.

    ros2 run main_ros box_target --ros-args -p model:=/home/puzzlebot/models/best.engine
    ros2 topic echo /box_target/target

The same executable also runs as the FAR detector (node name box_far, the full-frame
model boxes_seg_v1 and settings with the whole frame as ROI and a small min size):
it then publishes the nearest box seen anywhere in the image on /box_far/target.

Or without installing:  python3 box_target_node.py --ros-args -p model:=...

Parameters
    model         .engine / .onnx / .pt  (near_box.yaml is read from the same folder)
    settings      optional path to another near_box.yaml
    enabled       start processing images (default true); ~/enable switches it at runtime
    max_rate_hz   process at most this many frames per second (0 = every frame)
    backend       auto | tensorrt | onnxruntime | ultralytics
    image_topic   default /camera/image_raw - use the topic your TRAINING bags were recorded
                  from (raw vs colour-corrected look different to the network)
Publishes
    /box_target/target  std_msgs/String, JSON:
        {"found": true, "name": "box_drinks", "cls": 1, "conf": 0.91,
         "box": [x1, y1, x2, y2], "center": [u, v], "size_px": 212.0,
         "offset_x": -0.12,               # (u - W/2) / (W/2): <0 box left of centre, steer left
         "rect": {"center": [..], "size": [w, h], "angle_deg": a},   # outline fit (seg models)
         "touches": {"bottom": true, ...},  # box cut by the frame/ROI edge
         "stamp": 12.34, "ms": 31.2, "n_det": 3,
         "all": [[x1, y1, x2, y2, conf], ...]}  # every box above min_size_px, largest first
                                                 # (the far detector: the state machine skips
                                                 # boxes it already handled and takes the next)
        {"found": false, ...} when nothing near is seen.
    /box_target/debug   sensor_msgs/Image (bgr8), ONLY computed while someone subscribes
Images are processed newest-first with a depth-1 best-effort queue: if inference is slower
than the camera, old frames are dropped instead of piling up latency.
"""
import json
import os
import sys
import time

import cv2

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

try:
    from .near_box import NearBoxDetector, draw
except ImportError:                                    # run as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from near_box import NearBoxDetector, draw


def image_to_bgr(msg):
    """sensor_msgs/Image -> HxWx3 BGR uint8 (no cv_bridge needed)."""
    ch = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}.get(msg.encoding)
    if ch is None:
        raise ValueError("unsupported encoding " + msg.encoding)
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch)
    if msg.encoding == "rgb8":
        return a[:, :, ::-1]
    if msg.encoding == "bgra8":
        return a[:, :, :3]
    if msg.encoding == "rgba8":
        return a[:, :, 2::-1]
    if msg.encoding == "mono8":
        return np.repeat(a, 3, axis=2)
    return a


def bgr_to_image(img, header):
    m = Image()
    m.header = header
    m.height, m.width = img.shape[:2]
    m.encoding, m.is_bigendian, m.step = "bgr8", 0, img.shape[1] * 3
    m.data = np.ascontiguousarray(img).tobytes()
    return m


class BoxTargetNode(Node):
    def __init__(self):
        super().__init__("box_target")
        self.enabled = bool(self.declare_parameter("enabled", True).value)
        rate = float(self.declare_parameter("max_rate_hz", 0.0).value)
        self.min_period = 1.0 / rate if rate > 0 else 0.0
        self.last_t = 0.0
        model = self.declare_parameter("model", "").value
        settings = self.declare_parameter("settings", "").value or None
        backend = self.declare_parameter("backend", "auto").value
        topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        if not model:
            raise SystemExit("set the model parameter:  --ros-args -p model:=/path/best.engine")
        self.det = NearBoxDetector(os.path.expanduser(model), settings, backend)
        self.get_logger().info("backend %s | roi %s -> net %s | min size %s px | topic %s" % (
            self.det.be.name, self.det.s["roi"], self.det.s["net"], self.det.s["min_size_px"], topic))

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(Image, topic, self.on_image, qos)
        self.pub = self.create_publisher(String, "~/target", 10)
        self.pub_dbg = self.create_publisher(Image, "~/debug", 1)
        self.create_subscription(Bool, "~/enable", self.on_enable, 10)
        self.n, self.t_sum, self.t_log = 0, 0.0, time.time()

    def on_enable(self, msg):
        if bool(msg.data) != self.enabled:
            self.enabled = bool(msg.data)
            self.det.prev = None                    # forget the last target when switched
            self.get_logger().info("enabled" if self.enabled else "disabled (no inference)")
            if not self.enabled:
                self.pub.publish(String(data=json.dumps({"found": False, "disabled": True})))

    def on_image(self, msg):
        if not self.enabled:
            return
        now = time.monotonic()
        if self.min_period and now - self.last_t < self.min_period:
            return
        self.last_t = now
        try:
            frame = image_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        t0 = time.perf_counter()
        target = self.det.detect(frame)
        ms = 1e3 * (time.perf_counter() - t0)

        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        out = {"found": target is not None, "stamp": stamp, "ms": round(ms, 1), "n_det": self.det.n_det,
               "all": self.det.all_boxes}
        if target is not None:
            W = frame.shape[1]
            out.update({k: v for k, v in target.items() if k != "polygon"})
            poly = target.get("polygon")
            if poly is not None and len(poly) >= 3:
                poly = cv2.approxPolyDP(np.asarray(poly, np.float32).reshape(-1, 1, 2), 2.0, True)
                out["polygon"] = [[round(float(u), 1), round(float(v), 1)]
                                  for u, v in poly.reshape(-1, 2)[:60]]
            out["frame"] = [int(frame.shape[1]), int(frame.shape[0])]
            out["offset_x"] = (target["center"][0] - W / 2.0) / (W / 2.0)
        self.pub.publish(String(data=json.dumps(out, default=float)))

        if self.pub_dbg.get_subscription_count() > 0:      # drawing costs CPU: only when watched
            self.pub_dbg.publish(bgr_to_image(draw(frame, target, self.det.s), msg.header))

        self.n += 1
        self.t_sum += ms
        if time.time() - self.t_log > 10.0:
            self.get_logger().info("%.1f frames/s processed, detector %.1f ms/frame" % (
                self.n / (time.time() - self.t_log), self.t_sum / max(self.n, 1)))
            self.n, self.t_sum, self.t_log = 0, 0.0, time.time()


def main(args=None):
    rclpy.init(args=args)
    node = BoxTargetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()