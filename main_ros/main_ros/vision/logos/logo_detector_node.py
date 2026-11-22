#!/usr/bin/env python3
"""
logo_detector_node.py — company logos in the camera image (YOLO11n detect model)

Finds the company logos (soriene, emezon, walnut) that mark the docks. Used by the
landmark mapper while mapping (to place each logo on the map) and by the
restocking state machine to confirm the company at a dock before entering.

Subscribes  image_topic (default /camera/image_rect: the color-corrected image, the
            one the logo model was trained on)
            ~/enable (std_msgs/Bool): switch inference on/off (one network at a time)
Publishes   ~/detections (std_msgs/String, JSON):
              {"stamp": 12.3, "frame": [640, 480], "ms": 41.0,
               "detections": [{"company": "walnut", "cls": 2, "conf": 0.88,
                               "box": [x1, y1, x2, y2], "center": [u, v]}, ...]}

Parameters
    model        .engine / .onnx / .pt (TensorRT engine on the Jetson)
    companies    class index -> company name; overrides the names stored in the model,
                 so a renamed company needs no retraining (default soriene, emezon, walnut)
    backend      auto | tensorrt | onnxruntime | ultralytics
    net          network input [w, h] (default 640 x 640, as exported)
    conf, nms_iou, max_det, enabled, max_rate_hz

The pre/post-processing and the backends are shared with the box detector
(vision/boxes/near_box.py), so both run the same way on the Jetson.
"""
# Author: Raymundo Gomez

import json
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from main_ros.vision.boxes import near_box as nb
from main_ros.vision.boxes.box_target_node import image_to_bgr


class LogoDetector:
    """detect(frame) -> list of detections (all logos, not only one)."""

    def __init__(self, model, companies, backend='auto', net=(640, 640), conf=0.5,
                 nms_iou=0.5, max_det=10):
        self.companies = list(companies)
        self.s = {'classes': self.companies, 'frame': [640, 480], 'roi': [0, 0, 640, 480],
                  'net': list(net), 'conf': conf, 'nms_iou': nms_iou, 'max_det': max_det}
        self.be = nb.make_backend(model, backend)

    def _name(self, cls):
        return self.companies[cls] if 0 <= cls < len(self.companies) else str(cls)

    def detect(self, frame):
        H, W = frame.shape[:2]
        self.s['frame'], self.s['roi'] = [W, H], [0, 0, W, H]
        if isinstance(self.be, nb.UltralyticsBackend):
            nw, nh = self.s['net']
            r = self.be.model.predict(frame, imgsz=(nh, nw), conf=self.s['conf'],
                                      iou=self.s['nms_iou'], max_det=int(self.s['max_det']),
                                      verbose=False)[0]
            dets = [{'cls': int(c), 'conf': float(cf), 'box': b.tolist()}
                    for c, cf, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                        r.boxes.xyxy.cpu().numpy())]
        else:
            blob, meta = nb.preprocess(frame, self.s, self.be.fp16)
            outs = self.be(blob)
            dets = nb.decode(outs[0], len(self.companies), self.s, meta)
        out = []
        for d in dets:
            x1, y1, x2, y2 = [float(v) for v in d['box']]
            out.append({'company': self._name(d['cls']), 'cls': d['cls'],
                        'conf': round(float(d['conf']), 3), 'box': [x1, y1, x2, y2],
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]})
        return out


class LogoDetectorNode(Node):
    def __init__(self):
        super().__init__('logo_detector')
        p = self.declare_parameter
        model = os.path.expanduser(p('model', '').value)
        if not model:
            raise SystemExit('set the model parameter:  --ros-args -p model:=/path/logos.engine')
        companies = list(p('companies', ['soriene', 'emezon', 'walnut']).value)
        net = [int(v) for v in p('net', [640, 640]).value]
        self.det = LogoDetector(model, companies, p('backend', 'auto').value, net,
                                float(p('conf', 0.5).value), float(p('nms_iou', 0.5).value),
                                int(p('max_det', 10).value))
        self.enabled = bool(p('enabled', True).value)
        rate = float(p('max_rate_hz', 5.0).value)
        self.min_period = 1.0 / rate if rate > 0 else 0.0
        self.last_t = 0.0
        topic = p('image_topic', '/camera/image_rect').value
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, topic, self.on_image, qos)
        self.create_subscription(Bool, '~/enable', self.on_enable, 10)
        self.pub = self.create_publisher(String, '~/detections', 10)
        self.get_logger().info(f'logo detector: {self.det.be.name} | companies {companies} | '
                               f'{topic} | {"enabled" if self.enabled else "disabled"}')

    def on_enable(self, msg):
        if bool(msg.data) != self.enabled:
            self.enabled = bool(msg.data)
            self.get_logger().info('enabled' if self.enabled else 'disabled (no inference)')

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
        dets = self.det.detect(frame)
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.pub.publish(String(data=json.dumps({
            'stamp': stamp, 'frame': [int(frame.shape[1]), int(frame.shape[0])],
            'ms': round(1e3 * (time.perf_counter() - t0), 1), 'detections': dets})))


def main(args=None):
    rclpy.init(args=args)
    node = LogoDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
