#!/usr/bin/env python3
"""
qr_reader_node.py — read the QR codes on the boxes (OpenCV, no network)

Each box carries a QR code on one of its faces, about half the size of the face. The
code is one of the printed texts in config/qr_companies.yaml (e.g. "EMEZON-03");
this node reports its company (soriene, emezon or walnut). This node decodes every QR code
in the image and reports its text and its four corners, so the restocking state
machine can keep only the code that lies inside the target box's outline (and
never mix up the data of a box standing next to it).

Subscribes  image_topic (default /camera/image_raw)
            ~/enable (std_msgs/Bool): switch decoding on/off
Publishes   ~/codes (std_msgs/String, JSON):
              {"stamp": 12.3, "frame": [640, 480],
               "codes": [{"text": "walnut", "raw": "WALNUT-02", "corners": [[u, v] x 4]}],
               "located": [[[u, v] x 4], ...]}
            located: every QR pattern found, decoded or not (at a slant a code is often
            found but not read): it shows which face of a box is the front.
            An empty "codes" list when nothing was decoded (the state machine then
            knows the image was looked at).

Parameters  enabled (default false: only on while a box is inspected), max_rate_hz,
            upscale (decode on an enlarged image when codes are small, default 1.0),
            known_texts (only these companies are reported; [] = every company),
            companies_file ('' = config/qr_companies.yaml of the installed package;
            'none' = report the raw QR text, no table).
"""
# Author: Raymundo Gomez

import json
import time

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from main_ros.vision.boxes.box_target_node import image_to_bgr
from main_ros.vision.qr.decode import decode_qr, load_companies, locate_qr


class QrReaderNode(Node):
    def __init__(self):
        super().__init__('qr_reader')
        p = self.declare_parameter
        self.enabled = bool(p('enabled', False).value)
        rate = float(p('max_rate_hz', 5.0).value)
        self.min_period = 1.0 / rate if rate > 0 else 0.0
        self.upscale = float(p('upscale', 1.0).value)
        self.known = tuple(p('known_texts', ['soriene', 'emezon', 'walnut']).value)
        topic = p('image_topic', '/camera/image_raw').value
        table = p('companies_file', '').value
        self.companies = None
        if table != 'none':
            if not table:
                from ament_index_python.packages import get_package_share_directory
                table = f"{get_package_share_directory('main_ros')}/config/qr_companies.yaml"
            self.companies = load_companies(table)
        self.detector = cv2.QRCodeDetector()
        self.last_t = 0.0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, topic, self.on_image, qos)
        self.create_subscription(Bool, '~/enable', self.on_enable, 10)
        self.pub = self.create_publisher(String, '~/codes', 10)
        self.get_logger().info(
            f'QR reader on {topic} '
            f'({len(self.companies) if self.companies is not None else "no"} codes known; '
            f'{"enabled" if self.enabled else "disabled until enabled"})')

    def on_enable(self, msg):
        if bool(msg.data) != self.enabled:
            self.enabled = bool(msg.data)
            self.get_logger().info('enabled' if self.enabled else 'disabled')

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
        codes = decode_qr(frame, self.detector, self.upscale, self.known, self.companies)
        located = [c['corners'] for c in codes] or locate_qr(frame, self.detector)
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.pub.publish(String(data=json.dumps({
            'stamp': stamp, 'frame': [int(frame.shape[1]), int(frame.shape[0])], 'codes': codes,
            'located': located})))


def main(args=None):
    rclpy.init(args=args)
    node = QrReaderNode()
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
