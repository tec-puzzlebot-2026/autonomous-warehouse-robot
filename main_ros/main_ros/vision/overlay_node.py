#!/usr/bin/env python3
"""
overlay_node.py — one camera image with everything the robot sees drawn on it
=============================================================================
For Foxglove: the camera image with the latest result of every detector on top, so
one Image panel shows what the robot is looking at and what each model found.

  ArUco          green outline, "ArUco 21 - drop zone 1.42 m"
  company logo   magenta box, "walnut 0.88"
  far boxes      yellow boxes (every box the far detector sees), the chosen one thicker
  near box       cyan outline (segmentation polygon), "box_drinks 0.91"
  QR code        blue outline, its text ("WALNUT-02")

A detection is drawn while it is fresh (max_age_s of wall time); the corner legend
says which detector is running (a detector switched off draws nothing).

Subscribes  image_topic (default /camera/image_raw), /aruco/detections,
            /logo_detector/detections, /box_far/target, /box_target/target,
            /qr_reader/codes
Publishes   /vision/overlay (sensor_msgs/Image, bgr8) - only while someone subscribes

    ros2 run main_ros vision_overlay --ros-args -p image_topic:=/camera/image_raw
"""
# Author: Raymundo Gomez

import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


GREEN, MAGENTA, YELLOW = (60, 220, 60), (220, 60, 220), (0, 220, 255)
CYAN, BLUE, WHITE = (255, 220, 0), (255, 120, 30), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def image_to_bgr(msg):
    """sensor_msgs/Image -> HxWx3 BGR uint8 (no cv_bridge needed)."""
    ch = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, 'mono8': 1}.get(msg.encoding)
    if ch is None:
        raise ValueError('unsupported encoding ' + msg.encoding)
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch)
    if msg.encoding == 'rgb8':
        return a[:, :, ::-1]
    if msg.encoding == 'bgra8':
        return a[:, :, :3]
    if msg.encoding == 'rgba8':
        return a[:, :, 2::-1]
    if msg.encoding == 'mono8':
        return np.repeat(a, 3, axis=2)
    return a


def bgr_to_image(img, header):
    m = Image()
    m.header = header
    m.height, m.width = img.shape[:2]
    m.encoding, m.is_bigendian, m.step = 'bgr8', 0, img.shape[1] * 3
    m.data = np.ascontiguousarray(img).tobytes()
    return m


def label(img, text, org, color):
    """Text with a dark background, kept inside the image."""
    (w, h), base = cv2.getTextSize(text, FONT, 0.45, 1)
    x = int(min(max(org[0], 0), img.shape[1] - w - 2))
    y = int(min(max(org[1], h + 4), img.shape[0] - 2))
    cv2.rectangle(img, (x - 1, y - h - 4), (x + w + 2, y + base - 2), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y - 2), FONT, 0.45, color, 1, cv2.LINE_AA)


def poly(img, pts, color, width=2):
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [p], True, color, width, cv2.LINE_AA)


def rect(img, box, color, width=2):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, width)


class VisionOverlay(Node):
    def __init__(self):
        super().__init__('vision_overlay')
        p = self.declare_parameter
        image_topic = p('image_topic', '/camera/image_raw').value
        self.max_age = float(p('max_age_s', 1.0).value)
        names = p('aruco_names', ['21:drop zone']).value      # "id:name" labels
        self.aruco_names = {}
        for item in names:
            if ':' in str(item):
                k, v = str(item).split(':', 1)
                self.aruco_names[int(k)] = v
        self.last = {}                    # topic key -> (wall time, parsed JSON)
        for key, topic in (('aruco', '/aruco/detections'),
                           ('logos', '/logo_detector/detections'),
                           ('box_far', '/box_far/target'),
                           ('box_target', '/box_target/target'),
                           ('qr', '/qr_reader/codes')):
            self.create_subscription(String, topic, lambda m, k=key: self.on_json(k, m), 10)
        self.pub = self.create_publisher(Image, '/vision/overlay', 2)
        self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
        self.get_logger().info(f'vision overlay: {image_topic} -> /vision/overlay')

    def on_json(self, key, msg):
        try:
            self.last[key] = (time.monotonic(), json.loads(msg.data))
        except ValueError:
            pass

    def fresh(self, key):
        t, data = self.last.get(key, (0.0, None))
        if data is None or time.monotonic() - t > self.max_age:
            return None
        if isinstance(data, dict) and data.get('disabled'):
            return None
        return data

    def on_image(self, msg):
        if self.pub.get_subscription_count() == 0:
            return
        try:
            img = image_to_bgr(msg).copy()
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return
        on = []
        far = self.fresh('box_far')
        if far is not None:
            on.append('far boxes')
            for b in far.get('all', []) or []:
                rect(img, b, YELLOW, 1)
            if far.get('found') and far.get('box'):
                rect(img, far['box'], YELLOW, 2)
                label(img, f"{far.get('name', 'box')} {float(far.get('conf', 0)):.2f}",
                      (far['box'][0], far['box'][1] - 4), YELLOW)
        near = self.fresh('box_target')
        if near is not None:
            on.append('near box')
            if near.get('found'):
                if near.get('polygon'):
                    poly(img, near['polygon'], CYAN, 2)
                elif near.get('box'):
                    rect(img, near['box'], CYAN, 2)
                b = near.get('box') or [0, 20, 0, 0]
                label(img, f"{near.get('name', 'box')} {float(near.get('conf', 0)):.2f}",
                      (b[0], b[3] + 16), CYAN)
        logos = self.fresh('logos')
        if logos is not None:
            on.append('logos')
            for d in logos.get('detections', []):
                rect(img, d['box'], MAGENTA, 2)
                label(img, f"{d.get('company', '?')} {float(d.get('conf', 0)):.2f}",
                      (d['box'][0], d['box'][1] - 4), MAGENTA)
        qr = self.fresh('qr')
        if qr is not None:
            on.append('QR')
            for c in qr.get('codes', []):
                if c.get('corners'):
                    poly(img, c['corners'], BLUE, 2)
                    x, y = np.asarray(c['corners'], dtype=float).min(axis=0)
                    label(img, str(c.get('raw') or c.get('text')), (x, y - 4), BLUE)
        aruco = self.fresh('aruco')
        if aruco is not None:
            for d in aruco:
                if not d.get('corners'):
                    continue
                poly(img, d['corners'], GREEN, 2)
                name = self.aruco_names.get(int(d['id']))
                dist = d.get('distance')
                text = f"ArUco {d['id']}" + (f' - {name}' if name else '') + \
                    (f' {float(dist):.2f} m' if dist is not None else '')
                x, y = np.asarray(d['corners'], dtype=float).min(axis=0)
                label(img, text, (x, y - 4), GREEN)
        label(img, 'on: ' + (', '.join(on) if on else 'ArUco only'), (6, 18), WHITE)
        self.pub.publish(bgr_to_image(img, msg.header))


def main(args=None):
    rclpy.init(args=args)
    node = VisionOverlay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
