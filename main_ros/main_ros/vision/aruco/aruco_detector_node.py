"""ArUco detector node on the rectified image.

Subscribes: /camera/image_rect     (sensor_msgs/Image, color-corrected and rectified)
            /camera/camera_info    (sensor_msgs/CameraInfo)
Publishes:  /tf                    camera_optical_frame -> aruco_<id>
            /aruco/detections      std_msgs/String with JSON (one list per image)
            /aruco/debug_image     image with markers and axes (only if someone is viewing it)

Intrinsics: the rectified image has no distortion and its camera matrix is
P[:3,:3] of camera_info (not K). With use_rectified:=false, K and D are used on the
raw image.

Marker size: marker_size is the default side of the black square (m); per_marker_ids /
per_marker_size give a different size for each ID.

The JSON avoids creating an interface package (ament_cmake) for now. Each
element: id, frame_id, stamp, t [m] and q (x,y,z,w) of the marker in the optical frame,
distance [m], view_angle_deg, reproj_err_px, ambiguity (error 2nd solution / 1st;
~1 = ambiguous orientation), area_px.
"""
import json
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from main_ros.vision.aruco import aruco_core
from main_ros.vision.aruco import transforms as tf_
from main_ros.vision.aruco.transforms import matrix_to_quat
from main_ros.vision.image_conversions import bgr_to_imgmsg, imgmsg_to_bgr


def _finite(x):
    """JSON does not support NaN/inf: they are reported as null."""
    return float(x) if x is not None and math.isfinite(x) else None


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector')
        dp = self.declare_parameter
        image_topic = dp('image_topic', '/camera/image_rect').value
        info_topic = dp('camera_info_topic', '/camera/camera_info').value
        self.use_rectified = dp('use_rectified', True).value
        dictionary = dp('dictionary', aruco_core.DEFAULT_DICTIONARY).value
        # Side of the OUTER BLACK SQUARE of the ArUco, in meters (not the white paper).
        self.marker_size = float(dp('marker_size', 0.11).value)      # default size
        # Optional size per ID (parallel lists; ids < 0 are placeholders; sizes need a decimal point)
        self.sizes = tf_.per_marker_map(dp('per_marker_ids', [-1]).value,
                                        dp('per_marker_size', [0.0]).value, 'size')
        if any(v <= 0 for v in self.sizes.values()):
            raise ValueError('every per_marker_size must be > 0')
        ids = dp('marker_ids', '').value          # "" = all, or "21,22,23"
        self.allowed = {int(s) for s in str(ids).replace(' ', '').split(',') if s}
        self.publish_tf = dp('publish_tf', True).value
        self.min_area_px = float(dp('min_area_px', 100.0).value)

        if self.marker_size <= 0:
            raise ValueError('marker_size must be > 0')
        self.detector = aruco_core.MarkerDetector(dictionary)
        self.K = None
        self.D = None
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_det = self.create_publisher(String, '/aruco/detections', 10)
        self.pub_dbg = self.create_publisher(Image, '/aruco/debug_image', 2)
        self.create_subscription(CameraInfo, info_topic, self.on_info, 10)
        self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
        self.get_logger().info(
            'ArUco %s, side %.4f m (per ID: %s), ids %s, image %s (%s)' % (
                dictionary, self.marker_size, self.sizes or '-', sorted(self.allowed) or 'all',
                image_topic,
                'rectified: P, D=0' if self.use_rectified else 'raw: K, D'))

    def on_info(self, msg):
        if self.use_rectified:
            K = np.array(msg.p, dtype=float).reshape(3, 4)[:3, :3]
            D = np.zeros(5)
        else:
            K = np.array(msg.k, dtype=float).reshape(3, 3)
            D = np.array(msg.d, dtype=float)
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            self.get_logger().warn('camera_info not calibrated (fx or fy = 0); ignored',
                                   throttle_duration_sec=5.0)
            return
        self.K, self.D = K, D

    def on_image(self, msg):
        if self.K is None:
            self.get_logger().warn('Waiting for calibrated camera_info...', throttle_duration_sec=5.0)
            return
        img = imgmsg_to_bgr(msg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = []
        for marker_id, corners in self.detector.detect(gray):
            if self.allowed and marker_id not in self.allowed:
                continue
            if aruco_core.marker_area_px(corners) < self.min_area_px:
                continue
            try:
                det = aruco_core.describe_detection(
                    marker_id, corners, self.sizes.get(marker_id, self.marker_size),
                    self.K, self.D)
            except (cv2.error, ValueError, ZeroDivisionError) as e:
                self.get_logger().warn('Pose failed for id %d: %s' % (marker_id, e))
                continue
            if det is not None:
                detections.append(det)

        out = []
        for d in detections:
            q = matrix_to_quat(d['R'])
            out.append({
                'id': d['id'], 'frame_id': msg.header.frame_id,
                'stamp': msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec,
                't': [float(v) for v in d['t']], 'q': [float(v) for v in q],
                'distance': _finite(d['distance']),
                'view_angle_deg': _finite(d['view_angle_deg']),
                'reproj_err_px': _finite(d['reproj_err_px']),
                'ambiguity': _finite(d['ambiguity']),
                'area_px': _finite(d['area_px']),
            })
            if self.publish_tf:
                self.send_tf(msg.header, d['id'], d['t'], q)
        if out:
            self.pub_det.publish(String(data=json.dumps(out)))
        if self.pub_dbg.get_subscription_count() > 0:
            self.publish_debug(img, msg.header, detections)

    def send_tf(self, header, marker_id, t, q):
        ts = TransformStamped()
        ts.header = header
        ts.child_frame_id = 'aruco_%d' % marker_id
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = \
            float(t[0]), float(t[1]), float(t[2])
        (ts.transform.rotation.x, ts.transform.rotation.y,
         ts.transform.rotation.z, ts.transform.rotation.w) = [float(v) for v in q]
        self.tf_broadcaster.sendTransform(ts)

    def publish_debug(self, img, header, detections):
        for d in detections:
            pts = d['corners'].astype(int).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            if hasattr(cv2, 'drawFrameAxes'):
                cv2.drawFrameAxes(img, self.K, self.D, d['rvec'], d['t'],
                                  self.sizes.get(d['id'], self.marker_size) / 2)
            amb = d['ambiguity']
            text = 'id %d  %.2fm  %.0fdeg%s' % (
                d['id'], d['distance'], d['view_angle_deg'],
                '  AMBIGUOUS' if amb is not None and math.isfinite(amb) and amb < 1.5 else '')
            org = tuple(int(v) for v in d['corners'][0])
            cv2.putText(img, text, (org[0], max(org[1] - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        self.pub_dbg.publish(bgr_to_imgmsg(img, header, Image))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ArucoDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
