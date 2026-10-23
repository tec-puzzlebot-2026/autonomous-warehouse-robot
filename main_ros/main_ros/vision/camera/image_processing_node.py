"""ImageProcessing node - colour correction + rectification (MAS507 JetbotCamera).

Subscribes:  /camera/image_raw
Publishes:   /camera/image_cc    colour-corrected, original geometry
             /camera/image_rect  colour-corrected + rectified (apply_rectify:=true)

Order matters: the colour gains were measured on raw (distorted) white images,
so they are applied BEFORE rectification.

With rectifier:=image_proc in vision.launch.py this node runs with
apply_rectify:=false and image_proc::RectifyNode rectifies /camera/image_cc
using /camera/camera_info; both routes end up on /camera/image_rect.

Work is only done when someone subscribes to an output (saves CPU on the Nano).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image

import cv2

from main_ros.vision.camera.calibration import calib_io
from main_ros.vision.camera.image_conversions import bgr_to_imgmsg, imgmsg_to_bgr


class ImageProcessing(Node):

    def __init__(self):
        super().__init__('image_processing')
        dp = self.declare_parameter
        in_topic = dp('input_topic', '/camera/image_raw').value
        calib_dir = dp('calib_dir', calib_io.DEFAULT_CAMERA_CALIB_DIR).value
        self.apply_color = dp('apply_color', True).value
        self.apply_rectify = dp('apply_rectify', True).value
        color_file = dp('color_file', '').value
        intr_file = dp('intrinsics_file', '').value          # .yaml or MAS507 .npz
        alpha = dp('alpha', -1.0).value                      # <0: use P from yaml
        self.gains = None
        self.maps = None
        self.warned = set()

        if self.apply_color:
            path = calib_io.resolve(calib_dir, color_file, calib_io.COLOR_NPZ_FILE)
            try:
                self.gains = calib_io.load_color_gains(path)
                self.get_logger().info('Colour gains %dx%d from %s'
                                       % (self.gains.shape[1], self.gains.shape[0], path))
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn('No colour calibration (%s) - passing colours through' % e)

        if self.apply_rectify:
            path = calib_io.resolve(calib_dir, intr_file, calib_io.CAMERA_INFO_FILE)
            try:
                c = calib_io.load_intrinsics(path)
                if c['width'] is None:
                    raise ValueError('npz without image_size - set the size by '
                                     'recalibrating with calibrate_intrinsics')
                size = (c['width'], c['height'])
                use_alpha = None if (alpha < 0 and c['P'] is not None) else max(alpha, 0.0)
                m1, m2, newK = calib_io.build_rectify_maps(c['K'], c['D'], size,
                                                           c['R'], c['P'], use_alpha)
                self.maps = (m1, m2, size)
                self.get_logger().info('Rectification %dx%d from %s, new fx=%.1f'
                                       % (size[0], size[1], path, newK[0, 0]))
            except Exception as e:  # noqa: BLE001
                self.get_logger().error('Rectification disabled: %s' % e)

        qos = QoSProfile(depth=2, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.pub_cc = self.create_publisher(Image, '/camera/image_cc', qos)
        self.pub_rect = (self.create_publisher(Image, '/camera/image_rect', qos)
                         if self.apply_rectify else None)
        self.create_subscription(Image, in_topic, self.on_image, qos_profile_sensor_data)

    def warn_once(self, key, text):
        if key not in self.warned:
            self.warned.add(key)
            self.get_logger().warn(text)

    def on_image(self, msg):
        want_cc = self.pub_cc.get_subscription_count() > 0
        want_rect = self.pub_rect is not None and self.pub_rect.get_subscription_count() > 0
        if not (want_cc or want_rect):
            return
        img = imgmsg_to_bgr(msg)
        h, w = img.shape[:2]

        if self.gains is not None:
            if self.gains.shape[:2] == (h, w):
                img = calib_io.apply_color_gains(img, self.gains)
            else:
                self.warn_once('gsize', 'Colour gains are %dx%d but images are %dx%d - '
                               'redo colour calibration at this resolution'
                               % (self.gains.shape[1], self.gains.shape[0], w, h))
        if want_cc:
            self.pub_cc.publish(bgr_to_imgmsg(img, msg.header, Image))

        if want_rect and self.maps is not None:
            m1, m2, size = self.maps
            if size != (w, h):
                self.warn_once('rsize', 'Intrinsics are for %dx%d but images are %dx%d - '
                               'recalibrate at this resolution' % (size[0], size[1], w, h))
                return
            rect = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
            self.pub_rect.publish(bgr_to_imgmsg(rect, msg.header, Image))


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
