"""ImageSaver node - ROS 2 version of the MAS507 ImageSaving node.

Saves the latest *raw* camera frame as PNG when triggered:
    ros2 topic pub --once /save_image std_msgs/msg/Bool "{data: true}"
    ros2 service call /capture std_srvs/srv/Trigger
(--once removes the CTRL+C step from the MAS507 instructions.)

mode:=checkerboard  -> saves to <calib_dir>/images, refuses frames where the
                       board is not fully detected (require_board)
mode:=white         -> saves to <calib_dir>/whiteimages, for colour calibration

It also publishes /calibration/preview (bgr8) with the detected corners, image
sharpness, saved-image count and, in white mode, brightness/saturation, so
you can aim the camera from rqt_image_view or Foxglove.
"""
import re

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from main_ros.vision.camera.calibration import calib_io
from main_ros.vision.camera.image_conversions import bgr_to_imgmsg, imgmsg_to_bgr

PREVIEW_MAX_WIDTH = 640


def sharpness(gray):
    """Variance of the Laplacian - drops sharply with motion blur/defocus."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class ImageSaver(Node):

    def __init__(self):
        super().__init__('image_saver')
        dp = self.declare_parameter
        image_topic = dp('image_topic', '/camera/image_raw').value
        calib_dir = dp('calib_dir', calib_io.DEFAULT_CAMERA_CALIB_DIR).value
        self.mode = dp('mode', 'checkerboard').value        # checkerboard | white
        save_dir = dp('save_dir', '').value
        # Inner corners (a 8x6-square board has 7x5 inner corners)
        self.pattern = (dp('pattern_cols', 7).value, dp('pattern_rows', 5).value)
        self.require_board = dp('require_board', self.mode == 'checkerboard').value
        preview_rate = dp('preview_rate', 4.0).value
        self.prefix = dp('prefix', 'img').value

        default_sub = 'images' if self.mode == 'checkerboard' else 'whiteimages'
        self.save_dir = calib_io.resolve(calib_dir, save_dir, default_sub)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.index = self.next_index()
        self.saved_this_session = 0

        self.latest = None
        self.last_saved_stamp = None
        self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(Bool, 'save_image', self.on_save_topic, 10)
        self.create_service(Trigger, 'capture', self.on_capture)
        self.pub_preview = self.create_publisher(Image, '/calibration/preview', 2)
        if preview_rate > 0:
            self.create_timer(1.0 / preview_rate, self.publish_preview)

        self.get_logger().info(
            'Mode=%s, saving to %s (next index %d). Pattern %dx%d inner corners.'
            % (self.mode, self.save_dir, self.index, self.pattern[0], self.pattern[1]))

    def next_index(self):
        pat = re.compile(r'^%s_(\d+)$' % re.escape(self.prefix))
        nums = [int(m.group(1)) for m in
                (pat.match(p.stem) for p in calib_io.list_images(self.save_dir)) if m]
        return max(nums) + 1 if nums else 0

    def on_image(self, msg):
        self.latest = msg

    # ------------------------------------------------------------------ save
    def on_save_topic(self, msg):
        if msg.data:
            ok, text = self.save()
            (self.get_logger().info if ok else self.get_logger().warn)(text)

    def on_capture(self, _request, response):
        response.success, response.message = self.save()
        (self.get_logger().info if response.success else self.get_logger().warn)(
            response.message)
        return response

    def save(self):
        msg = self.latest
        if msg is None:
            return False, 'No image received yet - is the camera running?'
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self.last_saved_stamp:
            return False, 'No new frame since the last save'
        img = imgmsg_to_bgr(msg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.require_board:
            found, corners = cv2.findChessboardCorners(
                gray, self.pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not found:
                return False, ('Checkerboard (%dx%d inner corners) not fully visible - '
                               'not saved' % self.pattern)
            if not calib_io.corners_clear_of_border(corners, gray.shape[1], gray.shape[0]):
                return False, 'Checkerboard touches the image edge - move it inwards'
        path = self.save_dir / ('%s_%03d.png' % (self.prefix, self.index))
        if not cv2.imwrite(str(path), img):
            return False, 'Failed to write %s' % path
        self.index += 1
        self.saved_this_session += 1
        self.last_saved_stamp = stamp
        return True, 'Saved %s (%d this session, sharpness %.0f)' % (
            path.name, self.saved_this_session, sharpness(gray))

    # --------------------------------------------------------------- preview
    def publish_preview(self):
        msg = self.latest
        if msg is None or self.pub_preview.get_subscription_count() == 0:
            return
        img = imgmsg_to_bgr(msg)
        scale = min(1.0, PREVIEW_MAX_WIDTH / float(img.shape[1]))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = []
        if self.mode == 'checkerboard':
            found, corners = cv2.findChessboardCorners(
                gray, self.pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK)
            if corners is not None:
                cv2.drawChessboardCorners(img, self.pattern, corners, found)
            if found and not calib_io.corners_clear_of_border(
                    corners, gray.shape[1], gray.shape[0], margin=5):
                lines.append(('BOARD AT EDGE', False))
            else:
                lines.append(('BOARD OK' if found else 'NO BOARD', found))
            lines.append(('sharpness %.0f' % sharpness(gray), True))
        else:
            mean = float(gray.mean())
            sat = 100.0 * float(np.mean(img >= 250))
            ok = 120 <= mean <= 230 and sat < 1.0
            lines.append(('mean %.0f  saturated %.1f%%' % (mean, sat), ok))
        lines.append(('saved %d (next #%d)' % (self.saved_this_session, self.index), True))
        for i, (text, good) in enumerate(lines):
            color = (0, 200, 0) if good else (0, 0, 255)
            y = 22 + 22 * i
            cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        self.pub_preview.publish(bgr_to_imgmsg(img, msg.header, Image))


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
