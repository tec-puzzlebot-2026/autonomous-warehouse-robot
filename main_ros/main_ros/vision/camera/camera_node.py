"""Raspberry Pi Camera v2 (IMX219) driver for the Jetson Nano.

Publishes (in the node namespace, "camera" in the launch files):
    image_raw    sensor_msgs/Image (bgr8)  - untouched sensor image
    camera_info  sensor_msgs/CameraInfo    - from data/camera_info.yaml
Serves:
    set_camera_info  sensor_msgs/srv/SetCameraInfo - used by the image_pipeline
                     cameracalibrator COMMIT button; writes camera_info.yaml

The CSI camera is read through GStreamer (nvarguscamerasrc), so the OpenCV
used by python3 must be built with GStreamer support. Check with:
    python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
"""
import threading

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.srv import SetCameraInfo
from std_msgs.msg import Header

from main_ros.vision.camera.calibration import calib_io
from main_ros.vision.camera.image_conversions import bgr_to_imgmsg

# IMX219 sensor modes on the Jetson (width, height, max fps):
#   3264x2464@21  3264x1848@28  1920x1080@30  1640x1232@30  1280x720@60
# 1640x1232 keeps the full 4:3 field of view (binned) and is a good default.


def csi_pipeline(sensor_id, cap_w, cap_h, fps, out_w, out_h, flip, extra):
    return (
        'nvarguscamerasrc sensor-id=%d %s ! '
        'video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1, format=NV12 ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw, width=%d, height=%d, format=BGRx ! '
        'videoconvert ! video/x-raw, format=BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
        % (sensor_id, extra, cap_w, cap_h, fps, flip, out_w, out_h)
    )


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')
        dp = self.declare_parameter
        self.source = dp('source', 'csi').value              # csi | v4l2
        sensor_id = dp('sensor_id', 0).value
        cap_w = dp('capture_width', 1640).value
        cap_h = dp('capture_height', 1232).value
        self.width = dp('width', 640).value
        self.height = dp('height', 480).value
        fps = dp('framerate', 30).value
        flip = dp('flip_method', 0).value                    # 2 = rotate 180
        argus_extra = dp('argus_extra', '').value            # e.g. "wbmode=1 aelock=false"
        device = dp('device', '/dev/video0').value
        pipeline = dp('pipeline', '').value                  # full override
        self.frame_id = dp('frame_id', 'camera_optical_frame').value
        self.camera_name = dp('camera_name', 'pi_camera_v2').value
        calib_dir = dp('calib_dir', calib_io.DEFAULT_CAMERA_CALIB_DIR).value
        info_file = dp('camera_info_file', '').value
        self.info_path = calib_io.resolve(calib_dir, info_file, calib_io.CAMERA_INFO_FILE)

        # Open the camera
        if pipeline:
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        elif self.source == 'csi':
            pipeline = csi_pipeline(sensor_id, cap_w, cap_h, fps,
                                    self.width, self.height, flip, argus_extra)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            self.get_logger().fatal(
                'Could not open camera. Pipeline: %s\n'
                'Checks: (1) OpenCV built with GStreamer (see module docstring); '
                '(2) camera works with: gst-launch-1.0 nvarguscamerasrc ! fakesink; '
                '(3) if nvargus hangs: sudo systemctl restart nvargus-daemon' % pipeline)
            raise RuntimeError('camera open failed')
        self.get_logger().info('Camera opened: %s' % (pipeline or device))

        # Reliable publishers are compatible with both reliable and
        # best-effort subscribers (rqt_image_view, cameracalibrator, image_proc).
        qos = QoSProfile(depth=2, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.pub_img = self.create_publisher(Image, 'image_raw', qos)
        self.pub_info = self.create_publisher(CameraInfo, 'camera_info', qos)
        self.create_service(SetCameraInfo, 'set_camera_info', self.on_set_camera_info)

        self.info_lock = threading.Lock()
        self.camera_info = self.load_camera_info()

        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ camera_info
    def load_camera_info(self):
        msg = CameraInfo()
        msg.width, msg.height = self.width, self.height
        msg.distortion_model = 'plumb_bob'
        if not self.info_path.is_file():
            self.get_logger().warn(
                'No calibration at %s - publishing an uncalibrated camera_info' % self.info_path)
            return msg
        try:
            c = calib_io.read_camera_info_yaml(self.info_path)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('Failed to read %s: %s' % (self.info_path, e))
            return msg
        if (c['width'], c['height']) != (self.width, self.height):
            self.get_logger().warn(
                'Calibration is for %dx%d but the camera outputs %dx%d - recalibrate '
                'at this resolution.' % (c['width'], c['height'], self.width, self.height))
        msg.width, msg.height = c['width'], c['height']
        msg.distortion_model = c['distortion_model']
        msg.k = [float(x) for x in c['K'].ravel()]
        msg.d = [float(x) for x in c['D']]
        msg.r = [float(x) for x in c['R'].ravel()]
        msg.p = [float(x) for x in c['P'].ravel()]
        self.get_logger().info('Loaded calibration from %s' % self.info_path)
        return msg

    def on_set_camera_info(self, request, response):
        ci = request.camera_info
        try:
            calib_io.write_camera_info_yaml(
                self.info_path, ci.width, ci.height, list(ci.k), list(ci.d),
                list(ci.r), list(ci.p), camera_name=self.camera_name,
                distortion_model=ci.distortion_model or 'plumb_bob')
        except Exception as e:  # noqa: BLE001
            response.success = False
            response.status_message = 'Failed to write %s: %s' % (self.info_path, e)
            self.get_logger().error(response.status_message)
            return response
        with self.info_lock:
            self.camera_info = ci
        response.success = True
        response.status_message = 'Saved to %s' % self.info_path
        self.get_logger().info(response.status_message)
        return response

    # ---------------------------------------------------------------- capture
    def capture_loop(self):
        failures = 0
        while self.running and rclpy.ok():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                failures += 1
                if failures % 30 == 1:
                    self.get_logger().warn('Frame grab failed (%d)' % failures)
                continue
            failures = 0
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id
            self.pub_img.publish(bgr_to_imgmsg(frame, header, Image))
            with self.info_lock:
                info = self.camera_info
            info.header = header      # identical stamp -> exact-time sync works
            self.pub_info.publish(info)

    def destroy_node(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()