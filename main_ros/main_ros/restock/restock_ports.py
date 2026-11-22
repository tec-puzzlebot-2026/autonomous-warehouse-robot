"""
restock_ports.py — the ROS 2 side of the restocking state machine
==================================================================
RestockPorts gives restock_core.RestockCore everything it uses, on top of the
Program Manager's node (no extra node, no extra thread):

  motion      Sendero actions (/navigate_to_pose, /align), short odometry drives and
              turns of our own (/wheel/odom in, cmd_vel out: closing in on a box,
              into a trailer, the start-up search)
  forklift    /forklift/pick_box and /forklift/set_down_box (warehouse_msgs)
  poses       /slam/pose (Planisferio), /ekf/pose and /ekf/status (Chris's EKF),
              /slam/set_pose (seed), /ekf/relocalize
  perception  the JSON topics of box_far, box_target, qr_reader, logo_detector and the
              landmark mapper; each detector is switched with its ~/enable topic
  LiDAR       the raw scan (box face fit, open direction for the start-up search)
  carrying    /restock/carrying (scan filter mask)

Every topic name is a setting (config/restock.yaml, section ports).
"""
# Author: Raymundo Gomez

import collections
import json
import math
import time

import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, LaserScan
from std_msgs.msg import Bool, Empty, String

from main_ros.restock import geometry as G
from main_ros.restock.landmark_mapper_node import load_mount, yaw_of

DEFAULT_TOPICS = {
    'navigate_action': '/navigate_to_pose',
    'align_action': '/align',
    'pick_action': '/forklift/pick_box',
    'set_down_action': '/forklift/set_down_box',
    'slam_pose': '/slam/pose',
    'slam_set_pose': '/slam/set_pose',
    'ekf_pose': '/ekf/pose',
    'ekf_status': '/ekf/status',
    'ekf_relocalize': '/ekf/relocalize',
    'odom': '/wheel/odom',
    'scan': '/laser_scan',
    'camera_info': '/camera/camera_info',
    'carrying': '/restock/carrying',
    'landmarks': '/landmark_mapper/landmarks',
    'cmd_vel': '/cmd_vel',
    # stream: (output topic, enable topic)
    'box_far': ['/box_far/target', '/box_far/enable'],
    'box_target': ['/box_target/target', '/box_target/enable'],
    'qr': ['/qr_reader/codes', '/qr_reader/enable'],
    'logos': ['/logo_detector/detections', '/logo_detector/enable'],
}
STREAMS = ('box_far', 'box_target', 'qr', 'logos')
SUCCEEDED = 4                                   # action_msgs/GoalStatus.STATUS_SUCCEEDED


def _yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def _pose_stamped(node, pose, frame='map'):
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.header.stamp = node.get_clock().now().to_msg()
    ps.pose.position.x, ps.pose.position.y = float(pose[0]), float(pose[1])
    q = _yaw_quat(float(pose[2]))
    (ps.pose.orientation.x, ps.pose.orientation.y,
     ps.pose.orientation.z, ps.pose.orientation.w) = q
    return ps


class _Done:
    """An already finished handle."""

    def __init__(self, ok, msg=''):
        self._r = (ok, msg)

    def done(self):
        return True

    def result(self):
        return self._r

    def cancel(self):
        pass


class _ActionHandle:
    """send_goal_async -> done() / result() -> (ok, msg) / cancel()."""

    def __init__(self, client, goal, judge, feedback=None):
        self.judge = judge
        self.goal_handle = None
        self.result_future = None
        self._final = None
        self._cancel = False
        self.goal_future = client.send_goal_async(goal, feedback_callback=feedback)
        self.goal_future.add_done_callback(self._on_goal)

    def _on_goal(self, fut):
        try:
            gh = fut.result()
        except Exception as exc:                # noqa: BLE001
            self._final = (False, f'goal not sent: {exc}')
            return
        if gh is None or not gh.accepted:
            self._final = (False, 'goal rejected')
            return
        self.goal_handle = gh
        if self._cancel:
            gh.cancel_goal_async()
        self.result_future = gh.get_result_async()
        self.result_future.add_done_callback(self._on_result)

    def _on_result(self, fut):
        try:
            res = fut.result()
            self._final = self.judge(res.status, res.result)
        except Exception as exc:                # noqa: BLE001
            self._final = (False, f'no result: {exc}')

    def done(self):
        return self._final is not None

    def result(self):
        return self._final or (False, 'not finished')

    def cancel(self):
        self._cancel = True
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()


class _MotionJob:
    """A straight drive or an in-place turn by odometry (driven by the 20 Hz timer)."""

    def __init__(self, kind, amount, start, speed, centered=False):
        self.kind, self.amount, self.start = kind, float(amount), start
        self.speed, self.centered = speed, centered
        self._final = None
        self.turned = 0.0
        self.last_yaw = start[2]

    def done(self):
        return self._final is not None

    def result(self):
        return self._final or (False, 'not finished')

    def cancel(self):
        if self._final is None:
            self._final = (False, 'canceled')


class RestockPorts:
    """Ports of RestockCore on the Program Manager's node."""

    def __init__(self, node, cfg, log, mission_folder, slam_status):
        """
        node: the Program Manager node. cfg: config/restock.yaml (dict).
        log: function(text). mission_folder: function() -> folder path.
        slam_status: function() -> dict of Planisferio's /slam/status.
        """
        from rclpy.action import ActionClient
        self.node = node
        self.cfg = cfg
        self.log = log
        self._folder = mission_folder
        self._slam_status = slam_status
        pc = dict(cfg.get('ports', {}) or {})
        self.topics = dict(DEFAULT_TOPICS)
        self.topics.update(pc.get('topics', {}) or {})
        mc = pc.get('motion', {}) or {}
        self.drive_speed = float(mc.get('drive_speed', 0.06))
        self.turn_speed = float(mc.get('turn_speed', 0.5))
        self.center_gain = float(mc.get('center_gain', 1.5))
        self.pose_max_age = float(pc.get('pose_max_age_s', 1.0))
        self.laser_mount = tuple(float(v) for v in pc.get('laser_mount', [0.0, 0.0, 0.0]))
        mount = tuple(float(v) for v in pc.get('camera_mount', [0.10005, 0.0, 0.178, 0.0,
                                                                0.26, -0.0124]))
        base_h = float(pc.get('base_link_height', 0.047))
        aruco_cfg = pc.get('aruco_config', '')
        if aruco_cfg:
            try:
                mount, base_h = load_mount(aruco_cfg)
            except OSError as exc:
                self.log(f'restock: cannot read {aruco_cfg} ({exc}); using the camera_mount '
                         'setting')
        self.mount, self.base_h = mount, base_h
        self.cam = None

        t = self.topics
        self._nav_cls = self._align_cls = self._pick_cls = self._set_down_cls = None
        try:
            from nav2_msgs.action import NavigateToPose
            self._nav_cls = NavigateToPose
            self.nav_client = ActionClient(node, NavigateToPose, t['navigate_action'])
        except ImportError:
            self.nav_client = None
            self.log('restock: nav2_msgs is not installed: no navigation')
        try:
            from sendero_msgs.action import Align
            self._align_cls = Align
            self.align_client = ActionClient(node, Align, t['align_action'])
        except ImportError:
            self.align_client = None
            self.log('restock: sendero_msgs is not installed: alignment uses NavigateToPose')
        try:
            from warehouse_msgs.action import PickBox, SetDownBox
            self._pick_cls, self._set_down_cls = PickBox, SetDownBox
            self.pick_client = ActionClient(node, PickBox, t['pick_action'])
            self.set_down_client = ActionClient(node, SetDownBox, t['set_down_action'])
        except ImportError:
            self.pick_client = self.set_down_client = None
            self.log('restock: warehouse_msgs is not built: no forklift actions')

        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = node.create_publisher(Twist, t['cmd_vel'], 10)
        self.carry_pub = node.create_publisher(Bool, t['carrying'], latched)
        self.seed_pub = node.create_publisher(PoseWithCovarianceStamped, t['slam_set_pose'], 10)
        self.reloc_pub = node.create_publisher(Empty, t['ekf_relocalize'], 10)
        self.enable_pubs = {s: node.create_publisher(Bool, t[s][1], 10) for s in STREAMS}

        self.buffers = {s: collections.deque(maxlen=80) for s in STREAMS}
        for s in STREAMS:
            node.create_subscription(String, t[s][0], lambda m, s=s: self._on_stream(s, m), 20)
        self.landmarks_data = None
        node.create_subscription(String, t['landmarks'], self._on_landmarks, 5)
        self.slam = (None, 0.0)
        self.ekf = (None, 0.0)
        self.ekf_state = ('', 0.0)
        node.create_subscription(PoseWithCovarianceStamped, t['slam_pose'],
                                 lambda m: self._on_pose('slam', m), 20)
        node.create_subscription(PoseWithCovarianceStamped, t['ekf_pose'],
                                 lambda m: self._on_pose('ekf', m), 20)
        node.create_subscription(String, t['ekf_status'], self._on_ekf_status, 10)
        self.odom = None
        node.create_subscription(Odometry, t['odom'], self._on_odom, qos_profile_sensor_data)
        self.scan = None
        node.create_subscription(LaserScan, t['scan'], self._on_scan, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, t['camera_info'], self._on_info, 5)

        self.want = {s: False for s in STREAMS}
        self.carrying = False
        self.job = None
        self.actions = []                       # active navigation handles
        self.forklift = []                      # active forklift handles
        node.create_timer(0.05, self._motion_tick)
        node.create_timer(1.0, self._republish)

    # =====================================================================
    # subscriptions
    # =====================================================================
    def _on_stream(self, stream, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(data, dict) and not data.get('disabled'):
            self.buffers[stream].append((time.monotonic(), data))

    def _on_landmarks(self, msg):
        try:
            self.landmarks_data = json.loads(msg.data)
        except ValueError:
            pass

    def _on_pose(self, which, msg):
        p = msg.pose.pose
        pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        setattr(self, which, (pose, time.monotonic()))

    def _on_ekf_status(self, msg):
        self.ekf_state = (msg.data.strip().split()[0] if msg.data.strip() else '',
                          time.monotonic())

    def _on_odom(self, msg):
        p = msg.pose.pose
        self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

    def _on_scan(self, msg):
        self.scan = msg

    def _on_info(self, msg):
        if self.cam is None:
            self.cam = G.Camera.from_camera_info(list(msg.k), self.mount, self.base_h, list(msg.p))
            self.log(f'restock: camera fx {self.cam.fx:.1f} fy {self.cam.fy:.1f}')

    def _republish(self):
        """Repeat switches once a second (a node that started late still gets them)."""
        for s in STREAMS:
            self.enable_pubs[s].publish(Bool(data=self.want[s]))
        self.carry_pub.publish(Bool(data=self.carrying))

    # =====================================================================
    # ports: time, log, folder
    # =====================================================================
    def now(self):
        return time.monotonic()

    def mission_folder(self):
        return self._folder()

    # =====================================================================
    # ports: navigation
    # =====================================================================
    def navigate(self, pose, profile):
        if self.nav_client is None or not self.nav_client.server_is_ready():
            return _Done(False, 'Sendero /navigate_to_pose is not available')
        goal = self._nav_cls.Goal()
        goal.pose = _pose_stamped(self.node, pose)
        goal.behavior_tree = profile or ''
        h = _ActionHandle(self.nav_client, goal, self._judge_nav)
        self.actions.append(h)
        return h

    @staticmethod
    def _judge_nav(status, result):
        if status == SUCCEEDED:
            return True, ''
        msg = getattr(result, 'error_msg', '') or f'status {status}'
        return False, msg

    def align(self, target, standoff, profile):
        """Square up to `target` (x axis pointing out of the face), `standoff` m in front."""
        if self.align_client is None or not self.align_client.server_is_ready():
            goal_pose = G.pose_facing(target[:2], target[2], standoff)
            return self.navigate(goal_pose, profile)
        goal = self._align_cls.Goal()
        goal.target_topic = ''
        goal.target_pose = _pose_stamped(self.node, target)
        goal.robot_pose_in_target.position.x = float(standoff)
        q = _yaw_quat(math.pi)
        (goal.robot_pose_in_target.orientation.x, goal.robot_pose_in_target.orientation.y,
         goal.robot_pose_in_target.orientation.z, goal.robot_pose_in_target.orientation.w) = q
        goal.forward_only = False
        goal.profile = profile or ''
        h = _ActionHandle(self.align_client, goal, self._judge_align)
        self.actions.append(h)
        return h

    @staticmethod
    def _judge_align(status, result):
        if status == SUCCEEDED and int(getattr(result, 'error_code', 0)) == 0:
            return True, ''
        return False, getattr(result, 'error_msg', '') or f'status {status}'

    def drive_straight(self, distance, centered=False):
        return self._start_job('drive', distance, centered)

    def rotate(self, angle):
        return self._start_job('turn', angle, False)

    def _start_job(self, kind, amount, centered):
        if self.odom is None:
            return _Done(False, 'no wheel odometry')
        if self.job is not None:
            self.job.cancel()
        if abs(amount) < 1e-3:
            return _Done(True)
        speed = self.drive_speed if kind == 'drive' else self.turn_speed
        self.job = _MotionJob(kind, amount, self.odom, speed, centered)
        return self.job

    def _motion_tick(self):
        job = self.job
        if job is None:
            return
        if job.done():
            self.job = None
            self.cmd_pub.publish(Twist())
            return
        x, y, th = self.odom
        cmd = Twist()
        if job.kind == 'drive':
            sx, sy, sth = job.start
            along = (x - sx) * math.cos(sth) + (y - sy) * math.sin(sth)
            left = job.amount - along
            if abs(left) < 0.004 or (left * job.amount < 0):
                job._final = (True, '')
                self.cmd_pub.publish(Twist())
                self.job = None
                return
            v = math.copysign(min(job.speed, max(0.02, 1.5 * abs(left))), left)
            w = 2.0 * G.wrap(sth - th)                       # hold the heading
            if job.centered and job.amount > 0:
                w += self._centering()
            cmd.linear.x, cmd.angular.z = v, max(-0.4, min(0.4, w))
        else:
            job.turned += G.wrap(th - job.last_yaw)
            job.last_yaw = th
            left = job.amount - job.turned
            if abs(left) < math.radians(1.5):
                job._final = (True, '')
                self.cmd_pub.publish(Twist())
                self.job = None
                return
            cmd.angular.z = math.copysign(min(job.speed, max(0.15, 1.5 * abs(left))), left)
        self.cmd_pub.publish(cmd)

    def _centering(self):
        """Steer toward the middle of a trailer: compare the side walls on the LiDAR."""
        pts = self._scan_points()
        if pts is None:
            return 0.0
        side = pts[(np.abs(pts[:, 0]) < 0.15)]
        left = side[side[:, 1] > 0.05][:, 1]
        right = -side[side[:, 1] < -0.05][:, 1]
        if left.size < 3 or right.size < 3:
            return 0.0
        err = 0.5 * (float(np.median(left)) - float(np.median(right)))   # >0: too far right
        return max(-0.3, min(0.3, self.center_gain * err))

    def cancel_motion(self):
        for h in self.actions:
            if not h.done():
                h.cancel()
        self.actions = [h for h in self.actions if not h.done()]
        if self.job is not None:
            self.job.cancel()
            self.job = None
        for _ in range(3):
            self.cmd_pub.publish(Twist())

    def clear_costmaps(self):
        pub = getattr(self.node, 'nav_pub', None)
        if pub is not None:
            pub.publish(String(data='CLEAR_COSTMAPS'))

    # =====================================================================
    # ports: forklift
    # =====================================================================
    def pick(self, box_id, level, face_pose, gap):
        if self.pick_client is None or not self.pick_client.server_is_ready():
            return _Done(False, 'forklift /forklift/pick_box is not available')
        goal = self._pick_cls.Goal()
        goal.box_id = str(box_id)
        goal.level = int(level)
        # The action wants the face pose with x pointing INTO the box.
        goal.face_pose = _pose_stamped(self.node, (face_pose[0], face_pose[1],
                                                   G.wrap(face_pose[2] + math.pi)))
        goal.distance = float(gap)
        h = _ActionHandle(self.pick_client, goal, self._judge_forklift,
                          feedback=lambda fb: self.log(f'forklift: {fb.feedback.phase}'))
        self.forklift.append(h)
        return h

    def set_down(self, box_id, where):
        if self.set_down_client is None or not self.set_down_client.server_is_ready():
            return _Done(False, 'forklift /forklift/set_down_box is not available')
        goal = self._set_down_cls.Goal()
        goal.box_id = str(box_id)
        goal.where = str(where)
        h = _ActionHandle(self.set_down_client, goal, self._judge_forklift,
                          feedback=lambda fb: self.log(f'forklift: {fb.feedback.phase}'))
        self.forklift.append(h)
        return h

    @staticmethod
    def _judge_forklift(status, result):
        if status == SUCCEEDED and bool(getattr(result, 'success', False)):
            return True, ''
        return False, getattr(result, 'reason', '') or f'status {status}'

    def cancel_forklift(self):
        for h in self.forklift:
            if not h.done():
                h.cancel()
        self.forklift = [h for h in self.forklift if not h.done()]

    def set_carrying(self, on):
        self.carrying = bool(on)
        self.carry_pub.publish(Bool(data=self.carrying))

    # =====================================================================
    # ports: localization
    # =====================================================================
    def _fresh(self, value):
        pose, t = value
        return pose if pose is not None and time.monotonic() - t <= self.pose_max_age else None

    def slam_pose(self):
        return self._fresh(self.slam)

    def slam_localized(self):
        st = self._slam_status() or {}
        flag = str(st.get('localized', '1')).strip().lower() in ('1', 'true', 'yes')
        return flag and self.slam_pose() is not None

    def ekf_status(self):
        state, t = self.ekf_state
        return state if time.monotonic() - t < 3.0 else ''

    def ekf_pose(self):
        return self._fresh(self.ekf) if self.ekf_status() == 'TRACK' else None

    def seed_slam(self, pose):
        msg = PoseWithCovarianceStamped()
        ps = _pose_stamped(self.node, pose)
        msg.header, msg.pose.pose = ps.header, ps.pose
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.05 ** 2
        cov[35] = math.radians(5.0) ** 2
        msg.pose.covariance = cov
        self.seed_pub.publish(msg)
        self.log(f'restock: seeding Planisferio at ({pose[0]:.2f}, {pose[1]:.2f}, '
                 f'{math.degrees(pose[2]):.0f} deg)')

    def ekf_relocalize(self):
        self.reloc_pub.publish(Empty())

    # =====================================================================
    # ports: LiDAR
    # =====================================================================
    def _scan_points(self, max_age=0.5):
        s = self.scan
        if s is None:
            return None
        r = np.asarray(s.ranges, dtype=float)
        a = s.angle_min + s.angle_increment * np.arange(r.size) + self.laser_mount[2]
        ok = np.isfinite(r) & (r > max(s.range_min, 0.02)) & (r < s.range_max)
        x = self.laser_mount[0] + r[ok] * np.cos(a[ok])
        y = self.laser_mount[1] + r[ok] * np.sin(a[ok])
        return np.column_stack([x, y])

    def scan_face(self):
        pts = self._scan_points()
        return None if pts is None else G.face_from_scan(pts)

    def scan_free_direction(self, sector_deg=30.0):
        """(bearing, free distance) of the most open direction around the robot."""
        s = self.scan
        if s is None:
            return None
        r = np.asarray(s.ranges, dtype=float)
        r = np.where(np.isfinite(r), r, s.range_max)
        a = s.angle_min + s.angle_increment * np.arange(r.size) + self.laser_mount[2]
        a = np.arctan2(np.sin(a), np.cos(a))
        best = None
        half = math.radians(sector_deg) / 2
        for c in np.arange(-math.pi, math.pi, math.radians(15.0)):
            sel = np.abs(np.arctan2(np.sin(a - c), np.cos(a - c))) <= half
            if not sel.any():
                continue
            room = float(np.min(r[sel]))
            if best is None or room > best[1]:
                best = (float(c), room)
        return best

    # =====================================================================
    # ports: perception
    # =====================================================================
    def enable(self, stream, on):
        if self.want.get(stream) != bool(on):
            self.want[stream] = bool(on)
            self.enable_pubs[stream].publish(Bool(data=bool(on)))

    def messages(self, stream, since):
        return [(t, d) for t, d in self.buffers[stream] if t >= since]

    def camera(self):
        return self.cam

    def landmarks(self):
        return self.landmarks_data
