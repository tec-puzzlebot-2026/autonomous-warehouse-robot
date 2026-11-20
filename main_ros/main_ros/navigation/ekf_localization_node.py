"""Localization EKF with an ArUco map (Algorithm 6.1) to test Q_k and R_k in the arena.

Subscribes: /VelocityEncR, /VelocityEncL (std_msgs/Float32, rad/s)  -> prediction
            /aruco/detections (JSON from aruco_detector)             -> correction / start-up
            /ekf/checkpoint (geometry_msgs/Pose2D)  real pose measured in the map
            /ekf/relocalize (std_msgs/Empty)        forget the pose and localize again
Publishes:  /ekf/pose             PoseWithCovarianceStamped (map frame; only while tracking)
            /ekf/dead_reckoning   the same prediction WITHOUT corrections, for comparison
            /ekf/status           String: SEARCH | COLLECT | TRACK | FAILED (1 Hz)
            /cmd_vel              Twist, ONLY the search rotation (search_rotation: true)

Start-up (localization_mode):
  fixed  the filter starts at `initial_pose` (robot placed on a known mark).
  auto   the robot may start ANYWHERE in the arena: the pose is computed from the camera and
         the map (global_localization.py), optionally rotating in place to find markers; the
         EKF starts when the solution passes the chi-square consistency test. If the filter
         loses the robot (lost_rejects consecutive rejected corrections) it localizes again.

Measurement: the ArUco center in the optical frame is moved to base_link with the
calibrated mount X (camera_mount of the config) and converted to z = [ρ, α]; so g(m_i, s_k)
is exactly the course model. R_k = diag(σρ², σα²) comes from error_model.yaml
(depends on distance and view angle) plus the optional landmark_sigma / latency_s terms
(see ekf_core.inflate_R); Q_k comes from motion_model.yaml.

Log in <calib_dir>/ekf/ekf_<date>.csv for analyze_ekf_consistency:
  init       start-up / re-localization solution (state, source, n_markers, cost)
  update     each correction: innovation, NIS, accepted, distance, angle, Σ
  pose       at 5 Hz: EKF and dead reckoning with their σ
  checkpoint when you publish the real pose: error and NEES of EKF and dead reckoning
"""
import csv
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D, PoseWithCovarianceStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Empty, Float32, String

from main_ros.navigation.ekf_core import EKF, inflate_R, range_bearing_from_base
from main_ros.navigation.global_localization import GlobalLocalizer, StartupLocalizer
from main_ros.vision.aruco import experiment_io as xio
from main_ros.vision.aruco.paths import MotionModel, wrap, motion_dir
from main_ros.vision.aruco import error_model as em
from main_ros.vision.aruco import handeye as he
from main_ros.vision.aruco.paths import DEFAULT_CALIB_DIR, expand

LOG_COLUMNS = ['t', 'kind', 'id', 'z_rho', 'z_alpha', 'zhat_rho', 'zhat_alpha', 'nis',
               'accepted', 'distance', 'view_angle_deg', 'sigma_rho', 'sigma_alpha',
               'x', 'y', 'theta', 'sx', 'sy', 'stheta',
               'dr_x', 'dr_y', 'dr_theta', 'dr_sx', 'dr_sy', 'dr_stheta',
               'gt_x', 'gt_y', 'gt_theta', 'nees', 'dr_nees',
               'state', 'source', 'n_markers', 'cost']


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def nees(mu, Sigma, gt):
    e = np.array([gt[0] - mu[0], gt[1] - mu[1], wrap(gt[2] - mu[2])])
    try:
        return float(e @ np.linalg.solve(Sigma, e)), e
    except np.linalg.LinAlgError:
        return float('nan'), e


UNKNOWN_YAW = 100.0      # landmark_yaw entries >= this mean "facing direction not measured"


class EkfNode(Node):

    def __init__(self):
        super().__init__('ekf_localization')
        dp = self.declare_parameter
        calib = expand(dp('calib_dir', DEFAULT_CALIB_DIR).value)
        mm_file = dp('motion_model_file', str(motion_dir() / 'motion_model.yaml')).value
        em_file = dp('error_model_file', str(calib / 'aruco' / 'error_model.yaml')).value
        right_topic = dp('enc_right_topic', '/VelocityEncR').value
        left_topic = dp('enc_left_topic', '/VelocityEncL').value
        self.sign_R = float(dp('enc_right_sign', 1.0).value)
        self.sign_L = float(dp('enc_left_sign', 1.0).value)
        r = float(dp('wheel_radius', 0.05).value)
        L = float(dp('wheelbase', 0.19).value)
        self.frame = dp('frame_id', 'map').value
        ids = list(dp('landmark_ids', [21]).value)
        xs = list(dp('landmark_x', [0.0]).value)
        ys = list(dp('landmark_y', [0.0]).value)
        yaws = list(dp('landmark_yaw', [999.0]).value)
        mu0 = list(dp('initial_pose', [0.0, 0.0, 0.0]).value)
        s0 = list(dp('initial_sigma', [0.01, 0.01, 0.02]).value)
        self.apply_bias = dp('apply_bias_correction', True).value
        # Extra R terms the test grid does not contain (see ekf_core.inflate_R); 0 = off
        self.landmark_sigma = float(dp('landmark_sigma', 0.0).value)   # m, map survey error
        self.latency_s = float(dp('latency_s', 0.0).value)             # s, image -> filter delay
        # fallback if there is no error_model.yaml yet
        self.fb_rho = (float(dp('fallback_sigma_rho_c0', 0.01).value),
                       float(dp('fallback_sigma_rho_c2', 0.02).value))
        self.fb_alpha = math.radians(float(dp('fallback_sigma_alpha_deg', 1.0).value))
        mount = [float(v) for v in dp('camera_mount', [0.08, 0.0, 0.035, 0.0, 0.0, 0.0]).value]
        # ---- start-up / global localization
        self.mode = str(dp('localization_mode', 'fixed').value).lower()
        if self.mode not in ('fixed', 'auto'):
            raise ValueError("localization_mode must be 'fixed' or 'auto'")
        bounds = [float(v) for v in dp('arena_bounds', [0.0, 0.0, 0.0, 0.0]).value]
        rotate = bool(dp('search_rotation', False).value)
        self.cmd_topic = dp('cmd_vel_topic', '/cmd_vel').value
        aruco_cfg = dp('aruco_config', '').value
        n_frames = int(dp('init_frames', 30).value)
        r_scale = float(dp('init_r_scale', 2.0).value)
        allow_single = bool(dp('init_allow_single', True).value)
        max_sigma = float(dp('init_max_sigma', 0.3).value)
        settle_s = float(dp('settle_s', 0.6).value)
        lost_rejects = int(dp('lost_rejects', 8).value)
        search_speed = float(dp('search_speed', 0.3).value)
        search_timeout = float(dp('search_timeout_s', 60.0).value)

        if not (len(ids) == len(xs) == len(ys)):
            raise ValueError('landmark_ids, landmark_x and landmark_y must have the same length')
        self.landmarks = {int(i): (float(x), float(y)) for i, x, y in zip(ids, xs, ys)}
        self.X = he.mount_to_X(*mount)

        # facing direction of each marker in the map (rad), only needed for single-marker starts
        psi = {}
        if len(yaws) == len(ids):
            psi = {int(i): float(y) for i, y in zip(ids, yaws) if float(y) < UNKNOWN_YAW}
        elif not (len(yaws) == 1 and float(yaws[0]) >= UNKNOWN_YAW):
            raise ValueError('landmark_yaw must have one value per landmark_ids '
                             '(or a single 999.0 = unknown)')
        # height of each marker center above base_link (heights are per ID in aruco_localization.yaml)
        try:
            exp = xio.load_config(aruco_cfg or None)['experiment']
        except (OSError, ValueError, KeyError, AttributeError) as e:
            exp = {}
            self.get_logger().warn('No aruco_config (%s): default marker height for all IDs' % e)
        rel_h = {i: xio.marker_z(exp, i) for i in self.landmarks}

        try:
            self.mm = MotionModel.load(mm_file)
            self.get_logger().info('Motion model: %s' % mm_file)
        except (OSError, ValueError) as e:
            self.mm = MotionModel(r, L, (0.05, 0.05), (0.02, 0.02))
            self.get_logger().warn('No motion_model.yaml (%s): using the fallback σ = 0.05 + 0.02·|ω|' % e)
        try:
            self.err = em.load(em_file)
            self.get_logger().info('Camera error model: %s' % em_file)
        except OSError as e:
            self.err = None
            self.get_logger().warn('No error_model.yaml (%s): using the fallback R' % e)

        P0 = np.diag([v ** 2 for v in s0])
        self.ekf = EKF(self.mm, mu0, P0)
        self.dr = EKF(self.mm, mu0, P0, gate=None)          # prediction only
        self.last_L = None
        self.last_L_t = None
        self.last_wR = None
        self.last_pred_t = None
        self.yaw_odo = 0.0                                   # integrated odometry heading (rad)

        bias = (self.err or {}).get('bias', {}) if self.apply_bias else {}
        self.gl = GlobalLocalizer(
            self.landmarks, psi, rel_h, self.X,
            rb_cov_fn=self.measurement_R,
            pose_cov_fn=((lambda d, v, a: em.measurement_covariance(self.err, d, v)
                          if em.is_valid(self.err, d, v, a) else None)
                         if self.err is not None else None),
            valid_fn=((lambda d, v: em.is_valid(self.err, d, v)) if self.err is not None else None),
            bias=bias, bounds=bounds, n_frames=n_frames, r_scale=r_scale,
            allow_single=allow_single, max_sigma=max_sigma)
        self.startup = StartupLocalizer(
            self.gl, rotate=rotate, search_speed=search_speed, search_timeout_s=search_timeout,
            settle_s=settle_s, lost_rejects=lost_rejects if self.mode == 'auto' else 0)
        self.rotate = rotate
        self.failed_reported = False
        if self.mode == 'fixed':
            self.startup.state = self.startup.TRACK
        else:
            self.startup.start(self.now(), self.yaw_odo)
        if self.mode == 'auto' and len(self.landmarks) < 2 and not psi:
            self.get_logger().warn('auto mode with one marker needs landmark_yaw for it')

        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.log_path = calib / 'ekf' / ('ekf_%s.csv' % stamp)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_f = open(str(self.log_path), 'w', newline='')
        self.log = csv.DictWriter(self.log_f, fieldnames=LOG_COLUMNS, extrasaction='ignore')
        self.log.writeheader()

        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/ekf/pose', 10)
        self.pub_dr = self.create_publisher(PoseWithCovarianceStamped, '/ekf/dead_reckoning', 10)
        self.pub_status = self.create_publisher(String, '/ekf/status', 10)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, 10) if rotate else None
        self.create_subscription(Float32, right_topic, self.on_R, qos_profile_sensor_data)
        self.create_subscription(Float32, left_topic, self.on_L, qos_profile_sensor_data)
        self.create_subscription(String, '/aruco/detections', self.on_det, 10)
        self.create_subscription(Pose2D, '/ekf/checkpoint', self.on_checkpoint, 10)
        self.create_subscription(Empty, '/ekf/relocalize', self.on_relocalize, 10)
        self.create_timer(0.2, self.on_timer)
        self.create_timer(0.1, self.on_control)
        self.get_logger().info('EKF ready (%s mode): %d landmarks, log in %s' % (
            self.mode, len(self.landmarks), self.log_path))
        if self.mode == 'auto':
            self.get_logger().info('Searching for the robot pose%s' % (
                ': the robot will rotate in place (/cmd_vel)' if rotate else
                ': robot not moving, it must see >= 1 ArUco'))

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tracking(self):
        return self.startup.tracking()

    # ------------------------------------------------------------ prediction
    def on_L(self, msg):
        self.last_L = self.sign_L * float(msg.data)
        self.last_L_t = self.now()

    def on_R(self, msg):
        t = self.now()
        wR = self.sign_R * float(msg.data)
        self.last_wR = wR
        if self.last_L is None or t - self.last_L_t > 0.3:
            return
        if self.last_pred_t is not None:
            dt = t - self.last_pred_t
            if 0 < dt < 0.5:
                _, w = self.mm.wheels_to_twist(wR, self.last_L)
                self.yaw_odo += w * dt                   # integrates always (also while searching)
                if self.tracking():
                    self.ekf.predict(wR, self.last_L, dt)
                    self.dr.predict(wR, self.last_L, dt)
        self.last_pred_t = t

    def stationary(self):
        """True if the wheels are (almost) still. Without encoder data: only if not rotating."""
        if self.last_L is None or self.last_wR is None:
            return not self.rotate
        v, w = self.mm.wheels_to_twist(self.last_wR, self.last_L)
        return abs(w) < 0.03 and abs(v) < 0.02

    # ------------------------------------------------------------ correction
    def measurement_R(self, distance, view):
        if self.err is None:
            s_rho = self.fb_rho[0] + self.fb_rho[1] * distance ** 2
            return np.diag([s_rho ** 2, self.fb_alpha ** 2])
        return em.range_bearing_covariance(self.err, distance, view)

    def on_det(self, msg):
        try:
            dets = json.loads(msg.data)
        except ValueError:
            return
        if not self.tracking():
            self.on_det_startup(dets)
            return
        for d in dets:
            lm = self.landmarks.get(int(d.get('id', -1)))
            if lm is None:
                continue
            dist = d.get('distance')
            view = d.get('view_angle_deg')
            if dist is None or view is None:
                continue
            if self.err is not None and not em.is_valid(self.err, dist, view):
                continue                      # outside the range where the error was measured
            p_cam = np.array(list(d['t']) + [1.0])
            z = range_bearing_from_base(self.X @ p_cam)
            if self.apply_bias and self.err is not None:
                z[0] -= self.err['bias'].get('range', 0.0)
                z[1] = wrap(z[1] - self.err['bias'].get('bearing', 0.0))
            R = self.measurement_R(dist, view)
            v = w = 0.0
            if self.last_L is not None and self.last_wR is not None:
                v, w = self.mm.wheels_to_twist(self.last_wR, self.last_L)
            R = inflate_R(R, z[0], v, w, self.landmark_sigma, self.latency_s)
            try:
                res = self.ekf.update(z, lm, R)
            except ValueError:
                continue
            zhat = np.array([z[0] - res['nu'][0], wrap(z[1] - res['nu'][1])])
            self.write('update', id=d['id'], z_rho=z[0], z_alpha=z[1], zhat_rho=zhat[0],
                       zhat_alpha=zhat[1], nis=res['nis'], accepted=int(res['accepted']),
                       distance=dist, view_angle_deg=view, sigma_rho=math.sqrt(R[0, 0]),
                       sigma_alpha=math.sqrt(R[1, 1]))
            if self.startup.on_update(self.now(), self.yaw_odo, bool(res['accepted'])):
                self.get_logger().warn('Robot lost (%d rejected corrections in a row): '
                                       'localizing again' % self.startup.lost_rejects)
                self.write('lost', state=self.startup.state)
                self.failed_reported = False
                return

    def on_det_startup(self, dets):
        """No pose yet: feed the global localizer; start the EKF when a solution passes."""
        res = self.startup.on_detections(self.now(), dets, self.yaw_odo, self.stationary())
        if res is None:
            return
        if res['status'] != 'ok':
            self.get_logger().info('Start-up: %s with %d marker(s) (cost %.1f) -> %s' % (
                res['status'], res['n'], res['cost'] if res['cost'] == res['cost'] else -1.0,
                self.startup.state))
            return
        pose = np.array(res['pose'], dtype=float)
        P = np.asarray(res['cov'], dtype=float)
        for f in (self.ekf, self.dr):
            f.mu = pose.copy()
            f.Sigma = P.copy()
        self.last_pred_t = None
        self.write('init', state=self.startup.state, source=res['source'], n_markers=res['n'],
                   cost=res['cost'])
        self.log_f.flush()
        self.get_logger().info(
            'Start-up OK (%s): x = %.3f m, y = %.3f m, θ = %.1f°  (σ %.1f cm, %.1f°)' % (
                res['source'], pose[0], pose[1], math.degrees(pose[2]),
                100 * math.sqrt(P[0, 0]), math.degrees(math.sqrt(P[2, 2]))))

    def on_relocalize(self, _msg):
        self.get_logger().info('Relocalization requested')
        self.startup.start(self.now(), self.yaw_odo)
        self.failed_reported = False

    def on_control(self):
        """10 Hz: search timeouts and the rotation command (only with search_rotation)."""
        now = self.now()
        self.startup.tick(now, self.yaw_odo)
        if self.startup.state == self.startup.FAILED and not self.failed_reported:
            self.failed_reported = True
            self.get_logger().error(
                'Start-up FAILED: no consistent pose. Check that markers are visible from here, '
                'landmark_x/y, marker sizes, and that valid.max_distance covers the distance.')
            self.write('init', state=self.startup.state, source='failed')
        w = self.startup.command()
        if w is not None and self.pub_cmd is not None:
            msg = Twist()
            msg.angular.z = float(w)
            self.pub_cmd.publish(msg)

    # --------------------------------------------------------------- output
    def write(self, kind, **kw):
        row = {'t': '%.3f' % self.now(), 'kind': kind}
        mu, S = self.ekf.mu, self.ekf.Sigma
        dmu, dS = self.dr.mu, self.dr.Sigma
        row.update({'x': mu[0], 'y': mu[1], 'theta': mu[2],
                    'sx': math.sqrt(max(S[0, 0], 0)), 'sy': math.sqrt(max(S[1, 1], 0)),
                    'stheta': math.sqrt(max(S[2, 2], 0)),
                    'dr_x': dmu[0], 'dr_y': dmu[1], 'dr_theta': dmu[2],
                    'dr_sx': math.sqrt(max(dS[0, 0], 0)), 'dr_sy': math.sqrt(max(dS[1, 1], 0)),
                    'dr_stheta': math.sqrt(max(dS[2, 2], 0))})
        row.update(kw)
        self.log.writerow(row)

    def to_msg(self, f):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.pose.pose.position.x = float(f.mu[0])
        msg.pose.pose.position.y = float(f.mu[1])
        q = yaw_to_quat(f.mu[2])
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        cov = [0.0] * 36
        idx = (0, 1, 5)                          # x, y, yaw in the ROS 6x6 matrix
        for i in range(3):
            for j in range(3):
                cov[idx[i] * 6 + idx[j]] = float(f.Sigma[i, j])
        msg.pose.covariance = cov
        return msg

    def on_timer(self):
        self.pub_status.publish(String(data=self.startup.state))
        if not self.tracking():
            return                               # no valid pose: publish nothing
        self.pub.publish(self.to_msg(self.ekf))
        self.pub_dr.publish(self.to_msg(self.dr))
        self.write('pose')
        self.log_f.flush()

    def on_checkpoint(self, msg):
        gt = (msg.x, msg.y, msg.theta)
        n_ekf, e = nees(self.ekf.mu, self.ekf.Sigma, gt)
        n_dr, e_dr = nees(self.dr.mu, self.dr.Sigma, gt)
        self.write('checkpoint', gt_x=gt[0], gt_y=gt[1], gt_theta=gt[2], nees=n_ekf,
                   dr_nees=n_dr, state=self.startup.state)
        self.log_f.flush()
        self.get_logger().info(
            'Checkpoint: EKF error = (%+.3f m, %+.3f m, %+.1f°) NEES %.1f | dead reckoning = '
            '(%+.3f m, %+.3f m, %+.1f°) NEES %.1f' % (
                e[0], e[1], math.degrees(e[2]), n_ekf, e_dr[0], e_dr[1],
                math.degrees(e_dr[2]), n_dr))

    def destroy_node(self):
        try:
            if self.pub_cmd is not None:
                self.pub_cmd.publish(Twist())    # never leave the robot rotating
        except Exception:  # noqa: BLE001
            pass
        try:
            self.log_f.close()
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EkfNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

