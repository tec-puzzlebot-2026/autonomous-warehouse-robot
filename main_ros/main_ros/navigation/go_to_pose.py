#!/usr/bin/env python3
"""Go-to-pose controller on the EKF pose (map frame).

Subscribes: /ekf/pose (PoseWithCovarianceStamped), /ekf/status (String),
            /nav/goal (geometry_msgs/Pose2D)      go to x, y, theta (theta = 100.0: any heading)
            /nav/goal_line (geometry_msgs/Pose2D) straight line ending at x, y with heading theta
                                                  (bay entry: keeps the lateral error small)
            /nav/goto_marker (String "id,distance")  stop `distance` m in front of marker id, facing it
            /nav/cancel (Empty)
Publishes:  /cmd_vel_in (Twist; goes through the relay), /nav/state (String)

Safety: it stops (zero command) unless /ekf/status is TRACK, the pose is fresh (< pose_timeout_s)
and the position sigma is below max_sigma. The relay also stops after 0.5 s without commands.
The controller core (GoToController) has no ROS dependency and is unit-testable.
"""
import math

ANY = 100.0          # theta >= ANY means "do not care about the final heading"


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class GoToController:
    IDLE, TURN, DRIVE, FINAL, DONE = 'IDLE', 'TURN', 'DRIVE', 'FINAL', 'DONE'

    def __init__(self, v_max=0.15, w_max=0.8, v_min=0.06, w_min=0.35, k_v=0.8, k_w=1.5,
                 k_h=2.0, k_stanley=2.0, pos_tol=0.03, ang_tol=0.05, turn_enter=0.6,
                 turn_exit=0.15):
        self.v_max, self.w_max, self.v_min, self.w_min = v_max, w_max, v_min, w_min
        self.k_v, self.k_w, self.k_h, self.k_s = k_v, k_w, k_h, k_stanley
        self.pos_tol, self.ang_tol = pos_tol, ang_tol
        self.turn_enter, self.turn_exit = turn_enter, turn_exit
        self.state = self.IDLE
        self.goal = None
        self.line = False

    def set_goal(self, x, y, theta=ANY, line=False):
        self.goal = (float(x), float(y), float(theta))
        self.line = bool(line) and theta < ANY
        self.state = self.TURN
        self.start = None

    def cancel(self):
        self.state, self.goal = self.IDLE, None

    def _w(self, e):
        w = clamp(self.k_w * e, -self.w_max, self.w_max)
        if abs(w) < self.w_min:                      # motor dead zone: never command less
            w = math.copysign(self.w_min, e)
        return w

    def step(self, pose):
        """pose = (x, y, theta). Returns (v, w)."""
        if self.goal is None or self.state in (self.IDLE, self.DONE):
            return 0.0, 0.0
        x, y, th = pose
        gx, gy, gth = self.goal
        dx, dy = gx - x, gy - y
        d = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        e = wrap(bearing - th)

        if self.state == self.TURN:
            if d <= self.pos_tol:
                self.state = self.FINAL
            elif abs(e) < self.turn_exit:
                self.state = self.DRIVE
            else:
                return 0.0, self._w(e)

        if self.state == self.DRIVE:
            if d <= self.pos_tol or (d < 0.10 and abs(e) > math.pi / 2):     # reached / passed
                self.state = self.FINAL if gth < ANY else self.DONE
                if self.state == self.DONE:
                    return 0.0, 0.0
            elif abs(e) > self.turn_enter and not self.line:
                self.state = self.TURN
                return 0.0, self._w(e)
            else:
                if self.line:
                    ct = -math.sin(gth) * (x - gx) + math.cos(gth) * (y - gy)  # + = left of line
                    e_h = wrap(gth - math.atan(self.k_s * ct) - th)
                    scale = max(0.0, math.cos(e_h))
                else:
                    e_h = e
                    scale = max(0.0, math.cos(e))
                v = clamp(self.k_v * d, self.v_min, self.v_max) * scale
                if scale > 0.0:
                    v = max(v, self.v_min * 0.5)
                w = clamp(self.k_h * e_h, -self.w_max, self.w_max)
                return v, w

        if self.state == self.FINAL:
            eh = wrap(gth - th)
            if abs(eh) <= self.ang_tol:
                self.state = self.DONE
                return 0.0, 0.0
            return 0.0, self._w(eh)
        return 0.0, 0.0


# --------------------------------------------------------------------------- ROS node
def load_markers(map_file):
    import yaml
    with open(map_file) as f:
        p = yaml.safe_load(f)['ekf_localization']['ros__parameters']
    ids, xs, ys, yaws = p['landmark_ids'], p['landmark_x'], p['landmark_y'], p['landmark_yaw']
    return {int(i): (float(x), float(y), float(w)) for i, x, y, w in zip(ids, xs, ys, yaws)}


def main(args=None):
    import rclpy
    from geometry_msgs.msg import Pose2D, PoseWithCovarianceStamped, Twist
    from rclpy.node import Node
    from std_msgs.msg import Empty, String

    class GoToNode(Node):
        def __init__(self):
            super().__init__('go_to_pose')
            dp = self.declare_parameter
            self.ctl = GoToController(
                v_max=float(dp('v_max', 0.15).value), w_max=float(dp('w_max', 0.8).value),
                v_min=float(dp('v_min', 0.06).value), w_min=float(dp('w_min', 0.35).value),
                pos_tol=float(dp('pos_tol', 0.03).value), ang_tol=float(dp('ang_tol', 0.05).value))
            self.max_sigma = float(dp('max_sigma', 0.25).value)
            self.max_sigma_th = float(dp('max_sigma_theta', 0.12).value)     # rad (about 7 deg)
            self.pose_timeout = float(dp('pose_timeout_s', 0.6).value)
            self.out_topic = dp('cmd_vel_topic', '/cmd_vel_in').value
            map_file = dp('map_file', '').value
            self.markers = load_markers(map_file) if map_file else {}
            self.pose, self.pose_t, self.sigma, self.sigma_th, self.status = None, 0.0, 9.9, 9.9, ''
            self.pub = self.create_publisher(Twist, self.out_topic, 10)
            self.pub_state = self.create_publisher(String, '/nav/state', 10)
            self.create_subscription(PoseWithCovarianceStamped, '/ekf/pose', self.on_pose, 10)
            self.create_subscription(String, '/ekf/status', self.on_status, 10)
            self.create_subscription(Pose2D, '/nav/goal', lambda m: self.goal(m, False), 10)
            self.create_subscription(Pose2D, '/nav/goal_line', lambda m: self.goal(m, True), 10)
            self.create_subscription(String, '/nav/goto_marker', self.on_marker, 10)
            self.create_subscription(Empty, '/nav/cancel', lambda _m: self.stop('cancelled'), 10)
            self.create_timer(0.1, self.on_timer)
            self.get_logger().info('go_to_pose ready (%d markers), commands on %s' % (
                len(self.markers), self.out_topic))

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_status(self, m):
            self.status = m.data

        def on_pose(self, m):
            q = m.pose.pose.orientation
            th = 2.0 * math.atan2(q.z, q.w)
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, wrap(th))
            c = m.pose.covariance
            self.sigma = math.sqrt(max(c[0], c[7]))
            self.sigma_th = math.sqrt(max(c[35], 0.0))
            self.pose_t = self.now()

        def goal(self, m, line):
            self.ctl.set_goal(m.x, m.y, m.theta, line)
            self.get_logger().info('Goal (%.3f, %.3f, %s)%s' % (
                m.x, m.y, 'any' if m.theta >= ANY else '%.1f deg' % math.degrees(m.theta),
                ' straight line' if self.ctl.line else ''))

        def on_marker(self, m):
            try:
                i, d = m.data.split(',')
                mx, my, psi = self.markers[int(i)]
                d = float(d)
            except (ValueError, KeyError):
                self.get_logger().error('goto_marker needs "id,distance" with a known id')
                return
            gx, gy = mx + d * math.cos(psi), my + d * math.sin(psi)
            self.ctl.set_goal(gx, gy, wrap(psi + math.pi), False)
            self.get_logger().info('Goal %.2f m in front of marker %s: (%.3f, %.3f)' % (d, i, gx, gy))

        def stop(self, why):
            self.ctl.cancel()
            self.pub.publish(Twist())
            self.get_logger().warn('Stopped: %s' % why)

        def on_timer(self):
            self.pub_state.publish(String(data=self.ctl.state))
            if self.ctl.state in (GoToController.IDLE, GoToController.DONE):
                return
            if self.status != 'TRACK' or self.pose is None:
                self.pub.publish(Twist()); self.log_once('EKF not tracking'); return
            if self.now() - self.pose_t > self.pose_timeout:
                self.pub.publish(Twist()); self.log_once('pose is stale'); return
            if self.sigma > self.max_sigma:
                self.pub.publish(Twist()); self.log_once('position sigma %.2f m too large' % self.sigma); return
            if self.sigma_th > self.max_sigma_th:
                self.pub.publish(Twist())
                self.log_once('heading sigma %.1f deg too large: drive a little / use fixed mode'
                              % math.degrees(self.sigma_th))
                return
            v, w = self.ctl.step(self.pose)
            t = Twist()
            t.linear.x, t.angular.z = float(v), float(w)
            self.pub.publish(t)
            if self.ctl.state == GoToController.DONE:
                self.pub.publish(Twist())
                self.get_logger().info('Goal reached at (%.3f, %.3f, %.1f deg)' % (
                    self.pose[0], self.pose[1], math.degrees(self.pose[2])))

        _last = ''

        def log_once(self, msg):
            if msg != self._last:
                self.get_logger().warn('Paused: %s' % msg)
                self._last = msg

    rclpy.init(args=args)
    node = GoToNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub.publish(Twist())
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()