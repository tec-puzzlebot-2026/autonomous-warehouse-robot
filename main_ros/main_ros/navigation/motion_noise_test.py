"""Motion tests to measure noise for the prediction model.

Speed sweep(encoder noise vs. commanded speed):
    ros2 run main_ros motion_noise_test sweep
    ros2 run main_ros motion_noise_test sweep --only turn       # only in-place rotations
Validation against a tape measure (consistency of Q and calibration of r and L):
    ros2 run main_ros motion_noise_test straight --distance 1.0 --speed 0.15
    ros2 run main_ros motion_noise_test turn --angle 360 --speed 1.0

Publishes to /cmd_vel (geometry_msgs/Twist) and records /cmd_vel and the encoder data
(/VelocityEncR, /VelocityEncL; std_msgs/Float32 in rad/s) to a CSV file located in
<calib_dir>/motion/. 

You can also record a rosbag while the test is running:
    ros2 bag record -o sweep_01/cmd_vel/VelocityEncR/VelocityEncL

Safety: Pressing Ctrl+C stops the robot (by pubishing a zero-velocity command multiple times).
For straight-line segments, the robot moves out and back (+v followed by -v) so that it returns to
its strarting point; the duration of each segment is at most max_distance/|v|.
"""
import argparse
import csv
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32

from main_ros.navigation.motion_model import MotionModel, wrap
from main_ros.navigation.motion_config import RUNS_COLUMNS, load_config, motion_dir

class Runner(Node):
    def __init__(self, robot_cfg):
        super().__init__('motion_noise_test')
        rc = robot_cfg
        self.sign_R = float(rc.get('enc_right_sign', 1.0))
        self.sign_L = float(rc.get('enc_left_sign', 1.0))
        self.pub = self.create_publisher(Twist, rc.get('cmd_vel_topic', '/cmd_vel'), 10)
        self.create_subscription(Float32, rc.get('enc_right_topic', '/VelocityEncR'), self.on_R, qos_profile_sensor_data)
        self.create_subscription(Float32, rc.get('enc_left_topic', '/VelocityEncL'), self.on_L, qos_profile_sensor_data)
        self.rows = []
        self.last_R = self.last_L = None
        self.n_R = self.n_L = 0

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_R(self, msg):
        self.last_R = self.sign_R * float(msg.data)
        self.n_R += 1
        self.rows_append(self.now(), 'encR', self.last_R, '')

    def on_L(self, msg):
        self.last_L = self.sign_L * float(msg.data)
        self.n_L += 1
        self.rows_append(self.now(), 'encL', self.last_L, '')

    def send(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)
        self.rows.append(self.now(), 'cmd', float(v), float(w))

    def hold(self, v, w, seconds, rate=20.0, stop_fn=None):
        """Publishes (v, w) at 'rate' Hz during 'seconds' or until stop_fn() is True."""
        end = time.monotonic() + seconds
        period = 1.0 / rate
        nxt = time.monotonic()
        while time.monotonic() < end and rclpy.ok():
            if time.monotonic() >= nxt:
                self.send(v, w)
                nxt += period
            rclpy.spin_once(self, timeout_sec=0.005)
            if stop_fn is not None and stop_fn():
                return True
        return False

    def stop(self, seconds=1.0):
        self.hold(0.0, 0.0, seconds)

    def wait_encoders(self, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end and (self.n_R == 0 or self.n_L == 0):
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.n_R > 0 and self.n_L > 0
    
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'topic', 'a', 'b'])
            w.writerows(self.rows)
        return path

def build_plan(test, only=None):
    """(description, v, w, seconds) list."""
    seg_time = float(test.get('segment_time', 15.0))
    max_dist = float(test.get('max_distance'), 1.5)
    repeats = int(test.get('repeats', 1))
    plan = []
    if only in (None, 'straight'):
        for v in test.get('straight_speeds', []):
            dur = min(seg_time, max_dist / max(abs(v), 1e-6))
            for _ in range(repeats):
                plan.append(('straight forward v=%.2f m/s' % v, v, 0.0, dur))
                plan.append(('straight turn v=%.2f m/s' % v, -v, 0.0, dur))
    if only in (None, 'turn'):
        for w in test.get('turn_speeds', []):
            for _ in range(repeats):
                plan.append(('counterclockwise turn w=%.2f rad/s' % w, 0.0, w, seg_time))
                plan.append(('clockwise turn w=%.2f rad/s' % w, 0.0, -w, seg_time))
    return plan

def run_sweep(node, test, only, confirm):
    plan = build_plan(test, only)
    pause = float(test.get('pause_time', 2.0))
    total = sum(p[3] + pause for p in plan)
    print('Sweep: %d sections, ~%.1f min. Let ~%.1f m free in front and back.' % (len(plan), total / 60.0, float(test.get('max_distance', 1.5))))
    for i, (desc, v, w, dur) in enumerate(plan):
        if confirm:
            input(['[%d/%d] %s, %.1f s. Enter to start (Ctrl+C abort)' % (i + 1, len(plan), desc, dur)])
        else:
            print('[%d/%d] % s, %.1f s' % (i + 1, len(plan), desc, dur))
        node.hold(v, w, dur)
        node.stop(pause)

def run_validation(node, model, kind, target, speed):
    """Advance or turn until the odometry says: 'target'; returns the odometry pose."""
    s = [0.0, 0.0, 0.0]
    state = {'t': None, 'dist': 0.0, 'turn': 0.0}

    def integrate():
        t = node.now()
        if node.last_R is None or node.last_L is None:
            return False
        if state['t'] is not None:
            dt = t - state['t']
            if 0 < dt < 0.5:
                v, w = model.wheels_to_twist(node.last_R, node.last_L)
                s[0] += dt * v * math.cos(s[2])
                s[1] += dt * v * math.sin(s[2])
                s[2] = wrap(s[2] + dt * w)
                state['dist'] += dt * v
                state['turn'] += dt * w
        state['t'] = t
        if kind == 'straight':
            return abs(state['dist']) >= target
        return abs(state['turn']) >= math.radians(target)

    if kind == 'straight':
        v, w = speed, 0.0
        timeout = 3.0 * target / max(abs(speed), 1e-3) + 5.0
    else:
        v, w = 0.0, speed
        timeout = 3.0 * math.radians(target) / max(abs(speed), 1e-3) + 5.0
    reached = node.hold(v, w, timeout, rate=50.0, stop_fn=integrate)
    node.stop(1.5)      # Keeps integrating braking in CSV
    if not reached:
        print('WARNING: The objective couldn\'t be reached in %.0f s; revise the encoders and signs.' % timeout)
    return s, state


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['sweep', 'straight', 'turn'])
    p.add_argument('--only', choices=['straight', 'turn'], default=None)
    p.add_argument('--yes', action='store_true', help='do not press Enter before each segment')
    p.add_argument('--distance', type=float, default=1.0, help='m (straight)')
    p.add_argument('--angle', type=float, default=360.0, help='degrees (turn)')
    p.add_argument('--speed', type=float, default=None, help='m/s or rad/s')
    p.add_argument('--config', default=None)
    p.add_argument('--note', default='')
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    model = MotionModel(cfg['robot'].get('wheel_radius', 0.05), cfg['robot'].get('wheelbase', 0.19))
    out_dir = motion_dir()
    stamp = time.strftime('%Y%m%d_%H%M%S')

    rclpy.init()
    node = Runner(cfg['robot'])
    try:
        if not node.wait_encoders():
            sys.exit('No data is being received from the encoders (%s, %s). Is the Hackerboard connected? '
                    'and running micro-ROS?' % (cfg['robot'].get('enc_right_topic'),
                                                cfg['robot'].get('enc_left_topic')))
        if a.mode == 'sweep':
            run_sweep(node, cfg['test'], a.only, confirm=not a.yes)
            path = node.save(out_dir / ('sweep_%s.csv' % stamp))
            print('\nSaved %s \nAnalyze with: ros2 run main_ros analyze_motion_noise' % path)
            return
        speed = a.speed if a.speed is not None else (0.15 if a.mode == 'straight' else 1.0)
        target = a.distance if a.mode == 'straight' else a.angle
        input('Place the robot at the starting mark (axis over the origin, facing +x). '
            'Press Enter to start ')
        s, st = run_validation(node, model, a.mode, target, speed)
        runs_csv = out_dir / 'runs.csv'
        try:
            with open(str(runs_csv), newline='') as f:
                n_run = sum(1 for _ in csv.DictReader(f))
        except FileNotFoundError:
            n_run = 0
        path = node.save(out_dir / 'runs' / ('run_%03d.csv' % n_run))
        new = not runs_csv.is_file()
        with open(str(runs_csv), 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
            if new:
                w.writeheader()
            w.writerow({'run': n_run, 'type': a.mode, 'target': target, 'speed': speed,
                        'csv': path.name, 'odom_x': '%.4f' % s[0], 'odom_y': '%.4f' % s[1],
                        'odom_theta_deg': '%.2f' % math.degrees(s[2]),
                        'odom_dist': '%.4f' % st['dist'],
                        'odom_turn_deg': '%.2f' % math.degrees(st['turn']),
                        'real_x': '', 'real_y': '', 'real_theta_deg': '', 'note': a.note})
        print('\nRun %d: odometry x=%.3f m, y=%.3f m, θ=%.1f°' % (
            n_run, s[0], s[1], math.degrees(s[2])))
        print('Measure the ACTUAL pose of the wheel axle and record it in %s (columns real_x, '
        'real_y, real_theta_deg).' % runs_csv)
    except KeyboardInterrupt:
        print('\nAborted.')
    finally:
        try:
            for _ in range(5):
                node.send(0.0, 0.0)
                time.sleep(0.02)
        except Exception:
            pass
        if a.mode == 'sweep' and node.rows and not (out_dir / ('sweep_%s.csv' % stamp)).exists():
            print('Partial data in %s' % node.save(out_dir / ('sweep_%s_partial.csv' % stamp)))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()