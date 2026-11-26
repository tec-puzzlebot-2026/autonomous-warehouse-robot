#!/usr/bin/env python3
"""
forklift_bridge.py — PickBox / SetDownBox (our actions) served by control_forklift
==================================================================================
The Program Manager (restock_core) asks the forklift for two things, through the
actions in warehouse_msgs:

  PickBox     the robot stands about 10 cm from the box (forks toward it): pick it.
  SetDownBox  the robot carries a box: put it down where the robot stands.

The forklift teammate's control_forklift does the pick: `forklift_pick_back` is one
supervised Pick & Back per process (camera + LiDAR/IR, align, insert the forks, lift,
back out to the start pose), on top of `forklift_operate` (the real Tang board) or
`sim_forklift` (Isaac). This bridge:

  PickBox     starts `forklift_pick_back --level floor|first --execute` for the goal,
              relays its drive commands to the base, reports its phases as feedback,
              and answers success on "Done", failure on "Error: <reason>".
              Level 2 (the 200 mm shelf) has no automatic pick yet: it fails cleanly.
  SetDownBox  lowers the forks until the box rests on its runners
              (set_down_height_mm), and waits for ARRIVED. The base does not move:
              the Program Manager backs away afterwards (the forks slide out).

pick_back refuses to drive on a topic that anyone else publishes on, and our base
topic always has other publishers (Sendero, the Program Manager). So pick_back drives
on a topic nobody else uses (cmd_vel_topic in the pick config: /cmd_vel_in in the sim,
where the base listens on /cmd_vel), and while a pick runs the bridge copies those
commands to the base (base_cmd_topic); otherwise it is silent.

Parameters:
  pick_config         file name in share/main_ros/config/forklift/ (or a full path)
  base_cmd_topic      /cmd_vel in the sim, /cmd_vel_in on the robot (the relay)
  set_down_height_mm  lift height that leaves the box on its runners (default 5)
  pick_timeout_s      give up on a pick after this long (default 240, wall time)
"""
# Author: Raymundo Gomez

import json
import os
import signal
import subprocess
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64, String
from warehouse_msgs.action import PickBox, SetDownBox

LEVELS = {PickBox.Goal.LEVEL_FLOOR: 'floor', PickBox.Goal.LEVEL_1: 'first'}


def config_path(name):
    if os.path.isabs(name):
        return name
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('main_ros'), 'config', 'forklift', name)


class ForkliftBridge(Node):
    def __init__(self):
        super().__init__('forklift_bridge')
        p = self.declare_parameter
        self.cfg_file = config_path(p('pick_config', 'pick_sim.json').value)
        self.base_topic = p('base_cmd_topic', '/cmd_vel').value
        self.set_down_mm = float(p('set_down_height_mm', 5.0).value)
        self.pick_timeout = float(p('pick_timeout_s', 240.0).value)
        self.set_down_timeout = float(p('set_down_timeout_s', 30.0).value)
        with open(self.cfg_file) as f:
            self.pick_cfg = json.load(f)
        self.pick_cmd_topic = self.pick_cfg.get('cmd_vel_topic', '/cmd_vel_in')
        if self.pick_cmd_topic == self.base_topic:
            raise ValueError(f'{self.cfg_file}: cmd_vel_topic must differ from {self.base_topic} '
                             '(pick_back refuses a topic with other publishers)')
        self.sim_time = bool(self.get_parameter('use_sim_time').value)
        group = ReentrantCallbackGroup()
        self.base_pub = self.create_publisher(Twist, self.base_topic, 10)
        self.relaying = False
        self.create_subscription(Twist, self.pick_cmd_topic, self.on_pick_cmd, 10,
                                 callback_group=group)
        self.status = {'phase': '', 'result': None, 'lift': '', 'lift_ack': False}
        self.create_subscription(String, 'pick_back_real/status', self.on_status, 10,
                                 callback_group=group)
        self.create_subscription(String, 'pick_back_real/result', self.on_result, 10,
                                 callback_group=group)
        self.create_subscription(String, 'forklift/state', self.on_lift_state, 10,
                                 callback_group=group)
        self.lock = threading.Lock()
        ActionServer(self, PickBox, '/forklift/pick_box', self.execute_pick,
                     callback_group=group, goal_callback=self.accept,
                     cancel_callback=lambda _g: CancelResponse.ACCEPT)
        ActionServer(self, SetDownBox, '/forklift/set_down_box', self.execute_set_down,
                     callback_group=group, goal_callback=self.accept,
                     cancel_callback=lambda _g: CancelResponse.ACCEPT)
        self.get_logger().info(
            f'forklift bridge: PickBox -> forklift_pick_back ({os.path.basename(self.cfg_file)}), '
            f'drive {self.pick_cmd_topic} -> {self.base_topic}; SetDownBox -> lift to '
            f'{self.set_down_mm:.0f} mm')

    # ── topics ───────────────────────────────────────────────────────────
    def accept(self, _goal):
        return GoalResponse.REJECT if self.lock.locked() else GoalResponse.ACCEPT

    def on_pick_cmd(self, msg):
        if self.relaying:
            self.base_pub.publish(msg)

    def on_status(self, msg):
        try:
            data = json.loads(msg.data)
            self.status['phase'] = f"{data.get('estado', '')}: {data.get('detalle', '')}"[:120]
        except ValueError:
            self.status['phase'] = msg.data[:120]

    def on_result(self, msg):
        self.status['result'] = msg.data

    def on_lift_state(self, msg):
        if msg.data.startswith('GOAL_SENT'):
            self.status['lift_ack'] = True      # published once: do not miss it between polls
            self.status['lift'] = ''            # states before it belong to an older goal
        else:
            self.status['lift'] = msg.data

    def feedback(self, goal_handle, cls, text):
        fb = cls.Feedback()
        fb.phase = text
        goal_handle.publish_feedback(fb)

    # ── PickBox ──────────────────────────────────────────────────────────
    def execute_pick(self, goal_handle):
        g = goal_handle.request
        result = PickBox.Result()
        level = LEVELS.get(g.level)
        if level is None:
            goal_handle.succeed()
            result.success = False
            result.reason = (f'level {g.level}: pick_back has no automatic pick for this '
                             'level yet (floor and the 20 mm shelf only)')
            self.get_logger().warn(f'PickBox {g.box_id}: ' + result.reason)
            return result
        with self.lock:
            ok, reason, canceled = self.run_pick(goal_handle, g.box_id, level)
        if canceled:
            goal_handle.canceled()
            result.success, result.reason = False, 'canceled'
            return result
        goal_handle.succeed()
        result.success, result.reason = ok, reason
        self.get_logger().info(f'PickBox {g.box_id}: ' + ('done' if ok else 'FAILED: ' + reason))
        return result

    def run_pick(self, goal_handle, box_id, level):
        self.status.update(phase='starting pick_back', result=None)
        cmd = ['ros2', 'run', 'main_ros', 'forklift_pick_back', '--config', self.cfg_file,
               '--level', level, '--execute', '--ros-args',
               '-p', f'use_sim_time:={"true" if self.sim_time else "false"}']
        self.get_logger().info(f'PickBox {box_id}: level {level}: ' + ' '.join(cmd))
        proc = subprocess.Popen(cmd, start_new_session=True)
        self.relaying = True
        deadline = time.monotonic() + self.pick_timeout
        last_phase = None
        try:
            while True:
                if goal_handle.is_cancel_requested:
                    self.interrupt(proc)
                    return False, 'canceled', True
                if self.status['phase'] != last_phase:
                    last_phase = self.status['phase']
                    self.feedback(goal_handle, PickBox, last_phase)
                res = self.status['result']
                if res is not None:
                    self.wait_exit(proc, 5.0)
                    if res.strip() == 'Done':
                        return True, '', False
                    return False, res.replace('Error:', '').strip() or res, False
                code = proc.poll()
                if code is not None:
                    return (code == 0, '' if code == 0 else
                            f'pick_back exited ({code}) at "{last_phase}"', False)
                if time.monotonic() > deadline:
                    self.interrupt(proc)
                    return False, f'pick timed out after {self.pick_timeout:.0f} s', False
                time.sleep(0.05)
        finally:
            self.relaying = False
            self.base_pub.publish(Twist())          # the base stops whatever happened

    def interrupt(self, proc):
        try:
            os.killpg(proc.pid, signal.SIGINT)      # pick_back aborts: base stop, lift stop
        except ProcessLookupError:
            return
        self.wait_exit(proc, 5.0)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)

    @staticmethod
    def wait_exit(proc, seconds):
        try:
            proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            pass

    # ── SetDownBox ───────────────────────────────────────────────────────
    def execute_set_down(self, goal_handle):
        g = goal_handle.request
        result = SetDownBox.Result()
        with self.lock:
            ok, reason, canceled = self.lower(goal_handle)
        if canceled:
            goal_handle.canceled()
            result.success, result.reason = False, 'canceled'
            return result
        goal_handle.succeed()
        result.success, result.reason = ok, reason
        self.get_logger().info(f'SetDownBox {g.box_id} ({g.where}): '
                               + ('done' if ok else 'FAILED: ' + reason))
        return result

    def lower(self, goal_handle):
        # Created only now: pick_back refuses to run while the lift topic has another
        # publisher, so the bridge does not keep one.
        pub = self.create_publisher(Float64, 'forklift/target_height_mm', 10)
        try:
            t0 = time.monotonic()
            while pub.get_subscription_count() == 0 and time.monotonic() - t0 < 3.0:
                time.sleep(0.05)
            if pub.get_subscription_count() == 0:
                return False, 'no lift node (forklift_operate / sim_forklift)', False
            self.status['lift'], self.status['lift_ack'] = '', False
            pub.publish(Float64(data=self.set_down_mm))
            sent = time.monotonic()
            self.feedback(goal_handle, SetDownBox, f'lowering to {self.set_down_mm:.0f} mm')
            while True:
                if goal_handle.is_cancel_requested:
                    return False, 'canceled', True
                state, acked = self.status['lift'], self.status['lift_ack']
                if acked and state.startswith(('FAULT', 'GOAL_NOT_REACHED')):
                    return False, 'lift: ' + state, False
                if acked and state == 'ARRIVED':
                    self.feedback(goal_handle, SetDownBox, 'box on the floor')
                    return True, '', False
                if time.monotonic() - sent > self.set_down_timeout:
                    return False, f'lift did not arrive in {self.set_down_timeout:.0f} s', False
                time.sleep(0.05)
        finally:
            self.destroy_publisher(pub)


def main(args=None):
    rclpy.init(args=args)
    node = ForkliftBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
