#!/usr/bin/env python3
"""
forklift_placeholder.py — stand-in for the forklift state machine (PickBox, SetDownBox)

Until the real forklift script exists, this node serves the two actions the
Program Manager calls during restocking, so the whole mission can run end to end:

  /forklift/pick_box       warehouse_msgs/action/PickBox
  /forklift/set_down_box   warehouse_msgs/action/SetDownBox

Each goal "takes" a few seconds, publishes a few feedback phases and succeeds.
Failures can be injected to test the Program Manager's failure paths:

  ros2 run main_ros forklift_placeholder --ros-args -p pick_fail_first:=1
      the first PickBox fails, the next ones succeed
  -p pick_fail_every:=3     every third PickBox fails
  -p set_down_fail_first:=1

The real forklift node only has to serve the same two actions under the same
names; the Program Manager does not change. Cancelling a goal (P or X in the
monitor) ends it as canceled.
"""
# Author: Raymundo Gomez

import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from warehouse_msgs.action import PickBox, SetDownBox

PICK_PHASES = ('aligning with IR', 'lowering forks', 'inserting', 'lifting', 'backing out')
SET_DOWN_PHASES = ('lowering', 'releasing', 'forks up')


class ForkliftPlaceholder(Node):
    """Serves PickBox and SetDownBox with simulated durations."""

    def __init__(self):
        super().__init__('forklift_placeholder')
        p = self.declare_parameter
        self.pick_s = float(p('pick_duration_s', 4.0).value)
        self.set_down_s = float(p('set_down_duration_s', 3.0).value)
        self.pick_fail_first = int(p('pick_fail_first', 0).value)
        self.pick_fail_every = int(p('pick_fail_every', 0).value)
        self.set_down_fail_first = int(p('set_down_fail_first', 0).value)
        pick_name = p('pick_action', '/forklift/pick_box').value
        set_down_name = p('set_down_action', '/forklift/set_down_box').value
        self.n_pick = 0
        self.n_set_down = 0
        group = ReentrantCallbackGroup()
        self.pick_srv = ActionServer(
            self, PickBox, pick_name, self.execute_pick, callback_group=group,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT)
        self.set_down_srv = ActionServer(
            self, SetDownBox, set_down_name, self.execute_set_down, callback_group=group,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT)
        self.get_logger().info(
            f'Forklift PLACEHOLDER ready: {pick_name}, {set_down_name} '
            f'(pick {self.pick_s:.0f} s, set down {self.set_down_s:.0f} s). '
            'Replace with the real forklift node.')

    def _run(self, goal_handle, phases, duration, feedback_cls):
        """Walk through the phases; False if canceled."""
        step = duration / max(len(phases), 1)
        for phase in phases:
            if goal_handle.is_cancel_requested:
                return False
            fb = feedback_cls.Feedback()
            fb.phase = phase
            goal_handle.publish_feedback(fb)
            end = time.monotonic() + step
            while time.monotonic() < end:
                if goal_handle.is_cancel_requested:
                    return False
                time.sleep(0.05)
        return True

    def execute_pick(self, goal_handle):
        g = goal_handle.request
        self.n_pick += 1
        self.get_logger().info(f'PickBox {g.box_id}: level {g.level}, '
                               f'distance {g.distance:.3f} m')
        ok = self._run(goal_handle, PICK_PHASES, self.pick_s, PickBox)
        result = PickBox.Result()
        if not ok:
            goal_handle.canceled()
            result.success, result.reason = False, 'canceled'
            return result
        fail = (self.n_pick <= self.pick_fail_first
                or (self.pick_fail_every > 0 and self.n_pick % self.pick_fail_every == 0))
        result.success = not fail
        result.reason = 'placeholder: injected pick failure' if fail else ''
        goal_handle.succeed()
        self.get_logger().info(f'PickBox {g.box_id}: ' + ('FAILED (injected)' if fail else 'done'))
        return result

    def execute_set_down(self, goal_handle):
        g = goal_handle.request
        self.n_set_down += 1
        self.get_logger().info(f'SetDownBox {g.box_id}: {g.where}')
        ok = self._run(goal_handle, SET_DOWN_PHASES, self.set_down_s, SetDownBox)
        result = SetDownBox.Result()
        if not ok:
            goal_handle.canceled()
            result.success, result.reason = False, 'canceled'
            return result
        fail = self.n_set_down <= self.set_down_fail_first
        result.success = not fail
        result.reason = 'placeholder: injected set-down failure' if fail else ''
        goal_handle.succeed()
        self.get_logger().info(f'SetDownBox {g.box_id}: '
                               + ('FAILED (injected)' if fail else 'done'))
        return result


def main(args=None):
    """Entry point: serve both actions until Ctrl+C."""
    rclpy.init(args=args)
    node = ForkliftPlaceholder()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
