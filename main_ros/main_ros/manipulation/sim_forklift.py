#!/usr/bin/env python3
"""
sim_forklift.py — the forklift's ROS interface in Isaac Sim (stands in for `operar`)
====================================================================================
On the robot, `forklift_operate` (control_forklift/operate.py) talks to the Tang Nano
board that drives the lift motor. In our Isaac warehouse there is no Tang: the lift
is two prismatic joints driven by an OmniGraph (ROS2 Subscribe Joint State ->
Articulation Controller on /forklift/joint_command). This node offers the same topics
and service as `operar`, so control_forklift's `pick_back` runs unchanged in the sim:

  in   forklift/target_height_mm   std_msgs/Float64   lift goal, mm above the bottom
  out  forklift/current_height_mm  std_msgs/Float64   from /joint_states (both slides)
  out  forklift/state              std_msgs/String    READY_STOPPED, GOAL_SENT <mm> ...,
                                                      MOVING, ARRIVED, STOPPED,
                                                      GOAL_NOT_REACHED: <why>, FAULT: <why>
  srv  forklift/stop               std_srvs/Trigger   hold the lift where it is
  out  /forklift/joint_command     sensor_msgs/JointState   positions of the two slides (m)

The lift has two stages that move together (mast and fork slide), so a height h
means h/2 on each joint. The command is ramped at the real motor's speed
(about 70 mm/s), so the sim lifts as fast as the robot does, not instantly.

IR sensors are not simulated: in the sim, pick_back uses the camera to find the box
face (face_source: rgb) and the LiDAR to confirm the load (carry_sensor: lidar);
see config/forklift/pick_sim.json.
"""
# Author: Raymundo Gomez

import math

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, String
    from std_srvs.srv import Trigger
except ImportError:                     # the tests use LiftModel without ROS
    rclpy = None
    Node = object


class LiftModel:
    """The lift's state machine, without ROS (tested in test/test_sim_forklift.py)."""

    def __init__(self, max_height_mm=246.0, speed_mm_s=70.0, tolerance_mm=2.0,
                 settle_s=0.3, timeout_s=15.0):
        self.max_h = max_height_mm
        self.speed = speed_mm_s
        self.tol = tolerance_mm
        self.settle_s = settle_s
        self.timeout_s = timeout_s
        self.height = None              # measured, mm (None until /joint_states arrives)
        self.command = None             # ramped command sent to Isaac, mm
        self.goal = None
        self.goal_since = None
        self.near_since = None
        self.state = 'READY_STOPPED'
        self.events = []                # one-shot states (GOAL_SENT ...), published once

    def measure(self, height_mm):
        self.height = height_mm
        if self.command is None:
            self.command = height_mm

    def set_goal(self, goal_mm, now):
        if not math.isfinite(goal_mm) or not 0.0 <= goal_mm <= self.max_h:
            self.state = f'FAULT: goal {goal_mm} mm outside 0..{self.max_h:.0f} mm'
            self.goal = None
            return False
        if self.height is None:
            self.state = 'FAULT: no lift position yet (/joint_states)'
            self.goal = None
            return False
        self.goal = float(goal_mm)
        self.goal_since = now
        self.near_since = None
        self.events.append(f'GOAL_SENT {self.goal:.2f} mm (sim)')
        self.state = 'MOVING'
        return True

    def stop(self):
        if self.height is not None:
            self.command = self.height
        self.goal = None
        self.state = 'STOPPED'

    def step(self, now, dt):
        """Advance the ramp; returns the command to send (mm) or None."""
        if self.goal is None or self.command is None:
            return self.command
        delta = self.goal - self.command
        stride = self.speed * max(dt, 0.0)
        if abs(delta) <= stride:
            self.command = self.goal
        else:
            self.command += math.copysign(stride, delta)
        if self.height is not None and abs(self.height - self.goal) <= self.tol:
            if self.near_since is None:
                self.near_since = now
            if now - self.near_since >= self.settle_s:
                self.state = 'ARRIVED'
        else:
            self.near_since = None
            if self.state == 'ARRIVED':
                self.state = 'MOVING'
        if self.state != 'ARRIVED' and now - self.goal_since > self.timeout_s:
            self.state = (f'GOAL_NOT_REACHED: at {self.height:.1f} mm, goal {self.goal:.1f} mm '
                          f'after {self.timeout_s:.0f} s')
            self.goal = None
        return self.command


class SimForklift(Node):
    def __init__(self):
        super().__init__('sim_forklift')
        p = self.declare_parameter
        p('joint_command_topic', '/forklift/joint_command')
        p('joint_states_topic', '/joint_states')
        p('mast_joint', 'forklift_mast_slide_joint')
        p('fork_joint', 'forklift_fork_slide_joint')
        p('max_height_mm', 246.0)       # pick.json lift_max_height_mm (the joints reach 254)
        p('speed_mm_s', 70.0)           # the real motor's full speed (motor.json: 0.070 m/s)
        p('tolerance_mm', 2.0)
        p('timeout_s', 15.0)
        p('rate_hz', 20.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mast, self.fork = g('mast_joint'), g('fork_joint')
        self.model = LiftModel(float(g('max_height_mm')), float(g('speed_mm_s')),
                               float(g('tolerance_mm')), timeout_s=float(g('timeout_s')))
        self.cmd_pub = self.create_publisher(JointState, g('joint_command_topic'), 10)
        self.height_pub = self.create_publisher(Float64, 'forklift/current_height_mm', 10)
        self.state_pub = self.create_publisher(String, 'forklift/state', 10)
        self.create_subscription(Float64, 'forklift/target_height_mm', self.on_goal, 10)
        self.create_subscription(JointState, g('joint_states_topic'), self.on_joints, 10)
        self.create_service(Trigger, 'forklift/stop', self.on_stop)
        self.last_tick = self.now_s()
        self.create_timer(1.0 / max(1.0, float(g('rate_hz'))), self.tick)
        self.get_logger().info(
            f'sim forklift: {self.mast} + {self.fork} on {g("joint_command_topic")}, '
            f'0..{self.model.max_h:.0f} mm at {self.model.speed:.0f} mm/s')

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if self.mast in pos and self.fork in pos:
            self.model.measure(1000.0 * (pos[self.mast] + pos[self.fork]))

    def on_goal(self, msg):
        if self.model.set_goal(float(msg.data), self.now_s()):
            self.get_logger().info(f'lift goal {msg.data:.1f} mm')
        else:
            self.get_logger().warn(self.model.state)
        self.flush_events()

    def on_stop(self, _req, res):
        self.model.stop()
        self.send(self.model.command)
        res.success, res.message = True, 'lift held where it is (sim)'
        return res

    def flush_events(self):
        for text in self.model.events:
            self.state_pub.publish(String(data=text))
        self.model.events.clear()

    def send(self, height_mm):
        if height_mm is None:
            return
        half = max(0.0, height_mm) / 2000.0          # mm -> m, split between the stages
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [self.mast, self.fork]
        js.position = [half, half]
        self.cmd_pub.publish(js)

    def tick(self):
        now = self.now_s()
        dt, self.last_tick = now - self.last_tick, now
        self.send(self.model.step(now, dt))
        self.flush_events()
        if self.model.height is not None:
            self.height_pub.publish(Float64(data=float(self.model.height)))
        self.state_pub.publish(String(data=self.model.state))


def main(args=None):
    rclpy.init(args=args)
    node = SimForklift()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
