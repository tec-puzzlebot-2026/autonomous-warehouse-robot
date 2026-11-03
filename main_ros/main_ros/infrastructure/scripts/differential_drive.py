#!/usr/bin/env python3

"""
PuzzleBot velocity command relay (safety layer in front of the Hackerboard).

The Hackerboard firmware subscribes to /cmd_vel (geometry_msgs/Twist) and does
the differential-drive kinematics itself, so this node no longer computes wheel
speeds. It sits between the command sources (teleop, navigation, voice) and the
board, and adds what the board may not do on its own:

    - speed limits      (max_linear, max_angular)
    - command timeout   (no new command for cmd_timeout s -> robot stops)
    - fixed rate        (publishes to the board at publish_rate Hz)
    - stop on shutdown  (sends zeros when the node exits)

Subscribes:
    /cmd_vel_in   geometry_msgs/msg/Twist   (teleop / navigation publish here)

Publishes:
    /cmd_vel      geometry_msgs/msg/Twist   (Hackerboard, BEST_EFFORT QoS)

Run:
    ros2 run main_ros differential_drive
    ros2 run main_ros differential_drive --ros-args -p max_linear:=0.2
    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from geometry_msgs.msg import Twist


def clamp(value, limit):
    return max(-limit, min(limit, value))


class VelocityRelay(Node):

    def __init__(self):
        super().__init__("puzzlebot_differential_drive")

        self.declare_parameter("input_topic", "/cmd_vel_in")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("max_linear", 0.3)      # m/s
        self.declare_parameter("max_angular", 1.5)     # rad/s
        self.declare_parameter("cmd_timeout", 0.5)     # s
        self.declare_parameter("publish_rate", 20.0)   # Hz

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        rate = float(self.get_parameter("publish_rate").value)

        for name, value in (("max_linear", self.max_linear),
                            ("max_angular", self.max_angular),
                            ("cmd_timeout", self.cmd_timeout),
                            ("publish_rate", rate)):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if input_topic == output_topic:
            raise ValueError("input_topic and output_topic must differ")

        # The Hackerboard subscribes with BEST_EFFORT; match it.
        board_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cmd_pub = self.create_publisher(Twist, output_topic, board_qos)
        self.cmd_sub = self.create_subscription(Twist, input_topic, self.cmd_callback, 10)

        self.linear = 0.0
        self.angular = 0.0
        self.last_cmd_time = None
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f"Velocity relay: {input_topic} -> {output_topic} | "
            f"max {self.max_linear} m/s, {self.max_angular} rad/s | "
            f"timeout {self.cmd_timeout} s"
        )

    def cmd_callback(self, msg):
        linear = float(msg.linear.x)
        angular = float(msg.angular.z)
        if not math.isfinite(linear) or not math.isfinite(angular):
            self.get_logger().warning("Ignoring non-finite command.")
            return
        self.linear = clamp(linear, self.max_linear)
        self.angular = clamp(angular, self.max_angular)
        self.last_cmd_time = self.get_clock().now()

    def publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def control_loop(self):
        if self.last_cmd_time is not None:
            elapsed = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
            if elapsed > self.cmd_timeout:
                self.linear = 0.0
                self.angular = 0.0
        self.publish_cmd(self.linear, self.angular)

    def stop_motors(self):
        self.linear = 0.0
        self.angular = 0.0
        for _ in range(5):
            self.publish_cmd(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motors()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()