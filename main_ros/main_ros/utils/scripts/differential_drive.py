#!/usr/bin/env python3

"""
PuzzleBot differential-drive controller.

Subscribes:
    /cmd_vel       geometry_msgs/msg/Twist

Publishes:
    /VelocitySetL  std_msgs/msg/Float32
    /VelocitySetR  std_msgs/msg/Float32

IMPORTANT:
    Assumes Hackerboard velocity setpoints are in rad/s.
    Verify against the professor's firmware before driving.
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
from std_msgs.msg import Float32


class DifferentialDrive(Node):

    def __init__(self):
        super().__init__("puzzlebot_differential_drive")

        # These are example values, NOT verified robot dimensions.
        self.declare_parameter("wheel_radius", 0.02)
        self.declare_parameter("wheel_separation", 0.20)

        self.declare_parameter("max_wheel_speed", 2.0)
        self.declare_parameter("cmd_timeout", 0.5)

        self.declare_parameter("invert_left", False)
        self.declare_parameter("invert_right", False)

        self.wheel_radius = float(
            self.get_parameter("wheel_radius").value
        )

        self.wheel_separation = float(
            self.get_parameter("wheel_separation").value
        )

        self.max_wheel_speed = float(
            self.get_parameter("max_wheel_speed").value
        )

        self.cmd_timeout = float(
            self.get_parameter("cmd_timeout").value
        )

        self.invert_left = bool(
            self.get_parameter("invert_left").value
        )

        self.invert_right = bool(
            self.get_parameter("invert_right").value
        )

        if self.wheel_radius <= 0:
            raise ValueError("wheel_radius must be positive")

        if self.wheel_separation <= 0:
            raise ValueError("wheel_separation must be positive")

        if self.max_wheel_speed <= 0:
            raise ValueError("max_wheel_speed must be positive")

        if self.cmd_timeout <= 0:
            raise ValueError("cmd_timeout must be positive")

        motor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.left_pub = self.create_publisher(
            Float32,
            "/VelocitySetL",
            motor_qos,
        )

        self.right_pub = self.create_publisher(
            Float32,
            "/VelocitySetR",
            motor_qos,
        )

        self.cmd_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_callback,
            10,
        )

        self.left_speed = 0.0
        self.right_speed = 0.0

        self.last_cmd_time = None

        # Publish at 20 Hz.
        self.timer = self.create_timer(
            0.05,
            self.control_loop,
        )

        self.get_logger().info(
            "Differential-drive controller started."
        )

    def cmd_callback(self, msg):

        linear = float(msg.linear.x)
        angular = float(msg.angular.z)

        if not math.isfinite(linear) or not math.isfinite(angular):
            self.get_logger().warning(
                "Ignoring non-finite /cmd_vel command."
            )
            return

        # Differential-drive inverse kinematics.
        #
        # v_left  = v - omega * L / 2
        # v_right = v + omega * L / 2
        #
        # Convert wheel linear speed to angular speed:
        #
        # omega_wheel = v_wheel / wheel_radius

        left = (
            linear - angular * self.wheel_separation / 2.0
        ) / self.wheel_radius

        right = (
            linear + angular * self.wheel_separation / 2.0
        ) / self.wheel_radius

        # Clamp motor commands.
        left = max(
            -self.max_wheel_speed,
            min(self.max_wheel_speed, left),
        )

        right = max(
            -self.max_wheel_speed,
            min(self.max_wheel_speed, right),
        )

        if self.invert_left:
            left *= -1.0

        if self.invert_right:
            right *= -1.0

        self.left_speed = left
        self.right_speed = right

        self.last_cmd_time = (
            self.get_clock().now()
        )

    def publish_speeds(self, left, right):

        left_msg = Float32()
        left_msg.data = float(left)

        right_msg = Float32()
        right_msg.data = float(right)

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

    def control_loop(self):

        if self.last_cmd_time is None:
            self.publish_speeds(0.0, 0.0)
            return

        elapsed = (
            self.get_clock().now() - self.last_cmd_time
        ).nanoseconds / 1e9

        if elapsed > self.cmd_timeout:
            self.left_speed = 0.0
            self.right_speed = 0.0

        self.publish_speeds(
            self.left_speed,
            self.right_speed,
        )

    def stop_motors(self):

        self.left_speed = 0.0
        self.right_speed = 0.0

        for _ in range(5):
            self.publish_speeds(0.0, 0.0)


def main(args=None):

    rclpy.init(args=args)

    node = DifferentialDrive()

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