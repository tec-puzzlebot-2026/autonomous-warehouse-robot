#!/usr/bin/env python3
"""
encoder_bridge.py — the real robot's encoder topics, from the simulation's joint states

On the PuzzleBot the Hackerboard publishes each wheel's angular velocity on
/VelocityEncL and /VelocityEncR (std_msgs/Float32, rad/s); Chris's EKF predicts with
them. Isaac Sim publishes /joint_states instead. This node converts one into the
other, so the EKF (and anything else written for the real robot) runs unchanged in
the simulation.

Parameters: left_joint, right_joint (joint names in /joint_states), left_sign,
right_sign (-1.0 if a wheel turns backwards), topics.
Velocities come from the joint_states velocity field, or from differentiating the
positions when the simulator leaves it empty.
"""
# Author: Raymundo Gomez

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


class EncoderBridge(Node):
    def __init__(self):
        super().__init__('encoder_bridge')
        p = self.declare_parameter
        self.left = p('left_joint', 'wheel_left_joint').value
        self.right = p('right_joint', 'wheel_right_joint').value
        self.sign_l = float(p('left_sign', 1.0).value)
        self.sign_r = float(p('right_sign', 1.0).value)
        self.pub_l = self.create_publisher(Float32, p('left_topic', '/VelocityEncL').value, 10)
        self.pub_r = self.create_publisher(Float32, p('right_topic', '/VelocityEncR').value, 10)
        self.create_subscription(JointState, p('joint_states_topic', '/joint_states').value,
                                 self.on_joints, qos_profile_sensor_data)
        self.prev = None
        self.warned = False

    def on_joints(self, msg):
        names = list(msg.name)
        if self.left not in names or self.right not in names:
            if not self.warned:
                self.get_logger().warn(f'joints {self.left!r}/{self.right!r} not in '
                                       f'/joint_states {names}: set left_joint/right_joint')
                self.warned = True
            return
        li, ri = names.index(self.left), names.index(self.right)
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if len(msg.velocity) > max(li, ri):
            wl, wr = msg.velocity[li], msg.velocity[ri]
        else:
            pl, pr = msg.position[li], msg.position[ri]
            if self.prev is None or t <= self.prev[0]:
                self.prev = (t, pl, pr)
                return
            dt = t - self.prev[0]
            wl, wr = (pl - self.prev[1]) / dt, (pr - self.prev[2]) / dt
            self.prev = (t, pl, pr)
        # The EKF integrates when the RIGHT wheel arrives: publish left first.
        self.pub_l.publish(Float32(data=float(self.sign_l * wl)))
        self.pub_r.publish(Float32(data=float(self.sign_r * wr)))


def main(args=None):
    rclpy.init(args=args)
    node = EncoderBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
