#!/usr/bin/env python3
"""
scan_filter.py — hide the carried box from the LiDAR

A carried box sits 1 to 3 cm in front of the robot, about 12 cm wide, and blocks
the front of the LiDAR (the camera above it stays clear). Without this filter,
Sendero would see the box as an obstacle touching the robot and refuse to move,
and Planisferio would draw it into the map as a wall that moves with the robot.

This node sits between the LiDAR and everything that reads it:

    /laser_scan  ->  scan_filter  ->  /scan_nav   (read by Sendero and Planisferio)

While /restock/carrying is true, every return that falls inside the box's
rectangle in front of the robot (base_link: x from carry_x_min to carry_x_max,
|y| <= carry_half_width) is replaced by "no return" (inf). Otherwise the scan
passes through unchanged. The LiDAR mounting comes from TF base_link -> scan
frame, or from the laser_* parameters when that transform does not exist.
"""
# Author: Raymundo Gomez

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from main_ros.navigation.scan_mask import box_mask


class ScanFilter(Node):
    def __init__(self):
        super().__init__('scan_filter')
        p = self.declare_parameter
        self.in_topic = p('input_topic', '/laser_scan').value
        self.out_topic = p('output_topic', '/scan_nav').value
        self.base_frame = p('base_frame', 'base_link').value
        # Box rectangle in base_link while carrying [m]. Robot front at ~0.15 m (PuzzleBot),
        # box 1-3 cm ahead of it, 10 cm deep, 12 cm wide; with a margin.
        self.x_min = float(p('carry_x_min', 0.0).value)
        self.x_max = float(p('carry_x_max', 0.32).value)
        self.half_width = float(p('carry_half_width', 0.09).value)
        self.mount = (float(p('laser_x', 0.0).value), float(p('laser_y', 0.0).value),
                      float(p('laser_yaw', 0.0).value))
        self.carrying = bool(p('carrying', False).value)
        self.mount_from_tf = None
        try:
            import tf2_ros
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        except ImportError:
            self.tf_buffer = None
        self.pub = self.create_publisher(LaserScan, self.out_topic, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.in_topic, self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Bool, p('carrying_topic', '/restock/carrying').value,
                                 self.on_carrying, 10)
        self.get_logger().info(f'scan filter {self.in_topic} -> {self.out_topic} '
                               f'(box mask x {self.x_min:.2f}..{self.x_max:.2f} m, '
                               f'|y| <= {self.half_width:.2f} m while carrying)')

    def on_carrying(self, msg):
        if bool(msg.data) != self.carrying:
            self.carrying = bool(msg.data)
            self.get_logger().info('carrying: box masked' if self.carrying else 'not carrying')

    def _mount(self, frame):
        if self.mount_from_tf is not None or self.tf_buffer is None:
            return self.mount_from_tf or self.mount
        try:
            from rclpy.time import Time
            t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.mount_from_tf = (t.transform.translation.x, t.transform.translation.y, yaw)
            return self.mount_from_tf
        except Exception:                      # noqa: BLE001 (any TF error: use the parameters)
            return self.mount

    def on_scan(self, msg):
        if self.carrying:
            mount = self._mount(msg.header.frame_id)
            mask = box_mask(msg.ranges, msg.angle_min, msg.angle_increment, mount,
                            self.x_min, self.x_max, self.half_width)
            if mask.any():
                r = np.asarray(msg.ranges, dtype=np.float32)
                r[mask] = np.inf
                msg.ranges = r.tolist()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanFilter()
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
