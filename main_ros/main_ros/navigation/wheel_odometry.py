import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class WheelOdometry(Node):

    def __init__(self):
        super().__init__("wheel_odometry")

        # ==========================================
        # ROBOT PARAMETERS
        # ==========================================

        self.declare_parameter("wheel_radius", 0.05)
        self.declare_parameter("wheel_separation", 0.19)

        self.declare_parameter("left_joint", "wheel_left_joint")
        self.declare_parameter("right_joint", "wheel_right_joint")

        self.declare_parameter("left_sign", 1.0)
        self.declare_parameter("right_sign", 1.0)

        self.declare_parameter("publish_tf", True)

        self.radius = float(
            self.get_parameter("wheel_radius").value
        )

        self.separation = float(
            self.get_parameter("wheel_separation").value
        )

        self.left_joint = self.get_parameter("left_joint").value
        self.right_joint = self.get_parameter("right_joint").value

        self.left_sign = float(
            self.get_parameter("left_sign").value
        )

        self.right_sign = float(
            self.get_parameter("right_sign").value
        )

        self.publish_tf = bool(
            self.get_parameter("publish_tf").value
        )

        if self.radius <= 0 or self.separation <= 0:
            raise ValueError(
                "Wheel radius and separation must be positive"
            )

        if (
            self.left_sign not in (-1.0, 1.0)
            or self.right_sign not in (-1.0, 1.0)
        ):
            raise ValueError("Wheel signs must be +1 or -1")

        # ==========================================
        # ESTIMATED ROBOT POSE
        # ==========================================

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.previous = None
        self.last_log_time = time.monotonic()

        # ==========================================
        # ROS 2 COMMUNICATION
        # ==========================================

        self.odom_pub = self.create_publisher(
            Odometry,
            "/wheel/odom",
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joints,
            qos_profile_sensor_data
        )

        self.tf_pub = (
            TransformBroadcaster(self)
            if self.publish_tf
            else None
        )

        self.get_logger().info(
            f"Wheel odometry started: "
            f"r={self.radius} m, "
            f"track={self.separation} m"
        )

    # ==============================================
    # ENCODER CALLBACK
    # ==============================================

    def on_joints(self, msg):

        # Find wheels by name, never by array position.
        names = {
            name: i
            for i, name in enumerate(msg.name)
        }

        if (
            self.left_joint not in names
            or self.right_joint not in names
        ):
            return

        li = names[self.left_joint]
        ri = names[self.right_joint]

        if len(msg.position) <= max(li, ri):
            return

        left = float(msg.position[li])
        right = float(msg.position[ri])

        if not (
            math.isfinite(left)
            and math.isfinite(right)
        ):
            return

        # ==========================================
        # TIME
        # ==========================================

        stamp_ns = (
            msg.header.stamp.sec * 1_000_000_000
            + msg.header.stamp.nanosec
        )

        if self.previous is None:
            self.previous = (
                stamp_ns,
                left,
                right
            )

            self.publish(
                msg.header.stamp,
                0.0,
                0.0
            )
            return

        old_ns, old_left, old_right = self.previous

        self.previous = (
            stamp_ns,
            left,
            right
        )

        dt = (stamp_ns - old_ns) * 1e-9

        if dt <= 0:

            if dt < 0:
                self.x = 0.0
                self.y = 0.0
                self.theta = 0.0

                self.get_logger().warn(
                    "Simulation clock reset: odometry reset"
                )

            return

        # ==========================================
        # WHEEL ANGULAR DISPLACEMENT
        # ==========================================

        dleft = normalize_angle(
            left - old_left
        )

        dright = normalize_angle(
            right - old_right
        )

        # Reject implausible encoder jumps.
        if (
            abs(dleft / dt) > 30
            or abs(dright / dt) > 30
        ):
            self.get_logger().warn(
                "Implausible wheel jump ignored"
            )
            return

        # ==========================================
        # DIFFERENTIAL DRIVE KINEMATICS
        # ==========================================

        dl = (
            self.radius
            * self.left_sign
            * dleft
        )

        dr = (
            self.radius
            * self.right_sign
            * dright
        )

        # Forward displacement.
        ds = (dr + dl) / 2.0

        # Change in orientation.
        dtheta = (
            (dr - dl)
            / self.separation
        )

        # ==========================================
        # INTEGRATE ROBOT POSE
        # ==========================================

        mid_heading = (
            self.theta + dtheta / 2.0
        )

        self.x += (
            ds * math.cos(mid_heading)
        )

        self.y += (
            ds * math.sin(mid_heading)
        )

        self.theta = normalize_angle(
            self.theta + dtheta
        )

        # ==========================================
        # ESTIMATE VELOCITIES
        # ==========================================

        linear_velocity = ds / dt
        angular_velocity = dtheta / dt

        self.publish(
            msg.header.stamp,
            linear_velocity,
            angular_velocity
        )

        # ==========================================
        # DEBUG OUTPUT
        # ==========================================

        if time.monotonic() - self.last_log_time >= 2.0:

            self.get_logger().info(
                f"Wheel odom: "
                f"x={self.x:.3f} "
                f"y={self.y:.3f} "
                f"heading={math.degrees(self.theta):.1f}deg "
                f"v={linear_velocity:.3f}m/s "
                f"w={angular_velocity:.3f}rad/s"
            )

            self.last_log_time = time.monotonic()

    # ==============================================
    # PUBLISH ODOMETRY AND TF
    # ==============================================

    def publish(self, stamp, v, w):

        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)

        odom = Odometry()

        odom.header.stamp = stamp
        odom.header.frame_id = "odom"

        odom.child_frame_id = "base_link"

        # Position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        # Orientation quaternion
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Robot-frame velocities
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w

        # Placeholder covariances.
        # These must eventually be calibrated.
        for i in (0, 7):
            odom.pose.covariance[i] = 0.01
            odom.twist.covariance[i] = 0.04

        odom.pose.covariance[35] = 0.02
        odom.twist.covariance[35] = 0.04

        for i in (14, 21, 28):
            odom.pose.covariance[i] = 1000.0
            odom.twist.covariance[i] = 1000.0

        self.odom_pub.publish(odom)

        # Publish odom -> base_link transform.
        if self.tf_pub is not None:

            transform = TransformStamped()

            transform.header = odom.header
            transform.child_frame_id = "base_link"

            transform.transform.translation.x = self.x
            transform.transform.translation.y = self.y

            transform.transform.rotation.z = qz
            transform.transform.rotation.w = qw

            self.tf_pub.sendTransform(transform)


def main(args=None):

    rclpy.init(args=args)

    node = WheelOdometry()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()