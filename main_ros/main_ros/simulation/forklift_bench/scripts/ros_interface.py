"""Expose the simulated robot interfaces to a native ROS 2 Jazzy client."""
from functools import partial
from collections import deque
import math

from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan, Range
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from pxr import Gf

from physics import quaternion, rotation

TF_LINKS = {
    "base_link": "base_footprint",
    "wheel_left_link": "base_link",
    "wheel_right_link": "base_link",
    "laser_frame": "base_link",
    "forklift_mast_fixed_link": "base_link",
    "forklift_mast_slider_link": "forklift_mast_fixed_link",
    "forklift_fork_carriage_link": "forklift_mast_slider_link",
    "forklift_ir_left_link": "forklift_mast_fixed_link",
    "forklift_ir_right_link": "forklift_mast_fixed_link",
    "camera_link": "forklift_mast_slider_link",
    "camera_link_optical": "camera_link",
}
SCAN_RAYS = 360
BASE_TO_AXLE = .052


def stamp_from(time):
    sec, ns = divmod(round(time * 1_000_000_000), 1_000_000_000)
    return Time(sec=sec, nanosec=ns)


def tf(parent, child, stamp, position, attitude):
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, position)
    q = msg.transform.rotation
    q.x, q.y, q.z, q.w = map(float, (*attitude.GetImaginary(), attitude.GetReal()))
    return msg


class Interface(Node):
    def __init__(self, physics, scan_period, camera_indices, firmware=False):
        super().__init__("simforklift")
        self.sim = physics
        self.firmware = firmware
        self.pick_frames = deque(maxlen=60)
        self.scan_period = scan_period
        self.clock = self.create_publisher(Clock, "/clock", 10)
        self.tf = self.create_publisher(TFMessage, "/tf", 10)
        self.ports = {}
        self.angles = np.linspace(-math.pi, math.pi, SCAN_RAYS, endpoint=False)
        for i in sorted(self.sim.rows):
            ns = f"/puzzlebot_{i:02d}"
            self.create_subscription(Twist, ns+"/cmd_vel", partial(self.on_twist, i), 1)
            if not firmware:
                self.create_subscription(Float64, ns+"/forklift/command", partial(self.on_lift, i), 1)
                self.create_service(Trigger, ns+"/reset", partial(self.on_reset, i))
            topics = {"joints": (JointState, "/joint_states"), "odom": (Odometry, "/wheel/odom"),
                      "height": (Float64, "/forklift/height"), "scan": (LaserScan, "/scan")}
            if not firmware:
                topics.update(left=(Range, "/ir/left"), right=(Range, "/ir/right"))
            if i in camera_indices:
                topics.update(rgb=(Image, "/camera/image_raw"), info=(CameraInfo, "/camera/camera_info"))
            self.ports[i] = {key: self.create_publisher(typ, ns+topic, qos_profile_sensor_data)
                             for key, (typ, topic) in topics.items()}

    def on_twist(self, index, msg):
        self.sim.command_drive(index, msg.linear.x, msg.angular.z)

    def on_lift(self, index, msg):
        if not self.sim.command_lift(index, msg.data):
            self.get_logger().warning(f"Invalid lift command for puzzlebot_{index:02d}")

    def on_reset(self, index, request, response):
        self.sim.reset(index)
        response.success = True
        return response

    def publish_state(self):
        sim = self.sim
        stamp = stamp_from(sim.time)
        self.clock.publish(Clock(clock=stamp))
        positions = sim.robots.get_dof_positions()
        speeds = sim.robots.get_dof_velocities()
        link_poses = sim.robots.get_link_transforms()
        transforms = []
        for index, row in sim.rows.items():
            ports = self.ports[index]
            ns = f"puzzlebot_{index:02d}"
            base = f"{ns}/base_footprint"
            if self.firmware:
                root = sim.link_pose(index, "base_footprint", link_poses)
                base_rotation = np.asarray(Gf.Matrix3d(rotation(root)))
                frames = {}
                for child in ("laser_frame", "camera_link_optical"):
                    pose = sim.link_pose(index, child, link_poses)
                    mat = np.eye(4)
                    mat[:3, :3] = base_rotation @ np.asarray(Gf.Matrix3d(rotation(pose))).T
                    mat[:3, 3] = base_rotation @ (pose[:3] - root[:3])
                    frames[f"{ns}/{child}"] = mat
                self.pick_frames.append((sim.time, frames))
            wheels = JointState()
            wheels.header.stamp = stamp
            wheels.name = list(sim.names)
            wheels.position = positions[row].tolist()
            wheels.velocity = speeds[row].tolist()
            ports["joints"].publish(wheels)
            ports["height"].publish(Float64(data=float(sum(positions[row][sim.slides]))))
            x, y, theta = sim.odom[row]
            q = Gf.Quatd(math.cos(theta/2), Gf.Vec3d(0,0,math.sin(theta/2)))
            base_xy = (x+BASE_TO_AXLE*(1-math.cos(theta)), y-BASE_TO_AXLE*math.sin(theta), 0.)
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = f"{ns}/odom"
            odom.child_frame_id = base
            odom.pose.pose.position.x, odom.pose.pose.position.y = map(float, base_xy[:2])
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(theta/2), math.cos(theta/2)
            left, right = speeds[row][sim.wheels] * sim.odom_radius
            odom.twist.twist.linear.x = float((left+right)/2)
            odom.twist.twist.angular.z = float((right-left)/sim.odom_track)
            odom.twist.twist.linear.y = -BASE_TO_AXLE * odom.twist.twist.angular.z
            ports["odom"].publish(odom)
            transforms.append(tf(odom.header.frame_id, base, stamp, base_xy, q))
            active_links = TF_LINKS | {
                f"forklift_ir_{side}_optical_frame": f"forklift_ir_{side}_link"
                for side in ("left", "right")
                if f"forklift_ir_{side}_optical_frame" in sim.links
            }
            for child, parent in active_links.items():
                parent_pose = sim.link_pose(index, parent, link_poses)
                pose = sim.link_pose(index, child, link_poses)
                inv = quaternion(parent_pose).GetInverse()
                pos = Gf.Rotation(inv).TransformDir(Gf.Vec3d(*map(float, pose[:3]-parent_pose[:3])))
                att = inv * quaternion(pose)
                transforms.append(tf(f"{ns}/{parent}", f"{ns}/{child}", stamp, pos, att))
        self.tf.publish(TFMessage(transforms=transforms))

    def pick_transform(self, frame, stamp):
        if not self.pick_frames:
            raise ValueError("esperando TF del sensor")
        if stamp is None:
            _, frames = self.pick_frames[-1]
        else:
            seconds = stamp.sec + stamp.nanosec * 1e-9
            current, frames = min(self.pick_frames, key=lambda sample: abs(sample[0] - seconds))
            if abs(current-seconds) > .001:
                raise ValueError("esperando el TF de la imagen")
        if frame not in frames:
            raise ValueError(f"Falta TF del sensor {frame}")
        return frames[frame]

    def publish_ranges(self):
        sim = self.sim
        stamp = stamp_from(sim.time)
        link_poses = sim.robots.get_link_transforms()
        for index, ports in self.ports.items():
            ns = f"puzzlebot_{index:02d}"
            lidar = sim.link_pose(index, "laser_frame", link_poses)
            lidar_rot = rotation(lidar)
            scan = LaserScan()
            scan.header.stamp = stamp
            scan.header.frame_id = f"{ns}/laser_frame"
            scan.angle_min, scan.angle_max = float(self.angles[0]), float(self.angles[-1])
            scan.angle_increment = 2*math.pi/SCAN_RAYS
            scan.scan_time = self.scan_period
            scan.time_increment = 0.0  # All rays are sampled at the same simulation time.
            scan.range_min, scan.range_max = .05, 3.5
            for angle in self.angles:
                direction = lidar_rot.TransformDir(Gf.Vec3d(math.cos(angle),math.sin(angle),0.0))
                result = sim.range_ray(index, lidar[:3], direction, scan.range_max)
                scan.ranges.append(result if result >= scan.range_min else math.inf)
            ports["scan"].publish(scan)

            if self.firmware:
                continue
            for side in ("left", "right"):
                frame = f"forklift_ir_{side}_optical_frame"
                if frame not in sim.links:
                    frame = f"forklift_ir_{side}_link"
                pose = sim.link_pose(index, frame, link_poses)
                beam = rotation(pose).TransformDir(Gf.Vec3d(1,0,0))
                ir = Range()
                ir.header.stamp, ir.header.frame_id = stamp, f"{ns}/{frame}"
                ir.radiation_type = Range.INFRARED
                ir.field_of_view = .035
                ir.min_range, ir.max_range = .10, .80
                result = sim.range_ray(index, pose[:3], beam, ir.max_range, robust=True)
                ir.range = result if result >= ir.min_range else -math.inf
                ports[side].publish(ir)

    def publish_rgb(self, frames, width, height, intrinsics):
        stamp = stamp_from(self.sim.time)
        published = set()
        for index, rgba in frames.items():
            if not isinstance(rgba, np.ndarray) or rgba.shape not in ((height,width,3), (height,width,4)):
                continue
            frame = f"puzzlebot_{index:02d}/camera_link_optical"
            img = Image()
            img.header.stamp, img.header.frame_id = stamp, frame
            img.height, img.width, img.encoding, img.step = height, width, "rgb8", width*3
            img.data = np.ascontiguousarray(rgba[..., :3], dtype=np.uint8).tobytes()
            info = CameraInfo()
            info.header = img.header
            info.width, info.height = width, height
            info.distortion_model, info.d = "plumb_bob", [0.0]*5
            fx, fy, cx, cy = intrinsics[index]
            info.k = [fx,0.,cx, 0.,fy,cy, 0.,0.,1.]
            info.r = [1.,0.,0., 0.,1.,0., 0.,0.,1.]
            info.p = [fx,0.,cx,0., 0.,fy,cy,0., 0.,0.,1.,0.]
            self.ports[index]["rgb"].publish(img)
            self.ports[index]["info"].publish(info)
            published.add(index)
        return published
