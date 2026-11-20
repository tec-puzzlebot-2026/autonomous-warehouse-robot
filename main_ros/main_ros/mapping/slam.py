"""
Team Zero custom 2D LiDAR SLAM v2 — ROS 2 node.

Front-end (real time, every scan)
  wheel odometry prediction -> likelihood-field scan matching against the
  LOCAL map -> particle weighting / systematic resampling -> corrected pose
  -> keyframes inserted into the local and global occupancy grids.
Back-end (background thread)
  loop detection by (x, y) proximity + correlative scan matching -> loop edges
  -> pose-graph optimization (Gauss-Newton / LM, robust kernel, outlier
  rejection) -> global map rebuilt from all keyframes -> swapped in.

All estimation code lives in mapping/slam_core (no ROS imports). This node only
handles messages, time alignment, TF and publishing.

Inputs
  /laser_scan   sensor_msgs/LaserScan   (frame: laser_frame)
  /wheel/odom   nav_msgs/Odometry       (our wheel odometry, odom -> base_link)
  TF base_link -> laser_frame           (LiDAR mounting; params as fallback)

Outputs
  /map              nav_msgs/OccupancyGrid            (transient local, 1 Hz)
  /slam/pose        geometry_msgs/PoseWithCovarianceStamped
  /slam/path        nav_msgs/Path                     (keyframe trajectory, corrected)
  /slam/graph       visualization_msgs/MarkerArray    (pose graph + loop closures)
  /slam/particles   geometry_msgs/PoseArray           (debug)
  /slam/scan        sensor_msgs/LaserScan             (filtered scan, debug)
  /slam/status      std_msgs/String                   (key=value CSV, 1 Hz, for the
                                                       program manager and monitor)
  TF map -> odom
Services
  /slam/reset       std_srvs/Trigger
  /slam/save_map    std_srvs/Trigger  (PGM + YAML in the save_dir parameter)
  /slam/optimize    std_srvs/Trigger  (force a pose-graph optimization + map rebuild)

Ground truth from Isaac Sim is never used here.
Run with use_sim_time:=true so every stamp is simulation time.
"""

import collections
import math
import os
import time

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       QoSHistoryPolicy, qos_profile_sensor_data)
from rclpy.time import Time

from geometry_msgs.msg import (Point, Pose, PoseArray, PoseStamped,
                               PoseWithCovarianceStamped, TransformStamped)
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros

from main_ros.mapping.slam_core import (HAVE_SCIPY, LoopClosureConfig, OdometryBuffer,
                                        ScanPreprocessor, SlamConfig, SlamSystem)
from main_ros.mapping.slam_core.se2 import (quaternion_from_yaw,
                                            rotation_matrix_from_quaternion,
                                            yaw_from_quaternion)


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def make_pose(x, y, yaw):
    p = Pose()
    p.position.x, p.position.y = float(x), float(y)
    qx, qy, qz, qw = quaternion_from_yaw(float(yaw))
    p.orientation.x, p.orientation.y = qx, qy
    p.orientation.z, p.orientation.w = qz, qw
    return p


def rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


class SLAM(Node):

    def __init__(self):
        super().__init__("slam")

        # ---------------- Parameters ----------------
        d = SlamConfig()
        dl = LoopClosureConfig()
        p = self.declare_parameter
        self.scan_topic = p("scan_topic", "/laser_scan").value
        self.odom_topic = p("odom_topic", "/wheel/odom").value
        self.map_frame = p("map_frame", "map").value
        self.odom_frame_param = p("odom_frame", "").value      # "" = from /wheel/odom
        self.base_frame_param = p("base_frame", "").value      # "" = from /wheel/odom
        self.laser_tf_timeout = p("laser_tf_timeout", 3.0).value
        self.laser_fallback = (p("laser_x", 0.0).value, p("laser_y", 0.0).value,
                               p("laser_yaw", 0.0).value)
        self.max_odom_gap = p("max_odom_gap", 0.15).value
        self.map_publish_period = p("map_publish_period", 1.0).value
        self.graph_publish_period = p("graph_publish_period", 1.0).value
        self.tf_publish_rate = p("tf_publish_rate", 20.0).value
        self.tf_post_date = p("tf_post_date", 0.1).value
        self.save_dir = p("save_dir", os.path.expanduser("~/maps")).value
        self.status_period = p("status_period", 1.0).value
        # Saved maps: cells with occupancy <= save_free_max are free (white).
        # The map holds probabilities: a cell seen free once is about 40.
        self.save_free_max = p("save_free_max", 50).value

        loop = LoopClosureConfig(
            enabled=p("loop_closure", dl.enabled).value,
            threaded=p("loop_threaded", dl.threaded).value,
            min_path_m=p("loop_min_path", dl.min_path_m).value,
            search_radius=p("loop_search_radius", dl.search_radius).value,
            max_candidates=p("loop_max_candidates", dl.max_candidates).value,
            window=p("loop_match_window", dl.window).value,
            window_angle=p("loop_match_window_angle", dl.window_angle).value,
            resolution=p("loop_match_resolution", dl.resolution).value,
            sigma=p("loop_match_sigma", dl.sigma).value,
            submap_path_m=p("loop_submap_path", dl.submap_path_m).value,
            submap_max_keyframes=p("loop_submap_max_keyframes", dl.submap_max_keyframes).value,
            min_score=p("loop_min_score", dl.min_score).value,
            min_inliers=p("loop_min_inliers", dl.min_inliers).value,
            check_every=p("loop_check_every", dl.check_every).value,
            max_backlog=p("loop_max_backlog", dl.max_backlog).value,
            sigma_xy_min=p("loop_sigma_xy_min", dl.sigma_xy_min).value,
            sigma_theta_min=p("loop_sigma_theta_min", dl.sigma_theta_min).value,
            huber_delta=p("loop_huber_delta", dl.huber_delta).value,
            outlier_chi2=p("loop_outlier_chi2", dl.outlier_chi2).value,
            trigger_xy=p("loop_trigger_xy", dl.trigger_xy).value,
            trigger_theta=p("loop_trigger_theta", dl.trigger_theta).value,
            max_pending_closures=p("loop_max_pending", dl.max_pending_closures).value,
            rebuild_min_correction=p("loop_rebuild_min_correction",
                                     dl.rebuild_min_correction).value,
        )
        cfg = SlamConfig(
            resolution=p("resolution", d.resolution).value,
            initial_map_size_m=p("initial_map_size_m", d.initial_map_size_m).value,
            l_free=p("l_free", d.l_free).value,
            l_occ=p("l_occ", d.l_occ).value,
            l_min=p("l_min", d.l_min).value,
            l_max=p("l_max", d.l_max).value,
            occupied_threshold=p("occupied_threshold", d.occupied_threshold).value,
            sigma_hit=p("sigma_hit", d.sigma_hit).value,
            z_hit=p("z_hit", d.z_hit).value,
            z_rand=p("z_rand", d.z_rand).value,
            field_max_dist=p("field_max_dist", d.field_max_dist).value,
            field_half_size_m=p("field_half_size_m", d.field_half_size_m).value,
            num_particles=p("num_particles", d.num_particles).value,
            alphas=tuple(p("motion_alphas", list(d.alphas)).value),
            beta=p("likelihood_beta", d.beta).value,
            resample_ratio=p("resample_ratio", d.resample_ratio).value,
            update_min_d=p("update_min_d", d.update_min_d).value,
            update_min_a=p("update_min_a", d.update_min_a).value,
            keyframe_d=p("keyframe_d", d.keyframe_d).value,
            keyframe_a=p("keyframe_a", d.keyframe_a).value,
            frontend_local_map=p("frontend_local_map", d.frontend_local_map).value,
            local_map_window_m=p("local_map_window", d.local_map_window_m).value,
            local_map_min_keyframes=p("local_map_min_keyframes", d.local_map_min_keyframes).value,
            local_map_max_keyframes=p("local_map_max_keyframes", d.local_map_max_keyframes).value,
            seq_sigma_along=p("seq_sigma_along", d.seq_sigma_along).value,
            seq_sigma_along_per_m=p("seq_sigma_along_per_m", d.seq_sigma_along_per_m).value,
            seq_sigma_cross=p("seq_sigma_cross", d.seq_sigma_cross).value,
            seq_sigma_cross_per_m=p("seq_sigma_cross_per_m", d.seq_sigma_cross_per_m).value,
            seq_sigma_theta=p("seq_sigma_theta", d.seq_sigma_theta).value,
            seq_sigma_theta_per_rad=p("seq_sigma_theta_per_rad", d.seq_sigma_theta_per_rad).value,
            seq_sigma_theta_per_m=p("seq_sigma_theta_per_m", d.seq_sigma_theta_per_m).value,
            seq_use_particle_cov=p("seq_use_particle_cov", d.seq_use_particle_cov).value,
            loop=loop,
        )
        self.pre = ScanPreprocessor(
            map_beam_step=p("map_beam_step", 4).value,
            match_beams=p("match_beams", 180).value,
            map_max_range=p("map_max_range", 10.0).value,
            match_max_range=p("match_max_range", 10.0).value,
            use_no_return_free_space=p("use_no_return_free_space", True).value,
            loop_voxel=p("loop_voxel", 0.08).value,
            loop_max_points=p("loop_max_points", 400).value,
        )

        # ---------------- Core ----------------
        self.slam = SlamSystem(cfg)
        self.odom_buffer = OdometryBuffer(history_s=2.0, max_gap_s=self.max_odom_gap)
        self.odom_frame = self.odom_frame_param or "odom"
        self.base_frame = self.base_frame_param or "base_link"
        self.laser_ready = False
        self.first_scan_wall = None
        self.stats = dict(scans=0, used=0, skipped_odom=0, proc_ms=[])
        self.last_log = 0.0
        self.last_result = None
        self.graph_dirty = False
        self.loop_count = 0
        self._reset_status()

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---------------- I/O ----------------
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)

        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST)
        self.map_pub = self.create_publisher(OccupancyGridMsg, "/map", latched)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/slam/pose", 10)
        self.path_pub = self.create_publisher(Path, "/slam/path", latched)
        self.graph_pub = self.create_publisher(MarkerArray, "/slam/graph", latched)
        self.particles_pub = self.create_publisher(PoseArray, "/slam/particles", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/slam/scan",
                                              qos_profile_sensor_data)
        self.status_pub = self.create_publisher(String, "/slam/status", 10)

        self.create_service(Trigger, "/slam/reset", self.on_reset)
        self.create_service(Trigger, "/slam/save_map", self.on_save_map)
        self.create_service(Trigger, "/slam/optimize", self.on_optimize)

        self.create_timer(self.map_publish_period, self.publish_map)
        self.create_timer(self.graph_publish_period, self.publish_graph_if_dirty)
        self.create_timer(1.0 / self.tf_publish_rate, self.publish_map_to_odom)
        self.create_timer(0.5, self.forward_backend_events)
        if self.status_period > 0.0:
            self.create_timer(self.status_period, self.publish_status)

        self.get_logger().info(
            f"Custom SLAM v2 ready | in: {self.scan_topic} + {self.odom_topic} | "
            f"out: /map /slam/pose /slam/path /slam/graph, TF {self.map_frame}->odom | "
            f"loop closure {'ON' if loop.enabled else 'OFF'}"
            f"{' (background thread)' if loop.threaded else ''} | "
            f"front-end map: {'local' if cfg.frontend_local_map else 'global'} | "
            f"use_sim_time={self.get_parameter('use_sim_time').value}")
        if loop.enabled and not HAVE_SCIPY:
            self.get_logger().warn(
                "SciPy not found: pose-graph optimization falls back to a dense solver "
                "(fine up to ~1200 keyframes). Install it with: sudo apt install python3-scipy")

    # ==================================================================
    # Callbacks
    # ==================================================================
    def on_odom(self, msg):
        if not self.odom_frame_param and msg.header.frame_id:
            self.odom_frame = msg.header.frame_id
        if not self.base_frame_param and msg.child_frame_id:
            self.base_frame = msg.child_frame_id
        q = msg.pose.pose.orientation
        pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                yaw_from_quaternion(q.x, q.y, q.z, q.w))
        t = stamp_to_sec(msg.header.stamp)
        if t == 0.0:
            self.get_logger().warn("/wheel/odom has a zero timestamp; check use_sim_time "
                                   "in the odometry node", throttle_duration_sec=5.0)
            return
        self.odom_buffer.add(t, pose)

    def on_scan(self, scan):
        self.stats["scans"] += 1
        self.last_scan_wall = time.monotonic()
        ranges = np.asarray(scan.ranges, dtype=np.float64)

        # Debug republish of the filtered scan (same angular structure).
        out = LaserScan()
        out.header, out.angle_min, out.angle_max = scan.header, scan.angle_min, scan.angle_max
        out.angle_increment, out.time_increment = scan.angle_increment, scan.time_increment
        out.scan_time, out.range_min, out.range_max = scan.scan_time, scan.range_min, scan.range_max
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        out.ranges = np.where(ok, ranges, np.inf).astype(np.float32).tolist()
        self.scan_pub.publish(out)

        if not self.laser_ready and not self.resolve_laser_mounting(scan.header.frame_id):
            return

        t = stamp_to_sec(scan.header.stamp)
        odom_pose = self.odom_buffer.pose_at(t)
        if odom_pose is None:
            self.stats["skipped_odom"] += 1
            last_t, _ = self.odom_buffer.latest()
            reason = ("no /wheel/odom received yet" if last_t is None else
                      f"no odometry at scan time (scan t={t:.3f}, newest odom t={last_t:.3f})")
            self.get_logger().warn(f"Skipping scan: {reason}", throttle_duration_sec=2.0)
            return

        t0 = time.perf_counter()
        prepared = self.pre.prepare(ranges, scan.angle_min, scan.angle_increment,
                                    scan.range_min, scan.range_max)
        if prepared.n_valid < 10:
            self.get_logger().warn("Scan has fewer than 10 valid beams; skipped",
                                   throttle_duration_sec=2.0)
            return
        res = self.slam.process(t, odom_pose, prepared)
        self.stats["proc_ms"].append((time.perf_counter() - t0) * 1000.0)
        del self.stats["proc_ms"][:-200]
        self.stats["used"] += 1
        self.last_result = res
        self.used_wall.append(time.monotonic())

        self.publish_pose(scan.header.stamp, res)
        if res.filter_updated:
            self.publish_particles(scan.header.stamp)
        if res.keyframe:
            self.graph_dirty = True
        if res.corrected:
            T = res.correction
            self.last_correction = (math.hypot(T[0], T[1]), math.degrees(T[2]),
                                    time.monotonic())
            self.forward_backend_events()      # log what the back-end did first
            self.get_logger().info(
                f"Loop-closure correction applied: map, trajectory and pose moved "
                f"(current pose shifted {math.hypot(T[0], T[1]) * 100:.1f} cm, "
                f"{math.degrees(T[2]):.1f} deg)")
            self.publish_map()
            self.publish_graph()

        self.log_status(prepared, res)

    def forward_backend_events(self):
        for line in self.slam.backend.drain_events():
            if line.startswith("loop closure"):
                self.loop_count += 1
            if "error" in line:
                self.get_logger().error(line)
            else:
                self.get_logger().info(line)

    # ==================================================================
    # LiDAR mounting
    # ==================================================================
    def resolve_laser_mounting(self, laser_frame):
        if self.first_scan_wall is None:
            self.first_scan_wall = time.monotonic()
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, laser_frame, Time())
            tr, q = tf.transform.translation, tf.transform.rotation
            R3 = rotation_matrix_from_quaternion(q.x, q.y, q.z, q.w)
            if abs(R3[2, 2]) < 0.95:
                self.get_logger().warn(f"{laser_frame} is tilted relative to {self.base_frame}; "
                                       "2D projection may be inaccurate")
            self.pre.set_mounting(R3[:2, :2], (tr.x, tr.y))
            self.get_logger().info(
                f"LiDAR mounting from TF {self.base_frame}->{laser_frame}: "
                f"x={tr.x:.3f} y={tr.y:.3f} yaw={math.degrees(math.atan2(R3[1, 0], R3[0, 0])):.1f} deg"
                + (" (upside down: scan mirrored)" if R3[2, 2] < 0 else ""))
            self.laser_ready = True
            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            if time.monotonic() - self.first_scan_wall < self.laser_tf_timeout:
                self.get_logger().info(f"Waiting for TF {self.base_frame}->{laser_frame}...",
                                       throttle_duration_sec=1.0)
                return False
            x, y, yaw = self.laser_fallback
            c, s = math.cos(yaw), math.sin(yaw)
            self.pre.set_mounting([[c, -s], [s, c]], (x, y))
            self.get_logger().warn(
                f"TF {self.base_frame}->{laser_frame} unavailable ({e}). Using parameters "
                f"laser_x={x}, laser_y={y}, laser_yaw={yaw}")
            self.laser_ready = True
            return True

    # ==================================================================
    # Publishing
    # ==================================================================
    def publish_map_to_odom(self):
        if not self.slam.initialized:
            return
        corr = self.slam.map_to_odom()
        msg = TransformStamped()
        msg.header.stamp = (self.get_clock().now()
                            + Duration(seconds=self.tf_post_date)).to_msg()
        msg.header.frame_id = self.map_frame
        msg.child_frame_id = self.odom_frame
        msg.transform.translation.x, msg.transform.translation.y = float(corr[0]), float(corr[1])
        qx, qy, qz, qw = quaternion_from_yaw(float(corr[2]))
        msg.transform.rotation.x, msg.transform.rotation.y = qx, qy
        msg.transform.rotation.z, msg.transform.rotation.w = qz, qw
        self.tf_broadcaster.sendTransform(msg)

    def publish_pose(self, stamp, res):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp, msg.header.frame_id = stamp, self.map_frame
        msg.pose.pose = make_pose(*res.pose)
        cov = np.zeros((6, 6))
        cov[0:2, 0:2] = res.cov[0:2, 0:2]
        cov[0:2, 5] = res.cov[0:2, 2]
        cov[5, 0:2] = res.cov[2, 0:2]
        cov[5, 5] = res.cov[2, 2]
        msg.pose.covariance = cov.ravel().tolist()
        self.pose_pub.publish(msg)

    def publish_particles(self, stamp):
        msg = PoseArray()
        msg.header.stamp, msg.header.frame_id = stamp, self.map_frame
        msg.poses = [make_pose(*pt) for pt in self.slam.pf.particles]
        self.particles_pub.publish(msg)

    def publish_map(self):
        if not self.slam.initialized:
            return
        values, origin = self.slam.grid.to_occupancy_values(crop=True)
        self.map_info = (values.shape[1], values.shape[0],
                         float(np.count_nonzero(values >= 0)) * self.slam.grid.res ** 2)
        msg = OccupancyGridMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution = float(self.slam.grid.res)
        msg.info.height, msg.info.width = int(values.shape[0]), int(values.shape[1])
        msg.info.origin = make_pose(origin[0], origin[1], 0.0)
        msg.data = values.ravel().tolist()
        self.map_pub.publish(msg)

    def publish_graph_if_dirty(self):
        if self.graph_dirty:
            self.publish_graph()

    def publish_graph(self):
        """Corrected keyframe trajectory (/slam/path) and pose graph (/slam/graph)."""
        if not self.slam.initialized:
            return
        self.graph_dirty = False
        poses, edges = self.slam.graph_snapshot()
        stamp = self.get_clock().now().to_msg()

        path = Path()
        path.header.stamp, path.header.frame_id = stamp, self.map_frame
        for pose in poses:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose = make_pose(*pose)
            path.poses.append(ps)
        self.path_pub.publish(path)

        def pt(i):
            q = Point()
            q.x, q.y, q.z = float(poses[i, 0]), float(poses[i, 1]), 0.05
            return q

        nodes = Marker()
        nodes.header = path.header
        nodes.ns, nodes.id, nodes.type, nodes.action = "keyframes", 0, Marker.POINTS, Marker.ADD
        nodes.scale.x = nodes.scale.y = 0.06
        nodes.color = rgba(0.2, 0.4, 1.0)
        nodes.pose.orientation.w = 1.0
        nodes.points = [pt(i) for i in range(poses.shape[0])]

        seq = Marker()
        seq.header = path.header
        seq.ns, seq.id, seq.type, seq.action = "odometry_edges", 1, Marker.LINE_LIST, Marker.ADD
        seq.scale.x = 0.015
        seq.color = rgba(0.6, 0.6, 0.6, 0.8)
        seq.pose.orientation.w = 1.0

        loops = Marker()
        loops.header = path.header
        loops.ns, loops.id, loops.type, loops.action = "loop_closures", 2, Marker.LINE_LIST, Marker.ADD
        loops.scale.x = 0.03
        loops.color = rgba(1.0, 0.1, 0.6)
        loops.pose.orientation.w = 1.0
        for i, j, kind in edges:
            target = loops if kind == "loop" else seq
            target.points += [pt(i), pt(j)]
        self.graph_pub.publish(MarkerArray(markers=[nodes, seq, loops]))

    def log_status(self, prepared, res):
        now = time.monotonic()
        if now - self.last_log < 2.0:
            return
        ms = self.stats["proc_ms"][-50:]
        x, y, th = res.pose
        sx, sy = math.sqrt(max(res.cov[0, 0], 0)), math.sqrt(max(res.cov[1, 1], 0))
        ratio = "n/a" if math.isnan(res.match_ratio) else f"{res.match_ratio * 100:.0f}%"
        self.get_logger().info(
            f"pose x={x:.2f} y={y:.2f} th={math.degrees(th):.1f}deg (sd {sx:.3f},{sy:.3f} m) | "
            f"beams {prepared.n_valid}/{prepared.n_total} | match {ratio} | "
            f"N_eff {res.n_eff:.0f} | keyframes {len(self.slam.keyframes)} | "
            f"loops {self.loop_count} | corrections {self.slam.corrections} | "
            f"{np.mean(ms):.1f} ms/scan | scans used {self.stats['used']}/{self.stats['scans']}")
        self.last_log = now

    # ==================================================================
    # Services
    # ==================================================================
    def on_reset(self, request, response):
        self.slam.reset()
        self.loop_count = 0
        self.graph_dirty = True
        self._reset_status()
        response.success, response.message = True, "SLAM reset; next scan starts a new map"
        self.get_logger().info(response.message)
        return response

    def on_optimize(self, request, response):
        if not self.slam.initialized:
            response.success, response.message = False, "Map is empty"
            return response
        self.slam.request_optimization()
        response.success = True
        response.message = "Optimization requested; the corrected map is applied when ready"
        return response

    def on_save_map(self, request, response):
        if not self.slam.initialized:
            response.success, response.message = False, "Map is empty"
            return response
        values, origin = self.slam.grid.to_occupancy_values(crop=True)
        os.makedirs(self.save_dir, exist_ok=True)
        name = time.strftime("slam_map_%Y%m%d_%H%M%S")
        pgm, yml = (os.path.join(self.save_dir, name + ext) for ext in (".pgm", ".yaml"))
        # map_server convention: 0 = occupied (black), 254 = free, 205 = unknown.
        img = np.full(values.shape, 205, dtype=np.uint8)
        img[(values >= 0) & (values <= self.save_free_max)] = 254
        img[values >= 65] = 0
        img = np.flipud(img)   # PGM rows go top-down; grid row 0 is the bottom
        with open(pgm, "wb") as f:
            f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
            f.write(img.tobytes())
        with open(yml, "w") as f:
            f.write(f"image: {name}.pgm\nmode: trinary\nresolution: {self.slam.grid.res}\n"
                    f"origin: [{origin[0]:.4f}, {origin[1]:.4f}, 0.0]\nnegate: 0\n"
                    f"occupied_thresh: 0.65\nfree_thresh: 0.25\n")
        response.success, response.message = True, f"Saved {pgm} and {yml}"
        self.last_save = os.path.join(self.save_dir, name)
        self.get_logger().info(response.message)
        return response

    # ==================================================================
    # Status for the program manager / monitor
    # ==================================================================
    def _reset_status(self):
        """Forget the counters shown on /slam/status (start / reset)."""
        self.last_scan_wall = None
        self.used_wall = collections.deque(maxlen=200)
        self.last_correction = None
        self.map_info = None
        self.last_save = ""
        self.stats.update(scans=0, used=0, skipped_odom=0, proc_ms=[])

    def slam_state(self):
        """One word: what the SLAM is doing right now."""
        now = time.monotonic()
        if self.last_scan_wall is None:
            return "WAITING_FOR_SCANS"
        if not self.laser_ready:
            return "WAITING_FOR_LASER_TF"
        if not self.used_wall:
            return "WAITING_FOR_ODOM" if self.stats["skipped_odom"] else "STARTING"
        if now - self.used_wall[-1] > 2.0:
            return "STALLED"
        if getattr(self.slam.backend, "busy", False):
            return "OPTIMIZING"
        return "RUNNING"

    def publish_status(self):
        """Publish /slam/status: flat key=value pairs (like Frontera's stats)."""
        now = time.monotonic()
        recent = [t for t in self.used_wall if now - t <= 2.0]
        rate = len(recent) / 2.0
        kv = [("state", self.slam_state()),
              ("scan_hz", f"{rate:.1f}"),
              ("scans_used", self.stats["used"]),
              ("scans_total", self.stats["scans"]),
              ("skipped_odom", self.stats["skipped_odom"]),
              ("keyframes", len(self.slam.keyframes) if self.slam.initialized else 0),
              ("loops", self.loop_count),
              ("corrections", self.slam.corrections),
              ("backend_busy", int(bool(getattr(self.slam.backend, "busy", False))))]
        ms = self.stats["proc_ms"][-50:]
        if ms:
            kv.append(("proc_ms", f"{np.mean(ms):.1f}"))
        if self.last_correction is not None:
            d, a, t = self.last_correction
            kv += [("last_correction_cm", f"{d * 100.0:.1f}"),
                   ("last_correction_deg", f"{a:.1f}"),
                   ("last_correction_age_s", f"{now - t:.0f}")]
        if self.map_info is not None:
            w, h, area = self.map_info
            kv += [("map_w", w), ("map_h", h),
                   ("res", f"{self.slam.grid.res:.3f}"), ("known_m2", f"{area:.1f}")]
        res = self.last_result
        if res is not None:
            x, y, th = res.pose
            sd = math.sqrt(max(res.cov[0, 0], 0.0) + max(res.cov[1, 1], 0.0))
            kv += [("x", f"{x:.2f}"), ("y", f"{y:.2f}"),
                   ("yaw_deg", f"{math.degrees(th):.0f}"), ("sd_xy", f"{sd:.3f}"),
                   ("n_eff", f"{res.n_eff:.0f}")]
            if not math.isnan(res.match_ratio):
                kv.append(("match_pct", f"{res.match_ratio * 100.0:.0f}"))
        if self.last_save:
            kv.append(("last_save", self.last_save))
        self.status_pub.publish(String(data=",".join(f"{k}={v}" for k, v in kv)))

    def destroy_node(self):
        self.slam.shutdown()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SLAM()
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
