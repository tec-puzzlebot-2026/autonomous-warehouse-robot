#!/usr/bin/env python3
"""
landmark_mapper_node.py — put the ArUco markers and company logos on the SLAM map

Runs while the robot maps (Phase 1) and during a Phase 2 survey. Every sighting is
placed on the map with the robot's SLAM pose at that moment; many sightings per
marker / logo are averaged robustly (restock/landmarks.py).

  ArUco marker: the detector gives the marker's position in the camera's optical
      frame; the measured camera mount (camera_mount in aruco_localization.yaml,
      the same one Chris's EKF uses) moves it to base_link, the SLAM pose to the map.
  Logo: the logo detector gives the logo's image position; its bearing, cast from
      the camera into the current map, hits the wall the logo hangs on. The
      direction back toward the robot is kept as the wall-normal hint for the dock
      search (restock/docks.py).

Subscribes  /aruco/detections (JSON), /logo_detector/detections (JSON),
            /slam/pose, /map, camera_info_topic
Publishes   ~/landmarks (std_msgs/String, JSON, 1 Hz):
              {"markers": [{"id", "x", "y", "yaw", "sightings", "spread_m"}],
               "logos":   [{"id": company, "x", "y", "yaw", "sightings", "spread_m"}]}
            ~/zones (visualization_msgs/MarkerArray, latched): a size_m square in
              front of every marker, labeled by its ID (ID 21: the DROP ZONE;
              restock/zones.py, settings in restock.yaml, section zones)
Services    ~/reset (std_srvs/Trigger): forget everything (a new map was started)

The Program Manager reads ~/landmarks when it saves the mission folder.
"""
# Author: Raymundo Gomez

import json
import math
import os
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from main_ros.restock.geometry import Camera, transform_point, wrap
from main_ros.restock.grid import OccupancyMap
from main_ros.restock.landmarks import LandmarkSet, marker_yaw_from_optical
from main_ros.restock.zone_viz import zone_markers
from main_ros.restock.zones import ZoneConfig, zones_from_markers


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def load_mount(aruco_config):
    """camera_mount (x, y, z, roll, pitch, yaw) and base height from aruco_localization.yaml."""
    with open(os.path.expanduser(aruco_config)) as f:
        cfg = yaml.safe_load(f) or {}
    m = (cfg.get('camera_mount') or {}).get('ros__parameters', {})
    mount = tuple(float(m.get(k, 0.0)) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'))
    exp = (cfg.get('aruco_experiment') or {}).get('ros__parameters', {})
    return mount, float(exp.get('base_link_height', 0.047))


class LandmarkMapper(Node):
    def __init__(self):
        super().__init__('landmark_mapper')
        p = self.declare_parameter
        aruco_cfg = p('aruco_config', '').value
        mount = tuple(float(v) for v in p('camera_mount',
                                          [0.10005, 0.0, 0.178, 0.0, 0.26, -0.0124]).value)
        base_h = float(p('base_link_height', 0.047).value)
        if aruco_cfg and os.path.isfile(os.path.expanduser(aruco_cfg)):
            mount, base_h = load_mount(aruco_cfg)
        self.mount, self.base_h = mount, base_h
        self.max_marker_range = float(p('max_marker_range', 2.5).value)
        self.max_view_angle = float(p('max_view_angle_deg', 60.0).value)
        self.max_logo_range = float(p('max_logo_range', 3.0).value)
        self.max_pose_age = float(p('max_pose_age_s', 0.3).value)
        self.max_turn_rate = float(p('max_turn_rate', 0.5).value)
        self.min_logo_conf = float(p('min_logo_conf', 0.6).value)
        self.markers = LandmarkSet(int(p('min_marker_sightings', 5).value), 0.15)
        self.logos = LandmarkSet(int(p('min_logo_sightings', 3).value), 0.25)
        self.cam = None
        self.pose = None
        self.pose_t = None
        self.turn_rate = 0.0
        self.grid = None
        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseWithCovarianceStamped, p('pose_topic', '/slam/pose').value,
                                 self.on_pose, 20)
        self.create_subscription(OccupancyGrid, p('map_topic', '/map').value, self.on_map, latched)
        self.create_subscription(CameraInfo, p('camera_info_topic', '/camera/camera_info').value,
                                 self.on_info, 5)
        self.create_subscription(String, p('aruco_topic', '/aruco/detections').value,
                                 self.on_aruco, 20)
        self.create_subscription(String, p('logo_topic', '/logo_detector/detections').value,
                                 self.on_logos, 20)
        self.pub = self.create_publisher(String, '~/landmarks', 10)
        restock_cfg = p('restock_config', '').value
        self.zone_cfg = ZoneConfig()
        if restock_cfg and os.path.isfile(os.path.expanduser(restock_cfg)):
            with open(os.path.expanduser(restock_cfg)) as f:
                self.zone_cfg = ZoneConfig.from_dict((yaml.safe_load(f) or {}).get('zones'))
        self.zone_pub = self.create_publisher(MarkerArray, '~/zones', latched)
        self.zones = []
        self.zone_key = None            # (markers rounded, time) of the last zone update
        self.zone_t = 0.0
        self.create_service(Trigger, '~/reset', self.on_reset)
        self.create_timer(1.0, self.publish)
        self.get_logger().info(f'landmark mapper: camera mount {self.mount}')

    # ------------------------------------------------------------------ inputs
    def on_info(self, msg):
        if self.cam is None:
            self.cam = Camera.from_camera_info(list(msg.k), self.mount, self.base_h, list(msg.p))

    def on_map(self, msg):
        o = msg.info.origin.position
        self.grid = OccupancyMap.from_msg_fields(msg.info.width, msg.info.height,
                                                 msg.info.resolution, o.x, o.y, msg.data)

    def on_pose(self, msg):
        p = msg.pose.pose
        new = (p.position.x, p.position.y, yaw_of(p.orientation))
        now = time.monotonic()
        if self.pose is not None and self.pose_t is not None and now > self.pose_t:
            self.turn_rate = abs(wrap(new[2] - self.pose[2])) / (now - self.pose_t)
        self.pose, self.pose_t = new, now

    def _pose_ok(self):
        return (self.pose is not None and time.monotonic() - self.pose_t <= self.max_pose_age
                and self.turn_rate <= self.max_turn_rate)

    def on_aruco(self, msg):
        if not self._pose_ok():
            return
        try:
            dets = json.loads(msg.data)
        except ValueError:
            return
        T = Camera(1.0, 1.0, 0.0, 0.0, self.mount, self.base_h).T
        for d in dets:
            try:
                dist = float(d.get('distance', 99.0))
                view = abs(float(d.get('view_angle_deg', 0.0)))
                if dist > self.max_marker_range or view > self.max_view_angle:
                    continue
                p_base = T @ np.array(list(d['t']) + [1.0])
                x, y = transform_point(self.pose, (p_base[0], p_base[1]))
                yaw = marker_yaw_from_optical(d['q'], T, self.pose[2]) if d.get('q') else None
                self.markers.add(int(d['id']), x, y, yaw)
            except (KeyError, TypeError, ValueError):
                continue

    def on_logos(self, msg):
        if not self._pose_ok() or self.cam is None or self.grid is None:
            return
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        cam_xy = transform_point(self.pose, (self.cam.T[0, 3], self.cam.T[1, 3]))
        for d in data.get('detections', []):
            if float(d.get('conf', 0.0)) < self.min_logo_conf:
                continue
            u, v = (float(c) for c in d['center'][:2])
            bearing = wrap(self.pose[2] + self.cam.bearing(u, v))
            hit = self.grid.first_hit(cam_xy, bearing, self.max_logo_range)
            if hit is None:
                continue
            view_yaw = wrap(bearing + math.pi)          # from the logo back toward the robot
            self.logos.add(str(d['company']), hit[0], hit[1], view_yaw)

    # ------------------------------------------------------------------ outputs
    def snapshot(self):
        return {'markers': self.markers.to_list(), 'logos': self.logos.to_list(),
                'pending': {'markers': {str(k): v.count for k, v in self.markers.items.items()},
                            'logos': {k: v.count for k, v in self.logos.items.items()}}}

    def publish(self):
        snap = self.snapshot()
        self.pub.publish(String(data=json.dumps(snap)))
        self.update_zones(snap['markers'])

    def update_zones(self, markers):
        """Zones of the markers found so far: recomputed when a marker moves or every 10 s
        (the map grows), published only when they change."""
        if self.grid is None:
            return
        key = tuple(sorted((int(m['id']), round(m['x'], 2), round(m['y'], 2)) for m in markers))
        now = time.monotonic()
        if key == self.zone_key and now - self.zone_t < 10.0:
            return
        self.zone_key, self.zone_t = key, now
        zones = zones_from_markers(self.grid, markers, self.zone_cfg)
        new = [z.to_dict() for z in zones]
        if new == [z.to_dict() for z in self.zones] and self.zones:
            return
        for z in zones:
            if z.label not in [o.label for o in self.zones]:
                self.get_logger().info(f'zone {z.label} (ArUco {z.marker_id}) at '
                                       f'({z.center[0]:.2f}, {z.center[1]:.2f}), '
                                       f'{100 * z.inside:.0f} % inside the map')
        self.zones = zones
        self.zone_pub.publish(zone_markers(zones, self.get_clock().now().to_msg()))

    def on_reset(self, request, response):
        self.markers.items.clear()
        self.logos.items.clear()
        self.zones, self.zone_key = [], None
        self.zone_pub.publish(zone_markers([], self.get_clock().now().to_msg()))
        response.success, response.message = True, 'landmarks cleared'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkMapper()
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
