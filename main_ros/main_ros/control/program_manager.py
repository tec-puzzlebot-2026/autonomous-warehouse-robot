#!/usr/bin/env python3
# =============================================================================
# program_manager.py — Team Zero warehouse robot: Program Manager
# =============================================================================
# The Program Manager decides; every other script acts. It
#   * starts the robot's scripts one by one and waits until each is ready
#     (readiness gates, like the CALL-M program manager),
#   * runs the mission state machine (mission_core.py),
#   * sends short commands to Frontera (/exploration/control) and Sendero
#     (/navigation/control), and calls the SLAM services,
#   * watches every script and halts the robot if one dies,
#   * publishes one compact status for the monitor on the host PC.
# It never computes a map, a path or a velocity (except a zero velocity when
# it halts the robot).
#
# Everything runs on one machine (the Spark container in simulation, the
# Jetson on the robot). The monitor and RViz / Foxglove run on the host PC
# and talk to it over ROS 2 (DDS).
#
# Start:   ros2 launch main_ros main.launch.py mode:=sim      (or mode:=real)
#          map:=<mission folder | latest>  start from a saved map (skips stage 1)
# Control: the monitor (ros2 run main_ros mission_monitor, on the host), or
#          ros2 topic pub --once /mission/command std_msgs/String "{data: S}"
#
# Stages (mission_core.py): 1 mapping, 2 map quality, 3 route, 4 restocking.
# After each stage the robot stops and the monitor waits for a key.
# Commands (keys of the monitor):
#   S start: bring up / start the stage / next stage / resume
#   P pause            F finish the stage now (map, map tour)    K skip stage 2
#   M save the map     N reset (new mapping run, stage 1)        X stop (close all)
#
# Topics:
#   in : /mission/command (String)
#        /exploration/status, /exploration/stats, /exploration/map_save_request
#        /navigation/status, /navigation/stats, /slam/status
#   out: /mission/state (String, one word), /mission/status (String, JSON),
#        /mission/events (String, one line per event)
#        /exploration/control, /navigation/control, /exploration/map_save_status
#        /cmd_vel (zero only, when halting), /restock/status (Phase 2, JSON)
#
# Logs: main_ros/logs/program_manager_<mode>_<date>_<time>/  manager.txt, all.txt,
#       <script>.txt,
#       shutdown_report.txt
# =============================================================================
# Author: Raymundo Gomez

import json
import math
import os
import signal
import threading
import time

import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger

from main_ros.control import mission_core as mc
from main_ros.control.process_supervisor import ProcessSupervisor, run_log_dir
from main_ros.restock import mission_store as ms

try:
    from rosgraph_msgs.msg import Clock
except ImportError:  # pragma: no cover - always present in a ROS 2 install
    Clock = None

try:
    from rosidl_runtime_py.utilities import get_message
except ImportError:  # pragma: no cover
    get_message = None

# Fields of the Frontera / Sendero statistics forwarded to the monitor.
EXPLORER_KEYS = ('reached', 'failed', 'blacklist_count', 'frontier_count',
                 'map_known_area_m2', 'robot_x', 'robot_y', 'target_x',
                 'target_y', 'dist_target_m', 'dist_travelled_m')
NAVIGATION_KEYS = ('task', 'profile', 'distance_remaining', 'recoveries',
                   'safety', 'closest_obstacle_m', 'cmd_vx', 'cmd_wz',
                   'scan_age_s', 'last_plan', 'last_plan_ms')


def parse_kv(text):
    """'a=1,b=x' -> {'a': '1', 'b': 'x'}."""
    out = {}
    for part in str(text).split(','):
        if '=' in part:
            k, _, v = part.partition('=')
            out[k.strip()] = v.strip()
    return out


def find_config_dir():
    """The main_ros config directory: installed share first, then the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        d = os.path.join(get_package_share_directory('main_ros'), 'config')
        if os.path.isfile(os.path.join(d, 'program_manager.yaml')):
            return d
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.normpath(os.path.join(here, '..', '..', 'config'))
    return d


class _Job:
    """A function run in a background thread: done() and result() -> (ok, msg)."""

    def __init__(self):
        self._result = None

    def run(self, fn):
        def target():
            try:
                self._result = fn()
            except Exception as exc:              # noqa: BLE001 (reported to the operator)
                self._result = (False, f'{type(exc).__name__}: {exc}')
        threading.Thread(target=target, daemon=True).start()

    def done(self):
        return self._result is not None

    def result(self):
        return self._result or (False, 'not finished')

    def cancel(self):
        pass


class _ServiceHandle:
    """Result of an asynchronous Trigger call: done() and result() -> (ok, msg)."""

    def __init__(self, future):
        self._future = future

    def done(self):
        return self._future.done()

    def result(self):
        try:
            r = self._future.result()
            return bool(r.success), str(r.message)
        except Exception as exc:
            return False, str(exc)


def default_models_dir():
    """main_ros/main_ros/vision/models in the source tree.

    The models are not installed (too big, not in git); with --symlink-install the
    installed main_ros package links back to the source, so its real path finds them.
    """
    import main_ros
    return os.path.join(os.path.dirname(os.path.realpath(main_ros.__file__)), 'vision', 'models')


class ProgramManager(Node):
    """ROS 2 node: the ports of MissionCore plus status publishing."""

    def __init__(self):
        super().__init__('program_manager')
        self.mode = self.declare_parameter('mode', 'sim').value
        cfg_file = self.declare_parameter('config_file', '').value
        auto = self.declare_parameter('autostart', False).value
        self.autostart = str(auto).strip().lower() in ('true', '1', 'yes')
        map_arg = str(self.declare_parameter('map', '').value or '').strip()
        start_pose = str(self.declare_parameter('start_pose', '').value or '').strip()
        restock_only = str(self.declare_parameter('restock_only', False).value
                           ).strip().lower() in ('true', '1', 'yes')
        if self.mode not in ('sim', 'real'):
            raise ValueError(f'mode must be "sim" or "real", got "{self.mode}"')
        self.config_dir = os.path.dirname(cfg_file) if cfg_file else find_config_dir()
        cfg_file = cfg_file or os.path.join(self.config_dir, 'program_manager.yaml')
        with open(cfg_file) as f:
            self.cfg = yaml.safe_load(f)
        mode_cfg = self.cfg['modes'][self.mode]
        self.placeholders = {
            'mode': self.mode,
            'config_dir': self.config_dir,
            'use_sim_time': 'true' if mode_cfg.get('use_sim_time', False) else 'false',
            'maps_dir': os.path.expanduser(str(mode_cfg.get('maps_dir', '~/maps'))),
            'mission_dir': '',
            'slam_args': '',
        }
        extra = dict(mode_cfg.get('placeholders') or {})
        self.placeholders['models_dir'] = os.path.expanduser(
            str(extra.pop('models_dir', '') or default_models_dir()))
        for key, value in extra.items():
            value = os.path.expanduser(str(value))
            if key.endswith('_model') and not os.path.isabs(value):
                value = os.path.join(self.placeholders['models_dir'], value)
            self.placeholders[key] = value
        restock_file = os.path.join(self.config_dir, 'restock.yaml')
        self.restock_cfg = {}
        if os.path.isfile(restock_file):
            with open(restock_file) as f:
                self.restock_cfg = yaml.safe_load(f) or {}
        self.store = ms.MissionStore(os.path.join(self.placeholders['maps_dir'], 'missions'))
        self.mission_folder = None
        self.start_folder = None
        if map_arg:                                  # start from a saved map
            self.start_folder = self.store.resolve('' if map_arg == 'latest' else map_arg)
            self._use_mission_folder(self.start_folder)
            session = os.path.join(self.start_folder, ms.SESSION)
            # Planisferio loads the map and keeps mapping (stage 2 can still improve it).
            self.placeholders['slam_args'] = f'-p load_session:={session}'
        self.landmarks_json = None
        self.restock_ports = None
        self._plan_jobs = []
        self.log_dir = run_log_dir(f'program_manager_{self.mode}',
                                   str(mode_cfg.get('log_root', '') or ''))
        os.makedirs(self.placeholders['maps_dir'], exist_ok=True)
        self.placeholders['log_dir'] = self.log_dir
        self._mlog = open(os.path.join(self.log_dir, 'manager.txt'), 'a', buffering=1)
        if self.start_folder:
            self._log_line(f'map:={map_arg} -> {self.start_folder}'
                           + (f', start_pose {start_pose}' if start_pose else '')
                           + (', restock_only' if restock_only else ''))

        self.steps = self._load_steps()
        self.sup = ProcessSupervisor(self.log_dir, logger=self._log_line)
        topics = self.cfg['topics']
        self.topics = topics

        # ── publishers ───────────────────────────────────────────────────
        self.state_pub = self.create_publisher(String, topics['state'], 10)
        self.status_pub = self.create_publisher(String, topics['status'], 10)
        self.events_pub = self.create_publisher(String, topics['events'], 50)
        self.restock_pub = self.create_publisher(String, topics.get('restock_status',
                                                                    '/restock/status'), 10)
        self.explorer_pub = self.create_publisher(String, topics['explorer_control'], 10)
        self.nav_pub = self.create_publisher(String, topics['navigation_control'], 10)
        self.map_status_pub = self.create_publisher(String, topics['map_save_status'], 10)
        self.cmd_pub = self.create_publisher(Twist, topics['cmd_vel'], 10)

        # ── what the other scripts report ────────────────────────────────
        self.reports = {}
        for key in ('explorer_status', 'explorer_stats', 'navigation_status',
                    'navigation_stats', 'slam_status'):
            self.create_subscription(
                String, topics[key], lambda m, k=key: self._on_report(k, m), 10)
        self.create_subscription(String, topics['command'], self._on_command, 10)
        # Sendero's decisions (planned, blocked, recovery, aborted ...) in manager.txt only,
        # so a navigation problem can be read afterwards without a second terminal.
        self.create_subscription(String, topics.get('navigation_events', '/navigation/events'),
                                 self._on_nav_event, 50)
        self.create_subscription(String, topics['map_save_request'],
                                 lambda m: self.core.on_map_save_request(), 10)
        self.create_subscription(String, topics.get('landmarks', '/landmark_mapper/landmarks'),
                                 self._on_landmarks, 5)

        # ── readiness probes ─────────────────────────────────────────────
        self._msg_probes = {}           # topic -> {'seen', 'sub'}
        self._ready_steps = set()       # names of steps found ready (sticky while alive)
        self._graph_cache = (0.0, set(), set(), set())
        self._sim_clock = {'first': None, 'last': None, 'wall': 0.0}
        if Clock is not None:
            self.create_subscription(Clock, '/clock', self._on_clock,
                                     qos_profile_sensor_data)

        # ── SLAM services ────────────────────────────────────────────────
        self.clients_ = {key: self.create_client(Trigger, name)
                         for key, name in self.cfg['services'].items()}

        m = self.cfg.get('mission', {})
        settings = mc.MissionSettings(**{k: v for k, v in m.items()
                                         if k in mc.MissionSettings.__dataclass_fields__})
        if self.start_folder:                       # testing restocking from a saved map
            settings.start_pose = start_pose or settings.start_pose
            settings.restock_only = restock_only or settings.restock_only
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                                      '/initialpose', 1)
        self.core = mc.MissionCore(self.steps, self, settings, self.mode, self.start_folder)

        self.create_timer(1.0 / float(self.cfg.get('tick_hz', 10.0)), self._tick)
        self.create_timer(1.0 / float(self.cfg.get('status_hz', 2.0)), self._publish_status)
        self._last_state = None

        self._log_line('=' * 64)
        self._log_line(f'Program Manager — {self.mode.upper()} mode')
        if self.start_folder:
            self._log_line(f'saved map: {self.start_folder} (stage 1 skipped)')
        self._log_line(f'config: {cfg_file}')
        self._log_line(f'logs  : {self.log_dir}')
        for stage in mc.STAGES:
            names = [s.name for s in self.steps if s.stage == stage]
            if names:
                self._log_line(f'{stage:<11}: ' + ' -> '.join(names))
        self._log_line('Send S (monitor or /mission/command) to bring the system up.')
        self._log_line('=' * 64)
        if self.autostart:
            self.core.command('S')

    # =====================================================================
    # configuration
    # =====================================================================
    def _load_steps(self):
        """Steps of this mode, with {placeholders} filled in."""
        steps = []
        for raw in self.cfg['steps']:
            if self.mode not in raw.get('modes', ['sim', 'real']):
                continue
            cmd = str(raw.get('cmd', '') or '').strip()
            try:
                cmd.format(**self.placeholders)        # filled in again at launch time
            except KeyError as exc:
                raise ValueError(f'step {raw["name"]}: unknown placeholder {exc}') from exc
            if raw.get('todo'):
                cmd = ''
            ready = raw.get('ready', [])
            if isinstance(ready, dict):
                ready = [ready]
            steps.append(mc.Step(
                name=raw['name'], cmd=cmd, ready=ready,
                timeout=float(raw.get('timeout_s', 60.0)),
                critical=bool(raw.get('critical', True)),
                role=str(raw.get('role', '')),
                hint=str(raw.get('hint', '')),
                stage=str(raw.get('stage', 'core')),
                with_previous=bool(raw.get('with_previous', False))))
            if raw.get('todo'):
                steps[-1].hint = (f'This {self.mode} step is not configured yet '
                                  '(todo: true in program_manager.yaml). ' + steps[-1].hint)
        return steps

    # =====================================================================
    # ports used by MissionCore
    # =====================================================================
    def now(self):
        return time.monotonic()

    def prepare_bringup(self):
        """Before a bring-up from scratch, kill leftovers of an earlier run."""
        if any(self.sup.alive(s.name) for s in self.steps):
            return
        killed = self.sup.kill_stale(self.cfg.get('stale_process_patterns', []))
        for pid, cmdline in killed:
            self._log_line(f'killed leftover process {pid}: {cmdline[:120]}')

    def launch(self, step):
        # Placeholders are filled in now: {mission_dir} exists only after mapping.
        # A (re)started step must prove it is ready again with fresh messages.
        self._ready_steps.discard(step.name)
        for check in step.ready:
            (kind, value), = check.items()
            if kind == 'message':
                self._drop_probe(value)
        return self.sup.start(step.name, ' '.join(step.cmd.format(**self.placeholders).split()))

    def alive(self, step):
        return True if step.external else self.sup.alive(step.name)

    def was_launched(self, step):
        return self.sup.was_launched(step.name)

    def exit_code(self, step):
        return self.sup.exit_code(step.name)

    def log_path(self, step):
        return self.sup.log_path(step.name)

    def log_tail(self, step, n):
        """Last n non-empty lines of a script's log (for error messages)."""
        try:
            with open(self.sup.log_path(step.name), errors='replace') as f:
                lines = [ln.rstrip() for ln in f.readlines()[-200:]
                         if ln.strip() and not ln.startswith('=====')]
            return lines[-n:]
        except OSError:
            return []

    def stop_steps(self, steps):
        return self.sup.stop_many_async([s.name for s in steps if not s.external])

    def ready(self, step):
        """Readiness is checked until the step is ready once, then remembered while
        its process lives: the checks query the ROS graph and subscribe to topics,
        which costs CPU on the Jetson if repeated for every status update."""
        if step.name in self._ready_steps:
            if self.alive(step):
                return True
            self._ready_steps.discard(step.name)
        if all(self._check(c) for c in step.ready):
            self._ready_steps.add(step.name)
            return True
        return False

    def send_explorer(self, cmd):
        self.explorer_pub.publish(String(data=cmd))
        self._log_line(f'-> Frontera: {cmd}')

    def send_navigation(self, cmd):
        self.nav_pub.publish(String(data=cmd))
        self._log_line(f'-> Sendero: {cmd}')

    def zero_velocity(self):
        for _ in range(3):
            self.cmd_pub.publish(Twist())

    def call_service(self, key):
        client = self.clients_.get(key)
        if client is None or not client.service_is_ready():
            self._log_line(f'service {key} is not available')
            return None
        return _ServiceHandle(client.call_async(Trigger.Request()))

    def explorer_status(self):
        text, t = self.reports.get('explorer_status', ('', None))
        return text, (time.monotonic() - t) if t is not None else float('inf')

    def send_initial_pose(self, pose):
        """The robot's pose on the loaded map (as RViz 2D Pose Estimate does)."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(pose[0]), float(pose[1])
        msg.pose.pose.orientation.z = math.sin(pose[2] / 2.0)
        msg.pose.pose.orientation.w = math.cos(pose[2] / 2.0)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.05 ** 2
        cov[35] = 0.05 ** 2
        msg.pose.covariance = cov
        self.initial_pose_pub.publish(msg)

    def slam_status(self):
        text, _ = self.reports.get('slam_status', ('', None))
        return parse_kv(text)

    def publish_map_save_status(self, text):
        self.map_status_pub.publish(String(data=text))

    def on_state(self, state):
        if state != self._last_state:
            self._last_state = state
            self.state_pub.publish(String(data=state))

    def log(self, text):
        self._log_line(text)
        self.events_pub.publish(String(data=text))

    # ── phase 2 ──────────────────────────────────────────────────────────
    def _use_mission_folder(self, folder):
        self.mission_folder = folder
        self.placeholders['mission_dir'] = folder

    def _on_landmarks(self, msg):
        try:
            self.landmarks_json = json.loads(msg.data)
        except ValueError:
            pass

    def save_mission(self, raw_message):
        """End of mapping: the mission folder for Phase 2. -> (ok, text, folder)."""
        base = ms.parse_saved_base(raw_message or '')
        if not base:
            return False, f'no map path in "{raw_message}"', None
        r = self.restock_cfg.get('restock', {}) or {}
        companies = tuple(r.get('companies', ('soriene', 'emezon', 'walnut')))
        ids = list((self.restock_cfg.get('localization', {}) or {}).get('aruco_ids', []) or [])
        try:
            with open(os.path.join(self.config_dir, 'arena_map.yaml')) as f:
                template = ((yaml.safe_load(f) or {}).get('ekf_localization') or {}).get(
                    'ros__parameters', {})
        except OSError:
            template = {}
        overrides = {'search_rotation': False,
                     'cmd_vel_topic': self.placeholders.get('cmd_vel_in', '/cmd_vel')}
        try:
            from main_ros.restock.zones import ZoneConfig
            zone_cfg = ZoneConfig.from_dict(self.restock_cfg.get('zones'))
            folder, info = ms.save_mission_folder(self.store, base, self.landmarks_json, template,
                                                  companies, ids, overrides, zone_cfg)
        except (OSError, ValueError, KeyError) as exc:
            return False, str(exc), None
        self._use_mission_folder(folder)
        text = (f'{folder} (markers {info["markers"]}, zones {info["zones"]}, '
                f'docks {info["docks"]}'
                + (f', NO dock for {info["missing_docks"]}' if info['missing_docks'] else '')
                + ('' if info['session'] else ', no Planisferio session!') + ')')
        if not info['markers']:
            text += ' - no ArUco marker was found: the EKF cannot run in Phase 2'
        return True, text, folder

    def _ports(self):
        """RestockPorts on this node (created once: the map tour and restocking share it)."""
        from main_ros.restock.restock_ports import RestockPorts
        if self.restock_ports is None:
            cfg = dict(self.restock_cfg)
            ports = dict(cfg.get('ports', {}) or {})
            ports.setdefault('aruco_config', os.path.join(
                self.config_dir, self.placeholders.get('aruco_config', 'aruco_localization.yaml')))
            topics = dict(ports.get('topics', {}) or {})
            topics.setdefault('cmd_vel', self.placeholders.get('cmd_vel_in', '/cmd_vel'))
            topics.setdefault('scan', self.placeholders.get('scan_raw', '/laser_scan'))
            ports['topics'] = topics
            cfg['ports'] = ports
            self.restock_ports = RestockPorts(
                self, cfg, self.log, lambda: self.mission_folder, self.slam_status)
        return self.restock_ports

    def make_restock(self, mission, folder):
        """Stage 4: the restocking state machine."""
        from main_ros.restock.restock_core import RestockCore, RestockSettings
        settings = RestockSettings.from_dict(self.restock_cfg.get('restock', {}))
        from main_ros.restock.zones import ZoneConfig
        return RestockCore(self._ports(), settings, mission, folder or self.mission_folder,
                           ZoneConfig.from_dict(self.restock_cfg.get('zones')))

    def make_route_drive(self, folder):
        """Stage 4 with restock_mode: route_only: drive the stage-3 route, nothing else."""
        from main_ros.restock.route import RouteConfig
        from main_ros.restock.route_follower import RouteDriveSettings, RouteFollower
        rc = RouteConfig()
        for k, v in ((self.restock_cfg.get('restock', {}) or {}).get('route', {}) or {}).items():
            if hasattr(rc, k):
                setattr(rc, k, math.radians(v) if k == 'fov' and v > 3.2 else v)
        settings = RouteDriveSettings.from_dict(self.restock_cfg.get('route_drive', {}))
        return RouteFollower(self._ports(), settings, folder or self.mission_folder, rc)

    def make_tour(self):
        """Stage 2: the map-quality tour."""
        from main_ros.restock.map_tour import MapTour, TourSettings
        return MapTour(self._ports(), TourSettings.from_dict(
            self.restock_cfg.get('map_quality', {})))

    def preview_tour(self, folder):
        """After mapping: plan the stage-2 tour now (background thread) and show it on
        /restock/route_markers, so it can be checked in Foxglove before pressing S."""
        job = _Job()
        tour = self.make_tour()
        start = self._ports().slam_pose()

        def work():
            grid = self.load_grid(folder)
            if grid is None:
                return False, f'cannot read the map in {folder}'
            goals, r = tour.plan(grid, start)
            if not goals:
                return False, 'no room to drive a loop on this map'
            self._ports().show_route(goals, 'tour_preview')
            msg = (f'stage-2 tour preview on /restock/route_markers: {r["loops"]} loops '
                   f'({r["inner_loops"]} round obstacles), {len(goals)} waypoints, '
                   f'{r["length_m"]:.1f} m per pass')
            self._log_line(msg)
            return True, msg
        job.run(work)
        return job

    def load_grid(self, folder):
        """The occupancy grid saved in a mission folder (None if it cannot be read)."""
        from main_ros.restock.grid import OccupancyMap
        try:
            return OccupancyMap.from_files(os.path.join(folder, ms.MAP_YAML))
        except (OSError, ValueError, TypeError) as exc:
            self._log_line(f'cannot read the map in {folder}: {exc}')
            return None

    def plan_route(self, folder):
        """Stage 3 in a background thread: the patrol route -> <folder>/route.yaml."""
        from main_ros.restock.route import RouteConfig, plan_route
        job = _Job()
        ports = self._ports()
        pose = ports.slam_pose()
        cfg_route = (self.restock_cfg.get('restock', {}) or {}).get('route', {}) or {}

        def work():
            grid = self.load_grid(folder)
            if grid is None:
                return False, f'cannot read the map in {folder}'
            rc = RouteConfig()
            for k, v in cfg_route.items():
                if hasattr(rc, k):
                    setattr(rc, k, math.radians(v) if k == 'fov' and v > 3.2 else v)
            r = plan_route(grid, pose or (0.0, 0.0, 0.0), rc)
            if not r.viewpoints:
                return False, 'the map has no edge to patrol'
            ms.write_yaml(os.path.join(folder, ms.ROUTE), r.to_dict())
            ports.show_route(list(r.viewpoints), 'route')
            return True, (f'{len(r.viewpoints)} viewpoints, {r.length:.1f} m, '
                          f'{100 * r.coverage:.0f} % of the edges')
        job.run(work)
        return job

    # =====================================================================
    # readiness checks
    # =====================================================================
    def _graph_snapshot(self):
        """Service and node names (cached for half a second)."""
        t, services, nodes, topics = self._graph_cache
        if time.monotonic() - t > 1.0:
            services = {n for n, _ in self.get_service_names_and_types()}
            nodes = {('/' + ns.strip('/') + '/' + n).replace('//', '/')
                     for n, ns in self.get_node_names_and_namespaces()}
            topics = {n: ts for n, ts in self.get_topic_names_and_types()}
            self._graph_cache = (time.monotonic(), services, nodes, topics)
        return self._graph_cache

    def _check(self, check):
        """One readiness check, e.g. {'message': '/map'}."""
        (kind, value), = check.items()
        _, services, nodes, topics = self._graph_snapshot()
        if kind == 'topic':
            return self.count_publishers(value) > 0
        if kind == 'service':
            return value in services
        if kind == 'action':
            base = '/' + value.strip('/')
            return {base + '/_action/send_goal', base + '/_action/get_result'} <= services
        if kind == 'node':
            return ('/' + value.strip('/')) in nodes
        if kind == 'clock':
            c = self._sim_clock
            return (c['first'] is not None and c['last'] > c['first']
                    and time.monotonic() - c['wall'] < 5.0)
        if kind == 'message':
            return self._message_seen(value, topics)
        self._log_line(f'unknown readiness check {check}')
        return False

    def _message_seen(self, topic, topics):
        """True once a message arrived on `topic`. The probe subscription (raw, no
        decoding) is removed after the first message: staying subscribed to the
        scans, encoders and map would wake this node dozens of times a second."""
        probe = self._msg_probes.get(topic)
        if probe is None:
            types = topics.get(topic)
            if not types or get_message is None:
                return False
            probe = {'seen': False, 'sub': None}

            def on_msg(_msg, p=probe):
                p['seen'] = True
            try:
                probe['sub'] = self.create_subscription(
                    get_message(types[0]), topic, on_msg, qos_profile_sensor_data, raw=True)
            except Exception as exc:
                self._log_line(f'cannot probe {topic}: {exc}')
                return False
            self._msg_probes[topic] = probe
        if probe['seen'] and probe['sub'] is not None:
            self.destroy_subscription(probe['sub'])
            probe['sub'] = None
        return probe['seen']

    def _drop_probe(self, topic):
        probe = self._msg_probes.pop(topic, None)
        if probe and probe['sub'] is not None:
            self.destroy_subscription(probe['sub'])

    def _on_clock(self, msg):
        t = msg.clock.sec + 1e-9 * msg.clock.nanosec
        c = self._sim_clock
        if c['first'] is None or (c['last'] is not None and t < c['last']):
            c['first'] = t           # new simulation run
        c['last'] = t
        c['wall'] = time.monotonic()

    # =====================================================================
    # callbacks
    # =====================================================================
    def _on_nav_event(self, msg):
        line = f'[{time.strftime("%H:%M:%S")}] [sendero event] {msg.data[:400]}'
        try:
            self._mlog.write(line + '\n')
        except ValueError:
            pass

    def _on_report(self, key, msg):
        self.reports[key] = (msg.data, time.monotonic())

    def _on_command(self, msg):
        ok, text = self.core.command(msg.data)
        self._log_line(f'command "{msg.data}": {text}')

    def _tick(self):
        try:
            self.core.tick()
        except Exception as exc:     # never let the manager die silently
            self._log_line(f'ERROR in the state machine: {exc!r}')
            self.core._fault(f'internal error: {exc!r}')

    # =====================================================================
    # status for the monitor
    # =====================================================================
    def _report(self, key, keys=None):
        text, t = self.reports.get(key, ('', None))
        age = round(time.monotonic() - t, 1) if t is not None else None
        if keys is None:
            return text, age
        kv = parse_kv(text)
        return {k: kv[k] for k in keys if k in kv}, age

    def _publish_status(self):
        snap = self.core.snapshot()
        snap['steps'] = [{
            'name': s.name, 'external': s.external,
            'alive': self.alive(s), 'ready': self.ready(s) if self.alive(s) else False,
            'pid': (self.sup.get(s.name).pid if self.sup.get(s.name) else None),
            'starts': self.sup.starts.get(s.name, 0),
            'exit_code': None if s.external else self.sup.exit_code(s.name),
        } for s in self.core.steps]
        ex_status, ex_age = self._report('explorer_status')
        ex_stats, _ = self._report('explorer_stats', EXPLORER_KEYS)
        nav_status, nav_age = self._report('navigation_status')
        nav_stats, _ = self._report('navigation_stats', NAVIGATION_KEYS)
        slam_text, slam_age = self._report('slam_status')
        snap['explorer'] = dict(ex_stats, status=ex_status, age_s=ex_age)
        snap['navigation'] = dict(nav_stats, status=nav_status, age_s=nav_age)
        snap['slam'] = dict(parse_kv(slam_text), age_s=slam_age)
        c = self._sim_clock
        snap['sim_time_ok'] = (c['first'] is not None and time.monotonic() - c['wall'] < 2.0)
        snap['log_dir'] = self.log_dir
        snap['stamp'] = time.time()
        if snap.get('tour'):
            self.restock_pub.publish(String(data=json.dumps({'tour': snap['tour']}, default=str)))
        if snap.get('restock'):
            self.restock_pub.publish(String(data=json.dumps(snap['restock'], default=str)))
        self.status_pub.publish(String(data=json.dumps(snap, separators=(',', ':'), default=str)))
        self.state_pub.publish(String(data=snap['state']))

    # =====================================================================
    # logging and shutdown
    # =====================================================================
    def _log_line(self, text):
        line = f'[{time.strftime("%H:%M:%S")}] {text}'
        print(line, flush=True)
        try:
            self._mlog.write(line + '\n')
        except ValueError:
            pass

    def shutdown(self):
        """Halt the robot and close every script (Ctrl+C, or after X)."""
        self._log_line('Shutting down: halting the robot and closing every script...')
        try:
            if self.core.restock is not None:
                self.core.restock.abort('shutdown')
            self.send_explorer('STOP')
            self.send_navigation('CANCEL')
            self.zero_velocity()
        except Exception:
            pass
        time.sleep(0.3)
        self.sup.stop_all()
        report = ['Program Manager — shutdown report',
                  f'mode {self.mode}, final state {self.core.state}, '
                  f'stages {self.core.stage_progress()}',
                  f'mission folder: {self.core.mission_folder}',
                  f'last map: {self.core.last_map}', ''] + self.sup.report()
        with open(os.path.join(self.log_dir, 'shutdown_report.txt'), 'w') as f:
            f.write('\n'.join(report) + '\n')
        for line in report:
            self._log_line(line)
        self._log_line(f'Logs: {self.log_dir}')
        self.sup.close()


def main(args=None):
    """Spin until Ctrl+C / SIGTERM, then halt the robot and close every script."""
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init(args=args)
    node = ProgramManager()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    exit_at = None
    try:
        while rclpy.ok() and not stop.is_set():
            executor.spin_once(timeout_sec=0.1)
            # X finished: keep publishing STOPPED for a moment so the monitor
            # sees it (and closes itself), then end -> main.launch.py ends.
            if node.core.exit_requested:
                if exit_at is None:
                    exit_at = time.monotonic() + 3.0
                    node._log_line('Stop complete: closing the Program Manager.')
                elif time.monotonic() > exit_at:
                    break
    finally:
        node.shutdown()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
