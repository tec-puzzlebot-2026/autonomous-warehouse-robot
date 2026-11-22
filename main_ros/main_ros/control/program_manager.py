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
#          mission:=mapping | restocking | full   (default mapping)
#          map:=<mission folder>  (restocking: default the latest one)
# Control: the monitor (ros2 run main_ros mission_monitor, on the host), or
#          ros2 topic pub --once /mission/command std_msgs/String "{data: S}"
#
# Commands (keys of the monitor):
#   S start: bring the system up / start exploring / resume
#   P pause            F finish mapping now      M save the map
#   R restocking (mission:=full, after mapping)
#   N reset (new mapping run)                    X stop (close every script)
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
import os
import signal
import threading
import time

import rclpy
import yaml
from geometry_msgs.msg import Twist
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


class ProgramManager(Node):
    """ROS 2 node: the ports of MissionCore plus status publishing."""

    def __init__(self):
        super().__init__('program_manager')
        self.mode = self.declare_parameter('mode', 'sim').value
        cfg_file = self.declare_parameter('config_file', '').value
        auto = self.declare_parameter('autostart', False).value
        self.autostart = str(auto).strip().lower() in ('true', '1', 'yes')
        self.mission = str(self.declare_parameter('mission', 'mapping').value or 'mapping')
        map_arg = str(self.declare_parameter('map', '').value or '')
        if self.mode not in ('sim', 'real'):
            raise ValueError(f'mode must be "sim" or "real", got "{self.mode}"')
        if self.mission not in mc.MISSIONS:
            raise ValueError(f'mission must be one of {mc.MISSIONS}, got "{self.mission}"')
        self.config_dir = os.path.dirname(cfg_file) if cfg_file else find_config_dir()
        cfg_file = cfg_file or os.path.join(self.config_dir, 'program_manager.yaml')
        with open(cfg_file) as f:
            self.cfg = yaml.safe_load(f)
        mode_cfg = self.cfg['modes'][self.mode]
        self.placeholders = {
            'mode': self.mode,
            'mission': self.mission,
            'config_dir': self.config_dir,
            'use_sim_time': 'true' if mode_cfg.get('use_sim_time', False) else 'false',
            'maps_dir': os.path.expanduser(str(mode_cfg.get('maps_dir', '~/maps'))),
            'mission_dir': '',
            'slam_args': '',
        }
        for key, value in (mode_cfg.get('placeholders') or {}).items():
            self.placeholders[key] = os.path.expanduser(str(value))
        restock_file = os.path.join(self.config_dir, 'restock.yaml')
        self.restock_cfg = {}
        if os.path.isfile(restock_file):
            with open(restock_file) as f:
                self.restock_cfg = yaml.safe_load(f) or {}
        self.store = ms.MissionStore(os.path.join(self.placeholders['maps_dir'], 'missions'))
        self.mission_folder = None
        if self.mission == 'restocking':
            self.mission_folder = self.store.resolve(map_arg)      # clear error if none
            self._use_mission_folder(self.mission_folder)
        self.landmarks_json = None
        self.restock_ports = None
        self.log_dir = run_log_dir(f'program_manager_{self.mode}',
                                   str(mode_cfg.get('log_root', '') or ''))
        os.makedirs(self.placeholders['maps_dir'], exist_ok=True)
        self.placeholders['log_dir'] = self.log_dir
        self._mlog = open(os.path.join(self.log_dir, 'manager.txt'), 'a', buffering=1)

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
        self.create_subscription(String, topics['map_save_request'],
                                 lambda m: self.core.on_map_save_request(), 10)
        self.create_subscription(String, topics.get('landmarks', '/landmark_mapper/landmarks'),
                                 self._on_landmarks, 5)

        # ── readiness probes ─────────────────────────────────────────────
        self._msg_probes = {}           # topic -> {'count', 'last'}
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
        self.core = mc.MissionCore(self.steps, self, settings, self.mode, self.mission)
        self.core.mission_folder = self.mission_folder

        self.create_timer(1.0 / float(self.cfg.get('tick_hz', 10.0)), self._tick)
        self.create_timer(1.0 / float(self.cfg.get('status_hz', 2.0)), self._publish_status)
        self._last_state = None

        self._log_line('=' * 64)
        self._log_line(f'Program Manager — {self.mode.upper()} mode, mission {self.mission}')
        if self.mission_folder:
            self._log_line(f'mission folder: {self.mission_folder}')
        self._log_line(f'config: {cfg_file}')
        self._log_line(f'logs  : {self.log_dir}')
        self._log_line('steps : ' + ' -> '.join(s.name for s in self.steps))
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
                phase=str(raw.get('phase', 'both'))))
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
        return all(self._check(c) for c in step.ready)

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
        if self.mission == 'restocking':
            session = os.path.join(folder, ms.SESSION)
            self.placeholders['slam_args'] = (f'-p load_session:={session} '
                                              '-p localization_only:=true')

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
            folder, info = ms.save_mission_folder(self.store, base, self.landmarks_json, template,
                                                  companies, ids, overrides)
        except (OSError, ValueError, KeyError) as exc:
            return False, str(exc), None
        self._use_mission_folder(folder)
        text = (f'{folder} (markers {info["markers"]}, docks {info["docks"]}'
                + (f', NO dock for {info["missing_docks"]}' if info['missing_docks'] else '')
                + ('' if info['session'] else ', no Planisferio session!') + ')')
        if not info['markers']:
            text += ' - no ArUco marker was found: the EKF cannot run in Phase 2'
        return True, text, folder

    def make_restock(self, mission, folder):
        """The restocking state machine on this node's ports."""
        from main_ros.restock.restock_core import RestockCore, RestockSettings
        from main_ros.restock.restock_ports import RestockPorts
        if self.restock_ports is None:
            cfg = dict(self.restock_cfg)
            ports = dict(cfg.get('ports', {}) or {})
            ports.setdefault('aruco_config', os.path.join(self.config_dir,
                                                          'aruco_localization.yaml'))
            topics = dict(ports.get('topics', {}) or {})
            topics.setdefault('cmd_vel', self.placeholders.get('cmd_vel_in', '/cmd_vel'))
            topics.setdefault('scan', self.placeholders.get('scan_raw', '/laser_scan'))
            ports['topics'] = topics
            cfg['ports'] = ports
            self.restock_ports = RestockPorts(
                self, cfg, self.log, lambda: self.mission_folder, self.slam_status)
        settings = RestockSettings.from_dict(self.restock_cfg.get('restock', {}))
        return RestockCore(self.restock_ports, settings, mission, folder or self.mission_folder)

    # =====================================================================
    # readiness checks
    # =====================================================================
    def _graph_snapshot(self):
        """Service and node names (cached for half a second)."""
        t, services, nodes, topics = self._graph_cache
        if time.monotonic() - t > 0.5:
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
        """True once a message arrived on `topic` (raw subscription, no decoding)."""
        probe = self._msg_probes.get(topic)
        if probe is None:
            types = topics.get(topic)
            if not types or get_message is None:
                return False
            probe = {'count': 0, 'last': 0.0}

            def on_msg(_msg, p=probe):
                p['count'] += 1
                p['last'] = time.monotonic()
            try:
                self.create_subscription(get_message(types[0]), topic, on_msg,
                                         qos_profile_sensor_data, raw=True)
            except Exception as exc:
                self._log_line(f'cannot probe {topic}: {exc}')
                return False
            self._msg_probes[topic] = probe
        return probe['count'] > 0 and time.monotonic() - probe['last'] < 10.0

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
        } for s in self.steps]
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
                  f'mode {self.mode}, mission {self.mission}, final state {self.core.state}',
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
