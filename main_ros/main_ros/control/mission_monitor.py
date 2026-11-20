#!/usr/bin/env python3
"""
mission_monitor.py — Team Zero mission monitor (runs on the host PC)
====================================================================
A light terminal dashboard for the Program Manager running on the Spark
(simulation) or the Jetson (robot), next to RViz / Foxglove.

Light on purpose: it subscribes to ONE small topic, /mission/status (a JSON
string at 2 Hz that the Program Manager assembles from its own state and the
SLAM, Frontera and Sendero reports), and publishes key presses on
/mission/command. No map, scan or TF crosses the network for it.

Keys
    S  start: bring the system up / start exploring / resume
    P  pause (the robot holds)
    F  finish mapping now (optimize + save the map)
    M  save the map
    R  start restocking (after mapping; phase 2 is a placeholder for now)
    N  reset: new mapping run (press twice: it clears the map)
    X  stop: halt and close every script (press twice); then the Program
       Manager, this monitor and the host side close by themselves
    Q  quit the monitor (the robot keeps its state)

Opened by host.launch.py in its own window, or by hand on the host (same
ROS_DOMAIN_ID / DDS settings as for RViz):
    ros2 run main_ros mission_monitor
"""
# Author: Raymundo Gomez

import curses
import json
import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

UI_PERIOD_S = 0.25
LINK_TIMEOUT_S = 3.0
CONFIRM_WINDOW_S = 3.0
WIDTH = 80

KEYS = [  # (key, command, label)
    ('S', 'START', 'start'),
    ('P', 'PAUSE', 'pause'),
    ('F', 'FINISH', 'finish mapping'),
    ('M', 'SAVE_MAP', 'save map'),
    ('R', 'RESTOCK', 'restocking'),
    ('N', 'RESET', 'reset'),
    ('X', 'STOP', 'stop'),
]
NEEDS_CONFIRM = ('N', 'X')


class MonitorNode(Node):
    """Receives /mission/status and sends /mission/command."""

    def __init__(self):
        super().__init__('mission_monitor')
        status_topic = self.declare_parameter('status_topic', '/mission/status').value
        command_topic = self.declare_parameter('command_topic', '/mission/command').value
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE)
        self.cmd_pub = self.create_publisher(String, command_topic, qos)
        self.create_subscription(String, status_topic, self._on_status, qos)
        self.lock = threading.Lock()
        self.status = None
        self.status_wall = None
        self.started = time.time()
        self.last_sent = ''
        self.exit_at = None
        self.pending_confirm = None     # (key, wall time)

    def _on_status(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.status = data
            self.status_wall = time.time()

    def snapshot(self):
        with self.lock:
            age = None if self.status_wall is None else time.time() - self.status_wall
            return self.status, age

    def send(self, key):
        self.cmd_pub.publish(String(data=key))
        self.last_sent = f'{key} sent at {time.strftime("%H:%M:%S")}'


def put(scr, y, x, text, attr=0):
    """addstr that never raises on small terminals."""
    h, w = scr.getmaxyx()
    if 0 <= y < h and 0 <= x < w - 1:
        try:
            scr.addstr(y, x, str(text)[:max(0, w - x - 1)], attr)
        except curses.error:
            pass


def fnum(value, fmt='{:.2f}', default='–'):
    """Format a number that may be missing or a string."""
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return default


def draw(scr, node):
    curses.curs_set(0)
    scr.nodelay(True)
    curses.start_color()
    curses.use_default_colors()
    try:
        bg = 234 if curses.COLORS >= 256 else -1
        fg = 253 if curses.COLORS >= 256 else curses.COLOR_WHITE
        dim_fg = 245 if curses.COLORS >= 256 else curses.COLOR_WHITE
        pairs = [(fg, bg), (dim_fg, bg), (curses.COLOR_GREEN, bg),
                 (curses.COLOR_YELLOW, bg), (curses.COLOR_RED, bg),
                 (curses.COLOR_CYAN, bg)]
        for i, (f, b) in enumerate(pairs, start=1):
            curses.init_pair(i, f, b)
    except curses.error:
        pass
    text = curses.color_pair(1)
    bold = curses.color_pair(1) | curses.A_BOLD
    dim = curses.color_pair(2)
    green = curses.color_pair(3) | curses.A_BOLD
    amber = curses.color_pair(4) | curses.A_BOLD
    red = curses.color_pair(5) | curses.A_BOLD
    cyan = curses.color_pair(6) | curses.A_BOLD
    scr.bkgd(' ', text)

    state_color = {
        'IDLE': dim, 'BRINGUP': amber, 'READY': cyan, 'EXPLORING': green,
        'PAUSED': amber, 'FINISHING': amber, 'MAPPING_DONE': green,
        'RESTOCKING': cyan, 'RESETTING': amber, 'STOPPING': amber,
        'STOPPED': dim, 'ERROR': red}
    L, W = 1, WIDTH
    M = L + W // 2

    # Opened by the host manager: close when it removes this file.
    alive_file = os.environ.get('MISSION_MONITOR_ALIVE_FILE', '')
    last_full = 0.0

    while rclpy.ok():
        if alive_file and not os.path.exists(alive_file):
            break
        key = scr.getch()
        st, age = node.snapshot()
        # X finished and the Program Manager is closing: close the monitor too
        # (exit code 0 -> the host side closes its window and itself).
        if st and st.get('exiting'):
            if node.exit_at is None:
                node.exit_at = time.time() + 2.0
                node.last_sent = 'Program Manager closed. The monitor closes...'
            elif time.time() > node.exit_at:
                break
        allowed = set((st or {}).get('allowed', []))
        if key in (ord('q'), ord('Q')):
            break
        if 0 < key < 256 and chr(key).upper() in [k for k, _, _ in KEYS]:
            k = chr(key).upper()
            cmd = dict((kk, c) for kk, c, _ in KEYS)[k]
            now = time.time()
            if cmd not in allowed:
                node.last_sent = f'{k}: not possible now'
            elif k in NEEDS_CONFIRM and not (
                    node.pending_confirm and node.pending_confirm[0] == k
                    and now - node.pending_confirm[1] < CONFIRM_WINDOW_S):
                node.pending_confirm = (k, now)
                what = 'RESET (clears the map)' if k == 'N' else 'STOP everything'
                node.last_sent = f'press {k} again within {CONFIRM_WINDOW_S:.0f} s to {what}'
            else:
                node.pending_confirm = None
                node.send(k)

        if time.time() - last_full > 2.0:   # repaint everything now and then
            scr.clear()
            last_full = time.time()
        else:
            scr.erase()
        h, w = scr.getmaxyx()
        if w < W + 2 or h < 34:
            put(scr, 0, 0, f'Enlarge the terminal to at least {W + 2} x 34.', amber)
            scr.refresh()
            time.sleep(UI_PERIOD_S)
            continue

        def hline(y, title='', left='├', right='┤', mid_title=None):
            if mid_title is None:
                t = f' {title} ' if title else ''
                n = W - 2 - len(t)
                put(scr, y, L, left + '─' * (n // 2) + t + '─' * (n - n // 2) + right, dim)
            else:
                a, b = f' {title} ', f' {mid_title} '
                na, nb = M - L - 1 - len(a), L + W - 1 - M - 1 - len(b)
                put(scr, y, L, left + '─' * (na // 2) + a + '─' * (na - na // 2) + '┼'
                    + '─' * (nb // 2) + b + '─' * (nb - nb // 2) + right, dim)

        def sides(y, split=False):
            put(scr, y, L, '│', dim)
            put(scr, y, L + W - 1, '│', dim)
            if split:
                put(scr, y, M, '│', dim)

        def row(y, x, label, value, attr=text, lw=10):
            put(scr, y, x, label.ljust(lw), dim)
            put(scr, y, x + lw, value, attr)

        y = 0
        hline(y, 'TEAM ZERO · WAREHOUSE ROBOT · MISSION MONITOR', '┌', '┐')
        y += 1
        linked = st is not None and age is not None and age < LINK_TIMEOUT_S
        if not linked:
            sides(y)
            put(scr, y, L + 2, 'NO LINK TO THE ROBOT', red)
            y += 1
            sides(y)
            put(scr, y, L + 2, 'Is main.launch.py running? Same ROS_DOMAIN_ID / '
                'Fast DDS profile as RViz?', amber)
            y += 1
            sides(y)
            put(scr, y, L + 2, ('last status ' + (f'{age:.0f} s ago' if age else 'never')),
                dim)
            y += 1
            st = st or {}
        state = st.get('state', '?')
        up = int(st.get('uptime_s', 0) or 0)
        sides(y)
        put(scr, y, L + 2, 'MISSION', dim)
        put(scr, y, L + 11, state, state_color.get(state, bold))
        info = (f'{st.get("mode", "?").upper()} · up {up // 3600:02d}:{up % 3600 // 60:02d}:'
                f'{up % 60:02d} · link {fnum(age, "{:.1f}")} s')
        put(scr, y, L + W - 2 - len(info), info, dim)
        y += 1
        detail = st.get('detail', '')
        for i in range(2):
            sides(y)
            put(scr, y, L + 2, detail[i * (W - 4):(i + 1) * (W - 4)],
                red if state == 'ERROR' else text)
            y += 1
        lm = st.get('last_map') or {}
        sides(y)
        if st.get('saving_map'):
            put(scr, y, L + 2, 'map: saving...', amber)
        elif lm:
            put(scr, y, L + 2, f'map {lm.get("time", "")}: '
                f'{"saved " if lm.get("ok") else "FAILED "}{lm.get("message", "")}',
                green if lm.get('ok') else red)
        else:
            put(scr, y, L + 2, 'map: not saved yet', dim)
        y += 1

        # ── scripts | slam ───────────────────────────────────────────────
        hline(y, 'scripts', mid_title='slam')
        y += 1
        steps = st.get('steps', [])
        slam = st.get('slam', {}) or {}
        slam_rows = [
            ('state', slam.get('state', '–'),
             green if slam.get('state') == 'RUNNING' else amber),
            ('scans', f'{slam.get("scan_hz", "–")} Hz · used {slam.get("scans_used", "–")}', text),
            ('graph', f'{slam.get("keyframes", "–")} kf · {slam.get("loops", "–")} loops · '
                      f'{slam.get("corrections", "–")} corr', text),
            ('last fix', (f'{slam["last_correction_cm"]} cm, {slam.get("last_correction_age_s")} s ago'
                          if 'last_correction_cm' in slam else '–'), text),
            ('map', f'{slam.get("map_w", "–")}×{slam.get("map_h", "–")} · '
                    f'{slam.get("known_m2", "–")} m²', text),
            ('pose', f'({slam.get("x", "–")}, {slam.get("y", "–")}, '
                     f'{slam.get("yaw_deg", "–")}°) sd {slam.get("sd_xy", "–")}', text),
        ]
        n = max(len(steps), len(slam_rows))
        for i in range(n):
            sides(y, split=True)
            if i < len(steps):
                s = steps[i]
                if s.get('ready'):
                    dot, attr, word = '●', green, 'ready'
                elif s.get('alive'):
                    dot, attr, word = '●', amber, 'starting' if not s.get('external') else 'waiting'
                elif s.get('starts'):
                    dot, attr, word = '●', red, f'exited ({s.get("exit_code")})'
                else:
                    dot, attr, word = '○', dim, 'not started'
                put(scr, y, L + 2, dot, attr)
                put(scr, y, L + 4, s.get('name', '')[:15].ljust(16), text)
                put(scr, y, L + 20, word, attr)
            if i < len(slam_rows):
                lab, val, attr = slam_rows[i]
                row(y, M + 2, lab, val, attr)
            y += 1

        # ── exploration | navigation ─────────────────────────────────────
        hline(y, 'exploration (Frontera)', mid_title='navigation (Sendero)')
        y += 1
        ex = st.get('explorer', {}) or {}
        nav = st.get('navigation', {}) or {}
        ex_rows = [
            ('status', ex.get('status') or '–', green if ex.get('status') == 'EXPLORING' else text),
            ('goals', f'reached {ex.get("reached", "–")} · failed {ex.get("failed", "–")}', text),
            ('frontiers', f'{ex.get("frontier_count", "–")} · blacklist '
                          f'{ex.get("blacklist_count", "–")}', text),
            ('target', f'({fnum(ex.get("target_x"))}, {fnum(ex.get("target_y"))}) '
                       f'{fnum(ex.get("dist_target_m"), "{:.1f}")} m', text),
            ('travelled', f'{fnum(ex.get("dist_travelled_m"), "{:.1f}")} m · '
                          f'{fnum(ex.get("map_known_area_m2"), "{:.0f}")} m²', text),
        ]
        safety = nav.get('safety', '–')
        nav_rows = [
            ('status', nav.get('status') or '–',
             red if str(nav.get('status', '')).startswith(('BLOCKED', 'RECOVER', 'LOCAL'))
             else text),
            ('remaining', f'{fnum(nav.get("distance_remaining"))} m · '
                          f'recoveries {nav.get("recoveries", "–")}', text),
            ('safety', f'{safety} · closest {fnum(nav.get("closest_obstacle_m"))} m',
             green if safety == 'CLEAR' else amber),
            ('command', f'v {fnum(nav.get("cmd_vx"))}  w {fnum(nav.get("cmd_wz"))}', text),
            ('scan age', f'{fnum(nav.get("scan_age_s"))} s · {nav.get("profile", "–")}', text),
        ]
        for i in range(5):
            sides(y, split=True)
            lab, val, attr = ex_rows[i]
            row(y, L + 2, lab, val, attr)
            lab, val, attr = nav_rows[i]
            row(y, M + 2, lab, val, attr)
            y += 1

        # ── events ───────────────────────────────────────────────────────
        hline(y, 'events')
        y += 1
        events = st.get('events', [])[-8:]
        for i in range(8):
            sides(y)
            if i < len(events):
                t, msg = events[i]
                put(scr, y, L + 2, t, dim)
                put(scr, y, L + 11, msg[:W - 13],
                    red if msg.startswith(('ERROR', 'rejected')) else text)
            y += 1

        # ── keys ─────────────────────────────────────────────────────────
        hline(y, 'keys')
        y += 1
        sides(y)
        x = L + 2
        for k, cmd, label in KEYS:
            if k == 'S' and st.get('start_meaning'):
                label = st['start_meaning']
            on = cmd in allowed
            put(scr, y, x, f'[{k}]', bold if on else dim)
            put(scr, y, x + 4, label, text if on else dim)
            x += 5 + len(label) + 1
            if x > L + W - 14 and k != KEYS[-1][0]:
                y += 1
                sides(y)
                x = L + 2
        put(scr, y, L + W - 12, '[Q] quit', text)
        y += 1
        sides(y)
        put(scr, y, L + 2, node.last_sent,
            amber if 'again' in node.last_sent or 'not possible' in node.last_sent else dim)
        y += 1
        hline(y, '', '└', '┘')
        scr.refresh()
        time.sleep(UI_PERIOD_S)


def _redirect_stderr():
    """Send ROS / DDS error text to a file instead of the screen.

    A Humble PC talking to a Jazzy robot prints harmless discovery errors
    (ParticipantEntitiesInfo) on stderr; written into a curses screen they
    scroll it and break the layout.
    """
    try:
        log_dir = os.environ.get('MISSION_MONITOR_LOG_DIR', '')
        if not log_dir:
            from main_ros.control.process_supervisor import run_log_dir
            log_dir = run_log_dir('mission_monitor')
        path = os.path.join(log_dir, 'monitor_stderr.txt')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 2)
        os.close(fd)
    except OSError:
        pass


def main(args=None):
    _redirect_stderr()
    rclpy.init(args=args)
    node = MonitorNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        curses.wrapper(draw, node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
