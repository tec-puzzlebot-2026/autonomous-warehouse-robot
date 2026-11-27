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

Keys (one run, four stages: 1 map, 2 map quality, 3 route, 4 restocking;
after each stage the robot stops and waits for S)
    S  start: bring up / start mapping / next stage / resume
    P  pause (the robot holds)
    F  finish the stage now: mapping or the map tour (optimize + save the map)
    K  skip stage 2 (map quality): straight to the route
    M  save the map
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
WIDTH = 80                      # box width (the terminal needs 82 columns)

KEYS = [  # (key, command, label)
    ('S', 'START', 'start'),
    ('P', 'PAUSE', 'pause'),
    ('F', 'FINISH', 'finish'),
    ('K', 'SKIP', 'skip'),
    ('M', 'SAVE_MAP', 'save map'),
    ('N', 'RESET', 'reset'),
    ('X', 'STOP', 'stop'),
]
NEEDS_CONFIRM = ('N', 'X')
KEYS_SHORT = {'S': 'start', 'F': 'finish', 'M': 'save', 'K': 'skip'}
STAGE_MARK = {'done': '✓', 'skipped': '–', 'active': '▶', 'todo': '·'}
MAX_NODE_ROWS = 9               # at least; a taller window lists more (all at 84 x 50)
AREA_WINDOW_S = 60.0            # map growth is measured over this window


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
        self.last_sent_wall = 0.0
        self.pending_confirm = None     # (key, wall time)
        self.area_hist = []             # (wall time, known m²) for map growth

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
        self.last_sent_wall = time.time()

    def track_area(self, known_m2):
        """Remember the known map area to show how fast the map grows."""
        try:
            area = float(known_m2)
        except (TypeError, ValueError):
            return
        now = time.time()
        if not self.area_hist or now - self.area_hist[-1][0] >= 1.0:
            self.area_hist.append((now, area))
        while self.area_hist and now - self.area_hist[0][0] > AREA_WINDOW_S:
            self.area_hist.pop(0)

    def area_growth(self):
        """m² per minute over the last minute, or None."""
        if len(self.area_hist) < 2:
            return None
        (t0, a0), (t1, a1) = self.area_hist[0], self.area_hist[-1]
        if t1 - t0 < 5.0:
            return None
        return (a1 - a0) / (t1 - t0) * 60.0


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


def age_word(age, fresh=3.0):
    """'connected' / 'stale 12 s' / 'no data' for a report age."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return 'no data', 'dim'
    if a < fresh:
        return 'connected', 'green'
    return f'stale {a:.0f} s', 'amber'


def tour_rows(t, C):
    """Rows of the stage-2 map tour (5)."""
    text, green, amber, bold = C['text'], C['green'], C['amber'], C['bold']
    cov = t.get('coverage')
    marks = t.get('markers') or []
    logos = t.get('logos') or []
    return [
        ('goal', f'{t.get("goal", 0)}/{t.get("goals", 0)}'
                 + (' · paused' if t.get('paused') else ''), bold),
        ('skipped', t.get('skipped', 0), amber if t.get('skipped') else green),
        ('edges', f'{100 * cov:.0f} % · {t.get("length_m", 0)} m' if cov is not None else '–',
         text),
        ('ArUco', ', '.join(str(m) for m in marks) or 'none yet', green if marks else amber),
        ('logos', ', '.join(str(lg) for lg in logos) or 'none yet', green if logos else amber),
    ]


def restock_rows(rs, C):
    """Rows of the restocking section (5) and of the boxes section (5)."""
    text, green, amber, red, bold = C['text'], C['green'], C['amber'], C['red'], C['bold']
    route = rs.get('route') or {}
    cov = route.get('coverage')
    route_txt = (f'{min(route.get("next", 0) + 1, route.get("total", 0))}/{route.get("total", 0)}'
                 + (f' · {100 * cov:.0f} % edges' if cov is not None else '')
                 + (f' · {route["uncovered"]} skipped' if route.get('uncovered') else ''))
    box = rs.get('box') or {}
    if box:
        what = box.get('company') or ('data missing' if box.get('data_missing') else 'QR ?')
        box_txt = f'{box.get("id")} · {what} · level {box.get("level")}'
    else:
        box_txt = 'none'
    counts = rs.get('boxes') or {}
    count_txt = ' · '.join(f'{k} {v}' for k, v in sorted(counts.items())) or 'none yet'
    bad = any(k in counts for k in ('failed', 'lost', 'unknown', 'no_dock'))
    left = [
        ('state', f'{rs.get("state", "–")} / {rs.get("phase", "–")}',
         red if rs.get('error') else (amber if rs.get('paused') else bold)),
        ('route', route_txt, text),
        ('box', box_txt, text),
        ('boxes', count_txt, amber if bad else green),
        ('docks', ', '.join(rs.get('docks') or []) or 'none', green if rs.get('docks') else red),
    ]
    if rs.get('mode') == 'route_only':              # only driving the route: no boxes
        left[2] = ('box', 'not used (route only)', C['dim'])
        left[3] = ('boxes', 'not used (route only)', C['dim'])
        left[4] = ('docks', 'not used (route only)', C['dim'])
    delivered = rs.get('delivered') or {}
    events = rs.get('events') or []
    folder = str(rs.get('folder') or '–')
    if len(folder) > 50:
        folder = '…' + folder[-49:]
    bottom = [
        ('doing', rs.get('detail', '') or '–', text),
        ('delivered', ' · '.join(f'{c} {n}' for c, n in delivered.items()) or '–', green),
        ('problem', rs.get('error') or '–', red if rs.get('error') else C['dim']),
        ('last event', events[-1][1] if events else '–', text),
        ('mission folder', folder, C['dim']),
    ]
    return left, bottom


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
    C = {
        'text': curses.color_pair(1),
        'bold': curses.color_pair(1) | curses.A_BOLD,
        'dim': curses.color_pair(2),
        'green': curses.color_pair(3) | curses.A_BOLD,
        'amber': curses.color_pair(4) | curses.A_BOLD,
        'red': curses.color_pair(5) | curses.A_BOLD,
        'cyan': curses.color_pair(6) | curses.A_BOLD,
    }
    text, bold, dim = C['text'], C['bold'], C['dim']
    green, amber, red, cyan = C['green'], C['amber'], C['red'], C['cyan']
    scr.bkgd(' ', text)

    state_color = {
        'IDLE': dim, 'BRINGUP': amber, 'READY': cyan, 'EXPLORING': green,
        'PAUSED': amber, 'FINISHING': amber, 'MAPPING_DONE': green,
        'RESTOCKING': cyan, 'RESETTING': amber, 'STOPPING': amber,
        'STOPPED': dim, 'ERROR': red, 'STAGE_BRINGUP': amber, 'RESTOCK_PAUSED': amber,
        'RESTOCK_DONE': green, 'LOCALIZING': amber, 'QUALITY': green,
        'QUALITY_PAUSED': amber, 'QUALITY_DONE': green, 'PLANNING': amber,
        'ROUTE_READY': green}
    L, W = 1, WIDTH
    M = L + W // 2
    CW = W - 4                       # usable width inside the box

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
            node.last_sent_wall = now

        linked = st is not None and age is not None and age < LINK_TIMEOUT_S
        st = st or {}
        slam = st.get('slam', {}) or {}
        ex = st.get('explorer', {}) or {}
        nav = st.get('navigation', {}) or {}
        steps = st.get('steps', [])
        node.track_area(slam.get('known_m2'))

        # ── rows of the two split sections, built first to size the screen ──
        mode = st.get('mode', 'sim')
        conn = [('Spark' if mode == 'sim' else 'Jetson',) + (('connected', green) if linked else
                                    (('no link', red) if node.status_wall is None
                                     else (f'lost {age:.0f} s', red)))]
        if mode == 'sim':
            conn.append(('Isaac Sim', 'playing', green) if st.get('sim_time_ok')
                        else ('Isaac Sim', 'not playing', amber))
        for label, sub in (('SLAM', slam), ('Frontera', ex), ('Sendero', nav)):
            word, col = age_word(sub.get('age_s'))
            conn.append((label, word, C[col]))
        fox = next((s for s in steps if s.get('name') == 'foxglove_bridge'), None)
        if fox is not None:
            conn.append(('Foxglove', 'bridge up', green) if fox.get('ready')
                        else ('Foxglove', 'down', amber))
        nodes = []
        for i, s in enumerate(steps, start=1):
            if s.get('ready'):
                word, attr = ('playing' if s.get('external') else 'running'), green
            elif s.get('alive'):
                word, attr = ('waiting' if s.get('external') else 'starting'), amber
            elif s.get('starts'):
                word, attr = f'exited ({s.get("exit_code")})', red
            else:
                word, attr = 'off', dim
            nodes.append((f'Node {i}', s.get('name', '')[:17], word, attr))
        h, w = scr.getmaxyx()
        max_rows = max(MAX_NODE_ROWS, len(conn), h - 31)   # as many as the window fits
        if len(nodes) > max_rows:            # too short a window: problems first
            bad = [n for n in nodes if n[3] is not green]
            good = [n for n in nodes if n[3] is green]
            keep = (bad + good)[:max_rows - 1]
            rest = len(nodes) - len(keep)
            nodes = keep + [('', f'+ {rest} more', 'running', green)]
        n_split = max(len(conn), len(nodes))
        need_h = 31 + n_split

        if time.time() - last_full > 2.0:   # repaint everything now and then
            scr.clear()
            last_full = time.time()
        else:
            scr.erase()
        h, w = scr.getmaxyx()
        if w < W + 2 or h < need_h:
            put(scr, 0, 0, f'Enlarge the terminal to at least {W + 2} x {need_h}.', amber)
            scr.refresh()
            time.sleep(UI_PERIOD_S)
            continue

        def section(y, title='', mid_title=None, left='├', right='┤', mid='┼'):
            """Horizontal rule with a centered title (or one per half)."""
            if mid_title is None:
                t = f' {title} ' if title else ''
                n = W - 2 - len(t)
                put(scr, y, L, left + '─' * (n // 2), dim)
                put(scr, y, L + 1 + n // 2, t, dim)
                put(scr, y, L + 1 + n // 2 + len(t), '─' * (n - n // 2) + right, dim)
                return
            for x0, x1, t, lc in ((L, M, title, left), (M, L + W - 1, mid_title, mid)):
                t = f' {t} '
                n = x1 - x0 - 1 - len(t)
                put(scr, y, x0, lc + '─' * (n // 2) + t + '─' * (n - n // 2), dim)
            put(scr, y, L + W - 1, right, dim)

        def sides(y, split=False):
            put(scr, y, L, '│', dim)
            put(scr, y, L + W - 1, '│', dim)
            if split:
                put(scr, y, M, '│', dim)

        def row(y, x, label, value, attr=text, lw=11, width=None):
            put(scr, y, x, label.ljust(lw), dim)
            value = str(value)
            if width is not None:
                value = value[:max(0, width - lw)]
            put(scr, y, x + lw, value, attr)

        half = M - L - 3
        y = 0
        # ── title ───────────────────────────────────────────────────────
        put(scr, y, L, '┌' + '─' * (W - 2) + '┐', dim)
        y += 1
        sides(y)
        rs = st.get('restock') or None
        tour = st.get('tour') or None
        if tour and tour.get('done') and not rs:
            tour = tour if st.get('state') in ('FINISHING', 'QUALITY_DONE') else None
        put(scr, y, L + 2, 'Team Zero', bold)
        x = L + 13
        for _stage, title, mark in st.get('stages') or []:
            attr = {'done': green, 'active': cyan, 'skipped': dim}.get(mark, dim)
            put(scr, y, x, f'{STAGE_MARK.get(mark, "·")} {title}', attr)
            x += len(title) + 4
        up = int(st.get('uptime_s', 0) or 0)
        run = (f'{mode.upper()} · run {up // 3600:d}:{up % 3600 // 60:02d}:{up % 60:02d}'
               if up >= 3600 else f'{mode.upper()} · run {up // 60:02d}:{up % 60:02d}')
        put(scr, y, L + W - 2 - len(run), run, dim)
        y += 1

        # ── controls ────────────────────────────────────────────────────
        section(y, 'controls')
        y += 1
        sides(y)
        if linked:
            state, detail = st.get('state', '?'), st.get('detail', '')
        else:
            state = 'NO LINK'
            detail = ('no status yet' if age is None else f'last status {age:.0f} s ago') \
                + ' · is main.launch.py running on the robot?'
        put(scr, y, L + 2, state.ljust(13), state_color.get(state, red))
        put(scr, y, L + 15, ('· ' + detail)[:CW - 13],
            red if state in ('ERROR', 'NO LINK') else state_color.get(state, text))
        y += 1
        sides(y)
        items = [(k, label, KEYS_SHORT.get(k, label)) for k, _, label in KEYS]
        widths = sum(4 + len(sl) for _, _, sl in items)
        gap = 2 if widths + 2 * (len(items) - 1) <= CW else 1
        x = L + 2
        for k, cmd, _label in KEYS:
            short = KEYS_SHORT.get(k, _label)
            on = cmd in allowed
            kattr = (red if k == 'X' else bold) if on else dim
            put(scr, y, x, f'[{k.lower()}]', kattr)
            put(scr, y, x + 4, short, (red if k == 'X' else text) if on else dim)
            x += 4 + len(short) + gap
        y += 1
        sides(y)
        recent = node.last_sent and time.time() - node.last_sent_wall < 6.0
        if recent or node.exit_at is not None:
            msg = node.last_sent
            attr = amber if ('again' in msg or 'not possible' in msg) else cyan
            put(scr, y, L + 2, 'operator', dim)
        else:
            meaning = st.get('start_meaning', '')
            msg = f'Press S to {meaning}' if meaning and 'START' in allowed else ''
            if st.get('skip_meaning') and 'SKIP' in allowed:
                msg += f' · K to {st["skip_meaning"]}'
            if not msg and state == 'EXPLORING':
                msg = 'Exploring · P pause · F finish mapping'
            attr = cyan
            put(scr, y, L + 2, 'ready' if 'START' in allowed else '', dim)
        put(scr, y, L + 2 + max(10, (CW - len(msg)) // 2), msg[:CW - 10], attr)
        y += 1

        # ── connections | nodes ─────────────────────────────────────────
        section(y, 'connections', mid_title='nodes')
        y += 1
        for i in range(n_split):
            sides(y, split=True)
            if i < len(conn):
                lab, val, attr = conn[i]
                row(y, L + 2, lab, val, attr, lw=13, width=half)
            if i < len(nodes):
                num, name, word, attr = nodes[i]
                put(scr, y, M + 2, num.ljust(7), dim)
                put(scr, y, M + 9, name.ljust(18), text)
                put(scr, y, M + 27, word[:L + W - M - 28], attr)
            y += 1

        # ── exploration (Phase 1) or restocking (Phase 2) | navigation ─
        section(y, ('route' if rs.get('mode') == 'route_only' else 'restocking') if rs
                else ('map tour' if tour else 'exploration'),
                mid_title='navigation')
        y += 1
        ex_status = ex.get('status') or '–'
        nav_status = nav.get('status') or '–'
        safety = nav.get('safety', '–')
        ex_rows = [
            ('status', ex_status, bold),
            ('frontiers', ex.get('frontier_count', '–'), green),
            ('reached', ex.get('reached', '–'), green),
            ('failed', ex.get('failed', '–'),
             amber if str(ex.get('failed', '0')) not in ('0', '–') else green),
            ('blacklist', ex.get('blacklist_count', '–'), green),
        ]
        nav_rows = [
            ('position', f'{fnum(slam.get("x"))},{fnum(slam.get("y"))},'
                         f'{fnum(slam.get("yaw_deg"), "{:.0f}")}°', text),
            ('status', nav_status,
             red if str(nav_status).startswith(('BLOCKED', 'RECOVER', 'LOCAL')) else text),
            ('target', (f'({fnum(ex.get("target_x"))}, {fnum(ex.get("target_y"))})'
                        if ex.get('target_x') is not None else 'none'), text),
            ('dist tgt', (f'{fnum(ex.get("dist_target_m"), "{:.1f}")} m · '
                          f'travelled {fnum(ex.get("dist_travelled_m"), "{:.1f}")} m'), text),
            ('safety', f'{safety} · {fnum(nav.get("closest_obstacle_m"))} m',
             green if safety == 'CLEAR' else (dim if safety == '–' else amber)),
        ]
        if tour and not rs:
            ex_rows = tour_rows(tour, C)
            nav_rows[2] = ('profile', nav.get('profile', '–'), text)
            nav_rows[3] = ('remaining', f'{fnum(nav.get("distance_remaining"), "{:.2f}")} m · '
                                        f'{nav.get("task", "–")}', text)
        if rs:
            ex_rows, map_rows_r = restock_rows(rs, C)
            nav_rows[2] = ('profile', nav.get('profile', '–'), text)
            nav_rows[3] = ('remaining', f'{fnum(nav.get("distance_remaining"), "{:.2f}")} m · '
                                        f'{nav.get("task", "–")}', text)
        for i in range(5):
            sides(y, split=True)
            lab, val, attr = ex_rows[i]
            row(y, L + 2, lab, val, attr, width=half)
            lab, val, attr = nav_rows[i]
            row(y, M + 2, lab, val, attr, width=L + W - M - 4)
            y += 1

        # ── map (Phase 1) or boxes (Phase 2) ────────────────────────────
        section(y, 'boxes' if rs else 'map')
        y += 1
        try:
            res = float(slam['res'])
            extent = (f'{int(slam["map_w"]) * res:.1f} × {int(slam["map_h"]) * res:.1f} m '
                      f'@ {res:.3f} m/cell')
        except (KeyError, TypeError, ValueError):
            extent = '–'
        growth = node.area_growth()
        lm = st.get('last_map') or {}
        if st.get('saving_map'):
            saved, sattr = 'saving...', amber
        elif lm:
            saved = (f'{lm.get("time", "")} '
                     f'{"saved" if lm.get("ok") else "FAILED"} · {lm.get("message", "")}')
            sattr = green if lm.get('ok') else red
        else:
            saved, sattr = 'not saved yet', dim
        map_rows = [
            ('extent', extent, text),
            ('known mapped area', f'{fnum(slam.get("known_m2"), "{:.1f}")} m²', green),
            ('map growth', ('–' if growth is None else f'{growth:+.2f} m²/min'), text),
            ('slam', f'{slam.get("state", "–")} · {slam.get("scan_hz", "–")} Hz · '
                     f'{slam.get("keyframes", "–")} kf · {slam.get("loops", "–")} loops · '
                     f'{slam.get("corrections", "–")} corr',
             green if slam.get('state') == 'RUNNING' else amber),
            ('last save', saved, sattr),
        ]
        if rs:
            map_rows = map_rows_r
        for lab, val, attr in map_rows:
            sides(y)
            row(y, L + 2, lab, val, attr, lw=19, width=CW)
            y += 1

        # ── events ──────────────────────────────────────────────────────
        section(y, 'events')
        y += 1
        events = st.get('events', [])[-3:]
        for i in range(3):
            sides(y)
            if i < len(events):
                t, msg = events[i]
                put(scr, y, L + 2, t, dim)
                put(scr, y, L + 11, msg[:CW - 9],
                    red if msg.startswith(('ERROR', 'rejected')) else text)
            y += 1

        # ── system ──────────────────────────────────────────────────────
        section(y, 'system')
        y += 1
        sides(y)
        row(y, L + 2, 'monitor input', (f'/mission/status age {age:.1f} s' if age is not None
                                        else 'no status yet'),
            green if linked else red, lw=19, width=CW)
        y += 1
        sides(y)
        log_dir = str(st.get('log_dir', '') or '–')
        if len(log_dir) > CW - 19:
            log_dir = '…' + log_dir[-(CW - 20):]
        row(y, L + 2, 'logs', log_dir, dim, lw=19, width=CW)
        y += 1
        sides(y)
        row(y, L + 2, 'refresh', f'{1.0 / UI_PERIOD_S:.0f} Hz UI · status 2 Hz', text,
            lw=19, width=CW)
        put(scr, y, L + W - 11, '[q] quit', dim)
        y += 1
        put(scr, y, L, '└' + '─' * (W - 2) + '┘', dim)
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