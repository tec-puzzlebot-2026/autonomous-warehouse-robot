"""Tests of the restocking state machine with a simulated robot (no ROS needed).

The fake ports move the robot instantly (after a short delay) and synthesize what
the detectors would report: the far and near box detectors project the simulated
boxes through the same pinhole camera the state machine uses, the QR reader sees a
box's QR code only when its face looks at the camera, and the logo detector sees the
dock's company at its entrance.

Run:  cd main_ros && python3 -m pytest test/test_restock_core.py -v
"""
# Author: Raymundo Gomez

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import restock_world as RW  # noqa: E402

from main_ros.restock import geometry as G  # noqa: E402
from main_ros.restock import mission_store as MS  # noqa: E402
from main_ros.restock import restock_core as rc  # noqa: E402
from main_ros.restock.docks import find_docks  # noqa: E402
from main_ros.restock.geometry import wrap  # noqa: E402

FACE_ANGLE = {'front': 0.0, 'left': math.pi / 2, 'back': math.pi, 'right': -math.pi / 2}


class SimBox:
    def __init__(self, x, y, yaw, level, company, qr_face='front'):
        self.x, self.y, self.yaw, self.level = x, y, yaw, level
        self.company, self.qr_face = company, qr_face
        self.state = 'shelf'                    # shelf, carried, floor, dock


class Handle:
    def __init__(self, ports, delay, ok=True, msg='', on_done=None):
        self.p, self.t, self.ok, self.msg, self.on_done = ports, ports.t + delay, ok, msg, on_done
        self.canceled = False
        self._fired = False

    def done(self):
        if self.p.t >= self.t and not self._fired:
            self._fired = True
            if self.ok and self.on_done and not self.canceled:
                self.on_done()
        return self.p.t >= self.t or self.canceled

    def result(self):
        return (False, 'canceled') if self.canceled else (self.ok, self.msg)

    def cancel(self):
        self.canceled = True


class SimPorts:
    def __init__(self, folder, boxes, start=(0.6, 0.6, 0.0), localized=True, dock_logos=None):
        self.t = 0.0
        self.folder = folder
        self.boxes = boxes
        self.pose = list(start)
        self.localized = localized
        self.ekf_track_after = 3.0
        self.cam = G.Camera(500.0, 500.0, 320.0, 240.0, (0.10005, 0.0, 0.178, 0.0, 0.26, -0.0124))
        self.Tinv = np.linalg.inv(self.cam.T)
        self.enabled = {}
        self.enable_log = []
        self.logs = []
        self.calls = []
        self.carrying = False
        self.carried = None
        self.pick_fail = 0
        self.set_down_fail = 0
        self.nav_fail = set()
        self.dock_logos = dock_logos or {}       # dock company -> logo shown there
        self.spec = G.BoxSpec()
        self.front_x = 0.12
        self.seeded = 0
        self.relocalized = 0
        self.nav_delay = 1.0

    # ── time and log ─────────────────────────────────────────────────────
    def now(self):
        return self.t

    def log(self, text):
        self.logs.append(text)

    def mission_folder(self):
        return self.folder

    # ── motion ───────────────────────────────────────────────────────────
    def navigate(self, pose, profile):
        self.calls.append(('nav', tuple(round(v, 2) for v in pose), profile))
        key = (round(pose[0], 1), round(pose[1], 1))
        if key in self.nav_fail:
            return Handle(self, 0.5, False, 'ABORTED')

        def arrive():
            self.pose = list(pose)
        return Handle(self, self.nav_delay, on_done=arrive)

    def align(self, target, standoff, profile):
        self.calls.append(('align', tuple(round(v, 2) for v in target), standoff))

        def arrive():
            self.pose = list(G.pose_facing(target[:2], target[2], standoff))
        return Handle(self, 0.5, on_done=arrive)

    def drive_straight(self, distance, centered=False):
        self.calls.append(('drive', round(distance, 3)))

        def moved():
            self.pose[0] += distance * math.cos(self.pose[2])
            self.pose[1] += distance * math.sin(self.pose[2])
        return Handle(self, 0.3, on_done=moved)

    def rotate(self, angle):
        self.calls.append(('rotate', round(angle, 3)))

        def turned():
            self.pose[2] = wrap(self.pose[2] + angle)
        return Handle(self, 0.3, on_done=turned)

    def send_velocity(self, v, w):
        self.cmd = (float(v), float(w))

    def step(self, dt):
        """Integrate the last velocity command (visual servo) over dt."""
        v, w = getattr(self, 'cmd', (0.0, 0.0))
        self.pose[0] += v * dt * math.cos(self.pose[2])
        self.pose[1] += v * dt * math.sin(self.pose[2])
        self.pose[2] = wrap(self.pose[2] + w * dt)

    def cancel_motion(self):
        self.calls.append(('cancel_motion',))
        self.cmd = (0.0, 0.0)

    def clear_costmaps(self):
        pass

    # ── forklift ─────────────────────────────────────────────────────────
    def _box_ahead(self, max_gap=0.12):
        x, y, th = self.pose
        for b in self.boxes:
            if b.state not in ('shelf', 'floor'):
                continue
            d = math.hypot(b.x - x, b.y - y) - self.front_x - self.spec.depth / 2
            ang = abs(wrap(math.atan2(b.y - y, b.x - x) - th))
            if -0.02 <= d <= max_gap and ang < 0.3:
                return b
        return None

    def pick(self, box_id, level, face_pose, distance):
        self.calls.append(('pick', box_id, level, round(distance, 3)))
        if self.pick_fail > 0:
            self.pick_fail -= 1
            return Handle(self, 1.0, False, 'injected failure')
        b = self._box_ahead()
        if b is None:
            return Handle(self, 1.0, False, 'no box in front of the forks')

        def lift():
            b.state = 'carried'
            self.carried = b
        return Handle(self, 1.0, on_done=lift)

    def set_down(self, box_id, where):
        self.calls.append(('set_down', box_id, where))
        if self.set_down_fail > 0:
            self.set_down_fail -= 1
            return Handle(self, 1.0, False, 'injected failure')
        b = self.carried

        def drop():
            d = self.front_x + 0.02 + self.spec.depth / 2
            b.x = self.pose[0] + d * math.cos(self.pose[2])
            b.y = self.pose[1] + d * math.sin(self.pose[2])
            b.yaw = wrap(self.pose[2] + math.pi)       # the face it was picked by looks back
            b.level = 0
            b.state = 'dock' if where == 'dock' else 'floor'
            self.carried = None
        return Handle(self, 1.0, on_done=drop)

    def cancel_forklift(self):
        self.calls.append(('cancel_forklift',))

    def set_carrying(self, on):
        self.carrying = bool(on)

    # ── localization ─────────────────────────────────────────────────────
    def slam_pose(self):
        return tuple(self.pose) if self.localized else None

    def slam_localized(self):
        return self.localized

    def ekf_status(self):
        return 'TRACK' if self.t >= self.ekf_track_after else 'SEARCH'

    def ekf_pose(self):
        return tuple(self.pose) if self.ekf_status() == 'TRACK' else None

    def seed_slam(self, pose):
        self.seeded += 1
        self.localized = True

    def ekf_relocalize(self):
        self.relocalized += 1

    def scan_free_direction(self):
        return (0.0, 1.5)

    def landmarks(self):
        return {}

    def camera(self):
        return self.cam

    # ── perception (synthesized) ─────────────────────────────────────────
    def enable(self, stream, on):
        if self.enabled.get(stream) != on:
            self.enable_log.append((self.t, stream, on))
        self.enabled[stream] = on

    def _face(self, b):
        """The face of b that looks most toward the camera: (name, center, normal)."""
        x, y, _ = self.pose
        foot = G.BoxFootprint(b.x, b.y, b.yaw, self.spec)
        return max(foot.faces(), key=lambda f: math.cos(wrap(math.atan2(y - f[1][1], x - f[1][0])
                                                                 - f[2])))

    def _project(self, b):
        """Image box [x1, y1, x2, y2] of b's visible face, or None if not in view."""
        name, (fx, fy), normal = self._face(b)
        half = (self.spec.width if name in ('front', 'back') else self.spec.depth) / 2
        tx, ty = -math.sin(normal), math.cos(normal)
        z0 = self.spec.level_heights[b.level] - self.cam.base_height
        us, vs = [], []
        for s in (-half, half):
            for z in (z0, z0 + self.spec.height):
                pw = (fx + s * tx, fy + s * ty)
                pb = G.relative(tuple(self.pose), (pw[0], pw[1], 0.0))
                po = self.Tinv @ np.array([pb[0], pb[1], z, 1.0])
                if po[2] <= 0.05:
                    return None
                us.append(self.cam.cx + self.cam.fx * po[0] / po[2])
                vs.append(self.cam.cy + self.cam.fy * po[1] / po[2])
        if max(us) < 0 or min(us) > 640 or max(vs) < 0 or min(vs) > 480:
            return None
        return [max(min(us), 0), max(min(vs), 0), min(max(us), 640), min(max(vs), 480)]

    def _visible(self, max_range):
        out = []
        for b in self.boxes:
            if b.state in ('carried',):
                continue
            d = math.hypot(b.x - self.pose[0], b.y - self.pose[1])
            if d > max_range:
                continue
            bb = self._project(b)
            if bb is not None:
                out.append((d, b, bb))
        return sorted(out, key=lambda t: t[0])

    def messages(self, stream, since):
        stamps = [since + 0.1 * k for k in range(1, 4)]
        if stream == 'box_far':
            vis = self._visible(3.5)
            alls = [bb + [0.9] for _, _, bb in vis]
            return [(t, {'found': bool(vis), 'stamp': t, 'box': vis[0][2] if vis else None,
                         'all': alls}) for t in stamps]
        if stream == 'box_target':
            vis = [v for v in self._visible(0.9)]
            if not vis:
                return [(t, {'found': False, 'stamp': t}) for t in stamps]
            _, b, bb = vis[0]
            self._near = (b, bb)
            return [(t, {'found': True, 'stamp': t, 'box': bb, 'frame': [640, 480]})
                    for t in stamps]
        if stream == 'qr':
            vis = self._visible(0.9)
            codes = []
            if vis:
                _, b, bb = vis[0]
                name, (fx, fy), normal = self._face(b)
                view = math.atan2(self.pose[1] - fy, self.pose[0] - fx)
                if name == b.qr_face and abs(wrap(view - normal)) < math.radians(55):
                    cu, cv = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                    w, h = 0.25 * (bb[2] - bb[0]), 0.25 * (bb[3] - bb[1])
                    codes.append({'text': b.company, 'corners': [[cu - w, cv - h], [cu + w, cv - h],
                                                                 [cu + w, cv + h], [cu - w, cv + h]]})
            return [(t, {'stamp': t, 'codes': codes}) for t in stamps]
        if stream == 'logos':
            for c, d in self.world_docks.items():
                if math.hypot(d.entrance[0] - self.pose[0], d.entrance[1] - self.pose[1]) < 0.1:
                    shown = self.dock_logos.get(c, c)
                    return [(t, {'detections': [{'company': shown, 'conf': 0.9}]}) for t in stamps]
            return [(t, {'detections': []}) for t in stamps]
        return []

    def scan_ahead(self, half_width=0.10, max_range=1.5):
        """Nearest thing straight ahead on the LiDAR: a box face (for a rack box, the
        rack's edge in line with it), plus the extra obstacles in self.walls_ahead."""
        best = None
        for b in self.boxes:
            if b.state not in ('shelf', 'floor'):
                continue
            name, (fx, fy), normal = self._face(b)
            rel = G.relative(tuple(self.pose), (fx, fy, 0.0))
            if 0.0 < rel[0] < max_range and abs(rel[1]) < half_width:
                best = rel[0] if best is None else min(best, rel[0])
        for wx, wy in getattr(self, 'walls_ahead', []):
            rel = G.relative(tuple(self.pose), (wx, wy, 0.0))
            if 0.0 < rel[0] < max_range and abs(rel[1]) < half_width:
                best = rel[0] if best is None else min(best, rel[0])
        return best

    def scan_face(self):
        """LiDAR fit of a floor box face right in front of the robot."""
        x, y, th = self.pose
        for b in self.boxes:
            if b.state not in ('shelf', 'floor') or b.level != 0:
                continue
            name, (fx, fy), normal = self._face(b)
            rel = G.relative(tuple(self.pose), (fx, fy, 0.0))
            if 0.0 < rel[0] < 0.6 and abs(rel[1]) < 0.09:
                yaw_err = wrap(normal - th - math.pi)
                return rel[0] + rel[1] * math.tan(yaw_err), yaw_err
        return None


# =============================================================================
def make_folder(tmp_path, markers=True, drop_dock=None):
    grid = RW.make_world()
    folder = str(tmp_path / 'mission')
    os.makedirs(folder)
    base = str(tmp_path / 'saved')
    RW.write_map_files(base, grid)
    MS.copy_map_files(base, folder)
    docks = find_docks(grid, RW.logo_positions())
    if drop_dock:
        docks.pop(drop_dock)
    MS.write_yaml(os.path.join(folder, MS.DOCKS), {'docks': [d.to_dict() for d in docks.values()]})
    lms = [{'id': 7, 'x': 1.5, 'y': RW.H, 'yaw': -1.5708}] if markers else []
    MS.write_yaml(os.path.join(folder, MS.LANDMARKS), {'markers': lms})
    return folder, docks


def make_core(tmp_path, boxes, mission='full', settings=None, **kw):
    folder, docks = make_folder(tmp_path, drop_dock=kw.pop('drop_dock', None))
    ports = SimPorts(folder, boxes, **kw)
    ports.world_docks = docks
    # the full QR search (all faces, floor fallback); the no-QR shortcut is tested below
    s = settings or rc.RestockSettings(
        approach_mode='sendero', look_settle_s=0.2, look_s=0.5, read_s=0.5,
        acquire_s=0.4, logo_look_s=0.4, survey=False,
        inspect_other_faces=True, no_qr_company='')
    core = rc.RestockCore(ports, s, mission)
    return core, ports


def run(core, ports, until_done=True, max_s=3000.0, dt=0.1, stop_when=None):
    core.start()
    while ports.t < max_s:
        core.tick()
        ports.step(dt)
        ports.t = round(ports.t + dt, 3)
        if until_done and core.done:
            return
        if stop_when and stop_when():
            return
        if core.error:
            return


def run_on(core, ports, max_s=3000.0, dt=0.1, stop_when=None):
    """Keep ticking a core that run() already started."""
    while ports.t < max_s and not core.done and not core.error:
        core.tick()
        ports.step(dt)
        ports.t = round(ports.t + dt, 3)
        if stop_when and stop_when():
            return


def box_against_left_wall(company='walnut', qr='front'):
    return SimBox(0.0 + 0.05 + 0.06, 2.0, 0.0, 0, company, qr)         # pick face +x


def box_on_rack(company='emezon', qr='front'):
    return SimBox(2.3, 1.20 - 0.06, -math.pi / 2, 1, company, qr)      # rack edge, faces -y


# =============================================================================
def test_full_run_delivers_floor_and_rack_boxes(tmp_path):
    boxes = [box_against_left_wall('walnut'), box_on_rack('emezon')]
    core, ports = make_core(tmp_path, boxes)
    run(core, ports)
    assert core.done and core.outcome.startswith('COMPLETE'), core.outcome
    assert [b.state for b in boxes] == ['dock', 'dock'], [p for p in ports.logs[-15:]]
    docks = ports.world_docks
    for b in boxes:
        assert docks[b.company].contains(b.x, b.y), (b.company, b.x, b.y)
    assert core.per_company() == {'soriene': 0, 'emezon': 1, 'walnut': 1}
    assert not ports.carrying
    log = MS.read_yaml(os.path.join(ports.folder, MS.LOG))
    assert len(log['runs'][0]['boxes']) == 2
    assert os.path.isfile(os.path.join(ports.folder, MS.ROUTE))
    # one network at a time, besides the far box detector (boxes_always_on)
    on = set()
    for _, stream, state in ports.enable_log:
        (on.add if state else on.discard)(stream)
        assert len(on - {'qr', 'box_far'}) <= 1, on


VISION = dict(look_settle_s=0.2, look_s=0.5, read_s=0.5, acquire_s=0.4, logo_look_s=0.4,
              survey=False, approach_mode='vision', no_qr_company='walnut')


def test_vision_approach_servos_squares_reads_and_picks(tmp_path):
    """Classical CV approach: no Align action at the box (ALIGN is the Sendero path)."""
    boxes = [box_against_left_wall('emezon'), box_on_rack('soriene')]
    core, ports = make_core(tmp_path, boxes, settings=rc.RestockSettings(**VISION))
    phases = []
    orig = core._goto

    def spy(phase, **a):
        phases.append(phase)
        return orig(phase, **a)
    core._goto = spy
    run(core, ports)
    assert core.done and core.outcome.startswith('COMPLETE'), (core.outcome, ports.logs[-15:])
    assert [b.state for b in boxes] == ['dock', 'dock'], ports.logs[-15:]
    # the rack box is picked by its front face (QR); the wall box by the face the
    # robot reached: with no QR there it goes to walnut (no_qr_company)
    assert core.per_company()['soriene'] == 1
    assert sum(core.per_company().values()) == 2
    for p in ('SERVO', 'SQUARE', 'READ_QR', 'CLOSE_IN', 'PICK'):
        assert p in phases, p
    assert 'INSPECT' not in phases and 'ALIGN' not in phases


def test_vision_approach_squares_up_to_a_skewed_face(tmp_path):
    for deg in (25, -25):
        box = box_against_left_wall('emezon')
        box.yaw = math.radians(deg)                   # box turned on the floor
        d = tmp_path / str(deg)
        d.mkdir()
        core, ports = make_core(d, [box], settings=rc.RestockSettings(**VISION))
        run(core, ports, stop_when=lambda: core.phase == 'READ_QR')
        fit = ports.scan_face()
        assert fit is not None and abs(fit[1]) <= math.radians(3.0), (deg, fit)
        assert abs(core.ph.get('gap') or 0.0) < 0.3
        run_on(core, ports)
        assert core.done and box.state == 'dock', ports.logs[-15:]


def test_vision_approach_without_qr_goes_to_walnut(tmp_path):
    box = box_against_left_wall('emezon', qr='back')  # QR not on the front face
    core, ports = make_core(tmp_path, [box], settings=rc.RestockSettings(**VISION))
    run(core, ports)
    assert core.done and box.state == 'dock', ports.logs[-15:]
    assert ports.world_docks['walnut'].contains(box.x, box.y)


def test_vision_servo_takes_over_when_the_sendero_leg_never_ends(tmp_path):
    """Sendero gets the robot to the box but never reports the goal done (it hangs in
    its final turn): the servo takes over instead of waiting for ever."""
    box = box_against_left_wall('emezon')
    core, ports = make_core(tmp_path, [box], settings=rc.RestockSettings(
        **dict(VISION, servo_start_gap=0.3)))
    plain = ports.navigate

    def navigate(pose, profile):
        if core.phase == 'APPROACH':
            ports.calls.append(('nav', tuple(round(v, 2) for v in pose), profile))
            ports.pose = list(pose)                    # there, but never "succeeded"
            return Handle(ports, 1e9)
        return plain(pose, profile)
    ports.navigate = navigate
    run(core, ports, stop_when=lambda: core.phase == 'SERVO')
    assert core.phase == 'SERVO', ports.logs[-10:]
    assert any('visual servo takes over' in t for t in ports.logs), ports.logs[-10:]
    run_on(core, ports)
    assert core.done and box.state == 'dock', ports.logs[-15:]


def box_depth():
    return G.BoxSpec().depth


def test_vision_servo_stops_on_the_lidar_before_a_rack_edge(tmp_path):
    """Something sticks out in front of the box (a rack edge): the LiDAR corridor ahead
    stops the servo there, the robot never drives into it."""
    box = box_on_rack('soriene')
    core, ports = make_core(tmp_path, [box], settings=rc.RestockSettings(**VISION))
    run(core, ports, stop_when=lambda: core.phase == 'SERVO')
    x, y, _ = ports.pose
    a = math.atan2(box.y - y, box.x - x)                     # robot -> box
    d = math.hypot(box.x - x, box.y - y) - box_depth() / 2 - 0.12   # 12 cm before the face
    ports.walls_ahead = [(x + d * math.cos(a) - k * math.sin(a), y + d * math.sin(a) +
                          k * math.cos(a)) for k in (-0.05, 0.0, 0.05)]
    run_on(core, ports, stop_when=lambda: core.phase == 'SQUARE')
    assert any('by LiDAR' in t for t in ports.logs), ports.logs[-10:]
    edge = ports.scan_ahead()
    assert edge is not None and edge - core.s.robot_front_x > 0.15


def test_vision_servo_logs_what_it_sees_and_commands(tmp_path):
    box = box_against_left_wall('emezon')
    core, ports = make_core(tmp_path, [box], settings=rc.RestockSettings(**VISION))
    run(core, ports, stop_when=lambda: core.phase == 'SQUARE')
    lines = [t for t in ports.logs if ' servo: box ' in t]
    assert lines and all('-> v ' in t for t in lines)
    v = [float(t.split('-> v ')[1].split()[0]) for t in lines]
    assert all(x == 0.0 or x >= core.s.servo_min_v - 1e-9 for x in v)   # no creeping


def test_vision_approach_ends_lost_when_the_box_leaves_the_image(tmp_path):
    box = box_against_left_wall('emezon')
    core, ports = make_core(tmp_path, [box], settings=rc.RestockSettings(**VISION))
    run(core, ports, stop_when=lambda: core.phase == 'SERVO')
    box.state = 'carried'                             # gone from the image and the scan
    run_on(core, ports, stop_when=lambda: core.phase != 'SERVO', max_s=ports.t + 30)
    assert any('left the image' in t for t in ports.logs), ports.logs[-10:]
    assert ports.cmd == (0.0, 0.0)                    # stopped, not driving blind


def test_viewpoints_use_the_quick_profile_and_boxes_the_precise_one(tmp_path):
    s = rc.RestockSettings(
        approach_mode='sendero', look_settle_s=0.2, look_s=0.5, read_s=0.5, acquire_s=0.4,
        logo_look_s=0.4, survey=False, profile_viewpoint='patrol')
    core, ports = make_core(tmp_path, [box_against_left_wall('walnut')], settings=s)
    run(core, ports)
    assert core.done and core.outcome.startswith('COMPLETE'), core.outcome
    profiles = {c[2] for c in ports.calls if c[0] == 'nav'}
    assert 'patrol' in profiles                            # route viewpoints
    assert profiles - {'patrol'}                           # box faces / docks: the others
    route = {tuple(round(v, 2) for v in vp) for vp in core.route}
    assert all(c[2] == 'patrol' for c in ports.calls if c[0] == 'nav' and c[1] in route)


def test_qr_on_a_side_face_is_found_by_visiting_faces(tmp_path):
    b = SimBox(3.0, 0.05 + 0.05, math.pi / 2, 0, 'soriene', qr_face='left')   # bottom wall, faces +y
    core, ports = make_core(tmp_path, [b])
    run(core, ports)
    assert b.state == 'dock' and ports.world_docks['soriene'].contains(b.x, b.y)
    reg = core.boxes[0]
    assert reg.company == 'soriene' and not reg.data_missing


def test_qr_against_the_wall_uses_the_floor_fallback(tmp_path):
    b = box_against_left_wall('emezon', qr='back')        # QR face touches the wall
    core, ports = make_core(tmp_path, [b])
    run(core, ports)
    reg = core.boxes[0]
    assert reg.data_missing is True and reg.company == 'emezon', reg
    assert ('set_down', reg.id, 'floor') in ports.calls
    assert sum(1 for c in ports.calls if c[0] == 'pick') == 2
    assert b.state == 'dock' and ports.world_docks['emezon'].contains(b.x, b.y)


def test_box_already_in_a_dock_is_ignored(tmp_path):
    folder, docks = make_folder(tmp_path)
    d = docks['walnut']
    inside = SimBox(d.inside[0], d.inside[1], wrap(d.inside[2] + math.pi), 0, 'walnut')
    inside.state = 'dock'
    core, ports = make_core(tmp_path / 'x', [inside])
    run(core, ports)
    assert core.boxes == [] and core.done


def test_pick_failure_retries_once_then_gives_up(tmp_path):
    b = box_against_left_wall('walnut')
    core, ports = make_core(tmp_path, [b])
    ports.pick_fail = 1
    run(core, ports)
    assert b.state == 'dock'
    assert sum(1 for c in ports.calls if c[0] == 'pick') == 2

    b2 = box_against_left_wall('walnut')
    core2, ports2 = make_core(tmp_path / 'b', [b2])
    ports2.pick_fail = 5
    run(core2, ports2)
    assert core2.done and core2.boxes[0].status == 'failed'
    assert sum(1 for c in ports2.calls if c[0] == 'pick') == 2
    assert b2.state == 'shelf' and not ports2.carrying


def test_no_dock_for_the_company_leaves_the_box_in_the_free_area(tmp_path):
    b = box_against_left_wall('walnut')
    core, ports = make_core(tmp_path, [b], drop_dock='walnut')
    run(core, ports)
    assert core.boxes[0].status == 'no_dock'
    assert b.state == 'floor' and not ports.carrying
    assert ('set_down', 'box_01', 'floor') in ports.calls


def test_wrong_logo_at_the_dock_tries_the_other_docks(tmp_path):
    b = box_against_left_wall('walnut')
    # The walnut dock shows the soriene logo and the soriene dock the walnut logo.
    core, ports = make_core(tmp_path, [b], dock_logos={'walnut': 'soriene', 'soriene': 'walnut'})
    run(core, ports)
    assert b.state == 'dock'
    assert ports.world_docks['soriene'].contains(b.x, b.y)


def test_set_down_failure_at_the_dock_holds_with_the_box(tmp_path):
    b = box_against_left_wall('walnut')
    core, ports = make_core(tmp_path, [b])
    ports.set_down_fail = 2
    run(core, ports)
    assert core.error and 'holding the box' in core.error
    assert ports.carrying and b.state == 'carried'
    core.resume()                                 # the operator checked the forklift: S
    run_more(core, ports)
    assert b.state == 'dock' and not ports.carrying


def run_more(core, ports, max_s=3000.0, dt=0.1):
    while ports.t < max_s and not core.done and not core.error:
        core.tick()
        ports.step(dt)
        ports.t = round(ports.t + dt, 3)


def test_pause_cancels_the_forklift_and_resume_realigns(tmp_path):
    b = box_against_left_wall('walnut')
    core, ports = make_core(tmp_path, [b])
    run(core, ports, until_done=False, stop_when=lambda: core.phase == 'PICK')
    core.pause()
    assert ('cancel_forklift',) in ports.calls and core.paused
    for _ in range(20):
        core.tick()
        ports.t += 0.1
    assert core.phase == 'PICK'
    core.resume()
    assert core.phase == 'ALIGN'
    run_more(core, ports)
    assert b.state == 'dock'


def test_restocking_mode_localizes_from_the_ekf(tmp_path):
    core, ports = make_core(tmp_path, [], mission='restocking', localized=False)
    run(core, ports)
    assert ports.seeded >= 1 and core.done
    assert any('localized at' in t for t in ports.logs)


def test_no_aruco_found_stops_with_an_error(tmp_path):
    s = rc.RestockSettings(
        approach_mode='sendero', localize_timeout_s=60.0, search_pause_s=0.5, survey=False)
    core, ports = make_core(tmp_path, [], mission='restocking', localized=False, settings=s)
    ports.ekf_track_after = 1e9
    run(core, ports)
    assert core.error and 'no ArUco' in core.error
    assert ports.relocalized >= 2                     # it moved and searched again
    turns = [c for c in ports.calls if c[0] == 'rotate']
    assert len(turns) >= 12                           # at least one full turn in steps


def test_unreachable_viewpoint_is_skipped(tmp_path):
    core, ports = make_core(tmp_path, [])
    core.start()
    keys = []
    for p in core.route[1:]:                      # the first goal is already on its way
        k = (round(p[0], 1), round(p[1], 1))
        if k not in keys and k != (round(core.route[0][0], 1), round(core.route[0][1], 1)):
            keys.append(k)
    ports.nav_fail = set(keys[:2])
    expected = sum(1 for p in core.route[1:] if (round(p[0], 1), round(p[1], 1)) in ports.nav_fail)
    run_more(core, ports)
    assert core.done and core.uncovered == expected >= 2


def test_disagreement_triggers_relocalization(tmp_path):
    core, ports = make_core(tmp_path, [])
    ports.nav_delay = 8.0                         # a long drive: the poses drift apart on it
    core.start()
    real = ports.ekf_pose
    ports.ekf_pose = lambda: (ports.pose[0] + 1.0, ports.pose[1], ports.pose[2])
    for _ in range(200):
        core.tick()
        ports.t += 0.1
        if core.phase == 'LOCALIZE':
            break
    assert core.phase == 'LOCALIZE' and ports.relocalized == 1
    ports.ekf_pose = real
    run_more(core, ports)
    assert core.done


def test_survey_finds_a_missing_dock(tmp_path):
    b = box_against_left_wall('walnut')
    s = rc.RestockSettings(
        approach_mode='sendero', look_settle_s=0.2, look_s=0.5, read_s=0.5, acquire_s=0.4,
        logo_look_s=0.4, survey=True, survey_look_s=0.3)
    core, ports = make_core(tmp_path, [b], settings=s, drop_dock='walnut')
    lx, ly, lyaw = RW.logo_positions()['walnut']
    ports.landmarks = lambda: {'logos': [{'id': 'walnut', 'x': lx, 'y': ly, 'yaw': lyaw}]}
    run(core, ports)
    assert 'walnut' in core.world.docks
    assert b.state == 'dock' and core.world.docks['walnut'].contains(b.x, b.y)
    saved = MS.read_yaml(os.path.join(ports.folder, MS.DOCKS))
    assert 'walnut' in [d['company'] for d in saved['docks']]


def test_weak_far_detections_do_not_start_a_box_task(tmp_path):
    """Below min_far_conf (e.g. something yellow that is not a box) nothing is approached."""
    boxes = [box_against_left_wall('walnut')]
    core, ports = make_core(tmp_path, boxes)
    plain = ports.messages

    def weak(stream, since):
        out = plain(stream, since)
        if stream == 'box_far':
            for _, d in out:
                d['all'] = [b[:4] + [0.5] for b in d['all']]
                d['conf'] = 0.5
        return out
    ports.messages = weak
    run(core, ports)
    assert core.done and not core.boxes


def test_a_box_seen_while_driving_stops_the_patrol(tmp_path):
    boxes = [box_against_left_wall('walnut')]
    core, ports = make_core(tmp_path, boxes)
    run(core, ports)
    assert core.done and core.outcome.startswith('COMPLETE'), core.outcome
    assert any('spotted while driving' in line or 'spotted at' in line for line in ports.logs)


def test_a_goal_sendero_holds_without_moving_is_given_up(tmp_path):
    """The approach goal never ends and the robot does not move: after nav_stall_s it is
    retried once, then the box is given up and the patrol goes on."""
    boxes = [box_against_left_wall('walnut')]
    core, ports = make_core(tmp_path, boxes)
    core.s.nav_stall_s = 5.0
    plain = ports.navigate

    def navigate(pose, profile):
        if core.phase == 'APPROACH':
            ports.calls.append(('nav', tuple(round(v, 2) for v in pose), profile))
            return Handle(ports, 1e9)                  # held for ever, robot still
        return plain(pose, profile)
    ports.navigate = navigate
    run(core, ports)
    assert core.done
    assert any('no progress' in line for line in ports.logs)
    assert any(b.status == 'failed' for b in core.boxes)


def test_no_qr_on_the_front_face_goes_to_the_default_company(tmp_path):
    """The default settings: only the face the robot meets is read; a box without a
    readable QR there is picked anyway and delivered to no_qr_company (walnut)."""
    b = box_against_left_wall('emezon', qr='back')        # its QR is on another face
    s = rc.RestockSettings(
        approach_mode='sendero', look_settle_s=0.2, look_s=0.5, read_s=0.5, acquire_s=0.4,
        logo_look_s=0.4, survey=False)
    assert s.inspect_other_faces is False and s.no_qr_company == 'walnut'
    core, ports = make_core(tmp_path, [b], settings=s)
    run(core, ports)
    reg = core.boxes[0]
    assert reg.company == 'walnut' and not reg.data_missing, reg
    assert not any('face skipped' in line or 'going to the' in line for line in ports.logs)
    assert any('no_qr_company' in line for line in ports.logs)
    assert b.state == 'dock'
