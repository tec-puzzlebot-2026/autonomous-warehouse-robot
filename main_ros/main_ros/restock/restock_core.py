"""
restock_core.py — the restocking state machine (Phase 2, no ROS)
=================================================================
Runs inside the Program Manager's RESTOCKING state. Like mission_core.py it only
decides: it never computes a velocity or a path. It talks to the robot through a
`ports` object (restock_ports.py on the robot, a fake in test/test_restock_core.py)
and uses the ROS-free planners of this package (route, docks, geometry).

Mission level

  LOAD ─> LOCALIZE ─> [SURVEY] ─> PLAN_ROUTE ─> PATROL ⇄ BOX_TASK ─> RESUME ─> ... ─> COMPLETE

  LOAD        the mission folder: map, docks, markers (full mode: the folder just saved)
  LOCALIZE    restocking mode: the EKF searches for an ArUco (turning in place; moving to
              a new spot when that fails), its pose seeds Planisferio, Planisferio confirms
  SURVEY      only with survey: true and a company without a dock: look at the walls
              with the logo detector (survey: false: see OPENING below)
  PLAN_ROUTE  the inspection route along every obstacle edge (route.py)
  PATROL      drive to the next viewpoint, stop, collect the FAR box detector for ~2 s
              (with boxes_always_on it runs the whole time); the nearest new box
              starts a BOX_TASK
  RESUME      back to the viewpoint where the box was seen (more boxes may be there)

Box task (one box at a time)

  APPROACH ─> ACQUIRE ─> INSPECT (faces, QR) ─> ALIGN ─> CLOSE_IN ─> PICK ─> DELIVER
     DELIVER ─> CONFIRM_LOGO ─> DOCK_ALIGN ─> DOCK_IN ─> SET_DOWN ─> DOCK_OUT ─> DONE
     company's dock unknown: OPENING_GO (in front of the nearest dock opening by the
                drop zone) ─> OPENING_LOOK (logo detector names it) ─> its company:
                DOCK_ALIGN ...; another company: remembered, next opening
     no QR:     DELIVER to the drop zone (or a free area) ─> SET_DOWN (floor) ─> INSPECT around it ─> ALIGN ...
     no dock:   DELIVER to the drop zone (or a free area) ─> SET_DOWN (floor) ─> DONE (left there, marked)

Operator: pause() cancels motion and any forklift action and keeps the place in the
task; resume() re-enters the interrupted step (a forklift step restarts from ALIGN).
A failure the robot cannot handle alone (no ArUco found, a set-down that fails while
holding the box) sets `error`; the Program Manager pauses and S calls resume().

Distances in front of a box are GAPS: from the robot's front (robot_front_x ahead of
base_link) to the box face.
"""
# Author: Raymundo Gomez

import collections
import math
import os
import time

import numpy as np
from dataclasses import asdict, dataclass, field

from main_ros.restock import geometry as G
from main_ros.restock import mission_store as ms
from main_ros.restock.docks import Dock, find_docks, find_openings, free_area_near
from main_ros.restock.zones import (DROP_ZONE, Zone, ZoneConfig, drop_zone,
                                    zones_from_markers)
from main_ros.restock.geometry import wrap
from main_ros.restock.grid import OccupancyMap
from main_ros.restock.route import RouteConfig, plan_route

# Mission-level states (shown by the monitor); box-task phases are the sub-state.
LOAD, LOCALIZE, SURVEY, PLAN_ROUTE = 'LOAD', 'LOCALIZE', 'SURVEY', 'PLAN_ROUTE'
PATROL, BOX_TASK, RESUME, COMPLETE = 'PATROL', 'BOX_TASK', 'RESUME', 'COMPLETE'

PHASE_STATE = {
    'LOAD': LOAD, 'LOCALIZE': LOCALIZE, 'SURVEY_GO': SURVEY, 'SURVEY_LOOK': SURVEY,
    'SURVEY_PLAN': SURVEY, 'PLAN_ROUTE': PLAN_ROUTE, 'PATROL_GO': PATROL, 'PATROL_LOOK': PATROL,
    'RESUME_GO': RESUME, 'COMPLETE': COMPLETE,
}

# Where a paused / failed step starts again on resume() (default: the step itself).
RESUME_AT = {
    'CLOSE_IN': 'ALIGN', 'PICK': 'ALIGN', 'BACK_OFF': 'ALIGN', 'ALIGN_CHECK': 'ALIGN',
    'DOCK_ALIGN': 'DELIVER_GO', 'DOCK_IN': 'DELIVER_GO', 'DOCK_OUT': 'DELIVER_GO',
    'POST_PICK': 'DELIVER_GO', 'SURVEY_LOOK': 'SURVEY_GO', 'PATROL_LOOK': 'PATROL_LOOK',
    'OPENING_LOOK': 'OPENING_GO',
}
# SET_DOWN restarts itself (the box is still held), see _enter_SET_DOWN.

# Phases in which Planisferio and the EKF are compared (the robot drives on the map).
CROSSCHECK = ('PATROL_GO', 'RESUME_GO', 'APPROACH', 'INSPECT_GO', 'DELIVER_GO', 'SURVEY_GO',
              'OPENING_GO')

VISION = ('box_far', 'box_target', 'qr', 'logos')


@dataclass
class RestockSettings:
    """All limits and distances (config/restock.yaml, section restock)."""
    companies: tuple = ('soriene', 'emezon', 'walnut')
    # Robot geometry and the distances in front of a box (gaps from the robot's front).
    robot_front_x: float = 0.12        # m, base_link -> front of the robot (MEASURE)
    inspect_gap: float = 0.25          # m, gap while looking at a face (whole face in view)
    align_gap: float = 0.20            # m, gap of the aligned pose (camera + LiDAR see the face)
    ready_gap: float = 0.07            # m, gap handed to the forklift (PickBox)
    carry_gap: float = 0.02            # m, carried box: robot front -> box face
    back_off: float = 0.20             # m, reverse after a pick / a set-down on the floor
    align_yaw_tol_deg: float = 5.0
    align_tries: int = 2
    pick_tries: int = 2
    set_down_tries: int = 2
    max_off_axis_deg: float = 50.0     # faces seen this far off-axis count
    # Box and rack.
    box_width: float = 0.12
    box_depth: float = 0.10
    box_height: float = 0.12
    level_heights: tuple = (0.0, 0.19, 0.38)
    # Vision timing.
    look_settle_s: float = 0.6         # after stopping / enabling, before frames count
    look_s: float = 2.0                # far box detector at a viewpoint
    read_s: float = 2.0                # QR reading at a face
    acquire_s: float = 1.5             # near detector after the approach
    logo_look_s: float = 1.5
    # Docks without a logo position from stage 2: the openings near the drop zone are
    # found on the map; the logo detector reads each one's company from here.
    opening_search: float = 2.5        # m around the drop zone
    opening_look_standoff: float = 0.75  # m in front of the opening's mouth
    opening_look_turn_deg: float = 15.0  # turned toward the logo (left of the opening)
    # The far box detector runs the whole restocking (not only at the viewpoints).
    boxes_always_on: bool = True
    qr_pair_dt: float = 0.3            # s, QR and box outline must come from ~ the same frame
    logo_min_conf: float = 0.6
    require_logo: bool = False         # True: enter a dock only after its logo was seen
    # Where boxes go: 'docks' = the company's dock (the drop zone only when there is
    # none, or the QR cannot be read); 'drop_zone' = every box to the drop zone.
    deliver_to: str = 'docks'
    # Box selection.
    max_far_range: float = 3.0         # m, farther sightings are ignored (seen again later)
    min_far_conf: float = 0.90         # far detector confidence that starts a box task
    # With boxes_always_on: a box seen while driving to a viewpoint stops the patrol
    # there (the route resumes at that viewpoint after the box).
    watch_while_driving: bool = True
    dup_radius: float = 0.20           # m, a sighting this close to a known box is that box
    refine_jump: float = 0.25          # m, near estimate this far from the far one: re-approach
    max_boxes_per_view: int = 6
    # Navigation.
    profile_patrol: str = 'transport'     # driving to boxes, faces and docks (precise)
    profile_carrying: str = 'carrying'
    profile_viewpoint: str = ''           # route / survey viewpoints ('' = profile_patrol)
    nav_timeout_s: float = 120.0
    align_timeout_s: float = 45.0
    drive_timeout_s: float = 20.0
    pick_timeout_s: float = 120.0
    set_down_timeout_s: float = 90.0
    # Localization.
    localize_timeout_s: float = 300.0  # no ArUco after this: ERROR (move the robot)
    search_step_deg: float = 30.0      # turn this much, then stand still and look
    search_pause_s: float = 1.5        # still time per step (the EKF collects detections)
    collect_wait_s: float = 12.0       # EKF COLLECT (marker seen) may take this long
    ekf_stable_s: float = 1.0
    confirm_timeout_s: float = 10.0
    confirm_tol_m: float = 0.30
    search_move_max: float = 1.0       # m per move of the start-up search
    max_disagreement_m: float = 0.30   # Planisferio vs EKF ...
    disagreement_s: float = 3.0        # ... for this long: stop and re-localize
    # Survey.
    survey: bool = True
    survey_look_s: float = 1.5
    route: dict = field(default_factory=dict)      # RouteConfig overrides

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ('companies', 'level_heights'):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)

    @property
    def spec(self):
        return G.BoxSpec(self.box_width, self.box_depth, self.box_height,
                         tuple(self.level_heights))


@dataclass
class Box:
    """One box of the run (the registry)."""
    id: str
    x: float
    y: float
    level: int
    yaw: float = None              # outward normal of the pick face (map frame)
    company: str = None
    status: str = 'new'            # new, carried, delivered, failed, lost, unknown, no_dock
    data_missing: bool = False
    viewpoint: int = -1            # route index it was spotted from
    dock: str = None
    picks: int = 0
    notes: list = field(default_factory=list)

    def record(self):
        d = asdict(self)
        for k in ('x', 'y', 'yaw'):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        return d


class World:
    """Everything Phase 2 knows about the arena (from the mission folder)."""

    def __init__(self, folder, grid, docks, logos, markers, route_cfg, zones=()):
        self.folder = folder
        self.zones = list(zones)               # [Zone], one per ArUco (zones.py)
        self.drop_zone = drop_zone(self.zones)
        self.grid = grid
        self.docks = dict(docks)               # company -> Dock
        self.logos = dict(logos)               # company -> (x, y, view_yaw)
        self.markers = list(markers)
        self.route_cfg = route_cfg
        self.trav = grid.traversable(route_cfg.robot_radius, route_cfg.clearance_margin)
        self.dropped = []                      # (x, y) of boxes left on the floor
        self.openings = []                     # [Dock] whose company is not known yet

    def find_openings(self, search=2.5):
        """The dock openings near the drop zone (or anywhere) that no known dock is."""
        near = self.drop_zone.center if self.drop_zone is not None else None
        known = [d.entrance for d in self.docks.values()]
        self.openings = [o for o in find_openings(self.grid, near, search)
                         if all(math.hypot(o.entrance[0] - k[0], o.entrance[1] - k[1]) > 0.25
                                for k in known)]
        return self.openings

    @classmethod
    def from_folder(cls, folder, route_cfg, zone_cfg=None):
        grid = OccupancyMap.from_files(os.path.join(folder, ms.MAP_YAML))
        saved = ms.read_yaml(os.path.join(folder, ms.DOCKS), {}) or {}
        docks = {d['company']: Dock.from_dict(d) for d in saved.get('docks', [])}
        logos = {}
        for lg in (ms.read_yaml(os.path.join(folder, ms.LOGOS), {}) or {}).get('logos', []):
            logos[lg['id']] = (lg['x'], lg['y'], lg.get('yaw'))
        markers = (ms.read_yaml(os.path.join(folder, ms.LANDMARKS), {}) or {}).get('markers', [])
        saved_zones = (ms.read_yaml(os.path.join(folder, ms.ZONES), {}) or {}).get('zones')
        if saved_zones is not None:
            zones = [Zone.from_dict(z) for z in saved_zones]
        else:                                   # an older mission folder: work them out now
            zones = zones_from_markers(grid, markers, zone_cfg)
        return cls(folder, grid, docks, logos, markers, route_cfg, zones)

    def free_spot(self, spec):
        """Where to set a box down near the docks, avoiding boxes already left there."""
        trav = self.trav.copy()
        clear_r = 0.30
        for x, y in self.dropped:                       # keep away from earlier drops
            r0, c0 = self.grid.world_to_cell(x, y)
            n = int(math.ceil(clear_r / self.grid.res)) + 1
            for r in range(r0 - n, r0 + n + 1):
                for c in range(c0 - n, c0 + n + 1):
                    if self.grid.inside(r, c):
                        wx, wy = self.grid.cell_to_world(r, c)
                        if math.hypot(wx - x, wy - y) <= clear_r:
                            trav[r, c] = False
        z = self.drop_zone
        if z is not None:                               # inside the drop zone
            inner = trav.copy()
            rr, cc = np.nonzero(inner)
            for r, c in zip(rr, cc):
                if not z.contains(*self.grid.cell_to_world(r, c), margin=0.15):
                    inner[r, c] = False
            spot = free_area_near(self.grid, inner, [z.center], radius_needed=0.0,
                                  search=z.size)
            if spot is not None:
                return spot
        pts = [d.entrance[:2] for d in self.docks.values()] or [self._center()]
        return free_area_near(self.grid, trav, pts)

    def _center(self):
        x0, x1, y0, y1 = self.grid.bounds()
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def in_dock(self, x, y):
        """Inside a dock or the drop zone: a box there is delivered, not to be picked."""
        if self.drop_zone is not None and self.drop_zone.contains(x, y):
            return True
        return any(d.contains(x, y) for d in self.docks.values())

    def reachable(self, pose):
        r, c = self.grid.world_to_cell(pose[0], pose[1])
        return self.grid.inside(r, c) and bool(self.trav[r, c])


class RestockCore:
    """Phase-2 state machine. Call start(), then tick() at ~10 Hz."""

    def __init__(self, ports, settings=None, mission='restocking', folder=None, zone_cfg=None):
        self.p = ports
        self.zone_cfg = zone_cfg or ZoneConfig()      # older mission folders without zones.yaml
        self.s = settings or RestockSettings()
        self.mission = mission                  # 'restocking' (map loaded) or 'full'
        self.folder = folder
        self.spec = self.s.spec
        rc = RouteConfig()
        for k, v in (self.s.route or {}).items():
            if hasattr(rc, k):
                setattr(rc, k, math.radians(v) if k == 'fov' and v > 3.2 else v)
        self.route_cfg = rc
        self.world = None
        self.route = []
        self.route_info = {}
        self.idx = 0                            # next viewpoint
        self.uncovered = 0
        self.boxes = []
        self.box = None                         # the box of the current task
        self.phase = 'IDLE'
        self.ph = {}                            # data of the current phase
        self.wait = None                        # pending motion / action
        self.paused = False
        self.error = ''
        self.done = False
        self.outcome = ''
        self.detail = ''
        self.events = collections.deque(maxlen=60)
        self.log = None
        self.started = None
        self._disagree_since = None
        self._resume_after_localize = None
        self._localize_start = None
        self._nav_retry = {}
        self.boxes_here = 0
        self.inspect = None
        self.survey_views, self.survey_idx = [], 0

    # =====================================================================
    # operator interface (called by the Program Manager)
    # =====================================================================
    def start(self):
        self.started = self.p.now()
        self._goto('LOAD')

    def pause(self):
        """Hold the robot; resume() continues the interrupted step."""
        if self.paused or self.done:
            return
        self.paused = True
        self._stop_everything(cancel_forklift=True)
        self._event(f'paused in {self.phase}')

    def resume(self):
        """Continue after pause() or after an error the operator fixed."""
        if self.done:
            return
        self.paused = False
        if self.error:
            self._event(f'retrying after: {self.error}')
            self.error = ''
        if self.ph.get('_resume'):
            self._goto(self.ph['_resume'], **self.ph.get('_args', {}))
            return
        again = RESUME_AT.get(self.phase, self.phase)
        # Arguments belong to the interrupted step; a different restart step starts fresh.
        self._goto(again, **(self.ph.get('_args', {}) if again == self.phase else {}))

    def abort(self, why='stopped by the operator'):
        """X: stop for good (the Program Manager closes everything afterwards)."""
        self._stop_everything(cancel_forklift=True)
        if not self.done:
            self._finish(f'ABORTED: {why}')

    # =====================================================================
    # tick
    # =====================================================================
    def tick(self):
        if self.paused or self.error or self.done:
            return
        now = self.p.now()
        if self.wait is not None:
            self._poll_wait(now)
            if self.phase == 'PATROL_GO' and self.wait is not None:
                self._watch_far_box(now)
        else:
            handler = getattr(self, '_tick_' + self.phase, None)
            if handler is not None:
                handler(now)
        if not (self.paused or self.error or self.done):
            self._crosscheck(now)
            self._watch_drop_zone(now)

    def _watch_drop_zone(self, now):
        """No drop zone in the mission folder (its ArUco was not seen while mapping): take
        it from the landmark mapper as soon as the camera sees that marker now."""
        w = self.world
        if w is None or w.drop_zone is not None or now < getattr(self, '_dz_next', 0.0):
            return
        self._dz_next = now + 2.0
        lm = self.p.landmarks() or {}
        ids = [i for i, label in self.zone_cfg.labels.items() if label == DROP_ZONE]
        seen = [m for m in lm.get('markers', []) if int(m.get('id', -1)) in ids]
        if not seen:
            return
        z = zones_from_markers(w.grid, seen[:1], self.zone_cfg)[0]
        w.zones = [q for q in w.zones if q.marker_id != z.marker_id] + [z]
        w.drop_zone = z
        self._event(f'drop zone found: ArUco {z.marker_id} seen now, zone at '
                    f'({z.center[0]:.2f}, {z.center[1]:.2f})')
        show = getattr(self.p, 'show_zones', None)
        if show:
            show(w.zones)
        if any(c not in w.docks for c in self.s.companies):
            w.find_openings(self.s.opening_search)    # now near the drop zone
        try:
            ms.write_yaml(os.path.join(self.folder, ms.ZONES),
                          {'zones': [q.to_dict() for q in w.zones]})
        except OSError as exc:
            self._event(f'cannot write zones.yaml: {exc}')

    # =====================================================================
    # phase plumbing
    # =====================================================================
    def _goto(self, phase, **args):
        if phase != self.phase:
            self._event(f'{self.phase} -> {phase}')
        self.phase = phase
        self.ph = {'t0': self.p.now(), '_args': args}
        self.wait = None
        enter = getattr(self, '_enter_' + phase, None)
        if enter is not None:
            enter(**args)

    def _await(self, handle, timeout, ok, fail, what):
        """Wait for a motion / action handle; then ok() or fail(reason)."""
        if handle is None:
            fail(f'{what}: not available')
            return
        self.wait = {'h': handle, 'until': self.p.now() + timeout, 'ok': ok, 'fail': fail,
                     'what': what}

    def _poll_wait(self, now):
        w = self.wait
        h = w['h']
        if h.done():
            self.wait = None
            ok, msg = h.result()
            if ok:
                w['ok']()
            else:
                w['fail'](f'{w["what"]}: {msg}')
            return
        if now > w['until']:
            self.wait = None
            try:
                h.cancel()
            except Exception:                  # noqa: BLE001
                pass
            w['fail'](f'{w["what"]}: timed out')

    def _stop_everything(self, cancel_forklift):
        if self.wait is not None:
            try:
                self.wait['h'].cancel()
            except Exception:                  # noqa: BLE001
                pass
            self.wait = None
        self.p.cancel_motion()
        if cancel_forklift:
            self.p.cancel_forklift()
        for v in VISION:
            self.p.enable(v, False)

    def _vision(self, *on):
        if self.s.boxes_always_on and self.world is not None:
            on = tuple(on) + ('box_far',)
        for v in VISION:
            self.p.enable(v, v in on)

    def _hold_error(self, reason, retry_phase=None, **args):
        """Stop and wait for the operator (S retries `retry_phase`)."""
        self._stop_everything(cancel_forklift=False)
        self.error = reason
        self.ph['_resume'] = retry_phase or self.phase
        self.ph['_args'] = args
        self._event('ERROR: ' + reason)

    # ── navigation with one retry ────────────────────────────────────────
    def _nav(self, pose, profile, ok, fail, what):
        key = (round(pose[0], 2), round(pose[1], 2))

        def failed(reason):
            if self._nav_retry.get(key, 0) < 1:
                self._nav_retry[key] = 1
                self._event(f'{reason}; clearing the costmaps and trying once more')
                self.p.clear_costmaps()
                self._await(self.p.navigate(pose, profile), self.s.nav_timeout_s,
                            done, fail, what)
            else:
                fail(reason)

        def done():
            self._nav_retry.pop(key, None)
            ok()

        self._nav_retry.pop(key, None)
        self._await(self.p.navigate(pose, profile), self.s.nav_timeout_s, done, failed, what)

    # ── vision collection ────────────────────────────────────────────────
    def _start_look(self, streams, seconds):
        self._vision(*streams)
        self.ph['look_from'] = self.p.now() + self.s.look_settle_s
        self.ph['look_until'] = self.ph['look_from'] + seconds
        self.ph['streams'] = streams

    def _look_done(self, now):
        return now >= self.ph.get('look_until', 0.0)

    def _messages(self, stream):
        return self.p.messages(stream, self.ph.get('look_from', 0.0))

    # =====================================================================
    # LOAD
    # =====================================================================
    def _enter_LOAD(self):
        self.detail = 'Loading the mission folder...'
        try:
            folder = self.folder or self.p.mission_folder()
            self.world = World.from_folder(folder, self.route_cfg, self.zone_cfg)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._hold_error(f'cannot load the mission folder: {exc}', 'LOAD')
            return
        self.folder = self.world.folder
        self.log = ms.MissionLog(self.folder)
        self._vision()                          # every detector idle
        self.p.set_carrying(False)
        w = self.world
        missing = [c for c in self.s.companies if c not in w.docks]
        self._event(f'mission folder {self.folder}: {len(w.markers)} markers, '
                    f'docks {sorted(w.docks)}' + (f', NO dock for {missing}' if missing else '')
                    + (f', drop zone at ({w.drop_zone.center[0]:.2f}, '
                       f'{w.drop_zone.center[1]:.2f})' if w.drop_zone else ', NO drop zone'))
        show = getattr(self.p, 'show_zones', None)
        if show:
            show(w.zones)
        if missing:
            ops = w.find_openings(self.s.opening_search)
            self._event(f'{len(ops)} dock opening(s) without a known company'
                        + (' near the drop zone' if w.drop_zone else '')
                        + ': the logo detector reads them when a box goes there')
        if self.mission == 'full' or self.p.slam_localized():
            self._goto(self._after_localize())
        else:
            self._goto('LOCALIZE')

    def _viewpoint_profile(self):
        """Sendero profile for the route and survey viewpoints: quick stops (patrol in
        sendero_puzzlebot.yaml); boxes, faces and docks keep profile_patrol."""
        return self.s.profile_viewpoint or self.s.profile_patrol

    def _after_localize(self):
        missing = [c for c in self.s.companies if c not in self.world.docks]
        return 'SURVEY_PLAN' if (missing and self.s.survey) else 'PLAN_ROUTE'

    # =====================================================================
    # LOCALIZE: EKF finds an ArUco, seeds Planisferio, Planisferio confirms
    # =====================================================================
    def _enter_LOCALIZE(self, relocalize=False):
        self.ph.update(step='search', t_step=self.p.now(), turned=0.0, stable_since=None,
                       moves=0, seeds=0, collect_since=None)
        self._localize_start = self._localize_start or self.p.now()
        if relocalize:
            self.p.ekf_relocalize()
        if not self.world.markers:
            self.detail = ('No ArUco marker in the mission folder: give Planisferio the pose '
                           '(Foxglove 2D Pose Estimate on /initialpose).')
        else:
            self.detail = 'Localizing: looking for an ArUco marker...'

    def _tick_LOCALIZE(self, now):
        ph = self.ph
        if now - self._localize_start > self.s.localize_timeout_s:
            self._localize_start = None
            self._hold_error('no ArUco marker found during the start-up search '
                             f'({self.s.localize_timeout_s:.0f} s): move the robot where it '
                             'can see one, then press S', 'LOCALIZE')
            return
        if not self.world.markers:                      # only the operator can help
            if self.p.slam_localized():
                self._localized()
            return
        status = self.p.ekf_status()
        ekf = self.p.ekf_pose()
        if ph['step'] == 'search':
            if status == 'TRACK' and ekf is not None:
                ph['stable_since'] = ph['stable_since'] or now
                if now - ph['stable_since'] >= self.s.ekf_stable_s:
                    self.p.seed_slam(ekf)
                    ph['seeds'] += 1
                    ph['step'], ph['t_seed'] = 'confirm', now
                    self.detail = 'Localizing: EKF tracks; Planisferio confirms the pose...'
                return
            ph['stable_since'] = None
            if status == 'COLLECT':                 # a marker is in view: stand still
                ph['collect_since'] = ph['collect_since'] or now
                if now - ph['collect_since'] < self.s.collect_wait_s:
                    self.detail = 'Localizing: marker in view, the EKF is collecting...'
                    return
            ph['collect_since'] = None
            if status == 'FAILED':
                self.p.ekf_relocalize()
            if now - ph['t_step'] < self.s.search_pause_s:
                return
            step = math.radians(self.s.search_step_deg)
            if ph['turned'] + step > 2 * math.pi + 1e-6:
                self._search_move()                 # a full turn here without a marker
                return
            self.detail = (f'Localizing: turning to find a marker '
                           f'({math.degrees(ph["turned"]):.0f} deg so far)')

            def turned():
                ph['turned'] += step
                ph['t_step'] = self.p.now()

            def fail(reason):
                self._event(reason)
                turned()
            self._await(self.p.rotate(step), self.s.drive_timeout_s, turned, fail,
                        'search turn')
            return
        if ph['step'] == 'confirm':
            slam = self.p.slam_pose()
            if (self.p.slam_localized() and slam is not None and ekf is not None
                    and math.hypot(slam[0] - ekf[0], slam[1] - ekf[1]) <= self.s.confirm_tol_m):
                self._localized()
                return
            if now - ph['t_seed'] > self.s.confirm_timeout_s:
                if ph['seeds'] < 2 and ekf is not None:
                    self._event('Planisferio did not confirm the EKF pose: seeding again')
                    self.p.seed_slam(ekf)
                    ph['seeds'] += 1
                    ph['t_seed'] = now
                else:
                    self._event('Planisferio rejects the EKF pose: localizing again')
                    self.p.ekf_relocalize()
                    ph.update(step='search', t_step=now, turned=0.0, stable_since=None, seeds=0)

    def _search_move(self):
        """Drive to an open direction (LiDAR, no map pose needed), then search again."""
        free = self.p.scan_free_direction()
        ph = self.ph
        ph['moves'] += 1
        if free is None:
            self._event('start-up search: no open direction in the scan')
            self.p.ekf_relocalize()
            ph.update(t_step=self.p.now(), turned=0.0)
            return
        bearing, room = free
        dist = max(0.0, min(self.s.search_move_max, room - 0.40))
        self._event(f'start-up search: no marker here, moving {dist:.2f} m '
                    f'toward {math.degrees(bearing):.0f} deg')
        self.detail = 'Localizing: no marker in view, moving to a new spot...'

        def moved():
            self.p.ekf_relocalize()
            ph.update(t_step=self.p.now(), turned=0.0, step='search')

        def fail(reason):
            self._event(reason)
            moved()

        def turned():
            self._await(self.p.drive_straight(dist), self.s.drive_timeout_s, moved, fail,
                        'search move')
        self._await(self.p.rotate(bearing), self.s.drive_timeout_s, turned, fail, 'search turn')

    def _localized(self):
        self._localize_start = None
        pose = self.p.slam_pose()
        self._event('localized at ' + (f'({pose[0]:.2f}, {pose[1]:.2f}, '
                                       f'{math.degrees(pose[2]):.0f} deg)' if pose else '?'))
        back = self._resume_after_localize
        self._resume_after_localize = None
        if back:
            self._goto(back[0], **back[1])
        else:
            self._goto(self._after_localize())

    def _crosscheck(self, now):
        """Planisferio and the EKF disagree for too long: stop and re-localize."""
        if self.phase not in CROSSCHECK or self.p.ekf_status() != 'TRACK':
            self._disagree_since = None
            return
        a, b = self.p.slam_pose(), self.p.ekf_pose()
        if a is None or b is None:
            return
        if math.hypot(a[0] - b[0], a[1] - b[1]) <= self.s.max_disagreement_m:
            self._disagree_since = None
            return
        self._disagree_since = self._disagree_since or now
        if now - self._disagree_since >= self.s.disagreement_s:
            self._disagree_since = None
            self._event(f'Planisferio and the EKF disagree by '
                        f'{math.hypot(a[0] - b[0], a[1] - b[1]):.2f} m: re-localizing')
            back = (RESUME_AT.get(self.phase, self.phase), self.ph.get('_args', {}))
            self._stop_everything(cancel_forklift=False)
            self._resume_after_localize = back
            self._localize_start = None
            self._goto('LOCALIZE', relocalize=True)

    # =====================================================================
    # SURVEY: find the docks that are missing (logo detector at viewpoints)
    # =====================================================================
    def _enter_SURVEY_PLAN(self):
        start = self.p.slam_pose() or (0.0, 0.0, 0.0)
        r = plan_route(self.world.grid, start, self.route_cfg)
        self.ph['views'] = list(r.viewpoints)
        self.survey_views = list(r.viewpoints)
        self.survey_idx = 0
        missing = [c for c in self.s.companies if c not in self.world.docks]
        self._event(f'survey for the docks of {missing}: {len(r.viewpoints)} viewpoints')
        self._goto('SURVEY_GO')

    def _enter_SURVEY_GO(self):
        if self.survey_idx >= len(self.survey_views):
            self._event('survey finished; still no dock for '
                        f'{[c for c in self.s.companies if c not in self.world.docks]}')
            self._goto('PLAN_ROUTE')
            return
        pose = self.survey_views[self.survey_idx]
        self.detail = f'Survey: viewpoint {self.survey_idx + 1}/{len(self.survey_views)}'

        def skip(reason):
            self._event(reason)
            self.survey_idx += 1
            self._goto('SURVEY_GO')
        self._nav(pose, self._viewpoint_profile(), lambda: self._goto('SURVEY_LOOK'), skip,
                  'survey viewpoint')

    def _enter_SURVEY_LOOK(self):
        self._start_look(('logos',), self.s.survey_look_s)

    def _tick_SURVEY_LOOK(self, now):
        if not self._look_done(now):
            return
        self._vision()
        lm = self.p.landmarks() or {}
        logos = {lg['id']: (lg['x'], lg['y'], lg.get('yaw')) for lg in lm.get('logos', [])}
        new = {c: v for c, v in logos.items() if c not in self.world.docks}
        if new:
            found = find_docks(self.world.grid, new)
            for c, d in found.items():
                self.world.docks[c] = d
                self.world.logos[c] = new[c]
                self._event(f'survey: dock of {c} found at ({d.entrance[0]:.2f}, '
                            f'{d.entrance[1]:.2f})')
            if found:
                self._save_docks()
        self.survey_idx += 1
        if all(c in self.world.docks for c in self.s.companies):
            self._event('survey: every company has a dock')
            self._goto('PLAN_ROUTE')
        else:
            self._goto('SURVEY_GO')

    def _save_docks(self):
        try:
            ms.write_yaml(os.path.join(self.folder, ms.DOCKS),
                          {'docks': [d.to_dict() for d in self.world.docks.values()]})
            ms.write_yaml(os.path.join(self.folder, ms.LOGOS),
                          {'logos': [{'id': c, 'x': float(v[0]), 'y': float(v[1]),
                                      'yaw': None if v[2] is None else float(v[2])}
                                     for c, v in self.world.logos.items()]})
        except OSError as exc:
            self._event(f'cannot write docks.yaml: {exc}')

    # =====================================================================
    # PLAN_ROUTE / PATROL / RESUME
    # =====================================================================
    def _enter_PLAN_ROUTE(self):
        saved = ms.read_yaml(os.path.join(self.folder, ms.ROUTE), None) or {}
        if saved.get('viewpoints'):                 # planned in stage 3: use it as it is
            self.route = [tuple(float(v) for v in p) for p in saved['viewpoints']]
            self.route_info = saved
            self.idx = 0
            self._event(f'route from stage 3: {len(self.route)} viewpoints, '
                        f'{saved.get("length_m", 0):.1f} m')
            self._goto('PATROL_GO')
            return
        self.detail = 'Planning the inspection route...'
        start = self.p.slam_pose() or (0.0, 0.0, 0.0)
        r = plan_route(self.world.grid, start, self.route_cfg)
        self.route = list(r.viewpoints)
        self.route_info = r.to_dict()
        self.idx = 0
        try:
            ms.write_yaml(os.path.join(self.folder, ms.ROUTE), self.route_info)
        except OSError:
            pass
        self._event(f'route: {len(self.route)} viewpoints, {100 * r.coverage:.1f} % of the '
                    f'edges, {r.length:.1f} m')
        if not self.route:
            self._finish('COMPLETE: the map has no edge to inspect')
            return
        self._goto('PATROL_GO')

    def _enter_PATROL_GO(self):
        self._vision()
        self.boxes_here = 0
        self.ph['look_from'] = self.p.now() + self.s.look_settle_s   # sightings while driving
        if self.idx >= len(self.route):
            self._finish('COMPLETE: route finished')
            return
        self.detail = f'Patrol: driving to viewpoint {self.idx + 1}/{len(self.route)}'
        if self.log:
            self.log.progress(self.idx, len(self.route))

        def skip(reason):
            self.uncovered += 1
            self._event(f'viewpoint {self.idx + 1} skipped ({reason})')
            self.idx += 1
            self._goto('PATROL_GO')
        self._nav(self.route[self.idx], self._viewpoint_profile(),
                  lambda: self._goto('PATROL_LOOK'), skip, f'viewpoint {self.idx + 1}')

    def _watch_far_box(self, now):
        """Driving to a viewpoint with the far detector on: a box seen with enough
        confidence stops the patrol here and starts its box task."""
        if not (self.s.boxes_always_on and self.s.watch_while_driving):
            return
        if now < self.ph.get('watch_next', 0.0):
            return
        self.ph['watch_next'] = now + 0.5
        box = self._pick_far_box()
        if box is None:
            return
        w, self.wait = self.wait, None
        try:
            w['h'].cancel()
        except Exception:                      # noqa: BLE001
            pass
        self.p.cancel_motion()
        self.boxes.append(box)
        self.box = box
        self._event(f'{box.id} spotted while driving at ({box.x:.2f}, {box.y:.2f}), level '
                    f'{box.level}: patrol paused at viewpoint {self.idx + 1}')
        self._goto('APPROACH')

    def _enter_PATROL_LOOK(self):
        self.detail = f'Patrol: looking for boxes at viewpoint {self.idx + 1}/{len(self.route)}'
        self._start_look(('box_far',), self.s.look_s)

    def _tick_PATROL_LOOK(self, now):
        if not self._look_done(now):
            return
        self._vision()
        box = self._pick_far_box()
        if box is None:
            self.idx += 1
            self._goto('PATROL_GO')
            return
        self.boxes.append(box)
        self.box = box
        self.boxes_here = getattr(self, 'boxes_here', 0) + 1
        self._event(f'{box.id} spotted at ({box.x:.2f}, {box.y:.2f}), level {box.level}')
        self._goto('APPROACH')

    def _pick_far_box(self):
        """Nearest new box among the far detector's sightings at this viewpoint."""
        cam, pose = self.p.camera(), self.p.slam_pose()
        if cam is None or pose is None:
            return None
        best = None
        for _, d in self._messages('box_far'):
            boxes = [b[:4] for b in d.get('all') or []
                     if len(b) < 5 or float(b[4]) >= self.s.min_far_conf]
            if not boxes and d.get('found') and d.get('box') \
                    and float(d.get('conf', 1.0)) >= self.s.min_far_conf:
                boxes = [d['box']]
            for image_box in boxes:
                sg = G.locate_box(cam, pose, image_box, self.spec)
                if sg.distance > self.s.max_far_range or self.world.in_dock(sg.x, sg.y):
                    continue
                if self._known_box(sg.x, sg.y, sg.level):
                    continue
                if best is None or sg.distance < best.distance:
                    best = sg
        if best is None:
            return None
        return Box(id=f'box_{len(self.boxes) + 1:02d}', x=best.x, y=best.y, level=best.level,
                   viewpoint=self.idx)

    def _known_box(self, x, y, level):
        for b in self.boxes:
            if b.status == 'carried':
                continue
            if b.level == level and math.hypot(b.x - x, b.y - y) <= self.s.dup_radius:
                return True
        return False

    def _enter_RESUME_GO(self):
        self._vision()
        self.box = None
        if self.boxes_here >= self.s.max_boxes_per_view or self.idx >= len(self.route):
            self.idx += 1
            self._goto('PATROL_GO')
            return
        self.detail = f'Resuming the route at viewpoint {self.idx + 1}/{len(self.route)}'

        def skip(reason):
            self._event(reason)
            self.idx += 1
            self._goto('PATROL_GO')
        self._nav(self.route[self.idx], self._viewpoint_profile(),
                  lambda: self._goto('PATROL_LOOK'), skip, 'back to the route')

    # =====================================================================
    # BOX TASK: approach, acquire, inspect faces
    # =====================================================================
    def _base_dist(self, gap):
        """base_link -> box center for a given gap in front of a face."""
        return self.s.robot_front_x + gap + self.spec.depth / 2

    def _approach_pose(self, box, gap):
        """A reachable pose `gap` in front of the box, facing it, on the robot's side."""
        pose = self.p.slam_pose() or self.route[box.viewpoint]
        toward = math.atan2(pose[1] - box.y, pose[0] - box.x) if box.yaw is None else box.yaw
        d = self._base_dist(gap)
        for off in (0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9):
            a = wrap(toward + off)
            cand = (box.x + d * math.cos(a), box.y + d * math.sin(a), wrap(a + math.pi))
            if self.world.reachable(cand):
                return cand
        return None

    def _enter_APPROACH(self, again=False):
        box = self.box
        self._vision()
        pose = self._approach_pose(box, self.s.inspect_gap)
        if pose is None:
            self._box_end('failed', 'no reachable pose in front of the box')
            return
        self.ph['again'] = again
        self.detail = f'{box.id}: approaching'
        self._nav(pose, self.s.profile_patrol, lambda: self._goto('ACQUIRE', again=again),
                  lambda r: self._box_end('failed', r), f'{box.id} approach')

    def _enter_ACQUIRE(self, again=False, retried=False):
        self.ph.update(again=again, retried=retried)
        self.detail = f'{self.box.id}: near detector'
        self._start_look(('box_target',), self.s.acquire_s)

    def _tick_ACQUIRE(self, now):
        if not self._look_done(now):
            return
        box, ph = self.box, self.ph
        target = self._latest_target()
        if target is None:
            if not ph['retried']:
                self._event(f'{box.id}: not seen from the approach pose; looking again')
                self._goto('ACQUIRE', again=ph['again'], retried=True)
            else:
                self._box_end('lost', 'lost during the approach')
            return
        cam, pose = self.p.camera(), self.p.slam_pose()
        sg = G.locate_box(cam, pose, target['box'], self.spec)
        jump = math.hypot(sg.x - box.x, sg.y - box.y)
        box.x, box.y, box.level = sg.x, sg.y, sg.level
        box.yaw = wrap(pose[2] + math.pi)        # pick face looks at the robot (open floor)
        if box.level == 0:
            fit = self.p.scan_face()
            if fit is not None:
                dist, yaw_err = fit
                box.yaw = wrap(pose[2] + math.pi + yaw_err)
                c = dist + self.spec.depth / 2
                box.x = pose[0] + c * math.cos(pose[2])
                box.y = pose[1] + c * math.sin(pose[2])
        if jump > self.s.refine_jump and not ph['again']:
            self._event(f'{box.id}: {jump:.2f} m from the first estimate; approaching again')
            self._goto('APPROACH', again=True)
            return
        self._event(f'{box.id}: at ({box.x:.2f}, {box.y:.2f}), level {box.level}, pick face '
                    f'{math.degrees(box.yaw):.0f} deg')
        self._begin_inspect(on_floor=False)

    def _latest_target(self):
        msgs = [d for _, d in self._messages('box_target') if d.get('found')]
        return msgs[-1] if msgs else None

    def _begin_inspect(self, on_floor):
        box = self.box
        foot = G.BoxFootprint(box.x, box.y, box.yaw, self.spec)
        views = G.face_viewpoints(foot, self.world.grid, self.world.trav,
                                  standoff=self._base_dist(self.s.inspect_gap)
                                  - self.spec.depth / 2,
                                  max_off_axis=math.radians(self.s.max_off_axis_deg))
        here = self.p.slam_pose()
        others = [(n, p) for n, p in views if n != 'front' and p is not None]
        self.inspect = {'on_floor': on_floor, 'queue': G.order_viewpoints(here, others),
                        'tried': ['front'], 'reachable': 1 + len(others)}
        self._goto('INSPECT_READ', face='front')

    def _enter_INSPECT_GO(self):
        q = self.inspect['queue']
        if not q:
            self._inspect_finished()
            return
        name, pose = q[0]
        self.detail = f'{self.box.id}: going to the {name} face'

        def skip(reason):
            self._event(f'{self.box.id}: {name} face skipped ({reason})')
            q.pop(0)
            self._goto('INSPECT_GO')

        def arrived():
            q.pop(0)
            self.inspect['tried'].append(name)
            self._goto('INSPECT_READ', face=name)
        self._nav(pose, self.s.profile_patrol, arrived, skip, f'{name} face')

    def _enter_INSPECT_READ(self, face='front'):
        self.ph['face'] = face
        self.detail = f'{self.box.id}: reading the QR code ({face} face)'
        self._start_look(('box_target', 'qr'), self.s.read_s)

    def _tick_INSPECT_READ(self, now):
        if not self._look_done(now):
            return
        company = self._read_qr()
        if company:
            self.box.company = company
            self._event(f'{self.box.id}: QR "{company}" on the {self.ph["face"]} face')
            self._inspect_finished()
            return
        self._goto('INSPECT_GO')

    def _read_qr(self):
        """Company of the QR code that lies inside the target box's outline, or None."""
        targets = [(d.get('stamp', t), d) for t, d in self._messages('box_target')
                   if d.get('found')]
        if not targets:
            return None
        for t, d in self._messages('qr'):
            stamp = d.get('stamp', t)
            near = min(targets, key=lambda td: abs(td[0] - stamp))
            if abs(near[0] - stamp) > self.s.qr_pair_dt:
                continue
            poly = G.box_polygon(near[1])
            for code in d.get('codes', []):
                text = str(code.get('text', '')).strip().lower()
                if text in self.s.companies and G.qr_belongs_to_box(code.get('corners', []), poly):
                    return text
        return None

    def _inspect_finished(self):
        box, ins = self.box, self.inspect
        self._vision()
        if box.company is None:
            if ins['on_floor']:
                self._box_end('unknown', 'no QR code readable on the open floor: left in the '
                              'free area')
                return
            box.data_missing = True
            self._event(f'{box.id}: no QR code on {len(ins["tried"])}/{ins["reachable"]} '
                        'faces: "data missing", it is read again in the free area')
        self.ph['align_tries'] = 0
        self._goto('ALIGN', tries=0)

    # =====================================================================
    # BOX TASK: align, close in, pick
    # =====================================================================
    def _face_pose(self, box):
        """Center of the pick face, x axis pointing out of the box."""
        return (box.x + self.spec.depth / 2 * math.cos(box.yaw),
                box.y + self.spec.depth / 2 * math.sin(box.yaw), box.yaw)

    def _enter_ALIGN(self, tries=0, near=False):
        box = self.box
        self._vision()
        self.ph['tries'] = tries
        standoff = self.s.robot_front_x + self.s.align_gap
        face = self._face_pose(box)
        goal = G.pose_facing(face[:2], face[2], standoff)
        here = self.p.slam_pose()
        if not near and here is not None and (
                math.hypot(goal[0] - here[0], goal[1] - here[1]) > 0.25
                or abs(wrap(goal[2] - here[2])) > math.radians(35)):
            # Away from the pick face (e.g. after reading the back): drive there first.
            self.detail = f'{box.id}: back to the pick face'
            pose = (goal if self.world.reachable(goal)
                    else self._approach_pose(box, self.s.align_gap))
            if pose is None:
                self._box_end('failed', 'the pick face cannot be reached')
                return
            self._nav(pose, self.s.profile_patrol,
                      lambda: self._goto('ALIGN', tries=tries, near=True),
                      lambda r: self._box_end('failed', r), 'to the pick face')
            return
        self.detail = f'{box.id}: aligning with the pick face'
        h = self.p.align(face, standoff, self.s.profile_patrol)
        self._await(h, self.s.align_timeout_s, lambda: self._goto('ALIGN_CHECK', tries=tries),
                    lambda r: self._align_failed(r, tries), f'{box.id} align')

    def _align_failed(self, reason, tries):
        if tries + 1 < self.s.align_tries:
            self._event(f'{reason}; approaching again')
            pose = self._approach_pose(self.box, self.s.align_gap)
            if pose is None:
                self._box_end('failed', reason)
                return
            self._nav(pose, self.s.profile_patrol,
                      lambda: self._goto('ALIGN', tries=tries + 1, near=True),
                      lambda r: self._box_end('failed', r), 'realign')
        else:
            self._box_end('failed', reason)

    def _enter_ALIGN_CHECK(self, tries=0):
        box = self.box
        gap = self.s.align_gap
        if box.level == 0:
            fit = self.p.scan_face()
            if fit is not None:
                dist, yaw_err = fit
                gap = dist - self.s.robot_front_x
                if abs(yaw_err) > math.radians(self.s.align_yaw_tol_deg):
                    pose = self.p.slam_pose()
                    box.yaw = wrap(pose[2] + math.pi + yaw_err)
                    if tries + 1 < self.s.align_tries:
                        self._event(f'{box.id}: {math.degrees(yaw_err):.1f} deg off the face; '
                                    'aligning again')
                        self._goto('ALIGN', tries=tries + 1, near=True)
                        return
                    self._event(f'{box.id}: still {math.degrees(yaw_err):.1f} deg off; '
                                'continuing (the forklift fine-aligns)')
        self._goto('CLOSE_IN', gap=gap)

    def _enter_CLOSE_IN(self, gap=None):
        gap = self.s.align_gap if gap is None else gap
        dist = gap - self.s.ready_gap
        self.detail = f'{self.box.id}: closing in {100 * dist:.0f} cm by odometry'
        self._await(self.p.drive_straight(dist), self.s.drive_timeout_s,
                    lambda: self._goto('PICK'), lambda r: self._box_end('failed', r), 'close in')

    def _enter_PICK(self):
        box = self.box
        gap = self.s.ready_gap
        if box.level == 0:
            fit = self.p.scan_face()
            if fit is not None:
                gap = max(0.0, fit[0] - self.s.robot_front_x)
        box.picks += 1
        self.detail = f'{box.id}: PickBox (forklift), attempt {box.picks}'
        self._await(self.p.pick(box.id, box.level, self._face_pose(box), gap),
                    self.s.pick_timeout_s, self._picked, self._pick_failed, f'{box.id} PickBox')

    def _picked(self):
        box = self.box
        box.status = 'carried'
        self.p.set_carrying(True)
        self._event(f'{box.id}: picked')
        self._goto('POST_PICK')

    def _enter_POST_PICK(self):
        self._await(self.p.drive_straight(-self.s.back_off), self.s.drive_timeout_s,
                    lambda: self._goto('DELIVER_GO'),
                    lambda r: (self._event(r), self._goto('DELIVER_GO')), 'back off')

    def _pick_failed(self, reason):
        box = self.box
        self._event(f'{box.id}: {reason}')
        back = self.s.align_gap - self.s.ready_gap

        def again():
            if box.picks < self.s.pick_tries:
                self._goto('ALIGN', tries=0)
            else:
                self._box_end('failed', f'PickBox failed {box.picks} times')

        self._await(self.p.drive_straight(-back), self.s.drive_timeout_s, again,
                    lambda r: (self._event(r), again()), 'back off')

    # =====================================================================
    # DELIVERY
    # =====================================================================
    def _enter_DELIVER_GO(self, dock=None, tried=()):
        box = self.box
        self._vision()
        if self.s.deliver_to == 'drop_zone' and self.world.drop_zone is not None:
            self._goto('FLOOR_GO', why='to the drop zone')
            return
        if box.company is not None and box.company not in self.world.docks \
                and self.world.openings:
            self._goto('OPENING_GO', tried=list(tried))
            return
        if box.company is None or box.company not in self.world.docks:
            why = 'data missing' if box.company is None else f'no dock known for {box.company}'
            self._goto('FLOOR_GO', why=why)
            return
        d = self.world.docks[dock or box.company]
        tried = list(tried)
        self.detail = f'{box.id}: carrying it to the {d.company} dock'
        self._nav(d.entrance, self.s.profile_carrying,
                  lambda: self._goto('CONFIRM_LOGO', dock=d.company, tried=tried),
                  lambda r: self._hold_error(f'{box.id}: cannot reach the dock ({r})',
                                             'DELIVER_GO', dock=d.company, tried=tried),
                  'to the dock')

    # ── an opening whose company is not known: read its logo ─────────────
    def _opening_look_pose(self, o):
        th = o.entrance[2]                          # facing into the opening
        mouth = (0.5 * (o.entrance[0] + o.inside[0]), 0.5 * (o.entrance[1] + o.inside[1]))
        d = self.s.opening_look_standoff
        return (mouth[0] - d * math.cos(th), mouth[1] - d * math.sin(th),
                wrap(th + math.radians(self.s.opening_look_turn_deg)))

    def _enter_OPENING_GO(self, tried=()):
        box = self.box
        here = self.p.slam_pose() or (0.0, 0.0, 0.0)
        left = [o for o in self.world.openings if tuple(o.entrance) not in
                [tuple(t) for t in tried]]
        if not left:
            self._goto('FLOOR_GO', why=f'no opening shows the {box.company} logo')
            return
        o = min(left, key=lambda q: math.hypot(q.entrance[0] - here[0], q.entrance[1] - here[1]))
        self.ph['opening'] = o
        self.detail = f'{box.id}: to a dock opening, to read its logo'
        self._nav(self._opening_look_pose(o), self.s.profile_carrying,
                  lambda: self._goto('OPENING_LOOK', opening=o, tried=list(tried)),
                  lambda r: (self._event(f'{box.id}: opening unreachable ({r})'),
                             self._goto('OPENING_GO', tried=list(tried) + [o.entrance])),
                  'to the opening')

    def _enter_OPENING_LOOK(self, opening=None, tried=()):
        self.ph.update(opening=opening, tried=list(tried))
        self.detail = f'{self.box.id}: reading the logo of the opening'
        self._start_look(('logos',), self.s.logo_look_s)

    def _tick_OPENING_LOOK(self, now):
        if not self._look_done(now):
            return
        box, ph = self.box, self.ph
        counts = {}
        for _, d in self._messages('logos'):
            for det in d.get('detections', []):
                if float(det.get('conf', 0.0)) >= self.s.logo_min_conf:
                    c = str(det.get('company'))
                    counts[c] = counts.get(c, 0) + 1
        self._vision()
        o = ph['opening']
        tried = ph['tried'] + [o.entrance]
        if not counts:
            self._event(f'{box.id}: no logo read at the opening ({o.entrance[0]:.2f}, '
                        f'{o.entrance[1]:.2f})')
            self._goto('OPENING_GO', tried=tried)
            return
        company = max(counts, key=counts.get)
        o.company = company                           # this opening is now a known dock
        self.world.docks[company] = o
        self.world.openings = [q for q in self.world.openings if q is not o]
        self._save_docks()
        self._event(f'{box.id}: the opening at ({o.entrance[0]:.2f}, {o.entrance[1]:.2f}) is '
                    f'the {company} dock')
        if company == box.company:
            self._goto('DOCK_ALIGN', dock=company)
        else:
            self._goto('OPENING_GO', tried=tried)

    def _enter_CONFIRM_LOGO(self, dock=None, tried=()):
        self.ph.update(dock=dock or self.box.company, tried=list(tried))
        self.detail = f'{self.box.id}: confirming the {dock} logo'
        self._start_look(('logos',), self.s.logo_look_s)

    def _tick_CONFIRM_LOGO(self, now):
        if not self._look_done(now):
            return
        self._vision()
        box, ph = self.box, self.ph
        seen = set()
        for _, d in self._messages('logos'):
            for det in d.get('detections', []):
                if float(det.get('conf', 0.0)) >= self.s.logo_min_conf:
                    seen.add(str(det.get('company')))
        expected = box.company
        if expected in seen or (not seen and not self.s.require_logo):
            if not seen:
                self._event(f'{box.id}: no logo seen at the {ph["dock"]} dock; entering anyway')
            self._goto('DOCK_ALIGN', dock=ph['dock'])
            return
        tried = ph['tried'] + [ph['dock']]
        reads = sorted(seen) or 'nothing'
        self._event(f'{box.id}: the logo at the {ph["dock"]} dock reads {reads}'
                    f', expected {expected}')
        here = self.p.slam_pose()
        left = [d for c, d in self.world.docks.items() if c not in tried]
        if not left:
            self._goto('FLOOR_GO', why=f'no dock shows the {expected} logo')
            return
        nxt = min(left, key=lambda d: math.hypot(d.entrance[0] - here[0], d.entrance[1] - here[1]))
        dock = nxt
        self._nav(dock.entrance, self.s.profile_carrying,
                  lambda: self._goto('CONFIRM_LOGO', dock=dock.company, tried=tried),
                  lambda r: self._goto('FLOOR_GO', why=f'other dock unreachable ({r})'),
                  'to the next dock')

    def _enter_DOCK_ALIGN(self, dock=None):
        d = self.world.docks[dock]
        self.ph['dock'] = dock
        self.detail = f'{self.box.id}: squaring up to the {dock} trailer'
        mouth_d = 0.35
        mouth = (d.entrance[0] + mouth_d * math.cos(d.entrance[2]),
                 d.entrance[1] + mouth_d * math.sin(d.entrance[2]), wrap(d.entrance[2] + math.pi))
        self._await(self.p.align(mouth, mouth_d, self.s.profile_carrying), self.s.align_timeout_s,
                    lambda: self._goto('DOCK_IN', dock=dock),
                    lambda r: (self._event(f'{r}; entering from the entrance pose'),
                               self._goto('DOCK_IN', dock=dock)), 'dock align')

    def _dock_in_distance(self, dock):
        d = self.world.docks[dock]
        here = self.p.slam_pose() or d.entrance
        dist = math.hypot(d.inside[0] - here[0], d.inside[1] - here[1])
        return max(0.0, dist - (self.s.robot_front_x + self.s.carry_gap + self.spec.depth / 2))

    def _enter_DOCK_IN(self, dock=None, tries=0):
        dist = self._dock_in_distance(dock)
        self.ph['dock'] = dock
        self.detail = f'{self.box.id}: driving {100 * dist:.0f} cm into the {dock} trailer'
        self._await(self.p.drive_straight(dist, centered=True), self.s.drive_timeout_s,
                    lambda: self._goto('SET_DOWN', where='dock', dock=dock, dist=dist,
                                       tries=tries),
                    lambda r: self._hold_error(f'{self.box.id}: {r}', 'DELIVER_GO', dock=dock),
                    'into the trailer')

    def _enter_SET_DOWN(self, where='dock', dock=None, dist=0.0, tries=0, spot=None):
        box = self.box
        self.ph['_args'] = dict(where=where, dock=dock, dist=dist, tries=tries, spot=spot)
        self.detail = f'{box.id}: SetDownBox ({where})'
        self._await(self.p.set_down(box.id, where), self.s.set_down_timeout_s,
                    lambda: self._set_down_ok(where, dock, dist, spot),
                    lambda r: self._set_down_failed(r, where, dock, dist, tries, spot),
                    f'{box.id} SetDownBox')

    def _set_down_ok(self, where, dock, dist, spot=None):
        box = self.box
        self._event(f'{box.id}: set down ({where})')
        if where == 'dock':
            box.dock = dock
            box.status = 'in_dock'
            self._await(self.p.drive_straight(-dist), self.s.drive_timeout_s,
                        lambda: self._after_dock(), lambda r: (self._event(r), self._after_dock()),
                        'out of the trailer')
        else:
            self.p.set_carrying(False)
            box.status = 'on_floor'
            if spot is not None:
                box.x, box.y = spot
            self.world.dropped.append((box.x, box.y))
            self._goto('FLOOR_BACK')

    def _after_dock(self):
        self.p.set_carrying(False)
        self._box_end('delivered', f'delivered to the {self.box.dock} dock')

    def _set_down_failed(self, reason, where, dock, dist, tries, spot=None):
        box = self.box
        tries += 1
        self._event(f'{box.id}: {reason}')
        if tries >= self.s.set_down_tries:
            self._hold_error(f'{box.id}: SetDownBox failed {tries} times; the robot is still '
                             'holding the box', 'SET_DOWN', where=where, dock=dock, dist=dist,
                             tries=0, spot=spot)
            return
        if where == 'dock':
            self._await(self.p.drive_straight(-dist), self.s.drive_timeout_s,
                        lambda: self._goto('DOCK_IN', dock=dock, tries=tries),
                        lambda r: self._hold_error(r, 'SET_DOWN', where=where, dock=dock,
                                                   dist=dist, tries=0), 'back out')
        else:
            self._goto('SET_DOWN', where=where, dock=dock, dist=dist, tries=tries, spot=spot)

    # ── the free area by the docks (QR fallback, or no dock) ─────────────
    def _enter_FLOOR_GO(self, why=''):
        box = self.box
        spot = self.world.free_spot(self.spec)
        if spot is None:
            self._hold_error(f'{box.id}: no free area near the docks to set the box down',
                             'FLOOR_GO', why=why)
            return
        self.ph['why'] = why
        centroid = [d.entrance[:2] for d in self.world.docks.values()]
        z = self.world.drop_zone
        if z is not None and z.contains(*spot):
            heading = wrap(z.yaw + math.pi)              # from the arena toward the wall
        elif centroid:
            cx = sum(p[0] for p in centroid) / len(centroid)
            cy = sum(p[1] for p in centroid) / len(centroid)
            heading = math.atan2(spot[1] - cy, spot[0] - cx)
        else:
            heading = 0.0
        d = self.s.robot_front_x + self.s.carry_gap + self.spec.depth / 2
        pose = (spot[0] - d * math.cos(heading), spot[1] - d * math.sin(heading), heading)
        where = 'drop zone' if z is not None and z.contains(*spot) else 'free area'
        self._event(f'{box.id}: {why}: taking it to the {where} ({spot[0]:.2f}, {spot[1]:.2f})')
        self.detail = f'{box.id}: to the {where} ({why})'
        self._nav(pose, self.s.profile_carrying,
                  lambda: self._goto('SET_DOWN', where='floor', spot=spot),
                  lambda r: self._hold_error(f'{box.id}: cannot reach the free area ({r})',
                                             'FLOOR_GO', why=why), 'to the free area')

    def _enter_FLOOR_BACK(self):
        self._await(self.p.drive_straight(-self.s.back_off), self.s.drive_timeout_s,
                    self._floor_next, lambda r: (self._event(r), self._floor_next()), 'back off')

    def _floor_next(self):
        box = self.box
        if self.s.deliver_to == 'drop_zone' and self.world.drop_zone is not None:
            self._box_end('delivered', 'left in the drop zone')
            return
        if box.company is not None:                       # no dock for it: leave it here
            where = 'drop zone' if self.world.drop_zone is not None else 'free area'
            self._box_end('no_dock', f'left in the {where} (no dock for {box.company})')
            return
        pose = self.p.slam_pose()
        box.yaw = math.atan2(pose[1] - box.y, pose[0] - box.x)
        self._event(f'{box.id}: on the open floor; circling it to read the QR code')
        self._begin_inspect(on_floor=True)

    # =====================================================================
    # end of a box
    # =====================================================================
    def _box_end(self, status, reason):
        box = self.box
        if box is None:
            self._goto('RESUME_GO')
            return
        if box.status == 'carried':
            # Never forget a box that is still on the forks.
            self._hold_error(f'{box.id}: {reason} while carrying it', 'DELIVER_GO')
            return
        box.status = status
        box.notes.append(reason)
        self._vision()
        self._event(f'{box.id}: {status.upper()} - {reason}')
        if self.log:
            self.log.box(box.record())
        self._goto('RESUME_GO')

    def _finish(self, outcome):
        self._vision()
        self.outcome = outcome
        self.done = True
        self.phase = 'COMPLETE'
        self.detail = outcome
        if self.log:
            self.log.progress(min(self.idx, len(self.route)), len(self.route))
            self.log.finish(outcome)
        self._event(outcome)

    # =====================================================================
    # status
    # =====================================================================
    def _event(self, text):
        self.events.append((time.strftime('%H:%M:%S'), text))
        self.p.log('[restock] ' + text)
        if self.log and not text.startswith(('PATROL', 'IDLE')) and '->' not in text:
            self.log.event(text)

    @property
    def state(self):
        if self.done:
            return COMPLETE
        return PHASE_STATE.get(self.phase, BOX_TASK if self.box is not None else self.phase)

    def counts(self):
        out = collections.Counter(b.status for b in self.boxes)
        return dict(out)

    def per_company(self):
        out = {c: 0 for c in self.s.companies}
        for b in self.boxes:
            if b.status == 'delivered' and b.company in out:
                out[b.company] += 1
        return out

    def snapshot(self):
        box = self.box
        return {
            'state': self.state, 'phase': self.phase, 'detail': self.detail,
            'paused': self.paused, 'error': self.error, 'done': self.done,
            'outcome': self.outcome, 'folder': self.folder,
            'route': {'total': len(self.route), 'next': self.idx, 'uncovered': self.uncovered,
                      'coverage': self.route_info.get('coverage'),
                      'length_m': self.route_info.get('length_m')},
            'box': box.record() if box is not None else None,
            'boxes': self.counts(), 'delivered': self.per_company(),
            'docks': sorted(self.world.docks) if self.world else [],
            'events': list(self.events)[-8:],
        }
