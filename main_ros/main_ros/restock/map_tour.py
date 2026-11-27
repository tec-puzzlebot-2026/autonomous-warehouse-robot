"""
map_tour.py — stage 2: drive around the saved map to improve it (no ROS)
=========================================================================
After the exploration (stage 1) the robot knows the arena roughly. This tour drives
it along every obstacle edge (the same viewpoint planner as the restocking patrol,
restock/route.py) while Planisferio keeps mapping and the landmark mapper places
every ArUco marker and company logo it sees on the map.

The tour is built to give Planisferio LOOP CLOSURES, which is what straightens the
map: coming back to a place it mapped long ago lets the pose graph correct the drift
accumulated in between.
  - The viewpoints are ordered as one closed loop round the arena (route.py,
    closed=True) that ends back at the start: the big loop.
  - Every loop_every_m of driving, the tour detours back to an EARLIER viewpoint
    that is close by (loop_radius_m of driving) but was left at least loop_min_m of
    driving ago: a small loop. Then it carries on where it was.
  - passes: 2 goes round everything twice (more closures, more sightings per marker).

At each viewpoint the robot stops for dwell_s: the landmark mapper only uses
sightings taken while the robot stands still (sharp images, a settled pose).

Ports (restock_ports.RestockPorts on the robot, a fake in the tests):
    now(), log(text), navigate(pose, profile) -> handle, cancel_motion(),
    clear_costmaps(), slam_pose(), landmarks()
    handle: done(), result() -> (ok, msg), cancel()
"""
# Author: Raymundo Gomez

import collections
import math
import time
from dataclasses import dataclass, field

from main_ros.restock.route import RouteConfig, plan_route


@dataclass
class TourSettings:
    """config/restock.yaml, section map_quality."""
    profile: str = 'transport'
    dwell_s: float = 1.5              # stand still at every viewpoint (camera sightings)
    passes: int = 1                   # times round the whole tour
    return_to_start: bool = True      # end every pass where it started: a loop closure
    loop_every_m: float = 3.0         # m driven between detours back to an old place (0 = none)
    loop_min_m: float = 3.0           # m driven since that place: shorter is not a real loop
    loop_radius_m: float = 2.0        # m (driving) to that place: farther is not worth it
    nav_timeout_s: float = 120.0
    route: dict = field(default_factory=lambda: {'view_range': 2.0})   # the camera sees far

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MapTour:
    """Call start(grid), then tick() at ~10 Hz until done."""

    def __init__(self, ports, settings=None):
        self.p = ports
        self.s = settings or TourSettings()
        rc = RouteConfig()
        for k, v in (self.s.route or {}).items():
            if hasattr(rc, k):
                setattr(rc, k, math.radians(v) if k == 'fov' and v > 3.2 else v)
        self.route_cfg = rc
        self.goals = []                   # [(x, y, theta)] in visiting order, all passes
        self.idx = 0
        self.skipped = 0
        self.wait = None
        self.dwell_until = None
        self.retried = False
        self.paused = False
        self.done = False
        self.outcome = ''
        self.detail = ''
        self.route_info = {}
        self.loops = 0
        self.events = collections.deque(maxlen=30)

    # ── control ──────────────────────────────────────────────────────────
    def plan(self, grid, start=None):
        """The tour's goals in visiting order (all passes), without driving: start()
        uses it, and the Program Manager shows it as a preview after mapping."""
        start = tuple(start or self.p.slam_pose() or (0.0, 0.0, 0.0))
        r = plan_route(grid, start, self.route_cfg, closed=self.s.return_to_start)
        self.route_info = r.to_dict()
        one = list(r.viewpoints)
        if not one:
            return [], r
        one, self.loops = self._add_loops(grid, start, one)
        self.route_info['loops'] = self.loops
        if self.s.return_to_start:
            one.append(start)
        return one * max(1, int(self.s.passes)), r

    def _add_loops(self, grid, start, goals):
        """Insert detours back to earlier viewpoints (small loop closures)."""
        if self.s.loop_every_m <= 0 or len(goals) < 3:
            return list(goals), 0
        rc = self.route_cfg
        trav = grid.traversable(rc.robot_radius, rc.clearance_margin)

        def step(a, b):
            return math.hypot(b[0] - a[0], b[1] - a[1])

        out, driven = [], []              # goals so far, and the distance driven to each
        visited = [(tuple(start), 0.0)]   # places a detour can go back to
        used = set()
        total = since = 0.0
        prev = start
        loops = 0
        for g in goals:
            d = step(prev, g)
            total += d
            since += d
            out.append(g)
            driven.append(total)
            visited.append((tuple(g), total))
            prev = g
            if since < self.s.loop_every_m:
                continue
            old = [(i, p, t) for i, (p, t) in enumerate(visited)
                   if i not in used and total - t >= self.s.loop_min_m]
            if not old:
                continue
            field_ = grid.path_lengths(g[:2], trav)
            best = None
            for i, p, t in old:              # the oldest place within reach: the biggest loop
                cell = grid.world_to_cell(p[0], p[1])
                if not grid.inside(*cell):
                    continue
                dd = float(field_[cell])
                if math.isfinite(dd) and dd <= self.s.loop_radius_m and (best is None
                                                                         or t < best[2]):
                    best = (i, p, t, dd)
            if best is None:
                continue
            i, p, _t, dd = best
            used.add(i)
            out.append(p)                    # go back there (same pose: the same view)
            total += dd
            since = 0.0
            prev = p                         # and carry on from there
            loops += 1
        return out, loops

    def start(self, grid):
        goals, r = self.plan(grid)
        if not goals:
            self._finish('no edge to drive along on this map')
            return
        self.goals = goals
        self.idx = 0
        show = getattr(self.p, 'show_route', None)
        if show:
            show(self.goals, 'map_tour')
        self._event(f'map tour: {len(r.viewpoints)} viewpoints x {max(1, int(self.s.passes))} '
                    f'pass(es), {r.length:.1f} m per pass, {100 * r.coverage:.0f} % of the edges, '
                    f'{self.loops} loop closure detour(s) per pass')
        self._go()

    def pause(self):
        if self.done or self.paused:
            return
        self.paused = True
        if self.wait is not None:
            self.wait.cancel()
            self.wait = None
        self.p.cancel_motion()
        self._event(f'paused at goal {self.idx + 1}/{len(self.goals)}')

    def resume(self):
        if self.done:
            return
        self.paused = False
        self.retried = False
        self._go()

    def finish_early(self):
        """F: stop the tour here (the map is saved by the Program Manager)."""
        if self.wait is not None:
            self.wait.cancel()
            self.wait = None
        self.p.cancel_motion()
        self._finish(f'stopped by the operator at goal {self.idx + 1}/{len(self.goals)}')

    # ── tick ─────────────────────────────────────────────────────────────
    def tick(self):
        if self.done or self.paused:
            return
        now = self.p.now()
        if self.dwell_until is not None:
            if now >= self.dwell_until:
                self.dwell_until = None
                self.idx += 1
                self._go()
            return
        w = self.wait
        if w is None:
            return
        if w.done():
            self.wait = None
            ok, msg = w.result()
            if ok:
                self.retried = False
                self.dwell_until = now + self.s.dwell_s
                self.detail = (f'Map tour: looking around at goal {self.idx + 1}/'
                               f'{len(self.goals)}{self._markers_text()}')
                return
            self._failed(msg)
        elif now > self.wait_until:
            w.cancel()
            self.wait = None
            self._failed('timed out')

    # ── helpers ──────────────────────────────────────────────────────────
    def _go(self):
        if self.idx >= len(self.goals):
            self._finish(f'tour complete ({self.skipped} goals skipped)')
            return
        pose = self.goals[self.idx]
        self.wait = self.p.navigate(pose, self.s.profile)
        self.wait_until = self.p.now() + self.s.nav_timeout_s
        self.detail = (f'Map tour: driving to goal {self.idx + 1}/{len(self.goals)}'
                       f'{self._markers_text()}')

    def _failed(self, msg):
        if not self.retried:
            self.retried = True
            self._event(f'goal {self.idx + 1} failed ({msg}); clearing the costmaps, once more')
            self.p.clear_costmaps()
            self._go()
            return
        self.retried = False
        self.skipped += 1
        self._event(f'goal {self.idx + 1} skipped ({msg})')
        self.idx += 1
        self._go()

    def _markers_text(self):
        lm = self.p.landmarks() or {}
        m, lg = len(lm.get('markers', [])), len(lm.get('logos', []))
        return f' · {m} ArUco, {lg} logos so far' if lm else ''

    def _finish(self, outcome):
        self.done = True
        self.outcome = outcome
        self.detail = 'Map tour finished: ' + outcome
        self._event(self.detail)

    def _event(self, text):
        self.events.append((time.strftime('%H:%M:%S'), text))
        self.p.log('[map tour] ' + text)

    def snapshot(self):
        lm = self.p.landmarks() or {}
        return {'goal': min(self.idx + 1, len(self.goals)), 'goals': len(self.goals),
                'skipped': self.skipped, 'done': self.done, 'paused': self.paused,
                'outcome': self.outcome, 'detail': self.detail,
                'coverage': self.route_info.get('coverage'),
                'length_m': self.route_info.get('length_m'),
                'markers': [m.get('id') for m in lm.get('markers', [])],
                'logos': [lg.get('id') for lg in lm.get('logos', [])],
                'events': list(self.events)[-6:]}
