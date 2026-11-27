"""
map_tour.py — stage 2: drive around the saved map to improve it (no ROS)
=========================================================================
After the exploration (stage 1) the robot knows the arena roughly. This tour drives
it round the map in closed LOOPS (restock/loop_tour.py) while Planisferio keeps
mapping and the landmark mapper places every ArUco marker and company logo it sees
on the map: one loop round the whole arena along the walls, then one loop round
every obstacle (or row of racks), each ending where it began, then back to the
start. Coming back to places mapped long before is what lets Planisferio close
loops and straighten the map; with passes: 2 it does everything twice.

On the outer loop the robot turns to face the wall at every stop (face_walls), so
the camera sees the markers and logos on the walls.

At each waypoint the robot stops for dwell_s: the landmark mapper only uses
sightings taken while the robot stands still (sharp images, a settled pose).

Ports (restock_ports.RestockPorts on the robot, a fake in the tests):
    now(), log(text), navigate(pose, profile) -> handle, cancel_motion(),
    clear_costmaps(), slam_pose(), landmarks()
    handle: done(), result() -> (ok, msg), cancel()
"""
# Author: Raymundo Gomez

import collections
import time
from dataclasses import dataclass

from main_ros.restock.loop_tour import LoopConfig, plan_loops


@dataclass
class TourSettings:
    """config/restock.yaml, section map_quality."""
    profile: str = 'transport'
    dwell_s: float = 1.5              # stand still at every waypoint (camera sightings)
    passes: int = 1                   # times round all the loops
    return_to_start: bool = True      # end every pass where it started
    ring_clearance: float = 0.5       # m from walls and obstacles to the loops
    spacing: float = 0.8              # m between waypoints along a loop
    face_walls: bool = True           # outer loop: face the wall at every stop (camera)
    inner_loops: bool = True          # also loop round the obstacles in the arena
    nav_timeout_s: float = 60.0

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MapTour:
    """Call start(grid), then tick() at ~10 Hz until done."""

    def __init__(self, ports, settings=None):
        self.p = ports
        self.s = settings or TourSettings()
        self.loop_cfg = LoopConfig(ring_clearance=self.s.ring_clearance,
                                   spacing=self.s.spacing, face_walls=self.s.face_walls,
                                   inner_loops=self.s.inner_loops)
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
        one, info = plan_loops(grid, start, self.loop_cfg)
        self.route_info = dict(info)
        self.loops = info.get('loops', 0)
        if not one:
            return [], info
        if self.s.return_to_start:
            one.append(start)
        return one * max(1, int(self.s.passes)), info

    def start(self, grid):
        goals, r = self.plan(grid)
        if not goals:
            self._finish('no room to drive a loop on this map')
            return
        self.goals = goals
        self.idx = 0
        show = getattr(self.p, 'show_route', None)
        if show:
            show(self.goals, 'map_tour')
        self._event(f'map tour: {r["loops"]} loops ({r["inner_loops"]} round obstacles), '
                    f'{len(self.goals)} waypoints, {r["length_m"]:.1f} m per pass x '
                    f'{max(1, int(self.s.passes))} pass(es)')
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
                'loops': self.route_info.get('loops', 0),
                'length_m': self.route_info.get('length_m'),
                'markers': [m.get('id') for m in lm.get('markers', [])],
                'logos': [lg.get('id') for lg in lm.get('logos', [])],
                'events': list(self.events)[-6:]}
