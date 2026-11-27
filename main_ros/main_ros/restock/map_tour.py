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
import math
import time
from dataclasses import dataclass

from main_ros.restock.loop_tour import LoopConfig, plan_loops


@dataclass
class TourSettings:
    """config/restock.yaml, section map_quality."""
    profile: str = 'transport'
    dwell_s: float = 0.0              # stand still at a waypoint it stops at (0 = drive on)
    passes: int = 1                   # times round all the loops
    return_to_start: bool = True      # end every pass where it started
    ring_clearance: float = 0.5       # m from walls and obstacles to the loops
    spacing: float = 0.8              # m between waypoints along a loop
    loop_size: float = 1.6            # m, longest sector loop (0 = one loop round all)
    outer_loop: bool = True           # after the sectors, one loop round the whole arena
    face_walls: bool = False          # outer loop: face the nearest wall at every stop
    inner_loops: bool = True          # also loop round the obstacles in the arena
    robot_radius: float = 0.15        # m, the loops only run where the robot fits
    pass_radius: float = 0.35         # m, this close to a waypoint: on to the next (0 = stop)
    stall_s: float = 6.0              # s at a goal Sendero does not finish: count it reached
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
                                   spacing=self.s.spacing, loop_size=self.s.loop_size,
                                   outer_loop=self.s.outer_loop, face_walls=self.s.face_walls,
                                   inner_loops=self.s.inner_loops,
                                   robot_radius=self.s.robot_radius)
        self.near_since = None            # first time near the current goal
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
                self._reached(now)
                return
            self._failed(msg)
        elif self._passed(now):
            self.wait = None                  # the next goal replaces it in Sendero
            self.retried = False
            self.idx += 1
            self._go()
        elif now > self.wait_until:
            w.cancel()
            self.wait = None
            self._failed('timed out')

    # ── helpers ──────────────────────────────────────────────────────────
    def _reached(self, now):
        self.retried = False
        if self.s.dwell_s > 0:
            self.dwell_until = now + self.s.dwell_s
            self.detail = (f'Map tour: looking around at goal {self.idx + 1}/'
                           f'{len(self.goals)}{self._markers_text()}')
        else:
            self.idx += 1
            self._go()

    def _passed(self, now):
        """Close enough to the current goal to go on: drive through the waypoints of a
        loop without stopping (pass_radius), and never wait for ever at one Sendero
        does not finish (e.g. it cannot turn there): stall_s near it counts as reached."""
        pose = self.p.slam_pose()
        if pose is None or self.idx >= len(self.goals):
            return False
        g = self.goals[self.idx]
        d = math.hypot(pose[0] - g[0], pose[1] - g[1])
        last = self.idx == len(self.goals) - 1
        if self.s.pass_radius > 0 and not last and d <= self.s.pass_radius:
            return True
        near = d <= max(self.s.pass_radius, 0.25)
        if not near:
            self.near_since = None
            return False
        if self.near_since is None:
            self.near_since = now
            return False
        if now - self.near_since >= self.s.stall_s:
            self._event(f'goal {self.idx + 1}: at it for {self.s.stall_s:.0f} s without '
                        'Sendero finishing; going on')
            if self.wait is not None:
                self.wait.cancel()
            return True
        return False

    def _go(self):
        if self.idx >= len(self.goals):
            self._finish(f'tour complete ({self.skipped} goals skipped)')
            return
        pose = self.goals[self.idx]
        self.near_since = None
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
