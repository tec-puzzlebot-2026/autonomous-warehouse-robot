"""
route_follower.py — stage 4 without restocking: drive the inspection route (no ROS)
====================================================================================
Stage 3 plans the restocking route on the final map (restock/route.py): viewpoints
along the edge of every obstacle, because a box can only stand next to something
(on the floor against a wall or an obstacle, or on a rack). The full restocking
(restock_core.py) drives that route and, at every viewpoint, looks for boxes.

This class only DRIVES the route: no camera, no detectors, no forklift. It is what
stage 4 runs with mission.restock_mode: route_only in program_manager.yaml, to test
on the real robot that the route covers the arena and that the robot can reach every
viewpoint before the box handling is switched on.

    LOAD ─> GO (viewpoint i) ─> STOP (dwell_s, where the camera would look) ─> GO ...
        ─> [back to the start] ─> COMPLETE

A viewpoint the robot cannot reach is tried once more after clearing the costmaps,
then skipped (counted, shown on the monitor).

Interface (the same as RestockCore, so the Program Manager treats both alike):
    start(), tick(), pause(), resume(), abort(why)
    done, error, outcome, state, phase, detail, snapshot()

Ports (restock_ports.RestockPorts on the robot, a fake in the tests):
    now(), log(text), navigate(pose, profile) -> handle, cancel_motion(),
    clear_costmaps(), slam_pose()
    handle: done(), result() -> (ok, msg), cancel()
"""
# Author: Raymundo Gomez

import collections
import os
import time
from dataclasses import dataclass

from main_ros.restock import mission_store as ms
from main_ros.restock.grid import OccupancyMap
from main_ros.restock.route import RouteConfig, plan_route

LOAD, GO, STOP, COMPLETE = 'LOAD', 'GO', 'STOP', 'COMPLETE'


@dataclass
class RouteDriveSettings:
    """config/restock.yaml, section route_drive."""
    profile: str = 'transport'        # Sendero profile
    dwell_s: float = 1.0              # stand still at every viewpoint (where the camera looks)
    nav_timeout_s: float = 120.0      # per viewpoint
    return_to_start: bool = True      # drive back to where stage 4 started
    laps: int = 1                     # times round the whole route

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RouteFollower:
    """Call start(), then tick() at ~10 Hz until done."""

    mode = 'route_only'

    def __init__(self, ports, settings=None, folder=None, route_cfg=None):
        self.p = ports
        self.s = settings or RouteDriveSettings()
        self.folder = folder
        self.route_cfg = route_cfg or RouteConfig()
        self.route_info = {}
        self.viewpoints = []              # one lap, in visiting order
        self.goals = []                   # all laps (+ the start at the end)
        self.idx = 0
        self.skipped = 0
        self.retried = False
        self.wait = None
        self.wait_until = 0.0
        self.dwell_until = None
        self.paused = False
        self.done = False
        self.error = None
        self.outcome = ''
        self.state = LOAD
        self.phase = LOAD
        self.detail = 'Loading the route...'
        self.events = collections.deque(maxlen=30)

    # ── control ──────────────────────────────────────────────────────────
    def start(self):
        start = tuple(self.p.slam_pose() or (0.0, 0.0, 0.0))
        self.viewpoints = self._load_route(start)
        if not self.viewpoints:
            if not self.done:
                self._finish('COMPLETE: the map has no edge to drive along')
            return
        laps = max(1, int(self.s.laps))
        self.goals = list(self.viewpoints) * laps
        if self.s.return_to_start:
            self.goals.append(start)
        self.idx = 0
        self._event(f'route: {len(self.viewpoints)} viewpoints x {laps} lap(s)'
                    + (', then back to the start' if self.s.return_to_start else '')
                    + (f', {self.route_info.get("length_m", 0):.1f} m per lap'
                       if self.route_info.get('length_m') else ''))
        self._go()

    def pause(self):
        if self.done or self.paused:
            return
        self.paused = True
        if self.wait is not None:
            self.wait.cancel()
            self.wait = None
        self.dwell_until = None
        self.p.cancel_motion()
        self._event(f'paused at goal {self.idx + 1}/{len(self.goals)}')

    def resume(self):
        if self.done:
            return
        self.paused = False
        self.error = None
        self.retried = False
        self._event(f'resumed at goal {self.idx + 1}/{len(self.goals)}')
        self._go()

    def abort(self, why='stopped by the operator'):
        if self.wait is not None:
            self.wait.cancel()
            self.wait = None
        self.p.cancel_motion()
        if not self.done:
            self._finish(f'STOPPED at goal {self.idx + 1}/{len(self.goals)}: {why}')

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
                self.dwell_until = now + max(0.0, self.s.dwell_s)
                self._set(STOP, f'Route: stopped at goal {self.idx + 1}/{len(self.goals)}'
                          f'{self._lap_text()}')
                return
            self._failed(msg or 'navigation failed')
        elif now > self.wait_until:
            w.cancel()
            self.wait = None
            self._failed('timed out')

    # ── helpers ──────────────────────────────────────────────────────────
    def _load_route(self, start):
        """The route planned in stage 3 (route.yaml), or planned now if it is missing."""
        if self.folder:
            saved = ms.read_yaml(os.path.join(self.folder, ms.ROUTE), None) or {}
            if saved.get('viewpoints'):
                self.route_info = saved
                pts = [tuple(float(v) for v in p) for p in saved['viewpoints']]
                self._event(f'route from stage 3 ({ms.ROUTE})')
                return pts
        grid = None
        if self.folder:
            try:
                grid = OccupancyMap.from_files(os.path.join(self.folder, ms.MAP_YAML))
            except (OSError, ValueError, TypeError) as exc:
                self._event(f'cannot read the map: {exc}')
        if grid is None:
            self.error = 'no route.yaml and no map in the mission folder'
            self._finish('FAILED: ' + self.error)
            return []
        r = plan_route(grid, start, self.route_cfg)
        self.route_info = r.to_dict()
        try:
            ms.write_yaml(os.path.join(self.folder, ms.ROUTE), self.route_info)
        except OSError:
            pass
        self._event(f'route planned now: {len(r.viewpoints)} viewpoints, '
                    f'{100 * r.coverage:.0f} % of the edges')
        return list(r.viewpoints)

    def _go(self):
        if self.idx >= len(self.goals):
            self._finish(f'COMPLETE: route finished ({self.skipped} viewpoints skipped)')
            return
        pose = self.goals[self.idx]
        self.wait = self.p.navigate(pose, self.s.profile)
        self.wait_until = self.p.now() + self.s.nav_timeout_s
        home = self.s.return_to_start and self.idx == len(self.goals) - 1
        what = 'back to the start' if home else f'goal {self.idx + 1}/{len(self.goals)}'
        self._set(GO, f'Route: driving to {what}{self._lap_text()}')

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

    def _lap_text(self):
        n = len(self.viewpoints)
        laps = max(1, int(self.s.laps))
        if laps < 2 or not n or self.idx >= n * laps:
            return ''
        return f' (lap {self.idx // n + 1}/{laps})'

    def _set(self, phase, detail):
        self.phase = phase
        self.state = phase
        self.detail = detail

    def _finish(self, outcome):
        self.done = True
        self.outcome = outcome
        self.dwell_until = None
        self._set(COMPLETE, outcome)
        self._event(outcome)

    def _event(self, text):
        self.events.append((time.strftime('%H:%M:%S'), text))
        self.p.log('[route] ' + text)

    def snapshot(self):
        """The same keys as RestockCore.snapshot(), for the monitor."""
        cov = self.route_info.get('coverage')
        return {
            'mode': self.mode,
            'state': self.state, 'phase': self.phase, 'detail': self.detail,
            'paused': self.paused, 'error': self.error, 'done': self.done,
            'outcome': self.outcome,
            'route': {'next': min(self.idx, max(len(self.goals) - 1, 0)),
                      'total': len(self.goals),
                      'coverage': None if cov is None else float(cov),
                      'uncovered': self.skipped,
                      'length_m': self.route_info.get('length_m')},
            'box': None, 'boxes': {}, 'docks': [], 'delivered': {},
            'folder': self.folder or '',
            'events': list(self.events)[-6:],
        }
