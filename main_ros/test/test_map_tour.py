"""Tests of the stage-2 map tour (no ROS): it drives every viewpoint, stands still at
each, comes back to the start, skips goals it cannot reach, pauses and resumes.

Run:  cd main_ros && python3 -m pytest test/test_map_tour.py -v
"""
# Author: Raymundo Gomez

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import restock_world as RW  # noqa: E402

from main_ros.restock.map_tour import MapTour, TourSettings  # noqa: E402


class Handle:
    def __init__(self, ports, delay, ok=True, on_done=None):
        self.p, self.t, self.ok, self.on_done = ports, ports.t + delay, ok, on_done
        self.canceled = False
        self.fired = False

    def done(self):
        if self.p.t >= self.t and not self.fired:
            self.fired = True
            if self.ok and self.on_done and not self.canceled:
                self.on_done()
        return self.p.t >= self.t or self.canceled

    def result(self):
        return (False, 'canceled') if self.canceled else (self.ok, '' if self.ok else 'ABORTED')

    def cancel(self):
        self.canceled = True


class Ports:
    def __init__(self, start=(0.6, 0.6, 0.0)):
        self.t = 0.0
        self.pose = list(start)
        self.goals = []
        self.fail = set()
        self.logs = []
        self.cleared = 0
        self.canceled = 0
        self.shown = []              # (goals, kind) given to show_route

    def now(self):
        return self.t

    def log(self, text):
        self.logs.append(text)

    def navigate(self, pose, profile):
        self.goals.append((tuple(pose), self.t))
        key = (round(pose[0], 2), round(pose[1], 2))

        def arrive():
            self.pose = list(pose)
        return Handle(self, 1.0, key not in self.fail, arrive)

    def cancel_motion(self):
        self.canceled += 1

    def clear_costmaps(self):
        self.cleared += 1

    def show_route(self, poses, kind):
        self.shown.append((list(poses), kind))

    def slam_pose(self):
        return tuple(self.pose)

    def landmarks(self):
        return {'markers': [{'id': 7}], 'logos': []}


def run(tour, p, seconds=2000.0, dt=0.1):
    while p.t < seconds and not tour.done:
        tour.tick()
        p.t = round(p.t + dt, 3)


def test_tour_drives_every_waypoint_and_returns_to_the_start():
    p = Ports()
    tour = MapTour(p, TourSettings(dwell_s=1.0, pass_radius=0.0))
    tour.start(RW.make_world())
    n = len(tour.goals)
    assert n > 10
    assert tour.goals[-1] == (0.6, 0.6, 0.0)               # back to the start
    run(tour, p)
    assert tour.done and 'tour complete' in tour.outcome and tour.skipped == 0
    assert len(p.goals) == n
    gaps = [b[1] - a[1] for a, b in zip(p.goals, p.goals[1:])]
    assert min(gaps) >= 1.0 + 1.0 - 1e-6                   # 1 s driving + 1 s dwell
    assert tour.snapshot()['markers'] == [7]


def test_the_tour_is_closed_loops_round_the_arena_and_the_obstacles():
    from main_ros.restock.loop_tour import LoopConfig, plan_loops
    grid = RW.make_world()
    goals, info = plan_loops(grid, (0.6, 0.6, 0.0), LoopConfig(face_walls=False))
    assert info['loops'] >= 2 and info['inner_loops'] >= 1   # the arena + the rack / pillar
    # every loop ends where it began: the same pose shows up twice, one loop apart
    firsts = [i for i, g in enumerate(goals) if goals.index(g) != i]
    assert len(firsts) >= info['loops']
    # it drives smoothly: consecutive waypoints are close, the heading follows the loop
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(goals, goals[1:])]
    assert sorted(steps)[len(steps) // 2] < 1.0
    # the outer loop stays ring_clearance from the walls
    xs, ys = [g[0] for g in goals], [g[1] for g in goals]
    assert min(xs) > 0.4 and max(xs) < RW.W - 0.4 and min(ys) > 0.4 and max(ys) < RW.H - 0.4


def test_face_walls_turns_to_the_wall_on_the_outer_loop():
    from main_ros.restock.loop_tour import LoopConfig, plan_loops
    grid = RW.make_world()
    goals, _ = plan_loops(grid, (0.6, 0.6, 0.0), LoopConfig(inner_loops=False, loop_size=0,
                                                            face_walls=True))
    # on the bottom stretch (y ~ 0.5, driving +x counter-clockwise) it faces -y: the wall
    low = [g for g in goals if g[1] < 0.6 and 1.0 < g[0] < 3.5]
    assert low and all(math.sin(g[2]) < -0.9 for g in low)


def test_two_passes_and_unreachable_goals_are_skipped_after_one_retry():
    p = Ports()
    tour = MapTour(p, TourSettings(dwell_s=0.2, passes=2))
    tour.start(RW.make_world())
    one = len(tour.goals) // 2
    assert tour.goals[:one] == tour.goals[one:]
    bad = tour.goals[3]
    p.fail = {(round(bad[0], 2), round(bad[1], 2))}
    run(tour, p)
    assert tour.done and tour.skipped == 2                 # once per pass
    assert p.cleared == 2                                  # costmaps cleared before the retry


def test_pause_resume_and_finish_early():
    p = Ports()
    tour = MapTour(p, TourSettings(dwell_s=0.2))
    tour.start(RW.make_world())
    run(tour, p, seconds=5.0)
    goal = tour.idx
    tour.pause()
    assert p.canceled == 1
    for _ in range(50):
        tour.tick()
        p.t += 0.1
    assert tour.idx == goal and not tour.done
    tour.resume()
    assert p.goals[-1][0] == tour.goals[goal]              # the interrupted goal again
    tour.finish_early()
    assert tour.done and 'operator' in tour.outcome


def test_empty_map_finishes_at_once():
    import numpy as np
    from main_ros.restock.grid import OccupancyMap
    p = Ports()
    tour = MapTour(p)
    tour.start(OccupancyMap(np.full((40, 40), -1, dtype=np.int16), (0.0, 0.0), 0.05))
    assert tour.done and 'no room' in tour.outcome


def test_plan_gives_the_goals_without_driving_and_start_shows_them():
    p = Ports()
    tour = MapTour(p, TourSettings(passes=2))
    goals, info = tour.plan(RW.make_world(), (0.6, 0.6, 0.0))
    assert goals and not p.goals                           # planned, nothing sent
    assert len(goals) % 2 == 0 and goals[:len(goals) // 2] == goals[len(goals) // 2:]
    assert goals[len(goals) // 2 - 1] == (0.6, 0.6, 0.0)   # each pass ends at the start
    tour.start(RW.make_world())
    assert p.shown and p.shown[-1][1] == 'map_tour' and p.shown[-1][0] == tour.goals


def test_several_smaller_loops_then_one_round_everything():
    from main_ros.restock.loop_tour import LoopConfig, plan_loops
    grid = RW.make_world()
    goals, info = plan_loops(grid, (0.6, 0.6, 0.0), LoopConfig(loop_size=1.6))
    assert info['sector_loops'] >= 3 and info['loops'] >= info['sector_loops'] + 1
    one, info1 = plan_loops(grid, (0.6, 0.6, 0.0), LoopConfig(loop_size=0))
    assert info1['sector_loops'] == 0 and info1['loops'] >= 1


class MovingPorts(Ports):
    """The robot moves toward the goal a bit every tick; Sendero never finishes the
    last stretch of goal `stuck` (it cannot turn there, say)."""

    def __init__(self, stuck=None):
        super().__init__()
        self.stuck = stuck
        self.target = None

    def navigate(self, pose, profile):
        self.goals.append((tuple(pose), self.t))
        self.target = tuple(pose)
        p = self

        class H:
            canceled = False

            def done(self):
                g = p.target
                at = math.hypot(p.pose[0] - g[0], p.pose[1] - g[1]) < 0.05
                return self.canceled or (at and g != p.stuck)

            def result(self):
                return (not self.canceled, '')

            def cancel(self):
                self.canceled = True
        return H()

    def step(self, dt):
        if self.target is None:
            return
        dx, dy = self.target[0] - self.pose[0], self.target[1] - self.pose[1]
        d = math.hypot(dx, dy)
        k = min(1.0, 0.3 * dt / max(d, 1e-9))
        self.pose[0] += k * dx
        self.pose[1] += k * dy


def test_drives_through_waypoints_and_never_hangs_at_one():
    p = MovingPorts()
    tour = MapTour(p, TourSettings(pass_radius=0.35, stall_s=3.0))
    tour.start(RW.make_world())
    p.stuck = tour.goals[-1]                         # Sendero never finishes the last one
    while p.t < 3000 and not tour.done:
        tour.tick()
        p.step(0.1)
        p.t = round(p.t + 0.1, 3)
    assert tour.done and tour.skipped == 0
    assert any('without Sendero finishing' in line for line in p.logs)
