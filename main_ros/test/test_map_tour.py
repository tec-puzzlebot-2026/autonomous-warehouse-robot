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


def test_tour_visits_every_viewpoint_and_returns_to_the_start():
    p = Ports()
    tour = MapTour(p, TourSettings(dwell_s=1.0))
    tour.start(RW.make_world())
    n = len(tour.goals)
    assert n > 10
    assert tour.goals[-1] == (0.6, 0.6, 0.0)               # back to the start: a loop
    run(tour, p)
    assert tour.done and 'tour complete' in tour.outcome and tour.skipped == 0
    assert len(p.goals) == n
    # the robot stood still dwell_s at each goal before driving on
    gaps = [b[1] - a[1] for a, b in zip(p.goals, p.goals[1:])]
    assert min(gaps) >= 1.0 + 1.0 - 1e-6                   # 1 s driving + 1 s dwell
    assert tour.snapshot()['markers'] == [7]


def test_two_passes_and_unreachable_goals_are_skipped_after_one_retry():
    p = Ports()
    tour = MapTour(p, TourSettings(dwell_s=0.2, passes=2, loop_every_m=0.0))
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
    tour.start(OccupancyMap(np.zeros((40, 40), dtype=np.int16), (0.0, 0.0), 0.05))
    assert tour.done and 'no edge' in tour.outcome
    assert math.isfinite(len(tour.goals))


def test_plan_gives_the_goals_without_driving_and_start_shows_them():
    p = Ports()
    tour = MapTour(p, TourSettings(passes=2))
    goals, r = tour.plan(RW.make_world(), (0.6, 0.6, 0.0))
    assert goals and not p.goals                           # planned, nothing sent
    assert len(goals) == 2 * (len(r.viewpoints) + tour.loops + 1)   # + detours, start
    tour.start(RW.make_world())
    assert p.shown and p.shown[-1][1] == 'map_tour' and p.shown[-1][0] == tour.goals


def test_the_tour_goes_back_to_earlier_places_for_loop_closures():
    p = Ports()
    tour = MapTour(p, TourSettings(dwell_s=0.2, loop_every_m=3.0, loop_min_m=2.0,
                                   loop_radius_m=1.5))
    tour.start(RW.make_world())
    assert tour.loops >= 1
    # a detour goal repeats an earlier viewpoint (or the start) of the same pass
    seen, repeats = {(0.6, 0.6)}, 0
    for g in tour.goals[:-1]:
        key = (round(g[0], 3), round(g[1], 3))
        repeats += key in seen
        seen.add(key)
    assert repeats == tour.loops
    run(tour, p)
    assert tour.done and tour.skipped == 0


def test_no_detours_when_switched_off():
    p = Ports()
    tour = MapTour(p, TourSettings(loop_every_m=0.0))
    tour.start(RW.make_world())
    assert tour.loops == 0
    assert len(set((round(g[0], 3), round(g[1], 3)) for g in tour.goals[:-1])) \
        == len(tour.goals) - 1
