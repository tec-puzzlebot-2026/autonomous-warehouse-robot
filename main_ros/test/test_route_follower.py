"""Tests of stage 4 in route-only mode (no ROS): the robot drives the route planned
in stage 3 (route.yaml), stops at every viewpoint, comes back to the start, skips a
viewpoint it cannot reach after one retry, pauses, resumes and stops.

Run:  cd main_ros && python3 -m pytest test/test_route_follower.py -v
"""
# Author: Raymundo Gomez

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import restock_world as RW  # noqa: E402
from test_map_tour import Ports  # noqa: E402

from main_ros.restock import mission_store as ms  # noqa: E402
from main_ros.restock.route import RouteConfig, plan_route  # noqa: E402
from main_ros.restock.route_follower import RouteDriveSettings, RouteFollower  # noqa: E402

START = (0.6, 0.6, 0.0)


def folder_with_route(tmp_path, save_route=True):
    grid = RW.make_world()
    RW.write_map_files(str(tmp_path / os.path.splitext(ms.MAP_YAML)[0]), grid)
    r = plan_route(grid, START, RouteConfig())
    if save_route:
        ms.write_yaml(str(tmp_path / ms.ROUTE), r.to_dict())
    return str(tmp_path), r


def run(f, p, seconds=3000.0, dt=0.1):
    while p.t < seconds and not f.done:
        f.tick()
        p.t = round(p.t + dt, 3)


def test_drives_every_stage_3_viewpoint_then_back_to_the_start(tmp_path):
    folder, r = folder_with_route(tmp_path)
    p = Ports(START)
    f = RouteFollower(p, RouteDriveSettings(dwell_s=1.0), folder)
    f.start()
    assert f.goals[:-1] == [tuple(v) for v in r.to_dict()['viewpoints']]
    assert f.goals[-1] == START
    run(f, p)
    assert f.done and f.outcome.startswith('COMPLETE') and f.skipped == 0
    assert [g for g, _ in p.goals] == f.goals
    gaps = [b[1] - a[1] for a, b in zip(p.goals, p.goals[1:])]
    assert min(gaps) >= 2.0 - 1e-6                         # 1 s driving + 1 s stopped
    snap = f.snapshot()
    assert snap['mode'] == 'route_only' and snap['route']['total'] == len(f.goals)
    assert snap['route']['coverage'] is not None


def test_an_unreachable_viewpoint_is_retried_once_then_skipped(tmp_path):
    folder, _ = folder_with_route(tmp_path)
    p = Ports(START)
    f = RouteFollower(p, RouteDriveSettings(dwell_s=0.2, laps=2, return_to_start=False), folder)
    f.start()
    n = len(f.viewpoints)
    assert len(f.goals) == 2 * n
    bad = f.goals[2]
    p.fail = {(round(bad[0], 2), round(bad[1], 2))}
    run(f, p)
    assert f.done and f.skipped == 2 and p.cleared == 2    # once per lap
    assert f.snapshot()['route']['uncovered'] == 2


def test_pause_resume_and_abort(tmp_path):
    folder, _ = folder_with_route(tmp_path)
    p = Ports(START)
    f = RouteFollower(p, RouteDriveSettings(dwell_s=0.2), folder)
    f.start()
    run(f, p, seconds=5.0)
    goal = f.idx
    f.pause()
    assert p.canceled == 1 and f.paused
    for _ in range(30):
        f.tick()
        p.t += 0.1
    assert f.idx == goal and not f.done
    f.resume()
    assert p.goals[-1][0] == f.goals[goal]                 # the interrupted goal again
    f.abort('X')
    assert f.done and f.outcome.startswith('STOPPED')


def test_without_route_yaml_it_plans_the_route_itself(tmp_path):
    folder, r = folder_with_route(tmp_path, save_route=False)
    p = Ports(START)
    f = RouteFollower(p, RouteDriveSettings(), folder)
    f.start()
    assert abs(len(f.viewpoints) - len(r.viewpoints)) <= 2 and not f.done   # map read from the PGM
    assert os.path.exists(os.path.join(folder, ms.ROUTE))


def test_no_map_and_no_route_is_an_error(tmp_path):
    p = Ports(START)
    f = RouteFollower(p, RouteDriveSettings(), str(tmp_path))
    f.start()
    assert f.done and f.error and f.outcome.startswith('FAILED')
