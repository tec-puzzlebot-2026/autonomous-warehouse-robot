"""Tests of the simulated lift (sim_forklift.LiftModel, no ROS): ramp at the motor's
speed, GOAL_SENT once, ARRIVED after settling, faults, timeout, stop.

Run:  cd main_ros && python3 -m pytest test/test_sim_forklift.py -v
"""
# Author: Raymundo Gomez

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from main_ros.manipulation.sim_forklift import LiftModel  # noqa: E402


def run(m, t, until, dt=0.05, follow=True):
    """Step the model; the 'Isaac' joint follows the command exactly."""
    while t < until:
        cmd = m.step(t, dt)
        if follow and cmd is not None:
            m.measure(cmd)
        t = round(t + dt, 3)
    return t


def test_ramps_at_motor_speed_then_arrives():
    m = LiftModel(speed_mm_s=70.0)
    m.measure(0.0)
    assert m.set_goal(70.0, 0.0) and m.events == ['GOAL_SENT 70.00 mm (sim)']
    run(m, 0.0, 0.5)
    assert 30.0 <= m.command <= 40.0 and m.state == 'MOVING'     # ~35 mm after 0.5 s
    run(m, 0.5, 1.6)
    assert m.state == 'ARRIVED' and abs(m.height - 70.0) < 1e-6


def test_out_of_range_and_no_position_are_faults():
    m = LiftModel(max_height_mm=246.0)
    assert not m.set_goal(10.0, 0.0) and m.state.startswith('FAULT')      # no joint state yet
    m.measure(0.0)
    assert not m.set_goal(300.0, 0.0) and m.state.startswith('FAULT')


def test_a_stuck_lift_reports_goal_not_reached():
    m = LiftModel(timeout_s=2.0)
    m.measure(0.0)
    m.set_goal(50.0, 0.0)
    run(m, 0.0, 3.0, follow=False)                                # the joint never moves
    assert m.state.startswith('GOAL_NOT_REACHED') and m.goal is None


def test_stop_holds_where_it_is():
    m = LiftModel()
    m.measure(0.0)
    m.set_goal(100.0, 0.0)
    run(m, 0.0, 0.5)
    m.stop()
    held = m.command
    run(m, 0.5, 2.0)
    assert m.state == 'STOPPED' and m.command == held
