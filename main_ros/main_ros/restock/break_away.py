"""
break_away.py — the minimum speed a turn in place needs to start at all (no ROS)
================================================================================
A robot turning on the spot has to overcome static friction first: below some speed
the wheels do not turn the robot (the sim forklift does not move at 0.4 rad/s). Every
in-place turn command gets at least `floor`; when a commanded turn leaves the heading
still for `stall_s`, the floor goes up by `step` (up to `max_floor`) and stays there.

    ba = BreakAway(floor=0.5)
    w_sent = ba.command(v, w, heading, now)
"""
# Author: Raymundo Gomez

import math


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class BreakAway:
    def __init__(self, floor=0.5, max_floor=1.2, step=0.15, stall_s=1.0, log=None):
        self.floor, self.max_floor, self.step = float(floor), float(max_floor), float(step)
        self.stall_s = float(stall_s)
        self.log = log
        self._stall = None                      # (time, heading) of the last progress

    def command(self, v, w, heading, now):
        """Angular speed to send for (v, w); only turns in place are touched."""
        w = float(w)
        if abs(w) < 1e-6 or abs(v) > 0.01 or heading is None:
            self._stall = None
            return w
        w = math.copysign(max(abs(w), self.floor), w)
        if self._stall is None:
            self._stall = (now, heading)
        elif abs(_wrap(heading - self._stall[1])) > math.radians(1.0):
            self._stall = (now, heading)                  # it turns
        elif now - self._stall[0] > self.stall_s and self.floor < self.max_floor:
            self.floor = min(self.floor + self.step, self.max_floor)
            self._stall = (now, heading)
            if self.log:
                self.log(f'restock: a turn at {abs(w):.2f} rad/s does not move the robot: '
                         f'turning in place at >= {self.floor:.2f} rad/s from now on')
        return w
