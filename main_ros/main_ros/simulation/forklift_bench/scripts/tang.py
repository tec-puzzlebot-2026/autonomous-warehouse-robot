"""UART and timing model for the documented Tang/Nano interface."""
import errno
import math
import os
import pty
import re
import struct
import tty
from collections import deque
from functools import reduce
from operator import xor


MARKS = (120, 200, 300, 400, 500, 600, 700, 800)
NO_VALUE = 0xFFFF


def checksum(body):
    return reduce(xor, body, 0)


def signed32(value):
    return value - (1 << 32) if value & (1 << 31) else value


def ir_tables(header):
    defines = dict(re.findall(r'^`define (IR\w+) (\d+)', header.read_text(), re.MULTILINE))
    return tuple(tuple(int(defines[f"IR{i}_ADC_{mm // 10}"]) for mm in MARKS)
                 if defines[f"IR{i}_CALIBRATED"] == "1" else ()
                 for i in (1, 2))


def ir_mm(adc, unstable, table):
    if unstable:
        return 5, NO_VALUE
    if not table:
        return 6, NO_VALUE
    if adc > table[0]:
        return (1, 120) if adc <= table[0] + 12 else (3, NO_VALUE)
    if adc < table[-1]:
        return (2, 800) if adc >= table[-1] - 10 else (4, NO_VALUE)
    for a, b, lo, hi in zip(table, table[1:], MARKS, MARKS[1:]):
        if adc >= b:
            gain = ((hi - lo) * 1024 + (a - b) // 2) // (a - b)
            return 0, lo + ((a - adc) * gain + 512) // 1024
    raise AssertionError("invalid IR table")


class Nano:
    def __init__(self, tables):
        self.tables = tables
        self.windows = (deque(maxlen=9), deque(maxlen=9))
        self.sample_at = 0.
        self.tx_at = .300

    @staticmethod
    def adc(distance_m, table):
        mm = distance_m * 1000
        if mm < 120:
            return min(1023, table[0] + 13 + round(120 - mm))
        if mm > 800:
            return max(0, table[-1] - 11)
        for a, b, lo, hi in zip(table, table[1:], MARKS, MARKS[1:]):
            if mm <= hi:
                return round(a + (b - a) * (mm - lo) / (hi - lo))
        return table[-1]

    def step(self, now, ranges):
        if now + 1e-9 >= self.sample_at:
            self.sample_at = (now if now - self.sample_at >= .060 else self.sample_at) + .060
            for window, table, distance in zip(self.windows, self.tables, ranges()):
                window.append(self.adc(distance, table))
        if now + 1e-9 < self.tx_at:
            return None
        self.tx_at = (now if now - self.tx_at >= .300 else self.tx_at) + .300
        fields = bytearray(b'\xA9')
        for window in self.windows:
            ordered = sorted(window)
            spread = ordered[-2] - ordered[1] if len(ordered) == 9 else NO_VALUE
            fields.extend(struct.pack("<BHH", int(spread > 20), ordered[len(ordered) // 2], spread))
        fields.append(checksum(fields))
        return bytes(fields)


class TangProtocolModel:
    """Behavioral model of the supplied controller, stepped at the physics rate."""
    def __init__(self, tables=None):
        self.tables = tables
        self.mode = "D"
        self.span = self.margin = self.window = self.anchor = self.raw = self.token = 0
        self.goal = self.fault = self.drive = 0
        self.busy = self.hold = self.correcting_hold = False
        self.now = self.heartbeat = self.last_motion = self.last_stop = self.jog_at = 0.
        self.telemetry_at = 0.
        self.drive_at = self.drive_origin = 0
        self.pulse = False
        self.leads = {1: 5, -1: 7}
        self.pulse_steps = {1: 4, -1: 4}
        self.last_drive = 0
        self.sample_at = .020
        self.sample_rel = self.velocity = 0
        self.best_error = math.inf
        self.progress_at = 0.
        self.ir_received = None
        self.ir_at = -math.inf
        self.buffer = bytearray()

    @property
    def relative(self):
        return (1 if self.span >= 0 else -1) * (self.raw - self.anchor)

    def stop(self, clear_goal=False):
        self.drive = 0
        self.last_stop = self.now
        if clear_goal:
            self.busy = self.hold = False

    def trip(self, code):
        self.mode = "F"
        self.stop(clear_goal=True)
        self.fault = code

    def feed(self, data):
        for octet in data:
            if octet == 10:
                self.command(bytes(self.buffer).rstrip(b"\r"))
                self.buffer.clear()
            elif len(self.buffer) < 64:
                self.buffer.append(octet)
            else:
                self.buffer.clear()

    def command(self, line):
        # Frame lengths and XOR come from command_parser.v.
        if line in (b"D", b"M", b"Z", b"H", b"W", b"S", b"X"):
            name = line
            body = b""
        elif line[:1] in (b"K", b"A", b"I", b"G"):
            name, body = line[:1], line[1:-2]
            sizes = {b"K": 20, b"A": 8, b"I": 8, b"G": 12}
            if len(body) != sizes[name] or any(c not in b"0123456789ABCDEF" for c in line[1:]):
                return
            if checksum(line[:-2]) != int(line[-2:], 16):
                return
        else:
            return
        if name == b"H":
            self.heartbeat = self.now
        elif name == b"D":
            self.mode, self.span, self.goal, self.fault = "D", 0, 0, 0
            self.stop(clear_goal=True)
        elif name == b"M" and self.mode == "D":
            self.mode, self.fault, self.heartbeat = "M", 0, self.now
            self.stop(clear_goal=True)
        elif name == b"K" and self.mode in ("D", "C"):
            span = signed32(int(body[:8], 16))
            margin, tolerance, bottom = (int(body[i:i+4], 16) for i in (8, 12, 16))
            if 500 <= abs(span) <= 20000 and 1 <= margin <= 500 and 2 * margin < abs(span) and 1 <= tolerance <= min(50, margin) and bottom == tolerance:
                self.span, self.margin, self.window = span, margin, tolerance
                self.mode, self.fault, self.goal = "C", 0, 0
                self.heartbeat = self.now
                self.leads = {1: 5, -1: 7}
                self.pulse_steps = {1: 4, -1: 4}
                self.stop(clear_goal=True)
        elif name in (b"A", b"Z") and self.mode == "C" and self.now - max(self.last_stop, self.last_motion) >= .6:
            self.anchor = self.raw if name == b"Z" else signed32(int(body, 16))
            self.mode, self.heartbeat = "R", self.now
        elif name == b"I" and int(body, 16):
            self.token = int(body, 16)
        elif name == b"X":
            self.stop(clear_goal=True)
            self.fault = 0
        elif name == b"G" and self.mode == "R" and not self.busy and not self.drive:
            target, window = int(body[:8], 16), int(body[8:], 16)
            if target <= abs(self.span) - self.margin and window == self.window:
                self.goal, self.fault = target, 0
                self.busy = self.hold = True
                self.correcting_hold = False
                self.best_error, self.progress_at = math.inf, self.now
                self.last_drive = 0
                self.heartbeat = self.now
                self.last_stop = self.now
        elif name in (b"W", b"S") and self.mode == "M":
            direction = 1 if name == b"W" else -1
            self.heartbeat = self.jog_at = self.now
            if self.drive not in (0, direction):
                self.stop()
            elif self.drive or self.now - self.last_stop >= .020:
                self.drive = direction

    def accept_ir(self, frame):
        if not self.tables or len(frame) != 12 or frame[0] != 0xA9 or frame[-1] != checksum(frame[:-1]):
            return
        readings = (struct.unpack_from("<BHH", frame, i) for i in (1, 6))
        channels = tuple(readings)
        if any(q not in (0, 1) or adc > 1023 for q, adc, _ in channels):
            return
        self.ir_received = [
            (*ir_mm(adc, q == 1, table), adc, spread)
            for (q, adc, spread), table in zip(channels, self.tables)
        ]
        self.ir_at = self.now

    def control_goal(self):
        rel = self.relative
        error = self.goal - rel
        direction = 1 if error >= 0 else -1
        if not self.busy:
            if not self.hold or abs(error) <= self.window:
                return
            self.busy = self.correcting_hold = True
            self.best_error, self.progress_at = math.inf, self.now
            self.last_drive = 0
        if self.drive:
            if self.now - max(self.drive_at, self.last_motion) >= .5:
                self.trip(2)
            elif (rel - self.drive_origin) * self.drive < -8:
                self.trip(3)
            elif self.pulse:
                if self.now - self.drive_at >= self.pulse_steps[self.drive] * .020:
                    self.stop()
            elif (self.goal - (rel + self.velocity * self.leads[self.drive])) * self.drive <= 0:
                self.stop()
            return
        quiet = self.now - max(self.last_stop, self.last_motion)
        if self.last_drive:
            if quiet < .6:
                return
            if abs(error) <= self.window:
                self.busy = False
                return
            if self.last_drive * error >= 0:
                adjust = -1
            else:
                adjust = 1
            if self.pulse:
                self.pulse_steps[self.last_drive] = min(8, max(2, self.pulse_steps[self.last_drive] - adjust))
            else:
                self.leads[self.last_drive] = min(15, max(0, self.leads[self.last_drive] + adjust))
        elif abs(error) <= self.window:
            if quiet >= .6:
                self.busy = False
            return
        elif self.now - self.last_stop < .020:
            return
        excess = max(0, abs(error) - self.window)
        if excess <= self.best_error - 3:
            self.best_error, self.progress_at = excess, self.now
        elif self.now - self.progress_at >= 15:
            self.fault = 6 if self.correcting_hold else 5
            self.stop(clear_goal=True)
            return
        self.drive = self.last_drive = direction
        self.drive_at, self.drive_origin = self.now, rel
        self.pulse = abs(error) <= 40
        self.sample_rel, self.velocity, self.sample_at = rel, 0, self.now + .020

    def step(self, now, raw, nano_frame=None):
        self.now = now
        if nano_frame:
            self.accept_ir(nano_frame)
        if raw != self.raw:
            self.raw, self.last_motion = raw, now
        if now + 1e-9 >= self.sample_at:
            self.velocity = self.relative - self.sample_rel
            self.sample_rel, self.sample_at = self.relative, now + .020
        if self.mode not in ("D", "F") and now - self.heartbeat >= .3:
            self.command(b"D")
        if self.mode == "M" and self.drive and now - self.jog_at >= .3:
            self.stop()
        if self.mode == "R":
            if not -self.window <= self.relative <= abs(self.span) + self.window:
                self.trip(1)
            else:
                self.control_goal()
        moving = self.effective_drive()
        error = abs(self.goal - self.relative)
        duty = .75 if self.mode == "M" else .70 if self.pulse or error <= 40 else min(.75, .70 + .01 * (error // 32))
        packet = b""
        if now + 1e-9 >= self.telemetry_at:
            self.telemetry_at = now + .200
            packet = self.telemetry(bool(moving))
        return moving * duty, packet

    def effective_drive(self):
        if self.mode == "M":
            return self.drive
        if self.mode != "R":
            return 0
        if self.drive == 1 and self.relative >= abs(self.span) - self.margin:
            return 0
        if self.drive == -1 and self.relative <= -self.window:
            return 0
        return self.drive

    def telemetry(self, moving):
        packet = (f"@{self.raw & 0xffffffff:08X}\n#{self.relative & 0xffffffff:08X}\n"
                  f"!{self.mode}{int(moving)}{int(self.busy)}{self.fault:X}"
                  f"{self.span & 0xffffffff:08X}{self.goal:08X}\n${self.token:08X}\n")
        if not self.tables:
            return (packet + "?FFFFFFFFF\n").encode()
        fresh = self.now - self.ir_at < 2 and self.ir_received is not None
        for i, (prefix, spread_prefix) in enumerate((("?", "^"), ("&", "%"))):
            if fresh:
                status, mm, adc, spread = self.ir_received[i]
                packet += f"{prefix}{status:X}{mm:04X}{adc:04X}\n{spread_prefix}{spread:04X}\n"
            else:
                packet += f"{prefix}FFFFFFFFF\n{spread_prefix}FFFF\n"
        return packet.encode()


class VirtualSerial:
    def __init__(self):
        self.master, self.slave = pty.openpty()
        tty.setraw(self.slave)
        os.set_blocking(self.master, False)
        self.port = os.ttyname(self.slave)
        self.pending = bytearray()

    def read(self):
        try:
            return os.read(self.master, 4096)
        except BlockingIOError:
            return b""

    def write(self, packet):
        if len(self.pending) + len(packet) > 4096:
            self.pending.clear()
        self.pending.extend(packet)
        if self.pending:
            try:
                del self.pending[:os.write(self.master, self.pending)]
            except BlockingIOError:
                pass
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise

    def close(self):
        os.close(self.master)
        os.close(self.slave)
