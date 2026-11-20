#!/usr/bin/env python3
"""PuzzleBot bring-up: leaves everything READY, does not drive the robot.

Reuses debug_teleup_script.py (same folder) for SSH, the micro-ROS agent on the
Jetson and the /cmd_vel check, and adds a check of the encoder topics. It does
NOT start the velocity relay or the keyboard teleop, so it is safe to leave
running while you use motion_noise_test (which publishes on /cmd_vel itself):

    python3 puzzlebot_ready.py            # terminal 1: leave it open
    ros2 run main_ros motion_noise_test sweep      # terminal 2

Hackerboard reset: if the board does not connect to the agent (a common state after
the previous agent was killed), the script pulses the ESP32 reset line (RTS) over the
same USB serial port, through SSH, and retries once. Force it at start with --reset.
This relies on the board's USB-serial chip being wired to the ESP32 EN pin (standard
auto-reset circuit); if nothing happens, press the reset button by hand.

For teleoperation use debug_teleup_script.py instead. Never run both together:
the relay publishes zeros on /cmd_vel at 20 Hz and would fight the test.
Ctrl+C stops the agent only if this script started it.
"""
import shlex
import signal
import subprocess
import sys
import time

import debug_teleup_script as dts

ENCODER_TOPICS = ("/VelocityEncR", "/VelocityEncL")   # keep equal to puzzlebot.yaml
FORCE_RESET = "--reset" in sys.argv

# Runs ON THE JETSON (stdlib only). Same sequence as esptool's "hard reset":
# DTR released (GPIO0 high -> normal boot), RTS asserted (EN low) then released.
RESET_CODE = """
import fcntl, os, struct, termios, time
fd = os.open(%r, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
def clear(bit): fcntl.ioctl(fd, termios.TIOCMBIC, struct.pack("I", bit))
def setbit(bit): fcntl.ioctl(fd, termios.TIOCMBIS, struct.pack("I", bit))
clear(termios.TIOCM_DTR)
setbit(termios.TIOCM_RTS)
time.sleep(0.15)
clear(termios.TIOCM_RTS)
time.sleep(0.05)
os.close(fd)
"""


def reset_hackerboard(wait=3.0):
    """Pulse the ESP32 reset line through the Jetson's serial port. True if the pulse was sent."""
    dts.log("Resetting the Hackerboard (RTS pulse on %s)..." % dts.JETSON_SERIAL_PORT)
    r = dts.ssh_run("python3 -c " + shlex.quote(RESET_CODE % dts.JETSON_SERIAL_PORT), timeout=15)
    if r.returncode != 0:
        dts.log("Reset failed: " + (r.stderr.strip() or "exit %d" % r.returncode))
        return False
    dts.log("Reset pulse sent. The board takes a few seconds to boot and reconnect.")
    time.sleep(wait)
    return True


def publisher_count(env, topic):
    """Number of publishers of `topic` seen from this machine (None if ros2 timed out)."""
    try:
        r = subprocess.run(["ros2", "topic", "info", topic], env=env,
                           capture_output=True, text=True, timeout=8)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return 0                                   # "Unknown topic"
    for line in r.stdout.splitlines():
        if line.strip().startswith("Publisher count:"):
            try:
                return int(line.split(":", 1)[1])
            except ValueError:
                return 0
    return 0


def check_encoders(env, timeout=15):
    """Wait until every encoder topic has a publisher; return {topic: count}."""
    deadline = time.monotonic() + timeout
    counts = {}
    while True:
        counts = {t: publisher_count(env, t) for t in ENCODER_TOPICS}
        if all(c for c in counts.values()) or time.monotonic() >= deadline or not dts.running:
            return counts
        time.sleep(1)


def show_topics(env):
    try:
        r = subprocess.run(["ros2", "topic", "list", "-t"], env=env,
                           capture_output=True, text=True, timeout=10)
        dts.log("Topics seen from this machine:\n" + (r.stdout.strip() or r.stderr.strip()))
    except subprocess.TimeoutExpired:
        dts.log("ros2 topic list timed out")


def main():
    signal.signal(signal.SIGINT, dts.signal_handler)
    signal.signal(signal.SIGTERM, dts.signal_handler)
    try:
        dts.check_ssh()
        dts.start_jetson()
        if FORCE_RESET:
            reset_hackerboard()
        try:
            dts.check_hackerboard()
        except RuntimeError as exc:
            if FORCE_RESET:
                raise
            dts.log("Hackerboard did not connect (%s). Resetting it once and retrying..."
                    % str(exc).splitlines()[0])
            if not reset_hackerboard():
                raise
            dts.check_hackerboard()
        env = dts.host_ros_env()
        counts = check_encoders(env)
        if all(counts.values()):
            dts.log("READY: agent up, /cmd_vel has a subscriber, encoder topics have publishers.")
        else:
            dts.log("WARNING: no publisher for " + ", ".join(
                t for t, c in counts.items() if not c) +
                ". Teleop still works, but motion_noise_test needs the encoders.")
            show_topics(env)
        dts.log("Left running. Ctrl+C to stop.")
        while dts.running:
            if dts.remote_process is not None and dts.remote_process.poll() is not None:
                raise RuntimeError("Remote micro-ROS SSH session exited.")
            time.sleep(0.5)
        return 0
    except Exception as exc:                       # noqa: BLE001
        dts.log(f"ERROR: {exc}")
        return 1
    finally:
        dts.cleanup()


if __name__ == "__main__":
    sys.exit(main())