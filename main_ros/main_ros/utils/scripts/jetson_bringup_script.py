#!/usr/bin/env python3
"""PuzzleBot bringup, run LOCALLY on the Jetson.

Starts, in order (all on the Jetson, no SSH):
  1. micro-ROS agent      -> motors (/cmd_vel), encoders (/VelocityEncL/R,
                             /robot_vel) and LiDAR (/Lidar) from the Hackerboard
  2. camera + EKF launch  -> ros2 launch main_ros ekf_localization.launch.py
  3. velocity relay       -> differential_drive: /cmd_vel_in -> /cmd_vel
                             (speed limits + stop when commands stop)
Ctrl+C stops everything this script started.

Keyboard teleop is NOT started here: it needs its own interactive terminal.
Drive from the laptop or a second SSH session with:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in

Run on the Jetson:
    ros2 launch main_ros jetson_bringup.launch.py
    ros2 run main_ros jetson_bringup_script
"""
import os
import shlex
import signal
import subprocess
import sys
import time

ROS_SETUP = "/opt/ros/humble/setup.bash"
WORKSPACE_SETUP = "/home/puzzlebot/te3003b_ws/install/setup.bash"   # has main_ros + micro_ros_agent

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
BOARD_TOPICS = ("/cmd_vel",)          # topics the Hackerboard must be subscribed to

ENABLE_CAMERA_LAUNCH = True
CAMERA_LAUNCH_PACKAGE = "main_ros"
CAMERA_LAUNCH_FILE = "ekf_localization.launch.py"

ENABLE_RELAY = True
MAX_LINEAR = 0.3                      # m/s   (relay speed limits)
MAX_ANGULAR = 1.5                     # rad/s
CMD_TIMEOUT = 0.5                     # s without commands -> stop

LOG_DIR = os.path.expanduser("~/.ros/puzzlebot_bringup")   # agent output goes here

running = True
env = None
agent_process = None
agent_log = None
camera_process = None
relay_process = None


def log(message):
    print(f"[JETSON BRINGUP] {message}", flush=True)


# ----------------------------------------------------------------------------
def ros_env():
    """ROS environment from the setup files, independent of how we were started."""
    cmd = f"source {shlex.quote(ROS_SETUP)} && source {shlex.quote(WORKSPACE_SETUP)} && env -0"
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError("Cannot source ROS setup: " +
                           result.stderr.decode(errors="replace").strip())
    out = os.environ.copy()
    for entry in result.stdout.split(b"\0"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            out[key.decode(errors="replace")] = value.decode(errors="replace")
    return out


def pids_of(pattern, exact=False):
    args = ["pgrep", "-x" if exact else "-f", pattern]
    result = subprocess.run(args, capture_output=True, text=True)
    own = {os.getpid(), os.getppid()}
    return [int(x) for x in result.stdout.split() if x.isdigit() and int(x) not in own]


def start(cmd, label, **kwargs):
    log(f"Starting {label}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env, start_new_session=True, **kwargs)


def stop_process(proc, label, first=signal.SIGINT, grace=10):
    """SIGINT first (clean ROS shutdown), then SIGTERM, then SIGKILL."""
    if proc is None or proc.poll() is not None:
        return
    log(f"Stopping {label}...")
    for sig, wait in ((first, grace), (signal.SIGTERM, 4), (signal.SIGKILL, 3)):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return


# ----------------------------------------------------------------------------
def start_agent():
    global agent_process, agent_log
    existing = pids_of("micro_ros_agent", exact=True)
    if existing:
        log(f"micro-ROS agent already running (PID {existing}). Reusing it.")
        return
    if not os.path.exists(SERIAL_PORT):
        raise RuntimeError(f"{SERIAL_PORT} not found - is the Hackerboard connected?")
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "micro_ros_agent.log")
    agent_log = open(path, "w")
    agent_process = start(
        ["ros2", "run", "micro_ros_agent", "micro_ros_agent",
         "serial", "--dev", SERIAL_PORT, "-b", str(BAUD_RATE)],
        "micro-ROS agent", stdout=agent_log, stderr=subprocess.STDOUT)
    log(f"  agent output -> {path}")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and running:
        if agent_process.poll() is not None:
            raise RuntimeError(f"micro-ROS agent exited ({agent_process.returncode}); see {path}")
        if pids_of("micro_ros_agent", exact=True):
            log("micro-ROS agent running.")
            return
        time.sleep(0.5)
    raise RuntimeError("micro-ROS agent was not detected within 15 seconds.")


def check_hackerboard(timeout=25):
    log("Waiting for the Hackerboard /cmd_vel subscription...")
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline and running:
        ready = True
        for topic in BOARD_TOPICS:
            try:
                r = subprocess.run(["ros2", "topic", "info", topic],
                                   env=env, capture_output=True, text=True, timeout=6)
            except subprocess.TimeoutExpired:
                ready, last[topic] = False, "ros2 topic info timed out"
                continue
            last[topic] = r.stdout.strip() or r.stderr.strip()
            count = 0
            for line in r.stdout.splitlines():
                if line.strip().startswith("Subscription count:"):
                    try:
                        count = int(line.split(":", 1)[1])
                    except ValueError:
                        pass
            ready &= count >= 1
        if ready:
            log("Hackerboard connected: motors, encoders and LiDAR are live.")
            return
        time.sleep(1)
    raise RuntimeError("Hackerboard /cmd_vel subscription not visible. Check the board's "
                       "power/USB and the agent log.\n" +
                       "\n".join(f"{t}: {d}" for t, d in last.items()))


def start_camera_launch():
    global camera_process
    pattern = f"ros2 launch {CAMERA_LAUNCH_PACKAGE} {CAMERA_LAUNCH_FILE}"
    existing = pids_of(pattern)
    if existing:
        log(f"Camera/EKF launch already running (PID {existing}). Reusing it.")
        return
    camera_process = start(["ros2", "launch", CAMERA_LAUNCH_PACKAGE, CAMERA_LAUNCH_FILE],
                           "camera/EKF launch")
    time.sleep(3)
    if camera_process.poll() is not None:
        raise RuntimeError(f"Camera/EKF launch exited during startup ({camera_process.returncode}).")
    log("Camera/EKF launch running.")


def start_relay():
    global relay_process
    relay_process = start(
        ["ros2", "run", "main_ros", "differential_drive", "--ros-args",
         "-p", f"max_linear:={MAX_LINEAR}",
         "-p", f"max_angular:={MAX_ANGULAR}",
         "-p", f"cmd_timeout:={CMD_TIMEOUT}"],
        "velocity relay")
    time.sleep(1.5)
    if relay_process.poll() is not None:
        raise RuntimeError("Velocity relay exited during startup "
                           "(is main_ros built with the differential_drive entry point?).")
    log("Velocity relay running: publish /cmd_vel_in to drive.")


def cleanup():
    log("Shutting down...")
    stop_process(relay_process, "velocity relay", grace=4)
    stop_process(camera_process, "camera/EKF launch", grace=10)
    stop_process(agent_process, "micro-ROS agent", grace=4)
    if agent_log:
        agent_log.close()
    log("Shutdown complete.")


def signal_handler(_signum, _frame):
    global running
    running = False


def main():
    global env
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        env = ros_env()
        start_agent()
        check_hackerboard()
        if ENABLE_CAMERA_LAUNCH:
            start_camera_launch()
        if ENABLE_RELAY:
            start_relay()
        log("Everything running. Drive with (laptop or another SSH terminal):")
        log("  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in")
        log("Press Ctrl+C to stop.")
        while running:
            for proc, label in ((agent_process, "micro-ROS agent"),
                                (camera_process, "Camera/EKF launch"),
                                (relay_process, "Velocity relay")):
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(f"{label} exited ({proc.returncode}).")
            time.sleep(0.5)
        return 0
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())