#!/usr/bin/env python3
"""PuzzleBot bringup, run LOCALLY on the Jetson.

Starts, in order (all on the Jetson, no SSH):
  1. micro-ROS agent      -> motors (/cmd_vel), encoders (/VelocityEncL/R,
                             /robot_vel) from the Hackerboard
  2. LiDAR (RPLidar)      -> sllidar_ros2: /laser_scan in frame laser_frame
  3. camera + EKF launch  -> ros2 launch main_ros ekf_localization.launch.py
                             (--no-camera: the Program Manager starts the camera)
  4. velocity relay       -> differential_drive: /cmd_vel_in -> /cmd_vel
                             (speed limits + stop when commands stop)
Ctrl+C stops everything this script started.

The Program Manager runs it as its robot_bringup step (mode:=real):
    ros2 run main_ros jetson_bringup_script --no-camera
Options: --no-camera, --no-lidar, --no-relay.

Serial devices are opened by their /dev/serial/by-id name, so the Hackerboard
and the LiDAR never swap when the ttyUSB numbers change.

Keyboard teleop is NOT started here: it needs its own interactive terminal.
Drive from the laptop or a second SSH session with:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in

Run on the Jetson:
    ros2 launch main_ros jetson_bringup.launch.py
    ros2 run main_ros jetson_bringup_script
"""
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time

ROS_SETUP = "/opt/ros/humble/setup.bash"
WORKSPACE_SETUP = "/home/puzzlebot/te3003b_ws/install/setup.bash"   # has main_ros + micro_ros_agent

# Hackerboard (ESP32, CP2102N adapter)
SERIAL_PORT = ("/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_"
               "cc08d56b45e7ed11afbe666262c613ac-if00-port0")
BAUD_RATE = 115200

# RPLidar (SLAMTEC CP2102 adapter, serial 0001)
ENABLE_LIDAR = True
LIDAR_PORT = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
LIDAR_BAUD = 115200                   # RPLidar A1 / A2M8 (C1: 460800, A2M12/A3: 256000)
LIDAR_FRAME = "laser_frame"           # the LiDAR link in config/puzzlebot.urdf
LIDAR_TOPIC = "/laser_scan"
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
lidar_process = None


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


def start_lidar():
    global lidar_process
    existing = pids_of("sllidar_node")
    if existing:
        log(f"LiDAR driver already running (PID {existing}). Reusing it.")
        return
    if not os.path.exists(LIDAR_PORT):
        raise RuntimeError(f"{LIDAR_PORT} not found - is the LiDAR connected?")
    lidar_process = start(
        ["ros2", "run", "sllidar_ros2", "sllidar_node", "--ros-args",
         "-p", f"serial_port:={LIDAR_PORT}",
         "-p", f"serial_baudrate:={LIDAR_BAUD}",
         "-p", f"frame_id:={LIDAR_FRAME}",
         "-p", "angle_compensate:=true",
         "-r", f"scan:={LIDAR_TOPIC}"],
        "LiDAR driver")
    time.sleep(3)
    if lidar_process.poll() is not None:
        raise RuntimeError(f"LiDAR driver exited during startup ({lidar_process.returncode}); "
                           "check the port, the baud rate and the USB cable.")
    log(f"LiDAR running: {LIDAR_TOPIC} ({LIDAR_FRAME}).")


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
    """Stop everything at once (the Program Manager waits only a few seconds)."""
    log("Shutting down...")
    procs = [(p, label) for p, label in ((relay_process, "velocity relay"),
                                          (camera_process, "camera/EKF launch"),
                                          (lidar_process, "LiDAR driver"),
                                          (agent_process, "micro-ROS agent"))
             if p is not None and p.poll() is None]
    for sig, wait in ((signal.SIGINT, 3.0), (signal.SIGTERM, 1.5), (signal.SIGKILL, 1.0)):
        for p, label in procs:
            if p.poll() is None:
                if sig == signal.SIGINT:
                    log(f"Stopping {label}...")
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and any(p.poll() is None for p, _ in procs):
            time.sleep(0.1)
        procs = [(p, label) for p, label in procs if p.poll() is None]
        if not procs:
            break
    if agent_log:
        agent_log.close()
    log("Shutdown complete.")


def signal_handler(_signum, _frame):
    global running
    running = False


def main(argv=None):
    global env
    ap = argparse.ArgumentParser(description="PuzzleBot bringup on the Jetson")
    ap.add_argument("--no-camera", action="store_true",
                    help="do not start the camera/EKF launch (the Program Manager does)")
    ap.add_argument("--no-lidar", action="store_true", help="do not start the LiDAR")
    ap.add_argument("--no-relay", action="store_true", help="do not start the velocity relay")
    args, _ = ap.parse_known_args(argv)          # ignore --ros-args and the like
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        env = ros_env()
        start_agent()
        check_hackerboard()
        if ENABLE_LIDAR and not args.no_lidar:
            start_lidar()
        if ENABLE_CAMERA_LAUNCH and not args.no_camera:
            start_camera_launch()
        if ENABLE_RELAY and not args.no_relay:
            start_relay()
        log("Everything running. Drive with (laptop or another SSH terminal):")
        log("  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in")
        log("Press Ctrl+C to stop.")
        while running:
            for proc, label in ((agent_process, "micro-ROS agent"),
                                (lidar_process, "LiDAR driver"),
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