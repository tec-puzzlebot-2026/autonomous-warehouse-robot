#!/usr/bin/env python3
"""PuzzleBot host-side SSH, micro-ROS, and teleoperation manager."""
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

JETSON_HOST = "puzzlebot@jetson.local"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "StrictHostKeyChecking=accept-new"]

HOST_ROS_SETUP = "/home/chrisrvt/autonomous-warehouse-robot/install/setup.bash"
# HOST_ROS_SETUP = "/home/goma/ros2_ws/puzzlebot/autonomous-warehouse-robot/install/setup.bash"

HOST_WORKSPACE_SETUP = os.path.expanduser(
    "~/autonomous-warehouse-robot/install/setup.bash"
)
# HOST_WORKSPACE_SETUP = os.path.expanduser(
#     "~/ros2_ws/puzzlebot/autonomous-warehouse-robot/install/setup.bash"
# )

JETSON_ROS_SETUP = "/opt/ros/humble/setup.bash"
JETSON_WORKSPACE_SETUP = "/home/puzzlebot/ros2_packages_ws/install/setup.bash"
JETSON_SERIAL_PORT = "/dev/ttyUSB0"
JETSON_BAUD_RATE = 115200

# PLACEHOLDERS. Confirm physical dimensions and firmware command units first.
WHEEL_RADIUS = 0.05
WHEEL_SEPARATION = 0.20
MAX_WHEEL_SPEED = 2.0
CMD_TIMEOUT = 2.0
ENABLE_MOTOR_CONTROL = True

running = True
remote_process = None
controller_process = None
teleop_process = None
agent_started_by_us = False
agent_pid = None


def log(message):
    print(f"[DEBUG TELEOP] {message}", flush=True)


def ssh_run(command, timeout=15):
    return subprocess.run(
        ["ssh", "-T", *SSH_OPTIONS, JETSON_HOST, command],
        capture_output=True, text=True, timeout=timeout
    )


def check_ssh():
    log("Checking Jetson SSH connection...")
    result = ssh_run("printf 'JETSON_CONNECTED\\n'")
    if result.returncode != 0 or "JETSON_CONNECTED" not in result.stdout:
        raise RuntimeError(
            f"SSH failed (exit={result.returncode}): {result.stderr.strip()}"
        )
    log("Jetson connected successfully.")


def get_agent_pids():
    """Read executable PIDs; do not confuse the 'ros2 run' wrapper with agent."""
    result = ssh_run("pgrep -x micro_ros_agent || true")
    if result.returncode != 0:
        raise RuntimeError(f"Agent process lookup failed: {result.stderr.strip()}")
    return [int(x) for x in result.stdout.split() if x.isdigit()]


def build_remote_command():
    # exec replaces the remote bash with the ROS CLI. The CLI may launch
    # a child micro_ros_agent executable; we discover and record that PID.
    script = (
        "set -e; "
        f"source {shlex.quote(JETSON_ROS_SETUP)}; "
        f"source {shlex.quote(JETSON_WORKSPACE_SETUP)}; "
        'echo "[JETSON] Starting micro-ROS agent..."; '
        "exec ros2 run micro_ros_agent micro_ros_agent "
        f"serial --dev {shlex.quote(JETSON_SERIAL_PORT)} -b {JETSON_BAUD_RATE} -m 512"
    )
    return "bash -c " + shlex.quote(script)


def start_jetson():
    global remote_process, agent_started_by_us, agent_pid
    existing = get_agent_pids()
    if existing:
        log(f"Existing micro-ROS agent PID(s): {existing}. Reusing existing agent.")
        return

    log("Starting micro-ROS agent on Jetson...")
    remote_process = subprocess.Popen(
        ["ssh", "-tt", *SSH_OPTIONS, JETSON_HOST, build_remote_command()],
        start_new_session=True
    )
    agent_started_by_us = True
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and running:
        if remote_process.poll() is not None:
            raise RuntimeError(
                f"Remote agent SSH session exited ({remote_process.returncode})."
            )
        found = get_agent_pids()
        if found:
            agent_pid = found[0]
            log(f"micro-ROS agent started (Jetson PID {agent_pid}).")
            return
        time.sleep(0.5)
    raise RuntimeError("micro-ROS agent was not detected within 15 seconds.")


def host_ros_env():
    """Obtain ROS environment without depending on a login shell."""
    cmd = (
        f"source {shlex.quote(HOST_ROS_SETUP)} && "
        f"source {shlex.quote(HOST_WORKSPACE_SETUP)} && env -0"
    )
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(
            "Cannot source host ROS setup: " +
            result.stderr.decode(errors="replace").strip()
        )
    env = os.environ.copy()
    for entry in result.stdout.split(b"\0"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def check_hackerboard(timeout=25):
    """Confirm both motor-command subscriptions are visible from host DDS."""
    log("Waiting for Hackerboard motor subscriptions...")
    env = host_ros_env()
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline and running:
        ready = True
        for topic in ("/VelocitySetL", "/VelocitySetR"):
            try:
                result = subprocess.run(
                    ["ros2", "topic", "info", topic],
                    env=env, capture_output=True, text=True, timeout=6
                )
            except subprocess.TimeoutExpired:
                ready = False
                last[topic] = "ros2 topic info timed out"
                continue
            last[topic] = result.stdout.strip() or result.stderr.strip()
            count = 0
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.strip().startswith("Subscription count:"):
                        try:
                            count = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
            if count < 1:
                ready = False
        if ready:
            log("Both motor topics have subscribers visible from host.")
            return
        time.sleep(1)
    raise RuntimeError(
        "Hackerboard subscriptions not visible from host. "
        "Check ROS_DOMAIN_ID, ROS_LOCALHOST_ONLY, DDS discovery and agent.\n"
        + "\n".join(f"{topic}: {detail}" for topic, detail in last.items())
    )


def start_controller():
    global controller_process
    log("Starting differential-drive controller...")
    cmd = (
        f"source {shlex.quote(HOST_ROS_SETUP)} && "
        f"source {shlex.quote(HOST_WORKSPACE_SETUP)} && "
        "exec ros2 run main_ros differential_drive --ros-args "
        f"-p wheel_radius:={WHEEL_RADIUS} "
        f"-p wheel_separation:={WHEEL_SEPARATION} "
        f"-p max_wheel_speed:={MAX_WHEEL_SPEED} "
        f"-p cmd_timeout:={CMD_TIMEOUT}"
    )
    controller_process = subprocess.Popen(
        ["bash", "-c", cmd], start_new_session=True
    )
    time.sleep(1)
    if controller_process.poll() is not None:
        raise RuntimeError("Differential-drive controller exited during startup.")
    log("Differential-drive controller running.")


def start_teleop():
    global teleop_process
    if not shutil.which("gnome-terminal"):
        raise RuntimeError("gnome-terminal is not installed.")
    log("Opening keyboard teleoperation terminal...")
    cmd = (
        f"source {shlex.quote(HOST_ROS_SETUP)} && "
        f"source {shlex.quote(HOST_WORKSPACE_SETUP)} && "
        "echo 'PUZZLEBOT TELEOP — Ctrl+C to exit' && "
        "exec ros2 run teleop_twist_keyboard teleop_twist_keyboard "
        "--ros-args -r cmd_vel:=/cmd_vel"
    )
    teleop_process = subprocess.Popen(
        ["gnome-terminal", "--wait", "--title=PuzzleBot Teleoperation",
         "--", "bash", "-c", cmd],
        start_new_session=True
    )
    time.sleep(1)
    if teleop_process.poll() is not None:
        raise RuntimeError("Keyboard teleop terminal exited during startup.")
    log("Keyboard teleoperation terminal opened.")


def stop_process(proc, label):
    if proc is None or proc.poll() is not None:
        return
    log(f"Stopping {label}...")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=3)
    except ProcessLookupError:
        pass


def stop_remote_agent():
    """Only signal the exact agent PID recorded at startup."""
    global agent_started_by_us
    if not agent_started_by_us or agent_pid is None:
        return
    log(f"Stopping our Jetson agent PID {agent_pid}...")
    try:
        # Verify this PID still belongs to micro_ros_agent before signalling.
        script = (
            f"if [ \"$(cat /proc/{agent_pid}/comm 2>/dev/null)\" = "
            f"\"micro_ros_agent\" ]; then kill -INT {agent_pid}; fi"
        )
        result = ssh_run(script, timeout=8)
        if result.returncode != 0:
            log(f"Remote agent stop warning: {result.stderr.strip()}")
    except Exception as exc:
        log(f"Remote agent stop warning: {exc}")
    agent_started_by_us = False


def cleanup():
    log("Shutting down...")
    stop_process(teleop_process, "keyboard teleoperation")
    stop_process(controller_process, "differential-drive controller")
    stop_remote_agent()
    stop_process(remote_process, "Jetson SSH session")
    log("Shutdown complete.")


def signal_handler(_signum, _frame):
    global running
    running = False


def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        check_ssh()
        start_jetson()
        check_hackerboard()
        if not ENABLE_MOTOR_CONTROL:
            log("SSH, agent, and motor topic checks passed.")
            log("Motor control DISABLED: wheel parameters and firmware units unverified.")
            return 0
        start_controller()
        start_teleop()
        log("Teleoperation running. Press Ctrl+C to stop.")
        while running:
            if remote_process is not None and remote_process.poll() is not None:
                raise RuntimeError("Remote micro-ROS SSH session exited.")
            if controller_process is not None and controller_process.poll() is not None:
                raise RuntimeError("Differential-drive controller exited.")
            if teleop_process is not None and teleop_process.poll() is not None:
                log("Teleoperation terminal closed.")
                break
            time.sleep(0.5)
        return 0
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
