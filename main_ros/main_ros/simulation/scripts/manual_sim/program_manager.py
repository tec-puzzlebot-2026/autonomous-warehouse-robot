import os
import signal
import subprocess
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist


TELEOP_SESSION = "ray_teleop"


class ProgramManager(Node):

    def __init__(self):
        super().__init__("program_manager")

        self.processes = []
        self.teleop_started = False

        self.stop_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        try:
            # Check for an existing teleop session.
            existing = subprocess.run(
                [
                    "tmux",
                    "has-session",
                    "-t",
                    f"={TELEOP_SESSION}"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )

            if existing.returncode == 0:
                raise RuntimeError(
                    "A ray_teleop tmux session already exists. "
                    "Stop the previous launch first."
                )

            # ======================================
            # START MANUAL TELEOP
            # ======================================

            subprocess.run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    TELEOP_SESSION,
                    "ros2 run main_ros manual_teleop"
                ],
                check=True
            )

            self.teleop_started = True

            self.get_logger().info(
                "Teleop ready: "
                "docker exec -it ray-puzzlebot-ros2 "
                "tmux attach -t ray_teleop"
            )

            # ======================================
            # START WHEEL ODOMETRY
            # ======================================

            self.start(
                "wheel_odometry",
                [
                    "ros2",
                    "run",
                    "main_ros",
                    "wheel_odometry",
                    "--ros-args",
                    "-p",
                    "use_sim_time:=true"
                ]
            )

            # ======================================
            # START CUSTOM SLAM
            # ======================================

            self.start(
                "slam",
                [
                    "ros2",
                    "run",
                    "main_ros",
                    "slam",
                    "--ros-args",
                    "-p",
                    "use_sim_time:=true"
                ]
            )

        except Exception:
            self.stop_programs()
            raise

    # ==============================================
    # PROCESS MANAGEMENT
    # ==============================================

    def start(self, label, command):

        process = subprocess.Popen(
            command,
            start_new_session=True
        )

        self.processes.append(
            (label, process)
        )

        self.get_logger().info(
            f"Started {label} (PID {process.pid})"
        )

    def stop_programs(self):

        # Stop keyboard teleop.
        if self.teleop_started:

            subprocess.run(
                [
                    "tmux",
                    "kill-session",
                    "-t",
                    f"={TELEOP_SESSION}"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )

            self.teleop_started = False

        # Stop wheel odometry and SLAM.
        for _, process in self.processes:

            if process.poll() is None:

                try:
                    os.killpg(
                        process.pid,
                        signal.SIGINT
                    )

                except ProcessLookupError:
                    pass

        for _, process in self.processes:

            try:
                process.wait(timeout=3.0)

            except subprocess.TimeoutExpired:

                if process.poll() is None:
                    os.killpg(
                        process.pid,
                        signal.SIGTERM
                    )

        # Send stop commands to the robot.
        for _ in range(3):

            try:
                self.stop_pub.publish(Twist())

            except Exception:
                break

            time.sleep(0.05)


def main(args=None):

    rclpy.init(args=args)

    manager = None

    try:
        manager = ProgramManager()
        rclpy.spin(manager)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:

        if manager is not None:
            manager.stop_programs()
            manager.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()