import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist


# ==========================================
# ROBOT SPEED CONFIGURATION
# ==========================================

BASE_LINEAR_SPEED = 0.20       # m/s
BASE_ANGULAR_SPEED = 0.80      # rad/s

# Speed multiplier
SPEED_FACTOR = 1.0

MIN_SPEED_FACTOR = 0.25
MAX_SPEED_FACTOR = 3.0
SPEED_STEP = 0.25

# Safety timeout
DEADMAN_TIMEOUT = 0.7

# Command publishing frequency
PUBLISH_RATE = 20.0


def main():

    rclpy.init()

    node = rclpy.create_node("manual_teleop")

    publisher = node.create_publisher(
        Twist,
        "/cmd_vel",
        10
    )

    if not sys.stdin.isatty():
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError(
            "Manual teleop requires an interactive terminal"
        )

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # ==========================================
    # TELEOP STATE
    # ==========================================

    speed_factor = SPEED_FACTOR

    current_key = "x"
    last_key_time = 0.0

    linear_speed = BASE_LINEAR_SPEED * speed_factor
    angular_speed = BASE_ANGULAR_SPEED * speed_factor

    print("\n====================================")
    print("       PUZZLEBOT MANUAL TELEOP")
    print("====================================")

    print("\nMovement:")
    print("  W : Forward")
    print("  S : Backward")
    print("  A : Rotate left")
    print("  D : Rotate right")

    print("\nSpeed Control:")
    print("  H : Increase speed")
    print("  J : Decrease speed")

    print("\nOther:")
    print("  SPACE / X : Stop")
    print("  Q         : Quit")

    print("\n------------------------------------")
    print(f"Linear speed:  {linear_speed:.2f} m/s")
    print(f"Angular speed: {angular_speed:.2f} rad/s")
    print(f"Speed factor:  {speed_factor:.2f}x")
    print("------------------------------------")

    print("\nUse W/A/S/D to drive.")
    print("Commands stop automatically when keys stop arriving.\n")

    try:

        tty.setcbreak(fd)

        while rclpy.ok():

            # ==================================
            # READ KEYBOARD INPUT
            # ==================================

            if select.select(
                [sys.stdin],
                [],
                [],
                1.0 / PUBLISH_RATE
            )[0]:

                key = sys.stdin.read(1).lower()

                # ------------------------------
                # QUIT
                # ------------------------------

                if key == "q":
                    break

                # ------------------------------
                # INCREASE SPEED
                # ------------------------------

                elif key == "h":

                    speed_factor = min(
                        speed_factor + SPEED_STEP,
                        MAX_SPEED_FACTOR
                    )

                    linear_speed = (
                        BASE_LINEAR_SPEED * speed_factor
                    )

                    angular_speed = (
                        BASE_ANGULAR_SPEED * speed_factor
                    )

                    print(
                        f"\n[+] SPEED INCREASED | "
                        f"Linear: {linear_speed:.2f} m/s | "
                        f"Angular: {angular_speed:.2f} rad/s | "
                        f"Factor: {speed_factor:.2f}x"
                    )

                # ------------------------------
                # DECREASE SPEED
                # ------------------------------

                elif key == "j":

                    speed_factor = max(
                        speed_factor - SPEED_STEP,
                        MIN_SPEED_FACTOR
                    )

                    linear_speed = (
                        BASE_LINEAR_SPEED * speed_factor
                    )

                    angular_speed = (
                        BASE_ANGULAR_SPEED * speed_factor
                    )

                    print(
                        f"\n[-] SPEED DECREASED | "
                        f"Linear: {linear_speed:.2f} m/s | "
                        f"Angular: {angular_speed:.2f} rad/s | "
                        f"Factor: {speed_factor:.2f}x"
                    )

                # ------------------------------
                # MOVEMENT KEYS
                # ------------------------------

                elif key in ("w", "a", "s", "d", "x", " "):

                    current_key = key

                    last_key_time = time.monotonic()

            # ==================================
            # CREATE VELOCITY COMMAND
            # ==================================

            msg = Twist()

            # Deadman safety:
            # Stop if movement key is not refreshed.
            if (
                time.monotonic() - last_key_time
                < DEADMAN_TIMEOUT
            ):

                # Forward
                if current_key == "w":

                    msg.linear.x = linear_speed

                # Backward
                elif current_key == "s":

                    msg.linear.x = -linear_speed

                # Rotate left
                elif current_key == "a":

                    msg.angular.z = angular_speed

                # Rotate right
                elif current_key == "d":

                    msg.angular.z = -angular_speed

                # Stop
                elif current_key in ("x", " "):

                    msg.linear.x = 0.0
                    msg.angular.z = 0.0

            # ==================================
            # PUBLISH CMD_VEL
            # ==================================

            publisher.publish(msg)

    except KeyboardInterrupt:
        pass

    finally:

        # ======================================
        # STOP ROBOT ON EXIT
        # ======================================

        publisher.publish(Twist())

        time.sleep(0.1)

        termios.tcsetattr(
            fd,
            termios.TCSADRAIN,
            old_settings
        )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print("\nTeleop stopped. Robot commanded to stop.")


if __name__ == "__main__":
    main()