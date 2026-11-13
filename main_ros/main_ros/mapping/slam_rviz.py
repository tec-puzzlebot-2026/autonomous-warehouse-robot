import os
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def main():
    # Find the RViz configuration installed with the ROS 2 package.
    package_share = Path(get_package_share_directory("main_ros"))
    rviz_config = package_share / "rviz" / "slam.rviz"

    # Existing Fast DDS configuration on your laptop.
    fastdds_config = Path.home() / "fastdds_spark.xml"

    if not rviz_config.is_file():
        raise FileNotFoundError(
            f"RViz configuration not found: {rviz_config}"
        )

    if not fastdds_config.is_file():
        raise FileNotFoundError(
            f"Fast DDS configuration not found: {fastdds_config}"
        )

    # Configure ROS 2 connectivity for RViz.
    env = os.environ.copy()

    env["ROS_DOMAIN_ID"] = "0"
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    env["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(fastdds_config)

    env.pop("ROS_LOCALHOST_ONLY", None)
    env.pop("ROS_AUTOMATIC_DISCOVERY_RANGE", None)

    print("=" * 45, flush=True)
    print("PUZZLEBOT — SLAM RVIZ", flush=True)
    print("=" * 45, flush=True)
    print(f"Configuration: {rviz_config}", flush=True)
    print("LiDAR topic: /slam/scan", flush=True)
    print("Connecting to Spark using Fast DDS...", flush=True)

    # Launch RViz with the saved configuration.
    try:
        subprocess.run(
            [
                "rviz2",
                "-d",
                str(rviz_config),
                "--ros-args",
                "-p",
                "use_sim_time:=true",
            ],
            env=env,
            check=True,
        )

    except KeyboardInterrupt:
        pass

    except FileNotFoundError:
        print("ERROR: rviz2 is not installed.")

    except subprocess.CalledProcessError as exc:
        print(f"RViz exited with error code {exc.returncode}.")


if __name__ == "__main__":
    main()