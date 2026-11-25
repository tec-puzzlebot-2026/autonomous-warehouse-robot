# control_forklift — forklift control (Tang Nano 9K lift + IR sensors)

Written by our forklift teammate; the same code runs on the Jetson (real Tang over
USB serial) and in his forklift bench simulation (a virtual Tang:
`main_ros/simulation/forklift_bench`).

| Command | What it does |
|---|---|
| `ros2 run main_ros forklift_controller` | owns the Tang's serial port, heartbeat, lift position checkpoint (start first) |
| `ros2 run main_ros forklift_operate` | ROS interface: `forklift/target_height_mm` in, `forklift/current_height_mm`, `forklift/state`, `forklift/ir1_distance_mm`, `forklift/ir2_distance_mm` out, service `forklift/stop` |
| `ros2 run main_ros forklift_pick_back --config <pick.json> --level floor\|first --execute --wait-for-ready` | supervised pick: on `"Ready Pick Up"` (std_msgs/String on `pick_back_real/command`) it confirms the box (camera + IR), aligns, inserts the forks, lifts, backs out; answers `Done` or `Error: <reason>` on `pick_back_real/result` |

Files: `controller.py` (serial owner), `client.py` (its local socket client),
`link.py` + `protocol.py` (Tang UART frames), `calibration.py` (lift travel and
checkpoint), `operate.py` (ROS node), `pick_back.py` + `pick_geometry.py` (the pick).

The Program Manager calls the forklift through our `PickBox` / `SetDownBox` actions
(`warehouse_msgs`); a bridge from those actions to `pick_back` replaces
`forklift_placeholder` (next step). Requires `python3-serial` (pyserial).
