# control_forklift — forklift control (Tang Nano 9K lift + IR sensors)

Written by our forklift teammate; the same code runs on the Jetson (real Tang over
USB serial) and in his forklift bench simulation (a virtual Tang:
`main_ros/simulation/forklift_bench`).

| Command | What it does |
|---|---|
| `ros2 run main_ros forklift_controller` | owns the Tang's serial port, heartbeat, lift position checkpoint (start first) |
| `ros2 run main_ros forklift_operate` | ROS interface: `forklift/target_height_mm` in, `forklift/current_height_mm`, `forklift/state`, `forklift/ir1_distance_mm`, `forklift/ir2_distance_mm` out, service `forklift/stop` |
| `ros2 run main_ros forklift_pick_back --config <pick.json> --level floor\|first --execute --wait-for-ready` | supervised pick: on `"Ready Pick Up"` (std_msgs/String on `pick_back_real/command`) it confirms the box (camera + IR), aligns, inserts the forks, lifts, backs out; answers `Done` or `Error: <reason>` on `pick_back_real/result` |
| `ros2 run main_ros forklift_calibrate` | lift calibration by keys from an SSH terminal (the teammate's `calibrar`): marks the bottom and top of the travel and writes `calibracion_elevador.json` for the controller. Run it once on the robot before the first pick (with `forklift_controller` running) |

Files: `terminal.py` (lift calibration), `controller.py` (serial owner), `client.py` (its local socket client),
`link.py` + `protocol.py` (Tang UART frames), `calibration.py` (lift travel and
checkpoint), `operate.py` (ROS node), `pick_back.py` + `pick_geometry.py` (the pick).

The Program Manager calls the forklift through our `PickBox` / `SetDownBox` actions
(`warehouse_msgs`); a bridge from those actions to `pick_back` replaces
`forklift_placeholder` (next step). Requires `python3-serial` (pyserial).

## Firmware and the real robot's pick config

`firmware/nano/IR01.ino` (Arduino Nano: the two Sharp IR sensors, frame A9 to the
Tang) and `firmware/rtl/` (Tang Nano 9K Verilog: UART, command parser, quadrature
decoder, position controller, IR packet decoder) are the sources of the boards, as
the teammate delivered them (SimForklift_Real). `config/forklift/pick_real.example.json`
is the template of the real robot's pick config: fill the `null` values with the
PuzzleBot's measurements and save it as `pick_real.json` (axle frame: base_link).

## Deposit (SetDownBox)

From the teammate's physics test (pick, carry, deposit, all passing in Isaac): the
forks go down to the bottom, the box is left to settle, then the robot backs away
slowly and the forks slide out from under the pallet. `forklift_bridge` does the
first two (`set_down_height_mm` 0, `set_down_settle_s` 1.0); restocking backs out.
