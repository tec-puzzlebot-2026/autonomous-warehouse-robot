# forklift_bench — the forklift teammate's Isaac Sim bench

A small bench (floor, one two-level rack, one box, the forklift PuzzleBot) to test the
pick in isolation. Isaac runs headless (CPU) or with `--gui`; the lift motor and the
IR sensors run on a virtual Tang Nano, so `control_forklift`
(`main_ros/manipulation/control_forklift`) runs exactly as on the robot.

```bash
export ISAAC_SIM_ROOT=<your Isaac Sim 5.0 folder>
bash run.sh floor --start            # floor pick, starts right away
bash run.sh first                    # 20 mm shelf, waits for "Ready Pick Up" (ROS domain 74)
bash run.sh upper --gui              # 200 mm shelf, no automatic pick, with a window
```

Scenes: `worlds/floor.usda`, `first.usda` (20 mm shelf), `upper.usda` (200 mm shelf).
Assets (shared with our warehouse): `../assets/forklift/` —
`puzzlebot_forklift.usdc` (robot with forklift, IR sensors, camera),
`cargo.usdc` (box on pallet runners), `rack_two_levels.usda`, `rack_with_boxes.usda`.
`config/`: the virtual Tang (IR table, motor model, lift calibration) and the bench's
`pick.json` (robot geometry, topics under `/puzzlebot_00`).
