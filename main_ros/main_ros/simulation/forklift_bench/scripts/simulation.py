"""Run a warehouse and publish ROS 2 telemetry. RGB uses Isaac Replicator."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import sys

sys.dont_write_bytecode = True

WIDTH, HEIGHT = 320, 240
ROOT = Path(__file__).resolve().parents[1]


def options():
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=Path, default=ROOT/"worlds/floor.usda")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-rgb", action="store_true")
    p.add_argument("--cpu-rgb", action="store_true", help="Render the E80 bench geometry in CPU.")
    p.add_argument("--state-hz", type=float, default=20.)
    p.add_argument("--range-hz", type=float, default=5.)
    p.add_argument("--camera-hz", type=float, default=5.)
    p.add_argument("--camera-count", type=int, help="Publish RGB for the first N sets. Default: all.")
    p.add_argument("--duration", type=float, help="Stop after this many simulated seconds.")
    p.add_argument("--firmware", choices=("stable-model", "dual"))
    p.add_argument("--firmware-local", action="store_true", help="Run operar inside Kit using the same controller request contract.")
    p.add_argument("--motor-config", type=Path, default=ROOT/"config/motor.json", help="Measured motor response parameters in JSON.")
    p.add_argument("--calibration", type=Path, default=ROOT/"config/calibration.json")
    p.add_argument("--run-dir", type=Path, help="Fresh directory for the PTY and calibration checkpoint.")
    p.add_argument("--pick-config", type=Path)
    p.add_argument("--pick-level", choices=("floor", "first"), default="floor")
    p.add_argument("--execute-pick", action="store_true")
    p.add_argument("--wait-for-ready", action="store_true",
                   help="Wait for Ready Pick Up on the ROS command topic.")
    return p


def serve(stage, args, app=None):
    from pxr import UsdGeom
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor, TimeoutException
    from physics import DT, Physics
    from firmware_lift import FirmwareLift
    from rclpy.impl.implementation_singleton import rclpy_implementation
    from ros_interface import Interface
    stage.SetEditTarget(stage.GetSessionLayer())
    if any(not math.isfinite(rate) or not 0 < rate <= 120 for rate in
           (args.state_hz, args.range_hz, args.camera_hz)):
        raise ValueError("Rates must be in (0, 120] Hz")
    if args.camera_count is not None and args.camera_count < 1:
        raise ValueError("--camera-count must be positive")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("--duration must be positive and finite")
    state_period, range_period, camera_period = (
        max(1, round(1/(DT*rate))) for rate in (args.state_hz, args.range_hz, args.camera_hz)
    )
    if args.firmware_local and not args.firmware:
        raise ValueError("--firmware-local requiere --firmware")
    if args.pick_config and (not args.firmware_local or args.no_rgb):
        raise ValueError("--pick-config requiere --firmware-local y cámara RGB")
    if args.execute_pick and not args.pick_config:
        raise ValueError("--execute-pick requiere --pick-config")
    if args.wait_for_ready and not args.execute_pick:
        raise ValueError("--wait-for-ready requiere --execute-pick")
    if args.firmware and args.run_dir is None:
        raise ValueError("--firmware requiere --run-dir")
    if args.firmware:
        # control_forklift lives in main_ros/manipulation (the same code as on the robot).
        sys.path.insert(0, str(ROOT.parents[1] / "manipulation"))
        # pyserial is pure Python. Keep Kit's ROS and native extensions ahead of system modules.
        sys.path.append("/usr/lib/python3/dist-packages")
    sim = Physics(stage)
    ros = None
    executor = None
    rgb = {}
    cpu_cameras = {}
    firmware = None
    pick = None
    pick_result_at = None
    try:
        rclpy.init(args=[])
        indices = [] if args.no_rgb else sorted(sim.rows)[:args.camera_count]
        ros = Interface(sim, range_period * DT, indices, firmware=bool(args.firmware))
        executor = SingleThreadedExecutor()
        executor.add_node(ros)
        if args.firmware:
            firmware = FirmwareLift(sim, args.firmware, args.calibration, args.run_dir.resolve(),
                                    args.firmware_local, args.motor_config)
        if args.pick_config:
            from control_forklift.pick_back import PhysicalPickBack
            from control_forklift.pick_geometry import PickConfig
            config = PickConfig.load(args.pick_config)
            level = config.floor_level_mm if args.pick_level == "floor" else config.first_level_mm
            support = stage.GetPrimAtPath("/World/Sets/Set_00").GetAttribute("supportHeight").Get()
            if abs(1000 * support - level) > 1:
                raise ValueError("El nivel de Pick & Back no coincide con la repisa de la escena")
            row = sim.load_rows[0]
            initial_cargo = sim.loads.get_transforms()[row, :3].copy()

            def cargo_motion():
                delta = sim.loads.get_transforms()[row, :3] - initial_cargo
                return 1000 * float(delta[2]), 1000 * math.hypot(float(delta[0]), float(delta[1]))

            def check_cargo():
                raised, moved = cargo_motion()
                if raised < 12 or moved < 100:
                    raise ValueError("La carga física no coincide con la confirmación de los sensores")

            pick = PhysicalPickBack(config, level, args.execute_pick, ros_args=[
                "--ros-args", "-r", "__ns:=/puzzlebot_00", "-r",
                "/cmd_vel_in:=/puzzlebot_00/cmd_vel", "-p", "use_sim_time:=true",
            ], transform_provider=ros.pick_transform, wait_for_ready=args.wait_for_ready,
                completion_check=check_cargo)
            executor.add_node(pick)
        intrinsics = {}
        if not args.no_rgb:
            if args.cpu_rgb:
                from cpu_camera import CpuCamera
            else:
                import omni.replicator.core as rep
                rep.orchestrator.set_capture_on_play(False)
            for index in indices:
                path = sim.robot_path(index) + "/camera_link/rgb_camera"
                camera = UsdGeom.Camera.Get(stage, path)
                if not camera:
                    raise RuntimeError(f"Missing camera {path}")
                focal_length = camera.GetFocalLengthAttr().Get()
                h_ap = camera.GetHorizontalApertureAttr().Get()
                v_ap = camera.GetVerticalApertureAttr().Get()
                intrinsics[index] = (WIDTH * focal_length / h_ap, HEIGHT * focal_length / v_ap,
                                     WIDTH * .5, HEIGHT * .5)
                if args.cpu_rgb:
                    cpu_cameras[index] = CpuCamera(sim, index)
                else:
                    product = rep.create.render_product(path, (WIDTH, HEIGHT))
                    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                    annotator.attach([product])
                    rgb[index] = (product, annotator)
        print(f"ROS READY sets={sim.count} domain={os.environ.get('ROS_DOMAIN_ID')} cameras={len(indices)}", flush=True)
        first_frames = set(indices)
        captures = 0
        start = time.monotonic()
        while rclpy.ok() and (app is None or app.is_running()):
            if args.duration is not None and sim.time >= args.duration:
                break
            for _ in range(3*sim.count):
                try:
                    task, _, _ = executor.wait_for_ready_callbacks(timeout_sec=0)
                except TimeoutException:
                    break
                task()
                task.result()
            capture = bool(indices) and (sim.step_count+1) % camera_period == 0
            refresh = app is not None and (sim.step_count+1) % 4 == 0
            if firmware:
                firmware.before_step(DT, executor)
            sim.tick(update_usd=capture or refresh, firmware_lift=firmware is not None)
            if capture or sim.step_count % state_period == 0:
                ros.publish_state()
            if sim.step_count % range_period == 0:
                ros.publish_ranges()
            if capture:
                if cpu_cameras:
                    frames = {index: camera.render(WIDTH, HEIGHT, intrinsics[index])
                              for index, camera in cpu_cameras.items()}
                else:
                    rep.orchestrator.step(rt_subframes=1, pause_timeline=True, delta_time=0.0)
                    frames = {index: annotator.get_data() for index, (_, annotator) in rgb.items()}
                first_frames.difference_update(ros.publish_rgb(frames, WIDTH, HEIGHT, intrinsics))
                captures += 1
                if captures >= 10 and first_frames:
                    raise RuntimeError(f"No se obtuvieron imágenes RGB de los sets {sorted(first_frames)}")
            elif refresh:
                app.update()
            if pick:
                if pick.phase == "esperando_ready":
                    initial_cargo = sim.loads.get_transforms()[row, :3].copy()
                pick.tick()
                if pick.finished:
                    if pick_result_at is None:
                        pick_result_at = time.monotonic()
                    if time.monotonic() - pick_result_at > .4 and (
                            pick.stop_future is None or pick.stop_future.done() or
                            time.monotonic() - pick_result_at >= 2):
                        if pick.phase == "abortado":
                            raise RuntimeError(pick.detail)
                        raised, moved = cargo_motion()
                        print("PICK RESULT " + json.dumps({"phase": pick.phase, "level_mm": level,
                              "cargo_lift_mm": round(raised, 2), "cargo_travel_mm": round(moved, 2),
                              "return_error_mm": round(1000*pick.travel, 2)}), flush=True)
                        break
            wait = start + sim.time - time.monotonic()
            if wait > 0:
                time.sleep(wait)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    except rclpy_implementation.RCLError:
        if rclpy.ok():
            raise
    else:
        if pick and not pick.finished and rclpy.ok():
            reason = f"El banco terminó durante {pick.phase}: {pick.detail}"
            pick.abort(reason)
            raise RuntimeError(reason)
    finally:
        try:
            for product, annotator in rgb.values():
                annotator.detach([product])
                product.destroy()
        finally:
            if pick is not None:
                if rclpy.ok():
                    if pick.finished:
                        pick.drive_command()
                    else:
                        pick.abort("fin de la simulación")
                executor.remove_node(pick)
                pick.destroy_node()
            if firmware is not None:
                firmware.close(executor)
            if executor is not None:
                executor.shutdown()
            if ros is not None:
                ros.destroy_node()
            rclpy.try_shutdown()
            sim.close()


def main():
    args = options().parse_args()
    if not args.world.is_file():
        raise FileNotFoundError(args.world)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "sync_loads": True, "multi_gpu": False})
    try:
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension
        import carb
        enable_extension("omni.physx.tensors")
        if not args.no_rgb:
            enable_extension("omni.replicator.core")
        app.update()
        # Direct scripted stepping owns the timebase, PhysX and all ROS timestamps.
        carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        context = omni.usd.get_context()
        if not context.open_stage(str(args.world.resolve())):
            raise RuntimeError(f"Could not open {args.world}")
        app.update()
        if not args.headless:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=[3.2,-4.4,3.2], target=[0.0,0.0,0.0])
        serve(context.get_stage(), args, app)
    finally:
        app.close()


if __name__ == "__main__":
    main()
