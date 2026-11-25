"""Connect the simulated lift to the Jetson package through a virtual UART."""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
import os

from tang import Nano, TangProtocolModel, VirtualSerial, ir_tables


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "config" / "ir_calibration.vh"


@dataclass(frozen=True)
class MotorParameters:
    full_profile_speed_mps: float = .070
    powered_time_constant_s: float = .065
    coast_time_constant_s: float = .035

    @classmethod
    def load(cls, source):
        result = cls(**json.loads(source.read_text())) if source else cls()
        if not all(math.isfinite(x) and 0 < x <= lim for x, lim in (
                (result.full_profile_speed_mps, .25),
                (result.powered_time_constant_s, 1.),
                (result.coast_time_constant_s, 1.))):
            raise ValueError("Parámetros del motor fuera del intervalo del banco")
        return result


class LocalControllerClient:
    """Exercise the same controller and client codecs when Unix sockets are unavailable."""
    def __new__(cls, controller):
        from control_forklift.calibration import Calibration
        from control_forklift.client import ControllerClient

        class Client(ControllerClient):
            def __init__(self):
                self.client_id = "isaac-local"
                result = self._call("acquire", port=controller.port,
                                    calibration=str(controller.path), mode="operar")
                self.sequence = result["sequence"]
                self.calibration = Calibration(**result["calibration"])

            def _call(self, action, **kwargs):
                return controller.request({"action": action, **kwargs}, self.client_id)

            def close(self):
                if controller.owner == self.client_id:
                    self._call("release")

        return Client()


class FirmwareLift:
    def __init__(self, physics, profile, calibration_path, run_dir, local=False, motor_config=None):
        from control_forklift.calibration import load, save
        if sorted(physics.rows) != [0]:
            raise ValueError("El modo firmware requiere un mundo de una estación con Set_00")
        self.sim = physics
        self.motor = MotorParameters.load(motor_config)
        self.profile = profile
        self.row = physics.rows[0]
        self.calibration = load(calibration_path)
        if self.calibration.schema_version != 5:
            raise ValueError("El banco requiere calibración v5 con puntos medidos")
        if not 0 < self.calibration.travel_mm <= 254:
            raise ValueError("La carrera de la calibración excede el recorrido mecánico de Isaac")
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lease = run_dir / "connection.json"
        self.calibration_path = run_dir / "calibracion_sim.json"
        if self.lease.exists() or self.calibration_path.exists():
            raise ValueError("Usa un directorio de ejecución nuevo para evitar mezclar sesiones de calibración")
        # Start the virtual mechanism at B. The source checkpoint belongs to another physical run.
        save(self.calibration_path, replace(self.calibration,
             operational_bottom_counts=0, session_bottom_counts=0, last_raw_counts=0))
        self.nano = Nano(ir_tables(TABLE)) if profile == "dual" else None
        self.board = TangProtocolModel(self.nano.tables if self.nano else None)
        self.port = VirtualSerial()
        self.local = local
        self.controller = self.node = None
        self.last_heartbeat = -math.inf
        self.encoder_frozen = False
        self.motor_blocked = False
        self.nano_connected = True
        self.heartbeat_enabled = True
        self.raw = self.pwm = 0
        self.raw_origin = 0
        self.shaft_height = max(0., self.measured_height())
        self.shaft_speed = 0.
        self.limit_m = self.calibration.travel_mm / 1000
        self.socket_path = run_dir / "controller.sock"
        if not self.local and len(os.fsencode(self.socket_path)) >= 104:
            self.port.close()
            raise ValueError("Usa un directorio de ejecución más corto para el socket Unix")
        try:
            if local:
                from control_forklift.controller import Controller
                self.controller = Controller(self.port.port, str(self.calibration_path))
            self.lease.write_text(json.dumps({
                "port": self.port.port, "calibration": str(self.calibration_path),
                "socket": str(self.socket_path), "domain": int(os.environ.get("ROS_DOMAIN_ID", "73")),
                "namespace": "/puzzlebot_00", "profile": profile, "local": local,
                "isaac_pid": os.getpid(),
            }) + "\n", encoding="utf-8")
        except BaseException:
            if self.controller:
                self.controller.link.close()
            self.port.close()
            raise
        print(f"FIRMWARE READY {self.lease} ({profile}, {'local' if local else 'socket Unix'})", flush=True)

    def measured_height(self):
        positions = self.sim.robots.get_dof_positions()
        return float(sum(positions[self.row, self.sim.slides]))

    def reboot_board(self):
        if hasattr(self.board, "reboot"):
            self.board.reboot()
        else:
            self.board = TangProtocolModel(self.nano.tables if self.nano else None)
        self.raw_origin += self.raw
        self.raw = self.pwm = 0

    def read_ir(self):
        from pxr import Gf
        from physics import rotation
        result = []
        poses = self.sim.robots.get_link_transforms()
        for side in ("left", "right"):
            sensor = f"forklift_ir_{side}_optical_frame"
            if sensor not in self.sim.links:
                sensor = f"forklift_ir_{side}_link"
            pose = self.sim.link_pose(0, sensor, poses)
            beam = rotation(pose).TransformDir(Gf.Vec3d(1, 0, 0))
            result.append(self.sim.range_ray(0, pose[:3], beam, .85, robust=True))
        return result

    def before_step(self, dt, executor):
        from control_forklift.protocol import simple
        if self.controller is not None:
            self.controller.tick()
            if self.heartbeat_enabled and self.sim.time - self.last_heartbeat >= .05 - 1e-9:
                self.controller._send(simple("H"))
                self.last_heartbeat = self.sim.time
            if self.node is None and self.controller.token_acknowledged and self.controller.protocol_current:
                from control_forklift.operate import ForkliftNode
                self.node = ForkliftNode(self.port.port, str(self.calibration_path),
                                         client_factory=lambda *_: LocalControllerClient(self.controller),
                                         namespace="/puzzlebot_00")
                executor.add_node(self.node)
        if not self.encoder_frozen:
            direction = 1 if self.calibration.signed_span_counts > 0 else -1
            height_mm = min(self.calibration.travel_mm, max(0., self.measured_height() * 1000))
            self.raw = direction * self.calibration.mm_to_counts(height_mm) - self.raw_origin
        self.board.feed(self.port.read())
        frame = self.nano.step(self.sim.time, self.read_ir) if self.nano and self.nano_connected else None
        self.pwm, data = self.board.step(self.sim.time, self.raw, frame)
        self.port.write(data)

        # Average DC-motor response. The existing articulations retain contact physics.
        # Use measured displacement for the encoder, independently of motor command.
        if self.motor_blocked:
            self.shaft_speed = 0.
        else:
            speed = self.pwm * self.motor.full_profile_speed_mps / .75
            tau = self.motor.powered_time_constant_s if self.pwm else self.motor.coast_time_constant_s
            self.shaft_speed += (speed - self.shaft_speed) * (1 - math.exp(-dt / tau))
        self.shaft_height = min(self.limit_m, max(0., self.shaft_height + self.shaft_speed * dt))
        # The travel window is enforced by the board, the modeled gearbox and the USD limits.
        # The legacy meter command is disconnected while this adapter is active.
        self.sim.lift[self.row] = self.shaft_height

    def close(self, executor):
        if self.node:
            executor.remove_node(self.node)
            self.node.destroy_node()
        if self.controller:
            self.controller.checkpoint()
            self.controller.link.close()
        if hasattr(self.board, "close"):
            self.board.close()
        self.port.close()
        self.lease.unlink(missing_ok=True)
