"""Cliente local del proceso que conserva el puerto UART."""

from __future__ import annotations

from dataclasses import asdict
import json
import socket

from .calibration import Calibration
from .controller import default_socket
from .protocol import IRReading, Position, Session, Status


class ControllerClient:
    def __init__(self, port: str, calibration_file: str, mode: str,
                 socket_path: str | None = None) -> None:
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(2.0)
        try:
            self.socket.connect(socket_path or default_socket())
            self.stream = self.socket.makefile("rwb")
            result = self._call("acquire", port=port, calibration=calibration_file, mode=mode)
        except FileNotFoundError as exc:
            self.socket.close()
            raise ConnectionError("Inicia primero 'ros2 run main_ros forklift_controller'") from exc
        except Exception:
            self.socket.close()
            raise
        self.sequence = result["sequence"]
        self.calibration = (Calibration(**result["calibration"])
                            if result["calibration"] is not None else None)

    def _call(self, action: str, **kwargs) -> dict:
        self.stream.write(json.dumps({"action": action, **kwargs}).encode() + b"\n")
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise ConnectionError("Se perdió el controlador residente")
        result = json.loads(line)
        if not result.get("ok"):
            raise ValueError(result.get("error", "Petición rechazada"))
        return result

    def send(self, frame: bytes) -> None:
        self._call("send", frame=frame.hex())

    def poll(self) -> list[Position | Status | Session | IRReading]:
        result = self._call("poll", after=self.sequence)
        self.sequence = result["sequence"]
        events = []
        for item in result["events"]:
            if item["type"] == "position":
                events.append(Position(item["raw"], item["relative"]))
            elif item["type"] == "status":
                events.append(Status(item["mode"], item["moving"], item["span"],
                                     item["goal"], item.get("busy", False),
                                     item.get("extended", False),
                                     item.get("fault_code", 0)))
            elif item["type"] == "session":
                events.append(Session(item["token"]))
            elif item["type"] == "ir":
                events.append(IRReading(item["fresh"], item["status"],
                                        item["distance_mm"], item["adc"],
                                        item.get("sensor", 1)))
        return events

    def save_calibration(self, calibration: Calibration) -> Calibration:
        self._call("save_calibration", calibration=asdict(calibration))
        return self.get_calibration()

    def set_operational_bottom(self, raw: int) -> Calibration:
        self._call("set_operational_bottom", raw=raw)
        return self.get_calibration()

    def get_calibration(self) -> Calibration | None:
        data = self._call("calibration")["calibration"]
        return Calibration(**data) if data is not None else None

    def close(self) -> None:
        try:
            self._call("release")
        finally:
            self.stream.close()
            self.socket.close()
