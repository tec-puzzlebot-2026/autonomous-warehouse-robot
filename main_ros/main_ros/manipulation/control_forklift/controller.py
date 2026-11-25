"""Proceso residente: único dueño de UART, latido y checkpoint de posición."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

from .calibration import Calibration, SCHEMA_VERSION, SESSION_SCHEMA_VERSION, load, save
from .link import DEFAULT_PORT, MotorLink
from .protocol import IRReading, Position, Session, Status, simple, signed_frame


def default_socket() -> str:
    return f"/tmp/control_forklift-{os.getuid()}.sock"


class Controller:
    def __init__(self, port: str, calibration_file: str, link=None) -> None:
        self.port = port
        self.path = Path(calibration_file).expanduser().resolve()
        self.link = link if link is not None else MotorLink(port)
        self.lock = threading.RLock()
        self.tx_lock = threading.Lock()
        self.events: deque[tuple[int, dict]] = deque(maxlen=256)
        self.sequence = 0
        self.owner: str | None = None
        self.raw: int | None = None
        self.raw_verified = False
        self.mode: str | None = None
        self.moving = False
        self.protocol_current = False
        self.token: int | None = None
        self.last_checkpoint = 0.0
        self.checkpoint_pending = False
        self.running = True
        self.calibration = load(self.path) if self.path.is_file() else None
        self.initialized = False
        self.token_acknowledged = False
        self.token_set_at = 0.0
        self.mismatch_token: int | None = None
        self.mismatch_count = 0

    def _send(self, frame: bytes) -> None:
        with self.tx_lock:
            self.link.send(frame)

    def _event(self, event) -> None:
        if isinstance(event, Position):
            payload = {"type": "position", "raw": event.raw, "relative": event.relative}
        elif isinstance(event, Status):
            payload = {"type": "status", "mode": event.mode, "moving": event.moving,
                       "span": event.span, "goal": event.goal, "busy": event.busy,
                       "extended": event.extended, "fault_code": event.fault_code}
        elif isinstance(event, IRReading):
            payload = {"type": "ir", "fresh": event.fresh,
                       "status": event.status, "distance_mm": event.distance_mm,
                       "adc": event.adc, "sensor": event.sensor}
        else:
            payload = {"type": "session", "token": event.token}
        self.sequence += 1
        self.events.append((self.sequence, payload))

    def _initialize(self, observed_token: int) -> None:
        if self.raw is None:
            return
        saved = self.calibration
        if saved is not None and saved.schema_version >= SESSION_SCHEMA_VERSION:
            if observed_token != saved.session_token:
                # Se conserva la altura física del último checkpoint; el encoder nuevo
                # puede comenzar en cero o en otra cuenta si el firmware se reinició.
                delta = self.raw - saved.last_raw_counts
                saved = replace(saved,
                                session_bottom_counts=saved.session_bottom_counts + delta,
                                operational_bottom_counts=saved.operational_bottom_counts + delta,
                                last_raw_counts=self.raw)
                saved.validate()
                self.calibration = saved
                print("FPGA reiniciada: referencia recuperada suponiendo plataforma inmóvil.", flush=True)
            else:
                print("Misma sesión de FPGA: referencias del JSON conservadas.", flush=True)
        self.token = secrets.randbelow(2**32 - 1) + 1
        self._send(signed_frame("I", self.token))
        self.token_acknowledged = False
        self.token_set_at = time.monotonic()
        self.raw_verified = False
        if self.calibration is not None and self.calibration.schema_version >= SESSION_SCHEMA_VERSION:
            self.calibration = replace(self.calibration, session_token=self.token,
                                       last_raw_counts=self.raw)
            save(self.path, self.calibration)
        self.initialized = True

    def tick(self) -> None:
        with self.lock:
            now = time.monotonic()
            for event in self.link.poll():
                self._event(event)
                if isinstance(event, Position) and event.relative is None:
                    if self.raw != event.raw:
                        self.checkpoint_pending = True
                    self.raw = event.raw
                    self.raw_verified = False
                elif isinstance(event, Status):
                    self.protocol_current = event.extended
                    if self.moving and not event.moving:
                        self.checkpoint_pending = True
                        self.last_checkpoint = 0.0
                    self.moving = event.moving
                    self.mode = event.mode
                elif isinstance(event, Session):
                    if not self.initialized:
                        self._initialize(event.token)
                    elif event.token == self.token:
                        self.token_acknowledged = True
                        self.raw_verified = True
                        self.mismatch_count = 0
                    elif self.token_acknowledged:
                        self.mismatch_count = (self.mismatch_count + 1 if
                                               event.token == self.mismatch_token else 1)
                        self.mismatch_token = event.token
                        if self.mismatch_count >= 2:
                            self._send(simple("X"))
                            self._initialize(event.token)
                            self.mismatch_count = 0
            if (self.initialized and not self.token_acknowledged and
                    now - self.token_set_at > 1.0):
                self._send(signed_frame("I", self.token))
                self.token_set_at = now
            if (self.initialized and self.token_acknowledged and self.raw_verified and
                    self.mismatch_count == 0 and
                    self.checkpoint_pending and self.raw is not None
                    and now - self.last_checkpoint >= 1.0):
                self.checkpoint()

    def checkpoint(self) -> None:
        with self.lock:
            if (self.calibration is not None and self.calibration.schema_version >= SESSION_SCHEMA_VERSION
                    and self.raw is not None and self.token is not None and
                    self.token_acknowledged and self.raw_verified):
                updated = replace(self.calibration, last_raw_counts=self.raw,
                                  session_token=self.token)
                save(self.path, updated)
                self.calibration = updated
                self.checkpoint_pending = False
            self.last_checkpoint = time.monotonic()

    def save_calibration(self, data: dict) -> None:
        with self.lock:
            if not self.initialized or self.raw is None or self.token is None:
                raise ValueError("Espera telemetría y sesión de FPGA")
            calibration = Calibration(**data)
            if calibration.schema_version != SCHEMA_VERSION:
                raise ValueError("Formato de calibración anterior")
            calibration = replace(calibration, session_bottom_counts=calibration.reference_bottom_counts,
                                  operational_bottom_counts=calibration.reference_bottom_counts,
                                  last_raw_counts=self.raw, session_token=self.token)
            save(self.path, calibration)
            self.calibration = calibration
            self.checkpoint_pending = False

    def set_operational_bottom(self, raw: int) -> None:
        with self.lock:
            saved = self.calibration
            if saved is None or saved.schema_version < SESSION_SCHEMA_VERSION or self.raw is None:
                raise ValueError("No hay calibración actual")
            if self.moving or abs(self.raw - raw) > saved.arrival_counts:
                raise ValueError("El motor no está detenido o la posición cambió")
            if abs(raw - saved.session_bottom_counts) > saved.arrival_counts:
                raise ValueError("B está fuera de la ventana del ancla")
            self.calibration = replace(saved, operational_bottom_counts=raw,
                                       last_raw_counts=self.raw)
            save(self.path, self.calibration)
            self.checkpoint_pending = False

    def request(self, request: dict, client_id: str) -> dict:
        with self.lock:
            action = request.get("action")
            if action == "acquire":
                if self.owner is not None:
                    raise ValueError(f"Modo {self.owner} en uso; cierra ese cliente primero")
                if not self.initialized or not self.token_acknowledged:
                    raise ValueError("Controlador esperando telemetría de la FPGA")
                if not self.protocol_current:
                    raise ValueError("El bitstream de FPGA no tiene la telemetría del control nuevo")
                if request.get("port") != self.port or Path(request.get("calibration", "")).expanduser().resolve() != self.path:
                    raise ValueError("Puerto o ruta JSON distintos a los del controlador")
                if request.get("mode") not in ("calibrar", "operar"):
                    raise ValueError("Modo desconocido")
                if request["mode"] == "operar" and (self.calibration is None or
                        self.calibration.schema_version < SESSION_SCHEMA_VERSION):
                    raise ValueError("Se requiere nueva calibración con el firmware actual")
                self.owner = client_id
                return {"sequence": self.sequence, "calibration":
                        asdict(self.calibration) if self.calibration else None}
            if self.owner != client_id:
                raise ValueError("No tienes el control activo")
            if action == "poll":
                after = int(request.get("after", 0))
                if self.events and after < self.events[0][0] - 1:
                    raise ValueError("Se perdieron eventos de telemetría; detén y reinicia el modo")
                return {"sequence": self.sequence,
                        "events": [payload for number, payload in self.events if number > after],
                        "raw": self.raw, "mode": self.mode, "moving": self.moving}
            if action == "send":
                frame = bytes.fromhex(request["frame"])
                if frame[:1] not in (b"M", b"D", b"K", b"A", b"G", b"W", b"S", b"X"):
                    raise ValueError("Comando reservado al controlador")
                self._send(frame)
                return {}
            if action == "save_calibration":
                self.save_calibration(request["calibration"])
                return {}
            if action == "set_operational_bottom":
                self.set_operational_bottom(int(request["raw"]))
                return {}
            if action == "calibration":
                return {"calibration": asdict(self.calibration) if self.calibration else None}
            if action == "release":
                self._send(simple("X"))
                self.checkpoint()
                self.owner = None
                return {}
            raise ValueError("Petición desconocida")

    def client_loop(self, connection: socket.socket) -> None:
        client_id = secrets.token_hex(8)
        try:
            with connection, connection.makefile("rwb") as stream:
                while self.running:
                    line = stream.readline()
                    if not line:
                        break
                    try:
                        result = self.request(json.loads(line), client_id)
                        response = {"ok": True, **result}
                    except Exception as exc:
                        response = {"ok": False, "error": str(exc)}
                    stream.write(json.dumps(response).encode() + b"\n")
                    stream.flush()
        finally:
            with self.lock:
                if self.owner == client_id:
                    self._send(simple("X"))
                    self.checkpoint()
                    self.owner = None

    def serve(self, socket_path: str) -> None:
        path = Path(socket_path)
        if path.exists():
            try:
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.connect(socket_path)
                raise RuntimeError("Ya existe un controlador en ese socket")
            except ConnectionRefusedError:
                path.unlink()
        server = socket.socket(socket.AF_UNIX)
        server.bind(socket_path)
        os.chmod(socket_path, 0o600)
        server.listen()
        server.settimeout(0.1)
        worker = threading.Thread(target=self._serial_loop, daemon=True)
        heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        worker.start()
        heartbeat.start()
        print(f"Controlador activo: {socket_path}; JSON: {self.path}", flush=True)
        try:
            while self.running:
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self.client_loop, args=(connection,), daemon=True).start()
        finally:
            self.running = False
            worker.join(timeout=1.0)
            heartbeat.join(timeout=1.0)
            self.checkpoint()
            self.link.close()
            server.close()
            path.unlink(missing_ok=True)

    def _serial_loop(self) -> None:
        while self.running:
            try:
                self.tick()
            except Exception as exc:
                print(f"Error UART/JSON: {exc}", flush=True)
                self.running = False
                break
            time.sleep(0.02)

    def _heartbeat_loop(self) -> None:
        while self.running:
            try:
                self._send(simple("H"))
            except Exception as exc:
                print(f"Latido UART interrumpido: {exc}", flush=True)
                self.running = False
                break
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlador residente del elevador")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--socket", default=default_socket())
    args = parser.parse_args()
    controller = Controller(args.port, args.calibration)
    try:
        controller.serve(args.socket)
    except KeyboardInterrupt:
        controller.running = False


if __name__ == "__main__":
    main()
