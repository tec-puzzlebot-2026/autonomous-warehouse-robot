"""Calibración del elevador con teclas desde una terminal SSH."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import serial

from .calibration import (CONTROL_WINDOW_COUNTS, Calibration, SCHEMA_VERSION,
                          SESSION_SCHEMA_VERSION, load, save)
from .client import ControllerClient
from .link import DEFAULT_PORT
from .protocol import Position, Status, config, goal, signed_frame, simple


POLL_SECONDS = 0.01
HEARTBEAT_SECONDS = 0.05
JOG_SECONDS = 0.05
HOLD_TIMEOUT_SECONDS = 0.23
SETTLE_SECONDS = 0.6
AUTO_TOLERANCE_COUNTS = CONTROL_WINDOW_COUNTS
AUTO_MAX_SECONDS = 90.0
AUTO_MAX_PULSE_SECONDS = 0.12
AUTO_MIN_PULSE_SECONDS = 0.01


class TerminalCalibration:
    def __init__(self, port: str, output: str, link=None, socket_path: str | None = None) -> None:
        self.link = link if link is not None else ControllerClient(port, output, "calibrar", socket_path)
        self.output = Path(output)
        self.mode: str | None = None
        self.moving = False
        self.busy = False
        self.fpga_goal: int | None = None
        self.fpga_fault_code = 0
        self.raw: int | None = None
        self.relative: int | None = None
        self.last_relative = 0.0
        self.last_status = 0.0
        self.last_position = 0.0
        self.last_change = time.monotonic()
        self.last_stop = time.monotonic()
        self.last_jog = 0.0
        self.last_key = 0.0
        self.held: str | None = None
        self.auto: dict | None = None
        self.bottom: int | None = None
        self.top: int | None = None
        self.reference_bottom: int | None = None
        self.reference_top: int | None = None
        self.pending: Calibration | None = None
        self.confirmation: str | None = None
        self.entry: str | None = None
        self.entry_kind: str | None = None
        self.routine: dict | None = None
        self.zero_pending = False
        try:
            self.saved = (self.link.get_calibration() if hasattr(self.link, "get_calibration")
                          else load(self.output) if self.output.is_file() else None)
        except (OSError, ValueError) as exc:
            self.saved = None
            print(f"JSON existente no válido: {exc}")
        if self.saved is not None and self.saved.schema_version >= SESSION_SCHEMA_VERSION:
            self.reference_bottom = self.saved.session_bottom_counts
            self.reference_top = self.reference_bottom + self.saved.signed_span_counts

    def clear_marks(self) -> None:
        self.bottom = self.top = None
        self.reference_bottom = self.reference_top = None
        self.pending = None
        self.confirmation = None

    def stop(self, *, announce: bool = False) -> None:
        active = self.held is not None or self.auto is not None
        self.held = None
        self.auto = None
        if active or announce:
            self.last_stop = time.monotonic()
            self.link.send(simple("X"))
            if announce:
                print("\nMovimiento detenido.")

    def tick(self) -> None:
        now = time.monotonic()
        if not hasattr(self.link, "get_calibration"):
            # Compatibilidad con enlaces falsos de pruebas locales.
            if now - getattr(self, "last_heartbeat", 0.0) >= HEARTBEAT_SECONDS:
                self.link.send(simple("H"))
                self.last_heartbeat = now
        for event in self.link.poll():
            now = time.monotonic()
            if isinstance(event, Status):
                previous = self.mode
                self.mode = event.mode
                self.moving = event.moving
                self.busy = event.busy
                self.fpga_goal = event.goal
                self.fpga_fault_code = event.fault_code
                self.last_status = now
                if self.routine is not None:
                    pass  # La rutina usa temporalmente C y R con la B/T provisionales.
                elif event.mode != "M":
                    self.stop()
                    self.clear_marks()
                    if previous == "M":
                        print(f"\nFPGA en {event.mode}: B/T descartadas. Verifica la posición física.")
                elif previous != "M":
                    if self.saved is not None and self.saved.schema_version >= SESSION_SCHEMA_VERSION:
                        if hasattr(self.link, "get_calibration"):
                            self.saved = self.link.get_calibration()
                        self.reference_bottom = self.saved.session_bottom_counts
                        self.reference_top = self.reference_bottom + self.saved.signed_span_counts
                    print("\nFPGA en M: control manual disponible.")
            elif isinstance(event, Position) and event.relative is None:
                if event.raw != self.raw:
                    self.last_change = now
                self.raw = event.raw
                self.last_position = now
            elif isinstance(event, Position):
                self.relative = event.relative
                self.last_relative = now
        now = time.monotonic()
        if self.held is not None:
            if (self.mode != "M" or now - self.last_key > HOLD_TIMEOUT_SECONDS
                    or now - self.last_status > 0.8 or now - self.last_position > 0.6):
                self.stop()
                print(f"\nMotor detenido. Encoder: {self.raw}; verifica la posición física.")
            elif now - self.last_jog >= JOG_SECONDS:
                self.link.send(simple(self.held))
                self.last_jog = now
        self.advance_auto(now)
        self.advance_routine(now)
        if (self.zero_pending and self.ready(stable=True) and self.saved is not None
                and self.reference_bottom is not None):
            self.zero_pending = False
            if abs(self.raw - self.reference_bottom) <= self.saved.arrival_counts:
                if hasattr(self.link, "set_operational_bottom"):
                    self.saved = self.link.set_operational_bottom(self.raw)
                print(f"\nB alcanzada: cero operativo={self.raw}; ancla B={self.reference_bottom}.")
        if self.last_status and now - self.last_status > 1.0:
            if self.routine is not None:
                self.cancel_routine("Sin telemetría reciente; JSON sin cambios.", recover=False)
            self.stop()
            self.clear_marks()
            self.last_status = 0.0
            print("\nSin telemetría reciente: control detenido y marcas descartadas.")

    def ready(self, *, stable: bool = False) -> bool:
        now = time.monotonic()
        return (self.mode == "M" and not self.moving and self.held is None
                and self.auto is None and self.raw is not None
                and now - self.last_status <= 0.8
                and now - self.last_position <= 0.6
                and (not stable or (now - self.last_change >= SETTLE_SECONDS
                                    and now - self.last_stop >= SETTLE_SECONDS)))

    def require_ready(self, *, stable: bool = False) -> None:
        self.tick()
        if not self.ready(stable=stable):
            suffix = " y estable durante 0.6 s" if stable else ""
            raise ValueError(f"Espera telemetría reciente en M, con motor detenido{suffix}.")

    def jog_key(self, direction: str) -> None:
        if self.routine is not None:
            print("\nRutina activa: X cancela; W/S no están disponibles.")
            return
        self.tick()
        self.zero_pending = False
        if self.auto is not None:
            self.stop(announce=True)
            return
        if self.held != direction:
            self.stop()
            if not self.ready():
                return
            self.held = direction
            self.last_jog = 0.0
            self.pending = None
            self.confirmation = None
        self.last_key = time.monotonic()

    def mark(self, which: str) -> None:
        self.require_ready(stable=True)
        self.zero_pending = False
        if which == "B":
            self.bottom = self.raw
        else:
            self.top = self.raw
        self.pending = None
        self.confirmation = None
        print(f"\n{which}={self.raw} cuentas. B={self.bottom}, T={self.top}. JSON sin cambios.")

    def prepare_save(self, entry: str) -> None:
        self.require_ready(stable=True)
        if self.bottom is None or self.top is None:
            raise ValueError("Primero marca B y T en esta sesión.")
        parts = entry.split()
        if len(parts) not in (1, 2):
            raise ValueError("Escribe distancia_mm [margen_cuentas], por ejemplo: 117 25")
        span = self.top - self.bottom
        travel = float(parts[0])
        margin = int(parts[1]) if len(parts) == 2 else 100
        if not math.isfinite(travel) or travel <= 0 or not 500 <= abs(span) <= 20000:
            raise ValueError("Recorrido o distancia física inválidos")
        if not CONTROL_WINDOW_COUNTS <= margin <= 500 or margin * 2 >= abs(span):
            raise ValueError("Margen inválido: usa 20..500 y menos de la mitad del recorrido")
        effective = abs(span) - margin
        stations = [round(effective * index / 6) for index in range(1, 6)]
        points = [{"counts": 0, "mm": 0.0}] + [
            {"counts": count, "mm": travel * count / abs(span)} for count in stations
        ] + [{"counts": abs(span), "mm": travel}]
        provisional = Calibration(SCHEMA_VERSION, True, span, travel, margin, CONTROL_WINDOW_COUNTS,
                                  self.bottom, self.top, self.bottom, self.bottom,
                                  self.raw, 1, points)
        provisional.validate()
        self.pending = None
        self.confirmation = None
        self.routine = {"phase": "config", "started": time.monotonic(),
                        "configured_at": None, "reference_sent_at": None,
                        "stations": list(reversed(stations)), "index": 0,
                        "target_sent_at": None, "seen_busy": False,
                        "measure_counts": None, "samples": [], "provisional": provisional}
        self.link.send(simple("D"))
        self.link.send(config(span, margin, CONTROL_WINDOW_COUNTS, CONTROL_WINDOW_COUNTS))
        print(f"\nRecorrido: {span} cuentas / {travel:g} mm; margen: {margin} cuentas.")
        print("Rutina de cinco puntos iniciada. Mide con regla la altura desde B en cada parada.")
        print("X cancela y conserva el JSON anterior.")

    def confirm_save(self) -> None:
        calibration = self.pending
        if calibration is None or self.routine is None or self.routine["phase"] != "review":
            return
        self.tick()
        if (self.mode != "R" or self.moving or self.busy or self.raw is None or
                time.monotonic() - self.last_change < SETTLE_SECONDS):
            raise ValueError("Espera al motor detenido y a la posición estable")
        if (self.bottom, self.top) != (calibration.reference_bottom_counts,
                                    calibration.reference_top_counts):
            raise ValueError("Las marcas cambiaron; prepara de nuevo el guardado.")
        if hasattr(self.link, "save_calibration"):
            self.saved = self.link.save_calibration(calibration)
        else:
            save(self.output, calibration)
            self.saved = calibration
        self.reference_bottom = self.bottom
        self.reference_top = self.top
        self.pending = None
        self.confirmation = None
        self.routine = None
        self.link.send(simple("D"))
        self.link.send(simple("M"))
        print(f"Calibración guardada en {self.output}.")
        print("Modo manual solicitado; espera M o sal con Q para iniciar operar.")

    def cancel_routine(self, reason: str, *, recover: bool = True) -> None:
        if self.routine is None:
            return
        self.routine = None
        self.pending = None
        self.confirmation = None
        self.entry = None
        self.entry_kind = None
        self.bottom = self.top = None
        try:
            self.link.send(simple("X"))
            self.link.send(simple("D"))
            if recover:
                self.link.send(simple("M"))
        except (OSError, ValueError):
            pass
        print(f"\nRutina cancelada: {reason} JSON sin cambios.")

    def advance_routine(self, now: float) -> None:
        routine = self.routine
        if routine is None:
            return
        if self.mode == "F":
            self.cancel_routine("La FPGA informó una falla; comprueba la posición física.", recover=False)
            return
        if routine["phase"] == "config":
            if now - routine["started"] > 7.0:
                self.cancel_routine("La FPGA no confirmó la configuración.", recover=False)
                return
            if self.mode == "C":
                if routine["configured_at"] is None:
                    routine["configured_at"] = now
                if (now - routine["configured_at"] >= 0.75 and
                        (routine["reference_sent_at"] is None or
                         now - routine["reference_sent_at"] >= 1.0)):
                    self.link.send(signed_frame("A", routine["provisional"].reference_bottom_counts))
                    routine["reference_sent_at"] = now
            elif self.mode == "R" and routine["reference_sent_at"] is not None:
                routine["phase"] = "positioning"
            return
        if routine["phase"] != "positioning":
            return
        if self.mode == "R" and self.fpga_fault_code in (5, 6):
            self.cancel_routine("La FPGA no alcanzó la parada de calibración; revisa la posición física.")
            return
        if self.mode != "R" or now - self.last_status > 0.8 or now - self.last_position > 0.6:
            self.cancel_routine("Estado o encoder sin datos recientes.", recover=False)
            return
        target = routine["stations"][routine["index"]]
        if routine["target_sent_at"] is None:
            if self.moving or self.busy:
                return
            self.link.send(goal(target, CONTROL_WINDOW_COUNTS))
            routine["target_sent_at"] = now
            routine["seen_busy"] = False
            print(f"\nPunto {routine['index']+1}/5: moviendo a {target} cuentas desde B...")
            return
        if now - routine["target_sent_at"] > 40.0:
            self.cancel_routine("La meta no terminó a tiempo.", recover=False)
            return
        if self.fpga_goal == target and self.busy:
            routine["seen_busy"] = True
        if (routine["seen_busy"] and not self.busy and not self.moving and
                self.relative is not None and now - self.last_relative <= 0.5 and
                now - self.last_change >= SETTLE_SECONDS):
            if abs(self.relative - target) > AUTO_TOLERANCE_COUNTS:
                self.cancel_routine("La parada quedó fuera de 20 cuentas.", recover=False)
                return
            routine["measure_counts"] = self.relative
            routine["phase"] = "measuring"
            self.entry_kind = "measure"
            self.entry = ""
            print(f"\nPunto {routine['index']+1}/5 detenido en {self.relative} cuentas.")
            print("Altura física desde B en mm (Enter confirma, Esc cancela): ", end="", flush=True)

    def accept_measurement(self, entry: str) -> None:
        routine = self.routine
        if routine is None or routine["phase"] != "measuring":
            raise ValueError("No hay medición pendiente")
        height = float(entry)
        travel = routine["provisional"].travel_mm
        if not math.isfinite(height) or not 0 < height < travel:
            raise ValueError(f"La altura debe estar entre 0 y {travel:g} mm")
        if routine["samples"] and height >= routine["samples"][-1]["mm"]:
            raise ValueError("Al bajar, cada altura debe ser menor que la anterior")
        self.tick()
        if (self.mode != "R" or self.busy or self.moving or self.relative is None or
                time.monotonic() - self.last_change < SETTLE_SECONDS or
                abs(self.relative - routine["measure_counts"]) > CONTROL_WINDOW_COUNTS):
            raise ValueError("La posición cambió; espera estable o cancela la rutina")
        routine["samples"].append({"counts": routine["measure_counts"], "mm": height})
        routine["index"] += 1
        if routine["index"] < 5:
            routine["phase"] = "positioning"
            routine["target_sent_at"] = None
            routine["seen_busy"] = False
            print(f"Medición {routine['index']}/5 registrada: {routine['samples'][-1]}.")
            return
        provisional = routine["provisional"]
        points = ([{"counts": 0, "mm": 0.0}] +
                  sorted(routine["samples"], key=lambda point: point["counts"]) +
                  [{"counts": abs(provisional.signed_span_counts), "mm": provisional.travel_mm}])
        calibration = replace(provisional, calibration_points=points,
                              last_raw_counts=self.raw)
        try:
            calibration.validate()
        except ValueError:
            routine["samples"].pop()
            routine["index"] -= 1
            raise
        self.pending = calibration
        self.confirmation = "save"
        routine["phase"] = "review"
        print("\nMapa medido (cuentas desde B → altura en mm):")
        for point in points:
            print(f"  {point['counts']:>5} → {point['mm']:g}")
        print(f"Pulsa Y para sustituir {self.output}, o N para conservar el JSON anterior.")

    def confirm_bottom(self) -> None:
        self.require_ready(stable=True)
        if self.saved is None:
            raise ValueError("No hay JSON válido; marca B/T y guarda una calibración primero.")
        if self.saved.schema_version >= SESSION_SCHEMA_VERSION:
            print("\nLa B anclada ya está cargada; C no cambia el cero ni el JSON.")
            return
        self.reference_bottom = self.raw
        self.reference_top = self.raw + self.saved.signed_span_counts
        print(f"\nB actual={self.reference_bottom}, T calculada={self.reference_top}. JSON sin cambios.")

    def ask_restore(self) -> None:
        self.require_ready(stable=True)
        saved = self.saved
        if saved is None or saved.reference_bottom_counts is None or saved.reference_top_counts is None:
            raise ValueError("El JSON no contiene marcas B/T para recuperar.")
        if saved.schema_version >= SESSION_SCHEMA_VERSION:
            if hasattr(self.link, "get_calibration"):
                self.saved = self.link.get_calibration()
            self.reference_bottom = self.saved.session_bottom_counts
            self.reference_top = self.reference_bottom + self.saved.signed_span_counts
            print(f"\nB/T de sesión: {self.reference_bottom} / {self.reference_top}.")
            return
        bottom = saved.session_bottom_counts if saved.schema_version >= SESSION_SCHEMA_VERSION else saved.reference_bottom_counts
        top = bottom + saved.signed_span_counts
        low, high = sorted((bottom, top))
        if not low - AUTO_TOLERANCE_COUNTS <= self.raw <= high + AUTO_TOLERANCE_COUNTS:
            raise ValueError("El encoder está fuera del intervalo B/T guardado.")
        self.confirmation = "restore"
        print(f"\n¿La FPGA sigue en el MISMO encendido en que se guardó B="
              f"{bottom}, T={top}? "
              "Comprueba físicamente la posición. Pulsa Y para recuperar o N para cancelar.")

    def confirm_restore(self) -> None:
        self.require_ready(stable=True)
        saved = self.saved
        bottom = saved.session_bottom_counts if saved.schema_version >= SESSION_SCHEMA_VERSION else saved.reference_bottom_counts
        top = bottom + saved.signed_span_counts
        low, high = sorted((bottom, top))
        if not low - AUTO_TOLERANCE_COUNTS <= self.raw <= high + AUTO_TOLERANCE_COUNTS:
            raise ValueError("El encoder salió del intervalo B/T guardado.")
        self.reference_bottom = bottom
        self.reference_top = top
        self.confirmation = None
        print("\nB/T recuperadas para este encendido. Verifica físicamente antes de Ir a B/T.")

    def start_auto(self, which: str) -> None:
        self.require_ready(stable=True)
        self.zero_pending = False
        if self.saved is None or self.reference_bottom is None or self.reference_top is None:
            raise ValueError("Confirma B actual con C o recupera B/T con R antes de Ir a B/T.")
        low, high = sorted((self.reference_bottom, self.reference_top))
        boundary_tolerance = self.saved.arrival_counts if which == "B" else AUTO_TOLERANCE_COUNTS
        if not low - boundary_tolerance <= self.raw <= high + boundary_tolerance:
            raise ValueError("Encoder fuera de B/T; coloca la plataforma dentro manualmente.")
        target = (self.saved.session_bottom_counts if which == "B" and
                  self.saved.schema_version >= SESSION_SCHEMA_VERSION else
                  self.reference_bottom if which == "B" else self.reference_top)
        tolerance = self.saved.arrival_counts if which == "B" else AUTO_TOLERANCE_COUNTS
        if which == "B" and abs(self.raw - target) <= tolerance:
            self.zero_pending = True
            print(f"\nYa estás dentro de la ventana de B: diferencia de {self.raw - target} cuentas.")
            return
        boundary_tolerance = tolerance
        if which == "B":
            # Aproxima desde arriba: la inercia residual puede consumir la mitad
            # de la ventana. El punto de parada buscado queda antes del ancla.
            direction_to_top = 1 if self.saved.signed_span_counts > 0 else -1
            half_window = max(1, tolerance // 2)
            target += direction_to_top * half_window
            tolerance -= half_window
        direction = 1 if target > self.raw else -1
        remaining = (target - self.raw) * direction
        if remaining <= tolerance:
            if which == "B":
                self.zero_pending = True
            print(f"\nYa estás cerca de {which}: faltan {remaining} cuentas.")
            return
        self.auto = {"which": which, "target": target, "direction": direction,
                     "tolerance": tolerance, "boundary_tolerance": boundary_tolerance,
                     "command": "W" if (target - self.raw) * self.saved.signed_span_counts > 0 else "S",
                     "started": time.monotonic(), "phase": "settled", "rate": None,
                     "pulse_start": self.raw, "pulse_end": 0.0, "pulse_seconds": 0.0}
        print(f"\nHacia {which}; X cancela. Comprueba físicamente el punto alcanzado.")

    def finish_auto(self, message: str) -> None:
        self.stop()
        print(f"\n{message} Encoder: {self.raw}.")

    def advance_auto(self, now: float) -> None:
        move = self.auto
        if move is None:
            return
        if (self.mode != "M" or now - self.last_status > 0.8
                or now - self.last_position > 0.6
                or now - move["started"] > AUTO_MAX_SECONDS):
            self.finish_auto("Cancelado: estado/encoder sin datos recientes o tiempo agotado.")
            return
        low, high = sorted((self.reference_bottom, self.reference_top))
        if not low - move["boundary_tolerance"] <= self.raw <= high + move["boundary_tolerance"]:
            self.finish_auto("Detenido: encoder fuera de B/T.")
            return
        remaining = (move["target"] - self.raw) * move["direction"]
        if move["phase"] == "pulse":
            if remaining <= move["tolerance"] or now >= move["pulse_end"]:
                self.link.send(simple("X"))
                move["phase"] = "settling"
                move["pulse_end"] = now
            return
        if move["phase"] == "settling":
            if now - move["pulse_end"] > 2.5:
                self.finish_auto("Detenido: el encoder no se estabilizó.")
                return
            if (self.last_position <= move["pulse_end"] or
                    now - self.last_change < SETTLE_SECONDS):
                return
            if (move["which"] == "B" and
                    abs(self.raw - self.reference_bottom) <= move["boundary_tolerance"]):
                self.zero_pending = True
                self.finish_auto("B dentro de la ventana. Verifica físicamente.")
                return
            progress = (self.raw - move["pulse_start"]) * move["direction"]
            if progress <= 0:
                self.finish_auto("Detenido: el encoder no avanzó hacia la marca.")
                return
            move["rate"] = progress / move["pulse_seconds"]
            move["phase"] = "settled"
        if remaining < -move["tolerance"]:
            self.finish_auto("Detenido: se rebasó la marca. Revisa la posición física.")
            return
        if remaining <= move["tolerance"]:
            if move["which"] == "B":
                self.zero_pending = True
            self.finish_auto(f"Cerca de {move['which']}: diferencia de {remaining} cuentas. Verifica físicamente.")
            return
        if (move["rate"] is not None and
                move["rate"] * AUTO_MIN_PULSE_SECONDS * 1.5 > remaining + move["tolerance"]):
            self.finish_auto(
                f"Detenido antes de {move['which']}: el siguiente pulso podría rebasar "
                "la ventana. Comprueba la posición y aproxima manualmente si hace falta.")
            return
        desired_advance = max(1, (remaining - move["tolerance"]) / 2)
        pulse_seconds = (0.08 if move["rate"] is None else
                         min(AUTO_MAX_PULSE_SECONDS,
                             max(AUTO_MIN_PULSE_SECONDS, desired_advance / move["rate"])))
        move["pulse_start"] = self.raw
        move["pulse_seconds"] = pulse_seconds
        move["pulse_end"] = now + pulse_seconds
        move["phase"] = "pulse"
        self.link.send(simple(move["command"]))

    def status(self) -> None:
        self.tick()
        print(f"\nFPGA={self.mode or 'sin datos'}, motor={'activo' if self.moving else 'detenido'}, "
              f"meta={'activa' if self.busy else 'inactiva'}, "
              f"encoder={self.raw if self.raw is not None else 'sin datos'}, "
              f"B={self.bottom}, T={self.top}, B actual={self.reference_bottom}, "
              f"T actual={self.reference_top}")
        if self.saved is not None:
            print(f"JSON: {self.saved.travel_mm:g} mm / {self.saved.signed_span_counts} cuentas.")

    @staticmethod
    def help() -> None:
        print("\n=== CONTROLES DEL ELEVADOR ===")
        print("W / S  Subir / bajar mientras se mantiene la tecla")
        print("X      Detener")
        print("B / T  Marcar punto inferior / superior")
        print("G      Calibrar: distancia, margen y cinco alturas medidas")
        print("Y / N  Confirmar / cancelar cuando se solicite")
        print("C      Consultar B anclada; no cambia el cero")
        print("R      Actualizar B/T de esta sesión desde el JSON")
        print("1 / 2  Ir a B / ir a T")
        print("P      Ver estado y posición")
        print("M      Activar modo manual desde D o F")
        print("H      Mostrar este menú")
        print("Q      Salir")
        print("W/S se detienen si dejan de llegar pulsaciones repetidas.")

    def key(self, character: str) -> bool:
        key = character.lower()
        if self.routine is not None and key in ("x", "q"):
            self.cancel_routine("Cancelada por teclado.")
            return key != "q"
        if self.entry is not None:
            if character in ("\r", "\n"):
                entry, kind = self.entry, self.entry_kind
                self.entry = None
                self.entry_kind = None
                print()
                try:
                    if kind == "measure":
                        self.accept_measurement(entry)
                    else:
                        self.prepare_save(entry)
                except (OSError, ValueError) as exc:
                    print(f"No se aceptó la entrada: {exc}")
                    if kind == "measure" and self.routine is not None:
                        self.entry_kind = "measure"
                        self.entry = ""
                        print("Repite la altura en mm, o Esc cancela: ", end="", flush=True)
            elif character in ("\x7f", "\b"):
                if self.entry:
                    self.entry = self.entry[:-1]
                    print("\b \b", end="", flush=True)
            elif character == "\x1b":
                if self.routine is not None:
                    self.cancel_routine("Entrada cancelada.")
                else:
                    self.entry = None
                    self.entry_kind = None
                    print("\nEntrada cancelada.")
            elif character in "0123456789. -" and len(self.entry) < 32:
                self.entry += character
                print(character, end="", flush=True)
            return True
        if self.confirmation is not None:
            if key == "y":
                kind = self.confirmation
                try:
                    if kind == "save":
                        self.confirm_save()
                    else:
                        self.confirm_restore()
                except (OSError, ValueError) as exc:
                    print(f"\nNo se confirmó: {exc}")
                    self.confirmation = None
                    self.pending = None
                return True
            if key == "n" or key == "x":
                if self.routine is not None:
                    self.cancel_routine("Guardado cancelado.")
                else:
                    self.confirmation = None
                    self.pending = None
                    print("\nCancelado.")
                return True
            return True
        if key in ("w", "s"):
            self.jog_key(key.upper())
            return True
        if key == "x":
            self.zero_pending = False
            self.stop(announce=True)
            return True
        if key == "q":
            self.stop()
            return False
        if key == "g":
            if self.routine is not None:
                print("\nTermina o cancela la rutina actual con X.")
                return True
            self.stop()
            self.entry = ""
            self.entry_kind = "config"
            print("\nDistancia_mm [margen_cuentas] (Enter para preparar, Esc cancela): ",
                  end="", flush=True)
            return True
        try:
            if key in ("b", "t"):
                self.mark(key.upper())
            elif key == "c":
                self.confirm_bottom()
            elif key == "r":
                self.ask_restore()
            elif key in ("1", "2"):
                self.start_auto("B" if key == "1" else "T")
            elif key == "m":
                self.tick()
                if self.mode not in ("D", "F") or time.monotonic() - self.last_status > 0.8:
                    raise ValueError("Espera telemetría reciente en D o F antes de activar M.")
                self.link.send(simple("D"))
                self.link.send(simple("M"))
                print("\nActivación M solicitada; espera a verla con P.")
            elif key == "p":
                self.status()
            elif key == "h":
                self.help()
        except ValueError as exc:
            print(f"\nNo se aplicó la tecla: {exc}")
        return True

    def run(self) -> None:
        descriptor = sys.stdin.fileno()
        old_settings = termios.tcgetattr(descriptor)
        try:
            tty.setcbreak(descriptor)
            self.link.send(simple("D"))
            self.link.send(simple("M"))
            print("Calibración por SSH. Modo M sin límites físicos de recorrido.")
            self.help()
            print("El controlador residente conserva el puerto. Ctrl+C detiene y cierra este modo.")
            while True:
                self.tick()
                if select.select([descriptor], [], [], POLL_SECONDS)[0]:
                    data = os.read(descriptor, 32)
                    if not data:
                        break
                    for byte in data:
                        if not self.key(chr(byte)):
                            return
        finally:
            try:
                self.link.close()  # X y D al salir o si se corta la sesión SSH.
            finally:
                termios.tcsetattr(descriptor, termios.TCSADRAIN, old_settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrar elevador por SSH, sin interfaz gráfica")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--output", default="calibracion_elevador.json")
    parser.add_argument("--socket", default=None)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Se necesita una terminal SSH interactiva (TTY).")
    try:
        TerminalCalibration(args.port, args.output, socket_path=args.socket).run()
    except KeyboardInterrupt:
        print("\nInterrumpido. Motor detenido y puerto cerrado.")
    except (serial.SerialException, OSError) as exc:
        parser.exit(1, f"Error de puerto o archivo: {exc}\n")


if __name__ == "__main__":
    main()
