"""Geometría y última posición del elevador, en escritura atómica."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = 5
SESSION_SCHEMA_VERSION = 4
LEGACY_SCHEMA_VERSION = 2
CONTROL_WINDOW_COUNTS = 20


@dataclass(frozen=True)
class Calibration:
    schema_version: int
    complete: bool
    signed_span_counts: int
    travel_mm: float
    margin_counts: int
    tolerance_counts: int
    reference_bottom_counts: int | None = None
    reference_top_counts: int | None = None
    session_bottom_counts: int | None = None
    operational_bottom_counts: int | None = None
    last_raw_counts: int | None = None
    session_token: int | None = None
    calibration_points: list[dict] | None = None

    def validate(self) -> None:
        if self.schema_version not in (LEGACY_SCHEMA_VERSION, 3, SESSION_SCHEMA_VERSION,
                                       SCHEMA_VERSION) or self.complete is not True:
            raise ValueError("Calibración incompleta o versión incompatible")
        if not isinstance(self.signed_span_counts, int) or isinstance(self.signed_span_counts, bool):
            raise ValueError("signed_span_counts debe ser entero")
        span = abs(self.signed_span_counts)
        if not 500 <= span <= 20000:
            raise ValueError("El recorrido debe ser de 500 a 20000 cuentas")
        if not isinstance(self.travel_mm, (int, float)) or not math.isfinite(self.travel_mm) or self.travel_mm <= 0:
            raise ValueError("travel_mm debe ser positivo y finito")
        if not isinstance(self.margin_counts, int) or not 1 <= self.margin_counts <= 500:
            raise ValueError("margin_counts debe estar entre 1 y 500")
        if self.margin_counts * 2 >= span:
            raise ValueError("El margen consume demasiado recorrido")
        if not isinstance(self.tolerance_counts, int) or not 1 <= self.tolerance_counts <= min(50, self.margin_counts):
            raise ValueError("tolerance_counts inválido")
        if self.schema_version == LEGACY_SCHEMA_VERSION:
            if self.reference_bottom_counts is not None or self.reference_top_counts is not None:
                raise ValueError("La versión 2 no contiene marcas B/T")
        else:
            if (not isinstance(self.reference_bottom_counts, int)
                    or isinstance(self.reference_bottom_counts, bool)
                    or not isinstance(self.reference_top_counts, int)
                    or isinstance(self.reference_top_counts, bool)):
                raise ValueError("Las marcas B/T de referencia deben ser enteros")
            if self.reference_top_counts - self.reference_bottom_counts != self.signed_span_counts:
                raise ValueError("Las marcas B/T no coinciden con signed_span_counts")
        if self.schema_version >= SESSION_SCHEMA_VERSION:
            for value in (self.session_bottom_counts, self.operational_bottom_counts,
                          self.last_raw_counts):
                if not isinstance(value, int) or isinstance(value, bool) or not -2**31 <= value < 2**31:
                    raise ValueError("La posición de sesión debe ser signed32")
            if (not isinstance(self.session_token, int) or isinstance(self.session_token, bool)
                    or not 0 < self.session_token < 2**32):
                raise ValueError("session_token inválido")
        elif any(value is not None for value in (self.session_bottom_counts,
                self.operational_bottom_counts, self.last_raw_counts, self.session_token)):
            raise ValueError("La versión anterior no contiene estado de sesión")
        if self.schema_version == SCHEMA_VERSION:
            points = self.calibration_points
            if not isinstance(points, list) or len(points) != 7:
                raise ValueError("Se requieren B, cinco mediciones y T")
            last_count = -1
            last_mm = -1.0
            for point in points:
                if not isinstance(point, dict) or set(point) != {"counts", "mm"}:
                    raise ValueError("Punto de calibración inválido")
                count, height = point["counts"], point["mm"]
                if (not isinstance(count, int) or isinstance(count, bool) or
                        not isinstance(height, (int, float)) or isinstance(height, bool) or
                        not math.isfinite(height) or count <= last_count or height <= last_mm):
                    raise ValueError("Las mediciones deben crecer con las cuentas")
                last_count, last_mm = count, height
            if (points[0] != {"counts": 0, "mm": 0.0} or
                    points[-1]["counts"] != span or
                    abs(points[-1]["mm"] - self.travel_mm) > 1e-6):
                raise ValueError("B/T no coinciden con los extremos del mapa")
        elif self.calibration_points is not None:
            raise ValueError("La versión anterior no contiene mediciones intermedias")
        if self.schema_version >= SESSION_SCHEMA_VERSION:
            # Un JSON anterior podía guardar el cero dentro de ±3 mm. Se conserva
            # ese cero al actualizar el software; los nuevos ajustes usan ±20 cuentas.
            legacy_window = (round(3.0 * self.calibration_points[1]["counts"] /
                                   self.calibration_points[1]["mm"])
                             if self.schema_version == SCHEMA_VERSION
                             else round(3.0 * span / self.travel_mm))
            if abs(self.operational_bottom_counts - self.session_bottom_counts) > max(CONTROL_WINDOW_COUNTS, legacy_window):
                raise ValueError("El cero operativo supera la ventana de B")

    @property
    def arrival_counts(self) -> int:
        return CONTROL_WINDOW_COUNTS

    def arrival_counts_at(self, height_mm: float) -> int:
        """Una sola ventana de llegada para cualquier altura."""
        self.mm_to_counts(height_mm)
        return CONTROL_WINDOW_COUNTS

    def with_position(self, raw: int) -> "Calibration":
        return replace(self, last_raw_counts=raw)

    @property
    def max_goal_counts(self) -> int:
        return abs(self.signed_span_counts) - self.margin_counts

    @property
    def max_goal_mm(self) -> float:
        offset = 0
        if self.operational_bottom_counts is not None and self.session_bottom_counts is not None:
            direction = 1 if self.signed_span_counts > 0 else -1
            offset = direction * (self.operational_bottom_counts - self.session_bottom_counts)
        return self.counts_to_mm(self.max_goal_counts - offset)

    def mm_to_counts(self, height_mm: float) -> int:
        if not math.isfinite(height_mm) or height_mm < 0 or height_mm > self.travel_mm:
            raise ValueError(f"Altura fuera de 0..{self.max_goal_mm:.2f} mm")
        if self.schema_version == SCHEMA_VERSION and self.calibration_points:
            points = self.calibration_points
            for left, right in zip(points, points[1:]):
                if height_mm <= right["mm"]:
                    return round(left["counts"] + (height_mm - left["mm"]) *
                                 (right["counts"] - left["counts"]) /
                                 (right["mm"] - left["mm"]))
        return round(height_mm * abs(self.signed_span_counts) / self.travel_mm)

    def counts_to_mm(self, counts: int) -> float:
        if self.schema_version == SCHEMA_VERSION and self.calibration_points:
            points = self.calibration_points
            for left, right in zip(points, points[1:]):
                if counts <= right["counts"]:
                    return left["mm"] + (counts - left["counts"]) * (
                        right["mm"] - left["mm"]) / (right["counts"] - left["counts"])
            left, right = points[-2:]
            return left["mm"] + (counts - left["counts"]) * (
                right["mm"] - left["mm"]) / (right["counts"] - left["counts"])
        return counts * self.travel_mm / abs(self.signed_span_counts)


def load(path: str | Path) -> Calibration:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Campos de calibración inesperados")
    required = set(Calibration.__dataclass_fields__)
    if data.get("schema_version") == LEGACY_SCHEMA_VERSION:
        required -= {"reference_bottom_counts", "reference_top_counts"}
    if data.get("schema_version") in (LEGACY_SCHEMA_VERSION, 3):
        required -= {"session_bottom_counts", "operational_bottom_counts",
                     "last_raw_counts", "session_token"}
    if data.get("schema_version") != SCHEMA_VERSION:
        required -= {"calibration_points"}
    if set(data) != required:
        raise ValueError("Campos de calibración inesperados")
    result = Calibration(**data)
    result.validate()
    if result.margin_counts < CONTROL_WINDOW_COUNTS:
        raise ValueError("El margen guardado debe ser de al menos 20 cuentas para esta versión")
    # El campo persistido de versiones anteriores era la banda de parada.
    # Se migra en memoria y se escribe en el siguiente checkpoint.
    return replace(result, tolerance_counts=CONTROL_WINDOW_COUNTS)


def save(path: str | Path, calibration: Calibration) -> None:
    """Sustitución atómica en el mismo volumen, también para checkpoints."""
    if calibration.schema_version not in (SESSION_SCHEMA_VERSION, SCHEMA_VERSION):
        raise ValueError("Sólo se guardan calibraciones con estado de sesión")
    calibration.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=".calibration-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            payload = asdict(calibration)
            if calibration.schema_version == SESSION_SCHEMA_VERSION:
                del payload["calibration_points"]
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
