"""Tramas UART ASCII del bitstream ControlForklift."""

from __future__ import annotations

from dataclasses import dataclass


def checksum(body: bytes) -> int:
    result = 0
    for value in body:
        result ^= value
    return result


def simple(command: str) -> bytes:
    if command not in {"M", "D", "Z", "H", "W", "S", "X"}:
        raise ValueError("Comando simple desconocido")
    return command.encode("ascii") + b"\n"


def signed_frame(command: str, value: int) -> bytes:
    """A fija B en una cuenta absoluta; I guarda el identificador de sesión."""
    if command not in {"A", "I"} or not -2**31 <= value < 2**32:
        raise ValueError("Comando o valor fuera de rango")
    body = f"{command}{value & 0xFFFFFFFF:08X}".encode("ascii")
    return body + f"{checksum(body):02X}\n".encode("ascii")


def config(span: int, margin: int, tolerance: int, bottom_window: int = 0) -> bytes:
    if (not -2**31 <= span < 2**31 or not 0 <= margin <= 0xFFFF or
            not 0 <= tolerance <= 0xFFFF or not 0 <= bottom_window <= 0xFFFF):
        raise ValueError("Valor fuera del ancho de trama")
    body = f"K{span & 0xFFFFFFFF:08X}{margin:04X}{tolerance:04X}{bottom_window:04X}".encode("ascii")
    return body + f"{checksum(body):02X}\n".encode("ascii")


def goal(counts: int, arrival_counts: int) -> bytes:
    if not 0 <= counts <= 0xFFFFFFFF or not 1 <= arrival_counts <= 0xFFFF:
        raise ValueError("Meta o ventana fuera del ancho de trama")
    body = f"G{counts:08X}{arrival_counts:04X}".encode("ascii")
    return body + f"{checksum(body):02X}\n".encode("ascii")


@dataclass(frozen=True)
class Status:
    mode: str
    moving: bool
    span: int
    goal: int
    busy: bool = False
    extended: bool = False
    fault_code: int = 0


@dataclass(frozen=True)
class Position:
    raw: int
    relative: int | None = None


@dataclass(frozen=True)
class Session:
    token: int


@dataclass(frozen=True)
class IRReading:
    fresh: bool
    status: int | None
    distance_mm: int | None
    adc: int | None
    sensor: int = 1


def _signed32(value: int) -> int:
    return value - 2**32 if value & 0x80000000 else value


def _upper_hex(value: bytes) -> bool:
    return all(character in b"0123456789ABCDEF" for character in value)


class Parser:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._raw: int | None = None

    def feed(self, data: bytes) -> list[Status | Position | Session | IRReading]:
        events: list[Status | Position | Session | IRReading] = []
        for value in data:
            if value == 10:
                line = bytes(self._buffer)
                self._buffer.clear()
                try:
                    if len(line) == 9 and line[:1] == b"@" and _upper_hex(line[1:]):
                        self._raw = _signed32(int(line[1:], 16))
                        events.append(Position(self._raw))
                    elif len(line) == 9 and line[:1] == b"#" and self._raw is not None and _upper_hex(line[1:]):
                        events.append(Position(self._raw, _signed32(int(line[1:], 16))))
                    elif len(line) == 9 and line[:1] == b"$" and _upper_hex(line[1:]):
                        events.append(Session(int(line[1:], 16)))
                    elif (len(line) == 10 and line[:1] in (b"?", b"&") and
                          line[1:2] in (b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"F") and
                          _upper_hex(line[2:])):
                        status = None if line[1:2] == b"F" else int(line[1:2], 16)
                        mm = int(line[2:6], 16)
                        adc = int(line[6:10], 16)
                        if status is None:
                            if mm == 0xFFFF and adc == 0xFFFF:
                                events.append(IRReading(False, None, None, None,
                                                        1 if line[:1] == b"?" else 2))
                        elif (adc <= 1023 and
                              ((status == 0 and 120 <= mm <= 800) or
                               (status == 1 and mm == 120) or
                               (status == 2 and mm in (500, 800)) or
                               (status in (3, 4, 5, 6) and mm == 0xFFFF))):
                            events.append(IRReading(True, status,
                                                    None if mm == 0xFFFF else mm, adc,
                                                    1 if line[:1] == b"?" else 2))
                    elif (len(line) in (19, 20, 21) and line[:1] == b"!" and
                          line[1:2] in b"DMCRF" and line[2:3] in (b"0", b"1") and
                          (len(line) == 19 or line[3:4] in (b"0", b"1")) and
                          _upper_hex(line[3:] if len(line) == 19 else line[4:])):
                        offset = len(line) - 19
                        events.append(Status(chr(line[1]), line[2:3] == b"1",
                                             _signed32(int(line[3+offset:11+offset], 16)),
                                             int(line[11+offset:19+offset], 16),
                                             len(line) >= 20 and line[3:4] == b"1",
                                             len(line) == 21,
                                             int(line[4:5], 16) if len(line) == 21 else 0))
                except ValueError:
                    pass
            elif len(self._buffer) < 64:
                self._buffer.append(value)
            else:
                self._buffer.clear()
        return events
