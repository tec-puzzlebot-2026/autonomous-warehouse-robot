"""Acceso exclusivo al puerto serial de la Tang."""

from __future__ import annotations

import serial
from .protocol import IRReading, Parser, Position, Session, Status


DEFAULT_PORT = "/dev/serial/by-id/usb-SIPEED_JTAG_Debugger_FactoryAIOT_Pro-if01-port0"


class MotorLink:
    def __init__(self, port: str = DEFAULT_PORT) -> None:
        self.serial = serial.Serial(port, 115200, timeout=0, write_timeout=0.2, exclusive=True)
        self.parser = Parser()

    def send(self, frame: bytes) -> None:
        self.serial.write(frame)

    def poll(self) -> list[Status | Position | Session | IRReading]:
        return self.parser.feed(self.serial.read(self.serial.in_waiting or 1))

    def close(self) -> None:
        try:
            self.serial.write(b"X\nD\n")
            self.serial.flush()
        except serial.SerialException:
            pass
        finally:
            self.serial.close()
