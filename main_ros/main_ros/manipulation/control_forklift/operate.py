"""Nodo ROS 2: metas en mm y supervisión del elevador."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int32, String
from std_srvs.srv import Trigger

from .calibration import CONTROL_WINDOW_COUNTS
from .client import ControllerClient
from .link import DEFAULT_PORT
from .protocol import IRReading, Position, Status, config, goal, signed_frame, simple


class ForkliftNode(Node):
    def __init__(self, port: str, calibration_file: str,
                 settle_seconds: float = 0.6, socket_path: str | None = None, *,
                 client_factory=ControllerClient, namespace: str | None = None) -> None:
        super().__init__("control_forklift", namespace=namespace)
        self.link = client_factory(port, calibration_file, "operar", socket_path)
        self.calibration = self.link.calibration
        self.calibration.validate()
        if not math.isfinite(settle_seconds) or settle_seconds <= 0:
            raise ValueError("El tiempo de estabilización debe ser positivo y finito")
        self.arrival_tolerance_counts = CONTROL_WINDOW_COUNTS
        self.settle_seconds = settle_seconds
        self.mode = None
        self.moving = False
        self.busy = False
        self.last_status = 0.0
        self.latest_raw = None
        self.latest_raw_at = 0.0
        self.reference_sent_at = None
        self.expected_goal = None
        self.requested_mm = None
        self.goal_sent_at = None
        self.goal_acknowledged = False
        self.stopped_since = None
        self.arrived_goal = None
        self.drift_since = None
        self.fault = None
        self.last_recoverable_code = 0
        self.recoverable_reason = None
        self.ready_once = False
        self.startup_time = time.monotonic()
        self.latest_relative = None
        self.latest_relative_at = 0.0
        self.current_pub = self.create_publisher(Float64, "forklift/current_height_mm", 10)
        self.state_pub = self.create_publisher(String, "forklift/state", 10)
        self.ir_distance_pub = self.create_publisher(Float64, "forklift/ir_distance_mm", 10)
        self.ir_adc_pub = self.create_publisher(Int32, "forklift/ir_adc", 10)
        self.ir_state_pub = self.create_publisher(String, "forklift/ir_state", 10)
        self.ir_sensor_pubs = {
            sensor: (
                self.create_publisher(Float64, f"forklift/ir{sensor}_distance_mm", 10),
                self.create_publisher(Int32, f"forklift/ir{sensor}_adc", 10),
                self.create_publisher(String, f"forklift/ir{sensor}_state", 10),
            ) for sensor in (1, 2)
        }
        self.create_subscription(Float64, "forklift/target_height_mm", self.on_target, 10)
        self.create_service(Trigger, "forklift/stop", self.stop)
        self.create_timer(0.04, self.tick)
        self.link.send(simple("D"))
        self.link.send(config(self.calibration.signed_span_counts,
                              self.calibration.margin_counts,
                              self.arrival_tolerance_counts,
                              self.arrival_tolerance_counts))
        self.get_logger().info("Configuración enviada. La B anclada se cargará automáticamente; espera READY_STOPPED.")
        self.get_logger().info(
            f"Ventana única: ±{self.arrival_tolerance_counts} cuentas para parada "
            f"y llegada, en B y en cualquier altura. Verificación ROS "
            f"{self.settle_seconds:g} s después de estabilizarse.")

    def publish_state(self, state: str) -> None:
        self.state_pub.publish(String(data=state))

    def fail(self, message: str) -> None:
        if self.fault is not None:
            return
        self.fault = message
        self.get_logger().error(message)
        self.publish_state("FAULT: " + message)
        try:
            self.link.send(simple("D"))
        except Exception:
            pass

    def goal_not_reached(self, reason: str, cancel_hold: bool = False) -> None:
        """Detiene una meta sin perder la B cargada ni bloquear nuevas órdenes."""
        if cancel_hold:
            try:
                self.link.send(simple("X"))
            except Exception as exc:
                self.fail(f"No se pudo cancelar la meta: {exc}")
                return
        self.get_logger().warn(reason)
        self.recoverable_reason = reason
        self.publish_state("GOAL_NOT_REACHED: " + reason)
        self.expected_goal = None
        self.requested_mm = None
        self.goal_sent_at = None
        self.goal_acknowledged = False
        self.stopped_since = None
        self.arrived_goal = None
        self.drift_since = None

    def stop(self, request, response):
        del request
        try:
            self.link.send(simple("X"))
            self.expected_goal = None
            self.requested_mm = None
            self.goal_sent_at = None
            self.goal_acknowledged = False
            self.stopped_since = None
            self.arrived_goal = None
            self.drift_since = None
            self.recoverable_reason = None
            self.publish_state("STOPPED")
            response.success = True
            response.message = "Parada solicitada a la FPGA"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.fail("No se pudo enviar la orden de parada")
        return response

    def on_target(self, message: Float64) -> None:
        if self.fault:
            self.get_logger().warn("Meta ignorada: falla de protección activa")
            return
        if self.mode != "R" or time.monotonic() - self.last_status > 0.8:
            self.get_logger().warn("Meta ignorada: plataforma sin referencia o sin telemetría reciente")
            return
        if self.busy or self.goal_sent_at is not None:
            self.get_logger().warn("Meta ignorada: hay un movimiento en curso")
            return
        try:
            target = self.calibration.mm_to_counts(message.data)
            arrival_counts = self.arrival_tolerance_counts
            if target != 0:
                direction = 1 if self.calibration.signed_span_counts > 0 else -1
                target += direction * (self.calibration.operational_bottom_counts -
                                       self.calibration.session_bottom_counts)
            if not 0 <= target <= self.calibration.max_goal_counts:
                raise ValueError("La meta queda fuera del recorrido seguro desde el cero operativo")
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        try:
            self.link.send(goal(target, arrival_counts))
        except Exception as exc:
            self.fail(f"No se pudo enviar la meta: {exc}")
            return
        self.expected_goal = target
        self.requested_mm = message.data
        self.goal_sent_at = time.monotonic()
        self.goal_acknowledged = False
        self.stopped_since = None
        self.arrived_goal = None
        self.drift_since = None
        self.recoverable_reason = None
        self.publish_state(f"GOAL_SENT {message.data:.2f} mm ({target} cuentas; "
                           f"ventana {arrival_counts} cuentas)")

    def tick(self) -> None:
        if self.fault:
            return
        try:
            now = time.monotonic()
            for event in self.link.poll():
                if isinstance(event, IRReading):
                    labels = ("OK", "LIMITE_12_CM", "LIMITE_80_CM",
                              "MENOR_12_CM", "MAYOR_RANGO",
                              "LECTURA_INESTABLE", "SIN_CALIBRACION")
                    label = labels[event.status] if event.fresh else "SIN_DATOS"
                    if event.fresh and event.status == 2 and event.distance_mm == 500:
                        label = "LIMITE_50_CM"
                    sensor_pubs = self.ir_sensor_pubs.get(event.sensor)
                    if sensor_pubs is None:
                        continue
                    distance_pub, adc_pub, state_pub = sensor_pubs
                    state_pub.publish(String(data=label))
                    if event.sensor == 1:
                        self.ir_state_pub.publish(String(data=label))
                    if event.fresh and event.adc is not None:
                        adc_pub.publish(Int32(data=event.adc))
                        if event.sensor == 1:
                            self.ir_adc_pub.publish(Int32(data=event.adc))
                    if event.fresh and event.status == 0 and event.distance_mm is not None:
                        distance_pub.publish(Float64(data=float(event.distance_mm)))
                        if event.sensor == 1:
                            self.ir_distance_pub.publish(Float64(data=float(event.distance_mm)))
                elif isinstance(event, Position):
                    if event.relative is None:
                        self.latest_raw = event.raw
                        self.latest_raw_at = now
                    if event.relative is not None and self.mode == "R":
                        self.latest_relative = event.relative
                        self.latest_relative_at = now
                        direction = 1 if self.calibration.signed_span_counts > 0 else -1
                        height_counts = direction * (event.raw - self.calibration.operational_bottom_counts)
                        self.current_pub.publish(Float64(data=self.calibration.counts_to_mm(height_counts)))
                elif isinstance(event, Status):
                    self.last_status = now
                    self.mode = event.mode
                    self.moving = event.moving
                    self.busy = event.busy
                    if event.mode == "F":
                        reason = {
                            1: "posición fuera del recorrido",
                            2: "encoder sin avance durante 0.5 s",
                            3: "encoder retrocedió durante el movimiento",
                            4: "falla de protección de movimiento",
                        }.get(event.fault_code, f"motivo desconocido ({event.fault_code})")
                        self.fail(f"Falla FPGA: {reason}; meta={self.requested_mm} mm, "
                                  f"objetivo={event.goal}, encoder={self.latest_relative}")
                        return
                    if event.mode == "R" and event.fault_code in (5, 6):
                        if self.last_recoverable_code != event.fault_code:
                            self.last_recoverable_code = event.fault_code
                            kind = ("La meta no convergió" if event.fault_code == 5 else
                                    "No se pudo mantener la altura")
                            error = (abs(self.latest_relative - event.goal)
                                     if self.latest_relative is not None else "sin lectura")
                            self.goal_not_reached(
                                f"{kind}: meta={self.requested_mm} mm, "
                                f"objetivo={event.goal}, encoder={self.latest_relative}, "
                                f"error={error} cuentas; admite una nueva meta")
                        elif self.goal_sent_at is None and self.recoverable_reason is not None:
                            self.publish_state("GOAL_NOT_REACHED: " + self.recoverable_reason)
                        return
                    if event.fault_code == 0:
                        self.last_recoverable_code = 0
                    if self.ready_once and event.mode != "R":
                        self.fail(f"La FPGA salió de R (modo {event.mode}); revisa UART, latido y alimentación")
                        return
                    if not self.ready_once and event.mode == "D":
                        # Puede llegar una trama antigua después de K. Sólo se acepta durante el arranque.
                        self.publish_state("WAIT_CONFIG")
                    elif event.mode == "C":
                        if event.span != self.calibration.signed_span_counts:
                            self.fail("La FPGA no confirmó el recorrido del JSON")
                            return
                        self.publish_state("CONFIGURED: cargando B guardada")
                    elif event.mode == "R":
                        if event.span != self.calibration.signed_span_counts:
                            self.fail("El recorrido de FPGA no coincide con el JSON")
                            return
                        self.ready_once = True
                        if self.arrived_goal is not None and event.busy and event.goal == self.arrived_goal:
                            self.expected_goal = self.arrived_goal
                            self.arrived_goal = None
                            self.goal_sent_at = now
                            self.goal_acknowledged = True
                            self.stopped_since = None
                            self.drift_since = None
                            self.publish_state("CORRECTING")
                        if self.goal_sent_at is not None and event.goal == self.expected_goal:
                            self.goal_acknowledged = True
                        if self.expected_goal is not None and event.goal != self.expected_goal:
                            self.publish_state("GOAL_PENDING")
                        elif event.moving:
                            self.stopped_since = None
                            self.publish_state("MOVING")
                        elif event.busy:
                            self.stopped_since = None
                            self.publish_state("SETTLING")
                        elif self.goal_sent_at is not None and self.goal_acknowledged:
                            if self.stopped_since is None:
                                self.stopped_since = now
                            if now - self.stopped_since < self.settle_seconds:
                                self.publish_state("SETTLING")
                            elif (self.latest_relative is None or
                                  self.latest_relative_at < self.goal_sent_at or
                                  now - self.latest_relative_at > 0.5):
                                if now - self.stopped_since > max(2.0, self.settle_seconds + 1.0):
                                    self.fail("Sin posición reciente para verificar la llegada")
                                    return
                                self.publish_state("WAIT_POSITION")
                            else:
                                error_counts = abs(self.latest_relative - self.expected_goal)
                                if error_counts <= self.arrival_tolerance_counts:
                                    if self.expected_goal == 0:
                                        if (self.latest_raw is None or
                                                abs(self.latest_raw - self.calibration.session_bottom_counts)
                                                > self.arrival_tolerance_counts):
                                            self.fail("Llegada a B fuera de la ventana del ancla")
                                            return
                                        self.calibration = self.link.set_operational_bottom(self.latest_raw)
                                    self.arrived_goal = self.expected_goal
                                    self.goal_sent_at = None
                                    self.stopped_since = None
                                    self.drift_since = None
                                    self.publish_state("ARRIVED")
                                else:
                                    self.goal_not_reached(
                                        "Motor detenido fuera de la tolerancia de llegada: "
                                        f"meta={self.expected_goal}, posición={self.latest_relative}, "
                                        f"error={error_counts} cuentas, "
                                        f"aceptado={self.arrival_tolerance_counts} cuentas",
                                        cancel_hold=True)
                                    return
                        elif self.arrived_goal is not None:
                            if (self.latest_relative is not None and
                                    now - self.latest_relative_at <= 0.5 and
                                    abs(self.latest_relative - self.arrived_goal) > self.arrival_tolerance_counts):
                                if self.drift_since is None:
                                    self.drift_since = now
                                if now - self.drift_since > 1.0:
                                    self.goal_not_reached(
                                        "La plataforma se alejó de la meta y la FPGA no corrigió",
                                        cancel_hold=True)
                                    return
                                self.publish_state("DRIFT_WAIT")
                            else:
                                self.drift_since = None
                                self.publish_state("ARRIVED")
                        else:
                            if self.recoverable_reason is not None:
                                self.publish_state("GOAL_NOT_REACHED: " + self.recoverable_reason)
                            else:
                                self.publish_state("READY_STOPPED")
            if (self.mode == "C" and self.latest_raw is not None and
                    now - self.latest_raw_at <= 0.5 and now - self.startup_time >= 0.75 and
                    (self.reference_sent_at is None or now - self.reference_sent_at >= 1.0)):
                direction = 1 if self.calibration.signed_span_counts > 0 else -1
                relative = direction * (self.latest_raw - self.calibration.session_bottom_counts)
                if not -self.arrival_tolerance_counts <= relative <= abs(self.calibration.signed_span_counts) + self.arrival_tolerance_counts:
                    self.fail(
                        f"Posición actual {self.latest_raw} cuentas, B anclada "
                        f"{self.calibration.session_bottom_counts}: altura relativa "
                        f"{self.calibration.counts_to_mm(relative):.2f} mm fuera de B/T. "
                        "Usa calibrar para volver dentro del recorrido; "
                        "recalibra sólo si la referencia física cambió.")
                    return
                self.link.send(signed_frame("A", self.calibration.session_bottom_counts))
                self.reference_sent_at = now
            if not self.ready_once and now - self.startup_time > 5.0:
                self.fail("La FPGA no confirmó la configuración o la B absoluta")
                return
            if (self.goal_sent_at is not None and not self.goal_acknowledged and
                    now - self.goal_sent_at > 1.0):
                self.fail("La FPGA no confirmó la meta")
                return
            if self.last_status and now - self.last_status > 1.0:
                self.fail("Se perdió la telemetría UART")
        except Exception as exc:
            self.fail(f"Error de puerto serial: {exc}")

    def destroy_node(self):
        try:
            self.link.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description="Nodo ROS 2 del elevador")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--socket", default=None)
    parser.add_argument("--settle-seconds", type=float, default=0.6,
                        help="Espera tras la parada del motor antes de verificar la posición final")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = None
    try:
        node = ForkliftNode(args.port, args.calibration, args.settle_seconds, args.socket)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
