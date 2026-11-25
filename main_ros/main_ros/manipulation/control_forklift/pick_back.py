"""Supervised Pick & Back for physical PuzzleBot and Tang Nano 9K.

Default mode only observes. --execute enables base and lift commands after
sensor, geometry and ROS control preflight. --wait-for-ready waits for a ROS
command before starting preflight. The `controlador` process owns the Tang
serial port, and `operar` publishes the two IR readings in millimetres.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import signal
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Transform, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

from .pick_geometry import (PickConfig, find_box, front_obstacle, image_to_bgr,
                            pose_matrix, rear_clearance, scan_points, stamp_seconds,
                            target_from_box, yaw)


class PhysicalPickBack(Node):
    def __init__(self, cfg: PickConfig, level_mm: float, execute: bool, *, ros_args=None,
                 transform_provider=None, wait_for_ready=False, completion_check=None):
        if wait_for_ready and not execute:
            raise ValueError("--wait-for-ready requiere --execute")
        super().__init__("pick_back_real", cli_args=ros_args)
        self.cfg, self.level_mm, self.execute = cfg, level_mm, execute
        self.completion_check = completion_check
        self.insert_height_mm = cfg.insert_height_mm(level_mm)
        self.transport_height_mm = self.insert_height_mm + cfg.lift_travel_mm
        self.drive = self.lift = self.stop_client = None
        self.stop_future = None
        self.stop_sent_at = 0.
        self.controls_ready = False
        self.status = self.create_publisher(String, "pick_back_real/status",
                                            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.result = self.create_publisher(String, "pick_back_real/result",
                                            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.debug = self.create_publisher(Image, "pick_back_real/debug_image", qos_profile_sensor_data)
        self.transform_provider = transform_provider
        if transform_provider is None:
            from tf2_ros import Buffer, TransformException, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.transform_error = TransformException
        self.samples, self.arrivals = {}, {}
        self.observations = deque(maxlen=2000)
        self.last_image_stamp = -1.
        self.start_at = self.phase_started = time.monotonic()
        self.last_notice = 0.
        self.phase, self.detail = "observando", "esperando sensores"
        self.origin = None
        self.target_forward = self.target_lateral = None
        self.travel = 0.
        self.last_forward = None
        self.lift_goal = None
        self.lift_sent_at = 0.
        self.lift_ack = False
        self.settle_started = None
        self.retreat_start = None
        self.carry_reference_mm = None
        self.carry_reference_backed = 0.
        self.carry_hits = self.static_hits = 0
        self.last_carry_sample = self.last_carry_echo = 0.
        self.load_confirmed = False
        for kind, topic, key, sensor in (
            (Odometry, cfg.odom_topic, "odom", True),
            (LaserScan, cfg.scan_topic, "scan", True),
            (Image, cfg.image_topic, "image", True),
            (CameraInfo, cfg.camera_info_topic, "camera_info", True),
            (Float64, "forklift/current_height_mm", "height", False),
            (String, "forklift/state", "lift_state", False),
            (Float64, "forklift/ir1_distance_mm", "ir1", False),
            (String, "forklift/ir1_state", "ir1_state", False),
            (Float64, "forklift/ir2_distance_mm", "ir2", False),
            (String, "forklift/ir2_state", "ir2_state", False),
        ):
            self.create_subscription(kind, topic, lambda msg, name=key: self.receive(name, msg),
                                     qos_profile_sensor_data if sensor else 10)
        if wait_for_ready:
            self.create_subscription(String, "pick_back_real/command", self.on_command, 10)
            self.announce("esperando_ready", "esperando Ready Pick Up")
        else:
            if execute:
                self.enable_controls()
            self.announce("observando", "modo diagnóstico" if not execute else "validando antes de mover")

    def enable_controls(self):
        self.drive = self.create_publisher(Twist, self.cfg.cmd_vel_topic, 10)
        self.lift = self.create_publisher(Float64, "forklift/target_height_mm", 10)
        self.stop_client = self.create_client(Trigger, "forklift/stop")

    def on_command(self, msg):
        if self.phase != "esperando_ready" or msg.data != "Ready Pick Up":
            return
        self.enable_controls()
        self.start_at = time.monotonic()
        self.announce("observando", "Validando sensores y mandos tras recibir Ready Pick Up")

    @property
    def finished(self):
        return self.phase in ("diagnostico", "completado", "abortado")

    def receive(self, key, msg):
        now = time.monotonic()
        self.samples[key], self.arrivals[key] = msg, now
        if key == "lift_state" and self.lift_goal is not None and now > self.lift_sent_at:
            if msg.data.startswith("GOAL_SENT "):
                self.lift_ack = msg.data.split()[1] == f"{self.lift_goal:.2f}"
            if msg.data.startswith(("FAULT", "GOAL_NOT_REACHED")) and not self.finished:
                self.abort("elevador: " + msg.data)

    def announce(self, phase, detail):
        failed_phase = self.phase if phase == "abortado" else None
        self.phase, self.detail = phase, detail
        self.phase_started, self.settle_started = time.monotonic(), None
        payload = {"estado": phase, "detalle": detail, "nivel_mm": self.level_mm,
                   "avance_m": round(self.travel, 4), "carga_confirmada": self.load_confirmed,
                   "ejecucion": self.execute}
        if failed_phase is not None:
            payload["fase_fallo"] = failed_phase
        self.status.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        if self.drive is not None:
            if phase == "completado":
                self.result.publish(String(data="Done"))
            elif phase == "abortado":
                self.result.publish(String(data=f"Error: {detail}"))
        self.get_logger().info(f"{phase}: {detail}")

    def fresh(self, key, age, stamped=False):
        msg = self.samples.get(key)
        if msg is None or time.monotonic()-self.arrivals[key] > age:
            return None
        if stamped:
            stamp = stamp_seconds(msg.header.stamp)
            now_ros = self.get_clock().now().nanoseconds*1e-9
            if stamp <= 0 or not -.15 <= now_ros-stamp <= age+.25:
                return None
        return msg

    def fixed_tf(self, frame, stamp=None):
        if not frame:
            raise ValueError("Sensor sin frame_id")
        if self.transform_provider is not None:
            return self.transform_provider(frame, stamp)
        try:
            when = Time.from_msg(stamp) if stamp is not None else Time()
            result = self.tf_buffer.lookup_transform(self.cfg.base_frame, frame, when)
            return pose_matrix(result.transform)
        except self.transform_error as exc:
            pose = (self.cfg.static_transforms or {}).get(frame)
            if pose is None:
                raise ValueError(f"falta TF {self.cfg.base_frame} <- {frame}: {exc}") from exc
            transform = Transform()
            transform.translation.x, transform.translation.y, transform.translation.z = pose["translation"]
            (transform.rotation.x, transform.rotation.y,
             transform.rotation.z, transform.rotation.w) = pose["rotation"]
            return pose_matrix(transform)

    def drive_command(self, linear=0., angular=0.):
        if self.drive is not None:
            msg = Twist()
            msg.linear.x, msg.angular.z = float(linear), float(angular)
            self.drive.publish(msg)

    def abort(self, reason):
        self.drive_command()
        if self.execute and self.lift_goal is not None and not self.finished:
            if self.stop_client.service_is_ready():
                self.stop_sent_at = time.monotonic()
                self.stop_future = self.stop_client.call_async(Trigger.Request())
            else:
                self.get_logger().error("forklift/stop no disponible al abortar")
        if not self.finished:
            self.announce("abortado", reason)

    def lift_command(self, height_mm):
        if not 0 <= height_mm <= self.cfg.lift_max_height_mm:
            raise ValueError("Meta de elevador fuera de carrera")
        self.lift_goal, self.lift_ack = height_mm, False
        self.lift_sent_at = time.monotonic()
        self.lift.publish(Float64(data=float(height_mm)))

    def lift_arrived(self):
        state, height = self.fresh("lift_state", .8), self.fresh("height", .8)
        return (self.lift_ack and state is not None and state.data == "ARRIVED" and
                self.arrivals["lift_state"] > self.lift_sent_at and height is not None and
                self.arrivals["height"] > self.lift_sent_at and math.isfinite(height.data) and
                abs(height.data-self.lift_goal) <= self.cfg.lift_tolerance_mm)

    def settled(self, condition, seconds):
        if not condition:
            self.settle_started = None
            return False
        if self.settle_started is None:
            self.settle_started = time.monotonic()
        return time.monotonic()-self.settle_started >= seconds

    def odom_metrics(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        heading, speed = yaw(q), msg.twist.twist.linear.x
        if not all(map(math.isfinite, (p.x, p.y, heading, speed))):
            raise ValueError("Odometría no finita")
        if self.origin is None:
            return 0., 0., 0., speed
        x, y, theta = self.origin
        dx, dy = p.x-x, p.y-y
        return (dx*math.cos(theta)+dy*math.sin(theta),
                -dx*math.sin(theta)+dy*math.cos(theta),
                math.remainder(heading-theta, math.tau), speed)

    def ir1_mm(self):
        state, distance = self.fresh("ir1_state", .65), self.fresh("ir1", .65)
        if state is None or distance is None or state.data != "OK":
            return None
        return distance.data if math.isfinite(distance.data) and 120 <= distance.data <= 800 else None

    def observe(self, odom, points):
        image, info = self.fresh("image", .8, True), self.fresh("camera_info", 10.)
        if image is None or info is None:
            self.detail = "esperando RGB y camera_info"
            return
        stamp = stamp_seconds(image.header.stamp)
        if stamp <= self.last_image_stamp:
            return
        self.last_image_stamp = stamp
        if info.header.frame_id != image.header.frame_id or (info.width, info.height) != (image.width, image.height):
            raise ValueError("camera_info no corresponde a la imagen")
        if abs(self.odom_metrics(odom)[3]) > .008:
            self.detail = "esperando base detenida"
            self.observations.clear()
            return
        forward = None
        if self.cfg.face_source == "ir":
            ir_mm = self.ir1_mm()
            if ir_mm is None:
                self.detail = "IR1 sin estado OK y distancia recientes"
                self.observations.clear()
                return
            forward = self.cfg.ir1_forward_offset_m+ir_mm/1000.
        frame = image_to_bgr(image)
        if any(info.d):
            frame = cv2.undistort(frame, np.asarray(info.k, float).reshape(3, 3), np.asarray(info.d, float))
        box = find_box(frame, self.cfg)
        if self.debug.get_subscription_count():
            annotated = frame.copy()
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated, (x1, y1), (x2-1, y2-1), (0, 255, 0), 2)
            self.debug.publish(Image(header=image.header, width=image.width, height=image.height,
                                     encoding="bgr8", step=image.width*3, data=annotated.tobytes()))
        if box is None:
            self.detail = "RGB sin caja E80 o detección ambigua; ajusta HSV"
            self.observations.clear()
            return
        try:
            camera = self.fixed_tf(image.header.frame_id, image.header.stamp)
            lateral, width, forward = target_from_box(box, frame.shape, camera,
                                             (info.k[0], info.k[4], info.k[2], info.k[5]),
                                             forward, self.cfg)
        except ValueError as exc:
            self.detail = str(exc)
            self.observations.clear()
            return
        if not self.cfg.min_pick_distance_m <= forward <= self.cfg.max_pick_distance_m:
            self.detail = f"distancia {forward:.3f} m fuera de ventana de toma"
            self.observations.clear()
            return
        if front_obstacle(points, forward, self.cfg):
            self.detail = "LiDAR detecta obstáculo antes de la caja"
            self.observations.clear()
            return
        now = time.monotonic()
        self.observations.append((now, forward, lateral))
        while self.observations and now-self.observations[0][0] > self.cfg.observation_seconds:
            self.observations.popleft()
        if len(self.observations) < 8 or now-self.observations[0][0] < .9*self.cfg.observation_seconds:
            self.detail = f"confirmando caja con {len(self.observations)} imágenes"
            return
        faces = np.array([row[1] for row in self.observations])
        laterals = np.array([row[2] for row in self.observations])
        if np.ptp(faces) > self.cfg.observation_spread_m or np.ptp(laterals) > .018:
            self.detail = "posición RGB/IR varía demasiado"
            return
        self.target_forward, self.target_lateral = float(np.median(faces)), float(np.median(laterals))
        insert_travel = self.target_forward-self.cfg.fork_reach_m+self.cfg.insertion_depth_m
        if not .12 <= insert_travel <= .30:
            self.detail = f"recorrido de inserción {insert_travel:.3f} m fuera de 0.12–0.30 m"
            self.observations.clear()
            return
        ir2_state, ir2 = self.fresh("ir2_state", .65), self.fresh("ir2", .65)
        ir2_note = "IR2 sin lectura OK"
        if ir2_state is not None and ir2 is not None and ir2_state.data == "OK":
            difference = self.cfg.ir1_forward_offset_m+ir2.data/1000.-self.target_forward
            ir2_note = f"IR2 difiere {1000*difference:+.0f} mm"
        self.origin = (odom.pose.pose.position.x, odom.pose.pose.position.y,
                       yaw(odom.pose.pose.orientation))
        detail = (f"caja {self.target_forward:.3f} m; lateral {1000*self.target_lateral:+.0f} mm; "
                  f"ancho RGB {1000*width:.0f} mm; {ir2_note}")
        if not self.execute:
            self.announce("diagnostico", detail)
            return
        self.lift_command(self.insert_height_mm)
        self.announce("ajustando_altura", detail)

    def lidar_face(self, points, expected):
        x = points[(abs(points[:, 1] - self.target_lateral) < .045) &
                   (abs(points[:, 0] - expected) <= .027), 0]
        return float(np.median(x)) if len(x) >= 4 else None

    def carry_check(self, backed, scan, points):
        if backed < .055:
            return
        if self.cfg.carry_sensor == "lidar":
            stamp = stamp_seconds(scan.header.stamp)
            if stamp <= self.last_carry_sample:
                return
            self.last_carry_sample = stamp
            if self.lidar_face(points, self.carry_reference_mm / 1000) is not None:
                self.carry_hits += 1
                self.static_hits = 0
                self.last_carry_echo = time.monotonic()
                self.load_confirmed = self.carry_hits >= 3
            else:
                self.carry_hits = 0
                if self.lidar_face(points, self.carry_reference_mm / 1000 + backed) is not None:
                    self.static_hits += 1
                else:
                    self.static_hits = 0
                if self.static_hits >= 3:
                    self.abort("LiDAR indica que la caja quedó fija")
            return
        value = self.ir1_mm()
        if value is None or self.arrivals["ir1"] <= self.last_carry_sample:
            return
        self.last_carry_sample = self.arrivals["ir1"]
        if self.carry_reference_mm is None:
            self.carry_reference_mm, self.carry_reference_backed = value, backed
            return
        delta = value-self.carry_reference_mm
        if abs(delta) <= 25:
            self.carry_hits, self.static_hits = self.carry_hits+1, 0
        elif abs(delta-1000*(backed-self.carry_reference_backed)) <= 35:
            self.static_hits, self.carry_hits = self.static_hits+1, 0
        else:
            self.carry_hits = self.static_hits = 0
        if self.carry_hits >= 3:
            self.load_confirmed = True
            self.last_carry_echo = time.monotonic()
        if self.static_hits >= 3:
            self.abort("IR1 indica que la caja quedó fija al retroceder")

    def tick(self):
        if self.finished or self.phase == "esperando_ready":
            return
        try:
            self.update()
        except (ValueError, cv2.error) as exc:
            self.abort(str(exc))
        if self.phase == "observando" and time.monotonic()-self.last_notice > 4:
            self.last_notice = time.monotonic()
            self.get_logger().info("observando: " + self.detail)

    def update(self):
        now = time.monotonic()
        limits = {"observando": 60, "ajustando_altura": 20, "insertando": 18,
                  "deteniendo": 4, "levantando": 20, "regresando": 25, "estabilizando": 4}
        if now-self.start_at > 120 or now-self.phase_started > limits[self.phase]:
            self.abort(f"tiempo agotado en {self.phase}: {self.detail}")
            return
        if self.execute:
            if self.count_publishers(self.drive.topic_name) > 1 or self.count_publishers(self.lift.topic_name) > 1:
                self.abort("otro publicador controla base o elevador")
                return
            ready = (self.drive.get_subscription_count() and self.lift.get_subscription_count()
                     and self.stop_client.service_is_ready())
            if not ready:
                if self.controls_ready:
                    self.abort("se perdió la comunicación con la base o el elevador")
                else:
                    self.detail = "esperando base, operar y servicio forklift/stop"
                return
            self.controls_ready = True
        odom, scan = self.fresh("odom", .6, True), self.fresh("scan", .6, True)
        height, state = self.fresh("height", .8), self.fresh("lift_state", .8)
        if any(x is None for x in (odom, scan, height, state)):
            if self.phase != "observando":
                self.abort("odometría, LiDAR o elevador sin datos vigentes")
            else:
                self.detail = "esperando odometría, LiDAR, altura y estado del elevador"
            return
        if not math.isfinite(height.data):
            raise ValueError("Altura del elevador no finita")
        if state.data.startswith(("FAULT", "GOAL_NOT_REACHED")):
            self.abort("elevador: " + state.data)
            return
        if self.phase == "observando" and state.data not in ("READY_STOPPED", "ARRIVED"):
            self.detail = "esperando elevador READY_STOPPED o ARRIVED"
            return
        if odom.child_frame_id != self.cfg.base_frame:
            raise ValueError(f"odometría child_frame_id={odom.child_frame_id}; esperado {self.cfg.base_frame}")
        try:
            laser = self.fixed_tf(scan.header.frame_id)
        except ValueError as exc:
            if self.phase == "observando":
                self.detail = str(exc)
            else:
                self.abort(str(exc))
            return
        points = scan_points(scan, laser)
        rear = rear_clearance(scan, laser, self.cfg)
        if rear is None:
            if self.phase == "observando":
                self.detail = "LiDAR sin cobertura trasera válida"
            else:
                self.abort("LiDAR sin cobertura trasera válida")
            return
        if self.phase == "observando":
            self.observe(odom, points)
            return
        if self.fresh("image", .8, True) is None:
            self.abort("cámara RGB sin imágenes vigentes")
            return
        forward, lateral, heading, speed = self.odom_metrics(odom)
        self.travel = forward
        if abs(lateral) > .018 or abs(heading) > math.radians(5):
            self.abort("el robot salió del corredor alineado")
            return
        if self.last_forward is not None and abs(forward-self.last_forward) > .04:
            self.abort("salto de odometría")
            return
        self.last_forward = forward
        straight = float(np.clip(-2.5*heading, -.10, .10))
        if self.phase == "ajustando_altura":
            self.drive_command()
            if self.settled(self.lift_arrived() and abs(speed) <= .008, .3):
                self.announce("insertando", f"horquillas a {self.insert_height_mm:.1f} mm de consigna")
        elif self.phase == "insertando":
            expected_face = self.target_forward-forward
            if (self.cfg.check_ir_during_insertion and self.cfg.face_source == "ir" and
                    expected_face-self.cfg.ir1_forward_offset_m >= .14):
                ir_mm = self.ir1_mm()
                if ir_mm is None or abs(self.cfg.ir1_forward_offset_m+ir_mm/1000.-expected_face) > .045:
                    self.abort("IR1 no confirma el avance hacia la caja")
                    return
            if front_obstacle(points, expected_face, self.cfg):
                self.abort("obstáculo frontal antes de la caja")
                return
            remaining = self.target_forward-self.cfg.fork_reach_m+self.cfg.insertion_depth_m-forward
            if remaining <= .003:
                self.drive_command()
                self.announce("deteniendo", "profundidad de inserción alcanzada")
                return
            if forward > self.target_forward+.01:
                self.abort("sobrepasó la cara estimada de la caja")
                return
            self.drive_command(min(self.cfg.max_drive_speed_mps, max(.018, 1.4*remaining)), straight)
        elif self.phase == "deteniendo":
            self.drive_command()
            if self.settled(abs(speed) <= .008, .3):
                self.lift_command(self.transport_height_mm)
                self.announce("levantando", f"subiendo a {self.transport_height_mm:.1f} mm")
        elif self.phase == "levantando":
            self.drive_command()
            if self.settled(self.lift_arrived() and abs(speed) <= .008, .35):
                expected_face = self.target_forward - forward
                if self.cfg.carry_sensor == "lidar":
                    measurement = self.lidar_face(points, expected_face)
                    measured = None if measurement is None else 1000 * measurement
                else:
                    measured = self.ir1_mm()
                    if measured is not None and abs(self.cfg.ir1_forward_offset_m+measured/1000.-expected_face) > .045:
                        measured = None
                if measured is None:
                    self.abort(f"{self.cfg.carry_sensor} no confirma la cara de la caja antes del retroceso")
                    return
                self.retreat_start = forward
                self.carry_reference_mm = measured
                self.carry_reference_backed = 0.
                self.announce("regresando", "elevador estabilizado, comprobando espacio trasero")
        elif self.phase == "regresando":
            if forward <= .005:
                self.drive_command()
                self.announce("estabilizando", "comprobando pose inicial y carga")
                return
            if rear <= self.cfg.rear_clearance_m:
                self.abort(f"obstáculo trasero a {rear:.3f} m del borde")
                return
            self.carry_check(self.retreat_start-forward, scan, points)
            if not self.finished:
                self.drive_command(-min(self.cfg.max_reverse_speed_mps, max(self.cfg.min_reverse_speed_mps, 1.4*forward)), straight)
        else:
            self.drive_command()
            if abs(forward) > .012 or abs(lateral) > .012 or abs(heading) > math.radians(4):
                self.abort("pose final fuera de tolerancia")
                return
            self.carry_check(self.retreat_start-forward, scan, points)
            if self.finished:
                return
            if self.settled(abs(speed) <= .008, .35):
                if not self.load_confirmed or now-self.last_carry_echo > .8:
                    self.abort("regresó sin confirmación reciente de carga")
                else:
                    if self.completion_check is not None:
                        self.completion_check()
                    self.announce("completado", f"regresó al inicio; carga confirmada por {self.cfg.carry_sensor}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON con dimensiones medidas del robot")
    parser.add_argument("--level", choices=("floor", "first"), required=True)
    parser.add_argument("--execute", action="store_true", help="Autoriza comandos de base y elevador")
    parser.add_argument("--wait-for-ready", action="store_true",
                        help="Espera Ready Pick Up en pick_back_real/command antes de empezar")
    ros_args = sys.argv if argv is None else ["pick_back_real", *argv]
    args = parser.parse_args(remove_ros_args(ros_args)[1:])
    if args.wait_for_ready and not args.execute:
        parser.error("--wait-for-ready requiere --execute")
    try:
        cfg = PickConfig.load(args.config)
        level = cfg.floor_level_mm if args.level == "floor" else cfg.first_level_mm
        cfg.insert_height_mm(level)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = PhysicalPickBack(cfg, level, args.execute, wait_for_ready=args.wait_for_ready)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        while rclpy.ok() and not node.finished and not interrupted:
            executor.spin_once(timeout_sec=.04)
            node.tick()
        if interrupted:
            node.abort("cancelado por operador")
        return 130 if interrupted else 0 if node.phase in ("diagnostico", "completado") else 1
    finally:
        deadline = time.monotonic() + .3
        if node.stop_future is not None:
            deadline = node.stop_sent_at + 2.
        while rclpy.ok() and time.monotonic() < deadline:
            node.drive_command()
            executor.spin_once(timeout_sec=.04)
            if node.stop_future is not None and node.stop_future.done():
                break
        if node.stop_future is not None:
            if not node.stop_future.done():
                node.get_logger().error("La parada del elevador no recibió respuesta")
            elif not node.stop_future.result().success:
                node.get_logger().error("La parada del elevador fue rechazada")
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
