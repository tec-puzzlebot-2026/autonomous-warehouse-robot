"""Measured geometry and pure perception helpers for physical box pickup.

Distances are metres except fields ending in _mm. IR calibration lives in the
Tang bitstream and is intentionally not altered here.
"""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PickConfig:
    fork_tip_height_at_zero_mm: float
    fork_reach_m: float
    ir1_forward_offset_m: float
    box_width_m: float
    box_depth_m: float
    box_entry_gap_mm: float
    fork_thickness_mm: float
    base_half_width_m: float
    rear_overhang_m: float
    lift_max_height_mm: float
    floor_level_mm: float
    first_level_mm: float
    fork_clearance_mm: float = 3.0
    insertion_depth_m: float = .085
    lift_travel_mm: float = 31.0
    min_pick_distance_m: float = .27
    max_pick_distance_m: float = .42
    max_lateral_error_m: float = .018
    rear_clearance_m: float = .10
    front_clearance_m: float = .045
    max_drive_speed_mps: float = .045
    max_reverse_speed_mps: float = .045
    observation_seconds: float = 4.0
    observation_spread_m: float = .018
    box_width_tolerance_m: float = .045
    base_frame: str = "base_link"
    cmd_vel_topic: str = "/cmd_vel_in"
    odom_topic: str = "/wheel/odom"
    scan_topic: str = "/laser_scan"
    image_topic: str = "/camera/image_raw"
    camera_info_topic: str = "/camera/camera_info"
    hsv_lower: tuple = (8, 27, 55)
    hsv_upper: tuple = (40, 235, 255)
    vision_mode: str = "bands"
    static_transforms: dict | None = None
    face_source: str = "ir"
    carry_sensor: str = "ir"
    allow_vertical_crop: bool = False
    lift_tolerance_mm: float = 3.0
    check_ir_during_insertion: bool = True
    min_reverse_speed_mps: float = .020

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # Dataclasses use MISSING for both absent default and default_factory.
        from dataclasses import MISSING
        required = [f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING]
        missing = [name for name in required if data.get(name) is None]
        if missing:
            raise ValueError("Mide y rellena: " + ", ".join(missing))
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError("Campos desconocidos: " + ", ".join(unknown))
        for key in ("hsv_lower", "hsv_upper"):
            if key in data:
                data[key] = tuple(data[key])
        config = cls(**data)
        config.validate()
        return config

    def validate(self):
        numeric = [getattr(self, f.name) for f in fields(self)
                   if isinstance(getattr(self, f.name), (int, float))]
        if not all(map(math.isfinite, numeric)):
            raise ValueError("La geometría contiene un número no finito")
        positive = ("fork_reach_m", "box_width_m", "box_depth_m", "box_entry_gap_mm",
                    "fork_thickness_mm", "base_half_width_m", "rear_overhang_m",
                    "lift_max_height_mm", "insertion_depth_m", "lift_travel_mm",
                    "min_pick_distance_m", "max_pick_distance_m", "max_lateral_error_m",
                    "rear_clearance_m", "front_clearance_m", "max_drive_speed_mps",
                    "max_reverse_speed_mps", "observation_seconds", "observation_spread_m",
                    "box_width_tolerance_m", "lift_tolerance_mm")
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("Las longitudes, velocidades y tiempos deben ser positivos")
        if not 0 <= self.floor_level_mm < self.first_level_mm < self.lift_max_height_mm:
            raise ValueError("Niveles físicos fuera del recorrido")
        if not self.min_pick_distance_m < self.max_pick_distance_m <= 1:
            raise ValueError("Ventana de toma inválida")
        if not 0 < self.min_reverse_speed_mps <= self.max_reverse_speed_mps:
            raise ValueError("Velocidad mínima de retroceso inválida")
        if max(self.max_drive_speed_mps, self.max_reverse_speed_mps) > .10:
            raise ValueError("Velocidad máxima de prueba: 0.10 m/s")
        if not 0 < self.insertion_depth_m < .25 or not 0 < self.lift_travel_mm <= 60:
            raise ValueError("Inserción o elevación fuera del intervalo de prueba")
        if self.insertion_depth_m >= self.box_depth_m:
            raise ValueError("La inserción supera la profundidad de la caja")
        if self.fork_thickness_mm+self.fork_clearance_mm >= self.box_entry_gap_mm:
            raise ValueError("Las horquillas no caben en el hueco de entrada medido")
        if not 0 <= self.fork_clearance_mm <= 15:
            raise ValueError("Holgura de horquillas fuera de 0–15 mm")
        carry_range = self.fork_reach_m-self.insertion_depth_m-self.ir1_forward_offset_m
        if self.carry_sensor == "ir" and not .12 <= carry_range <= .75:
            raise ValueError("IR1 quedaría fuera de rango al insertar; ajusta geometría/profundidad")
        if self.observation_seconds < 2:
            raise ValueError("Se requieren al menos 2 s de observación")
        if self.cmd_vel_topic not in ("/cmd_vel_in", "/cmd_vel"):
            raise ValueError("cmd_vel_topic debe ser /cmd_vel_in o /cmd_vel")
        if not self.base_frame or self.base_frame.startswith("/"):
            raise ValueError("base_frame inválido")
        for name in ("odom_topic", "scan_topic", "image_topic", "camera_info_topic"):
            if not getattr(self, name).startswith("/"):
                raise ValueError(f"{name} requiere nombre absoluto")
        for hsv in (self.hsv_lower, self.hsv_upper):
            if len(hsv) != 3 or any(not isinstance(x, int) or not 0 <= x <= 255 for x in hsv):
                raise ValueError("HSV inválido")
        if any(a > b for a, b in zip(self.hsv_lower, self.hsv_upper)):
            raise ValueError("hsv_lower debe ser <= hsv_upper")
        if self.face_source not in ("ir", "rgb") or self.carry_sensor not in ("ir", "lidar"):
            raise ValueError("Fuentes de cara o confirmación inválidas")
        if self.vision_mode not in ("bands", "solid"):
            raise ValueError("vision_mode debe ser bands o solid")
        if self.static_transforms is not None:
            if not isinstance(self.static_transforms, dict):
                raise ValueError("static_transforms debe ser un objeto JSON")
            for frame, pose in self.static_transforms.items():
                if (not frame or not isinstance(pose, dict) or set(pose) != {"translation", "rotation"} or
                    not isinstance(pose["translation"], list) or not isinstance(pose["rotation"], list) or
                    len(pose["translation"]) != 3 or len(pose["rotation"]) != 4):
                    raise ValueError(f"TF estática inválida para {frame}")
                if not all(isinstance(x, (int, float)) and math.isfinite(x)
                           for x in pose["translation"] + pose["rotation"]):
                    raise ValueError(f"TF estática no finita para {frame}")
                if sum(x*x for x in pose["rotation"]) < 1e-6:
                    raise ValueError(f"TF estática sin rotación para {frame}")

    def insert_height_mm(self, level_mm):
        goal = level_mm + self.fork_clearance_mm - self.fork_tip_height_at_zero_mm
        if not 0 <= goal <= self.lift_max_height_mm or goal+self.lift_travel_mm > self.lift_max_height_mm:
            raise ValueError("La altura de inserción/transporte excede la carrera del elevador")
        return goal


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec*1e-9


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def pose_matrix(t):
    p, q = t.translation, t.rotation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-6 or not np.isfinite(norm) or not all(map(math.isfinite, (p.x, p.y, p.z))):
        raise ValueError("TF inválida")
    x, y, z, w = quat/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), p.x],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), p.y],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y), p.z],
                     [0, 0, 0, 1]], dtype=float)


def image_to_bgr(msg):
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}
    n = channels.get(msg.encoding)
    if n is None or msg.width < 8 or msg.height < 8 or msg.step < msg.width*n:
        raise ValueError(f"Imagen no admitida: {msg.encoding}")
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if len(raw) < msg.height*msg.step:
        raise ValueError("Imagen ROS truncada")
    pixels = raw[:msg.height*msg.step].reshape(msg.height, msg.step)[:, :msg.width*n]
    pixels = pixels.reshape(msg.height, msg.width, n)
    conversions = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
                   "bgra8": cv2.COLOR_BGRA2BGR}
    return np.ascontiguousarray(pixels if msg.encoding == "bgr8" else
                                cv2.cvtColor(pixels, conversions[msg.encoding]))


def find_box(frame, cfg):
    """Configurable version of the ZIP's warm-band E80 image detector."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    warm = cv2.inRange(hsv, cfg.hsv_lower, cfg.hsv_upper)
    warm[:h//12] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w//80), max(5, h//10)))
    regions = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(regions)
    candidates = []
    for x, y, width, height, area in stats[1:]:
        if height < .065*h or width < .06*w or not .32 <= width/height <= 2.9 or width > .91*w:
            continue
        if area < .18*width*height:
            continue
        inset = max(1, width//4)
        rows = np.mean(warm[y:y+height, x+inset:x+width-inset] != 0, axis=1)
        stripes = rows > .30
        runs = np.count_nonzero(stripes & ~np.r_[False, stripes[:-1]])
        if cfg.vision_mode == "bands" and (runs < 2 or not .07 <= np.mean(stripes) <= .83):
            continue
        candidates.append((width*height, (int(x), int(y), int(x+width), int(y+height))))
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[1][0] > .65*candidates[0][0]:
        return None
    return candidates[0][1] if candidates else None


def target_from_box(bounds, shape, camera_pose, intrinsics, forward, cfg):
    x1, y1, x2, y2 = bounds
    h, w = shape[:2]
    if x1 <= 3 or x2 >= w-3 or (not cfg.allow_vertical_crop and (y1 <= 3 or y2 >= h-3)):
        raise ValueError("Caja recortada en RGB")
    fx, fy, cx, cy = intrinsics
    if min(fx, fy) <= 0 or not all(map(math.isfinite, intrinsics)):
        raise ValueError("Intrínsecos RGB inválidos")
    v = (y1+y2)/2
    rays = [camera_pose[:3, :3] @ np.array([(u-cx)/fx, (v-cy)/fy, 1.])
            for u in (x1, (x1+x2)/2, x2)]
    if any(ray[0] <= .05 for ray in rays):
        raise ValueError("Cámara no orientada hacia la caja")
    if forward is None:
        width_per_m = abs(rays[2][1]/rays[2][0] - rays[0][1]/rays[0][0])
        if width_per_m < 1e-6:
            raise ValueError("RGB sin ancho frontal resoluble")
        forward = camera_pose[0, 3] + cfg.box_width_m / width_per_m
    points = []
    for ray in rays:
        scale = (forward-camera_pose[0, 3])/ray[0]
        if scale <= 0:
            raise ValueError("Caja detrás de la cámara")
        points.append(camera_pose[:3, 3]+scale*ray)
    width = abs(points[2][1]-points[0][1])
    if abs(width-cfg.box_width_m) > cfg.box_width_tolerance_m:
        raise ValueError(f"Ancho RGB {width:.3f} m difiere del medido")
    lateral = float(points[1][1])
    if abs(lateral) > cfg.max_lateral_error_m:
        raise ValueError(f"Caja desalineada {1000*lateral:+.0f} mm")
    return lateral, width, forward


def scan_points(scan, laser_pose):
    if not 0 < scan.angle_increment <= math.tau or not 0 < scan.range_min < scan.range_max:
        raise ValueError("LiDAR sin geometría válida")
    ranges = np.asarray(scan.ranges, dtype=float)
    angles = scan.angle_min+np.arange(len(ranges))*scan.angle_increment
    valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
    n = np.count_nonzero(valid)
    coords = np.stack((ranges[valid]*np.cos(angles[valid]), ranges[valid]*np.sin(angles[valid]),
                       np.zeros(n), np.ones(n)))
    return (laser_pose @ coords).T[:, :2]


def rear_clearance(scan, laser_pose, cfg):
    angles = scan.angle_min+np.arange(len(scan.ranges))*scan.angle_increment
    directions = laser_pose[:2, :2] @ np.stack((np.cos(angles), np.sin(angles)))
    rear = (directions[0] < -.7) & (np.abs(directions[1]) <= -directions[0])
    ranges = np.asarray(scan.ranges, dtype=float)
    if np.count_nonzero(rear) < 5 or np.mean(~np.isnan(ranges[rear]) & (ranges[rear] >= scan.range_min)) < .9:
        return None
    if np.any(rear & np.isfinite(ranges) & (ranges < scan.range_min)):
        return 0.
    points = scan_points(scan, laser_pose)
    corridor = points[(points[:, 0] < -cfg.rear_overhang_m) &
                      (np.abs(points[:, 1]) <= cfg.base_half_width_m)]
    return float(np.min(-corridor[:, 0]-cfg.rear_overhang_m)) if len(corridor) else math.inf


def front_obstacle(points, predicted_face, cfg):
    corridor = points[(np.abs(points[:, 1]) <= cfg.base_half_width_m*.7) & (points[:, 0] > .09)]
    return bool(len(corridor) and np.any(corridor[:, 0] < predicted_face-cfg.front_clearance_m))
