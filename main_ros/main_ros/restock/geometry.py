"""
geometry.py — camera, box and pose geometry for restocking (no ROS).

Everything the box task computes from an image or the map:

  * Camera: pixel -> ray in base_link, from the intrinsics and the measured mount
    (camera_mount in aruco_localization.yaml, the same convention as the EKF).
  * Box from one image: all boxes are the same size, so the box's height in pixels
    gives its distance; its bottom edge gives its level (floor / rack level 1 / 2).
  * Box faces: the four faces of a box on the map, the pose in front of each, and
    whether the robot can reach it and see the face from there.
  * QR association: a QR code belongs to the target box only if all four of its
    corners lie inside the box's outline in the image.

Poses are (x, y, theta) in the map frame unless stated otherwise.
"""
# Author: Raymundo Gomez

import math
from dataclasses import dataclass, field

import numpy as np

from main_ros.vision.aruco import handeye


def wrap(a):
    """Angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def compose(a, b):
    """Pose a (+) pose b."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def relative(a, b):
    """Pose of b in the frame of a."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2]))


def transform_point(pose, p):
    """Point p (x, y) in the frame of `pose` -> map."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (pose[0] + c * p[0] - s * p[1], pose[1] + s * p[0] + c * p[1])


# =============================================================================
# Camera
# =============================================================================
@dataclass
class Camera:
    """Pinhole camera with its mount on the robot.

    fx, fy, cx, cy: intrinsics of the image the detectors see [px].
    mount: (x, y, z, roll, pitch, yaw) of camera_link in base_link, as measured for
    the EKF (aruco_localization.yaml, camera_mount); z is above the wheel axle.
    base_height: height of base_link (the wheel axle) above the floor [m].
    """
    fx: float
    fy: float
    cx: float
    cy: float
    mount: tuple = (0.10, 0.0, 0.178, 0.0, 0.26, 0.0)
    base_height: float = 0.047
    T: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.T = handeye.mount_to_X(*self.mount)      # base_link <- optical

    @classmethod
    def from_camera_info(cls, k, mount, base_height=0.047, p=None):
        """From sensor_msgs/CameraInfo K (9 values) or, for a rectified image, P (12)."""
        if p is not None and len(p) == 12 and p[0] > 0:
            return cls(p[0], p[5], p[2], p[6], tuple(mount), base_height)
        return cls(k[0], k[4], k[2], k[5], tuple(mount), base_height)

    def ray_base(self, u, v):
        """Unit direction in base_link of the ray through pixel (u, v), and the origin."""
        d_opt = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        d = self.T[:3, :3] @ d_opt
        return self.T[:3, 3].copy(), d / np.linalg.norm(d)

    def point_base(self, u, v, depth):
        """3D point in base_link at optical depth `depth` [m] along pixel (u, v)."""
        p_opt = np.array([(u - self.cx) / self.fx * depth, (v - self.cy) / self.fy * depth,
                          depth, 1.0])
        return (self.T @ p_opt)[:3]

    def bearing(self, u, v=None):
        """Horizontal bearing [rad] in base_link of pixel (u, v) (left = positive).

        With the camera pitched down, an image column is not a constant bearing: give v
        (the row of the object) for an exact value; without it the center row is used.
        """
        _, d = self.ray_base(u, self.cy if v is None else v)
        return math.atan2(d[1], d[0])


# =============================================================================
# Box from one image
# =============================================================================
@dataclass
class BoxSpec:
    """Box size [m] and rack level heights (top surface of each level above the floor)."""
    width: float = 0.12          # the face the forklift picks (and the QR faces)
    depth: float = 0.10
    height: float = 0.12
    level_heights: tuple = (0.0, 0.19, 0.38)


@dataclass
class BoxSighting:
    """Where one camera detection puts the box (all in the map frame)."""
    x: float                     # box center
    y: float
    bottom_z: float              # height of the box's bottom above the floor [m]
    level: int                   # 0 floor, 1, 2
    distance: float              # horizontal distance robot -> visible face [m]
    bearing: float               # in the robot frame [rad]


def locate_box(cam, robot_pose, image_box, spec=BoxSpec()):
    """Box position and level from its image bounding box [x1, y1, x2, y2] [px].

    The box is 12 cm tall whichever face is seen, so its height in pixels gives the
    optical depth (pinhole: depth = fy * H / h_px). The bottom edge's pixel then
    gives the height of the box's bottom above the floor, i.e. its level.
    """
    x1, y1, x2, y2 = [float(v) for v in image_box]
    h_px = max(y2 - y1, 1.0)
    depth = cam.fy * spec.height / h_px
    u_mid = 0.5 * (x1 + x2)
    face = cam.point_base(u_mid, 0.5 * (y1 + y2), depth)
    bottom = cam.point_base(u_mid, y2, depth)
    bottom_z = float(bottom[2] + cam.base_height)
    level = int(np.argmin([abs(bottom_z - h) for h in spec.level_heights]))
    dist = math.hypot(face[0], face[1])
    bearing = math.atan2(face[1], face[0])
    # The visible face is half a box in front of the center.
    center_robot = ((dist + 0.5 * spec.depth) * math.cos(bearing),
                    (dist + 0.5 * spec.depth) * math.sin(bearing))
    cx, cy = transform_point(robot_pose, center_robot)
    return BoxSighting(cx, cy, bottom_z, level, dist, bearing)


def quad_normal_yaw(cam, corners):
    """Yaw [rad, base_link] of the outward normal (toward the camera) of a flat square
    seen as the 4 image corners `corners` (a QR code on a box face), or None.

    A square's pose from its 4 corners (OpenCV IPPE) gives its plane whatever its size;
    the size only scales the distance, which is not used here.
    """
    import cv2
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != 4:
        return None
    s = 0.05
    obj = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])
    K = np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]])
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, pts, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]
    if float(n @ tvec.ravel()) > 0.0:          # make it point back at the camera
        n = -n
    nb = cam.T[:3, :3] @ n
    if math.hypot(nb[0], nb[1]) < 0.3:         # a face seen from above: no yaw
        return None
    return math.atan2(nb[1], nb[0])


def point_at_height(cam, u, v, z_floor):
    """base_link (x, y) where the ray of pixel (u, v) is z_floor above the floor, or
    None (the ray never gets there in front of the camera). With a box's bottom edge
    and its level's height this gives the distance to the box even when its top is cut
    off by the image border (a rack box close to the camera)."""
    o = cam.point_base(u, v, 0.0)
    r = cam.point_base(u, v, 1.0) - o
    dz = z_floor - cam.base_height - o[2]
    if abs(r[2]) < 1e-6 or dz / r[2] <= 0.0:
        return None
    d = dz / r[2]
    return float(o[0] + d * r[0]), float(o[1] + d * r[1])


def face_center_offset(cam, image_box, forward, spec=BoxSpec(), frame_w=None):
    """Sideways offset [m, + = left] of a face's center from base_link's x axis, for a
    face `forward` m ahead (base_link x; the LiDAR fit when the robot is parallel to it).

    Each image column is a ray; it meets the plane x = forward at a known y. A box edge
    cut by the image border is no edge, so then the other edge and the face's known
    width (12 cm) give the center.
    """
    x1, y1, x2, y2 = [float(v) for v in image_box[:4]]
    w = frame_w if frame_w else 2.0 * cam.cx
    v = 0.5 * (y1 + y2)

    def y_at(u):
        o = cam.point_base(u, v, 0.0)
        r = cam.point_base(u, v, 1.0) - o
        if abs(r[0]) < 1e-6:
            return None
        return float(o[1] + (forward - o[0]) / r[0] * r[1])

    left_cut, right_cut = x1 <= 2, x2 >= w - 2
    half = spec.width / 2
    if left_cut and right_cut:
        return None
    if left_cut:                     # image left = robot left (+y): the right edge is real
        yr = y_at(x2)
        return None if yr is None else yr + half
    if right_cut:
        yl = y_at(x1)
        return None if yl is None else yl - half
    yl, yr = y_at(x1), y_at(x2)
    return None if yl is None or yr is None else 0.5 * (yl + yr)


# =============================================================================
# Box faces on the map
# =============================================================================
@dataclass
class BoxFootprint:
    """A box on the map: center and the outward normal of its pick face."""
    x: float
    y: float
    yaw: float                   # direction the pick face looks toward (out of the box)
    spec: BoxSpec = field(default_factory=BoxSpec)

    def faces(self):
        """The four faces: list of (name, center (x, y), outward normal angle)."""
        out = []
        for name, d_angle, half in (('front', 0.0, self.spec.depth / 2),
                                    ('left', math.pi / 2, self.spec.width / 2),
                                    ('back', math.pi, self.spec.depth / 2),
                                    ('right', -math.pi / 2, self.spec.width / 2)):
            a = wrap(self.yaw + d_angle)
            out.append((name, (self.x + half * math.cos(a), self.y + half * math.sin(a)), a))
        return out

    def cells(self, grid, margin=0.02):
        """Grid cells covered by the box (with a margin): (rows, cols)."""
        hw, hd = self.spec.width / 2 + margin, self.spec.depth / 2 + margin
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        cells = set()
        step = grid.res / 2
        for a in np.arange(-hd, hd + 1e-9, step):
            for b in np.arange(-hw, hw + 1e-9, step):
                x = self.x + c * a - s * b
                y = self.y + s * a + c * b
                cells.add(grid.world_to_cell(x, y))
        return cells


def pose_facing(point, normal, standoff):
    """Pose `standoff` m out along `normal` from `point`, facing the point."""
    return (point[0] + standoff * math.cos(normal), point[1] + standoff * math.sin(normal),
            wrap(normal + math.pi))


def face_viewpoints(box, grid, traversable, standoff=0.25, max_off_axis=math.radians(50),
                    step=math.radians(25)):
    """For each face, a reachable pose from which it can be seen, or None.

    Tries the pose straight in front of the face first, then poses turned off-axis
    by up to max_off_axis (a box on a rack shows its sides from the aisle that way).
    A pose counts if the robot fits there (traversable) and nothing on the map
    blocks the line of sight to the face (the box's own cells do not block).
    Returns a list of (name, pose or None).
    """
    blocked = grid.occupied().copy()
    for r, c in box.cells(grid):
        if grid.inside(r, c):
            blocked[r, c] = False
    offs = [0.0]
    k = 1
    while k * step <= max_off_axis + 1e-9:
        offs += [k * step, -k * step]
        k += 1
    out = []
    for name, center, normal in box.faces():
        found = None
        for off in offs:
            pose = pose_facing(center, wrap(normal + off), standoff)
            r, c = grid.world_to_cell(pose[0], pose[1])
            if not grid.inside(r, c) or not traversable[r, c]:
                continue
            if grid.line_of_sight((pose[0], pose[1]), center, blocked, ignore_last=0):
                found = pose
                break
        out.append((name, found))
    return out


def order_viewpoints(start, named_poses):
    """Visit order for face viewpoints: nearest first, then nearest to the last one."""
    left = [(n, p) for n, p in named_poses if p is not None]
    order, cur = [], start
    while left:
        k = min(range(len(left)), key=lambda i: math.hypot(left[i][1][0] - cur[0],
                                                           left[i][1][1] - cur[1]))
        order.append(left.pop(k))
        cur = order[-1][1]
    return order


# =============================================================================
# QR association
# =============================================================================
def rect_polygon(rect):
    """Corners of a rotated rectangle {'center': [u, v], 'size': [w, h], 'angle_deg': a}."""
    (u, v), (w, h) = rect['center'], rect['size']
    a = math.radians(rect.get('angle_deg', 0.0))
    c, s = math.cos(a), math.sin(a)
    pts = []
    for du, dv in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((u + c * du - s * dv, v + s * du + c * dv))
    return pts


def box_polygon(target):
    """Outline of the target box from a /box_target/target JSON: the fitted rectangle
    of the segmentation (seg models) or, failing that, the bounding box."""
    if target.get('polygon'):
        return [tuple(p) for p in target['polygon']]
    if target.get('rect'):
        return rect_polygon(target['rect'])
    x1, y1, x2, y2 = target['box']
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def point_in_polygon(pt, poly):
    """Ray-casting test; points on the edge count as inside."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x <= xc:
                inside = not inside
    return inside


def polygon_area(poly):
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def qr_belongs_to_box(qr_corners, box_poly, min_ratio=0.05, max_ratio=0.8, margin_px=3.0):
    """True if the QR code (4 corners) lies inside the box outline and has a plausible
    size. The QR codes are about half the face; a QR of a neighboring box falls (at
    least partly) outside the target's outline and is rejected."""
    if len(qr_corners) < 4 or len(box_poly) < 3:
        return False
    cu = sum(p[0] for p in box_poly) / len(box_poly)
    cv = sum(p[1] for p in box_poly) / len(box_poly)
    grown = []
    for u, v in box_poly:                    # small tolerance for the outline fit
        d = math.hypot(u - cu, v - cv) or 1.0
        grown.append((u + (u - cu) / d * margin_px, v + (v - cv) / d * margin_px))
    if not all(point_in_polygon(p, grown) for p in qr_corners):
        return False
    ratio = polygon_area(qr_corners) / max(polygon_area(box_poly), 1.0)
    return min_ratio <= ratio <= max_ratio


# =============================================================================
# Fitting the box face in the LiDAR scan
# =============================================================================
def ransac_line(points, iterations=100, tol=0.01, min_inliers=6, seed=0):
    """Fit a line to 2D points with RANSAC.

    Returns (point_on_line, unit_direction, inlier_mask) or None. Used to square the
    robot to a box face when the face is at LiDAR height (boxes on the floor).
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < max(min_inliers, 2):
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        i, j = rng.choice(pts.shape[0], 2, replace=False)
        d = pts[j] - pts[i]
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        d /= n
        normal = np.array([-d[1], d[0]])
        inl = np.abs((pts - pts[i]) @ normal) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_inliers:
        return None
    q = pts[best]
    c = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - c)
    return c, vt[0], best


def face_from_scan(points_base, max_range=0.6, half_width=0.09, tol=0.01):
    """Box face in front of the robot from LiDAR points in base_link.

    Returns (distance_to_face, face_yaw_error) or None: distance along the robot's x
    axis, and the angle between the face and the robot's y axis (0 = parallel).
    """
    pts = np.asarray(points_base, dtype=float).reshape(-1, 2)
    sel = pts[(pts[:, 0] > 0.0) & (pts[:, 0] < max_range) & (np.abs(pts[:, 1]) < half_width)]
    fit = ransac_line(sel, tol=tol)
    if fit is None:
        return None
    c, d, _ = fit
    yaw_err = wrap(math.atan2(d[1], d[0]) - math.pi / 2)
    if yaw_err > math.pi / 2:
        yaw_err -= math.pi
    if yaw_err < -math.pi / 2:
        yaw_err += math.pi
    # Distance along x where the line crosses y = 0.
    if abs(d[1]) < 1e-6:
        return None
    t = -c[1] / d[1]
    return float(c[0] + t * d[0]), float(yaw_err)
