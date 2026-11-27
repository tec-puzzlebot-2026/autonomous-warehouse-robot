"""
loop_tour.py — stage 2's drive: loops round the arena and round every obstacle (no ROS)
========================================================================================
Planisferio straightens a map when the robot comes back to a place it mapped long
before (a loop closure). So the map tour is made of closed LOOPS, nothing else:

  1. The outer loop: round the whole arena, ring_clearance m from the walls, back to
     where it began. Counter-clockwise, so the walls are on the robot's right.
  2. One loop round every obstacle (or group of obstacles closer together than
     2 x ring_clearance, e.g. a row of racks), nearest first, each back to where it
     began.
  3. Back to the start.

The loops are the contours of "the floor at least ring_clearance from anything" of
the robot's part of the map, so they follow the arena's real shape. If a narrow
passage splits that floor in two at ring_clearance, the clearance is lowered (in
5 cm steps) until it is one piece, so the outer loop goes round the whole arena. Waypoints every
spacing_m along them, each facing along the loop, so the robot drives smoothly
instead of turning back and forth. With face_walls, the robot turns to the wall at
every stop of the outer loop (toward the nearest wall) so the camera sees the ArUco
markers and logos on it, and turns back when it drives on.

    goals, info = plan_loops(grid, start, LoopConfig())
"""
# Author: Raymundo Gomez

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LoopConfig:
    ring_clearance: float = 0.5       # m from walls and obstacles to the loops
    spacing: float = 0.8              # m between waypoints along a loop
    min_loop_length: float = 1.5      # m, shorter loops (round a tiny obstacle) are skipped
    face_walls: bool = True           # outer loop: face the wall at every stop (camera)
    inner_loops: bool = True          # also loop round the obstacles inside the arena
    robot_radius: float = 0.15
    clearance_margin: float = 0.02


def _contours(mask):
    """Outer contours and holes of a binary mask: [(points Nx2 (col, row), parent)]."""
    import cv2
    found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = found[-2], found[-1]
    if hierarchy is None:
        return []
    return [(c.reshape(-1, 2), int(h[3])) for c, h in zip(contours, hierarchy[0])]


def _to_world(grid, pts):
    return np.array([grid.cell_to_world(int(r), int(c)) for c, r in pts], dtype=float)


def _signed_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _resample(xy, spacing):
    """Points every `spacing` m along the closed polyline xy (first point kept)."""
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    total = float(seg.sum())
    n = max(3, int(round(total / spacing)))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for k in range(n):
        s = k * total / n
        i = int(np.searchsorted(cum, s, side='right') - 1)
        i = min(i, len(xy) - 1)
        t = (s - cum[i]) / max(seg[i], 1e-9)
        a, b = xy[i], xy[(i + 1) % len(xy)]
        out.append(a + t * (b - a))
    return np.array(out), total


def _loop_poses(xy, outward_turn=None):
    """(x, y, yaw) along a closed loop, yaw along the loop (or turned by outward_turn)."""
    poses = []
    n = len(xy)
    for k in range(n):
        a, b = xy[k], xy[(k + 1) % n]
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        if outward_turn is not None:
            yaw = math.atan2(math.sin(yaw + outward_turn), math.cos(yaw + outward_turn))
        poses.append((float(a[0]), float(a[1]), yaw))
    return poses


def _face_nearest_wall(grid, poses, reach):
    """Each pose turned toward the nearest wall (the camera looks straight at it)."""
    occ = grid.occupied()
    out = []
    for x, y, yaw in poses:
        best = None
        for k in range(32):
            a = k * math.pi / 16
            for i in range(1, int(reach / grid.res) + 1):
                d = i * grid.res
                r, c = grid.world_to_cell(x + d * math.cos(a), y + d * math.sin(a))
                if grid.inside(r, c) and occ[r, c]:
                    if best is None or d < best[0]:
                        best = (d, a)
                    break
        out.append((x, y, yaw if best is None else math.atan2(math.sin(best[1]),
                                                              math.cos(best[1]))))
    return out


def _rotate_to(poses, xy):
    """Start the loop at its waypoint nearest to xy."""
    k = min(range(len(poses)),
            key=lambda i: math.hypot(poses[i][0] - xy[0], poses[i][1] - xy[1]))
    return poses[k:] + poses[:k]


def _pick_ring(grid, start, reach, clear, cfg):
    """The largest clearance (up to ring_clearance) at which the floor the loops run on
    is still ONE piece: a narrow passage (between racks, say) splits it at a large
    clearance and the outer loop would only go round part of the arena."""
    import cv2
    ring = float(cfg.ring_clearance)
    floor = cfg.robot_radius + 0.05
    best = None
    while ring >= floor - 1e-9:
        mask = reach & (clear >= ring)
        if mask.any():
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),
                                                                connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            if best is None:
                best = (ring, mask)
            if areas.max() >= 0.9 * areas.sum():
                return ring, mask
        ring = round(ring - 0.05, 3)
    return best if best is not None else (cfg.ring_clearance, reach & (clear >= floor))


def plan_loops(grid, start, cfg=None):
    """The tour's goals in order, and a summary dict (loops, length)."""
    cfg = cfg or LoopConfig()
    trav = grid.traversable(cfg.robot_radius, cfg.clearance_margin)
    reach = np.isfinite(grid.path_lengths(start[:2], trav))
    if not reach.any():                       # the start cell itself is not drivable
        reach = trav
    clear = grid.clearance(treat_unknown_as_obstacle=True)
    ring, mask = _pick_ring(grid, start, reach, clear, cfg)
    info = {'loops': 0, 'inner_loops': 0, 'length_m': 0.0, 'ring_clearance': round(ring, 2)}
    if not mask.any():
        return [], info
    conts = _contours(mask)
    outers = [(i, c) for i, (c, parent) in enumerate(conts) if parent < 0 and len(c) >= 3]
    if not outers:
        return [], info
    # The outer loop: the region the robot is in (the biggest, if it is in none).
    sr, sc = grid.world_to_cell(*start[:2])

    def contains(c, r, col):
        import cv2
        return cv2.pointPolygonTest(c.reshape(-1, 1, 2).astype(np.int32),
                                    (float(col), float(r)), False) >= 0

    mine = [(i, c) for i, c in outers if contains(c, sr, sc)]
    oi, oc = (mine or sorted(outers, key=lambda ic: -len(ic[1])))[0]
    goals = []
    here = tuple(start[:2])
    length = 0.0
    loops = []
    xy = _to_world(grid, oc)
    if _signed_area(xy) < 0:
        xy = xy[::-1]                         # counter-clockwise: walls on the right
    pts, total = _resample(xy, cfg.spacing)
    if total >= cfg.min_loop_length:
        loops.append(('outer', pts, total))
    if cfg.inner_loops:
        for c, parent in conts:
            if parent != oi or len(c) < 3:
                continue
            hxy = _to_world(grid, c)
            if _signed_area(hxy) < 0:
                hxy = hxy[::-1]
            hp, ht = _resample(hxy, cfg.spacing)
            if ht >= cfg.min_loop_length:
                loops.append(('inner', hp, ht))
    outer = [lp for lp in loops if lp[0] == 'outer']
    inner = [lp for lp in loops if lp[0] == 'inner']
    order = outer[:]
    while inner:                              # nearest loop next
        last = order[-1][1] if order else np.array([here])
        k = min(range(len(inner)), key=lambda i: float(np.min(
            np.hypot(inner[i][1][:, 0] - last[-1][0], inner[i][1][:, 1] - last[-1][1]))))
        order.append(inner.pop(k))
    for kind, pts_, total_ in order:
        poses = _loop_poses(pts_)
        if kind == 'outer' and cfg.face_walls:
            poses = _face_nearest_wall(grid, poses, ring + 0.5)
        poses = _rotate_to(poses, here)
        poses.append(poses[0])                # close the loop: back where it began
        goals.extend(poses)
        here = poses[0][:2]
        length += total_
    info.update(loops=len(order), inner_loops=sum(1 for o in order if o[0] == 'inner'),
                length_m=round(length, 2))
    return goals, info
