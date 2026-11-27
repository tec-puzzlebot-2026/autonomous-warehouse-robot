"""
loop_tour.py — stage 2's drive: many small closed loops, then one round everything (no ROS)
==========================================================================================
Planisferio straightens a map when the robot comes back to a place it mapped long
before (a loop closure). So the map tour is made of closed LOOPS, nothing else:

  1. Sector loops: the arena is cut across its long side into sectors about
     loop_size m long; the robot drives one closed loop round each sector, one
     after the other. Neighbouring loops run along the same cut line (in opposite
     directions), so every new loop comes back over ground the previous one mapped:
     one loop closure per sector, plus one when each loop closes on itself.
  2. One loop round every obstacle (or group of obstacles closer together than
     2 x ring_clearance, e.g. a row of racks), right after the sector it is in.
  3. The big loop round the whole arena (outer_loop), which ties the sectors
     together, and back to the start.

The loops are contours of "the floor at least ring_clearance from anything" of the
robot's part of the map, so they follow the arena's real shape. If a narrow passage
splits that floor at ring_clearance, the clearance is lowered (5 cm steps) until it
is one piece. Waypoints every spacing m, each facing along the loop, so the robot
drives on smoothly (the map tour passes through them, map_tour.py); with
face_walls the outer-loop waypoints face the nearest wall instead (the robot turns
at every stop: slower).

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
    loop_size: float = 1.6            # m, longest sector (0 = no sectors: one loop round all)
    outer_loop: bool = True           # after the sectors, one loop round the whole arena
    min_loop_length: float = 1.5      # m, shorter loops (round a tiny obstacle) are skipped
    face_walls: bool = False          # outer-loop waypoints face the nearest wall
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
        i = min(int(np.searchsorted(cum, s, side='right') - 1), len(xy) - 1)
        t = (s - cum[i]) / max(seg[i], 1e-9)
        a, b = xy[i], xy[(i + 1) % len(xy)]
        out.append(a + t * (b - a))
    return np.array(out), total


def _loop(grid, contour, spacing):
    """A contour as a counter-clockwise loop of waypoints: (points, length). The
    contour is straightened first (corners kept, cell steps gone)."""
    import cv2
    eps = max(1.0, 0.15 / grid.res)           # 15 cm
    simple = cv2.approxPolyDP(contour.reshape(-1, 1, 2).astype(np.int32), eps, True)
    if len(simple) >= 3:
        contour = simple.reshape(-1, 2)
    xy = _to_world(grid, contour)
    if _signed_area(xy) < 0:
        xy = xy[::-1]
    return _resample(xy, spacing)


def _loop_poses(xy):
    """(x, y, yaw) along a closed loop, yaw toward the next waypoint."""
    n = len(xy)
    return [(float(xy[k][0]), float(xy[k][1]),
             math.atan2(xy[(k + 1) % n][1] - xy[k][1], xy[(k + 1) % n][0] - xy[k][0]))
            for k in range(n)]


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


def _pick_ring(reach, clear, cfg):
    """The largest clearance (up to ring_clearance) at which the floor the loops run on
    is still ONE piece: a narrow passage (between racks, say) splits it at a large
    clearance and the loops would only cover part of the arena."""
    import cv2
    ring = float(cfg.ring_clearance)
    floor = cfg.robot_radius + 0.05
    best = None
    while ring >= floor - 1e-9:
        mask = reach & (clear >= ring)
        if mask.any():
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),
                                                              connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            if best is None:
                best = (ring, mask)
            if areas.max() >= 0.9 * areas.sum():
                return ring, mask
        ring = round(ring - 0.05, 3)
    return best if best is not None else (cfg.ring_clearance, reach & (clear >= floor))


def _start_component(mask, grid, start):
    """The connected piece of `mask` the robot is in (the biggest if it is in none)."""
    import cv2
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    r, c = grid.world_to_cell(*start[:2])
    k = int(lab[r, c]) if grid.inside(r, c) else 0
    if k == 0:                                # not inside: the nearest / biggest piece
        k = int(np.argmax(np.bincount(lab[lab > 0].ravel())))
    return lab == k


def _cuts(mask, grid, loop_size):
    """Where to cut the floor across its long side: (by_cols, [cell index], lo, hi)."""
    rows, cols = np.nonzero(mask)
    by_cols = (cols.max() - cols.min()) * 1.0 >= (rows.max() - rows.min())
    lo, hi = (int(cols.min()), int(cols.max())) if by_cols else (int(rows.min()),
                                                                 int(rows.max()))
    if loop_size <= 0:
        return by_cols, [], lo, hi
    n = max(1, int(math.ceil((hi - lo + 1) * grid.res / loop_size - 0.1)))
    count = mask.sum(axis=0) if by_cols else mask.sum(axis=1)
    span = (hi - lo + 1) / n
    edges = []
    for k in range(1, n):                     # each cut where the floor is most open
        e0 = int(round(lo + k * span))
        w = max(1, int(0.25 * span))
        cand = list(range(max(lo + 2, e0 - w), min(hi - 1, e0 + w) + 1))
        edges.append(max(cand, key=lambda e: (count[e], -abs(e - e0))) if cand else e0)
    return by_cols, edges, lo, hi


def _snap(grid, mask, xy, radius):
    """Every point moved to the nearest cell of `mask` (dropped if none within radius);
    points closer than 15 cm to the previous one are dropped."""
    rr, cc = np.nonzero(mask)
    wx = grid.origin[0] + (cc + 0.5) * grid.res
    wy = grid.origin[1] + (rr + 0.5) * grid.res
    out = []
    for x, y in xy:
        d = np.hypot(wx - x, wy - y)
        k = int(np.argmin(d))
        if d[k] > radius:
            continue
        p = (float(wx[k]), float(wy[k]))
        if out and math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) < 0.15:
            continue
        out.append(p)
    if len(out) > 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 0.15:
        out.pop()
    return np.array(out)


def _rectangle(x0, x1, y0, y1, spacing):
    """A counter-clockwise rectangle as points every ~spacing m, and its length."""
    corners = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=float)
    return _resample(corners, spacing)


def plan_loops(grid, start, cfg=None):
    """The tour's goals in order, and a summary dict (loops, length)."""
    cfg = cfg or LoopConfig()
    trav = grid.traversable(cfg.robot_radius, cfg.clearance_margin)
    reach = np.isfinite(grid.path_lengths(start[:2], trav))
    if not reach.any():                       # the start cell itself is not drivable
        reach = trav
    clear = grid.clearance(treat_unknown_as_obstacle=True)
    ring, mask = _pick_ring(reach, clear, cfg)
    info = {'loops': 0, 'sector_loops': 0, 'inner_loops': 0, 'length_m': 0.0,
            'ring_clearance': round(ring, 2)}
    if not mask.any():
        return [], info
    mask = _start_component(mask, grid, start)

    # [(kind, points, length)] in driving order
    loops = []
    rows, cols = np.nonzero(mask)
    X0, Y0 = grid.cell_to_world(int(rows.min()), int(cols.min()))
    X1, Y1 = grid.cell_to_world(int(rows.max()), int(cols.max()))
    by_cols, edges, lo, hi = _cuts(mask, grid, cfg.loop_size)
    snap_r = ring + 0.3

    def add_rect(kind, x0, x1, y0, y1):
        pts, _ = _rectangle(x0, x1, y0, y1, cfg.spacing)
        pts = _snap(grid, mask, pts, snap_r)
        if len(pts) >= 3:
            total = float(np.sum(np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)))
            if total >= cfg.min_loop_length:
                loops.append((kind, pts, total))

    # holes: loops round the obstacles inside the floor, after their sector
    holes = []
    if cfg.inner_loops:
        for c, p in _contours(mask):
            if p >= 0 and len(c) >= 3:
                hp, ht = _loop(grid, c, cfg.spacing)
                if ht >= cfg.min_loop_length:
                    holes.append((hp, ht))
    if edges:
        half = grid.res                       # the two loops at a cut: 2 cells apart
        bounds = [lo] + edges + [hi]
        world = [grid.cell_to_world(0, b)[0] if by_cols else grid.cell_to_world(b, 0)[1]
                 for b in bounds]
        spans = list(zip(world, world[1:]))
        here_axis = start[0] if by_cols else start[1]
        if abs(here_axis - world[-1]) < abs(here_axis - world[0]):
            spans = spans[::-1]               # start from the end the robot is nearest
        ends = (world[0], world[-1])
        for a, b in spans:
            lo_, hi_ = min(a, b), max(a, b)
            lo_ += half if lo_ not in ends else 0.0
            hi_ -= half if hi_ not in ends else 0.0
            if by_cols:
                add_rect('sector', lo_, hi_, Y0, Y1)
            else:
                add_rect('sector', X0, X1, lo_, hi_)
            keep = []
            for hp, ht in holes:              # obstacles in this sector
                m = hp.mean(axis=0)
                v = m[0] if by_cols else m[1]
                if lo_ - half <= v <= hi_ + half:
                    loops.append(('inner', hp, ht))
                else:
                    keep.append((hp, ht))
            holes = keep
    for hp, ht in holes:
        loops.append(('inner', hp, ht))
    if not edges or cfg.outer_loop:
        add_rect('outer', X0, X1, Y0, Y1)     # round the whole arena

    goals = []
    here = tuple(start[:2])
    for kind, pts, _total in loops:
        poses = _loop_poses(pts)
        if kind == 'outer' and cfg.face_walls:
            poses = _face_nearest_wall(grid, poses, ring + 0.5)
        poses = _rotate_to(poses, here)
        poses.append(poses[0])                # close the loop: back where it began
        goals.extend(poses)
        here = poses[0][:2]
    info.update(loops=len(loops),
                sector_loops=sum(1 for lp in loops if lp[0] == 'sector'),
                inner_loops=sum(1 for lp in loops if lp[0] == 'inner'),
                length_m=round(sum(lp[2] for lp in loops), 2))
    return goals, info
