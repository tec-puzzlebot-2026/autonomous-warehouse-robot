"""
docks.py — find each company's dock on the map from its logo (no ROS).

The arena has one dock (a trailer opening in a wall) per company, with the
company's logo on the wall just to the LEFT of the opening (as seen from inside
the arena). Nothing about the docks is stored beforehand: after mapping, for
every logo found, this module

  1. finds the wall the logo is on and its inward normal (toward the free floor),
  2. walks along the wall to the RIGHT of the logo until the wall opens into a
     recess at least `min_depth` deep and `min_width` wide (the trailer),
  3. returns the dock: the entrance pose (centered in front of the opening, facing
     in) and the inside pose (a fraction of the recess depth, facing in).

If logos or docks move, the next mapping run finds the new layout.
"""
# Author: Raymundo Gomez

import math
from dataclasses import dataclass

import numpy as np

from main_ros.restock.geometry import wrap


@dataclass
class Dock:
    company: str
    entrance: tuple          # (x, y, theta): in front of the opening, facing in
    inside: tuple            # (x, y, theta): where the box is set down
    width: float             # opening width [m]
    depth: float             # recess depth measured on the map [m]
    logo: tuple              # (x, y) of the logo

    def to_dict(self):
        r = lambda p: [round(float(v), 4) for v in p]       # noqa: E731
        return {'company': self.company, 'entrance': r(self.entrance), 'inside': r(self.inside),
                'width': round(self.width, 3), 'depth': round(self.depth, 3),
                'logo': r(self.logo)}

    def contains(self, x, y, margin=0.10):
        """True if (x, y) is inside the trailer (beyond the mouth, within its width).
        Boxes seen there were already delivered and must be left alone."""
        th = self.entrance[2]                      # facing into the trailer
        mouth_x = 0.5 * (self.entrance[0] + self.inside[0])
        mouth_y = 0.5 * (self.entrance[1] + self.inside[1])
        dx, dy = x - self.entrance[0], y - self.entrance[1]
        along = dx * math.cos(th) + dy * math.sin(th)
        across = -dx * math.sin(th) + dy * math.cos(th)
        mdx, mdy = mouth_x - self.entrance[0], mouth_y - self.entrance[1]
        mouth_along = mdx * math.cos(th) + mdy * math.sin(th)
        return along >= mouth_along - margin and abs(across) <= self.width / 2 + margin

    @classmethod
    def from_dict(cls, d):
        return cls(d['company'], tuple(d['entrance']), tuple(d['inside']),
                   float(d['width']), float(d['depth']), tuple(d['logo']))


def wall_normal(grid, point, radius=0.5):
    """Inward normal of the wall at `point`: mean direction from it to the free cells
    around it (the side the robot saw the logo from). None if no free cell nearby."""
    free = grid.free()
    r0, c0 = grid.world_to_cell(*point)
    n = int(math.ceil(radius / grid.res))
    vx = vy = 0.0
    for r in range(r0 - n, r0 + n + 1):
        for c in range(c0 - n, c0 + n + 1):
            if grid.inside(r, c) and free[r, c]:
                x, y = grid.cell_to_world(r, c)
                d = math.hypot(x - point[0], y - point[1])
                if 0 < d <= radius:
                    vx += (x - point[0]) / d
                    vy += (y - point[1]) / d
    if vx == 0.0 and vy == 0.0:
        return None
    return math.atan2(vy, vx)


def refine_normal(grid, point, hint, span=math.radians(30), half_len=0.4):
    """Best wall normal near `hint`: the direction whose wall line (through `point`,
    perpendicular to the normal) runs over the most occupied cells. Robust to a
    trailer's side wall next to the logo, which would bias a plain average."""
    occ = grid.occupied()
    best, best_score = hint, -1
    for k in range(-15, 16):
        n = wrap(hint + k * span / 15)
        t = n + math.pi / 2
        score = 0
        for off in np.arange(-0.15, 0.151, grid.res / 2):        # the wall may be a bit off
            cnt = 0
            for a in np.arange(-half_len, half_len + 1e-9, grid.res):
                x = point[0] + a * math.cos(t) - off * math.cos(n)
                y = point[1] + a * math.sin(t) - off * math.sin(n)
                r, c = grid.world_to_cell(x, y)
                if grid.inside(r, c) and occ[r, c]:
                    cnt += 1
            score = max(score, cnt)
        if score > best_score:
            best, best_score = n, score
    return best


def snap_to_wall(grid, point, normal, search=0.3):
    """Move `point` along the normal onto the nearest occupied cell (the wall surface)."""
    occ = grid.occupied()
    best = None
    steps = int(search / (grid.res / 2))
    for k in range(-steps, steps + 1):
        t = k * grid.res / 2
        x = point[0] - t * math.cos(normal)
        y = point[1] - t * math.sin(normal)
        r, c = grid.world_to_cell(x, y)
        if grid.inside(r, c) and occ[r, c]:
            if best is None or abs(t) < abs(best[0]):
                best = (t, (x, y))
    return point if best is None else best[1]


def recess_depth(grid, wall_point, into, max_depth=1.5):
    """How far one can go from a point on the wall line into the wall direction
    (`into`) before hitting an obstacle. Unknown cells count as open (the inside of a
    trailer is often only partly seen), but at least one free cell is required."""
    occ = grid.occupied()
    free = grid.free()
    step = grid.res / 2
    depth, seen_free = 0.0, False
    t = step
    while t <= max_depth:
        x = wall_point[0] + t * math.cos(into)
        y = wall_point[1] + t * math.sin(into)
        r, c = grid.world_to_cell(x, y)
        if not grid.inside(r, c) or occ[r, c]:
            break
        seen_free = seen_free or bool(free[r, c])
        depth = t
        t += step
    return depth if seen_free else 0.0


def find_dock(grid, company, logo_xy, min_depth=0.25, min_width=0.18, search=1.5,
              inside_fraction=0.6, entrance_standoff=0.35, max_inside=0.6, normal_hint=None):
    """The dock to the right of a logo, or None if no opening is found.

    normal_hint: direction from the logo toward where the robot saw it from (the
    landmark mapper records it); it makes the wall direction unambiguous.
    """
    normal = normal_hint if normal_hint is not None else wall_normal(grid, logo_xy)
    if normal is None:
        return None
    normal = refine_normal(grid, logo_xy, normal)
    wall = snap_to_wall(grid, logo_xy, normal)
    into = wrap(normal + math.pi)                     # facing the wall
    right = wrap(into - math.pi / 2)                  # right-hand side when facing it
    step = grid.res / 2
    # The wall line is probed slightly in front of the wall surface, then into it.
    base = (wall[0] + 0.5 * grid.res * math.cos(normal),
            wall[1] + 0.5 * grid.res * math.sin(normal))
    opening, best = [], None
    s = step
    while s <= search:
        p = (base[0] + s * math.cos(right), base[1] + s * math.sin(right))
        d = recess_depth(grid, p, into)
        if d >= min_depth:
            opening.append((s, d))
        elif opening:
            if opening[-1][0] - opening[0][0] >= min_width:
                best = opening
                break
            opening = []
        s += step
    if best is None and opening and opening[-1][0] - opening[0][0] >= min_width:
        best = opening
    if best is None:
        return None
    s_mid = 0.5 * (best[0][0] + best[-1][0])
    width = best[-1][0] - best[0][0] + step
    depth = float(np.median([d for _, d in best]))
    mouth = (base[0] + s_mid * math.cos(right), base[1] + s_mid * math.sin(right))
    entrance = (mouth[0] + entrance_standoff * math.cos(normal),
                mouth[1] + entrance_standoff * math.sin(normal), into)
    go_in = min(inside_fraction * depth, max_inside)
    inside = (mouth[0] + go_in * math.cos(into), mouth[1] + go_in * math.sin(into), into)
    return Dock(company, entrance, inside, width, depth, (float(wall[0]), float(wall[1])))


def find_docks(grid, logos, **kw):
    """{company: Dock} for every logo whose dock is found.

    logos: {company: (x, y)} or {company: (x, y, view_yaw)}, where view_yaw points from
    the logo toward the robot that saw it (used as the wall normal hint).
    """
    out = {}
    for company, v in logos.items():
        hint = v[2] if len(v) > 2 and v[2] is not None else None
        d = find_dock(grid, company, (v[0], v[1]), normal_hint=hint, **kw)
        if d is not None:
            out[company] = d
    return out


def free_area_near(grid, traversable, points, radius_needed=0.45, search=3.0):
    """A spot near `points` (the docks) with the most open floor around it: the
    center of the largest clear circle within `search` m of their centroid. Used to
    set a box down and drive around it. Returns (x, y) or None."""
    if not points:
        return None
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    clear = grid.clearance()
    best, best_score = None, -1.0
    rr, cc = np.nonzero(traversable)
    for r, c in zip(rr, cc):
        x, y = grid.cell_to_world(r, c)
        d = math.hypot(x - cx, y - cy)
        if d > search or clear[r, c] < radius_needed:
            continue
        score = clear[r, c] - 0.2 * d           # open space first, then close to the docks
        if score > best_score:
            best, best_score = (x, y), score
    return best


def find_openings(grid, near=None, search=2.5, min_area=0.06, min_depth=0.25,
                  entrance_standoff=0.35, inside_fraction=0.6, max_inside=0.6):
    """Every dock opening of the map (a recess off the main room), without logos.

    The room is the free floor that a disk ~0.5 m wide can sweep; a trailer is
    narrower than that, so what is left of the free floor beyond the room's walls
    are the trailers. near: only openings within `search` m of this point (the drop
    zone). Returns [Dock] with company '' (the logo at the opening names it later).
    """
    import cv2
    free = grid.free().astype(np.uint8)
    rad = max(1, int(math.ceil(0.25 / grid.res)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    opened = cv2.morphologyEx(free, cv2.MORPH_OPEN, disk)
    n, lab = cv2.connectedComponents(opened, connectivity=8)
    if n <= 1:
        return []
    room = lab == int(np.argmax(np.bincount(lab[lab > 0].ravel())))
    rr, cc = np.nonzero(room)
    r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
    recess = (free > 0) & ~room
    m, rlab = cv2.connectedComponents(recess.astype(np.uint8), connectivity=8)
    touch = cv2.dilate(room.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = []
    for k in range(1, m):
        cells = rlab == k
        if cells.sum() * grid.res ** 2 < min_area:
            continue
        ys, xs = np.nonzero(cells)
        cy, cx = ys.mean(), xs.mean()
        if r0 <= cy <= r1 and c0 <= cx <= c1:
            continue                               # a gap inside the room, not a trailer
        mouth = cells & touch
        if not mouth.any():
            continue
        my, mx = np.nonzero(mouth)
        mouth_xy = (grid.origin[0] + (mx.mean() + 0.5) * grid.res,
                    grid.origin[1] + (my.mean() + 0.5) * grid.res)
        cen = (grid.origin[0] + (cx + 0.5) * grid.res, grid.origin[1] + (cy + 0.5) * grid.res)
        into = math.atan2(cen[1] - mouth_xy[1], cen[0] - mouth_xy[0])
        # snap the direction to the room's axes (walls are straight)
        into = round(into / (math.pi / 2)) * (math.pi / 2)
        wx = grid.origin[0] + (xs + 0.5) * grid.res
        wy = grid.origin[1] + (ys + 0.5) * grid.res
        along = (wx - mouth_xy[0]) * math.cos(into) + (wy - mouth_xy[1]) * math.sin(into)
        across = -(wx - mouth_xy[0]) * math.sin(into) + (wy - mouth_xy[1]) * math.cos(into)
        depth = float(along.max()) + grid.res / 2
        width = float(across.max() - across.min()) + grid.res
        if depth < min_depth:
            continue
        mid = float(0.5 * (across.max() + across.min()))
        mouth_xy = (mouth_xy[0] - mid * math.sin(into), mouth_xy[1] + mid * math.cos(into))
        if near is not None and math.hypot(mouth_xy[0] - near[0],
                                           mouth_xy[1] - near[1]) > search:
            continue
        entrance = (mouth_xy[0] - entrance_standoff * math.cos(into),
                    mouth_xy[1] - entrance_standoff * math.sin(into), wrap(into))
        go_in = min(inside_fraction * depth, max_inside)
        inside = (mouth_xy[0] + go_in * math.cos(into), mouth_xy[1] + go_in * math.sin(into),
                  wrap(into))
        out.append(Dock('', entrance, inside, width, depth, mouth_xy))
    return out
